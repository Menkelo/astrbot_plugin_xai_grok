import asyncio
import base64
import io
import re
from functools import partial
from typing import Any, List, Optional, Tuple

import httpx
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Image as AstrImage, At, Reply

try:
    from PIL import Image as PILImage
    from PIL import ImageFilter
except ImportError:
    PILImage = None
    ImageFilter = None


class ImageService:
    def __init__(self, http_client: httpx.AsyncClient, context, max_image_size=5 * 1024 * 1024):
        self.http_client = http_client
        self.context = context
        self.max_image_size = max_image_size

    @staticmethod
    def _format_base64(base64_str: str) -> str:
        base64_str = base64_str.replace("\n", "").replace("\r", "")
        if not base64_str.startswith("data:"):
            return f"data:image/jpeg;base64,{base64_str}"
        return base64_str

    @staticmethod
    def _fit_image_to_aspect(img, target_ratio: float):
        width, height = img.size
        if width <= 0 or height <= 0 or target_ratio <= 0:
            return img

        ratio = width / height
        if abs(ratio - target_ratio) < 0.01:
            return img

        if target_ratio >= ratio:
            canvas_h = height
            canvas_w = max(width, int(round(canvas_h * target_ratio)))
        else:
            canvas_w = width
            canvas_h = max(height, int(round(canvas_w / target_ratio)))

        bg = img.copy()
        bg_ratio = width / height
        if bg_ratio > target_ratio:
            bg_h = canvas_h
            bg_w = int(round(bg_h * bg_ratio))
        else:
            bg_w = canvas_w
            bg_h = int(round(bg_w / bg_ratio))
        bg = bg.resize((bg_w, bg_h), PILImage.Resampling.LANCZOS)
        left = max(0, (bg_w - canvas_w) // 2)
        top = max(0, (bg_h - canvas_h) // 2)
        bg = bg.crop((left, top, left + canvas_w, top + canvas_h))
        if ImageFilter:
            bg = bg.filter(ImageFilter.GaussianBlur(radius=18))

        fg = img.copy()
        fg.thumbnail((canvas_w, canvas_h), PILImage.Resampling.LANCZOS)
        x = (canvas_w - fg.width) // 2
        y = (canvas_h - fg.height) // 2
        bg.paste(fg, (x, y))
        return bg

    def _process_image_sync(
        self,
        base64_str: str,
        crop_for_video=False,
        target_aspect_ratio: Optional[float] = None
    ) -> str:
        if not PILImage:
            return self._format_base64(base64_str)

        try:
            if "," in base64_str:
                _, data = base64_str.split(",", 1)
            else:
                data = base64_str

            try:
                image_data = base64.b64decode(data)
            except Exception:
                data = re.sub(r"[^a-zA-Z0-9+/=]", "", data)
                image_data = base64.b64decode(data)

            original_size = len(image_data)
            is_too_large = original_size > self.max_image_size

            if not crop_for_video and not is_too_large:
                return self._format_base64(base64_str)

            with io.BytesIO(image_data) as input_buffer:
                img = PILImage.open(input_buffer)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                if crop_for_video and target_aspect_ratio:
                    img = self._fit_image_to_aspect(img, target_aspect_ratio)
                    logger.info(
                        f"[video.image] fit reference image to target_ratio={target_aspect_ratio:.4f}, "
                        f"output_size={img.size[0]}x{img.size[1]}"
                    )

                elif crop_for_video:
                    width, height = img.size
                    ratio = width / height
                    if 0.85 <= ratio <= 1.15:
                        target_ratio = 1.0
                    elif ratio > 1.15:
                        target_ratio = 3 / 2 if ratio < 1.6 else 16 / 9
                    else:
                        target_ratio = 2 / 3 if ratio > 0.62 else 9 / 16

                    if ratio > target_ratio:
                        new_width = int(height * target_ratio)
                        left = (width - new_width) // 2
                        img = img.crop((left, 0, left + new_width, height))
                    elif ratio < target_ratio:
                        new_height = int(width / target_ratio)
                        top = (height - new_height) // 2
                        img = img.crop((0, top, width, top + new_height))

                save_kwargs = {"format": "JPEG"}
                needs_resize = is_too_large or max(img.size) > 2048
                if needs_resize:
                    img.thumbnail((2048, 2048), PILImage.Resampling.LANCZOS)
                    save_kwargs["quality"] = 80 if is_too_large else 92
                else:
                    save_kwargs["quality"] = 95
                    save_kwargs["subsampling"] = 0

                with io.BytesIO() as output_buffer:
                    img.save(output_buffer, **save_kwargs)
                    jpeg_data = output_buffer.getvalue()
                    new_base64 = base64.b64encode(jpeg_data).decode("utf-8")
                    return f"data:image/jpeg;base64,{new_base64}"

        except Exception as e:
            logger.error(f"图片处理失败: {e}，将使用原图")
            return self._format_base64(base64_str)

    async def _fetch_url_base64(self, url: str) -> Optional[str]:
        try:
            r = await self.http_client.get(url, timeout=httpx.Timeout(30.0))
            if r.status_code == 200:
                return base64.b64encode(r.content).decode()
        except Exception:
            pass
        return None

    def _component_to_candidate(self, comp, allow_at=True):
        if isinstance(comp, AstrImage):
            return ("b64", comp.convert_to_base64())
        if allow_at and isinstance(comp, At) and comp.qq:
            return ("url", f"https://q.qlogo.cn/headimg_dl?dst_uin={comp.qq}&spec=640")
        return None

    def _collect_candidates_from_chain(self, chain, add_candidate, allow_at=True) -> bool:
        if not chain:
            return False
        for comp in chain:
            cand = self._component_to_candidate(comp, allow_at)
            if cand and add_candidate(cand):
                return True
        return False

    async def _collect_candidates_from_reply(self, event: AstrMessageEvent, add_candidate) -> bool:
        reply_comp = next((c for c in event.message_obj.message if isinstance(c, Reply)), None)
        if not reply_comp:
            return False

        if reply_comp.chain:
            if self._collect_candidates_from_chain(reply_comp.chain, add_candidate, allow_at=False):
                return True

        if reply_comp.id:
            try:
                bot = event.bot or (self.context.get_bot() if hasattr(self.context, "get_bot") else None)
                if bot:
                    resp = await bot.api.call_action("get_msg", message_id=int(reply_comp.id))
                    content = resp.get("message")

                    img_urls = []
                    if isinstance(content, list):
                        for seg in content:
                            if isinstance(seg, dict) and seg.get("type") == "image":
                                data = seg.get("data", {})
                                u = data.get("url") or data.get("file")
                                if u and str(u).startswith("http"):
                                    img_urls.append(u)
                    elif isinstance(content, str):
                        urls = re.findall(r"url=(http[^,\]]+)", content)
                        img_urls.extend([u.replace("&amp;", "&") for u in urls])

                    for u in img_urls:
                        if add_candidate(("url", u)):
                            return True
            except Exception:
                pass
        return False

    async def _resolve_candidates(
        self,
        candidates: List[Tuple[str, Any]],
        crop_for_video: bool,
        target_aspect_ratio: Optional[float] = None
    ) -> List[str]:
        if not candidates:
            return []
        loop = asyncio.get_running_loop()

        async def resolve_one(cand):
            try:
                kind, data = cand
                if kind == "b64":
                    b64 = await data if asyncio.iscoroutine(data) else data
                else:
                    b64 = await self._fetch_url_base64(data)

                if not b64:
                    return None
                fn = partial(
                    self._process_image_sync,
                    b64,
                    crop_for_video,
                    target_aspect_ratio
                )
                return await loop.run_in_executor(None, fn)
            except Exception:
                return None

        results = await asyncio.gather(*(resolve_one(c) for c in candidates), return_exceptions=True)
        return [r for r in results if isinstance(r, str)]

    async def extract_images_from_message(
        self,
        event: AstrMessageEvent,
        crop_for_video=False,
        target_index=-1,
        target_aspect_ratio: Optional[float] = None
    ) -> List[str]:
        if not hasattr(event, "message_obj") or not event.message_obj:
            return []

        candidates = []
        current_idx = 0

        def add_candidate(cand):
            nonlocal current_idx
            if target_index != -1:
                if current_idx == target_index:
                    candidates.append(cand)
                    current_idx += 1
                    return True
                current_idx += 1
                return False
            candidates.append(cand)
            current_idx += 1
            return False

        found = self._collect_candidates_from_chain(event.message_obj.message, add_candidate, allow_at=True)
        if found:
            return await self._resolve_candidates(candidates, crop_for_video, target_aspect_ratio)

        if candidates and target_index == -1:
            return await self._resolve_candidates(candidates, crop_for_video, target_aspect_ratio)

        await self._collect_candidates_from_reply(event, add_candidate)
        return await self._resolve_candidates(candidates, crop_for_video, target_aspect_ratio)
