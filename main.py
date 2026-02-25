import asyncio
import json
import re
import uuid
import io
import base64
import mimetypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Any
from urllib.parse import urljoin, urlparse

import httpx
import aiofiles
from astrbot.api import logger
from astrbot.api.all import *
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.api.message_components import Video, Image as AstrImage, Plain, Reply, At

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None
    logger.warning("未检测到 Pillow 库，图片处理功能不可用，建议安装: pip install Pillow")


@dataclass
class ProviderRuntime:
    provider_id: str
    api_type: str
    base_url: str  # 规范化为 .../v1
    api_key: str
    model: str


class GrokMediaPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config

        # 仅保留 provider 选择
        self.video_provider_id = config.get("video_provider_id", "")
        self.image_provider_id = config.get("image_provider_id", "")
        self.nsfw_provider_id = config.get("nsfw_provider_id", "")

        self.timeout_seconds = 180
        self.max_retry_attempts = 3
        self.max_image_size = 5 * 1024 * 1024
        self.save_video_enabled = False

        try:
            plugin_data_dir = Path(StarTools.get_data_dir("astrbot_plugin_xai_grok"))
            self.data_dir = plugin_data_dir / "downloads"
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.data_dir = self.data_dir.resolve()
        except Exception as e:
            logger.warning(f"无法使用StarTools数据目录: {e}")
            self.data_dir = Path(__file__).parent / "downloads"
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.data_dir = self.data_dir.resolve()

        self.http_client = httpx.AsyncClient(follow_redirects=True)

        logger.info(
            "Grok多媒体插件已初始化，providers: "
            f"video={self.video_provider_id or '-'}, "
            f"image={self.image_provider_id or '-'}, "
            f"nsfw={self.nsfw_provider_id or '-'}"
        )

    async def on_unload(self):
        if getattr(self, "http_client", None):
            await self.http_client.aclose()

    def _format_base64(self, base64_str: str) -> str:
        base64_str = base64_str.replace("\n", "").replace("\r", "")
        if not base64_str.startswith("data:"):
            return f"data:image/jpeg;base64,{base64_str}"
        return base64_str

    async def _reply_error(self, event: AstrMessageEvent, text: str):
        try:
            await event.send(event.chain_result([
                Reply(id=str(event.message_obj.message_id)),
                Plain(text)
            ]))
        except Exception:
            await event.send(event.plain_result(text))

    def _process_image_sync(self, base64_str: str, crop_for_video: bool = False) -> str:
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

                if crop_for_video:
                    width, height = img.size
                    ratio = width / height
                    if 0.85 <= ratio <= 1.15:
                        target_ratio = 1.0
                        logger_msg = "1:1 方形"
                    elif ratio > 1.15:
                        if ratio < 1.6:
                            target_ratio = 3 / 2
                            logger_msg = "3:2 横屏"
                        else:
                            target_ratio = 16 / 9
                            logger_msg = "16:9 横屏"
                    else:
                        if ratio > 0.62:
                            target_ratio = 2 / 3
                            logger_msg = "2:3 竖屏"
                        else:
                            target_ratio = 9 / 16
                            logger_msg = "9:16 竖屏"

                    if ratio > target_ratio:
                        new_width = int(height * target_ratio)
                        left = (width - new_width) // 2
                        img = img.crop((left, 0, left + new_width, height))
                    elif ratio < target_ratio:
                        new_height = int(width / target_ratio)
                        top = (height - new_height) // 2
                        img = img.crop((0, top, width, top + new_height))
                    logger.info(f"图片已自动裁剪为 {logger_msg}")

                save_kwargs = {"format": "JPEG"}
                if is_too_large:
                    img.thumbnail((2048, 2048), PILImage.Resampling.LANCZOS)
                    save_kwargs["quality"] = 80
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

    def _component_to_candidate(self, comp, allow_at: bool = True):
        if isinstance(comp, AstrImage):
            return ("b64", comp.convert_to_base64())
        if allow_at and isinstance(comp, At) and comp.qq:
            url = f"https://q.qlogo.cn/headimg_dl?dst_uin={comp.qq}&spec=640"
            return ("url", url)
        return None

    def _collect_candidates_from_chain(self, chain, add_candidate, allow_at=True) -> bool:
        if not chain:
            return False
        for comp in chain:
            cand = self._component_to_candidate(comp, allow_at=allow_at)
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

    async def _resolve_candidates(self, candidates: List[Tuple[str, Any]], crop_for_video: bool) -> List[str]:
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
                return await loop.run_in_executor(None, self._process_image_sync, b64, crop_for_video)
            except Exception:
                return None

        results = await asyncio.gather(*(resolve_one(c) for c in candidates), return_exceptions=True)
        return [r for r in results if isinstance(r, str)]

    async def _extract_images_from_message(
        self, event: AstrMessageEvent, crop_for_video: bool = False, target_index: int = -1
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

        found = self._collect_candidates_from_chain(
            event.message_obj.message, add_candidate, allow_at=True
        )

        if found:
            return await self._resolve_candidates(candidates, crop_for_video)

        if candidates and target_index == -1:
            return await self._resolve_candidates(candidates, crop_for_video)

        await self._collect_candidates_from_reply(event, add_candidate)
        return await self._resolve_candidates(candidates, crop_for_video)

    async def _run_task_times(self, event, prompt, task_type, image_base64, times: int):
        times = max(1, times)
        for i in range(times):
            async for res in self._process_task(event, prompt, task_type, image_base64, show_status=(i == 0)):
                yield res

    def _extract_prompt_after_command(self, event: AstrMessageEvent, command: str) -> str:
        text = (event.message_str or "").strip()
        m = re.match(rf"^[\\/!]?{re.escape(command)}\s*([\s\S]*)$", text, re.IGNORECASE)
        return (m.group(1) if m else "").strip()

    # ========= Provider 解析（参考 Gemini 插件）=========

    def _extract_model_from_provider_id(self, provider_id: str) -> str:
        if provider_id and "/" in provider_id:
            return provider_id.split("/", 1)[1].strip()
        return ""

    def _normalize_openai_base(self, base_url: str) -> str:
        b = (base_url or "").strip().rstrip("/")
        if not b:
            return ""
        if not b.endswith("/v1"):
            b = b + "/v1"
        return b

    def _build_endpoint(self, base_v1: str, path_after_v1: str) -> str:
        return f"{base_v1}/{path_after_v1.lstrip('/')}"

    def _parse_provider(self, provider_id: str) -> Tuple[Optional[ProviderRuntime], Optional[str]]:
        if not provider_id:
            return None, "未配置提供商"

        provider = None
        if hasattr(self.context, "get_provider_by_id"):
            try:
                provider = self.context.get_provider_by_id(provider_id)
            except Exception:
                provider = None

        if not provider and "/" in provider_id and hasattr(self.context, "get_provider_by_id"):
            provider_name = provider_id.split("/", 1)[0]
            try:
                provider = self.context.get_provider_by_id(provider_name)
            except Exception:
                provider = None

        if not provider:
            return None, f"找不到提供商 ID: {provider_id}"

        p_conf = getattr(provider, "provider_config", {}) or {}

        base_url = (
            getattr(provider, "api_base", "")
            or p_conf.get("api_base")
            or p_conf.get("api_base_url")
            or p_conf.get("base_url")
            or p_conf.get("openai_api_base")
            or ""
        )
        base_url = str(base_url).strip().rstrip("/")

        api_key = ""
        for k in ["key", "keys", "api_key", "access_token", "token"]:
            val = p_conf.get(k)
            if val:
                if isinstance(val, str):
                    api_key = val
                elif isinstance(val, list) and val:
                    api_key = val[0]
                if api_key:
                    break

        model = self._extract_model_from_provider_id(provider_id)
        if not model:
            model = (
                getattr(provider, "model", "")
                or p_conf.get("model")
                or p_conf.get("model_id")
                or ""
            )
        model = str(model).strip()

        api_type = "openai"

        if not base_url:
            return None, f"提供商缺少 base_url: {provider_id}"
        if not api_key:
            return None, f"提供商缺少 api_key: {provider_id}"
        if not model:
            return None, f"提供商缺少 model: {provider_id}"

        rt = ProviderRuntime(
            provider_id=provider_id,
            api_type=api_type,
            base_url=self._normalize_openai_base(base_url),
            api_key=api_key.strip(),
            model=model,
        )
        return rt, None

    # ========= API 调用 =========

    async def _call_generation_api(
        self,
        prompt: str,
        model: str,
        base_url: str,  # .../v1
        api_key: str,
        is_nsfw: bool = False
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        endpoint = "images/generations/nsfw" if is_nsfw else "images/generations"
        api_url = self._build_endpoint(base_url, endpoint)

        payload = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "response_format": "url"
        }

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        timeout_config = httpx.Timeout(connect=20.0, read=self.timeout_seconds, write=60.0, pool=60.0)
        last_error = "未知错误"

        for attempt in range(self.max_retry_attempts):
            try:
                logger.info(f"调用{'NSFW' if is_nsfw else 'Image'} API (模型: {model}, 尝试 {attempt + 1})")
                response = await self.http_client.post(api_url, json=payload, headers=headers, timeout=timeout_config)

                if response.status_code == 200:
                    try:
                        result = response.json()
                        if "data" in result and isinstance(result["data"], list):
                            urls = [item.get("url") for item in result["data"] if item.get("url")]
                            if urls:
                                return urls, None
                        if "url" in result:
                            return [result["url"]], None
                        last_error = f"响应格式无法解析: {str(result)[:120]}"
                    except json.JSONDecodeError:
                        last_error = "JSON解析失败"

                elif response.status_code == 429:
                    last_error = "触发限流 (429)，正在重试..."
                    await asyncio.sleep(2)
                    continue
                else:
                    try:
                        err_json = response.json()
                        err_msg = err_json.get("error", {}).get("message") or err_json.get("error")
                        last_error = f"API错误({response.status_code}): {err_msg}"
                    except Exception:
                        last_error = f"API请求失败({response.status_code})"

            except Exception as e:
                last_error = f"请求异常: {str(e)}"

        return None, last_error

    async def _call_grok_api(
        self,
        prompt: str,
        image_base64: Optional[str],
        model: str,
        base_url: str,  # .../v1
        api_key: str
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        content = [{"type": "text", "text": prompt}]
        if image_base64:
            content.append({"type": "image_url", "image_url": {"url": image_base64}})

        payload = {"model": model, "messages": [{"role": "user", "content": content}]}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        chat_api_url = self._build_endpoint(base_url, "chat/completions")

        timeout_config = httpx.Timeout(
            connect=20.0,
            read=self.timeout_seconds,
            write=60.0,
            pool=self.timeout_seconds + 10
        )
        last_error = "未知错误"

        for attempt in range(self.max_retry_attempts):
            try:
                logger.info(f"调用Grok Chat API (模型: {model}, 尝试 {attempt + 1})")
                response = await self.http_client.post(
                    chat_api_url, json=payload, headers=headers, timeout=timeout_config
                )

                if response.status_code == 200:
                    try:
                        result = response.json()
                        if "error" in result:
                            err_msg = str(result.get("error"))
                            if "internal_server_error" in err_msg or "服务器内部错误" in err_msg:
                                continue
                            last_error = f"服务端错误: {result.get('error')}"
                            continue

                        urls, parse_error = self._extract_media_url_from_response(result)
                        if urls:
                            return urls, None
                        last_error = parse_error or "未找到媒体链接"

                    except json.JSONDecodeError:
                        last_error = "JSON解析失败"

                elif response.status_code == 429:
                    last_error = "触发限流 (429)，正在重试..."
                    continue

                elif response.status_code == 500:
                    error_text = response.text
                    if "429" in error_text:
                        last_error = "上游服务限流 (429)"
                    elif "void *" in error_text or "NoneType" in error_text:
                        continue
                    else:
                        last_error = f"服务端错误(500): {error_text[:100]}"
                else:
                    last_error = f"API请求失败({response.status_code})"

            except Exception as e:
                last_error = f"请求异常: {str(e)}"

        return None, last_error

    def _extract_media_url_from_response(self, response_data: dict) -> Tuple[Optional[List[str]], Optional[str]]:
        try:
            if not isinstance(response_data, dict) or "choices" not in response_data:
                return None, "无效响应"

            choice = response_data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")

            if "video_url" in response_data:
                return [response_data["video_url"]], None
            if "image_url" in response_data:
                img = response_data["image_url"]
                return [img] if isinstance(img, str) else img, None
            if "video_url" in message:
                return [message["video_url"]], None

            if "render_searched_image" in content:
                return None, "Grok 执行了搜索而非生成，请尝试更具体的提示词。"

            urls = []
            urls += re.findall(r"!\[.*?\]\((https?://[^\s<>\"']+)\)", content)
            urls += re.findall(r"""(?:src|href)=["'](https?://[^"']+)["']""", content, re.IGNORECASE)
            urls += re.findall(r"https?://[^\s<>\"')\]]+", content)

            valid_exts = {".mp4", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".mov", ".webm"}
            trusted_domains = ["assets.grok.com", "assets.x.ai", "grok.com", "x.ai"]

            final_urls = []
            for u in urls:
                clean_url = u.rstrip(".,;:]}")
                parsed = urlparse(clean_url)
                if parsed.path.endswith("/"):
                    continue
                if any(parsed.path.lower().endswith(ext) for ext in valid_exts) or \
                   any(d in parsed.netloc for d in trusted_domains):
                    if clean_url not in final_urls:
                        final_urls.append(clean_url)

            return (final_urls, None) if final_urls else (None, "未提取到有效的媒体链接")
        except Exception as e:
            return None, f"提取异常: {e}"

    async def _download_file(self, url: str, base_url: str, api_key: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            if url.endswith("/") or urlparse(url).path.endswith("/"):
                return None, "无效媒体链接(目录URL)"

            parsed = urlparse(url)
            ext = Path(parsed.path).suffix.lower() or ".mp4"
            filename = f"grok_media_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}{ext}"
            file_path = self.data_dir / filename

            headers = {"User-Agent": "Mozilla/5.0 ..."}
            cookies = {}

            is_self_hosted = False
            try:
                if parsed.netloc == urlparse(base_url).netloc:
                    is_self_hosted = True
            except Exception:
                pass

            if is_self_hosted:
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                headers["Referer"] = base_url
            elif "grok.com" in parsed.netloc and api_key and len(api_key) > 50:
                cookies = {"sso": api_key, "sso-rw": api_key}

            async def do_download(target_url):
                response = await self.http_client.get(
                    target_url, headers=headers, cookies=cookies, timeout=httpx.Timeout(300.0)
                )
                response.raise_for_status()

                final_path = file_path
                content_type = response.headers.get("Content-Type", "")
                if content_type:
                    guess_ext = mimetypes.guess_extension(content_type.split(";")[0])
                    if guess_ext and guess_ext != ext and guess_ext not in [".html", ".htm"]:
                        final_path = file_path.with_suffix(guess_ext)

                async with aiofiles.open(final_path, "wb") as f:
                    await f.write(response.content)

                return str(final_path.resolve()), content_type

            try:
                return await do_download(url)
            except Exception as e:
                if is_self_hosted:
                    return None, f"下载失败({e})"
                try:
                    fallback_url = urljoin(base_url + "/", parsed.path.lstrip("/"))
                    return await do_download(fallback_url)
                except Exception:
                    pass
                return None, str(e)

        except Exception as e:
            return None, str(e)

    async def _cleanup_file(self, path: Optional[str]):
        if not path or self.save_video_enabled:
            return
        try:
            p = Path(path)
            if p.exists():
                p.unlink()
        except Exception:
            pass

    async def _try_apply_preset(self, prompt: str, event: AstrMessageEvent) -> Tuple[str, Optional[str], bool]:
        preset_hub = getattr(self.context, "preset_hub", None)
        if not preset_hub:
            if hasattr(self.context, "get_star"):
                preset_hub = self.context.get_star("astrbot_plugin_preset_hub")
            elif hasattr(self.context, "get_plugin"):
                preset_hub = self.context.get_plugin("astrbot_plugin_preset_hub")
        if not preset_hub:
            return prompt, None, False

        try:
            if hasattr(preset_hub, "get_full_prompt"):
                user_id = event.unified_msg_origin
                new_prompt = await preset_hub.get_full_prompt(prompt, user_id) if asyncio.iscoroutinefunction(
                    preset_hub.get_full_prompt
                ) else preset_hub.get_full_prompt(prompt, user_id)
                if new_prompt and new_prompt != prompt:
                    return new_prompt, "系统预设", False

            if hasattr(preset_hub, "get_all_keys"):
                all_keys = preset_hub.get_all_keys()
                all_keys.sort(key=len, reverse=True)
                prompt_lower = prompt.lower()
                for key in all_keys:
                    if prompt_lower == key.lower() or prompt_lower.startswith(key.lower() + " "):
                        preset_val = preset_hub.resolve_preset(key)
                        if preset_val:
                            extra = prompt[len(key):].strip()
                            return (f"{preset_val}, {extra}" if extra else preset_val), key, bool(extra)
        except Exception:
            pass
        return prompt, None, False

    async def _process_task(
        self,
        event: AstrMessageEvent,
        prompt: str,
        task_type: str,
        image_base64: Optional[str] = None,
        show_status: bool = True
    ):
        task_id = str(uuid.uuid4())[:8]
        prompt = prompt.replace("用户：", "").replace("User:", "").strip()

        if task_type in ("image", "edit", "nsfw"):
            prompt, preset_name, _ = await self._try_apply_preset(prompt, event)
        else:
            preset_name = None

        prompt = re.sub(r"\s+", " ", prompt).strip()

        try:
            if task_type == "video":
                action_name = "生成视频"
                icon = "📺"
                final_prompt = f"Animate this image: {prompt}"
            elif task_type == "nsfw":
                action_name = "生成涩图"
                icon = "🔞"
                final_prompt = prompt
            else:
                action_name = "修改图片" if task_type == "edit" else "生成图片"
                icon = "🎨"
                final_prompt = f"generate an image of {prompt}" if task_type == "image" else prompt

            status_msg = f"{icon} 正在{action_name}{f'「预设：{preset_name}」' if preset_name else ''}..."
            if show_status:
                yield event.plain_result(status_msg)

            asyncio.create_task(self._async_core(event, final_prompt, image_base64, task_id, task_type))

        except Exception as e:
            yield event.plain_result(f"❌ 错误: {e}")

    async def _safe_send_chain(self, event: AstrMessageEvent, chain: List):
        try:
            await event.send(event.chain_result(chain))
        except Exception as e:
            msg = str(e)
            if "retcode=1200" in msg or "Timeout" in msg or "ActionFailed" in msg:
                logger.warning("检测到发送超时(retcode=1200)，忽略报错。")
            else:
                logger.error(f"发送异常: {e}，尝试转Base64补救...")
                await self._fallback_send_images(event, chain)

    async def _fallback_send_images(self, event: AstrMessageEvent, chain: List):
        for comp in chain:
            if isinstance(comp, AstrImage) and comp.file and Path(comp.file).exists():
                try:
                    async with aiofiles.open(comp.file, "rb") as f:
                        content = await f.read()
                    b64 = base64.b64encode(content).decode()
                    await event.send(event.chain_result([AstrImage(file=f"base64://{b64}")]))
                    await asyncio.sleep(0.5)
                except Exception:
                    pass

    async def _async_core(
        self,
        event: AstrMessageEvent,
        prompt: str,
        image_base64: Optional[str],
        task_id: str,
        task_type: str
    ):
        _ = task_id
        local_paths = []

        try:
            if task_type == "video":
                provider_id = self.video_provider_id
            elif task_type == "nsfw":
                provider_id = self.nsfw_provider_id
            else:
                provider_id = self.image_provider_id

            runtime, perr = self._parse_provider(provider_id)
            if perr or not runtime:
                await self._reply_error(event, f"❌ {perr}")
                return

            base_url = runtime.base_url
            api_key = runtime.api_key
            model = runtime.model

            if task_type == "nsfw":
                urls, error = await self._call_generation_api(
                    prompt=prompt,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    is_nsfw=True
                )
            else:
                urls, error = await self._call_grok_api(
                    prompt=prompt,
                    image_base64=image_base64,
                    model=model,
                    base_url=base_url,
                    api_key=api_key
                )

            if error:
                await self._reply_error(event, f"❌ {error}")
                return

            if not isinstance(urls, list):
                urls = [urls]

            for u in urls:
                path, _ct = await self._download_file(u, base_url, api_key)
                if path:
                    local_paths.append(path)

            if not local_paths:
                await self._reply_error(event, "⚠️ 资源已生成，但下载失败。")
                return

            if task_type in ("image", "edit", "nsfw"):
                for p in local_paths:
                    ext = Path(p).suffix.lower()
                    if ext in [".mp4", ".mov", ".webm", ".avi", ".mkv"]:
                        await event.send(event.chain_result([Video.fromFileSystem(p)]))
                    else:
                        await self._safe_send_chain(
                            event,
                            [Reply(id=str(event.message_obj.message_id)), AstrImage.fromFileSystem(p)]
                        )
            else:
                video_files = [
                    p for p in local_paths
                    if Path(p).suffix.lower() in [".mp4", ".mov", ".webm", ".avi", ".mkv"]
                ]
                if video_files:
                    for p in video_files:
                        try:
                            await event.send(event.chain_result([Video.fromFileSystem(p)]))
                        except Exception as e:
                            await self._reply_error(event, f"⚠️ 视频发送失败: {e}")
                else:
                    await self._reply_error(event, "⚠️ 生成结果中未包含视频文件。")

        except Exception as e:
            logger.error(f"任务异常: {e}")
            await self._reply_error(event, f"❌ 异常: {e}")
        finally:
            if not self.save_video_enabled:
                await asyncio.sleep(5)
                for p in local_paths:
                    await self._cleanup_file(p)

    @filter.command("视频")
    async def cmd_video_main(self, event: AstrMessageEvent):
        prompt = self._extract_prompt_after_command(event, "视频")
        images = await self._extract_images_from_message(event, crop_for_video=True, target_index=0)
        if not images:
            yield event.plain_result("❌ 视频生成需要提供图片")
            return
        async for res in self._process_task(event, prompt, "video", images[0]):
            yield res

    @filter.regex(r"[\\/!]?视频(\d+)")
    async def cmd_video_repeat(self, event: AstrMessageEvent):
        text = (event.message_str or "").strip()
        m = re.search(r"[\\/!]?视频(\d+)\s*([\s\S]*)", text, re.S)
        if not m:
            return
        count = int(m.group(1))
        prompt = (m.group(2) or "").strip()

        images = await self._extract_images_from_message(event, crop_for_video=True, target_index=0)
        if not images:
            yield event.plain_result("❌ 视频生成需要提供图片")
            return
        async for res in self._run_task_times(event, prompt, "video", images[0], count):
            yield res

    @filter.regex(r"[\\/!]?画图(\d+)")
    async def cmd_image_repeat(self, event: AstrMessageEvent):
        text = (event.message_str or "").strip()
        m = re.search(r"[\\/!]?画图(\d+)\s*([\s\S]*)", text, re.S)
        if not m:
            return
        count = int(m.group(1))
        prompt = (m.group(2) or "").strip()

        images = await self._extract_images_from_message(event, crop_for_video=False, target_index=0)
        if images:
            async for res in self._run_task_times(event, prompt, "edit", images[0], count):
                yield res
        else:
            async for res in self._run_task_times(event, prompt, "image", None, count):
                yield res

    @filter.command("画图")
    async def cmd_image_gen(self, event: AstrMessageEvent):
        prompt = self._extract_prompt_after_command(event, "画图")
        images = await self._extract_images_from_message(event, crop_for_video=False, target_index=0)
        if images:
            async for res in self._process_task(event, prompt, "edit", images[0]):
                yield res
        else:
            async for res in self._process_task(event, prompt, "image", None):
                yield res

    @filter.command("涩图")
    async def cmd_nsfw_gen(self, event: AstrMessageEvent):
        prompt = self._extract_prompt_after_command(event, "涩图")
        async for res in self._process_task(event, prompt, "nsfw", None):
            yield res

    @filter.regex(r"[\\/!]?涩图(\d+)")
    async def cmd_nsfw_repeat(self, event: AstrMessageEvent):
        text = (event.message_str or "").strip()
        m = re.search(r"[\\/!]?涩图(\d+)\s*([\s\S]*)", text, re.S)
        if not m:
            return
        count = int(m.group(1))
        prompt = (m.group(2) or "").strip()

        async for res in self._run_task_times(event, prompt, "nsfw", None, count):
            yield res
