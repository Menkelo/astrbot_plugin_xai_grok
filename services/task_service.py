import asyncio
import json
import re
from pathlib import Path
from typing import Optional, List, Tuple

from astrbot.api import logger
from astrbot.api.message_components import Video, Image as AstrImage, Reply


class TaskService:
    ALLOWED_SIZES = {"1024x1024", "1024x1792", "1280x720", "1792x1024", "720x1280"}

    def __init__(self, plugin, provider_resolver, api_client, media_service, send_service):
        self.plugin = plugin
        self.provider_resolver = provider_resolver
        self.api_client = api_client
        self.media_service = media_service
        self.send_service = send_service

        # 下载重试配置（可在插件 config 里覆盖）
        conf = getattr(plugin, "config", {}) or {}
        self.download_retry_attempts = int(conf.get("download_retry_attempts", 3))
        self.download_retry_delay_seconds = float(conf.get("download_retry_delay_seconds", 1.2))
        if self.download_retry_attempts < 1:
            self.download_retry_attempts = 1
        if self.download_retry_delay_seconds <= 0:
            self.download_retry_delay_seconds = 1.0

    def _select_provider_id(self, task_type: str) -> str:
        if task_type == "video":
            return self.plugin.video_provider_id
        if task_type == "edit":
            return self.plugin.image_edit_provider_id
        return self.plugin.image_gen_provider_id

    @staticmethod
    def _is_tool_usage_card_response(resp: dict) -> bool:
        try:
            if not isinstance(resp, dict):
                return False
            choices = resp.get("choices", [])
            if not isinstance(choices, list) or not choices:
                return False
            msg = (choices[0] or {}).get("message", {}) or {}
            content = msg.get("content", "")
            return isinstance(content, str) and "<xai:tool_usage_card>" in content
        except Exception:
            return False

    @staticmethod
    def _unsupported_size_message(invalid_size: str) -> str:
        return (
            f"❌ 尺寸不支持: {invalid_size}\n"
            "支持比例: 1:1 / 2:3 / 16:9 / 3:2 / 9:16\n"
            "对应尺寸: 1024x1024 / 1024x1792 / 1280x720 / 1792x1024 / 720x1280"
        )

    def _extract_size_for_image(self, prompt: str) -> Tuple[str, Optional[str], Optional[str]]:
        if not prompt:
            return "", None, None

        text = prompt
        size = None
        invalid_size = None

        # 显式尺寸
        m_size = re.search(r"(?<!\d)(\d{2,5})\s*[xX×]\s*(\d{2,5})(?!\d)", text)
        if m_size:
            candidate = f"{m_size.group(1)}x{m_size.group(2)}"
            text = text[:m_size.start()] + " " + text[m_size.end():]
            if candidate in self.ALLOWED_SIZES:
                size = candidate
            else:
                invalid_size = candidate

        # 比例映射尺寸
        if not size and not invalid_size:
            m_ratio = re.search(r"(\d{1,2})\s*[:：]\s*(\d{1,2})", text)
            if m_ratio:
                ratio = f"{int(m_ratio.group(1))}:{int(m_ratio.group(2))}"
                ratio_map = {
                    "1:1": "1024x1024",
                    "2:3": "1024x1792",
                    "3:2": "1792x1024",
                    "16:9": "1280x720",
                    "9:16": "720x1280",
                }
                if ratio in ratio_map:
                    size = ratio_map[ratio]
                    text = text[:m_ratio.start()] + " " + text[m_ratio.end():]

        # 中文别名
        if not size and not invalid_size:
            alias_map = [
                (r"方图|方形", "1024x1024"),
                (r"竖图|竖屏", "1024x1792"),
                (r"横图|横屏", "1792x1024"),
                (r"宽屏", "1280x720"),
                (r"长竖图", "720x1280"),
            ]
            for p, mapped in alias_map:
                m = re.search(p, text, flags=re.I)
                if m:
                    size = mapped
                    text = re.sub(p, " ", text, flags=re.I, count=1)
                    break

        text = re.sub(r"\s+", " ", text).strip()
        return text, size, invalid_size

    @staticmethod
    def _strip_ratio_or_size_tokens(text: str) -> str:
        if not text:
            return ""
        s = text
        s = re.sub(r"(?<!\d)\d{1,2}\s*[:：]\s*\d{1,2}(?!\d)", " ", s)       # ratio
        s = re.sub(r"(?<!\d)\d{2,5}\s*[xX×]\s*\d{2,5}(?!\d)", " ", s)        # size
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _extract_aspect_ratio_for_video(prompt: str) -> Tuple[str, Optional[str]]:
        text = str(prompt or "")
        aspect_ratio = None

        m = re.search(r"(\d{1,2})\s*[:：]\s*(\d{1,2})", text)
        if m:
            ratio = f"{int(m.group(1))}:{int(m.group(2))}"
            allowed = {"1:1", "2:3", "3:2", "16:9", "9:16", "4:3", "3:4"}
            if ratio in allowed:
                aspect_ratio = ratio
                text = text[:m.start()] + " " + text[m.end():]

        if not aspect_ratio:
            alias_map = [
                (r"横屏|宽屏", "16:9"),
                (r"竖屏|竖图", "9:16"),
                (r"方图|方形", "1:1"),
            ]
            for p, ar in alias_map:
                mm = re.search(p, text, flags=re.I)
                if mm:
                    aspect_ratio = ar
                    text = re.sub(p, " ", text, flags=re.I, count=1)
                    break

        text = re.sub(r"\s+", " ", text).strip()
        return text, aspect_ratio

    async def _download_with_retry(self, url: str, base_url: str, api_key: str) -> Tuple[Optional[str], Optional[str]]:
        last_err = None
        for i in range(self.download_retry_attempts):
            path, err = await self.media_service.download_file(url, base_url, api_key)
            if path:
                if i > 0:
                    logger.info(f"[下载重试] 第 {i + 1} 次成功: {url}")
                return path, None

            last_err = err or "未知错误"
            if i < self.download_retry_attempts - 1:
                wait = self.download_retry_delay_seconds * (2 ** i)
                logger.warning(
                    f"[下载重试] 第 {i + 1} 次失败: {url} | err={last_err} | {wait:.1f}s 后重试"
                )
                await asyncio.sleep(wait)

        return None, last_err

    async def run_async_core(
        self,
        event,
        prompt: str,
        image_base64: Optional[str],
        task_id: str,
        task_type: str
    ):
        _ = task_id
        local_paths: List[str] = []

        try:
            provider_id = self._select_provider_id(task_type)
            runtime, perr = self.provider_resolver.parse(provider_id)
            if perr or not runtime:
                await self.send_service.reply_error(event, f"❌ {perr}")
                return

            resp = None
            urls = None
            perr = None

            if task_type == "image":
                prompt_clean, gen_size, invalid_size = self._extract_size_for_image(prompt)

                if invalid_size:
                    await self.send_service.reply_error(event, self._unsupported_size_message(invalid_size))
                    return

                if not gen_size:
                    gen_size = "1024x1792"

                logger.info(f"任务路由: task_type=image, model={runtime.model}, size={gen_size}")

                resp, error = await self.api_client.call_generation(
                    prompt=prompt_clean,
                    model=runtime.model,
                    base_url=runtime.base_url,
                    api_key=runtime.api_key,
                    size=gen_size
                )
                if error:
                    await self.send_service.reply_error(event, f"❌ {error}")
                    return
                urls, perr = self.media_service.extract_media_url_from_generation_response(resp)

            elif task_type == "edit":
                if not image_base64:
                    await self.send_service.reply_error(event, "❌ 图生图需要提供参考图片")
                    return

                # 图生图不改比例
                edit_prompt_clean = self._strip_ratio_or_size_tokens(prompt)
                logger.info(f"任务路由: task_type=edit, model={runtime.model}, size=follow-source")

                resp, error = await self.api_client.call_image_edit(
                    prompt=edit_prompt_clean,
                    image_base64=image_base64,
                    model=runtime.model,
                    base_url=runtime.base_url,
                    api_key=runtime.api_key
                )
                if error:
                    await self.send_service.reply_error(event, f"❌ {error}")
                    return
                urls, perr = self.media_service.extract_media_url_from_generation_response(resp)

            elif task_type == "video":
                video_prompt = prompt
                video_aspect_ratio = None

                if not image_base64:
                    video_prompt, video_aspect_ratio = self._extract_aspect_ratio_for_video(prompt)

                logger.info(
                    f"[task.video] input_prompt={prompt!r}, parsed_prompt={video_prompt!r}, "
                    f"aspect_ratio={video_aspect_ratio or 'default'}"
                )
                logger.info(
                    f"任务路由: task_type=video, model={runtime.model}, "
                    f"mode={'i2v' if image_base64 else 't2v'}, aspect_ratio={video_aspect_ratio or 'default'}"
                )

                resp, error = await self.api_client.call_chat(
                    prompt=video_prompt,
                    image_base64=image_base64,
                    model=runtime.model,
                    base_url=runtime.base_url,
                    api_key=runtime.api_key,
                    aspect_ratio=video_aspect_ratio
                )
                if error:
                    await self.send_service.reply_error(event, f"❌ {error}")
                    return

                if self._is_tool_usage_card_response(resp):
                    await self.send_service.reply_error(
                        event,
                        "❌ 当前模型返回了工具调用卡片（未实际生成媒体）。\n请更换视频模型提供商。"
                    )
                    return

                urls, perr = self.media_service.extract_media_url_from_chat_response(resp)

            else:
                await self.send_service.reply_error(event, f"❌ 不支持的任务类型: {task_type}")
                return

            if perr:
                try:
                    logger.warning(f"媒体提取失败，原始响应片段: {json.dumps(resp, ensure_ascii=False)[:1600]}")
                except Exception:
                    logger.warning(f"媒体提取失败，原始响应(非JSON): {str(resp)[:1600]}")
                await self.send_service.reply_error(event, f"❌ {perr}")
                return

            if not isinstance(urls, list):
                urls = [urls]
            urls = [u for u in urls if u]

            if not urls:
                await self.send_service.reply_error(event, "❌ 生成结果为空（未返回可下载资源）")
                return

            failed_urls = []
            for u in urls:
                path, err = await self._download_with_retry(u, runtime.base_url, runtime.api_key)
                if path:
                    local_paths.append(path)
                else:
                    failed_urls.append((u, err))

            if not local_paths:
                detail = failed_urls[0][1] if failed_urls else "未知错误"
                await self.send_service.reply_error(event, f"⚠️ 资源已生成，但下载失败（已重试）: {detail}")
                return

            if failed_urls:
                logger.warning(f"部分资源下载失败: {failed_urls[:3]}")

            if task_type in ("image", "edit"):
                for p in local_paths:
                    ext = Path(p).suffix.lower()
                    if ext in self.media_service.VIDEO_EXTS:
                        await event.send(event.chain_result([Video.fromFileSystem(p)]))
                    else:
                        await self.send_service.safe_send_chain(
                            event,
                            [Reply(id=str(event.message_obj.message_id)), AstrImage.fromFileSystem(p)]
                        )
            else:
                video_files = [p for p in local_paths if Path(p).suffix.lower() in self.media_service.VIDEO_EXTS]
                if video_files:
                    for p in video_files:
                        try:
                            await event.send(event.chain_result([Video.fromFileSystem(p)]))
                        except Exception as e:
                            await self.send_service.reply_error(event, f"⚠️ 视频发送失败: {e}")
                else:
                    await self.send_service.reply_error(event, "⚠️ 生成结果中未包含视频文件。")

        except Exception as e:
            logger.error(f"任务异常: {e}")
            await self.send_service.reply_error(event, f"❌ 异常: {e}")
        finally:
            if not self.plugin.save_video_enabled:
                await asyncio.sleep(5)
                for p in local_paths:
                    await self.media_service.cleanup_file(p)