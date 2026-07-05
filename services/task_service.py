import asyncio
import json
import re
from pathlib import Path
from typing import Optional, List, Tuple

from astrbot.api import logger
from astrbot.api.message_components import Video, Image as AstrImage, Reply


class TaskService:
    ALLOWED_SIZES = {"1024x1024", "1024x1792", "1280x720", "1792x1024", "720x1280"}
    LEGACY_VIDEO_DURATIONS = {6, 10, 12, 16, 20}
    XAI_VIDEO_15_MIN_DURATION = 1
    XAI_VIDEO_15_MAX_DURATION = 15
    VIDEO_RATIO_SIZE_MAP = {
        "1:1": "1024x1024",
        "16:9": "1280x720",
        "9:16": "720x1280",
        "2:3": "1024x1792",
        "3:2": "1792x1024",
        "4:3": "1792x1024",
        "3:4": "1024x1792",
    }
    VIDEO_SIZE_RATIO_MAP = {
        "1024x1024": "1:1",
        "1280x720": "16:9",
        "720x1280": "9:16",
        "1024x1792": "2:3",
        "1792x1024": "3:2",
    }

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

    def _select_provider_id(self, task_type: str, image_base64: Optional[str] = None) -> str:
        if task_type == "video":
            if image_base64:
                return self.plugin.video_i2v_provider_id or self.plugin.video_provider_id
            return self.plugin.video_t2v_provider_id or self.plugin.video_provider_id
        if task_type == "edit":
            return self.plugin.image_edit_provider_id
        return self.plugin.image_gen_provider_id

    @staticmethod
    def _select_provider_role(task_type: str, image_base64: Optional[str] = None) -> str:
        if task_type == "video":
            return "i2v" if image_base64 else "t2v"
        if task_type == "edit":
            return "edit"
        return "image"

    @staticmethod
    def _is_grok_41(model: str) -> bool:
        m = str(model or "").strip().lower()
        return "grok-4.1" in m or "grok41" in m

    @staticmethod
    def _is_xai_video_15(model: str) -> bool:
        m = str(model or "").strip().lower()
        return "grok-imagine-video-1.5" in m

    @staticmethod
    def _uses_xai_video_generation_api(model: str) -> bool:
        m = str(model or "").strip().lower()
        return m.startswith("grok-imagine-video")

    @staticmethod
    def _is_image_only_video_model(model: str) -> bool:
        m = str(model or "").strip().lower()
        return "grok-imagine-video-1.5-preview" in m

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
    def _extract_aspect_ratio_for_video(prompt: str, strip_token: bool = True) -> Tuple[str, Optional[str]]:
        text = str(prompt or "")
        aspect_ratio = None

        m = re.search(r"(\d{1,2})\s*[:：]\s*(\d{1,2})", text)
        if m:
            ratio = f"{int(m.group(1))}:{int(m.group(2))}"
            allowed = {"1:1", "2:3", "3:2", "16:9", "9:16", "4:3", "3:4"}
            if ratio in allowed:
                aspect_ratio = ratio
                if strip_token:
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
                    if strip_token:
                        text = re.sub(p, " ", text, flags=re.I, count=1)
                    break

        text = re.sub(r"\s+", " ", text).strip()
        return text, aspect_ratio

    @classmethod
    def _video_size_for_aspect_ratio(cls, aspect_ratio: Optional[str]) -> Optional[str]:
        if not aspect_ratio:
            return None
        return cls.VIDEO_RATIO_SIZE_MAP.get(aspect_ratio)

    @classmethod
    def _extract_video_shape(cls, prompt: str, strip_token: bool = True) -> Tuple[str, Optional[str], Optional[str]]:
        text = str(prompt or "")
        aspect_ratio = None
        video_size = None

        m_size = re.search(r"(?<!\d)(\d{2,5})\s*[xX×]\s*(\d{2,5})(?!\d)", text)
        if m_size:
            candidate = f"{m_size.group(1)}x{m_size.group(2)}"
            mapped_ratio = cls.VIDEO_SIZE_RATIO_MAP.get(candidate)
            if mapped_ratio:
                video_size = candidate
                aspect_ratio = mapped_ratio
                if strip_token:
                    text = text[:m_size.start()] + " " + text[m_size.end():]

        if not aspect_ratio:
            text, aspect_ratio = cls._extract_aspect_ratio_for_video(text, strip_token=strip_token)
            video_size = cls._video_size_for_aspect_ratio(aspect_ratio)

        text = re.sub(r"\s+", " ", text).strip()
        return text, aspect_ratio, video_size

    @staticmethod
    def _extract_video_resolution(prompt: str) -> Tuple[str, Optional[str]]:
        text = str(prompt or "")
        resolution = None

        m = re.search(r"(?<!\d)(480|720|1080)\s*p(?![a-zA-Z0-9])", text, flags=re.I)
        if m:
            resolution = f"{m.group(1)}p".lower()
            text = text[:m.start()] + " " + text[m.end():]

        text = re.sub(r"\s+", " ", text).strip()
        return text, resolution

    @classmethod
    def _extract_duration_for_video(cls, prompt: str) -> Tuple[str, Optional[int]]:
        text = str(prompt or "")
        duration_seconds = None

        m = re.search(
            r"(?<!\d)(\d{1,3})\s*(?:seconds?|secs?|s|秒(?:钟)?)(?![a-zA-Z0-9])",
            text,
            flags=re.I
        )
        if m:
            candidate = int(m.group(1))
            duration_seconds = candidate
            text = text[:m.start()] + " " + text[m.end():]

        text = re.sub(r"\s+", " ", text).strip()
        return text, duration_seconds

    @classmethod
    def _backend_duration_for_video(
        cls,
        duration_seconds: Optional[int],
        use_xai_video_generation_api: bool
    ) -> Tuple[Optional[int], Optional[str]]:
        if not duration_seconds:
            return None, None

        if use_xai_video_generation_api:
            if cls.XAI_VIDEO_15_MIN_DURATION <= duration_seconds <= cls.XAI_VIDEO_15_MAX_DURATION:
                return duration_seconds, None
            return None, (
                "xAI 视频生成接口仅支持 1-15 秒视频时长，"
                f"当前输入: {duration_seconds}s"
            )

        if duration_seconds == 15:
            return 16, None
        if duration_seconds in cls.LEGACY_VIDEO_DURATIONS:
            return duration_seconds, None
        allowed = "6 / 10 / 12 / 16 / 20"
        return None, f"当前视频后端仅支持 {allowed} 秒；15s 会自动按 16s 兼容"

    @staticmethod
    def _should_fallback_to_legacy_video(error: Optional[str]) -> bool:
        if not error:
            return False
        e = str(error).lower()
        return "404" in e or "405" in e or "not found" in e or "method not allowed" in e

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
            provider_id = self._select_provider_id(task_type, image_base64)
            runtime, perr = self.provider_resolver.parse(provider_id)
            if perr or not runtime:
                await self.send_service.reply_error(event, f"❌ {perr}")
                return

            resp = None
            urls = None
            perr = None

            if task_type == "image":
                # 文生图按模型路由
                if self._is_grok_41(runtime.model):
                    logger.info(f"任务路由: task_type=image, api=chat/completions, model={runtime.model}")

                    resp, error = await self.api_client.call_chat(
                        prompt=prompt,
                        image_base64=None,
                        model=runtime.model,
                        base_url=runtime.base_url,
                        api_key=runtime.api_key,
                        aspect_ratio=None
                    )
                    if error:
                        await self.send_service.reply_error(event, f"❌ {error}")
                        return

                    if self._is_tool_usage_card_response(resp):
                        await self.send_service.reply_error(
                            event,
                            "❌ 当前模型返回了工具调用卡片（未实际生成媒体）。\n请更换生图模型提供商。"
                        )
                        return

                    urls, perr = self.media_service.extract_media_url_from_chat_response(resp)

                else:
                    # 默认走 generation（grok-imagine 系列）
                    prompt_clean, gen_size, invalid_size = self._extract_size_for_image(prompt)

                    if invalid_size:
                        await self.send_service.reply_error(event, self._unsupported_size_message(invalid_size))
                        return

                    if not gen_size:
                        gen_size = "1024x1792"

                    logger.info(
                        f"任务路由: task_type=image, api=images/generations, "
                        f"model={runtime.model}, size={gen_size}"
                    )

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

                # 图生图统一去掉比例/尺寸词，不改原图比例
                edit_prompt_clean = self._strip_ratio_or_size_tokens(prompt)

                if self._is_grok_41(runtime.model):
                    # grok-4.1 图生图走 chat/completions（携带参考图）
                    logger.info(
                        f"任务路由: task_type=edit, api=chat/completions, "
                        f"model={runtime.model}, mode=i2i(chat)"
                    )

                    resp, error = await self.api_client.call_chat(
                        prompt=edit_prompt_clean,
                        image_base64=image_base64,
                        model=runtime.model,
                        base_url=runtime.base_url,
                        api_key=runtime.api_key,
                        aspect_ratio=None
                    )
                    if error:
                        await self.send_service.reply_error(event, f"❌ {error}")
                        return

                    if self._is_tool_usage_card_response(resp):
                        await self.send_service.reply_error(
                            event,
                            "❌ 当前模型返回了工具调用卡片（未实际生成媒体）。\n请更换图生图模型提供商。"
                        )
                        return

                    urls, perr = self.media_service.extract_media_url_from_chat_response(resp)

                else:
                    # grok-imagine 图生图走 images/edits
                    logger.info(
                        f"任务路由: task_type=edit, api=images/edits, "
                        f"model={runtime.model}, size=follow-source"
                    )

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
                video_duration_seconds = None
                backend_duration_seconds = None
                video_size = None
                video_resolution = None
                is_xai_video_15 = self._is_xai_video_15(runtime.model)
                use_xai_video_generation_api = self._uses_xai_video_generation_api(runtime.model)
                is_i2v_only_video_model = self._is_image_only_video_model(runtime.model)

                if is_i2v_only_video_model and not image_base64:
                    await self.send_service.reply_error(
                        event,
                        "❌ 当前视频模型仅支持图生视频。请发送/引用图片，或在配置中为文生视频选择 grok-imagine-video。"
                    )
                    return

                video_prompt, video_duration_seconds = self._extract_duration_for_video(video_prompt)
                video_prompt, video_resolution = self._extract_video_resolution(video_prompt)
                backend_duration_seconds, duration_error = self._backend_duration_for_video(
                    video_duration_seconds,
                    use_xai_video_generation_api
                )
                if duration_error:
                    await self.send_service.reply_error(event, f"❌ {duration_error}")
                    return

                video_prompt, video_aspect_ratio, video_size = self._extract_video_shape(
                    video_prompt,
                    strip_token=True
                )

                logger.info(
                    f"[task.video] input_prompt={prompt!r}, parsed_prompt={video_prompt!r}, "
                    f"aspect_ratio={video_aspect_ratio or 'default'}, video_size={video_size or 'default'}, "
                    f"resolution={video_resolution or 'default'}, "
                    f"duration_requested={video_duration_seconds or 'default'}, "
                    f"duration_backend={backend_duration_seconds or 'default'}, "
                    f"xai_video_15={is_xai_video_15}, "
                    f"xai_video_api={use_xai_video_generation_api}, "
                    f"i2v_only={is_i2v_only_video_model}"
                )
                logger.info(
                    f"任务路由: task_type=video, model={runtime.model}, "
                    f"mode={'i2v' if image_base64 else 't2v'}, "
                    f"aspect_ratio={video_aspect_ratio or 'default'}, video_size={video_size or 'default'}, "
                    f"resolution={video_resolution or 'default'}, "
                    f"duration_requested={video_duration_seconds or 'default'}, "
                    f"duration_backend={backend_duration_seconds or 'default'}, "
                    f"api={'videos/generations' if use_xai_video_generation_api else 'chat/completions'}"
                )

                if use_xai_video_generation_api:
                    resp, error = await self.api_client.call_video_generation(
                        prompt=video_prompt,
                        image_base64=image_base64,
                        model=runtime.model,
                        base_url=runtime.base_url,
                        api_key=runtime.api_key,
                        duration_seconds=backend_duration_seconds,
                        aspect_ratio=video_aspect_ratio,
                        resolution=video_resolution
                    )

                    if error and self._should_fallback_to_legacy_video(error):
                        logger.warning(
                            f"[task.video] videos/generations 不可用，尝试回退 chat/completions: {error}"
                        )
                        legacy_duration_seconds, legacy_duration_error = self._backend_duration_for_video(
                            video_duration_seconds,
                            False
                        )
                        if legacy_duration_error:
                            await self.send_service.reply_error(
                                event,
                                f"❌ xAI 视频接口不可用（{error}），且旧视频链路不支持该时长：{legacy_duration_error}"
                            )
                            return
                        resp, error = await self.api_client.call_chat(
                            prompt=video_prompt,
                            image_base64=image_base64,
                            model=runtime.model,
                            base_url=runtime.base_url,
                            api_key=runtime.api_key,
                            aspect_ratio=video_aspect_ratio,
                            duration_seconds=legacy_duration_seconds,
                            video_size=video_size
                        )
                else:
                    resp, error = await self.api_client.call_chat(
                        prompt=video_prompt,
                        image_base64=image_base64,
                        model=runtime.model,
                        base_url=runtime.base_url,
                        api_key=runtime.api_key,
                        aspect_ratio=video_aspect_ratio,
                        duration_seconds=backend_duration_seconds,
                        video_size=video_size
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
