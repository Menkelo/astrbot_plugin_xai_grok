import asyncio
import re
import uuid
from typing import Optional, AsyncGenerator

from astrbot.api import logger
from ..utils.text_utils import normalize_spaces


class OrchestratorService:
    def __init__(self, plugin, preset_service, task_service):
        self.plugin = plugin
        self.preset_service = preset_service
        self.task_service = task_service

    @staticmethod
    def _detect_ratio_token(text: str) -> Optional[str]:
        """
        提取比例标记，仅用于状态展示
        支持：1:1 / 2:3 / 3:2 / 16:9 / 9:16 / 4:3 / 3:4（含中文冒号）
        """
        s = str(text or "")
        m = re.search(r"(\d{1,2})\s*[:：]\s*(\d{1,2})", s)
        if not m:
            return None
        ratio = f"{int(m.group(1))}:{int(m.group(2))}"
        allowed = {"1:1", "2:3", "3:2", "16:9", "9:16", "4:3", "3:4"}
        return ratio if ratio in allowed else None

    @staticmethod
    def _detect_video_duration_token(text: str) -> Optional[int]:
        """
        提取视频时长标记，仅用于状态展示。
        """
        s = str(text or "")
        m = re.search(
            r"(?<!\d)(\d{1,3})\s*(?:seconds?|secs?|s|秒(?:钟)?)(?![a-zA-Z0-9])",
            s,
            flags=re.I
        )
        if not m:
            return None
        duration = int(m.group(1))
        return duration if duration == 15 else None

    async def start_once(
        self,
        event,
        prompt: str,
        task_type: str,
        image_base64: Optional[str] = None,
        show_status: bool = True
    ) -> AsyncGenerator:
        prompt = (prompt or "").replace("用户：", "").replace("User:", "").strip()

        detected_ratio = self._detect_ratio_token(prompt)
        detected_duration = self._detect_video_duration_token(prompt) if task_type == "video" else None
        logger.info(
            f"[orchestrator] raw_prompt={prompt!r}, detected_ratio={detected_ratio}, "
            f"detected_duration={detected_duration}"
        )

        # 预设逻辑保持：仅 edit 可用
        if task_type == "edit":
            prompt, preset_name, has_extra = await self.preset_service.apply(prompt, event)
        else:
            preset_name, has_extra = None, False

        prompt = normalize_spaces(prompt)
        task_id = str(uuid.uuid4())[:8]

        if task_type == "video":
            final_prompt = f"Animate this image: {prompt}" if image_base64 else prompt
        else:
            final_prompt = f"generate an image of {prompt}" if task_type == "image" else prompt

        if show_status:
            if task_type == "video":
                status_msg = "📺 正在生成视频"
            elif task_type == "edit":
                status_msg = "🎨 正在修改图片"
            else:
                status_msg = "🎨 正在生成图片"

            if preset_name:
                status_msg += f"「预设：{preset_name}」"
            if has_extra:
                status_msg += "(已衔接额外提示词)"

            # 文生图显示比例标签
            if task_type == "image" and detected_ratio:
                status_msg += f" [{detected_ratio}]"
            # 文生视频显示比例标签（图生视频不显示）
            if task_type == "video" and (not image_base64) and detected_ratio:
                status_msg += f" [{detected_ratio}]"
            if task_type == "video" and detected_duration:
                status_msg += f" [{detected_duration}s]"

            status_msg += "..."
            yield event.plain_result(status_msg)

        asyncio.create_task(
            self.task_service.run_async_core(
                event=event,
                prompt=final_prompt,
                image_base64=image_base64,
                task_id=task_id,
                task_type=task_type
            )
        )
