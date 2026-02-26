import re
from pathlib import Path

import httpx
from astrbot.api import logger
from astrbot.api.all import *
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools

from .utils.text_utils import extract_prompt_after_command
from .services.provider_resolver import ProviderResolver
from .services.api_client import ApiClient
from .services.image_service import ImageService
from .services.media_service import MediaService
from .services.preset_service import PresetService
from .services.send_service import SendService
from .services.task_service import TaskService
from .services.orchestrator_service import OrchestratorService


class GrokMediaPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config

        # Provider 配置
        self.video_provider_id = config.get("video_provider_id", "")
        self.image_gen_provider_id = config.get("image_gen_provider_id", "")
        self.image_edit_provider_id = config.get("image_edit_provider_id", "")

        self.timeout_seconds = 180
        self.max_retry_attempts = 3
        self.max_image_size = 5 * 1024 * 1024
        self.save_video_enabled = False

        try:
            plugin_data_dir = Path(StarTools.get_data_dir("astrbot_plugin_xai_grok"))
            self.data_dir = (plugin_data_dir / "downloads").resolve()
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"无法使用StarTools数据目录: {e}")
            self.data_dir = (Path(__file__).parent / "downloads").resolve()
            self.data_dir.mkdir(parents=True, exist_ok=True)

        self.http_client = httpx.AsyncClient(follow_redirects=True)

        # services
        self.provider_resolver = ProviderResolver(self.context)
        self.api_client = ApiClient(self.http_client, self.timeout_seconds, self.max_retry_attempts)
        self.image_service = ImageService(self.http_client, self.context, self.max_image_size)
        self.media_service = MediaService(self.http_client, self.data_dir)
        self.preset_service = PresetService(self.context)
        self.send_service = SendService(lambda: self.save_video_enabled)

        self.task_service = TaskService(
            plugin=self,
            provider_resolver=self.provider_resolver,
            api_client=self.api_client,
            media_service=self.media_service,
            send_service=self.send_service
        )
        self.orchestrator = OrchestratorService(
            plugin=self,
            preset_service=self.preset_service,
            task_service=self.task_service
        )

        logger.info(
            "Grok-Imagine已初始化: "
            f"video={self.video_provider_id or '-'}, "
            f"image_gen={self.image_gen_provider_id or '-'}, "
            f"image_edit={self.image_edit_provider_id or '-'}"
        )

    async def on_unload(self):
        if getattr(self, "http_client", None):
            await self.http_client.aclose()

    @filter.command("视频")
    async def cmd_video_main(self, event: AstrMessageEvent, *, prompt: str = ""):
        # 关键修复：强制从原始消息提取，避免框架参数吞掉 “1:1”
        prompt = extract_prompt_after_command(event.message_str, "视频")

        if not (prompt or "").strip():
            yield event.plain_result("❌ 请输入视频提示词，例如：/视频 一只猫在跑步 1:1")
            return

        images = await self.image_service.extract_images_from_message(
            event, crop_for_video=True, target_index=0
        )
        image_base64 = images[0] if images else None

        if image_base64:
            logger.info("[视频] 检测到参考图，走图生视频")
        else:
            logger.info("[视频] 未检测到参考图，走文生视频")

        logger.info(f"[视频] final_prompt={prompt!r}")

        async for res in self.orchestrator.start_once(
            event=event,
            prompt=prompt,
            task_type="video",
            image_base64=image_base64,
            show_status=True
        ):
            yield res

    @filter.regex(r"[\\/!]?视频(\d+)")
    async def cmd_video_repeat(self, event: AstrMessageEvent):
        count, prompt = self.orchestrator.parse_repeat_command(event.message_str, "视频")
        if not count:
            return

        # 兜底提取，避免某些平台 regex 内容丢失
        if not (prompt or "").strip():
            prompt = extract_prompt_after_command(event.message_str, "视频")

        if not (prompt or "").strip():
            yield event.plain_result("❌ 请输入视频提示词，例如：/视频3 一只猫在跑步 1:1")
            return

        images = await self.image_service.extract_images_from_message(
            event, crop_for_video=True, target_index=0
        )
        image_base64 = images[0] if images else None

        if image_base64:
            logger.info(f"[视频批量] 检测到参考图，次数={count}，走图生视频")
        else:
            logger.info(f"[视频批量] 未检测到参考图，次数={count}，走文生视频")

        logger.info(f"[视频批量] final_prompt={prompt!r}")

        async for res in self.orchestrator.start_repeat(
            event=event,
            prompt=prompt,
            task_type="video",
            image_base64=image_base64,
            times=count
        ):
            yield res

    @filter.command("画图")
    async def cmd_image_gen(self, event: AstrMessageEvent, *, prompt: str = ""):
        prompt = extract_prompt_after_command(event.message_str, "画图")

        if not (prompt or "").strip():
            yield event.plain_result("❌ 请输入图片提示词，例如：/画图 一只猫 1:1")
            return

        images = await self.image_service.extract_images_from_message(
            event, crop_for_video=False, target_index=0
        )
        if images:
            # 图生图 -> /v1/images/edits
            async for res in self.orchestrator.start_once(
                event=event,
                prompt=prompt,
                task_type="edit",
                image_base64=images[0],
                show_status=True
            ):
                yield res
        else:
            # 文生图 -> /v1/images/generations
            async for res in self.orchestrator.start_once(
                event=event,
                prompt=prompt,
                task_type="image",
                image_base64=None,
                show_status=True
            ):
                yield res

    @filter.regex(r"[\\/!]?画图(\d+)")
    async def cmd_image_repeat(self, event: AstrMessageEvent):
        count, prompt = self.orchestrator.parse_repeat_command(event.message_str, "画图")
        if not count:
            return

        if not (prompt or "").strip():
            prompt = extract_prompt_after_command(event.message_str, "画图")

        if not (prompt or "").strip():
            yield event.plain_result("❌ 请输入图片提示词，例如：/画图3 一只猫 1:1")
            return

        images = await self.image_service.extract_images_from_message(
            event, crop_for_video=False, target_index=0
        )
        if images:
            async for res in self.orchestrator.start_repeat(
                event=event,
                prompt=prompt,
                task_type="edit",
                image_base64=images[0],
                times=count
            ):
                yield res
        else:
            async for res in self.orchestrator.start_repeat(
                event=event,
                prompt=prompt,
                task_type="image",
                image_base64=None,
                times=count
            ):
                yield res
