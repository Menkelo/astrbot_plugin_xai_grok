from pathlib import Path

import httpx
from astrbot.api import logger
from astrbot.api.all import *
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools

from .utils.text_utils import extract_prompt_from_components
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

        # Provider 配置（图片 / 视频 两个槽位，兼容旧字段回退）
        self.image_provider_id = config.get("image_provider_id", "") or (
            config.get("image_gen_provider_id", "")
            or config.get("image_edit_provider_id", "")
        )
        self.video_provider_id = config.get("video_provider_id", "") or (
            config.get("video_t2v_provider_id", "")
            or config.get("video_i2v_provider_id", "")
        )

        # 视频默认时长（秒），配置面板滑动条控制，未配置时使用后端默认
        try:
            self.video_default_duration = int(config.get("video_default_duration", 0) or 0)
        except (TypeError, ValueError):
            self.video_default_duration = 0
        if self.video_default_duration < 1:
            self.video_default_duration = 0

        # 视频默认格式（480p/720p/1080p），配置面板下拉控制，未配置时后端默认（720p）
        try:
            self.video_default_resolution = str(config.get("video_default_resolution", "") or "").strip().lower()
        except Exception:
            self.video_default_resolution = ""
        if self.video_default_resolution not in ("480p", "720p", "1080p"):
            self.video_default_resolution = ""

        self.timeout_seconds = 180
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
        self.api_client = ApiClient(self.http_client, self.timeout_seconds)
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
            f"image={self.image_provider_id or '-'}, "
            f"video={self.video_provider_id or '-'}, "
            f"video_default_duration={self.video_default_duration or 'default'}, "
            f"video_default_resolution={self.video_default_resolution or 'default'}"
        )

    async def on_unload(self):
        if getattr(self, "http_client", None):
            await self.http_client.aclose()

    @filter.command("视频")
    async def cmd_video_main(self, event: AstrMessageEvent, *, prompt: str = ""):
        # 关键修复：强制从原始消息提取，避免框架参数吞掉 “1:1”
        # 从消息组件链取 Plain 文本，避免 @用户 被渲染为 " @昵称(QQ号) " 混入提示词
        prompt = extract_prompt_from_components(event.message_obj.message, "视频")

        if not (prompt or "").strip():
            yield event.plain_result("❌ 请输入视频提示词，例如：/视频 一只猫在跑步 1:1")
            return

        target_aspect_ratio = None
        _, video_aspect_ratio, video_size = self.task_service._extract_video_shape(
            prompt,
            strip_token=False
        )
        if video_aspect_ratio and ":" in video_aspect_ratio:
            try:
                w, h = video_aspect_ratio.split(":", 1)
                target_aspect_ratio = float(w) / float(h)
            except Exception:
                target_aspect_ratio = None
        if target_aspect_ratio:
            logger.info(
                f"[瑙嗛] target_aspect_ratio={video_aspect_ratio}, "
                f"video_size={video_size or 'default'}"
            )

        images = await self.image_service.extract_images_from_message(
            event,
            crop_for_video=True,
            target_index=0,
            target_aspect_ratio=target_aspect_ratio
        )
        image_ref = images[0] if images else None
        image_base64 = (image_ref or {}).get("b64")
        image_url = (image_ref or {}).get("url")

        if image_base64 or image_url:
            logger.info("[视频] 检测到参考图，走图生视频")
        else:
            logger.info("[视频] 未检测到参考图，走文生视频")

        logger.info(f"[视频] final_prompt={prompt!r}")

        async for res in self.orchestrator.start_once(
            event=event,
            prompt=prompt,
            task_type="video",
            image_base64=image_base64,
            image_url=image_url,
            show_status=True
        ):
            yield res

    @filter.command("grok")
    async def cmd_image_gen(self, event: AstrMessageEvent, *, prompt: str = ""):
        # 从消息组件链取 Plain 文本，避免 @用户 被渲染为 " @昵称(QQ号) " 混入提示词
        prompt = extract_prompt_from_components(event.message_obj.message, "grok")

        if not (prompt or "").strip():
            yield event.plain_result("❌ 请输入图片提示词，例如：/grok 一只猫 1:1")
            return

        images = await self.image_service.extract_images_from_message(
            event, crop_for_video=False, target_index=0
        )
        image_ref = images[0] if images else None
        image_base64 = (image_ref or {}).get("b64")
        image_url = (image_ref or {}).get("url")

        if image_base64 or image_url:
            # 图生图 -> /v1/images/edits 或 chat/completions
            async for res in self.orchestrator.start_once(
                event=event,
                prompt=prompt,
                task_type="edit",
                image_base64=image_base64,
                image_url=image_url,
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
