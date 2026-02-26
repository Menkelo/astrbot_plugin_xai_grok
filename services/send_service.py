import base64
from pathlib import Path
from typing import List

import aiofiles
from astrbot.api import logger
from astrbot.api.message_components import Image as AstrImage, Reply, Plain


class SendService:
    def __init__(self, save_video_enabled_getter):
        """
        save_video_enabled_getter: callable，返回当前是否保留文件
        """
        self._save_video_enabled_getter = save_video_enabled_getter

    async def reply_error(self, event, text: str):
        try:
            await event.send(event.chain_result([
                Reply(id=str(event.message_obj.message_id)),
                Plain(text)
            ]))
        except Exception:
            await event.send(event.plain_result(text))

    async def safe_send_chain(self, event, chain: List):
        try:
            await event.send(event.chain_result(chain))
        except Exception as e:
            msg = str(e)
            if "retcode=1200" in msg or "Timeout" in msg or "ActionFailed" in msg:
                logger.warning("检测到发送超时(retcode=1200)，忽略报错。")
            else:
                logger.error(f"发送异常: {e}，尝试转Base64补救...")
                await self.fallback_send_images(event, chain)

    async def fallback_send_images(self, event, chain: List):
        for comp in chain:
            if isinstance(comp, AstrImage) and comp.file and Path(comp.file).exists():
                try:
                    async with aiofiles.open(comp.file, "rb") as f:
                        content = await f.read()
                    b64 = base64.b64encode(content).decode()
                    await event.send(event.chain_result([AstrImage(file=f"base64://{b64}")]))
                except Exception:
                    pass