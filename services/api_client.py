import asyncio
import json
import base64
import re
from typing import Optional, Tuple

import httpx
from astrbot.api import logger


class ApiClient:
    def __init__(self, http_client: httpx.AsyncClient, timeout_seconds=180, max_retry_attempts=3):
        self.http_client = http_client
        self.timeout_seconds = timeout_seconds
        self.max_retry_attempts = max_retry_attempts

    @staticmethod
    def endpoint(base_v1: str, path_after_v1: str) -> str:
        return f"{base_v1.rstrip('/')}/{path_after_v1.lstrip('/')}"

    @staticmethod
    def _decode_data_url(image_base64: str) -> tuple[Optional[bytes], str]:
        if not image_base64:
            return None, "image/jpeg"

        s = str(image_base64).strip()
        mime = "image/jpeg"

        try:
            if s.startswith("data:"):
                m = re.match(r"^data:([^;]+);base64,(.*)$", s, re.I | re.S)
                if not m:
                    return None, mime
                mime = m.group(1).lower().strip()
                data = m.group(2).strip()
            else:
                data = s

            try:
                raw = base64.b64decode(data)
            except Exception:
                data = re.sub(r"[^a-zA-Z0-9+/=]", "", data)
                raw = base64.b64decode(data)

            return raw, mime
        except Exception:
            return None, mime

    async def call_chat(
        self,
        prompt: str,
        image_base64: Optional[str],
        model: str,
        base_url: str,
        api_key: str,
        aspect_ratio: Optional[str] = None
    ) -> Tuple[Optional[dict], Optional[str]]:
        """
        用于视频等 chat/completions
        文生视频支持 aspect_ratio
        """
        url = self.endpoint(base_url, "chat/completions")

        strict_prompt = (
            "Generate media result directly. "
            "Do NOT call any tools/functions/chatroom actions. "
            "Return only final media output.\n\n"
            f"{prompt}"
        )

        content = [{"type": "text", "text": strict_prompt}]
        if image_base64:
            content.append({"type": "image_url", "image_url": {"url": image_base64}})

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "tool_choice": "none"
        }

        # 兼容不同后端：两种位置都传
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
            payload["video_config"] = {"aspect_ratio": aspect_ratio}

        logger.info(f"[api.chat] aspect_ratio={aspect_ratio}, payload_keys={list(payload.keys())}")

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        timeout = httpx.Timeout(connect=20.0, read=self.timeout_seconds, write=60.0, pool=self.timeout_seconds + 10)
        last_error = "未知错误"

        for i in range(self.max_retry_attempts):
            try:
                logger.info(
                    f"调用 Chat API (模型: {model}, aspect_ratio: {aspect_ratio or 'default'}, 尝试 {i + 1})"
                )
                r = await self.http_client.post(url, json=payload, headers=headers, timeout=timeout)

                if r.status_code == 200:
                    try:
                        return r.json(), None
                    except json.JSONDecodeError:
                        last_error = "JSON解析失败"
                        continue

                if r.status_code == 429:
                    last_error = "触发限流 (429)，正在重试..."
                    await asyncio.sleep(2)
                    continue

                if r.status_code == 500:
                    t = r.text
                    if "void *" in t or "NoneType" in t:
                        continue
                    last_error = f"服务端错误(500): {t[:120]}"
                    continue

                try:
                    err = r.json()
                    emsg = err.get("error", {}).get("message") or err.get("error")
                    last_error = f"API错误({r.status_code}): {emsg}"
                except Exception:
                    last_error = f"API请求失败({r.status_code})"

            except Exception as e:
                last_error = f"请求异常: {e}"

        return None, last_error

    async def call_generation(
        self,
        prompt: str,
        model: str,
        base_url: str,
        api_key: str,
        size: Optional[str] = None
    ) -> Tuple[Optional[dict], Optional[str]]:
        url = self.endpoint(base_url, "images/generations")

        payload = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "response_format": "url"
        }
        if size:
            payload["size"] = size

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        timeout = httpx.Timeout(connect=20.0, read=self.timeout_seconds, write=60.0, pool=60.0)
        last_error = "未知错误"

        for i in range(self.max_retry_attempts):
            try:
                logger.info(f"调用 Image Generation API (模型: {model}, size: {size or 'default'}, 尝试 {i + 1})")
                r = await self.http_client.post(url, json=payload, headers=headers, timeout=timeout)

                if r.status_code == 200:
                    try:
                        return r.json(), None
                    except json.JSONDecodeError:
                        last_error = "JSON解析失败"
                        continue

                if r.status_code == 429:
                    last_error = "触发限流 (429)，正在重试..."
                    await asyncio.sleep(2)
                    continue

                try:
                    err = r.json()
                    emsg = err.get("error", {}).get("message") or err.get("error")
                    last_error = f"API错误({r.status_code}): {emsg}"
                except Exception:
                    last_error = f"API请求失败({r.status_code})"

            except Exception as e:
                last_error = f"请求异常: {e}"

        return None, last_error

    async def call_image_edit(
        self,
        prompt: str,
        image_base64: str,
        model: str,
        base_url: str,
        api_key: str
    ) -> Tuple[Optional[dict], Optional[str]]:
        url = self.endpoint(base_url, "images/edits")
        headers = {"Authorization": f"Bearer {api_key}"}
        timeout = httpx.Timeout(connect=20.0, read=self.timeout_seconds, write=60.0, pool=60.0)
        last_error = "未知错误"

        raw, mime = self._decode_data_url(image_base64)
        if not raw:
            return None, "参考图无效：无法解析为图片数据"

        ext_map = {
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/gif": "gif",
            "image/webp": "webp"
        }
        ext = ext_map.get(mime, "jpg")
        filename = f"input.{ext}"

        for i in range(self.max_retry_attempts):
            try:
                logger.info(f"调用 Image Edit API (模型: {model}, size: follow-source, 尝试 {i + 1})")

                data = {
                    "model": model,
                    "prompt": prompt,
                    "n": "1"
                }
                files = {"image": (filename, raw, mime)}

                r = await self.http_client.post(
                    url,
                    data=data,
                    files=files,
                    headers=headers,
                    timeout=timeout
                )

                if r.status_code == 200:
                    try:
                        return r.json(), None
                    except json.JSONDecodeError:
                        last_error = "JSON解析失败"
                        continue

                if r.status_code == 429:
                    last_error = "触发限流 (429)，正在重试..."
                    await asyncio.sleep(2)
                    continue

                try:
                    err = r.json()
                    emsg = err.get("error", {}).get("message") or err.get("error")
                    last_error = f"API错误({r.status_code}): {emsg}"
                except Exception:
                    last_error = f"API请求失败({r.status_code})"

            except Exception as e:
                last_error = f"请求异常: {e}"

        return None, last_error