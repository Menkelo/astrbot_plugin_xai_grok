import asyncio
import json
from typing import Optional, Tuple

import httpx
from astrbot.api import logger


class ApiClient:
    def __init__(self, http_client: httpx.AsyncClient, timeout_seconds=180):
        self.http_client = http_client
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def endpoint(base_v1: str, path_after_v1: str) -> str:
        return f"{base_v1.rstrip('/')}/{path_after_v1.lstrip('/')}"

    async def call_chat(
        self,
        prompt: str,
        image_base64: Optional[str],
        model: str,
        base_url: str,
        api_key: str,
        aspect_ratio: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        video_size: Optional[str] = None,
        resolution: Optional[str] = None,
        allow_tools: bool = False
    ) -> Tuple[Optional[dict], Optional[str]]:
        """
        用于视频等 chat/completions
        视频支持 aspect_ratio / size / seconds / resolution
        allow_tools=True 时放开工具调用（Grok 对话模型生图依赖 imagine 工具调用）。
        """
        url = self.endpoint(base_url, "chat/completions")

        if allow_tools:
            if image_base64:
                strict_prompt = (
                    "An image is attached as reference. Please use the media generation tool "
                    "to apply the requested edit to the reference image and return the final "
                    "generated media output.\n\n"
                    f"{prompt}"
                )
            else:
                strict_prompt = (
                    "Please use the media generation tool to generate the requested image "
                    "and return the final generated media output.\n\n"
                    f"{prompt}"
                )
        else:
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
        }
        if not allow_tools:
            payload["tool_choice"] = "none"

        video_config = {}

        # 兼容不同后端：Grok2API 视频链路实际读取 video_config.size/seconds；
        # 其他 OpenAI 兼容后端可能读取 aspect_ratio 或 duration。
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
            video_config["aspect_ratio"] = aspect_ratio
        if video_size:
            payload["size"] = video_size
            video_config["size"] = video_size
        if resolution:
            payload["resolution"] = resolution
            video_config["resolution"] = resolution
            video_config["resolution_name"] = resolution
        if duration_seconds:
            payload["seconds"] = duration_seconds
            payload["duration"] = duration_seconds
            payload["duration_seconds"] = duration_seconds
            video_config["seconds"] = duration_seconds
            video_config["duration"] = duration_seconds
            video_config["duration_seconds"] = duration_seconds
        if video_config:
            payload["video_config"] = video_config

        logger.info(
            f"[api.chat] aspect_ratio={aspect_ratio or 'default'}, "
            f"video_size={video_size or 'default'}, duration={duration_seconds or 'default'}, "
            f"resolution={resolution or 'default'}, "
            f"video_config={video_config or 'default'}, "
            f"payload_keys={list(payload.keys())}"
        )

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        timeout = httpx.Timeout(connect=20.0, read=self.timeout_seconds, write=60.0, pool=self.timeout_seconds + 10)

        try:
            logger.info(
                f"调用 Chat API (模型: {model}, aspect_ratio: {aspect_ratio or 'default'}, "
                f"video_size: {video_size or 'default'}, "
                f"duration: {duration_seconds or 'default'}, "
                f"resolution: {resolution or 'default'})"
            )
            r = await self.http_client.post(url, json=payload, headers=headers, timeout=timeout)

            if r.status_code == 200:
                try:
                    return r.json(), None
                except json.JSONDecodeError:
                    return None, "JSON解析失败"

            try:
                err = r.json()
                emsg = err.get("error", {}).get("message") or err.get("error")
                return None, f"API错误({r.status_code}): {emsg}"
            except Exception:
                return None, f"API请求失败({r.status_code})"

        except Exception as e:
            return None, f"请求异常: {e}"

    async def call_video(
        self,
        prompt: str,
        image_base64: Optional[str],
        model: str,
        base_url: str,
        api_key: str,
        aspect_ratio: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        video_size: Optional[str] = None,
        resolution: Optional[str] = None
    ) -> Tuple[Optional[dict], Optional[str]]:
        """旧后端 chat/completions 视频链路（保留作兜底）"""
        return await self.call_chat(
            prompt=prompt,
            image_base64=image_base64,
            model=model,
            base_url=base_url,
            api_key=api_key,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            video_size=video_size,
            resolution=resolution
        )

    async def submit_video_job(
        self,
        prompt: str,
        model: str,
        base_url: str,
        api_key: str,
        image_url: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        duration_seconds: Optional[int] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Grok2API 新版视频任务：POST /v1/videos/generations
        返回 (request_id, err)。
        """
        url = self.endpoint(base_url, "videos/generations")

        payload = {
            "model": model,
            "prompt": prompt,
        }
        if image_url:
            payload["image"] = {"url": image_url}
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        if resolution:
            payload["resolution"] = resolution
        if duration_seconds:
            payload["duration"] = int(duration_seconds)

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        timeout = httpx.Timeout(connect=20.0, read=60.0, write=60.0, pool=60.0)

        try:
            logger.info(
                f"提交视频任务 (模型: {model}, image_url={'有' if image_url else '无'}, "
                f"aspect_ratio={aspect_ratio or 'default'}, resolution={resolution or 'default'}, "
                f"duration={duration_seconds or 'default'})"
            )
            r = await self.http_client.post(url, json=payload, headers=headers, timeout=timeout)

            if r.status_code == 200:
                try:
                    data = r.json()
                except json.JSONDecodeError:
                    return None, "JSON解析失败"
                request_id = (data or {}).get("request_id") or (data or {}).get("id")
                if request_id:
                    return str(request_id), None
                return None, f"视频任务响应缺少 request_id: {str(data)[:200]}"

            if r.status_code == 404:
                return None, "后端不支持 /v1/videos/generations，请升级 Grok2API 或更换视频模型"

            try:
                err = r.json()
                emsg = err.get("error", {}).get("message") or err.get("error")
                return None, f"API错误({r.status_code}): {emsg}"
            except Exception:
                return None, f"API请求失败({r.status_code})"

        except Exception as e:
            return None, f"请求异常: {e}"

    async def poll_video_job(
        self,
        request_id: str,
        model: str,
        base_url: str,
        api_key: str,
        poll_interval: float = 5.0,
        max_wait_seconds: float = 300.0
    ) -> Tuple[Optional[dict], Optional[str]]:
        """
        轮询视频任务：GET /v1/videos/{request_id}
        返回 (job_result, err)。job_result 含 status=pending/done/failed。
        done 时含 video.url 可直接下载。
        """
        url = self.endpoint(base_url, f"videos/{request_id}")
        headers = {"Authorization": f"Bearer {api_key}"}
        timeout = httpx.Timeout(connect=20.0, read=60.0, write=60.0, pool=60.0)
        last_error = "未知错误"

        waited = 0.0
        while waited < max_wait_seconds:
            try:
                r = await self.http_client.get(url, headers=headers, timeout=timeout)

                if r.status_code == 200:
                    try:
                        job = r.json()
                    except json.JSONDecodeError:
                        last_error = "视频任务响应JSON解析失败"
                        await asyncio.sleep(poll_interval)
                        waited += poll_interval
                        continue

                    status = str((job or {}).get("status", "")).lower()
                    if status == "done":
                        return job, None
                    if status == "failed":
                        err_body = (job or {}).get("error", {}) or {}
                        message = err_body.get("message") if isinstance(err_body, dict) else err_body
                        return None, f"视频生成失败: {message or '未知错误'}"

                    progress = (job or {}).get("progress")
                    logger.info(
                        f"[video.job] request_id={request_id}, status={status}, "
                        f"progress={progress}, waited={waited:.0f}s/{max_wait_seconds:.0f}s"
                    )

                elif r.status_code == 404:
                    return None, f"视频任务不存在或已过期: {request_id}"
                elif r.status_code in (429, 503):
                    logger.warning(f"[video.job] 轮询触发 {r.status_code}，稍后继续...")
                else:
                    try:
                        err = r.json()
                        emsg = err.get("error", {}).get("message") or err.get("error")
                        last_error = f"API错误({r.status_code}): {emsg}"
                    except Exception:
                        last_error = f"API请求失败({r.status_code})"

            except Exception as e:
                last_error = f"请求异常: {e}"

            await asyncio.sleep(poll_interval)
            waited += poll_interval

        return None, f"视频生成超时（已等待 {max_wait_seconds:.0f}s），请稍后查询。request_id={request_id}"

    async def call_video_job(
        self,
        prompt: str,
        model: str,
        base_url: str,
        api_key: str,
        image_url: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        poll_interval: float = 5.0,
        max_wait_seconds: float = 300.0
    ) -> Tuple[Optional[dict], Optional[str]]:
        """提交视频任务并轮询直至完成，返回 (done_job, err)"""
        request_id, err = await self.submit_video_job(
            prompt=prompt,
            model=model,
            base_url=base_url,
            api_key=api_key,
            image_url=image_url,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration_seconds=duration_seconds
        )
        if err:
            return None, err

        logger.info(f"视频任务已提交: request_id={request_id}, model={model}")
        job, perr = await self.poll_video_job(
            request_id=request_id,
            model=model,
            base_url=base_url,
            api_key=api_key,
            poll_interval=poll_interval,
            max_wait_seconds=max_wait_seconds
        )
        if perr:
            return None, perr
        job["request_id"] = request_id
        return job, None

    async def call_generation(
        self,
        prompt: str,
        model: str,
        base_url: str,
        api_key: str,
        size: Optional[str] = None,
        resolution: Optional[str] = "2k"
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
        if resolution:
            payload["resolution"] = resolution

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        timeout = httpx.Timeout(connect=20.0, read=self.timeout_seconds, write=60.0, pool=60.0)

        try:
            logger.info(
                f"调用 Image Generation API (模型: {model}, size: {size or 'default'}, "
                f"resolution: {resolution or 'default'})"
            )
            r = await self.http_client.post(url, json=payload, headers=headers, timeout=timeout)

            if r.status_code == 200:
                try:
                    return r.json(), None
                except json.JSONDecodeError:
                    return None, "JSON解析失败"

            try:
                err = r.json()
                emsg = err.get("error", {}).get("message") or err.get("error")
                return None, f"API错误({r.status_code}): {emsg}"
            except Exception:
                return None, f"API请求失败({r.status_code})"

        except Exception as e:
            return None, f"请求异常: {e}"

    async def call_image_edit(
        self,
        prompt: str,
        image_url: str,
        model: str,
        base_url: str,
        api_key: str,
        resolution: Optional[str] = "2k"
    ) -> Tuple[Optional[dict], Optional[str]]:
        """
        Grok2API 新版 /v1/images/edits 为 JSON 接口，参考图通过 image.url 传递。
        url 可以是公网 http(s) 地址，也可传 data URL（兼容 xAI 官方，Grok2API 会拒绝非 http(s)）。
        """
        url = self.endpoint(base_url, "images/edits")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        timeout = httpx.Timeout(connect=20.0, read=self.timeout_seconds, write=60.0, pool=60.0)
        last_error = "未知错误"

        if not image_url:
            return None, "参考图无效：缺少 image url"

        payload = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "image": {"url": image_url},
        }
        if resolution:
            payload["resolution"] = resolution

        try:
            logger.info(
                f"调用 Image Edit API (模型: {model}, size: follow-source, "
                f"resolution: {resolution or 'default'})"
            )

            r = await self.http_client.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout
            )

            if r.status_code == 200:
                try:
                    return r.json(), None
                except json.JSONDecodeError:
                    return None, "JSON解析失败"

            try:
                err = r.json()
                emsg = err.get("error", {}).get("message") or err.get("error")
                return None, f"API错误({r.status_code}): {emsg}"
            except Exception:
                return None, f"API请求失败({r.status_code})"

        except Exception as e:
            return None, f"请求异常: {e}"
