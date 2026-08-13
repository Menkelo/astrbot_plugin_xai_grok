import asyncio
import json
import time
from typing import List, Optional, Set, Tuple

import httpx
from astrbot.api import logger


class ApiClient:
    MODEL_CACHE_TTL_SECONDS = 300
    MODEL_PROBE_TIMEOUT = 15

    def __init__(self, http_client: httpx.AsyncClient, timeout_seconds=180, max_retry_attempts=3):
        self.http_client = http_client
        self.timeout_seconds = timeout_seconds
        self.max_retry_attempts = max_retry_attempts
        self._models_cache: Optional[Set[str]] = None
        self._models_cache_ts = 0.0
        self._models_cache_lock = asyncio.Lock()

    @staticmethod
    def endpoint(base_v1: str, path_after_v1: str) -> str:
        return f"{base_v1.rstrip('/')}/{path_after_v1.lstrip('/')}"

    async def fetch_available_models(self, base_url: str, api_key: str) -> Optional[Set[str]]:
        """探测当前后端可用模型列表（GET /v1/models），带短时缓存。

        探测失败（接口不存在 / 网络异常）返回 None，由调用方决定保持原模型。
        """
        now = time.time()
        if self._models_cache is not None and now - self._models_cache_ts < self.MODEL_CACHE_TTL_SECONDS:
            return self._models_cache

        url = self.endpoint(base_url, "models")
        headers = {"Authorization": f"Bearer {api_key}"}
        timeout = httpx.Timeout(connect=20.0, read=self.MODEL_PROBE_TIMEOUT, write=15.0, pool=15.0)

        async with self._models_cache_lock:
            if self._models_cache is not None and now - self._models_cache_ts < self.MODEL_CACHE_TTL_SECONDS:
                return self._models_cache
            try:
                r = await self.http_client.get(url, headers=headers, timeout=timeout)
                if r.status_code != 200:
                    logger.warning(f"[model.probe] GET /v1/models → HTTP {r.status_code}: {r.text[:200]}")
                    return None
                data = r.json()
            except Exception as e:
                logger.warning(f"[model.probe] GET /v1/models 失败: {e}")
                return None

            model_ids: Set[str] = set()
            for item in data.get("data", []) if isinstance(data, dict) else []:
                if isinstance(item, dict):
                    mid = str(item.get("id", "")).strip()
                    if mid:
                        model_ids.add(mid)
            if not model_ids:
                logger.warning("[model.probe] /v1/models 未返回任何模型")
                return None

            self._models_cache = model_ids
            self._models_cache_ts = time.time()
            logger.info(f"[model.probe] 可用模型 {len(model_ids)} 个: {sorted(model_ids)}")
            return model_ids

    async def resolve_model(
        self,
        configured_model: str,
        fallback_models: List[str],
        base_url: str,
        api_key: str,
        scene: str,
    ) -> str:
        """根据 /v1/models 探测结果自动选择可用模型，配置模型不可用时按候选回退。

        探测失败或候选全部不可用时不改写，返回配置模型原值。
        """
        preferred = str(configured_model or "").strip()
        if not preferred and fallback_models:
            preferred = fallback_models[0]

        candidates: List[str] = []
        for name in [preferred, *fallback_models]:
            name = str(name or "").strip()
            if name and name not in candidates:
                candidates.append(name)
        if not candidates:
            return preferred

        available = await self.fetch_available_models(base_url, api_key)
        if not available:
            return preferred

        for candidate in candidates:
            if candidate in available:
                if candidate != candidates[0]:
                    logger.warning(f"[model.resolve] [{scene}] 配置模型 {candidates[0]} 不可用，自动回退为: {candidate}")
                return candidate

        logger.warning(f"[model.resolve] [{scene}] 候选均不可用，继续使用: {candidates[0]}")
        return candidates[0]

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
        resolution: Optional[str] = None
    ) -> Tuple[Optional[dict], Optional[str]]:
        """
        用于视频等 chat/completions
        视频支持 aspect_ratio / size / seconds / resolution
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
        last_error = "未知错误"

        for i in range(self.max_retry_attempts):
            try:
                logger.info(
                    f"调用 Chat API (模型: {model}, aspect_ratio: {aspect_ratio or 'default'}, "
                    f"video_size: {video_size or 'default'}, "
                    f"duration: {duration_seconds or 'default'}, "
                    f"resolution: {resolution or 'default'}, 尝试 {i + 1})"
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
        last_error = "未知错误"

        for i in range(self.max_retry_attempts):
            try:
                logger.info(
                    f"提交视频任务 (模型: {model}, image_url={'有' if image_url else '无'}, "
                    f"aspect_ratio={aspect_ratio or 'default'}, resolution={resolution or 'default'}, "
                    f"duration={duration_seconds or 'default'}, 尝试 {i + 1})"
                )
                r = await self.http_client.post(url, json=payload, headers=headers, timeout=timeout)

                if r.status_code == 200:
                    try:
                        data = r.json()
                    except json.JSONDecodeError:
                        last_error = "JSON解析失败"
                        continue
                    request_id = (data or {}).get("request_id") or (data or {}).get("id")
                    if request_id:
                        return str(request_id), None
                    last_error = f"视频任务响应缺少 request_id: {str(data)[:200]}"
                    continue

                if r.status_code == 404:
                    return None, "后端不支持 /v1/videos/generations，请升级 Grok2API 或更换视频模型"

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
                elif r.status_code == 429:
                    logger.warning("[video.job] 轮询触发限流(429)，稍后继续...")
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
        image_url: str,
        model: str,
        base_url: str,
        api_key: str
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

        for i in range(self.max_retry_attempts):
            try:
                logger.info(f"调用 Image Edit API (模型: {model}, size: follow-source, 尝试 {i + 1})")

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
                        last_error = "JSON解析失败"
                        continue

                if r.status_code == 404:
                    return None, "后端不支持 /v1/images/edits JSON 接口，请升级 Grok2API"

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
