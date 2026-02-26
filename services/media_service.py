import re
import json
import uuid
import base64
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse, urljoin

import aiofiles
import httpx


class MediaService:
    VIDEO_EXTS = [".mp4", ".mov", ".webm", ".avi", ".mkv"]

    def __init__(self, http_client: httpx.AsyncClient, data_dir: Path):
        self.http_client = http_client
        self.data_dir = data_dir

    def extract_media_url_from_chat_response(self, response_data: dict):
        """
        返回: (list[str], err)
        list 可包含:
        - http(s) URL
        - data:image/...;base64,...
        """
        try:
            if not isinstance(response_data, dict):
                return None, "无效响应"

            urls = []

            def add_url(u: str):
                if not u:
                    return
                u = str(u).strip()
                if u and u not in urls:
                    urls.append(u)

            def add_b64(b64: str, mime: str = "image/png"):
                if not b64:
                    return
                b64 = str(b64).strip()
                if b64:
                    add_url(f"data:{mime};base64,{b64}")

            def walk(obj):
                if isinstance(obj, dict):
                    # 常见直出字段
                    for k in ("url", "image_url", "video_url"):
                        v = obj.get(k)
                        if isinstance(v, str):
                            add_url(v)
                        elif isinstance(v, list):
                            for x in v:
                                if isinstance(x, str):
                                    add_url(x)
                        elif isinstance(v, dict) and isinstance(v.get("url"), str):
                            add_url(v.get("url"))

                    # 常见 base64 字段
                    for k in ("b64_json", "image_base64", "base64"):
                        v = obj.get(k)
                        if isinstance(v, str):
                            add_b64(v)

                    for v in obj.values():
                        walk(v)

                elif isinstance(obj, list):
                    for it in obj:
                        walk(it)

                elif isinstance(obj, str):
                    s = obj.replace("\\/", "/")
                    # markdown 图片
                    for u in re.findall(r"!\[.*?\]\((https?://[^\s<>\"']+)\)", s):
                        add_url(u.rstrip(".,;:]}"))
                    # 普通链接
                    for u in re.findall(r"https?://[^\s<>\"')\]]+", s):
                        add_url(u.rstrip(".,;:]}"))
                    # data url
                    for d in re.findall(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", s):
                        add_url(d.replace(" ", ""))

            # 先递归扫描结构
            walk(response_data)

            # 再扫描整体JSON字符串（兜底）
            raw = json.dumps(response_data, ensure_ascii=False)
            raw = raw.replace("\\/", "/")
            for u in re.findall(r"https?://[^\s<>\"')\]]+", raw):
                add_url(u.rstrip(".,;:]}"))
            for d in re.findall(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", raw):
                add_url(d.replace(" ", ""))

            return (urls, None) if urls else (None, "未提取到有效的媒体链接")

        except Exception as e:
            return None, f"提取异常: {e}"

    def extract_media_url_from_generation_response(self, result: dict):
        if not isinstance(result, dict):
            return None, "响应格式无效"

        urls = []

        if isinstance(result.get("data"), list):
            for item in result["data"]:
                if not isinstance(item, dict):
                    continue
                if item.get("url"):
                    urls.append(item["url"])
                elif item.get("b64_json"):
                    urls.append(f"data:image/png;base64,{item['b64_json']}")

        if result.get("url"):
            urls.append(result["url"])

        # 兜底扫描
        if not urls:
            raw = json.dumps(result, ensure_ascii=False).replace("\\/", "/")
            for u in re.findall(r"https?://[^\s<>\"')\]]+", raw):
                uu = u.rstrip(".,;:]}")
                if uu not in urls:
                    urls.append(uu)

        return (urls, None) if urls else (None, "响应中未找到可用资源链接")

    async def download_file(self, url: str, base_url: str, api_key: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            # 支持 data URL
            if isinstance(url, str) and url.startswith("data:"):
                try:
                    header, b64data = url.split(",", 1)
                    mime = "image/png"
                    m = re.match(r"data:([^;]+);base64", header, re.I)
                    if m:
                        mime = m.group(1).lower()

                    ext = mimetypes.guess_extension(mime) or ".png"
                    filename = f"grok_media_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}{ext}"
                    file_path = self.data_dir / filename

                    try:
                        raw = base64.b64decode(b64data)
                    except Exception:
                        b64data = re.sub(r"[^a-zA-Z0-9+/=]", "", b64data)
                        raw = base64.b64decode(b64data)

                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(raw)

                    return str(file_path.resolve()), mime
                except Exception as e:
                    return None, f"dataURL解析失败: {e}"

            # 常规URL下载
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

    async def cleanup_file(self, path: Optional[str]):
        if not path:
            return
        try:
            p = Path(path)
            if p.exists():
                p.unlink()
        except Exception:
            pass