import inspect
import json
import re
from typing import Optional, Tuple, Any
from astrbot.api import logger


class PresetService:
    def __init__(self, context):
        self.context = context

    def _get_preset_hub(self):
        preset_hub = getattr(self.context, "preset_hub", None)
        if preset_hub:
            return preset_hub

        if hasattr(self.context, "get_star"):
            try:
                preset_hub = self.context.get_star("astrbot_plugin_preset_hub")
                if preset_hub:
                    return preset_hub
            except Exception:
                pass

        if hasattr(self.context, "get_plugin"):
            try:
                preset_hub = self.context.get_plugin("astrbot_plugin_preset_hub")
                if preset_hub:
                    return preset_hub
            except Exception:
                pass

        return None

    async def _call_compat(self, fn, *candidates):
        last_err = None
        for args in candidates:
            try:
                ret = fn(*args)
                if inspect.isawaitable(ret):
                    ret = await ret
                return ret
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                break
        if last_err:
            raise last_err
        return None

    async def _resolve_preset_value(self, preset_hub, key: str) -> Any:
        fn = getattr(preset_hub, "resolve_preset", None)
        if not fn:
            return None
        ret = fn(key)
        if inspect.isawaitable(ret):
            ret = await ret
        return ret

    async def _get_all_keys(self, preset_hub):
        fn = getattr(preset_hub, "get_all_keys", None)
        if not fn:
            return []
        ret = fn()
        if inspect.isawaitable(ret):
            ret = await ret
        return ret or []

    @staticmethod
    def _parse_preset_value(preset_val: Any) -> str:
        if preset_val is None:
            return ""

        if isinstance(preset_val, dict):
            prompt = str(preset_val.get("prompt", "")).strip()
            ar = str(preset_val.get("aspect_ratio", "")).strip()
            return f"{prompt} {ar}".strip()

        if isinstance(preset_val, str):
            s = preset_val.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        prompt = str(obj.get("prompt", "")).strip()
                        ar = str(obj.get("aspect_ratio", "")).strip()
                        merged = f"{prompt} {ar}".strip()
                        if merged:
                            return merged
                except Exception:
                    pass
            return s

        return str(preset_val).strip()

    @staticmethod
    def _remove_at_tokens(text: str) -> str:
        s = str(text or "")

        # CQ at
        s = re.sub(r"\[CQ:at,[^\]]+\]", " ", s, flags=re.I)

        # OneBot 可能变体
        s = re.sub(r"\[at:[^\]]+\]", " ", s, flags=re.I)

        # HTML/XML at
        s = re.sub(r"<at[^>]*>", " ", s, flags=re.I)

        # 关键：无论是否有空格，直接去掉 @后连续非空白
        # 例：娘化@晗子 / @晗子 / @123456
        s = re.sub(r"@[^\s,，;；]+", " ", s)

        # @全体
        s = re.sub(r"@(全体成员|all)\b", " ", s, flags=re.I)

        return re.sub(r"\s+", " ", s).strip()

    @classmethod
    def _strip_nonsemantic_tokens_for_flag(cls, text: str) -> str:
        s = cls._remove_at_tokens(text)

        # 比例 1:1 / 16:9
        s = re.sub(r"(?<!\d)\d{1,2}\s*[:：]\s*\d{1,2}(?!\d)", " ", s)

        # 尺寸 1024x1792 / 1024×1792
        s = re.sub(r"(?<!\d)\d{2,5}\s*[xX×]\s*\d{2,5}(?!\d)", " ", s)

        # 清标点
        s = re.sub(r"[，,。.!！?？;；:：|/\\\-\_\(\)\[\]\{\}\"'`~]+", " ", s)

        return re.sub(r"\s+", " ", s).strip()

    @classmethod
    def _clean_extra_for_prompt(cls, text: str) -> str:
        # 真正给模型的 extra：去掉 @，保留正常语义
        s = cls._remove_at_tokens(text)
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _match_key(prompt: str, key: str) -> bool:
        """
        支持这些形式：
        - 娘化
        - 娘化 xxx
        - 娘化@晗子
        - 娘化,xxx / 娘化，xxx
        """
        p = (prompt or "").strip().lower()
        k = (key or "").strip().lower()
        if not p or not k:
            return False
        if p == k:
            return True
        return bool(re.match(rf"^{re.escape(k)}(?:\s+|[@,，]|$)", p))

    async def apply(self, prompt: str, event) -> Tuple[str, Optional[str], bool]:
        """
        return: (new_prompt, preset_name, has_extra)
        """
        prompt = (prompt or "").strip()
        if not prompt:
            return prompt, None, False

        preset_hub = self._get_preset_hub()
        if not preset_hub:
            return prompt, None, False

        # 1) 系统预设
        try:
            if hasattr(preset_hub, "get_full_prompt"):
                fn = preset_hub.get_full_prompt
                user_id = getattr(event, "unified_msg_origin", "") or ""
                new_prompt = await self._call_compat(fn, (prompt, user_id), (prompt,), tuple())
                if new_prompt and isinstance(new_prompt, str) and new_prompt.strip() != prompt:
                    return new_prompt.strip(), "系统预设", False
        except Exception as e:
            logger.warning(f"PresetService: get_full_prompt 调用失败: {e}")

        # 2) 关键词预设
        try:
            keys = await self._get_all_keys(preset_hub)
            if not isinstance(keys, list):
                keys = list(keys) if keys else []
            keys = [str(k).strip() for k in keys if str(k).strip()]
            keys.sort(key=len, reverse=True)

            for key in keys:
                if not self._match_key(prompt, key):
                    continue

                preset_val = await self._resolve_preset_value(preset_hub, key)
                preset_text = self._parse_preset_value(preset_val)
                if not preset_text:
                    continue

                raw_extra = prompt[len(key):].strip() if len(prompt) >= len(key) else ""

                extra_for_prompt = self._clean_extra_for_prompt(raw_extra)
                flag_text = self._strip_nonsemantic_tokens_for_flag(raw_extra)
                has_extra = bool(flag_text)

                merged = f"{preset_text}, {extra_for_prompt}" if extra_for_prompt else preset_text

                logger.info(
                    f"[preset] key={key!r}, raw_extra={raw_extra!r}, "
                    f"extra_for_prompt={extra_for_prompt!r}, has_extra={has_extra}"
                )
                return merged.strip(), key, has_extra

        except Exception as e:
            logger.warning(f"PresetService: 关键词预设匹配失败: {e}")

        return prompt, None, False