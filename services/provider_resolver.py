from typing import Optional, Tuple
from ..models.provider import ProviderRuntime


class ProviderResolver:
    def __init__(self, context):
        self.context = context

    @staticmethod
    def _extract_model_from_provider_id(provider_id: str) -> str:
        if provider_id and "/" in provider_id:
            return provider_id.split("/", 1)[1].strip()
        return ""

    @staticmethod
    def _normalize_openai_base(base_url: str) -> str:
        b = (base_url or "").strip().rstrip("/")
        if not b:
            return ""
        if not b.endswith("/v1"):
            b += "/v1"
        return b

    def parse(self, provider_id: str) -> Tuple[Optional[ProviderRuntime], Optional[str]]:
        if not provider_id:
            return None, "未配置提供商"

        provider = None
        if hasattr(self.context, "get_provider_by_id"):
            try:
                provider = self.context.get_provider_by_id(provider_id)
            except Exception:
                provider = None

        if not provider and "/" in provider_id and hasattr(self.context, "get_provider_by_id"):
            pname = provider_id.split("/", 1)[0]
            try:
                provider = self.context.get_provider_by_id(pname)
            except Exception:
                provider = None

        if not provider:
            return None, f"找不到提供商 ID: {provider_id}"

        p_conf = getattr(provider, "provider_config", {}) or {}

        base_url = (
            getattr(provider, "api_base", "")
            or p_conf.get("api_base")
            or p_conf.get("api_base_url")
            or p_conf.get("base_url")
            or p_conf.get("openai_api_base")
            or ""
        )

        api_key = ""
        for k in ["key", "keys", "api_key", "access_token", "token"]:
            val = p_conf.get(k)
            if val:
                api_key = val if isinstance(val, str) else (val[0] if isinstance(val, list) and val else "")
                if api_key:
                    break

        model = self._extract_model_from_provider_id(provider_id) or (
            getattr(provider, "model", "")
            or p_conf.get("model")
            or p_conf.get("model_id")
            or ""
        )

        base_url = self._normalize_openai_base(str(base_url))
        api_key = str(api_key).strip()
        model = str(model).strip()

        if not base_url:
            return None, f"提供商缺少 base_url: {provider_id}"
        if not api_key:
            return None, f"提供商缺少 api_key: {provider_id}"
        if not model:
            return None, f"提供商缺少 model: {provider_id}"

        return ProviderRuntime(
            provider_id=provider_id,
            api_type="openai",
            base_url=base_url,
            api_key=api_key,
            model=model
        ), None