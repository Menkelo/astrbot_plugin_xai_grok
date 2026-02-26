from dataclasses import dataclass


@dataclass
class ProviderRuntime:
    provider_id: str
    api_type: str
    base_url: str  # 规范化为 .../v1
    api_key: str
    model: str