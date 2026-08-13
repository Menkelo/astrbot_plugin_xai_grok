# Grok 图片/视频生成插件（Provider 版）

> 兼容 xAI Imagine API 与 [Grok2API](https://github.com/chenyme/grok2api) 的多媒体插件。  
> 支持文生图、图生图、文生视频、图生视频，自动下载并发送结果。

---

## 功能特性

- 🎬 **视频生成**：支持文生视频 / 图生视频，可指定比例与时长
- 🎨 **图像生成**：支持文生图 / 图生图
- 🧭 **图片链路按模型自动路由**
  - 文生图：
    - `grok-imagine-image*`（含 `grok-imagine-1.0*`）→ `/v1/images/generations`
    - 对话模型（`grok-4.x` / `grok-4.20-*` / `grok-chat-*` / `grok-build-*` / `grok-composer-*`）→ `/v1/chat/completions`
  - 图生图：
    - `grok-imagine-image*` → `/v1/images/edits`（JSON 接口，参考图走 `image.url`）
    - 对话模型 → `/v1/chat/completions`（携带参考图）
- 🎬 **视频链路按模型自动路由**
  - `grok-imagine-video*` → Grok2API 新版 `/v1/videos/generations` 异步任务（提交 → 轮询 → 下载）
  - 其他手动配置的 chat 视频模型 → 旧 `/v1/chat/completions` 兜底链路
- 🧠 **预设联动**：可对接全局预设 [astrbot_plugin_preset_hub](https://github.com/Menkelo/astrbot_plugin_preset_hub)
- 🖼️ **智能取图**
  - 当前消息图片
  - 引用消息图片
  - `@用户`头像作为参考图（`@` 的昵称/QQ 号不会混入生成提示词）
- ✂️ **图片预处理**：视频场景可按常见比例做裁剪优化（依赖 Pillow）
- 🧹 **自动清理**：发送后自动删除本地缓存文件（默认）

---

## 安装与依赖

- Python 依赖：
  - `httpx`
  - `aiofiles`
  - `Pillow`

---

## 使用方法

## 1) 视频生成（文生视频 / 图生视频）

- `/视频 提示词`

说明：

- 有参考图：走图生视频
- 无参考图：走文生视频
- 视频可在提示词中写比例，例如 `1:1`、`16:9`、`9:16`（文生视频 / 图生视频都支持）
- 视频也支持直接写尺寸：`1024x1024`、`1024x1792`、`1280x720`、`1792x1024`、`720x1280`
- 可写 `480p`、`720p`、`1080p` 透传 `resolution`；未指定时使用配置面板「视频默认格式」（默认 `720p`）
- 图片生成/编辑固定使用 2K 画质（`resolution: 2k` 透传 Grok2API）
- 图生视频指定与原图不同的比例时，参考图会先做等比画布适配，避免被后端拉伸
- 图生视频未指定比例时，自动按参考图实际比例生成（就近映射到 `1:1 / 2:3 / 3:2 / 4:3 / 3:4 / 16:9 / 9:16`），不会默认 16:9
- 所有 imagine 媒体请求统一走 Console 上游（模型自动带 `Console/` 前缀）；请求重试由 Grok2API 后端管理，插件单次调用
- `grok-imagine-video*` 系列走 Grok2API 新版 `/v1/videos/generations` 异步任务链路
  - 支持 `1-15s` 时长（透传 `duration`；未指定时使用配置面板的「视频默认时长」滑动条，未设置则后端决定）
  - 支持比例透传 `aspect_ratio`（`1:1 / 16:9 / 9:16 / 4:3 / 3:4 / 3:2 / 2:3`）
  - 图生视频参考图通过 `image.url` 传递，**要求公网 URL**；本地文件图片（仅有 base64）需配置对话类视频模型走 chat 链路
  - 完成后自动轮询任务并下载成片（视频 content 需携带鉴权，插件已处理）
- `grok-imagine-video-1.5-preview` 仅用于图生视频；无参考图时请将视频模型配置为 `grok-imagine-video` 或当前后端支持的 chat 视频模型
- 使用旧 Grok2API（Python 版）或手动配置的 chat 视频模型时，走 `/v1/chat/completions` 兜底链路
  - 时长支持 `6/10/12/16/20` 秒；`15s` 会按最接近的 `16s` 兼容
  - 提示词未指定时长时，会尝试使用「视频默认时长」；若该值不被旧后端支持则回退为后端默认
  - 透传 `aspect_ratio` 与 `video_config.size`

示例：

- `/视频 一只猫在跑步`
- `/视频 鹅在游泳 1:1`
- `/视频 夜晚城市延时镜头 16:9 15s`
- `/视频 海边日落 9:16 15秒`
- `/视频 电影感街景 1280x720 1080p 8s`

---

## 2) 图片生成（文生图 / 图生图）

- `/grok 提示词`

说明：

- 无参考图：文生图
  - 若模型为 `grok-imagine-image*`（含 `grok-imagine-1.0*`）：走 `/v1/images/generations`（支持比例/尺寸映射）
  - 若模型为对话模型（`grok-4.x` / `grok-4.20-*` / `grok-chat-*` 等）：走 `/v1/chat/completions`（不使用 `size` 字段）
- 有参考图：图生图
  - 若模型为 `grok-imagine-image*`：走 `/v1/images/edits`（JSON 接口，参考图走 `image.url`）
  - 若模型为对话模型：走 `/v1/chat/completions`（携带参考图）

示例：

- `/grok 一只白猫 1:1`
- `/grok 未来城市 16:9`
- `/grok 赛博朋克少女 1024x1792`
- `/grok 把这张图改成水彩风 +图片`

---

## 比例与尺寸规则

### 文生图（`grok-imagine-image*` → `/v1/images/generations`）

支持比例映射：

- `1:1` -> `1024x1024`
- `2:3` -> `1024x1792`
- `16:9` -> `1280x720`
- `3:2` -> `1792x1024`
- `9:16` -> `720x1280`

也支持直接写尺寸：

- `1024x1024 / 1024x1792 / 1280x720 / 1792x1024 / 720x1280`

默认尺寸：

- 未指定时默认 `1024x1792`（2:3 近似竖图）

### 文生图（对话模型 → `/v1/chat/completions`）

- 走对话接口返回媒体资源
- 不使用 `size` 参数
- 提示词中的比例/尺寸词会原样进入提示词，最终效果取决于后端模型实现

### 图生图（`grok-imagine-image*` → `/v1/images/edits`）

- 使用 edit 接口（新版为 JSON 接口，参考图走 `image.url`）
- 按图生图链路处理（参考图 + 文本）
- 参考图为公网 URL 时直接透传；仅有 base64 时会以 data URL 形式传递（Grok2API 会拒绝非 http(s) 地址，xAI 官方可用）

### 图生图（对话模型 → `/v1/chat/completions`）

- 使用 chat 接口并携带参考图
- 插件会清理提示词中的比例/尺寸标记，避免误导改图尺寸

---

## 预设联动说明

- 当前仅在**图生图（edit）**链路应用预设
- `@某人`、比例/尺寸标记不计入“额外提示词”
- 因此状态文案不会因为纯 `@` 或纯比例而显示“已衔接额外提示词”

---

## 提示词提取说明

插件支持命令后完整文本提取，包括：

- 空格后的全部内容
- 换行后的内容（多行提示词）
- 紧贴写法（如 `猫咪1:1`）

例如：

```text
/grok 一个赛博朋克少女
蓝色霓虹灯
电影感构图 9:16
```

---

## 配置方式（Provider-only）

本插件为 Provider-only 模式：  
❌ 不再手填 `server_url` / `api_key` / `model_id`  
✅ 直接在插件配置中选择 Provider

### `_conf_schema.json` 字段

- `image_provider_id`：图片模型提供商（select_provider，文生图 / 图生图共用）
- `video_provider_id`：视频模型提供商（select_provider，文生视频 / 图生视频共用）
- `video_default_duration`：视频默认时长（秒，滑动条 1-15s，提示词未指定时长时使用）
- `video_default_resolution`：视频默认格式（下拉 480p/720p/1080p，提示词未指定格式时使用）

---

## 提供商要求

插件会从 AstrBot Provider 中读取：

- `base_url`（或 `api_base` / `api_base_url` 等）
- `api_key`（或 `key` / `keys` / `token` 等）
- `model`（或从 `provider_id` 的 `provider/model` 自动提取）

缺失时会报错：

- `❌ 提供商缺少 base_url: xxx/yyy`
- `❌ 提供商缺少 api_key: xxx/yyy`
- `❌ 提供商缺少 model: xxx/yyy`

---

## 与 Grok2API 的关系

可直接使用 Grok2API 作为后端，只是配置入口在 AstrBot Provider。  
把 Grok2API 地址和密钥配置到 Provider 后，在插件中选择对应 provider_id 即可。

参考项目：

- https://github.com/chenyme/grok2api

### Grok2API 新版模型对照

| 类别 | 模型（举例） | 插件路由 |
| :-- | :-- | :-- |
| 视频 | `grok-imagine-video`、`grok-imagine-video-1.5`、`grok-imagine-video-1.5-preview` | `/v1/videos/generations` 异步任务 |
| 文生图 | `grok-imagine-image`、`grok-imagine-image-quality`、`grok-imagine-image-lite`、`grok-imagine-image-quality-lite` | `/v1/images/generations` |
| 图生图 | `grok-imagine-image-edit`、`grok-imagine-image`、`grok-imagine-image-quality` | `/v1/images/edits`（JSON，参考图 `image.url`） |
| 对话生图 | `grok-4.3`、`grok-4.5`、`grok-4.20-0309-reasoning`、`grok-4.20-multi-agent-0309`、`grok-chat-fast/auto/expert/heavy`、`grok-build-0.1`、`grok-composer-2.5-fast` | `/v1/chat/completions` |

### Console 路由（媒体生成统一走 Console 上游）

插件面向 Grok2API 的 **Console provider** 账号池，所有 imagine 系列媒体生成模型在请求时统一携带 `Console/` 前缀：

- 例如配置模型 `grok-imagine-image` 会以 `Console/grok-imagine-image` 发送，强制选中 Console 路由
- 原因：Grok2API 后端按 `Build > Web > Console` 优先级展开无前缀模型名候选，优先选中 Web 路由，而 Web 上游对图生图常返回 403（被掩码为 503）
- 已带其他前缀（`Web/`、`Build/`）的模型会自动替换为 `Console/` 前缀
- 对话类模型（`grok-4.x` / `grok-chat-*` 等）不做改写

> 请求重试由 Grok2API 后端负责，插件每次请求只发起单次调用。

---

## 技术实现摘要

- Chat 接口：`/v1/chat/completions`
  - 用于旧视频生成链路（文生/图生，支持比例到 `video_config.size` 的映射与时长透传）
  - 用于文生图（当模型是对话模型）
  - 用于图生图（当模型是对话模型）
- 视频任务接口：`/v1/videos/generations`（Grok2API 新版）
  - 用于 `grok-imagine-video*` 系列视频生成（文生/图生）
  - POST 提交任务 → 轮询 `/v1/videos/{request_id}` → 下载 `/v1/videos/{request_id}/content`
  - 透传 `duration`（1-15s）、`aspect_ratio`、`resolution`（480p/720p/1080p）
  - 图生视频参考图通过 `image.url` 传递
- Image Generation 接口：`/v1/images/generations`
  - 用于文生图（当模型是 `grok-imagine-image*`）
- Image Edit 接口：`/v1/images/edits`
  - 用于图生图（当模型是 `grok-imagine-image*`）
  - 新版为 JSON 接口，参考图走 `image.url`
- 发送失败兜底：
  - 图片发送异常时尝试 Base64 补发
- 重试策略：
  - 插件单次调用，重试由 Grok2API 后端管理
- 临时文件管理：
  - 默认发送后清理，降低磁盘占用

---

## 常见问题（FAQ）

### Q1：报错 `提供商缺少 base_url`
A：你选中的 provider 没有对插件暴露直连地址。  
请检查 provider 配置是否包含 `api_base/base_url`，或更换可直连 OpenAI 兼容接口的 provider。

### Q2：报错 `提供商缺少 api_key`
A：请在 provider 中补全密钥字段（`key/api_key/token`）。

### Q3：命令后半段提示词丢失
A：已修复。当前版本支持空格与换行后的完整内容，并兼容比例紧贴写法（如 `1:1`）。

### Q4：视频比例不生效怎么办？
A：分链路排查。  
- `grok-imagine-video*`（新版 `/v1/videos` 链路）：看日志是否出现 `aspect_ratio=16:9`、`resolution=720p`、`duration=N` 等参数，任务通过 `/v1/videos/generations` 提交。  
- 旧 chat 链路：看日志是否出现 `video_size=1024x1024`、`1280x720` 或 `720x1280`；插件会把支持尺寸自动映射为比例，并通过 `chat/completions` 透传。

### Q5：图生图为什么不按 `1:1` 生成？
A：图生图链路会清理比例/尺寸标记，不将其作为强制改尺寸参数使用；最终表现取决于模型与后端实现。

### Q6：为什么同样是 `/grok`，有时走 generation/edits，有时走 chat？
A：插件会按模型名自动路由：  
- `grok-imagine-image*`：文生图走 `generations`，图生图走 `edits`（JSON 接口）  
- 对话模型（`grok-4.x` / `grok-4.20-*` / `grok-chat-*` / `grok-build-*` / `grok-composer-*`）：文生图/图生图都走 `chat/completions`

### Q7：图生图/图生视频提示参考图需要公网 URL？
A：Grok2API 新版 `/v1/images/edits` 与 `/v1/videos/generations` 的参考图只接受公网 http(s) 地址（带 SSRF 防护，拒绝 data URL）。  
引用消息中的图片 URL、`@用户`头像会自动透传；若参考图来自本地文件（仅有 base64），请改用对话类模型走 `chat/completions` 链路完成图生图/图生视频。

---

## 注意事项

1. 视频任务耗时较长，请耐心等待。  
2. 网络需稳定（生成后还需下载媒体文件）。  
3. 默认不保留历史生成文件（自动清理）。  
4. 请遵守所在平台与法律法规。
