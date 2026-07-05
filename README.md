# Grok 图片/视频生成插件（Provider 版）

> 兼容 xAI Imagine API 与 [Grok2API](https://github.com/chenyme/grok2api) 的多媒体插件。  
> 支持文生图、图生图、文生视频、图生视频，自动下载并发送结果。

---

## 功能特性

- 🎬 **视频生成**：支持文生视频 / 图生视频，可指定比例与时长
- 🎨 **图像生成**：支持文生图 / 图生图
- 🧭 **图片链路按模型自动路由**
  - 文生图：
    - `grok-imagine*` → `/v1/images/generations`
    - `grok-4.1*` → `/v1/chat/completions`
  - 图生图：
    - `grok-imagine*` → `/v1/images/edits`
    - `grok-4.1*` → `/v1/chat/completions`（携带参考图）
- 🧠 **预设联动**：可对接全局预设 [astrbot_plugin_preset_hub](https://github.com/Menkelo/astrbot_plugin_preset_hub)
- 🖼️ **智能取图**
  - 当前消息图片
  - 引用消息图片
  - `@用户`头像作为参考图
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
- 使用 `grok-imagine-video-1.5*` / `grok-imagine-video-1.5-preview` 时，时长支持 `1-15s`，例如 `15s` 会精确透传为 `duration=15`
- 使用旧 Grok2API 视频链路时，时长支持 `6/10/12/16/20`；`15s` 会按最接近的 `16s` 兼容

示例：

- `/视频 一只猫在跑步`
- `/视频 鹅在游泳 1:1`
- `/视频 夜晚城市延时镜头 16:9 15s`
- `/视频 海边日落 9:16 15秒`

---

## 2) 图片生成（文生图 / 图生图）

- `/画图 提示词`

说明：

- 无参考图：文生图
  - 若模型为 `grok-imagine*`：走 `/v1/images/generations`（支持比例/尺寸映射）
  - 若模型为 `grok-4.1*`：走 `/v1/chat/completions`（不使用 `size` 字段）
- 有参考图：图生图
  - 若模型为 `grok-imagine*`：走 `/v1/images/edits`
  - 若模型为 `grok-4.1*`：走 `/v1/chat/completions`（携带参考图）

示例：

- `/画图 一只白猫 1:1`
- `/画图 未来城市 16:9`
- `/画图 赛博朋克少女 1024x1792`
- `/画图 把这张图改成水彩风 +图片`

---

## 比例与尺寸规则

### 文生图（`grok-imagine*` → `/v1/images/generations`）

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

### 文生图（`grok-4.1*` → `/v1/chat/completions`）

- 走对话接口返回媒体资源
- 不使用 `size` 参数
- 提示词中的比例/尺寸词会原样进入提示词，最终效果取决于后端模型实现

### 图生图（`grok-imagine*` → `/v1/images/edits`）

- 使用 edit 接口
- 按图生图链路处理（参考图 + 文本）

### 图生图（`grok-4.1*` → `/v1/chat/completions`）

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
/画图 一个赛博朋克少女
蓝色霓虹灯
电影感构图 9:16
```

---

## 配置方式（Provider-only）

本插件为 Provider-only 模式：  
❌ 不再手填 `server_url` / `api_key` / `model_id`  
✅ 直接在插件配置中选择 Provider

### `_conf_schema.json` 字段

- `video_provider_id`：视频模型提供商（select_provider）
- `image_gen_provider_id`：文生图模型提供商（select_provider）
- `image_edit_provider_id`：图生图模型提供商（select_provider）

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

---

## 技术实现摘要

- Chat 接口：`/v1/chat/completions`
  - 用于旧视频生成链路（文生/图生，支持比例到 `video_config.size` 的映射与时长透传）
  - 用于文生图（当模型是 `grok-4.1*`）
  - 用于图生图（当模型是 `grok-4.1*`）
- Video Generation 接口：`/v1/videos/generations`
  - 用于 `grok-imagine-video-1.5*` / `grok-imagine-video-1.5-preview`
  - 支持 `duration` 1-15 秒，图生视频通过 `image.url` 传参考图
- Image Generation 接口：`/v1/images/generations`
  - 用于文生图（当模型是 `grok-imagine*`）
- Image Edit 接口：`/v1/images/edits`
  - 用于图生图（当模型是 `grok-imagine*`）
- 自动重试：
  - 针对 429 / 部分 5xx 做有限重试
- 发送失败兜底：
  - 图片发送异常时尝试 Base64 补发
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
A：旧 Grok2API 视频链路看日志是否出现 `video_size=1024x1024`、`1280x720` 或 `720x1280`。  
`grok-imagine-video-1.5*` 官方接口未公开比例参数，插件会保留提示词中的 `16:9 / 9:16 / 1:1` 让模型按文本理解。

### Q5：图生图为什么不按 `1:1` 生成？
A：图生图链路会清理比例/尺寸标记，不将其作为强制改尺寸参数使用；最终表现取决于模型与后端实现。

### Q6：为什么同样是 `/画图`，有时走 generation/edits，有时走 chat？
A：插件会按模型名自动路由：  
- `grok-imagine*`：文生图走 `generations`，图生图走 `edits`  
- `grok-4.1*`：文生图/图生图都走 `chat/completions`

---

## 注意事项

1. 视频任务耗时较长，请耐心等待。  
2. 网络需稳定（生成后还需下载媒体文件）。  
3. 默认不保留历史生成文件（自动清理）。  
4. 请遵守所在平台与法律法规。
