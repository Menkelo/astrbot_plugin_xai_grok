# Grok 图片/视频生成插件（Provider 版）

> 基于 AstrBot 提供商体系（`select_provider`）的 Grok 多媒体插件。  
> 支持图生视频、文生图、图生图、NSFW 生图，自动下载并发送结果。

---

## 功能特性

- 🎬 **图生视频**：基于输入图片 + 提示词生成视频
- 🎨 **图像生成**：支持文生图 / 图生图
- 🔞 **NSFW 生图**：独立命令与独立模型提供商
- 🔁 **批量任务**：支持 `视频N / 画图N / 涩图N`
- 🧠 **预设联动**：可自动对接全局预设 [astrbot_plugin_preset_hub](https://github.com/Menkelo/astrbot_plugin_preset_hub)
- 🖼️ **智能取图**：
  - 当前消息图片
  - 引用消息图片
  - `@用户`头像作为参考图
- ✂️ **视频裁剪优化**：图生视频时自动按常见比例居中裁剪
- 🧹 **自动清理**：发送后自动删除本地缓存文件（默认）

---

## 安装与依赖

- Python 依赖：
  - `httpx`
  - `aiofiles`
  - （可选）`Pillow`：用于图片裁剪/压缩优化
- 如未安装 Pillow，插件仍可运行，但会跳过部分图片预处理。

```bash
pip install httpx aiofiles Pillow
```

---

## 使用方法

### 1) 视频生成（需要图片）

- `/视频 提示词`
- `/视频3 提示词`（连续生成 3 次）

> 需要当前消息或引用消息中有图片。  
> 也支持 `@某人`，会尝试使用头像作为输入图。

---

### 2) 图片生成（官方默认每次生成返回两次结果）

- `/画图 提示词`（无图：文生图；有图：图生图）
- `/画图4 提示词`（连续 4 次）

---

### 3) NSFW 生成

- `/涩图 提示词`
- `/涩图2 提示词`（连续生成 2 次）

---

## 提示词说明（已优化）

插件已支持命令后**完整文本提取**，包括：

- 空格后的全部内容
- 换行后的内容（多行提示词）

例如：

```text
/画图 一个赛博朋克少女
蓝色霓虹灯
电影感构图
```

---

## 配置方式（重要）

本插件已改为 **Provider-only** 模式：  
❌ 不再需要手动填写 `server_url` / `api_key` / `model_id`  
✅ 直接在插件配置中选择提供商模型

### `_conf_schema.json` 推荐字段

- `video_provider_id`：视频模型提供商（select_provider）
- `image_provider_id`：常规生图提供商（select_provider）
- `nsfw_provider_id`：NSFW 生图提供商（select_provider）

---

## 提供商要求

插件会从 AstrBot Provider 中读取以下信息：

- `base_url`（或 `api_base` / `api_base_url` 等）
- `api_key`（或 `key` / `keys` / `token` 等）
- `model`（或从 `provider_id` 的 `provider/model` 自动提取）

若缺失会报错，例如：

- `❌ 提供商缺少 base_url: xxx/yyy`
- `❌ 提供商缺少 api_key: xxx/yyy`

---

## 与 Grok2API 的关系

你依然可以使用 Grok2API 作为后端，只是配置入口变成了 AstrBot Provider。  
也就是说：  
把 Grok2API 地址与密钥配置到 AstrBot 提供商里，然后在本插件中“选择提供商”即可。

注意：需关闭流式响应。

参考项目：

- https://github.com/chenyme/grok2api
- https://github.com/Tomiya233/grok2api
- https://github.com/XeanYu/grok2api-rs

---

## 技术实现摘要

- Chat 接口：`/v1/chat/completions`
  - 用于视频生成、图生图、常规图像任务（按当前实现）
- Image Generation 接口：`/v1/images/generations`、`/v1/images/generations/nsfw`
  - 用于 NSFW 图像任务
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
请检查该 provider 的配置是否包含 `api_base/base_url`，或更换可直连 OpenAI 兼容接口的 provider。

### Q2：报错 `提供商缺少 api_key`
A：请在 provider 中补全密钥字段（`key/api_key/token`）。

### Q3：命令后半段提示词丢失
A：已修复。当前版本支持空格与换行后的完整内容。

### Q4：视频命令提示“需要提供图片”
A：请确保消息中包含图片、引用了带图消息，或使用 `@用户` 头像作为输入。

---

## 注意事项

1. 视频任务耗时较长，请耐心等待。
2. 网络需稳定（生成后还需下载媒体文件）。
3. 默认不保留历史生成文件（自动清理）。
4. NSFW 功能请遵守所在平台与法律法规。

