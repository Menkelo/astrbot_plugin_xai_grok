# [v1.1.19] - 修复 @用户 的 QQ 号混入提示词

* **🐛 提示词提取改为只取 Plain 文本**
  * 平台适配器（aiocqhttp）会把 `@用户` 渲染进 `message_str` 为 `" @昵称(QQ号) "`
  * 引用头像作为底图时，被 @ 的人的 QQ 号此前会残留到生成提示词
  * 现在从消息组件链中只拼接 `Plain` 文本段重建提示词，`At` / `Image` / `Reply` 组件一律跳过

---

# [v1.1.18] - 视频默认格式 + 图片强制 2K 画质

* **🖼️ 图片生成/编辑强制 2K 画质**
  * `/v1/images/generations` 与 `/v1/images/edits` 固定透传 `resolution: 2k`（Grok2API 后端默认 1k）
  * 图生图 / 文生图统一生效，`size` / `aspect_ratio` 不受影响

* **🎬 视频默认格式配置**
  * 配置面板新增「视频默认格式」下拉（`480p / 720p / 1080p`，默认 `720p`）
  * 提示词中显式写 `480p/720p/1080p` 时优先使用提示词；否则使用配置默认值
  * 注意：Console 上游 `grok-imagine-video` 仅支持 `480p/720p`，选择 `1080p` 可能被后端拒绝

---

# [v1.1.17] - 媒体生成统一走 Console 路由

* **🧭 强制 Console 路由（修复图生图跑到 Web 上游返回 403/503）**
  * Grok2API 后端按 `Build > Web > Console` 优先级展开无前缀模型名候选并优先选中 Web 路由，而 Web 上游对图生图常返回 403（被掩码为 503）
  * 所有 imagine 系列媒体模型在请求时统一携带 `Console/` 前缀（如 `Console/grok-imagine-image`），强制选中 Console 路由
  * 已带其他前缀（`Web/`、`Build/`）的模型自动替换为 `Console/`；对话类模型不改写
  * 图生图 / 文生图 / 视频链路统一生效

* **🧹 移除模型列表探测**
  * 不再调用 `GET /v1/models` 探测可用模型，去掉模型探测缓存与候选回退逻辑

* **🔁 重试交由后端管理**
  * 插件侧移除 429/503 等多次重试循环，每次请求只发起单次调用
  * 重试策略由 Grok2API 后端统一管理

---

# [v1.1.16] - 上游 503 自动重试

* **🔁 上游服务暂不可用(503)自动重试**
  * `call_chat` / `call_generation` / `call_image_edit` / `submit_video_job` 遇 503 自动重试（最多重试至配置次数，间隔 3s）
  * Grok2API 后端会把上游 401/403/额度耗尽统一掩码为 503「上游服务暂不可用」，多为瞬时状态，重试后常可恢复
  * 视频任务轮询遇 503 不再中断，继续轮询直至任务完成或超时
  * 全部重试失败时错误信息保留 503 与上游原因，便于排查

---

# [v1.1.15] - 图生视频按参考图比例生成

* **🖼️ 图生视频跟随参考图比例（修复默认 16:9）**
  * 图生视频时若提示词未指定比例，自动探测参考图实际比例并透传给视频接口
  * 支持 URL 参考图（探测原始 URL）与本地 base64 参考图（探测裁剪后比例），均与后端实际接收图一致
  * 探测比例就近映射到后端支持集合：`1:1 / 2:3 / 3:2 / 4:3 / 3:4 / 16:9 / 9:16`
  * 新视频链路（`/v1/videos/generations`）与旧 chat 兜底链路统一生效
  * 提示词显式指定比例时仍以提示词为准

---

# [v1.1.14] - 自动模型探测与回退

* **🧭 自动模型探测与回退（解决「模型不存在(404)」）**
  * 调用图片/图生图/视频接口前，探测 `GET /v1/models` 获取后端当前可用模型列表（短时缓存 5 分钟）
  * 配置的 imagine 系列模型不可用（后端未启用 / 名称不一致）时，自动回退到同类候选模型
  * 回退顺序：图片 `grok-imagine-image*` 系列 → 图生图 `grok-imagine-image-edit` 系列 → 视频 `grok-imagine-video*` 系列
  * 探测失败或候选全部不可用时不改写模型，沿用配置原值（不破坏旧后端链路）
  * 探测到的模型列表写入日志，便于排查 404 根因
  * 对话类模型生图不受影响，不做探测回退

---

# [v1.1.13] - 配置面板简化与视频默认时长

* **⚙️ 配置面板简化**
  * 移除 `video_t2v_provider_id` / `video_i2v_provider_id` / `image_gen_provider_id` / `image_edit_provider_id` 四个槽位
  * 现在只需配置两个模型提供商：`image_provider_id`（文生图/图生图共用）与 `video_provider_id`（文生视频/图生视频共用）
  * 旧配置字段会自动回退迁移到新槽位，无需重新填写

* **⏱️ 新增「视频默认时长」配置**
  * `video_default_duration`（滑动条 1-15s）
  * 提示词未指定时长时使用该默认时长
  * 新版 `/v1/videos` 链路直接透传；旧 chat 视频链路仅在支持范围（6/10/12/16/20）内生效，否则回退为后端默认

---

# [v1.1.12] - 兼容 Grok2API 新版模型与接口

* **🎬 视频链路切换（重要）**
  * `grok-imagine-video*` 系列改走 Grok2API 新版 `/v1/videos/generations` 异步任务接口
  * 提交任务 → 轮询 `/v1/videos/{request_id}` → 下载 `/v1/videos/{request_id}/content`
  * 视频 content 下载自动携带鉴权（Authorization）
  * 时长按 `1-15s` 校验并透传 `duration`，比例透传 `aspect_ratio`，分辨率透传 `resolution`（480p/720p/1080p）
  * 旧后端（手动配置的 chat 视频模型）保留 `/v1/chat/completions` 兜底链路

* **🖼️ 图片链路更新**
  * `/v1/images/edits` 改用 Grok2API 新版 JSON 接口，参考图通过 `image.url` 传递
  * 参考图保留原始 URL（引用消息图片 / `@用户`头像），Grok2API 仅接受公网 http(s) 地址
  * 仅有 base64 的本地图片会以 data URL 传递（xAI 官方可用，Grok2API 会拒绝并提示）

* **🧭 模型路由扩展（按 Grok2API 最新模型目录）**
  * 新增对话模型生图路由：`grok-4.3` / `grok-4.5` / `grok-4.20-0309-*` / `grok-4.20-multi-agent-0309` / `grok-chat-fast|auto|expert|heavy` / `grok-build-0.1` / `grok-composer-2.5-fast`
  * 新增图片模型路由：`grok-imagine-image` / `grok-imagine-image-quality` / `grok-imagine-image-lite` / `grok-imagine-image-quality-lite` / `grok-imagine-image-edit`
  * 新增视频模型路由：`grok-imagine-video` / `grok-imagine-video-1.5` / `grok-imagine-video-1.5-preview`

* **📄 文档同步**
  * 更新 `README.md` 与 `_conf_schema.json` 的模型与接口说明

---

# [v1.1.11] - 修复图生视频跨比例参考图被压扁

* **图生视频**
  * 取图时会先解析提示词中的视频比例/尺寸
  * 参考图会按目标比例生成画布，原图等比居中，背景模糊填充
  * 避免 2:3 参考图指定 16:9 视频时被后端直接拉伸压扁

---

# [v1.1.10] - 修复视频 chat 链路时长与尺寸透传

* **视频生成**
  * `/v1/chat/completions` 视频请求新增顶层 `seconds` 字段
  * `/v1/chat/completions` 视频请求新增顶层 `size` 字段
  * 保留 `video_config.seconds` / `video_config.size`，兼容不同后端读取方式
  * 用于修复 `grok-imagine-video-1.5-preview` 默认 6s、比例不生效的问题

---

# [v1.1.9] - 恢复 grok-imagine-video chat 路由

* **视频生成**
  * `grok-imagine-video` / `grok-imagine-video-latest` 恢复走 `/v1/chat/completions`
  * 继续不调用 `/v1/videos`
  * 保留 `grok-imagine-video-1.5-preview` 无参考图时的图生视频限制提示

---

# [v1.1.8] - 禁用 /v1/videos 并拦截 grok-imagine-video 404

* **视频生成**
  * 不再调用 `/v1/videos`
  * `grok-imagine-video` / `grok-imagine-video-latest` 不再发送到 `chat/completions`，避免后端返回模型 404
  * 当配置了不兼容的视频模型时，提前返回可读的配置提示
  * 视频生成链路继续只使用 `/v1/chat/completions`

---

# [v1.1.7] - 兼容 grok-imagine-video-1.5-preview 图生视频接口

* **视频生成**
  * `grok-imagine-video-1.5-preview` 携带参考图时改走 `POST /v1/videos` multipart 表单
  * 按后端要求传递 `input_reference` 图片字段
  * 时长改用 `seconds` 字段，支持将提示词中的 `1-15s` 传给后端
  * 尺寸继续使用 `size` 字段，例如 `720x1280`

---

# [v1.1.6] - 视频接口回退到 chat/completions

* **🎬 视频生成**
  * 不再调用 `/v1/videos/generations`
  * 所有视频模型统一走 `/v1/chat/completions`
  * 继续透传 `duration / aspect_ratio / video_config.size / video_config.resolution`

---

# [v1.1.5] - 拆分文生视频与图生视频槽位

* **🎬 视频配置**
  * 新增 `video_t2v_provider_id` 文生视频槽位和 `video_i2v_provider_id` 图生视频槽位
  * `video_provider_id` 保留为默认回退槽位，兼容旧配置
  * `grok-imagine-video-1.5-preview` 无参考图时会提示仅支持图生视频

* **🎞️ 视频参数**
  * 官方 `grok-imagine-video*` 视频接口优先走 `/v1/videos/generations`
  * 官方 xAI 视频接口透传 `aspect_ratio` 和 `resolution`
  * 支持从 `1024x1024 / 1024x1792 / 1280x720 / 1792x1024 / 720x1280` 映射视频比例
  * 旧后端不支持 `/v1/videos/generations` 时会回退到 `chat/completions` 视频链路

---

# [v1.1.4] - 兼容 grok-imagine-video-1.5

* **🎬 视频生成**
  * `grok-imagine-video-1.5*` / `grok-imagine-video-1.5-preview` 改走 xAI `POST /v1/videos/generations`
  * 支持官方 `duration` 参数范围 `1-15` 秒，`15s` 不再映射为 `16s`
  * 图生视频通过 `image.url` 传参考图，并轮询 `/v1/videos/{request_id}` 获取成片

---

# [v1.1.3] - 修复视频比例与时长透传

* **🎬 视频生成**
  * 图生视频也会解析提示词中的比例，并映射到 Grok2API 使用的 `video_config.size`
  * `15s / 15秒 / 15秒钟` 会按 Grok2API 当前支持的最接近时长 `16s` 透传
  * 视频请求会同时携带 `seconds / duration` 兼容字段，便于不同后端识别

---

# [v1.1.2] - 支持 15 秒视频并移除批量生成

* **🎬 视频生成**
  * 支持在提示词中使用 `15s` / `15秒` / `15秒钟` 指定 15 秒视频时长
  * 视频请求会将时长透传到 `video_config.seconds`

* **🧹 功能收敛**
  * 移除 `/视频N` 与 `/画图N` 批量生成命令

---

# [v1.1.1] - 生图/图生图接口按模型自动路由

* **🧭 图片链路接口自动路由（按模型名）**
  * 当模型为 **`grok-imagine`** 系列时：
    * 文生图（image）走 `POST /v1/images/generations`
    * 图生图（edit）走 `POST /v1/images/edits`
  * 当模型为 **`grok-4.1`** 对话系列时：
    * 文生图（image）走 `POST /v1/chat/completions`
    * 图生图（edit）走 `POST /v1/chat/completions`（携带参考图）

* **🎨 行为细节统一**
  * `images/generations` 路径继续支持比例/尺寸映射与默认尺寸 `1024x1792`
  * `chat/completions` 生图路径不传 `size` 参数，由模型侧生成媒体输出
  * 图生图在两条路径下都保持“按原图语义改图”定位；并保留对比例/尺寸词的清理策略（用于 edit 提示词净化）

* **🛡️ 稳定性与可观测性**
  * 保留工具卡片拦截（`<xai:tool_usage_card>`）提示，避免“未实际出图”被误判成功
  * 增加路由日志，便于排查当前到底命中了 `generation / edits / chat` 哪条链路

---
<details>
<summary>📋 点击查看历史更新日志</summary>

# [v1.1.0] - 功能与行为统一更新

* **🎬 视频能力升级**
  * 支持**文生视频 / 图生视频**（不再强制必须带参考图）
  * 文生视频支持从提示词解析比例并透传 `aspect_ratio`（如 `1:1 / 16:9 / 9:16`）
  * 状态文案在文生视频场景下可显示比例标签（如 `[1:1]`）

* **🎨 图片能力调整**
  * 文生图默认尺寸调整为 `1024x1792`（2:3 近似）
  * 文生图支持比例词到合法尺寸映射（`1:1 / 2:3 / 16:9 / 3:2 / 9:16`）
  * 图生图明确为**按原图比例处理**，不支持通过提示词改比例

* **🧠 预设联动优化**
  * 预设仍保持仅在 `edit`（图生图）链路应用
  * `@用户`、比例词、尺寸词不再被识别为“额外提示词”
  * 修复 `@` 黏连写法（如 `娘化@某人`）导致误显示“已衔接额外提示词”的问题

* **📝 提示词提取增强**
  * 命令提示词统一从 `event.message_str` 提取，避免比例词（如 `1:1`）在指令参数解析中丢失
  * 改进多行文本、空白符、中文冒号等兼容性

* **🔁 稳定性改进**
  * 资源下载新增自动重试（指数退避）
  * 当出现“资源已生成但下载失败”时会先重试再报错
  * 增加关键链路日志（比例解析、请求参数、下载重试）便于排障

* **🧹 功能收敛**
  * 移除 NSFW（涩图）相关逻辑与指令支持

# [v1.0.1] - 模型填写迭代

* **🛠️ 优化选择**: 全面对接模型提供商，不再手动填写

# [v1.0.0] - 初始版本

* **🎉 发布**: 插件初始版本发布
* **✨ 功能列表**:
  * 文/图生图
  * 图生视频

</details>
