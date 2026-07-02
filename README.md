# Photo Retouch Agent MVP

一个对照片进行自动 P 图美化、也支持自定义美化的照片美化 Agent MVP。当前版本使用 Python/FastAPI 后端和原生 HTML/CSS/JavaScript 前端，先聚焦人像照片美化，同时按 domain/provider 结构预留风景、商品图和真实图像模型扩展。

## 当前能力

- 上传 JPG、PNG、WebP 图片，限制 10MB。
- 自动识别人像、风景、商品和通用照片，生成结构化分析。
- 人像提供自然美化、精致头像、氛围风格方案；其他照片提供自然增强、清晰通透、氛围风格方案。
- Agent 大脑支持 Qwen Vision、GLM Vision、豆包 Vision、DeepSeek 和本地规则。
- Agent 行动支持 Qwen Image、Wan Image、Seedream 和 Pillow 本地模拟。
- Agent 大脑和 Agent 行动必须显式选择并配置完成，工作流才可执行。
- 支持服务端环境变量 API Key，或用户在当前页面临时提供 API Key。
- 查询任务状态和结果。
- 基于上一张结果继续二次修改。
- FastAPI 托管前端页面，打开后端根路径即可使用。
- 标签化人像工作台，按皮肤、脸型、眼部、五官、身形、妆容和整体色调分类。
- 本地实时美化模式使用浏览器 Canvas 完成像素调整、区域柔化和基础形变，不调用云端模型。
- AI 精修模式把标签强度、风格预设和自然语言要求组合为修图指令，调用已选择的分析与图像编辑模型。
- 支持原图/结果滑动对比，并分别导出本地美化结果或 AI 生成结果。

## 项目结构

```text
backend/
  app/
    api/routes/        # HTTP 接口
    brains/            # DeepSeek/GLM/本地 Agent 大脑
    core/              # 配置和错误处理
    domains/           # portrait/landscape/product 等领域扩展点
    providers/         # 图像编辑模型适配层
    schemas/           # Pydantic 数据结构
    services/          # 存储、任务、修图编排
  tests/               # 后端单元测试
  pyproject.toml

frontend/
  index.html
  css/styles.css
  js/api.js
  js/main.js

data/
  uploads/             # 运行时生成，git 忽略
  outputs/             # 运行时生成，git 忽略
  jobs/                # 运行时生成，git 忽略
```

仓库根目录下的 `index.html`、`styles.css`、`app.js` 是最早期静态原型；当前可运行工程以前端 `frontend/` 和后端 `backend/` 为准。

## 人像美化工作台

前端提供两种互补的修图方式：

- **实时美化**：参数在浏览器本地处理，适合光滑肌肤、肤色、美白、光影、基础脸型/身形调整和整体色调。预览过程不会把图片发送给模型。
- **AI 精修**：用户选择视觉分析模型与图像编辑模型，标签强度会被转换为结构化修图要求，再按上传、分析、方案、生成的顺序执行后端工作流。

当前本地几何调整采用单人居中人像的启发式区域，属于 MVP 实现。后续可接入本地人脸关键点和人体分割能力，提高侧脸、多人以及复杂姿态下的区域精度。

AI 模式下临时输入的 API Key 只保存在页面内存中，不写入 Local Storage、任务 JSON 或下载文件；页面刷新后自动清空。

## 本地启动

```bash
cd backend
python3 -m venv .venv
.venv/bin/python -m pip install fastapi "uvicorn[standard]" pillow python-dotenv python-multipart pytest httpx
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

然后访问：

```text
http://127.0.0.1:8000/
```

API 文档：

```text
http://127.0.0.1:8000/docs
```

## 模型分层

项目将模型分为两层，避免把文本推理能力误当成图像编辑能力：

| 层级 | 可选项 | 职责 |
| --- | --- | --- |
| Agent 大脑 | Qwen Vision、GLM Vision、豆包 Vision、DeepSeek、本地规则 | 识别照片内容并优化图像编辑 prompt |
| Agent 行动 | Qwen Image、Wan Image、Seedream、Pillow mock | 接收优化后的 prompt，基于原图生成结果 |

Qwen Vision、GLM Vision 和豆包 Vision 使用图片 Base64 输入，能够直接观察照片。DeepSeek 官方 API 当前只提供文本 ChatCompletions，因此在本项目中会接收本地结构化照片分析并优化指令，界面明确标记为“分析辅助”，不会声称它直接看到了图片。

GLM 使用 `glm-5v-turbo` 负责视觉分析和 prompt 优化。视觉理解模型只输出文本，实际图片编辑仍由 Agent 行动层完成。

官方文档：

```text
https://api-docs.deepseek.com/
https://help.aliyun.com/en/model-studio/qwen-vl-compatible-with-openai
https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5v-turbo
https://www.volcengine.com/docs/82379/1362913
```

### Agent 大脑

```dotenv
PHOTO_AGENT_BRAIN_PROVIDER=qwen
DASHSCOPE_API_KEY=sk-your-dashscope-key
DASHSCOPE_WORKSPACE_ID=your-workspace-id
DASHSCOPE_VISION_MODEL=qwen3-vl-plus

DEEPSEEK_API_KEY=sk-your-deepseek-key
DEEPSEEK_MODEL=deepseek-v4-flash

ZHIPU_API_KEY=your-zhipu-api-key
ZHIPU_MODEL=glm-5v-turbo

ARK_API_KEY=your-ark-api-key
ARK_VISION_MODEL=doubao-seed-2-0-lite-260215
```

也可以在前端临时输入对应 API Key。Agent Key 使用 `X-Agent-API-Key` 请求头发送；Qwen 大脑的临时 Workspace ID 使用 `X-Agent-Workspace-Id`。

## 接入 Agent 行动

Qwen Image 和 Wan Image 使用阿里云百炼图像编辑 HTTP API；Seedream 使用火山方舟图片生成编辑 API。三者都接收 Base64 原图并返回有效期有限的结果 URL，后端会立即下载到本地 `data/outputs/`。

官方文档：

```text
https://help.aliyun.com/en/model-studio/qwen-image-edit-api
https://help.aliyun.com/en/model-studio/wan-image-edit
https://www.volcengine.com/docs/82379/1824692
```

### 方式一：服务端环境变量

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env`：

```dotenv
PHOTO_AGENT_IMAGE_PROVIDER=qwen
DASHSCOPE_API_KEY=sk-your-api-key
DASHSCOPE_WORKSPACE_ID=your-workspace-id
DASHSCOPE_REGION=beijing
DASHSCOPE_IMAGE_MODEL=qwen-image-2.0-pro
DASHSCOPE_WAN_MODEL=wan2.7-image-pro

ARK_API_KEY=your-ark-api-key
ARK_IMAGE_MODEL=doubao-seedream-5-0-260128
```

重启 FastAPI。前端显式选择一项 Agent 大脑和一项 Agent 行动；服务端已配置对应 Key 时，页面 API Key 输入框可以留空。

北京和新加坡地域的 API Key、Workspace ID 和请求地址不能混用。使用新加坡地域时：

```dotenv
DASHSCOPE_REGION=singapore
```

也可以通过 `DASHSCOPE_ENDPOINT` 在服务端配置完整接口地址。出于 SSRF 防护考虑，前端不能覆盖 Endpoint。

### 方式二：用户临时提供 API Key

前端选择行动模型并输入对应 API Key。Qwen Image 或 Wan Image 可同时填写 Workspace ID。Key 通过 `X-Action-API-Key` 请求头发送，只在当前页面内存和单次后端请求中使用：

- 不写入 job JSON。
- 不写入浏览器 Local Storage。
- 不出现在 URL 和正常日志中。
- 页面刷新后自动清空。

生产部署必须使用 HTTPS，不应通过公网明文 HTTP 传输用户 API Key。更成熟的产品应由服务端统一保管模型凭据，用户侧 API Key 模式只适合内部测试或 BYOK 场景。

## 测试

```bash
cd backend
.venv/bin/python -m pytest
```

前端当前使用原生 ES Module，可用 Node 做语法检查：

```bash
node --check frontend/js/api.js
node --check frontend/js/main.js
```

## API

```text
GET  /api/health
POST /api/photos/upload
POST /api/photos/analyze
POST /api/retouch/plans
GET  /api/retouch/providers
POST /api/retouch/jobs
GET  /api/retouch/jobs/{jobId}
POST /api/retouch/jobs/{jobId}/refine
```

## 扩展方向

新增照片类型时，在 `backend/app/domains/` 下增加对应 domain：

```text
domains/landscape/analyzer.py
domains/landscape/planner.py
domains/landscape/retoucher.py
```

然后在 `domains/registry.py` 注册即可。接口层不需要重写。

新增其他真实模型时，实现 `ImageEditProvider` 并在 factory 注册：

```text
providers/seedream_provider.py
```

当前 `MockImageProvider` 只负责离线闭环和测试，不代表最终图像质量。
