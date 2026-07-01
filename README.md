# Photo Retouch Agent MVP

一个对照片进行自动 P 图美化、也支持自定义美化的照片美化 Agent MVP。当前版本使用 Python/FastAPI 后端和原生 HTML/CSS/JavaScript 前端，先聚焦人像照片美化，同时按 domain/provider 结构预留风景、商品图和真实图像模型扩展。

## 当前能力

- 上传 JPG、PNG、WebP 图片，限制 10MB。
- 生成人像照片结构化分析。
- 提供自然美化、精致头像、氛围风格三个 RetouchPlan。
- 支持 Pillow mock provider 和阿里百炼 Qwen Image provider。
- 支持服务端环境变量 API Key，或用户在当前页面临时提供 API Key。
- 查询任务状态和结果。
- 基于上一张结果继续二次修改。
- FastAPI 托管前端页面，打开后端根路径即可使用。

## 项目结构

```text
backend/
  app/
    api/routes/        # HTTP 接口
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

## 接入 Qwen Image

Qwen provider 使用阿里云百炼图像编辑 HTTP API，默认模型为 `qwen-image-2.0-pro`。官方接口支持将本地图片编码为 Base64 输入，并返回有效期有限的结果图片 URL；后端会立即下载结果到本地 `data/outputs/`。

官方文档：

```text
https://help.aliyun.com/en/model-studio/qwen-image-edit-api
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
```

重启 FastAPI。前端选择“自动选择”或“Qwen Image”，API Key 输入框可以留空。

北京和新加坡地域的 API Key、Workspace ID 和请求地址不能混用。使用新加坡地域时：

```dotenv
DASHSCOPE_REGION=singapore
```

也可以通过 `DASHSCOPE_ENDPOINT` 在服务端配置完整接口地址。出于 SSRF 防护考虑，前端不能覆盖 Endpoint。

### 方式二：用户临时提供 API Key

前端选择“Qwen Image”，输入 API Key；如果服务端没有配置 Workspace ID，同时填写对应 Workspace ID。Key 通过 `X-AI-API-Key` 请求头发送，只在当前页面内存和单次后端请求中使用：

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
