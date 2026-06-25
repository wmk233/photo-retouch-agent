# Photo Retouch Agent MVP

一个对照片进行自动 P 图美化、也支持自定义美化的照片美化 Agent MVP。当前版本使用 Python/FastAPI 后端和原生 HTML/CSS/JavaScript 前端，先聚焦人像照片美化，同时按 domain/provider 结构预留风景、商品图和真实图像模型扩展。

## 当前能力

- 上传 JPG、PNG、WebP 图片，限制 10MB。
- 生成人像照片结构化分析。
- 提供自然美化、精致头像、氛围风格三个 RetouchPlan。
- 创建美化任务，使用 Pillow mock provider 生成结果图。
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
.venv/bin/python -m pip install fastapi "uvicorn[standard]" pillow python-multipart pytest httpx
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

接入真实模型时，新增 provider：

```text
providers/qwen_provider.py
providers/seedream_provider.py
```

并替换 `RetouchService` 默认使用的 `MockImageProvider`。当前 `MockImageProvider` 只负责 MVP 本地闭环，不代表最终图像质量。
