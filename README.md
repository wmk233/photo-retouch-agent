# Photo Retouch Agent MVP

一个对照片进行自动 P 图美化、也支持自定义美化的照片美化 Agent 静态原型。当前版本用浏览器本地 Canvas 模拟修图链路，覆盖上传预览、Agent 分析、三种一键方案、结果对比、下载和二次修改。

## 使用方式

直接打开 `index.html`，或在目录内启动任意静态服务：

```bash
python3 -m http.server 4173
```

然后访问 `http://127.0.0.1:4173/`。

## 当前范围

- 支持 JPG、PNG、WebP 上传预览。
- 输出模拟的结构化照片分析。
- 提供自然美化、精致头像、氛围风格三个方案。
- 用 Canvas 生成可下载的模拟美化结果。
- 支持基于上一个结果继续输入自然语言修改。

## 后续接入点

- `POST /api/photos/upload`
- `POST /api/photos/analyze`
- `POST /api/retouch/plans`
- `POST /api/retouch/jobs`
- `GET /api/retouch/jobs/:id`
- `POST /api/retouch/jobs/:id/refine`
