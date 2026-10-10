# 生图工作台「反推提示词」— 真实付费验证记录

- 仓库: `/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent`
- 验证时间: 2026-10-10
- 真实付费请求: **1 次**（`Qwen/Qwen3-VL-32B-Instruct`，`POST /api/studio/interrogate`）
- 网络调用: 报价阶段 **0 次**，只有确认后才发出那 1 次
- 离线测试: 后端 **394 passed**；前端 **41 pass / 0 fail**（视觉调用全部由 `httpx.MockTransport` 接住）

## 一、验证路径

1. `POST /api/photos/upload` 上传一张 904x1190 的本地人像 → `img_cf78e3670dbc`。
2. `POST /api/studio/interrogate/quote`：返回 `paidRequests: 1`、`requiresConfirmation: true`，不联网、不花钱。
3. `POST /api/studio/interrogate`（`confirmPaid: true`）：HTTP 200，耗时约 12s，返回主提示词 1 段、negativePrompt 1 段、备选 2 条。

## 二、判定

- 返回结构符合契约：`prompt` / `negativePrompt` / `alternatives` 三段齐备；备选自动去重，与主提示词重复的说法被丢掉（上限 3 条）。
- 提示词按约定顺序展开（主体与动作 → 环境与背景 → 光线与时间 → 构图与镜头 → 色调质感风格），给出的是可执行词汇（如 85mm、浅景深、暖调），不是「很美」这类空话。
- 身份设防生效：只写画面里看得见的发型、妆造、衣着、光线，没有推断年龄、职业、民族、宗教。
- 报价与实付一致：`paidRequests` 报 1，实际只发出 1 次请求；失败只上报、不重试，不会偷偷变成第 2 次。

## 三、隐私说明

反推会把照片交给云端视觉模型，画面内容以文字形式离开本机——这正是它只属于工作台、而不是美颜管线的原因（美颜管线仍然只在本机量事实）。

原始返回体（含对这张照片的文字描述）**未提交到 git**，避免把对真人外貌的描述写进公开仓库；本地留在 `evidence/interrogate-live-result.json`。
