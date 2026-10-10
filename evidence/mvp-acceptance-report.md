# MVP 验收 —— 免费全流程 20/20、真实付费流程复现到中止点、本地无付费预检

本轮做三件事：把**免费全流程**从头走一遍并逐项核对；把**真实付费流程**再走一遍，
记录它走到哪一步、花了哪几笔钱；在**不花一分钱**的前提下用本地预检脚本验证
RunningHub 出图工作流与提示词是否与选中的造型一致。

## 付费口径（本轮一共花了什么）

| 流程 | 真实付费调用 | 说明 |
| --- | --- | --- |
| 免费全流程 | **0** | 图像生成走本地模拟（`pillow-mock-retoucher`），页面自述 0 次付费 |
| 真实付费流程 | **2 次 DeepSeek** | 人物事实分析 + 造型设计，两次都返回 200；之后走查中止，未出图 |
| 本地预检 | **0** | 未上传 RunningHub、未创建任务（`paidRequests=0`、`taskId=None`） |

图像生成费用：**0**。`data/outputs/studio/` 至今不存在，说明生图工作台还没有任何一次真实出图。

## 固定素材

- 原始照片：`data/inputs/IMG_3651.jpg`（904×1190）
- 免费流程上传后照片 ID：`img_7bb5d3a49a73`
- 预检流程照片 ID：`img_f1b17ea96eda`；提示词预检照片 ID：`img_e753ed69f6a8`

---

## 1. 免费全流程：20/20 通过

记录：`free-e2e-summary.json`（2026-10-09T07:0x）。结果页徽标是
**「技术检查通过 · 未完成视觉验收」** —— 也就是说，本轮证明的是流程与像素级改动，
不是「这张图好不好看」。

| 检查 | 结果 |
| --- | --- |
| 默认入口是「AI 造型设计」，默认服务商是免费本地模拟 | PASS（`action=mock`，`brain=deepseek`） |
| 未选照片时说真话，不假装检测到人脸 | PASS（`face=—`、`label=未选择照片`） |
| 上传照片：前端预览 + 后端留存 + 前端 ID 与后端一致 | PASS（`img_7bb5d3a49a73`） |
| 三套真实造型（模板免费、文案可用） | PASS（`urban_short` / `artistic_wave` / `bold_street`） |
| 确认前告知 0 费用 | PASS（`1 个本地任务（不计费） · 0 次付费（本地模拟 1 次）`） |
| 出图完成、结果页可渲染 | PASS |
| 结果**不是**原图 | PASS（904×1190，87.9% 像素发生变化；`lock_18b81daa6a6b.png`，1,152,137 bytes） |
| 前后对比、说明文字、下载链接、历史记录齐备 | PASS（历史 19 条） |
| 皮肤「AI 美化」同样 0 付费且真出图 | PASS（`changed=87.9%`） |
| 取消确认后不发付费设计请求 | PASS（`proposals 1 -> 1`） |
| 全程 0 条页面 JS 报错 | PASS（`errors=0`） |

## 2. 文案与口径核对：15/15 通过

记录：`free_copy_check.json`（2026-10-09T07:01:17Z，15 项全 PASS）。重点核对的是
「有没有把 3 次说成 2 次」「有没有暗示 DeepSeek 会看图」这类会让人误解付款与能力的问题：

- AI 说明里写明 **DeepSeek 没有图像输入能力、不会看到原图**，只依据本地测量的 FaceFacts 设计。
- 费用说明写明**完整流程 3 次付费调用**（1 次人物事实分析 + 1 次造型设计 + 1 次出图提示词规划），有缓存时只计 2 次。
- 按钮文案从「1 次分析 + 1 次设计」改为 **「本步 2 次 · 全程最多 3 次」**；模板按钮仍标 **0 次调用**。

## 3. 真实付费流程：走到「三套造型已渲染」后中止

记录：`real_e2e_summary.json`（2026-10-09T07:13:38Z → 07:14:34Z）。

走到并且通过的部分：

1. 能力自检：DeepSeek 已配置（`deepseek-v4-flash`），RunningHub 已配置（`runninghub-crop-stitch`），全流程付费次数自报 3。
2. 照片上传成功。
3. **费用确认弹窗内容正确**：明确写「最多 3 次付费请求」「不会自动重试」「只有点击确定才提交」。
4. **2 次真实 DeepSeek 调用成功**（`deepseek_calls.jsonl`）：分析 3,293 bytes、造型 19,858 bytes，两次都 `status: 200`，模型 `deepseek-v4-flash`。

中止点：

```
ABORT: no /styling/proposals response captured
```

走查脚本靠一个本地抓包代理记录浏览器发出的 `/styling/proposals` 响应，代理没抓到，
于是脚本主动中止。`capture_proxy.log` 里只有一行 `setsid: command not found`，
说明代理没有按预期方式启动 —— 本轮**没有**继续追查这一点。

中止带来的后果是可控的：

- 未进入出图步骤 → **图像生成费用 0**，未创建 RunningHub 任务。
- 记录器自身是好的：`recorder_selftest.json` 里 `pass: true`，能记录 200 与 404 两种响应；`proxy_selftest.jsonl` 证明代理能转发并落盘一次 chat/completions。

结论：**真实付费路径没有被端到端验证过**，已知的是「分析 + 造型」两笔花钱的调用真实可用，
出图那一步仍未走完。下次继续时应先修掉抓包代理的启动方式，再跑一遍。

## 4. 本地无付费预检：RunningHub 工作流 11/11

记录：`preflight-runninghub-local.json`（照片 `img_f1b17ea96eda`，造型 `artistic_wave`）。

| 检查 | 结果 |
| --- | --- |
| 上传原图 / 本地分析（0 次模型调用）/ 三套免费模板 | PASS |
| 选中的造型带真实 `editPrompt`（不是模板串台） | PASS（`style_id=artistic_wave`） |
| SCHP + MediaPipe 蒙版构建 | PASS（`editableShare=0.443`，`collarSplit=True`） |
| 发型 / 服装提示词与所选造型一致 | PASS |
| 工作流对着**线上节点表**校验 | PASS（`objectInfo=yes`，`errors=[]`） |
| 门禁接受该造型 | PASS（`denoise={'clothing':0.75,'hair':0.7}`，画幅 904×1190） |
| 本轮**未发生任何计费请求** | PASS（`paidRequests=0`，`taskId=None`） |

蒙版几何（`gate.mask_geometry`）：hair `painted_pixels=189900`、clothing `305536`，
两者 `painted_over_facial_core=0` —— 五官核心区没有被涂抹。

同一套预检在**AI 提示词**上也跑过一次：`preflight_ai_prompt.json`，`passed=True`，3 条 proposal。

已知差距（预检脚本自己列的 `gaps`，都指向「需要人眼确认」而不是「已知错误」）：

- `collar_margin_recovery_is_tone_based` / `collar_split_is_tone_based`：下巴以下划给服装区的那 302 px 是按肤色判的，不是按服装类判的 —— 遇到暗部脖子或肤色衣领会误判。
- `parsing_boundary_is_approximate`：ATR 在 512×512 上跑再上采样，发际线与领口只有几像素精度。
- `hairline_is_not_a_hair_matte`：发型蒙版是剪影 + 边缘扩张，碎发可能落在外面。
- `speckle_hair_islands_dropped`（11 个岛，806 px）/ `speckle_garment_islands_dropped`（9 个岛，3661 px）：画幅边缘的背景碎片被剔除。
- `glasses_guard_is_geometric`：无墨镜像素（清晰镜片不在 ATR 分类里），镜片带来自 MediaPipe 眼环，属几何而非分类结果。

## 5. 未做 / 未动

- 未重跑真实付费流程，未追查抓包代理的 `setsid` 缺失。
- 未做出图，未创建 RunningHub 任务，未改后端/前端代码。
- 未做「好看不好看」的视觉验收。

## 6. 本轮产物

保留在版本库里的（无图像数据）：`free-e2e-summary.json`、`free_copy_check.json`、
`real_e2e_summary.json`、`recorder_selftest.json`、`preflight-runninghub-local.json`、
`preflight_ai_prompt.json`、`deepseek_calls.jsonl`、`proxy_selftest.jsonl`，以及走查脚本
`e2e_free_flow.mjs`、`ui_walkthrough.mjs`、`real_e2e_acceptance.mjs`、`initial_state_shot.mjs`、
`free_copy_check.mjs`、`recorder_selftest.mjs`、`upload_probe.mjs`、`upload_verify.mjs`、
`load_requests_probe.mjs`、`mock_e2e.py`、`deepseek_capture_proxy.py`、
`preflight_runninghub_local.py`、`preflight_ai_prompt.py`。

**只留在本地**（含本人照片，已在 `.gitignore` 中排除）：`evidence/mvp-acceptance/*.png`（29 张）、
`test-portrait.png`、`preflight-work/`（照片副本与工作包）、`*.log`、`*.bak`。

复现（需先起本地服务，且不要把 RunningHub 的 `--submit` 混进来）：

```bash
cd backend && ../.venv/bin/python -m uvicorn app.main:app --port 8000   # 另开一个终端
node evidence/mvp-acceptance/free_copy_check.mjs
node evidence/mvp-acceptance/e2e_free_flow.mjs
../.venv/bin/python evidence/mvp-acceptance/preflight_runninghub_local.py   # 0 付费
```
