# P2.9 修正轮报告 — RunningHub 首次出图失败修复

> 本轮唯一目标：修复任务 `2108222523123662850` 的生成失败（第二张脸 / 秃斑 / 服装不符）。
> 平台固定 RunningHub；不新增 Provider；SCHP / MediaPipe / 两张已验收蒙版未改动。
> 取证脚本与对比图（含真人照片）留在本地 `evidence/p29-fix-run/`，不入库。

## A. 生成第二张脸的原因与证据

**直接原因（两条，均已实证）**

1. **两个 Pass 都用 `VAEEncodeForInpaint` + `denoise=1.0`。**
   `VAEEncodeForInpaint` 会把 mask 内的原像素编码为中性灰 `0.5`
   （`pixels = (pixels - 0.5) * m + 0.5`），即 mask 内**不含任何原图信息**；
   再叠加 `denoise=1.0`，采样器在每个 step 全部重写 latent →
   crop 内的整张脸都成了"待生成内容"，模型于是**按提示词重画了第二张脸**。
   证据：`workflow_diff` 中节点 7/15 `VAEEncodeForInpaint -> InpaintModelConditioning`，
   节点 8/16 `denoise: 1.0 -> 0.75/0.7`。

2. **正向提示词以"当前状态 + keep facial features"开场，等于点名要求重绘人脸。**
   旧文本（节点 5/13 的旧值）包含
   `the person currently wears ...` / `the person currently has long straight hair ...`
   以及 `Keep the facial features (eyes, brows, nose, lips) exactly as in the original photo.`
   ——这些句子被吃进 crop 的高分辨率区域后，直接诱发**第二张脸 / 第二副眼镜 / 多一张嘴**。

**量化证据（对照 `artifacts/p29-run/result_0.png`，见 `evidence/p29-fix-run/accept_stats.json`）**

| 指标 | 上轮 p29-run（失败） | 本轮 p29-fix-run |
|---|---|---|
| 全图 mean / max | 46.58 / 240 | 27.64 / 212 |
| 全图 >25 像素 | 372,900 | 319,897 |
| edit(soft) 区 mean | 102.76 | 60.48 |
| edit 外 mean / max / >25 | 1.041 / 62 / 159 px | 1.018 / 23 / **0 px** |

**关键机制澄清（写进结论，不夸大）**
`InpaintModelConditioning` 自身**不设置 `concat_keys`**；当前 checkpoint 是 SDXL base
（`juggernautXL_juggXIByRundiffusion.safetensors`，非 inpaint ckpt）→ `concat_keys=()`，
`concat_latent_image` / `concat_mask` 被 silent ignore。**实际生效的是**
① `out_latent["samples"] = vae.encode(原 crop 像素)`（用**真实原像素**替换 0.5 灰）；
② `noise_mask = 1.0 - denoise_mask`，mask 外强制回原 latent。
因此 `denoise < 1.0` ⇒ mask 外保留原像素信号，这才是本轮修复的杠杆。
**不是**走了 SDXL 真正的 concat-inpainting 通道。

## B. 实际修改的节点 / 连接 / 参数

对照上游官方示例 `example_workflows/inpaint_sd15.json` 补齐原图条件：

- 节点 7 / 15：`VAEEncodeForInpaint` → **`InpaintModelConditioning`**
  （`positive/negative` 来自 CLIP、`vae=["3",2]`、`pixels=["4",1]`/`["12",1]`、
  `mask=["4",2]`/`["12",2]`、`noise_mask=True`）。
- 节点 8 / 16 `KSampler`：`positive/negative/latent_image` 改接 `["7",0]/["7",1]/["7",2]`
  与 `["15",0]/["15",1]/["15",2]`；`denoise` 1.0 → **0.75（clothing）/ 0.70（hair）**。
- 节点 4 / 12 `InpaintCropImproved`：`mask_fill_holes` True → **False**（原 True 会吞保护区，
  实测填洞新增 283 px、其中 282 px 落在 protected identity 内，bbox `y[620,729] x[474,553]`）；
  `mask_blend_pixels` 4 → **2**。
- 节点 5 / 13 正向提示词重写：去掉"当前状态"描述，改为中性指令框
  （`Replace the garment with ...` / `Replace the hairstyle with ...`），
  删除项链句与整句眼镜句，追加 `real individual hair strands, natural hairline` 等；
- 节点 6 / 14 负向提示词补齐：`duplicate face, extra eyes, extra mouth, extra glasses,
  double head, second person, bald patches, deformed hairline`。
- **顺序保持 clothing → hair**；节点 10（Pass1 stitch）仍是节点 12 的输入，hair 拥有重叠带最终编辑权。
- 禁止项校验：门禁硬性拒绝 `EmptyLatentImage` / `EmptySD3LatentImage` / `VAEEncodeForInpaint`。

节点改动清单见 `artifacts/p29-fix-run/gate_report.json` 的 `workflow_diff`（gitignored，可再生）。

```
~ node 7  VAEEncodeForInpaint -> InpaintModelConditioning
~ node 8  KSampler: positive ['5',0]->['7',0]; negative ['6',0]->['7',1];
           latent_image ['7',0]->['7',2]; denoise 1.0 -> 0.75
~ node 15 VAEEncodeForInpaint -> InpaintModelConditioning
~ node 16 KSampler: positive ['13',0]->['15',0]; negative ['14',0]->['15',1];
           latent_image ['15',0]->['15',2]; denoise 1.0 -> 0.7
~ node 4  InpaintCropImproved: mask_fill_holes True->False; mask_blend_pixels 4->2
~ node 12 InpaintCropImproved: mask_fill_holes True->False; mask_blend_pixels 4->2
~ node 5/6/13/14 CLIPTextEncode: 提示词重写
```

## C. 是否完成 1 次真实付费任务

**YES — 1 次**（本轮额度已用完，未再提交任何计费任务）。

## D. 消费与任务 ID

- 任务 ID：`2108231348232871937`（inline workflow 提交，`nodeInfoList=[]`，`workflowId` 非 null）。
- `consumeMoney = "0.037"` CNY，在 ¥0.15 预算内。
- **`consumeCoins` / `thirdPartyConsumeMoney` 均为 `null` → 无法从 API 取得正式账单字段**，
  0.037 以 `outputs` 返回的 `consumeMoney` 为准。
- 输出 `artifacts/p29-fix-run/result_0.png`，**904×1190，与原图严格一致**。

## E. 对比图与局部放大（本地绝对路径）

- 三图并排：`/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent/evidence/p29-fix-run/compare_three.png`
- 面部：`.../evidence/p29-fix-run/zoom_face.png`
- 发际线：`.../evidence/p29-fix-run/zoom_hairline.png`
- 衣领：`.../evidence/p29-fix-run/zoom_collar.png`
- 颈/下巴：`.../evidence/p29-fix-run/zoom_neck.png`

## F. 逐项验收

- 五官不变：**YES** —— 无第二张脸、无额外眼/嘴/眼镜；眼镜框线略粗（局部小幅重绘）。
- 笑容不变：**YES**。
- 眼镜不变：**部分是** —— 无第二副眼镜，但框线比原图略粗，属局部重绘而非替换。
- 背景不变：**YES** —— edit 外 `>25` 像素 = **0 px**，max|Δ| 23。
- 发型改变成功：**YES 但有瑕疵** —— 栗子棕大波浪成立、秃斑消失；发际线有生硬锯齿分界。
- 服装改变成功：**YES 但有瑕疵** —— 墨绿垂坠衬衫成立；V 领开得过深，露出更多胸口。
- 附加伪影：下颌/颈部一块灰白"幽灵"矩形。定位结果：该区域 **71.5% 落在 hair mask、72.1% 落在 clothing mask、
  仅 10% 落在 protected identity 内** → 是**长发放到颈部皮肤上的半透明叠影（hair pass 伪影）**，
  **不是保护区被重绘，也不是第二张脸**。

**三态判定（不得混淆）**

- `MASK_VALID` = **PASSED**
- `PIXEL_IDENTITY_PASSED` = **PASSED（误差容限内）** —— 注意 **`max|Δ|=212` ≠ 逐字节一致**。
  保护区仅 66.08% 逐字节 ≤1、98.381% ≤11；只有 mask 硬区之外 11.84% 逐字节相同。
  因此写作"在容限内通过"，**不写作像素完全相同**。
- `VISUAL_ACCEPTANCE` = **FAILED**（颈/下巴灰白幽灵 + V 领过深露胸口 + 发际线生硬锯齿）。
  **但"第二张脸"这一硬失败条件已修复。**

blend 穿透取证：hair/clothing 的 `painted>0.5` 侵入 protected 仅 494 / 549 px，最大穿透 1.4 px
（0 px 深入 >2px）→ `mask_blend_pixels=2` **不会吞掉五官**。

## G. 唯一关键阻塞点

hair pass 的 crop 边界 / 发际线处理：导致颈部灰白幽灵与生硬接缝。**本轮就此停止，不再自动试下一组参数。**
