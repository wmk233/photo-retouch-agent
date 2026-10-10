# P2.9 领口残留 — 固定最佳成片 + 单 Pass 局部修复预检

- 仓库：`/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent`
- 基线 HEAD：`211c354`；本报告对应提交为 `211c354` 之上的本轮 commit
- stylekit 只读；未访问 WORLD RADAR / AI Interior Studio 或任何无关项目
- **本轮真实付费请求：0 次。未调用任何付费接口，未上传输入文件。**
- 本轮只做到「可运行预检」，**不宣称修复已经视觉成功**，等待操作者授权付费

---

## A. 上轮失败报告与对比工具的提交

| 项 | 值 |
|---|---|
| commit | `2f73d10` — `docs(p29): record collar-residual regeneration failure and keep the compare tooling` |
| 内容 | `evidence/p29-collar-final-run-report.md`、`evidence/p29-collar-final-run/compare_renders.py`、`compare_renders.json` |
| 上轮失败任务 | RunningHub task `2108343890091409409`（仅用于失败分析） |
| 上轮结论 | 整套重生成（Clothing Pass → Hair Pass）**未通过**：旧白领残留未消失、新生成一大片奶油色内搭、发型与衬衫样式被破坏、额头新增拼贴接缝 |

上轮失败任务 `2108343890091409409` **仅作为失败证据**，不得作为下一次修复的底图。

---

## B. 最佳 Final 的明确来源与校验信息

| 项 | 值 |
|---|---|
| 来源任务 | RunningHub task `2108330225116082177`（status SUCCESS，上轮保留） |
| 文件 | `evidence/p29-submit-run/final.png` |
| 尺寸 | **904 × 1190** |
| SHA256 | `1b9634d1eb3f24738d9caf67d1009b21053a9790a1947400c67449bc5331590a` |
| 原图 | `data/inputs/IMG_3651.jpg`，904 × 1190，SHA256 `b87df5a0ac5158a7097652b4a8ea62104159699963bf9fc969d0e239da357d1d` |

这张照片已基本达标：本人五官与笑容保持、背景保持、栗子棕大波浪、绿色法式衬衫、发际线基本自然。
因此本轮把它**冻结为唯一修复底图**：以后不再通过重新生成整套造型来修复小范围领口问题。
门禁 gate1 会对该文件做 SHA256 校验，与冻结值不一致时直接失败，防止误用失败任务产物。

---

## C. 颈部白色残留的实际范围与修复蒙版

判定与生成工具：`backend/tools/build_collar_repair_mask.py`（离线，无网络、无密钥）。
它是对**已生成的最佳 Final** 做局部修复的释放蒙版（白 = 允许单次 inpaint 重绘，黑 = 保留 Final 原像素），
不是对最初白色毛衣重做整体换装。

复现命令（在 `backend/` 下）：

```bash
../.venv/bin/python -m tools.build_collar_repair_mask \
    --final ../evidence/p29-submit-run/final.png \
    --original ../data/inputs/IMG_3651.jpg \
    --labels ../evidence/p29-collar-repair/labels_fresh.png \
    --out ../evidence/p29-collar-repair
```

### 残留定位（`collar_repair_stats.json`）

| 指标 | 值 |
|---|---|
| 领口修复窗口 bbox | `[327, 594, 678, 752]`（y 由下巴下方起，窗口 55,968 px） |
| 下巴锚点 | `(486.61, 594.19)`，脸宽 256.72 px |
| 浅色衣料像素（final） | 120,027 |
| 原图为浅色针织物像素 | 209,309 |
| 领口残留（residual） | 3,282 |
| **可安全释放（release）** | **3,291 px**，bbox `[444, 594, 560, 709]` |
| 其中落在 SCHP 衣领类内 | 2,583 px |
| 落在服装类内 | 3,291 px（= 全部） |
| 释放区在头发上 | **0 px** |
| 释放区在背景上 | **0 px** |
| 释放区在身份保护区内 | **0 px** |
| 保留（no-edit）像素 | 963,857 |

释放区由两个连通块构成：

| 连通块 | 像素 | bbox |
|---|---|---|
| 主斜带 | 2,761 | `[444, 630, 96, 80]` |
| 下巴下方小块 | 530 | `[526, 594, 35, 25]` |

### 蒙版预览（`evidence/p29-collar-repair/`，PNG 因含真人照片被 `.gitignore` 排除，不提交）

| 文件 | 含义 |
|---|---|
| `collar_repair_mask.png` | 修复蒙版本体：白 = 可重绘残留，黑 = 保留 Final 原像素 |
| `collar_repair_no_edit.png` | 保护区（身份/眼镜/下巴/颈部皮肤/头发/背景/橄榄绿服装主体） |
| `collar_repair_guarded_pale.png` | 见 D 节：被皮肤保护挡住、工具拒绝裁决的残留 |
| `collar_repair_preview.png` | 三面板预览：Final / 红色=安全释放+黄色=受保护残留 / 蒙版 |

预览由 `make_repair_preview.py` 生成，可随时重跑复现（脚本已提交，PNG 不提交）。
视觉确认：红色安全区正好落在领口残留上，未覆盖皮肤、头发、背景或橄榄绿服装主体；黄色块位于下巴正下方。

---

## D. 因身份 / 皮肤保护不能安全编辑的像素

必须区分两类数字，二者不能混为一谈：

| 类别 | 像素数 | 说明 |
|---|---|---|
| 直接与身份保护区（脸/牙齿/眼镜/下巴/颈部皮肤）冲突的释放 | **0 px** | `blocked_by_identity_px = 0`；无任何释放落在保护区内 |
| **被「本人肤色保护」挡住的同带浅色残留** | **3,007 px**，bbox `[437, 594, 566, 726]` | `guarded_pale_px` |

3,007 px 的中央浅色块位于下巴正下方。在原图上，该处是褶皱阴影，被自适应肤色模型判为**本人皮肤**；
而它在 Final 上同时是浅色衣料。于是「保留原图真实皮肤像素」与「去除白色残留」两条要求在此区**无法同时满足**。
这是不可消除的保护冲突，工具**不会**为了赢而悄悄放宽蒙版：该区保持黑色、不参与重绘，如实上报给操作者裁决。

因此本轮**不声称所有残留均可自动去除**：可安全自动处理的只有左侧斜带约 **3,291 px**；
中央约 **3,007 px** 需要操作者给出处理意见（是否接受保留，或另行授权针对该块放宽皮肤保护）。

---

## E. 单 Pass 局部修复工作流是否准备好

工具：`backend/tools/runninghub_collar_repair_run.py`（离线构建 + 门禁；上传/提交需显式开关）。

节点图（**只有一个 KSampler**，不再执行完整 Clothing → Hair 两趟流程）：

```
1 LoadImage(final) → 2 LoadImageMask(collar_repair_mask) → 3 CheckpointLoaderSimple
→ 4 InpaintCropImproved → 5/6 CLIPTextEncode → 7 InpaintModelConditioning(noise_mask)
→ 8 KSampler(denoise 0.70) → 9 VAEDecode → 10 InpaintStitchImproved → 11 SaveImage
```

- 复用上次成功配置（`import` 自 `tools.runninghub_crop_stitch_run`）：`juggernautXL_juggXIByRundiffusion.safetensors`、`CROP_DEFAULTS`、`SAMPLER`、`DENOISE_BAND`。
- 极简提示词，正向只描述「把蒙版区重绘为与周围一致的深橄榄绿布料，延续成自然领口」；
  负向为短清单（white/cream/beige + 内搭/围巾/蕾丝/缎带/项链 + duplicated neckline + blurry/lowres/watermark）。
  **不再叠加数十条负向词**，且不含任何面部词（word-boundary 校验通过）。

### 八项离线门禁（`--dry-run` 实测）

| # | 门禁 | 结果 |
|---|---|---|
| 1 | 修复输入是否最佳 Final | 通过：SHA256 与冻结值 `1b9634d1…` 一致，904×1190 |
| 2 | 是否只有一个局部生成 Pass | 通过：KSampler 仅 `['8']`，passes = 1 |
| 3 | 蒙版是否局限于领口残留 | 通过：raw bbox `[444,594,560,709]` ⊂ 窗口 `[327,594,678,752]` |
| 4 | 是否与五官/眼镜/颈部皮肤保护区互斥 | 通过：raw mask ∩ guards = **0 px**；羽化边缘 255 px 且 ≤ 2.0 px（`mask_blend_pixels` 过渡，叠加在原图上而非替换）；未侵入核心 |
| 5 | 是否不覆盖正确处理过的发型/服装主体 | 通过：hair 0 px、background 0 px、olive_blouse 仅 147 px（深度 0.0、核心 0 px，为抗锯齿边缘） |
| 6 | 是否保持 904×1190 原图坐标 | 通过：crop 426,577,152,152 → 1024×1024，均匀缩放，stitch 还原画布 |
| 7 | 节点/模型是否与上次成功配置兼容 | 通过：`validate_workflow` 0 错误，checkpoint 无 inpainting UNet（走 masked-latent 路径，与上次成功一致） |
| 8 | 「保留原像素」与「去除白残留」是否冲突 | **未消解**：3,007 px @ `[437,594,566,726]`（见 D 节） |

**默认行为**：gate8 存在不可消解冲突时 **HALT（exit 2）** 并打印具体原因与像素数，不生成工作流。
`--accept-guarded-residual` 可在接受该残留前提下构建工作流（实测 exit 0，生成 `artifacts/p29-collar-repair/workflow.json`）。

> 结论：门禁 1–7 全部通过，**工作流已可运行**；是否放行第 8 项冲突由操作者决定。
> 真实付费请求仍为 **0**，未上传、未创建任何任务。

---

## F. 本地测试与 Git 状态

```
cd backend && ../.venv/bin/python -m pytest -q
→ 351 passed（1 条与本轮无关的 StarletteDeprecationWarning）
```

- `git status`：本轮提交后工作树 **clean**；未 push、未打 tag。
- 提交内容仅限：本报告、`backend/tools/build_collar_repair_mask.py`、
  `backend/tools/runninghub_collar_repair_run.py`、`evidence/p29-collar-repair/*.json`、`evidence/p29-collar-repair/*.py`。
- 未提交（按 `.gitignore` 排除）：真人照片、`*.png` 预览、`artifacts/` 运行产物、含签名 URL 的 `outputs.json`、任何密钥/令牌。

---

## 下一步（需操作者授权后才执行）

1. 由操作者裁决 gate8 的 3,007 px 中央残留（接受保留 / 另行授权放宽该块皮肤保护）。
2. 授权后在 `--accept-guarded-residual` 下执行一次上传 + 一次付费提交（**仅 1 个生成 Pass**）。
3. 修复结果仍需人工视觉验收；本轮**不宣称任何视觉成功**。
