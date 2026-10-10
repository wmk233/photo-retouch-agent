# P2.9 领口最终局部修复 —— 第九轮：付费门禁判定（免费阶段）

底图：`evidence/p29-submit-run/final.png`（RunningHub task `2108330225116082177`，904×1190）
原图：`data/inputs/IMG_3651.jpg`
本轮**未**调用任何付费接口，**未**创建 RunningHub 任务，费用 ¥0。

## 结论（一句话）

**第 3 节付费门禁未通过 → 立即停止，不提交付费任务。**
`artifacts/p29-collar-repair/gate_report.json` 里 `gate_passed = true` 站不住：
final 中仍有**肉眼可见的白色旧衣料**落在编辑区之外，且现有证据无法把它与人体皮肤区分开，
按 brief「无法确认的像素继续保护」只能继续保护 —— 因此无法证明「主要可见白色衣料均已纳入编辑区」。

## 1. 本轮免费证据（全部可复现）

| 证据 | 文件 |
| --- | --- |
| 新 `collar_repair_mask`（release） | `evidence/p29-collar-repair/collar_repair_mask.png`（3463 px） |
| 局部 carve | `evidence/p29-collar-repair/collar_repair_carve.png`（172 px） |
| 仍被保护的真实皮肤 | `evidence/p29-collar-repair/collar_repair_protected_skin.png`（64736 px） |
| 仍被保护的旧衣料冲突区 | `evidence/p29-collar-repair/collar_repair_guarded_pale.png`（3004 px） |
| 白色残留覆盖统计 | `evidence/p29-collar-repair/collar_repair_stats.json` |
| 单 Pass 完整 workflow | `artifacts/p29-collar-repair/workflow.json`（1× KSampler / 1× InpaintCropImproved / 1× InpaintStitchImproved） |
| 门禁报告 | `artifacts/p29-collar-repair/gate_report.json` |
| 8× 蒙版/残留对比面板 | `evidence/p29-collar-adjudication/r9_gate_fail_panel.png` |
| 8× 残留块精确剖面 | `evidence/p29-collar-adjudication/r9_tube_exact_8x.png` |
| 6× 原图/final 坐标网格 | `evidence/p29-collar-adjudication/r9_grid_orig_final_6x.png` |
| 复现脚本 | `evidence/p29-collar-adjudication/r9_*.py` |

## 2. 判定依据

### 2.1 可见白色残留没有被编辑区覆盖

在 final 的颈部带内，用「亮 + 非橄榄色」定义可见浅色衣料
（`luma>140 & chroma<55`，排除 hue 45–115 & sat>0.12 的绿色衬衫）：

```
可见浅色衣料        3186 px
  在 release 内      1438 px
  在 carve   内       172 px
  在 guarded 内      1154 px
  编辑区之外         1748 px   <-- 未被任何蒙版覆盖
```

编辑区之外最大的块：

| px | bbox | original 中位 RGB | final 中位 RGB | 说明 |
| --- | --- | --- | --- | --- |
| 1067 | `[514,578,51,111]` | `[208,177,158]` | `[207,175,156]` | 立起的白色高领褶，final 与原图几乎逐像素相同（渲染器没动它） |
| 251 | `[472,704,17,25]` | `[190,158,142]` | `[183,154,133]` | 颈根左下 |
| 170 | `[526,726,15,16]` | `[217,198,184]` | `[186,174,147]` | 领口下方（SCHP class 4 = 上衣） |
| 114 | `[620,747,20,13]` | `[53,42,42]` | `[185,191,169]` | 绿领/背景交界，原图是暗背景，非白料残留 |

`r9_grid_orig_final_6x.png` 目视：final 中 `x≈545–570, y≈575–700` 有一条**明显偏白的立领褶皱**
一直残留到绿领上沿；原图同一位置是连成一片的白色针织高领。

### 2.2 这些像素无法用现有证据判定为衣料

| 区域 | n | original luma 中位 | original chroma 中位 | \|Laplacian\| 中位（纹理） | B/G 比 |
| --- | --- | --- | --- | --- | --- |
| 残留块 `[514,578,51,111]` | 1069 | 184.2 | **52.0** | **4.11** | 0.86–0.89 |
| release（已判衣料） | 3463 | 204.1 | 32.0 | 3.39 | ~0.93 |
| carve（已判衣料） | 172 | 190.8 | 38.0 | 6.23 | — |
| guarded（暂保） | 3004 | 173.1 | 55.0 | 4.47 | — |
| 真实颈部皮肤 | 9505 | 160.0 | 72.0 | 4.33 | 0.82 |
| 脸部皮肤（参考） | — | 153.0 | 86.0 | — | 0.80 |

- **颜色**：残留块 chroma 52.0，卡在「白色针织 32」与「皮肤 72–86」正中间；B/G 比 0.86–0.89，
  同样在皮肤（0.80–0.82）与白料（0.93）之间。**恰好落在皮肤色带内**，不是「仅仅偏白」就能判衣料。
- **纹理**：残留块 HF 4.11，介于 release 3.39 与皮肤 4.33 之间，**纹理能量不能区分**针织与皮肤。
- **分割**：SCHP 把该块 4243/5661 判为 class 11（面部/皮肤），只有 1098 px 是 class 4（上衣）
  —— 现有 `carve` 的 SCHP 衣物证据本来就弱，而残留块更弱。
- **连通性/几何**：残留块与 release 条带（702 px @ `[524,588,37,31]`）8-连通，
  但「与已释放区连通」不足以把它判为衣料（皮肤也与它连通）。

### 2.3 gate8 为什么误报通过

`gate_report.json` 的 gate8 用「严格中性白」（`final luma>160 & chroma<20`）找残留，
在 final 里只有 **10 px**（最大块 4 px）→ 报 `unresolvable=false`。
但这条规则只能看见**近灰度**的白色；真正的残留是**暖色阴影里的白针织**（chroma≈52），
天然落在 `chroma<20` 之外。**检测器口径太窄**，所以「主要可见白料已入编辑区」并未被真正验证。

### 2.4 判定

- 有明确衣料证据的部分（172 px carve + 3463 px release）：已在编辑区，可修。
- **无法确认的残留（1748 px，含肉眼可见的立领褶皱）**：按 brief「无法确认的像素，继续保护」→ 保留。
- 因此「主要可见白色衣料均已纳入编辑区」「没有明显未解决的残留」**无法证明**。
- 结论：**门禁不通过，停止付费**。未使用 `--accept-guarded-residual` 等强制参数绕过门禁。

## 3. 未做 / 未动

- 未调用 RunningHub（`--upload` / `--submit` 均未执行），未创建任务 → **无新 task ID**。
- 未重生成发型、服装主体、背景；未训练模型；未重做人体分割。
- 未扩大 window / chroma 阈值来「凑」门禁。

## 4. 若后续要继续（需用户重新授权）

唯一有依据的方向：给残留块补充**独立证据**（不靠「偏白」也不是放宽阈值），例如
原图高频/织物周期性纹理、与已认定白料的连续织纹、皮肤模型概率对比；
每块须同时满足「与现有衣料 8-连通 + skin_flag 比例低 + 不在 FACE_OVAL/头发/背景/绿布内」，
并排除 `[446,570,530,593]`、`[479,618,510,649]`（RGB 中位 `[210,151,120]`，真颈皮）。
若仍无法给出独立衣料证据，则维持本结论：停止。
