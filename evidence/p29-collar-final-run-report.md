# P29 衣领残留修复 — 最终候选真实出图验证（round 3）

- 仓库：`/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent`
- Git HEAD：`211c354`（工作树 clean，未 push、未打 tag）
- Plan：`ai_elegant_refined`
- 本轮真实任务：**1 次**，task ID `2108343890091409409`，status SUCCESS
- 真实扣费（账户余额差额）：**¥0.032**（19.880 → 19.848 CNY，整轮两阶段合计；不要按 stage 相加）
- 上一轮保留任务：`2108330225116082177`（`evidence/p29-submit-run/`）

## 一、付费前免费预检（全部通过）

- 由 HEAD 重新构建包 `artifacts/local-edit-package/20261010T-preflight`，四张蒙版与 `20261009T-collarfix` **逐字节一致**（sha256 相同）→ 确认未使用旧缓存。
- clothing mask = 上一版 + 302 px（302,487 → 302,789）；`clothing_added == protected_removed == 302`。
- hair mask 不变（187,091）；identity 21,873 / glasses 20,492 不变；neck skin 59,630 前后一致；`freed_over_skin 0`。
- protected 75,762 → 75,460；editable 476,523；hair∩clothing 13,357（未新增）。
- 提示词节点 5 含深橄榄绿同色领里 clause；节点 6/14 含 9 条新 negative 词。
- 执行顺序 clothing(node 4/10) → hair(node 12 读 node 10) 不变；SaveImage node 20 = clothing pass，node 19 = final。
- 原图与全部蒙版 904×1190。RunningHub 余额 19.880 CNY、当前任务 0；节点表重新拉取后为 **19,387** 个节点（上轮 19,188，表已变化，重新拉取必要），dry-run `errors: []`。

## 二、本轮实测（像素/数值）

| 指标 | 上一轮 final | 本轮 clothing pass | 本轮 final |
|---|---|---|---|
| 带内 pale-where-original-knit | 33,254 (37.8%) | 49,645 (56.4%) | **48,730 (55.3%)** |
| clothing mask 内最大 pale 连通块 | 10,163 px | 43,781 px | **43,695 px**（bbox 412,618–606,999） |
| 领区条带(594:729,478:562) 平均亮度 | 146.6 | 165.2 | 165.2 |

带 = y520–1000, x240–820；原始带内浅色针织物 88,055 px；PALE>165 / KNIT>150。

- **浅色残留来源判定：Clothing Pass 已经生成，不是 Hair Pass。** 49,645 → 48,730，两个阶段几乎相同（差异 <1,000 px），Hair Pass 只是把同一块浅色区域轻微平滑。按任务书第二部分的规则（“如果 Clothing Pass 已经生成浅色领里：只修改 clothing prompt”），本轮已带 collar clause 修改仍失败。
- 旧白领残片仍在：条带内本轮 final 仍有 6,330 px > PALE。
- 蒙版侧（离线、确定）：released_total = 302，**全部 302 px 都是 pale(>150) 且 100% 落在 clothing mask 内**，bbox 恰好 = y594–729 × x478–562（即任务书所说“条带”）。上一版在条带内仍保护 5,197 px，新版剩 4,897 px（释放 300 px，另有 2 px 在条带外）。
- **“670 保留像素”无法复现**：现存证据里没有任何自然定义等于 670（released=302；条带内 kept=11,040；条带内 kept 且原始为浅色=8,361；条带内仍受保护=4,897）。该数字可能来自上轮不同的条带定义，本轮以实测数字为准。

## 三、与上一轮的真实差异

- prev_final vs new_final：平均 |Δ|（RGB 求和）= 63.17；变化 >30 的像素 **396,930**。差异集中在整块头发与整件上衣，背景≈0。→ 本轮不是“定点修衣领”，而是把头发轮廓和整件衣服全部重新生成了一遍。
- 相对原图平均 |Δ|：上一轮 18.01 → 本轮 23.99。

区域保真（final vs 原图，mean|Δ| / max）：

| 区域 | 上一轮 | 本轮 |
|---|---|---|
| 背景左 | 0.51 / 3 | 0.51 / 3 |
| 背景右 | 0.36 / 3 | 0.36 / 3 |
| 嘴巴+牙齿 | 0.91 / 5 | **0.91 / 5** |
| 眼镜/眼周 | 3.20 / 91 | 3.38 / 95 |
| 颈部皮肤 | 0.98 / 64 | 1.39 / 66 |
| 额头/发际线 | 7.80 / 91 | **20.97 / 153** |

→ 背景、口腔牙齿、眼镜、颈部皮肤与上一轮一样保持得很好；**发际线区域明显变差**（可见暗色拼贴带）。

## 四、视觉验收结论：**未通过**

1. 旧白领残片未消失（条带内 6,330 px 浅色，放大图可见米色翻片）。
2. 新生成的浅色领里不但没消失，反而**明显变大**：本轮 clothing pass 直接把脖子以下画成一整片奶油/象牙色内搭（最大连通块 43,695 px，从下巴一直延伸到画面底部），深橄榄绿只剩两侧。
3. 已验收造型被破坏：栗子棕**蓬松大波浪 → 变短、贴头的波浪/波波头**；浅橄榄绿衬衫 → **深色缎面交叠上衣**。
4. 新增拼贴痕迹：额头/发际线一条暗色接缝带。
5. 未保住：发型体量、衬衫样式与衣身结构。

保住的：五官/笑容/牙齿（mean|Δ|=0.91，与上轮完全一致）、眼镜、颈部皮肤、背景、姿势。

## 五、证据文件（`evidence/p29-collar-final-run/`）

- `clothing_pass.png`（本轮 Clothing Pass，node 20）
- `final.png`（本轮 Final，node 19）
- `four_up_1x.png`（原图 / 上轮 Final / 本轮 Clothing Pass / 本轮 Final）
- `neck_8x_orig_prev_new.png`（领口 8×：原图 / 上轮 / 本轮）
- `neck_8x_clothingpass_vs_final.png`（8×：本轮 Clothing Pass vs Final，浅色来源）
- `face_hairline_3x.png`（3×：面部与发际线）
- `diff_prev_vs_new.png`（上轮 vs 本轮差异热图）
- `compare_renders.py` / `compare_renders.json`（本报告全部数值）

上一轮证据仍在 `evidence/p29-submit-run/`；`artifacts/p29-collar-final-run/task.json` 已脱敏（移除签名 `netWssUrl` 与 `clientId`）。

## 六、下一步（未经授权不执行）

按任务书判定，浅色来源在 Clothing Pass，则应继续改 clothing prompt（把“领口及褶皱同色”的约束提到最强权重，并显式禁止“内搭/围巾/高领/白色翻片/奶油色/象牙色”），而不是继续动蒙版；同时需要修掉本轮新出现的发际线接缝与发型漂移。本轮上限 0 付费，未再提交任何任务。
