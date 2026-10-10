# CropAndStitch 审计取证说明（provenance）

本轮 P2.9 对上游组件做了**只读审计**。上游代码为 **GPL-3.0**，而本仓库为
**Apache-2.0**，因此仓库**不收录上游源码正文**（避免把 GPL-3.0 代码并入
Apache-2.0 项目的分发）。此处仅登记可复现的来源、版本钉值与本地校验和；
审计结论见 `../P2.9-report.md`。

- 上游仓库: https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch
- 审计引用: `main` @ tree `8584b08d851762965df898b421a39075fc5357ae`
- 上游 LICENSE: GPL-3.0（blob `f288702d2fa16d3cdf0035b15a9fcbc552cd88e7`, 35,149 B）

## 审计对象（上游 blob 与本地副本校验和）

| 上游路径 | blob sha | 上游大小 (B) | 本地副本 sha256 |
| --- | --- | --- | --- |
| `inpaint_cropandstitch.py` | `ea413236190ca522b8c5277254831169bbd0152e` | 99,875 | `3ac0453d65ec9aff6f176c070d15c73715073c63a530975f5ffa2e1d4143bbf8` |
| `README.md` | `69b447c4e10019401ddc57a526a3fa7f30a3ca6d` | 12,905 | `8e50962e0043e99c3c5b02cfcc3759f9cd2afb2aa228a222742f4ace35f66538` |
| `.github` 树清单 (`tree.json`) | `8584b08d851762965df898b421a39075fc5357ae` | — | `cc75a3d0b9e65839b3ebadfcec95f392823274708c21dda2d87c7fb091639e0e` |

## 不入库的本地副本（`.gitignore` 已排除，可随时重新拉取）

- `inpaint_cropandstitch.py`（上游源码正文，GPL-3.0）
- `upstream-README.md`（上游 README 正文，GPL-3.0）
- `readme-api.json`（GitHub API 响应，内嵌 README base64 正文）

重新获取方式：

```bash
curl -sSL -o inpaint_cropandstitch.py \
  https://raw.githubusercontent.com/lquesada/ComfyUI-Inpaint-CropAndStitch/8584b08d851762965df898b421a39075fc5357ae/inpaint_cropandstitch.py
curl -sSL -o upstream-README.md \
  https://raw.githubusercontent.com/lquesada/ComfyUI-Inpaint-CropAndStitch/8584b08d851762965df898b421a39075fc5357ae/README.md
# 校验：sha256 应分别为 3ac0453d…3bbf8 / 8e50962e…f66538
```

## 审计结论（摘要，详见 `../P2.9-report.md`）

- 两节点契约：`InpaintCropImproved(image, mask?, optional_context_mask?) -> (stitcher, image, mask)`；
  `InpaintStitchImproved(stitcher, inpainted_image) -> 拼回原图坐标`。
- 成立：不修改未选中区域（不经过 VAE encode/decode）、仅编辑 mask 区域、裁剪→预缩放→采样→混合拼回原图。
- 注意：`optional_context_mask` 是**扩大裁剪上下文**用，不是保护区，`protected_identity_mask` 不可喂入。
- 风险：若开启 `output_resize_to_target_size`，重绘区会被缩放，**不得**把缩放结果判为保脸成功。
