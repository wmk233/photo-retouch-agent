# P2.9 — RunningHub 实盘核对（有 Key，0 次付费生成）

- 日期: 2026-10-08；核对时 HEAD: `7586710`
- 本轮**只发只读 / 校验类请求**：账户信息、APIKEY 列表、`/object_info`、一次注定失败的 `create` 校验。
  **未提交任何计费任务、未上传任何真人照片、未充值。** 账户 `remainCoins` 前后均为 100、`currentTaskCounts` 均为 0。
- API Key 只存放于仓库外 `~/.runninghub_key`（`umask 077`，mode 600），**未写入本仓库任何文件、未打印**。
- 本文件取代 `p29-runninghub-audit.md` 第三节的「无 Key 阻塞」条目；该文件其余结论仍然有效。

## 一、凭据与账户（已核实）

`POST /uc/openapi/accountStatus`（body `{"apikey": ...}`）→

```json
{"code":0,"msg":"success","data":{"remainCoins":"100","currentTaskCounts":"0",
 "remainMoney":null,"currency":null,"apiType":"SHARED"}}
```

`GET /openapi/v2/api-key/list`（`Authorization: Bearer <key>`）→ 该账号 11 个 Key，本次使用的 Key
`a316****5baf`，`status:1`，`quotaUsed:0.000`，建号日期 2026-10-08（新账号）。

结论：**Key 有效、鉴权通过、账户可读**。Key 类型为 `SHARED`（共享型实例）；`remainMoney` 为 `null`，即**钱包余额为空**。

## 二、运行时节点清单（决定性核实）

RunningHub 提供原生 ComfyUI 代理 `GET https://www.runninghub.cn/proxy/<API_KEY>/object_info`
（文档《原生ComfyUI接口支持说明》，等同本地 `127.0.0.1:8188`）。该请求只读、不产生任务、不扣费。

实测返回 **HTTP 200，32,253,161 字节，19,201 个节点定义**。其中：

```
InpaintCropImproved   category=inpaint  python_module=custom_nodes.ComfyUI-Inpaint-CropAndStitch  output=[STITCHER, IMAGE, MASK]
InpaintStitchImproved category=inpaint  python_module=custom_nodes.ComfyUI-Inpaint-CropAndStitch  output=[IMAGE]
```

→ **平台运行时已安装 `lquesada/ComfyUI-Inpaint-CropAndStitch`**（节点类名与上游 fork 一致，
非同名近似节点：`InpaintCrop`/`InpaintStitch` 原版亦并存，但我们需要的 Improved 版本存在）。
`p29-runninghub-audit.md` 第三节第 2 条「是否已安装未核实」由此**解除**。

完整输入契约（抽取自 `object_info`，见 `p29-runninghub-live/node-contract-cropandstitch.json`）：

`InpaintCropImproved`
- 必填: `image`(IMAGE)、`downscale_algorithm`、`upscale_algorithm`、`preresize`、
  `preresize_mode`、`preresize_min_width/height`、`preresize_max_width/height`、
  `mask_fill_holes`(默认 True)、`mask_expand_pixels`(默认 0)、`mask_invert`(默认 False)、
  `mask_blend_pixels`(默认 32)、`mask_hipass_filter`(默认 0.1)、
  `extend_for_outpainting` + `extend_{up,down,left,right}_factor`、
  `context_from_mask_extend_factor`(默认 1.2)、
  **`output_resize_to_target_size`(默认 True)、`output_target_width/height`(默认 512)**、
  `output_padding`(默认 32)、`device_mode`(默认 `cpu (compatible)`)
- 可选: `mask`(MASK)、`optional_context_mask`(MASK)
- 输出: `STITCHER`、`IMAGE`(裁剪块)、`MASK`

`InpaintStitchImproved`
- 必填: `stitcher`(STITCHER)、`inpainted_image`(IMAGE)
- 输出: `IMAGE`

→ 契约与本地标准包严格对应：`image` ← `original_image`；`mask` ← `hair_edit_mask` / `clothing_edit_mask`；
`inpainted_image` ← 采样器输出；`stitcher` 把结果**按原坐标系拼回**（`output_resize_to_target_size`
只影响送进采样器的裁剪块尺寸，不影响回贴坐标）。这满足「输出尺寸与原始照片严格对应」。

## 三、API 契约（已逐页核对官方文档）

| 用途 | 方法与路径 | 关键请求 | 关键响应 |
| --- | --- | --- | --- |
| 上传原图/蒙版 | `POST /openapi/v2/media/upload/binary`（multipart `file`，Bearer） | PNG/JPG/JPEG/WEBP | `data.fileName`（喂 `LoadImage.image` 用法）、`download_url`、`type` |
| 发起任务 | `POST /task/openapi/create` | `apiKey`、`workflowId`、`nodeInfoList[{nodeId,fieldName,fieldValue}]`、`instanceType`、`webhookUrl`、`retainSeconds`，**或 `workflow`（整张工作流 JSON，指定后忽略 `workflowId`）** | `data.taskId`、`clientId`、`netWssUrl` |
| 查状态 | `POST /task/openapi/status` | `apiKey`、`taskId` | `data` = `QUEUED`/`RUNNING`/`SUCCESS`/`FAILED` |
| 取结果 | `POST /task/openapi/outputs` | `apiKey`、`taskId` | `data[]` = `fileUrl`、`fileType`、`nodeId`、`consumeCoins` |
| 取结果 V2 | `POST /openapi/v2/query` | `taskId` | `status`、`results[].url`、`errorCode` |
| 取工作流 JSON | `POST /api/openapi/getJsonApiFormat` | — | 工作流 JSON |

计费口径：ComfyUI 任务按实例时长扣 **coins**（文档示例 `consumeCoins: 17` / 83 秒）；
`instanceType` 分 `default`(24G)/`plus`(48G)/`ultra`(84G)，`retainSeconds`(10~180s) 另计。
`/openapi/v2/price-preview/**` 只覆盖模型 API，**不覆盖 ComfyUI 工作流任务**。

## 四、实测阻塞：`605 NOT_ENOUGH_BALANCE`

`POST /task/openapi/create` 在本账号上被**余额门禁**拦下，且与 `workflowId` 无关，可复现：

| 请求 | 返回 |
| --- | --- |
| `{"apiKey":K,"workflowId":"1"}` | `{"code":605,"msg":"NOT_ENOUGH_BALANCE"}` |
| `{"apiKey":K,"workflowId":"1980237776367083521"}`（文档示例 ID） | 同上 |
| `{"apiKey":K,"workflowId":"1904136902449209346"}`（文档示例 ID） | 同上 |
| 加 `Authorization: Bearer`、`nodeInfoList:[]`、`instanceType:"default"` | 同上 |

- `605 / NOT_ENOUGH_BALANCE` **不在**官方《接口错误码说明》表内（该表只到 301/380/412/416/421/433/435/436/5xx/8xx/9xx）；
  最接近的是 `416 TASK_CREATE_FAILED_BY_NOT_ENOUGH_WALLET`（钱包余额不足，建议充值）。
- 账户 `remainCoins = 100` 但 `remainMoney = null`。判断：**100 创作币不构成 API 任务的可扣余额**，
  共享型 API 任务需要钱包余额（RMB）。
- 对照组：同一把 Key 调 `/openapi/v2/price-preview/alibaba/qwen-image-2.0/image-edit`
  返回 `1007 field 'imageUrls' is required` —— 说明 **Key 对 v2 接口鉴权正常**，
  `605` 是 ComfyUI 任务创建这一条路径的余额门禁，不是鉴权失败。
- 该次失败请求**未创建任务、未扣费**（`currentTaskCounts` 仍为 0，`remainCoins` 仍为 100）。

## 五、结论

1. **节点可用性：通过。** 平台运行时装有 `ComfyUI-Inpaint-CropAndStitch`
   （`InpaintCropImproved` / `InpaintStitchImproved`），输入输出契约与本地标准包严格对应，
   支持「原图 + 外部 mask + prompt 输入 + 局部生成 + 拼回原坐标」。
2. **现成可调用工作流：不存在（按验收口径）。** 平台没有可离线枚举的工作流库
   （列表页客户端渲染、`robots.txt` 禁止 `/api/`），且我们**未曾**在该账号工作台保存过任何工作流，
   因此**没有任何一个已核对过真实节点输入的 `workflowId` 可以给出**。
   可替代路径是 `workflow` 内联提交（官方文档明确支持，指定后忽略 `workflowId`），
   这条路径不受「平台是否预置工作流」限制 —— 但需先构造并本地校验一张 ComfyUI 图。
3. **提交能力：阻塞。** 即便工作流就绪，本账号当前也**提交不了任务**（`605 NOT_ENOUGH_BALANCE`）。
   这是本轮的硬阻塞。

## 六、距离一次真实生成还缺什么

1. **账户余额**：RunningHub 钱包充值（或确认哪种余额可用于共享型 API 任务），
   否则任何 `/task/openapi/create` 都返回 `605`。这是唯一挡在「能不能提交」前面的东西。
2. **一张已核对的工作流**：二选一
   - (a) 在 RunningHub 网页工作台里搭一次 `LoadImage → InpaintCropImproved → 采样 → InpaintStitchImproved`
     并**保存+手动运行一次**（错误码 `810 WORKFLOW_NOT_SAVED_OR_NOT_RUNNING` 要求如此），
     导出 API JSON，用 `/api/openapi/getJsonApiFormat` 逐节点复核；
   - (b) 直接内联 `workflow` JSON 提交，跳过工作台。
3. **API 输入映射**（已就绪，见 `p29-runninghub-audit.md` 第二节表）：
   `original_image` / `*_edit_mask` → 上传取 `fileName` → `LoadImage.image`；prompt → `CLIPTextEncode.text`。
4. **仍缺**：一次付费授权（本轮为 0 次）。

## 七、复现命令（不打印 Key）

```bash
umask 077 && printf '%s' '<KEY>' > ~/.runninghub_key && chmod 600 ~/.runninghub_key
K=$(cat ~/.runninghub_key)

# 账户信息（只读）
curl -sS -X POST https://www.runninghub.cn/uc/openapi/accountStatus \
  -H 'Content-Type: application/json' -d "{\"apikey\":\"$K\"}"

# 运行时节点清单（只读，约 32MB；用于确认 CropAndStitch 是否安装）
curl -sS "https://www.runninghub.cn/proxy/$K/object_info" -o object_info.json
grep -c '"InpaintCropImproved"' object_info.json
```

文档页抓取（Apifox 站点，`window.__remixContext` 只是外壳，正文需 `strip.py` 去标签）：
`https://www.runninghub.cn/runninghub-api-doc-cn/api-425749013`（高级创建）、`-425749012`（简易）、
`-425749007`（上传）、`-425749003`（状态）、`-425749004`（结果）、`-425767306`（结果 V2）、
`doc-8287337`（原生 ComfyUI 接口）、`doc-8287342`（工作流完整接入示例）、`doc-8287338`（错误码）。

## 八、安全备注

- 本轮使用的 Key 已随用户消息进入会话文本；**建议在平台轮换该 Key**（工作台 → API 调用）。
- Key 未写入仓库任何文件；`~/.runninghub_key` 在仓库外且权限 600。
- 原始文档 HTML 与 32MB `object_info.json` 属第三方内容/超大文件，已 gitignore，仅本地保留。
