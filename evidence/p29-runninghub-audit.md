# P2.9 — RunningHub 真实出图通道审计（只读，0 次调用）

- 日期: 2026-10-08；审计时 HEAD: `9d14a5a`
- 目标组件: `lquesada/ComfyUI-Inpaint-CropAndStitch`（见 `p29-cropandstitch-audit/PROVENANCE.md`）
- 本轮**未注册账号、未申请 Key、未发起任何任务**；下列结论全部来自 RunningHub 公开文档与公开页面。

## 一、平台能力：ComfyUI 工作流 API（已核实）

官方文档（`https://www.runninghub.cn/runninghub-api-doc-cn/api-425749013`，高级接口）明确：

- 请求: `POST https://www.runninghub.cn/task/openapi/create`，`Content-Type: application/json`
- 必填: `apiKey`（用户 API 密钥）、`workflowId`（工作流模板 ID）、`nodeInfoList`
- `nodeInfoList` 元素: `{nodeId, fieldName, fieldValue}` —— 即**逐节点、逐字段**改参：
  - `nodeId` 节点编号（来自工作流 JSON）、`fieldName` 输入 key（如 `text`/`seed`/`steps`）、`fieldValue` 新值
- 可选: `accessPassword`、`addMetadata`(默认 true)、`webhookUrl`、`instanceType`(`default` 24G / `plus` 48G / `ultra` 84G)、`usePersonalQueue`、`retainSeconds`(10~180s)
- ★ 可选 `workflow`（完整工作流 JSON；**指定后忽略 `workflowId`**）—— 即可以**内联提交整张 ComfyUI 图**，无需预先在平台保存模板
- 配套接口（同批文档）: `/task/openapi/upload`（上传资源取 `fileName`）、`/task/openapi/status`、`/task/openapi/outputs`、`/task/openapi/cancel`、`/api/openapi/getJsonApiFormat`（取工作流 JSON）
- 另有 `POST /task/openapi/ai-app/run`（按 AI 应用 ID 运行），用于平台已上架的 AI 应用

图片来源（官方《关于nodeInfoList》）：① `LoadImageFromUrl` 节点传 URL；② 用上传接口拿 `fileName` 喂 `LoadImage`。
我们在 `https://www.runninghub.cn/ai-detail/1894616400458330114` 的 SSR 内嵌节点清单中核实了该寻址方式真实存在
（该页真实节点含 `nodeId 23 / LoadImage / fieldName "image"`，与文档一致）。

另有两份官方文档：`api-425749012`（简易接口）、`api-425749014`（获取工作流 JSON）、`api-425749015`（取消任务）；
以及《原生 ComfyUI 接口支持说明》——平台提供 `https://www.runninghub.cn/proxy/<your-api-key>`
（24G，`/proxy-plus/<key>` 为 48G），行为等同本地 `http://127.0.0.1:8188`。

## 二、与本地标准包的映射（若走自建工作流）

| 我们的产物 | 目标节点 | `fieldName` | 取值方式 |
| --- | --- | --- | --- |
| `original_image` | `LoadImage`(图像) | `image` | `/task/openapi/upload` 返回的 `fileName` |
| `hair_edit_mask` / `clothing_edit_mask` | `LoadImage`(蒙版) | `image` | 同上（两张 255 白/0 黑 PNG） |
| `prompt_bundle.hair_prompt` / `clothing_prompt` | `CLIPTextEncode` | `text` | 直接字符串 |
| `negative_prompt` | `CLIPTextEncode`(negative) | `text` | 直接字符串 |
| 两步顺序 | `InpaintCropImproved` → 采样 → `InpaintStitchImproved` | — | 第 2 步的 `image` 输入接第 1 步输出 |

关键前提：工作流里必须**已存在** `InpaintCropImproved` / `InpaintStitchImproved` 两个自定义节点。

## 三、未能核实的事项（阻塞点）

1. **平台上不存在「已核实可调用」的现成 CropAndStitch 工作流。**
   公开的 AI 应用 / 工作流列表页（`/ai-apps`、`/page-workflow`、`/search`）都是**客户端渲染**，
   SSR HTML 里没有条目清单（实测：`/page-workflow` 与 `/search?keyword=局部重绘` 均无 SSR 结果）；
   驱动这些列表的是 `/api/` 下的接口，而 `robots.txt` 明确 `Disallow: /api/`，未抓取。
   因此**无法在不登录的前提下枚举平台工作流库**，也就无法确认任何一个 `workflowId` 的真实节点输入。
2. **`ComfyUI-Inpaint-CropAndStitch` 是否已在 RunningHub 运行时安装，未核实。**
   官方文档只说「模板缺少**模型**可上传收藏」，没有公开自定义节点白名单/自动安装说明。
3. **单次费用未核实。** 计费口径要在登录后的账户/任务接口（`/api-425748943` 获取账户信息、`/api-432926239` 队列状态）才能读到；本轮无 Key，未读取。
4. 平台另有 `instanceType` 档位与 `retainSeconds` 计费项，均为账户侧信息。

## 四、结论

- **机制可行**：RunningHub 的 `/task/openapi/create` 支持「原图 + 外部 mask + prompt 输入 + 逐节点改参 + 可内联整张工作流 JSON」，
  与我们的本地标准包契约一一对应；`InpaintCropImproved/InpaintStitchImproved` 的「只改 mask 区、拼回原坐标系」语义也满足要求。
- **但按验收标准（必须核对真实节点输入、必须给出确定的 `workflowId`）判定：BLOCKED。**
  阻塞原因是**账户侧信息不可得**，不是平台不支持：
  1. 无 RunningHub 账号 / API Key → 无法读取工作流库与账户信息；
  2. 平台工作流清单非 SSR，`/api/` 被 robots 禁止抓取 → 无法离线枚举；
  3. 现成工作流是否含 CropAndStitch 节点、单次费用，均需登录后核对。

## 五、解锁所需（最小集）

1. 用户提供 RunningHub 账号并生成 API Key（`Authorization: Bearer` / 请求体 `apiKey`）。
2. 确认 `ComfyUI-Inpaint-CropAndStitch` 在该账号可用（导入一张含该节点的最小工作流并保存，取回 `workflowId`）。
3. 用 `/api/openapi/getJsonApiFormat` 拉回该工作流 JSON，**逐节点核对**输入是否与第二节映射一致。
4. 再用 `获取账户信息` / `队列状态` 读到真实计费口径，才可以申请一次付费授权。

> 本轮**不新建适配器、不接新平台**；以上仅为解锁清单，未执行。
