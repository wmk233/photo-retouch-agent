# macOS 应用包（图片工作台.app）

把本地工作台封装成一个可以双击运行的原生 macOS 应用：启动时自动拉起本地
FastAPI 服务，并在原生窗口（WKWebView）里打开界面；关闭窗口即停止服务。

应用默认落在「生图工作台」，页面顶部切换器可进入「美颜工作台」。

## 产物

| 位置 | 说明 |
| --- | --- |
| `/Applications/图片工作台.app` | 安装后的应用，可从启动台 / 聚焦搜索直接打开 |
| `packaging/macos/build/` | 构建中间产物（图标、未安装的 bundle） |

日志：

- `~/Library/Logs/图片工作台/app.log` —— 服务与窗口日志
- `~/Library/Logs/图片工作台/launcher.log` —— 启动器日志

## 重新构建

```bash
./packaging/macos/build_app.sh              # 生成并安装到 /Applications
./packaging/macos/build_app.sh --no-install # 只生成到 packaging/macos/build
```

## 运行依赖

- 应用通过 `~/AI-beauty-style-lab/repos/photo-retouch-agent/.venv/bin/python` 运行，
  仓库换路径时用环境变量 `BEAUTY_LAB_REPO` 覆盖（启动器会读取）。
- 原生窗口依赖 `.venv` 中的 `pywebview`（含 pyobjc）：

  ```bash
  uv pip install --python .venv/bin/python pywebview
  ```

  若该依赖缺失，应用会自动退回到「默认浏览器打开界面」模式，功能不受影响。
- 端口按 `8000 → 8015` 顺序尝试：已在运行的服务会被复用，不会重复启动。
- 起始页为 `/studio.html`（生图工作台），切换器指向 `index.html`（美颜工作台）。

## 组成

- `launcher.sh` —— bundle 的主可执行文件，负责校验环境并拉起 `app_runtime.py`
- `app_runtime.py` —— 启动/复用本地服务，创建原生窗口，退出时回收服务
- `make_icon.py` —— 生成 `AppIcon.icns`（纯 Pillow 绘制，无外部素材）
- `build_app.sh` —— 组装 bundle、写入 `Info.plist`、ad-hoc 签名并安装
