#!/usr/bin/env python3
"""图片工作台 —— macOS 原生桌面外壳。

双击「图片工作台.app」后：
  1. 复用或启动本地 FastAPI 服务（127.0.0.1）；
  2. 打开原生窗口（WKWebView）加载工作台界面；
  3. 关闭窗口 / 按 ⌘Q 时，若服务由本进程启动则一并关闭。

日志：~/Library/Logs/图片工作台/app.log
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

APP_NAME = "图片工作台"
DEFAULT_REPO = Path.home() / "AI-beauty-style-lab" / "repos" / "photo-retouch-agent"
HOST = "127.0.0.1"
LANDING_PAGE = "studio.html"
CANDIDATE_PORTS = tuple(range(8000, 8016))
READY_TIMEOUT_SECONDS = 240.0

REPO_DIR = Path(os.environ.get("BEAUTY_LAB_REPO", str(DEFAULT_REPO))).expanduser().resolve()
BACKEND_DIR = REPO_DIR / "backend"
LOG_DIR = Path.home() / "Library" / "Logs" / APP_NAME
LOG_PATH = LOG_DIR / "app.log"

SPLASH_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; }
  body {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 26px; font-family: -apple-system, "PingFang SC", "Helvetica Neue", sans-serif;
    background: radial-gradient(120% 120% at 50% 0%, #2b2440 0%, #16121f 55%, #0d0b13 100%);
    color: #f4f1ff;
  }
  .mark {
    width: 108px; height: 108px; border-radius: 28px;
    display: flex; align-items: center; justify-content: center;
    font-size: 40px; font-weight: 700; letter-spacing: 1px;
    background: linear-gradient(140deg, #b48cff 0%, #7c5cff 45%, #4a2fd6 100%);
    box-shadow: 0 18px 48px rgba(124, 92, 255, .42);
  }
  h1 { margin: 0; font-size: 22px; font-weight: 600; letter-spacing: .4px; }
  p { margin: 0; font-size: 13px; color: rgba(244, 241, 255, .62); }
  .bar { width: 220px; height: 4px; border-radius: 99px; background: rgba(255,255,255,.12); overflow: hidden; }
  .bar span {
    display: block; width: 40%; height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #b48cff, #7c5cff);
    animation: slide 1.25s ease-in-out infinite;
  }
  @keyframes slide { 0% { transform: translateX(-110%); } 100% { transform: translateX(320%); } }
</style>
</head>
<body>
  <div class="mark">AB</div>
  <h1>图片工作台</h1>
  <div class="bar"><span></span></div>
  <p id="stage">正在启动本地服务…</p>
  <script>
    window.__setStage = function (text) { document.getElementById('stage').textContent = text; };
  </script>
</body>
</html>
"""


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def redirect_output() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stream = open(LOG_PATH, "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = stream
    sys.stderr = stream


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((HOST, port))
        except OSError:
            return False
    return True


def health_ok(port: int, timeout: float = 0.8) -> bool:
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/api/health", timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


class LocalService:
    """本地 Web 服务：可能复用已有实例，也可能由本进程启动。"""

    def __init__(self, port: int, owned: bool, server: object | None = None) -> None:
        self.port = port
        self.owned = owned
        self.server = server

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}/"

    @property
    def page_url(self) -> str:
        """外壳默认落在生图工作台；美颜工作台由页面顶部的切换器进入。"""
        return f"{self.url}{LANDING_PAGE}"

    def wait_until_ready(self, on_tick=None) -> bool:
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if health_ok(self.port):
                return True
            if self.owned and getattr(self.server, "should_exit", False):
                return False
            if on_tick:
                on_tick()
            time.sleep(0.4)
        return False

    def shutdown(self) -> None:
        if self.owned and self.server is not None:
            self.server.should_exit = True


def start_service() -> LocalService:
    if not BACKEND_DIR.exists():
        raise RuntimeError(f"找不到后端目录：{BACKEND_DIR}")

    for port in CANDIDATE_PORTS:
        if health_ok(port):
            log(f"复用已在运行的服务：http://{HOST}:{port}/")
            return LocalService(port, owned=False)
        if port_is_free(port):
            return _launch_service(port)
    raise RuntimeError(f"{CANDIDATE_PORTS[0]}–{CANDIDATE_PORTS[-1]} 端口全部被占用。")


def _launch_service(port: int) -> LocalService:
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    os.chdir(BACKEND_DIR)

    import uvicorn

    from app.main import app as fastapi_app

    config = uvicorn.Config(
        fastapi_app,
        host=HOST,
        port=port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="beauty-lab-server", daemon=True)
    thread.start()
    log(f"已启动本地服务：http://{HOST}:{port}/ （pid={os.getpid()}）")
    return LocalService(port, owned=True, server=server)


def open_in_browser_fallback(service: LocalService) -> None:
    import subprocess

    subprocess.run(["open", service.url], check=False)
    log("未安装原生窗口组件（pywebview），已在默认浏览器中打开界面。")
    try:
        while not getattr(service.server, "should_exit", True):
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        service.shutdown()


def main() -> int:
    redirect_output()
    log("=" * 56)
    log(f"启动 {APP_NAME} · 仓库 {REPO_DIR}")

    try:
        import webview
    except Exception as error:  # noqa: BLE001 - 任何导入失败都退回到浏览器
        log(f"pywebview 不可用（{error}），使用浏览器兜底模式。")
        service = start_service()
        service.wait_until_ready()
        open_in_browser_fallback(service)
        return 0

    window = webview.create_window(
        APP_NAME,
        html=SPLASH_HTML,
        width=1480,
        height=960,
        min_size=(1080, 720),
        background_color="#16121f",
        text_select=False,
        zoomable=False,
    )

    state: dict[str, object] = {"service": None}

    def boot() -> None:
        try:
            window.evaluate_js("window.__setStage && window.__setStage('正在启动本地服务…')")
            service = start_service()
            state["service"] = service
            hint = "正在连接已有服务…" if not service.owned else "正在加载生图工作台…"
            window.evaluate_js(f"window.__setStage && window.__setStage({hint!r})")
            if not service.wait_until_ready():
                window.evaluate_js(
                    "window.__setStage && window.__setStage('服务启动失败，请查看日志后重试')"
                )
                log("服务未能在超时时间内就绪。")
                return
            log(f"服务就绪，加载界面：{service.page_url}")
            window.load_url(service.page_url)
        except Exception as error:  # noqa: BLE001
            log(f"启动失败：{error!r}")
            safe = str(error).replace("\\", "\\\\").replace("'", "\\'")
            window.evaluate_js(f"window.__setStage && window.__setStage('启动失败：{safe}')")

    try:
        webview.start(boot)
    finally:
        service = state.get("service")
        if isinstance(service, LocalService):
            service.shutdown()
        log(f"{APP_NAME} 已退出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
