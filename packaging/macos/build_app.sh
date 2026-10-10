#!/bin/zsh
# 组装「图片工作台.app」并安装到 /Applications。
#
# 用法：
#   ./packaging/macos/build_app.sh              # 生成 + 安装到 /Applications
#   ./packaging/macos/build_app.sh --no-install # 只在 packaging/macos/build 下生成
set -euo pipefail

PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$PKG_DIR/../.." && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"
APP_NAME="图片工作台"
BUILD_DIR="$PKG_DIR/build"
APP_BUNDLE="$BUILD_DIR/$APP_NAME.app"
INSTALL_DIR="/Applications"
INSTALL=false
[ "${1:-}" = "--no-install" ] || INSTALL=true

echo "▶ 生成图标"
"$PYTHON" "$PKG_DIR/make_icon.py" "$BUILD_DIR" >/dev/null

echo "▶ 组装 bundle"
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources"

cp "$PKG_DIR/launcher.sh" "$APP_BUNDLE/Contents/MacOS/AIBeautyStyleLab"
chmod 755 "$APP_BUNDLE/Contents/MacOS/AIBeautyStyleLab"
cp "$PKG_DIR/app_runtime.py" "$APP_BUNDLE/Contents/Resources/app_runtime.py"
cp "$BUILD_DIR/AppIcon.icns" "$APP_BUNDLE/Contents/Resources/AppIcon.icns"

cat > "$APP_BUNDLE/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key>
	<string>zh_CN</string>
	<key>CFBundleDisplayName</key>
	<string>图片工作台</string>
	<key>CFBundleExecutable</key>
	<string>AIBeautyStyleLab</string>
	<key>CFBundleIconFile</key>
	<string>AppIcon</string>
	<key>CFBundleIdentifier</key>
	<string>lab.image.workbench</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundleName</key>
	<string>图片工作台</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>0.3.0</string>
	<key>CFBundleVersion</key>
	<string>1</string>
	<key>LSMinimumSystemVersion</key>
	<string>11.0</string>
	<key>LSMultipleInstancesProhibited</key>
	<true/>
	<key>NSAppTransportSecurity</key>
	<dict>
		<key>NSAllowsLocalNetworking</key>
		<true/>
	</dict>
	<key>NSHighResolutionCapable</key>
	<true/>
	<key>NSPrincipalClass</key>
	<string>NSApplication</string>
</dict>
</plist>
PLIST

/usr/bin/codesign --force --deep --sign - "$APP_BUNDLE" >/dev/null 2>&1 \
  && echo "  已附加 ad-hoc 签名" \
  || echo "  （跳过签名，不影响本机运行）"

if [ "$INSTALL" = true ]; then
  echo "▶ 安装到 $INSTALL_DIR"
  rm -rf "$INSTALL_DIR/$APP_NAME.app"
  /usr/bin/ditto "$APP_BUNDLE" "$INSTALL_DIR/$APP_NAME.app"
  /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
    -f "$INSTALL_DIR/$APP_NAME.app" >/dev/null 2>&1 || true
  echo "✅ 完成：$INSTALL_DIR/$APP_NAME.app"
else
  echo "✅ 完成：$APP_BUNDLE"
fi
