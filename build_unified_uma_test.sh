#!/bin/bash

set -e  # 遇到错误立即退出

# 配置
NDK_ROOT="/Users/yinrun/Library/Android/sdk/ndk/29.0.13113456"
QNN_SDK_ROOT="/Users/yinrun/Workspace/qairt/2.40.0.251030"
TOOLCHAIN="$NDK_ROOT/toolchains/llvm/prebuilt/darwin-x86_64"
CXX="$TOOLCHAIN/bin/aarch64-linux-android21-clang++"

# 不再需要 Hexagon SDK，因为使用 QNN 内置的 element-wise 算子

# 构建目录
BUILD_DIR="build"
mkdir -p $BUILD_DIR

# 文件配置
TARGET="$BUILD_DIR/unified_uma_demo"
SRC="unified_uma_demo.cpp"

# 编译参数
CXXFLAGS="-std=c++17 -Wall -O2 -fPIC -stdlib=libc++"
INCLUDES="-I$(pwd)/include -I$QNN_SDK_ROOT/include/QNN"
LIBS="-L$(pwd)/libs -lOpenCL -ldl"

echo "=== 编译统一 UMA 测试程序 ==="
echo "使用 QNN 内置 ElementWiseMultiply 算子（无需自定义 OpPackage）"
echo ""

# 编译主程序
echo "编译主程序..."
$CXX $CXXFLAGS $INCLUDES -o $TARGET $SRC $LIBS
echo "✓ $TARGET"
echo ""

# 检查设备连接
DEVICE=$(adb devices | grep -w device | awk '{print $1}' || echo "")
[ -z "$DEVICE" ] && { echo "错误: 未找到已连接的设备"; exit 1; }
echo "设备: $DEVICE"
echo ""

# 推送文件到设备
echo "推送文件到设备..."
adb push $TARGET /data/local/tmp/$(basename $TARGET) > /dev/null
adb push fill_array.cl /data/local/tmp/fill_array.cl > /dev/null
adb shell chmod 755 /data/local/tmp/$(basename $TARGET) > /dev/null
echo "✓ 推送成功"
echo ""

# 运行测试
echo "=== 运行测试 ==="
adb shell "cd /data/local/tmp && export LD_LIBRARY_PATH=/data/local/tmp:\$LD_LIBRARY_PATH && /data/local/tmp/$(basename $TARGET)"
