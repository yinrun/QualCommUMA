#!/bin/bash

set -e  # 遇到错误立即退出

# 配置
NDK_ROOT="/Users/yinrun/Library/Android/sdk/ndk/29.0.13113456"
TOOLCHAIN="$NDK_ROOT/toolchains/llvm/prebuilt/darwin-x86_64"
CXX="$TOOLCHAIN/bin/aarch64-linux-android21-clang++"

# 构建目录
BUILD_DIR="build"
mkdir -p $BUILD_DIR

# 文件配置
TARGET="$BUILD_DIR/opencl_uma_demo"
SRC="opencl_uma_demo.cpp"

# 编译参数
CXXFLAGS="-std=c++11 -Wall -O2 -fPIC"
INCLUDES="-I$(pwd)/include"
LIBS="-L$(pwd)/libs -lOpenCL"

echo "=== 编译 OpenCL UMA Demo ==="

# 编译
echo "编译 OpenCL UMA Demo..."
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
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/vendor/lib64:/system/lib64 /data/local/tmp/$(basename $TARGET)"
