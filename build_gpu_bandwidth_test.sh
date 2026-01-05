#!/bin/bash

set -e

# 配置
NDK_ROOT="/Users/yinrun/Library/Android/sdk/ndk/29.0.13113456"
TOOLCHAIN="$NDK_ROOT/toolchains/llvm/prebuilt/darwin-x86_64"
CXX="$TOOLCHAIN/bin/aarch64-linux-android21-clang++"

# 构建目录
BUILD_DIR="build"
mkdir -p $BUILD_DIR

# 文件配置
TARGET="$BUILD_DIR/gpu_bandwidth_test"
SRC="gpu_bandwidth_test.cpp"

# 编译参数
CXXFLAGS="-std=c++17 -Wall -O2 -fPIC -stdlib=libc++"
INCLUDES="-I$(pwd)/include"
LIBS="-L$(pwd)/libs -lOpenCL -ldl"

echo "=== 编译带宽测试程序 ==="
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
adb push gpu_bandwidth_test.cl /data/local/tmp/gpu_bandwidth_test.cl > /dev/null
adb shell chmod 755 /data/local/tmp/$(basename $TARGET) > /dev/null
echo "✓ 推送成功"
echo ""

# 运行测试
echo "=== 运行 GPU 带宽测试 ==="
echo "用法: gpu_bandwidth_test [数据大小MB] [迭代次数]"
echo "默认: 1024MB (1GB), 10次迭代"
echo ""
adb shell "cd /data/local/tmp && export LD_LIBRARY_PATH=/data/local/tmp:\$LD_LIBRARY_PATH && /data/local/tmp/$(basename $TARGET) $@"
