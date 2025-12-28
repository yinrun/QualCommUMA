#!/bin/bash

# Android NDK 配置
export ANDROID_NDK_HOME="/opt/homebrew/share/android-ndk"
NDK_ROOT="$ANDROID_NDK_HOME"

# 设备架构 (arm64-v8a)
ANDROID_ABI="arm64-v8a"
ANDROID_PLATFORM="android-21"

# 工具链路径
TOOLCHAIN="$NDK_ROOT/toolchains/llvm/prebuilt/darwin-x86_64"
CXX="$TOOLCHAIN/bin/aarch64-linux-android${ANDROID_PLATFORM#android-}-clang++"

# 检查工具链是否存在
if [ ! -f "$CXX" ]; then
    echo "错误: 找不到编译器 $CXX"
    echo "请检查 NDK 安装路径: $NDK_ROOT"
    exit 1
fi

echo "=== 编译 OpenCL UMA Demo ==="
echo "NDK 路径: $NDK_ROOT"
echo "编译器: $CXX"
echo "目标架构: $ANDROID_ABI"
echo ""

# 编译参数
CXXFLAGS="-std=c++11 -Wall -O2 -fPIC"
INCLUDES="-I$(pwd)/include"
LIBS="-L$(pwd)/libs -lOpenCL"
TARGET="uma_demo"
SRC="uma_demo.cpp"

# 如果指定了 no_unmap 参数，编译测试版本
if [ "$1" == "no_unmap" ]; then
    TARGET="uma_demo_no_unmap"
    SRC="uma_demo_no_unmap.cpp"
fi

# 编译
echo "正在编译..."
$CXX $CXXFLAGS $INCLUDES -o $TARGET $SRC $LIBS

if [ $? -ne 0 ]; then
    echo "编译失败！"
    exit 1
fi

echo "编译成功: $TARGET"
echo ""

# 检查设备连接
DEVICE=$(adb devices | grep -w device | awk '{print $1}')
if [ -z "$DEVICE" ]; then
    echo "错误: 未找到已连接的设备"
    exit 1
fi

echo "设备 ID: $DEVICE"
echo ""

# 推送可执行文件到设备
echo "正在推送可执行文件到设备..."
adb push $TARGET /data/local/tmp/$TARGET
adb shell chmod 755 /data/local/tmp/$TARGET

if [ $? -ne 0 ]; then
    echo "推送失败！"
    exit 1
fi

echo "推送成功"
echo ""

# 运行测试
echo "=== 在设备上运行 UMA Demo ==="
adb shell "LD_LIBRARY_PATH=/vendor/lib64:/system/lib64 /data/local/tmp/$TARGET"

echo ""
echo "=== 完成 ==="

