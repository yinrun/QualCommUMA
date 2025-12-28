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

echo "=== 编译 QNN UMA 测试程序 ==="
echo "NDK 路径: $NDK_ROOT"
echo "编译器: $CXX"
echo "目标架构: $ANDROID_ABI"
echo ""

# QNN SDK 路径
QNN_SDK_ROOT="/Users/lvyinrun/workspace/qairt/2.41.0.251128"

# 编译参数
CXXFLAGS="-std=c++11 -Wall -O2 -fPIC"
INCLUDES="-I$(pwd)/include -I$QNN_SDK_ROOT/include/QNN"
LIBS="-ldl"  # 链接 dlopen

# 默认使用 real 版本（真正使用 QNN SDK API）
TARGET="qnn_uma_real_demo"
SRC="qnn_uma_real_demo.cpp"
OP_PACKAGE_SRC="custom_multiply_op_package.cpp"

# 如果指定了参数，使用对应版本
if [ "$1" == "real" ] || [ -z "$1" ]; then
    TARGET="qnn_uma_real_demo"
    SRC="qnn_uma_real_demo.cpp"
    OP_PACKAGE_SRC="custom_multiply_op_package.cpp"
elif [ "$1" == "operator" ]; then
    TARGET="qnn_uma_operator_demo"
    SRC="qnn_uma_operator_demo.cpp"
    OP_PACKAGE_SRC=""
    INCLUDES="-I$(pwd)/include"  # operator 版本不需要 QNN SDK 头文件
elif [ -n "$1" ]; then
    echo "未知参数: $1"
    echo "使用方法: $0 [real|operator]"
    echo "  real: 使用真实 QNN SDK API 版本（默认）"
    echo "  operator: 使用简化版本（CPU 模拟算子）"
    exit 1
fi

# 编译
echo "正在编译..."
if [ -n "$OP_PACKAGE_SRC" ]; then
    # 编译包含 OpPackage 模块的版本
    $CXX $CXXFLAGS $INCLUDES -o $TARGET $SRC $OP_PACKAGE_SRC $LIBS
else
    # 编译不包含 OpPackage 的版本
    $CXX $CXXFLAGS $INCLUDES -o $TARGET $SRC $LIBS
fi

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
echo "=== 在设备上运行 QNN UMA 测试 ==="
adb shell "/data/local/tmp/$TARGET"

echo ""
echo "=== 完成 ==="


