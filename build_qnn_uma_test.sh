#!/bin/bash

set -e  # 遇到错误立即退出

# 配置
NDK_ROOT="/home/yinrun/Android/Sdk/ndk/29.0.13599879"
QNN_SDK_ROOT="/home/yinrun/software/qualcomm/qairt/2.40.0.251030"
TOOLCHAIN="$NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64"
CXX="$TOOLCHAIN/bin/aarch64-linux-android21-clang++"

# 构建目录
BUILD_DIR="build"
mkdir -p $BUILD_DIR

# 文件配置
OP_PACKAGE_LIB="$BUILD_DIR/libCustomOpPackage.so"
OP_PACKAGE_SRC="custom_op_package.cpp"
TARGET="$BUILD_DIR/qnn_uma_demo"
SRC="qnn_uma_demo.cpp"

# 编译参数
CXXFLAGS="-std=c++11 -Wall -O2 -fPIC"
INCLUDES="-I$(pwd)/include -I$QNN_SDK_ROOT/include/QNN"
LIBS="-ldl"

echo "=== 编译 QNN UMA 测试程序 ==="

# 编译 OpPackage 共享库
echo "编译 OpPackage 共享库..."
$CXX $CXXFLAGS $INCLUDES -shared -o $OP_PACKAGE_LIB $OP_PACKAGE_SRC $LIBS
echo "✓ $OP_PACKAGE_LIB"

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
adb push $OP_PACKAGE_LIB /data/local/tmp/$(basename $OP_PACKAGE_LIB) > /dev/null
adb shell chmod 755 /data/local/tmp/$(basename $TARGET) /data/local/tmp/$(basename $OP_PACKAGE_LIB) > /dev/null
echo "✓ 推送成功"
echo ""

# 运行测试
echo "=== 运行测试 ==="
adb shell "export LD_LIBRARY_PATH=/data/local/tmp:\$LD_LIBRARY_PATH && /data/local/tmp/$(basename $TARGET)"
