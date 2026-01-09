#!/bin/bash

set -e

# 配置
NDK_ROOT="/home/yinrun/Android/Sdk/ndk/29.0.13599879"
QNN_SDK_ROOT="/home/yinrun/software/qualcomm/qairt/2.40.0.251030"
TOOLCHAIN="$NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64"
CXX="$TOOLCHAIN/bin/aarch64-linux-android21-clang++"

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 构建目录
BUILD_DIR="build"
mkdir -p $BUILD_DIR

# 文件配置
TARGET="$BUILD_DIR/qnn_htp_elementwise_bandwidth_test"
SRC="$PROJECT_ROOT/src/qnn_htp_elementwise_bandwidth_test.cpp"

# QNN SDK 头文件路径
QNN_INCLUDE="$QNN_SDK_ROOT/include/QNN"

# 编译参数
CXXFLAGS="-std=c++11 -Wall -O2 -fPIC"
INCLUDES="-I$QNN_INCLUDE"
LIBS="-ldl -lm"

echo "=== 编译 QNN HTP Element-wise Add 带宽测试程序 ==="
echo ""

# 检查 QNN SDK 头文件
if [ ! -d "$QNN_INCLUDE" ]; then
    echo "错误: 未找到 QNN SDK 头文件目录: $QNN_INCLUDE"
    exit 1
fi

# 检查源文件
if [ ! -f "$SRC" ]; then
    echo "错误: 未找到源文件: $SRC"
    exit 1
fi

# 编译主程序
echo "编译主程序..."
echo "  源文件: $SRC"
echo "  目标文件: $TARGET"
echo "  编译器: $CXX"
echo ""

$CXX $CXXFLAGS $INCLUDES -o $TARGET $SRC $LIBS

if [ $? -eq 0 ]; then
    echo "✓ 编译成功: $TARGET"
    file $TARGET
    echo ""
else
    echo "错误: 编译失败"
    exit 1
fi

# 检查设备连接
DEVICE=$(adb devices | grep -w device | awk '{print $1}' || echo "")
if [ -z "$DEVICE" ]; then
    echo "警告: 未找到已连接的设备"
    echo "请连接设备后手动推送文件:"
    echo "  adb push $TARGET /data/local/tmp/$(basename $TARGET)"
    exit 0
fi

echo "设备: $DEVICE"
echo ""

# 推送文件到设备
echo "推送文件到设备..."
adb push $TARGET /data/local/tmp/$(basename $TARGET) > /dev/null
echo "✓ 推送可执行文件成功"

# 推送 QNN 模型文件（如果存在）
QNN_MODEL_DIR="$PROJECT_ROOT/models/qnn_models_elementwise"
if [ -d "$QNN_MODEL_DIR" ]; then
    echo "推送 QNN 模型文件..."
    adb shell "mkdir -p /data/local/tmp/qnn_models_elementwise/aarch64-android" > /dev/null
    adb push "$QNN_MODEL_DIR"/*.bin /data/local/tmp/qnn_models_elementwise/ > /dev/null 2>&1 || true
    adb push "$QNN_MODEL_DIR"/*context*.bin /data/local/tmp/qnn_models_elementwise/ > /dev/null 2>&1 || true
    # 优先推送aarch64-android版本
    if [ -f "$QNN_MODEL_DIR/aarch64-android/libelementwise_add.so" ]; then
        adb shell "mkdir -p /data/local/tmp/qnn_models_elementwise/aarch64-android" > /dev/null 2>&1
        adb push "$QNN_MODEL_DIR/aarch64-android/libelementwise_add.so" /data/local/tmp/qnn_models_elementwise/aarch64-android/libelementwise_add.so > /dev/null 2>&1 || true
    fi
    # 如果没有aarch64版本，尝试x86_64版本
    if [ -d "$QNN_MODEL_DIR/x86_64-linux-clang" ]; then
        adb shell "mkdir -p /data/local/tmp/qnn_models_elementwise/x86_64-linux-clang" > /dev/null
        adb push "$QNN_MODEL_DIR/x86_64-linux-clang"/*.so /data/local/tmp/qnn_models_elementwise/x86_64-linux-clang/ > /dev/null 2>&1 || true
    fi
    adb push "$QNN_MODEL_DIR"/*.json /data/local/tmp/qnn_models_elementwise/ > /dev/null 2>&1 || true
    echo "✓ 推送模型文件成功"
fi

adb shell chmod 755 /data/local/tmp/$(basename $TARGET) > /dev/null
echo ""

# 运行测试
echo "=== 运行 QNN HTP Element-wise Add 带宽测试 ==="
echo "用法: qnn_htp_elementwise_bandwidth_test [数据大小MB] [迭代次数] [模型路径]"
echo "默认: 512MB, 3次迭代, /data/local/tmp/qnn_models_elementwise/aarch64-android/libelementwise_add.so"
echo ""
echo "示例:"
echo "  ./build_elementwise_bandwidth_test.sh 512 3  # 512MB, 3次迭代"
echo ""

# 如果提供了参数，则运行测试
if [ $# -gt 0 ]; then
    # 传递环境变量（如果设置了QNN_MEM_MODE）
    ENV_VARS=""
    if [ -n "$QNN_MEM_MODE" ]; then
        ENV_VARS="export QNN_MEM_MODE=$QNN_MEM_MODE && "
    fi
    adb shell "cd /data/local/tmp && export LD_LIBRARY_PATH=/vendor/lib64:/system/lib64:\$LD_LIBRARY_PATH && $ENV_VARS /data/local/tmp/$(basename $TARGET) $@"
else
    echo "提示: 运行以下命令执行测试:"
    echo "  QNN_MEM_MODE=RAW ./build_elementwise_bandwidth_test.sh 1 1000  # RAW模式"
    echo "  QNN_MEM_MODE=MEMHANDLE ./build_elementwise_bandwidth_test.sh 1 1000  # MEMHANDLE模式"
fi
