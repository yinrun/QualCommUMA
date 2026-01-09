#!/bin/bash

# QNN HTP Element-wise Add 逐步验证脚本
# 用于验证每个步骤是否正常工作

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== QNN HTP Element-wise Add 逐步验证 ==="
echo ""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查设备连接
echo "步骤 1: 检查设备连接..."
DEVICE=$(adb devices | grep -w device | awk '{print $1}' || echo "")
if [ -z "$DEVICE" ]; then
    echo -e "${RED}✗ 未找到已连接的设备${NC}"
    echo "请连接Android设备后重试"
    exit 1
fi
echo -e "${GREEN}✓ 设备已连接: $DEVICE${NC}"
echo ""

# 检查模型文件
echo "步骤 2: 检查模型文件..."
MODEL_SO="/data/local/tmp/qnn_models_elementwise/aarch64-android/libelementwise_add.so"
if adb shell "test -f $MODEL_SO" 2>/dev/null; then
    echo -e "${GREEN}✓ 模型文件存在: $MODEL_SO${NC}"
    adb shell "ls -lh $MODEL_SO"
else
    echo -e "${YELLOW}⚠ 模型文件不存在，需要先推送${NC}"
    echo "运行: ./scripts/build_elementwise_bandwidth_test.sh"
    exit 1
fi
echo ""

# 检查可执行文件
echo "步骤 3: 检查可执行文件..."
EXECUTABLE="/data/local/tmp/qnn_htp_elementwise_bandwidth_test"
if adb shell "test -f $EXECUTABLE" 2>/dev/null; then
    echo -e "${GREEN}✓ 可执行文件存在: $EXECUTABLE${NC}"
    adb shell "ls -lh $EXECUTABLE"
    adb shell "file $EXECUTABLE"
else
    echo -e "${YELLOW}⚠ 可执行文件不存在，需要先编译和推送${NC}"
    echo "运行: ./scripts/build_elementwise_bandwidth_test.sh"
    exit 1
fi
echo ""

# 检查库文件
echo "步骤 4: 检查必要的库文件..."
LIBS=(
    "/vendor/lib64/libcdsprpc.so"
    "/vendor/lib64/libQnnHtp.so"
    "/system/lib64/libcdsprpc.so"
)

for lib in "${LIBS[@]}"; do
    if adb shell "test -f $lib" 2>/dev/null; then
        echo -e "${GREEN}✓ 库文件存在: $lib${NC}"
    else
        echo -e "${YELLOW}⚠ 库文件不存在: $lib${NC}"
    fi
done
echo ""

# 测试512MB数据（3次迭代）
echo "步骤 5: 运行512MB测试（3次迭代）..."
echo "这用于验证基本功能是否正常"
echo ""

adb shell "cd /data/local/tmp && export LD_LIBRARY_PATH=/vendor/lib64:/system/lib64:\$LD_LIBRARY_PATH && $EXECUTABLE 512 3" || {
    echo -e "${RED}✗ 512MB测试失败${NC}"
    echo "请检查错误信息"
    exit 1
}

echo ""
echo -e "${GREEN}✓ 512MB测试成功${NC}"
echo ""
echo -e "${GREEN}✓ 逐步验证完成${NC}"
echo ""
echo "测试配置：512MB数据，3次迭代"
