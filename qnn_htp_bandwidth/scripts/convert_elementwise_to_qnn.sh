#!/bin/bash

# QNN Element-wise Add 模型转换脚本
# 将 PyTorch 模型转换为 QNN 格式（HTP backend）
# 使用 QNN PyTorch Converter

set -e

# QNN SDK 路径
QNN_SDK_ROOT="/home/yinrun/software/qualcomm/qairt/2.40.0.251030"

# 检查 qairt conda 环境
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda 命令"
    exit 1
fi

# 激活 qairt 环境
echo "激活 qairt conda 环境..."
eval "$(conda shell.bash hook)"
conda activate qairt

# 设置 QNN SDK Python 路径
export PYTHONPATH="$QNN_SDK_ROOT/lib/python:$PYTHONPATH"

# 检查 QNN PyTorch converter
QNN_PYTORCH_CONVERTER="$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-pytorch-converter"
if [ ! -f "$QNN_PYTORCH_CONVERTER" ]; then
    echo "错误: 未找到 QNN PyTorch converter: $QNN_PYTORCH_CONVERTER"
    exit 1
fi

echo "=== 转换 Element-wise Add PyTorch 模型到 QNN 格式 ==="
echo "QNN SDK: $QNN_SDK_ROOT"
echo "PyTorch Converter: $QNN_PYTORCH_CONVERTER"
echo ""

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Element-wise Add模型转换
# 优先使用TorchScript模型（.pt），如果没有则使用完整模型（.pth）
TORCHSCRIPT_MODEL="$PROJECT_ROOT/models/qnn_elementwise_add_model.pt"
PYTORCH_MODEL="$PROJECT_ROOT/models/qnn_elementwise_add_model_full.pth"
OUTPUT_DIR="$PROJECT_ROOT/models/qnn_models_elementwise"
MODEL_NAME="elementwise_add"
INPUT_LIST="$PROJECT_ROOT/data/input_list_elementwise.txt"

# 检查输入文件（优先使用TorchScript）
if [ -f "$TORCHSCRIPT_MODEL" ]; then
    PYTORCH_MODEL="$TORCHSCRIPT_MODEL"
    echo "✓ 使用 TorchScript 模型: $TORCHSCRIPT_MODEL"
elif [ -f "$PYTORCH_MODEL" ]; then
    echo "✓ 使用完整 PyTorch 模型: $PYTORCH_MODEL"
else
    echo "错误: 未找到 PyTorch 模型文件"
    echo "请先运行 qnn_htp_elementwise_bandwidth_model.py 生成 PyTorch 模型"
    exit 1
fi

if [ ! -f "$INPUT_LIST" ]; then
    echo "错误: 未找到输入列表文件: $INPUT_LIST"
    exit 1
fi

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 转换命令
# 使用 HTP backend，INT8量化
# 输入维度: 使用64MB维度
echo "正在转换模型（HTP backend，INT8量化，导出为cpp格式）..."
echo "✓ 使用INT8量化以优化HTP性能"
echo "✓ 输入维度: 1,67108864 (64MB INT8数据)"
# 转换器会生成.bin和.cpp文件，指定基础输出路径
OUTPUT_BASE="$OUTPUT_DIR/$MODEL_NAME"
# 转换器需要在数据目录中运行才能找到输入数据文件
DATA_DIR="$PROJECT_ROOT/data"
cd "$DATA_DIR"
$QNN_PYTORCH_CONVERTER \
    --input_network $PYTORCH_MODEL \
    --input_list $(basename $INPUT_LIST) \
    -d input 1,67108864 \
    -o ${OUTPUT_BASE}.cpp \
    --act_bitwidth 8 \
    --weights_bitwidth 8 \
    --float_bias_bitwidth 8 \
    --float_bitwidth 8 \
    --act_quantizer_calibration min-max \
    --act_quantizer_schema asymmetric \
    --param_quantizer_calibration min-max \
    --param_quantizer_schema asymmetric \
    --overwrite_model_prefix \
    --debug
cd "$PROJECT_ROOT"

if [ $? -eq 0 ]; then
    echo "✓ 模型转换成功"
    echo "输出目录: $OUTPUT_DIR"
    ls -lh $OUTPUT_DIR/
else
    echo "错误: 模型转换失败"
    exit 1
fi

# 检查并重命名cpp文件
CPP_FILE="$OUTPUT_DIR/${MODEL_NAME}.cpp"
CPP_FILE_NO_EXT="$OUTPUT_DIR/${MODEL_NAME}"
if [ -f "$CPP_FILE" ]; then
    echo "✓ 找到cpp文件: $CPP_FILE"
elif [ -f "$CPP_FILE_NO_EXT" ] && [ ! -f "$OUTPUT_DIR/${MODEL_NAME}.bin" ] || [ "$CPP_FILE_NO_EXT" != "$OUTPUT_DIR/${MODEL_NAME}.bin" ]; then
    file_type=$(file -b "$CPP_FILE_NO_EXT" 2>/dev/null | grep -i "c source\|text" || echo "")
    if [ -n "$file_type" ]; then
        echo "重命名cpp文件: $CPP_FILE_NO_EXT -> $CPP_FILE"
        mv "$CPP_FILE_NO_EXT" "$CPP_FILE"
    fi
fi

# 生成 QNN 模型库（.so 文件）
QNN_MODEL_LIB_GENERATOR="$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator"
if [ ! -f "$QNN_MODEL_LIB_GENERATOR" ]; then
    echo "警告: 未找到 QNN Model Lib Generator: $QNN_MODEL_LIB_GENERATOR"
    echo "跳过模型库生成步骤"
else
    if [ -f "$CPP_FILE" ]; then
        echo ""
        echo "正在生成 QNN 模型库..."
        NDK_ROOT="/home/yinrun/Android/Sdk/ndk/29.0.13599879"
        if [ -d "$NDK_ROOT" ] && [ -f "$NDK_ROOT/ndk-build" ]; then
            export ANDROID_NDK_ROOT="$NDK_ROOT"
            export PATH="$NDK_ROOT:$PATH"
            echo "使用NDK生成Android版本..."
            $QNN_MODEL_LIB_GENERATOR \
                -c $CPP_FILE \
                -b "$OUTPUT_DIR/${MODEL_NAME}.bin" \
                -o $OUTPUT_DIR \
                -t aarch64-android

            if [ $? -eq 0 ]; then
                echo "✓ QNN 模型库生成成功"
                ls -lh $OUTPUT_DIR/*.so 2>/dev/null || true

                # 使用 qnn-context-binary-generator 生成 context binary（设置 VTCM 大小为 0）
                QNN_CONTEXT_BINARY_GENERATOR="$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator"
                MODEL_SO="$OUTPUT_DIR/aarch64-android/libelementwise_add.so"
                HTP_BACKEND_SO_X86="$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so"
                VTCM_CONFIG_FILE="$PROJECT_ROOT/configs/htp_vtcm_config.json"

                # 创建配置目录和文件（如果不存在）
                mkdir -p "$PROJECT_ROOT/configs"
                if [ ! -f "$VTCM_CONFIG_FILE" ]; then
                    cat > "$VTCM_CONFIG_FILE" << 'EOF'
{
  "htp_config": {
    "vtcm_mb": 0
  }
}
EOF
                    echo "✓ 创建 VTCM 配置文件: $VTCM_CONFIG_FILE"
                fi

                if [ -f "$QNN_CONTEXT_BINARY_GENERATOR" ] && [ -f "$MODEL_SO" ] && [ -f "$VTCM_CONFIG_FILE" ]; then
                    echo ""
                    echo "正在生成 QNN Context Binary（VTCM 大小设为 0）..."
                    if [ -f "$HTP_BACKEND_SO_X86" ]; then
                        # 注意：由于架构限制，这可能在主机上失败，但配置文件已准备好
                        # 可以在目标设备上运行时使用此配置文件
                        $QNN_CONTEXT_BINARY_GENERATOR \
                            --model "$MODEL_SO" \
                            --backend "$HTP_BACKEND_SO_X86" \
                            --output_dir "$OUTPUT_DIR" \
                            --config_file "$VTCM_CONFIG_FILE" 2>&1 || echo "注意: 在主机上生成失败（架构不匹配），配置文件已准备好，可在设备上使用"

                        if [ $? -eq 0 ]; then
                            echo "✓ Context Binary 生成成功（vtcm_mb=0）"
                        fi
                    else
                        echo "注意: 未找到 x86_64 版本的 HTP backend"
                        echo "      VTCM 配置文件已创建: $VTCM_CONFIG_FILE"
                        echo "      可以在目标设备上使用 qnn-context-binary-generator 生成 context binary"
                    fi
                fi
            else
                echo "警告: QNN 模型库生成失败"
            fi
        else
            echo "警告: 未找到NDK，跳过模型库生成"
        fi
    else
        echo "注意: 未找到 .cpp 文件，无法生成模型库"
    fi
fi

echo ""
echo "✓ 模型转换完成"
echo "输出文件: $OUTPUT_DIR/${MODEL_NAME}.bin"
ls -lh $OUTPUT_DIR/
echo ""
echo "下一步: 运行 build_elementwise_bandwidth_test.sh 编译测试程序"
