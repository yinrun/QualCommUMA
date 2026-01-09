#!/usr/bin/env python3
"""
QNN HTP Element-wise Add 带宽测试模型生成脚本
创建简单的 element-wise Add 算子模型并导出为 TorchScript 格式
用于测试 QNN HTP 的最高算子带宽
"""

import sys
import numpy as np
import torch
import torch.nn as nn

class ElementwiseAddModel(nn.Module):
    """简单的 element-wise Add 模型，用于带宽测试
    使用单输入 + 常量权重的方式，避免多输入转换问题
    使用float32，QNN转换器会将其量化为INT8
    """
    def __init__(self, add_value=1.0):
        super(ElementwiseAddModel, self).__init__()
        # 使用可学习的参数作为第二个操作数（会被转换为常量）
        self.add_value = nn.Parameter(torch.tensor(add_value))

    def forward(self, x):
        # Element-wise addition: x + constant
        return x + self.add_value

if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) > 1:
        input_size = int(sys.argv[1])
    else:
        # 默认 64MB for INT8 (67108864 elements = 64 * 1024 * 1024)
        input_size = 67108864

    print("生成Element-wise Add模型...")
    model = ElementwiseAddModel()
    print("算子类型: Add (Element-wise)")

    print(f"=== 生成 QNN HTP Element-wise Add 带宽测试模型 ===")
    print(f"输入大小: {input_size} elements")
    print(f"  每个输入: {input_size * 4 / 1024 / 1024:.2f} MB (FP32)")
    print(f"  每个输入: {input_size * 1 / 1024 / 1024:.2f} MB (INT8量化后)")
    print(f"算子类型: Add (Element-wise)")
    print(f"数据类型: FP32 (PyTorch) -> INT8 (QNN转换器量化)")
    print()

    # 创建模型
    model = ElementwiseAddModel()
    model.eval()

    # 创建虚拟输入（使用float32，QNN PyTorch转换器会量化为INT8）
    dummy_input = torch.randn(1, input_size, dtype=torch.float32)

    # 获取项目根目录（脚本所在目录的上级目录）
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 保存 PyTorch 模型（用于pytorch converter）
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 保存state_dict（用于加载权重）
    pytorch_path = os.path.join(models_dir, "qnn_elementwise_add_model.pth")
    print(f"正在保存 PyTorch state_dict到: {pytorch_path}")
    torch.save(model.state_dict(), pytorch_path)
    print(f"✓ PyTorch state_dict保存成功: {pytorch_path}")

    # 导出为TorchScript（推荐用于QNN转换器）
    print(f"正在导出 TorchScript 模型...")
    model.eval()
    try:
        # 使用torch.jit.trace（更可靠）
        traced_model = torch.jit.trace(model, dummy_input)
        torchscript_path = os.path.join(models_dir, "qnn_elementwise_add_model.pt")
        traced_model.save(torchscript_path)
        print(f"✓ TorchScript 模型导出成功: {torchscript_path}")
    except Exception as e:
        print(f"警告: TorchScript trace失败: {e}")
        print("尝试使用torch.jit.script...")
        try:
            scripted_model = torch.jit.script(model)
            torchscript_path = os.path.join(models_dir, "qnn_elementwise_add_model.pt")
            scripted_model.save(torchscript_path)
            print(f"✓ TorchScript 模型导出成功: {torchscript_path}")
        except Exception as e2:
            print(f"警告: TorchScript script也失败: {e2}")
            # 作为备选，保存完整模型（需要weights_only=False）
            model_full_path = os.path.join(models_dir, "qnn_elementwise_add_model_full.pth")
            torch.save(model, model_full_path, _use_new_zipfile_serialization=False)
            print(f"✓ 完整 PyTorch 模型保存成功: {model_full_path} (需要weights_only=False加载)")


    # 生成输入数据文件（float32格式，用于QNN转换器量化校准）
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    input_data_path = os.path.join(data_dir, "input_elementwise_data_float.raw")
    dummy_input.numpy().tofile(input_data_path)
    print(f"✓ 输入数据保存到: {input_data_path} (FP32格式，用于量化校准)")

    # 生成输入列表文件
    input_list_path = os.path.join(data_dir, "input_list_elementwise.txt")
    with open(input_list_path, "w") as f:
        f.write(f"input_elementwise_data_float.raw\n")
    print(f"✓ 输入列表文件生成: {input_list_path}")
    print()
    print("下一步: 运行 convert_elementwise_to_qnn.sh 将 PyTorch 模型转换为 QNN 格式")
