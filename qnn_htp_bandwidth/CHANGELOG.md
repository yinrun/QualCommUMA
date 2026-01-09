# 更新日志

## 2026-01-09 - 仅保留PyTorch转换方式

### 变更内容

1. **移除ONNX转换方式**
   - 删除ONNX模型导出代码
   - 移除ONNX转换相关脚本
   - 统一使用PyTorch Converter

2. **优化模型生成脚本**
   - 主要导出TorchScript格式（`.pt`文件）
   - 保留state_dict（`.pth`文件）作为备用
   - 移除ONNX导出代码

3. **统一转换脚本**
   - `convert_elementwise_to_qnn.sh` 现在使用PyTorch Converter
   - 自动检测并使用TorchScript模型（优先）或完整PyTorch模型

### 优势

- **简化流程**：无需中间ONNX格式，直接从PyTorch转换
- **更好的兼容性**：TorchScript格式更稳定可靠
- **减少依赖**：不需要ONNX相关的依赖

### 使用方法

```bash
# 1. 生成PyTorch模型
conda activate qairt
python scripts/qnn_htp_elementwise_bandwidth_model.py

# 2. 转换为QNN格式
./scripts/convert_elementwise_to_qnn.sh

# 3. 运行测试
./scripts/build_elementwise_bandwidth_test.sh 1 10
```

### 文件变更

- ✅ `scripts/qnn_htp_elementwise_bandwidth_model.py` - 移除ONNX导出
- ✅ `scripts/convert_elementwise_to_qnn.sh` - 改为使用PyTorch Converter
- ✅ `README.md` - 更新文档说明
- ❌ `scripts/convert_elementwise_pytorch_to_qnn.sh` - 已删除（合并到主脚本）
