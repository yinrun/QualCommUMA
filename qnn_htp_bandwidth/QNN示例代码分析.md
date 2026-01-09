# QNN SDK示例代码分析 - TCM处理方式

## 关键发现

### 1. 示例代码的使用方式

从 `/home/yinrun/software/qualcomm/qairt/2.40.0.251030/examples/QNN/SampleApp/SampleApp/src/QnnSampleApp.cpp` 中可以看到：

**正确的使用流程**：
1. `composeGraphs()` - 调用模型库的composeGraphs函数
2. `finalizeGraphs()` - 直接调用 `graphFinalize(graphHandle, profileHandle, nullptr)`
3. `executeGraphs()` - 使用 `setupInputAndOutputTensors()` 设置tensor

**关键点**：
- **不修改tensor维度**：直接使用 `composeGraphs` 返回的 `graphInfo` 中的tensor维度
- **不修改graphInfo**：完全信任模型库返回的配置
- **graphFinalize参数**：`graphFinalize(graphHandle, profileHandle, nullptr)` - 第三个参数是 `nullptr`，表示使用默认配置

### 2. setupTensors的实现

从 `IOTensor.cpp` 的 `setupTensors()` 函数可以看到：

```cpp
// 1. 使用deepCopyQnnTensorInfo复制tensor信息（包括维度）
deepCopyQnnTensorInfo(((*tensors) + tensorIdx), &wrapperTensor)

// 2. 根据graphInfo中的维度分配buffer
fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(wrapperTensor), QNN_TENSOR_GET_RANK(wrapperTensor));
allocateBuffer(reinterpret_cast<uint8_t**>(&clientBuffer.data), dims, ...);
```

**关键**：
- 维度来自 `graphInfo.inputTensors` 和 `graphInfo.outputTensors`
- **不修改维度**，直接使用
- 根据维度分配对应大小的buffer

### 3. deepCopyQnnTensorInfo的实现

从 `QnnSampleAppUtils.cpp` 可以看到：

```cpp
// 复制维度（第238-246行）
QNN_TENSOR_SET_DIMENSIONS(dst, nullptr);
if (QNN_TENSOR_GET_RANK(src) > 0) {
    QNN_TENSOR_SET_DIMENSIONS(dst, (uint32_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t)));
    if (QNN_TENSOR_GET_DIMENSIONS(dst)) {
        pal::StringOp::memscpy(QNN_TENSOR_GET_DIMENSIONS(dst),
                               QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t),
                               QNN_TENSOR_GET_DIMENSIONS(src),
                               QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t));
    }
}
```

**关键**：
- **完全复制**源tensor的维度，不做任何修改
- 维度来自 `composeGraphs` 返回的 `graphInfo`

## 结论

### 我们的代码已经正确

根据示例代码，我们的实现方式是正确的：
1. ✅ 调用 `composeGraphs` 获取 `graphInfo`
2. ✅ 使用 `deepCopyQnnTensorInfo` 复制tensor配置（包括维度）
3. ✅ 不修改tensor维度
4. ✅ 直接调用 `graphFinalize(graphHandle, NULL, NULL)`

### 问题所在

**TCM错误发生在 `graphFinalize` 阶段**，这说明：
1. **模型转换时使用的512MB维度**可能确实超过了TCM限制
2. **QNN工具链应该自动处理**，但可能需要：
   - 转换时的特殊配置（如 `ir_optimizer_config`）
   - 或者运行时通过 `graphFinalize` 的第三个参数传递优化配置

### 可能的解决方案

#### 方案1: 检查graphFinalize的第三个参数

`graphFinalize` 的签名是：
```cpp
Qnn_ErrorHandle_t graphFinalize(Qnn_GraphHandle_t graph,
                                 Qnn_ProfileHandle_t profileHandle,
                                 const QnnGraph_Config_t* config);
```

第三个参数 `config` 可能是 `QnnHtpGraph_Config_t`，可以传递优化选项。

#### 方案2: 检查模型转换配置

可能需要通过转换器的 `--ir_optimizer_config` 或 `--optimization_pass_mode_config` 来配置如何处理大数据。

#### 方案3: 查看QNN SDK文档

需要查看：
- `optimization_grammar.html` - 优化语法文档
- `htp_vtcm_sharing.html` - VTCM共享文档
- `QnnHtpGraph_OptimizationOption_t` - HTP优化选项API

## 下一步

1. 查看 `graphFinalize` 的第三个参数是否可以传递优化配置
2. 查看转换器是否有处理大数据的配置选项
3. 查看QNN SDK文档中关于TCM和tiling的说明
