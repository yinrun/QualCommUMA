# RMSNorm GPU vs NPU 基准测试

## 背景

论文 **HeteroInfer** (SOSP 2025) 的关键设计决策：**RMSNorm 等归一化算子应调度到 GPU 而非 NPU**。

论文原文并未提供 RMSNorm 的独立微基准数据，仅在异构调度框架中隐含了该决策。本实验通过在 **Snapdragon 8 Gen 5 (SM8850)** 上独立测试 GPU 和 NPU 执行 FP16 RMSNorm，直接验证该结论。

## 实测结果

**平台**: Snapdragon 8 Gen 5 (SM8850), Adreno 840 (12 CU) + Hexagon V81, LPDDR5X-5300
**理论峰值带宽**: 84.8 GB/s
**GPU**: FP16 RMSNorm (OpenCL)
**NPU**: FP16 RMSNorm (QNN HTP Native `QNN_OP_RMS_NORM`)

### GPU vs NPU FP16 RMSNorm 性能对比

| 场景 | batch | hidden | GPU(us) | GPU(GB/s) | NPU(us) | NPU(GB/s) | GPU/NPU |
|------|-------|--------|---------|-----------|---------|-----------|---------|
| decode | 1 | 2048 | 62.6 | 0.18 | 278.2 | 0.04 | **4.4x** |
| decode | 1 | 3200 | 61.4 | 0.29 | 276.6 | 0.06 | **4.5x** |
| decode | 1 | 4096 | 62.1 | 0.37 | 278.3 | 0.08 | **4.5x** |
| prefill-16 | 16 | 4096 | 62.3 | 4.04 | 290.8 | 0.87 | **4.7x** |
| prefill-64 | 64 | 4096 | 69.6 | 14.14 | 346.3 | 2.84 | **5.0x** |
| prefill-256 | 256 | 4096 | 111.3 | 35.15 | 449.0 | 8.72 | **4.0x** |
| prefill-512 | 512 | 4096 | 230.5 | 33.92 | 636.0 | 12.29 | **2.8x** |
| prefill-1k | 1024 | 4096 | 487.4 | 32.08 | 1266.9 | 12.34 | **2.6x** |

### 分析

**1. NPU 调度开销是致命瓶颈**
- NPU 基础调度延迟约 **276-278 us**（从 batch=1 的三组 decode 测试可见）
- GPU 基础调度延迟约 **61-63 us**，仅为 NPU 的 1/4
- LLM decode 阶段（batch=1），NPU 调度开销完全掩盖了计算时间

**2. GPU 带宽利用率远优于 NPU**
- GPU batch=256: 35.2 GB/s（理论带宽的 41%）
- NPU FP16 RMSNorm 最高仅 12.3 GB/s（prefill-1k），约为 GPU 的 35%

**3. 加速比随 batch 增大收敛但 GPU 始终更快**
- batch=1: 4.4-4.5x（NPU 调度开销主导）
- batch=256: 4.0x
- batch=1024: 2.6x（GPU 仍保持 ~3x 优势）

## 结论

### 论文结论验证: 完全验证

GPU 执行 FP16 RMSNorm 比 NPU 快 **2.6x–5.0x**：
- decode（batch=1）: GPU 快 **~4.5x**
- prefill（batch=1024）: GPU 快 **~2.6x**

| 因素 | GPU (Adreno 840) | NPU (Hexagon V81) |
|------|-------------------|-------------------|
| 调度延迟 | ~62 us | ~278 us (4.5x 更高) |
| FP16 RMSNorm | 支持 | 支持 |
| 峰值带宽利用 | 35.2 GB/s (41%) | 12.3 GB/s (15%) |
| 计算模型 | 灵活 SIMD，适合 reduction | Systolic array，适合矩阵乘 |

## 遇到的问题

### 问题 1: QNN tensor 参数必须注册为图张量

**现象**: 调用 `graphAddNode(RmsNorm)` 返回 error 6001 (`QNN_GRAPH_ERROR_INVALID_HANDLE`)，
所有数据类型 (FP16, UINT8, INT16) 均失败。

**诊断过程**:
1. 开启 QNN verbose 日志 (`QNN_LOG_LEVEL_VERBOSE`)
2. 日志显示: `Failed to create InputDef for Op rmsnorm_native param axes`
3. 问题定位到 `axes` 张量参数，而非数据类型

**根因**: QNN API 的 `QNN_PARAMTYPE_TENSOR` 类型参数中嵌入的张量，必须先通过 `tensorCreateGraphTensor()` 注册为图张量，
再作为参数传递给 `graphAddNode()`。这在 SDK 的 `QnnModel.cpp` 参考实现中有体现（`addNode()` → `addTensor()` → `tensorCreateGraphTensor()`），
但 API 文档和头文件注释中未明确说明。

**修复**:
```cpp
// 错误: 直接在 param 中使用未注册的张量
Qnn_Param_t axes_param;
axes_param.tensorParam = axes_tensor;  // axes_tensor 未注册

// 正确: 先注册为图张量
g_qnn->tensorCreateGraphTensor(g_graph, &axes_tensor);
Qnn_Param_t axes_param;
axes_param.tensorParam = axes_tensor;  // axes_tensor 已注册，包含有效 id
```

**影响范围**: 所有使用 tensor 类型参数的算子，包括 `RmsNorm.axes`、`ReduceMean.axes`、`Conv2d.dilation` 等。

### 问题 2: RmsNorm 需要 3 个输入 (含 beta)

**现象**: 只传 2 个输入 (input, gamma) 时 `graphAddNode` 失败。

**根因**: HTP 后端要求 RmsNorm 的 3 个输入 (input, gamma, beta) 全部提供，beta 不可省略。
即使 beta 全为零也必须显式传入。

### 问题 3: gamma/beta 应为 rank-1 张量

**现象**: 使用 rank-4 的 gamma/beta (`[1,1,1,hidden]`) 时行为异常。

**根因**: MasterOpDef 文档明确指出 gamma/beta 的 rank = M = size(axes)。
当 axes=[3]（仅通道维度）时 M=1，gamma/beta 应为 shape `[hidden_dim]` 的一维张量。

## 目录结构

```
rmsnorm_benchmark/
├── README.md
├── CMakeLists.txt
├── build_android.sh
├── run_on_device.sh
├── kernels/
│   └── rmsnorm.cl              # GPU OpenCL FP16 RMSNorm 内核
└── src/
    ├── common.h                # 共享类型: RMSNormConfig, RMSNormResult, ION 工具
    ├── gpu_rmsnorm.h/.cpp      # GPU OpenCL 实现
    ├── npu_rmsnorm.h/.cpp      # NPU QNN 实现 (Native/Decomposed FP16)
    └── main.cpp                # 测试驱动: GPU vs NPU FP16 对比
```

## 构建与运行

```bash
export ANDROID_NDK=/path/to/android-ndk-r25c
export QNN_SDK_ROOT=/path/to/qualcomm/qairt/2.42.0.251225

bash build_android.sh
bash run_on_device.sh
bash run_on_device.sh --hidden-dim 8192 --batch 128 --iters 500
```

## 参考

- **HeteroInfer** (SOSP 2025): *Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference*
- Qualcomm QNN SDK 2.42.0 — `QnnOpDef.h:640` (`QNN_OP_RMS_NORM`)
- Qualcomm QNN SDK 2.42.0 — `share/QNN/converter/jni/QnnModel.cpp` (tensor param 注册模式)
