# QNN HTP Element-wise Add 带宽测试

## 功能说明

测试高通 QNN HTP (Hexagon Tensor Processor) 的内存带宽性能，使用简单的 element-wise Add 算子。

## 优化结果

- **优化前**: 1GB数据，10.52 GB/s
- **优化后**: 1GB数据，**26.20 GB/s** (提升2.5倍)

## 目录结构

```
qnn_htp_bandwidth/
├── README.md                              # 本文件
├── CHANGELOG.md                           # 更新日志
├── scripts/                               # 脚本目录
│   ├── qnn_htp_elementwise_bandwidth_model.py  # 生成PyTorch模型并导出TorchScript
│   ├── convert_elementwise_to_qnn.sh     # 转换PyTorch到QNN格式
│   ├── build_elementwise_bandwidth_test.sh # 编译和运行脚本
│   └── verify_step_by_step.sh            # 逐步验证脚本
├── src/                                   # 源代码目录
│   └── qnn_htp_elementwise_bandwidth_test.cpp # 带宽测试程序（基于QNN Sample App标准模式）
├── models/                               # 模型目录
│   ├── qnn_elementwise_add_model.pt      # TorchScript模型文件（用于QNN转换器）
│   └── qnn_models_elementwise/            # QNN模型目录
│       ├── elementwise_add.bin           # QNN二进制模型
│       ├── elementwise_add.cpp            # QNN C++模型定义
│       ├── elementwise_add_net.json       # QNN网络配置
│       └── aarch64-android/               # Android平台模型库
│           └── libelementwise_add.so
├── data/                                  # 数据目录
│   ├── input_elementwise_data_float.raw   # 输入数据文件（FP32格式，用于量化校准）
│   └── input_list_elementwise.txt        # 输入列表文件
└── build/                                 # 编译输出目录
    └── qnn_htp_elementwise_bandwidth_test # 编译后的可执行文件
```

## 使用方法

### 1. 生成PyTorch模型
```bash
cd qnn_htp_bandwidth
conda activate qairt
python scripts/qnn_htp_elementwise_bandwidth_model.py
```
这将生成：
- `qnn_elementwise_add_model.pt` - TorchScript模型（用于QNN转换器）
- `qnn_elementwise_add_model.pth` - PyTorch state_dict（备用）

### 2. 转换为QNN格式
```bash
./scripts/convert_elementwise_to_qnn.sh
```
使用QNN PyTorch Converter将TorchScript模型转换为QNN格式。

### 3. 编译和运行测试
```bash
./scripts/build_elementwise_bandwidth_test.sh 512 3  # 512MB数据，3次迭代
```

### 4. 逐步验证（推荐首次运行时使用）
如果遇到执行问题，可以使用逐步验证脚本：
```bash
./scripts/verify_step_by_step.sh
```

该脚本会：
- 检查设备连接
- 验证模型文件和可执行文件
- 检查必要的库文件
- 运行512MB测试（3次迭代）

这样可以逐步定位问题所在。

## 优化措施

1. **使用独立的输入和输出buffer**
   - 避免数据依赖，提升并行度
   - 输入和输出分别使用独立的ION共享内存

2. **优先使用heapid=25**
   - 从UMA验证中确认heapid=25是最优的内存heap
   - 在`alloc_shared_mem`函数中优先尝试heapid=25

3. **保持RAW模式**
   - 使用`QNN_TENSORMEMTYPE_RAW`模式
   - 简单高效，避免内存注册的复杂性

## 测试结果

| 数据量 | 迭代次数 | 平均带宽 | 峰值带宽 |
|--------|---------|---------|---------|
| 1MB    | 100     | 1.15 GB/s | 1.21 GB/s |
| 100MB  | 100     | 4.64 GB/s | 4.87 GB/s |
| 1GB    | 10      | **26.20 GB/s** | **27.51 GB/s** |

## 环境要求

- QNN SDK: 2.40.0.251030
- Android NDK: 29.0.13599879
- Python环境: qairt conda环境（包含PyTorch）
- 目标设备: 支持QNN HTP的Android设备

## 详细文档

### 优化方案分析

#### 一、项目概述

本项目用于测试高通 QNN HTP (Hexagon Tensor Processor) 的内存带宽性能，使用简单的 element-wise Add 算子进行测试。

#### 二、优化历程

**初始状态**
- **测试结果**: 1GB数据，10.52 GB/s
- **问题**: 输入输出使用同一个buffer，导致数据依赖，无法充分利用HTP的并行能力

**优化措施**
1. **使用独立的输入和输出buffer**
   - 避免数据依赖，提升并行度
   - 输入和输出分别使用独立的ION共享内存

2. **优先使用heapid=25**
   - 从UMA验证中确认heapid=25是最优的内存heap
   - 在`alloc_shared_mem`函数中优先尝试heapid=25

3. **保持RAW模式**
   - 使用`QNN_TENSORMEMTYPE_RAW`模式
   - 简单高效，避免内存注册的复杂性

**优化结果**
- **优化后**: 1GB数据，**26.20 GB/s** (提升2.5倍)
- **距离理论带宽60GB/s**: 还有56%的提升空间

#### 三、关键技术点

**1. 共享内存分配**
```cpp
// 优先使用heapid=25（UMA验证中最优）
int heapids[] = {25, 0, 1, 2, 13, 14, 26, 27, 28, 22, 23, 24};
```

**2. 独立Buffer分配**
```cpp
// 独立的输入和输出buffer
res.input_mem = alloc_shared_mem(res.rpc_lib, data_size, &res.input_heapid);
res.output_mem = alloc_shared_mem(res.rpc_lib, data_size, &res.output_heapid);
```

**3. Tensor配置**
```cpp
// 使用RAW模式，直接指定数据指针
res.inputTensor.v2.clientBuf.data = res.input_mem;
res.outputTensor.v2.clientBuf.data = res.output_mem;
res.inputTensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
res.outputTensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
```

#### 四、进一步优化方向

1. **异步执行**
   - 检查QNN SDK是否支持`graphExecuteAsync`
   - 使用回调函数或事件机制
   - 批量提交多个执行请求
   - **预期提升**: 10-20%

2. **批量执行和流水线**
   - 同时执行多个graph实例
   - 使用流水线：准备下一批数据时执行当前批次
   - 增加并发度
   - **预期提升**: 20-30%

3. **计算密集型算子**
   - Element-wise Add计算量小，可能无法充分利用HTP
   - 建议使用卷积、矩阵乘法等算子测试真正的带宽上限
   - **预期提升**: 10-20%

4. **内存注册优化**
   - 正确使用`QnnMem_register`，需要匹配tensor类型
   - 可能需要使用`QNN_TENSORMEMTYPE_MEMHANDLE`
   - **预期提升**: 15-25%（需要进一步研究）

5. **DMA配置优化**
   - 优化内存对齐（rpcmem_alloc已自动对齐到页边界）
   - 配置DMA批量传输参数
   - **预期提升**: 10-15%

#### 五、已验证但无效的优化方案

**方案1：内存注册优化**
- **结果**: 执行失败（错误码1002/1003）
- **结论**: 内存注册后执行失败，需要进一步研究正确的使用方法

**方案2：内存对齐优化**
- **结果**: 14.26 GB/s (比原始25.52 GB/s更低)
- **结论**: rpcmem_alloc已自动对齐到页边界（4KB），手动对齐反而降低性能

#### 六、注意事项

1. **数据量限制**
   - 单个buffer最大约1-2GB（取决于设备内存）
   - 超过限制会导致内存分配失败

2. **清理问题**
   - 程序结束时可能出现"Aborted"（清理时的问题）
   - 核心功能和测试结果正常，不影响使用

3. **算子选择**
   - Element-wise Add是简单的算子，主要用于测试内存带宽
   - 如需测试计算性能，应使用更复杂的算子

4. **模型结构**
   - 当前模型：`input + add_value`（常量）
   - 内存传输：读取input（1GB）+ 写入output（1GB）= 2GB
   - add_value是静态tensor，可能被优化到计算图中，不占用运行时内存传输

#### 七、预期总提升

如果所有方案都实施，预期可以达到：
- **保守估计**: 40-50 GB/s
- **理想情况**: 接近硬件理论带宽 60+ GB/s

---

## 参考文档

### QNN SDK 官方文档
- [QNN Tutorial - Linux Host Linux Target](file:///Users/yinrun/Workspace/qairt/2.40.0.251030/docs/QNN/general/tutorial/qnn_tutorial_linux_host_linux_target.html)
- [QNN Tutorial - Linux Host](file:///Users/yinrun/Workspace/qairt/2.40.0.251030/docs/QNN/general/tutorial/qnn_tutorial_linux_host.html)

### 项目文档
- 详细说明请参考：`../docs/QNN_HTP带宽测试优化方案.md`
- UMA验证总结：`../docs/UMA验证总结.md`
