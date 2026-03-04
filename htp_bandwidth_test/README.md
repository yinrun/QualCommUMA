# HTP Memory Bandwidth Test

通过 QNN HTP 后端执行 ElementWiseAdd 测量 Hexagon Tensor Processor 的最大内存带宽。

- **目标平台**: Snapdragon 8 Gen 5 (Hexagon V81 / SM8850)
- **测试方法**: 两个 128MB 输入张量 (UFIXED_POINT_8) 逐元素相加，计算有效带宽 (2读 + 1写 = 3x data)
- **测试结果**: **50+ GB/s** (BURST mode, SDK 2.42)

## 关键优化点

### 1. Power Config 正确配置 (6.4 → 34.5 GB/s, 5.4x)

```cpp
// contextId 必须绑定到 createPowerConfigId 返回的 ID
dcvsConfig.dcvsV3Config.contextId = powerConfigId;  // 不能用 0
// 禁用 DCVS，锁定最高频率
dcvsConfig.dcvsV3Config.dcvsEnable = 0;
// 不要在执行前调用 destroyPowerConfigId
```

### 2. 使用 UFIXED_POINT_8 数据类型 (34.5 → 50+ GB/s, +45%)

```cpp
// INT_8 会导致 HTP 插入隐式 Cast 节点，增加额外内存读写
// UFIXED_POINT_8 + 显式量化参数是 HTP 原生类型，无需转换
tensor.v1.dataType = QNN_DATATYPE_UFIXED_POINT_8;
tensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
tensor.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
tensor.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
```

## 目录结构

- `src/main.cpp`: 直接用 QNN C API 构建图并执行带宽测试
- `build_android.sh`: NDK 交叉编译脚本
- `run_on_device.sh`: 推送依赖并在设备上运行

## 前置条件

- Android NDK (通过 ANDROID_SDK_ROOT 自动检测)
- QNN SDK 2.42+ (`QNN_SDK_ROOT` 环境变量)
- 设备已连接，`adb devices` 可见

## 编译与运行

```bash
cd htp_bandwidth_test
export ANDROID_SDK_ROOT=/path/to/Android/Sdk
export QNN_SDK_ROOT=/path/to/qairt/2.42.0.251225
./build_android.sh
./run_on_device.sh
```

输出示例:

```
[QNN] HTP Add demo starting...
[QNN] Using provider API 2.32.0
[QNN] HTP num_cores=1
[QNN] HTP num_hvx_threads=8
[QNN] Done. max_error=0 sample_out=3 total_ms=7.48 per_tile_ms=1.87 bandwidth_GBps=50.11
```

## 技术细节

- 张量布局: NHWC `[32, 1, 1, C]`，4 tiles x 32MB = 128MB 总数据
- 内存分配: `rpcmem_alloc` (heapid=25) + `QnnMem_register` (ION zero-copy)
- 性能模式: DCVS 禁用, MAX 电压角, HMX MAX, sleep 禁用
- 运算类型: `QNN_OP_ELEMENT_WISE_ADD`
