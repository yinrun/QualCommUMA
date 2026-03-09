# Triton RMSNorm via Hexagon-MLIR

第三种 HTP RMSNorm 实现：使用 [hexagon-mlir](https://github.com/qualcomm/hexagon-mlir)（Triton backend for Hexagon）编写 Triton kernel，编译到 HVX 指令在 HTP 上执行。

## 三种 HTP RMSNorm 方案对比

| 维度 | QNN Native | Custom HVX | Triton (hexagon-mlir) |
|------|-----------|------------|----------------------|
| 实现方式 | `QNN_OP_RMS_NORM` 内置算子 | HVX intrinsics C++ | Triton DSL (Python) |
| 代码量 | ~50 行 QNN C API 调用 | ~300 行 HVX + ~200 行 op package | ~30 行 kernel + ~50 行 harness |
| 优化手段 | QNN runtime 自动优化 | 手写 SIMD、手动 tiling | 编译器自动 VTCM tiling、DMA、多线程 |
| 多线程 | 自动 AUTOSPLIT | 需手写 splitting rule | `num_programs` 控制 |
| 开发难度 | 低（黑盒调用） | 高（需熟悉 HVX ISA） | 中（Triton 语法 + 编译器选项） |
| 可移植性 | 仅 QNN 平台 | 仅 Hexagon | Triton 语法可移植到 GPU/其他后端 |

## 文件结构

```
triton_rmsnorm/
├── rmsnorm_kernel.py   # Triton RMSNorm kernel + benchmark harness
├── setup_env.sh        # 环境配置（构建 hexagon-mlir 工具链）
└── README.md           # 本文档
```

## 环境准备

```bash
# 首次：构建完整工具链（~30-60分钟）
source setup_env.sh

# 后续：仅加载环境变量
source setup_env.sh --skip-build
```

### 前置依赖

- Python 3.8+
- Hexagon SDK 6.4+ (build script 自动下载)
- Android NDK (设备执行)
- adb 连接的 SM8850 设备
- ~50 GB 磁盘空间 (LLVM + Triton 构建)

### V81 架构支持

SM8850 使用 Hexagon V81。hexagon-mlir FAQ 仅列出 v73/v75/v79 为已验证版本。
默认尝试 `HEXAGON_ARCH_VERSION=81`，如编译失败可回退：

```bash
export HEXAGON_ARCH_VERSION=79
source setup_env.sh --skip-build
```

## 运行测试

```bash
# 全部场景
pytest -sv rmsnorm_kernel.py

# 仅 decode 场景
pytest -sv rmsnorm_kernel.py -k "decode"

# 仅 FP16 + 4线程
pytest -sv rmsnorm_kernel.py -k "fp16 and 4T-base"

# 仅特定 hidden size
pytest -sv rmsnorm_kernel.py -k "decode-4k and fp32 and 1T"
```

## Kernel 说明

Triton kernel 逻辑与 `hexagon-mlir/test/python/triton/test_rms_norm.py` 一致：

```python
@triton.jit
def rms_norm_fwd_kernel(x_ptr, y_ptr, weights_ptr, ...):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < NUM_COLS
    for row in range(pid, NUM_ROWS, programs):
        x = tl.load(x_ptr + row * NUM_COLS + block, mask=mask)
        mean_sq = tl.sum(x * x, axis=0) / NUM_COLS
        rms = tl.sqrt(mean_sq + EPSILON)
        g = tl.load(weights_ptr + block, mask=mask)
        y = (x / rms) * g
        tl.store(y_ptr + row * NUM_COLS + block, y, mask=mask)
```

### 优化选项

hexagon-mlir 编译器支持以下优化 pass：

| 选项 | 说明 |
|------|------|
| `enableMultiThreading` | 启用 HVX 多线程并行 |
| `enableVTCMTiling` | 启用 VTCM (Vector TCM) 自动 tiling |
| `enableConvertToHexagonmem` | 转换为 Hexagon memory 操作 |
| `enableHexagonmemCopyToDMA` | 使用 DMA 引擎搬运数据 |

benchmark 会扫描 5 种组合，找到最优配置。

## 性能对比

> 待实测数据填入

```
Scene        batch   hid | Triton(us) BW(GB/s) | Custom(us) BW(GB/s) | Native(us) BW(GB/s) | GPU(us) BW(GB/s)
--------------------------------------------------------------------------------------------------------------
decode-2k        1  2048 |       TODO      TODO |      808.0     0.02 |      807.3     0.02 |    62.7     0.18
decode-3.2k      1  3200 |       TODO      TODO |      813.8     0.02 |      820.2     0.02 |    60.3     0.30
decode-4k        1  4096 |       TODO      TODO |      819.2     0.03 |      837.5     0.03 |    61.4     0.37
prefill-16      16  4096 |       TODO      TODO |     1294.0     0.21 |     1079.8     0.25 |    60.3     4.17
prefill-64      64  4096 |       TODO      TODO |     2404.1     0.44 |     1881.6     0.56 |    74.4    13.23
pf-256         256  4096 |       TODO      TODO |     6446.1     0.65 |     4450.0     0.94 |   110.5    35.40
pf-512         512  4096 |       TODO      TODO |    18152.0     0.46 |     7628.7     1.10 |   230.7    33.89
pf-1k         1024  4096 |       TODO      TODO |    30840.0     0.54 |    12567.8     1.34 |   483.9    32.31
```

## 开发体验对比

| 维度 | QNN Native | Custom HVX | Triton |
|------|-----------|------------|--------|
| 初始上手 | 简单：查 API 文档 | 困难：学习 HVX ISA | 中等：安装工具链耗时 |
| 调试 | 黑盒，靠日志 | GDB + hexagon-sim | Python pytest |
| 迭代速度 | 快（改参数重编） | 慢（改 C++ 重编 .so） | 快（改 Python 重跑） |
| 性能调优 | 无法干预 | 完全可控 | 编译器选项 + kernel 写法 |
