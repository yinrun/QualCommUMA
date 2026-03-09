# Fast Sync 原型：GPU↔NPU 同步开销测试

## 背景

论文 **HeteroInfer** (SOSP 2025) 提出的 fast sync 机制是 decode 阶段获得 2-4x 加速的关键。

**问题**: 移动 SoC 上 `clFinish()` 同步开销约 **~400 us**，在 decode 阶段（kernel 执行仅几百微秒）占比极大。

**论文方案** (Section 4.3):
> "A flag bit is added alongside the output tensor and is updated once the output tensor
> is completely populated. The CPU core only needs to poll this flag bit for a few
> microseconds and can immediately notify the NPU for subsequent execution."

核心思路是 **数据级同步**：
1. GPU kernel 写完输出后，在共享内存中写一个 **flag bit**
2. CPU 先 `usleep(predicted_time)` 粗等待
3. CPU 直接从 ION buffer 的 CPU 映射地址 **读 flag**（不经过 OpenCL 驱动）
4. Flag 置位 → GPU 完成，通知 NPU

这完全绕过了 `clFinish()` / `cl_event` 的驱动路径，实现 ~1us 的同步开销。

## 测试场景

模拟 LLM decoder layer 的 GPU↔NPU pipeline（batch=1, hidden_dim=4096, FP16）：

```
GPU(RMSNorm) → sync → NPU(RMSNorm) → sync → GPU(RMSNorm) → sync → NPU(RMSNorm) → ...
```

GPU 和 NPU 通过 ION 共享内存实现零拷贝数据传递。

## 六种同步模式

### Mode 1: Sequential Blocking（基线）

```
主线程: clEnqueue + clFinish() → graphExecute() → 重复
```

GPU 和 NPU 串行阻塞执行。`clFinish()` 的开销直接体现在 pipeline 延迟中。

### Mode 2: Threaded + clFinish

```
主线程: clEnqueue + clFinish() → signal(npu_start) → spin_wait(npu_done)
NPU线程: spin_wait(npu_start) → graphExecute() → signal(npu_done)
```

NPU 在独立线程执行，但 GPU 仍用 `clFinish()`。目的：隔离线程机制本身的开销。

### Mode 3: Event Poll（cl_event 轮询，对照组）

```
主线程: clEnqueue + clFlush() → poll(clGetEventInfo) → signal(npu_start) → spin_wait(npu_done)
NPU线程: spin_wait(npu_start) → graphExecute() → signal(npu_done)
```

用 `clGetEventInfo()` 替代 `clFinish()`。但 `clGetEventInfo()` 仍经过 OpenCL 驱动，
与 `clFinish` 开销差异很小。**这不是论文的方案**。保留做对照。

### Mode 4: Fast Sync（论文方案）

```
主线程: clEnqueue + clFlush() → usleep(hint) → poll(共享内存 flag) → signal(npu_start) → spin_wait(npu_done)
NPU线程: spin_wait(npu_start) → graphExecute() → signal(npu_done)
```

- **GPU kernel 修改**: 写完输出数据后，在共享内存 flag buffer 写 `1`
- **CPU 同步**: `usleep(predicted_time)` 粗等待 + 直接读 ION 映射地址的 flag
- **不经过 OpenCL 驱动**：flag 是 GPU kernel 的输出，CPU 直接读共享内存

### Mode 5: Fast Sync Direct（NPU 线程直接 poll，无主线程中继）

```
主线程: clEnqueue + clFlush() → signal(gpu_submitted) → spin_wait(npu_done)
NPU线程: spin_wait(gpu_submitted) → poll(共享内存 flag) → graphExecute() → signal(npu_done)
```

与 Mode 4 的关键区别：**NPU 线程直接轮询 GPU flag**，主线程在 GPU 提交后立即释放（无需等待 GPU 完成再通知 NPU）。消除了"GPU flag 检测 → 主线程唤醒 NPU 线程"之间的中继延迟（约 1-5us）。

### Mode 6: Parallel Sync（GPU+NPU 并行启动 + DSP 端 SyncWait）

```
主线程: clEnqueue + clFlush() → signal(gpu_submitted)
NPU线程: signal → graphExecute() [graph: SyncWait → RmsNorm]
         DSP: SyncWait op HAP_mmap_get(ion_fd) → dcinva DDR polling → RmsNorm
```

目标：GPU 执行期间 NPU RPC overhead 与 GPU 并行，DSP 端直接感知 GPU 完成。
通过 `HAP_mmap_get(flag_ion_fd)` 绕过 QNN TCM DMA 拷贝，DSP 直接轮询 ION DDR 地址。

**关键实现**：SyncWait 接收静态参数 `flag_ion_fd`（ION fd），DSP 侧用 `HAP_mmap_get(fd)` 获取 DSP 虚拟地址，直接轮询原始 ION DDR 内存（非 QNN DMA 副本）。

## 关键设计

### GPU Kernel Flag 写入

```opencl
__kernel void rmsnorm(
    __global scalar_t* output,
    __global const scalar_t* input,
    __global const scalar_t* gamma,
    const int hidden_dim,
    const float epsilon,
    __local float* sdata,
    __global volatile uint* done_flag)   // ← 新增：完成标志
{
  // ... Phase 1-4: 计算 RMSNorm，写入 output ...

  // Phase 5: 写完成标志
  barrier(CLK_GLOBAL_MEM_FENCE);   // 确保所有输出写入已提交
  if (get_local_id(0) == 0)
    *done_flag = 1u;
}
```

关键点：
- `barrier(CLK_GLOBAL_MEM_FENCE)` 确保所有 work-item 的数据写在 flag 之前
- `volatile` 防止 GPU 编译器优化掉写入
- batch=1 时只有一个 work-group，work-group barrier 足够

### CPU 侧 Flag 轮询

```cpp
// Flag buffer: 独立的 4 字节 ION buffer
volatile uint32_t* flag_ptr = (volatile uint32_t*)ion_flag.ptr;
*flag_ptr = 0;  // 重置

gpu_submit();   // clEnqueue + clFlush（非阻塞）

// 直接读共享内存，不经过 OpenCL 驱动
if (usleep_hint > 0) usleep(usleep_hint);
while (*flag_ptr == 0) cpu_pause();
// GPU 完成！
```

### ARM UMA 缓存一致性

SM8850 使用 ACE 协议的 coherent interconnect：
- GPU 和 CPU 在同一个一致性域内
- `CL_MEM_HOST_UNCACHED_QCOM` 确保 CPU 读绕过 CPU cache
- `volatile` 确保 C++ 编译器不优化 CPU 读操作
- `barrier(CLK_GLOBAL_MEM_FENCE)` 确保 GPU 侧内存序

### ION 共享内存（零拷贝）

三个 ION buffer：

```
ion_buf0: GPU 读 (input)  → NPU 写 (output)   [hidden_dim * 2 bytes]
ion_buf1: GPU 写 (output) → NPU 读 (input)    [hidden_dim * 2 bytes]
ion_flag: GPU 写 flag → CPU 读 flag            [4 bytes]
```

- GPU 端: `CL_MEM_ION_HOST_PTR_QCOM` Qualcomm 扩展导入
- NPU 端: `QNN_MEM_TYPE_ION` 注册

### 测量指标

| 指标 | 来源 |
|------|------|
| gpu_compute_us | OpenCL profiling (COMMAND_END - COMMAND_START) |
| gpu_sync_us | 同步机制耗时减去 compute |
| npu_compute_us | graphExecute 返回耗时 |
| npu_sync_us | 等待 NPU 完成的轮询时间 |
| step_total_us | 端到端单步延迟 |

统计量: min / P50 / avg / P99 / max

### GPU 同步诊断

对比四种 GPU 完成检测方式（GPU-only，无 NPU 干扰）：

```
1. clFinish (blocking)        — OpenCL 驱动阻塞等待
2. clFlush + Event Poll    — OpenCL 驱动级轮询（clGetEventInfo）
3. clFlush + WaitForEvents    — OpenCL OS 级等待
4. clFlush + flag poll        — 共享内存直接轮询（论文方案）← ground truth

clFinish 真正开销 = clFinish 总时间 - flag 轮询总时间
```

flag 轮询检测的是 GPU 硬件完成时刻（kernel 写 flag 到共享内存），不经过驱动。
与 clFinish 的差值才是驱动引入的真正开销。

### OpenCL Profiling 时间线

利用 `CL_PROFILING_COMMAND_QUEUED/SUBMIT/START/END` 四个硬件时间戳分解 GPU 命令流水线：

| 时间戳 | 含义 |
|--------|------|
| COMMAND_QUEUED | 命令在 host 入队时刻 |
| COMMAND_SUBMIT | 命令提交到 GPU 硬件时刻 |
| COMMAND_START | GPU 开始执行 kernel 时刻 |
| COMMAND_END | GPU 执行完成时刻 |

注意：profiling 时间戳用的是 GPU 设备时钟，与 CPU host 时钟不同，两者不能直接相减。
但各阶段的**差值**（如 SUBMIT→START）是可靠的设备侧延迟度量。

## 关键设计：Custom HTP Op Package

### SyncWait + RmsNorm 联合包（heteroedge_op/）

Mode 6 需要在 QNN 计算图中插入自定义 `SyncWait` op，在 DSP 端轮询 GPU flag。

**图结构**：
```
sw_input[ION, FP16] + sw_flag[ION, UINT32] → SyncWait → sw_out[NATIVE] → RmsNorm → output[ION]
```

`SyncWait` 是 data passthrough op（2 输入：数据 + flag，1 输出：数据拷贝），在 DSP 执行时：
1. 用 `dcinva` cache invalidation 轮询 flag tensor
2. cache invalidate 整个 data buffer
3. memcpy data → output（建立与 RmsNorm 的 tensor 依赖）

**关键实现细节**：必须为 SyncWait 注册 `PlainFloat16Tensor` 变体。若仅有 generic `Tensor` 实现，HTP planner 会为下游 RmsNorm 选择 scalar reference 实现（无 HVX），导致 ~8x 开销。

### QNN 图开销单元测试（test_graph_overhead）

独立测试 6 种图配置，分析自定义算子的开销来源：

```
Config A: Native QNN RmsNorm (1 op)                     min=267  p50=278  avg=288
Config B: Custom HVX RmsNorm (1 op)                     min=265  p50=274  avg=279
Config C: SyncWait only (passthrough, flag=1)            min=311  p50=317  avg=351
Config D: SyncWait + RmsNorm (2 ops, separate packages)  min=1746 p50=2263 avg=2205  ← 8.3x 慢！
Config E: 2x Custom RmsNorm (2 ops chained, same pkg)    min=267  p50=289  avg=298
Config F: SyncWait+RmsNorm combined pkg (PlainFP16)      min=317  p50=327  avg=329
Config G: SyncWait+RmsNorm (flag=0, QNN DMA copy demo)   ~410000 us                  ← 超时
```

**发现 1（Config D vs E vs F）**：不同包之间的边界不是瓶颈（Config F 与 D 开销相同）。
根本原因：SyncWait 原始实现只有 generic `Tensor` 变体，导致 RmsNorm 退化为 scalar 实现。
**修复**：添加 `PlainFloat16Tensor` 变体 → Config F = 327us（正常）。

**发现 2（Config F vs G）**：flag 预设为 1（DMA 拷贝时已是 1）→ 327us ✓；
flag 从 0 开始（GPU 之后写 1）→ DSP 轮询的是 QNN DMA 拷贝的副本（始终为 0）→ 超时 410ms ✗。

**图开销模型**（所有 config 共享 ~270us 固定开销 = RPC setup + ARM 侧处理）：
- Native RmsNorm (HVX): +4us DSP compute → 274us total
- SyncWait (PlainFP16 memcpy): +57us DSP compute → 327us total
- 2x RmsNorm: +8us DSP compute → 289us total

## 关键发现：QNN DMA 拷贝问题与 HAP_mmap_get 解决方案

### 根本问题

QNN HTP 在 `graphExecute` 启动时，将所有 `APP_WRITE` 输入 tensor 数据 **DMA 拷贝到 DSP 内部内存（TCM/VTCM）**。DSP op 的 `raw_data_const()` 返回的是 DSP 内部副本的指针，而非原始 ION buffer 地址。

```
t=0:   CPU 调用 graphExecute
        → QNN DMA: ion_flag(=0) → DSP TCM copy(=0)
t=80:  DSP 开始执行 SyncWait op
        pflag = raw_data_const() = DSP TCM 副本地址
        dcinva(TCM 地址) 无效  ← TCM 是 DSP 内部内存，cache invalidate 无法使其看到 GPU 的 ION 写入
t=120: GPU 写 ION flag = 1  → 但 DSP 看的是 TCM 副本(=0)，永远读不到 1
t=410ms: DSP 超时返回
```

**验证**：test_graph_overhead Config F（flag 预设 1）= 327us；Config G（flag=0）= ~410ms。

### 解决方案：HAP_mmap_get + 静态参数传 ION fd

绕过 QNN tensor 机制，通过 Hexagon DSP SDK 的 `HAP_mmap_get(int fd, void **vaddr, uint64 *paddr)` API 直接获取 ION buffer 的 DSP 虚拟地址：

1. **CPU 侧**：将 `ion_flag.fd`（`rpcmem_to_fd()` 返回的 ION fd）作为 `Qnn_Param_t` UINT32 静态参数传给 SyncWait op
2. **DSP 侧**（`HeteroEdgeSyncWait.cpp`）：从静态参数读取 ion_fd，调用 `HAP_mmap_get(ion_fd, &vaddr, &paddr)` 获取 DSP VA，直接在 ION DDR 地址上做 `dcinva` + poll

```
t=0:   CPU 调用 graphExecute (ion_flag.fd 作为 SyncWait 静态参数)
        → QNN DMA: ion_flag(=0) → DSP TCM copy(=0)  [仍然发生，但我们绕过它]
t=80:  DSP 执行 SyncWait op
        ion_fd = static_param["flag_ion_fd"]
        HAP_mmap_get(ion_fd, &vaddr, &paddr)   ← 获取 ION DDR 的 DSP VA
        volatile uint32_t* pflag = vaddr;       ← 直接指向原始 ION DDR
        dcinva(pflag); if (*pflag == 1) break;  ← 感知到 GPU 的写入！
t=224: GPU 写 ION flag = 1  → DSP 的 dcinva + poll 检测到
t=240: RmsNorm 执行
t=270: graphExecute 返回
```

**前提条件**：ION buffer 必须先通过 `g_qnn->memRegister()` 注册（`QNN_MEM_TYPE_ION`），这一步会在 CDSP 建立 VA 映射，使 `HAP_mmap_get` 可以成功。

**验证数据**（standalone sync_op_test）：
- Scenario A（flag 预设 1）: 324us — 正常
- Scenario B（CPU 在 graphExecute 后 50us 写 1）: **488us** — DSP 正确感知到 CPU 写入！
- Scenario C（永不写）: 2.3s timeout — DDR ~230ns/dcinva（vs TCM ~41ns）证明确实在轮询 DDR

## 实测结果（SM8850, Adreno 840 + Hexagon V81）

### GPU 同步诊断

单独测试 GPU kernel（hidden=4096, FP16, batch=1, 100 次迭代）：

```
Method                    total_p50   extra
clFinish (blocking)         676.3 us
clFlush+cl_event poll       687.4 us  submit=159.3 poll=527.2 polls=1205
clFlush+WaitForEvents       654.8 us
clFlush+flag poll ★         292.2 us  polls=5956  ← ground truth
```

**Flag 轮询比 clFinish 快 384us** — 这就是 clFinish 的驱动端后处理开销。
cl_event 轮询与 clFinish 开销相近（两者都经过 OpenCL 驱动，~395us 开销），
只有 flag 方案（直接读共享内存）才能绕过驱动。

### OpenCL Profiling 时间线分析

利用 `CL_PROFILING_COMMAND_QUEUED/SUBMIT/START/END` 分解 GPU 命令流水线各阶段延迟：

```
GPU command pipeline (device clock, p50):

  Host (CPU)                        Device (GPU)
  ─────────────                     ───────────────
  clEnqueueNDRange ─┐
                    │   98.4 us     QUEUED → SUBMIT    驱动翻译：命令 → GPU 硬件指令
                    ├──────────────►
  clFlush ──────────┘               │
                                    │  244.7 us         GPU 调度：command stream → 执行单元
                                    ├──────────────────►
                                    │  START
                                    │   18.7 us         kernel 执行
                                    │  END
                                    ├──────────────────►
                                    │                    flag 写入共享内存 ← CPU 可检测 (292us)
                                    │
  clFinish 返回 ◄───────────────────┘  +384 us 驱动后处理
  (总计 676us)
```

**关键发现：**

| 阶段 | 耗时 (p50) | 说明 |
|------|-----------|------|
| QUEUED→SUBMIT | 98.4 us | OpenCL 驱动翻译命令给 GPU 硬件 |
| SUBMIT→START | 244.7 us | GPU 硬件调度延迟（最大瓶颈） |
| START→END | 18.7 us | kernel 真正执行时间 |
| QUEUED→END | 362.3 us | 设备侧总时间 |
| Flag wall time | 292.2 us | Host 视角：提交 → flag 可检测 |
| clFinish 返回 | 676.3 us | Host 视角：提交 → clFinish 返回 |
| **clFinish 开销** | **384.1 us** | **clFinish 返回 - flag 检测 = 纯驱动后处理** |

**292us 的 flag 检测时间分解**：kernel compute 仅 18.7us，剩余 ~274us 是 host 侧提交 + GPU 硬件调度延迟。这部分是不可避免的（命令必须经过驱动到达 GPU），但 flag 方案省掉了 clFinish 返回前的 384us 驱动后处理。

### Pipeline 模式对比（无绑核）

```
Mode                   step_p50   gpu_sync   npu_sync   sync_tot    speedup
------------------------------------------------------------------------
Seq Blocking            935.1 us    632.6 us      0.0 us    632.6 us     1.00x
Thread+clFinish        1101.7 us    824.4 us      1.5 us    825.9 us     0.85x
Event Poll             1169.8 us    860.5 us      1.7 us    862.2 us     0.80x
Fast Sync ★             841.1 us    356.0 us      2.3 us    358.3 us     1.11x
```

不绑核时 Fast Sync 仅 1.11x 加速。线程模式下 spin-loop 与 NPU graphExecute 竞争同一 CPU 核，
导致 NPU compute 从 283us 膨胀到 476us（+68%），严重削弱加速效果。

### Pipeline 模式对比（绑核：main→core 7, npu→core 6）

SM8850 CPU 拓扑：Core 0-5 @ 3.63GHz (performance), Core 6-7 @ 4.61GHz (prime)。
将主线程（flag 轮询）绑到 prime core 7，NPU 线程绑到 prime core 6，消除 CPU 竞争：

```
Mode                   step_p50   gpu_sync   npu_sync   sync_tot    speedup
------------------------------------------------------------------------
Seq Blocking            761.3 us    547.1 us      0.0 us    547.1 us     1.00x
Thread+clFinish         749.3 us    549.0 us      0.4 us    549.4 us     1.02x
Event Poll              657.0 us    480.6 us      0.2 us    480.8 us     1.16x
Fast Sync               260.8 us     97.7 us      0.2 us     97.9 us     2.92x
Fast Sync Direct        270.5 us     44.1 us      0.0 us     44.1 us     2.81x
Parallel Sync ★         273.0 us      0.0 us      0.2 us      0.2 us     2.79x
```

**所有三种 fast sync 模式均达到 ~2.8x 加速**（step p50: ~265-273us vs 761us）。

**Parallel Sync（Mode 6）现已工作**，通过 HAP_mmap_get 绕过 QNN TCM DMA 拷贝问题：
- `gpu_sync = 0.0 us`：主线程零 GPU 同步开销，GPU 完成检测完全由 DSP 内部完成
- `npu_compute = 220us`：包含 RPC 启动（~80us）+ DSP SyncWait 轮询 + RmsNorm 计算
- GPU（~224us flag 时间）和 NPU（~240us graphExecute）并行重叠

Mode 5 (Fast Sync Direct) 较 Mode 4 的改进：NPU 线程直接轮询 GPU flag，省去主线程检测到 flag 后唤醒 NPU 线程的中继延迟。npu_sync 降到接近 0。

**三种 fast sync 的性能对比分析**：
- Mode 4 (Fast Sync): CPU poll → relay → NPU，gpu_sync = 97.7us（含 CPU flag 检测延迟）
- Mode 5 (Fast Sync Direct): NPU 线程直接 poll，消除中继，gpu_sync 降至 44.1us
- Mode 6 (Parallel Sync): DSP 内部 poll，主线程 gpu_sync = 0，但 graphExecute 含 RPC overhead (~80us)

### 绑核配置综合对比

测试了四种 CPU 绑核配置下 Fast Sync 的表现，每种配置同时跑 Seq Blocking 基线：

SM8850 CPU 拓扑：
- Core 0-5 @ 3.63GHz（performance 核）
- Core 6-7 @ 4.61GHz（prime 核）

```
                         Fast Sync                     Seq Baseline
Config              step_p50  gpu_sync  npu_compute   step_p50   speedup
--------------------------------------------------------------------------
No pinning           841.1 us   356.0 us   476.0 us    935.1 us    1.11x
Little+Little (0,1)  752.0 us   364.1 us   383.6 us    941.1 us    1.25x
Big+Little (7,0)     719.2 us   285.9 us   429.8 us    765.3 us    1.06x
Big+Big (7,6) ★      301.5 us   119.9 us   179.8 us    762.4 us    2.53x
```

**关键发现：**

1. **Big+Big (7,6) 是唯一有效配置**：2.53x 加速，其他配置均 < 1.3x
2. **NPU compute 对 CPU 频率极度敏感**：
   - Prime 核 (core 6): 179.8 us
   - Performance 核 (core 0): 429.8 us（+139%）
   - 无绑核: 476.0 us（+165%，spin-loop 竞争 + 调度不确定）
3. **GPU sync 只对主线程核频率敏感**：
   - Big 核 (core 7): 119.9-285.9 us
   - Little 核 (core 0): 364.1 us
4. **NPU graphExecute() 是 CPU RPC 调度到 DSP**：频率越低 RPC 延迟越大

### 绑核效果分析（Fast Sync: 无绑核 vs Big+Big）

| 指标 | 不绑核 | Big+Big (7,6) | 变化 |
|------|--------|---------------|------|
| step_p50 | 841.1 us | 301.5 us | **-64%** |
| gpu_sync | 356.0 us | 119.9 us | -66% |
| npu_compute | 476.0 us | 179.8 us | **-62%** |
| npu_sync | 2.3 us | 0.4 us | -83% |
| speedup | 1.11x | **2.53x** | |

两个主要改善来源：

1. **NPU compute 大幅下降**（476.0→179.8us, -296us）：之前 spin-loop 和 NPU graphExecute
   竞争同一个 CPU 核，导致 DSP 调度严重受阻。绑核后两个线程各自独占 prime 核，消除竞争。
2. **GPU sync 也下降**（356.0→119.9us, -236us）：主线程绑到 prime 核后，flag 轮询响应更快
   （不被抢占），submit 路径也更快。

### 开销来源分析

```
Sequential Blocking (762 us, Big+Big 绑核):
  gpu: clEnqueue + clFinish (含驱动后处理) = 547us (gpu_sync)
  npu: graphExecute = 197us
  合计: 547 + 197 ≈ 762us (串行)

Fast Sync (302 us, Big+Big 绑核):
  gpu: clEnqueue + clFlush + flag_detect = 120us (gpu_sync)
  npu: graphExecute = 180us (绑核消除 CPU 竞争)
  sync: npu_sync ≈ 0us
  合计: 120 + 180 ≈ 302us (gpu→npu 串行，省掉 clFinish 开销 + 消除 CPU 竞争)
```

**绑核后省掉了 ~460us**：clFinish 驱动开销 ~427us + CPU 竞争消除 ~17us NPU 加速。

## 目录结构

```
fast_sync_test/
├── README.md                     # 本文档
├── CMakeLists.txt
├── build_android.sh
├── run_on_device.sh
├── kernels/
│   └── rmsnorm.cl                # GPU FP16 RMSNorm + 完成 flag 写入
├── src/
│   ├── common.h                  # ION/rpcmem + SyncMode/StepTiming/Stats 类型
│   ├── gpu_engine.h/.cpp         # GPU OpenCL: blocking + nonblocking + flag-based
│   ├── npu_engine.h/.cpp         # NPU QNN: standard graph + sync graph (SyncWait)
│   ├── pipeline.h/.cpp           # 六种同步模式 + GPU 诊断
│   ├── main.cpp                  # CLI + 结果输出
│   └── test_graph_overhead.cpp   # 单元测试：分析 QNN 图开销（Config A-G）
└── heteroedge_op/                # 联合 HTP op package（SyncWait + RmsNorm）
    ├── HeteroEdgeInterface.cpp   # 注册接口（heteroedge.HvxOpPackage）
    ├── HeteroEdgeSyncWait.cpp    # SyncWait: dcinva poll + memcpy passthrough
    ├── HeteroEdgeRmsNorm.cpp     # HVX FP16 RmsNorm
    ├── Makefile
    └── build.sh
```

### CPU 绑核

通过 `sched_setaffinity()` 将主线程和 NPU 工作线程分别绑定到不同 CPU 核，
消除 spin-loop 轮询与 NPU graphExecute 的 CPU 竞争。

SM8850 推荐配置：`--main-core 7 --npu-core 6`（两个 prime 核 @ 4.61GHz）。

## 构建与运行

```bash
export ANDROID_NDK=/path/to/android-ndk-r25c
export QNN_SDK_ROOT=/path/to/qualcomm/qairt/2.42.0.251225
export HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.5.0.0

# 编译主程序 + test_graph_overhead
bash build_android.sh

# 编译联合 HTP op package（SyncWait + RmsNorm）
cd heteroedge_op && bash build.sh && cd ..

# 推送到设备并运行（推荐：绑核 Big+Big）
bash run_on_device.sh --main-core 7 --npu-core 6

# 单独测试 Fast Sync Direct（最优模式）
bash run_on_device.sh --mode direct --main-core 7 --npu-core 6

# QNN 图开销单元测试（Config A-G，包括 DMA 拷贝验证）
# 注意：Config G 运行 1 步耗时 ~410ms（QNN timeout 验证）
adb shell "cd /data/local/tmp/fast_sync_test && ./test_graph_overhead --steps 100"
```

## 结论

1. **clFinish 开销确认**：SM8850 上 clFinish 的驱动后处理开销约 **384us**，与论文 ~400us 一致
2. **cl_event 无法替代 flag**：clGetEventInfo 轮询与 clFinish 开销相同（~395us），两者共用驱动代码路径
3. **Flag-based 共享内存轮询有效**：绕过 OpenCL 驱动，GPU 完成检测从 ~676us 降到 ~305us
4. **CPU 绑核至关重要**：不绑核时 spin-loop 与 NPU 竞争导致 NPU compute +68%，绑核后消除竞争
5. **必须绑定到 prime 核**：四种绑核配置中只有 Big+Big (7,6) 有效，NPU RPC 调度对 CPU 频率极敏感
6. **三种 fast sync 模式均达到 ~2.8x 加速**：step_p50 从 761us 降到 261-273us，符合论文 2-4x 预测
7. **主要瓶颈在 GPU 调度**：SUBMIT→START 的 244us 调度延迟不可避免，占 flag wall time 的 80%
8. **QNN DMA 拷贝问题已解决（HAP_mmap_get）**：QNN 在 graphExecute 启动时将 APP_WRITE tensor 拷贝到 DSP TCM，导致 DSP 无法感知 GPU 后续写入（超时 410ms）。解决方案：将 ION fd 作为静态参数传入 SyncWait，DSP 侧用 `HAP_mmap_get(fd)` 获取 ION DDR 的 DSP VA 直接轮询，绕过 QNN tensor 机制。Parallel Sync（Mode 6）现在工作正常（273us, 2.79x）
9. **Parallel Sync 实现 GPU+NPU 真正并行**：主线程 gpu_sync = 0us，GPU 完成检测完全由 DSP 内部 SyncWait op 处理，RPC launch overhead（~80us）与 GPU 执行时间（~224us）重叠
10. **SyncWait 自定义 HTP op 完整实现**：联合包（SyncWait+RmsNorm），`flag_ion_fd` 静态参数 + HAP_mmap_get DDR polling + PlainFloat16Tensor 变体（避免 RmsNorm 退化为 scalar 实现）

## 参考

- **HeteroInfer** (SOSP 2025), Section 4.3: Fast Synchronization
- Qualcomm QNN SDK 2.42.0 — `graphExecute()` (blocking API)
- OpenCL 2.0 — `clFlush()`, `clGetEventInfo()`, `CL_QUEUE_PROFILING_ENABLE`, `CL_PROFILING_COMMAND_*`
- Qualcomm OpenCL extension — `CL_MEM_ION_HOST_PTR_QCOM` (zero-copy ION import)
