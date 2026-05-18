# HeteroInfer 复现完整方案（基于 ExecuTorch）

## 关联代码

- **Repository**: `git.n.xiaomi.com:lvyinrun/executorch.git`
- **Branch**: `feat/qwen3-hetero-baseline`
- **Base commit**: `a49171dff` (upstream main, 2026-04-29)
- **MR/PR**: https://git.n.xiaomi.com/lvyinrun/executorch/-/merge_requests/new?merge_request[source_branch]=feat/qwen3-hetero-baseline

### 当前已实现（commits）

| Commit | 描述 |
|--------|------|
| `ff74c2801` | feat(qwen3): static_qwen3 model 独立化 + KV8 quant variant |
| `4376f0b59` | feat(hetero): W2 in-channel partition prototype + QnnMemManager 修复 |

对应方案章节：
- §1-§2 背景与架构选型 → 设计阶段
- §3 Phase 0-2 → 已部分实现（W2 切分原型 + Qwen3 模型独立化）
- §3 Phase 3 → 未开始（HeteroBackend 真正并行）

## 1. 背景与目标

### 1.1 论文方案概述

HeteroInfer (SOSP'25) 在移动 SoC 上利用 GPU+NPU 异构并行加速 LLM 推理：

- **Layer-level 异构**：Matmul/FFN → NPU；RMSNorm/SwiGLU/Attention → GPU；按算子亲和性分派
- **Tensor-level 异构**：在同一算子内拆分 weight/activation，让 GPU 和 NPU **并行计算**
- **Fast Sync**：UMA shared buffer + flag polling，~1μs 级同步
- **Profiler + Solver**：离线测量算子性能，自动求解最优切分方案
- **架构**：三组件 — Profiler、Solver、Inference Engine

### 1.2 三种 Tensor 切分策略（论文 §4.2）

| 策略 | 适用阶段 | 切分维度 | 静/动态 |
|------|----------|----------|---------|
| **Weight-centric** | Prefill+Decode | Weight 行维度 | Static |
| **Activation-centric** | Prefill | Activation seq_len 维度 | Dynamic |
| **Hybrid** | Prefill 中等长度 | Weight 行 + Activation seq_len | 混合 |

### 1.3 复现目标

- **平台**：SM8850 (Snapdragon 8 Elite) + ExecuTorch QNN backend
- **模型**：Qwen3 0.6B（28 层标准 Transformer + GQA）
- **基准**：当前 baseline decode 118 tok/s（max_seq_len=1024, taskset c0）
- **目标**：layer-level 加速 ≥ baseline；tensor-level 在 prefill 阶段 ≥ 1.3× layer-level

---

## 2. 架构设计

### 2.1 核心架构选型：HeteroBackend + HeteroPartitioner

**当前 W2 切分实验为什么没有性能收益：**

切分思路本身（weight in-channel partition）是论文 §4.2.1 的方案，没问题。问题在于实现方式——**通过 ExecuTorch Partitioner 把切分点暴露成多个 delegate 边界**：

- `hetero.split_w2_cpu` 被标记为 skip → ExecuTorch 把它独立成一个 CPU op
- 每层 w2 切分变成：`[QNN delegate: ...w2_a]` → `[CPU op: split_w2_cpu]` → `[QNN delegate: add...]`
- ExecuTorch runtime 是单线程指令循环，逐个调度 `DelegateCall` 和 `KernelCall`，全部阻塞同步执行
- 28 层 × 3 次 dispatch = 84 次切换，每次有 ~69μs launch overhead，无任何并行可能

实测 decode 速度 23 tok/s（baseline 118 tok/s），慢 5.1×。

**HeteroBackend 方案如何解决：**

同样做 weight 切分，但切分点**在 backend 内部**，对 ExecuTorch runtime 不可见：
- ExecuTorch 看到的：整层 transformer = 1 个 delegate call
- HeteroBackend 内部：多线程同时调度 NPU graphExecute + GPU OpenCL enqueue
- Fast Sync 在 backend 内部完成（不通过 runtime）

伪代码对比：

```cpp
// 当前 W2 实验（图层面切分）
ExecuTorch runtime 调度：
  qnn_graph_execute(graph_A);   // 阻塞等待
  cpu_op_split_w2(...);          // 阻塞等待
  qnn_graph_execute(graph_B);   // 阻塞等待
// NPU 和 CPU 必然串行

// HeteroBackend 方案（同样的 weight 切分）
HeteroBackend::execute() 内部：
  std::thread t_npu([]{ qnn_graph_execute(w2_a_graph); });
  opencl_enqueue(w2_b_kernel);  // GPU 同时启动
  fast_sync_wait_both();         // ~1μs
// NPU 和 GPU 真正并行
```

**为什么不能用 Plan.md 的图层面切分路线：**
- 已实验证明（Hetero Norm split 58 子图、W2 split 29 子图）：每个子图边界引入 ~69μs launch overhead
- ExecuTorch runtime 是单线程，**delegate call 阻塞**，无法实现 GPU/NPU 并行
- 不管多少种切分方式（按 norm 切、按 weight 切、按 op 类型切），只要切分暴露成 delegate 边界，结果都一样

**为什么需要 HeteroBackend：**
- 整层 transformer 作为单个 delegate，对 ExecuTorch 图不透明
- Backend 内部用多线程同时调度 GPU OpenCL 和 NPU QNN
- 内部仍然有"图切分"——但这个切分对 runtime 不可见，是 backend 自己控制的执行时序
- 跨层 pipeline 重叠可能（Figure 10）

### 2.2 整体组件

```
[ Offline ]                          [ Runtime ]
  ┌─────────────┐                      ┌─────────────────────────┐
  │ Profiler    │ → perf_metrics.json  │ ExecuTorch .pte         │
  │ (per-op     │                      │   ├─ embedding (CPU)    │
  │  GPU/NPU    │ ↓                    │   ├─ HeteroBackend ─────┤
  │  perf       │                      │   │     init():         │
  │  measurement)│ ┌───────────┐       │   │       OpenCL ctx    │
  └─────────────┘ │ Solver    │       │   │       QNN HTP ctx   │
        ↓         │ (decide   │       │   │       Fast Sync     │
                  │  partition│       │   │     execute():      │
  ┌─────────────┐ │  ratio per│       │   │       28-layer loop │
  │ Export      │ │  op)      │       │   │         GPU ‖ NPU   │
  │ Pipeline    │ │           │       │   │                     │
  │  - Partit-  │ └───────────┘       │   └─ lm_head (CPU)      │
  │    ioner    │       ↓              └─────────────────────────┘
  │  - Backend  │   partition_plan.json
  │    Details  │       ↓
  │  - Quant    │ ┌───────────────┐
  │    (W4A16)  │ │ Code Gen      │
  └─────────────┘ │ Strategy =    │
                  │ {Weight,Act,  │
                  │  Hybrid}      │
                  └───────────────┘
```

### 2.3 模块映射

| 论文模块 | 复现实现 |
|----------|----------|
| Profiler | Python 脚本 + 设备端 micro-benchmark |
| Solver | Python LP/grid search |
| Inference Engine | C++ HeteroBackend + ExecuTorch runtime |
| GPU OpenCL kernels | 复用 `fast_sync_test/kernels/` + 新写 |
| NPU QNN HTP | 复用 ExecuTorch QNN backend 的 HTP 图构建 |
| Fast Sync | 复用 `fast_sync_test/` 6 模式之一（Pipeline + flag polling）|
| W4A16 量化 | 复用 ExecuTorch `Qwen3_0_6B_KV8QuantRecipe` |

---

## 3. 实现计划（4 个 Phase）

### Phase 0: 基础验证与 Profiler（2 周）

**目标**：建立 per-op 性能数据库，验证子图拆分正确性。

#### 0.1 Profiler 实现

为每个目标 op（QKV/O/gate_up/down 的 matmul）构建独立 micro-benchmark：

```cpp
// HeteroEdge/profiler/op_bench.cpp
struct OpProfile {
  std::string op_type;     // "matmul"
  int64_t M, N, K;          // tensor shape
  std::string backend;      // "gpu" | "npu"
  double latency_us;
  double bandwidth_gbps;
};
```

测量集合：
- 所有 Qwen3 0.6B 实际 weight shape
- Activation 长度：1, 32, 64, 128, 256, 512, 1024
- 每个 (op, shape) 在 GPU 和 NPU 各跑 100 次取 p50

输出：`profile_results.json` 供 Solver 使用。

#### 0.2 Attention/MLP 拆分验证

复用 `hetero_split_test/`：
- 把每层拆为 Attention QNN graph + MLP QNN graph
- 验证拆分输出 = 未拆分输出（前 20 token 一致）
- 测量两个子图的独立延迟（用于 Solver）

**验收**：
- 数值一致性（greedy decode）
- 单层 NPU MLP TFLOPS 实测数据

#### 0.3 Fast Sync 模式选定

复用 `fast_sync_test/` 已有 6 种模式，选最优：
- 目标：< 50μs/sync（论文报告 ~1μs）
- 评估：连续 28 层 sync 累积 vs `clFinish` baseline

---

### Phase 1: HeteroBackend C++ 框架（4 周）

**目标**：实现 ExecuTorch BackendInterface，支持 layer-level 异构（不并行，先验证正确性）。

#### 1.1 HeteroBackend 类骨架

```cpp
// HeteroEdge/runtime/hetero_backend.h
class HeteroBackend final : public BackendInterface {
  bool is_available() const override { return true; }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,    // serialized blob
      ArrayRef<CompileSpec> compile_specs) const override;

  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle,
      Span<EValue*> args) const override;

  void destroy(DelegateHandle* handle) const override;
};

// 注册
static HeteroBackend hetero_backend;
auto _ = register_backend({"HeteroBackend", &hetero_backend});
```

#### 1.2 Init 阶段做的事

```cpp
struct HeteroGraph {  // DelegateHandle
  // Per-layer 资源
  std::vector<OpenCLContext*> gpu_ctxs;       // GPU side
  std::vector<QnnGraph*> npu_mlp_graphs;      // NPU side
  std::vector<IonBuffer> shared_buffers;      // ION zero-copy
  
  // Solver 决定的执行计划
  PartitionPlan plan;                         // layer × {weight, activation, hybrid, none}
  
  // Fast Sync
  FastSyncManager sync;
};

Result<DelegateHandle*> HeteroBackend::init(...) {
  // 1. 反序列化 blob
  auto graph = deserialize(processed);
  
  // 2. 初始化 OpenCL（复用 fast_sync_test/gpu_engine.cpp）
  graph->gpu_ctxs = init_opencl_per_layer();
  
  // 3. 初始化 QNN HTP（复用 ExecuTorch 的 QnnManager）
  graph->npu_mlp_graphs = init_qnn_per_layer(blob.npu_data);
  
  // 4. 分配 ION shared buffer 池
  graph->shared_buffers = allocate_ion_pool(...);
  
  return graph;
}
```

#### 1.3 Execute 阶段（先实现串行版本）

```cpp
Error HeteroBackend::execute(...) {
  auto* graph = static_cast<HeteroGraph*>(handle);
  
  // 从 args 取输入 tensor，写到 ION shared buffer
  ion_copy_in(args[0], graph->shared_buffers.input);
  
  for (int L = 0; L < 28; ++L) {
    // 串行版本：先 GPU attention，再 NPU MLP
    gpu_run_attention(graph->gpu_ctxs[L], ...);
    qnn_graph_execute(graph->npu_mlp_graphs[L], ...);
  }
  
  ion_copy_out(graph->shared_buffers.output, args[1]);
  return Error::Ok;
}
```

#### 1.4 GPU OpenCL Kernels

需要实现的 kernel（论文 §3.1，Figure 6）：

| Kernel | 用途 | 来源 |
|--------|------|------|
| `rmsnorm.cl` | RMSNorm | 复用 `fast_sync_test/kernels/` |
| `qkv_split.cl` | Q/K/V split | 新写 |
| `rope.cl` | Rotary Position Embedding | 新写 |
| `softmax.cl` | Attention softmax | 新写 |
| `attention_mvm.cl` | Attention matmul | 新写 |
| `swiglu.cl` | SiLU(gate) * up | 新写 |
| `gemv_w4a16.cl` | Decode GEMV (W4A16) | 新写 |
| `gemm_w4a16.cl` | Prefill GEMM (W4A16) | 新写 |

#### 1.5 NPU QNN HTP 图

复用 ExecuTorch QnnManager + node_visitor，但在 HeteroBackend 内部直接构建（不通过 ExecuTorch partitioner）：

```cpp
QnnGraph* build_mlp_graph(int layer_idx, const Weights& w) {
  // 用 QNN C++ API 直接构建：
  // RMSNorm → MatMul(gate_up) → SiLU → Mul → MatMul(down)
  auto graph = qnn_create_graph(...);
  qnn_add_op(graph, "RMSNorm", ...);
  qnn_add_op(graph, "MatMul", ...);  // gate_up
  qnn_add_op(graph, "Sigmoid", ...);  // SiLU 部分
  qnn_add_op(graph, "Mul", ...);
  qnn_add_op(graph, "MatMul", ...);  // down
  qnn_finalize_graph(graph);
  return graph;
}
```

**Phase 1 验收**：单层 GPU+NPU 串行版本端到端跑通，输出与 baseline 一致。

---

### Phase 2: Export Pipeline（2 周）

**目标**：让 `.pte` 文件能被 HeteroBackend 加载执行。

#### 2.1 HeteroPartitioner

```python
# executorch/backends/hetero/hetero_partitioner.py
class HeteroPartitioner(Partitioner):
    """把整个 transformer 标记为 hetero_block，让 HeteroBackend 接管。"""
    
    def partition(self, exported_program) -> PartitionResult:
        graph = exported_program.graph
        
        # 标记所有属于 transformer 层的节点
        for node in graph.nodes:
            if is_in_transformer_layer(node):
                node.meta["delegation_tag"] = "hetero_block"
        
        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags={
                "hetero_block": DelegationSpec(
                    "HeteroBackend",
                    [self.compile_spec],
                )
            },
        )
```

#### 2.2 HeteroBackendDetails (preprocess)

```python
class HeteroBackendDetails(BackendDetails):
    @staticmethod
    def preprocess(edge_program, compile_specs) -> PreprocessResult:
        # 1. 收集每层的权重并 W4A16 量化
        layer_weights = quantize_w4a16(edge_program)
        
        # 2. 加载 Solver 输出的 partition_plan.json
        plan = load_partition_plan(compile_specs)
        
        # 3. 序列化为 blob：
        #    - 每层 weight (W4A16)
        #    - 每层 GPU/NPU partition ratio
        #    - Tensor shape metadata
        blob = serialize_hetero_graph(layer_weights, plan)
        
        return PreprocessResult(processed_bytes=blob)
```

#### 2.3 导出脚本

```bash
python export_hetero.py \
  --model qwen3-0_6b-kv8 \
  --soc SM8850 \
  --partition_plan partition_plan.json \
  --output qwen3_hetero.pte
```

**Phase 2 验收**：导出 .pte，HeteroBackend 能加载并执行。

---

### Phase 3: 并行执行 + Tensor-level 切分（4 周）

**目标**：实现真正的 GPU/NPU 并行 + 三种切分策略。

#### 3.1 并行执行（论文 Figure 10）

**Prefill 阶段（NPU-dominant）**：
```cpp
for (int L = 0; L < 28; ++L) {
  // 主线程: 启动 NPU
  qnn_graph_execute_async(npu_mlp[L]);  // 不等待返回
  
  // GPU 同时算下一层 attention（提前一层）
  if (L + 1 < 28)
    opencl_enqueue(attention_kernel[L+1]);
  
  // Fast Sync: 等 NPU 完成
  fast_sync_wait(npu_mlp[L].done_flag);
}
```

**Decode 阶段（GPU-dominant）**：
```cpp
for (int L = 0; L < 28; ++L) {
  // GPU 算 Attention
  opencl_enqueue(attention_kernel[L]);
  
  // NPU 同时辅助算 MLP 的一部分（weight-centric partition）
  qnn_graph_execute_async(npu_mlp_partial[L]);
  
  // GPU 算 MLP 的另一部分
  opencl_enqueue(gpu_mlp_partial_kernel[L]);
  
  fast_sync_merge(...);
}
```

#### 3.2 Weight-centric partition（论文 §4.2.1）

```cpp
struct WeightCentricPlan {
  int npu_size;   // 例如 4096
  int gpu_size;   // 例如 4096，partition_ratio = 1:1
};

void execute_weight_centric(layer L, WeightCentricPlan plan) {
  // NPU 算 weight 的 [0:npu_size] 行
  qnn_graph_execute_async(npu_partial[L]);  // weight: [npu_size, K]
  
  // GPU 算 weight 的 [npu_size:end] 行  
  opencl_enqueue(gpu_partial[L]);  // weight: [gpu_size, K]
  
  fast_sync_wait_both();
  
  // Concat output: [npu_out, gpu_out] → [out_size, *]
  ion_concat(npu_out, gpu_out, final_out);
}
```

#### 3.3 Activation-centric partition（论文 §4.2.2）

```cpp
struct ActivationCentricPlan {
  int npu_seq_size;   // 必须是 NPU 标准 shape，如 256
  int gpu_seq_size;   // 任意
};

void execute_activation_centric(layer L, int seq_len, ActivationCentricPlan plan) {
  // NPU 用 pre-generated graph (静态 shape) 算前 npu_seq_size 个 token
  qnn_graph_execute_async(npu_static[L]);  // activation: [npu_seq_size, K]
  
  // GPU 算剩余 gpu_seq_size 个 token（动态 shape）
  opencl_enqueue(gpu_dynamic[L]);  // activation: [gpu_seq_size, K]
  
  fast_sync_wait_both();
  ion_concat_seq(...);
}
```

#### 3.4 Hybrid partition（论文 §4.2.3）

混合 weight + activation 切分。核心想法：先用 weight-centric 把 op 切到 NPU 友好的 shape，再用 activation-centric 处理动态部分。

#### 3.5 Solver 实现

输入：profiler 数据 + 每层 op shape
输出：每个 op 的最优 partition strategy + ratio

```python
def solve_partition(op_shape, gpu_perf, npu_perf, sync_overhead):
    # 论文公式：
    # T_total = max(T_gpu(partition1), T_npu(partition2)) + T_sync + T_copy
    # s.t. partition1 + partition2 = All
    
    candidates = []
    for ratio in [1:0, 3:1, 1:1, 1:3, 0:1]:
        t_gpu = predict_gpu(op_shape, ratio.gpu_size, gpu_perf)
        t_npu = predict_npu(op_shape, ratio.npu_size, npu_perf)
        t_total = max(t_gpu, t_npu) + sync_overhead
        candidates.append((ratio, t_total))
    
    return min(candidates, key=lambda x: x[1])
```

**Phase 3 验收**：
- Prefill 加速 ≥ 1.5× layer-level
- Decode 加速 ≥ 1.2× layer-level
- 所有切分策略数值正确

---

## 4. 文件结构

```
HeteroEdge/
├── docs/
│   ├── heteroinfer-replication-plan.md    # 本文件
│   ├── hetero-weight-partition-plan.md    # Phase 3.2 子方案
│   └── workflow-baseline-export-verify.md
├── profiler/
│   ├── op_bench.cpp                       # 设备端 micro-benchmark
│   ├── run_profiler.sh
│   └── profile_results.json               # 输出
├── solver/
│   ├── solve_partition.py                 # Solver 实现
│   ├── partition_plan.json                # 输出
│   └── partition_strategies.py            # 三种策略
├── runtime/                               # HeteroBackend C++
│   ├── hetero_backend.cpp/.h
│   ├── hetero_graph.h                     # DelegateHandle 数据结构
│   ├── gpu_engine.cpp/.h                  # OpenCL 封装（复用 fast_sync_test）
│   ├── npu_engine.cpp/.h                  # QNN HTP 封装
│   ├── fast_sync.cpp/.h                   # 复用 fast_sync_test
│   ├── ion_pool.cpp/.h                    # Shared buffer 管理
│   └── partition_executor.cpp/.h          # 三种切分策略执行
├── kernels/                               # OpenCL kernels
│   ├── rmsnorm.cl                         # 复用
│   ├── rope.cl
│   ├── attention.cl
│   ├── swiglu.cl
│   ├── gemv_w4a16.cl
│   └── gemm_w4a16.cl
├── export/                                # Python
│   ├── hetero_partitioner.py              # ExecuTorch Partitioner
│   ├── hetero_backend_details.py          # ExecuTorch BackendDetails
│   ├── export_hetero.py                   # 导出脚本
│   └── quantize_w4a16.py                  # 复用 ExecuTorch 量化
├── runner/
│   ├── main.cpp                           # 基于 ET LLM runner
│   └── CMakeLists.txt
└── tests/
    ├── test_hetero_backend.cpp            # 单元测试
    └── test_partition_strategies.py       # Python 测试
```

---

## 5. 时间线与里程碑

| Phase | 周数 | 累计 | 关键里程碑 |
|-------|------|------|-----------|
| 0. Profiler + 子图验证 | 2 | 2 | per-op 性能数据库；Attention/MLP 拆分正确性 |
| 1. HeteroBackend 串行版 | 4 | 6 | 整层串行执行端到端正确 |
| 2. Export Pipeline | 2 | 8 | .pte 加载成功，HeteroBackend init 通过 |
| 3.1 GPU/NPU 并行 | 1 | 9 | Layer-level 并行加速验证 |
| 3.2 Weight-centric | 1 | 10 | Decode tensor-level 加速 ≥ 1.2× |
| 3.3 Activation-centric | 1 | 11 | Prefill 动态长度支持 |
| 3.4 Solver | 1 | 12 | 自动决策 + 端到端最优配置 |

**总计：12 周（3 个月）**

---

## 6. 已有可复用资源

| 组件 | 路径 | 复用度 |
|------|------|--------|
| Fast Sync 6 模式 | `HeteroEdge/fast_sync_test/` | 直接复用 |
| GPU OpenCL 框架 | `HeteroEdge/fast_sync_test/src/gpu_engine.cpp` | 直接复用 |
| RMSNorm CL kernel | `HeteroEdge/fast_sync_test/kernels/rmsnorm.cl` | 直接复用 |
| ION buffer 工具 | `HeteroEdge/fast_sync_test/src/common.h` | 直接复用 |
| QNN HTP 图构建 | `executorch/backends/qualcomm/runtime/QnnManager.cpp` | 修改复用 |
| QNN node_visitor (105 ops) | `executorch/backends/qualcomm/builders/` | 修改复用 |
| W4A16 量化 | `executorch` Qwen3_0_6B_KV8QuantRecipe | 直接复用 |
| 子图拆分 runner | `HeteroEdge/hetero_split_test/` | Phase 0 起点 |
| Qwen3 baseline 数据 | 118 tok/s decode | 对照基准 |

---

## 7. 风险与降级方案

| 风险 | 影响 | 降级方案 |
|------|------|---------|
| GPU OpenCL kernels 性能不达标 | Layer-level 加速失败 | 用 NEON CPU 替代某些 GPU op |
| Fast Sync 实测 > 100μs | 并行收益消失 | 回退到 layer-level，不做 tensor-level |
| W4A16 在 GPU 上精度损失 | 输出乱码 | GPU 用 fp16，只 NPU 量化 |
| QNN HTP 静态 shape 限制 | Activation-centric 失败 | 只实现 Weight-centric |
| ExecuTorch BackendInterface 限制 | HeteroBackend 集成困难 | 退化为独立 C++ runner，不依赖 ExecuTorch |

---

## 8. 与已有方案的对应

| 论文章节 | 现有文档/代码 | 状态 |
|---------|--------------|------|
| §3 SoC characterization | `HeteroEdge/README.md`、`UMA验证总结.md` | ✅ 已完成 |
| §4.1 Layer-level | `HeteroEdge/Plan.md`、`hetero_layer_plan.md` | 设计完成 |
| §4.2.1 Weight-centric | `HeteroEdge/docs/hetero-weight-partition-plan.md` + `replace_w2_conv.py` | ⚠️ 当前实现是死路（图层面切分），需重写到 HeteroBackend 内部 |
| §4.2.2 Activation-centric | - | 未开始 |
| §4.2.3 Hybrid | - | 未开始 |
| §4.3 Fast Sync | `HeteroEdge/fast_sync_test/` | ✅ 已完成 |
| §4.4 Solver | - | 未开始 |

---

## 9. 一句话总结

**论文的核心是 GPU+NPU 并行执行 + Fast Sync，实现路径必须是 HeteroBackend（自定义 ExecuTorch backend，整层接管）。当前的 W2 切分实验属于图层面切分路线，已证明无法实现并行，需要全部重新实现到 HeteroBackend 内部。**

12 周可实现端到端 layer-level + tensor-level 异构推理，目标性能：prefill 加速 5-7×，decode 加速 1.5-2×（参考论文 Llama-8B 数据）。
