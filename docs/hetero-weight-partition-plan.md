# Hetero Weight Partition 验证方案

## 关联代码

- **Repository**: `git.n.xiaomi.com:lvyinrun/executorch.git`
- **Branch**: `feat/qwen3-hetero-baseline`
- **Commit**: `4376f0b59` (feat(hetero): W2 in-channel partition prototype)
- **关键文件**:
  - `extension/llm/custom_ops/op_split_w2_cpu.cpp` — CPU portable kernel
  - `examples/qualcomm/oss_scripts/llama/hetero_partition/replace_w2_conv.py` — Python source transform
  - `examples/qualcomm/oss_scripts/llama/wrappers/llm_wrappers.py` — `--hetero_split_w2` flag handling

总方案见 [`heteroinfer-replication-plan.md`](./heteroinfer-replication-plan.md)。

## 背景

论文 HeteroInfer §4.2 提出 Weight-centric Partition：将 FFN down_proj (w2) 的权重沿 in_channels 切分，NPU 和 CPU/GPU 各算一半，结果相加。数学等价：

```
y = W·x = [W_a | W_b] · [x_a; x_b] = W_a·x_a + W_b·x_b
```

当前实现（`upstream/hetero_split_w2`）已在 ExecuTorch 上跑通：
- Python source transform：`replace_w2_conv.py` 将 `w2_conv` 拆为 `w2_a` (NPU) + `w2_b` (CPU)
- C++ portable kernel：`op_split_w2_cpu.cpp` 执行 CPU 半的 fp32 1x1 conv2d
- QNN partitioner 通过 `skip_node_op_set` 跳过 CPU 半

## 目标

验证 W2 in-channel 切分在 SM8850 上的：
1. **正确性**：切分后输出与 baseline 一致（greedy decode, temp=0）
2. **性能开销**：量化子图边界引入的 launch overhead
3. **切分比例影响**：不同 split ratio 对精度和性能的影响

## 实验配置

### 模型与设备
- 模型：Qwen3 0.6B（28 层，hidden_dim=3072, dim=1024）
- 量化：W4A16 + kv8（`qwen3-0_6b-kv8` recipe）
- 设备：SM8850 (204cbd30)
- 模式：hybrid mode, max_seq_len=128, prefill_ar_len=128
- Runner：含 `op_split_w2_cpu` + `op_attention_norm` 的 `qnn_llama_runner`

### 实验矩阵

| 实验 | 切分层数 | split 比例 | 目的 |
|------|---------|-----------|------|
| E0 | 0（baseline） | - | 对照基准 |
| E1 | layer 0 only | 50/50 (1536) | 单层正确性 + 最小开销 |
| E2 | 全 28 层 | 50/50 (1536) | 全模型开销 |
| E3 | 全 28 层 | 75/25 (2304/768) | NPU 多算，CPU 少算 |
| E4 | 全 28 层 | 25/75 (768/2304) | NPU 少算，CPU 多算 |

### 评估指标

| 指标 | 方法 |
|------|------|
| 正确性 | greedy decode 前 50 token 与 baseline 逐 token 对比 |
| Decode 速度 | KV mode (eval_mode=0), 相同 prompt, seq_len=100, 取 3 次中位数 |
| Prefill 速度 | hybrid mode (eval_mode=1), 相同 prompt |
| 子图数 | 从 runner log 统计 "Deserializing processed data" 出现次数 |
| 内存 | RSS from runner stats |

## 验证步骤

### Step 0: 环境准备

```bash
# 确认当前工作树状态
cd /home/yinrun/workspace/HeteroEdge/executorch
git status  # 应在 main + hetero patches

# 确认 runner 已编译（含 op_split_w2_cpu）
file build-android/examples/qualcomm/oss_scripts/llama/qnn_llama_runner
# → ELF 64-bit LSB pie executable, ARM aarch64

# 确认设备连接
adb -s 204cbd30 shell "echo ok"
```

### Step 1: 导出 Baseline (E0)

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate qwen3dp
export QNN_SDK_ROOT=/home/yinrun/software/qualcomm/qairt/2.42.0.251225
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export PYTHONPATH=$QNN_SDK_ROOT/lib/python:$PYTHONPATH

cd /home/yinrun/workspace/HeteroEdge/executorch

python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android -m SM8850 \
  -a /home/yinrun/workspace/HeteroEdge/qwen3_perf/artifacts/hetero_baseline_kv8_fresh \
  --decoder_model qwen3-0_6b-kv8 \
  --model_mode hybrid --prefill_ar_len 128 --max_seq_len 128 \
  --prompt "What is the capital of France?" \
  --compile_only
```

### Step 2: 导出 SplitW2 (E1-E4)

修改 `wrappers/llm_wrappers.py` 中的 `layer_indices` 和 `split` 参数：

```python
# E1: layer 0 only, split=1536
layer_indices = [0]
replace_w2_conv(decoder, layer_indices=layer_indices, split=1536)

# E2: 全 28 层, split=1536
layer_indices = list(range(len(decoder.layers)))
replace_w2_conv(decoder, layer_indices=layer_indices, split=1536)

# E3: 全 28 层, split=2304 (75% NPU)
replace_w2_conv(decoder, layer_indices=layer_indices, split=2304)

# E4: 全 28 层, split=768 (25% NPU)
replace_w2_conv(decoder, layer_indices=layer_indices, split=768)
```

导出命令加 `--hetero_split_w2`：
```bash
python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android -m SM8850 \
  -a /home/yinrun/workspace/HeteroEdge/qwen3_perf/artifacts/hetero_split_w2_E2 \
  --decoder_model qwen3-0_6b-kv8 \
  --model_mode hybrid --prefill_ar_len 128 --max_seq_len 128 \
  --prompt "What is the capital of France?" \
  --compile_only --hetero_split_w2
```

### Step 3: 设备推理

```bash
ADB="adb -s 204cbd30"
DEV=/data/local/tmp/hetero_w2_test

# Push runner + libs (只需一次)
$ADB push build-android/examples/qualcomm/oss_scripts/llama/qnn_llama_runner $DEV/
$ADB push build-android/backends/qualcomm/libqnn_executorch_backend.so $DEV/
$ADB push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so $DEV/
$ADB push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV81Stub.so $DEV/
$ADB push $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so $DEV/
$ADB push $QNN_SDK_ROOT/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so $DEV/

# Push pte + tokenizer
$ADB push <artifact_dir>/hybrid_llama_qnn.pte $DEV/model.pte
$ADB push <artifact_dir>/tokenizer.json $DEV/

# 运行 (KV mode for decode benchmark)
$ADB shell "cd $DEV && \
  export LD_LIBRARY_PATH=$DEV && \
  export ADSP_LIBRARY_PATH=$DEV && \
  taskset f0 ./qnn_llama_runner \
    --model_path model.pte \
    --tokenizer_path tokenizer.json \
    --prompt 'What is the capital of France?' \
    --temperature 0 --seq_len 100 \
    --shared_buffer --eval_mode 0 \
    --decoder_model_version qwen3 \
    -output_path output.txt"
```

### Step 4: 正确性对比

```bash
# 拉取输出
$ADB pull $DEV/output.txt ./output_E1.txt

# 逐 token 对比（人工检查前 50 token）
diff <(head -5 output_baseline.txt) <(head -5 output_E1.txt)
```

判定标准：
- **通过**：前 30 token 完全一致（允许后续 diverge）
- **警告**：前 10 token 一致但 30 token 内分叉
- **失败**：前 10 token 就不一致

## 预期结果

| 实验 | 预期 Decode tok/s | 预期子图数 | 预期正确性 |
|------|------------------|-----------|-----------|
| E0 (baseline) | ~88 | 1 | ✅ |
| E1 (layer 0) | ~80-88 | 2 | ✅ 前 30 token 一致 |
| E2 (全 28 层) | ~20-40 | 29 | ✅ 前 10 token 一致 |
| E3 (75/25) | ~20-40 | 29 | ✅ |
| E4 (25/75) | ~15-30 | 29 | ⚠️ CPU 负担重 |

### 性能退化分析模型

```
decode_overhead = N_subgraphs × launch_overhead_per_call
               = 29 × 0.069ms (RPC polling)
               = 2.0ms/token

baseline_decode_time = 1/88 = 11.4ms/token
split_w2_decode_time ≈ 11.4 + 2.0 + cpu_matmul_time
                     ≈ 11.4 + 2.0 + 1536×1024/(1 GFLOPS) ≈ 11.4 + 2.0 + 1.6
                     ≈ 15.0ms/token → ~67 tok/s (理论上限)
```

实际可能更慢（CPU matmul 无 SIMD 优化、cache miss 等）。

## 已知问题与注意事项

1. **Prefill shape bug（已修复）**：`op_split_w2_cpu.cpp` 需支持 `[bsz, in_ch, 1, seq_len]`
2. **Prompt 引号**：导出时 `--prompt` 必须加引号，否则校准数据异常导致乱码
3. **Cold/Warm 差异**：首次加载 model_load ~6s，warm run ~0.8s；decode 速度也有差异，需 warmup 后取数
4. **Token 数不同**：SplitW2 可能提前 hit EOS，导致平均 decode rate 偏高（KV cache 小时更快）。对比时需控制相同 token 数或用足够长的 seq_len
5. **QnnMemManager 全局缓存**：已 cherry-pick `2e99cad69` + `bc1293d2a` 修复多子图 shared buffer 问题

## 后续方向

验证完成后，根据结果决定：
- 如果 E2 decode 在 40+ tok/s → 子图边界开销可接受，继续优化 CPU kernel（NEON SIMD）
- 如果 E2 decode < 20 tok/s → 确认图层面切分是死路，转向 HeteroBackend 方案（单 delegate 内部编排）
- 如果 E3/E4 显示切分比例对精度有影响 → 需要 calibration-aware split point 选择

最终目标：将 CPU 半替换为 GPU OpenCL kernel，实现 NPU+GPU 真正并行（Fast Sync ~1μs 同步）。
