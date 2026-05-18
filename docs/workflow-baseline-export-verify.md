# Workflow: Baseline 导出与验证

## 环境

- 台式机：`yinrun-ThinkCentre-M760t`
- Conda：`qwen3dp`
- QNN SDK：`/home/yinrun/software/qualcomm/qairt/2.42.0.251225`
- ExecuTorch：`/home/yinrun/workspace/HeteroEdge/executorch` (main 分支)
- 设备：SM8850 (adb serial: 204cbd30)
- 产物目录：`/home/yinrun/workspace/HeteroEdge/qwen3_perf/artifacts/`

## Step 1: 导出

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate qwen3dp
export QNN_SDK_ROOT=/home/yinrun/software/qualcomm/qairt/2.42.0.251225
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export PYTHONPATH=$QNN_SDK_ROOT/lib/python:$PYTHONPATH

cd /home/yinrun/workspace/HeteroEdge/executorch

python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android \
  -m SM8850 \
  -a /home/yinrun/workspace/HeteroEdge/qwen3_perf/artifacts/baseline_kv8 \
  --decoder_model qwen3-0_6b-kv8 \
  --model_mode hybrid \
  --prefill_ar_len 128 \
  --max_seq_len 1024 \
  --prompt "What is the capital of France?" \
  --compile_only
```

耗时约 12 分钟。产物：`hybrid_llama_qnn.pte` (~666MB) + `tokenizer.json` + `chat_template.jinja`

## Step 2: Push 到设备

```bash
ADB="adb -s 204cbd30"
DEV=/data/local/tmp/hetero_test

$ADB shell "mkdir -p $DEV"

# Runner + QNN libs（只需首次 push）
$ADB push build-android/examples/qualcomm/oss_scripts/llama/qnn_llama_runner $DEV/
$ADB push build-android/backends/qualcomm/libqnn_executorch_backend.so $DEV/
$ADB push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so $DEV/
$ADB push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV81Stub.so $DEV/
$ADB push $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so $DEV/
$ADB push $QNN_SDK_ROOT/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so $DEV/

# 模型 + tokenizer
$ADB push artifacts/baseline_kv8/hybrid_llama_qnn.pte $DEV/baseline.pte
$ADB push artifacts/baseline_kv8/tokenizer.json $DEV/
```

## Step 3: 设备推理

```bash
$ADB shell "cd $DEV && \
  export LD_LIBRARY_PATH=$DEV && \
  export ADSP_LIBRARY_PATH=$DEV && \
  chmod +x qnn_llama_runner && \
  taskset c0 ./qnn_llama_runner \
    --model_path baseline.pte \
    --tokenizer_path tokenizer.json \
    --prompt 'What is the capital of France?' \
    --temperature 0 \
    --seq_len 100 \
    --shared_buffer \
    --eval_mode 0 \
    --decoder_model_version qwen3 \
    -output_path output.txt"
```

参数说明：
- `taskset c0`：绑定大核 6+7
- `--eval_mode 0`：KV mode（纯 decode，AR-1 逐 token）
- `--eval_mode 1`：hybrid mode（AR-128 prefill + decode）
- `--shared_buffer`：启用 ION shared buffer
- `--seq_len 100`：生成上限（prompt + output）

## Step 4: 检查结果

```bash
# 查看性能
$ADB shell "cat $DEV/output.txt"  # 生成文本

# 从 runner stdout 中提取 PyTorchObserver JSON
# prefill_token_per_sec / decode_token_per_sec
```

## 预期结果

| 配置 | Prefill tok/s | Decode tok/s |
|------|---------------|-------------|
| max_seq_len=1024, kv8, taskset c0 | ~116 | ~118 |
| max_seq_len=128, kv8, taskset c0 | ~133 | ~135 |
| max_seq_len=1024, kv8, taskset f0 | ~93 | ~88 |

## 注意事项

1. **Prompt 必须加引号**：否则校准数据异常导致量化参数错误
2. **Cold/Warm 差异**：首次 model_load ~5-6s，后续 ~0.8s；decode 速度也有差异
3. **Token 数影响**：KV cache 越大 decode 越慢，对比时需控制相同 seq_len
4. **Runner 需含 custom ops**：CMakeLists 中需链接 `op_split_w2_cpu.cpp` 和 `op_attention_norm.cpp`
