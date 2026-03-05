#pragma once
#include "common.h"

enum class NpuMode {
  NATIVE,      // Use QNN_OP_RMS_NORM fused op (FP16)
  DECOMPOSED   // Manual decomposition into 6 FP16 ops
};

bool npu_rmsnorm_init(const RMSNormConfig& config, NpuMode mode);
RMSNormResult npu_rmsnorm_run(const RMSNormConfig& config, int num_warmup, int num_iters);
bool npu_rmsnorm_read_output(void* dst, size_t bytes);
void npu_rmsnorm_print_info();
void npu_rmsnorm_cleanup();
