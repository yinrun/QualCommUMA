#pragma once
#include "common.h"

enum class NpuMode {
  NATIVE,      // Use QNN_OP_RMS_NORM fused op (FP16)
  DECOMPOSED   // Manual decomposition into 6 FP16 ops
};

bool npu_rmsnorm_init(const RMSNormConfig& config, NpuMode mode);
// Init via context binary cache: build graph → serialize → deserialize → graphRetrieve
// Reports cache load time via out_cache_load_us
bool npu_rmsnorm_init_cached(const RMSNormConfig& config, NpuMode mode, double& out_cache_load_us);
RMSNormResult npu_rmsnorm_run(const RMSNormConfig& config, int num_warmup, int num_iters);
bool npu_rmsnorm_read_output(void* dst, size_t bytes);
void npu_rmsnorm_print_info();
void npu_rmsnorm_cleanup();
