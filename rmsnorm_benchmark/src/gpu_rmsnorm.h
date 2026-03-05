#pragma once
#include "common.h"

// Initialize OpenCL: device, compile kernel, create buffers.
bool gpu_rmsnorm_init(const RMSNormConfig& config, const char* kernel_path);

// Run benchmark: warmup + timed iterations. Returns average latency and bandwidth.
RMSNormResult gpu_rmsnorm_run(const RMSNormConfig& config, int num_warmup, int num_iters);

// Read output buffer to host (for correctness verification).
bool gpu_rmsnorm_read_output(void* dst, size_t bytes);

// Print GPU device info.
void gpu_rmsnorm_print_info();

// Release all resources.
void gpu_rmsnorm_cleanup();
