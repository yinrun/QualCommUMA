#pragma once
#include "common.h"

// Initialize OpenCL: import ION buffers, compile kernel.
bool gpu_init(const IonBuffer& A, const IonBuffer& B, const IonBuffer& C,
              const char* kernel_path);

// Run bandwidth test.  If barrier != nullptr, waits on it after warmup.
BandwidthResult gpu_run(int num_warmup, int num_iters, SpinBarrier* barrier);

// Print device info.
void gpu_print_info();

// Release all OpenCL resources.
void gpu_cleanup();
