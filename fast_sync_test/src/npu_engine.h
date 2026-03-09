#pragma once
#include "common.h"

// NPU RMSNorm engine using QNN HTP.
// Accepts external ION buffers for zero-copy sharing with GPU.
// Init must be called from main thread (QNN is not thread-safe for init).
// execute_blocking() can be called from any thread.

// Standard init: Input[ION] → RmsNorm → Output[ION]
bool npu_init(int hidden_dim, float epsilon,
              const IonBuffer& ion_input, const IonBuffer& ion_output);

// Sync init: Input[ION] + GPUFlag[ION] → SyncWait → Data[native] → RmsNorm → Output[ION]
// DSP polls gpu_flag before executing RmsNorm, enabling GPU+NPU parallel launch.
// Requires SyncWait custom op .so files at runtime (set via ADSP_LIBRARY_PATH).
bool npu_init_with_sync(int hidden_dim, float epsilon,
                        const IonBuffer& ion_input, const IonBuffer& ion_output,
                        const IonBuffer& ion_gpu_flag);

// Blocking: calls graphExecute. Returns wall-clock time in us.
double npu_execute_blocking();

void npu_print_info();
void npu_cleanup();
