#pragma once
#include "common.h"

// Initialize QNN HTP: dlopen, backend, device, power, graph, register buffers.
// The IonBuffer A/B/C are allocated externally (main) and registered here.
bool htp_init(const IonBuffer& A, const IonBuffer& B, const IonBuffer& C);

// Run bandwidth test.  If barrier != nullptr, waits on it after warmup.
BandwidthResult htp_run(int num_warmup, int num_iters, SpinBarrier* barrier);

// Print device info.
void htp_print_info();

// Release all QNN resources (deregister mem, free graph/context/backend, dlclose).
void htp_cleanup();
