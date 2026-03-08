#include "common.h"
#include "gpu_rmsnorm.h"
#include "npu_rmsnorm.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

static void print_usage(const char* prog) {
  printf("Usage: %s [options]\n", prog);
  printf("  --hidden-dim N      hidden dimension (default: 4096)\n");
  printf("  --batch N           single batch size to test\n");
  printf("  --iters N           iterations per test (default: auto)\n");
  printf("  --warmup N          warmup iterations (default: 10)\n");
}

static int auto_iters(int batch, int hidden, double estimated_bw_gbps) {
  double bytes_per_call = (double)batch * hidden * 2 * 3;
  double secs_per_call  = (bytes_per_call / (1024.0*1024.0*1024.0)) / estimated_bw_gbps;
  if (secs_per_call < 1e-9) secs_per_call = 1e-6;
  int iters = std::max(50, (int)(0.2 / secs_per_call));
  return std::min(iters, 10000);
}

struct TestCase { int batch; int hidden; const char* label; };

int main(int argc, char* argv[]) {
  int hidden_dim   = 4096;
  int single_batch = 0;
  int user_iters   = 0;
  int warmup       = 10;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--hidden-dim") && i+1 < argc) hidden_dim = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--batch") && i+1 < argc) single_batch = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--iters") && i+1 < argc) user_iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup") && i+1 < argc) warmup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--help")) { print_usage(argv[0]); return 0; }
  }

  printf("=== RMSNorm Benchmark: GPU vs NPU vs NPU-Cached ===\n");
  printf("平台: SM8850, Adreno 840 + Hexagon V81\n");
  printf("理论峰值带宽: %.1f GB/s (LPDDR5X-5300)\n\n", kTheoreticalBandwidthGBps);

  // Print device info
  {
    printf("--- 设备信息 ---\n");
    RMSNormConfig tmp = {1, 4096, 1e-6f};
    if (gpu_rmsnorm_init(tmp, "kernels/rmsnorm.cl")) {
      gpu_rmsnorm_print_info();
      gpu_rmsnorm_cleanup();
    }
    if (npu_rmsnorm_init(tmp, NpuMode::NATIVE)) {
      npu_rmsnorm_print_info();
      npu_rmsnorm_cleanup();
    }
    printf("\n");
  }

  // Build test matrix
  std::vector<TestCase> cases;
  if (single_batch > 0) {
    cases.push_back({single_batch, hidden_dim, "custom"});
  } else {
    cases.push_back({1,    2048, "decode"});
    cases.push_back({1,    3200, "decode"});
    cases.push_back({1,    4096, "decode"});
    cases.push_back({16,   4096, "prefill-16"});
    cases.push_back({64,   4096, "prefill-64"});
    cases.push_back({256,  4096, "prefill-256"});
    cases.push_back({512,  4096, "prefill-512"});
    cases.push_back({1024, 4096, "prefill-1k"});
  }

  // Header
  printf("--- GPU RMSNorm (FP16) vs NPU Dynamic vs NPU Cached ---\n\n");
  printf("%-12s %5s %6s | %9s %9s | %9s %9s | %9s %9s | %7s %7s\n",
         "场景", "batch", "hidden",
         "GPU(us)", "GPU(GB/s)",
         "NPU(us)", "NPU(GB/s)",
         "Cache(us)", "Cac(GB/s)",
         "GPU/NPU", "GPU/Cac");
  for (int i = 0; i < 115; ++i) printf("-");
  printf("\n");

  for (auto& tc : cases) {
    RMSNormConfig cfg = {tc.batch, tc.hidden, 1e-6f};
    int iters = user_iters > 0 ? user_iters : auto_iters(tc.batch, tc.hidden, 20.0);

    // GPU
    RMSNormResult gpu = {};
    if (gpu_rmsnorm_init(cfg, "kernels/rmsnorm.cl"))
      gpu = gpu_rmsnorm_run(cfg, warmup, iters);
    gpu_rmsnorm_cleanup();

    // NPU dynamic
    RMSNormResult npu = {};
    if (npu_rmsnorm_init(cfg, NpuMode::NATIVE))
      npu = npu_rmsnorm_run(cfg, warmup, iters);
    npu_rmsnorm_cleanup();

    // NPU cached
    RMSNormResult npu_cached = {};
    double cache_load_us = 0;
    if (npu_rmsnorm_init_cached(cfg, NpuMode::NATIVE, cache_load_us)) {
      printf("[NPU] Cache load: %.1f us\n", cache_load_us);
      npu_cached = npu_rmsnorm_run(cfg, warmup, iters);
    }
    npu_rmsnorm_cleanup();

    // Print row
    printf("%-12s %5d %6d", tc.label, tc.batch, tc.hidden);

    if (gpu.success) printf(" | %9.1f %9.2f", gpu.latency_us, gpu.bandwidth_gbps);
    else             printf(" | %9s %9s", "FAIL", "-");

    if (npu.success) printf(" | %9.1f %9.2f", npu.latency_us, npu.bandwidth_gbps);
    else             printf(" | %9s %9s", "FAIL", "-");

    if (npu_cached.success) printf(" | %9.1f %9.2f", npu_cached.latency_us, npu_cached.bandwidth_gbps);
    else                    printf(" | %9s %9s", "FAIL", "-");

    if (gpu.success && npu.success)
      printf(" | %5.1fx", npu.latency_us / gpu.latency_us);
    else
      printf(" | %7s", "-");

    if (gpu.success && npu_cached.success)
      printf(" %5.1fx", npu_cached.latency_us / gpu.latency_us);
    else
      printf(" %7s", "-");

    printf("\n");
  }

  printf("\n--- 结论 ---\n");
  printf("GPU/NPU: NPU dynamic 延迟 / GPU 延迟 (含 FastRPC launch 开销)\n");
  printf("GPU/Cac: NPU cached 延迟 / GPU 延迟 (context binary 反序列化, 消除 launch 开销)\n");

  return 0;
}
