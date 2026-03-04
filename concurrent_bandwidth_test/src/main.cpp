#include "common.h"
#include "gpu_bandwidth.h"
#include "htp_bandwidth.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <algorithm>

// ── Helpers ─────────────────────────────────────────────────────────────────

static void print_result(const char* label, const BandwidthResult& r) {
  if (!r.success) {
    printf("  %s: FAILED (%s)\n", label, r.error.c_str());
    return;
  }
  printf("  %s: %.2f GB/s (%d iters, %.3fs)\n",
         label, r.bandwidth_gbps, r.num_iterations, r.elapsed_seconds);
}

enum class Mode { GPU, NPU, CONCURRENT, ALL };

static void print_usage(const char* prog) {
  printf("Usage: %s [options]\n", prog);
  printf("  --mode gpu|npu|concurrent|all   (default: all)\n");
  printf("  --total-size-mb N               total data MB (default: 256)\n");
  printf("  --gpu-ratio R                   GPU partition 0.0-1.0 (default: 0.5)\n");
  printf("  --gpu-iters N                   GPU iterations (default: auto)\n");
  printf("  --npu-iters N                   NPU iterations (default: auto)\n");
}

// Auto-compute iterations targeting ~100ms runtime
static int auto_iters(size_t partition_bytes, double estimated_bw_gbps) {
  double bytes_per_iter = partition_bytes * 3.0;  // 2 read + 1 write
  double secs_per_iter  = (bytes_per_iter / (1024.0*1024.0*1024.0)) / estimated_bw_gbps;
  int iters = std::max(1, (int)(0.1 / secs_per_iter));  // target 100ms
  return iters;
}

// ── Test runners ────────────────────────────────────────────────────────────

static BandwidthResult run_gpu_only(size_t size_bytes, int iters) {
  IonBuffer A, B, C;
  if (!allocIonBuffer(size_bytes, 1, A) ||
      !allocIonBuffer(size_bytes, 2, B) ||
      !allocIonBuffer(size_bytes, 0, C)) {
    freeIonBuffer(A); freeIonBuffer(B); freeIonBuffer(C);
    return {.error = "ION alloc failed"};
  }

  if (!gpu_init(A, B, C, "kernels/element_add.cl")) {
    gpu_cleanup();
    freeIonBuffer(A); freeIonBuffer(B); freeIonBuffer(C);
    return {.error = "GPU init failed"};
  }

  auto res = gpu_run(3, iters, nullptr);
  gpu_cleanup();
  freeIonBuffer(A); freeIonBuffer(B); freeIonBuffer(C);
  return res;
}

static BandwidthResult run_npu_only(size_t size_bytes, int iters) {
  IonBuffer A, B, C;
  if (!allocIonBuffer(size_bytes, 1, A) ||
      !allocIonBuffer(size_bytes, 2, B) ||
      !allocIonBuffer(size_bytes, 0, C)) {
    freeIonBuffer(A); freeIonBuffer(B); freeIonBuffer(C);
    return {.error = "ION alloc failed"};
  }

  if (!htp_init(A, B, C)) {
    htp_cleanup();
    freeIonBuffer(A); freeIonBuffer(B); freeIonBuffer(C);
    return {.error = "HTP init failed"};
  }

  auto res = htp_run(3, iters, nullptr);
  htp_cleanup();
  freeIonBuffer(A); freeIonBuffer(B); freeIonBuffer(C);
  return res;
}

struct ConcurrentResult {
  BandwidthResult gpu;
  BandwidthResult npu;
  double wall_seconds;
};

static ConcurrentResult run_concurrent(
    size_t gpu_bytes, size_t npu_bytes,
    int gpu_iters, int npu_iters) {

  ConcurrentResult cr = {};

  // Allocate ION buffers for GPU partition
  IonBuffer gA, gB, gC;
  if (!allocIonBuffer(gpu_bytes, 1, gA) ||
      !allocIonBuffer(gpu_bytes, 2, gB) ||
      !allocIonBuffer(gpu_bytes, 0, gC)) {
    freeIonBuffer(gA); freeIonBuffer(gB); freeIonBuffer(gC);
    cr.gpu.error = "GPU ION alloc failed";
    return cr;
  }

  // Allocate ION buffers for NPU partition
  IonBuffer nA, nB, nC;
  if (!allocIonBuffer(npu_bytes, 1, nA) ||
      !allocIonBuffer(npu_bytes, 2, nB) ||
      !allocIonBuffer(npu_bytes, 0, nC)) {
    freeIonBuffer(gA); freeIonBuffer(gB); freeIonBuffer(gC);
    freeIonBuffer(nA); freeIonBuffer(nB); freeIonBuffer(nC);
    cr.npu.error = "NPU ION alloc failed";
    return cr;
  }

  // Initialize both (sequentially — dlopen not thread-safe)
  if (!gpu_init(gA, gB, gC, "kernels/element_add.cl")) {
    cr.gpu.error = "GPU init failed";
    gpu_cleanup();
    freeIonBuffer(gA); freeIonBuffer(gB); freeIonBuffer(gC);
    freeIonBuffer(nA); freeIonBuffer(nB); freeIonBuffer(nC);
    return cr;
  }

  if (!htp_init(nA, nB, nC)) {
    cr.npu.error = "HTP init failed";
    gpu_cleanup(); htp_cleanup();
    freeIonBuffer(gA); freeIonBuffer(gB); freeIonBuffer(gC);
    freeIonBuffer(nA); freeIonBuffer(nB); freeIonBuffer(nC);
    return cr;
  }

  // Launch concurrent threads with barrier
  SpinBarrier barrier(2);

  double wall_t0 = now_seconds();

  std::thread gpu_thread([&]() {
    cr.gpu = gpu_run(3, gpu_iters, &barrier);
  });
  std::thread npu_thread([&]() {
    cr.npu = htp_run(3, npu_iters, &barrier);
  });

  gpu_thread.join();
  npu_thread.join();

  cr.wall_seconds = now_seconds() - wall_t0;

  // Cleanup
  gpu_cleanup();
  htp_cleanup();
  freeIonBuffer(gA); freeIonBuffer(gB); freeIonBuffer(gC);
  freeIonBuffer(nA); freeIonBuffer(nB); freeIonBuffer(nC);
  return cr;
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  Mode mode           = Mode::ALL;
  size_t total_mb     = 256;
  double gpu_ratio    = 0.5;
  int gpu_iters_arg   = 0;  // 0 = auto
  int npu_iters_arg   = 0;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--mode") && i+1 < argc) {
      ++i;
      if (!strcmp(argv[i], "gpu"))        mode = Mode::GPU;
      else if (!strcmp(argv[i], "npu"))   mode = Mode::NPU;
      else if (!strcmp(argv[i], "concurrent")) mode = Mode::CONCURRENT;
      else                                mode = Mode::ALL;
    } else if (!strcmp(argv[i], "--total-size-mb") && i+1 < argc) {
      total_mb = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--gpu-ratio") && i+1 < argc) {
      gpu_ratio = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--gpu-iters") && i+1 < argc) {
      gpu_iters_arg = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--npu-iters") && i+1 < argc) {
      npu_iters_arg = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--help")) {
      print_usage(argv[0]); return 0;
    }
  }

  size_t total_bytes = total_mb * 1024ULL * 1024ULL;
  size_t gpu_bytes   = static_cast<size_t>(total_bytes * gpu_ratio);
  size_t npu_bytes   = total_bytes - gpu_bytes;

  // Align to 1MB boundary (for QNN tensor shape compatibility)
  gpu_bytes = (gpu_bytes / (1024*1024)) * (1024*1024);
  npu_bytes = (npu_bytes / (1024*1024)) * (1024*1024);
  if (gpu_bytes == 0) gpu_bytes = 1024*1024;
  if (npu_bytes == 0) npu_bytes = 1024*1024;

  int gpu_iters = gpu_iters_arg > 0 ? gpu_iters_arg : auto_iters(gpu_bytes, 50.0);
  int npu_iters = npu_iters_arg > 0 ? npu_iters_arg : auto_iters(npu_bytes, 45.0);
  int gpu_full_iters = gpu_iters_arg > 0 ? gpu_iters_arg : auto_iters(total_bytes, 50.0);
  int npu_full_iters = npu_iters_arg > 0 ? npu_iters_arg : auto_iters(total_bytes, 45.0);

  printf("=== 异构并发带宽测试 ===\n");
  printf("理论峰值: %.1f GB/s (LPDDR5X-5300, 4ch x 16bit)\n", kTheoreticalBandwidthGBps);
  printf("任务: Element-wise Add (C = A + B)\n");
  printf("总大小: %zu MB, GPU: %zu MB (%.0f%%), NPU: %zu MB (%.0f%%)\n\n",
         total_mb, gpu_bytes/(1024*1024), gpu_ratio*100,
         npu_bytes/(1024*1024), (1.0-gpu_ratio)*100);

  BandwidthResult gpu_solo = {}, npu_solo = {};
  ConcurrentResult concurrent = {};

  // ── GPU-only baseline ───────────────────────────────────────────────────
  if (mode == Mode::GPU || mode == Mode::ALL) {
    printf("--- GPU 设备信息 ---\n");
    {
      // Quick init just to print info, then cleanup
      IonBuffer tmp;
      allocIonBuffer(1024*1024, 0, tmp);
      if (gpu_init(tmp, tmp, tmp, "kernels/element_add.cl")) {
        gpu_print_info();
        gpu_cleanup();
      }
      freeIonBuffer(tmp);
    }
    printf("\n=== GPU-Only 基线 (全量 %zu MB) ===\n", total_mb);
    gpu_solo = run_gpu_only(total_bytes, gpu_full_iters);
    print_result("GPU", gpu_solo);
    if (gpu_solo.success)
      printf("  利用率: %.1f%%\n", gpu_solo.bandwidth_gbps / kTheoreticalBandwidthGBps * 100);
    printf("\n");
  }

  // ── NPU-only baseline ──────────────────────────────────────────────────
  if (mode == Mode::NPU || mode == Mode::ALL) {
    printf("--- NPU 设备信息 ---\n");
    {
      // Use small separate buffers (like GPU info path) to avoid memRegister failure
      IonBuffer tA, tB, tC;
      allocIonBuffer(1024*1024, 1, tA);
      allocIonBuffer(1024*1024, 2, tB);
      allocIonBuffer(1024*1024, 0, tC);
      if (htp_init(tA, tB, tC)) {
        htp_print_info();
        htp_cleanup();
      }
      freeIonBuffer(tA); freeIonBuffer(tB); freeIonBuffer(tC);
    }
    printf("\n=== NPU-Only 基线 (全量 %zu MB) ===\n", total_mb);
    npu_solo = run_npu_only(total_bytes, npu_full_iters);
    print_result("NPU", npu_solo);
    if (npu_solo.success)
      printf("  利用率: %.1f%%\n", npu_solo.bandwidth_gbps / kTheoreticalBandwidthGBps * 100);
    printf("\n");
  }

  // ── Concurrent ────────────────────────────────────────────────────────
  if (mode == Mode::CONCURRENT || mode == Mode::ALL) {
    printf("=== GPU + NPU 并发 (GPU %zuMB + NPU %zuMB) ===\n",
           gpu_bytes/(1024*1024), npu_bytes/(1024*1024));
    concurrent = run_concurrent(gpu_bytes, npu_bytes, gpu_iters, npu_iters);
    print_result("GPU", concurrent.gpu);
    print_result("NPU", concurrent.npu);

    if (concurrent.gpu.success && concurrent.npu.success) {
      double aggregate = concurrent.gpu.bandwidth_gbps + concurrent.npu.bandwidth_gbps;
      double overlap = 0;
      double tg = concurrent.gpu.elapsed_seconds;
      double tn = concurrent.npu.elapsed_seconds;
      double tw = concurrent.wall_seconds;
      if (std::min(tg, tn) > 0)
        overlap = (tg + tn - tw) / std::min(tg, tn) * 100.0;

      printf("  Wall clock: %.3fs, Overlap: %.1f%%\n", tw, overlap);
      printf("  聚合带宽: %.2f GB/s (%.1f%%)\n",
             aggregate, aggregate / kTheoreticalBandwidthGBps * 100);
    }
    printf("\n");
  }

  // ── Summary table ─────────────────────────────────────────────────────
  if (mode == Mode::ALL) {
    printf("=== 总结 ===\n");
    printf("+-----------+-----------+-----------+---------+---------+\n");
    printf("| 模式      | GPU GB/s  | NPU GB/s  | 聚合    | 利用率  |\n");
    printf("+-----------+-----------+-----------+---------+---------+\n");

    if (gpu_solo.success)
      printf("| GPU only  | %9.2f |     --    | %7.2f | %5.1f%%  |\n",
             gpu_solo.bandwidth_gbps, gpu_solo.bandwidth_gbps,
             gpu_solo.bandwidth_gbps / kTheoreticalBandwidthGBps * 100);

    if (npu_solo.success)
      printf("| NPU only  |     --    | %9.2f | %7.2f | %5.1f%%  |\n",
             npu_solo.bandwidth_gbps, npu_solo.bandwidth_gbps,
             npu_solo.bandwidth_gbps / kTheoreticalBandwidthGBps * 100);

    if (concurrent.gpu.success && concurrent.npu.success) {
      double agg = concurrent.gpu.bandwidth_gbps + concurrent.npu.bandwidth_gbps;
      printf("| 并发      | %9.2f | %9.2f | %7.2f | %5.1f%%  |\n",
             concurrent.gpu.bandwidth_gbps, concurrent.npu.bandwidth_gbps,
             agg, agg / kTheoreticalBandwidthGBps * 100);
    }
    printf("+-----------+-----------+-----------+---------+---------+\n");

    if (gpu_solo.success && npu_solo.success &&
        concurrent.gpu.success && concurrent.npu.success) {
      double solo_sum = gpu_solo.bandwidth_gbps + npu_solo.bandwidth_gbps;
      double agg = concurrent.gpu.bandwidth_gbps + concurrent.npu.bandwidth_gbps;
      printf("竞争因子: %.2f (%.2f / %.2f)\n", agg / solo_sum, agg, solo_sum);
    }
  }

  return 0;
}
