#pragma once
#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

// ── Timing ──────────────────────────────────────────────────────────────────
inline double now_seconds() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// ── ION buffer descriptor ───────────────────────────────────────────────────
struct IonBuffer {
  void*  ptr  = nullptr;
  int    fd   = -1;
  size_t size = 0;
};

// ── rpcmem helpers ──────────────────────────────────────────────────────────
struct RpcMemApi {
  void* libHandle = nullptr;
  void* (*alloc)(int, int, int) = nullptr;
  void  (*freeMem)(void*)       = nullptr;
  int   (*toFd)(void*)          = nullptr;
};

inline RpcMemApi& getRpcMemApi() {
  static RpcMemApi api;
  static bool init = false;
  if (!init) {
    init = true;
    const char* candidates[] = {
        "libcdsprpc.so",
        "/vendor/lib64/libcdsprpc.so",
        "/system/lib64/libcdsprpc.so",
        nullptr};
    for (const char** p = candidates; *p && !api.libHandle; ++p)
      api.libHandle = dlopen(*p, RTLD_LAZY | RTLD_LOCAL);
    if (api.libHandle) {
      api.alloc   = reinterpret_cast<void*(*)(int,int,int)>(dlsym(api.libHandle, "rpcmem_alloc"));
      api.freeMem = reinterpret_cast<void(*)(void*)>(dlsym(api.libHandle, "rpcmem_free"));
      api.toFd    = reinterpret_cast<int(*)(void*)>(dlsym(api.libHandle, "rpcmem_to_fd"));
    }
  }
  return api;
}

constexpr int RPCMEM_HEAP_ID_SYSTEM = 25;

inline bool allocIonBuffer(size_t size, uint8_t fillValue, IonBuffer& out) {
  auto& rpc = getRpcMemApi();
  if (!rpc.alloc || !rpc.toFd) return false;

  out.ptr = rpc.alloc(RPCMEM_HEAP_ID_SYSTEM, 0, static_cast<int>(size));
  if (!out.ptr) return false;
  out.size = size;
  std::memset(out.ptr, fillValue, size);

  out.fd = rpc.toFd(out.ptr);
  if (out.fd < 0) {
    rpc.freeMem(out.ptr);
    out.ptr = nullptr;
    return false;
  }
  return true;
}

inline void freeIonBuffer(IonBuffer& buf) {
  if (buf.ptr) {
    auto& rpc = getRpcMemApi();
    if (rpc.freeMem) rpc.freeMem(buf.ptr);
    buf.ptr = nullptr;
    buf.fd  = -1;
    buf.size = 0;
  }
}

// ── Theoretical peak ────────────────────────────────────────────────────────
constexpr double kTheoreticalBandwidthGBps = 84.8;  // LPDDR5X-5300 4ch x 16bit

// ── RMSNorm configuration ───────────────────────────────────────────────────
struct RMSNormConfig {
  int   batch_size;   // 1 for decode, seq_len for prefill
  int   hidden_dim;   // 2048, 3200, 4096, etc.
  float epsilon;      // typically 1e-6
};

// ── RMSNorm benchmark result ────────────────────────────────────────────────
struct RMSNormResult {
  double latency_us      = 0.0;   // average latency per call (microseconds)
  double bandwidth_gbps  = 0.0;   // effective memory bandwidth (GB/s)
  int    num_iterations  = 0;
  bool   success         = false;
  std::string error;
};
