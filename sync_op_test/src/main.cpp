//=============================================================================
// SyncWait Op 独立测试：CPU 写 flag，DSP 读 flag
//
// 不依赖 GPU，用 CPU 写 ION shared memory flag 来测试 SyncWait op
// 能否在 DSP 端看到 flag 的变化。
//
// 核心机制：
//   将 flag ION buffer 的文件描述符（fd）作为静态参数传给 SyncWait op。
//   DSP 端用 HAP_mmap_get(fd, &vaddr) 获取 DSP VA，
//   用 dcinva(vaddr) 直接轮询 DDR，绕过 QNN 的 TCM DMA 拷贝。
//
// 测试场景：
//   A: flag=1 在 graphExecute 之前写入（验证基础功能，不需要 HAP_mmap_get）
//   B: flag=0 在 graphExecute 之前，CPU 线程在 graphExecute 期间写 1
//      → HAP_mmap_get 成功：DSP 直接看到 CPU 写入 → ~50us 完成
//      → HAP_mmap_get 失败：DSP 看 TCM 副本 → ~410ms 超时
//   C: flag=0 永不写入（验证超时行为）
//=============================================================================

#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <thread>
#include <vector>

#include "QNN/QnnBackend.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnDevice.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnInterface.h"
#include "QNN/QnnLog.h"
#include "QNN/QnnMem.h"
#include "QNN/QnnTensor.h"
#include "QNN/HTP/QnnHtpDevice.h"
#include "QNN/HTP/QnnHtpGraph.h"
#include "QNN/HTP/QnnHtpPerfInfrastructure.h"

// ── Timing ──────────────────────────────────────────────────────────────────
static double now_us() {
  return std::chrono::duration<double, std::micro>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ── ION buffer (rpcmem) ─────────────────────────────────────────────────────
struct IonBuffer { void* ptr = nullptr; int fd = -1; size_t size = 0; };

struct RpcMemApi {
  void* lib = nullptr;
  void* (*alloc)(int, int, int) = nullptr;
  void  (*freeMem)(void*) = nullptr;
  int   (*toFd)(void*) = nullptr;
};

static RpcMemApi& rpcmem() {
  static RpcMemApi api;
  static bool init = false;
  if (!init) {
    init = true;
    const char* libs[] = {"libcdsprpc.so", "/vendor/lib64/libcdsprpc.so", nullptr};
    for (auto* p = libs; *p && !api.lib; ++p)
      api.lib = dlopen(*p, RTLD_LAZY | RTLD_LOCAL);
    if (api.lib) {
      api.alloc   = (void*(*)(int,int,int))dlsym(api.lib, "rpcmem_alloc");
      api.freeMem = (void(*)(void*))dlsym(api.lib, "rpcmem_free");
      api.toFd    = (int(*)(void*))dlsym(api.lib, "rpcmem_to_fd");
    }
  }
  return api;
}

static bool allocIon(size_t size, IonBuffer& out) {
  auto& r = rpcmem();
  if (!r.alloc || !r.toFd) return false;
  out.ptr = r.alloc(25, 0, (int)size);
  if (!out.ptr) return false;
  out.size = size;
  memset(out.ptr, 0, size);
  out.fd = r.toFd(out.ptr);
  if (out.fd < 0) { r.freeMem(out.ptr); out.ptr = nullptr; return false; }
  return true;
}

static void freeIon(IonBuffer& b) {
  if (b.ptr) { rpcmem().freeMem(b.ptr); b.ptr = nullptr; b.fd = -1; b.size = 0; }
}

// No CPU-side DSP VA translation needed.
// The ION fd is passed directly to the SyncWait op as a static parameter.
// On DSP: HAP_mmap_get(fd, &vaddr, &paddr) returns the DSP VA for direct polling.

// ── FP16 conversion ─────────────────────────────────────────────────────────
static uint16_t f32_to_f16(float f) {
  uint32_t x; memcpy(&x, &f, 4);
  uint16_t sign = (x >> 16) & 0x8000;
  int exp = ((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFF;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7C00;
  return sign | (exp << 10) | (mant >> 13);
}

// ── QNN helpers ─────────────────────────────────────────────────────────────
static const QNN_INTERFACE_VER_TYPE* g_qnn = nullptr;
static void* g_lib = nullptr;

static void logCb(const char* fmt, QnnLog_Level_t level, uint64_t, va_list args) {
  if (level != QNN_LOG_LEVEL_ERROR) return;
  printf("[QNN-E] "); vprintf(fmt, args); printf("\n");
}

static bool ok(Qnn_ErrorHandle_t s, const char* w) {
  if (s != QNN_SUCCESS) { printf("[ERR] %s: %lu\n", w, (unsigned long)s); return false; }
  return true;
}

static Qnn_Tensor_t makeFp16(const char* name, Qnn_TensorType_t type,
                              uint32_t* dims, uint32_t rank = 4) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = QNN_TENSOR_VERSION_1;
  t.v1.name = name; t.v1.type = type;
  t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  t.v1.dataType = QNN_DATATYPE_FLOAT_16;
  t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  t.v1.rank = rank; t.v1.dimensions = dims;
  t.v1.memType = QNN_TENSORMEMTYPE_RAW;
  t.v1.clientBuf = QNN_CLIENT_BUFFER_INIT;
  return t;
}

static Qnn_Tensor_t makeUint32(const char* name, Qnn_TensorType_t type,
                                uint32_t* dims, uint32_t rank = 4) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = QNN_TENSOR_VERSION_1;
  t.v1.name = name; t.v1.type = type;
  t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  t.v1.dataType = QNN_DATATYPE_UINT_32;
  t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  t.v1.rank = rank; t.v1.dimensions = dims;
  t.v1.memType = QNN_TENSORMEMTYPE_RAW;
  t.v1.clientBuf = QNN_CLIENT_BUFFER_INIT;
  return t;
}

static void setHighPerf() {
  QnnDevice_Infrastructure_t infra = nullptr;
  if (QNN_SUCCESS != g_qnn->deviceGetInfrastructure(&infra) || !infra) return;
  auto* hi = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
  if (hi->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
  auto& perf = hi->perfInfra;
  uint32_t pid = 0;
  if (QNN_SUCCESS != perf.createPowerConfigId(0, 0, &pid)) return;

  QnnHtpPerfInfrastructure_PowerConfig_t dcvs = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  dcvs.dcvsV3Config.contextId = pid;
  dcvs.dcvsV3Config.setDcvsEnable = 1; dcvs.dcvsV3Config.dcvsEnable = 0;
  dcvs.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  dcvs.dcvsV3Config.setSleepDisable = 1; dcvs.dcvsV3Config.sleepDisable = 1;
  dcvs.dcvsV3Config.setBusParams = 1;
  dcvs.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.setCoreParams = 1;
  dcvs.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&dcvs, nullptr};
  perf.setPowerConfig(pid, cfgs);
}

// ── Build graph: SyncWait → RmsNorm ─────────────────────────────────────────
struct TestGraph {
  Qnn_GraphHandle_t graph = nullptr;
  Qnn_Tensor_t execIn[2];   // [data, flag]
  Qnn_Tensor_t execOut[1];  // [output]
};

static bool buildGraph(Qnn_ContextHandle_t ctx, const char* name,
                        uint32_t* dimsIO, uint32_t* dimsFlag, uint32_t* dimsGamma,
                        size_t gamma_bytes, void* gamma_ptr,
                        Qnn_MemHandle_t hIn, Qnn_MemHandle_t hFlag, Qnn_MemHandle_t hOut,
                        uint32_t flag_ion_fd,
                        TestGraph& tg) {
  // Graph config: FP16 precision, HVX threads
  QnnHtpGraph_CustomConfig_t htpCfgs[2] = {
    QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT,
    QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT};
  htpCfgs[0].option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
  htpCfgs[0].precision = QNN_PRECISION_FLOAT16;
  htpCfgs[1].option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
  htpCfgs[1].numHvxThreads = 8;

  QnnGraph_Config_t gc = QNN_GRAPH_CONFIG_INIT;
  gc.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  gc.customConfig = htpCfgs;
  const QnnGraph_Config_t* gcList[] = {&gc, nullptr};

  if (!ok(g_qnn->graphCreate(ctx, name, gcList, &tg.graph), "graphCreate"))
    return false;

  // Tensors
  Qnn_Tensor_t sw_in   = makeFp16("sw_in",   QNN_TENSOR_TYPE_APP_WRITE, dimsIO);
  Qnn_Tensor_t sw_flag = makeUint32("sw_flag", QNN_TENSOR_TYPE_APP_WRITE, dimsFlag);
  Qnn_Tensor_t sw_out  = makeFp16("sw_out",   QNN_TENSOR_TYPE_NATIVE,    dimsIO);
  Qnn_Tensor_t output  = makeFp16("output",   QNN_TENSOR_TYPE_APP_READ,  dimsIO);
  Qnn_Tensor_t gamma   = makeFp16("gamma",    QNN_TENSOR_TYPE_STATIC, dimsGamma, 1);
  gamma.v1.clientBuf.data = gamma_ptr;
  gamma.v1.clientBuf.dataSize = (uint32_t)gamma_bytes;

  g_qnn->tensorCreateGraphTensor(tg.graph, &sw_in);
  g_qnn->tensorCreateGraphTensor(tg.graph, &sw_flag);
  g_qnn->tensorCreateGraphTensor(tg.graph, &sw_out);
  g_qnn->tensorCreateGraphTensor(tg.graph, &gamma);
  g_qnn->tensorCreateGraphTensor(tg.graph, &output);

  // SyncWait node with flag_ion_fd static parameter
  // DSP uses HAP_mmap_get(fd) to get the DSP VA for direct DDR polling.
  {
    Qnn_Param_t p = QNN_PARAM_INIT;
    p.paramType = QNN_PARAMTYPE_SCALAR;
    p.name = "flag_ion_fd";
    p.scalarParam.dataType = QNN_DATATYPE_UINT_32;
    p.scalarParam.uint32Value = flag_ion_fd;
    Qnn_Param_t params[] = {p};

    Qnn_Tensor_t in[] = {sw_in, sw_flag};
    Qnn_Tensor_t out[] = {sw_out};
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT; op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = "syncwait"; op.v1.packageName = "heteroedge.HvxOpPackage";
    op.v1.typeName = "SyncWait";
    op.v1.numOfParams = 1; op.v1.params = params;
    op.v1.numOfInputs = 2; op.v1.inputTensors = in;
    op.v1.numOfOutputs = 1; op.v1.outputTensors = out;
    if (!ok(g_qnn->graphAddNode(tg.graph, op), "addNode SyncWait")) return false;
  }

  // RmsNorm node
  {
    Qnn_Param_t eps = QNN_PARAM_INIT;
    eps.paramType = QNN_PARAMTYPE_SCALAR; eps.name = "epsilon";
    eps.scalarParam.dataType = QNN_DATATYPE_FLOAT_32; eps.scalarParam.floatValue = 1e-6f;
    Qnn_Param_t params[] = {eps};
    Qnn_Tensor_t in[] = {sw_out, gamma};
    Qnn_Tensor_t out[] = {output};
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT; op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = "rmsnorm"; op.v1.packageName = "heteroedge.HvxOpPackage";
    op.v1.typeName = "RmsNorm";
    op.v1.numOfParams = 1; op.v1.params = params;
    op.v1.numOfInputs = 2; op.v1.inputTensors = in;
    op.v1.numOfOutputs = 1; op.v1.outputTensors = out;
    if (!ok(g_qnn->graphAddNode(tg.graph, op), "addNode RmsNorm")) return false;
  }

  if (!ok(g_qnn->graphFinalize(tg.graph, nullptr, nullptr), "graphFinalize"))
    return false;

  // Bind ION memHandles for execution
  tg.execIn[0] = sw_in;
  tg.execIn[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
  tg.execIn[0].v1.memHandle = hIn;
  tg.execIn[1] = sw_flag;
  tg.execIn[1].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
  tg.execIn[1].v1.memHandle = hFlag;
  tg.execOut[0] = output;
  tg.execOut[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
  tg.execOut[0].v1.memHandle = hOut;

  return true;
}

// ── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
  int hidden = 4096;
  int steps = 20;
  int warmup = 5;
  int delay_us = 50;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--hidden") && i+1 < argc) hidden = atoi(argv[++i]);
    if (!strcmp(argv[i], "--steps") && i+1 < argc) steps = atoi(argv[++i]);
    if (!strcmp(argv[i], "--delay") && i+1 < argc) delay_us = atoi(argv[++i]);
  }

  printf("=== SyncWait Op 独立测试（CPU 写 flag，DSP 直接 DDR 轮询）===\n");
  printf("hidden=%d, steps=%d, warmup=%d, delay=%dus\n\n", hidden, steps, warmup, delay_us);

  size_t data_bytes = (size_t)hidden * 2;  // FP16
  size_t gamma_bytes = data_bytes;

  // Allocate ION buffers
  IonBuffer ionIn, ionOut, ionFlag, ionGamma;
  if (!allocIon(data_bytes, ionIn) || !allocIon(data_bytes, ionOut) ||
      !allocIon(sizeof(uint32_t), ionFlag) || !allocIon(gamma_bytes, ionGamma)) {
    printf("ION alloc failed\n"); return 1;
  }

  // Fill data + gamma
  uint16_t one = f32_to_f16(1.0f);
  uint16_t* gp = (uint16_t*)ionGamma.ptr;
  uint16_t* ip = (uint16_t*)ionIn.ptr;
  for (int i = 0; i < hidden; ++i) { gp[i] = one; ip[i] = f32_to_f16(0.5f); }

  // Pass the ION fd to SyncWait. On DSP, HAP_mmap_get(fd) returns the DSP VA.
  // No CPU-side DSP VA translation needed.
  uint32_t flag_ion_fd = (uint32_t)ionFlag.fd;
  printf("flag ION fd = %d — DSP 将用 HAP_mmap_get(%d) 获取 DSP VA\n\n",
         (int)flag_ion_fd, (int)flag_ion_fd);

  // Init QNN
  g_lib = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
  if (!g_lib) { printf("dlopen failed: %s\n", dlerror()); return 1; }

  auto getProviders = (decltype(&QnnInterface_getProviders))
      dlsym(g_lib, "QnnInterface_getProviders");
  const QnnInterface_t** providers = nullptr; uint32_t np = 0;
  getProviders(&providers, &np);
  g_qnn = &providers[0]->QNN_INTERFACE_VER_NAME;

  Qnn_LogHandle_t log = nullptr;
  if (g_qnn->logCreate) g_qnn->logCreate(logCb, QNN_LOG_LEVEL_ERROR, &log);

  Qnn_BackendHandle_t backend = nullptr;
  ok(g_qnn->backendCreate(log, nullptr, &backend), "backendCreate");

  Qnn_DeviceHandle_t device = nullptr;
  if (g_qnn->deviceCreate) g_qnn->deviceCreate(nullptr, nullptr, &device);
  setHighPerf();

  Qnn_ContextHandle_t ctx = nullptr;
  ok(g_qnn->contextCreate(backend, device, nullptr, &ctx), "contextCreate");

  // Register heteroedge op package
  if (!ok(g_qnn->backendRegisterOpPackage(
        backend, "./libQnnHtpHeteroEdgeOpPackage.so",
        "heteroedgeInterfaceProvider", "CPU"), "regPkg CPU")) {
    printf("Failed to register CPU op package\n"); return 1;
  }
  if (!ok(g_qnn->backendRegisterOpPackage(
        backend, "./htp/libQnnHtpHeteroEdgeOpPackage.so",
        "heteroedgeInterfaceProvider", "HTP"), "regPkg HTP")) {
    printf("Failed to register HTP op package\n"); return 1;
  }
  printf("HeteroEdge op package registered OK\n\n");

  // Register ION buffers with QNN
  uint32_t dimsIO[4] = {1, 1, 1, (uint32_t)hidden};
  uint32_t dimsFlag[4] = {1, 1, 1, 1};
  uint32_t dimsGamma[1] = {(uint32_t)hidden};

  auto regBuf = [&](const IonBuffer& ion, const uint32_t* dims, uint32_t nd,
                     Qnn_DataType_t dt) -> Qnn_MemHandle_t {
    Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
    desc.memShape.numDim = nd; desc.memShape.dimSize = const_cast<uint32_t*>(dims);
    desc.dataType = dt; desc.memType = QNN_MEM_TYPE_ION; desc.ionInfo.fd = ion.fd;
    Qnn_MemHandle_t h = nullptr;
    g_qnn->memRegister(ctx, &desc, 1, &h);
    return h;
  };

  Qnn_MemHandle_t hIn   = regBuf(ionIn,   dimsIO,   4, QNN_DATATYPE_FLOAT_16);
  Qnn_MemHandle_t hOut  = regBuf(ionOut,  dimsIO,   4, QNN_DATATYPE_FLOAT_16);
  Qnn_MemHandle_t hFlag = regBuf(ionFlag, dimsFlag, 4, QNN_DATATYPE_UINT_32);

  // Build graph — pass flag_ion_fd as static param to SyncWait
  TestGraph tg;
  if (!buildGraph(ctx, "syncwait_test", dimsIO, dimsFlag, dimsGamma,
                  gamma_bytes, ionGamma.ptr, hIn, hFlag, hOut, flag_ion_fd, tg)) {
    printf("Build graph failed\n"); return 1;
  }
  printf("Graph built: SyncWait(flag_ion_fd=%d) → RmsNorm\n\n", (int)flag_ion_fd);

  volatile uint32_t* flag_ptr = (volatile uint32_t*)ionFlag.ptr;

  // ──────────────────────────────────────────────────────────────────────────
  // 场景 A：flag=1 在 graphExecute 之前写入
  // 预期：~320us（DMA 拷贝时 flag 已为 1，SyncWait 立即通过）
  // ──────────────────────────────────────────────────────────────────────────
  printf("--- 场景 A：flag=1 预设（graphExecute 前写入）---\n");
  printf("  预期：~320us（基础功能验证，DMA 拷贝 flag=1）\n\n");

  for (int i = 0; i < warmup; ++i) {
    *flag_ptr = 1;
    g_qnn->graphExecute(tg.graph, tg.execIn, 2, tg.execOut, 1, nullptr, nullptr);
  }

  std::vector<double> timesA;
  for (int i = 0; i < steps; ++i) {
    *flag_ptr = 1;
    double t0 = now_us();
    g_qnn->graphExecute(tg.graph, tg.execIn, 2, tg.execOut, 1, nullptr, nullptr);
    timesA.push_back(now_us() - t0);
  }

  std::sort(timesA.begin(), timesA.end());
  printf("  A: flag=1 预设   min=%.0f  p50=%.0f  avg=%.0f  max=%.0f us\n\n",
         timesA.front(), timesA[timesA.size()/2],
         std::accumulate(timesA.begin(), timesA.end(), 0.0) / timesA.size(),
         timesA.back());

  // ──────────────────────────────────────────────────────────────────────────
  // 场景 B：flag=0 在 graphExecute 之前，CPU 线程延迟后写 1
  //
  // 有 DSP VA → DSP 直接轮询 DDR → 看到 CPU 写入 → ~delay_us 后完成
  // 无 DSP VA → DSP 看 TCM 副本 (=0) → ~410ms 超时
  // ──────────────────────────────────────────────────────────────────────────
  printf("--- 场景 B：flag=0，CPU 线程 %dus 后写 1 ---\n", delay_us);
  printf("  HAP_mmap_get(%d) 成功 → 预期 ~%d us + overhead（DSP 直接看到 CPU 写入）\n",
         (int)flag_ion_fd, delay_us);
  printf("  HAP_mmap_get 失败 → 预期 ~410ms 超时（TCM DMA 拷贝）\n\n");

  // Warmup with flag=1
  for (int i = 0; i < warmup; ++i) {
    *flag_ptr = 1;
    g_qnn->graphExecute(tg.graph, tg.execIn, 2, tg.execOut, 1, nullptr, nullptr);
  }

  std::vector<double> timesB;
  int steps_b = steps;
  for (int i = 0; i < steps_b; ++i) {
    *flag_ptr = 0;

    std::thread writer([flag_ptr, delay_us]() {
      usleep(delay_us);
      *flag_ptr = 1;
    });

    double t0 = now_us();
    g_qnn->graphExecute(tg.graph, tg.execIn, 2, tg.execOut, 1, nullptr, nullptr);
    double elapsed = now_us() - t0;
    timesB.push_back(elapsed);

    writer.join();

    if (steps_b <= 5)  // print each step when few
      printf("  B[%d]: %.0f us %s\n", i, elapsed,
             elapsed > 100000 ? "[TIMEOUT]" : elapsed > 10000 ? "[SLOW]" : "[OK]");
  }

  if (steps_b > 5) {
    std::sort(timesB.begin(), timesB.end());
    printf("  B: CPU 延迟写  min=%.0f  p50=%.0f  avg=%.0f  max=%.0f us\n",
           timesB.front(), timesB[timesB.size()/2],
           std::accumulate(timesB.begin(), timesB.end(), 0.0) / timesB.size(),
           timesB.back());
  }
  printf("\n");

  // ──────────────────────────────────────────────────────────────────────────
  // 场景 C：flag=0 永不写入（验证超时行为）
  // 预期：~410ms（10M 次 dcinva 超时）
  // ──────────────────────────────────────────────────────────────────────────
  printf("--- 场景 C：flag=0 永不写入（超时验证）---\n");
  printf("  预期：~410ms（无论有无 DSP VA，flag 始终为 0）\n\n");

  *flag_ptr = 0;
  double t0 = now_us();
  g_qnn->graphExecute(tg.graph, tg.execIn, 2, tg.execOut, 1, nullptr, nullptr);
  double elapsed_c = now_us() - t0;
  printf("  C: flag=0 永不写入  %.0f us  [%s]\n\n", elapsed_c,
         elapsed_c > 100000 ? "TIMEOUT（预期）" : "UNEXPECTED");

  // ──────────────────────────────────────────────────────────────────────────
  // 总结
  // ──────────────────────────────────────────────────────────────────────────
  printf("=== 总结 ===\n");
  printf("  flag ION fd:          %d\n", (int)flag_ion_fd);
  printf("  场景 A (flag=1 预设): p50 = %.0f us\n", timesA[timesA.size()/2]);
  printf("  场景 B (CPU 延迟写):  ");
  if (steps_b <= 5) {
    for (double t : timesB) printf("%.0f ", t);
    printf("us\n");
  } else {
    std::sort(timesB.begin(), timesB.end());
    printf("p50 = %.0f us\n", timesB[timesB.size()/2]);
  }
  printf("  场景 C (永不写):      %.0f us\n\n", elapsed_c);

  std::sort(timesB.begin(), timesB.end());
  bool b_ok = !timesB.empty() && timesB[timesB.size()/2] < 10000;
  if (b_ok) {
    printf("结论：HAP_mmap_get 成功，DSP 直接看到 CPU 的 ION 写入！\n");
    printf("  场景 B p50 ≈ %.0f us（≈%d us delay + overhead）\n",
           timesB[timesB.size()/2], delay_us);
    printf("  QNN TCM DMA 拷贝已绕过，DSP 直接轮询 DDR。\n");
    printf("  下一步：将此机制应用到 GPU→NPU pipeline。\n");
  } else {
    printf("结论：场景 B 超时（%.0f us）。\n", timesB[timesB.size()/2]);
    printf("  HAP_mmap_get 可能失败（DSP 回退到 TCM 拷贝路径）。\n");
    printf("  可能原因：\n");
    printf("  1. QNN memRegister 未建立 CDSP 映射（HAP_mmap_get fd 未知）\n");
    printf("  2. 需要先调用 fastrpc_mmap(CDSP_DOMAIN_ID, fd, ...) 建立映射\n");
    printf("  3. 内存类型不匹配（uncached 需要不同的 HAP_mmap 参数）\n");
  }

  // Cleanup
  std::vector<Qnn_MemHandle_t> handles = {hIn, hOut, hFlag};
  g_qnn->memDeRegister(handles.data(), (uint32_t)handles.size());
  g_qnn->contextFree(ctx, nullptr);
  if (device && g_qnn->deviceFree) g_qnn->deviceFree(device);
  g_qnn->backendFree(backend);
  if (log && g_qnn->logFree) g_qnn->logFree(log);
  dlclose(g_lib);
  freeIon(ionIn); freeIon(ionOut); freeIon(ionFlag); freeIon(ionGamma);

  return 0;
}
