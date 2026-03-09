// RMSNorm benchmark: GPU (OpenCL FP16) vs NPU (QNN HTP)
// Platform: SM8850, Adreno 840 + Hexagon V81
//
// Tests multiple NPU modes:
//   1. Native QNN_OP_RMS_NORM (FP16)
//   2. Decomposed RmsNorm via element-wise ops (FP16)
//   3. ElementWiseAdd baseline (UINT8) — measures NPU dispatch overhead

#define CL_TARGET_OPENCL_VERSION 200

#include <CL/cl.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

#include "QNN/QnnBackend.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnDevice.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnInterface.h"
#include "QNN/QnnLog.h"
#include "QNN/QnnOpDef.h"
#include "QNN/QnnMem.h"
#include "QNN/QnnTensor.h"
#include "QNN/HTP/QnnHtpDevice.h"
#include "QNN/HTP/QnnHtpGraph.h"
#include "QNN/HTP/QnnHtpPerfInfrastructure.h"

// ─── Timing ────────────────────────────────────────────────────────────────
static inline double now_sec() {
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ─── FP16 helpers ──────────────────────────────────────────────────────────
static uint16_t f32_to_f16(float f) {
  uint32_t x; memcpy(&x, &f, 4);
  uint16_t s = (x >> 16) & 0x8000;
  int e = ((x >> 23) & 0xFF) - 127 + 15;
  uint32_t m = x & 0x7FFFFF;
  if (e <= 0) return s;
  if (e >= 31) return s | 0x7C00;
  return s | (e << 10) | (m >> 13);
}

static float f16_to_f32(uint16_t h) {
  uint32_t s = (h & 0x8000) << 16;
  uint32_t e = (h >> 10) & 0x1F;
  uint32_t m = h & 0x3FF;
  if (e == 0) { if (m == 0) { float r; uint32_t v = s; memcpy(&r, &v, 4); return r; }
    e = 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF; }
  else if (e == 31) { uint32_t v = s | 0x7F800000 | (m << 13); float r; memcpy(&r, &v, 4); return r; }
  uint32_t v = s | ((e + 127 - 15) << 23) | (m << 13);
  float r; memcpy(&r, &v, 4); return r;
}

// ─── ION buffer ────────────────────────────────────────────────────────────
struct IonBuf { void* ptr = nullptr; int fd = -1; size_t size = 0; };

struct RpcMemApi {
  void* lib = nullptr;
  void* (*alloc)(int, int, int) = nullptr;
  void  (*free_)(void*) = nullptr;
  int   (*toFd)(void*) = nullptr;
};

static RpcMemApi& rpcApi() {
  static RpcMemApi a;
  static bool init = false;
  if (!init) {
    init = true;
    for (auto p : {"libcdsprpc.so", "/vendor/lib64/libcdsprpc.so"}) {
      a.lib = dlopen(p, RTLD_LAZY);
      if (a.lib) break;
    }
    if (a.lib) {
      a.alloc = (void*(*)(int,int,int))dlsym(a.lib, "rpcmem_alloc");
      a.free_ = (void(*)(void*))dlsym(a.lib, "rpcmem_free");
      a.toFd  = (int(*)(void*))dlsym(a.lib, "rpcmem_to_fd");
    }
  }
  return a;
}

static bool ionAlloc(size_t sz, uint8_t fill, IonBuf& b) {
  auto& r = rpcApi();
  if (!r.alloc || !r.toFd) return false;
  b.ptr = r.alloc(25, 0, (int)sz);
  if (!b.ptr) return false;
  b.size = sz;
  memset(b.ptr, fill, sz);
  b.fd = r.toFd(b.ptr);
  if (b.fd < 0) { r.free_(b.ptr); b.ptr = nullptr; return false; }
  return true;
}

static void ionFree(IonBuf& b) {
  if (b.ptr) { rpcApi().free_(b.ptr); b.ptr = nullptr; b.fd = -1; b.size = 0; }
}

// ─── Constants ─────────────────────────────────────────────────────────────
constexpr double kPeakBW = 84.8;  // GB/s LPDDR5X-5300

struct BenchResult {
  double latency_us = 0;
  double bw_gbps = 0;
  int iters = 0;
  bool ok = false;
  std::string err;
};

// ─── QNN Error Names ───────────────────────────────────────────────────────
static const char* qnnErrName(Qnn_ErrorHandle_t e) {
  switch (e) {
    case 0: return "SUCCESS";
    case 6000: return "GRAPH_INVALID_ARGUMENT";
    case 6001: return "GRAPH_INVALID_HANDLE";
    case 6002: return "GRAPH_DOES_NOT_EXIST";
    case 6003: return "GRAPH_INVALID_NAME";
    case 6004: return "GRAPH_INVALID_TENSOR";
    case 6005: return "GRAPH_INVALID_OP_CONFIG";
    case 6006: return "GRAPH_SET_PROFILE";
    case 6007: return "GRAPH_UNCONNECTED_NODE";
    case 6020: return "GRAPH_CREATE_FAILED";
    case 6021: return "GRAPH_OPTIMIZATION_FAILED";
    case 6022: return "GRAPH_FINALIZE_FAILED";
    default:   return "UNKNOWN";
  }
}

static bool qok(Qnn_ErrorHandle_t s, const char* w) {
  if (s != QNN_SUCCESS) {
    printf("[NPU] %s failed: %u (%s)\n", w, (unsigned)s, qnnErrName(s));
    return false;
  }
  return true;
}

// QNN log callback for debug output
static void qnnLogCallback(const char* fmt, QnnLog_Level_t level, uint64_t /*ts*/, va_list args) {
  const char* lvl = "???";
  switch (level) {
    case QNN_LOG_LEVEL_ERROR:   lvl = "ERR"; break;
    case QNN_LOG_LEVEL_WARN:    lvl = "WRN"; break;
    case QNN_LOG_LEVEL_INFO:    lvl = "INF"; break;
    case QNN_LOG_LEVEL_VERBOSE: lvl = "VRB"; break;
    case QNN_LOG_LEVEL_DEBUG:   lvl = "DBG"; break;
    default: break;
  }
  printf("[QNN-%s] ", lvl);
  vprintf(fmt, args);
  if (fmt[strlen(fmt)-1] != '\n') printf("\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// NPU Session: each test creates a fresh backend/context/graph
// ═══════════════════════════════════════════════════════════════════════════
enum class NpuMode { NATIVE, DECOMPOSED, ELEMENT_ADD };

struct NpuSession {
  void* libHtp = nullptr;
  const QNN_INTERFACE_VER_TYPE* q = nullptr;
  Qnn_LogHandle_t logger = nullptr;
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  Qnn_ContextHandle_t ctx = nullptr;
  Qnn_GraphHandle_t graph = nullptr;
  uint32_t coreCount = 0;
  bool verbose = false;

  IonBuf ionIn, ionOut, ionGamma, ionBeta, ionB;
  Qnn_MemHandle_t regIn = nullptr, regOut = nullptr, regB = nullptr;

  Qnn_Tensor_t execIn[2], execOut[1];
  int numIn = 1;

  NpuMode mode = NpuMode::NATIVE;
  int batch = 0, hidden = 0;
  size_t elemBytes = 2;
  bool quiet = false;  // suppress strategy prints during benchmark

  uint32_t dimsIO[4], dimsGamma[1], dimsScalar[4], dimsMean[4], dimsAxes[1];

  void cleanup() {
    if (q && regIn)  { Qnn_MemHandle_t h[] = {regIn}; q->memDeRegister(h, 1); regIn = nullptr; }
    if (q && regOut) { Qnn_MemHandle_t h[] = {regOut}; q->memDeRegister(h, 1); regOut = nullptr; }
    if (q && regB)   { Qnn_MemHandle_t h[] = {regB}; q->memDeRegister(h, 1); regB = nullptr; }
    if (q && graph)   { /* no graphFree in v1 API */ }
    if (q && ctx)     q->contextFree(ctx, nullptr);
    if (q && device && q->deviceFree) q->deviceFree(device);
    if (q && backend) q->backendFree(backend);
    if (q && logger)  q->logFree(logger);
    if (libHtp)       dlclose(libHtp);
    ctx = nullptr; device = nullptr; backend = nullptr;
    graph = nullptr; q = nullptr; logger = nullptr; libHtp = nullptr;
    ionFree(ionIn); ionFree(ionOut); ionFree(ionGamma); ionFree(ionBeta); ionFree(ionB);
  }

  bool loadBackend() {
    libHtp = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
    if (!libHtp) { printf("[NPU] dlopen: %s\n", dlerror()); return false; }

    auto getProv = (decltype(&QnnInterface_getProviders))dlsym(libHtp, "QnnInterface_getProviders");
    if (!getProv) return false;

    const QnnInterface_t** provs = nullptr; uint32_t np = 0;
    if (getProv(&provs, &np) != QNN_SUCCESS || np == 0) return false;

    const QnnInterface_t* best = provs[0];
    for (uint32_t i = 0; i < np; i++)
      if (provs[i] && provs[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR)
        if (!best || provs[i]->apiVersion.coreApiVersion.minor > best->apiVersion.coreApiVersion.minor)
          best = provs[i];
    q = &best->QNN_INTERFACE_VER_NAME;

    // Create logger for verbose mode
    if (verbose && q->logCreate) {
      auto ls = q->logCreate(qnnLogCallback, QNN_LOG_LEVEL_DEBUG, &logger);
      if (ls != QNN_SUCCESS) printf("[NPU] logCreate failed: %u (non-fatal)\n", (unsigned)ls);
    }

    if (!qok(q->backendCreate(logger, nullptr, &backend), "backendCreate")) return false;

    if (q->deviceCreate) {
      auto s = q->deviceCreate(nullptr, nullptr, &device);
      if (s != QNN_SUCCESS && s != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) return false;
    }

    setHighPerf();

    if (!qok(q->contextCreate(backend, device, nullptr, &ctx), "contextCreate")) return false;

    coreCount = queryCores();
    return true;
  }

  void setHighPerf() {
    if (!q->deviceGetInfrastructure) return;
    QnnDevice_Infrastructure_t infra = nullptr;
    if (QNN_SUCCESS != q->deviceGetInfrastructure(&infra) || !infra) return;
    auto* hi = (QnnHtpDevice_Infrastructure_t*)infra;
    if (hi->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
    auto& p = hi->perfInfra;
    if (!p.createPowerConfigId || !p.setPowerConfig) return;

    uint32_t pid = 0;
    if (QNN_SUCCESS != p.createPowerConfigId(0, 0, &pid)) return;

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
    p.setPowerConfig(pid, cfgs);
  }

  uint32_t queryCores() {
    if (!q->deviceGetPlatformInfo) return 0;
    const QnnDevice_PlatformInfo_t* info = nullptr;
    if (QNN_SUCCESS != q->deviceGetPlatformInfo(nullptr, &info) || !info) return 0;
    uint32_t c = 0;
    if (info->version == QNN_DEVICE_PLATFORM_INFO_VERSION_1)
      for (uint32_t i = 0; i < info->v1.numHwDevices; i++) {
        auto& d = info->v1.hwDevices[i];
        if (d.version != QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1) continue;
        auto* ext = (QnnHtpDevice_DeviceInfoExtension_t*)d.v1.deviceInfoExtension;
        if (ext && ext->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) { c = d.v1.numCores; break; }
      }
    if (q->deviceFreePlatformInfo) q->deviceFreePlatformInfo(nullptr, info);
    return c;
  }

  bool regBuf(const IonBuf& ion, const uint32_t* dims, uint32_t nd, Qnn_DataType_t dt, Qnn_MemHandle_t& h) {
    Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
    desc.memShape.numDim = nd;
    desc.memShape.dimSize = const_cast<uint32_t*>(dims);
    desc.dataType = dt;
    desc.memType = QNN_MEM_TYPE_ION;
    desc.ionInfo.fd = ion.fd;
    return qok(q->memRegister(ctx, &desc, 1, &h), "memRegister");
  }

  // Build graph config with optional FP16 precision
  bool createGraph(const char* name, bool fp16Precision) {
    // Build HTP custom configs - properly terminated
    QnnHtpGraph_CustomConfig_t htpCfg[4];
    for (int i = 0; i < 4; i++) htpCfg[i] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    int n = 0;

    if (fp16Precision) {
      htpCfg[n].option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
      htpCfg[n].precision = QNN_PRECISION_FLOAT16;
      n++;
    }
    if (coreCount > 0) {
      htpCfg[n].option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_CORES;
      htpCfg[n].numCores = coreCount;
      n++;
    }
    htpCfg[n].option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
    htpCfg[n].numHvxThreads = 8;
    n++;
    // htpCfg[n] stays as QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT (terminator)

    QnnGraph_Config_t gc = QNN_GRAPH_CONFIG_INIT;
    gc.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    gc.customConfig = htpCfg;
    const QnnGraph_Config_t* gcl[] = {&gc, nullptr};

    return qok(q->graphCreate(ctx, name, gcl, &graph), "graphCreate");
  }

  // ── Tensor helpers ──
  Qnn_Tensor_t mkTensor(const char* nm, Qnn_TensorType_t tp, uint32_t* d, uint32_t rank,
                          Qnn_DataType_t dtype) {
    Qnn_Tensor_t t = QNN_TENSOR_INIT;
    t.version = QNN_TENSOR_VERSION_1;
    t.v1.name = nm; t.v1.type = tp;
    t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType = dtype;
    t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    t.v1.rank = rank; t.v1.dimensions = d;
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    t.v1.clientBuf = QNN_CLIENT_BUFFER_INIT;
    return t;
  }

  Qnn_Tensor_t mkFp16(const char* nm, Qnn_TensorType_t tp, uint32_t* d, uint32_t rank) {
    return mkTensor(nm, tp, d, rank, QNN_DATATYPE_FLOAT_16);
  }

  Qnn_Tensor_t mkFp32(const char* nm, Qnn_TensorType_t tp, uint32_t* d, uint32_t rank) {
    return mkTensor(nm, tp, d, rank, QNN_DATATYPE_FLOAT_32);
  }

  Qnn_Tensor_t mkU8(const char* nm, Qnn_TensorType_t tp, uint32_t* d) {
    Qnn_Tensor_t t = QNN_TENSOR_INIT;
    t.version = QNN_TENSOR_VERSION_1;
    t.v1.name = nm; t.v1.type = tp;
    t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType = QNN_DATATYPE_UFIXED_POINT_8;
    t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
    t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    t.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
    t.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
    t.v1.rank = 4; t.v1.dimensions = d;
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    t.v1.clientBuf = QNN_CLIENT_BUFFER_INIT;
    return t;
  }

  // ── Build Native RmsNorm graph ──
  // useFp32: use FLOAT_32 data type (HTP will internally use FP16)
  bool buildNative(bool withBeta, bool useFp32 = false) {
    auto mk = useFp32 ? &NpuSession::mkFp32 : &NpuSession::mkFp16;

    Qnn_Tensor_t input  = (this->*mk)("input",  QNN_TENSOR_TYPE_APP_WRITE, dimsIO, 4);
    Qnn_Tensor_t output = (this->*mk)("output", QNN_TENSOR_TYPE_APP_READ,  dimsIO, 4);

    // Gamma: rank-1 [hidden_dim] per MasterOpDef (rank = size(axes) = 1)
    Qnn_Tensor_t gamma  = (this->*mk)("gamma",  QNN_TENSOR_TYPE_STATIC, dimsGamma, 1);
    gamma.v1.clientBuf.data = ionGamma.ptr;
    gamma.v1.clientBuf.dataSize = (uint32_t)ionGamma.size;

    if (!qok(q->tensorCreateGraphTensor(graph, &input), "tensor:input")) return false;
    if (!qok(q->tensorCreateGraphTensor(graph, &gamma), "tensor:gamma")) return false;

    Qnn_Tensor_t beta;
    if (withBeta) {
      beta = (this->*mk)("beta", QNN_TENSOR_TYPE_STATIC, dimsGamma, 1);
      beta.v1.clientBuf.data = ionBeta.ptr;
      beta.v1.clientBuf.dataSize = (uint32_t)ionBeta.size;
      if (!qok(q->tensorCreateGraphTensor(graph, &beta), "tensor:beta")) return false;
    }

    if (!qok(q->tensorCreateGraphTensor(graph, &output), "tensor:output")) return false;

    // Params: epsilon + axes
    Qnn_Param_t epsP = QNN_PARAM_INIT;
    epsP.paramType = QNN_PARAMTYPE_SCALAR;
    epsP.name = QNN_OP_RMS_NORM_PARAM_EPSILON;
    epsP.scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
    epsP.scalarParam.floatValue = 1e-6f;

    // Axes tensor - MUST be registered as graph tensor before use in param
    uint32_t axesData[] = {3};
    dimsAxes[0] = 1;
    Qnn_Tensor_t axesTensor = QNN_TENSOR_INIT;
    axesTensor.version = QNN_TENSOR_VERSION_1;
    axesTensor.v1.name = "axes";
    axesTensor.v1.type = QNN_TENSOR_TYPE_STATIC;
    axesTensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    axesTensor.v1.dataType = QNN_DATATYPE_UINT_32;
    axesTensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    axesTensor.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    axesTensor.v1.rank = 1;
    axesTensor.v1.dimensions = dimsAxes;
    axesTensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    axesTensor.v1.clientBuf.data = axesData;
    axesTensor.v1.clientBuf.dataSize = sizeof(axesData);
    if (!qok(q->tensorCreateGraphTensor(graph, &axesTensor), "tensor:axes")) return false;

    Qnn_Param_t axesP = QNN_PARAM_INIT;
    axesP.paramType = QNN_PARAMTYPE_TENSOR;
    axesP.name = QNN_OP_RMS_NORM_PARAM_AXES;
    axesP.tensorParam = axesTensor;

    Qnn_Param_t params[] = {epsP, axesP};

    Qnn_Tensor_t in3[] = {input, gamma, beta};
    Qnn_Tensor_t in2[] = {input, gamma};
    Qnn_Tensor_t out[] = {output};

    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = "rmsnorm"; op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName = QNN_OP_RMS_NORM;
    op.v1.numOfParams = 2; op.v1.params = params;
    op.v1.numOfInputs = withBeta ? 3 : 2;
    op.v1.inputTensors = withBeta ? in3 : in2;
    op.v1.numOfOutputs = 1; op.v1.outputTensors = out;

    if (!qok(q->graphAddNode(graph, op), "graphAddNode(RmsNorm)")) return false;

    execIn[0] = input; execOut[0] = output; numIn = 1;
    return true;
  }

  // ── Build Decomposed RmsNorm graph (FP16) ──
  bool buildDecomposed() {
    Qnn_Tensor_t input  = mkFp16("input",  QNN_TENSOR_TYPE_APP_WRITE, dimsIO, 4);
    Qnn_Tensor_t output = mkFp16("output", QNN_TENSOR_TYPE_APP_READ,  dimsIO, 4);
    Qnn_Tensor_t gamma  = mkFp16("gamma",  QNN_TENSOR_TYPE_STATIC, dimsGamma, 1);
    gamma.v1.clientBuf.data = ionGamma.ptr;
    gamma.v1.clientBuf.dataSize = (uint32_t)ionGamma.size;

    // For broadcasting, gamma needs rank-4 for element-wise multiply
    uint32_t dimsGamma4[4] = {1, 1, 1, (uint32_t)hidden};
    Qnn_Tensor_t gamma4 = mkFp16("gamma4", QNN_TENSOR_TYPE_STATIC, dimsGamma4, 4);
    gamma4.v1.clientBuf.data = ionGamma.ptr;
    gamma4.v1.clientBuf.dataSize = (uint32_t)ionGamma.size;

    uint16_t epsVal = f32_to_f16(1e-6f);
    Qnn_Tensor_t eps = mkFp16("eps", QNN_TENSOR_TYPE_STATIC, dimsScalar, 4);
    eps.v1.clientBuf.data = &epsVal;
    eps.v1.clientBuf.dataSize = sizeof(epsVal);

    Qnn_Tensor_t x_sq       = mkFp16("x_sq",       QNN_TENSOR_TYPE_NATIVE, dimsIO, 4);
    Qnn_Tensor_t mean_sq    = mkFp16("mean_sq",     QNN_TENSOR_TYPE_NATIVE, dimsMean, 4);
    Qnn_Tensor_t mean_eps   = mkFp16("mean_eps",    QNN_TENSOR_TYPE_NATIVE, dimsMean, 4);
    Qnn_Tensor_t inv_rms    = mkFp16("inv_rms",     QNN_TENSOR_TYPE_NATIVE, dimsMean, 4);
    Qnn_Tensor_t normalized = mkFp16("normalized",  QNN_TENSOR_TYPE_NATIVE, dimsIO, 4);

    // Create all tensors
    for (auto* t : {&input, &gamma4, &eps, &x_sq, &mean_sq, &mean_eps, &inv_rms, &normalized, &output})
      if (!qok(q->tensorCreateGraphTensor(graph, t), "tensor:decomposed")) return false;

    // Node 1: x_sq = Mul(input, input)
    {
      Qnn_Tensor_t i[] = {input, input}; Qnn_Tensor_t o[] = {x_sq};
      Qnn_OpConfig_t c = QNN_OPCONFIG_INIT; c.version = QNN_OPCONFIG_VERSION_1;
      c.v1.name = "mul_sq"; c.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
      c.v1.typeName = QNN_OP_ELEMENT_WISE_MULTIPLY;
      c.v1.numOfInputs = 2; c.v1.inputTensors = i;
      c.v1.numOfOutputs = 1; c.v1.outputTensors = o;
      if (!qok(q->graphAddNode(graph, c), "node:mul_sq")) return false;
    }

    // Node 2: mean_sq = ReduceMean(x_sq, axes=[3], keep_dims=true)
    {
      uint32_t ax[] = {3}; uint32_t axD[] = {1};
      Qnn_Tensor_t axT = QNN_TENSOR_INIT;
      axT.version = QNN_TENSOR_VERSION_1; axT.v1.name = "red_axes";
      axT.v1.type = QNN_TENSOR_TYPE_STATIC;
      axT.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
      axT.v1.dataType = QNN_DATATYPE_UINT_32;
      axT.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
      axT.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
      axT.v1.rank = 1; axT.v1.dimensions = axD;
      axT.v1.memType = QNN_TENSORMEMTYPE_RAW;
      axT.v1.clientBuf.data = ax; axT.v1.clientBuf.dataSize = sizeof(ax);
      if (!qok(q->tensorCreateGraphTensor(graph, &axT), "tensor:red_axes")) return false;

      Qnn_Param_t p1 = QNN_PARAM_INIT;
      p1.paramType = QNN_PARAMTYPE_TENSOR; p1.name = QNN_OP_REDUCE_MEAN_PARAM_AXES;
      p1.tensorParam = axT;
      Qnn_Param_t p2 = QNN_PARAM_INIT;
      p2.paramType = QNN_PARAMTYPE_SCALAR; p2.name = QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS;
      p2.scalarParam.dataType = QNN_DATATYPE_BOOL_8; p2.scalarParam.bool8Value = 1;
      Qnn_Param_t pp[] = {p1, p2};

      Qnn_Tensor_t i[] = {x_sq}; Qnn_Tensor_t o[] = {mean_sq};
      Qnn_OpConfig_t c = QNN_OPCONFIG_INIT; c.version = QNN_OPCONFIG_VERSION_1;
      c.v1.name = "reduce_mean"; c.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
      c.v1.typeName = QNN_OP_REDUCE_MEAN;
      c.v1.numOfParams = 2; c.v1.params = pp;
      c.v1.numOfInputs = 1; c.v1.inputTensors = i;
      c.v1.numOfOutputs = 1; c.v1.outputTensors = o;
      if (!qok(q->graphAddNode(graph, c), "node:reduce_mean")) return false;
    }

    // Node 3: mean_eps = Add(mean_sq, eps)
    {
      Qnn_Tensor_t i[] = {mean_sq, eps}; Qnn_Tensor_t o[] = {mean_eps};
      Qnn_OpConfig_t c = QNN_OPCONFIG_INIT; c.version = QNN_OPCONFIG_VERSION_1;
      c.v1.name = "add_eps"; c.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
      c.v1.typeName = QNN_OP_ELEMENT_WISE_ADD;
      c.v1.numOfInputs = 2; c.v1.inputTensors = i;
      c.v1.numOfOutputs = 1; c.v1.outputTensors = o;
      if (!qok(q->graphAddNode(graph, c), "node:add_eps")) return false;
    }

    // Node 4: inv_rms = Rsqrt(mean_eps)
    {
      Qnn_Tensor_t i[] = {mean_eps}; Qnn_Tensor_t o[] = {inv_rms};
      Qnn_OpConfig_t c = QNN_OPCONFIG_INIT; c.version = QNN_OPCONFIG_VERSION_1;
      c.v1.name = "rsqrt"; c.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
      c.v1.typeName = QNN_OP_ELEMENT_WISE_RSQRT;
      c.v1.numOfInputs = 1; c.v1.inputTensors = i;
      c.v1.numOfOutputs = 1; c.v1.outputTensors = o;
      if (!qok(q->graphAddNode(graph, c), "node:rsqrt")) return false;
    }

    // Node 5: normalized = Mul(input, inv_rms)
    {
      Qnn_Tensor_t i[] = {input, inv_rms}; Qnn_Tensor_t o[] = {normalized};
      Qnn_OpConfig_t c = QNN_OPCONFIG_INIT; c.version = QNN_OPCONFIG_VERSION_1;
      c.v1.name = "mul_norm"; c.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
      c.v1.typeName = QNN_OP_ELEMENT_WISE_MULTIPLY;
      c.v1.numOfInputs = 2; c.v1.inputTensors = i;
      c.v1.numOfOutputs = 1; c.v1.outputTensors = o;
      if (!qok(q->graphAddNode(graph, c), "node:mul_norm")) return false;
    }

    // Node 6: output = Mul(normalized, gamma4)
    {
      Qnn_Tensor_t i[] = {normalized, gamma4}; Qnn_Tensor_t o[] = {output};
      Qnn_OpConfig_t c = QNN_OPCONFIG_INIT; c.version = QNN_OPCONFIG_VERSION_1;
      c.v1.name = "mul_gamma"; c.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
      c.v1.typeName = QNN_OP_ELEMENT_WISE_MULTIPLY;
      c.v1.numOfInputs = 2; c.v1.inputTensors = i;
      c.v1.numOfOutputs = 1; c.v1.outputTensors = o;
      if (!qok(q->graphAddNode(graph, c), "node:mul_gamma")) return false;
    }

    execIn[0] = input; execOut[0] = output; numIn = 1;
    return true;
  }

  // ── Build ElementAdd graph (UINT8 baseline) ──
  bool buildElementAdd() {
    Qnn_Tensor_t in0 = mkU8("in0", QNN_TENSOR_TYPE_APP_WRITE, dimsIO);
    Qnn_Tensor_t in1 = mkU8("in1", QNN_TENSOR_TYPE_APP_WRITE, dimsIO);
    Qnn_Tensor_t out = mkU8("out", QNN_TENSOR_TYPE_APP_READ,  dimsIO);

    if (!qok(q->tensorCreateGraphTensor(graph, &in0), "tensor:in0")) return false;
    if (!qok(q->tensorCreateGraphTensor(graph, &in1), "tensor:in1")) return false;
    if (!qok(q->tensorCreateGraphTensor(graph, &out), "tensor:out")) return false;

    Qnn_Tensor_t opIn[] = {in0, in1}; Qnn_Tensor_t opOut[] = {out};
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT; op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = "add"; op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName = QNN_OP_ELEMENT_WISE_ADD;
    op.v1.numOfInputs = 2; op.v1.inputTensors = opIn;
    op.v1.numOfOutputs = 1; op.v1.outputTensors = opOut;

    if (!qok(q->graphAddNode(graph, op), "graphAddNode(Add)")) return false;

    execIn[0] = in0; execIn[1] = in1; execOut[0] = out; numIn = 2;
    return true;
  }

  // ── Initialize a complete session ──
  bool init(int b, int h, NpuMode m) {
    batch = b; hidden = h; mode = m;
    bool isU8 = (m == NpuMode::ELEMENT_ADD);
    elemBytes = isU8 ? 1 : 2;

    size_t tBytes = (size_t)batch * hidden * elemBytes;
    size_t gBytes = (size_t)hidden * elemBytes;

    // Dimensions
    if (isU8) {
      uint32_t C = 1048576; if (tBytes < C) C = (uint32_t)tBytes;
      dimsIO[0] = (uint32_t)(tBytes / C); dimsIO[1] = 1; dimsIO[2] = 1; dimsIO[3] = C;
    } else {
      dimsIO[0] = batch; dimsIO[1] = 1; dimsIO[2] = 1; dimsIO[3] = hidden;
    }
    dimsGamma[0] = hidden;
    dimsScalar[0] = 1; dimsScalar[1] = 1; dimsScalar[2] = 1; dimsScalar[3] = 1;
    dimsMean[0] = batch; dimsMean[1] = 1; dimsMean[2] = 1; dimsMean[3] = 1;

    // Alloc ION
    if (isU8) {
      if (!ionAlloc(tBytes, 1, ionIn) || !ionAlloc(tBytes, 0, ionOut)) return false;
      if (!ionAlloc(tBytes, 2, ionB)) return false;
    } else {
      // Allocate large enough for FP32 (4 bytes) since Native mode tries both FP32 and FP16
      size_t maxTBytes = (size_t)batch * hidden * 4;
      size_t maxGBytes = (size_t)hidden * 4;
      if (!ionAlloc(maxTBytes, 0, ionIn) || !ionAlloc(maxTBytes, 0, ionOut)) return false;
      if (!ionAlloc(maxGBytes, 0, ionGamma) || !ionAlloc(maxGBytes, 0, ionBeta)) return false;

      // Store FP32 values (works for both FP32 and FP16 tests)
      std::mt19937 rng(42);
      std::uniform_real_distribution<float> dist(0.1f, 1.0f);
      auto* pf = (float*)ionIn.ptr;
      for (int i = 0; i < batch * hidden; i++) pf[i] = dist(rng);

      auto* gf = (float*)ionGamma.ptr;
      for (int i = 0; i < hidden; i++) gf[i] = 1.0f;

      memset(ionBeta.ptr, 0, maxGBytes);
    }

    // Load QNN backend
    if (!loadBackend()) return false;

    // Register buffers - for FP16/FP32 we'll re-register per strategy
    Qnn_DataType_t dt = isU8 ? QNN_DATATYPE_UFIXED_POINT_8 : QNN_DATATYPE_FLOAT_16;
    if (isU8) {
      if (!regBuf(ionIn, dimsIO, 4, dt, regIn) || !regBuf(ionOut, dimsIO, 4, dt, regOut))
        return false;
      if (!regBuf(ionB, dimsIO, 4, dt, regB)) return false;
    }

    // Try building graph - for NATIVE mode, try multiple strategies
    bool graphOk = false;

    if (m == NpuMode::NATIVE) {
      // Strategy: try different configs, each with FRESH context
      // fp32dt: use FLOAT_32 data type (HTP supplement lists FLOAT_32 configs)
      struct Strat { const char* name; bool fp16prec; bool beta; bool fp32dt; };
      Strat strats[] = {
        // FP32 data type configs (HTP may internally convert to FP16)
        {"FP32dt+FP16prec+3in",   true,  true,  true},
        {"FP32dt+FP16prec+2in",   true,  false, true},
        {"FP32dt+NoPrec+3in",     false, true,  true},
        {"FP32dt+NoPrec+2in",     false, false, true},
        // FP16 data type configs
        {"FP16dt+FP16prec+3in",   true,  true,  false},
        {"FP16dt+FP16prec+2in",   true,  false, false},
        {"FP16dt+NoPrec+3in",     false, true,  false},
        {"FP16dt+NoPrec+2in",     false, false, false},
      };
      for (auto& s : strats) {
        if (!quiet) printf("[NPU] Try Native: %s\n", s.name);
        // Register buffers with correct data type for this strategy
        Qnn_DataType_t stratDt = s.fp32dt ? QNN_DATATYPE_FLOAT_32 : QNN_DATATYPE_FLOAT_16;
        if (!regIn) {
          if (!regBuf(ionIn, dimsIO, 4, stratDt, regIn) || !regBuf(ionOut, dimsIO, 4, stratDt, regOut))
            return false;
        }
        if (createGraph(s.name, s.fp16prec) && buildNative(s.beta, s.fp32dt)) {
          if (qok(q->graphFinalize(graph, nullptr, nullptr), "graphFinalize")) {
            if (!quiet) printf("[NPU] Native '%s' OK!\n", s.name);
            graphOk = true; break;
          }
        }
        // Failed - must destroy context and recreate to avoid stale state
        if (!quiet) printf("[NPU] Native '%s' failed, recreating context\n", s.name);
        if (regIn) { Qnn_MemHandle_t h[] = {regIn}; q->memDeRegister(h, 1); regIn = nullptr; }
        if (regOut) { Qnn_MemHandle_t h[] = {regOut}; q->memDeRegister(h, 1); regOut = nullptr; }
        q->contextFree(ctx, nullptr); ctx = nullptr; graph = nullptr;
        if (!qok(q->contextCreate(backend, device, nullptr, &ctx), "contextCreate(retry)"))
          return false;
      }
    } else if (m == NpuMode::DECOMPOSED) {
      bool fp16_opts[] = {true, false};
      for (bool fp16 : fp16_opts) {
        if (!quiet) printf("[NPU] Try Decomposed: fp16=%d\n", fp16);
        if (!regIn) {
          if (!regBuf(ionIn, dimsIO, 4, QNN_DATATYPE_FLOAT_16, regIn) ||
              !regBuf(ionOut, dimsIO, 4, QNN_DATATYPE_FLOAT_16, regOut))
            return false;
        }
        if (createGraph("decomposed", fp16) && buildDecomposed()) {
          if (qok(q->graphFinalize(graph, nullptr, nullptr), "graphFinalize")) {
            if (!quiet) printf("[NPU] Decomposed OK!\n");
            graphOk = true; break;
          }
        }
        if (!quiet) printf("[NPU] Decomposed fp16=%d failed, recreating context\n", fp16);
        if (regIn) { Qnn_MemHandle_t h[] = {regIn}; q->memDeRegister(h, 1); regIn = nullptr; }
        if (regOut) { Qnn_MemHandle_t h[] = {regOut}; q->memDeRegister(h, 1); regOut = nullptr; }
        q->contextFree(ctx, nullptr); ctx = nullptr; graph = nullptr;
        if (!qok(q->contextCreate(backend, device, nullptr, &ctx), "contextCreate(retry)"))
          return false;
      }
    } else {
      if (createGraph("add_graph", false) && buildElementAdd())
        graphOk = qok(q->graphFinalize(graph, nullptr, nullptr), "graphFinalize");
    }

    if (!graphOk) return false;

    // Bind memory handles
    execIn[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    execIn[0].v1.memHandle = regIn;
    if (numIn > 1) { execIn[1].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE; execIn[1].v1.memHandle = regB; }
    execOut[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    execOut[0].v1.memHandle = regOut;
    return true;
  }

  BenchResult run(int warmup, int iters) {
    BenchResult r; r.iters = iters;
    for (int i = 0; i < warmup; i++)
      if (!qok(q->graphExecute(graph, execIn, numIn, execOut, 1, nullptr, nullptr), "warmup"))
        { r.err = "warmup failed"; return r; }

    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
      if (!qok(q->graphExecute(graph, execIn, numIn, execOut, 1, nullptr, nullptr), "exec"))
        { r.err = "exec failed"; return r; }
    double t1 = now_sec();

    double bpc;
    if (mode == NpuMode::ELEMENT_ADD)
      bpc = (double)batch * hidden * elemBytes * 3.0;
    else
      bpc = (double)batch * hidden * elemBytes * 2.0 + (double)hidden * elemBytes;

    r.latency_us = ((t1 - t0) / iters) * 1e6;
    r.bw_gbps = (bpc * iters / (1024.0*1024.0*1024.0)) / (t1 - t0);
    r.ok = true;
    return r;
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// GPU Session (OpenCL FP16 RMSNorm)
// ═══════════════════════════════════════════════════════════════════════════
struct GpuSession {
  cl_platform_id plat = nullptr;
  cl_device_id dev = nullptr;
  cl_context ctx = nullptr;
  cl_command_queue queue = nullptr;
  cl_program prog = nullptr;
  cl_kernel kern = nullptr;
  cl_mem bufIn = nullptr, bufOut = nullptr, bufGamma = nullptr;
  int batch = 0, hidden = 0;

  void cleanup() {
    if (kern) clReleaseKernel(kern);
    if (bufIn) clReleaseMemObject(bufIn);
    if (bufOut) clReleaseMemObject(bufOut);
    if (bufGamma) clReleaseMemObject(bufGamma);
    if (prog) clReleaseProgram(prog);
    if (queue) clReleaseCommandQueue(queue);
    if (ctx) clReleaseContext(ctx);
    kern = nullptr; bufIn = nullptr; bufOut = nullptr; bufGamma = nullptr;
    prog = nullptr; queue = nullptr; ctx = nullptr;
  }

  void printInfo() {
    if (!dev) return;
    char name[256]; cl_uint cu; cl_ulong mem;
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
    printf("  GPU: %s, %u CU, %.2f GB\n", name, cu, mem / (1024.0*1024.0*1024.0));
  }

  bool init(int b, int h, const char* kpath) {
    batch = b; hidden = h;
    cl_int err;
    err = clGetPlatformIDs(1, &plat, nullptr);
    if (err) { printf("[GPU] platform: %d\n", err); return false; }
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    if (err) { printf("[GPU] device: %d\n", err); return false; }
    ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (err) return false;
    queue = clCreateCommandQueueWithProperties(ctx, dev, nullptr, &err);
    if (err) return false;

    // Read kernel
    FILE* f = fopen(kpath, "r");
    if (!f) { printf("[GPU] cannot read %s\n", kpath); return false; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* src = (char*)malloc(sz + 1);
    size_t n = fread(src, 1, sz, f); src[n] = '\0'; fclose(f);
    size_t sn = n;
    prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, &sn, &err);
    free(src);
    if (err) return false;

    err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
    if (err) {
      size_t lsz; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &lsz);
      char* log = (char*)malloc(lsz);
      clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, lsz, log, nullptr);
      printf("[GPU] Build:\n%s\n", log); free(log); return false;
    }
    kern = clCreateKernel(prog, "rmsnorm", &err);
    if (err) return false;

    size_t tBytes = (size_t)b * h * 2;
    size_t gBytes = (size_t)h * 2;
    bufIn = clCreateBuffer(ctx, CL_MEM_READ_ONLY, tBytes, nullptr, &err); if (err) return false;
    bufOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, tBytes, nullptr, &err); if (err) return false;
    bufGamma = clCreateBuffer(ctx, CL_MEM_READ_ONLY, gBytes, nullptr, &err); if (err) return false;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);
    std::vector<uint16_t> hIn(b * h);
    for (auto& v : hIn) v = f32_to_f16(dist(rng));
    clEnqueueWriteBuffer(queue, bufIn, CL_TRUE, 0, tBytes, hIn.data(), 0, nullptr, nullptr);

    std::vector<uint16_t> hG(h, f32_to_f16(1.0f));
    clEnqueueWriteBuffer(queue, bufGamma, CL_TRUE, 0, gBytes, hG.data(), 0, nullptr, nullptr);
    return true;
  }

  BenchResult run(int warmup, int iters) {
    BenchResult r; r.iters = iters;
    size_t local = 256;
    size_t mwg; clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(mwg), &mwg, nullptr);
    if (local > mwg) local = mwg;
    size_t global = (size_t)batch * local;

    float eps = 1e-6f;
    clSetKernelArg(kern, 0, sizeof(cl_mem), &bufOut);
    clSetKernelArg(kern, 1, sizeof(cl_mem), &bufIn);
    clSetKernelArg(kern, 2, sizeof(cl_mem), &bufGamma);
    clSetKernelArg(kern, 3, sizeof(int), &hidden);
    clSetKernelArg(kern, 4, sizeof(float), &eps);
    clSetKernelArg(kern, 5, local * sizeof(float), nullptr);

    for (int i = 0; i < warmup; i++)
      clEnqueueNDRangeKernel(queue, kern, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    clFinish(queue);

    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
      clEnqueueNDRangeKernel(queue, kern, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    clFinish(queue);
    double t1 = now_sec();

    double bpc = (double)batch * hidden * 2.0 * 2 + (double)hidden * 2;
    r.latency_us = ((t1 - t0) / iters) * 1e6;
    r.bw_gbps = (bpc * iters / (1024.0*1024.0*1024.0)) / (t1 - t0);
    r.ok = true;
    return r;
  }

  bool readOutput(void* dst, size_t bytes) {
    return CL_SUCCESS == clEnqueueReadBuffer(queue, bufOut, CL_TRUE, 0, bytes, dst, 0, nullptr, nullptr);
  }
};

// ─── Auto-compute iterations ──────────────────────────────────────────────
static int autoIters(int b, int h, double estBW) {
  double bpc = (double)b * h * 2 * 3;
  double spc = (bpc / (1024.0*1024.0*1024.0)) / estBW;
  if (spc < 1e-9) spc = 1e-6;
  return std::clamp((int)(0.2 / spc), 50, 10000);
}

// ─── CPU Reference ────────────────────────────────────────────────────────
static void cpuRmsNorm(const uint16_t* in, const uint16_t* gamma, uint16_t* out,
                       int batch, int hidden, float eps) {
  for (int b = 0; b < batch; b++) {
    const uint16_t* x = in + b * hidden;
    uint16_t* y = out + b * hidden;
    float ss = 0;
    for (int i = 0; i < hidden; i++) { float v = f16_to_f32(x[i]); ss += v * v; }
    float rms = 1.0f / sqrtf(ss / hidden + eps);
    for (int i = 0; i < hidden; i++) {
      float v = f16_to_f32(x[i]) * rms * f16_to_f32(gamma[i]);
      y[i] = f32_to_f16(v);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
  int warmup = 10;
  int userIters = 0;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--iters") && i+1 < argc) userIters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup") && i+1 < argc) warmup = atoi(argv[++i]);
  }

  printf("=== RMSNorm Benchmark: GPU vs NPU (SM8850) ===\n");
  printf("LPDDR5X-5300 peak: %.1f GB/s\n\n", kPeakBW);

  // ─── Device Info ──
  printf("--- Device Info ---\n");
  {
    GpuSession g;
    if (g.init(1, 4096, "kernels/rmsnorm.cl")) { g.printInfo(); g.cleanup(); }
    else { printf("  GPU: init failed\n"); g.cleanup(); }
  }
  {
    NpuSession n;
    if (n.init(1, 4096, NpuMode::ELEMENT_ADD)) {
      printf("  NPU: Hexagon V81, %u core(s)\n", n.coreCount);
      n.cleanup();
    } else { printf("  NPU: init failed\n"); n.cleanup(); }
  }
  printf("\n");

  // ─── Correctness Verification ──
  printf("--- Correctness Verification (batch=1, hidden=4096) ---\n");
  {
    const int B = 1, H = 4096;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);
    std::vector<uint16_t> in(B * H), gamma(H, f32_to_f16(1.0f)), ref(B * H);
    for (auto& v : in) v = f32_to_f16(dist(rng));
    cpuRmsNorm(in.data(), gamma.data(), ref.data(), B, H, 1e-6f);

    // GPU verify
    {
      GpuSession g;
      if (g.init(B, H, "kernels/rmsnorm.cl")) {
        auto r = g.run(2, 5);
        if (r.ok) {
          std::vector<uint16_t> out(B * H);
          g.readOutput(out.data(), out.size() * 2);
          float maxErr = 0;
          for (int i = 0; i < B * H; i++)
            maxErr = std::max(maxErr, fabsf(f16_to_f32(out[i]) - f16_to_f32(ref[i])));
          printf("  GPU FP16: max_err=%.6f %s\n", maxErr, maxErr < 0.01f ? "PASS" : "FAIL");
        }
        g.cleanup();
      }
    }
  }
  printf("\n");

  // ─── NPU FP16 RmsNorm Support Detection ──
  printf("--- NPU RmsNorm Support Detection ---\n");
  bool nativeOk = false, decomposedOk = false;
  {
    // Test Native
    {
      NpuSession n;
      if (n.init(1, 4096, NpuMode::NATIVE)) {
        auto r = n.run(2, 5);
        if (r.ok) { printf("  Native QNN_OP_RMS_NORM (FP16): SUPPORTED (%.1f us)\n", r.latency_us); nativeOk = true; }
        else printf("  Native QNN_OP_RMS_NORM (FP16): init OK but exec failed (%s)\n", r.err.c_str());
      } else {
        printf("  Native QNN_OP_RMS_NORM (FP16): NOT SUPPORTED\n");
      }
      n.cleanup();
    }

    // Test Decomposed
    {
      NpuSession n;
      if (n.init(1, 4096, NpuMode::DECOMPOSED)) {
        auto r = n.run(2, 5);
        if (r.ok) { printf("  Decomposed RmsNorm (FP16):     SUPPORTED (%.1f us)\n", r.latency_us); decomposedOk = true; }
        else printf("  Decomposed RmsNorm (FP16):     init OK but exec failed (%s)\n", r.err.c_str());
      } else {
        printf("  Decomposed RmsNorm (FP16):     NOT SUPPORTED\n");
      }
      n.cleanup();
    }

    printf("  ElementWiseAdd (UINT8):        SUPPORTED (baseline)\n");

    if (nativeOk || decomposedOk)
      printf("\n  >>> NPU FP16 RmsNorm available <<<\n");
    else
      printf("\n  >>> NPU cannot do FP16 RMSNorm, using UINT8 Add as baseline <<<\n");
  }
  printf("\n");

  // ─── Benchmark Matrix ──
  NpuMode npuRmsMode = nativeOk ? NpuMode::NATIVE : decomposedOk ? NpuMode::DECOMPOSED : NpuMode::NATIVE;
  bool hasNpuRms = nativeOk || decomposedOk;

  struct TC { int batch; int hidden; const char* label; };
  std::vector<TC> cases = {
    {1, 2048, "decode"},    {1, 3200, "decode"},    {1, 4096, "decode"},
    {16, 4096, "prefill-16"}, {64, 4096, "prefill-64"},
    {256, 4096, "pf-256"}, {512, 4096, "pf-512"}, {1024, 4096, "pf-1k"},
  };

  if (hasNpuRms) {
    const char* ml = nativeOk ? "NPU-RMS" : "NPU-Dec";
    printf("--- Benchmark: GPU FP16 vs %s FP16 vs NPU-Add UINT8 ---\n\n", ml);
    printf("%-10s %5s %5s | %9s %9s | %9s %9s | %9s %9s | %7s\n",
           "Scene", "batch", "hid", "GPU(us)", "GPU(GB/s)", ml, "BW(GB/s)", "Add(us)", "BW(GB/s)", "GPU/NPU");
    for (int i = 0; i < 110; i++) printf("-");
    printf("\n");
  } else {
    printf("--- Benchmark: GPU FP16 RMSNorm vs NPU UINT8 Add (baseline) ---\n\n");
    printf("%-10s %5s %5s | %9s %9s | %9s %9s | %7s\n",
           "Scene", "batch", "hid", "GPU(us)", "GPU(GB/s)", "NPU(us)", "NPU(GB/s)", "Ratio");
    for (int i = 0; i < 90; i++) printf("-");
    printf("\n");
  }

  for (auto& tc : cases) {
    int iters = userIters > 0 ? userIters : autoIters(tc.batch, tc.hidden, 20.0);

    // GPU
    BenchResult gpuR;
    { GpuSession g;
      if (g.init(tc.batch, tc.hidden, "kernels/rmsnorm.cl"))
        gpuR = g.run(warmup, iters);
      g.cleanup();
    }

    if (hasNpuRms) {
      // NPU RmsNorm FP16
      BenchResult npuR;
      { NpuSession n; n.quiet = true;
        if (n.init(tc.batch, tc.hidden, npuRmsMode))
          npuR = n.run(warmup, iters);
        n.cleanup();
      }

      // NPU Add UINT8
      BenchResult addR;
      { NpuSession n; n.quiet = true;
        if (n.init(tc.batch, tc.hidden, NpuMode::ELEMENT_ADD))
          addR = n.run(warmup, iters);
        n.cleanup();
      }

      printf("%-10s %5d %5d", tc.label, tc.batch, tc.hidden);
      if (gpuR.ok) printf(" | %9.1f %9.2f", gpuR.latency_us, gpuR.bw_gbps);
      else printf(" | %9s %9s", "FAIL", "-");
      if (npuR.ok) printf(" | %9.1f %9.2f", npuR.latency_us, npuR.bw_gbps);
      else printf(" | %9s %9s", "FAIL", "-");
      if (addR.ok) printf(" | %9.1f %9.2f", addR.latency_us, addR.bw_gbps);
      else printf(" | %9s %9s", "FAIL", "-");
      if (gpuR.ok && npuR.ok) printf(" | %5.2fx", npuR.latency_us / gpuR.latency_us);
      else printf(" | %7s", "-");
      printf("\n");
    } else {
      // Fallback: GPU vs NPU Add only
      BenchResult addR;
      { NpuSession n; n.quiet = true;
        if (n.init(tc.batch, tc.hidden, NpuMode::ELEMENT_ADD))
          addR = n.run(warmup, iters);
        n.cleanup();
      }

      printf("%-10s %5d %5d", tc.label, tc.batch, tc.hidden);
      if (gpuR.ok) printf(" | %9.1f %9.2f", gpuR.latency_us, gpuR.bw_gbps);
      else printf(" | %9s %9s", "FAIL", "-");
      if (addR.ok) printf(" | %9.1f %9.2f", addR.latency_us, addR.bw_gbps);
      else printf(" | %9s %9s", "FAIL", "-");
      if (gpuR.ok && addR.ok) printf(" | %5.2fx", addR.latency_us / gpuR.latency_us);
      else printf(" | %7s", "-");
      printf("\n");
    }
  }

  printf("\n--- Summary ---\n");
  if (nativeOk)
    printf("NPU supports Native RmsNorm (FP16) via QNN_OP_RMS_NORM\n");
  else if (decomposedOk)
    printf("NPU supports Decomposed RmsNorm (FP16) via element-wise ops\n");
  else
    printf("NPU does NOT support FP16 RmsNorm (neither native nor decomposed)\n");

  printf("GPU: Adreno 840 OpenCL FP16 RMSNorm\n");
  printf("Theoretical peak BW: %.1f GB/s\n", kPeakBW);

  return 0;
}
