//=============================================================================
// Unit test: QNN graph overhead analysis for custom ops
//
// Tests different graph configurations to isolate where the 9.4x overhead
// comes from when combining two custom ops (SyncWait + RmsNorm) in one graph.
//
// Configs:
//   A: Native QNN RmsNorm (1 op)         — baseline
//   B: Custom HVX RmsNorm (1 op)         — custom op baseline
//   C: SyncWait only (1 op, passthrough)  — SyncWait overhead
//   D: SyncWait + Custom RmsNorm (2 ops)  — the slow case
//=============================================================================

#include "common.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

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

static const QNN_INTERFACE_VER_TYPE* g_qnn = nullptr;
static void* g_libHandle = nullptr;

static void logCb(const char* fmt, QnnLog_Level_t level, uint64_t, va_list args) {
  if (level != QNN_LOG_LEVEL_ERROR) return;
  printf("[QNN-E] "); vprintf(fmt, args); printf("\n");
}

static bool ok(Qnn_ErrorHandle_t s, const char* w) {
  if (s != QNN_SUCCESS) { printf("[ERR] %s: %lu\n", w, (unsigned long)s); return false; }
  return true;
}

constexpr uint32_t kRank = 4;

static Qnn_Tensor_t makeFp16(const char* name, Qnn_TensorType_t type,
                              uint32_t* dims, uint32_t rank = kRank) {
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
                                uint32_t* dims, uint32_t rank = kRank) {
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

struct TestGraph {
  const char* name;
  Qnn_GraphHandle_t graph = nullptr;
  Qnn_Tensor_t execIn[2];
  Qnn_Tensor_t execOut[1];
  uint32_t numIn = 0;
  double finalize_us = 0;
};

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

static Qnn_GraphHandle_t createGraphHandle(Qnn_ContextHandle_t ctx, const char* name) {
  QnnHtpGraph_CustomConfig_t htpCfgs[3] = {
    QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT,
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

  Qnn_GraphHandle_t graph = nullptr;
  if (!ok(g_qnn->graphCreate(ctx, name, gcList, &graph), "graphCreate")) return nullptr;
  return graph;
}

int main(int argc, char* argv[]) {
  int hidden = 4096;
  int steps = 100;
  int warmup = 20;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--hidden") && i+1 < argc) hidden = atoi(argv[++i]);
    if (!strcmp(argv[i], "--steps") && i+1 < argc) steps = atoi(argv[++i]);
  }

  printf("=== QNN Graph Overhead Test ===\n");
  printf("hidden=%d, steps=%d, warmup=%d\n\n", hidden, steps, warmup);

  size_t tensor_bytes = (size_t)hidden * 2;
  size_t gamma_bytes = tensor_bytes;

  // Allocate ION buffers
  IonBuffer ionIn, ionOut, ionFlag, ionGamma;
  if (!allocIonBuffer(tensor_bytes, 0, ionIn) ||
      !allocIonBuffer(tensor_bytes, 0, ionOut) ||
      !allocIonBuffer(sizeof(uint32_t), 0, ionFlag) ||
      !allocIonBuffer(gamma_bytes, 0, ionGamma)) {
    printf("ION alloc failed\n"); return 1;
  }

  // Fill input + gamma
  uint16_t one = float_to_half(1.0f);
  uint16_t* gp = reinterpret_cast<uint16_t*>(ionGamma.ptr);
  uint16_t* ip = reinterpret_cast<uint16_t*>(ionIn.ptr);
  for (int i = 0; i < hidden; ++i) { gp[i] = one; ip[i] = float_to_half(0.5f); }
  *reinterpret_cast<uint32_t*>(ionFlag.ptr) = 1;  // flag=1 so SyncWait doesn't block

  // Init QNN
  g_libHandle = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
  if (!g_libHandle) { printf("dlopen failed: %s\n", dlerror()); return 1; }

  auto getProviders = reinterpret_cast<decltype(&QnnInterface_getProviders)>(
      dlsym(g_libHandle, "QnnInterface_getProviders"));
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

  // Register custom op packages
  bool hasSyncWait = true, hasRmsNorm = true, hasHeteroEdge = true;
  if (QNN_SUCCESS != g_qnn->backendRegisterOpPackage(
        backend, "./libQnnHtpSyncWaitOpPackage.so", "syncwaitInterfaceProvider", "CPU"))
    hasSyncWait = false;
  if (hasSyncWait && QNN_SUCCESS != g_qnn->backendRegisterOpPackage(
        backend, "./htp/libQnnHtpSyncWaitOpPackage.so", "syncwaitInterfaceProvider", "HTP"))
    hasSyncWait = false;

  if (QNN_SUCCESS != g_qnn->backendRegisterOpPackage(
        backend, "./libQnnHtpRmsNormOpPackage.so", "rmsnormInterfaceProvider", "CPU"))
    hasRmsNorm = false;
  if (hasRmsNorm && QNN_SUCCESS != g_qnn->backendRegisterOpPackage(
        backend, "./htp/libQnnHtpRmsNormOpPackage.so", "rmsnormInterfaceProvider", "HTP"))
    hasRmsNorm = false;

  // Combined package (SyncWait + RmsNorm in one .so, package name: heteroedge.HvxOpPackage)
  if (QNN_SUCCESS != g_qnn->backendRegisterOpPackage(
        backend, "./libQnnHtpHeteroEdgeOpPackage.so", "heteroedgeInterfaceProvider", "CPU"))
    hasHeteroEdge = false;
  if (hasHeteroEdge && QNN_SUCCESS != g_qnn->backendRegisterOpPackage(
        backend, "./htp/libQnnHtpHeteroEdgeOpPackage.so", "heteroedgeInterfaceProvider", "HTP"))
    hasHeteroEdge = false;

  printf("SyncWait op package  (separate): %s\n", hasSyncWait   ? "OK" : "FAILED");
  printf("RmsNorm  op package  (separate): %s\n", hasRmsNorm    ? "OK" : "FAILED");
  printf("HeteroEdge op package (combined): %s\n\n", hasHeteroEdge ? "OK" : "FAILED");

  // Register ION buffers
  uint32_t dimsIO[4] = {1, 1, 1, (uint32_t)hidden};
  uint32_t dimsFlag[4] = {1, 1, 1, 1};
  uint32_t dimsGamma[1] = {(uint32_t)hidden};

  struct RegMem { Qnn_MemHandle_t h = nullptr; };
  auto regBuf = [&](const IonBuffer& ion, const uint32_t* dims, uint32_t nd,
                     Qnn_DataType_t dt) -> Qnn_MemHandle_t {
    Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
    desc.memShape.numDim = nd;
    desc.memShape.dimSize = const_cast<uint32_t*>(dims);
    desc.dataType = dt;
    desc.memType = QNN_MEM_TYPE_ION;
    desc.ionInfo.fd = ion.fd;
    Qnn_MemHandle_t h = nullptr;
    g_qnn->memRegister(ctx, &desc, 1, &h);
    return h;
  };

  Qnn_MemHandle_t hIn    = regBuf(ionIn,    dimsIO,   4, QNN_DATATYPE_FLOAT_16);
  Qnn_MemHandle_t hOut   = regBuf(ionOut,   dimsIO,   4, QNN_DATATYPE_FLOAT_16);
  Qnn_MemHandle_t hFlag  = regBuf(ionFlag,  dimsFlag, 4, QNN_DATATYPE_UINT_32);

  // Build test graphs
  std::vector<TestGraph> tests;

  // ---- Config A: Native QNN RmsNorm (1 op) ----
  {
    TestGraph tg; tg.name = "A: Native RmsNorm (1 op)";
    tg.graph = createGraphHandle(ctx, "graph_a_native");
    if (tg.graph) {
      Qnn_Tensor_t input  = makeFp16("a_input",  QNN_TENSOR_TYPE_APP_WRITE, dimsIO);
      Qnn_Tensor_t output = makeFp16("a_output", QNN_TENSOR_TYPE_APP_READ,  dimsIO);
      Qnn_Tensor_t gamma  = makeFp16("a_gamma",  QNN_TENSOR_TYPE_STATIC, dimsGamma, 1);
      Qnn_Tensor_t beta   = makeFp16("a_beta",   QNN_TENSOR_TYPE_STATIC, dimsGamma, 1);
      gamma.v1.clientBuf.data = ionGamma.ptr;
      gamma.v1.clientBuf.dataSize = (uint32_t)gamma_bytes;
      IonBuffer ionBetaA; allocIonBuffer(gamma_bytes, 0, ionBetaA);
      beta.v1.clientBuf.data = ionBetaA.ptr;
      beta.v1.clientBuf.dataSize = (uint32_t)gamma_bytes;

      g_qnn->tensorCreateGraphTensor(tg.graph, &input);
      g_qnn->tensorCreateGraphTensor(tg.graph, &gamma);
      g_qnn->tensorCreateGraphTensor(tg.graph, &beta);
      g_qnn->tensorCreateGraphTensor(tg.graph, &output);

      // axes tensor
      uint32_t dimsAxes[1] = {1}; uint32_t axesData[1] = {3};
      Qnn_Tensor_t axes = QNN_TENSOR_INIT;
      axes.version = QNN_TENSOR_VERSION_1;
      axes.v1.name = "a_axes"; axes.v1.type = QNN_TENSOR_TYPE_STATIC;
      axes.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
      axes.v1.dataType = QNN_DATATYPE_UINT_32;
      axes.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
      axes.v1.rank = 1; axes.v1.dimensions = dimsAxes;
      axes.v1.memType = QNN_TENSORMEMTYPE_RAW;
      axes.v1.clientBuf.data = axesData; axes.v1.clientBuf.dataSize = 4;
      g_qnn->tensorCreateGraphTensor(tg.graph, &axes);

      Qnn_Param_t eps = QNN_PARAM_INIT;
      eps.paramType = QNN_PARAMTYPE_SCALAR; eps.name = QNN_OP_RMS_NORM_PARAM_EPSILON;
      eps.scalarParam.dataType = QNN_DATATYPE_FLOAT_32; eps.scalarParam.floatValue = 1e-6f;
      Qnn_Param_t axp = QNN_PARAM_INIT;
      axp.paramType = QNN_PARAMTYPE_TENSOR; axp.name = QNN_OP_RMS_NORM_PARAM_AXES;
      axp.tensorParam = axes;
      Qnn_Param_t params[] = {eps, axp};
      Qnn_Tensor_t opIn[] = {input, gamma, beta};
      Qnn_Tensor_t opOut[] = {output};
      Qnn_OpConfig_t op = QNN_OPCONFIG_INIT; op.version = QNN_OPCONFIG_VERSION_1;
      op.v1.name = "rmsnorm_a"; op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
      op.v1.typeName = QNN_OP_RMS_NORM;
      op.v1.numOfParams = 2; op.v1.params = params;
      op.v1.numOfInputs = 3; op.v1.inputTensors = opIn;
      op.v1.numOfOutputs = 1; op.v1.outputTensors = opOut;
      g_qnn->graphAddNode(tg.graph, op);

      double t0 = now_us();
      bool fin = ok(g_qnn->graphFinalize(tg.graph, nullptr, nullptr), "finalize A");
      tg.finalize_us = now_us() - t0;
      if (fin) {
        tg.execIn[0] = input;
        tg.execIn[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        tg.execIn[0].v1.memHandle = hIn;
        tg.execOut[0] = output;
        tg.execOut[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        tg.execOut[0].v1.memHandle = hOut;
        tg.numIn = 1;
        tests.push_back(tg);
      }
      freeIonBuffer(ionBetaA);
    }
  }

  // ---- Config B: Custom HVX RmsNorm (1 op) ----
  if (hasRmsNorm) {
    TestGraph tg; tg.name = "B: Custom RmsNorm (1 op)";
    tg.graph = createGraphHandle(ctx, "graph_b_custom_rms");
    if (tg.graph) {
      Qnn_Tensor_t input  = makeFp16("b_input",  QNN_TENSOR_TYPE_APP_WRITE, dimsIO);
      Qnn_Tensor_t output = makeFp16("b_output", QNN_TENSOR_TYPE_APP_READ,  dimsIO);
      Qnn_Tensor_t gamma  = makeFp16("b_gamma",  QNN_TENSOR_TYPE_STATIC, dimsGamma, 1);
      gamma.v1.clientBuf.data = ionGamma.ptr;
      gamma.v1.clientBuf.dataSize = (uint32_t)gamma_bytes;

      g_qnn->tensorCreateGraphTensor(tg.graph, &input);
      g_qnn->tensorCreateGraphTensor(tg.graph, &gamma);
      g_qnn->tensorCreateGraphTensor(tg.graph, &output);

      Qnn_Param_t eps = QNN_PARAM_INIT;
      eps.paramType = QNN_PARAMTYPE_SCALAR; eps.name = "epsilon";
      eps.scalarParam.dataType = QNN_DATATYPE_FLOAT_32; eps.scalarParam.floatValue = 1e-6f;
      Qnn_Param_t params[] = {eps};
      Qnn_Tensor_t opIn[] = {input, gamma};
      Qnn_Tensor_t opOut[] = {output};
      Qnn_OpConfig_t op = QNN_OPCONFIG_INIT; op.version = QNN_OPCONFIG_VERSION_1;
      op.v1.name = "rmsnorm_b"; op.v1.packageName = "rmsnorm.HvxOpPackage";
      op.v1.typeName = "RmsNorm";
      op.v1.numOfParams = 1; op.v1.params = params;
      op.v1.numOfInputs = 2; op.v1.inputTensors = opIn;
      op.v1.numOfOutputs = 1; op.v1.outputTensors = opOut;
      g_qnn->graphAddNode(tg.graph, op);

      double t0 = now_us();
      bool fin = ok(g_qnn->graphFinalize(tg.graph, nullptr, nullptr), "finalize B");
      tg.finalize_us = now_us() - t0;
      if (fin) {
        tg.execIn[0] = input;
        tg.execIn[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        tg.execIn[0].v1.memHandle = hIn;
        tg.execOut[0] = output;
        tg.execOut[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        tg.execOut[0].v1.memHandle = hOut;
        tg.numIn = 1;
        tests.push_back(tg);
      }
    }
  }

  // ---- Config C: SyncWait only (1 op, passthrough) ----
  if (hasSyncWait) {
    TestGraph tg; tg.name = "C: SyncWait only (1 op)";
    tg.graph = createGraphHandle(ctx, "graph_c_syncwait");
    if (tg.graph) {
      Qnn_Tensor_t data_in = makeFp16("c_data",   QNN_TENSOR_TYPE_APP_WRITE, dimsIO);
      Qnn_Tensor_t flag_in = makeUint32("c_flag",  QNN_TENSOR_TYPE_APP_WRITE, dimsFlag);
      Qnn_Tensor_t data_out= makeFp16("c_output",  QNN_TENSOR_TYPE_APP_READ,  dimsIO);

      g_qnn->tensorCreateGraphTensor(tg.graph, &data_in);
      g_qnn->tensorCreateGraphTensor(tg.graph, &flag_in);
      g_qnn->tensorCreateGraphTensor(tg.graph, &data_out);

      Qnn_Tensor_t swIn[]  = {data_in, flag_in};
      Qnn_Tensor_t swOut[] = {data_out};
      Qnn_OpConfig_t op = QNN_OPCONFIG_INIT; op.version = QNN_OPCONFIG_VERSION_1;
      op.v1.name = "syncwait_c"; op.v1.packageName = "syncwait.HvxOpPackage";
      op.v1.typeName = "SyncWait";
      op.v1.numOfParams = 0; op.v1.params = nullptr;
      op.v1.numOfInputs = 2; op.v1.inputTensors = swIn;
      op.v1.numOfOutputs = 1; op.v1.outputTensors = swOut;
      g_qnn->graphAddNode(tg.graph, op);

      double t0 = now_us();
      bool fin = ok(g_qnn->graphFinalize(tg.graph, nullptr, nullptr), "finalize C");
      tg.finalize_us = now_us() - t0;
      if (fin) {
        tg.execIn[0] = data_in;
        tg.execIn[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        tg.execIn[0].v1.memHandle = hIn;
        tg.execIn[1] = flag_in;
        tg.execIn[1].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        tg.execIn[1].v1.memHandle = hFlag;
        tg.execOut[0] = data_out;
        tg.execOut[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        tg.execOut[0].v1.memHandle = hOut;
        tg.numIn = 2;
        tests.push_back(tg);
      }
    }
  }

  // ---- Config D: SyncWait + Custom RmsNorm (2 ops) ----
  if (hasSyncWait && hasRmsNorm) {
    TestGraph tg; tg.name = "D: SyncWait + Custom RmsNorm (2 ops)";
    tg.graph = createGraphHandle(ctx, "graph_d_sync_rms");
    if (tg.graph) {
      Qnn_Tensor_t sw_in   = makeFp16("d_data",    QNN_TENSOR_TYPE_APP_WRITE, dimsIO);
      Qnn_Tensor_t sw_flag = makeUint32("d_flag",   QNN_TENSOR_TYPE_APP_WRITE, dimsFlag);
      Qnn_Tensor_t sw_out  = makeFp16("d_sw_out",   QNN_TENSOR_TYPE_NATIVE,    dimsIO);
      Qnn_Tensor_t output  = makeFp16("d_output",   QNN_TENSOR_TYPE_APP_READ,  dimsIO);
      Qnn_Tensor_t gamma   = makeFp16("d_gamma",    QNN_TENSOR_TYPE_STATIC, dimsGamma, 1);
      gamma.v1.clientBuf.data = ionGamma.ptr;
      gamma.v1.clientBuf.dataSize = (uint32_t)gamma_bytes;

      g_qnn->tensorCreateGraphTensor(tg.graph, &sw_in);
      g_qnn->tensorCreateGraphTensor(tg.graph, &sw_flag);
      g_qnn->tensorCreateGraphTensor(tg.graph, &sw_out);
      g_qnn->tensorCreateGraphTensor(tg.graph, &gamma);
      g_qnn->tensorCreateGraphTensor(tg.graph, &output);

      // SyncWait node
      {
        Qnn_Tensor_t in[] = {sw_in, sw_flag};
        Qnn_Tensor_t out[] = {sw_out};
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT; op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "syncwait_d"; op.v1.packageName = "syncwait.HvxOpPackage";
        op.v1.typeName = "SyncWait";
        op.v1.numOfParams = 0; op.v1.params = nullptr;
        op.v1.numOfInputs = 2; op.v1.inputTensors = in;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = out;
        g_qnn->graphAddNode(tg.graph, op);
      }
      // Custom RmsNorm node
      {
        Qnn_Param_t eps = QNN_PARAM_INIT;
        eps.paramType = QNN_PARAMTYPE_SCALAR; eps.name = "epsilon";
        eps.scalarParam.dataType = QNN_DATATYPE_FLOAT_32; eps.scalarParam.floatValue = 1e-6f;
        Qnn_Param_t params[] = {eps};
        Qnn_Tensor_t in[] = {sw_out, gamma};
        Qnn_Tensor_t out[] = {output};
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT; op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "rmsnorm_d"; op.v1.packageName = "rmsnorm.HvxOpPackage";
        op.v1.typeName = "RmsNorm";
        op.v1.numOfParams = 1; op.v1.params = params;
        op.v1.numOfInputs = 2; op.v1.inputTensors = in;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = out;
        g_qnn->graphAddNode(tg.graph, op);
      }

      double t0 = now_us();
      bool fin = ok(g_qnn->graphFinalize(tg.graph, nullptr, nullptr), "finalize D");
      tg.finalize_us = now_us() - t0;
      if (fin) {
        tg.execIn[0] = sw_in;
        tg.execIn[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        tg.execIn[0].v1.memHandle = hIn;
        tg.execIn[1] = sw_flag;
        tg.execIn[1].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        tg.execIn[1].v1.memHandle = hFlag;
        tg.execOut[0] = output;
        tg.execOut[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        tg.execOut[0].v1.memHandle = hOut;
        tg.numIn = 2;
        tests.push_back(tg);
      }
    }
  }

  // ---- Config E: 2x Custom RmsNorm (2 identical ops chained) ----
  if (hasRmsNorm) {
    TestGraph tg; tg.name = "E: 2x Custom RmsNorm (2 ops chained)";
    tg.graph = createGraphHandle(ctx, "graph_e_double_rms");
    if (tg.graph) {
      Qnn_Tensor_t input   = makeFp16("e_input",   QNN_TENSOR_TYPE_APP_WRITE, dimsIO);
      Qnn_Tensor_t mid     = makeFp16("e_mid",     QNN_TENSOR_TYPE_NATIVE,    dimsIO);
      Qnn_Tensor_t output  = makeFp16("e_output",  QNN_TENSOR_TYPE_APP_READ,  dimsIO);
      Qnn_Tensor_t gamma1  = makeFp16("e_gamma1",  QNN_TENSOR_TYPE_STATIC, dimsGamma, 1);
      Qnn_Tensor_t gamma2  = makeFp16("e_gamma2",  QNN_TENSOR_TYPE_STATIC, dimsGamma, 1);
      gamma1.v1.clientBuf.data = ionGamma.ptr;
      gamma1.v1.clientBuf.dataSize = (uint32_t)gamma_bytes;
      gamma2.v1.clientBuf.data = ionGamma.ptr;
      gamma2.v1.clientBuf.dataSize = (uint32_t)gamma_bytes;

      g_qnn->tensorCreateGraphTensor(tg.graph, &input);
      g_qnn->tensorCreateGraphTensor(tg.graph, &mid);
      g_qnn->tensorCreateGraphTensor(tg.graph, &gamma1);
      g_qnn->tensorCreateGraphTensor(tg.graph, &gamma2);
      g_qnn->tensorCreateGraphTensor(tg.graph, &output);

      // First RmsNorm
      {
        Qnn_Param_t eps = QNN_PARAM_INIT;
        eps.paramType = QNN_PARAMTYPE_SCALAR; eps.name = "epsilon";
        eps.scalarParam.dataType = QNN_DATATYPE_FLOAT_32; eps.scalarParam.floatValue = 1e-6f;
        Qnn_Param_t params[] = {eps};
        Qnn_Tensor_t in[] = {input, gamma1};
        Qnn_Tensor_t out[] = {mid};
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT; op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "rmsnorm_e1"; op.v1.packageName = "rmsnorm.HvxOpPackage";
        op.v1.typeName = "RmsNorm";
        op.v1.numOfParams = 1; op.v1.params = params;
        op.v1.numOfInputs = 2; op.v1.inputTensors = in;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = out;
        g_qnn->graphAddNode(tg.graph, op);
      }
      // Second RmsNorm
      {
        Qnn_Param_t eps = QNN_PARAM_INIT;
        eps.paramType = QNN_PARAMTYPE_SCALAR; eps.name = "epsilon";
        eps.scalarParam.dataType = QNN_DATATYPE_FLOAT_32; eps.scalarParam.floatValue = 1e-6f;
        Qnn_Param_t params[] = {eps};
        Qnn_Tensor_t in[] = {mid, gamma2};
        Qnn_Tensor_t out[] = {output};
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT; op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "rmsnorm_e2"; op.v1.packageName = "rmsnorm.HvxOpPackage";
        op.v1.typeName = "RmsNorm";
        op.v1.numOfParams = 1; op.v1.params = params;
        op.v1.numOfInputs = 2; op.v1.inputTensors = in;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = out;
        g_qnn->graphAddNode(tg.graph, op);
      }

      double t0 = now_us();
      bool fin = ok(g_qnn->graphFinalize(tg.graph, nullptr, nullptr), "finalize E");
      tg.finalize_us = now_us() - t0;
      if (fin) {
        tg.execIn[0] = input;
        tg.execIn[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        tg.execIn[0].v1.memHandle = hIn;
        tg.execOut[0] = output;
        tg.execOut[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        tg.execOut[0].v1.memHandle = hOut;
        tg.numIn = 1;
        tests.push_back(tg);
      }
    }
  }

  // ---- Config F: SyncWait + RmsNorm from COMBINED heteroedge package ----
  if (hasHeteroEdge) {
    TestGraph tg; tg.name = "F: SyncWait+RmsNorm combined package";
    tg.graph = createGraphHandle(ctx, "graph_f_combined");
    if (tg.graph) {
      Qnn_Tensor_t sw_in   = makeFp16("f_data",    QNN_TENSOR_TYPE_APP_WRITE, dimsIO);
      Qnn_Tensor_t sw_flag = makeUint32("f_flag",   QNN_TENSOR_TYPE_APP_WRITE, dimsFlag);
      Qnn_Tensor_t sw_out  = makeFp16("f_sw_out",   QNN_TENSOR_TYPE_NATIVE,    dimsIO);
      Qnn_Tensor_t output  = makeFp16("f_output",   QNN_TENSOR_TYPE_APP_READ,  dimsIO);
      Qnn_Tensor_t gamma   = makeFp16("f_gamma",    QNN_TENSOR_TYPE_STATIC, dimsGamma, 1);
      gamma.v1.clientBuf.data = ionGamma.ptr;
      gamma.v1.clientBuf.dataSize = (uint32_t)gamma_bytes;

      g_qnn->tensorCreateGraphTensor(tg.graph, &sw_in);
      g_qnn->tensorCreateGraphTensor(tg.graph, &sw_flag);
      g_qnn->tensorCreateGraphTensor(tg.graph, &sw_out);
      g_qnn->tensorCreateGraphTensor(tg.graph, &gamma);
      g_qnn->tensorCreateGraphTensor(tg.graph, &output);

      // SyncWait from combined package
      {
        Qnn_Tensor_t in[] = {sw_in, sw_flag};
        Qnn_Tensor_t out[] = {sw_out};
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT; op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "syncwait_f"; op.v1.packageName = "heteroedge.HvxOpPackage";
        op.v1.typeName = "SyncWait";
        op.v1.numOfParams = 0; op.v1.params = nullptr;
        op.v1.numOfInputs = 2; op.v1.inputTensors = in;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = out;
        g_qnn->graphAddNode(tg.graph, op);
      }
      // RmsNorm from combined package
      {
        Qnn_Param_t eps = QNN_PARAM_INIT;
        eps.paramType = QNN_PARAMTYPE_SCALAR; eps.name = "epsilon";
        eps.scalarParam.dataType = QNN_DATATYPE_FLOAT_32; eps.scalarParam.floatValue = 1e-6f;
        Qnn_Param_t params[] = {eps};
        Qnn_Tensor_t in[] = {sw_out, gamma};
        Qnn_Tensor_t out[] = {output};
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT; op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "rmsnorm_f"; op.v1.packageName = "heteroedge.HvxOpPackage";
        op.v1.typeName = "RmsNorm";
        op.v1.numOfParams = 1; op.v1.params = params;
        op.v1.numOfInputs = 2; op.v1.inputTensors = in;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = out;
        g_qnn->graphAddNode(tg.graph, op);
      }

      double t0 = now_us();
      bool fin = ok(g_qnn->graphFinalize(tg.graph, nullptr, nullptr), "finalize F");
      tg.finalize_us = now_us() - t0;
      if (fin) {
        tg.execIn[0] = sw_in;
        tg.execIn[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        tg.execIn[0].v1.memHandle = hIn;
        tg.execIn[1] = sw_flag;
        tg.execIn[1].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        tg.execIn[1].v1.memHandle = hFlag;
        tg.execOut[0] = output;
        tg.execOut[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        tg.execOut[0].v1.memHandle = hOut;
        tg.numIn = 2;
        tests.push_back(tg);
      }
    }
  }

  // ---- Config G: QNN DMA Copy Proof (SyncWait+RmsNorm, flag=0, 1 step) ----
  // Demonstrates that DSP cannot see GPU's live ION write via QNN tensor input.
  // When flag starts at 0 and is never written, DSP polls a QNN-internal TCM
  // copy (also 0) forever → times out after ~410ms.
  // This config runs only 1 step (no warmup) due to the long timeout.
  struct DmaCopyTest {
    Qnn_GraphHandle_t graph = nullptr;
    Qnn_Tensor_t execIn[2];
    Qnn_Tensor_t execOut[1];
  } dmaTest;

  if (hasHeteroEdge) {
    // Allocate a separate flag buffer that stays 0 (never written to 1)
    IonBuffer ionFlagZero;
    bool dmaOk = allocIonBuffer(sizeof(uint32_t), 0, ionFlagZero);
    if (dmaOk) {
      *reinterpret_cast<uint32_t*>(ionFlagZero.ptr) = 0;  // flag stays 0
      Qnn_MemHandle_t hFlagZero = regBuf(ionFlagZero, dimsFlag, 4, QNN_DATATYPE_UINT_32);

      dmaTest.graph = createGraphHandle(ctx, "graph_g_dma");
      if (dmaTest.graph) {
        Qnn_Tensor_t sw_in   = makeFp16("g_data",   QNN_TENSOR_TYPE_APP_WRITE, dimsIO);
        Qnn_Tensor_t sw_flag = makeUint32("g_flag",  QNN_TENSOR_TYPE_APP_WRITE, dimsFlag);
        Qnn_Tensor_t sw_out  = makeFp16("g_sw_out",  QNN_TENSOR_TYPE_NATIVE,    dimsIO);
        Qnn_Tensor_t output  = makeFp16("g_output",  QNN_TENSOR_TYPE_APP_READ,  dimsIO);
        Qnn_Tensor_t gamma   = makeFp16("g_gamma",   QNN_TENSOR_TYPE_STATIC, dimsGamma, 1);
        gamma.v1.clientBuf.data = ionGamma.ptr;
        gamma.v1.clientBuf.dataSize = (uint32_t)gamma_bytes;

        g_qnn->tensorCreateGraphTensor(dmaTest.graph, &sw_in);
        g_qnn->tensorCreateGraphTensor(dmaTest.graph, &sw_flag);
        g_qnn->tensorCreateGraphTensor(dmaTest.graph, &sw_out);
        g_qnn->tensorCreateGraphTensor(dmaTest.graph, &gamma);
        g_qnn->tensorCreateGraphTensor(dmaTest.graph, &output);

        {  // SyncWait
          Qnn_Tensor_t in[] = {sw_in, sw_flag};
          Qnn_Tensor_t out[] = {sw_out};
          Qnn_OpConfig_t op = QNN_OPCONFIG_INIT; op.version = QNN_OPCONFIG_VERSION_1;
          op.v1.name = "syncwait_g"; op.v1.packageName = "heteroedge.HvxOpPackage";
          op.v1.typeName = "SyncWait";
          op.v1.numOfParams = 0; op.v1.params = nullptr;
          op.v1.numOfInputs = 2; op.v1.inputTensors = in;
          op.v1.numOfOutputs = 1; op.v1.outputTensors = out;
          g_qnn->graphAddNode(dmaTest.graph, op);
        }
        {  // RmsNorm
          Qnn_Param_t eps = QNN_PARAM_INIT;
          eps.paramType = QNN_PARAMTYPE_SCALAR; eps.name = "epsilon";
          eps.scalarParam.dataType = QNN_DATATYPE_FLOAT_32; eps.scalarParam.floatValue = 1e-6f;
          Qnn_Param_t params[] = {eps};
          Qnn_Tensor_t in[] = {sw_out, gamma};
          Qnn_Tensor_t out[] = {output};
          Qnn_OpConfig_t op = QNN_OPCONFIG_INIT; op.version = QNN_OPCONFIG_VERSION_1;
          op.v1.name = "rmsnorm_g"; op.v1.packageName = "heteroedge.HvxOpPackage";
          op.v1.typeName = "RmsNorm";
          op.v1.numOfParams = 1; op.v1.params = params;
          op.v1.numOfInputs = 2; op.v1.inputTensors = in;
          op.v1.numOfOutputs = 1; op.v1.outputTensors = out;
          g_qnn->graphAddNode(dmaTest.graph, op);
        }

        if (ok(g_qnn->graphFinalize(dmaTest.graph, nullptr, nullptr), "finalize G")) {
          dmaTest.execIn[0] = sw_in;
          dmaTest.execIn[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
          dmaTest.execIn[0].v1.memHandle = hIn;
          dmaTest.execIn[1] = sw_flag;
          dmaTest.execIn[1].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
          dmaTest.execIn[1].v1.memHandle = hFlagZero;  // flag=0, never written
          dmaTest.execOut[0] = output;
          dmaTest.execOut[0].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
          dmaTest.execOut[0].v1.memHandle = hOut;
        } else {
          dmaTest.graph = nullptr;
        }
      }
      // Note: ionFlagZero and hFlagZero will be cleaned up with context
    }
  }

  // ---- Execute and measure ----
  printf("\n--- Finalize Times ---\n");
  for (auto& tg : tests)
    printf("  %-42s  %10.1f us\n", tg.name, tg.finalize_us);

  printf("\n--- Execution Times (graphExecute wall time, %d iters) ---\n", steps);
  printf("  %-42s  %8s %8s %8s %8s\n", "Config", "min", "p50", "avg", "max");

  for (auto& tg : tests) {
    // Warmup
    for (int i = 0; i < warmup; ++i)
      g_qnn->graphExecute(tg.graph, tg.execIn, tg.numIn, tg.execOut, 1, nullptr, nullptr);

    // Measure
    std::vector<double> times;
    for (int i = 0; i < steps; ++i) {
      double t0 = now_us();
      g_qnn->graphExecute(tg.graph, tg.execIn, tg.numIn, tg.execOut, 1, nullptr, nullptr);
      times.push_back(now_us() - t0);
    }

    Stats s = compute_stats(times);
    printf("  %-42s  %7.1f %8.1f %8.1f %8.1f\n", tg.name, s.min, s.p50, s.avg, s.max);
  }

  // Config G: QNN DMA copy proof (1 step only — each step takes ~410ms timeout)
  if (dmaTest.graph) {
    printf("\n--- Config G: QNN DMA Copy Proof (flag=0, 1 step, expect ~410ms) ---\n");
    printf("  Demonstrates: QNN copies input tensors to DSP-internal TCM at graphExecute start.\n");
    printf("  DSP polls TCM copy (value=0); GPU's ION write of 1 is never visible to DSP.\n");
    printf("  SyncWait DSP op times out after 10M × ~41ns ≈ 410ms.\n");
    printf("  (Compare: Config F with flag=1 pre-set → 327us — DMA copies 1, immediate exit)\n\n");
    double t0 = now_us();
    g_qnn->graphExecute(dmaTest.graph, dmaTest.execIn, 2, dmaTest.execOut, 1, nullptr, nullptr);
    double elapsed = now_us() - t0;
    printf("  G: SyncWait+RmsNorm (flag=0, QNN DMA copy)  %10.0f us  [TIMEOUT]\n", elapsed);
  }

  // Cleanup
  std::vector<Qnn_MemHandle_t> handles = {hIn, hOut, hFlag};
  g_qnn->memDeRegister(handles.data(), (uint32_t)handles.size());
  g_qnn->contextFree(ctx, nullptr);
  if (device && g_qnn->deviceFree) g_qnn->deviceFree(device);
  g_qnn->backendFree(backend);
  if (log && g_qnn->logFree) g_qnn->logFree(log);
  dlclose(g_libHandle);
  freeIonBuffer(ionIn); freeIonBuffer(ionOut); freeIonBuffer(ionFlag); freeIonBuffer(ionGamma);

  return 0;
}
