#include "npu_rmsnorm.h"

#include <dlfcn.h>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

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

namespace {

constexpr uint32_t kTensorRank = 4;

void*                        g_libHandle  = nullptr;
const QNN_INTERFACE_VER_TYPE* g_qnn       = nullptr;
Qnn_LogHandle_t              g_log        = nullptr;
Qnn_BackendHandle_t          g_backend    = nullptr;
Qnn_DeviceHandle_t           g_device     = nullptr;
Qnn_ContextHandle_t          g_context    = nullptr;
Qnn_GraphHandle_t            g_graph      = nullptr;
uint32_t                     g_coreCount  = 0;
NpuMode                      g_mode       = NpuMode::NATIVE;

IonBuffer g_ionInput, g_ionOutput, g_ionGamma, g_ionBeta;

struct RegMem { Qnn_MemHandle_t handle = nullptr; };
RegMem g_regInput, g_regOutput;

Qnn_Tensor_t g_execInputs[1];
Qnn_Tensor_t g_execOutputs[1];

uint32_t g_dimsIO[kTensorRank];        // [batch, 1, 1, hidden_dim]
uint32_t g_dimsGamma[kTensorRank];     // [1, 1, 1, hidden_dim] (rank-4, for decomposed)
uint32_t g_dimsGamma1D[1];            // [hidden_dim] (rank-1, for native RmsNorm)
uint32_t g_dimsScalar[kTensorRank];   // [1, 1, 1, 1]
uint32_t g_dimsMean[kTensorRank];     // [batch, 1, 1, 1]
uint32_t g_dimsAxes[1];              // [1]

int g_batch  = 0;
int g_hidden = 0;
size_t g_elem_bytes = 2;

void qnnLogCallback(const char* fmt, QnnLog_Level_t level,
                     uint64_t /*timestamp*/, va_list args) {
  const char* tag = (level == QNN_LOG_LEVEL_ERROR) ? "E" :
                    (level == QNN_LOG_LEVEL_WARN)  ? "W" : "I";
  printf("[QNN-%s] ", tag);
  vprintf(fmt, args);
  printf("\n");
}

bool check(Qnn_ErrorHandle_t s, const char* w) {
  if (s != QNN_SUCCESS) {
    printf("[NPU] %s failed: %lu\n", w, (unsigned long)s);
    return false;
  }
  return true;
}

static uint16_t float_to_half(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint16_t sign = (x >> 16) & 0x8000;
  int exp = ((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFF;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7C00;
  return sign | (exp << 10) | (mant >> 13);
}

bool registerBuffer(const IonBuffer& ion, const uint32_t* dims, uint32_t ndims,
                    Qnn_DataType_t dtype, RegMem& out) {
  Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
  desc.memShape.numDim  = ndims;
  desc.memShape.dimSize = const_cast<uint32_t*>(dims);
  desc.dataType         = dtype;
  desc.memType          = QNN_MEM_TYPE_ION;
  desc.ionInfo.fd       = ion.fd;
  if (QNN_SUCCESS != g_qnn->memRegister(g_context, &desc, 1, &out.handle)) {
    printf("[NPU] memRegister failed (fd=%d)\n", ion.fd);
    return false;
  }
  return true;
}

void deregisterAll() {
  if (!g_qnn || !g_qnn->memDeRegister) return;
  std::vector<Qnn_MemHandle_t> handles;
  if (g_regInput.handle)  handles.push_back(g_regInput.handle);
  if (g_regOutput.handle) handles.push_back(g_regOutput.handle);
  if (!handles.empty())
    g_qnn->memDeRegister(handles.data(), static_cast<uint32_t>(handles.size()));
  g_regInput.handle = g_regOutput.handle = nullptr;
}

void setHighPerformanceMode() {
  if (!g_qnn->deviceGetInfrastructure) return;
  QnnDevice_Infrastructure_t infra = nullptr;
  if (QNN_SUCCESS != g_qnn->deviceGetInfrastructure(&infra) || !infra) return;
  auto* htpInfra = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
  if (htpInfra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
  auto& perf = htpInfra->perfInfra;
  if (!perf.createPowerConfigId || !perf.setPowerConfig) return;

  uint32_t powerConfigId = 0;
  if (QNN_SUCCESS != perf.createPowerConfigId(0, 0, &powerConfigId)) return;

  QnnHtpPerfInfrastructure_PowerConfig_t dcvs = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  dcvs.dcvsV3Config.contextId              = powerConfigId;
  dcvs.dcvsV3Config.setDcvsEnable          = 1;
  dcvs.dcvsV3Config.dcvsEnable             = 0;
  dcvs.dcvsV3Config.powerMode              = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  dcvs.dcvsV3Config.setSleepDisable        = 1;
  dcvs.dcvsV3Config.sleepDisable           = 1;
  dcvs.dcvsV3Config.setBusParams           = 1;
  dcvs.dcvsV3Config.busVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.busVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.setCoreParams          = 1;
  dcvs.dcvsV3Config.coreVoltageCornerMin   = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.coreVoltageCornerTarget= DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.coreVoltageCornerMax   = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

  QnnHtpPerfInfrastructure_PowerConfig_t hmx = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  hmx.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2;
  hmx.hmxV2Config.hmxPickDefault         = 0;
  hmx.hmxV2Config.hmxVoltageCornerMin    = DCVS_EXP_VCORNER_MAX;
  hmx.hmxV2Config.hmxVoltageCornerTarget = DCVS_EXP_VCORNER_MAX;
  hmx.hmxV2Config.hmxVoltageCornerMax    = DCVS_EXP_VCORNER_MAX;
  hmx.hmxV2Config.hmxPerfMode            = QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH;

  const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&dcvs, &hmx, nullptr};
  perf.setPowerConfig(powerConfigId, cfgs);
}

uint32_t queryCoreCount() {
  if (!g_qnn->deviceGetPlatformInfo) return 0;
  const QnnDevice_PlatformInfo_t* info = nullptr;
  if (QNN_SUCCESS != g_qnn->deviceGetPlatformInfo(nullptr, &info) || !info) return 0;
  uint32_t cores = 0;
  if (info->version == QNN_DEVICE_PLATFORM_INFO_VERSION_1) {
    for (uint32_t i = 0; i < info->v1.numHwDevices; ++i) {
      auto& dev = info->v1.hwDevices[i];
      if (dev.version != QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1) continue;
      auto* ext = reinterpret_cast<QnnHtpDevice_DeviceInfoExtension_t*>(dev.v1.deviceInfoExtension);
      if (ext && ext->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) { cores = dev.v1.numCores; break; }
    }
  }
  if (g_qnn->deviceFreePlatformInfo) g_qnn->deviceFreePlatformInfo(nullptr, info);
  return cores;
}

Qnn_Tensor_t makeFp16Tensor(const char* name, Qnn_TensorType_t type,
                              uint32_t* dims, uint32_t rank = kTensorRank) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version      = QNN_TENSOR_VERSION_1;
  t.v1.id        = 0;
  t.v1.name      = name;
  t.v1.type      = type;
  t.v1.dataFormat     = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  t.v1.dataType       = QNN_DATATYPE_FLOAT_16;
  t.v1.quantizeParams.encodingDefinition   = QNN_DEFINITION_UNDEFINED;
  t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  t.v1.rank           = rank;
  t.v1.dimensions     = dims;
  t.v1.memType        = QNN_TENSORMEMTYPE_RAW;
  t.v1.clientBuf      = QNN_CLIENT_BUFFER_INIT;
  return t;
}

// Create a static UINT32 axes tensor and register it as a graph tensor.
// QNN requires tensor-type params to be registered via tensorCreateGraphTensor()
// before passing to graphAddNode(). See QnnModel.cpp in SDK for reference.
bool createAxesTensor(Qnn_Tensor_t& out, uint32_t* axes_data, uint32_t num_axes) {
  g_dimsAxes[0] = num_axes;
  out = QNN_TENSOR_INIT;
  out.version = QNN_TENSOR_VERSION_1;
  out.v1.name = "axes";
  out.v1.type = QNN_TENSOR_TYPE_STATIC;
  out.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  out.v1.dataType = QNN_DATATYPE_UINT_32;
  out.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  out.v1.rank = 1;
  out.v1.dimensions = g_dimsAxes;
  out.v1.memType = QNN_TENSORMEMTYPE_RAW;
  out.v1.clientBuf.data     = axes_data;
  out.v1.clientBuf.dataSize = num_axes * sizeof(uint32_t);
  return check(g_qnn->tensorCreateGraphTensor(g_graph, &out), "tensor axes");
}

// Build graph with native QNN_OP_RMS_NORM (FP16, 3 inputs, rank-1 gamma/beta)
bool buildNativeGraph() {
  Qnn_Tensor_t input  = makeFp16Tensor("input",  QNN_TENSOR_TYPE_APP_WRITE, g_dimsIO);
  Qnn_Tensor_t output = makeFp16Tensor("output", QNN_TENSOR_TYPE_APP_READ,  g_dimsIO);
  Qnn_Tensor_t gamma  = makeFp16Tensor("gamma",  QNN_TENSOR_TYPE_STATIC,    g_dimsGamma1D, 1);
  Qnn_Tensor_t beta   = makeFp16Tensor("beta",   QNN_TENSOR_TYPE_STATIC,    g_dimsGamma1D, 1);
  gamma.v1.clientBuf.data     = g_ionGamma.ptr;
  gamma.v1.clientBuf.dataSize = static_cast<uint32_t>(g_ionGamma.size);
  beta.v1.clientBuf.data      = g_ionBeta.ptr;
  beta.v1.clientBuf.dataSize  = static_cast<uint32_t>(g_ionBeta.size);

  if (!check(g_qnn->tensorCreateGraphTensor(g_graph, &input),  "tensor input") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &gamma),  "tensor gamma") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &beta),   "tensor beta") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &output), "tensor output"))
    return false;

  // Epsilon param (scalar)
  Qnn_Param_t eps_param = QNN_PARAM_INIT;
  eps_param.paramType    = QNN_PARAMTYPE_SCALAR;
  eps_param.name         = QNN_OP_RMS_NORM_PARAM_EPSILON;
  eps_param.scalarParam.dataType   = QNN_DATATYPE_FLOAT_32;
  eps_param.scalarParam.floatValue = 1e-6f;

  // Axes param (tensor) — normalize on channel dim
  uint32_t axes_data[] = {3};
  Qnn_Tensor_t axes_tensor;
  if (!createAxesTensor(axes_tensor, axes_data, 1))
    return false;

  Qnn_Param_t axes_param = QNN_PARAM_INIT;
  axes_param.paramType   = QNN_PARAMTYPE_TENSOR;
  axes_param.name        = QNN_OP_RMS_NORM_PARAM_AXES;
  axes_param.tensorParam = axes_tensor;

  Qnn_Param_t params[] = {eps_param, axes_param};
  Qnn_Tensor_t opIn[]  = {input, gamma, beta};
  Qnn_Tensor_t opOut[] = {output};

  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = "rmsnorm";
  op.v1.packageName  = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName     = QNN_OP_RMS_NORM;
  op.v1.numOfParams  = 2; op.v1.params = params;
  op.v1.numOfInputs  = 3; op.v1.inputTensors  = opIn;
  op.v1.numOfOutputs = 1; op.v1.outputTensors = opOut;

  if (!check(g_qnn->graphAddNode(g_graph, op), "graphAddNode(RmsNorm)"))
    return false;

  g_execInputs[0] = input;
  g_execOutputs[0] = output;

  return true;
}

// Build decomposed RmsNorm graph: Mul -> ReduceMean -> Add -> Rsqrt -> Mul -> Mul (FP16)
bool buildDecomposedGraph() {
  Qnn_Tensor_t input  = makeFp16Tensor("input",  QNN_TENSOR_TYPE_APP_WRITE, g_dimsIO);
  Qnn_Tensor_t output = makeFp16Tensor("output", QNN_TENSOR_TYPE_APP_READ,  g_dimsIO);
  Qnn_Tensor_t gamma  = makeFp16Tensor("gamma",  QNN_TENSOR_TYPE_STATIC,    g_dimsGamma);
  gamma.v1.clientBuf.data     = g_ionGamma.ptr;
  gamma.v1.clientBuf.dataSize = static_cast<uint32_t>(g_ionGamma.size);

  uint16_t eps_fp16 = float_to_half(1e-6f);
  Qnn_Tensor_t eps = makeFp16Tensor("eps_static", QNN_TENSOR_TYPE_STATIC, g_dimsScalar);
  eps.v1.clientBuf.data     = &eps_fp16;
  eps.v1.clientBuf.dataSize = sizeof(eps_fp16);

  Qnn_Tensor_t x_sq       = makeFp16Tensor("x_sq",       QNN_TENSOR_TYPE_NATIVE, g_dimsIO);
  Qnn_Tensor_t mean_sq    = makeFp16Tensor("mean_sq",    QNN_TENSOR_TYPE_NATIVE, g_dimsMean);
  Qnn_Tensor_t mean_eps   = makeFp16Tensor("mean_eps",   QNN_TENSOR_TYPE_NATIVE, g_dimsMean);
  Qnn_Tensor_t inv_rms    = makeFp16Tensor("inv_rms",    QNN_TENSOR_TYPE_NATIVE, g_dimsMean);
  Qnn_Tensor_t normalized = makeFp16Tensor("normalized", QNN_TENSOR_TYPE_NATIVE, g_dimsIO);

  if (!check(g_qnn->tensorCreateGraphTensor(g_graph, &input), "input") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &gamma), "gamma") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &eps), "eps") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &x_sq), "x_sq") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &mean_sq), "mean_sq") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &mean_eps), "mean_eps") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &inv_rms), "inv_rms") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &normalized), "normalized") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &output), "output"))
    return false;

  // Node 1: x_sq = Mul(input, input)
  { Qnn_Tensor_t i[] = {input, input}; Qnn_Tensor_t o[] = {x_sq};
    Qnn_OpConfig_t c = QNN_OPCONFIG_INIT; c.version = QNN_OPCONFIG_VERSION_1;
    c.v1.name = "mul_sq"; c.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    c.v1.typeName = QNN_OP_ELEMENT_WISE_MULTIPLY;
    c.v1.numOfInputs = 2; c.v1.inputTensors = i;
    c.v1.numOfOutputs = 1; c.v1.outputTensors = o;
    if (!check(g_qnn->graphAddNode(g_graph, c), "node mul_sq")) return false; }

  // Node 2: mean_sq = ReduceMean(x_sq, axes=[3], keep_dims=true)
  { uint32_t ax[] = {3};
    Qnn_Tensor_t ax_t;
    if (!createAxesTensor(ax_t, ax, 1)) return false;

    Qnn_Param_t p1 = QNN_PARAM_INIT;
    p1.paramType = QNN_PARAMTYPE_TENSOR; p1.name = QNN_OP_REDUCE_MEAN_PARAM_AXES;
    p1.tensorParam = ax_t;
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
    if (!check(g_qnn->graphAddNode(g_graph, c), "node reduce_mean")) return false; }

  // Node 3: mean_eps = Add(mean_sq, epsilon)
  { Qnn_Tensor_t i[] = {mean_sq, eps}; Qnn_Tensor_t o[] = {mean_eps};
    Qnn_OpConfig_t c = QNN_OPCONFIG_INIT; c.version = QNN_OPCONFIG_VERSION_1;
    c.v1.name = "add_eps"; c.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    c.v1.typeName = QNN_OP_ELEMENT_WISE_ADD;
    c.v1.numOfInputs = 2; c.v1.inputTensors = i;
    c.v1.numOfOutputs = 1; c.v1.outputTensors = o;
    if (!check(g_qnn->graphAddNode(g_graph, c), "node add_eps")) return false; }

  // Node 4: inv_rms = Rsqrt(mean_eps)
  { Qnn_Tensor_t i[] = {mean_eps}; Qnn_Tensor_t o[] = {inv_rms};
    Qnn_OpConfig_t c = QNN_OPCONFIG_INIT; c.version = QNN_OPCONFIG_VERSION_1;
    c.v1.name = "rsqrt"; c.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    c.v1.typeName = QNN_OP_ELEMENT_WISE_RSQRT;
    c.v1.numOfInputs = 1; c.v1.inputTensors = i;
    c.v1.numOfOutputs = 1; c.v1.outputTensors = o;
    if (!check(g_qnn->graphAddNode(g_graph, c), "node rsqrt")) return false; }

  // Node 5: normalized = Mul(input, inv_rms)
  { Qnn_Tensor_t i[] = {input, inv_rms}; Qnn_Tensor_t o[] = {normalized};
    Qnn_OpConfig_t c = QNN_OPCONFIG_INIT; c.version = QNN_OPCONFIG_VERSION_1;
    c.v1.name = "mul_norm"; c.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    c.v1.typeName = QNN_OP_ELEMENT_WISE_MULTIPLY;
    c.v1.numOfInputs = 2; c.v1.inputTensors = i;
    c.v1.numOfOutputs = 1; c.v1.outputTensors = o;
    if (!check(g_qnn->graphAddNode(g_graph, c), "node mul_norm")) return false; }

  // Node 6: output = Mul(normalized, gamma)
  { Qnn_Tensor_t i[] = {normalized, gamma}; Qnn_Tensor_t o[] = {output};
    Qnn_OpConfig_t c = QNN_OPCONFIG_INIT; c.version = QNN_OPCONFIG_VERSION_1;
    c.v1.name = "mul_gamma"; c.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    c.v1.typeName = QNN_OP_ELEMENT_WISE_MULTIPLY;
    c.v1.numOfInputs = 2; c.v1.inputTensors = i;
    c.v1.numOfOutputs = 1; c.v1.outputTensors = o;
    if (!check(g_qnn->graphAddNode(g_graph, c), "node mul_gamma")) return false; }

  g_execInputs[0] = input;
  g_execOutputs[0] = output;

  return true;
}

// Create graph, build op nodes, and finalize
bool buildGraph(NpuMode mode, const char* gname) {
  QnnHtpGraph_CustomConfig_t htpCfgs[4] = {QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT,
                                             QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT,
                                             QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT,
                                             QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT};
  uint32_t cfgCount = 0;

  htpCfgs[cfgCount].option    = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
  htpCfgs[cfgCount].precision = QNN_PRECISION_FLOAT16;
  cfgCount++;
  if (g_coreCount > 0) {
    htpCfgs[cfgCount].option   = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_CORES;
    htpCfgs[cfgCount].numCores = g_coreCount;
    cfgCount++;
  }
  htpCfgs[cfgCount].option        = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
  htpCfgs[cfgCount].numHvxThreads = 8;
  cfgCount++;

  QnnGraph_Config_t graphCfg = QNN_GRAPH_CONFIG_INIT;
  graphCfg.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graphCfg.customConfig = htpCfgs;
  const QnnGraph_Config_t* graphCfgList[] = {&graphCfg, nullptr};

  if (!check(g_qnn->graphCreate(g_context, gname, graphCfgList, &g_graph), "graphCreate"))
    return false;

  bool ok = false;
  switch (mode) {
    case NpuMode::NATIVE:      ok = buildNativeGraph(); break;
    case NpuMode::DECOMPOSED:  ok = buildDecomposedGraph(); break;
  }
  if (!ok) { g_graph = nullptr; return false; }

  if (!check(g_qnn->graphFinalize(g_graph, nullptr, nullptr), "graphFinalize")) {
    g_graph = nullptr; return false;
  }
  return true;
}

}  // namespace

void npu_rmsnorm_print_info() {
  const char* mode_str = "Unknown";
  switch (g_mode) {
    case NpuMode::NATIVE:      mode_str = "Native RmsNorm (FP16)"; break;
    case NpuMode::DECOMPOSED:  mode_str = "Decomposed (FP16)"; break;
  }
  printf("  NPU: Hexagon V81, %u core(s), %s\n", g_coreCount, mode_str);
}

bool npu_rmsnorm_init(const RMSNormConfig& config, NpuMode mode) {
  g_batch  = config.batch_size;
  g_hidden = config.hidden_dim;
  g_mode   = mode;
  g_elem_bytes = 2;  // FP16

  size_t tensor_bytes = (size_t)g_batch * g_hidden * g_elem_bytes;
  size_t gamma_bytes  = (size_t)g_hidden * g_elem_bytes;

  // Set tensor dimensions
  g_dimsIO[0] = g_batch; g_dimsIO[1] = 1; g_dimsIO[2] = 1; g_dimsIO[3] = g_hidden;
  g_dimsGamma[0] = 1; g_dimsGamma[1] = 1; g_dimsGamma[2] = 1; g_dimsGamma[3] = g_hidden;
  g_dimsGamma1D[0] = g_hidden;
  g_dimsScalar[0] = 1; g_dimsScalar[1] = 1; g_dimsScalar[2] = 1; g_dimsScalar[3] = 1;
  g_dimsMean[0] = g_batch; g_dimsMean[1] = 1; g_dimsMean[2] = 1; g_dimsMean[3] = 1;

  // Allocate ION buffers
  if (!allocIonBuffer(tensor_bytes, 1, g_ionInput) ||
      !allocIonBuffer(tensor_bytes, 0, g_ionOutput) ||
      !allocIonBuffer(gamma_bytes, 0, g_ionGamma) ||
      !allocIonBuffer(gamma_bytes, 0, g_ionBeta)) {
    printf("[NPU] Failed to alloc ION buffers\n"); return false;
  }

  // Init input with random FP16 (same seed as GPU)
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.1f, 1.0f);
  uint16_t* ptr = reinterpret_cast<uint16_t*>(g_ionInput.ptr);
  for (int i = 0; i < g_batch * g_hidden; ++i)
    ptr[i] = float_to_half(dist(rng));
  // Init gamma = 1.0
  uint16_t one = float_to_half(1.0f);
  uint16_t* gp = reinterpret_cast<uint16_t*>(g_ionGamma.ptr);
  for (int i = 0; i < g_hidden; ++i) gp[i] = one;

  // dlopen QNN backend
  g_libHandle = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
  if (!g_libHandle) { printf("[NPU] dlopen failed: %s\n", dlerror()); return false; }

  using GetProvidersFn = decltype(&QnnInterface_getProviders);
  auto getProviders = reinterpret_cast<GetProvidersFn>(
      dlsym(g_libHandle, "QnnInterface_getProviders"));
  if (!getProviders) { printf("[NPU] getProviders not found\n"); return false; }

  const QnnInterface_t** providers = nullptr;
  uint32_t numProviders = 0;
  if (getProviders(&providers, &numProviders) != QNN_SUCCESS || numProviders == 0) {
    printf("[NPU] No providers\n"); return false;
  }

  const QnnInterface_t* best = nullptr;
  for (uint32_t i = 0; i < numProviders; ++i) {
    if (!providers[i]) continue;
    if (providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR) {
      if (!best || providers[i]->apiVersion.coreApiVersion.minor >
                       best->apiVersion.coreApiVersion.minor)
        best = providers[i];
    }
  }
  if (!best) best = providers[0];
  g_qnn = &best->QNN_INTERFACE_VER_NAME;

  if (g_qnn->logCreate)
    g_qnn->logCreate(qnnLogCallback, QNN_LOG_LEVEL_ERROR, &g_log);

  if (!check(g_qnn->backendCreate(g_log, nullptr, &g_backend), "backendCreate"))
    return false;
  if (g_qnn->deviceCreate) {
    auto s = g_qnn->deviceCreate(nullptr, nullptr, &g_device);
    if (s != QNN_SUCCESS && s != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
      printf("[NPU] deviceCreate failed: %lu\n", (unsigned long)s); return false;
    }
  }

  setHighPerformanceMode();

  if (!check(g_qnn->contextCreate(g_backend, g_device, nullptr, &g_context), "contextCreate"))
    return false;

  g_coreCount = queryCoreCount();

  // Register ION buffers
  if (!registerBuffer(g_ionInput,  g_dimsIO, kTensorRank, QNN_DATATYPE_FLOAT_16, g_regInput) ||
      !registerBuffer(g_ionOutput, g_dimsIO, kTensorRank, QNN_DATATYPE_FLOAT_16, g_regOutput))
    return false;

  // Build graph
  if (!buildGraph(mode, "rmsnorm_graph"))
    return false;

  // Bind registered memory handles
  g_execInputs[0].v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  g_execInputs[0].v1.memHandle = g_regInput.handle;
  g_execOutputs[0].v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  g_execOutputs[0].v1.memHandle = g_regOutput.handle;

  return true;
}

RMSNormResult npu_rmsnorm_run(const RMSNormConfig& config, int num_warmup, int num_iters) {
  RMSNormResult res;
  res.num_iterations = num_iters;

  for (int i = 0; i < num_warmup; ++i) {
    if (!check(g_qnn->graphExecute(g_graph, g_execInputs, 1,
                                   g_execOutputs, 1, nullptr, nullptr), "warmup")) {
      res.error = "warmup failed"; return res;
    }
  }

  double t0 = now_seconds();
  for (int i = 0; i < num_iters; ++i) {
    if (!check(g_qnn->graphExecute(g_graph, g_execInputs, 1,
                                   g_execOutputs, 1, nullptr, nullptr), "exec")) {
      res.error = "exec failed at iter " + std::to_string(i); return res;
    }
  }
  double t1 = now_seconds();

  double elapsed = t1 - t0;
  // RMSNorm: read input + write output + read gamma
  double bytes_per_call = (double)g_batch * g_hidden * g_elem_bytes * 2.0
                        + (double)g_hidden * g_elem_bytes;
  double total_bytes = bytes_per_call * num_iters;

  res.latency_us     = (elapsed / num_iters) * 1e6;
  res.bandwidth_gbps = (total_bytes / (1024.0*1024.0*1024.0)) / elapsed;
  res.success = true;
  return res;
}

bool npu_rmsnorm_read_output(void* dst, size_t bytes) {
  if (!g_ionOutput.ptr) return false;
  memcpy(dst, g_ionOutput.ptr, bytes);
  return true;
}

void npu_rmsnorm_cleanup() {
  deregisterAll();
  if (g_qnn && g_context) g_qnn->contextFree(g_context, nullptr);
  if (g_qnn && g_device && g_qnn->deviceFree) g_qnn->deviceFree(g_device);
  if (g_qnn && g_backend) g_qnn->backendFree(g_backend);
  if (g_qnn && g_log && g_qnn->logFree) g_qnn->logFree(g_log);
  if (g_libHandle) dlclose(g_libHandle);
  g_context = nullptr; g_device = nullptr; g_backend = nullptr;
  g_graph = nullptr; g_qnn = nullptr; g_libHandle = nullptr; g_log = nullptr;

  freeIonBuffer(g_ionInput);
  freeIonBuffer(g_ionOutput);
  freeIonBuffer(g_ionGamma);
  freeIonBuffer(g_ionBeta);
}
