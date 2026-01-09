#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/time.h>
#include <math.h>

// QNN SDK 头文件
#include "QnnInterface.h"
#include "QnnTypes.h"
#include "QnnMem.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnOpDef.h"
#include "QnnBackend.h"
#include "QnnOpPackage.h"
#include "System/QnnSystemDlc.h"
#include "System/QnnSystemInterface.h"
#include "System/QnnSystemContext.h"

// ============================================================================
// GraphInfo结构体定义（与QNN工具生成的模型库一致）
// ============================================================================
typedef struct {
    Qnn_GraphHandle_t graph;
    char* graphName;
    Qnn_Tensor_t* inputTensors;
    uint32_t numInputTensors;
    Qnn_Tensor_t* outputTensors;
    uint32_t numOutputTensors;
} GraphInfo_t;
typedef GraphInfo_t* GraphInfoPtr_t;

// ============================================================================
// 资源管理结构体
// ============================================================================
typedef struct {
    // 库句柄
    void* rpc_lib;
    void* qnn_backend_lib;
    void* model_lib;

    // 共享内存
    void* input_mem;
    void* output_mem;
    int input_heapid;
    int output_heapid;

    // QNN句柄
    Qnn_LogHandle_t logHandle;
    Qnn_BackendHandle_t backendHandle;
    Qnn_DeviceHandle_t deviceHandle;
    Qnn_ContextHandle_t contextHandle;
    Qnn_GraphHandle_t graphHandle;

    // Graph信息（从composeGraphs返回）
    void** graphsInfo;
    uint32_t numGraphsInfo;
    GraphInfo_t* graphInfo;

    // Tensor配置（从graphInfo复制）
    Qnn_Tensor_t inputTensor;
    Qnn_Tensor_t outputTensor;

    // QNN接口实现
    QNN_INTERFACE_VER_TYPE* qnnInterfaceImpl;
} Resources;

// ============================================================================
// 工具函数
// ============================================================================

// 获取高精度时间（秒）
static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// 深度复制QNN tensor信息（参考QNN SampleApp实现）
static bool deepCopyQnnTensorInfo(Qnn_Tensor_t* dst, const Qnn_Tensor_t* src) {
    if (!dst || !src) {
        printf("错误: deepCopyQnnTensorInfo收到NULL指针\n");
        return false;
    }

    // 设置tensor版本
    dst->version = src->version;

    // 复制tensor名称
    if (src->v2.name) {
        size_t name_len = strlen(src->v2.name);
        dst->v2.name = (char*)malloc(name_len + 1);
        if (!dst->v2.name) {
            printf("错误: 无法分配tensor名称内存\n");
            return false;
        }
        memcpy((void*)dst->v2.name, (const void*)src->v2.name, name_len + 1);
    } else {
        dst->v2.name = NULL;
    }

    // 复制基本属性
    dst->v2.id = src->v2.id;
    dst->v2.type = src->v2.type;
    dst->v2.dataFormat = src->v2.dataFormat;
    dst->v2.dataType = src->v2.dataType;
    dst->v2.rank = src->v2.rank;

    // 复制量化参数
    dst->v2.quantizeParams = src->v2.quantizeParams;

    // 复制维度（分配新内存并复制值）
    if (src->v2.rank > 0 && src->v2.dimensions) {
        dst->v2.dimensions = (uint32_t*)malloc(src->v2.rank * sizeof(uint32_t));
        if (!dst->v2.dimensions) {
            printf("错误: 无法分配tensor维度内存\n");
            if (dst->v2.name) free((void*)dst->v2.name);
            return false;
        }
        memcpy(dst->v2.dimensions, src->v2.dimensions, src->v2.rank * sizeof(uint32_t));
    } else {
        dst->v2.dimensions = NULL;
    }

    // 复制isDynamicDimensions
    if (src->v2.isDynamicDimensions && src->v2.rank > 0) {
        dst->v2.isDynamicDimensions = (uint8_t*)malloc(src->v2.rank * sizeof(uint8_t));
        if (!dst->v2.isDynamicDimensions) {
            printf("错误: 无法分配isDynamicDimensions内存\n");
            if (dst->v2.name) free((void*)dst->v2.name);
            if (dst->v2.dimensions) free((void*)dst->v2.dimensions);
            return false;
        }
        memcpy(dst->v2.isDynamicDimensions, src->v2.isDynamicDimensions,
               src->v2.rank * sizeof(uint8_t));
    } else {
        dst->v2.isDynamicDimensions = NULL;
    }

    // 复制sparseParams
    dst->v2.sparseParams = src->v2.sparseParams;

    // 内存类型和buffer在调用后设置
    dst->v2.memType = QNN_TENSORMEMTYPE_RAW;
    dst->v2.clientBuf.data = NULL;
    dst->v2.clientBuf.dataSize = 0;
    dst->v2.memHandle = NULL;

    return true;
}

// ============================================================================
// 资源清理函数
// ============================================================================
static void cleanup(Resources* res) {
    if (!res) return;

    // 使用局部变量保存需要清理的资源，避免在清理过程中访问已释放的内存
    void* rpc_lib = res->rpc_lib;
    void* qnn_backend_lib = res->qnn_backend_lib;
    void* model_lib = res->model_lib;
    void* input_mem = res->input_mem;
    void* output_mem = res->output_mem;
    void** graphsInfo = res->graphsInfo;
    uint32_t numGraphsInfo = res->numGraphsInfo;
    QNN_INTERFACE_VER_TYPE* qnnInterfaceImpl = res->qnnInterfaceImpl;
    Qnn_ContextHandle_t contextHandle = res->contextHandle;
    Qnn_DeviceHandle_t deviceHandle = res->deviceHandle;
    Qnn_BackendHandle_t backendHandle = res->backendHandle;
    Qnn_LogHandle_t logHandle = res->logHandle;

    typedef void (*rpcmem_free_fn)(void*);
    rpcmem_free_fn rpcmem_free = NULL;

    // 清理共享内存
    if (rpc_lib) {
        rpcmem_free = (rpcmem_free_fn)dlsym(rpc_lib, "rpcmem_free");
        if (rpcmem_free) {
            if (input_mem) rpcmem_free(input_mem);
            if (output_mem) rpcmem_free(output_mem);
        }
        dlclose(rpc_lib);
        res->rpc_lib = NULL;
    }

    // 释放deepCopyQnnTensorInfo分配的内存（安全释放，检查指针有效性）
    if (res->inputTensor.v2.name) {
        free((void*)res->inputTensor.v2.name);
        res->inputTensor.v2.name = NULL;
    }
    if (res->inputTensor.v2.dimensions) {
        free(res->inputTensor.v2.dimensions);
        res->inputTensor.v2.dimensions = NULL;
    }
    if (res->inputTensor.v2.isDynamicDimensions) {
        free(res->inputTensor.v2.isDynamicDimensions);
        res->inputTensor.v2.isDynamicDimensions = NULL;
    }
    res->inputTensor.v2.clientBuf.data = NULL;
    res->inputTensor.v2.clientBuf.dataSize = 0;

    if (res->outputTensor.v2.name) {
        free((void*)res->outputTensor.v2.name);
        res->outputTensor.v2.name = NULL;
    }
    if (res->outputTensor.v2.dimensions) {
        free(res->outputTensor.v2.dimensions);
        res->outputTensor.v2.dimensions = NULL;
    }
    if (res->outputTensor.v2.isDynamicDimensions) {
        free(res->outputTensor.v2.isDynamicDimensions);
        res->outputTensor.v2.isDynamicDimensions = NULL;
    }
    res->outputTensor.v2.clientBuf.data = NULL;
    res->outputTensor.v2.clientBuf.dataSize = 0;

    // 清理graphsInfo（在contextFree之前，避免contextFree后访问无效内存）
    // 安全处理：检查model_lib是否仍然有效
    // 注意：如果graphFinalize失败，graphsInfo可能已经无效，所以这里要小心处理
    if (model_lib && graphsInfo && numGraphsInfo > 0) {
        // 先保存函数指针，避免在dlclose后访问
        typedef int (*FreeGraphsInfoFn_t)(void**, uint32_t);
        FreeGraphsInfoFn_t freeGraphsInfo = NULL;

        // 尝试获取freeGraphsInfo函数，但如果失败也不影响清理
        freeGraphsInfo = (FreeGraphsInfoFn_t)dlsym(model_lib, "ElementwiseAdd_freeGraphsInfo");
        if (freeGraphsInfo) {
            freeGraphsInfo(graphsInfo, numGraphsInfo);
        }
        res->graphsInfo = NULL;
        res->numGraphsInfo = 0;
        res->graphInfo = NULL;
    }

    // 清理QNN资源（按照sample_app的顺序：context -> device -> backend -> log）
    // 注意：如果graphFinalize失败，某些资源可能已经无效，所以要小心处理
    // 简化处理：只清理我们确定有效的资源，避免阻塞操作
    if (qnnInterfaceImpl && qnn_backend_lib) {
        // 只有在backend库仍然有效时才尝试清理QNN资源
        // 使用简化的清理顺序，避免可能的阻塞
        if (contextHandle && qnnInterfaceImpl->contextFree) {
            qnnInterfaceImpl->contextFree(contextHandle, NULL);
            res->contextHandle = NULL;
        }
        if (deviceHandle && qnnInterfaceImpl->deviceFree) {
            qnnInterfaceImpl->deviceFree(deviceHandle);
            res->deviceHandle = NULL;
        }
        if (backendHandle && qnnInterfaceImpl->backendFree) {
            qnnInterfaceImpl->backendFree(backendHandle);
            res->backendHandle = NULL;
        }
        if (logHandle && qnnInterfaceImpl->logFree) {
            qnnInterfaceImpl->logFree(logHandle);
            res->logHandle = NULL;
        }
        res->qnnInterfaceImpl = NULL;
    }

    // 清理库（最后清理）
    if (model_lib) {
        dlclose(model_lib);
        res->model_lib = NULL;
    }
    if (qnn_backend_lib) {
        dlclose(qnn_backend_lib);
        res->qnn_backend_lib = NULL;
    }
}

// ============================================================================
// 库加载函数
// ============================================================================

// 加载共享内存库
static void* load_rpc_lib() {
    const char* paths[] = {
        "libcdsprpc.so",
        "/vendor/lib64/libcdsprpc.so",
        "/system/lib64/libcdsprpc.so",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        printf("尝试加载: %s\n", paths[i]);
        fflush(stdout);
        void* lib = dlopen(paths[i], RTLD_LAZY);
        if (lib) {
            printf("✓ 加载共享内存库: %s\n", paths[i]);
            fflush(stdout);
            return lib;
        } else {
            const char* err = dlerror();
            if (err) {
                printf("  失败: %s\n", err);
                fflush(stdout);
            }
        }
    }
    return NULL;
}

// 分配共享内存（优先使用heapid=25，这是UMA验证中最优的）
static void* alloc_shared_mem(void* rpc_lib, size_t size, int* heapid) {
    typedef void* (*rpcmem_alloc_fn)(int, int, int);
    rpcmem_alloc_fn rpcmem_alloc = (rpcmem_alloc_fn)dlsym(rpc_lib, "rpcmem_alloc");
    if (!rpcmem_alloc) {
        printf("错误: 无法找到 rpcmem_alloc\n");
        fflush(stdout);
        return NULL;
    }

    // 优先使用heapid=25（UMA验证中最优），然后尝试其他heapid
    int heapids[] = {25, 0, 1, 2, 13, 14, 26, 27, 28, 22, 23, 24};
    for (int i = 0; i < sizeof(heapids)/sizeof(heapids[0]); i++) {
        printf("  尝试 heapid=%d...\n", heapids[i]);
        fflush(stdout);
        void* mem = rpcmem_alloc(heapids[i], 0, size);
        if (mem) {
            *heapid = heapids[i];
            printf("  ✓ 成功 (heapid=%d)\n", heapids[i]);
            fflush(stdout);
            return mem;
        }
    }
    printf("错误: 所有heapid都失败\n");
    fflush(stdout);
    return NULL;
}

// 加载 QNN backend
static void* load_qnn_backend() {
    const char* paths[] = {
        "/vendor/lib64/libQnnHtp.so",
        "/vendor/lib64/libQnnHtpStub.so",
        "/system/lib64/libQnnHtp.so",
        "libQnnHtp.so",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        void* lib = dlopen(paths[i], RTLD_LAZY);
        if (lib) {
            printf("✓ 加载 QNN HTP backend: %s\n", paths[i]);
            return lib;
        }
    }
    return NULL;
}

// ============================================================================
// QNN初始化函数（按照sample_app的标准流程）
// ============================================================================

// 步骤1: 初始化QNN Backend
static Qnn_ErrorHandle_t initializeQnnBackend(Resources* res) {
    printf("\n=== 步骤 1: 初始化 QNN Backend ===\n");
    fflush(stdout);

    // 加载QNN backend库
    printf("加载QNN backend库...\n");
    fflush(stdout);
    res->qnn_backend_lib = load_qnn_backend();
    fflush(stdout);
    if (!res->qnn_backend_lib) {
        printf("错误: 无法加载 QNN backend 库\n");
        return (Qnn_ErrorHandle_t)1;
    }

    // 获取QNN接口提供者
    typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t***, uint32_t*);
    QnnInterfaceGetProvidersFn_t getInterfaceProviders =
        (QnnInterfaceGetProvidersFn_t)dlsym(res->qnn_backend_lib, "QnnInterface_getProviders");
    if (!getInterfaceProviders) {
        printf("错误: 无法找到 QnnInterface_getProviders\n");
        return (Qnn_ErrorHandle_t)1;
    }

    const QnnInterface_t** interfaceProviders = NULL;
    uint32_t numProviders = 0;
    Qnn_ErrorHandle_t qnn_err = getInterfaceProviders(&interfaceProviders, &numProviders);
    if (qnn_err != QNN_SUCCESS || !interfaceProviders || numProviders == 0) {
        printf("错误: 无法获取 QNN 接口提供者\n");
        return (Qnn_ErrorHandle_t)1;
    }

    // 获取接口实现
    const QnnInterface_t* qnnInterface = interfaceProviders[0];
    if (qnnInterface->apiVersion.coreApiVersion.major == 2) {
        res->qnnInterfaceImpl = (QNN_INTERFACE_VER_TYPE*)&(qnnInterface->QNN_INTERFACE_VER_NAME);
    }
    if (!res->qnnInterfaceImpl) {
        printf("错误: 无法获取接口实现\n");
        return (Qnn_ErrorHandle_t)1;
    }

    // 创建Log句柄
    if (res->qnnInterfaceImpl->logCreate) {
        qnn_err = res->qnnInterfaceImpl->logCreate(NULL, QNN_LOG_LEVEL_INFO, &res->logHandle);
        if (qnn_err != QNN_SUCCESS) {
            printf("警告: logCreate 失败，继续执行\n");
        }
    }

    // 创建Backend句柄
    if (res->qnnInterfaceImpl->backendCreate) {
        qnn_err = res->qnnInterfaceImpl->backendCreate(res->logHandle, NULL, &res->backendHandle);
        if (qnn_err != QNN_BACKEND_NO_ERROR) {
            printf("错误: 无法创建 backend (错误码: 0x%lx)\n", (unsigned long)qnn_err);
            return qnn_err;
        }
    }
    printf("✓ Backend 创建成功\n");

    // 创建Device句柄
    if (!res->qnnInterfaceImpl->deviceCreate) {
        printf("错误: deviceCreate API 不可用\n");
        return (Qnn_ErrorHandle_t)1;
    }
    qnn_err = res->qnnInterfaceImpl->deviceCreate(res->logHandle, NULL, &res->deviceHandle);
    if (qnn_err != QNN_DEVICE_NO_ERROR) {
        printf("错误: 无法创建 device (错误码: 0x%lx)\n", (unsigned long)qnn_err);
        return qnn_err;
    }
    printf("✓ Device 创建成功\n");

    // 创建Context句柄
    if (!res->qnnInterfaceImpl->contextCreate) {
        printf("错误: contextCreate API 不可用\n");
        return (Qnn_ErrorHandle_t)1;
    }
    qnn_err = res->qnnInterfaceImpl->contextCreate(
        res->backendHandle,
        res->deviceHandle,
        NULL,
        &res->contextHandle);
    if (qnn_err != QNN_CONTEXT_NO_ERROR) {
        printf("错误: 无法创建 context (错误码: 0x%lx)\n", (unsigned long)qnn_err);
        return qnn_err;
    }
    printf("✓ Context 创建成功\n");

    return QNN_SUCCESS;
}

// 步骤2: 加载模型并composeGraphs（按照sample_app模式）
static Qnn_ErrorHandle_t composeGraphs(Resources* res, const char* model_path) {
    printf("\n=== 步骤 2: 加载模型并 Compose Graphs ===\n");

    // 加载模型库
    const char* so_paths[] = {
        model_path,
        "/data/local/tmp/qnn_models_elementwise/aarch64-android/libelementwise_add.so",
        "/data/local/tmp/qnn_models_elementwise/libelementwise_add.so",
        NULL
    };

    for (int i = 0; so_paths[i]; i++) {
        res->model_lib = dlopen(so_paths[i], RTLD_LAZY);
        if (res->model_lib) {
            printf("✓ 加载模型库成功: %s\n", so_paths[i]);
            break;
        }
    }

    if (!res->model_lib) {
        printf("错误: 无法加载模型库文件\n");
        return (Qnn_ErrorHandle_t)1;
    }

    // 获取composeGraphs函数
    typedef int (*ComposeGraphsFn_t)(Qnn_BackendHandle_t, QNN_INTERFACE_VER_TYPE,
                                     Qnn_ContextHandle_t,
                                     const void*, uint32_t, void***, uint32_t*,
                                     bool, void*, int);
    ComposeGraphsFn_t composeGraphs =
        (ComposeGraphsFn_t)dlsym(res->model_lib, "ElementwiseAdd_composeGraphs");
    if (!composeGraphs) {
        printf("错误: 无法找到 ElementwiseAdd_composeGraphs 函数\n");
        printf("提示: dlerror = %s\n", dlerror());
        return (Qnn_ErrorHandle_t)1;
    }
    printf("✓ 找到 composeGraphs 函数\n");

    // 复制QNN接口结构体（按值传递）
    QNN_INTERFACE_VER_TYPE qnnInterface;
    memcpy(&qnnInterface, res->qnnInterfaceImpl, sizeof(QNN_INTERFACE_VER_TYPE));

    // 调用composeGraphs（按照sample_app模式，传递NULL作为graphsConfigInfo）
    void** graphsInfo = NULL;
    uint32_t numGraphsInfo = 0;

    printf("调用 composeGraphs（使用默认配置）...\n");
    int model_err = composeGraphs(
        res->backendHandle,
        qnnInterface,
        res->contextHandle,
        NULL,  // graphsConfigInfo - 使用默认配置
        0,     // numGraphsConfigInfo
        &graphsInfo,
        &numGraphsInfo,
        false, // debug
        NULL,  // logCallback
        0      // maxLogLevel
    );

    if (model_err != 0) {
        printf("错误: composeGraphs 返回错误码: %d\n", model_err);
        return (Qnn_ErrorHandle_t)1;
    }

    if (!graphsInfo || numGraphsInfo == 0) {
        printf("错误: graphsInfo 为 NULL 或 numGraphsInfo 为 0\n");
        return (Qnn_ErrorHandle_t)1;
    }

    printf("✓ composeGraphs 成功 (graphs: %u)\n", numGraphsInfo);

    // 保存graphsInfo
    res->graphsInfo = graphsInfo;
    res->numGraphsInfo = numGraphsInfo;

    // 提取第一个graph的信息
    GraphInfoPtr_t* graphInfoArray = (GraphInfoPtr_t*)graphsInfo;
    if (!graphInfoArray[0]) {
        printf("错误: graphInfoArray[0] 为 NULL\n");
        return (Qnn_ErrorHandle_t)1;
    }

    res->graphInfo = graphInfoArray[0];
    res->graphHandle = res->graphInfo->graph;

    printf("✓ 获取 graph handle: %p\n", (void*)res->graphHandle);
    printf("  Graph名称: %s\n", res->graphInfo->graphName ? res->graphInfo->graphName : "NULL");
    printf("  输入tensor数量: %u\n", res->graphInfo->numInputTensors);
    printf("  输出tensor数量: %u\n", res->graphInfo->numOutputTensors);

    return QNN_SUCCESS;
}

// 步骤3: finalizeGraphs（按照sample_app模式）
static Qnn_ErrorHandle_t finalizeGraphs(Resources* res) {
    printf("\n=== 步骤 3: Finalize Graphs ===\n");

    if (!res->qnnInterfaceImpl->graphFinalize) {
        printf("错误: graphFinalize API 不可用\n");
        return (Qnn_ErrorHandle_t)1;
    }

    // 按照sample_app模式，传递nullptr作为第三个参数（使用默认配置）
    Qnn_ErrorHandle_t qnn_err = res->qnnInterfaceImpl->graphFinalize(
        res->graphHandle,
        NULL,  // profileHandle
        NULL   // config - 使用默认配置
    );

    if (qnn_err != QNN_GRAPH_NO_ERROR) {
        printf("错误: graphFinalize 失败 (错误码: 0x%lx)\n", (unsigned long)qnn_err);
        printf("提示: 请检查数据大小是否超过TCM限制\n");
        return qnn_err;
    }

    printf("✓ Graph Finalize 成功\n");
    return QNN_SUCCESS;
}

// 步骤4: 准备输入输出tensors（按照sample_app模式）
static Qnn_ErrorHandle_t setupInputAndOutputTensors(Resources* res, size_t data_size) {
    printf("\n=== 步骤 4: 准备输入输出 Tensors ===\n");

    if (!res->graphInfo || !res->graphInfo->inputTensors || !res->graphInfo->outputTensors) {
        printf("错误: 无法从graphInfo获取tensor配置\n");
        return (Qnn_ErrorHandle_t)1;
    }

    // 打印所有输入tensor信息
    printf("检查所有输入tensors:\n");
    for (uint32_t i = 0; i < res->graphInfo->numInputTensors; i++) {
        Qnn_Tensor_t* tensor = &res->graphInfo->inputTensors[i];
        printf("  输入tensor[%u]: name=%s, type=%d\n",
               i, tensor->v2.name ? tensor->v2.name : "NULL", tensor->v2.type);
    }

    // 找到输入tensor（使用第一个输入tensor）
    Qnn_Tensor_t* inputTensorFromGraph = NULL;
    if (res->graphInfo->numInputTensors > 0) {
        inputTensorFromGraph = &res->graphInfo->inputTensors[0];
        printf("使用第一个输入tensor: %s\n",
               inputTensorFromGraph->v2.name ? inputTensorFromGraph->v2.name : "NULL");
    }

    // 找到输出tensor
    Qnn_Tensor_t* outputTensorFromGraph = NULL;
    if (res->graphInfo->numOutputTensors > 0) {
        outputTensorFromGraph = &res->graphInfo->outputTensors[0];
        printf("使用第一个输出tensor: %s\n",
               outputTensorFromGraph->v2.name ? outputTensorFromGraph->v2.name : "NULL");
    }

    if (!inputTensorFromGraph || !outputTensorFromGraph) {
        printf("错误: 无法找到输入或输出tensor\n");
        return (Qnn_ErrorHandle_t)1;
    }

    // 使用deepCopyQnnTensorInfo复制tensor配置（按照sample_app模式）
    // 重要：不修改tensor维度，完全信任composeGraphs返回的配置
    res->inputTensor = QNN_TENSOR_INIT;
    res->outputTensor = QNN_TENSOR_INIT;

    if (!deepCopyQnnTensorInfo(&res->inputTensor, inputTensorFromGraph)) {
        printf("错误: 无法复制输入tensor信息\n");
        return (Qnn_ErrorHandle_t)1;
    }

    if (!deepCopyQnnTensorInfo(&res->outputTensor, outputTensorFromGraph)) {
        printf("错误: 无法复制输出tensor信息\n");
        return (Qnn_ErrorHandle_t)1;
    }

    // 设置内存类型和数据指针（RAW模式）
    res->inputTensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
    res->inputTensor.v2.clientBuf.data = res->input_mem;
    res->inputTensor.v2.clientBuf.dataSize = data_size;

    res->outputTensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
    res->outputTensor.v2.clientBuf.data = res->output_mem;
    res->outputTensor.v2.clientBuf.dataSize = data_size;

    // 打印tensor信息
    if (res->inputTensor.v2.dimensions && res->inputTensor.v2.rank >= 2) {
        printf("  输入tensor维度: [%u", res->inputTensor.v2.dimensions[0]);
        for (uint32_t i = 1; i < res->inputTensor.v2.rank; i++) {
            printf(", %u", res->inputTensor.v2.dimensions[i]);
        }
        printf("]\n");
    }
    if (res->outputTensor.v2.dimensions && res->outputTensor.v2.rank >= 2) {
        printf("  输出tensor维度: [%u", res->outputTensor.v2.dimensions[0]);
        for (uint32_t i = 1; i < res->outputTensor.v2.rank; i++) {
            printf(", %u", res->outputTensor.v2.dimensions[i]);
        }
        printf("]\n");
    }
    printf("  使用RAW模式\n");

    printf("✓ Tensor 准备完成\n");
    return QNN_SUCCESS;
}

// ============================================================================
// 带宽测试函数
// ============================================================================

static double test_bandwidth(Resources* res, size_t data_size, int num_iterations) {
    printf("\n=== 步骤 5: 执行带宽测试 ===\n");

    if (!res->qnnInterfaceImpl->graphExecute) {
        printf("错误: graphExecute API 不可用\n");
        return 0.0;
    }

    // 准备执行tensors
    const Qnn_Tensor_t execInputTensors[] = {res->inputTensor};
    Qnn_Tensor_t execOutputTensors[] = {res->outputTensor};

    // 预热
    printf("预热运行 (3次)...\n");
    int warmup_failures = 0;
    for (int i = 0; i < 3; i++) {
        Qnn_ErrorHandle_t err = res->qnnInterfaceImpl->graphExecute(
            res->graphHandle,
            execInputTensors, 1,
            execOutputTensors, 1,
            NULL, NULL);
        if (err != QNN_GRAPH_NO_ERROR) {
            warmup_failures++;
        }
    }
    if (warmup_failures > 0) {
        printf("警告: %d/%d 次预热失败，但继续测试\n", warmup_failures, 3);
    } else {
        printf("✓ 所有预热迭代成功\n");
    }

    // 实际测试
    printf("开始带宽测试 (%d 次迭代)...\n", num_iterations);
    double start_time = get_time();

    for (int i = 0; i < num_iterations; i++) {
        Qnn_ErrorHandle_t err = res->qnnInterfaceImpl->graphExecute(
            res->graphHandle,
            execInputTensors, 1,
            execOutputTensors, 1,
            NULL, NULL);

        if (err != QNN_GRAPH_NO_ERROR) {
            printf("错误: 执行失败 (迭代 %d/%d, 错误码: 0x%lx)\n",
                   i+1, num_iterations, (unsigned long)err);
            return 0.0;
        }
    }
    double end_time = get_time();

    double elapsed = end_time - start_time;
    double total_data = (double)data_size * num_iterations * 2;  // 读+写

    // 计算有效带宽
    double effective_bandwidth = (total_data / (1024.0 * 1024.0 * 1024.0)) / elapsed;

    printf("✓ 带宽测试完成\n");
    printf("  数据大小: %.2f MB\n", (double)data_size / (1024.0 * 1024.0));
    printf("  迭代次数: %d\n", num_iterations);
    printf("  总数据量: %.3f GB (读+写)\n", total_data / (1024.0 * 1024.0 * 1024.0));
    printf("  总执行时间: %.6f 秒\n", elapsed);
    printf("  平均执行时间: %.6f 秒/迭代 (%.3f ms/迭代)\n",
           elapsed / num_iterations, (elapsed / num_iterations) * 1000);
    printf("  有效带宽: %.2f GB/s (内存+计算的总传输率)\n", effective_bandwidth);

    return effective_bandwidth;
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    printf("=== QNN HTP Element-wise Add 带宽测试 ===\n");
    fflush(stdout);
    printf("基于 QNN SDK Sample App 标准模式实现\n\n");
    fflush(stdout);

    // 解析命令行参数
    size_t data_size_mb = (argc > 1) ? atoi(argv[1]) : 64;
    size_t data_size = data_size_mb * 1024 * 1024;
    size_t num_elements = data_size / sizeof(int8_t);
    int num_iterations = (argc > 2) ? atoi(argv[2]) : 3;
    const char* model_path = (argc > 3) ? argv[3] :
        "/data/local/tmp/qnn_models_elementwise/aarch64-android/libelementwise_add.so";

    printf("测试配置:\n");
    fflush(stdout);
    printf("  算子类型: Element-wise Add\n");
    fflush(stdout);
    printf("  数据大小: %zu MB (%zu INT8 elements)\n", data_size_mb, num_elements);
    fflush(stdout);
    printf("  迭代次数: %d\n", num_iterations);
    fflush(stdout);
    printf("  模型路径: %s\n", model_path);
    fflush(stdout);
    printf("\n");
    fflush(stdout);

    Resources res = {0};
    Qnn_ErrorHandle_t qnn_err = QNN_SUCCESS;

    // 初始化共享内存库
    printf("=== 初始化共享内存 ===\n");
    fflush(stdout);
    printf("开始加载共享内存库...\n");
    fflush(stdout);
    res.rpc_lib = load_rpc_lib();
    fflush(stdout);
    if (!res.rpc_lib) {
        printf("错误: 无法加载共享内存库\n");
        fflush(stdout);
        cleanup(&res);
        return 1;
    }
    printf("共享内存库加载成功\n");
    fflush(stdout);

    printf("查找 rpcmem_alloc 函数...\n");
    fflush(stdout);
    typedef void* (*rpcmem_alloc_fn)(int, int, int);
    rpcmem_alloc_fn rpcmem_alloc = (rpcmem_alloc_fn)dlsym(res.rpc_lib, "rpcmem_alloc");
    if (!rpcmem_alloc) {
        const char* err = dlerror();
        printf("错误: 无法找到 rpcmem_alloc 函数");
        if (err) printf(": %s", err);
        printf("\n");
        fflush(stdout);
        cleanup(&res);
        return 1;
    }
    printf("✓ 找到 rpcmem_alloc 函数\n");
    fflush(stdout);

    // 分配独立的输入和输出buffer（优化：避免数据依赖）
    printf("分配输入共享内存 (%zu bytes)...\n", data_size);
    fflush(stdout);
    res.input_mem = alloc_shared_mem(res.rpc_lib, data_size, &res.input_heapid);
    fflush(stdout);
    printf("分配输出共享内存 (%zu bytes)...\n", data_size);
    fflush(stdout);
    res.output_mem = alloc_shared_mem(res.rpc_lib, data_size, &res.output_heapid);
    fflush(stdout);

    if (!res.input_mem || !res.output_mem) {
        printf("错误: 无法分配共享内存\n");
        cleanup(&res);
        return 1;
    }
    printf("✓ 分配输入共享内存成功 (heapid=%d, size=%zu MB)\n",
           res.input_heapid, data_size_mb);
    fflush(stdout);
    printf("✓ 分配输出共享内存成功 (heapid=%d, size=%zu MB)\n",
           res.output_heapid, data_size_mb);
    fflush(stdout);

    // 初始化测试数据
    printf("初始化测试数据 (%zu elements)...\n", num_elements);
    fflush(stdout);
    int8_t* input_data = (int8_t*)res.input_mem;
    for (size_t i = 0; i < num_elements; i++) {
        input_data[i] = (int8_t)(i % 127);
    }
    printf("✓ 初始化测试数据完成\n\n");
    fflush(stdout);

    // 按照sample_app的标准流程执行
    // 步骤1: 初始化QNN Backend
    printf("开始初始化QNN Backend...\n");
    fflush(stdout);
    qnn_err = initializeQnnBackend(&res);
    fflush(stdout);
    if (qnn_err != QNN_SUCCESS) {
        printf("QNN Backend初始化失败\n");
        fflush(stdout);
        cleanup(&res);
        return 1;
    }
    printf("QNN Backend初始化完成\n");
    fflush(stdout);

    // 步骤2: 加载模型并composeGraphs
    printf("开始加载模型并composeGraphs...\n");
    fflush(stdout);
    qnn_err = composeGraphs(&res, model_path);
    fflush(stdout);
    if (qnn_err != QNN_SUCCESS) {
        printf("composeGraphs失败\n");
        fflush(stdout);
        cleanup(&res);
        return 1;
    }
    printf("composeGraphs完成\n");
    fflush(stdout);

    // 步骤3: finalizeGraphs
    printf("开始finalizeGraphs...\n");
    fflush(stdout);
    qnn_err = finalizeGraphs(&res);
    fflush(stdout);
    if (qnn_err != QNN_SUCCESS) {
        printf("finalizeGraphs失败，开始清理资源...\n");
        fflush(stdout);
        cleanup(&res);
        printf("资源清理完成\n");
        fflush(stdout);
        return 1;
    }
    printf("finalizeGraphs完成\n");
    fflush(stdout);

    // 步骤4: 准备输入输出tensors
    printf("开始准备输入输出tensors...\n");
    fflush(stdout);
    qnn_err = setupInputAndOutputTensors(&res, data_size);
    fflush(stdout);
    if (qnn_err != QNN_SUCCESS) {
        printf("setupInputAndOutputTensors失败\n");
        fflush(stdout);
        cleanup(&res);
        return 1;
    }
    printf("setupInputAndOutputTensors完成\n");
    fflush(stdout);

    // 步骤5: 执行带宽测试
    printf("开始执行带宽测试...\n");
    fflush(stdout);
    double bandwidth = test_bandwidth(&res, data_size, num_iterations);
    fflush(stdout);
    printf("带宽测试完成，结果: %.2f GB/s\n", bandwidth);
    fflush(stdout);

    // 输出结果
    if (bandwidth > 0.0) {
        printf("\n=== 测试结果 ===\n");
        printf("算子类型: Element-wise Add\n");
        printf("输入大小: %zu INT8 elements\n", num_elements);
        printf("输出大小: %zu INT8 elements\n", num_elements);
        printf("迭代次数: %d\n", num_iterations);
        printf("总数据传输: %.3f GB (读+写)\n",
               (double)data_size * num_iterations * 2 / (1024.0 * 1024.0 * 1024.0));
        printf("⚠️  这是总执行性能，不是纯内存带宽\n");
        printf("总执行效率: %.2f GB/s (包含内存传输+计算)\n", bandwidth);
        printf("\n");
        fflush(stdout);
    } else {
        printf("错误: 带宽测试失败\n");
        fflush(stdout);
        cleanup(&res);
        return 1;
    }

    printf("开始清理资源...\n");
    fflush(stdout);
    cleanup(&res);
    printf("资源清理完成\n");
    fflush(stdout);
    return 0;
}
