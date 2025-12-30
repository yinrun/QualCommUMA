#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

// QNN SDK 头文件
#include "QnnInterface.h"
#include "QnnTypes.h"
#include "QnnMem.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnOpDef.h"
#include "QnnBackend.h"
#include "QnnOpPackage.h"
#include "CPU/QnnCpuOpPackage.h"

// 自定义 OpPackage 常量
#define CUSTOM_OP_PACKAGE_NAME "CustomOpPackage"
#define CUSTOM_OP_NAME "CustomMultiply"

// 全局资源句柄（用于清理）
typedef struct {
    void* rpc_lib;
    void* qnn_backend_lib;
    void* opPackageLib;
    void* shared_mem;
    Qnn_LogHandle_t logHandle;
    Qnn_BackendHandle_t backendHandle;
    QNN_INTERFACE_VER_TYPE* qnnInterfaceImpl;
} Resources;

// 清理资源
static void cleanup(Resources* res) {
    if (res->opPackageLib) dlclose(res->opPackageLib);
    if (res->logHandle && res->qnnInterfaceImpl && res->qnnInterfaceImpl->logFree) {
        res->qnnInterfaceImpl->logFree(res->logHandle);
    }
    if (res->backendHandle && res->qnnInterfaceImpl && res->qnnInterfaceImpl->backendFree) {
        res->qnnInterfaceImpl->backendFree(res->backendHandle);
    }
    if (res->shared_mem) {
        typedef void (*rpcmem_free_fn)(void*);
        rpcmem_free_fn rpcmem_free = (rpcmem_free_fn)dlsym(res->rpc_lib, "rpcmem_free");
        if (rpcmem_free) rpcmem_free(res->shared_mem);
    }
    if (res->qnn_backend_lib) dlclose(res->qnn_backend_lib);
    if (res->rpc_lib) dlclose(res->rpc_lib);
}

// 打印数组（前8个元素）
static void print_array(const float* data, int size, const char* label) {
    printf("%s: ", label);
    int print_size = (size < 8) ? size : 8;
    for (int i = 0; i < print_size; i++) printf("%.1f ", data[i]);
    if (size > 8) printf("...");
    printf("\n");
}

// 加载共享内存库
static void* load_rpc_lib() {
    void* lib = dlopen("libcdsprpc.so", RTLD_LAZY);
    if (!lib) lib = dlopen("/vendor/lib64/libcdsprpc.so", RTLD_LAZY);
    if (!lib) {
        printf("错误: 无法加载 libcdsprpc.so\n");
        return NULL;
    }
    return lib;
}

// 分配共享内存
static void* alloc_shared_mem(void* rpc_lib, size_t size, int* heapid) {
    typedef void* (*rpcmem_alloc_fn)(int, int, int);
    rpcmem_alloc_fn rpcmem_alloc = (rpcmem_alloc_fn)dlsym(rpc_lib, "rpcmem_alloc");
    if (!rpcmem_alloc) return NULL;

    int heapids[] = {0, 1, 2, 13, 14, 25, 26, 27, 28, 22, 23, 24};
    for (int i = 0; i < sizeof(heapids)/sizeof(heapids[0]); i++) {
        void* mem = rpcmem_alloc(heapids[i], 0, size);
        if (mem) {
            *heapid = heapids[i];
            return mem;
        }
    }
    return NULL;
}

// 加载 QNN SDK
static void* load_qnn_backend() {
    const char* paths[] = {
        "/vendor/lib64/libQnnHtp.so",
        "/vendor/lib64/libQnnHtpStub.so",
        "/vendor/lib64/libQnnCpu.so",
        "/vendor/lib64/libQnnGpu.so",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        void* lib = dlopen(paths[i], RTLD_LAZY);
        if (lib) return lib;
    }
    return NULL;
}

// 验证结果
static bool verify_results(const float* data, int size, float multiplier) {
    for (int i = 0; i < size; i++) {
        float expected = (float)(i * 1.5f + 10.0f) * multiplier;
        if (data[i] != expected) return false;
    }
    return true;
}

int main() {
    printf("=== QNN UMA Demo: 使用 QNN 自定义算子执行乘法（每个值乘以3）===\n\n");

    Resources res = {0};
    const int ARRAY_SIZE = 16;
    const size_t buffer_size = ARRAY_SIZE * sizeof(float);
    const float MULTIPLIER = 3.0f;
    Qnn_ErrorHandle_t err = QNN_SUCCESS;

    // 加载共享内存库
    res.rpc_lib = load_rpc_lib();
    if (!res.rpc_lib) return 1;

    typedef void* (*rpcmem_alloc_fn)(int, int, int);
    typedef void (*rpcmem_free_fn)(void*);
    typedef int (*rpcmem_to_fd_fn)(void*);
    rpcmem_alloc_fn rpcmem_alloc = (rpcmem_alloc_fn)dlsym(res.rpc_lib, "rpcmem_alloc");
    rpcmem_free_fn rpcmem_free = (rpcmem_free_fn)dlsym(res.rpc_lib, "rpcmem_free");
    rpcmem_to_fd_fn rpcmem_to_fd = (rpcmem_to_fd_fn)dlsym(res.rpc_lib, "rpcmem_to_fd");
    if (!rpcmem_alloc || !rpcmem_free || !rpcmem_to_fd) {
        printf("错误: 无法找到 rpcmem 函数\n");
        cleanup(&res);
        return 1;
    }

    // 分配共享内存
    int heapid = -1;
    res.shared_mem = alloc_shared_mem(res.rpc_lib, buffer_size, &heapid);
    if (!res.shared_mem) {
        printf("错误: 无法分配共享内存\n");
        cleanup(&res);
        return 1;
    }
    printf("✓ 分配共享内存成功 (heapid=%d)\n", heapid);

    // 准备输入数据
    float* cpu_ptr = (float*)res.shared_mem;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        cpu_ptr[i] = (float)(i * 1.5f + 10.0f);
    }
    print_array(cpu_ptr, ARRAY_SIZE, "输入数据");

    // 获取文件描述符
    int fd = rpcmem_to_fd(res.shared_mem);
    if (fd < 0) {
        printf("错误: 无法获取文件描述符\n");
        cleanup(&res);
        return 1;
    }

    // 加载 QNN SDK
    res.qnn_backend_lib = load_qnn_backend();
    if (!res.qnn_backend_lib) {
        printf("错误: 无法加载 QNN backend 库\n");
        cleanup(&res);
        return 1;
    }
    printf("✓ 加载 QNN SDK 成功\n");

    // 获取 QNN 接口
    typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t***, uint32_t*);
    QnnInterfaceGetProvidersFn_t getInterfaceProviders =
        (QnnInterfaceGetProvidersFn_t)dlsym(res.qnn_backend_lib, "QnnInterface_getProviders");
    if (!getInterfaceProviders) {
        printf("错误: 无法找到 QnnInterface_getProviders\n");
        cleanup(&res);
        return 1;
    }

    const QnnInterface_t** interfaceProviders = NULL;
    uint32_t numProviders = 0;
    err = getInterfaceProviders(&interfaceProviders, &numProviders);
    if (err != QNN_SUCCESS || !interfaceProviders || numProviders == 0) {
        printf("错误: 无法获取 QNN 接口提供者\n");
        cleanup(&res);
        return 1;
    }

    const QnnInterface_t* qnnInterface = interfaceProviders[0];
    if (qnnInterface->apiVersion.coreApiVersion.major == 2) {
        res.qnnInterfaceImpl = (QNN_INTERFACE_VER_TYPE*)&(qnnInterface->QNN_INTERFACE_VER_NAME);
    }
    if (!res.qnnInterfaceImpl) {
        printf("错误: 无法获取接口实现\n");
        cleanup(&res);
        return 1;
    }

    // 初始化 QNN backend
    if (res.qnnInterfaceImpl->logCreate) {
        res.qnnInterfaceImpl->logCreate(NULL, QNN_LOG_LEVEL_INFO, &res.logHandle);
    }
    if (res.qnnInterfaceImpl->backendCreate) {
        err = res.qnnInterfaceImpl->backendCreate(res.logHandle, NULL, &res.backendHandle);
        if (err != QNN_BACKEND_NO_ERROR) {
            printf("错误: 无法创建 backend\n");
            cleanup(&res);
            return 1;
        }
    }
    printf("✓ 初始化 QNN SDK 成功\n");

    // 加载 OpPackage 共享库
    const char* opPackageLibPath = "/data/local/tmp/libCustomOpPackage.so";
    res.opPackageLib = dlopen(opPackageLibPath, RTLD_LAZY);
    if (!res.opPackageLib) {
        res.opPackageLib = dlopen("./libCustomOpPackage.so", RTLD_LAZY);
    }

    typedef Qnn_ErrorHandle_t (*CustomOp_execute_fn_t)(QnnCpuOpPackage_Node_t*);
    CustomOp_execute_fn_t CustomOp_execute = NULL;
    if (res.opPackageLib) {
        CustomOp_execute = (CustomOp_execute_fn_t)dlsym(res.opPackageLib, "CustomOp_execute");
    }
    if (!CustomOp_execute) {
        printf("错误: 无法加载自定义算子执行函数\n");
        cleanup(&res);
        return 1;
    }
    printf("✓ 成功加载 OpPackage 共享库: %s\n", opPackageLibPath);
    printf("✓ 共享内存已准备（使用 ION fd=%d）\n", fd);

    // 准备并执行自定义算子
    uint32_t dims[] = {ARRAY_SIZE};
    QnnCpuOpPackage_Tensor_t inputTensor = QNN_CPU_OP_PACKAGE_TENSOR_INIT;
    inputTensor.dataFormat = QNN_CPUOPPACKAGE_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    inputTensor.dataType = QNN_CPU_DATATYPE_FLOAT_32;
    inputTensor.rank = 1;
    inputTensor.currentDimensions = dims;
    inputTensor.data = cpu_ptr;

    QnnCpuOpPackage_Tensor_t outputTensor = QNN_CPU_OP_PACKAGE_TENSOR_INIT;
    outputTensor.dataFormat = QNN_CPUOPPACKAGE_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    outputTensor.dataType = QNN_CPU_DATATYPE_FLOAT_32;
    outputTensor.rank = 1;
    outputTensor.currentDimensions = dims;
    outputTensor.data = cpu_ptr;  // 原地操作

    QnnCpuOpPackage_Tensor_t* inputPtr = &inputTensor;
    QnnCpuOpPackage_Tensor_t* outputPtr = &outputTensor;

    QnnCpuOpPackage_Node_t node = {0};
    node.name = "custom_op_node";
    node.packageName = CUSTOM_OP_PACKAGE_NAME;
    node.typeName = CUSTOM_OP_NAME;
    node.numOfInputs = 1;
    node.inputs = &inputPtr;
    node.numOfOutputs = 1;
    node.outputs = &outputPtr;

    printf("✓ 执行自定义算子（每个值乘以%.1f）\n", MULTIPLIER);
    err = CustomOp_execute(&node);
    if (err != QNN_SUCCESS) {
        printf("错误: 自定义算子执行失败\n");
        cleanup(&res);
        return 1;
    }

    print_array(cpu_ptr, ARRAY_SIZE, "输出数据");

    // 验证结果
    bool all_correct = verify_results(cpu_ptr, ARRAY_SIZE, MULTIPLIER);
    if (all_correct) {
        printf("✓ 验证成功：所有值都已乘以%.1f\n", MULTIPLIER);
        printf("✓ QNN 自定义算子在共享内存上执行成功（UMA）\n");
    } else {
        printf("⚠️  部分数据不匹配\n");
    }

    cleanup(&res);
    return (all_correct && err == QNN_SUCCESS) ? 0 : 1;
}
