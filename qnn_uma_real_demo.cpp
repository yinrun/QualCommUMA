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

int main() {
    printf("=== QNN UMA Demo: 使用 QNN SDK 执行 ElementWiseMultiply（每个值乘以3）===\n\n");
    
    // 加载共享内存库
    void* rpc_lib = dlopen("libcdsprpc.so", RTLD_LAZY);
    if (rpc_lib == NULL) {
        rpc_lib = dlopen("/vendor/lib64/libcdsprpc.so", RTLD_LAZY);
    }
    if (rpc_lib == NULL) {
        printf("错误: 无法加载 libcdsprpc.so\n");
        return 1;
    }
    
    typedef void* (*rpcmem_alloc_fn)(int, int, int);
    typedef void (*rpcmem_free_fn)(void*);
    typedef int (*rpcmem_to_fd_fn)(void*);
    
    rpcmem_alloc_fn rpcmem_alloc = (rpcmem_alloc_fn)dlsym(rpc_lib, "rpcmem_alloc");
    rpcmem_free_fn rpcmem_free = (rpcmem_free_fn)dlsym(rpc_lib, "rpcmem_free");
    rpcmem_to_fd_fn rpcmem_to_fd = (rpcmem_to_fd_fn)dlsym(rpc_lib, "rpcmem_to_fd");
    
    if (!rpcmem_alloc || !rpcmem_free || !rpcmem_to_fd) {
        printf("错误: 无法找到 rpcmem 函数\n");
        dlclose(rpc_lib);
        return 1;
    }
    
    // 分配共享内存
    const int ARRAY_SIZE = 16;
    const size_t buffer_size = ARRAY_SIZE * sizeof(float);
    
    void* shared_mem = NULL;
    int successful_heapid = -1;
    int heapids[] = {0, 1, 2, 13, 14, 25, 26, 27, 28, 22, 23, 24};
    
    for (int i = 0; i < sizeof(heapids)/sizeof(heapids[0]); i++) {
        shared_mem = rpcmem_alloc(heapids[i], 0, buffer_size);
        if (shared_mem != NULL) {
            successful_heapid = heapids[i];
            break;
        }
    }
    
    if (shared_mem == NULL) {
        printf("错误: 无法分配共享内存\n");
        dlclose(rpc_lib);
        return 1;
    }
    
    printf("✓ 分配共享内存成功 (heapid=%d)\n", successful_heapid);
    
    // 准备输入数据
    float* cpu_ptr = (float*)shared_mem;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        cpu_ptr[i] = (float)(i * 1.5f + 10.0f);
    }
    
    printf("输入数据: ");
    for (int i = 0; i < 8; i++) printf("%.1f ", cpu_ptr[i]);
    printf("...\n");
    
    // 获取文件描述符
    int fd = rpcmem_to_fd(shared_mem);
    if (fd < 0) {
        printf("错误: 无法获取文件描述符\n");
        rpcmem_free(shared_mem);
        dlclose(rpc_lib);
        return 1;
    }
    
    // 加载 QNN SDK 库
    const char* qnn_backend_paths[] = {
        "/vendor/lib64/libQnnHtp.so",
        "/vendor/lib64/libQnnHtpStub.so",
        "/vendor/lib64/libQnnCpu.so",
        "/vendor/lib64/libQnnGpu.so",
        NULL
    };
    
    void* qnn_backend_lib = NULL;
    for (int i = 0; qnn_backend_paths[i] != NULL; i++) {
        qnn_backend_lib = dlopen(qnn_backend_paths[i], RTLD_LAZY);
        if (qnn_backend_lib != NULL) break;
    }
    
    if (qnn_backend_lib == NULL) {
        printf("错误: 无法加载 QNN backend 库\n");
        rpcmem_free(shared_mem);
        dlclose(rpc_lib);
        return 1;
    }
    
    printf("✓ 加载 QNN SDK 成功\n");
    
    // 获取 QNN 接口
    typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList,
                                                              uint32_t* numProviders);
    
    QnnInterfaceGetProvidersFn_t getInterfaceProviders = 
        (QnnInterfaceGetProvidersFn_t)dlsym(qnn_backend_lib, "QnnInterface_getProviders");
    
    if (getInterfaceProviders == NULL) {
        printf("错误: 无法找到 QnnInterface_getProviders\n");
        dlclose(qnn_backend_lib);
        rpcmem_free(shared_mem);
        dlclose(rpc_lib);
        return 1;
    }
    
    const QnnInterface_t** interfaceProviders = NULL;
    uint32_t numProviders = 0;
    
    Qnn_ErrorHandle_t err = getInterfaceProviders(&interfaceProviders, &numProviders);
    if (err != QNN_SUCCESS || interfaceProviders == NULL || numProviders == 0) {
        printf("错误: 无法获取 QNN 接口提供者\n");
        dlclose(qnn_backend_lib);
        rpcmem_free(shared_mem);
        dlclose(rpc_lib);
        return 1;
    }
    
    const QnnInterface_t* qnnInterface = interfaceProviders[0];
    QNN_INTERFACE_VER_TYPE* qnnInterfaceImpl = NULL;
    
    if (qnnInterface->apiVersion.coreApiVersion.major == 2) {
        qnnInterfaceImpl = (QNN_INTERFACE_VER_TYPE*)&(qnnInterface->QNN_INTERFACE_VER_NAME);
    }
    
    if (qnnInterfaceImpl == NULL) {
        printf("错误: 无法获取接口实现\n");
        dlclose(qnn_backend_lib);
        rpcmem_free(shared_mem);
        dlclose(rpc_lib);
        return 1;
    }
    
    // 声明所有变量（避免 goto 跳过初始化）
    Qnn_BackendHandle_t backendHandle = NULL;
    Qnn_LogHandle_t logHandle = NULL;
    Qnn_ContextHandle_t context = NULL;
    Qnn_DeviceHandle_t deviceHandle = NULL;
    Qnn_MemDescriptor_t memDescriptor = QNN_MEM_DESCRIPTOR_INIT;
    uint32_t dim_size = ARRAY_SIZE;
    Qnn_MemHandle_t memHandle = NULL;
    Qnn_GraphHandle_t graphHandle = NULL;
    float* constant_data = NULL;
    bool all_correct = false;
    
    // 初始化 QNN backend
    if (qnnInterfaceImpl->logCreate) {
        qnnInterfaceImpl->logCreate(NULL, QNN_LOG_LEVEL_INFO, &logHandle);
    }
    
    if (qnnInterfaceImpl->backendCreate) {
        err = qnnInterfaceImpl->backendCreate(logHandle, NULL, &backendHandle);
        if (err != QNN_BACKEND_NO_ERROR) {
            printf("错误: 无法创建 backend\n");
            if (logHandle && qnnInterfaceImpl->logFree) qnnInterfaceImpl->logFree(logHandle);
            dlclose(qnn_backend_lib);
            rpcmem_free(shared_mem);
            dlclose(rpc_lib);
            return 1;
        }
    }
    
    // 创建 context
    if (qnnInterfaceImpl->deviceCreate) {
        err = qnnInterfaceImpl->deviceCreate(logHandle, NULL, &deviceHandle);
        if (err != QNN_DEVICE_NO_ERROR) {
            printf("错误: 无法创建 device\n");
            if (backendHandle && qnnInterfaceImpl->backendFree) qnnInterfaceImpl->backendFree(backendHandle);
            if (logHandle && qnnInterfaceImpl->logFree) qnnInterfaceImpl->logFree(logHandle);
            dlclose(qnn_backend_lib);
            rpcmem_free(shared_mem);
            dlclose(rpc_lib);
            return 1;
        }
    }
    
    if (qnnInterfaceImpl->contextCreate && deviceHandle) {
        err = qnnInterfaceImpl->contextCreate(backendHandle, deviceHandle, NULL, &context);
        if (err != QNN_CONTEXT_NO_ERROR) {
            printf("错误: 无法创建 context\n");
            if (deviceHandle && qnnInterfaceImpl->deviceFree) qnnInterfaceImpl->deviceFree(deviceHandle);
            if (backendHandle && qnnInterfaceImpl->backendFree) qnnInterfaceImpl->backendFree(backendHandle);
            if (logHandle && qnnInterfaceImpl->logFree) qnnInterfaceImpl->logFree(logHandle);
            dlclose(qnn_backend_lib);
            rpcmem_free(shared_mem);
            dlclose(rpc_lib);
            return 1;
        }
    }
    
    printf("✓ 初始化 QNN SDK 成功\n");
    
    // 注册共享内存到 QNN
    memDescriptor.memShape.numDim = 1;
    memDescriptor.memShape.dimSize = &dim_size;
    memDescriptor.dataType = QNN_DATATYPE_FLOAT_32;
    memDescriptor.memType = QNN_MEM_TYPE_ION;
    memDescriptor.ionInfo.fd = fd;
    
    err = qnnInterfaceImpl->memRegister(context, &memDescriptor, 1, &memHandle);
    if (err != QNN_SUCCESS) {
        printf("错误: 无法注册共享内存\n");
        if (context && qnnInterfaceImpl->contextFree) qnnInterfaceImpl->contextFree(context, NULL);
        if (deviceHandle && qnnInterfaceImpl->deviceFree) qnnInterfaceImpl->deviceFree(deviceHandle);
        if (backendHandle && qnnInterfaceImpl->backendFree) qnnInterfaceImpl->backendFree(backendHandle);
        if (logHandle && qnnInterfaceImpl->logFree) qnnInterfaceImpl->logFree(logHandle);
        dlclose(qnn_backend_lib);
        rpcmem_free(shared_mem);
        dlclose(rpc_lib);
        return 1;
    }
    
    printf("✓ 注册共享内存成功\n");
    
    // 创建 QNN Graph 并执行 ElementWiseMultiply（每个值乘以3）
    err = qnnInterfaceImpl->graphCreate(context, "multiply_by_2_graph", NULL, &graphHandle);
    if (err != QNN_SUCCESS) {
        printf("错误: 无法创建 graph\n");
        if (qnnInterfaceImpl->memDeRegister) qnnInterfaceImpl->memDeRegister(&memHandle, 1);
        if (context && qnnInterfaceImpl->contextFree) qnnInterfaceImpl->contextFree(context, NULL);
        if (deviceHandle && qnnInterfaceImpl->deviceFree) qnnInterfaceImpl->deviceFree(deviceHandle);
        if (backendHandle && qnnInterfaceImpl->backendFree) qnnInterfaceImpl->backendFree(backendHandle);
        if (logHandle && qnnInterfaceImpl->logFree) qnnInterfaceImpl->logFree(logHandle);
        dlclose(qnn_backend_lib);
        rpcmem_free(shared_mem);
        dlclose(rpc_lib);
        return 1;
    }
    
    // 准备 tensor
    dim_size = ARRAY_SIZE;
    
    Qnn_Tensor_t inputTensor = QNN_TENSOR_INIT;
    inputTensor.version = QNN_TENSOR_VERSION_1;
    inputTensor.v1.name = "input";
    inputTensor.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
    inputTensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    inputTensor.v1.dataType = QNN_DATATYPE_FLOAT_32;
    inputTensor.v1.rank = 1;
    inputTensor.v1.dimensions = &dim_size;
    inputTensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    inputTensor.v1.clientBuf.data = NULL;
    inputTensor.v1.clientBuf.dataSize = 0;
    
    constant_data = (float*)malloc(ARRAY_SIZE * sizeof(float));
    for (int i = 0; i < ARRAY_SIZE; i++) {
        constant_data[i] = -10.1f;
    }
    
    Qnn_Tensor_t constantTensor = QNN_TENSOR_INIT;
    constantTensor.version = QNN_TENSOR_VERSION_1;
    constantTensor.v1.name = "constant_2";
    constantTensor.v1.type = QNN_TENSOR_TYPE_STATIC;
    constantTensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    constantTensor.v1.dataType = QNN_DATATYPE_FLOAT_32;
    constantTensor.v1.rank = 1;
    constantTensor.v1.dimensions = &dim_size;
    constantTensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    constantTensor.v1.clientBuf.data = constant_data;
    constantTensor.v1.clientBuf.dataSize = ARRAY_SIZE * sizeof(float);
    
    Qnn_Tensor_t outputTensor = QNN_TENSOR_INIT;
    outputTensor.version = QNN_TENSOR_VERSION_1;
    outputTensor.v1.name = "output";
    outputTensor.v1.type = QNN_TENSOR_TYPE_APP_READ;
    outputTensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    outputTensor.v1.dataType = QNN_DATATYPE_FLOAT_32;
    outputTensor.v1.rank = 1;
    outputTensor.v1.dimensions = &dim_size;
    outputTensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    outputTensor.v1.clientBuf.data = NULL;
    outputTensor.v1.clientBuf.dataSize = 0;
    
    // 创建 tensor
    err = qnnInterfaceImpl->tensorCreateGraphTensor(graphHandle, &inputTensor);
    if (err != QNN_SUCCESS) {
        printf("错误: 无法创建输入 tensor\n");
        free(constant_data);
        if (qnnInterfaceImpl->memDeRegister) qnnInterfaceImpl->memDeRegister(&memHandle, 1);
        if (context && qnnInterfaceImpl->contextFree) qnnInterfaceImpl->contextFree(context, NULL);
        if (deviceHandle && qnnInterfaceImpl->deviceFree) qnnInterfaceImpl->deviceFree(deviceHandle);
        if (backendHandle && qnnInterfaceImpl->backendFree) qnnInterfaceImpl->backendFree(backendHandle);
        if (logHandle && qnnInterfaceImpl->logFree) qnnInterfaceImpl->logFree(logHandle);
        dlclose(qnn_backend_lib);
        rpcmem_free(shared_mem);
        dlclose(rpc_lib);
        return 1;
    }
    
    err = qnnInterfaceImpl->tensorCreateGraphTensor(graphHandle, &constantTensor);
    if (err != QNN_SUCCESS) {
        printf("错误: 无法创建常量 tensor\n");
        free(constant_data);
        if (qnnInterfaceImpl->memDeRegister) qnnInterfaceImpl->memDeRegister(&memHandle, 1);
        if (context && qnnInterfaceImpl->contextFree) qnnInterfaceImpl->contextFree(context, NULL);
        if (deviceHandle && qnnInterfaceImpl->deviceFree) qnnInterfaceImpl->deviceFree(deviceHandle);
        if (backendHandle && qnnInterfaceImpl->backendFree) qnnInterfaceImpl->backendFree(backendHandle);
        if (logHandle && qnnInterfaceImpl->logFree) qnnInterfaceImpl->logFree(logHandle);
        dlclose(qnn_backend_lib);
        rpcmem_free(shared_mem);
        dlclose(rpc_lib);
        return 1;
    }
    
    err = qnnInterfaceImpl->tensorCreateGraphTensor(graphHandle, &outputTensor);
    if (err != QNN_SUCCESS) {
        printf("错误: 无法创建输出 tensor\n");
        free(constant_data);
        if (qnnInterfaceImpl->memDeRegister) qnnInterfaceImpl->memDeRegister(&memHandle, 1);
        if (context && qnnInterfaceImpl->contextFree) qnnInterfaceImpl->contextFree(context, NULL);
        if (deviceHandle && qnnInterfaceImpl->deviceFree) qnnInterfaceImpl->deviceFree(deviceHandle);
        if (backendHandle && qnnInterfaceImpl->backendFree) qnnInterfaceImpl->backendFree(backendHandle);
        if (logHandle && qnnInterfaceImpl->logFree) qnnInterfaceImpl->logFree(logHandle);
        dlclose(qnn_backend_lib);
        rpcmem_free(shared_mem);
        dlclose(rpc_lib);
        return 1;
    }
    
    // 创建 ElementWiseMultiply 节点
    Qnn_Tensor_t mulInputs[] = {inputTensor, constantTensor};
    Qnn_Tensor_t mulOutputs[] = {outputTensor};
    
    Qnn_OpConfig_t opConfig = QNN_OPCONFIG_INIT;
    opConfig.version = QNN_OPCONFIG_VERSION_1;
    opConfig.v1.name = "multiply_by_2_op";
    opConfig.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    opConfig.v1.typeName = QNN_OP_ELEMENT_WISE_MULTIPLY;
    opConfig.v1.numOfParams = 0;
    opConfig.v1.params = NULL;
    opConfig.v1.numOfInputs = 2;
    opConfig.v1.inputTensors = mulInputs;
    opConfig.v1.numOfOutputs = 1;
    opConfig.v1.outputTensors = mulOutputs;
    
    err = qnnInterfaceImpl->graphAddNode(graphHandle, opConfig);
    if (err != QNN_SUCCESS) {
        printf("错误: 无法添加节点\n");
        free(constant_data);
        if (qnnInterfaceImpl->memDeRegister) qnnInterfaceImpl->memDeRegister(&memHandle, 1);
        if (context && qnnInterfaceImpl->contextFree) qnnInterfaceImpl->contextFree(context, NULL);
        if (deviceHandle && qnnInterfaceImpl->deviceFree) qnnInterfaceImpl->deviceFree(deviceHandle);
        if (backendHandle && qnnInterfaceImpl->backendFree) qnnInterfaceImpl->backendFree(backendHandle);
        if (logHandle && qnnInterfaceImpl->logFree) qnnInterfaceImpl->logFree(logHandle);
        dlclose(qnn_backend_lib);
        rpcmem_free(shared_mem);
        dlclose(rpc_lib);
        return 1;
    }
    
    err = qnnInterfaceImpl->graphFinalize(graphHandle, NULL, NULL);
    if (err != QNN_SUCCESS) {
        printf("错误: 无法 finalize graph\n");
        free(constant_data);
        if (qnnInterfaceImpl->memDeRegister) qnnInterfaceImpl->memDeRegister(&memHandle, 1);
        if (context && qnnInterfaceImpl->contextFree) qnnInterfaceImpl->contextFree(context, NULL);
        if (deviceHandle && qnnInterfaceImpl->deviceFree) qnnInterfaceImpl->deviceFree(deviceHandle);
        if (backendHandle && qnnInterfaceImpl->backendFree) qnnInterfaceImpl->backendFree(backendHandle);
        if (logHandle && qnnInterfaceImpl->logFree) qnnInterfaceImpl->logFree(logHandle);
        dlclose(qnn_backend_lib);
        rpcmem_free(shared_mem);
        dlclose(rpc_lib);
        return 1;
    }
    
    // 执行 graph
    Qnn_Tensor_t executeInputs[] = {inputTensor};
    executeInputs[0].v1.clientBuf.data = cpu_ptr;
    executeInputs[0].v1.clientBuf.dataSize = buffer_size;
    
    Qnn_Tensor_t executeOutputs[] = {outputTensor};
    executeOutputs[0].v1.clientBuf.data = cpu_ptr;
    executeOutputs[0].v1.clientBuf.dataSize = buffer_size;
    
    err = qnnInterfaceImpl->graphExecute(graphHandle, executeInputs, 1, executeOutputs, 1, NULL, NULL);
    if (err != QNN_SUCCESS) {
        printf("错误: Graph 执行失败\n");
        free(constant_data);
        if (qnnInterfaceImpl->memDeRegister) qnnInterfaceImpl->memDeRegister(&memHandle, 1);
        if (context && qnnInterfaceImpl->contextFree) qnnInterfaceImpl->contextFree(context, NULL);
        if (deviceHandle && qnnInterfaceImpl->deviceFree) qnnInterfaceImpl->deviceFree(deviceHandle);
        if (backendHandle && qnnInterfaceImpl->backendFree) qnnInterfaceImpl->backendFree(backendHandle);
        if (logHandle && qnnInterfaceImpl->logFree) qnnInterfaceImpl->logFree(logHandle);
        dlclose(qnn_backend_lib);
        rpcmem_free(shared_mem);
        dlclose(rpc_lib);
        return 1;
    }
    
    printf("输出数据: ");
    for (int i = 0; i < 8; i++) printf("%.1f ", cpu_ptr[i]);
    printf("...\n");
    
    // 验证结果
    all_correct = true;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        float expected = (float)(i * 1.5f + 10.0f) * 3.0f;
        if (cpu_ptr[i] != expected) {
            all_correct = false;
            break;
        }
    }
    
    if (all_correct) {
        printf("✓ 验证成功：所有值都已乘以3\n");
    } else {
        printf("⚠️  部分数据不匹配\n");
    }
    
    // 清理
    free(constant_data);
    if (qnnInterfaceImpl->memDeRegister) qnnInterfaceImpl->memDeRegister(&memHandle, 1);
    if (context && qnnInterfaceImpl->contextFree) qnnInterfaceImpl->contextFree(context, NULL);
    if (deviceHandle && qnnInterfaceImpl->deviceFree) qnnInterfaceImpl->deviceFree(deviceHandle);
    if (backendHandle && qnnInterfaceImpl->backendFree) qnnInterfaceImpl->backendFree(backendHandle);
    if (logHandle && qnnInterfaceImpl->logFree) qnnInterfaceImpl->logFree(logHandle);
    rpcmem_free(shared_mem);
    dlclose(qnn_backend_lib);
    dlclose(rpc_lib);
    
    return (all_correct && err == QNN_SUCCESS) ? 0 : 1;
}
