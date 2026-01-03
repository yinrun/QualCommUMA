#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>

// 高通 ION 扩展定义
// 参考: https://registry.khronos.org/OpenCL/extensions/qcom/cl_qcom_ion_host_ptr.txt
#ifndef CL_MEM_ION_HOST_PTR_QCOM
#define CL_MEM_ION_HOST_PTR_QCOM 0x40A8
#endif
#ifndef CL_MEM_EXT_HOST_PTR_QCOM
#define CL_MEM_EXT_HOST_PTR_QCOM (1 << 29)
#endif
#ifndef CL_MEM_HOST_UNCACHED_QCOM
#define CL_MEM_HOST_UNCACHED_QCOM 0
#endif

typedef struct _cl_mem_ext_host_ptr {
    cl_uint allocation_type;
    cl_uint host_cache_policy;
} cl_mem_ext_host_ptr;

typedef struct _cl_mem_ion_host_ptr {
    cl_mem_ext_host_ptr ext_host_ptr;
    int ion_filedesc;
    void* ion_hostptr;
} cl_mem_ion_host_ptr;
#include "QnnInterface.h"
#include "QnnTypes.h"
#include "QnnMem.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnOpDef.h"
#include "QnnBackend.h"
#include "QnnOpPackage.h"

// 全局资源句柄
typedef struct {
    void* rpc_lib;
    void* qnn_backend_lib;
    void* shared_mem;
    int heapid;
    int fd;
    Qnn_LogHandle_t logHandle;
    Qnn_BackendHandle_t backendHandle;
    Qnn_DeviceHandle_t deviceHandle;
    Qnn_ContextHandle_t contextHandle;
    Qnn_GraphHandle_t graphHandle;
    Qnn_Tensor_t inputTensor;
    Qnn_Tensor_t multiplierTensor;
    Qnn_Tensor_t outputTensor;
    QNN_INTERFACE_VER_TYPE* qnnInterfaceImpl;
    cl_context cl_context;
    cl_command_queue cl_queue;
    cl_program cl_program;
    cl_kernel cl_kernel;
    cl_mem cl_buffer;
} Resources;

// 从文件读取 Kernel 源代码
static char* read_kernel_source(const char* filename, size_t* source_size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("错误: 无法打开 kernel 文件: %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* source = (char*)malloc(file_size + 1);
    if (!source) {
        fclose(file);
        return NULL;
    }

    size_t read_size = fread(source, 1, file_size, file);
    source[read_size] = '\0';
    fclose(file);

    if (source_size) {
        *source_size = read_size;
    }

    return source;
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

static void cleanup(Resources* res) {
    if (res->cl_buffer) clReleaseMemObject(res->cl_buffer);
    if (res->cl_kernel) clReleaseKernel(res->cl_kernel);
    if (res->cl_program) clReleaseProgram(res->cl_program);
    if (res->cl_queue) clReleaseCommandQueue(res->cl_queue);
    if (res->cl_context) clReleaseContext(res->cl_context);
    if (res->contextHandle && res->qnnInterfaceImpl && res->qnnInterfaceImpl->contextFree) {
        res->qnnInterfaceImpl->contextFree(res->contextHandle, NULL);
    }
    if (res->deviceHandle && res->qnnInterfaceImpl && res->qnnInterfaceImpl->deviceFree) {
        res->qnnInterfaceImpl->deviceFree(res->deviceHandle);
    }
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

int main() {
    printf("=== 统一 UMA 内存 Demo: GPU + NPU + CPU 共享同一块内存 ===\n\n");

    Resources res = {0};
    const int ARRAY_SIZE = 16;
    const size_t buffer_size = ARRAY_SIZE * sizeof(float);
    const float GPU_FILL_VALUE = 10.0f;
    const float NPU_MULTIPLIER = 3.0f;
    cl_int cl_err = CL_SUCCESS;
    Qnn_ErrorHandle_t qnn_err = QNN_SUCCESS;

    printf("=== 步骤 1: 分配 ION 共享内存 ===\n");
    res.rpc_lib = load_rpc_lib();
    if (!res.rpc_lib) {
        cleanup(&res);
        return 1;
    }

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

    res.shared_mem = alloc_shared_mem(res.rpc_lib, buffer_size, &res.heapid);
    printf("shared_mem 地址: %p\n", res.shared_mem);
    if (!res.shared_mem) {
        printf("错误: 无法分配共享内存\n");
        cleanup(&res);
        return 1;
    }
    printf("✓ 分配 ION 共享内存成功 (heapid=%d)\n", res.heapid);

    res.fd = rpcmem_to_fd(res.shared_mem);
    if (res.fd < 0) {
        printf("错误: 无法获取文件描述符\n");
        cleanup(&res);
        return 1;
    }
    printf("✓ 获取文件描述符成功 (fd=%d)\n", res.fd);

    printf("\n=== 步骤 2: 初始化 OpenCL (GPU) ===\n");
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;

    cl_err = clGetPlatformIDs(1, &platform, NULL);
    if (cl_err != CL_SUCCESS) {
        printf("错误: 无法获取 OpenCL 平台 (错误码: %d)\n", cl_err);
        cleanup(&res);
        return 1;
    }

    cl_err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (cl_err != CL_SUCCESS || device == NULL) {
        printf("错误: 无法获取 GPU 设备\n");
        cleanup(&res);
        return 1;
    }

    cl_bool host_unified_memory;
    clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &host_unified_memory, NULL);
    if (!host_unified_memory) {
        printf("错误: 设备不支持 GPU UMA\n");
        cleanup(&res);
        return 1;
    }

    res.cl_context = clCreateContext(NULL, 1, &device, NULL, NULL, &cl_err);
    if (cl_err != CL_SUCCESS) {
        printf("错误: 无法创建 OpenCL 上下文 (错误码: %d)\n", cl_err);
        cleanup(&res);
        return 1;
    }

    res.cl_queue = clCreateCommandQueueWithProperties(res.cl_context, device, NULL, &cl_err);
    if (cl_err != CL_SUCCESS) {
        printf("错误: 无法创建命令队列 (错误码: %d)\n", cl_err);
        cleanup(&res);
        return 1;
    }

    size_t kernel_source_size = 0;
    char* kernel_source = read_kernel_source("fill_array.cl", &kernel_source_size);
    if (!kernel_source) {
        printf("错误: 无法读取 kernel 文件\n");
        cleanup(&res);
        return 1;
    }

    res.cl_program = clCreateProgramWithSource(res.cl_context, 1,
                                               (const char**)&kernel_source,
                                               &kernel_source_size, &cl_err);
    free(kernel_source);
    if (cl_err != CL_SUCCESS) {
        printf("错误: 无法创建程序 (错误码: %d)\n", cl_err);
        cleanup(&res);
        return 1;
    }

    cl_err = clBuildProgram(res.cl_program, 1, &device, NULL, NULL, NULL);
    if (cl_err != CL_SUCCESS) {
        printf("错误: 程序编译失败 (错误码: %d)\n", cl_err);
        size_t log_size;
        clGetProgramBuildInfo(res.cl_program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(res.cl_program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("编译日志:\n%s\n", log);
        free(log);
        cleanup(&res);
        return 1;
    }

    res.cl_kernel = clCreateKernel(res.cl_program, "fill_array", &cl_err);
    if (cl_err != CL_SUCCESS) {
        printf("错误: 无法创建 kernel (错误码: %d)\n", cl_err);
        cleanup(&res);
        return 1;
    }

    float* shared_mem_ptr = (float*)res.shared_mem;
    for (int i = 0; i < ARRAY_SIZE; i++) shared_mem_ptr[i] = 1.0f;
    print_array(shared_mem_ptr, ARRAY_SIZE, "初始数据 (CPU)");

    // 使用 cl_qcom_ion_host_ptr 扩展创建 OpenCL Buffer
    // 参考: https://registry.khronos.org/OpenCL/extensions/qcom/cl_qcom_ion_host_ptr.txt
    cl_mem_ion_host_ptr ion_mem = {{0}};
    ion_mem.ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_QCOM;
    ion_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    ion_mem.ion_filedesc = res.fd;
    ion_mem.ion_hostptr = res.shared_mem;

    res.cl_buffer = clCreateBuffer(res.cl_context,
                                   CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM | CL_MEM_READ_WRITE,
                                   buffer_size, &ion_mem, &cl_err);
    if (cl_err != CL_SUCCESS) {
        printf("错误: 无法创建 OpenCL Buffer\n");
        cleanup(&res);
        return 1;
    }
    printf("✓ OpenCL Buffer 创建成功\n\n");

    printf("=== 步骤 3: 初始化 QNN (NPU) ===\n");
    res.qnn_backend_lib = load_qnn_backend();
    if (!res.qnn_backend_lib) {
        printf("错误: 无法加载 QNN backend 库\n");
        cleanup(&res);
        return 1;
    }

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
    qnn_err = getInterfaceProviders(&interfaceProviders, &numProviders);
    if (qnn_err != QNN_SUCCESS || !interfaceProviders || numProviders == 0) {
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

    if (res.qnnInterfaceImpl->logCreate) {
        res.qnnInterfaceImpl->logCreate(NULL, QNN_LOG_LEVEL_INFO, &res.logHandle);
    }
    if (res.qnnInterfaceImpl->backendCreate) {
        qnn_err = res.qnnInterfaceImpl->backendCreate(res.logHandle, NULL, &res.backendHandle);
        if (qnn_err != QNN_BACKEND_NO_ERROR) {
            printf("错误: 无法创建 backend\n");
            cleanup(&res);
            return 1;
        }
    }

    // 创建 Device (HTP)
    if (!res.qnnInterfaceImpl->deviceCreate) {
        printf("错误: deviceCreate API 不可用\n");
        cleanup(&res);
        return 1;
    }
    qnn_err = res.qnnInterfaceImpl->deviceCreate(res.logHandle, NULL, &res.deviceHandle);
    if (qnn_err != QNN_DEVICE_NO_ERROR) {
        printf("错误: 无法创建 device (错误码: %lu)\n", (unsigned long)qnn_err);
        cleanup(&res);
        return 1;
    }
    printf("✓ 创建 Device 成功\n");

    // 创建 Context
    if (!res.qnnInterfaceImpl->contextCreate) {
        printf("错误: contextCreate API 不可用\n");
        cleanup(&res);
        return 1;
    }
    qnn_err = res.qnnInterfaceImpl->contextCreate(res.backendHandle,
                                                   res.deviceHandle,
                                                   NULL, &res.contextHandle);
    if (qnn_err != QNN_CONTEXT_NO_ERROR) {
        printf("错误: 无法创建 context (错误码: 0x%lx)\n", (unsigned long)qnn_err);
        cleanup(&res);
        return 1;
    }
    printf("✓ 创建 Context 成功\n");

    if (!res.qnnInterfaceImpl->memRegister) {
        printf("错误: memRegister API 不可用\n");
        cleanup(&res);
        return 1;
    }
    Qnn_MemHandle_t memHandle = NULL;
    Qnn_MemDescriptor_t memDesc = QNN_MEM_DESCRIPTOR_INIT;
    memDesc.memType = QNN_MEM_TYPE_ION;
    memDesc.dataType = QNN_DATATYPE_FLOAT_32;
    memDesc.memShape.numDim = 1;
    uint32_t memDim = ARRAY_SIZE;
    memDesc.memShape.dimSize = &memDim;
    memDesc.ionInfo.fd = res.fd;

    qnn_err = res.qnnInterfaceImpl->memRegister(res.contextHandle, &memDesc, 1, &memHandle);
    if (qnn_err != QNN_MEM_NO_ERROR) {
        printf("错误: 无法注册共享内存\n");
        cleanup(&res);
        return 1;
    }
    printf("✓ 注册共享内存成功\n");

    printf("✓ QNN 初始化成功\n\n");

    printf("=== 步骤 4: NPU 处理数据 ===\n");
    if (!res.qnnInterfaceImpl->graphCreate) {
        printf("错误: Graph API 不可用\n");
        cleanup(&res);
        return 1;
    }

    qnn_err = res.qnnInterfaceImpl->graphCreate(res.contextHandle,
                                                 "custom_multiply_graph",
                                                 NULL, &res.graphHandle);
    if (qnn_err != QNN_GRAPH_NO_ERROR) {
        printf("错误: 无法创建 graph (错误码: %lu)\n", (unsigned long)qnn_err);
        cleanup(&res);
        return 1;
    }
    printf("✓ 创建 Graph 成功\n");

    uint32_t dims[] = {ARRAY_SIZE};
    memset(&res.inputTensor, 0, sizeof(Qnn_Tensor_t));
    res.inputTensor.version = QNN_TENSOR_VERSION_2;
    res.inputTensor.v2.name = "input_tensor";
    res.inputTensor.v2.type = QNN_TENSOR_TYPE_APP_WRITE;
    res.inputTensor.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    res.inputTensor.v2.dataType = QNN_DATATYPE_FLOAT_32;
    res.inputTensor.v2.rank = 1;
    res.inputTensor.v2.dimensions = dims;
    res.inputTensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
    res.inputTensor.v2.clientBuf.data = NULL;
    res.inputTensor.v2.clientBuf.dataSize = buffer_size;
    memset(&res.inputTensor.v2.quantizeParams, 0, sizeof(Qnn_QuantizeParams_t));

    qnn_err = res.qnnInterfaceImpl->tensorCreateGraphTensor(res.graphHandle, &res.inputTensor);
    if (qnn_err != QNN_TENSOR_NO_ERROR) {
        printf("错误: 无法创建输入 tensor (错误码: %lu)\n", (unsigned long)qnn_err);
        cleanup(&res);
        return 1;
    }
    printf("✓ 创建输入 Tensor 成功\n");

    static float multiplier_array[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        multiplier_array[i] = NPU_MULTIPLIER;
    }
    memset(&res.multiplierTensor, 0, sizeof(Qnn_Tensor_t));
    res.multiplierTensor.version = QNN_TENSOR_VERSION_2;
    res.multiplierTensor.v2.name = "multiplier_tensor";
    res.multiplierTensor.v2.type = QNN_TENSOR_TYPE_STATIC;
    res.multiplierTensor.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    res.multiplierTensor.v2.dataType = QNN_DATATYPE_FLOAT_32;
    res.multiplierTensor.v2.rank = 1;
    res.multiplierTensor.v2.dimensions = dims;
    res.multiplierTensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
    res.multiplierTensor.v2.clientBuf.data = multiplier_array;
    res.multiplierTensor.v2.clientBuf.dataSize = buffer_size;
    memset(&res.multiplierTensor.v2.quantizeParams, 0, sizeof(Qnn_QuantizeParams_t));

    qnn_err = res.qnnInterfaceImpl->tensorCreateGraphTensor(res.graphHandle, &res.multiplierTensor);
    if (qnn_err != QNN_TENSOR_NO_ERROR) {
        printf("错误: 无法创建乘数 tensor (错误码: %lu)\n", (unsigned long)qnn_err);
        cleanup(&res);
        return 1;
    }
    printf("✓ 创建乘数 Tensor 成功 (value=%.1f)\n", NPU_MULTIPLIER);

    // 创建输出 Tensor
    memset(&res.outputTensor, 0, sizeof(Qnn_Tensor_t));
    res.outputTensor.version = QNN_TENSOR_VERSION_2;
    res.outputTensor.v2.name = "output_tensor";
    res.outputTensor.v2.type = QNN_TENSOR_TYPE_APP_READ;
    res.outputTensor.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    res.outputTensor.v2.dataType = QNN_DATATYPE_FLOAT_32;
    res.outputTensor.v2.rank = 1;
    res.outputTensor.v2.dimensions = dims;
    res.outputTensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
    res.outputTensor.v2.clientBuf.data = NULL;
    res.outputTensor.v2.clientBuf.dataSize = buffer_size;
    memset(&res.outputTensor.v2.quantizeParams, 0, sizeof(Qnn_QuantizeParams_t));

    qnn_err = res.qnnInterfaceImpl->tensorCreateGraphTensor(res.graphHandle, &res.outputTensor);
    if (qnn_err != QNN_TENSOR_NO_ERROR) {
        printf("错误: 无法创建输出 tensor (错误码: %lu)\n", (unsigned long)qnn_err);
        cleanup(&res);
        return 1;
    }
    printf("✓ 创建输出 Tensor 成功\n");

    // 创建 OpConfig
    Qnn_OpConfig_t opConfig = QNN_OPCONFIG_INIT;
    opConfig.version = QNN_OPCONFIG_VERSION_1;
    opConfig.v1.name = "elementwise_multiply_node";
    opConfig.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    opConfig.v1.typeName = QNN_OP_ELEMENT_WISE_MULTIPLY;
    opConfig.v1.numOfInputs = 2;
    Qnn_Tensor_t opInputTensors[] = {res.inputTensor, res.multiplierTensor};
    opConfig.v1.inputTensors = opInputTensors;
    opConfig.v1.numOfOutputs = 1;
    Qnn_Tensor_t opOutputTensors[] = {res.outputTensor};
    opConfig.v1.outputTensors = opOutputTensors;

    if (!res.qnnInterfaceImpl->graphAddNode) {
        printf("错误: graphAddNode API 不可用\n");
        cleanup(&res);
        return 1;
    }

    qnn_err = res.qnnInterfaceImpl->graphAddNode(res.graphHandle, opConfig);
    if (qnn_err != QNN_GRAPH_NO_ERROR) {
        printf("错误: 无法添加节点到 graph (错误码: %lu)\n", (unsigned long)qnn_err);
        cleanup(&res);
        return 1;
    }
    printf("✓ 添加节点到 Graph 成功\n");

    if (!res.qnnInterfaceImpl->graphFinalize) {
        printf("错误: graphFinalize API 不可用\n");
        cleanup(&res);
        return 1;
    }

    qnn_err = res.qnnInterfaceImpl->graphFinalize(res.graphHandle, NULL, NULL);
    if (qnn_err != QNN_GRAPH_NO_ERROR) {
        printf("错误: 无法 finalize graph (错误码: %lu)\n", (unsigned long)qnn_err);
        cleanup(&res);
        return 1;
    }
    printf("✓ Graph Finalize 成功\n");

    printf("\n=== 步骤 4.1: GPU 第一次处理数据 ===\n");
    cl_err = clSetKernelArg(res.cl_kernel, 0, sizeof(cl_mem), &res.cl_buffer);
    cl_err |= clSetKernelArg(res.cl_kernel, 1, sizeof(float), &GPU_FILL_VALUE);
    cl_err |= clSetKernelArg(res.cl_kernel, 2, sizeof(int), &ARRAY_SIZE);
    if (cl_err != CL_SUCCESS) {
        printf("错误: 设置 kernel 参数失败 (错误码: %d)\n", cl_err);
        cleanup(&res);
        return 1;
    }

    size_t work_size = ARRAY_SIZE;
    cl_err = clEnqueueNDRangeKernel(res.cl_queue, res.cl_kernel, 1, NULL,
                                    &work_size, NULL, 0, NULL, NULL);
    if (cl_err != CL_SUCCESS) {
        printf("错误: 执行 kernel 失败 (错误码: %d)\n", cl_err);
        cleanup(&res);
        return 1;
    }

    clFinish(res.cl_queue);
    #if defined(__GNUC__) || defined(__clang__)
        __sync_synchronize();
    #endif
    print_array((float*)res.shared_mem, ARRAY_SIZE, "GPU 第一次处理后的数据");

    if (!res.qnnInterfaceImpl->graphExecute) {
        printf("错误: graphExecute API 不可用\n");
        cleanup(&res);
        return 1;
    }

    res.inputTensor.v2.clientBuf.data = res.shared_mem;
    res.inputTensor.v2.clientBuf.dataSize = buffer_size;
    res.outputTensor.v2.clientBuf.data = res.shared_mem;
    res.outputTensor.v2.clientBuf.dataSize = buffer_size;

    const Qnn_Tensor_t execInputTensors[] = {res.inputTensor};
    Qnn_Tensor_t execOutputTensors[] = {res.outputTensor};

    qnn_err = res.qnnInterfaceImpl->graphExecute(res.graphHandle,
                                                   execInputTensors, 1,
                                                   execOutputTensors, 1,
                                                   NULL, NULL);
    if (qnn_err != QNN_GRAPH_NO_ERROR) {
        printf("错误: 执行 graph 失败 (错误码: %lu)\n", (unsigned long)qnn_err);
        cleanup(&res);
        return 1;
    }
    printf("✓ NPU Graph 执行完成（每个值乘以%.1f）\n", NPU_MULTIPLIER);
    print_array((float*)res.shared_mem, ARRAY_SIZE, "NPU 处理后的数据");

    clFinish(res.cl_queue);
    #if defined(__GNUC__) || defined(__clang__)
        __sync_synchronize();
    #endif
    printf("\n=== 步骤 5: GPU 第二次处理数据 ===\n");

    cl_err = clSetKernelArg(res.cl_kernel, 0, sizeof(cl_mem), &res.cl_buffer);
    cl_err |= clSetKernelArg(res.cl_kernel, 1, sizeof(float), &GPU_FILL_VALUE);
    cl_err |= clSetKernelArg(res.cl_kernel, 2, sizeof(int), &ARRAY_SIZE);
    if (cl_err != CL_SUCCESS) {
        printf("错误: 设置 kernel 参数失败 (错误码: %d)\n", cl_err);
        cleanup(&res);
        return 1;
    }

    size_t global_work_size = ARRAY_SIZE;
    cl_err = clEnqueueNDRangeKernel(res.cl_queue, res.cl_kernel, 1, NULL,
                                    &global_work_size, NULL, 0, NULL, NULL);
    if (cl_err != CL_SUCCESS) {
        printf("错误: 执行 kernel 失败 (错误码: %d)\n", cl_err);
        cleanup(&res);
        return 1;
    }

    clFinish(res.cl_queue);
    #if defined(__GNUC__) || defined(__clang__)
        __sync_synchronize();
    #endif
    print_array((float*)res.shared_mem, ARRAY_SIZE, "GPU 处理后的数据");

    printf("\n=== 步骤 6: 验证结果 ===\n");
    // 计算流程: 初始值(1.0) -> GPU第一次(+10.0+id+0.3) -> NPU(*3.0) -> GPU第二次(+10.0+id+0.3)
    float expected_0 = ((1.0f + GPU_FILL_VALUE + 0.0f + 0.3f) * NPU_MULTIPLIER) + GPU_FILL_VALUE + 0.0f + 0.3f;
    float* result_ptr = (float*)res.shared_mem;
    if (abs(result_ptr[0] - expected_0) < 0.1f) {
        printf("✓ 验证成功：预期值[0]=%.1f, 实际值[0]=%.1f\n", expected_0, result_ptr[0]);
    } else {
        printf("⚠️  验证失败：预期值[0]=%.1f, 实际值[0]=%.1f\n", expected_0, result_ptr[0]);
    }

    printf("\n=== Demo 完成 ===\n");
    printf("✓ CPU、GPU、NPU 共享同一块 UMA 内存，实现零拷贝\n");

    cleanup(&res);
    return 0;
}
