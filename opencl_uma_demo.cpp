#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 从文件读取 Kernel 源代码
static char* read_kernel_source(const char* filename, size_t* source_size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("错误: 无法打开 kernel 文件: %s\n", filename);
        return NULL;
    }

    // 获取文件大小
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // 分配内存并读取文件
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

int main() {
    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem buffer = NULL;

    printf("=== OpenCL UMA (统一内存架构) Demo ===\n\n");

    // 1. 获取平台
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf("错误: 无法获取 OpenCL 平台 (错误码: %d)\n", err);
        return 1;
    }

    // 2. 获取设备 (优先 GPU)
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("警告: 无法获取 GPU 设备，尝试 CPU\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    }
    if (err != CL_SUCCESS || device == NULL) {
        printf("错误: 无法获取 OpenCL 设备\n");
        return 1;
    }

    // 打印 GPU 设备信息
    printf("=== GPU 设备信息 ===\n");

    // 设备名称
    size_t name_size;
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &name_size);
    char* device_name = (char*)malloc(name_size);
    clGetDeviceInfo(device, CL_DEVICE_NAME, name_size, device_name, NULL);
    printf("设备名称: %s\n", device_name);
    free(device_name);

    // OpenCL 版本
    size_t version_size;
    clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &version_size);
    char* version = (char*)malloc(version_size);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, version_size, version, NULL);
    printf("OpenCL 版本: %s\n", version);
    free(version);

    // 设备类型
    cl_device_type device_type;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
    printf("设备类型: ");
    if (device_type & CL_DEVICE_TYPE_GPU) printf("GPU ");
    if (device_type & CL_DEVICE_TYPE_CPU) printf("CPU ");
    if (device_type & CL_DEVICE_TYPE_ACCELERATOR) printf("Accelerator ");
    printf("\n");

    // 检查 UMA 支持
    cl_bool host_unified_memory;
    clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool),
                   &host_unified_memory, NULL);
    printf("UMA 支持 (CL_DEVICE_HOST_UNIFIED_MEMORY): %s\n",
           host_unified_memory ? "是" : "否");

    printf("\n");

    if (!host_unified_memory) {
        printf("警告: 设备可能不完全支持 UMA，但继续测试...\n\n");
    }

    // 3. 创建上下文
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("错误: 无法创建上下文 (错误码: %d)\n", err);
        return 1;
    }

    // 4. 创建命令队列
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf("错误: 无法创建命令队列 (错误码: %d)\n", err);
        clReleaseContext(context);
        return 1;
    }

    // 5. 从文件读取 Kernel 源代码
    printf("读取 Kernel 源代码文件...\n");
    size_t kernel_source_size = 0;
    char* kernel_source = read_kernel_source("fill_array.cl", &kernel_source_size);
    if (!kernel_source) {
        printf("错误: 无法读取 kernel 文件\n");
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    printf("✓ Kernel 源代码读取成功 (%zu 字节)\n", kernel_source_size);

    // 6. 创建程序
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    if (err != CL_SUCCESS) {
        printf("错误: 无法创建程序 (错误码: %d)\n", err);
        free(kernel_source);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    free(kernel_source);

    // 7. 编译程序
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("错误: 程序编译失败 (错误码: %d)\n", err);
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("编译日志:\n%s\n", log);
        free(log);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    printf("✓ 程序编译成功\n");

    // 8. 创建 kernel
    kernel = clCreateKernel(program, "fill_array", &err);
    if (err != CL_SUCCESS) {
        printf("错误: 无法创建 kernel (错误码: %d)\n", err);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    printf("✓ Kernel 创建成功\n");

    // 9. 分配数据大小
    const int ARRAY_SIZE = 16;
    const size_t buffer_size = ARRAY_SIZE * sizeof(float);

    printf("\n=== 步骤 1: 在 Host 上使用 UMA 分配内存 ===\n");
    printf("数组大小: %d 个 float (共 %zu 字节)\n", ARRAY_SIZE, buffer_size);

    // 10. 使用 CL_MEM_ALLOC_HOST_PTR 创建 UMA 缓冲区
    //
    // 在 UMA 架构下，这块内存是 CPU 和 GPU 共享的同一块物理内存
    // 使用 CL_MEM_ALLOC_HOST_PTR 分配主机可访问的内存
    //
    buffer = clCreateBuffer(context,
                           CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                           buffer_size, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("错误: 无法创建 UMA 缓冲区 (错误码: %d)\n", err);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    printf("✓ UMA 缓冲区创建成功\n");

    // 11. 映射缓冲区获取 CPU 可访问的指针
    // 注意：这里只 map 一次，之后使用 clFinish 来同步，不再 unmap/remap
    printf("\n=== 步骤 2: 映射缓冲区到主机内存 ===\n");
    float* host_ptr = (float*)clEnqueueMapBuffer(queue, buffer, CL_TRUE,
                                                 CL_MAP_WRITE | CL_MAP_READ,
                                                 0, buffer_size,
                                                 0, NULL, NULL, &err);
    if (err != CL_SUCCESS || host_ptr == NULL) {
        printf("错误: 无法映射缓冲区 (错误码: %d)\n", err);
        clReleaseMemObject(buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    printf("✓ 缓冲区映射成功，主机指针: %p\n", host_ptr);

    // 12. 在主机上初始化数据
    printf("\n=== 步骤 3: 在 Host 上初始化数据 ===\n");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        host_ptr[i] = 1.0f;  // 初始化为 0
    }
    printf("初始数据: ");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%.1f ", host_ptr[i]);
    }
    printf("\n");

    // 13. 使用 clFinish 确保 CPU 写入完成并刷新缓存
    // 在 UMA 架构下，clFinish 可以触发缓存刷新和同步
    // 不需要 unmap，直接使用 clFinish 来确保数据一致性
    clFinish(queue);
    printf("✓ CPU 写入完成，缓存已刷新（使用 clFinish 同步）\n");

    // 14. 设置 kernel 参数
    printf("\n=== 步骤 4: 在 OpenCL Kernel 中赋值 ===\n");
    float fill_value = 10.0f;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(float), &fill_value);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &ARRAY_SIZE);
    if (err != CL_SUCCESS) {
        printf("错误: 设置 kernel 参数失败 (错误码: %d)\n", err);
        clReleaseMemObject(buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    printf("✓ Kernel 参数设置成功 (fill_value = %.1f)\n", fill_value);

    // 15. 执行 kernel
    size_t global_work_size = ARRAY_SIZE;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
                                 &global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("错误: 执行 kernel 失败 (错误码: %d)\n", err);
        clReleaseMemObject(buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    printf("✓ Kernel 执行成功\n");

    // 16. 使用 clFinish 等待 GPU 执行完成
    // clFinish 会确保 GPU 的所有操作完成，并触发缓存同步
    clFinish(queue);
    printf("✓ GPU 执行完成（使用 clFinish 同步）\n");

    // 17. 在 CPU 上读取结果
    // 由于缓冲区一直保持映射状态，可以直接读取
    // clFinish 已经确保了 GPU 的写入已经刷新到共享内存
    printf("\n=== 步骤 5: 在 CPU 上读取结果 ===\n");
    printf("GPU 计算后的数据: ");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%.1f ", host_ptr[i]);
    }
    printf("\n");

    // 18. 清理：取消映射并释放资源
    clEnqueueUnmapMemObject(queue, buffer, host_ptr, 0, NULL, NULL);
    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    printf("\n=== Demo 完成 ===\n");
    return 0;
}
