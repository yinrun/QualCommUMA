#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// OpenCL Kernel 源代码
const char* kernel_source = 
"__kernel void fill_array(__global float* data, float value, int size) {\n"
"    int id = get_global_id(0);\n"
"    if (id < size) {\n"
"        data[id] = value + id;  // 赋值: value + 索引\n"
"    }\n"
"}\n";

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
    
    // 检查 UMA 支持
    cl_bool host_unified_memory;
    clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), 
                   &host_unified_memory, NULL);
    printf("UMA 支持 (CL_DEVICE_HOST_UNIFIED_MEMORY): %s\n\n", 
           host_unified_memory ? "是" : "否");
    
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
    
    // 5. 创建程序
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("错误: 无法创建程序 (错误码: %d)\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    
    // 6. 编译程序
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
    
    // 7. 创建 kernel
    kernel = clCreateKernel(program, "fill_array", &err);
    if (err != CL_SUCCESS) {
        printf("错误: 无法创建 kernel (错误码: %d)\n", err);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    printf("✓ Kernel 创建成功\n");
    
    // 8. 分配数据大小
    const int ARRAY_SIZE = 16;
    const size_t buffer_size = ARRAY_SIZE * sizeof(float);
    
    printf("\n=== 步骤 1: 在 Host 上使用 UMA 分配内存 ===\n");
    printf("数组大小: %d 个 float (共 %zu 字节)\n", ARRAY_SIZE, buffer_size);
    
    // 9. 使用 CL_MEM_ALLOC_HOST_PTR 创建 UMA 缓冲区
    // 这个标志告诉 OpenCL 分配主机可访问的内存
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
    
    // 10. 将缓冲区映射到主机内存 (UMA 的关键步骤)
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
    
    // 11. 在主机上初始化数据 (可选，演示主机访问)
    printf("\n=== 步骤 3: 在 Host 上初始化数据 ===\n");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        host_ptr[i] = 0.0f;  // 初始化为 0
    }
    printf("✓ 数据已初始化为 0\n");
    printf("初始数据: ");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%.1f ", host_ptr[i]);
    }
    printf("\n");
    
    // 12. 取消映射 (在 GPU 执行前)
    // 
    // 为什么需要取消映射？
    // 1. 内存一致性：确保 CPU 写入的数据已刷新到共享内存
    // 2. 缓存刷新：将 CPU 缓存中的数据同步到物理内存
    // 3. 同步点：创建明确的 CPU→GPU 同步边界
    // 4. 访问权限：明确转移内存访问权限给 GPU
    // 5. 规范要求：符合 OpenCL 规范的最佳实践
    //
    // 如果不取消映射，GPU 可能读取到：
    // - 还在 CPU 缓存中的数据（未刷新）
    // - 不一致或未初始化的数据
    // - 导致数据竞争和未定义行为
    //
    err = clEnqueueUnmapMemObject(queue, buffer, host_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("警告: 取消映射失败 (错误码: %d)\n", err);
    }
    printf("✓ 缓冲区已取消映射，准备 GPU 访问\n");
    printf("  (CPU 缓存已刷新，GPU 可以安全访问共享内存)\n");
    
    // 13. 设置 kernel 参数
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
    
    // 14. 执行 kernel
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
    
    // 15. 等待执行完成
    clFinish(queue);
    printf("✓ GPU 执行完成\n");
    
    // 16. 重新映射缓冲区以在 CPU 上读取
    printf("\n=== 步骤 5: 在 CPU 上读取结果 ===\n");
    host_ptr = (float*)clEnqueueMapBuffer(queue, buffer, CL_TRUE,
                                          CL_MAP_READ,
                                          0, buffer_size,
                                          0, NULL, NULL, &err);
    if (err != CL_SUCCESS || host_ptr == NULL) {
        printf("错误: 无法重新映射缓冲区 (错误码: %d)\n", err);
        clReleaseMemObject(buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    
    // 17. 在 CPU 上读取并验证结果
    printf("✓ 缓冲区重新映射成功\n");
    printf("GPU 计算后的数据: ");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%.1f ", host_ptr[i]);
    }
    printf("\n");
    
    // 验证结果
    printf("\n=== 验证结果 ===\n");
    bool correct = true;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        float expected = fill_value + i;
        if (host_ptr[i] != expected) {
            printf("错误: data[%d] = %.1f, 期望 = %.1f\n", i, host_ptr[i], expected);
            correct = false;
        }
    }
    if (correct) {
        printf("✓ 所有数据正确！UMA 工作正常。\n");
        printf("  每个元素的值 = %.1f + 索引\n", fill_value);
    }
    
    // 18. 清理
    clEnqueueUnmapMemObject(queue, buffer, host_ptr, 0, NULL, NULL);
    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    printf("\n=== Demo 完成 ===\n");
    return 0;
}

