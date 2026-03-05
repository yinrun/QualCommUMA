#define CL_TARGET_OPENCL_VERSION 200
#include "gpu_rmsnorm.h"
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

namespace {

cl_platform_id   g_platform = nullptr;
cl_device_id     g_device   = nullptr;
cl_context       g_context  = nullptr;
cl_command_queue  g_queue    = nullptr;
cl_program       g_program  = nullptr;
cl_kernel        g_kernel   = nullptr;
cl_mem           g_bufInput = nullptr;
cl_mem           g_bufOutput= nullptr;
cl_mem           g_bufGamma = nullptr;
size_t           g_elem_size = 0;  // bytes per element (2 for fp16, 4 for fp32)
int              g_batch     = 0;
int              g_hidden    = 0;

static char* read_file(const char* path, size_t* out_size) {
  FILE* f = fopen(path, "r");
  if (!f) return nullptr;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* buf = (char*)malloc(sz + 1);
  size_t n = fread(buf, 1, sz, f);
  buf[n] = '\0';
  fclose(f);
  if (out_size) *out_size = n;
  return buf;
}

// Convert float to IEEE 754 half-precision
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

}  // namespace

void gpu_rmsnorm_print_info() {
  if (!g_device) return;
  char name[256];
  clGetDeviceInfo(g_device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
  cl_uint cu;
  clGetDeviceInfo(g_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
  cl_ulong mem;
  clGetDeviceInfo(g_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
  printf("  GPU: %s, %u CU, %.2f GB\n", name, cu, mem / (1024.0*1024.0*1024.0));
}

bool gpu_rmsnorm_init(const RMSNormConfig& config, const char* kernel_path) {
  cl_int err;
  g_batch  = config.batch_size;
  g_hidden = config.hidden_dim;
  g_elem_size = 2;  // FP16

  // Platform & device
  err = clGetPlatformIDs(1, &g_platform, nullptr);
  if (err != CL_SUCCESS) { printf("[GPU] clGetPlatformIDs: %d\n", err); return false; }
  err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_GPU, 1, &g_device, nullptr);
  if (err != CL_SUCCESS) { printf("[GPU] clGetDeviceIDs: %d\n", err); return false; }

  // Context & queue
  g_context = clCreateContext(nullptr, 1, &g_device, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) { printf("[GPU] clCreateContext: %d\n", err); return false; }
  g_queue = clCreateCommandQueueWithProperties(g_context, g_device, nullptr, &err);
  if (err != CL_SUCCESS) { printf("[GPU] clCreateCommandQueue: %d\n", err); return false; }

  // Compile kernel with FP16 flag
  size_t src_size = 0;
  char* src = read_file(kernel_path, &src_size);
  if (!src) { printf("[GPU] Cannot read %s\n", kernel_path); return false; }

  g_program = clCreateProgramWithSource(g_context, 1, (const char**)&src, &src_size, &err);
  free(src);
  if (err != CL_SUCCESS) { printf("[GPU] clCreateProgramWithSource: %d\n", err); return false; }

  const char* build_opts = "-DUSE_FP16";
  err = clBuildProgram(g_program, 1, &g_device, build_opts, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_sz;
    clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
    char* log = (char*)malloc(log_sz);
    clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, log_sz, log, nullptr);
    printf("[GPU] Build error:\n%s\n", log);
    free(log);
    return false;
  }

  g_kernel = clCreateKernel(g_program, "rmsnorm", &err);
  if (err != CL_SUCCESS) { printf("[GPU] clCreateKernel: %d\n", err); return false; }

  // Allocate buffers
  size_t tensor_bytes = (size_t)g_batch * g_hidden * g_elem_size;
  size_t gamma_bytes  = (size_t)g_hidden * g_elem_size;

  g_bufInput  = clCreateBuffer(g_context, CL_MEM_READ_ONLY,  tensor_bytes, nullptr, &err);
  if (err != CL_SUCCESS) { printf("[GPU] input buffer: %d\n", err); return false; }
  g_bufOutput = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, tensor_bytes, nullptr, &err);
  if (err != CL_SUCCESS) { printf("[GPU] output buffer: %d\n", err); return false; }
  g_bufGamma  = clCreateBuffer(g_context, CL_MEM_READ_ONLY,  gamma_bytes,  nullptr, &err);
  if (err != CL_SUCCESS) { printf("[GPU] gamma buffer: %d\n", err); return false; }

  // Initialize input with random FP16 values in [0.1, 1.0]
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.1f, 1.0f);

  std::vector<uint16_t> host_input(g_batch * g_hidden);
  for (auto& v : host_input) v = float_to_half(dist(rng));
  clEnqueueWriteBuffer(g_queue, g_bufInput, CL_TRUE, 0, tensor_bytes, host_input.data(), 0, nullptr, nullptr);

  // Initialize gamma = 1.0
  std::vector<uint16_t> host_gamma(g_hidden, float_to_half(1.0f));
  clEnqueueWriteBuffer(g_queue, g_bufGamma, CL_TRUE, 0, gamma_bytes, host_gamma.data(), 0, nullptr, nullptr);

  return true;
}

RMSNormResult gpu_rmsnorm_run(const RMSNormConfig& config, int num_warmup, int num_iters) {
  RMSNormResult res;
  res.num_iterations = num_iters;

  // Work sizes: one work-group per batch row
  size_t local  = 256;
  size_t max_wg;
  clGetDeviceInfo(g_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr);
  if (local > max_wg) local = max_wg;
  size_t global = (size_t)g_batch * local;

  // Set kernel args
  float eps = config.epsilon;
  int hd = g_hidden;
  clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &g_bufOutput);
  clSetKernelArg(g_kernel, 1, sizeof(cl_mem), &g_bufInput);
  clSetKernelArg(g_kernel, 2, sizeof(cl_mem), &g_bufGamma);
  clSetKernelArg(g_kernel, 3, sizeof(int), &hd);
  clSetKernelArg(g_kernel, 4, sizeof(float), &eps);
  clSetKernelArg(g_kernel, 5, local * sizeof(float), nullptr);  // local memory

  // Warmup
  for (int i = 0; i < num_warmup; ++i)
    clEnqueueNDRangeKernel(g_queue, g_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
  clFinish(g_queue);

  // Timed run
  double t0 = now_seconds();
  for (int i = 0; i < num_iters; ++i)
    clEnqueueNDRangeKernel(g_queue, g_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
  clFinish(g_queue);
  double t1 = now_seconds();

  double elapsed = t1 - t0;
  // Bandwidth: read input + read gamma + write output = 3 tensors
  // (gamma is small and likely cached, but count it conservatively)
  double bytes_per_call = (double)g_batch * g_hidden * g_elem_size * 2.0  // read input + write output
                        + (double)g_hidden * g_elem_size;                  // read gamma
  double total_bytes = bytes_per_call * num_iters;

  res.latency_us     = (elapsed / num_iters) * 1e6;
  res.bandwidth_gbps = (total_bytes / (1024.0*1024.0*1024.0)) / elapsed;
  res.success = true;
  return res;
}

bool gpu_rmsnorm_read_output(void* dst, size_t bytes) {
  if (!g_queue || !g_bufOutput) return false;
  cl_int err = clEnqueueReadBuffer(g_queue, g_bufOutput, CL_TRUE, 0, bytes, dst, 0, nullptr, nullptr);
  return err == CL_SUCCESS;
}

void gpu_rmsnorm_cleanup() {
  if (g_kernel)    clReleaseKernel(g_kernel);
  if (g_bufInput)  clReleaseMemObject(g_bufInput);
  if (g_bufOutput) clReleaseMemObject(g_bufOutput);
  if (g_bufGamma)  clReleaseMemObject(g_bufGamma);
  if (g_program)   clReleaseProgram(g_program);
  if (g_queue)     clReleaseCommandQueue(g_queue);
  if (g_context)   clReleaseContext(g_context);
  g_kernel = nullptr; g_bufInput = nullptr; g_bufOutput = nullptr; g_bufGamma = nullptr;
  g_program = nullptr; g_queue = nullptr; g_context = nullptr;
  g_platform = nullptr; g_device = nullptr;
}
