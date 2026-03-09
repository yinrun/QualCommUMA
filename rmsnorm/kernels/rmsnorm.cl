// RMSNorm: y[i] = (x[i] / sqrt(mean(x^2) + eps)) * gamma[i]
// One work-group per row (batch element).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void rmsnorm(
    __global half*       output,
    __global const half* input,
    __global const half* gamma,
    const int hidden_dim,
    const float epsilon,
    __local float* sdata)
{
  int row = get_group_id(0);
  int lid = get_local_id(0);
  int lsz = get_local_size(0);

  __global const half* x = input  + row * hidden_dim;
  __global half*       y = output + row * hidden_dim;

  float partial = 0.0f;
  for (int i = lid; i < hidden_dim; i += lsz) {
    float val = vload_half(i, (__global const half*)x);
    partial += val * val;
  }
  sdata[lid] = partial;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = lsz >> 1; s > 0; s >>= 1) {
    if (lid < s)
      sdata[lid] += sdata[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  float rms_inv = rsqrt(sdata[0] / (float)hidden_dim + epsilon);

  for (int i = lid; i < hidden_dim; i += lsz) {
    float val = vload_half(i, (__global const half*)x);
    float g   = vload_half(i, (__global const half*)gamma);
    vstore_half(val * rms_inv * g, i, (__global half*)y);
  }
}
