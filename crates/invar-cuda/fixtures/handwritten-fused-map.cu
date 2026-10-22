// Reviewed fixed M3-03 comparison: one input, three materialized wrapping-u32
// stages, scalar contiguous loads/stores, explicit grid stride and tail mask.
extern "C" __global__ void invar_map_v1(
    const unsigned int* input0,
    unsigned int* output0,
    unsigned int* output1,
    unsigned int* output2,
    unsigned int n) {
  unsigned int base = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (unsigned int i = base; i < n; i += stride) {
    unsigned int first = input0[i] + 3u;
    unsigned int second = first * 7u;
    unsigned int third = second + 11u;
    output0[i] = first;
    output1[i] = second;
    output2[i] = third;
  }
}
