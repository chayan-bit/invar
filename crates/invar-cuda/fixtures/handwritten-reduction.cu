// Reviewed M3-04 wrapping-sum comparator: 16 coalesced items per lane,
// warp shuffle trees, one warp-summary merge, ordered block propagation.
#define ITEMS 4096u
#define LANES 256u
extern "C" __global__ void invar_block_v1(const unsigned int* input, unsigned int* summaries, unsigned int n) {
  unsigned int t = threadIdx.x;
  unsigned int base = blockIdx.x * ITEMS + t;
  unsigned int value = 0u;
  for (unsigned int row = 0u; row < ITEMS / LANES; ++row) {
    unsigned int i = base + row * LANES;
    if (i < n) value += input[i];
  }
  for (unsigned int offset = 16u; offset > 0u; offset >>= 1u)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  __shared__ unsigned int warps[8];
  if ((t & 31u) == 0u) warps[t >> 5u] = value;
  __syncthreads();
  if (t == 0u) {
    unsigned int total = 0u;
    for (unsigned int w = 0u; w < 8u; ++w) total += warps[w];
    summaries[blockIdx.x] = total;
  }
}
extern "C" __global__ void invar_propagate_v1(const unsigned int* summaries, unsigned int* total, unsigned int blocks) {
  if (blockIdx.x == 0u) {
    unsigned int t = threadIdx.x;
    unsigned int value = 0u;
    for (unsigned int i = t; i < blocks; i += LANES) value += summaries[i];
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u)
      value += __shfl_down_sync(0xffffffffu, value, offset);
    __shared__ unsigned int warps[8];
    if ((t & 31u) == 0u) warps[t >> 5u] = value;
    __syncthreads();
    if (t == 0u) {
      unsigned int sum = 0u;
      for (unsigned int w = 0u; w < 8u; ++w) sum += warps[w];
      total[0] = sum;
    }
  }
}
extern "C" __global__ void invar_writeback_v1(const unsigned int* total, unsigned int* output, unsigned int seed) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) output[0] = seed + total[0];
}
