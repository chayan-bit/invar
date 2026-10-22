// Reviewed M3-04 ordered affine comparator: each lane owns 16 consecutive
// source items, with ordered trees for summaries and direct output recompute.
#define ITEMS 4096u
#define LANES 256u
typedef struct { unsigned int a; unsigned int b; } Pair;
__device__ __forceinline__ Pair identity() { return {1u, 0u}; }
__device__ __forceinline__ Pair then(Pair left, Pair right) {
  return {right.a * left.a, right.a * left.b + right.b};
}
__device__ __forceinline__ unsigned int apply(Pair value, unsigned int seed) {
  return value.a * seed + value.b;
}
extern "C" __global__ void invar_block_v1(const unsigned int* aa, const unsigned int* bb, Pair* summaries, unsigned int n) {
  unsigned int t = threadIdx.x;
  unsigned int base = blockIdx.x * ITEMS + t * 16u;
  Pair aggregate = identity();
  for (unsigned int j = 0u; j < 16u; ++j) {
    unsigned int i = base + j;
    Pair item = i < n ? Pair{aa[i], bb[i]} : identity();
    aggregate = then(aggregate, item);
  }
  __shared__ Pair tree[LANES];
  tree[t] = aggregate;
  __syncthreads();
  for (unsigned int offset = 1u; offset < LANES; offset <<= 1u) {
    if (((t + 1u) % (2u * offset)) == 0u)
      tree[t] = then(tree[t - offset], tree[t]);
    __syncthreads();
  }
  if (t == LANES - 1u) summaries[blockIdx.x] = tree[t];
}
extern "C" __global__ void invar_propagate_v1(const Pair* summaries, Pair* prefix, unsigned int blocks) {
  if (blockIdx.x == 0u) {
    unsigned int t = threadIdx.x;
    unsigned int chunk = (blocks + LANES - 1u) / LANES;
    unsigned int start = t * chunk;
    Pair aggregate = identity();
    for (unsigned int j = 0u; j < chunk && start + j < blocks; ++j)
      aggregate = then(aggregate, summaries[start + j]);
    __shared__ Pair ping[LANES];
    __shared__ Pair pong[LANES];
    ping[t] = aggregate;
    __syncthreads();
    for (unsigned int offset = 1u; offset < LANES; offset <<= 1u) {
      Pair value = ping[t];
      pong[t] = t >= offset ? then(ping[t - offset], value) : value;
      __syncthreads();
      ping[t] = pong[t];
      __syncthreads();
    }
    Pair carry = t == 0u ? identity() : ping[t - 1u];
    for (unsigned int j = 0u; j < chunk && start + j < blocks; ++j) {
      prefix[start + j] = carry;
      carry = then(carry, summaries[start + j]);
    }
  }
}
extern "C" __global__ void invar_writeback_v1(const unsigned int* aa, const unsigned int* bb, const Pair* summaries, const Pair* prefix, unsigned int* output, unsigned int seed, unsigned int n, unsigned int blocks) {
  unsigned int t = threadIdx.x;
  unsigned int base = blockIdx.x * ITEMS + t * 16u;
  Pair rows[16];
  Pair aggregate = identity();
  for (unsigned int j = 0u; j < 16u; ++j) {
    unsigned int i = base + j;
    Pair item = i < n ? Pair{aa[i], bb[i]} : identity();
    aggregate = then(aggregate, item);
    rows[j] = aggregate;
  }
  __shared__ Pair ping[LANES];
  __shared__ Pair pong[LANES];
  ping[t] = aggregate;
  __syncthreads();
  for (unsigned int offset = 1u; offset < LANES; offset <<= 1u) {
    Pair value = ping[t];
    pong[t] = t >= offset ? then(ping[t - offset], value) : value;
    __syncthreads();
    ping[t] = pong[t];
    __syncthreads();
  }
  Pair before = t == 0u ? prefix[blockIdx.x] : then(prefix[blockIdx.x], ping[t - 1u]);
  for (unsigned int j = 0u; j < 16u; ++j) {
    unsigned int i = base + j;
    if (i < n) output[i] = apply(then(before, rows[j]), seed);
  }
  if (base == 0u) output[n] = apply(then(prefix[blocks - 1u], summaries[blocks - 1u]), seed);
}
