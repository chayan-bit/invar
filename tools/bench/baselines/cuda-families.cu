// Independent CUDA baselines for the M5-01 families, swept to 2^28 elements.
// Library calls: CUB DeviceTransform (map, CCCL 2.8+), CUB DeviceReduce::Sum
// (reduction), CUB DeviceScan::InclusiveSum and InclusiveScan with an affine
// compose operator (scan), CUB DeviceSegmentedReduce::Sum (segmented), CUB
// DeviceSelect::If (stable compaction) and Thrust gather (gather/scatter). Transpose is a port of
// NVIDIA's reference tiled shared-memory transpose (transposeNoBankConflicts,
// NVIDIA-developer-blog/code-samples series/cuda-cpp/transpose, BSD-3-Clause),
// extended with bounds checks for rectangular shapes. Stencil and frontier have
// no library primitive and are reviewed handwritten kernels pinned by this file's
// content hash. Matrix uses the separate cuBLAS fixture (cuda-cublas-gemm.cu).
// Build (native sm_120; see bench/README.md for the CUDA 12.9 include and library paths):
// clang++ -std=c++17 -O2 -x cuda --cuda-path=CUDA_ROOT --cuda-gpu-arch=sm_120 bench/baselines/cuda-families.cu -lcudart -o cuda-families
// Usage: cuda-families <transpose|gather|stencil|segsum|compact|frontier|map_lcg|sum_lcg|scan_lcg|affine_lcg> <n> <output.bin>
// Inputs match metal-families.mm: x[i] = i*1664525 + 1013904223 (u32), slot k adds 17k;
// map_lcg is (x0+x1)*7+11; affine_lcg scans pairs (x0,x1) as y = x0*y + x1 from y = 17; gather
// index i*2654435761 mod n; segments of 16; compaction keeps x < 2^31 in order;
// frontier n is the adjacency cell count (a square, at most 2^28) of the dense
// graph with v = sqrt(n) vertices and edges u->u+1, u->5u+1 (mod v), started at
// vertex 0 and run for at most 16 level-synchronous rounds.
#include <cub/cub.cuh>
#include <cuda/std/tuple>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/version.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_OK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); return 2; } } while (0)

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

// NVIDIA reference transposeNoBankConflicts, rows x cols input, cols x rows output.
__global__ void transpose_tiled(uint32_t* out, const uint32_t* in, uint32_t rows, uint32_t cols) {
  __shared__ uint32_t tile[TILE_DIM][TILE_DIM + 1];
  uint32_t x = blockIdx.x * TILE_DIM + threadIdx.x;
  uint32_t y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    if (x < cols && y + j < rows) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * cols + x];
  __syncthreads();
  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    if (x < rows && y + j < cols) out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
}

// Five-point stencil with absent neighbours contributing zero (wrapping u32).
__global__ void stencil(const uint32_t* a, uint32_t* b, uint32_t rows, uint32_t cols) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= rows || c >= cols) return;
  size_t i = size_t(r) * cols + c;
  uint32_t v = a[i];
  if (r > 0) v += a[i - cols];
  if (r + 1 < rows) v += a[i + cols];
  if (c > 0) v += a[i - 1];
  if (c + 1 < cols) v += a[i + 1];
  b[i] = v;
}

// One warp per vertex: an active, unseen vertex marks itself seen and scans its
// adjacency row with coalesced loads, setting next[w] for every edge.
__global__ void frontier_round(const uint32_t* edges, uint32_t* seen, const uint32_t* frontier,
                               uint32_t* next, uint32_t* live, uint32_t v) {
  uint32_t u = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  // The whole warp shares u, so it returns or continues together.
  if (u >= v || frontier[u] == 0 || seen[u] != 0) return;
  if (lane == 0) seen[u] = 1;
  const uint32_t* row = edges + size_t(u) * v;
  for (uint32_t w = lane; w < v; w += 32) {
    if (row[w] == 1) { next[w] = 1; *live = 1; }
  }
}

struct Affine { uint32_t a, b; };
struct Compose {
  __host__ __device__ Affine operator()(Affine left, Affine right) const {
    return {right.a * left.a, right.a * left.b + right.b};
  }
};
struct MapOp {
  __host__ __device__ uint32_t operator()(uint32_t a, uint32_t b) const { return (a + b) * 7u + 11u; }
};
__global__ void apply_seed(const Affine* p, uint32_t* out, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = p[i].a * 17u + p[i].b;
}

struct Below {
  __host__ __device__ bool operator()(uint32_t x) const { return x < 0x80000000u; }
};

static uint32_t lcg(uint32_t i) { return i * 1664525u + 1013904223u; }

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  std::string family = argv[1];
  char* end = nullptr;
  unsigned long parsed = std::strtoul(argv[2], &end, 10);
  if (!end || *end || parsed < 1 || parsed > (1ul << 28)) return 2;
  uint32_t n = uint32_t(parsed), cells = n;
  bool transpose = family == "transpose", gather = family == "gather", st = family == "stencil";
  bool segsum = family == "segsum", compact = family == "compact", frontier = family == "frontier";
  bool map = family == "map_lcg", sum = family == "sum_lcg", scan = family == "scan_lcg", affine = family == "affine_lcg";
  if (!(transpose || gather || st || segsum || compact || frontier || map || sum || scan || affine)) return 2;
#if CUB_VERSION < 200800
  if (map) return 2;  // CUB DeviceTransform needs CCCL 2.8 (CUDA 12.9)
#endif
  if (segsum && n % 16 != 0) return 2;
  if (frontier) {
    n = uint32_t(std::sqrt(double(cells)));
    if (n * n != cells || n > 16384) return 2;
  }
  auto cold_start = std::chrono::steady_clock::now();
  int runtime = 0, driver = 0, dev = 0;
  CUDA_OK(cudaRuntimeGetVersion(&runtime));
  CUDA_OK(cudaDriverGetVersion(&driver));
  CUDA_OK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CUDA_OK(cudaGetDeviceProperties(&prop, dev));
  uint32_t rows = uint32_t(std::sqrt(double(n)));
  while (n % rows != 0) rows--;
  uint32_t cols = n / rows;
  // Host inputs.
  // Two input slots for map and affine: a[0..n) is slot 0, a[n..2n) slot 1.
  size_t elements = frontier ? size_t(n) * n : (map || affine) ? size_t(2) * n : n;
  std::vector<uint32_t> host_a(elements), host_ix(gather ? n : 0);
  for (size_t i = 0; i < elements; i++) host_a[i] = frontier ? 0u : lcg(uint32_t(i % n)) + (i >= n ? 17u : 0u);
  if (affine) {  // interleave (x0, x1) pairs for the Affine scan
    std::vector<uint32_t> pairs(elements);
    for (uint32_t i = 0; i < n; i++) { pairs[2 * size_t(i)] = host_a[i]; pairs[2 * size_t(i) + 1] = host_a[n + i]; }
    host_a.swap(pairs);
  }
  if (frontier) for (uint32_t u = 0; u < n; u++) { host_a[size_t(u) * n + (u + 1) % n] = 1; host_a[size_t(u) * n + (u * 5ull + 1) % n] = 1; }
  for (uint32_t i = 0; i < (gather ? n : 0); i++) host_ix[i] = uint32_t((uint64_t(i) * 2654435761ull) % n);
  uint32_t out_words = segsum ? n / 16 : sum ? 1 : n, segments = n / 16;
  std::vector<uint32_t> host_offsets(segsum ? segments + 1 : 0);
  for (uint32_t s = 0; s < host_offsets.size(); s++) host_offsets[s] = s * 16;
  // Device buffers and library workspace, allocated once before timing.
  uint32_t *a = nullptr, *ix = nullptr, *out = nullptr, *offsets = nullptr, *scratch = nullptr, *count = nullptr;
  CUDA_OK(cudaMalloc(&a, elements * 4));
  CUDA_OK(cudaMalloc(&out, size_t(out_words) * 4));
  CUDA_OK(cudaMalloc(&count, 8));
  if (gather) CUDA_OK(cudaMalloc(&ix, size_t(n) * 4));
  if (segsum) CUDA_OK(cudaMalloc(&offsets, host_offsets.size() * 4));
  if (frontier || affine) CUDA_OK(cudaMalloc(&scratch, size_t(2) * n * 4));
  if (segsum) CUDA_OK(cudaMemcpy(offsets, host_offsets.data(), host_offsets.size() * 4, cudaMemcpyHostToDevice));
  auto compile_start = std::chrono::steady_clock::now();
  void* temp = nullptr;
  size_t temp_bytes = 0;
  if (segsum) CUDA_OK(cub::DeviceSegmentedReduce::Sum(temp, temp_bytes, a, out, int(segments), offsets, offsets + 1));
  if (compact) CUDA_OK(cub::DeviceSelect::If(temp, temp_bytes, a, out, count, int(n), Below{}));
  if (sum) CUDA_OK(cub::DeviceReduce::Sum(temp, temp_bytes, a, out, int(n)));
  if (scan) CUDA_OK(cub::DeviceScan::InclusiveSum(temp, temp_bytes, a, out, int(n)));
  if (affine) CUDA_OK(cub::DeviceScan::InclusiveScan(temp, temp_bytes, (Affine*)a, (Affine*)scratch, Compose{}, int(n)));
  if (temp_bytes) CUDA_OK(cudaMalloc(&temp, temp_bytes));
  CUDA_OK(cudaDeviceSynchronize());
  double compilation_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - compile_start).count();
  cudaEvent_t start_event, stop_event;
  CUDA_OK(cudaEventCreate(&start_event));
  CUDA_OK(cudaEventCreate(&stop_event));
  std::vector<uint32_t> host_out;
  // One call: upload, every launch (device time = summed event intervals), download.
  auto timed = [&](auto launch) -> double {
    if (cudaEventRecord(start_event) != cudaSuccess) return -1.0;
    if (!launch()) return -1.0;
    if (cudaEventRecord(stop_event) != cudaSuccess || cudaEventSynchronize(stop_event) != cudaSuccess) return -1.0;
    float ms = 0;
    if (cudaEventElapsedTime(&ms, start_event, stop_event) != cudaSuccess) return -1.0;
    return ms;
  };
  auto call = [&]() -> double {
    if (cudaMemcpy(a, host_a.data(), elements * 4, cudaMemcpyHostToDevice) != cudaSuccess) return -1.0;
    if (gather && cudaMemcpy(ix, host_ix.data(), size_t(n) * 4, cudaMemcpyHostToDevice) != cudaSuccess) return -1.0;
    double gpu = 0;
    uint32_t words = out_words;
    if (transpose) {
      dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM), block(TILE_DIM, BLOCK_ROWS);
      gpu = timed([&] { transpose_tiled<<<grid, block>>>(out, a, rows, cols); return cudaGetLastError() == cudaSuccess; });
    } else if (gather) {
      gpu = timed([&] { thrust::gather(thrust::device, ix, ix + n, a, out); return cudaGetLastError() == cudaSuccess; });
    } else if (st) {
      dim3 grid((cols + 15) / 16, (rows + 15) / 16), block(16, 16);
      gpu = timed([&] { stencil<<<grid, block>>>(a, out, rows, cols); return cudaGetLastError() == cudaSuccess; });
    } else if (segsum) {
      gpu = timed([&] { return cub::DeviceSegmentedReduce::Sum(temp, temp_bytes, a, out, int(segments), offsets, offsets + 1) == cudaSuccess; });
    } else if (map) {
#if CUB_VERSION >= 200800
      gpu = timed([&] { return cub::DeviceTransform::Transform(cuda::std::tuple{a, a + n}, out, int(n), MapOp{}) == cudaSuccess; });
#endif
    } else if (sum) {
      gpu = timed([&] { return cub::DeviceReduce::Sum(temp, temp_bytes, a, out, int(n)) == cudaSuccess; });
    } else if (scan) {
      gpu = timed([&] { return cub::DeviceScan::InclusiveSum(temp, temp_bytes, a, out, int(n)) == cudaSuccess; });
    } else if (affine) {
      gpu = timed([&] {
        if (cub::DeviceScan::InclusiveScan(temp, temp_bytes, (Affine*)a, (Affine*)scratch, Compose{}, int(n)) != cudaSuccess) return false;
        apply_seed<<<(n + 255) / 256, 256>>>((const Affine*)scratch, out, n);
        return cudaGetLastError() == cudaSuccess;
      });
    } else if (compact) {
      gpu = timed([&] { return cub::DeviceSelect::If(temp, temp_bytes, a, out, count, int(n), Below{}) == cudaSuccess; });
      uint32_t selected = 0;
      if (cudaMemcpy(&selected, count, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return -1.0;
      words = selected;
    } else {
      // scratch holds frontier[n] then next[n]; out holds seen; count[0] is live.
      uint32_t one = 1;
      if (cudaMemset(out, 0, size_t(n) * 4) != cudaSuccess || cudaMemset(scratch, 0, size_t(2) * n * 4) != cudaSuccess
          || cudaMemcpy(scratch, &one, 4, cudaMemcpyHostToDevice) != cudaSuccess) return -1.0;
      uint32_t blocks = uint32_t((uint64_t(n) * 32 + 255) / 256);
      for (int round = 0; round < 16 && gpu >= 0; round++) {
        uint32_t live = 0;
        // The whole round is timed: the live-count reset, the round kernel and
        // the device-side frontier rotation (next becomes frontier, next is cleared).
        double t = timed([&] {
          if (cudaMemsetAsync(count, 0, 4) != cudaSuccess) return false;
          frontier_round<<<blocks, 256>>>(a, out, scratch, scratch + n, count, n);
          return cudaGetLastError() == cudaSuccess
              && cudaMemcpyAsync(scratch, scratch + n, size_t(n) * 4, cudaMemcpyDeviceToDevice) == cudaSuccess
              && cudaMemsetAsync(scratch + n, 0, size_t(n) * 4) == cudaSuccess;
        });
        gpu = t < 0 ? -1.0 : gpu + t;
        if (t < 0 || cudaMemcpy(&live, count, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return -1.0;
        if (live == 0) break;
      }
    }
    host_out.resize(words);
    if (words && cudaMemcpy(host_out.data(), out, size_t(words) * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return -1.0;
    return gpu;
  };
  if (call() < 0) return 2;
  double cold_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cold_start).count();
  for (int i = 0; i < 10; i++) if (call() < 0) return 2;
  std::vector<double> device_ms, host_ms;
  for (int i = 0; i < 30; i++) {
    auto start = std::chrono::steady_clock::now();
    double gpu = call();
    if (gpu < 0) return 2;
    device_ms.push_back(gpu);
    host_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
  }
  FILE* file = std::fopen(argv[3], "wb");
  if (!file) return 2;
  size_t bytes = host_out.size() * 4;
  bool wrote = bytes == 0 || std::fwrite(host_out.data(), 1, bytes, file) == bytes;
  if (std::fclose(file) != 0 || !wrote) return 2;
  const char* name = transpose ? "NVIDIA reference tiled transpose (port)" : gather ? "NVIDIA Thrust gather"
      : st ? "Handwritten CUDA stencil" : segsum ? "NVIDIA CUB DeviceSegmentedReduce"
      : compact ? "NVIDIA CUB DeviceSelect" : map ? "NVIDIA CUB DeviceTransform" : sum ? "NVIDIA CUB DeviceReduce"
      : scan ? "NVIDIA CUB DeviceScan" : affine ? "NVIDIA CUB DeviceScan affine compose"
      : "Handwritten CUDA level-synchronous frontier";
  unsigned long input_bytes = (unsigned long)elements * 4 + (gather ? n * 4ul : 0ul);
  std::printf("{\"schema_version\":3,\"baseline\":\"%s\",\"vendor_versions\":{\"cuda_runtime\":%d,\"cuda_headers\":%d,\"cuda_driver\":%d,\"cub\":%d,\"thrust\":%d,\"device\":\"%s\",\"compute_capability\":\"sm_%d%d\",\"framework\":\"CUDA runtime with CCCL\"},\"family\":\"%s\",\"n\":%u,\"warmups\":10,\"samples\":30,\"input_recipe\":\"lcg1664525_1013904223_v1\",\"input_bytes\":%lu,\"output_bytes\":%lu,\"compilation_ms\":%.9g,\"cold_process_ms\":%.9g,\"device_resident_boundary\":\"sum of cuda event intervals around the library calls or kernels\",\"host_to_host_boundary\":\"pageable host input copy through host output copy, after workspace sizing\",\"device_samples_ms\":[",
              name, runtime, CUDART_VERSION, driver, CUB_VERSION, THRUST_VERSION, prop.name, prop.major, prop.minor, argv[1], cells,
              input_bytes, (unsigned long)bytes, compilation_ms, cold_ms);
  for (size_t i = 0; i < device_ms.size(); i++) std::printf("%s%.9g", i ? "," : "", device_ms[i]);
  std::printf("],\"host_samples_ms\":[");
  for (size_t i = 0; i < host_ms.size(); i++) std::printf("%s%.9g", i ? "," : "", host_ms[i]);
  std::printf("]}\n");
  return 0;
}
