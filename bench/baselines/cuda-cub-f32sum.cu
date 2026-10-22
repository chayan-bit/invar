// External vendor comparison fixture for the v0.2 fsum lane, not Invar code.
// Build: nvcc -std=c++17 -O2 -arch=sm_120 bench/baselines/cuda-cub-f32sum.cu -o cuda-cub-f32sum
// Usage: cuda-cub-f32sum N   (prints the resident median of 30 timed
// cub::DeviceReduce::Sum calls after 10 warmups, in milliseconds)
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_OK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); return 2; } } while (0)

int main(int argc, char** argv) {
  if (argc != 2) { std::fprintf(stderr, "usage: cuda-cub-f32sum N\n"); return 2; }
  long n = std::strtol(argv[1], nullptr, 10);
  if (n <= 0 || n > (1L << 28)) return 2;
  std::vector<float> host(n);
  for (long i = 0; i < n; ++i) host[i] = 1.0f + float(i % 1024) / 1024.0f;
  float *input = nullptr, *output = nullptr;
  CUDA_OK(cudaMalloc(&input, n * sizeof(float)));
  CUDA_OK(cudaMalloc(&output, sizeof(float)));
  CUDA_OK(cudaMemcpy(input, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  void* scratch = nullptr;
  size_t scratch_bytes = 0;
  CUDA_OK(cub::DeviceReduce::Sum(scratch, scratch_bytes, input, output, int(n)));
  CUDA_OK(cudaMalloc(&scratch, scratch_bytes));
  cudaEvent_t start, end;
  CUDA_OK(cudaEventCreate(&start));
  CUDA_OK(cudaEventCreate(&end));
  std::vector<float> times;
  for (int sample = 0; sample < 40; ++sample) {
    CUDA_OK(cudaEventRecord(start));
    CUDA_OK(cub::DeviceReduce::Sum(scratch, scratch_bytes, input, output, int(n)));
    CUDA_OK(cudaEventRecord(end));
    CUDA_OK(cudaEventSynchronize(end));
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, start, end));
    if (sample >= 10) times.push_back(ms);
  }
  std::sort(times.begin(), times.end());
  std::printf("cub f32sum n=%ld median_ms=%.4f\n", n, (times[14] + times[15]) / 2.0f);
  return 0;
}
