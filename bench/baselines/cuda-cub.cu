// External vendor comparison fixture, not Invar-generated code.
// Build: nvcc -std=c++17 -O2 -arch=sm_86 bench/baselines/cuda-cub.cu -o cuda-cub
// Usage: cuda-cub <map|sum|scan|affine|f32sum|f32scan> <n<=1048576> <output.bin>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#if CUB_VERSION >= 200800
#include <cuda/std/tuple>
#endif
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_OK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); return 2; } } while (0)

struct Affine { uint32_t a, b; };
struct MapOp {
  __host__ __device__ uint32_t operator()(uint32_t a, uint32_t b) const {
    return (a+b)*7u+11u;
  }
};
struct Compose {
  __host__ __device__ Affine operator()(Affine left, Affine right) const {
    return {right.a * left.a, right.a * left.b + right.b};
  }
};
__global__ void apply_seed(const Affine* pairs, uint32_t* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = pairs[i].a * 17u + pairs[i].b;
}
static float percentile(std::vector<float> v, size_t index) {
  std::sort(v.begin(), v.end());
  return v[index];
}

int main(int argc, char** argv) {
  if (argc != 4) { std::fprintf(stderr, "usage: cuda-cub FAMILY N OUTPUT\n"); return 2; }
  const char* family = argv[1];
  char* end = nullptr;
  unsigned long parsed = std::strtoul(argv[2], &end, 10);
  if (!end || *end || parsed == 0 || parsed > 1048576) return 2;
  int n = static_cast<int>(parsed);
  bool map = std::strcmp(family,"map") == 0;
  bool sum = std::strcmp(family,"sum") == 0;
  bool scan = std::strcmp(family,"scan") == 0;
  bool affine = std::strcmp(family,"affine") == 0;
  bool f32sum = std::strcmp(family,"f32sum") == 0;
  bool f32scan = std::strcmp(family,"f32scan") == 0;
  if (!map && !sum && !scan && !affine && !f32sum && !f32scan) return 2;
  auto cold_start = std::chrono::steady_clock::now();
  int driver_version=0; CUDA_OK(cudaDriverGetVersion(&driver_version));
  size_t element = affine ? sizeof(Affine) : sizeof(uint32_t);
  size_t input_bytes = size_t(n) * element;
  size_t output_bytes = (sum || f32sum) ? sizeof(uint32_t) : size_t(n) * sizeof(uint32_t);
  std::vector<uint32_t> input(n), output((sum || f32sum) ? 1 : n);
  std::vector<uint32_t> second;
  std::vector<float> float_input;
  std::vector<Affine> pairs;
  for (int i=0;i<n;i++) input[i] = uint32_t(i)*17u+3u;
  if (map) { second.resize(n); for (int i=0;i<n;i++) second[i]=uint32_t(i)*7u+5u; }
  if (f32sum || f32scan) { float_input.resize(n); for (int i=0;i<n;i++) float_input[i] = float(i%17)/17.0f; }
  if (affine) { pairs.resize(n); for (int i=0;i<n;i++) pairs[i]={uint32_t(i)*3u+2u,uint32_t(i)*7u+5u}; }
  void *d_input=nullptr,*d_input2=nullptr,*d_output=nullptr,*d_temp=nullptr,*d_affine_output=nullptr;
  CUDA_OK(cudaMalloc(&d_input,input_bytes));
  if (map) CUDA_OK(cudaMalloc(&d_input2,input_bytes));
  CUDA_OK(cudaMalloc(&d_output,output_bytes));
  if (affine) CUDA_OK(cudaMalloc(&d_affine_output,input_bytes));
  size_t temp_bytes=0;
  if (sum) CUDA_OK(cub::DeviceReduce::Sum(nullptr,temp_bytes,(uint32_t*)d_input,(uint32_t*)d_output,n));
  if (scan) CUDA_OK(cub::DeviceScan::InclusiveSum(nullptr,temp_bytes,(uint32_t*)d_input,(uint32_t*)d_output,n));
  if (affine) CUDA_OK(cub::DeviceScan::InclusiveScan(nullptr,temp_bytes,(Affine*)d_input,(Affine*)d_affine_output,Compose{},n));
  if (f32sum) CUDA_OK(cub::DeviceReduce::Sum(nullptr,temp_bytes,(float*)d_input,(float*)d_output,n));
  if (f32scan) CUDA_OK(cub::DeviceScan::InclusiveSum(nullptr,temp_bytes,(float*)d_input,(float*)d_output,n));
  size_t vendor_temp_bytes = map ? 0 : std::max<size_t>(temp_bytes,1);
  if (vendor_temp_bytes) CUDA_OK(cudaMalloc(&d_temp,vendor_temp_bytes));
  size_t explicit_intermediate_bytes = affine ? input_bytes : 0;
  size_t total_scratch_bytes = vendor_temp_bytes + explicit_intermediate_bytes;
  cudaEvent_t start,stop;
  CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
  auto launch = [&]() -> cudaError_t {
    if (map) {
#if CUB_VERSION >= 200800
      return cub::DeviceTransform::Transform(cuda::std::tuple{(uint32_t*)d_input,(uint32_t*)d_input2},(uint32_t*)d_output,n,MapOp{});
#else
      thrust::transform(thrust::device,(uint32_t*)d_input,(uint32_t*)d_input+n,(uint32_t*)d_input2,(uint32_t*)d_output,MapOp{});
      return cudaGetLastError();
#endif
    }
    if (sum) return cub::DeviceReduce::Sum(d_temp,temp_bytes,(uint32_t*)d_input,(uint32_t*)d_output,n);
    if (scan) return cub::DeviceScan::InclusiveSum(d_temp,temp_bytes,(uint32_t*)d_input,(uint32_t*)d_output,n);
    if (affine) {
      auto e=cub::DeviceScan::InclusiveScan(d_temp,temp_bytes,(Affine*)d_input,(Affine*)d_affine_output,Compose{},n);
      if (e != cudaSuccess) return e;
      apply_seed<<<(n+255)/256,256>>>((Affine*)d_affine_output,(uint32_t*)d_output,n);
      return cudaGetLastError();
    }
    if (f32sum) return cub::DeviceReduce::Sum(d_temp,temp_bytes,(float*)d_input,(float*)d_output,n);
    return cub::DeviceScan::InclusiveSum(d_temp,temp_bytes,(float*)d_input,(float*)d_output,n);
  };
  const void* host_input = affine ? (const void*)pairs.data() : (f32sum || f32scan ? (const void*)float_input.data() : (const void*)input.data());
  CUDA_OK(cudaMemcpy(d_input,host_input,input_bytes,cudaMemcpyHostToDevice));
  if (map) CUDA_OK(cudaMemcpy(d_input2,second.data(),input_bytes,cudaMemcpyHostToDevice));
  CUDA_OK(launch()); CUDA_OK(cudaDeviceSynchronize());
  float cold_ms=std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-cold_start).count();
  for (int i=0;i<10;i++) { CUDA_OK(launch()); CUDA_OK(cudaDeviceSynchronize()); }
  std::vector<float> device_ms, host_ms;
  for (int i=0;i<30;i++) {
    auto host_start=std::chrono::steady_clock::now();
    CUDA_OK(cudaMemcpy(d_input,host_input,input_bytes,cudaMemcpyHostToDevice));
    if (map) CUDA_OK(cudaMemcpy(d_input2,second.data(),input_bytes,cudaMemcpyHostToDevice));
    CUDA_OK(cudaEventRecord(start)); CUDA_OK(launch()); CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float elapsed=0; CUDA_OK(cudaEventElapsedTime(&elapsed,start,stop));
    CUDA_OK(cudaMemcpy(output.data(),d_output,output_bytes,cudaMemcpyDeviceToHost));
    auto host_end=std::chrono::steady_clock::now();
    device_ms.push_back(elapsed);
    host_ms.push_back(std::chrono::duration<float,std::milli>(host_end-host_start).count());
  }
  FILE* file=std::fopen(argv[3],"wb");
  if (!file || std::fwrite(output.data(),1,output_bytes,file)!=output_bytes || std::fclose(file)!=0) return 2;
  std::printf("{\"schema_version\":1,\"baseline\":\"%s\",\"vendor_versions\":{\"cub\":%d,\"cuda_runtime\":%d,\"driver\":%d},\"family\":\"%s\",\"n\":%d,\"warmups\":10,\"samples\":30,\"input_bytes\":%zu,\"output_bytes\":%zu,\"vendor_temporary_bytes\":%zu,\"explicit_intermediate_bytes\":%zu,\"total_scratch_bytes\":%zu,\"workspace_bytes\":%zu,\"workspace_scope\":\"vendor temporary allocation plus explicit fixture intermediate\",\"device_median_ms\":%.9g,\"host_median_ms\":%.9g,\"device_samples_ms\":[",map?
#if CUB_VERSION >= 200800
    "CUB DeviceTransform"
#else
    "Thrust transform (CUB DeviceTransform unavailable)"
#endif
    :"CUB",CUB_VERSION,CUDART_VERSION,driver_version,family,n,map?input_bytes*2:input_bytes,output_bytes,vendor_temp_bytes,explicit_intermediate_bytes,total_scratch_bytes,total_scratch_bytes,percentile(device_ms,15),percentile(host_ms,15));
  for (size_t i=0;i<device_ms.size();i++) std::printf("%s%.9g",i?",":"",device_ms[i]);
  std::printf("],\"host_samples_ms\":[");
  for (size_t i=0;i<host_ms.size();i++) std::printf("%s%.9g",i?",":"",host_ms[i]);
  std::printf("],\"cold_process_ms\":%.9g}\n",cold_ms);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  cudaFree(d_temp); cudaFree(d_affine_output); cudaFree(d_output); cudaFree(d_input2); cudaFree(d_input);
  return 0;
}
