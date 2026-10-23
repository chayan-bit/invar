// External cuBLAS FP32 comparison fixture; strict ordered policy is incompatible.
// Build: nvcc -std=c++17 -O2 -arch=sm_86 bench/baselines/cuda-cublas-gemm.cu -lcublas -o cuda-cublas-gemm
// Usage: cuda-cublas-gemm f32gemm <square dimension 1..128> <output.bin>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_OK(call) do { cudaError_t e=(call); if (e!=cudaSuccess) { \
  std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); return 2; } } while (0)
#define BLAS_OK(call) do { cublasStatus_t e=(call); if (e!=CUBLAS_STATUS_SUCCESS) { \
  std::fprintf(stderr,"%s: cuBLAS status %d\n",#call,(int)e); return 2; } } while (0)

int main(int argc,char** argv) {
  if (argc!=4 || std::strcmp(argv[1],"f32gemm")!=0) return 2;
  char* end=nullptr;
  unsigned long parsed=std::strtoul(argv[2],&end,10);
  if (!end || *end || parsed<1 || parsed>128) return 2;
  int n=(int)parsed;
  auto cold_start=std::chrono::steady_clock::now();
  int driver_version=0; CUDA_OK(cudaDriverGetVersion(&driver_version));
  size_t bytes=size_t(n)*size_t(n)*sizeof(float);
  std::vector<float> host_a(n*n),host_b(n*n),host_c(n*n);
  for (int i=0;i<n*n;i++) { host_a[i]=float(i%13)/13.0f; host_b[i]=float(i%11)/11.0f; }
  float *a=nullptr,*b=nullptr,*c=nullptr;
  CUDA_OK(cudaMalloc(&a,bytes)); CUDA_OK(cudaMalloc(&b,bytes)); CUDA_OK(cudaMalloc(&c,bytes));
  cublasHandle_t handle=nullptr;
  BLAS_OK(cublasCreate(&handle));
  BLAS_OK(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
  int cublas_version=0; BLAS_OK(cublasGetVersion(handle,&cublas_version));
  cudaEvent_t start,stop; CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
  const float alpha=1.0f,beta=0.0f;
  auto launch=[&]() -> cublasStatus_t {
    // Column-major B^T*A^T has the same bytes as row-major A*B.
    return cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&alpha,b,n,a,n,&beta,c,n);
  };
  CUDA_OK(cudaMemcpy(a,host_a.data(),bytes,cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(b,host_b.data(),bytes,cudaMemcpyHostToDevice));
  BLAS_OK(launch()); CUDA_OK(cudaDeviceSynchronize());
  double cold_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cold_start).count();
  for (int i=0;i<10;i++) { BLAS_OK(launch()); CUDA_OK(cudaDeviceSynchronize()); }
  std::vector<double> device_ms,host_ms;
  for (int i=0;i<30;i++) {
    auto host_start=std::chrono::steady_clock::now();
    CUDA_OK(cudaMemcpy(a,host_a.data(),bytes,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b,host_b.data(),bytes,cudaMemcpyHostToDevice));
    CUDA_OK(cudaEventRecord(start)); BLAS_OK(launch()); CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float elapsed=0; CUDA_OK(cudaEventElapsedTime(&elapsed,start,stop));
    CUDA_OK(cudaMemcpy(host_c.data(),c,bytes,cudaMemcpyDeviceToHost));
    auto host_end=std::chrono::steady_clock::now();
    device_ms.push_back(elapsed);
    host_ms.push_back(std::chrono::duration<double,std::milli>(host_end-host_start).count());
  }
  FILE* file=std::fopen(argv[3],"wb");
  if (!file) return 2;
  bool wrote=std::fwrite(host_c.data(),1,bytes,file)==bytes;
  if (std::fclose(file)!=0 || !wrote) return 2;
  auto median=[](std::vector<double> v) { std::sort(v.begin(),v.end()); return v[15]; };
  std::printf("{\"schema_version\":1,\"baseline\":\"cuBLAS SGEMM\",\"vendor_versions\":{\"cublas\":%d,\"cuda_runtime\":%d,\"driver\":%d,\"math_mode\":\"pedantic\"},\"family\":\"f32gemm\",\"n\":%d,\"warmups\":10,\"samples\":30,\"input_bytes\":%zu,\"output_bytes\":%zu,\"workspace_bytes\":0,\"device_median_ms\":%.9g,\"host_median_ms\":%.9g,\"device_samples_ms\":[",cublas_version,CUDART_VERSION,driver_version,n,bytes*2,bytes,median(device_ms),median(host_ms));
  for (size_t i=0;i<device_ms.size();i++) std::printf("%s%.9g",i?",":"",device_ms[i]);
  std::printf("],\"host_samples_ms\":[");
  for (size_t i=0;i<host_ms.size();i++) std::printf("%s%.9g",i?",":"",host_ms[i]);
  std::printf("],\"cold_process_ms\":%.9g}\n",cold_ms);
  cudaEventDestroy(start); cudaEventDestroy(stop); cublasDestroy(handle);
  cudaFree(c); cudaFree(b); cudaFree(a);
  return 0;
}
