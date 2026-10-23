// External handwritten CUDA robotics baselines (GNET-05 comparison D), not Invar-generated code.
// Build: nvcc -std=c++17 -O3 --fmad=false -arch=native -Ibench/baselines bench/baselines/cuda-robotics-handwritten.cu -o cuda-robotics-handwritten
// When nvcc cannot target the GPU (sm_120 needs CUDA 12.8), ship PTX for the driver to JIT:
// replace -arch=native with -gencode arch=compute_90,code=compute_90 (the PTX has no fma,
// so the JIT has nothing to contract).
// Usage: cuda-robotics-handwritten <family> <panda|ur5e|none> <trig 0|1> <seed> <n> <m> <output.bin>
// --fmad=false keeps every multiply and add separately rounded (no contraction), and the
// rational control divides with __fdiv_rn, so outputs are bit-identical to the reference.
#include "robotics-kernels.h"
#include "robotics-tables.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

__global__ void rk_configs(const u32* model, RkParams p, float* out, const float* obs, const u32* count, u32 family) {
  __shared__ float slices[RK_CONFIGS * RK_SLICE];
  u32 local = threadIdx.x / RK_LANES;
  rk_config(family, slices + local * RK_SLICE, model, p, out, obs, count, blockIdx.x * RK_CONFIGS + local,
            threadIdx.x % RK_LANES);
}
__global__ void rk_rollouts(const u32* model, RkParams p, float* out) {
  __shared__ float slices[RK_CONFIGS * RK_SLICE];
  u32 local = threadIdx.x / RK_LANES;
  rk_rollout(slices + local * RK_SLICE, model, p, out, blockIdx.x * RK_CONFIGS + local, threadIdx.x % RK_LANES);
}
__global__ void rk_prologue(float* aux, RkParams p, u32 what) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (what == 0) rk_spheres(aux, p); else rk_sensor(aux, p);
}
__global__ void rk_cloud_scan(float* pts, const float* fr, RkParams p, u32* flags, u32* prefix, u32* totals) {
  __shared__ u32 first[RK_CLOUD_BLOCK], second[RK_CLOUD_BLOCK];
  u32 lane = threadIdx.x, i = blockIdx.x * RK_CLOUD_BLOCK + lane;
  u32 f = i < p.points ? rk_sample(pts, fr, p, i) : 0u;
  first[lane] = f;
  __syncthreads();
  for (u32 offset = 1; offset < RK_CLOUD_BLOCK; offset *= 2) {
    second[lane] = first[lane] + (lane >= offset ? first[lane - offset] : 0u);
    __syncthreads();
    first[lane] = second[lane];
    __syncthreads();
  }
  if (i < p.points) { flags[i] = f; prefix[i] = first[lane] - f; }
  if (lane == RK_CLOUD_BLOCK - 1) totals[blockIdx.x] = first[lane];
}
__global__ void rk_cloud_blocks(u32* totals, u32* count, RkParams p) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  u32 run = 0;
  for (u32 g = 0; g < p.blocks; g++) { u32 t = totals[g]; totals[g] = run; run += t; }
  count[0] = run;
}
__global__ void rk_cloud_scatter(const float* pts, const u32* flags, const u32* prefix, const u32* offsets,
                                 float* dst, RkParams p, u32 base) {
  u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.points || flags[i] == 0u) return;
  u32 w = offsets[i / RK_CLOUD_BLOCK] + prefix[i];
  for (u32 c = 0; c < 3; c++) dst[base + 3 * w + c] = pts[3 * i + c];
}
__global__ void rk_cloud_neighbours(float* out, const u32* count, RkParams p) {
  u32 a = blockIdx.x * blockDim.x + threadIdx.x;
  u32 kept = count[0];
  if (a == 0) out[0] = (float)kept;
  if (a >= p.m) return;
  out[1 + 3 * kept + a] = a < kept ? rk_neighbours(out + 1, kept, a) : 0.0f;
}

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); return 2; } } while (0)

int main(int argc, char** argv) {
  if (argc != 8 && argc != 9) {
    std::fprintf(stderr, "usage: <family> <arm> <trig> <seed> <n> <m> <output.bin> [batch]\n");
    return 2;
  }
  // Each timed sample runs `batch` complete calls back to back; the recorded
  // per-call times are the sample totals divided by `batch`.
  u32 batch = argc == 9 ? (u32)std::strtoul(argv[8], nullptr, 10) : 1;
  if (batch < 1 || batch > 100000) return 2;
  std::string family = argv[1], arm = argv[2];
  u32 trig = (u32)std::strtoul(argv[3], nullptr, 10), seed = (u32)std::strtoul(argv[4], nullptr, 10);
  u32 n = (u32)std::strtoul(argv[5], nullptr, 10), m = (u32)std::strtoul(argv[6], nullptr, 10);
  const RoboticsModel& model = arm == "panda" ? PANDA : UR5E;
  // ponytail: local quality-control caps; raise only after a memory preflight.
  if (n < 1 || n > 1048576 || m > 4096 || trig > 1) return 2;
  u32 fam = family == "fk" || family == "fkpre" ? 0 : family == "jacobian" ? 1 : family == "mass" ? 2
          : family == "collision" ? 3 : family == "pipeline" ? 4 : family == "rollout" ? 5
          : family == "pointcloud" ? 6 : 99;
  if (fam == 99) return 2;
  RkParams p = {seed, n, m, trig, model.nb, model.nj, model.ng, model.tb, model.par, model.fix, model.ax,
                model.jt, model.jx, model.lo, model.span, model.hi, model.mv, model.bt, model.ar, model.gi,
                model.gt, model.tool, 0, 0, 0, 0, 4.0f, 2.0f, 4.0f, 2.0f, 2.0f,
                1.5f, 1.5f, 0.05f, 1.8f, 0.04f, 4.0f};
  u32 points = 0;
  if (fam == 6) { p.cloud_range = 1; points = n; }
  if (fam == 4) {
    p.sx = 2.0f; p.ox = 1.0f; p.sy = 2.0f; p.oy = 1.0f; p.sz = 1.2f;
    p.crop_x = 0.9f; p.crop_y = 0.9f; p.crop_z0 = 0.0f; p.crop_z1 = 1.1f;
    points = m;
  }
  p.points = points;
  p.blocks = (points + 255) / 256;
  size_t words = fam == 0 ? (size_t)n * 12 * model.nb : fam == 1 ? (size_t)n * 6 * model.nj
               : fam == 2 ? (size_t)n * model.nj * model.nj : fam == 3 ? (size_t)n * model.ng * m
               : fam == 4 ? (size_t)n * (model.ng + 1) : fam == 5 ? (size_t)n * m * (model.nj + 1)
               : 1 + 3 * (size_t)n + m;
  auto cold_start = std::chrono::steady_clock::now();
  int runtime = 0, driver = 0;
  CHECK(cudaRuntimeGetVersion(&runtime));
  CHECK(cudaDriverGetVersion(&driver));
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  auto alloc = [](size_t bytes, void** ptr) { return cudaMalloc(ptr, std::max<size_t>(bytes, 16)); };
  u32 *model_d, *flags, *prefix, *totals, *count;
  float *out, *aux, *pts, *kept_points;
  CHECK(alloc(model.count * 4, (void**)&model_d));
  CHECK(cudaMemcpy(model_d, model.words, model.count * 4, cudaMemcpyHostToDevice));
  CHECK(alloc(words * 4, (void**)&out));
  CHECK(alloc(std::max<size_t>(4 * m, 5) * 4, (void**)&aux));
  CHECK(alloc(3 * (size_t)points * 4, (void**)&pts));
  CHECK(alloc((size_t)points * 4, (void**)&flags));
  CHECK(alloc((size_t)points * 4, (void**)&prefix));
  CHECK(alloc((size_t)p.blocks * 4, (void**)&totals));
  CHECK(alloc(4, (void**)&count));
  CHECK(alloc(3 * (size_t)points * 4, (void**)&kept_points));
  size_t workspace_bytes = (std::max<size_t>(4 * m, 5) + 3 * (size_t)points * 2 + 2 * (size_t)points + p.blocks + 1) * 4;
  cudaEvent_t begin, end;
  CHECK(cudaEventCreate(&begin));
  CHECK(cudaEventCreate(&end));
  u32 launches = 0;
  auto launch = [&](u32 reps) -> float {
    launches = 0;
    cudaEventRecord(begin);
    for (u32 rep = 0; rep < reps; rep++) {
    if (fam == 6 || fam == 4) {
      rk_prologue<<<1, 1>>>(aux, p, 1u);
      rk_cloud_scan<<<p.blocks, RK_CLOUD_BLOCK>>>(pts, aux, p, flags, prefix, totals);
      rk_cloud_blocks<<<1, 1>>>(totals, count, p);
      rk_cloud_scatter<<<p.blocks, RK_CLOUD_BLOCK>>>(pts, flags, prefix, totals, fam == 6 ? out : kept_points, p,
                                                     fam == 6 ? 1u : 0u);
      launches += 4;
    }
    if (fam == 6) {
      rk_cloud_neighbours<<<std::max<u32>(1u, (m + 63) / 64), 64>>>(out, count, p);
      launches++;
    } else if (fam == 5) {
      rk_rollouts<<<(n + 15) / 16, RK_THREADS>>>(model_d, p, out);
      launches++;
    } else {
      if (fam == 3) { rk_prologue<<<1, 1>>>(aux, p, 0u); launches++; }
      rk_configs<<<(n + 15) / 16, RK_THREADS>>>(model_d, p, out, fam == 4 ? kept_points : aux, count, fam);
      launches++;
    }
    }
    launches /= reps;
    cudaEventRecord(end);
    if (cudaEventSynchronize(end) != cudaSuccess || cudaGetLastError() != cudaSuccess) return -1.0f;
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, begin, end);
    return ms;
  };
  std::vector<float> host(words);
  size_t produced = words;
  auto download = [&]() -> bool {
    if (fam == 6) {
      u32 kept = 0;
      if (cudaMemcpy(&kept, count, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
      produced = 1 + 3 * (size_t)kept + m;
    }
    return cudaMemcpy(host.data(), out, produced * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
  };
  // Inputs are scalars; the warm host-to-host scope is launch, completion and full download.
  if (launch(1) <= 0 || !download()) return 2;
  double cold_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cold_start).count();
  for (int i = 0; i < 10; i++) if (launch(batch) <= 0) return 2;
  // Resident: `batch` calls between two events. Host-to-host: `batch`
  // complete warm calls (launch, completion, download) back to back.
  std::vector<double> device_ms, host_ms;
  double min_sample_gpu_ms = 1e300;
  for (int i = 0; i < 30; i++) {
    float elapsed = launch(batch);
    if (elapsed <= 0) return 2;
    min_sample_gpu_ms = std::min(min_sample_gpu_ms, (double)elapsed);
    device_ms.push_back((double)elapsed / batch);
    auto start = std::chrono::steady_clock::now();
    for (u32 rep = 0; rep < batch; rep++)
      if (launch(1) <= 0 || !download()) return 2;
    host_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / batch);
  }
  FILE* file = std::fopen(argv[7], "wb");
  if (!file) return 2;
  bool wrote = std::fwrite(host.data(), 4, produced, file) == produced;
  if (std::fclose(file) != 0 || !wrote) return 2;
  auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return (v[14] + v[15]) / 2.0; };
  std::printf("{\"schema_version\":1,\"baseline\":\"Handwritten CUDA robotics (cuRobo-style chain)\",\"vendor_versions\":{\"cuda_runtime\":%d,\"cuda_driver\":%d,\"device\":\"%s\",\"compute_capability\":\"%d.%d\"},\"family\":\"%s\",\"arm\":\"%s\",\"trig\":%u,\"seed\":%u,\"n\":%u,\"m\":%u,\"output_words\":%zu,\"warmups\":10,\"samples\":30,\"batch\":%u,\"min_sample_gpu_ms\":%.9g,\"input_bytes\":%zu,\"output_bytes\":%zu,\"workspace_bytes\":%zu,\"launches_per_call\":%u,\"allocation_policy\":\"all buffers preallocated and reused for warm calls\",\"cold_boundary\":\"fixture main through first completed call; excludes process load\",\"compilation_ms\":0,\"cold_process_ms\":%.9g,\"device_samples_ms\":[",
    runtime, driver, prop.name, prop.major, prop.minor, family.c_str(), arm.c_str(), trig, seed, n, m, produced,
    batch, min_sample_gpu_ms, sizeof(p), produced * 4, workspace_bytes, launches, cold_ms);
  for (size_t i = 0; i < device_ms.size(); i++) std::printf("%s%.9g", i ? "," : "", device_ms[i]);
  std::printf("],\"host_samples_ms\":[");
  for (size_t i = 0; i < host_ms.size(); i++) std::printf("%s%.9g", i ? "," : "", host_ms[i]);
  std::printf("],\"device_median_ms\":%.9g,\"host_median_ms\":%.9g}\n", median(device_ms), median(host_ms));
  return 0;
}
