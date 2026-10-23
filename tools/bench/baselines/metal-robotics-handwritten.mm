// External handwritten Metal robotics baselines (GNET-05 comparison D), not Invar-generated code.
// Build: xcrun clang++ -std=c++17 -O2 -fobjc-arc -Ibench/baselines bench/baselines/metal-robotics-handwritten.mm -framework Foundation -framework Metal -o metal-robotics-handwritten
// Usage: metal-robotics-handwritten <robotics-kernels.h> <family> <panda|ur5e|none> <trig 0|1> <seed> <n> <m> <output.bin>
// Kernels: robotics-kernels.h (shared with CUDA); tables: robotics-tables.h (generated).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "robotics-tables.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Host mirror of RkParams (four-byte fields only).
struct RkParams {
  uint32_t seed, n, m, trig;
  uint32_t nb, nj, ng, tb;
  uint32_t par, fix, ax, jt, jx, lo, span, hi, mv, bt, ar, gi, gt, tool;
  uint32_t cloud_range, blocks, points, pad1;
  float sx, ox, sy, oy, sz;
  float crop_x, crop_y, crop_z0, crop_z1, crop_r0, crop_r1;
};

static const char* entries = R"metal(
kernel void rk_configs(device const uint* model [[buffer(0)]], constant RkParams& p [[buffer(1)]],
                       device float* out [[buffer(2)]], device const float* obs [[buffer(3)]],
                       device const uint* count [[buffer(4)]], constant uint& family [[buffer(5)]],
                       uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
  threadgroup float slices[RK_CONFIGS * RK_SLICE];
  uint local = tid / RK_LANES;
  rk_config(family, slices + local * RK_SLICE, model, p, out, obs, count,
            group * RK_CONFIGS + local, tid % RK_LANES);
}
kernel void rk_rollouts(device const uint* model [[buffer(0)]], constant RkParams& p [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
  threadgroup float slices[RK_CONFIGS * RK_SLICE];
  uint local = tid / RK_LANES;
  rk_rollout(slices + local * RK_SLICE, model, p, out, group * RK_CONFIGS + local, tid % RK_LANES);
}
kernel void rk_prologue(device float* aux [[buffer(0)]], constant RkParams& p [[buffer(1)]],
                        constant uint& what [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i != 0) return;
  if (what == 0) rk_spheres(aux, p); else rk_sensor(aux, p);
}
kernel void rk_cloud_scan(device float* pts [[buffer(0)]], device const float* fr [[buffer(1)]],
                          constant RkParams& p [[buffer(2)]], device uint* flags [[buffer(3)]],
                          device uint* prefix [[buffer(4)]], device uint* totals [[buffer(5)]],
                          uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
  threadgroup uint first[RK_CLOUD_BLOCK], second[RK_CLOUD_BLOCK];
  uint i = group * RK_CLOUD_BLOCK + lane;
  uint f = i < p.points ? rk_sample(pts, fr, p, i) : 0u;
  first[lane] = f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint offset = 1; offset < RK_CLOUD_BLOCK; offset *= 2) {
    second[lane] = first[lane] + (lane >= offset ? first[lane - offset] : 0u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    first[lane] = second[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (i < p.points) { flags[i] = f; prefix[i] = first[lane] - f; }
  if (lane == RK_CLOUD_BLOCK - 1) totals[group] = first[lane];
}
kernel void rk_cloud_blocks(device uint* totals [[buffer(0)]], device uint* count [[buffer(1)]],
                            constant RkParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i != 0) return;
  uint run = 0;
  for (uint g = 0; g < p.blocks; g++) { uint t = totals[g]; totals[g] = run; run += t; }
  count[0] = run;
}
kernel void rk_cloud_scatter(device const float* pts [[buffer(0)]], device const uint* flags [[buffer(1)]],
                             device const uint* prefix [[buffer(2)]], device const uint* offsets [[buffer(3)]],
                             device float* dst [[buffer(4)]], constant RkParams& p [[buffer(5)]],
                             constant uint& base [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.points || flags[i] == 0u) return;
  uint w = offsets[i / RK_CLOUD_BLOCK] + prefix[i];
  for (uint c = 0; c < 3; c++) dst[base + 3 * w + c] = pts[3 * i + c];
}
kernel void rk_cloud_neighbours(device float* out [[buffer(0)]], device const uint* count [[buffer(1)]],
                                constant RkParams& p [[buffer(2)]], uint a [[thread_position_in_grid]]) {
  uint kept = count[0];
  if (a == 0) out[0] = (float)kept;
  if (a >= p.m) return;
  out[1 + 3 * kept + a] = a < kept ? rk_neighbours(out + 1, kept, a) : 0.0f;
}
)metal";

static int fail(const char* what) { std::fprintf(stderr, "%s\n", what); return 2; }

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc != 9 && argc != 10)
      return fail("usage: <kernels.h> <family> <arm> <trig> <seed> <n> <m> <output.bin> [batch]");
    // Each timed sample runs `batch` complete calls back to back; the recorded
    // per-call times are the sample totals divided by `batch`.
    uint32_t batch = argc == 10 ? (uint32_t)std::strtoul(argv[9], nullptr, 10) : 1;
    if (batch < 1 || batch > 100000) return fail("batch out of range");
    std::ifstream header_file(argv[1]);
    if (!header_file) return fail("cannot read kernels header");
    std::stringstream header;
    header << header_file.rdbuf();
    std::string family = argv[2], arm = argv[3];
    uint32_t trig = (uint32_t)std::strtoul(argv[4], nullptr, 10);
    uint32_t seed = (uint32_t)std::strtoul(argv[5], nullptr, 10);
    uint32_t n = (uint32_t)std::strtoul(argv[6], nullptr, 10);
    uint32_t m = (uint32_t)std::strtoul(argv[7], nullptr, 10);
    const RoboticsModel& model = arm == "panda" ? PANDA : UR5E;
    // ponytail: local quality-control caps; raise only after a memory preflight.
    if (n < 1 || n > 1048576 || m > 4096 || trig > 1) return fail("input out of range");
    uint32_t fam = family == "fk" || family == "fkpre" ? 0 : family == "jacobian" ? 1 : family == "mass" ? 2
                 : family == "collision" ? 3 : family == "pipeline" ? 4 : family == "rollout" ? 5
                 : family == "pointcloud" ? 6 : 99;
    if (fam == 99) return fail("unknown family");
    RkParams p = {seed, n, m, trig, model.nb, model.nj, model.ng, model.tb, model.par, model.fix, model.ax,
                  model.jt, model.jx, model.lo, model.span, model.hi, model.mv, model.bt, model.ar, model.gi,
                  model.gt, model.tool, 0, 0, 0, 0, 4.0f, 2.0f, 4.0f, 2.0f, 2.0f,
                  1.5f, 1.5f, 0.05f, 1.8f, 0.04f, 4.0f};
    uint32_t points = 0;
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
    NSOperatingSystemVersion os = [[NSProcessInfo processInfo] operatingSystemVersion];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyApple7]) return fail("no Apple7 Metal device");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    auto compile_start = std::chrono::steady_clock::now();
    MTLCompileOptions* options = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
      options.mathMode = MTLMathModeSafe;
      options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    NSString* source = [NSString stringWithUTF8String:(header.str() + entries).c_str()];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) { std::fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 2; }
    auto pipeline = [&](NSString* name) -> id<MTLComputePipelineState> {
      id<MTLFunction> function = [library newFunctionWithName:name];
      return function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
    };
    id<MTLComputePipelineState> configs = pipeline(@"rk_configs"), rollouts = pipeline(@"rk_rollouts"),
        prologue = pipeline(@"rk_prologue"), scan = pipeline(@"rk_cloud_scan"), blocks = pipeline(@"rk_cloud_blocks"),
        scatter = pipeline(@"rk_cloud_scatter"), neighbours = pipeline(@"rk_cloud_neighbours");
    if (!configs || !rollouts || !prologue || !scan || !blocks || !scatter || !neighbours) return fail("pipeline");
    if (configs.maxTotalThreadsPerThreadgroup < 64 || scan.maxTotalThreadsPerThreadgroup < 256) return fail("threads");
    double compilation_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-compile_start).count();
    auto buffer = [&](size_t bytes) { return [device newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared]; };
    id<MTLBuffer> model_buffer = buffer(model.count * 4);
    std::memcpy(model_buffer.contents, model.words, model.count * 4);
    id<MTLBuffer> out = buffer(words * 4), aux = buffer(std::max<size_t>(4 * m, 5) * 4), pts = buffer(3 * (size_t)points * 4),
        flags = buffer((size_t)points * 4), prefix = buffer((size_t)points * 4), totals = buffer((size_t)p.blocks * 4),
        count = buffer(4), kept_points = buffer(3 * (size_t)points * 4);
    size_t workspace_bytes = (size_t)(aux.length + pts.length + flags.length + prefix.length + totals.length + count.length + kept_points.length);
    uint32_t launches = 0;
    auto launch = [&](uint32_t reps) -> double {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      launches = 0;
      for (uint32_t rep = 0; rep < reps; rep++) {
      auto encode = [&](id<MTLComputePipelineState> state, MTLSize groups, MTLSize threads, void (^bind)(id<MTLComputeCommandEncoder>)) {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:state];
        bind(encoder);
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        [encoder endEncoding];
        launches++;
      };
      MTLSize one = MTLSizeMake(1, 1, 1);
      if (fam == 6 || fam == 4) {
        uint32_t what = 1, base = fam == 6 ? 1 : 0;
        encode(prologue, one, one, ^(id<MTLComputeCommandEncoder> e) {
          [e setBuffer:aux offset:0 atIndex:0]; [e setBytes:&p length:sizeof(p) atIndex:1]; [e setBytes:&what length:4 atIndex:2]; });
        encode(scan, MTLSizeMake(p.blocks, 1, 1), MTLSizeMake(256, 1, 1), ^(id<MTLComputeCommandEncoder> e) {
          [e setBuffer:pts offset:0 atIndex:0]; [e setBuffer:aux offset:0 atIndex:1]; [e setBytes:&p length:sizeof(p) atIndex:2];
          [e setBuffer:flags offset:0 atIndex:3]; [e setBuffer:prefix offset:0 atIndex:4]; [e setBuffer:totals offset:0 atIndex:5]; });
        encode(blocks, one, one, ^(id<MTLComputeCommandEncoder> e) {
          [e setBuffer:totals offset:0 atIndex:0]; [e setBuffer:count offset:0 atIndex:1]; [e setBytes:&p length:sizeof(p) atIndex:2]; });
        id<MTLBuffer> dst = fam == 6 ? out : kept_points;
        encode(scatter, MTLSizeMake(p.blocks, 1, 1), MTLSizeMake(256, 1, 1), ^(id<MTLComputeCommandEncoder> e) {
          [e setBuffer:pts offset:0 atIndex:0]; [e setBuffer:flags offset:0 atIndex:1]; [e setBuffer:prefix offset:0 atIndex:2];
          [e setBuffer:totals offset:0 atIndex:3]; [e setBuffer:dst offset:0 atIndex:4]; [e setBytes:&p length:sizeof(p) atIndex:5];
          [e setBytes:&base length:4 atIndex:6]; });
      }
      if (fam == 6) {
        encode(neighbours, MTLSizeMake(m == 0 ? 1 : (m + 63) / 64, 1, 1), MTLSizeMake(64, 1, 1), ^(id<MTLComputeCommandEncoder> e) {
          [e setBuffer:out offset:0 atIndex:0]; [e setBuffer:count offset:0 atIndex:1]; [e setBytes:&p length:sizeof(p) atIndex:2]; });
      } else if (fam == 5) {
        encode(rollouts, MTLSizeMake((n + 15) / 16, 1, 1), MTLSizeMake(64, 1, 1), ^(id<MTLComputeCommandEncoder> e) {
          [e setBuffer:model_buffer offset:0 atIndex:0]; [e setBytes:&p length:sizeof(p) atIndex:1]; [e setBuffer:out offset:0 atIndex:2]; });
      } else {
        if (fam == 3) {
          uint32_t what = 0;
          encode(prologue, one, one, ^(id<MTLComputeCommandEncoder> e) {
            [e setBuffer:aux offset:0 atIndex:0]; [e setBytes:&p length:sizeof(p) atIndex:1]; [e setBytes:&what length:4 atIndex:2]; });
        }
        id<MTLBuffer> points_in = fam == 4 ? kept_points : aux;
        encode(configs, MTLSizeMake((n + 15) / 16, 1, 1), MTLSizeMake(64, 1, 1), ^(id<MTLComputeCommandEncoder> e) {
          [e setBuffer:model_buffer offset:0 atIndex:0]; [e setBytes:&p length:sizeof(p) atIndex:1]; [e setBuffer:out offset:0 atIndex:2];
          [e setBuffer:points_in offset:0 atIndex:3]; [e setBuffer:count offset:0 atIndex:4]; [e setBytes:&fam length:4 atIndex:5]; });
      }
      }
      launches /= reps;
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) return -1.0;
      return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
    };
    std::vector<float> host(words);
    size_t produced = words;
    auto download = [&]() {
      if (fam == 6) produced = 1 + 3 * (size_t)((const uint32_t*)count.contents)[0] + m;
      std::memcpy(host.data(), out.contents, produced * 4);
    };
    // Inputs are scalars (seed and sizes); the warm host-to-host scope is encode, launch,
    // completion and the full output download.
    if (launch(1) <= 0) return fail("first launch failed");
    download();
    double cold_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cold_start).count();
    for (int i = 0; i < 10; i++) if (launch(batch) <= 0) return fail("warmup failed");
    // Resident: one command buffer of `batch` calls. Host-to-host: `batch`
    // complete warm calls (encode, commit, wait, download) back to back.
    std::vector<double> device_ms, host_ms;
    double min_sample_gpu_ms = 1e300;
    for (int i = 0; i < 30; i++) {
      double elapsed = launch(batch);
      if (elapsed <= 0) return fail("sample failed");
      min_sample_gpu_ms = std::min(min_sample_gpu_ms, elapsed);
      device_ms.push_back(elapsed / batch);
      auto start = std::chrono::steady_clock::now();
      for (uint32_t rep = 0; rep < batch; rep++) {
        if (launch(1) <= 0) return fail("sample failed");
        download();
      }
      host_ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count() / batch);
    }
    FILE* file = std::fopen(argv[8], "wb");
    if (!file) return fail("cannot write output");
    bool wrote = std::fwrite(host.data(), 4, produced, file) == produced;
    if (std::fclose(file) != 0 || !wrote) return fail("short write");
    auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return (v[14] + v[15]) / 2.0; };
    std::printf("{\"schema_version\":1,\"baseline\":\"Handwritten Metal robotics (cuRobo-style chain)\",\"vendor_versions\":{\"macos\":[%ld,%ld,%ld],\"framework\":\"Metal system framework\",\"device\":\"%s\"},\"family\":\"%s\",\"arm\":\"%s\",\"trig\":%u,\"seed\":%u,\"n\":%u,\"m\":%u,\"output_words\":%zu,\"warmups\":10,\"samples\":30,\"batch\":%u,\"min_sample_gpu_ms\":%.9g,\"input_bytes\":%zu,\"output_bytes\":%zu,\"workspace_bytes\":%zu,\"launches_per_call\":%u,\"allocation_policy\":\"all buffers preallocated and reused for warm calls\",\"cold_boundary\":\"fixture main through first completed call; excludes process load\",\"compilation_ms\":%.9g,\"cold_process_ms\":%.9g,\"device_samples_ms\":[",
      (long)os.majorVersion, (long)os.minorVersion, (long)os.patchVersion, device.name.UTF8String, family.c_str(), arm.c_str(),
      trig, seed, n, m, produced, batch, min_sample_gpu_ms, sizeof(p), produced * 4, workspace_bytes, launches, compilation_ms, cold_ms);
    for (size_t i = 0; i < device_ms.size(); i++) std::printf("%s%.9g", i ? "," : "", device_ms[i]);
    std::printf("],\"host_samples_ms\":[");
    for (size_t i = 0; i < host_ms.size(); i++) std::printf("%s%.9g", i ? "," : "", host_ms[i]);
    std::printf("],\"device_median_ms\":%.9g,\"host_median_ms\":%.9g}\n", median(device_ms), median(host_ms));
    return 0;
  }
}
