// External handwritten Metal affine-scan quality control, not Invar-generated code.
// Build: xcrun clang++ -std=c++17 -O2 -fobjc-arc bench/baselines/metal-affine-handwritten.mm -framework Foundation -framework Metal -o metal-affine-handwritten
// Usage: metal-affine-handwritten affine <n 1..1048576> <output.bin>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct Pair { uint32_t a, b; };

static NSString* const shader = @R"metal(
#include <metal_stdlib>
using namespace metal;
struct Pair { uint a, b; };
inline Pair compose(Pair left, Pair right) {
  return {right.a * left.a, right.a * left.b + right.b};
}
kernel void scan_blocks(const device Pair* input [[buffer(0)]],
                        device Pair* prefix [[buffer(1)]],
                        device Pair* totals [[buffer(2)]],
                        constant uint& n [[buffer(3)]],
                        uint lane [[thread_index_in_threadgroup]],
                        uint group [[threadgroup_position_in_grid]]) {
  threadgroup Pair first[256], second[256];
  uint i = group * 256 + lane;
  first[lane] = i < n ? input[i] : Pair{1, 0};
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint offset=1; offset<256; offset*=2) {
    Pair prior = lane >= offset ? first[lane-offset] : Pair{1, 0};
    second[lane] = compose(prior, first[lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    first[lane] = second[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (i < n) prefix[i] = first[lane];
  if (lane == 255) totals[group] = first[lane];
}
kernel void add_block_prefix(device Pair* prefix [[buffer(0)]],
                             const device Pair* scanned_totals [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  uint group = i / 256;
  if (group) prefix[i] = compose(scanned_totals[group-1], prefix[i]);
}
kernel void emit_result(const device Pair* prefix [[buffer(0)]],
                        device uint* output [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
  if (i < n) output[i] = prefix[i].a * 17u + prefix[i].b;
}
)metal";

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc != 4 || std::strcmp(argv[1], "affine") != 0) return 2;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(argv[2], &end, 10);
    // ponytail: local quality-control cap; raise only after target memory preflight for the final sweep.
    if (!end || *end || parsed < 1 || parsed > 1048576) return 2;
    NSUInteger n = (NSUInteger)parsed;
    NSUInteger input_bytes = n * sizeof(Pair), output_bytes = n * sizeof(uint32_t);
    auto cold_start = std::chrono::steady_clock::now();
    NSOperatingSystemVersion os = [[NSProcessInfo processInfo] operatingSystemVersion];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyApple7]) return 2;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return 2;
    auto compile_start = std::chrono::steady_clock::now();
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:shader options:nil error:&error];
    if (!library) { std::fprintf(stderr, "Metal library compilation failed: %s\n", error.localizedDescription.UTF8String); return 2; }
    auto pipeline = [&](NSString* name) -> id<MTLComputePipelineState> {
      id<MTLFunction> function = [library newFunctionWithName:name];
      return function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
    };
    id<MTLComputePipelineState> scan = pipeline(@"scan_blocks");
    id<MTLComputePipelineState> add = pipeline(@"add_block_prefix");
    id<MTLComputePipelineState> emit = pipeline(@"emit_result");
    if (!scan || !add || !emit || scan.maxTotalThreadsPerThreadgroup < 256) return 2;
    double compilation_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-compile_start).count();
    id<MTLBuffer> input = [device newBufferWithLength:input_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
    if (!input || !output) return 2;
    NSMutableArray<id<MTLBuffer>>* prefixes = [NSMutableArray array];
    NSMutableArray<id<MTLBuffer>>* totals = [NSMutableArray array];
    std::vector<NSUInteger> counts;
    NSUInteger workspace_bytes = 0;
    for (NSUInteger count=n;;) {
      NSUInteger groups = (count+255)/256;
      id<MTLBuffer> prefix = [device newBufferWithLength:count*sizeof(Pair) options:MTLResourceStorageModeShared];
      id<MTLBuffer> total = [device newBufferWithLength:groups*sizeof(Pair) options:MTLResourceStorageModeShared];
      if (!prefix || !total) return 2;
      [prefixes addObject:prefix]; [totals addObject:total]; counts.push_back(count);
      workspace_bytes += (count+groups)*sizeof(Pair);
      if (groups == 1) break;
      count = groups;
    }
    std::vector<Pair> host_input(n);
    std::vector<uint32_t> host_output(n);
    for (NSUInteger i=0;i<n;i++) host_input[i] = {(uint32_t)i*3u+2u,(uint32_t)i*7u+5u};
    auto upload = [&]() { std::memcpy(input.contents,host_input.data(),input_bytes); };
    auto launch = [&]() -> double {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      for (NSUInteger level=0;level<counts.size();level++) {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:scan];
        [encoder setBuffer:level ? totals[level-1] : input offset:0 atIndex:0];
        [encoder setBuffer:prefixes[level] offset:0 atIndex:1];
        [encoder setBuffer:totals[level] offset:0 atIndex:2];
        uint32_t count = (uint32_t)counts[level];
        [encoder setBytes:&count length:sizeof(count) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake((count+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding];
      }
      for (NSUInteger level=counts.size()-1;level>0;level--) {
        NSUInteger lower = level-1;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:add];
        [encoder setBuffer:prefixes[lower] offset:0 atIndex:0];
        [encoder setBuffer:prefixes[level] offset:0 atIndex:1];
        uint32_t count = (uint32_t)counts[lower];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake((count+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding];
      }
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:emit];
      [encoder setBuffer:prefixes[0] offset:0 atIndex:0];
      [encoder setBuffer:output offset:0 atIndex:1];
      uint32_t count = (uint32_t)n;
      [encoder setBytes:&count length:sizeof(count) atIndex:2];
      [encoder dispatchThreadgroups:MTLSizeMake((n+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [encoder endEncoding];
      [command commit]; [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) return -1.0;
      return (command.GPUEndTime-command.GPUStartTime)*1000.0;
    };
    auto download = [&]() { std::memcpy(host_output.data(),output.contents,output_bytes); };
    upload(); if (launch() <= 0) return 2; download();
    double cold_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cold_start).count();
    for (int i=0;i<10;i++) if (launch() <= 0) return 2;
    std::vector<double> device_ms, host_ms;
    for (int i=0;i<30;i++) {
      auto start = std::chrono::steady_clock::now();
      upload(); double elapsed=launch(); if (elapsed <= 0) return 2; download();
      device_ms.push_back(elapsed);
      host_ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    FILE* file = std::fopen(argv[3],"wb");
    if (!file) return 2;
    bool wrote = std::fwrite(host_output.data(),1,output_bytes,file) == output_bytes;
    if (std::fclose(file) != 0 || !wrote) return 2;
    auto median = [](std::vector<double> v) {std::sort(v.begin(),v.end());return (v[14]+v[15])/2.0;};
    std::printf("{\"schema_version\":1,\"baseline\":\"Handwritten Metal affine quality control\",\"vendor_versions\":{\"macos\":[%ld,%ld,%ld],\"framework\":\"Metal system framework\"},\"family\":\"affine\",\"n\":%lu,\"warmups\":10,\"samples\":30,\"input_bytes\":%lu,\"output_bytes\":%lu,\"workspace_bytes\":%lu,\"launches_per_call\":%lu,\"workspace_scope\":\"explicit pair prefixes and block totals\",\"allocation_policy\":\"all buffers preallocated and reused for warm calls\",\"cold_boundary\":\"fixture main through first completed call; excludes process load\",\"compilation_ms\":%.9g,\"cold_process_ms\":%.9g,\"device_samples_ms\":[",(long)os.majorVersion,(long)os.minorVersion,(long)os.patchVersion,(unsigned long)n,(unsigned long)input_bytes,(unsigned long)output_bytes,(unsigned long)workspace_bytes,(unsigned long)(2*counts.size()),compilation_ms,cold_ms);
    for (size_t i=0;i<device_ms.size();i++) std::printf("%s%.9g",i?",":"",device_ms[i]);
    std::printf("],\"host_samples_ms\":[");
    for (size_t i=0;i<host_ms.size();i++) std::printf("%s%.9g",i?",":"",host_ms[i]);
    std::printf("],\"device_median_ms\":%.9g,\"host_median_ms\":%.9g}\n",median(device_ms),median(host_ms));
    return 0;
  }
}
