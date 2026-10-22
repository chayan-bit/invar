// Independent Apple GPU baselines for the M5-01 families beyond map/reduction/scan.
// transpose and gather use MPSGraph; gather_hw (the M4-05 contract's named
// baseline), stencil, segmented sum, stable compaction and frontier are reviewed
// handwritten Metal kernels pinned by this file's content hash.
// Matrix uses the separate MPSMatrixMultiplication fixture (metal-mps-gemm.mm).
// Build: xcrun clang++ -std=c++17 -O2 -fobjc-arc bench/baselines/metal-families.mm -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -o metal-families
// Usage: metal-families <transpose|gather|gather_hw|stencil|segsum|compact|frontier> <n> <output.bin>
// Inputs: x[i] = i*1664525 + 1013904223 (u32); gather index i*2654435761 mod n;
// segments of 16; compaction keeps x < 2^31 in order; frontier n is the
// adjacency cell count (a square, at most 2^28) of the dense graph with
// v = sqrt(n) vertices and edges u->u+1, u->5u+1 (mod v), started at vertex 0
// and run for at most 16 level-synchronous rounds.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

static const char* kKernels = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void stencil(device const uint* a [[buffer(0)]], device uint* b [[buffer(1)]],
                    constant uint2& dims [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  uint rows = dims.x, cols = dims.y;
  if (p.y >= rows || p.x >= cols) return;
  uint r = p.y, c = p.x, i = r * cols + c;
  uint v = a[i];
  if (r > 0) v += a[i - cols];
  if (r + 1 < rows) v += a[i + cols];
  if (c > 0) v += a[i - 1];
  if (c + 1 < cols) v += a[i + 1];
  b[i] = v;
}
kernel void gather_hw(device const uint* a [[buffer(0)]], device const uint* ix [[buffer(1)]],
                      device uint* out [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i < n) out[i] = a[ix[i]];
}
kernel void segsum(device const uint* a [[buffer(0)]], device uint* out [[buffer(1)]],
                   constant uint& segments [[buffer(2)]], uint s [[thread_position_in_grid]]) {
  if (s >= segments) return;
  uint acc = 0;
  for (uint i = s * 16; i < s * 16 + 16; i++) acc += a[i];
  out[s] = acc;
}
kernel void compact_count(device const uint* a [[buffer(0)]], device uint* counts [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]],
                          uint l [[thread_position_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
  threadgroup uint part[256];
  part[l] = (i < n && a[i] < 0x80000000u) ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 128; s > 0; s >>= 1) {
    if (l < s) part[l] += part[l + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (l == 0) counts[g] = part[0];
}
// One threadgroup of 1024 threads: chunked sums, a threadgroup scan of the
// 1024 chunk totals, then exclusive offsets written back per chunk.
kernel void compact_offsets(device uint* counts [[buffer(0)]], device uint* total [[buffer(1)]],
                            constant uint& groups [[buffer(2)]], uint l [[thread_position_in_threadgroup]]) {
  threadgroup uint scan[1024];
  uint chunk = (groups + 1023) / 1024, lo = min(l * chunk, groups), hi = min(lo + chunk, groups);
  uint sum = 0;
  for (uint g = lo; g < hi; g++) sum += counts[g];
  scan[l] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 1; s < 1024; s <<= 1) {
    uint add = l >= s ? scan[l - s] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scan[l] += add;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint run = scan[l] - sum;
  for (uint g = lo; g < hi; g++) { uint c = counts[g]; counts[g] = run; run += c; }
  if (l == 1023) total[0] = scan[1023];
}
kernel void compact_scatter(device const uint* a [[buffer(0)]], device const uint* offsets [[buffer(1)]],
                            device uint* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                            uint i [[thread_position_in_grid]], uint l [[thread_position_in_threadgroup]],
                            uint g [[threadgroup_position_in_grid]]) {
  threadgroup uint scan[256];
  bool keep = i < n && a[i] < 0x80000000u;
  scan[l] = keep ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 1; s < 256; s <<= 1) {
    uint add = l >= s ? scan[l - s] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scan[l] += add;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (keep) out[offsets[g] + scan[l] - 1] = a[i];
}
kernel void frontier_round(device const uint* edges [[buffer(0)]], device uint* seen [[buffer(1)]],
                           device const uint* frontier [[buffer(2)]], device atomic_uint* next [[buffer(3)]],
                           device atomic_uint* live [[buffer(4)]], constant uint& v [[buffer(5)]],
                           uint u [[thread_position_in_grid]]) {
  if (u >= v || frontier[u] == 0 || seen[u] != 0) return;
  seen[u] = 1;
  for (uint w = 0; w < v; w++) {
    if (edges[u * v + w] == 1) {
      atomic_store_explicit(&next[w], 1u, memory_order_relaxed);
      atomic_store_explicit(&live[0], 1u, memory_order_relaxed);
    }
  }
}
)MSL";

static uint32_t lcg(uint32_t i) { return i * 1664525u + 1013904223u; }

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc != 4) return 2;
    std::string family = argv[1];
    char* end = nullptr;
    unsigned long parsed = std::strtoul(argv[2], &end, 10);
    if (!end || *end || parsed < 1 || parsed > (1ul << 28)) return 2;
    uint32_t n = (uint32_t)parsed, cells = n;
    bool transpose = family == "transpose", graph_gather = family == "gather", gather_hw = family == "gather_hw";
    bool gather = graph_gather || gather_hw, stencil = family == "stencil";
    bool segsum = family == "segsum", compact = family == "compact", frontier = family == "frontier";
    if (!(transpose || gather || stencil || segsum || compact || frontier)) return 2;
    if (segsum && n % 16 != 0) return 2;
    if (frontier) {
      n = (uint32_t)std::sqrt((double)cells);
      if (n * n != cells || n > 16384) return 2;
    }
    auto cold_start = std::chrono::steady_clock::now();
    NSOperatingSystemVersion os = [[NSProcessInfo processInfo] operatingSystemVersion];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyApple7]) return 2;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    uint32_t rows = (uint32_t)std::sqrt((double)n);
    while (n % rows != 0) rows--;
    uint32_t cols = n / rows;
    uint32_t groups = (n + 255) / 256;
    // Host inputs and buffers.
    uint32_t elements = frontier ? n * n : n;
    std::vector<uint32_t> host_a(elements), host_ix(gather ? n : 0);
    for (uint32_t i = 0; i < elements; i++) host_a[i] = frontier ? 0u : lcg(i);
    if (frontier) for (uint32_t u = 0; u < n; u++) { host_a[u * n + (u + 1) % n] = 1; host_a[u * n + (u * 5 + 1) % n] = 1; }
    for (uint32_t i = 0; i < (gather ? n : 0); i++) host_ix[i] = (uint32_t)(((uint64_t)i * 2654435761ull) % n);
    uint32_t out_words = segsum ? n / 16 : n;
    id<MTLBuffer> a = [device newBufferWithLength:elements * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> ix = gather ? [device newBufferWithLength:n * 4 options:MTLResourceStorageModeShared] : nil;
    id<MTLBuffer> out = [device newBufferWithLength:out_words * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> scratch = [device newBufferWithLength:(compact ? groups : 2 * n) * 4 + 8 options:MTLResourceStorageModeShared];
    id<MTLBuffer> total = [device newBufferWithLength:8 options:MTLResourceStorageModeShared];
    if (!a || !out || !scratch || !total || (gather && !ix)) return 2;
    // Pipelines or graphs, compiled once and timed.
    auto compile_start = std::chrono::steady_clock::now();
    MPSGraphExecutable* executable = nil;
    MPSGraphTensorData *a_data = nil, *ix_data = nil, *out_data = nil;
    id<MTLComputePipelineState> p1 = nil, p2 = nil, p3 = nil;
    if (transpose || graph_gather) {
      MPSGraph* graph = [MPSGraph new];
      NSArray<NSNumber*>* shape = transpose ? @[@(rows), @(cols)] : @[@(n)];
      MPSGraphTensor* x = [graph placeholderWithShape:shape dataType:MPSDataTypeUInt32 name:@"x"];
      MPSGraphTensor* idx = gather ? [graph placeholderWithShape:@[@(n)] dataType:MPSDataTypeInt32 name:@"ix"] : nil;
      MPSGraphTensor* result = transpose
          ? [graph transposeTensor:x dimension:0 withDimension:1 name:nil]
          : [graph gatherWithUpdatesTensor:x indicesTensor:idx axis:0 batchDimensions:0 name:nil];
      MPSGraphShapedType* xt = [[MPSGraphShapedType alloc] initWithShape:shape dataType:MPSDataTypeUInt32];
      NSDictionary* feeds = gather ? @{x: xt, idx: [[MPSGraphShapedType alloc] initWithShape:@[@(n)] dataType:MPSDataTypeInt32]} : @{x: xt};
      executable = [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device] feeds:feeds
                              targetTensors:@[result] targetOperations:nil compilationDescriptor:nil];
      if (!executable) return 2;
      a_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:a shape:shape dataType:MPSDataTypeUInt32];
      if (gather) ix_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:ix shape:@[@(n)] dataType:MPSDataTypeInt32];
      out_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:out shape:transpose ? @[@(cols), @(rows)] : @[@(n)] dataType:MPSDataTypeUInt32];
    } else {
      NSError* error = nil;
      MTLCompileOptions* options = [MTLCompileOptions new];
      id<MTLLibrary> library = [device newLibraryWithSource:@(kKernels) options:options error:&error];
      if (!library) return 2;
      auto pipe = [&](const char* name) {
        id<MTLFunction> f = [library newFunctionWithName:@(name)];
        return f ? [device newComputePipelineStateWithFunction:f error:nil] : nil;
      };
      if (stencil) p1 = pipe("stencil");
      if (gather_hw) p1 = pipe("gather_hw");
      if (segsum) p1 = pipe("segsum");
      if (compact) { p1 = pipe("compact_count"); p2 = pipe("compact_offsets"); p3 = pipe("compact_scatter"); }
      if (frontier) p1 = pipe("frontier_round");
      if (!p1 || (compact && (!p2 || !p3))) return 2;
    }
    double compilation_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - compile_start).count();
    // One call: upload, every launch, download. Returns summed GPU time or < 0.
    std::vector<uint32_t> host_out;
    auto commit = [&](std::function<void(id<MTLCommandBuffer>)> encode) -> double {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      encode(command);
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) return -1.0;
      return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
    };
    auto dispatch = [&](id<MTLCommandBuffer> command, id<MTLComputePipelineState> p, std::vector<id<MTLBuffer>> buffers,
                        std::vector<uint32_t> constants, MTLSize grid, MTLSize group) {
      id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
      [enc setComputePipelineState:p];
      NSUInteger slot = 0;
      for (id<MTLBuffer> b : buffers) [enc setBuffer:b offset:0 atIndex:slot++];
      if (!constants.empty()) [enc setBytes:constants.data() length:constants.size() * 4 atIndex:slot];
      [enc dispatchThreads:grid threadsPerThreadgroup:group];
      [enc endEncoding];
    };
    auto call = [&]() -> double {
      std::memcpy(a.contents, host_a.data(), elements * 4);
      if (gather) std::memcpy(ix.contents, host_ix.data(), n * 4);
      double gpu = 0;
      uint32_t words = out_words;
      if (transpose || graph_gather) {
        gpu = commit([&](id<MTLCommandBuffer> c) {
          MPSCommandBuffer* m = [MPSCommandBuffer commandBufferWithCommandBuffer:c];
          [executable encodeToCommandBuffer:m inputsArray:gather ? @[a_data, ix_data] : @[a_data]
                               resultsArray:@[out_data] executionDescriptor:nil];
        });
      } else if (gather_hw) {
        gpu = commit([&](id<MTLCommandBuffer> c) {
          dispatch(c, p1, {a, ix, out}, {n}, MTLSizeMake(n, 1, 1), MTLSizeMake(std::min(n, 256u), 1, 1));
        });
      } else if (stencil) {
        gpu = commit([&](id<MTLCommandBuffer> c) {
          dispatch(c, p1, {a, out}, {rows, cols}, MTLSizeMake(cols, rows, 1), MTLSizeMake(16, 16, 1));
        });
      } else if (segsum) {
        gpu = commit([&](id<MTLCommandBuffer> c) {
          dispatch(c, p1, {a, out}, {n / 16}, MTLSizeMake(n / 16, 1, 1), MTLSizeMake(256, 1, 1));
        });
      } else if (compact) {
        gpu = commit([&](id<MTLCommandBuffer> c) {
          dispatch(c, p1, {a, scratch}, {n}, MTLSizeMake(groups * 256, 1, 1), MTLSizeMake(256, 1, 1));
          dispatch(c, p2, {scratch, total}, {groups}, MTLSizeMake(1024, 1, 1), MTLSizeMake(1024, 1, 1));
          dispatch(c, p3, {a, scratch, out}, {n}, MTLSizeMake(groups * 256, 1, 1), MTLSizeMake(256, 1, 1));
        });
        words = ((uint32_t*)total.contents)[0];
      } else {
        // scratch holds frontier[n] then next[n]; out holds seen; total[0] is live.
        uint32_t* s = (uint32_t*)scratch.contents;
        std::memset(out.contents, 0, n * 4);
        std::memset(s, 0, 2 * n * 4);
        s[0] = 1;
        for (uint32_t round = 0; round < 16 && gpu >= 0; round++) {
          // The whole round runs in one timed command buffer: the live-count
          // reset, the round kernel and the GPU-side frontier rotation.
          double t = commit([&](id<MTLCommandBuffer> c) {
            id<MTLBlitCommandEncoder> reset = [c blitCommandEncoder];
            [reset fillBuffer:total range:NSMakeRange(0, 4) value:0];
            [reset endEncoding];
            id<MTLComputeCommandEncoder> enc = [c computeCommandEncoder];
            [enc setComputePipelineState:p1];
            [enc setBuffer:a offset:0 atIndex:0];
            [enc setBuffer:out offset:0 atIndex:1];
            [enc setBuffer:scratch offset:0 atIndex:2];
            [enc setBuffer:scratch offset:n * 4 atIndex:3];
            [enc setBuffer:total offset:0 atIndex:4];
            [enc setBytes:&n length:4 atIndex:5];
            [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min(n, 256u), 1, 1)];
            [enc endEncoding];
            id<MTLBlitCommandEncoder> rotate = [c blitCommandEncoder];
            [rotate copyFromBuffer:scratch sourceOffset:n * 4 toBuffer:scratch destinationOffset:0 size:n * 4];
            [rotate fillBuffer:scratch range:NSMakeRange(n * 4, n * 4) value:0];
            [rotate endEncoding];
          });
          gpu = t < 0 ? -1.0 : gpu + t;
          if (((uint32_t*)total.contents)[0] == 0) break;
        }
      }
      host_out.assign((uint32_t*)out.contents, (uint32_t*)out.contents + words);
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
    const char* name = transpose ? "Apple MPSGraph transpose" : graph_gather ? "Apple MPSGraph gather"
        : gather_hw ? "Handwritten Metal gather"
        : stencil ? "Handwritten Metal stencil" : segsum ? "Handwritten Metal segmented sum"
        : compact ? "Handwritten Metal stable compaction" : "Handwritten Metal level-synchronous frontier";
    const char* framework = (transpose || graph_gather) ? "MetalPerformanceShadersGraph system framework" : "Metal system framework";
    unsigned long input_bytes = (unsigned long)elements * 4 + (gather ? n * 4ul : 0ul);
    std::printf("{\"schema_version\":3,\"baseline\":\"%s\",\"vendor_versions\":{\"macos\":[%ld,%ld,%ld],\"framework\":\"%s\"},\"family\":\"%s\",\"n\":%u,\"warmups\":10,\"samples\":30,\"input_recipe\":\"lcg1664525_1013904223_v1\",\"input_bytes\":%lu,\"output_bytes\":%lu,\"compilation_ms\":%.9g,\"cold_process_ms\":%.9g,\"device_resident_boundary\":\"sum of metal command buffer gpu start to end\",\"host_to_host_boundary\":\"host input copy through host output copy, after compile\",\"device_samples_ms\":[",
                name, (long)os.majorVersion, (long)os.minorVersion, (long)os.patchVersion, framework, argv[1], cells, input_bytes,
                (unsigned long)bytes, compilation_ms, cold_ms);
    for (size_t i = 0; i < device_ms.size(); i++) std::printf("%s%.9g", i ? "," : "", device_ms[i]);
    std::printf("],\"host_samples_ms\":[");
    for (size_t i = 0; i < host_ms.size(); i++) std::printf("%s%.9g", i ? "," : "", host_ms[i]);
    std::printf("]}\n");
    return 0;
  }
}
