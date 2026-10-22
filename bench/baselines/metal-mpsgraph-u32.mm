// External Apple MPSGraph u32 map/reduce/modular-scan/affine-scan comparison.
// Build: xcrun clang++ -std=c++17 -O2 -fobjc-arc bench/baselines/metal-mpsgraph-u32.mm -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -o metal-mpsgraph-u32
// Usage: metal-mpsgraph-u32 <map|sum|scan|affine> <n 1..1048576> <output.bin>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc != 4) return 2;
    bool map = std::strcmp(argv[1], "map") == 0;
    bool sum = std::strcmp(argv[1], "sum") == 0;
    bool scan = std::strcmp(argv[1], "scan") == 0;
    bool affine = std::strcmp(argv[1], "affine") == 0;
    if (!map && !sum && !scan && !affine) return 2;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(argv[2], &end, 10);
    if (!end || *end || parsed < 1 || parsed > 1048576) return 2;
    NSUInteger n = (NSUInteger)parsed;
    NSUInteger input_bytes = n * sizeof(uint32_t);
    NSUInteger output_bytes = (sum ? 1 : n) * sizeof(uint32_t);
    auto cold_start = std::chrono::steady_clock::now();
    NSOperatingSystemVersion os = [[NSProcessInfo processInfo] operatingSystemVersion];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyApple7]) return 2;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return 2;
    MPSGraph* graph = [MPSGraph new];
    MPSGraphTensor* first = [graph placeholderWithShape:@[@(n)] dataType:MPSDataTypeUInt32 name:@"first"];
    MPSGraphTensor* second = (map || affine) ? [graph placeholderWithShape:@[@(n)] dataType:MPSDataTypeUInt32 name:@"second"] : nil;
    MPSGraphTensor* result = nil;
    if (map) {
      MPSGraphTensor* added = [graph additionWithPrimaryTensor:first secondaryTensor:second name:nil];
      MPSGraphTensor* seven = [graph constantWithScalar:7.0 dataType:MPSDataTypeUInt32];
      MPSGraphTensor* eleven = [graph constantWithScalar:11.0 dataType:MPSDataTypeUInt32];
      MPSGraphTensor* multiplied = [graph multiplicationWithPrimaryTensor:added secondaryTensor:seven name:nil];
      result = [graph additionWithPrimaryTensor:multiplied secondaryTensor:eleven name:nil];
    } else if (affine) {
      MPSGraphTensor* a = first;
      MPSGraphTensor* b = second;
      for (NSUInteger offset=1; offset<n; offset*=2) {
        MPSGraphTensor* identity_a = [graph constantWithScalar:1.0 shape:@[@(offset)] dataType:MPSDataTypeUInt32];
        MPSGraphTensor* identity_b = [graph constantWithScalar:0.0 shape:@[@(offset)] dataType:MPSDataTypeUInt32];
        MPSGraphTensor* prior_a = [graph concatTensors:@[identity_a,[graph sliceTensor:a dimension:0 start:0 length:n-offset name:nil]] dimension:0 name:nil];
        MPSGraphTensor* prior_b = [graph concatTensors:@[identity_b,[graph sliceTensor:b dimension:0 start:0 length:n-offset name:nil]] dimension:0 name:nil];
        MPSGraphTensor* next_b = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:a secondaryTensor:prior_b name:nil] secondaryTensor:b name:nil];
        a = [graph multiplicationWithPrimaryTensor:a secondaryTensor:prior_a name:nil];
        b = next_b;
      }
      MPSGraphTensor* seed = [graph constantWithScalar:17.0 dataType:MPSDataTypeUInt32];
      result = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:a secondaryTensor:seed name:nil] secondaryTensor:b name:nil];
    } else if (sum) {
      result = [graph reductionSumWithTensor:first axis:0 name:nil];
    } else {
      result = [graph cumulativeSumWithTensor:first axis:0 name:nil];
    }
    if (!result) return 2;
    MPSGraphShapedType* input_type = [[MPSGraphShapedType alloc] initWithShape:@[@(n)] dataType:MPSDataTypeUInt32];
    NSDictionary* feeds = (map || affine) ? @{first:input_type, second:input_type} : @{first:input_type};
    auto compile_start = std::chrono::steady_clock::now();
    MPSGraphExecutable* executable = [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                                       feeds:feeds targetTensors:@[result]
                                            targetOperations:nil compilationDescriptor:nil];
    if (!executable) return 2;
    double compilation_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-compile_start).count();
    id<MTLBuffer> a = [device newBufferWithLength:input_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = (map || affine) ? [device newBufferWithLength:input_bytes options:MTLResourceStorageModeShared] : nil;
    id<MTLBuffer> out = [device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
    if (!a || ((map || affine) && !b) || !out) return 2;
    MPSGraphTensorData* a_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:a shape:@[@(n)] dataType:MPSDataTypeUInt32];
    MPSGraphTensorData* b_data = (map || affine) ? [[MPSGraphTensorData alloc] initWithMTLBuffer:b shape:@[@(n)] dataType:MPSDataTypeUInt32] : nil;
    MPSGraphTensorData* out_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:out shape:@[@(sum ? 1 : n)] dataType:MPSDataTypeUInt32];
    if (!a_data || ((map || affine) && !b_data) || !out_data) return 2;
    NSArray<MPSGraphTensorData*>* inputs = (map || affine) ? @[a_data,b_data] : @[a_data];
    std::vector<uint32_t> host_a(n), host_b((map || affine) ? n : 0), host_out(sum ? 1 : n);
    for (NSUInteger i=0;i<n;i++) {
      host_a[i] = affine ? (uint32_t)i * 3u + 2u : (uint32_t)i * 17u + 3u;
      if (map || affine) host_b[i] = (uint32_t)i * 7u + 5u;
    }
    auto upload = [&]() {
      std::memcpy(a.contents, host_a.data(), input_bytes);
      if (map || affine) std::memcpy(b.contents, host_b.data(), input_bytes);
    };
    auto launch = [&]() -> double {
      MPSCommandBuffer* command = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
      [executable encodeToCommandBuffer:command inputsArray:inputs resultsArray:@[out_data] executionDescriptor:nil];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) return -1.0;
      return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
    };
    auto download = [&]() {std::memcpy(host_out.data(), out.contents, output_bytes);};
    upload();
    if (launch() <= 0) return 2;
    download();
    double cold_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cold_start).count();
    for (int i=0;i<10;i++) if (launch() <= 0) return 2;
    std::vector<double> device_ms, host_ms;
    for (int i=0;i<30;i++) {
      auto host_start = std::chrono::steady_clock::now();
      upload();
      double elapsed = launch();
      if (elapsed <= 0) return 2;
      download();
      auto host_end = std::chrono::steady_clock::now();
      device_ms.push_back(elapsed);
      host_ms.push_back(std::chrono::duration<double,std::milli>(host_end-host_start).count());
    }
    FILE* file = std::fopen(argv[3], "wb");
    if (!file) return 2;
    bool wrote = std::fwrite(host_out.data(),1,output_bytes,file) == output_bytes;
    if (std::fclose(file) != 0 || !wrote) return 2;
    auto median = [](std::vector<double> samples) {std::sort(samples.begin(),samples.end());return (samples[14]+samples[15])/2.0;};
    const char* name = map ? "Apple MPSGraph elementwise" : (sum ? "Apple MPSGraph reductionSum" : (scan ? "Apple MPSGraph cumulativeSum" : "Apple MPSGraph affinePrefix"));
    std::printf("{\"schema_version\":1,\"baseline\":\"%s\",\"vendor_versions\":{\"macos\":[%ld,%ld,%ld],\"framework\":\"MetalPerformanceShadersGraph system framework\"},\"family\":\"%s\",\"n\":%lu,\"warmups\":10,\"samples\":30,\"input_bytes\":%lu,\"output_bytes\":%lu,\"workspace_bytes\":0,\"workspace_scope\":\"explicit buffers only; framework internals unknown\",\"allocation_policy\":\"explicit input and output buffers reused; framework internals opaque\",\"cold_boundary\":\"fixture main through first completed call; excludes process load\",\"compilation_ms\":%.9g,\"cold_process_ms\":%.9g,\"device_samples_ms\":[",name, (long)os.majorVersion,(long)os.minorVersion,(long)os.patchVersion,argv[1],(unsigned long)n,(unsigned long)(input_bytes*((map||affine)?2:1)),(unsigned long)output_bytes,compilation_ms,cold_ms);
    for (size_t i=0;i<device_ms.size();i++) std::printf("%s%.9g",i?",":"",device_ms[i]);
    std::printf("],\"host_samples_ms\":[");
    for (size_t i=0;i<host_ms.size();i++) std::printf("%s%.9g",i?",":"",host_ms[i]);
    std::printf("],\"device_median_ms\":%.9g,\"host_median_ms\":%.9g}\n",median(device_ms),median(host_ms));
    return 0;
  }
}
