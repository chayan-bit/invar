// Provisional v2 MPSGraph reduction comparison for the generated u32 recipe.
// Build: xcrun clang++ -std=c++17 -O2 -fobjc-arc bench/baselines/metal-mpsgraph-reduction-v2.mm -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -o metal-mpsgraph-reduction-v2
// Usage: metal-mpsgraph-reduction-v2 sum <n 1..1048576> <output.bin> <input.bin> [--cold-child]
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
    bool cold_child = argc == 6 && std::strcmp(argv[5], "--cold-child") == 0;
    if ((argc != 5 && !cold_child) || std::strcmp(argv[1], "sum") != 0) return 2;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(argv[2], &end, 10);
    if (!end || *end || parsed < 1 || parsed > 1048576) return 2;
    NSUInteger n = (NSUInteger)parsed;
    NSUInteger input_bytes = n * sizeof(uint32_t);
    auto cold_start = std::chrono::steady_clock::now();
    NSOperatingSystemVersion os = [[NSProcessInfo processInfo] operatingSystemVersion];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyApple7]) return 2;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    MPSGraph* graph = [MPSGraph new];
    MPSGraphTensor* first = [graph placeholderWithShape:@[@(n)] dataType:MPSDataTypeUInt32 name:@"first"];
    MPSGraphTensor* reduced = [graph reductionSumWithTensor:first axis:0 name:nil];
    MPSGraphTensor* seed = [graph constantWithScalar:17.0 dataType:MPSDataTypeUInt32];
    MPSGraphTensor* result = [graph additionWithPrimaryTensor:reduced secondaryTensor:seed name:nil];
    NSDictionary* feeds = @{first:[[MPSGraphShapedType alloc] initWithShape:@[@(n)] dataType:MPSDataTypeUInt32]};
    auto compile_start = std::chrono::steady_clock::now();
    MPSGraphExecutable* executable = [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                                       feeds:feeds targetTensors:@[result]
                                            targetOperations:nil compilationDescriptor:nil];
    if (!queue || !executable) return 2;
    double compilation_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-compile_start).count();
    std::vector<uint32_t> host_input(n), host_output(1);
    for (NSUInteger i=0;i<n;i++) host_input[i] = (uint32_t)i * 1664525u + 1013904223u;
    FILE* input_file = std::fopen(argv[4], "wb");
    if (!input_file) return 2;
    bool input_written = std::fwrite(host_input.data(),1,input_bytes,input_file) == input_bytes;
    if (std::fclose(input_file) != 0 || !input_written) return 2;
    auto first_host_start = std::chrono::steady_clock::now();
    id<MTLBuffer> resident_input = [device newBufferWithLength:input_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> resident_output = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    if (!resident_input || !resident_output) return 2;
    std::memcpy(resident_input.contents, host_input.data(), input_bytes);
    auto launch = [&](id<MTLBuffer> in, id<MTLBuffer> out) -> double {
      MPSGraphTensorData* in_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:in shape:@[@(n)] dataType:MPSDataTypeUInt32];
      MPSGraphTensorData* out_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:out shape:@[@1] dataType:MPSDataTypeUInt32];
      if (!in_data || !out_data) return -1.0;
      MPSCommandBuffer* command = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
      [executable encodeToCommandBuffer:command inputsArray:@[in_data] resultsArray:@[out_data] executionDescriptor:nil];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) return -1.0;
      return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
    };
    double first_gpu_ms = launch(resident_input,resident_output);
    if (first_gpu_ms <= 0) return 2;
    std::memcpy(host_output.data(),resident_output.contents,4);
    double first_host_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-first_host_start).count();
    double cold_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cold_start).count();
    std::vector<double> device_ms, host_ms;
    if (cold_child) {
      device_ms.push_back(first_gpu_ms);
      host_ms.push_back(first_host_ms);
    } else {
      for (int i=0;i<10;i++) if (launch(resident_input,resident_output) <= 0) return 2;
      for (int i=0;i<30;i++) {
        double elapsed = launch(resident_input,resident_output);
        if (elapsed <= 0) return 2;
        device_ms.push_back(elapsed);
      }
      auto host_call = [&]() -> double {
        auto started = std::chrono::steady_clock::now();
        id<MTLBuffer> in = [device newBufferWithLength:input_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        if (!in || !out) return -1.0;
        std::memcpy(in.contents,host_input.data(),input_bytes);
        if (launch(in,out) <= 0) return -1.0;
        std::memcpy(host_output.data(),out.contents,4);
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
      };
      for (int i=0;i<10;i++) if (host_call() <= 0) return 2;
      for (int i=0;i<30;i++) {
        double elapsed = host_call();
        if (elapsed <= 0) return 2;
        host_ms.push_back(elapsed);
      }
    }
    FILE* output_file = std::fopen(argv[3], "wb");
    if (!output_file) return 2;
    bool output_written = std::fwrite(host_output.data(),1,4,output_file) == 4;
    if (std::fclose(output_file) != 0 || !output_written) return 2;
    std::printf("{\"schema_version\":2,\"baseline\":\"Apple MPSGraph reductionSum v2\",\"vendor_versions\":{\"macos\":[%ld,%ld,%ld],\"framework\":\"MetalPerformanceShadersGraph system framework\"},\"device_registry_id\":%llu,\"family\":\"sum\",\"n\":%lu,\"warmups\":%d,\"samples\":%d,\"input_bytes\":%lu,\"output_bytes\":4,\"workspace_bytes\":0,\"workspace_scope\":\"explicit buffers only; framework internals unknown\",\"allocation_policy\":\"resident buffers reused; host calls allocate fresh buffers after graph compilation\",\"input_recipe\":\"lcg1664525_1013904223_plus_slot17_v1\",\"device_resident_boundary\":\"metal_command_buffer_gpu_start_end_v1\",\"host_to_host_boundary\":\"metal_runner_after_compile_through_host_output_copy_v1\",\"cold_boundary\":\"fixture main through first completed call; excludes process load\",\"compilation_ms\":%.9g,\"cold_process_ms\":%.9g,\"device_samples_ms\":[",(long)os.majorVersion,(long)os.minorVersion,(long)os.patchVersion,(unsigned long long)[device registryID],(unsigned long)n,cold_child?0:10,cold_child?1:30,(unsigned long)input_bytes,compilation_ms,cold_ms);
    for (size_t i=0;i<device_ms.size();i++) std::printf("%s%.9g",i?",":"",device_ms[i]);
    std::printf("],\"host_samples_ms\":[");
    for (size_t i=0;i<host_ms.size();i++) std::printf("%s%.9g",i?",":"",host_ms[i]);
    std::printf("]}\n");
    return 0;
  }
}
