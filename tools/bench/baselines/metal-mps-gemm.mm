// External Apple MPS comparison fixture; f32 policy compatibility is pending M4-01.
// Build: xcrun clang++ -std=c++17 -O2 -fobjc-arc bench/baselines/metal-mps-gemm.mm -framework Foundation -framework Metal -framework MetalPerformanceShaders -o metal-mps-gemm
// Usage: metal-mps-gemm f32gemm <square dimension 1..128> <output.bin>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc!=4 || std::strcmp(argv[1],"f32gemm")!=0) return 2;
    char* end=nullptr;
    unsigned long parsed=std::strtoul(argv[2],&end,10);
    if (!end || *end || parsed<1 || parsed>128) return 2;
    NSUInteger n=(NSUInteger)parsed;
    auto cold_start=std::chrono::steady_clock::now();
    NSOperatingSystemVersion os=[[NSProcessInfo processInfo] operatingSystemVersion];
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyApple7]) { std::fprintf(stderr,"Apple7 Metal GPU unavailable\n"); return 2; }
    id<MTLCommandQueue> queue=[device newCommandQueue];
    if (!queue) return 2;
    NSUInteger bytes=n*n*sizeof(float);
    id<MTLBuffer> a=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> b=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> c=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!a || !b || !c) return 2;
    std::vector<float> host_a(n*n),host_b(n*n),host_c(n*n);
    for (NSUInteger i=0;i<n*n;i++) { host_a[i]=float(i%13)/13.0f; host_b[i]=float(i%11)/11.0f; }
    MPSMatrixDescriptor* descriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:n columns:n rowBytes:n*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrix* left=[[MPSMatrix alloc] initWithBuffer:a descriptor:descriptor];
    MPSMatrix* right=[[MPSMatrix alloc] initWithBuffer:b descriptor:descriptor];
    MPSMatrix* result=[[MPSMatrix alloc] initWithBuffer:c descriptor:descriptor];
    MPSMatrixMultiplication* multiply=[[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:NO resultRows:n resultColumns:n interiorColumns:n alpha:1.0 beta:0.0];
    if (!left || !right || !result || !multiply) return 2;
    auto launch=[&]() -> id<MTLCommandBuffer> {
      id<MTLCommandBuffer> command=[queue commandBuffer];
      [multiply encodeToCommandBuffer:command leftMatrix:left rightMatrix:right resultMatrix:result];
      [command commit];
      [command waitUntilCompleted];
      return command;
    };
    std::memcpy([a contents],host_a.data(),bytes);
    std::memcpy([b contents],host_b.data(),bytes);
    if ([launch() status]!=MTLCommandBufferStatusCompleted) return 2;
    double cold_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cold_start).count();
    for (int i=0;i<10;i++) if ([launch() status]!=MTLCommandBufferStatusCompleted) return 2;
    std::vector<double> device_ms,host_ms;
    for (int i=0;i<30;i++) {
      auto host_start=std::chrono::steady_clock::now();
      std::memcpy([a contents],host_a.data(),bytes);
      std::memcpy([b contents],host_b.data(),bytes);
      id<MTLCommandBuffer> command=launch();
      if ([command status]!=MTLCommandBufferStatusCompleted) return 2;
      std::memcpy(host_c.data(),[c contents],bytes);
      auto host_end=std::chrono::steady_clock::now();
      double elapsed=([command GPUEndTime]-[command GPUStartTime])*1000.0;
      if (elapsed<=0) { std::fprintf(stderr,"Metal GPU timestamps unavailable\n"); return 2; }
      device_ms.push_back(elapsed);
      host_ms.push_back(std::chrono::duration<double,std::milli>(host_end-host_start).count());
    }
    FILE* file=std::fopen(argv[3],"wb");
    if (!file) return 2;
    bool wrote=std::fwrite(host_c.data(),1,bytes,file)==bytes;
    if (std::fclose(file)!=0 || !wrote) return 2;
    auto median=[](std::vector<double> values) {std::sort(values.begin(),values.end());return values[15];};
    std::printf("{\"schema_version\":1,\"baseline\":\"Apple MPSMatrixMultiplication\",\"vendor_versions\":{\"macos\":[%ld,%ld,%ld],\"framework\":\"MetalPerformanceShaders system framework\"},\"family\":\"f32gemm\",\"n\":%lu,\"warmups\":10,\"samples\":30,\"input_bytes\":%lu,\"output_bytes\":%lu,\"workspace_bytes\":0,\"device_median_ms\":%.9g,\"host_median_ms\":%.9g,\"device_samples_ms\":[",(long)os.majorVersion,(long)os.minorVersion,(long)os.patchVersion,(unsigned long)n,(unsigned long)(2*bytes),(unsigned long)bytes,median(device_ms),median(host_ms));
    for (size_t i=0;i<device_ms.size();i++) std::printf("%s%.9g",i?",":"",device_ms[i]);
    std::printf("],\"host_samples_ms\":[");
    for (size_t i=0;i<host_ms.size();i++) std::printf("%s%.9g",i?",":"",host_ms[i]);
    std::printf("],\"cold_process_ms\":%.9g}\n",cold_ms);
    return 0;
  }
}
