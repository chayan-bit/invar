// External Apple MPS image reduce/integral comparison for 1D f32 data.
// Build: xcrun clang++ -std=c++17 -O2 -fobjc-arc bench/baselines/metal-mps-image.mm -framework Foundation -framework Metal -framework MetalPerformanceShaders -o metal-mps-image
// Usage: metal-mps-image <f32sum|f32scan> <n 2..4096> <output.bin>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc,char** argv) {
  @autoreleasepool {
    if (argc!=4) return 2;
    bool sum=std::strcmp(argv[1],"f32sum")==0;
    bool scan=std::strcmp(argv[1],"f32scan")==0;
    if (!sum && !scan) return 2;
    char* end=nullptr;
    unsigned long parsed=std::strtoul(argv[2],&end,10);
    if (!end || *end || parsed<2 || parsed>4096) return 2;
    NSUInteger n=(NSUInteger)parsed;
    auto cold_start=std::chrono::steady_clock::now();
    NSOperatingSystemVersion os=[[NSProcessInfo processInfo] operatingSystemVersion];
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyApple7]) return 2;
    id<MTLCommandQueue> queue=[device newCommandQueue];
    if (!queue) return 2;
    MTLPixelFormat pixel_format=MTLPixelFormatR32Float;
    MTLTextureDescriptor* input_desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format width:n height:1 mipmapped:NO];
    input_desc.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
    input_desc.storageMode=MTLStorageModeShared;
    MTLTextureDescriptor* output_desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format width:(sum?2:n) height:1 mipmapped:NO];
    output_desc.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
    output_desc.storageMode=MTLStorageModeShared;
    id<MTLTexture> input=[device newTextureWithDescriptor:input_desc];
    id<MTLTexture> output=[device newTextureWithDescriptor:output_desc];
    if (!input || !output) return 2;
    MPSUnaryImageKernel* kernel=sum? (MPSUnaryImageKernel*)[[MPSImageReduceRowSum alloc] initWithDevice:device]
                                   : (MPSUnaryImageKernel*)[[MPSImageIntegral alloc] initWithDevice:device];
    if (!kernel) return 2;
    std::vector<float> float_input(n),float_output(sum?2:n);
    for (NSUInteger i=0;i<n;i++) float_input[i]=float(i%17)/17.0f;
    const void* host_input=float_input.data();
    void* host_output=float_output.data();
    NSUInteger output_elements=sum?2:n;
    auto upload=[&]() { [input replaceRegion:MTLRegionMake2D(0,0,n,1) mipmapLevel:0 withBytes:host_input bytesPerRow:n*sizeof(uint32_t)]; };
    auto launch=[&]() -> id<MTLCommandBuffer> {
      id<MTLCommandBuffer> command=[queue commandBuffer];
      [kernel encodeToCommandBuffer:command sourceTexture:input destinationTexture:output];
      [command commit]; [command waitUntilCompleted];
      return command;
    };
    auto download=[&]() { [output getBytes:host_output bytesPerRow:output_elements*sizeof(uint32_t) fromRegion:MTLRegionMake2D(0,0,output_elements,1) mipmapLevel:0]; };
    upload();
    if ([launch() status]!=MTLCommandBufferStatusCompleted) return 2;
    double cold_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cold_start).count();
    for (int i=0;i<10;i++) if ([launch() status]!=MTLCommandBufferStatusCompleted) return 2;
    std::vector<double> device_ms,host_ms;
    for (int i=0;i<30;i++) {
      auto host_start=std::chrono::steady_clock::now();
      upload();
      id<MTLCommandBuffer> command=launch();
      if ([command status]!=MTLCommandBufferStatusCompleted) return 2;
      download();
      auto host_end=std::chrono::steady_clock::now();
      double elapsed=([command GPUEndTime]-[command GPUStartTime])*1000.0;
      if (elapsed<=0) return 2;
      device_ms.push_back(elapsed);
      host_ms.push_back(std::chrono::duration<double,std::milli>(host_end-host_start).count());
    }
    FILE* file=std::fopen(argv[3],"wb");
    if (!file) return 2;
    size_t output_bytes=(sum?1:n)*sizeof(float);
    bool wrote=std::fwrite(host_output,1,output_bytes,file)==output_bytes;
    if (std::fclose(file)!=0 || !wrote) return 2;
    auto median=[](std::vector<double> v) {std::sort(v.begin(),v.end());return v[15];};
    std::printf("{\"schema_version\":1,\"baseline\":\"%s\",\"vendor_versions\":{\"macos\":[%ld,%ld,%ld],\"framework\":\"MetalPerformanceShaders system framework\"},\"family\":\"%s\",\"n\":%lu,\"warmups\":10,\"samples\":30,\"input_bytes\":%lu,\"output_bytes\":%zu,\"workspace_bytes\":0,\"device_median_ms\":%.9g,\"host_median_ms\":%.9g,\"device_samples_ms\":[",sum?"Apple MPSImageReduceRowSum":"Apple MPSImageIntegral",(long)os.majorVersion,(long)os.minorVersion,(long)os.patchVersion,argv[1],(unsigned long)n,(unsigned long)(n*sizeof(float)),output_bytes,median(device_ms),median(host_ms));
    for (size_t i=0;i<device_ms.size();i++) std::printf("%s%.9g",i?",":"",device_ms[i]);
    std::printf("],\"host_samples_ms\":[");
    for (size_t i=0;i<host_ms.size();i++) std::printf("%s%.9g",i?",":"",host_ms[i]);
    std::printf("],\"cold_process_ms\":%.9g}\n",cold_ms);
    return 0;
  }
}
