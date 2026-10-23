// External NVIDIA CUB baseline; host orchestration only, no Invar kernel code.
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <openssl/sha.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CUDA_OK(call) do { auto error=(call); if(error!=cudaSuccess){std::fprintf(stderr,"CUDA baseline failed: %s\n",cudaGetErrorString(error));return 2;} } while(0)
double load_1m() { double load=-1; auto file=std::fopen("/proc/loadavg","r"); if(file) { if(std::fscanf(file,"%lf",&load)!=1)load=-1;std::fclose(file); } return load; }
int main(int argc,char **argv) {
    unsigned maximum=argc>1?std::strtoul(argv[1],nullptr,10):26;
    unsigned samples=argc>2?std::strtoul(argv[2],nullptr,10):5;
    if(maximum<16||maximum>26||samples<3||samples>21)return 2;
    for(unsigned power=16;power<=maximum;++power){
        size_t n=size_t(1)<<power,bytes=n*sizeof(unsigned);unsigned seed=0x5eeda11u^power,state=seed;
        std::vector<unsigned> input(n),ids(n),out(n),values(n);
        for(size_t i=0;i<n;++i){state^=state<<13;state^=state>>17;state^=state<<5;input[i]=i%8==0?(state&255u):state;ids[i]=unsigned(i);}
        unsigned *keys_in,*keys_out,*ids_in,*ids_out;void *temp=nullptr;size_t temp_bytes=0;
        CUDA_OK(cudaMalloc(&keys_in,bytes));CUDA_OK(cudaMalloc(&keys_out,bytes));CUDA_OK(cudaMalloc(&ids_in,bytes));CUDA_OK(cudaMalloc(&ids_out,bytes));
        CUDA_OK(cub::DeviceRadixSort::SortPairs(temp,temp_bytes,keys_in,keys_out,ids_in,ids_out,int(n),0,32));CUDA_OK(cudaMalloc(&temp,temp_bytes));
        cudaEvent_t start,end;CUDA_OK(cudaEventCreate(&start));CUDA_OK(cudaEventCreate(&end));
        std::vector<float> times;std::vector<double> load_before,load_after;unsigned char digest[SHA256_DIGEST_LENGTH];
        for(unsigned sample=0;sample<=samples;++sample){
            double before=load_1m();if(!(before>=0&&before<8)){std::fprintf(stderr,"benchmark withheld: load %.2f must be under 8\n",before);return 75;}load_before.push_back(before);
            CUDA_OK(cudaMemcpy(keys_in,input.data(),bytes,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(ids_in,ids.data(),bytes,cudaMemcpyHostToDevice));
            CUDA_OK(cudaEventRecord(start));CUDA_OK(cub::DeviceRadixSort::SortPairs(temp,temp_bytes,keys_in,keys_out,ids_in,ids_out,int(n),0,32));CUDA_OK(cudaEventRecord(end));CUDA_OK(cudaEventSynchronize(end));
            float ms;CUDA_OK(cudaEventElapsedTime(&ms,start,end));
            double after=load_1m();if(!(after>=0&&after<8)){std::fprintf(stderr,"benchmark withheld: load %.2f reached 8\n",after);return 75;}load_after.push_back(after);
            if(sample){times.push_back(ms);}
            {
                CUDA_OK(cudaMemcpy(out.data(),keys_out,bytes,cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(values.data(),ids_out,bytes,cudaMemcpyDeviceToHost));
                std::vector<bool> seen(n);SHA256_CTX hash;SHA256_Init(&hash);
                for(size_t i=0;i<n;++i){unsigned id=values[i];if(id>=n||seen[id]||input[id]!=out[i]||(i&&!(out[i-1]<out[i]||(out[i-1]==out[i]&&values[i-1]<id))))return 3;seen[id]=true;
                    unsigned pair[2]={out[i],id};SHA256_Update(&hash,pair,sizeof(pair));}
                SHA256_Final(digest,&hash);
            }
        }
        auto ordered=times;std::sort(ordered.begin(),ordered.end());
        std::printf("{\"schema\":\"invar-sort-benchmark-v2\",\"backend\":\"nvcc-cub-12.9.27\",\"power\":%u,\"elements\":%zu,\"seed\":%u,\"samples\":%u,\"warmups\":1,\"median_ms\":%.9g,\"timed_ms\":[",power,n,seed,samples,ordered[samples/2]);
        for(size_t i=0;i<times.size();++i)std::printf("%s%.9g",i?",":"",times[i]);
        std::printf("],\"calibration_id\":\"sort.sort_by_key.u32\",\"load_1m_before\":[");for(size_t i=0;i<load_before.size();++i)std::printf("%s%.2f",i?",":"",load_before[i]);
        std::printf("],\"load_1m_after\":[");for(size_t i=0;i<load_after.size();++i)std::printf("%s%.2f",i?",":"",load_after[i]);
        std::printf("],\"output_sha256\":\"");for(auto b:digest)std::printf("%02x",b);std::printf("\",\"timed_boundary\":\"CUDA events around DeviceRadixSort::SortPairs, excludes reset, allocation, upload, download\"}\n");std::fflush(stdout);
        CUDA_OK(cudaEventDestroy(start));CUDA_OK(cudaEventDestroy(end));CUDA_OK(cudaFree(temp));CUDA_OK(cudaFree(keys_in));CUDA_OK(cudaFree(keys_out));CUDA_OK(cudaFree(ids_in));CUDA_OK(cudaFree(ids_out));
    }
}
