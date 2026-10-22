// Stable multi-pass radix sort. Each group owns a contiguous 256-item tile.
// Integer encodings avoid floating arithmetic and device FTZ differences.
__device__ unsigned sort_canonical(unsigned x,unsigned kind) {
    return kind==2 && (x&0x7f800000u)==0x7f800000u && (x&0x007fffffu) ? 0x7fc00000u : x;
}
__device__ unsigned sort_digit(unsigned x,const unsigned *p) {
    x=sort_canonical(x,p[3]);
    if(p[3]==1) x^=0x80000000u;
    else if(p[3]==2) x=(x&0x80000000u)?~x:(x^0x80000000u);
    return (x>>p[4])&15u;
}
extern "C" __global__ void sort_histogram(const unsigned *keys,unsigned *hist,const unsigned *p) {
    __shared__ unsigned local[16];
    unsigned t=threadIdx.x,b=blockIdx.x,i=b*256+t;
    if(t<16) local[t]=0;
    __syncthreads();
    if(i<p[0]) atomicAdd(&local[sort_digit(keys[i],p)],1u);
    __syncthreads();
    if(t<16) hist[t*p[1]+b]=local[t];
}
extern "C" __global__ void sort_scan(const unsigned *hist,unsigned *prefix,unsigned *sums,const unsigned *p) {
    __shared__ unsigned local[256];
    unsigned t=threadIdx.x,bucket=blockIdx.x/p[2],chunk=blockIdx.x%p[2],g=chunk*256+t;
    unsigned value=g<p[1]?hist[bucket*p[1]+g]:0;
    local[t]=value;
    __syncthreads();
    for(unsigned step=1;step<256;step*=2) {
        unsigned add=t>=step?local[t-step]:0;
        __syncthreads();local[t]+=add;__syncthreads();
    }
    if(g<p[1]) prefix[bucket*p[1]+g]=local[t]-value;
    if(t==255) sums[bucket*p[2]+chunk]=local[t];
}
extern "C" __global__ void sort_carries(const unsigned *sums,unsigned *carry,const unsigned *p) {
    __shared__ unsigned totals[16];
    unsigned t=threadIdx.x,total=0;
    if(t<16) {for(unsigned c=0;c<p[2];++c) total+=sums[t*p[2]+c];totals[t]=total;}
    __syncthreads();
    if(t<16) {
        unsigned next=0;for(unsigned d=0;d<t;++d) next+=totals[d];
        for(unsigned c=0;c<p[2];++c) {carry[t*p[2]+c]=next;next+=sums[t*p[2]+c];}
    }
}
extern "C" __global__ void sort_scatter(const unsigned *keys,const unsigned *values,const unsigned *prefix,const unsigned *carry,unsigned *out_keys,unsigned *out_values,const unsigned *p) {
    __shared__ unsigned warp_counts[8*16];
    unsigned t=threadIdx.x,b=blockIdx.x,i=b*256+t,lane=t&31u,warp=t>>5;
    bool valid=i<p[0];unsigned key=valid?keys[i]:0,digit=sort_digit(key,p),rank=0;
    for(unsigned d=0;d<16;++d) {
        unsigned mask=__ballot_sync(0xffffffffu,valid && digit==d);
        if(lane==0) warp_counts[warp*16+d]=__popc(mask);
        if(valid && digit==d) rank=__popc(mask&((1u<<lane)-1u));
    }
    __syncthreads();
    if(valid) {
        for(unsigned w=0;w<warp;++w) rank+=warp_counts[w*16+digit];
        unsigned destination=prefix[digit*p[1]+b]+carry[digit*p[2]+b/256]+rank;
        out_keys[destination]=sort_canonical(key,p[3]);
        out_values[destination]=values[i];
    }
}
