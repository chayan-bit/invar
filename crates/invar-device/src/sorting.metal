#include <metal_stdlib>
using namespace metal;
// The runtime admits this kernel only with 32-wide SIMD groups and 256 lanes.
uint sort_canonical(uint x,uint kind) {
    return kind==2 && (x&0x7f800000u)==0x7f800000u && (x&0x007fffffu) ? 0x7fc00000u : x;
}
uint sort_digit(uint x,device const uint *p) {
    x=sort_canonical(x,p[3]);
    if(p[3]==1) x^=0x80000000u;
    else if(p[3]==2) x=(x&0x80000000u)?~x:(x^0x80000000u);
    return (x>>p[4])&15u;
}
kernel void sort_histogram(device const uint *keys[[buffer(0)]],device uint *hist[[buffer(1)]],device const uint *p[[buffer(2)]],uint t[[thread_index_in_threadgroup]],uint b[[threadgroup_position_in_grid]]) {
    threadgroup atomic_uint local[16];uint i=b*256+t;
    if(t<16) atomic_store_explicit(&local[t],0,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(i<p[0]) atomic_fetch_add_explicit(&local[sort_digit(keys[i],p)],1,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t<16) hist[t*p[1]+b]=atomic_load_explicit(&local[t],memory_order_relaxed);
}
kernel void sort_scan(device const uint *hist[[buffer(0)]],device uint *prefix[[buffer(1)]],device uint *sums[[buffer(2)]],device const uint *p[[buffer(3)]],uint t[[thread_index_in_threadgroup]],uint b[[threadgroup_position_in_grid]]) {
    threadgroup uint local[256];uint bucket=b/p[2],chunk=b%p[2],g=chunk*256+t;
    uint value=g<p[1]?hist[bucket*p[1]+g]:0;local[t]=value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint step=1;step<256;step*=2) {
        uint add=t>=step?local[t-step]:0;
        threadgroup_barrier(mem_flags::mem_threadgroup);local[t]+=add;threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(g<p[1]) prefix[bucket*p[1]+g]=local[t]-value;
    if(t==255) sums[bucket*p[2]+chunk]=local[t];
}
kernel void sort_carries(device const uint *sums[[buffer(0)]],device uint *carry[[buffer(1)]],device const uint *p[[buffer(2)]],uint t[[thread_index_in_threadgroup]]) {
    threadgroup uint totals[16];uint total=0;
    if(t<16) {for(uint c=0;c<p[2];++c) total+=sums[t*p[2]+c];totals[t]=total;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t<16) {
        uint next=0;for(uint d=0;d<t;++d) next+=totals[d];
        for(uint c=0;c<p[2];++c) {carry[t*p[2]+c]=next;next+=sums[t*p[2]+c];}
    }
}
kernel void sort_scatter(device const uint *keys[[buffer(0)]],device const uint *values[[buffer(1)]],device const uint *prefix[[buffer(2)]],device const uint *carry[[buffer(3)]],device uint *out_keys[[buffer(4)]],device uint *out_values[[buffer(5)]],device const uint *p[[buffer(6)]],uint t[[thread_index_in_threadgroup]],uint b[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]],uint warp[[simdgroup_index_in_threadgroup]]) {
    threadgroup uint warp_counts[8*16];uint i=b*256+t;bool valid=i<p[0];uint key=valid?keys[i]:0,digit=sort_digit(key,p),rank=0;
    for(uint d=0;d<16;++d) {
        uint flag=valid && digit==d;
        uint before=simd_prefix_exclusive_sum(flag),total=simd_sum(flag);
        if(lane==0) warp_counts[warp*16+d]=total;
        if(flag) rank=before;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(valid) {
        for(uint w=0;w<warp;++w) rank+=warp_counts[w*16+digit];
        uint destination=prefix[digit*p[1]+b]+carry[digit*p[2]+b/256]+rank;
        out_keys[destination]=sort_canonical(key,p[3]);out_values[destination]=values[i];
    }
}
