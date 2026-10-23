/*
The MIT License (MIT)

Copyright (c) 2019-present, cuNSearch contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
// Spatial uniform-grid organization adapted from cuNSearch (MIT), commit
// 4dd5598824bb7dfceae86d89758a6be74e7a9ac4. Copyright 2019-present
// cuNSearch contributors (github.com/InteractiveComputerGraphics/cuNSearch); MIT notice above.
// Invar uses exact signed cell keys, stable radix ordering, inclusive radius,
// self inclusion, true counts, strict binary32 and original-index tie breaking.
typedef unsigned int U;
typedef unsigned long long W;
__device__ bool finite_word(U a) { return (a&0x7fffffffu)<0x7f800000u; }
__device__ U significand(U a) { return (a&0x7fffffu)|((a&0x7f800000u)?0x800000u:0u); }
__device__ int exponent(U a) { return (a&0x7f800000u)?int((a>>23)&255)-150:-149; }
__device__ bool cell_index(U a,U cell,U *out) {
    W num=significand(a), den=significand(cell);
    if (!num) { *out=0; return true; }
    int shift=exponent(a)-exponent(cell); bool neg=(a>>31)!=0;
    W limit=neg?0x80000000ull:0x7fffffffull;
    if(shift>=0) {
        // Compare before shifting; den*limit fits 55 bits. No overflowing shift.
        // Admit the shared magnitude ceiling before shifting. The final
        // floor/ceil check applies the asymmetric positive/negative i32 limits.
        W bound=den*0x80000000ull;
        if(shift>=64 || num>(bound>>shift)) return false;
        num<<=shift;
    } else {
        if(shift<=-24) { *out=neg?0xffffffffu:0u; return true; }
        den<<=-shift;
    }
    W magnitude=num/den+W(neg && num%den!=0);
    if(magnitude>limit) return false;
    *out=neg?U(0u-U(magnitude)):U(magnitude); return true;
}
#define ABI const U *a,const U *b,const U *ids,const U *ranges,U *counts,U *indices,U *distances,const U *p
extern "C" __global__ void spatial_map(ABI) {
    U i=blockIdx.x*blockDim.x+threadIdx.x,n=p[0]; if(i>=n) return;
    U x=a[i],y=a[n+i],z=a[2*n+i];
    counts[i]=0;
    if(!finite_word(x)||!finite_word(y)||!finite_word(z)) { counts[i]=1; return; }
    U cell=p[1];
    if(cell==0 || cell>=0x7f800000u) { counts[i]=2; return; }
    if(!cell_index(x,cell,indices+i)||!cell_index(y,cell,indices+n+i)||!cell_index(z,cell,indices+2*n+i)) counts[i]=2;
}
extern "C" __global__ void spatial_boundaries(ABI) {
    U i=blockIdx.x*blockDim.x+threadIdx.x,n=p[0]; if(i>=n) return;
    counts[i]=i==0 || a[i]!=a[i-1] || a[n+i]!=a[n+i-1] || a[2*n+i]!=a[2*n+i-1];
}
extern "C" __global__ void spatial_query(ABI) {
    U q=blockIdx.x*blockDim.x+threadIdx.x,n=p[0],qn=p[1],k=p[3]; if(q>=qn) return;
    U base=q*k; counts[q]=0;
    for(U j=0;j<k;j++) { indices[base+j]=0xffffffffu; distances[base+j]=0x7f800000u; }
    if(!finite_word(b[q])||!finite_word(b[qn+q])||!finite_word(b[2*qn+q])) return;
    U rb=p[5];
    if(p[4] && ((rb&0x7fffffffu)>0x7f800000u || ((rb>>31) && (rb&0x7fffffffu)))) return;
    if(!p[4] && !k) return;
    float r=__uint_as_float(rb), threshold=__fmul_rn(r,r);
    float x=__uint_as_float(b[q]),y=__uint_as_float(b[qn+q]),z=__uint_as_float(b[2*qn+q]);
    // ponytail: exhaustive occupied cells. Add proven strict-f32 bounds before
    // performance admission; no fixed 27-cell or first-shell approximation.
    for(U c=0;c<p[2];c++) for(U i=ranges[2*c];i<ranges[2*c+1];i++) {
        float dx=__fsub_rn(__uint_as_float(a[i]),x),dy=__fsub_rn(__uint_as_float(a[n+i]),y),dz=__fsub_rn(__uint_as_float(a[2*n+i]),z);
        float d2=__fadd_rn(__fadd_rn(__fmul_rn(dx,dx),__fmul_rn(dy,dy)),__fmul_rn(dz,dz));
        if(p[4] && !(d2<=threshold)) continue;
        counts[q]++;
        U bits=__float_as_uint(d2),id=ids[i],pos=k;
        for(U j=0;j<k;j++) if(bits<distances[base+j] || (bits==distances[base+j] && id<indices[base+j])) { pos=j; break; }
        if(pos==k) continue;
        for(U j=k-1;j>pos;j--) { distances[base+j]=distances[base+j-1]; indices[base+j]=indices[base+j-1]; }
        distances[base+pos]=bits; indices[base+pos]=id;
    }
}
