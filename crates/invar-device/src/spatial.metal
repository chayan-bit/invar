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
// cuNSearch contributors. See docs/design/v02-spatial.md and upstream MIT notice.
// Invar uses exact signed cell keys, stable radix ordering, inclusive radius,
// self inclusion, true counts, strict binary32 and original-index tie breaking.
typedef uint U;
typedef ulong W;
inline bool finite_word(U a) { return (a&0x7fffffffu)<0x7f800000u; }
inline U significand(U a) { return (a&0x7fffffu)|((a&0x7f800000u)?0x800000u:0u); }
inline int exponent(U a) { return (a&0x7f800000u)?int((a>>23)&255)-150:-149; }
inline bool cell_index(U a,U cell,thread U *out) {
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
#define ABI device const U *a [[buffer(0)]],device const U *b [[buffer(1)]],device const U *ids [[buffer(2)]],device const U *ranges [[buffer(3)]],device U *counts [[buffer(4)]],device U *indices [[buffer(5)]],device U *distances [[buffer(6)]],constant U *p [[buffer(7)]],U gid [[thread_position_in_grid]]

kernel void spatial_map(ABI) {
    U i=gid,n=p[0]; if(i>=n) return;
    U x=a[i],y=a[n+i],z=a[2*n+i]; counts[i]=0;
    if(!finite_word(x)||!finite_word(y)||!finite_word(z)) { counts[i]=1; return; }
    U cell=p[1]; if(cell==0 || cell>=0x7f800000u) { counts[i]=2; return; }
    U kx=0,ky=0,kz=0;
    if(!cell_index(x,cell,&kx)||!cell_index(y,cell,&ky)||!cell_index(z,cell,&kz)) counts[i]=2;
    indices[i]=kx; indices[n+i]=ky; indices[2*n+i]=kz;
}
kernel void spatial_boundaries(ABI) {
    U i=gid,n=p[0]; if(i>=n) return;
    counts[i]=i==0 || a[i]!=a[i-1] || a[n+i]!=a[n+i-1] || a[2*n+i]!=a[2*n+i-1];
}
// Native operations are safe only in this proved normal-or-zero window:
// every nonzero subtraction has magnitude >=2^-63, its square >=2^-126;
// the three-square sum is <2^124. Thus no intermediate can flush or overflow.
// Contract(off), fast-math off and the per-process normal-operation D1 probe
// are additionally required. All other inputs use the shared integer core.
inline bool spatial_normal_domain(U a) {
    a &= 0x7fffffffu; return a==0 || (a>=0x2b800000u && a<=0x5d800000u);
}
inline U spatial_distance(U px,U py,U pz,U qx,U qy,U qz,bool native) {
    if(native && spatial_normal_domain(px) && spatial_normal_domain(py) && spatial_normal_domain(pz)
        && spatial_normal_domain(qx) && spatial_normal_domain(qy) && spatial_normal_domain(qz)) {
        float dx=as_type<float>(px)-as_type<float>(qx);
        float dy=as_type<float>(py)-as_type<float>(qy);
        float dz=as_type<float>(pz)-as_type<float>(qz);
        float xx=dx*dx,yy=dy*dy,zz=dz*dz;
        float xy=xx+yy; return as_type<U>(xy+zz);
    }
    gnet_u dx=gnet_f32_sub(px,qx),dy=gnet_f32_sub(py,qy),dz=gnet_f32_sub(pz,qz);
    return U(gnet_f32_add(gnet_f32_add(gnet_f32_mul(dx,dx),gnet_f32_mul(dy,dy)),gnet_f32_mul(dz,dz)));
}
kernel void spatial_query(ABI) {
    U q=gid,n=p[0],qn=p[1],k=p[3]; if(q>=qn) return;
    U base=q*k; counts[q]=0;
    for(U j=0;j<k;j++) { indices[base+j]=0xffffffffu; distances[base+j]=0x7f800000u; }
    if(!finite_word(b[q])||!finite_word(b[qn+q])||!finite_word(b[2*qn+q])) return;
    U rb=p[5];
    if(p[4] && ((rb&0x7fffffffu)>0x7f800000u || ((rb>>31) && (rb&0x7fffffffu)))) return;
    if(!p[4] && !k) return;
    U threshold=U(gnet_f32_mul(rb,rb));
    // ponytail: exhaustive occupied cells; qualify conservative pruning before
    // claiming production performance. Never stop on a first shell or tie.
    for(U c=0;c<p[2];c++) for(U i=ranges[2*c];i<ranges[2*c+1];i++) {
        U bits=spatial_distance(a[i],a[n+i],a[2*n+i],b[q],b[qn+q],b[2*qn+q],p[7]!=0);
        if(p[4] && bits>threshold) continue;
        counts[q]++;
        U id=ids[i],pos=k;
        for(U j=0;j<k;j++) if(bits<distances[base+j] || (bits==distances[base+j] && id<indices[base+j])) { pos=j; break; }
        if(pos==k) continue;
        for(U j=k-1;j>pos;j--) { distances[base+j]=distances[base+j-1]; indices[base+j]=indices[base+j-1]; }
        distances[base+pos]=bits; indices[base+pos]=id;
    }
}
