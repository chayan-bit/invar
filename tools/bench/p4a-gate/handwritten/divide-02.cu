#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>
using uint=unsigned int;
#define REF
using wide_t=long long;using uwide_t=unsigned long long;
#define WIDE wide_t
#define UWIDE uwide_t
template<class T> __device__ T max(T a,T b){return a>b?a:b;}
template<class T> __device__ T min(T a,T b){return a<b?a:b;}

struct O {int value;uint trap,span,end,key;};
struct N {int leaf,value,m1,m2;};
__device__ O zero(){O o={0,0,0,0,0xffffffffu};return o;}
__device__ int ck(int a,int b,uint op,uint span,REF O &o){
 if(o.trap)return 0;
 if(op>=3 && b==0){o.trap=2;o.span=span;o.end=span+1;return 0;}
 if(op==4 && a==(-2147483647-1) && b==-1)return 0;
 WIDE v=op==0?WIDE(a)+b:op==1?WIDE(a)-b:op==2?WIDE(a)*b:op==3?WIDE(a)/b:WIDE(a)%b;
 if(v<(-2147483647L-1L)||v>2147483647L){o.trap=1;o.span=span;o.end=span+1;return 0;}return int(v);
}
__device__ int leaf1(int x, REF O &o) {int z0=ck(x,2,4,34,o);int z1=(z0 == 0);if (z1) {int z2=ck(x,2,3,55,o);int z3=ck(z2,16,0,59,o);int z4=ck(z3,10007,4,65,o);return z4;}int z5=ck(x,3,2,90,o);int z6=ck(z5,1,0,94,o);int z7=ck(z6,10007,4,99,o);return z7;}
__device__ N node1(int lo,int hi,int d,REF O &o) {N n={0,0,0,0};int z8=ck(hi,lo,1,161,o);int z9=(z8 <= 3);if (z9) {int z10=ck(lo,6,2,188,o);int z11=ck(z10,hi,0,192,o);int z12=leaf1(z11,o);n.leaf=1;n.value=z12;return n;}int z13=(d <= 0);if (z13) {int z14=ck(hi,lo,1,232,o);int z15=ck(z14,8,0,237,o);int z16=leaf1(z15,o);n.leaf=1;n.value=z16;return n;}int z17=ck(lo,1,0,261,o);int z18=ck(hi,lo,1,272,o);int z19=ck(z18,2,1,277,o);int z20=ck(lo,2,4,293,o);int z21=ck(z20,2,0,297,o);int z22=ck(z21,2,4,302,o);int z23=ck(2,z22,0,287,o);int z24=ck(z19,z23,2,282,o);int z25=ck(z24,7,3,308,o);int z26=ck(z17,z25,0,265,o);int m1=z26;n.m1=m1;return n;}
__device__ O join1(O aa,O bb,O cc) {if(aa.trap)return aa;if(bb.trap)return bb;if(cc.trap)return cc;O o=zero();int a=aa.value,b=bb.value,c=cc.value;int z27=ck(a,3,2,389,o);int z28=ck(z27,b,0,393,o);int z29=ck(z28,100003,4,398,o);o.value=z29;return o;}
__device__ int leaf0(int x, REF O &o) {int z0=(x % 2);int z1=(z0 == 0);if (z1) {int z2=(x / 2);int z3=(z2 + 16);int z4=(z3 % 10007);return z4;}int z5=(x * 3);int z6=(z5 + 1);int z7=(z6 % 10007);return z7;}
__device__ N node0(int lo,int hi,int d,REF O &o) {N n={0,0,0,0};int z8=(hi - lo);int z9=(z8 <= 3);if (z9) {int z10=(lo * 6);int z11=(z10 + hi);int z12=leaf0(z11,o);n.leaf=1;n.value=z12;return n;}int z13=(d <= 0);if (z13) {int z14=(hi - lo);int z15=(z14 + 8);int z16=leaf0(z15,o);n.leaf=1;n.value=z16;return n;}int z17=(lo + 1);int z18=(hi - lo);int z19=(z18 - 2);int z20=(lo % 2);int z21=(z20 + 2);int z22=(z21 % 2);int z23=(2 + z22);int z24=(z19 * z23);int z25=(z24 / 7);int z26=(z17 + z25);int m1=z26;n.m1=m1;return n;}
__device__ O join0(O aa,O bb,O cc) {if(aa.trap)return aa;if(bb.trap)return bb;if(cc.trap)return cc;O o=zero();int a=aa.value,b=bb.value,c=cc.value;int z27=(a * 3);int z28=(z27 + b);int z29=(z28 % 100003);o.value=z29;return o;}
struct F {int lo,hi,d,phase,m1,m2;O a,b;};
__device__ O serial(int lo,int hi,int d) {
 F f[64];uint sp=0;f[0]={lo,hi,d,0,0,0,zero(),zero()};O r=zero();
 for(uint work=0;work<200000;work++){
  if(f[sp].phase==0){O o=zero();N n=node1(f[sp].lo,f[sp].hi,f[sp].d,o);if(o.trap)return o;
   if(n.leaf){r.value=n.value;if(sp==0)return r;sp--;}
   else{f[sp].m1=n.m1;f[sp].m2=n.m2;f[sp].phase=1;F ch={f[sp].lo,n.m1,f[sp].d-1,0,0,0,zero(),zero()};if(sp==63){r.trap=3;return r;}f[++sp]=ch;continue;}
  }
  for(;;){
   if(f[sp].phase==1){f[sp].a=r;f[sp].phase=2;F ch={f[sp].m1,f[sp].hi,f[sp].d-1,0,0,0,zero(),zero()};f[++sp]=ch;break;}
   
   r=join1(f[sp].a,r,zero());if(sp==0)return r;sp--;
  }
 }r.trap=3;return r;
}
__device__ O fallback(int a,int b){O o=zero();int hi=ck(33,a,0,459,o);if(o.trap)return o;return serial(33,hi,b);}

__device__ bool guarded(int a,int b){return a>=0&&a<=20000&&b>=0&&b<=30;}
__device__ N nodep0(int lo,int hi,int d,uint key,REF O &o) {N n={0,0,0,0};int z8=(hi - lo);int z9=(z8 <= 3);if (z9) {if(uint(lo)!=key){n.leaf=1;return n;}int z10=(lo * 6);int z11=(z10 + hi);int z12=leaf0(z11,o);n.leaf=1;n.value=z12;return n;}int z13=(d <= 0);if (z13) {if(uint(lo)!=key){n.leaf=1;return n;}int z14=(hi - lo);int z15=(z14 + 8);int z16=leaf0(z15,o);n.leaf=1;n.value=z16;return n;}int z17=(lo + 1);int z18=(hi - lo);int z19=(z18 - 2);int z20=(lo % 2);int z21=(z20 + 2);int z22=(z21 % 2);int z23=(2 + z22);int z24=(z19 * z23);int z25=(z24 / 7);int z26=(z17 + z25);int m1=z26;n.m1=m1;return n;}
__device__ N nodep1(int lo,int hi,int d,uint key,REF O &o) {N n={0,0,0,0};int z8=ck(hi,lo,1,161,o);int z9=(z8 <= 3);if (z9) {if(uint(lo)!=key){n.leaf=1;return n;}int z10=ck(lo,6,2,188,o);int z11=ck(z10,hi,0,192,o);int z12=leaf1(z11,o);n.leaf=1;n.value=z12;return n;}int z13=(d <= 0);if (z13) {if(uint(lo)!=key){n.leaf=1;return n;}int z14=ck(hi,lo,1,232,o);int z15=ck(z14,8,0,237,o);int z16=leaf1(z15,o);n.leaf=1;n.value=z16;return n;}int z17=ck(lo,1,0,261,o);int z18=ck(hi,lo,1,272,o);int z19=ck(z18,2,1,277,o);int z20=ck(lo,2,4,293,o);int z21=ck(z20,2,0,297,o);int z22=ck(z21,2,4,302,o);int z23=ck(2,z22,0,287,o);int z24=ck(z19,z23,2,282,o);int z25=ck(z24,7,3,308,o);int z26=ck(z17,z25,0,265,o);int m1=z26;n.m1=m1;return n;}
__device__ O path(int a,int b,uint gid,bool wide){
 O o=zero();if(!guarded(a,b)){if(gid==0)return fallback(a,b);return o;}
 if(gid>=uint(max(a,1)))return o;
 int lo=33,hi=33+a,d=b;uint k=33+gid;UWIDE weight=1;
 for(uint depth=0;depth<64;depth++){N n=wide?nodep1(lo,hi,d,k,o):nodep0(lo,hi,d,k,o);if(o.trap){o.key=uint(lo-33)*64+depth;return o;}
  if(n.leaf){if(uint(lo)!=k)return zero();o.value=int((weight*UWIDE(n.value))%100003);return o;}
  if(k<uint(n.m1)){hi=n.m1;d--;weight=(weight*3)%100003;}
  
  else{lo=n.m1;d-=1;weight=(weight*1)%100003;}
 }o.trap=3;return o;
}
__device__ O addo(O a,O b){if(a.trap&&(!b.trap||a.key<=b.key))return a;if(b.trap)return b;O o=zero();o.value=int((UWIDE(a.value)+UWIDE(b.value))%100003);return o;}
__device__ O path_fast(int a,int b,uint gid,bool wide){
 O o=zero();if(!guarded(a,b)){if(gid==0)return fallback(a,b);return o;}
 if(gid>=uint(max(a,1)))return o;
 int lo=33,hi=33+a,d=b;uint k=33+gid;uint weight=1;
 for(uint depth=0;depth<64;depth++){N n=nodep0(lo,hi,d,k,o);if(o.trap){o.key=uint(lo-33)*64+depth;return o;}
  if(n.leaf){if(uint(lo)!=k)return zero();o.value=int((weight*uint(n.value))%100003);return o;}
  if(k<uint(n.m1)){hi=n.m1;d--;weight=(weight*3)%100003;}
  
  else{lo=n.m1;d-=1;weight=(weight*1)%100003;}
 }o.trap=3;return o;
}

struct Args {int a,b;uint D,groups;};
__device__ O replay(Args arg,uint gid,REF uint &stop){
 O o=zero();int lo=33,hi=0,d=arg.b;stop=0;
 if(!guarded(arg.a,arg.b)){if(gid==0)return fallback(arg.a,arg.b);return o;}
 hi=33+arg.a;for(uint depth=0;depth<arg.D;depth++){N n=node1(lo,hi,d,o);if(o.trap){stop=depth;return o;}if(n.leaf){o.value=n.value;stop=depth;return o;}
  uint digit=(gid >> ((arg.D-1-depth)*1))&1;
  if(digit>=2){stop=depth+1;return zero();}
  if(digit==0){hi=n.m1;d--;}
  
  else{lo=n.m1;d-=1;}
 }stop=arg.D;return serial(lo,hi,d);
}
extern "C" __global__ void A(const Args arg,O *part,O *out,uint *stops){
 if(!guarded(arg.a,arg.b)){if(threadIdx.x==0)part[blockIdx.x]=blockIdx.x==0?fallback(arg.a,arg.b):zero();return;}
 __shared__ O v[256];uint gid=blockIdx.x*256+threadIdx.x;
 v[threadIdx.x]=path(arg.a,arg.b,gid,true);__syncthreads();
 for(uint s=128;s;s/=2){if(threadIdx.x<s)v[threadIdx.x]=addo(v[threadIdx.x],v[threadIdx.x+s]);__syncthreads();}
 if(threadIdx.x==0)part[blockIdx.x]=v[0];
}
extern "C" __global__ void reduce(const Args arg,O *part,O *out,uint *stops){
 if(!guarded(arg.a,arg.b)){if(threadIdx.x==0)out[0]=part[0];return;}
 __shared__ O v[256];O o=zero();for(uint i=threadIdx.x;i<arg.groups;i+=256)o=addo(o,part[i]);v[threadIdx.x]=o;__syncthreads();
 for(uint s=128;s;s/=2){if(threadIdx.x<s)v[threadIdx.x]=addo(v[threadIdx.x],v[threadIdx.x+s]);__syncthreads();}
 if(threadIdx.x==0)out[0]=v[0];
}
extern "C" __global__ void C2(const Args arg,O *part,O *out,uint *stops){
 if(!guarded(arg.a,arg.b)){if(threadIdx.x==0)part[blockIdx.x]=blockIdx.x==0?fallback(arg.a,arg.b):zero();return;}
 __shared__ uint v[256];uint gid=blockIdx.x*256+threadIdx.x;O o=path_fast(arg.a,arg.b,gid,false);v[threadIdx.x]=uint(o.value);__syncthreads();
 for(uint s=128;s;s/=2){if(threadIdx.x<s)v[threadIdx.x]=(v[threadIdx.x]+v[threadIdx.x+s])%100003;__syncthreads();}
 if(threadIdx.x==0){O r=zero();r.value=int(v[0]);part[blockIdx.x]=r;}
}
extern "C" __global__ void C1(const Args arg,O *part,O *out,uint *stops){
 if(!guarded(arg.a,arg.b)){if(threadIdx.x==0)out[0]=fallback(arg.a,arg.b);return;}
 __shared__ uint v[256];uint sum=0;for(uint gid=threadIdx.x;gid<uint(max(arg.a,1));gid+=256)sum=(sum+uint(path_fast(arg.a,arg.b,gid,false).value))%100003;
 v[threadIdx.x]=sum;__syncthreads();for(uint s=128;s;s/=2){if(threadIdx.x<s)v[threadIdx.x]=(v[threadIdx.x]+v[threadIdx.x+s])%100003;__syncthreads();}
 if(threadIdx.x==0){O r=zero();r.value=int(v[0]);out[0]=r;}
}
extern "C" __global__ void B(const Args arg,O *part,O *out,uint *stops){
 __shared__ O v[256];__shared__ uint stop[256];uint gid=blockIdx.x*256+threadIdx.x;uint count=1u<<(arg.D*1);
 uint st=0;v[threadIdx.x]=gid<count?replay(arg,gid,st):zero();stop[threadIdx.x]=st;__syncthreads();
 uint levels=min(arg.D,8u);
 for(uint level=0,stride=1;level<levels;level++,stride*=2){
  uint depth=arg.D-1-level;
  if(threadIdx.x%(stride*2)==0 && stop[threadIdx.x]>depth){v[threadIdx.x]=join1(v[threadIdx.x],v[threadIdx.x+stride],zero());}
  __syncthreads();
 }if(threadIdx.x==0){part[blockIdx.x]=v[0];stops[blockIdx.x]=stop[0];}
}
extern "C" __global__ void Btop(const Args arg,O *part,O *out,uint *stops){
 __shared__ O v[256];__shared__ uint stop[256];v[threadIdx.x]=threadIdx.x<arg.groups?part[threadIdx.x]:zero();stop[threadIdx.x]=threadIdx.x<arg.groups?stops[threadIdx.x]:0;__syncthreads();
 uint levels=arg.D>8?arg.D-8:0;
 for(uint level=0,stride=1;level<levels;level++,stride*=2){uint depth=levels-1-level;
  if(threadIdx.x%(stride*2)==0&&stop[threadIdx.x]>depth)v[threadIdx.x]=join1(v[threadIdx.x],v[threadIdx.x+stride],zero());__syncthreads();
 }if(threadIdx.x==0)out[0]=v[0];
}

extern "C" __global__ void C0(const Args arg,O *part,O *out,uint *stops){if(threadIdx.x==0)out[0]=fallback(arg.a,arg.b);}
extern "C" __global__ void A_fused(const Args arg,O *part,O *out,uint *stops){
 if(!guarded(arg.a,arg.b)){if(threadIdx.x==0)out[0]=fallback(arg.a,arg.b);return;}
 __shared__ O v[256];O o=zero();for(uint gid=threadIdx.x;gid<uint(max(arg.a,1));gid+=256)o=addo(o,path(arg.a,arg.b,gid,true));v[threadIdx.x]=o;__syncthreads();
 for(uint s=128;s;s/=2){if(threadIdx.x<s)v[threadIdx.x]=addo(v[threadIdx.x],v[threadIdx.x+s]);__syncthreads();}if(threadIdx.x==0)out[0]=v[0];
}
extern "C" __global__ void A_no_checks(const Args arg,O *part,O *out,uint *stops){
 if(!guarded(arg.a,arg.b)){if(threadIdx.x==0)part[blockIdx.x]=blockIdx.x==0?fallback(arg.a,arg.b):zero();return;}
 __shared__ O v[256];uint gid=blockIdx.x*256+threadIdx.x;
 v[threadIdx.x]=path(arg.a,arg.b,gid,false);__syncthreads();
 for(uint s=128;s;s/=2){if(threadIdx.x<s)v[threadIdx.x]=addo(v[threadIdx.x],v[threadIdx.x+s]);__syncthreads();}
 if(threadIdx.x==0)part[blockIdx.x]=v[0];
}
extern "C" __global__ void A_narrow(const Args arg,O *part,O *out,uint *stops){
 if(!guarded(arg.a,arg.b)){if(threadIdx.x==0)part[blockIdx.x]=blockIdx.x==0?fallback(arg.a,arg.b):zero();return;}
 __shared__ O v[256];uint gid=blockIdx.x*256+threadIdx.x;
 v[threadIdx.x]=path_fast(arg.a,arg.b,gid,false);__syncthreads();
 for(uint s=128;s;s/=2){if(threadIdx.x<s)v[threadIdx.x]=addo(v[threadIdx.x],v[threadIdx.x+s]);__syncthreads();}
 if(threadIdx.x==0)part[blockIdx.x]=v[0];
}
__device__ uint narrow_path(int a,int b,uint gid,bool unused){
 O o=zero();
 if(gid>=uint(max(a,1)))return 0;
 int lo=33,hi=33+a,d=b;uint k=33+gid;uint weight=1;
 for(uint depth=0;depth<64;depth++){N n=nodep0(lo,hi,d,k,o);
  if(n.leaf){if(uint(lo)!=k)return 0;return (weight*uint(n.value))%100003;}
  if(k<uint(n.m1)){hi=n.m1;d--;weight=(weight*3)%100003;}
  
  else{lo=n.m1;d-=1;weight=(weight*1)%100003;}
 }o.trap=3;return 0;
}

extern "C" __global__ void AProot(const Args arg,O *part,O *out,uint *stops){if(threadIdx.x==0){O o=zero();N n=node0(33,33+arg.a,arg.b,o);o.value=n.value;out[0]=o;}}
extern "C" __global__ void APfused(const Args arg,O *part,O *out,uint *stops){
 __shared__ uint v[256];uint sum=0;
 for(uint gid=threadIdx.x;gid<uint(max(arg.a,1));gid+=256)sum=(sum+narrow_path(arg.a,arg.b,gid,false))%100003;
 v[threadIdx.x]=sum;__syncthreads();for(uint s=128;s;s/=2){if(threadIdx.x<s)v[threadIdx.x]=(v[threadIdx.x]+v[threadIdx.x+s])%100003;__syncthreads();}
 if(threadIdx.x==0){O o=zero();o.value=int(v[0]);out[0]=o;}
}
extern "C" __global__ void APparallel(const Args arg,O *part,O *out,uint *stops){
 __shared__ uint v[256];v[threadIdx.x]=narrow_path(arg.a,arg.b,blockIdx.x*256+threadIdx.x,false);__syncthreads();
 for(uint s=128;s;s/=2){if(threadIdx.x<s)v[threadIdx.x]=(v[threadIdx.x]+v[threadIdx.x+s])%100003;__syncthreads();}
 if(threadIdx.x==0)part[blockIdx.x].value=int(v[0]);
}
extern "C" __global__ void APreduce(const Args arg,O *part,O *out,uint *stops){
 __shared__ uint v[256];v[threadIdx.x]=threadIdx.x<arg.groups?uint(part[threadIdx.x].value):0;__syncthreads();
 for(uint s=128;s;s/=2){if(threadIdx.x<s)v[threadIdx.x]=(v[threadIdx.x]+v[threadIdx.x+s])%100003;__syncthreads();}
 if(threadIdx.x==0){O o=zero();o.value=int(v[0]);out[0]=o;}
}
