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
__device__ int w1(int x, REF O &o) {int z0=ck(x,101,4,36,o);int s=z0;int i=0;for (uint guard=0;guard<64;guard++) {int z1=ck(x,11,4,74,o);int z2=(i < z1);if (!(z2) || o.trap) break;int z3=ck(s,2,2,96,o);int z4=ck(z3,i,0,100,o);int z5=ck(z4,10007,4,105,o);s=z5;int z6=ck(i,1,0,128,o);i=z6;}return s;}
__device__ int t01(int n,int s,REF O &o) {return w1(s,o);}
__device__ int t1(int n,int s,REF O &o) {int z7=(n <= 0);if (z7) {int z8=w1(s,o);return z8;}int z9=ck(s,2,2,233,o);int z10=ck(n,4,2,241,o);int z11=ck(z9,z10,0,237,o);int z12=ck(z11,4,4,246,o);int sel=z12;int z13=(sel == 0);if (z13) {int z14=ck(n,1,1,292,o);int z15=ck(s,15,2,300,o);int z16=ck(z15,23,0,305,o);int z17=ck(z16,1009,4,311,o);int z18=t01(z14,z17,o);int v0_0=z18;int z19=ck(n,1,1,343,o);int z20=ck(s,24,2,351,o);int z21=ck(z20,80,0,356,o);int z22=ck(z21,1009,4,362,o);int z23=t01(z19,z22,o);int v0_1=z23;int z24=ck(v0_0,9,2,392,o);int z25=ck(z24,v0_1,0,396,o);int z26=ck(z25,10007,4,404,o);return z26;}int z27=(sel == 1);if (z27) {int z28=ck(n,2,1,460,o);int z29=ck(s,4,2,468,o);int z30=ck(z29,56,0,472,o);int z31=ck(z30,1009,4,478,o);int z32=t01(z28,z31,o);int v1_0=z32;int z33=ck(n,2,1,510,o);int z34=ck(s,29,2,518,o);int z35=ck(z34,90,0,523,o);int z36=ck(z35,1009,4,529,o);int z37=t01(z33,z36,o);int v1_1=z37;int z38=ck(v1_0,9,2,559,o);int z39=ck(z38,v1_1,0,563,o);int z40=ck(z39,10007,4,571,o);return z40;}int z41=(sel == 2);if (z41) {int z42=ck(n,2,1,627,o);int z43=ck(s,28,2,635,o);int z44=ck(z43,40,0,640,o);int z45=ck(z44,1009,4,646,o);int z46=t01(z42,z45,o);int v2_0=z46;int z47=ck(n,2,1,678,o);int z48=ck(s,6,2,686,o);int z49=ck(z48,54,0,690,o);int z50=ck(z49,1009,4,696,o);int z51=t01(z47,z50,o);int v2_1=z51;int z52=ck(n,1,1,728,o);int z53=ck(s,8,2,736,o);int z54=ck(z53,28,0,740,o);int z55=ck(z54,1009,4,746,o);int z56=t01(z52,z55,o);int v2_2=z56;int z57=ck(v2_0,8,2,776,o);int z58=ck(z57,v2_1,0,780,o);int z59=ck(z58,v2_2,0,787,o);int z60=ck(z59,10007,4,795,o);return z60;}int z61=ck(n,1,1,825,o);int z62=ck(s,18,2,833,o);int z63=ck(z62,47,0,838,o);int z64=ck(z63,1009,4,844,o);int z65=t01(z61,z64,o);return z65;}
__device__ N node1(int lo,int hi,int d,REF O &o) {N n={0,0,0,0};int z66=ck(hi,lo,1,906,o);int z67=(z66 <= 2);if (z67) {int z68=ck(lo,8,2,931,o);int z69=ck(z68,hi,0,935,o);int z70=ck(z69,2,4,941,o);int z71=ck(lo,8,2,950,o);int z72=ck(z71,hi,0,954,o);int z73=ck(z72,1009,4,960,o);int z74=t1(z70,z73,o);n.leaf=1;n.value=z74;return n;}int z75=(d <= 0);if (z75) {int z76=ck(hi,lo,1,1000,o);int z77=ck(z76,9,0,1005,o);int z78=ck(z77,2,4,1010,o);int z79=ck(hi,lo,1,1019,o);int z80=ck(z79,9,0,1024,o);int z81=ck(z80,1009,4,1029,o);int z82=t1(z78,z81,o);n.leaf=1;n.value=z82;return n;}int z83=ck(lo,1,0,1056,o);int z84=ck(hi,lo,1,1067,o);int z85=ck(z84,2,1,1072,o);int z86=ck(lo,1,4,1088,o);int z87=ck(z86,1,0,1092,o);int z88=ck(z87,1,4,1097,o);int z89=ck(2,z88,0,1082,o);int z90=ck(z85,z89,2,1077,o);int z91=ck(z90,6,3,1103,o);int z92=ck(z83,z91,0,1060,o);int m1=z92;n.m1=m1;int z93=ck(m1,1,0,1124,o);int z94=ck(hi,m1,1,1135,o);int z95=ck(z94,2,1,1140,o);int z96=ck(z95,3,2,1145,o);int z97=ck(z96,6,3,1150,o);int z98=ck(z93,z97,0,1128,o);int m2=z98;n.m2=m2;return n;}
__device__ O join1(O aa,O bb,O cc) {if(aa.trap)return aa;if(bb.trap)return bb;if(cc.trap)return cc;O o=zero();int a=aa.value,b=bb.value,c=cc.value;int z99=ck(a,1,2,1262,o);int z100=ck(z99,b,0,1266,o);int z101=ck(c,4,2,1274,o);int z102=ck(z100,z101,0,1270,o);int z103=ck(z102,100003,4,1279,o);o.value=z103;return o;}
__device__ int w0(int x, REF O &o) {int z0=(x % 101);int s=z0;int i=0;for (uint guard=0;guard<64;guard++) {int z1=(x % 11);int z2=(i < z1);if (!(z2) || o.trap) break;int z3=(s * 2);int z4=(z3 + i);int z5=(z4 % 10007);s=z5;int z6=(i + 1);i=z6;}return s;}
__device__ int t00(int n,int s,REF O &o) {return w0(s,o);}
__device__ int t0(int n,int s,REF O &o) {int z7=(n <= 0);if (z7) {int z8=w0(s,o);return z8;}int z9=(s * 2);int z10=(n * 4);int z11=(z9 + z10);int z12=(z11 % 4);int sel=z12;int z13=(sel == 0);if (z13) {int z14=(n - 1);int z15=(s * 15);int z16=(z15 + 23);int z17=(z16 % 1009);int z18=t00(z14,z17,o);int v0_0=z18;int z19=(n - 1);int z20=(s * 24);int z21=(z20 + 80);int z22=(z21 % 1009);int z23=t00(z19,z22,o);int v0_1=z23;int z24=(v0_0 * 9);int z25=(z24 + v0_1);int z26=(z25 % 10007);return z26;}int z27=(sel == 1);if (z27) {int z28=(n - 2);int z29=(s * 4);int z30=(z29 + 56);int z31=(z30 % 1009);int z32=t00(z28,z31,o);int v1_0=z32;int z33=(n - 2);int z34=(s * 29);int z35=(z34 + 90);int z36=(z35 % 1009);int z37=t00(z33,z36,o);int v1_1=z37;int z38=(v1_0 * 9);int z39=(z38 + v1_1);int z40=(z39 % 10007);return z40;}int z41=(sel == 2);if (z41) {int z42=(n - 2);int z43=(s * 28);int z44=(z43 + 40);int z45=(z44 % 1009);int z46=t00(z42,z45,o);int v2_0=z46;int z47=(n - 2);int z48=(s * 6);int z49=(z48 + 54);int z50=(z49 % 1009);int z51=t00(z47,z50,o);int v2_1=z51;int z52=(n - 1);int z53=(s * 8);int z54=(z53 + 28);int z55=(z54 % 1009);int z56=t00(z52,z55,o);int v2_2=z56;int z57=(v2_0 * 8);int z58=(z57 + v2_1);int z59=(z58 + v2_2);int z60=(z59 % 10007);return z60;}int z61=(n - 1);int z62=(s * 18);int z63=(z62 + 47);int z64=(z63 % 1009);int z65=t00(z61,z64,o);return z65;}
__device__ N node0(int lo,int hi,int d,REF O &o) {N n={0,0,0,0};int z66=(hi - lo);int z67=(z66 <= 2);if (z67) {int z68=(lo * 8);int z69=(z68 + hi);int z70=(z69 % 2);int z71=(lo * 8);int z72=(z71 + hi);int z73=(z72 % 1009);int z74=t0(z70,z73,o);n.leaf=1;n.value=z74;return n;}int z75=(d <= 0);if (z75) {int z76=(hi - lo);int z77=(z76 + 9);int z78=(z77 % 2);int z79=(hi - lo);int z80=(z79 + 9);int z81=(z80 % 1009);int z82=t0(z78,z81,o);n.leaf=1;n.value=z82;return n;}int z83=(lo + 1);int z84=(hi - lo);int z85=(z84 - 2);int z86=(lo % 1);int z87=(z86 + 1);int z88=(z87 % 1);int z89=(2 + z88);int z90=(z85 * z89);int z91=(z90 / 6);int z92=(z83 + z91);int m1=z92;n.m1=m1;int z93=(m1 + 1);int z94=(hi - m1);int z95=(z94 - 2);int z96=(z95 * 3);int z97=(z96 / 6);int z98=(z93 + z97);int m2=z98;n.m2=m2;return n;}
__device__ O join0(O aa,O bb,O cc) {if(aa.trap)return aa;if(bb.trap)return bb;if(cc.trap)return cc;O o=zero();int a=aa.value,b=bb.value,c=cc.value;int z99=(a * 1);int z100=(z99 + b);int z101=(c * 4);int z102=(z100 + z101);int z103=(z102 % 100003);o.value=z103;return o;}
struct F {int lo,hi,d,phase,m1,m2;O a,b;};
__device__ O serial(int lo,int hi,int d) {
 F f[64];uint sp=0;f[0]={lo,hi,d,0,0,0,zero(),zero()};O r=zero();
 for(uint work=0;work<200000;work++){
  if(f[sp].phase==0){O o=zero();N n=node1(f[sp].lo,f[sp].hi,f[sp].d,o);if(o.trap)return o;
   if(n.leaf){r.value=n.value;if(sp==0)return r;sp--;}
   else{f[sp].m1=n.m1;f[sp].m2=n.m2;f[sp].phase=1;F ch={f[sp].lo,n.m1,f[sp].d-1,0,0,0,zero(),zero()};if(sp==63){r.trap=3;return r;}f[++sp]=ch;continue;}
  }
  for(;;){
   if(f[sp].phase==1){f[sp].a=r;f[sp].phase=2;F ch={f[sp].m1,f[sp].m2,f[sp].d-1,0,0,0,zero(),zero()};f[++sp]=ch;break;}
   if(f[sp].phase==2){f[sp].b=r;f[sp].phase=3;F ch={f[sp].m2,f[sp].hi,f[sp].d-1,0,0,0,zero(),zero()};f[++sp]=ch;break;}
   r=join1(f[sp].a,f[sp].b,r);if(sp==0)return r;sp--;
  }
 }r.trap=3;return r;
}
__device__ O fallback(int a,int b){O o=zero();int hi=ck(48,a,0,1340,o);if(o.trap)return o;return serial(48,hi,b);}

__device__ bool guarded(int a,int b){return a>=0&&a<=3000&&b>=0&&b<=14;}
__device__ N nodep0(int lo,int hi,int d,uint key,REF O &o) {N n={0,0,0,0};int z66=(hi - lo);int z67=(z66 <= 2);if (z67) {if(uint(lo)!=key){n.leaf=1;return n;}int z68=(lo * 8);int z69=(z68 + hi);int z70=(z69 % 2);int z71=(lo * 8);int z72=(z71 + hi);int z73=(z72 % 1009);int z74=t0(z70,z73,o);n.leaf=1;n.value=z74;return n;}int z75=(d <= 0);if (z75) {if(uint(lo)!=key){n.leaf=1;return n;}int z76=(hi - lo);int z77=(z76 + 9);int z78=(z77 % 2);int z79=(hi - lo);int z80=(z79 + 9);int z81=(z80 % 1009);int z82=t0(z78,z81,o);n.leaf=1;n.value=z82;return n;}int z83=(lo + 1);int z84=(hi - lo);int z85=(z84 - 2);int z86=(lo % 1);int z87=(z86 + 1);int z88=(z87 % 1);int z89=(2 + z88);int z90=(z85 * z89);int z91=(z90 / 6);int z92=(z83 + z91);int m1=z92;n.m1=m1;int z93=(m1 + 1);int z94=(hi - m1);int z95=(z94 - 2);int z96=(z95 * 3);int z97=(z96 / 6);int z98=(z93 + z97);int m2=z98;n.m2=m2;return n;}
__device__ N nodep1(int lo,int hi,int d,uint key,REF O &o) {N n={0,0,0,0};int z66=ck(hi,lo,1,906,o);int z67=(z66 <= 2);if (z67) {if(uint(lo)!=key){n.leaf=1;return n;}int z68=ck(lo,8,2,931,o);int z69=ck(z68,hi,0,935,o);int z70=ck(z69,2,4,941,o);int z71=ck(lo,8,2,950,o);int z72=ck(z71,hi,0,954,o);int z73=ck(z72,1009,4,960,o);int z74=t1(z70,z73,o);n.leaf=1;n.value=z74;return n;}int z75=(d <= 0);if (z75) {if(uint(lo)!=key){n.leaf=1;return n;}int z76=ck(hi,lo,1,1000,o);int z77=ck(z76,9,0,1005,o);int z78=ck(z77,2,4,1010,o);int z79=ck(hi,lo,1,1019,o);int z80=ck(z79,9,0,1024,o);int z81=ck(z80,1009,4,1029,o);int z82=t1(z78,z81,o);n.leaf=1;n.value=z82;return n;}int z83=ck(lo,1,0,1056,o);int z84=ck(hi,lo,1,1067,o);int z85=ck(z84,2,1,1072,o);int z86=ck(lo,1,4,1088,o);int z87=ck(z86,1,0,1092,o);int z88=ck(z87,1,4,1097,o);int z89=ck(2,z88,0,1082,o);int z90=ck(z85,z89,2,1077,o);int z91=ck(z90,6,3,1103,o);int z92=ck(z83,z91,0,1060,o);int m1=z92;n.m1=m1;int z93=ck(m1,1,0,1124,o);int z94=ck(hi,m1,1,1135,o);int z95=ck(z94,2,1,1140,o);int z96=ck(z95,3,2,1145,o);int z97=ck(z96,6,3,1150,o);int z98=ck(z93,z97,0,1128,o);int m2=z98;n.m2=m2;return n;}
__device__ O path(int a,int b,uint gid,bool wide){
 O o=zero();if(!guarded(a,b)){if(gid==0)return fallback(a,b);return o;}
 if(gid>=uint(max(a,1)))return o;
 int lo=48,hi=48+a,d=b;uint k=48+gid;UWIDE weight=1;
 for(uint depth=0;depth<64;depth++){N n=wide?nodep1(lo,hi,d,k,o):nodep0(lo,hi,d,k,o);if(o.trap){o.key=uint(lo-48)*64+depth;return o;}
  if(n.leaf){if(uint(lo)!=k)return zero();o.value=int((weight*UWIDE(n.value))%100003);return o;}
  if(k<uint(n.m1)){hi=n.m1;d--;weight=(weight*1)%100003;}
  else if(k<uint(n.m2)){lo=n.m1;hi=n.m2;d--;}
  else{lo=n.m2;d-=1;weight=(weight*4)%100003;}
 }o.trap=3;return o;
}
__device__ O addo(O a,O b){if(a.trap&&(!b.trap||a.key<=b.key))return a;if(b.trap)return b;O o=zero();o.value=int((UWIDE(a.value)+UWIDE(b.value))%100003);return o;}
__device__ O path_fast(int a,int b,uint gid,bool wide){
 O o=zero();if(!guarded(a,b)){if(gid==0)return fallback(a,b);return o;}
 if(gid>=uint(max(a,1)))return o;
 int lo=48,hi=48+a,d=b;uint k=48+gid;uint weight=1;
 for(uint depth=0;depth<64;depth++){N n=nodep0(lo,hi,d,k,o);if(o.trap){o.key=uint(lo-48)*64+depth;return o;}
  if(n.leaf){if(uint(lo)!=k)return zero();o.value=int((weight*uint(n.value))%100003);return o;}
  if(k<uint(n.m1)){hi=n.m1;d--;weight=(weight*1)%100003;}
  else if(k<uint(n.m2)){lo=n.m1;hi=n.m2;d--;}
  else{lo=n.m2;d-=1;weight=(weight*4)%100003;}
 }o.trap=3;return o;
}

struct Args {int a,b;uint D,groups;};
__device__ O replay(Args arg,uint gid,REF uint &stop){
 O o=zero();int lo=48,hi=0,d=arg.b;stop=0;
 if(!guarded(arg.a,arg.b)){if(gid==0)return fallback(arg.a,arg.b);return o;}
 hi=48+arg.a;for(uint depth=0;depth<arg.D;depth++){N n=node1(lo,hi,d,o);if(o.trap){stop=depth;return o;}if(n.leaf){o.value=n.value;stop=depth;return o;}
  uint digit=(gid >> ((arg.D-1-depth)*2))&3;
  if(digit>=3){stop=depth+1;return zero();}
  if(digit==0){hi=n.m1;d--;}
  else if(digit==1){lo=n.m1;hi=n.m2;d--;}
  else{lo=n.m2;d-=1;}
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
 __shared__ O v[256];__shared__ uint stop[256];uint gid=blockIdx.x*256+threadIdx.x;uint count=1u<<(arg.D*2);
 uint st=0;v[threadIdx.x]=gid<count?replay(arg,gid,st):zero();stop[threadIdx.x]=st;__syncthreads();
 uint levels=min(arg.D,4u);
 for(uint level=0,stride=1;level<levels;level++,stride*=4){
  uint depth=arg.D-1-level;
  if(threadIdx.x%(stride*4)==0 && stop[threadIdx.x]>depth){v[threadIdx.x]=join1(v[threadIdx.x],v[threadIdx.x+stride],v[threadIdx.x+stride*2]);}
  __syncthreads();
 }if(threadIdx.x==0){part[blockIdx.x]=v[0];stops[blockIdx.x]=stop[0];}
}
extern "C" __global__ void Btop(const Args arg,O *part,O *out,uint *stops){
 __shared__ O v[256];__shared__ uint stop[256];v[threadIdx.x]=threadIdx.x<arg.groups?part[threadIdx.x]:zero();stop[threadIdx.x]=threadIdx.x<arg.groups?stops[threadIdx.x]:0;__syncthreads();
 uint levels=arg.D>4?arg.D-4:0;
 for(uint level=0,stride=1;level<levels;level++,stride*=4){uint depth=levels-1-level;
  if(threadIdx.x%(stride*4)==0&&stop[threadIdx.x]>depth)v[threadIdx.x]=join1(v[threadIdx.x],v[threadIdx.x+stride],v[threadIdx.x+stride*2]);__syncthreads();
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
 int lo=48,hi=48+a,d=b;uint k=48+gid;uint weight=1;
 for(uint depth=0;depth<64;depth++){N n=nodep0(lo,hi,d,k,o);
  if(n.leaf){if(uint(lo)!=k)return 0;return (weight*uint(n.value))%100003;}
  if(k<uint(n.m1)){hi=n.m1;d--;weight=(weight*1)%100003;}
  else if(k<uint(n.m2)){lo=n.m1;hi=n.m2;d--;}
  else{lo=n.m2;d-=1;weight=(weight*4)%100003;}
 }o.trap=3;return 0;
}

extern "C" __global__ void AProot(const Args arg,O *part,O *out,uint *stops){if(threadIdx.x==0){O o=zero();N n=node0(48,48+arg.a,arg.b,o);o.value=n.value;out[0]=o;}}
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
