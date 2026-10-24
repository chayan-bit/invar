#include <metal_stdlib>
using namespace metal;
#define REF thread
#define WIDE long
#define UWIDE ulong

struct O {int value;uint trap,span,end,key;};
struct N {int leaf,value,m1,m2;};
O zero(){O o={0,0,0,0,0xffffffffu};return o;}
int ck(int a,int b,uint op,uint span,REF O &o){
 if(o.trap)return 0;
 if(op>=3 && b==0){o.trap=2;o.span=span;o.end=span+1;return 0;}
 if(op==4 && a==(-2147483647-1) && b==-1)return 0;
 WIDE v=op==0?WIDE(a)+b:op==1?WIDE(a)-b:op==2?WIDE(a)*b:op==3?WIDE(a)/b:WIDE(a)%b;
 if(v<(-2147483647L-1L)||v>2147483647L){o.trap=1;o.span=span;o.end=span+1;return 0;}return int(v);
}
int w1(int x, REF O &o) {int z0=ck(x,2,4,31,o);int z1=(z0 == 0);if (z1) {int z2=ck(x,2,3,52,o);int z3=ck(z2,36,0,56,o);int z4=ck(z3,10007,4,62,o);return z4;}int z5=ck(x,3,2,87,o);int z6=ck(z5,1,0,91,o);int z7=ck(z6,10007,4,96,o);return z7;}
int t01(int n,int s,REF O &o) {return w1(s,o);}
int t1(int n,int s,REF O &o) {int z8=(n <= 0);if (z8) {int z9=w1(s,o);return z9;}int z10=ck(s,31,2,185,o);int z11=ck(n,2,2,194,o);int z12=ck(z10,z11,0,190,o);int z13=ck(z12,3,4,199,o);int sel=z13;int z14=(sel == 0);if (z14) {int z15=ck(n,1,1,245,o);int z16=ck(s,22,2,253,o);int z17=ck(z16,84,0,258,o);int z18=ck(z17,1009,4,264,o);int z19=t01(z15,z18,o);int v0_0=z19;int z20=ck(n,2,1,296,o);int z21=ck(s,23,2,304,o);int z22=ck(z21,19,0,309,o);int z23=ck(z22,1009,4,315,o);int z24=t01(z20,z23,o);int v0_1=z24;int z25=ck(v0_0,3,2,345,o);int z26=ck(z25,v0_1,0,349,o);int z27=ck(z26,10007,4,357,o);return z27;}int z28=(sel == 1);if (z28) {int z29=ck(n,2,1,413,o);int z30=ck(s,2,2,421,o);int z31=ck(z30,28,0,425,o);int z32=ck(z31,1009,4,431,o);int z33=t01(z29,z32,o);int v1_0=z33;int z34=ck(n,2,1,463,o);int z35=ck(s,29,2,471,o);int z36=ck(z35,76,0,476,o);int z37=ck(z36,1009,4,482,o);int z38=t01(z34,z37,o);int v1_1=z38;int z39=ck(n,2,1,514,o);int z40=ck(s,2,2,522,o);int z41=ck(z40,48,0,526,o);int z42=ck(z41,1009,4,532,o);int z43=t01(z39,z42,o);int v1_2=z43;int z44=ck(v1_0,4,2,562,o);int z45=ck(z44,v1_1,0,566,o);int z46=ck(z45,v1_2,0,573,o);int z47=ck(z46,10007,4,581,o);return z47;}int z48=ck(n,1,1,611,o);int z49=ck(s,19,2,619,o);int z50=ck(z49,13,0,624,o);int z51=ck(z50,1009,4,630,o);int z52=t01(z48,z51,o);return z52;}
N node1(int lo,int hi,int d,REF O &o) {N n={0,0,0,0};int z53=ck(hi,lo,1,692,o);int z54=(z53 <= 3);if (z54) {int z55=ck(lo,8,2,717,o);int z56=ck(z55,hi,0,721,o);int z57=ck(z56,2,4,727,o);int z58=ck(lo,8,2,736,o);int z59=ck(z58,hi,0,740,o);int z60=ck(z59,1009,4,746,o);int z61=t1(z57,z60,o);n.leaf=1;n.value=z61;return n;}int z62=(d <= 0);if (z62) {int z63=ck(hi,lo,1,786,o);int z64=ck(z63,9,0,791,o);int z65=ck(z64,2,4,796,o);int z66=ck(hi,lo,1,805,o);int z67=ck(z66,9,0,810,o);int z68=ck(z67,1009,4,815,o);int z69=t1(z65,z68,o);n.leaf=1;n.value=z69;return n;}int z70=ck(lo,1,0,842,o);int z71=ck(hi,lo,1,853,o);int z72=ck(z71,2,1,858,o);int z73=ck(lo,3,4,874,o);int z74=ck(z73,3,0,878,o);int z75=ck(z74,3,4,883,o);int z76=ck(2,z75,0,868,o);int z77=ck(z72,z76,2,863,o);int z78=ck(z77,8,3,889,o);int z79=ck(z70,z78,0,846,o);int m1=z79;n.m1=m1;int z80=ck(m1,1,0,910,o);int z81=ck(hi,m1,1,921,o);int z82=ck(z81,2,1,926,o);int z83=ck(z82,2,2,931,o);int z84=ck(z83,8,3,936,o);int z85=ck(z80,z84,0,914,o);int m2=z85;n.m2=m2;return n;}
O join1(O aa,O bb,O cc) {if(aa.trap)return aa;if(bb.trap)return bb;if(cc.trap)return cc;O o=zero();int a=aa.value,b=bb.value,c=cc.value;int z86=ck(a,6,2,1048,o);int z87=ck(z86,b,0,1052,o);int z88=ck(c,5,2,1060,o);int z89=ck(z87,z88,0,1056,o);int z90=ck(z89,100003,4,1065,o);o.value=z90;return o;}
int w0(int x, REF O &o) {int z0=(x % 2);int z1=(z0 == 0);if (z1) {int z2=(x / 2);int z3=(z2 + 36);int z4=(z3 % 10007);return z4;}int z5=(x * 3);int z6=(z5 + 1);int z7=(z6 % 10007);return z7;}
int t00(int n,int s,REF O &o) {return w0(s,o);}
int t0(int n,int s,REF O &o) {int z8=(n <= 0);if (z8) {int z9=w0(s,o);return z9;}int z10=(s * 31);int z11=(n * 2);int z12=(z10 + z11);int z13=(z12 % 3);int sel=z13;int z14=(sel == 0);if (z14) {int z15=(n - 1);int z16=(s * 22);int z17=(z16 + 84);int z18=(z17 % 1009);int z19=t00(z15,z18,o);int v0_0=z19;int z20=(n - 2);int z21=(s * 23);int z22=(z21 + 19);int z23=(z22 % 1009);int z24=t00(z20,z23,o);int v0_1=z24;int z25=(v0_0 * 3);int z26=(z25 + v0_1);int z27=(z26 % 10007);return z27;}int z28=(sel == 1);if (z28) {int z29=(n - 2);int z30=(s * 2);int z31=(z30 + 28);int z32=(z31 % 1009);int z33=t00(z29,z32,o);int v1_0=z33;int z34=(n - 2);int z35=(s * 29);int z36=(z35 + 76);int z37=(z36 % 1009);int z38=t00(z34,z37,o);int v1_1=z38;int z39=(n - 2);int z40=(s * 2);int z41=(z40 + 48);int z42=(z41 % 1009);int z43=t00(z39,z42,o);int v1_2=z43;int z44=(v1_0 * 4);int z45=(z44 + v1_1);int z46=(z45 + v1_2);int z47=(z46 % 10007);return z47;}int z48=(n - 1);int z49=(s * 19);int z50=(z49 + 13);int z51=(z50 % 1009);int z52=t00(z48,z51,o);return z52;}
N node0(int lo,int hi,int d,REF O &o) {N n={0,0,0,0};int z53=(hi - lo);int z54=(z53 <= 3);if (z54) {int z55=(lo * 8);int z56=(z55 + hi);int z57=(z56 % 2);int z58=(lo * 8);int z59=(z58 + hi);int z60=(z59 % 1009);int z61=t0(z57,z60,o);n.leaf=1;n.value=z61;return n;}int z62=(d <= 0);if (z62) {int z63=(hi - lo);int z64=(z63 + 9);int z65=(z64 % 2);int z66=(hi - lo);int z67=(z66 + 9);int z68=(z67 % 1009);int z69=t0(z65,z68,o);n.leaf=1;n.value=z69;return n;}int z70=(lo + 1);int z71=(hi - lo);int z72=(z71 - 2);int z73=(lo % 3);int z74=(z73 + 3);int z75=(z74 % 3);int z76=(2 + z75);int z77=(z72 * z76);int z78=(z77 / 8);int z79=(z70 + z78);int m1=z79;n.m1=m1;int z80=(m1 + 1);int z81=(hi - m1);int z82=(z81 - 2);int z83=(z82 * 2);int z84=(z83 / 8);int z85=(z80 + z84);int m2=z85;n.m2=m2;return n;}
O join0(O aa,O bb,O cc) {if(aa.trap)return aa;if(bb.trap)return bb;if(cc.trap)return cc;O o=zero();int a=aa.value,b=bb.value,c=cc.value;int z86=(a * 6);int z87=(z86 + b);int z88=(c * 5);int z89=(z87 + z88);int z90=(z89 % 100003);o.value=z90;return o;}
struct F {int lo,hi,d,phase,m1,m2;O a,b;};
O serial(int lo,int hi,int d) {
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
O fallback(int a,int b){O o=zero();int hi=ck(5,a,0,1124,o);if(o.trap)return o;return serial(5,hi,b);}

bool guarded(int a,int b){return a>=0&&a<=3000&&b>=0&&b<=14;}
N nodep0(int lo,int hi,int d,uint key,REF O &o) {N n={0,0,0,0};int z53=(hi - lo);int z54=(z53 <= 3);if (z54) {if(uint(lo)!=key){n.leaf=1;return n;}int z55=(lo * 8);int z56=(z55 + hi);int z57=(z56 % 2);int z58=(lo * 8);int z59=(z58 + hi);int z60=(z59 % 1009);int z61=t0(z57,z60,o);n.leaf=1;n.value=z61;return n;}int z62=(d <= 0);if (z62) {if(uint(lo)!=key){n.leaf=1;return n;}int z63=(hi - lo);int z64=(z63 + 9);int z65=(z64 % 2);int z66=(hi - lo);int z67=(z66 + 9);int z68=(z67 % 1009);int z69=t0(z65,z68,o);n.leaf=1;n.value=z69;return n;}int z70=(lo + 1);int z71=(hi - lo);int z72=(z71 - 2);int z73=(lo % 3);int z74=(z73 + 3);int z75=(z74 % 3);int z76=(2 + z75);int z77=(z72 * z76);int z78=(z77 / 8);int z79=(z70 + z78);int m1=z79;n.m1=m1;int z80=(m1 + 1);int z81=(hi - m1);int z82=(z81 - 2);int z83=(z82 * 2);int z84=(z83 / 8);int z85=(z80 + z84);int m2=z85;n.m2=m2;return n;}
N nodep1(int lo,int hi,int d,uint key,REF O &o) {N n={0,0,0,0};int z53=ck(hi,lo,1,692,o);int z54=(z53 <= 3);if (z54) {if(uint(lo)!=key){n.leaf=1;return n;}int z55=ck(lo,8,2,717,o);int z56=ck(z55,hi,0,721,o);int z57=ck(z56,2,4,727,o);int z58=ck(lo,8,2,736,o);int z59=ck(z58,hi,0,740,o);int z60=ck(z59,1009,4,746,o);int z61=t1(z57,z60,o);n.leaf=1;n.value=z61;return n;}int z62=(d <= 0);if (z62) {if(uint(lo)!=key){n.leaf=1;return n;}int z63=ck(hi,lo,1,786,o);int z64=ck(z63,9,0,791,o);int z65=ck(z64,2,4,796,o);int z66=ck(hi,lo,1,805,o);int z67=ck(z66,9,0,810,o);int z68=ck(z67,1009,4,815,o);int z69=t1(z65,z68,o);n.leaf=1;n.value=z69;return n;}int z70=ck(lo,1,0,842,o);int z71=ck(hi,lo,1,853,o);int z72=ck(z71,2,1,858,o);int z73=ck(lo,3,4,874,o);int z74=ck(z73,3,0,878,o);int z75=ck(z74,3,4,883,o);int z76=ck(2,z75,0,868,o);int z77=ck(z72,z76,2,863,o);int z78=ck(z77,8,3,889,o);int z79=ck(z70,z78,0,846,o);int m1=z79;n.m1=m1;int z80=ck(m1,1,0,910,o);int z81=ck(hi,m1,1,921,o);int z82=ck(z81,2,1,926,o);int z83=ck(z82,2,2,931,o);int z84=ck(z83,8,3,936,o);int z85=ck(z80,z84,0,914,o);int m2=z85;n.m2=m2;return n;}
O path(int a,int b,uint gid,bool wide){
 O o=zero();if(!guarded(a,b)){if(gid==0)return fallback(a,b);return o;}
 if(gid>=uint(max(a,1)))return o;
 int lo=5,hi=5+a,d=b;uint k=5+gid;UWIDE weight=1;
 for(uint depth=0;depth<64;depth++){N n=wide?nodep1(lo,hi,d,k,o):nodep0(lo,hi,d,k,o);if(o.trap){o.key=uint(lo-5)*64+depth;return o;}
  if(n.leaf){if(uint(lo)!=k)return zero();o.value=int((weight*UWIDE(n.value))%100003);return o;}
  if(k<uint(n.m1)){hi=n.m1;d--;weight=(weight*6)%100003;}
  else if(k<uint(n.m2)){lo=n.m1;hi=n.m2;d--;}
  else{lo=n.m2;d-=1;weight=(weight*5)%100003;}
 }o.trap=3;return o;
}
O addo(O a,O b){if(a.trap&&(!b.trap||a.key<=b.key))return a;if(b.trap)return b;O o=zero();o.value=int((UWIDE(a.value)+UWIDE(b.value))%100003);return o;}
O path_fast(int a,int b,uint gid,bool wide){
 O o=zero();if(!guarded(a,b)){if(gid==0)return fallback(a,b);return o;}
 if(gid>=uint(max(a,1)))return o;
 int lo=5,hi=5+a,d=b;uint k=5+gid;uint weight=1;
 for(uint depth=0;depth<64;depth++){N n=nodep0(lo,hi,d,k,o);if(o.trap){o.key=uint(lo-5)*64+depth;return o;}
  if(n.leaf){if(uint(lo)!=k)return zero();o.value=int((weight*uint(n.value))%100003);return o;}
  if(k<uint(n.m1)){hi=n.m1;d--;weight=(weight*6)%100003;}
  else if(k<uint(n.m2)){lo=n.m1;hi=n.m2;d--;}
  else{lo=n.m2;d-=1;weight=(weight*5)%100003;}
 }o.trap=3;return o;
}

struct Args {int a,b;uint D,groups;};
O replay(Args arg,uint gid,REF uint &stop){
 O o=zero();int lo=5,hi=0,d=arg.b;stop=0;
 if(!guarded(arg.a,arg.b)){if(gid==0)return fallback(arg.a,arg.b);return o;}
 hi=5+arg.a;for(uint depth=0;depth<arg.D;depth++){N n=node1(lo,hi,d,o);if(o.trap){stop=depth;return o;}if(n.leaf){o.value=n.value;stop=depth;return o;}
  uint digit=(gid >> ((arg.D-1-depth)*2))&3;
  if(digit>=3){stop=depth+1;return zero();}
  if(digit==0){hi=n.m1;d--;}
  else if(digit==1){lo=n.m1;hi=n.m2;d--;}
  else{lo=n.m2;d-=1;}
 }stop=arg.D;return serial(lo,hi,d);
}
kernel void A(constant Args &arg [[buffer(0)]],device O *part [[buffer(1)]],device O *out [[buffer(2)]],device uint *stops [[buffer(3)]],uint TID [[thread_index_in_threadgroup]],uint GROUP [[threadgroup_position_in_grid]]){
 if(!guarded(arg.a,arg.b)){if(TID==0)part[GROUP]=GROUP==0?fallback(arg.a,arg.b):zero();return;}
 threadgroup O v[256];uint gid=GROUP*256+TID;
 v[TID]=path(arg.a,arg.b,gid,true);threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint s=128;s;s/=2){if(TID<s)v[TID]=addo(v[TID],v[TID+s]);threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(TID==0)part[GROUP]=v[0];
}
kernel void reduce(constant Args &arg [[buffer(0)]],device O *part [[buffer(1)]],device O *out [[buffer(2)]],device uint *stops [[buffer(3)]],uint TID [[thread_index_in_threadgroup]],uint GROUP [[threadgroup_position_in_grid]]){
 if(!guarded(arg.a,arg.b)){if(TID==0)out[0]=part[0];return;}
 threadgroup O v[256];O o=zero();for(uint i=TID;i<arg.groups;i+=256)o=addo(o,part[i]);v[TID]=o;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint s=128;s;s/=2){if(TID<s)v[TID]=addo(v[TID],v[TID+s]);threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(TID==0)out[0]=v[0];
}
kernel void C2(constant Args &arg [[buffer(0)]],device O *part [[buffer(1)]],device O *out [[buffer(2)]],device uint *stops [[buffer(3)]],uint TID [[thread_index_in_threadgroup]],uint GROUP [[threadgroup_position_in_grid]]){
 if(!guarded(arg.a,arg.b)){if(TID==0)part[GROUP]=GROUP==0?fallback(arg.a,arg.b):zero();return;}
 threadgroup uint v[256];uint gid=GROUP*256+TID;O o=path_fast(arg.a,arg.b,gid,false);v[TID]=uint(o.value);threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint s=128;s;s/=2){if(TID<s)v[TID]=(v[TID]+v[TID+s])%100003;threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(TID==0){O r=zero();r.value=int(v[0]);part[GROUP]=r;}
}
kernel void C1(constant Args &arg [[buffer(0)]],device O *part [[buffer(1)]],device O *out [[buffer(2)]],device uint *stops [[buffer(3)]],uint TID [[thread_index_in_threadgroup]],uint GROUP [[threadgroup_position_in_grid]]){
 if(!guarded(arg.a,arg.b)){if(TID==0)out[0]=fallback(arg.a,arg.b);return;}
 threadgroup uint v[256];uint sum=0;for(uint gid=TID;gid<uint(max(arg.a,1));gid+=256)sum=(sum+uint(path_fast(arg.a,arg.b,gid,false).value))%100003;
 v[TID]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint s=128;s;s/=2){if(TID<s)v[TID]=(v[TID]+v[TID+s])%100003;threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(TID==0){O r=zero();r.value=int(v[0]);out[0]=r;}
}
kernel void B(constant Args &arg [[buffer(0)]],device O *part [[buffer(1)]],device O *out [[buffer(2)]],device uint *stops [[buffer(3)]],uint TID [[thread_index_in_threadgroup]],uint GROUP [[threadgroup_position_in_grid]]){
 threadgroup O v[256];threadgroup uint stop[256];uint gid=GROUP*256+TID;uint count=1u<<(arg.D*2);
 uint st=0;v[TID]=gid<count?replay(arg,gid,st):zero();stop[TID]=st;threadgroup_barrier(mem_flags::mem_threadgroup);
 uint levels=min(arg.D,4u);
 for(uint level=0,stride=1;level<levels;level++,stride*=4){
  uint depth=arg.D-1-level;
  if(TID%(stride*4)==0 && stop[TID]>depth){v[TID]=join1(v[TID],v[TID+stride],v[TID+stride*2]);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }if(TID==0){part[GROUP]=v[0];stops[GROUP]=stop[0];}
}
kernel void Btop(constant Args &arg [[buffer(0)]],device O *part [[buffer(1)]],device O *out [[buffer(2)]],device uint *stops [[buffer(3)]],uint TID [[thread_index_in_threadgroup]],uint GROUP [[threadgroup_position_in_grid]]){
 threadgroup O v[256];threadgroup uint stop[256];v[TID]=TID<arg.groups?part[TID]:zero();stop[TID]=TID<arg.groups?stops[TID]:0;threadgroup_barrier(mem_flags::mem_threadgroup);
 uint levels=arg.D>4?arg.D-4:0;
 for(uint level=0,stride=1;level<levels;level++,stride*=4){uint depth=levels-1-level;
  if(TID%(stride*4)==0&&stop[TID]>depth)v[TID]=join1(v[TID],v[TID+stride],v[TID+stride*2]);threadgroup_barrier(mem_flags::mem_threadgroup);
 }if(TID==0)out[0]=v[0];
}

kernel void C0(constant Args &arg [[buffer(0)]],device O *part [[buffer(1)]],device O *out [[buffer(2)]],device uint *stops [[buffer(3)]],uint TID [[thread_index_in_threadgroup]],uint GROUP [[threadgroup_position_in_grid]]){if(TID==0)out[0]=fallback(arg.a,arg.b);}
kernel void A_fused(constant Args &arg [[buffer(0)]],device O *part [[buffer(1)]],device O *out [[buffer(2)]],device uint *stops [[buffer(3)]],uint TID [[thread_index_in_threadgroup]],uint GROUP [[threadgroup_position_in_grid]]){
 if(!guarded(arg.a,arg.b)){if(TID==0)out[0]=fallback(arg.a,arg.b);return;}
 threadgroup O v[256];O o=zero();for(uint gid=TID;gid<uint(max(arg.a,1));gid+=256)o=addo(o,path(arg.a,arg.b,gid,true));v[TID]=o;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint s=128;s;s/=2){if(TID<s)v[TID]=addo(v[TID],v[TID+s]);threadgroup_barrier(mem_flags::mem_threadgroup);}if(TID==0)out[0]=v[0];
}
kernel void A_no_checks(constant Args &arg [[buffer(0)]],device O *part [[buffer(1)]],device O *out [[buffer(2)]],device uint *stops [[buffer(3)]],uint TID [[thread_index_in_threadgroup]],uint GROUP [[threadgroup_position_in_grid]]){
 if(!guarded(arg.a,arg.b)){if(TID==0)part[GROUP]=GROUP==0?fallback(arg.a,arg.b):zero();return;}
 threadgroup O v[256];uint gid=GROUP*256+TID;
 v[TID]=path(arg.a,arg.b,gid,false);threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint s=128;s;s/=2){if(TID<s)v[TID]=addo(v[TID],v[TID+s]);threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(TID==0)part[GROUP]=v[0];
}
kernel void A_narrow(constant Args &arg [[buffer(0)]],device O *part [[buffer(1)]],device O *out [[buffer(2)]],device uint *stops [[buffer(3)]],uint TID [[thread_index_in_threadgroup]],uint GROUP [[threadgroup_position_in_grid]]){
 if(!guarded(arg.a,arg.b)){if(TID==0)part[GROUP]=GROUP==0?fallback(arg.a,arg.b):zero();return;}
 threadgroup O v[256];uint gid=GROUP*256+TID;
 v[TID]=path_fast(arg.a,arg.b,gid,false);threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint s=128;s;s/=2){if(TID<s)v[TID]=addo(v[TID],v[TID+s]);threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(TID==0)part[GROUP]=v[0];
}
uint narrow_path(int a,int b,uint gid,bool unused){
 O o=zero();
 if(gid>=uint(max(a,1)))return 0;
 int lo=5,hi=5+a,d=b;uint k=5+gid;uint weight=1;
 for(uint depth=0;depth<64;depth++){N n=nodep0(lo,hi,d,k,o);
  if(n.leaf){if(uint(lo)!=k)return 0;return (weight*uint(n.value))%100003;}
  if(k<uint(n.m1)){hi=n.m1;d--;weight=(weight*6)%100003;}
  else if(k<uint(n.m2)){lo=n.m1;hi=n.m2;d--;}
  else{lo=n.m2;d-=1;weight=(weight*5)%100003;}
 }o.trap=3;return 0;
}

kernel void AProot(constant Args &arg [[buffer(0)]],device O *part [[buffer(1)]],device O *out [[buffer(2)]],device uint *stops [[buffer(3)]],uint TID [[thread_index_in_threadgroup]],uint GROUP [[threadgroup_position_in_grid]]){if(TID==0){O o=zero();N n=node0(5,5+arg.a,arg.b,o);o.value=n.value;out[0]=o;}}
kernel void APfused(constant Args &arg [[buffer(0)]],device O *part [[buffer(1)]],device O *out [[buffer(2)]],device uint *stops [[buffer(3)]],uint TID [[thread_index_in_threadgroup]],uint GROUP [[threadgroup_position_in_grid]]){
 threadgroup uint v[256];uint sum=0;
 for(uint gid=TID;gid<uint(max(arg.a,1));gid+=256)sum=(sum+narrow_path(arg.a,arg.b,gid,false))%100003;
 v[TID]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint s=128;s;s/=2){if(TID<s)v[TID]=(v[TID]+v[TID+s])%100003;threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(TID==0){O o=zero();o.value=int(v[0]);out[0]=o;}
}
kernel void APparallel(constant Args &arg [[buffer(0)]],device O *part [[buffer(1)]],device O *out [[buffer(2)]],device uint *stops [[buffer(3)]],uint TID [[thread_index_in_threadgroup]],uint GROUP [[threadgroup_position_in_grid]]){
 threadgroup uint v[256];v[TID]=narrow_path(arg.a,arg.b,GROUP*256+TID,false);threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint s=128;s;s/=2){if(TID<s)v[TID]=(v[TID]+v[TID+s])%100003;threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(TID==0)part[GROUP].value=int(v[0]);
}
kernel void APreduce(constant Args &arg [[buffer(0)]],device O *part [[buffer(1)]],device O *out [[buffer(2)]],device uint *stops [[buffer(3)]],uint TID [[thread_index_in_threadgroup]],uint GROUP [[threadgroup_position_in_grid]]){
 threadgroup uint v[256];v[TID]=TID<arg.groups?uint(part[TID].value):0;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint s=128;s;s/=2){if(TID<s)v[TID]=(v[TID]+v[TID+s])%100003;threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(TID==0){O o=zero();o.value=int(v[0]);out[0]=o;}
}
