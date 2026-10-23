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
int leaf1(int x, REF O &o) {int z0=ck(x,2,4,34,o);int z1=(z0 == 0);if (z1) {int z2=ck(x,2,3,55,o);int z3=ck(z2,16,0,59,o);int z4=ck(z3,10007,4,65,o);return z4;}int z5=ck(x,3,2,90,o);int z6=ck(z5,1,0,94,o);int z7=ck(z6,10007,4,99,o);return z7;}
N node1(int lo,int hi,int d,REF O &o) {N n={0,0,0,0};int z8=ck(hi,lo,1,161,o);int z9=(z8 <= 3);if (z9) {int z10=ck(lo,6,2,188,o);int z11=ck(z10,hi,0,192,o);int z12=leaf1(z11,o);n.leaf=1;n.value=z12;return n;}int z13=(d <= 0);if (z13) {int z14=ck(hi,lo,1,232,o);int z15=ck(z14,8,0,237,o);int z16=leaf1(z15,o);n.leaf=1;n.value=z16;return n;}int z17=ck(lo,1,0,261,o);int z18=ck(hi,lo,1,272,o);int z19=ck(z18,2,1,277,o);int z20=ck(lo,2,4,293,o);int z21=ck(z20,2,0,297,o);int z22=ck(z21,2,4,302,o);int z23=ck(2,z22,0,287,o);int z24=ck(z19,z23,2,282,o);int z25=ck(z24,7,3,308,o);int z26=ck(z17,z25,0,265,o);int m1=z26;n.m1=m1;return n;}
O join1(O aa,O bb,O cc) {if(aa.trap)return aa;if(bb.trap)return bb;if(cc.trap)return cc;O o=zero();int a=aa.value,b=bb.value,c=cc.value;int z27=ck(a,3,2,389,o);int z28=ck(z27,b,0,393,o);int z29=ck(z28,100003,4,398,o);o.value=z29;return o;}
int leaf0(int x, REF O &o) {int z0=(x % 2);int z1=(z0 == 0);if (z1) {int z2=(x / 2);int z3=(z2 + 16);int z4=(z3 % 10007);return z4;}int z5=(x * 3);int z6=(z5 + 1);int z7=(z6 % 10007);return z7;}
N node0(int lo,int hi,int d,REF O &o) {N n={0,0,0,0};int z8=(hi - lo);int z9=(z8 <= 3);if (z9) {int z10=(lo * 6);int z11=(z10 + hi);int z12=leaf0(z11,o);n.leaf=1;n.value=z12;return n;}int z13=(d <= 0);if (z13) {int z14=(hi - lo);int z15=(z14 + 8);int z16=leaf0(z15,o);n.leaf=1;n.value=z16;return n;}int z17=(lo + 1);int z18=(hi - lo);int z19=(z18 - 2);int z20=(lo % 2);int z21=(z20 + 2);int z22=(z21 % 2);int z23=(2 + z22);int z24=(z19 * z23);int z25=(z24 / 7);int z26=(z17 + z25);int m1=z26;n.m1=m1;return n;}
O join0(O aa,O bb,O cc) {if(aa.trap)return aa;if(bb.trap)return bb;if(cc.trap)return cc;O o=zero();int a=aa.value,b=bb.value,c=cc.value;int z27=(a * 3);int z28=(z27 + b);int z29=(z28 % 100003);o.value=z29;return o;}
struct F {int lo,hi,d,phase,m1,m2;O a,b;};
O serial(int lo,int hi,int d) {
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
O fallback(int a,int b){O o=zero();int hi=ck(33,a,0,459,o);if(o.trap)return o;return serial(33,hi,b);}

bool guarded(int a,int b){return a>=0&&a<=20000&&b>=0&&b<=30;}
N nodep0(int lo,int hi,int d,uint key,REF O &o) {N n={0,0,0,0};int z8=(hi - lo);int z9=(z8 <= 3);if (z9) {if(uint(lo)!=key){n.leaf=1;return n;}int z10=(lo * 6);int z11=(z10 + hi);int z12=leaf0(z11,o);n.leaf=1;n.value=z12;return n;}int z13=(d <= 0);if (z13) {if(uint(lo)!=key){n.leaf=1;return n;}int z14=(hi - lo);int z15=(z14 + 8);int z16=leaf0(z15,o);n.leaf=1;n.value=z16;return n;}int z17=(lo + 1);int z18=(hi - lo);int z19=(z18 - 2);int z20=(lo % 2);int z21=(z20 + 2);int z22=(z21 % 2);int z23=(2 + z22);int z24=(z19 * z23);int z25=(z24 / 7);int z26=(z17 + z25);int m1=z26;n.m1=m1;return n;}
N nodep1(int lo,int hi,int d,uint key,REF O &o) {N n={0,0,0,0};int z8=ck(hi,lo,1,161,o);int z9=(z8 <= 3);if (z9) {if(uint(lo)!=key){n.leaf=1;return n;}int z10=ck(lo,6,2,188,o);int z11=ck(z10,hi,0,192,o);int z12=leaf1(z11,o);n.leaf=1;n.value=z12;return n;}int z13=(d <= 0);if (z13) {if(uint(lo)!=key){n.leaf=1;return n;}int z14=ck(hi,lo,1,232,o);int z15=ck(z14,8,0,237,o);int z16=leaf1(z15,o);n.leaf=1;n.value=z16;return n;}int z17=ck(lo,1,0,261,o);int z18=ck(hi,lo,1,272,o);int z19=ck(z18,2,1,277,o);int z20=ck(lo,2,4,293,o);int z21=ck(z20,2,0,297,o);int z22=ck(z21,2,4,302,o);int z23=ck(2,z22,0,287,o);int z24=ck(z19,z23,2,282,o);int z25=ck(z24,7,3,308,o);int z26=ck(z17,z25,0,265,o);int m1=z26;n.m1=m1;return n;}
O path(int a,int b,uint gid,bool wide){
 O o=zero();if(!guarded(a,b)){if(gid==0)return fallback(a,b);return o;}
 if(gid>=uint(max(a,1)))return o;
 int lo=33,hi=33+a,d=b;uint k=33+gid;UWIDE weight=1;
 for(uint depth=0;depth<64;depth++){N n=wide?nodep1(lo,hi,d,k,o):nodep0(lo,hi,d,k,o);if(o.trap){o.key=uint(lo-33)*64+depth;return o;}
  if(n.leaf){if(uint(lo)!=k)return zero();o.value=int((weight*UWIDE(n.value))%100003);return o;}
  if(k<uint(n.m1)){hi=n.m1;d--;weight=(weight*3)%100003;}
  
  else{lo=n.m1;d-=1;weight=(weight*1)%100003;}
 }o.trap=3;return o;
}
O addo(O a,O b){if(a.trap&&(!b.trap||a.key<=b.key))return a;if(b.trap)return b;O o=zero();o.value=int((UWIDE(a.value)+UWIDE(b.value))%100003);return o;}
O path_fast(int a,int b,uint gid,bool wide){
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
O replay(Args arg,uint gid,REF uint &stop){
 O o=zero();int lo=33,hi=0,d=arg.b;stop=0;
 if(!guarded(arg.a,arg.b)){if(gid==0)return fallback(arg.a,arg.b);return o;}
 hi=33+arg.a;for(uint depth=0;depth<arg.D;depth++){N n=node1(lo,hi,d,o);if(o.trap){stop=depth;return o;}if(n.leaf){o.value=n.value;stop=depth;return o;}
  uint digit=(gid >> ((arg.D-1-depth)*1))&1;
  if(digit>=2){stop=depth+1;return zero();}
  if(digit==0){hi=n.m1;d--;}
  
  else{lo=n.m1;d-=1;}
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
 threadgroup O v[256];threadgroup uint stop[256];uint gid=GROUP*256+TID;uint count=1u<<(arg.D*1);
 uint st=0;v[TID]=gid<count?replay(arg,gid,st):zero();stop[TID]=st;threadgroup_barrier(mem_flags::mem_threadgroup);
 uint levels=min(arg.D,8u);
 for(uint level=0,stride=1;level<levels;level++,stride*=2){
  uint depth=arg.D-1-level;
  if(TID%(stride*2)==0 && stop[TID]>depth){v[TID]=join1(v[TID],v[TID+stride],zero());}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }if(TID==0){part[GROUP]=v[0];stops[GROUP]=stop[0];}
}
kernel void Btop(constant Args &arg [[buffer(0)]],device O *part [[buffer(1)]],device O *out [[buffer(2)]],device uint *stops [[buffer(3)]],uint TID [[thread_index_in_threadgroup]],uint GROUP [[threadgroup_position_in_grid]]){
 threadgroup O v[256];threadgroup uint stop[256];v[TID]=TID<arg.groups?part[TID]:zero();stop[TID]=TID<arg.groups?stops[TID]:0;threadgroup_barrier(mem_flags::mem_threadgroup);
 uint levels=arg.D>8?arg.D-8:0;
 for(uint level=0,stride=1;level<levels;level++,stride*=2){uint depth=levels-1-level;
  if(TID%(stride*2)==0&&stop[TID]>depth)v[TID]=join1(v[TID],v[TID+stride],zero());threadgroup_barrier(mem_flags::mem_threadgroup);
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
 int lo=33,hi=33+a,d=b;uint k=33+gid;uint weight=1;
 for(uint depth=0;depth<64;depth++){N n=nodep0(lo,hi,d,k,o);
  if(n.leaf){if(uint(lo)!=k)return 0;return (weight*uint(n.value))%100003;}
  if(k<uint(n.m1)){hi=n.m1;d--;weight=(weight*3)%100003;}
  
  else{lo=n.m1;d-=1;weight=(weight*1)%100003;}
 }o.trap=3;return 0;
}

kernel void AProot(constant Args &arg [[buffer(0)]],device O *part [[buffer(1)]],device O *out [[buffer(2)]],device uint *stops [[buffer(3)]],uint TID [[thread_index_in_threadgroup]],uint GROUP [[threadgroup_position_in_grid]]){if(TID==0){O o=zero();N n=node0(33,33+arg.a,arg.b,o);o.value=n.value;out[0]=o;}}
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
