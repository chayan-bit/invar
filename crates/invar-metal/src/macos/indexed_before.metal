kernel void invar_frontier_expand_v1(device const uint* values [[buffer(0)]], device const uint* indices [[buffer(1)]], device uint* output [[buffer(2)]], device uint* frontier [[buffer(3)]], device uint* next [[buffer(4)]], device uint* queue [[buffer(5)]], device uint* control [[buffer(6)]], constant uint& round [[buffer(7)]], uint tid [[thread_position_in_grid]]) {
  if (tid!=0 || round>=16u || control[1]!=0) return;
  for (unsigned int v=0;v<@V@u;++v) next[v]=0;
  for (unsigned int k=0;k<control[0];++k) {
    unsigned int u=queue[k];
    if (frontier[u]!=1 || output[u]!=0) continue;
    output[u]=1;
    for (unsigned int v=0;v<@V@u;++v) if (values[u*@V@u+v]==1 && output[v]==0) next[v]=1;
  }
}
kernel void invar_frontier_compact_v1(device const uint* values [[buffer(0)]], device const uint* indices [[buffer(1)]], device uint* output [[buffer(2)]], device uint* frontier [[buffer(3)]], device uint* next [[buffer(4)]], device uint* queue [[buffer(5)]], device uint* control [[buffer(6)]], constant uint& round [[buffer(7)]], uint tid [[thread_position_in_grid]]) {
  if (tid!=0 || round>=16u || control[1]!=0) return;
  unsigned int count=0;
  for (unsigned int v=0;v<@V@u;++v) {
    frontier[v]=next[v];
    output[@V@u*(round+1u)+v]=next[v];
    if (next[v]==1) { if (count>=@V@u) { control[1]=1; return; } queue[count++]=v; }
  }
  control[0]=count;
}
