// Gather and frontier arithmetic from M5-01 06d2633c metal-families.mm.
// Baked sizes replace constant arguments. Scatter/histogram are supplementary
// independent direct kernels: that branch has no M5-01 baseline for them.
#include <metal_stdlib>
using namespace metal;
kernel void gather_hw(device const uint* a [[buffer(0)]], device const uint* ix [[buffer(1)]], device uint* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < @N@u) out[i] = a[ix[i]];
}
kernel void scatter_hw(device const uint* a [[buffer(0)]], device const uint* ix [[buffer(1)]], device uint* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < @N@u) out[ix[i]] = a[i];
}
kernel void histogram_hw(device const uint* a [[buffer(0)]], device const uint* ix [[buffer(1)]], device atomic_uint* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < @N@u) atomic_fetch_add_explicit(&out[ix[i]], a[i], memory_order_relaxed);
}
kernel void frontier_round(device const uint* edges [[buffer(0)]], device uint* seen [[buffer(1)]], device const uint* frontier [[buffer(2)]], device atomic_uint* next [[buffer(3)]], device atomic_uint* live [[buffer(4)]], uint u [[thread_position_in_grid]]) {
  if (u >= @V@u || frontier[u] == 0 || seen[u] != 0) return;
  seen[u] = 1;
  for (uint w = 0; w < @V@u; w++) {
    if (edges[u * @V@u + w] == 1) {
      atomic_store_explicit(&next[w], 1u, memory_order_relaxed);
      atomic_store_explicit(&live[0], 1u, memory_order_relaxed);
    }
  }
}
