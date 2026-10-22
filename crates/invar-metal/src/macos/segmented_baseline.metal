// Adapted signatures/constants from M5-01 metal-families.mm (SHA-256 92401663dfb24ec184a677e67aa239f5a59245069db6f84ab6135489e90439f5).
// Arithmetic and baseline schedules unchanged; map16 is an independent oracle.
#include <metal_stdlib>
using namespace metal;
constant uint N = @N@u;
constant uint segments = N / 16u;
constant uint groups = (N + 255u) / 256u;
constant uint n = N;
kernel void segsum(device const uint* a [[buffer(0)]], device uint* out [[buffer(1)]],
                   uint s [[thread_position_in_grid]]) {
  if (s >= segments) return;
  uint acc = 0;
  for (uint i = s * 16; i < s * 16 + 16; i++) acc += a[i];
  out[s] = acc;
}
kernel void compact_count(device const uint* a [[buffer(0)]], device uint* counts [[buffer(1)]],
                          uint i [[thread_position_in_grid]],
                          uint l [[thread_position_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
  threadgroup uint part[256];
  part[l] = (i < n && a[i] < 0x80000000u) ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 128; s > 0; s >>= 1) {
    if (l < s) part[l] += part[l + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (l == 0) counts[g] = part[0];
}
// One threadgroup of 1024 threads: chunked sums, a threadgroup scan of the
// 1024 chunk totals, then exclusive offsets written back per chunk.
kernel void compact_offsets(device uint* counts [[buffer(0)]], device uint* total [[buffer(1)]],
                            uint l [[thread_position_in_threadgroup]]) {
  threadgroup uint scan[1024];
  uint chunk = (groups + 1023) / 1024, lo = min(l * chunk, groups), hi = min(lo + chunk, groups);
  uint sum = 0;
  for (uint g = lo; g < hi; g++) sum += counts[g];
  scan[l] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 1; s < 1024; s <<= 1) {
    uint add = l >= s ? scan[l - s] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scan[l] += add;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint run = scan[l] - sum;
  for (uint g = lo; g < hi; g++) { uint c = counts[g]; counts[g] = run; run += c; }
  if (l == 1023) total[0] = scan[1023];
}
kernel void compact_scatter(device const uint* a [[buffer(0)]], device const uint* offsets [[buffer(1)]],
                            device uint* out [[buffer(2)]], uint i [[thread_position_in_grid]], uint l [[thread_position_in_threadgroup]],
                            uint g [[threadgroup_position_in_grid]]) {
  threadgroup uint scan[256];
  bool keep = i < n && a[i] < 0x80000000u;
  scan[l] = keep ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 1; s < 256; s <<= 1) {
    uint add = l >= s ? scan[l - s] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scan[l] += add;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (keep) out[offsets[g] + scan[l] - 1] = a[i];
}

kernel void map16(device const uint* a [[buffer(0)]], device uint* out [[buffer(1)]], uint s [[thread_position_in_grid]]) {
  if (s >= segments) return;
  for (uint i = s * 16; i < s * 16 + 16; i++) out[i] = (a[i] + 7u) * 2654435761u + s;
}
