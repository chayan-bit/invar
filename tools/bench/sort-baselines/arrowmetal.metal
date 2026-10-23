// Adapted from ArrowMetal SortSource.swift at 1f2111540e68edfdf1adf87df8821ca9cd1958c0.
// Apache-2.0; see arrowmetal-LICENSE and NOTICE.md.
#include <metal_stdlib>
using namespace metal;
#define TG 256u
    #define RADIX 256u
    #define DIGIT_BITS 8u
    #define DIGIT_MASK 0xFFu
    // Apple GPUs are 32 lanes wide, which the scan below already assumes (`simdTotals[32]`).
    #define SIMDS (TG / 32u)

    // Histogram of digit `shift` per block of `elemsPerBlock` elements, counts laid out digit-major:
    // counts[d * blocks + b]. The same sweep records the bitwise OR and AND of the block's keys, which
    // together say which bits differ anywhere in the column and so which whole passes are identity
    // permutations that can be skipped (see `argsort`). It is free here: the keys are already loaded.
    kernel void radix_histogram(device const uint* keys [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                                constant uint& shift [[buffer(2)]], constant uint& elemsPerBlock [[buffer(3)]],
                                constant uint& blocks [[buffer(4)]], device atomic_uint* counts [[buffer(5)]],
                                device uint* spanOr [[buffer(6)]], device uint* spanAnd [[buffer(7)]],
                                uint lid [[thread_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]]) {
        threadgroup atomic_uint hist[RADIX];
        threadgroup atomic_uint orLo, orHi, andLo, andHi;
        for (uint d = lid; d < RADIX; d += TG) atomic_store_explicit(&hist[d], 0u, memory_order_relaxed);
        if (lid == 0) {
            atomic_store_explicit(&orLo, 0u, memory_order_relaxed);
            atomic_store_explicit(&orHi, 0u, memory_order_relaxed);
            atomic_store_explicit(&andLo, 0xFFFFFFFFu, memory_order_relaxed);
            atomic_store_explicit(&andHi, 0xFFFFFFFFu, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint n = *nPtr, start = tgid * elemsPerBlock, len = (start < n) ? min(elemsPerBlock, n - start) : 0u;
        uint o = 0, a = ~(uint)0;
        for (uint off = lid; off < len; off += TG) { uint i = start + off;
            uint k = keys[i];
            o |= k; a &= k;
            atomic_fetch_add_explicit(&hist[(uint)((k >> shift) & DIGIT_MASK)], 1u, memory_order_relaxed);
        }
        atomic_fetch_or_explicit(&orLo, (uint)o, memory_order_relaxed);
        atomic_fetch_and_explicit(&andLo, (uint)a, memory_order_relaxed);
        if (sizeof(uint) > 4) {
            atomic_fetch_or_explicit(&orHi, (uint)((ulong)o >> 32), memory_order_relaxed);
            atomic_fetch_and_explicit(&andHi, (uint)((ulong)a >> 32), memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = lid; d < RADIX; d += TG)
            atomic_store_explicit(&counts[d * blocks + tgid], atomic_load_explicit(&hist[d], memory_order_relaxed), memory_order_relaxed);
        if (lid == 0) {
            ulong ov = (ulong)atomic_load_explicit(&orLo, memory_order_relaxed);
            ulong av = (ulong)atomic_load_explicit(&andLo, memory_order_relaxed);
            if (sizeof(uint) > 4) {
                ov |= (ulong)atomic_load_explicit(&orHi, memory_order_relaxed) << 32;
                av |= (ulong)atomic_load_explicit(&andHi, memory_order_relaxed) << 32;
            }
            // An empty block must not claim every bit is constant, so it reports "nothing set" for the
            // OR and "nothing clear" for the AND, which is what the identities above already give.
            spanOr[tgid] = (uint)ov; spanAnd[tgid] = (uint)av;
        }
    }
    // Exclusive scan over the digit-major counts table (RADIX * blocks entries), single threadgroup.
    kernel void radix_scan(device uint* counts [[buffer(0)]], constant uint& total [[buffer(1)]],
                           uint lid [[thread_index_in_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
        threadgroup uint simdTotals[32];
        uint per = (total + TG - 1) / TG;
        uint lo = lid * per, hi = min(total, lo + per);
        uint local = 0;
        for (uint i = lo; i < hi; i++) local += counts[i];
        uint pre = simd_prefix_exclusive_sum(local);
        uint t = simd_sum(local);
        if (lane == 0) simdTotals[sgid] = t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint prefix = 0;
        for (uint k = 0; k < sgid; k++) prefix += simdTotals[k];
        uint run = prefix + pre;
        for (uint i = lo; i < hi; i++) { uint c = counts[i]; counts[i] = run; run += c; }
    }
    kernel void radix_scatter(device const uint* keys [[buffer(0)]], device const uint* vals [[buffer(1)]],
                              device const uint* nPtr [[buffer(2)]], constant uint& shift [[buffer(3)]],
                              constant uint& elemsPerBlock [[buffer(4)]], constant uint& blocks [[buffer(5)]],
                              device const uint* offsets [[buffer(6)]],
                              device uint* outKeys [[buffer(7)]], device uint* outVals [[buffer(8)]],
                              uint lid [[thread_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]],
                              uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
        threadgroup uint base[RADIX];
        threadgroup uchar peerCount[SIMDS * RADIX];
        for (uint d = lid; d < RADIX; d += TG) base[d] = offsets[d * blocks + tgid];
        for (uint j = lid; j < SIMDS * RADIX; j += TG) peerCount[j] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint n = *nPtr, start = tgid * elemsPerBlock, len = (start < n) ? min(elemsPerBlock, n - start) : 0u, end = start + len;
        for (uint off = 0u; off < len; off += TG) {
            uint chunk = start + off, i = chunk + lid;
            bool active = off + lid < len;
            uint key = active ? keys[i] : (uint)0;
            uint val = active ? vals[i] : 0u;
            uint d = (uint)((key >> shift) & DIGIT_MASK);
            // The lanes of this SIMD group that are inside the block, then those of them holding this
            // digit. `simd_ballot` must be reached by every lane, so the mask is arithmetic, not a vote.
            int avail = (int)end - (int)(chunk + sgid * 32u);
            uint peers = avail >= 32 ? 0xFFFFFFFFu : (avail <= 0 ? 0u : (0xFFFFFFFFu >> (32 - (uint)avail)));
            for (uint b = 0u; b < DIGIT_BITS; b++) {
                bool one = ((d >> b) & 1u) != 0u;
                uint ones = (uint)((simd_vote::vote_t)simd_ballot(one));
                peers &= one ? ones : ~ones;
            }
            uint rank = popcount(peers & ((1u << lane) - 1u));
            bool leader = active && rank == 0u;
            if (leader) peerCount[sgid * RADIX + d] = (uchar)popcount(peers);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint before = 0u, total = 0u;
            if (active) {
                for (uint s = 0u; s < SIMDS; s++) {
                    uint c = peerCount[s * RADIX + d];
                    before += (s < sgid) ? c : 0u;
                    total += c;
                }
                uint pos = base[d] + before + rank;
                outKeys[pos] = key;
                outVals[pos] = val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (leader) {
                peerCount[sgid * RADIX + d] = 0;        // ready for the next chunk, no bulk clear needed
                if (before == 0u) base[d] += total;     // the digit's first SIMD group advances the base
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
