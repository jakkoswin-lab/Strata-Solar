// src/glm4/glm4_engine.cu - the Solar-Open-100B (GGUF architecture glm4moe) engine path.
//
// One GPU, the routed experts either in VRAM or (the 100B file on a 16 GB card) in the GGUF's mmap, one expert blob
// crossing PCIe per expert per matrix.  The quantized matmuls run on Strata's i-quant MMVQ (IQ2_S / IQ3_S / IQ3_XXS /
// Q5_K ...), so a weight means what it means in llama.cpp; the reference path in glm4ref.cu is the same arithmetic,
// one token at a time, checked against llama.cpp's logits.
//
// What this engine adds over the reference: a prompt chunk (T rows share one matmul per matrix and one attention
// kernel), a KV cache per layer, sessions (--batch N) whose caches and positions are swapped by select(), KV reuse
// across requests (the serve protocol's match_len), the sampler, and the serve protocol of serve/server.py.
//
// The serve protocol (one request at a time on the control lines):
//   GEN <max_new> [key=value ...] <id,id,...>        -- read the prompt, then decode
//   BGEN <slot> <max_new> [keys] <ids>               -- into a batch slot (--batch N >= 2)
//   BSTOP <slot> / BYIELD / STOP                     -- as in the qwen35 engine
// Replies: READY <ctx> stop, PP <done> <total> <ms> <tok_s>, T <id>, BT <slot> <id>, DONE ..., BADM, ERR.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/glm4/engine.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/qwen35/sampling.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-glm4: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__,         \
                         __LINE__);                                                                                    \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

// ------------------------------------------------------------------------------------------------ kernels
enum : int { T_F32 = 0, T_F16 = 1, T_BF16 = 30 };

__device__ __forceinline__ float h2f(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) {
        const float v = ldexpf((float) m, -24);
        return s ? -v : v;
    }
    const uint32_t f = e == 31 ? (s | 0x7f800000u | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
    return __uint_as_float(f);
}
__device__ __forceinline__ uint16_t ld16(const uint8_t* p) { return (uint16_t) (p[0] | (p[1] << 8)); }

template <int T> __device__ void deq32(const uint8_t* row, int c, float* out) {
    if constexpr (T == T_F32) {
        const float* r = (const float*) row + 32 * c;
        for (int l = 0; l < 32; ++l) out[l] = r[l];
    } else if constexpr (T == T_F16) {
        const uint8_t* r = row + 64 * c;
        for (int l = 0; l < 32; ++l) out[l] = h2f(ld16(r + 2 * l));
    } else if constexpr (T == T_BF16) {
        const uint8_t* r = row + 64 * c;
        for (int l = 0; l < 32; ++l) out[l] = __uint_as_float((uint32_t) ld16(r + 2 * l) << 16);
    }
}

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

__device__ float block_max(float v, float* sh);   // defined with the attention kernels below

constexpr int kThreads = 128;

template <int T>
__global__ void k_gemv(const uint8_t* W, size_t row_bytes, size_t expert_bytes, const int* ids, int n_in, const float* x,
                       size_t x_stride, float* y, int n_out) {
    __shared__ float sh[kThreads];
    const int o = blockIdx.x, sel = blockIdx.y;
    const int e = ids ? ids[sel] : 0;
    const uint8_t* row = W + (size_t) e * expert_bytes + (size_t) o * row_bytes;
    const float* xs = x + (size_t) sel * x_stride;
    float acc = 0.f, v[32];
    for (int c = threadIdx.x; c < n_in / 32; c += blockDim.x) {
        deq32<T>(row, c, v);
        const float* xc = xs + 32 * c;
        for (int l = 0; l < 32; ++l) acc += v[l] * xc[l];
    }
    acc = block_sum(acc, sh);
    if (threadIdx.x == 0) y[(size_t) sel * n_out + o] = acc;
}

/// y[row][o] = W[o] . x[row] for the f32 router (W [n_out][n_in] row-major).
__global__ void k_gemv_f32_rows(const float* W, const float* x, float* y, int n_in, int n_out, int xstride) {
    __shared__ float sh[kThreads];
    const int o = blockIdx.x, r = blockIdx.y;
    const float* xr = x + (size_t) r * xstride;
    const float* wr = W + (size_t) o * n_in;
    float acc = 0.f;
    if (n_in == 4096 && blockDim.x == 128) {
        // Solar's only router width.  32 register loads issued together instead of 32 dependent round trips, and the
        // sum is still taken in the same order, so the logits are bit-identical.
        float a[32], b[32];
#pragma unroll
        for (int k = 0; k < 32; ++k) a[k] = wr[threadIdx.x + 128 * k];
#pragma unroll
        for (int k = 0; k < 32; ++k) b[k] = xr[threadIdx.x + 128 * k];
#pragma unroll
        for (int k = 0; k < 32; ++k) acc += a[k] * b[k];
    } else {
        for (int i = threadIdx.x; i < n_in; i += blockDim.x) acc += wr[i] * xr[i];
    }
    acc = block_sum(acc, sh);
    if (threadIdx.x == 0) y[(size_t) r * n_out + o] = acc;
}

/// Router selection on the device: sigmoid, add the selection bias, keep the top `Kp` (descending score, ties to the
/// lower expert index - byte for byte the ordering the host's partial_sort used) and renormalise the *unbiased*
/// weights.  One thread per row is enough for 128 experts, and it takes the host out of the loop: the only thing it
/// still needs is the id list, for the expert-cache lookup.
__global__ void k_route_topk(const float* __restrict__ logits, const float* __restrict__ bias, int* __restrict__ ids,
                             float* __restrict__ weights, int ne, int Kp, float route_scale, int weights_norm) {
    __shared__ float score[256], raw[256];
    const int row = blockIdx.x;
    const float* lg = logits + (size_t) row * ne;
    const int lim = ne < 256 ? ne : 256;
    for (int i = threadIdx.x; i < lim; i += blockDim.x) {
        const float p = 1.0f / (1.0f + expf(-lg[i]));
        raw[i] = p;
        score[i] = p + bias[i];
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    const int k = Kp < 8 ? Kp : 8;
    // The best eight live in registers.  This used to be a dynamically indexed int[32]/float[32], which the compiler
    // puts in local memory - a single thread doing ~1100 dependent local-memory accesses cost **48 us per layer**
    // (2.3 ms/token), more than the rest of the router put together.  Slots are filled with -inf/INT_MAX so a new
    // entry can always be dropped into the last slot and bubbled up with compile-time indices.
    float tv[8];
    int ti[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        tv[j] = -INFINITY;
        ti[j] = 0x7fffffff;
    }
    for (int i = 0; i < lim; ++i) {
        const float v = score[i];
        if (!(v > tv[7] || (v == tv[7] && i < ti[7]))) continue;
        tv[7] = v;
        ti[7] = i;
#pragma unroll
        for (int j = 7; j > 0; --j) {
            if (tv[j] > tv[j - 1] || (tv[j] == tv[j - 1] && ti[j] < ti[j - 1])) {
                const float a = tv[j - 1];
                tv[j - 1] = tv[j];
                tv[j] = a;
                const int b = ti[j - 1];
                ti[j - 1] = ti[j];
                ti[j] = b;
            }
        }
    }
    float ws = 0.f;
    for (int j = 0; j < k; ++j) ws += raw[ti[j]];
    for (int j = 0; j < k; ++j) {
        const int e = ti[j];
        const float p = raw[e];
        ids[(size_t) row * Kp + j] = e;
        weights[(size_t) row * Kp + j] = weights_norm ? p / ws : p * route_scale;
    }
}

// blocks per segment of one expert blob in k_fetch: 8 misses x 3 segments x this many blocks is the whole grid,
// and the copy is memory-latency bound, so more blocks mean more bytes in flight
static const int kFetchChunks = [] {
    const char* v = std::getenv("STRATA_GLM4_FETCH_CHUNKS");
    const int k = v ? std::atoi(v) : 4;
    return k > 0 && k <= 64 ? k : 4;
}();

/// Decode's expert-cache directory on the device.  For each of the Kp selected experts: find its slot, or take an
/// LRU victim, update the directory and record a miss; then write the two weight-pointer tables the matvecs read.
/// One thread, because Kp is 8 and the LRU scan is over ~2000 ints - cheaper than the host round-trip it replaces.
__global__ void k_resolve(const int32_t* __restrict__ ids, int Kp, int layer, int ne, int slots, size_t stride,
                          size_t off_d, uint8_t* __restrict__ cache, int32_t* __restrict__ sl,
                          int32_t* __restrict__ se, int32_t* __restrict__ so, unsigned long long* __restrict__ used,
                          unsigned long long* __restrict__ tick, uint32_t* __restrict__ freq,
                          uint8_t** __restrict__ wp_g, uint8_t** __restrict__ wp_d,
                          int32_t* __restrict__ miss_e, int32_t* __restrict__ miss_slot, int* __restrict__ miss_n,
                          int* __restrict__ hits, int* __restrict__ misses) {
    // One entry per (layer, expert), 39 slots per layer at this VRAM size, and a token touches 8 experts per layer:
    // a global LRU therefore only remembers ~5 tokens back, and every layer competes with the other 47.  A frequency
    // count with periodic halving keeps the experts a layer keeps coming back to instead, and falls back to recency
    // among equals.  `freq` null selects plain LRU.
    //
    // The victim search is the whole block's job: with one thread it was a walk over ~1900 slots in VRAM that cost
    // 215 us per layer (10 ms/token at 1500 positions) because the two arrays are read one dependent load at a time.
    __shared__ uint32_t sh_f[128];
    __shared__ unsigned long long sh_u[128];
    __shared__ int sh_i[128], sh_slot;
    const int tid = threadIdx.x;
    int m = 0;
    if (tid == 0 && freq && ((++*tick) & 8191ull) == 0) {
        for (int i = 0; i < slots; ++i) freq[i] >>= 1;
    }
    __syncthreads();
    for (int c = 0; c < Kp; ++c) {
        if (tid == 0) {
            const int e = ids[c];
            const int s = so[layer * ne + e];
            sh_slot = s;
            if (s >= 0) {
                ++*hits;
                used[s] = ++*tick;
                if (freq && freq[s] < (1u << 30)) ++freq[s];
            } else {
                ++*misses;
            }
        }
        __syncthreads();
        if (sh_slot < 0) {
            // (freq, used, index) minimum over the whole cache; a never-used slot has used == 0 and so always wins
            uint32_t bf = ~0u;
            unsigned long long bu = ~0ull;
            int bi = slots;
            for (int i = tid; i < slots; i += blockDim.x) {
                const uint32_t f = freq ? freq[i] : 0u;
                const unsigned long long u = used[i];
                if (f < bf || (f == bf && (u < bu || (u == bu && i < bi)))) {
                    bf = f;
                    bu = u;
                    bi = i;
                }
            }
            sh_f[tid] = bf;
            sh_u[tid] = bu;
            sh_i[tid] = bi;
            __syncthreads();
            for (int st = blockDim.x / 2; st > 0; st >>= 1) {
                if (tid < st) {
                    const uint32_t f2 = sh_f[tid + st];
                    const unsigned long long u2 = sh_u[tid + st];
                    const int i2 = sh_i[tid + st];
                    if (f2 < sh_f[tid] || (f2 == sh_f[tid] && (u2 < sh_u[tid] || (u2 == sh_u[tid] && i2 < sh_i[tid])))) {
                        sh_f[tid] = f2;
                        sh_u[tid] = u2;
                        sh_i[tid] = i2;
                    }
                }
                __syncthreads();
            }
            if (tid == 0) {
                const int v = sh_i[0];
                const int e = ids[c];
                if (sl[v] >= 0) so[sl[v] * ne + se[v]] = -1;
                sl[v] = layer;
                se[v] = e;
                used[v] = ++*tick;
                if (freq) freq[v] = 1;
                so[layer * ne + e] = v;
                miss_e[m] = e;
                miss_slot[m] = v;
                ++m;
                sh_slot = v;
            }
            __syncthreads();
        }
        if (tid == 0) {
            wp_g[c] = cache + (size_t) sh_slot * stride;
            wp_d[c] = wp_g[c] + off_d;
        }
        __syncthreads();
    }
    if (tid == 0) *miss_n = m;
}

/// Fill the slots k_resolve just marked, straight out of the pinned expert region.  One block per (miss, segment,
/// chunk): the three segments are gate, up and down, and each block copies a slice of one of them.
__global__ void k_fetch(const int32_t* __restrict__ miss_e, const int32_t* __restrict__ miss_slot,
                        const int* __restrict__ miss_n, const uint8_t* gsrc, size_t gbytes, const uint8_t* usrc,
                        size_t ubytes, const uint8_t* dsrc, size_t dbytes, size_t off_u, size_t off_d,
                        uint8_t* __restrict__ cache, size_t stride, int chunks, int seg0) {
    const int m = blockIdx.x / chunks, c = blockIdx.x % chunks, seg = seg0 + blockIdx.y;
    if (m >= *miss_n) return;
    const int e = miss_e[m];
    const uint8_t* src = seg == 0 ? gsrc + (size_t) e * gbytes
                                  : seg == 1 ? usrc + (size_t) e * ubytes : dsrc + (size_t) e * dbytes;
    const size_t bytes = seg == 0 ? gbytes : seg == 1 ? ubytes : dbytes;
    uint8_t* dst = cache + (size_t) miss_slot[m] * stride + (seg == 0 ? 0 : seg == 1 ? off_u : off_d);
    const size_t lo = bytes * (size_t) c / (size_t) chunks, hi = bytes * (size_t) (c + 1) / (size_t) chunks;
    const size_t n4 = (hi - lo) / 16;
    for (size_t i = threadIdx.x; i < n4; i += blockDim.x)
        ((uint4*) (dst + lo))[i] = ((const uint4*) (src + lo))[i];
    for (size_t i = n4 * 16 + threadIdx.x; i < hi - lo; i += blockDim.x)
        dst[lo + i] = src[lo + i];
}

/// y = x / sqrt(mean(x^2) + eps) * w, per row (grid = rows); w may be null.
__global__ void k_rms_rows(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xg = x + (size_t) blockIdx.x * n;
    float* yg = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    if (n == 4096 && blockDim.x == 128) {
        float v[32];
#pragma unroll
        for (int k = 0; k < 32; ++k) v[k] = xg[threadIdx.x + 128 * k];
#pragma unroll
        for (int k = 0; k < 32; ++k) s += v[k] * v[k];
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) s += xg[i] * xg[i];
    }
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yg[i] = xg[i] * r * (w ? w[i] : 1.0f);
}

__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

/// NEOX RoPE over the first n_rot dims of each head, q in place, k (and v) appended to the cache at pos0 + row.
__global__ void k_qk_prep_rows(float* qk, const float* v, __half* kc, __half* vc, int H, int HKV, int D, int n_rot,
                               const int64_t* pos0p, int ctx, float base) {
    __shared__ float buf[512];
    const int per = H + HKV;
    const int row = blockIdx.x / per, head = blockIdx.x % per;
    const bool isq = head < H;
    const float* src = qk + ((size_t) row * per + head) * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) buf[i] = src[i];
    __syncthreads();
    const int64_t pos = *pos0p + row;
    __half* dst = isq ? nullptr : kc + (((size_t) (head - H) * ctx + pos) * D);
    const int half = n_rot / 2;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float w = buf[i];
        if (i < n_rot) {
            const int p = i < half ? i : i - half;
            const float theta = (float) pos * powf(base, -2.0f * (float) p / (float) n_rot);
            const float cs = cosf(theta), sn = sinf(theta);
            w = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
        }
        if (isq) qk[((size_t) row * per + head) * D + i] = w;
        else dst[i] = __float2half(w);
    }
    if (!isq)
        for (int i = threadIdx.x; i < D; i += blockDim.x)
            vc[((size_t) (head - H) * ctx + pos) * D + i] =
                __float2half(v[((size_t) row * HKV + (head - H)) * D + i]);
}

/// Causal attention for row `blockIdx.y` (position pos0 + row) over the cache's positions 0 .. pos0 + row.
__global__ void k_attn_rows(const float* q, const __half* kc, const __half* vc, float* out, int H, int HKV, int D,
                            int64_t pos0, int ctx, float scale) {
    extern __shared__ float sc[];
    __shared__ float sh[kThreads];
    const int h = blockIdx.x, row = blockIdx.y, kvh = h / (H / HKV);
    const int n_pos = (int) (pos0 + row + 1);
    const float* qh = q + ((size_t) row * (H + HKV) + h) * D;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) {
        const __half* kp = kc + ((size_t) kvh * ctx + p) * D;
        float s = 0.f;
        for (int i = 0; i < D; ++i) s += qh[i] * __half2float(kp[i]);
        sc[p] = s * scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) mx = fmaxf(mx, sc[p]);
    sh[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    mx = sh[0];
    __syncthreads();
    float sum = 0.f;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) {
        sc[p] = expf(sc[p] - mx);
        sum += sc[p];
    }
    sum = block_sum(sum, sh);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float acc = 0.f;
        for (int p = 0; p < n_pos; ++p) acc += sc[p] * __half2float(vc[((size_t) kvh * ctx + p) * D + i]);
        out[((size_t) row * H + h) * D + i] = acc / sum;
    }
}

// ---------------------------------------------------------------- GEMM-style decode attention (n == 1, D == 128)
// Four kernels instead of one serial walk.  The 8 query heads that share a KV head are computed together, so K and V
// are read once per token:
//   A scores: block per (kv head, 128-position tile); K's tile staged in shared; thread t = position p0+t computes
//     the 8 heads' dots (no per-position warp reduction).
//   B softmax per query head over the positions.
//   C values: block per (kv head, position split); 128 threads cover D; V[p][d] is read once and applied to the 8
//     heads' probabilities.
//   D combine: sum the splits.
__global__ void k_attn_scores(const float* qk, const __half* kc, float* sc, int H, int HKV, int D, int ctx,
                              const int64_t* pos0p, float scale) {
    const int n_pos = (int) (*pos0p) + 1;
    __shared__ __half tile[128][130];
    __shared__ float qs[8][128];   // this KV head's query rows: the inner loop then touches shared memory only
    const int kvh = blockIdx.x, t = blockIdx.y;
    const int p0 = t * 128;
    if (p0 >= n_pos) return;
    const int d = threadIdx.x, R = H / HKV;
    const __half* kb = kc + (size_t) kvh * ctx * D;
    for (int i = 0; i < 128; ++i)
        tile[i][d] = (p0 + i < n_pos) ? kb[(size_t) (p0 + i) * D + d] : __float2half(0.f);
    for (int i = d; i < R * D; i += blockDim.x) qs[i / D][i % D] = qk[(size_t) kvh * R * D + i];
    __syncthreads();
    const int i = threadIdx.x;   // one thread per position of the tile
    if (p0 + i >= n_pos) return;
    // One accumulator per head and four elements per step: eight independent chains to hide the shared-memory
    // latency, against one dependent chain before.  Every fmaf is written out in the original order, so the dot is
    // bit-identical to the scalar loop.
    float acc[8];
    for (int h = 0; h < R; ++h) acc[h] = 0.f;
    const __half* kr = tile[i];
    for (int e = 0; e + 3 < D; e += 4) {
        const float k0 = __half2float(kr[e]), k1 = __half2float(kr[e + 1]);
        const float k2 = __half2float(kr[e + 2]), k3 = __half2float(kr[e + 3]);
        for (int h = 0; h < R; ++h) {
            acc[h] = fmaf(qs[h][e], k0, acc[h]);
            acc[h] = fmaf(qs[h][e + 1], k1, acc[h]);
            acc[h] = fmaf(qs[h][e + 2], k2, acc[h]);
            acc[h] = fmaf(qs[h][e + 3], k3, acc[h]);
        }
    }
    for (int e = D & ~3; e < D; ++e) {
        const float k0 = __half2float(kr[e]);
        for (int h = 0; h < R; ++h) acc[h] = fmaf(qs[h][e], k0, acc[h]);
    }
    for (int h = 0; h < R; ++h) sc[(size_t) (kvh * R + h) * ctx + p0 + i] = acc[h] * scale;
}

__global__ void k_attn_softmax(float* sc, int H, int ctx, const int64_t* pos0p) {
    const int n_pos = (int) (*pos0p) + 1;
    __shared__ float sh[kThreads];
    float* row = sc + (size_t) blockIdx.x * ctx;
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) mx = fmaxf(mx, row[p]);
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) {
        row[p] = expf(row[p] - mx);
        sum += row[p];
    }
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) row[p] *= inv;
}

__global__ void k_attn_values(const float* sc, const __half* vc, float* part, int H, int HKV, int D, int ctx,
                              const int64_t* pos0p, int nsplits) {
    const int n_pos = (int) (*pos0p) + 1;
    // The probabilities are head-major, so reading eight of them per position was eight scattered loads.  The
    // split's slice is small (positions/nsplits ~ 64), so stage it in shared once and read it as one block.
    __shared__ float ps[8][256];
    const int kvh = blockIdx.x, s = blockIdx.y, d = threadIdx.x, R = H / HKV;
    const int tiles_now = (n_pos + 127) / 128;
    const int cap = tiles_now * 2;
    const int nsplit = nsplits < cap ? (nsplits > 1 ? nsplits : 1) : (cap > 1 ? cap : 1);
    const int p0 = s < nsplit ? (int) ((int64_t) n_pos * s / nsplit) : n_pos;
    const int p1 = s < nsplit ? (int) ((int64_t) n_pos * (s + 1) / nsplit) : n_pos;
    const int np = p1 - p0;
    for (int i = d; i < R * np; i += blockDim.x) {
        const int h = i / np, q = i - h * np;
        ps[h][q] = sc[(size_t) (kvh * R + h) * ctx + p0 + q];
    }
    __syncthreads();
    float acc[8];
    for (int h = 0; h < R; ++h) acc[h] = 0.f;
    const __half* vb = vc + (size_t) kvh * ctx * D;
    // Eight positions in flight.  The loop used to be one dependent global load per position - V[p][d] and V[p+1][d]
    // are 256 bytes apart, so 62 positions in a row cost 62 memory latencies (measured 62 us per layer).  The
    // accumulation is still done in ascending position order, one fmaf at a time, so the result is unchanged.
    int p = 0;
    for (; p + 7 < np; p += 8) {
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = __half2float(vb[(size_t) (p0 + p + j) * D + d]);
        for (int h = 0; h < R; ++h) {
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[h] = fmaf(ps[h][p + j], v[j], acc[h]);
        }
    }
    for (; p < np; ++p) {
        const float v = __half2float(vb[(size_t) (p0 + p) * D + d]);
        for (int h = 0; h < R; ++h) acc[h] = fmaf(ps[h][p], v, acc[h]);
    }
    for (int h = 0; h < R; ++h) part[((size_t) (kvh * R + h) * nsplits + s) * D + d] = acc[h];
}

__global__ void k_attn_combine(const float* part, float* out, int H, int D, int nsplits, int qstride) {
    const int h = blockIdx.x, d = threadIdx.x;
    float acc = 0.f;
    for (int s = 0; s < nsplits; ++s) acc += part[((size_t) h * nsplits + s) * D + d];
    out[(size_t) h * qstride + d] = acc;
}

/// dst[e] = src[idx[e]] for `qb` bytes (one q8_1 column per entry), for the grouped-expert path.
__global__ void k_gather_cols(uint8_t* __restrict__ dst, const uint8_t* __restrict__ src,
                              const int* __restrict__ idx, int qb) {
    const int e = blockIdx.x;
    const int off = blockIdx.y * blockDim.x + threadIdx.x;
    if (off < qb) dst[(size_t) e * qb + off] = src[(size_t) idx[e] * qb + off];
}

/// dst[idx[e]][0..width) = src[e][0..width), the grouped matmul's rows back to their routing slots.
__global__ void k_scatter_rows(float* __restrict__ dst, const float* __restrict__ src, const int* __restrict__ idx,
                               int width, int stride) {
    const int e = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) dst[(size_t) idx[e] * stride + i] = src[(size_t) e * width + i];
}

__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

/// The same causal attention as k_attn_rows, with coalesced cache reads: a warp owns a position and its 32 lanes
/// cover the head's 128 dims (4 each), instead of one thread walking a whole 128-float row at a 512-byte stride.
/// D must be 128.  Four warps split the positions for the score pass and the value pass.
__global__ void k_attn_rows_fast(const float* q, const __half* kc, const __half* vc, float* out, int H, int HKV, int D,
                                 int64_t pos0, int ctx, float scale, int mode = 0) {
    extern __shared__ float sc[];        // the n_pos scores, then the probabilities
    __shared__ float sh[kThreads];
    __shared__ float red[4][128];
    const int h = blockIdx.x, row = blockIdx.y, kvh = h / (H / HKV);
    const int n_pos = (int) (pos0 + row + 1);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* qh = q + ((size_t) row * (H + HKV) + h) * D;
    float qreg[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) qreg[i] = qh[lane + i * 32];
    // Four positions per warp iteration.  The prompt pass is quadratic in the position count and this loop was one
    // position at a time, i.e. five dependent shuffles per position - the whole warp stalls on each reduction.  The
    // four dots and their four reductions are independent, so they overlap.  Each position's own dot keeps the
    // original i = 0..3 order, so every score is bit-identical.
    // The four positions of a group are addressed unconditionally: with the `p0 + j < n_pos` clamp inside the loop the
    // compiler cannot issue the sixteen K loads together, and the QK step - 92% of the prompt attention - costs about
    // 170 cycles per four MACs of work, i.e. it is waiting on one load at a time.  Groups that fit are handled
    // unconditionally; the last one or two per warp fall to the scalar tail (one position per iteration, no stride, so
    // every position is covered exactly once across the four warps).
    int pfull = warp * 4;
    for (; pfull + 3 < n_pos; pfull += 16) {
        const __half* kp0 = kc + ((size_t) kvh * ctx + pfull) * D;
        const __half* kp1 = kp0 + (size_t) D;
        const __half* kp2 = kp1 + (size_t) D;
        const __half* kp3 = kp2 + (size_t) D;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int e = lane + i * 32;
            const float q = qreg[i];
            s0 = fmaf(q, __half2float(kp0[e]), s0);
            s1 = fmaf(q, __half2float(kp1[e]), s1);
            s2 = fmaf(q, __half2float(kp2[e]), s2);
            s3 = fmaf(q, __half2float(kp3[e]), s3);
        }
        for (int o = 16; o > 0; o >>= 1) {
            s0 += __shfl_xor_sync(0xffffffffu, s0, o);
            s1 += __shfl_xor_sync(0xffffffffu, s1, o);
            s2 += __shfl_xor_sync(0xffffffffu, s2, o);
            s3 += __shfl_xor_sync(0xffffffffu, s3, o);
        }
        if (lane == 0) {
            sc[pfull] = s0 * scale;
            sc[pfull + 1] = s1 * scale;
            sc[pfull + 2] = s2 * scale;
            sc[pfull + 3] = s3 * scale;
        }
    }
    for (int p = pfull; p < n_pos; ++p) {
        const __half* kp = kc + ((size_t) kvh * ctx + p) * D;
        float sv = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) sv = fmaf(qreg[i], __half2float(kp[lane + i * 32]), sv);
        for (int o = 16; o > 0; o >>= 1) sv += __shfl_xor_sync(0xffffffffu, sv, o);
        if (lane == 0) sc[p] = sv * scale;
    }
    __syncthreads();
    if (mode == 1) {   // diagnostic (STRATA_GLM4_ATTN_SKIP): stop after the scores, to time the two halves
        if (threadIdx.x == 0) out[((size_t) row * H + h) * D] = sc[0];
        return;
    }
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) mx = fmaxf(mx, sc[p]);
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) {
        sc[p] = expf(sc[p] - mx);
        sum += sc[p];
    }
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) sc[p] *= inv;
    __syncthreads();
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int p = warp; p < n_pos; p += 4) {
        const float e = sc[p];
        const __half* vp = vc + ((size_t) kvh * ctx + p) * D;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] += e * __half2float(vp[lane + i * 32]);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) red[warp][lane + i * 32] = acc[i];
    __syncthreads();
    if (warp == 0) {
        float* ow = out + ((size_t) row * H + h) * D;
        for (int i = lane; i < D; i += 32) ow[i] = red[0][i] + red[1][i] + red[2][i] + red[3][i];
    }
}


/// Prompt attention in flash form: tiles of 64 positions, online softmax, no per-position score array.  Four of the
/// eight query heads that share a KV head are handled per block, so the K and V of a position are read twice per row
/// instead of eight times, and the running max / sum / value accumulators live in registers and shared (about 9 KB
/// total, against the 40 KB an all-heads version with a stored score row needs - that one measured 2.3x slower for
/// exactly that reason).
__global__ void k_attn_flash(const float* q, const __half* kc, const __half* vc, float* out, int H, int HKV, int D,
                             int64_t pos0, int ctx, float scale) {
    constexpr int RB = 4;                    // query heads per block (8 measured slower: double the shared)
    constexpr int TILE = 64;                 // positions per tile
    const int kvh = blockIdx.x >> 1, hg = blockIdx.x & 1;
    const int R = H / HKV, h0 = kvh * R + hg * RB;
    const int row = blockIdx.y;
    const int n_pos = (int) (pos0 + row + 1);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __shared__ float st[RB][TILE];            // this tile's scores, then exp(score - max)
    __shared__ float res[4][RB];              // per-head rescale factor (written by thread 0)
    __shared__ float mrun[RB], lrun[RB];
    __shared__ float red[4][RB][128];         // cross-warp value partials
    float qreg[RB][4];
    for (int h = 0; h < RB; ++h)
        for (int i = 0; i < 4; ++i)
            qreg[h][i] = q[((size_t) row * (H + HKV) + h0 + h) * D + lane + i * 32];
    if (threadIdx.x < RB) {
        mrun[threadIdx.x] = -INFINITY;
        lrun[threadIdx.x] = 0.f;
    }
    float acc[RB][4];
    for (int h = 0; h < RB; ++h)
        for (int i = 0; i < 4; ++i) acc[h][i] = 0.f;
    __syncthreads();
    for (int t0 = 0; t0 < n_pos; t0 += TILE) {
        const int tn = min(TILE, n_pos - t0);
        for (int j = warp; j < tn; j += 4) {
            const __half* kp = kc + ((size_t) kvh * ctx + t0 + j) * D;
            float kv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) kv[i] = __half2float(kp[lane + i * 32]);
            for (int h = 0; h < RB; ++h) {
                float sv = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) sv = fmaf(qreg[h][i], kv[i], sv);
                for (int o = 16; o > 0; o >>= 1) sv += __shfl_xor_sync(0xffffffffu, sv, o);
                if (lane == 0) st[h][j] = sv * scale;
            }
        }
        __syncthreads();
        if (threadIdx.x < RB) {
            float tm = -INFINITY;
            for (int j = 0; j < tn; ++j) tm = fmaxf(tm, st[threadIdx.x][j]);
            const float mnew = fmaxf(mrun[threadIdx.x], tm);
            const float rsc = expf(mrun[threadIdx.x] - mnew);
            float sum = 0.f;
            for (int j = 0; j < tn; ++j) {
                const float e = expf(st[threadIdx.x][j] - mnew);
                st[threadIdx.x][j] = e;
                sum += e;
            }
            lrun[threadIdx.x] = lrun[threadIdx.x] * rsc + sum;
            mrun[threadIdx.x] = mnew;
            res[0][threadIdx.x] = rsc;
        }
        __syncthreads();
        for (int h = 0; h < RB; ++h) {
            const float rsc = res[0][h];
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[h][i] *= rsc;
        }
        for (int j = warp; j < tn; j += 4) {
            const __half* vp = vc + ((size_t) kvh * ctx + t0 + j) * D;
            float pv[RB];
            for (int h = 0; h < RB; ++h) pv[h] = st[h][j];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float v = __half2float(vp[lane + i * 32]);
                for (int h = 0; h < RB; ++h) acc[h][i] = fmaf(pv[h], v, acc[h][i]);
            }
        }
        __syncthreads();
    }
    for (int h = 0; h < RB; ++h)
        for (int i = 0; i < 4; ++i) red[warp][h][lane + i * 32] = acc[h][i];
    __syncthreads();
    if (warp == 0)
        for (int h = 0; h < RB; ++h) {
            const float inv = 1.0f / lrun[h];
            float* ow = out + ((size_t) row * H + h0 + h) * D;
            for (int i = lane; i < D; i += 32)
                ow[i] = ((red[0][h][i] + red[1][h][i]) + (red[2][h][i] + red[3][h][i])) * inv;
        }
}

/// SwiGLU from the merged gate|up layout: entry e is [gate (FF) | up (FF)] back to back.
__global__ void k_swiglu_gu(const float* gu, float* h, int entries, int FF) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= entries * FF) return;
    const int e = i / FF, k = i - e * FF;
    const float* g = gu + (size_t) e * 2 * FF + k;
    h[i] = silu(g[0]) * g[FF];
}

__global__ void k_swiglu(const float* g, const float* u, float* h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
__global__ void k_add_rows(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
/// x[row] += sum_k w[row][k] * y[row][k] + ys[row]
__global__ void k_moe_sum_rows(float* x, const float* y, const float* w, int k, const float* ys, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (i >= n) return;
    float acc = 0.f;
    for (int s = 0; s < k; ++s) acc += w[(size_t) row * k + s] * y[((size_t) row * k + s) * n + i];
    x[(size_t) row * n + i] += acc + ys[(size_t) row * n + i];
}

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;
    int type = -1;
    int n_in = 0, n_out = 0;
    size_t row_bytes = 0;
};

struct HostMat {
    const uint8_t* p = nullptr;
    int type = -1;
    int n_in = 0, n_out = 0, ne = 0;
    size_t row_bytes = 0, expert_bytes = 0;
    // distance between consecutive experts in memory; 0 means expert_bytes (the mmap layout, one tensor at a time).
    // The pinned copy uses stride = a whole cache slot, so gate | up | down of one expert are contiguous there and a
    // cache miss is a single transfer.
    size_t stride = 0;
    size_t est() const { return stride ? stride : expert_bytes; }
};

struct Layer {
    const float *attn_norm = nullptr, *post_norm = nullptr;
    Mat wq, wk, wv, wo;
    Mat qk;                 // attn_q rows followed by attn_k rows: one matmul for both
    bool qk_merged = false;
    Mat sh_gate, sh_up, sh_down;
    Mat gate_exps, up_exps, down_exps;   // when every expert fits in VRAM (else src_g / src_u / src_d and the hosts)
    const float *router = nullptr, *prob_bias = nullptr;
    HostMat hq, hk, hv, ho, hg, hu, hd, hsg, hsu, hsd;   // set instead of the Mat when it lives in the mmap
    const uint8_t *src_g = nullptr, *src_u = nullptr, *src_d = nullptr;
    size_t gsz = 0, dsz = 0;
};

struct Model {
    strata::core::ModelGeometry g;
    std::shared_ptr<strata::GgufModel> file;
    int64_t top_k = 8;
    float eps = 1e-5f, rope_base = 1e6f, route_scale = 1.0f;
    int n_rot = 128;
    bool weights_norm = true, experts_host = false;
    std::vector<Layer> layers;
    const float* out_norm = nullptr;
    Mat head;
    HostMat hhead;
    const uint8_t* embd = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
    size_t max_expert_bytes = 0;
};

size_t row_bytes_of(int type, int64_t n) {
    if (type == T_F32) return (size_t) n * 4;
    if (type == T_F16 || type == T_BF16) return (size_t) n * 2;
    return strata::kernels::iq_row_bytes(type, n);
}
size_t q8_bytes(int64_t n) { return (size_t) (n / 32) * 36; }

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else {
            uint32_t mm = m, sh = 0;
            while ((mm & 0x400) == 0) { mm <<= 1; ++sh; }
            u = s | ((uint32_t) (127 - 14 - sh) << 23) | ((mm & 0x3ff) << 13);
        }
    } else if (e == 0x1f) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e + 112) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

float* upload_f32(const strata::TensorInfo& t, const uint8_t* src, uint64_t& bytes) {
    std::vector<float> h((size_t) t.elements());
    if (t.type == T_F32) std::memcpy(h.data(), src, h.size() * 4);
    else if (t.type == T_BF16)
        for (size_t i = 0; i < h.size(); ++i) h[i] = bf16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == T_F16)
        for (size_t i = 0; i < h.size(); ++i) h[i] = f16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else {
        std::fprintf(stderr, "strata-glm4: %s is %s, expected F32, BF16 or F16\n", t.name.c_str(), t.type_name());
        std::exit(1);
    }
    float* d = nullptr;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    bytes += h.size() * 4;
    return d;
}

bool load(const std::string& path, Model& m, std::string& err) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.file;
    if (!strata::core::geometry_from_gguf(gm.meta(), m.g, m.top_k, err)) return false;
    if (m.g.family != strata::ModelFamily::Glm4Moe) {
        err = "strata-glm4 runs Solar-Open-100B (glm4moe) files only";
        return false;
    }
    size_t checked = 0;
    if (!strata::core::check_gguf_layout(gm, m.g, err, &checked)) return false;
    const auto& g = m.g;
    if (g.head_dim > 512 || g.n_expert > 512 || m.top_k > 32) {
        err = "a geometry this engine's buffers do not take (head <= 512, <= 512 experts, top-k <= 32)";
        return false;
    }
    const std::string p = "glm4moe.";
    if (const strata::MetaValue* v = gm.meta().get(p + "attention.layer_norm_rms_epsilon")) m.eps = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.freq_base")) m.rope_base = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.dimension_count")) m.n_rot = (int) v->u;
    if (const strata::MetaValue* v = gm.meta().get(p + "expert_weights_scale")) m.route_scale = (float) v->num();
    m.weights_norm = g.expert_weights_norm;

    uint64_t total = 0;
    for (size_t s = 0; s < gm.size(); ++s)
        for (const strata::TensorInfo& ti : gm.shard(s).tensors()) total += strata::tensor_payload_bytes(ti);
    size_t free_vram = 0, total_vram = 0;
    CK(cudaMemGetInfo(&free_vram, &total_vram));
    m.experts_host = total + (uint64_t) 2 << 30 > (uint64_t) free_vram;

    uint64_t bytes = 0;
    auto find = [&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) {
            std::fprintf(stderr, "strata-glm4: missing %s\n", name.c_str());
            std::exit(1);
        }
        return t;
    };
    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        return upload_f32(*t, gm.shard(s).tensor_data(*t), bytes);
    };
    auto mat = [&](const std::string& name, HostMat* host, Mat* dev) {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        const uint8_t* src = gm.shard(s).tensor_data(*t);
        const bool is_expert = t->shape.size() == 3;
        if (!is_expert && !strata::kernels::iq_supported((int) t->type) &&
            strata::kernels::embed_type_supported((int) t->type) == false && t->type != T_F32)
            std::fprintf(stderr, "strata-glm4: warning: %s is %s (the i-quant MMVQ may not take it)\n", name.c_str(),
                         t->type_name());
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        w.row_bytes = row_bytes_of(w.type, w.n_in);
        if (m.experts_host && is_expert) {
            host->p = src;
            host->type = w.type;
            host->n_in = w.n_in;
            host->n_out = w.n_out;
            host->ne = (int) t->shape[2];
            host->row_bytes = row_bytes_of(w.type, w.n_in);
            host->expert_bytes = host->row_bytes * (size_t) w.n_out;
            m.max_expert_bytes = std::max(m.max_expert_bytes, host->expert_bytes);
            bytes += (uint64_t) host->expert_bytes * host->ne;
            return;
        }
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, src, nb, cudaMemcpyHostToDevice));
        w.d = d;
        *dev = w;
        bytes += nb;
    };
    auto L = [](int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    m.layers.resize((size_t) g.n_layers);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        Layer& y = m.layers[(size_t) l];
        y.attn_norm = f32(L(l, "attn_norm.weight"));
        y.post_norm = f32(L(l, "post_attention_norm.weight"));
        mat(L(l, "attn_q.weight"), &y.hq, &y.wq);
        mat(L(l, "attn_k.weight"), &y.hk, &y.wk);
        // q and k share a row width and a type in every Solar quant: pack them so one matmul fills both (see forward)
        if (y.wq.type == y.wk.type && y.wq.d && y.wk.d) {
            const size_t rb = y.wq.row_bytes;
            const size_t nq = rb * (size_t) g.n_head * g.head_dim, nk = rb * (size_t) g.n_head_kv * g.head_dim;
            uint8_t* pk = nullptr;
            CK(cudaMalloc(&pk, nq + nk));
            CK(cudaMemcpy(pk, y.wq.d, nq, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(pk + nq, y.wk.d, nk, cudaMemcpyDeviceToDevice));
            y.qk = y.wq;
            y.qk.n_out = (int) ((size_t) g.n_head + g.n_head_kv) * (int) g.head_dim;
            y.qk.d = pk;
            y.qk_merged = true;
            CK(cudaFree((void*) y.wq.d));   // give the separate copies back to the expert cache
            CK(cudaFree((void*) y.wk.d));
            y.wq.d = y.wk.d = nullptr;
        }
        mat(L(l, "attn_v.weight"), &y.hv, &y.wv);
        mat(L(l, "attn_output.weight"), &y.ho, &y.wo);
        y.router = f32(L(l, "ffn_gate_inp.weight"));
        y.prob_bias = f32(L(l, "exp_probs_b.bias"));
        mat(L(l, "ffn_gate_shexp.weight"), &y.hsg, &y.sh_gate);
        mat(L(l, "ffn_up_shexp.weight"), &y.hsu, &y.sh_up);
        mat(L(l, "ffn_down_shexp.weight"), &y.hsd, &y.sh_down);
        mat(L(l, "ffn_gate_exps.weight"), &y.hg, &y.gate_exps);
        mat(L(l, "ffn_up_exps.weight"), &y.hu, &y.up_exps);
        mat(L(l, "ffn_down_exps.weight"), &y.hd, &y.down_exps);
    }
    m.out_norm = f32("output_norm.weight");
    mat("output.weight", &m.hhead, &m.head);
    {
        size_t s = 0;
        const strata::TensorInfo* t = find("token_embd.weight", s);
        m.embd_type = (int) t->type;
        m.embd_row = row_bytes_of(m.embd_type, (int64_t) t->shape[0]);
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        m.embd = d;
        bytes += nb;
    }
    std::fprintf(stderr,
                 "strata-glm4: %s: %zu tensors (layout checked), %.2f GiB %s in %.1f s%s\n", path.c_str(), checked,
                 (double) bytes / (1 << 30), m.experts_host ? "(routed experts in the mmap)" : "on the GPU",
                 ms_since(t0) / 1000.0,
                 m.experts_host ? "; one expert blob crosses PCIe per expert per matrix" : "");
    return true;
}

// ------------------------------------------------------------------------------------------------ the engine
float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}
template <class T> T* dalloc_t(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

struct Engine {
    const Model* m = nullptr;
    cudaStream_t st = nullptr;
    int ctx = 8192, tmax = 512;
    int64_t pos = 0;
    // activations [tmax][...]
    float *x, *h, *y, *qkf, *vx, *att, *router, *gu, *eh, *ey, *sg, *su, *shh, *sy, *wts, *logits;
    float *att_sc = nullptr, *att_part = nullptr;   // GEMM attention: scores [H][ctx], split accumulators [H][S][D]
    int att_splits = 32;   // STRATA_GLM4_ATTN_SPLITS: position splits for the value accumulation
    bool att_gemm = false;   // force the GEMM path
    int att_thresh = 640;    // ... or use it once the context passes this many positions (STRATA_GLM4_ATTN=fast disables)

    void *q8 = nullptr, *q8d = nullptr;
    uint8_t *exp_g = nullptr, *exp_u = nullptr, *exp_d = nullptr;   // one expert's gate / up / down blob, staged
    // The expert cache: (layer, expert) blobs resident in VRAM, LRU-evicted.  A hit reads nothing over PCIe, which is
    // the whole cost of a decode step when the experts live in the GGUF's mmap.
    uint8_t* xcache = nullptr;
    size_t xc_slots = 0, xc_g = 0, xc_u = 0, xc_d = 0, xc_stride = 0;
    std::vector<int32_t> xc_layer, xc_expert;
    std::vector<uint64_t> xc_used;
    std::unordered_map<uint64_t, int> xc_map;
    uint64_t xc_tick = 0;
    long long xc_hits = 0, xc_misses = 0;
    int64_t xc_mib = -1;   // --expert-cache-mib (-1: from the free VRAM)
    bool xc_part = false;   // partitioned: a fixed slice per layer (measured worse than the global LRU)
    bool xc_lfu = false;    // STRATA_GLM4_CACHE_LFU: evict the least FREQUENTLY used (with aging) instead of the oldest
    std::vector<uint32_t> xc_freq;
    // The cache directory also lives on the device, so a decode step never round-trips through the host: the slot
    // tables, the (layer, expert) -> slot map, the LRU clock, the miss list and the counters are all device state.
    // The host vectors above are the prefill path's view and are refreshed around each n > 1 chunk.
    // CUDA graph of one decode step: every launch in it costs ~9.7 us of driver overhead on this machine against
    // 0.5 us inside a graph, and a token is ~1000 launches.  The graph bakes in kernel arguments, so the token id and
    // the position live in fixed host buffers and the position reaches the kernels through a device counter.
    // Decode's down projection is only needed after the SwiGLU, so its third of the fill runs on its own stream and
    // hides behind the gate/up matvec; ~4.5 ms/token at 1500 positions.
    cudaStream_t st2 = nullptr;
    cudaEvent_t ev_fork = nullptr, ev_join = nullptr;
    // prompt pass: the expert fills of a layer are issued on st2, one event each, and the matvecs wait per group, so
    // the 1.4 s of copying overlaps the 1.3 s of matvec instead of preceding it
    bool prefill_pipe = true;
    std::vector<cudaEvent_t> ev_grp;
    bool fetch_split = true;
    bool graph_enabled = false;
    bool capturing = false;
    cudaGraphExec_t gexec = nullptr;
    int graph_sess = -1;
    int64_t* d_pos = nullptr;
    int64_t pos_stage = 0;
    int32_t tok_stage = 0;
    int attn_skip = 0;     // STRATA_GLM4_ATTN_SKIP=1: stop the prefill attention after the scores (timing)
    bool att_flash = true;    // STRATA_GLM4_NO_ATTN_FLASH=1 falls back to the per-head prompt kernel
    int att_tiles = 1;     // fixed at the maximum over the context, so the grids never change
    int att_nsplits = 1;
    bool dev_cache = false;
    int32_t *d_xc_layer = nullptr, *d_xc_expert = nullptr, *d_xc_of = nullptr;
    unsigned long long *d_xc_used = nullptr, *d_xc_tick = nullptr;
    int32_t *d_miss_e = nullptr, *d_miss_slot = nullptr, *d_miss_n = nullptr;
    int32_t *d_xc_hits = nullptr, *d_xc_misses = nullptr;
    uint32_t* d_xc_freq = nullptr;   // device side of the frequency count (null: plain LRU)
    long long dev_hits = 0, dev_misses = 0;
    // STRATA_GLM4_ROUTECHECK: verify the device selection against the host's for one chunk, then stop
    int routecheck_left = 0;
    long long routecheck_bad = 0, routecheck_seen = 0;
    uint64_t xc_calls = 0;
    size_t xc_per = 0;     // slots per layer in the partitioned mode
    // stage profiler (STRATA_GLM4_PROF=1): one event between stages, the deltas summed per bucket over the tokens
    bool prof = false;

    std::vector<cudaEvent_t> pev;
    int pei = 0;
    double prof_sum[7] = {}, prof_total = 0, prof_copy_ms = 0, prof_router_ms = 0;
    double prof_rsync = 0, prof_rtopk = 0, prof_rup = 0, prof_miss_ms = 0;
    std::vector<cudaEvent_t> pev_f;   // STRATA_GLM4_PROF: two pairs per layer, around k_resolve and k_fetch
    std::vector<cudaEvent_t> pev_r;
    std::vector<cudaEvent_t> pev_a;   // 4 pairs per layer: scores / softmax / values / combine
    std::vector<cudaEvent_t> pev_g;   // 3 pairs per layer: rms / router gemv / route top-k
    double prof_fetch_ms = 0, prof_fetch_bytes = 0, prof_resolve_ms = 0;
    double prof_attn_k[4] = {};
    double prof_route_k[3] = {};
    long long prof_tokens = 0, prof_misses0 = 0;
    bool prof_started = false;
    ~Engine() {
        refresh_dev_counters();
        if (xc_slots) {
            const long long h = xc_hits + dev_hits, m2 = xc_misses + dev_misses;
            std::fprintf(stderr, "strata-glm4: expert cache: %lld hits, %lld misses (%.1f%% hit rate) over %zu slots\n",
                         h, m2, h + m2 ? 100.0 * (double) h / (double) (h + m2) : 0.0, xc_slots);
        }
        if (prof_tokens) {
            static const char* const names[7] = {"embed", "attn", "router", "gate/up", "down", "shared", "head"};
            const double n = (double) prof_tokens;
            std::fprintf(stderr, "strata-glm4: profile over %lld decode tokens: %.2f ms/token, %.1f misses/token\n",
                         prof_tokens, prof_total / n, (double) (dev_misses + xc_misses - prof_misses0) / n);
            std::fprintf(stderr,
                         "strata-glm4: profile   host blocked by expert copies %6.2f ms/token, router round-trip "
                         "%6.2f ms/token (copy+sync %.2f, top-k %.2f, upload %.2f), miss copies %.2f ms/token\n",
                         prof_copy_ms / n, prof_router_ms / n, prof_rsync / n, prof_rtopk / n, prof_rup / n,
                         prof_miss_ms / n);
            {
                const double fms = prof_fetch_ms / n;
                const double per_tok_miss = (double) (dev_misses + xc_misses - prof_misses0) / n;
                const double gb = per_tok_miss * (double) xc_stride;
                std::fprintf(stderr,
                             "strata-glm4: profile   k_fetch %.2f ms/token, %.0f MB/token (%.1f misses x %.2f MiB) "
                             "= %.1f GB/s\n",
                             fms, gb / 1e6, per_tok_miss, (double) xc_stride / 1048576.0, fms > 0 ? gb / (fms * 1e6) : 0.0);
                std::fprintf(stderr, "strata-glm4: profile   k_resolve %.2f ms/token (%.1f us/layer)\n",
                             prof_resolve_ms / n, prof_resolve_ms / n / (double) m->layers.size() * 1000.0);
                static const char* const rn[3] = {"rms", "rt-gemv", "rt-topk"};
                for (int j = 0; j < 3; ++j)
                    std::fprintf(stderr, "strata-glm4: profile     router %-8s %5.2f ms/token (%5.1f us/layer)\n", rn[j],
                                 prof_route_k[j] / n, prof_route_k[j] / n / (double) m->layers.size() * 1000.0);
                static const char* const an[4] = {"scores", "softmax", "values", "combine"};
                for (int j = 0; j < 4; ++j)
                    std::fprintf(stderr, "strata-glm4: profile     attn %-8s %6.2f ms/token (%6.1f us/layer)\n", an[j],
                                 prof_attn_k[j] / n,
                                 prof_attn_k[j] / n / (double) m->layers.size() * 1000.0);
            }
            for (int i = 0; i < 7; ++i)
                std::fprintf(stderr, "strata-glm4: profile   %-8s %6.2f ms/token  %4.1f%%\n", names[i],
                             prof_sum[i] / n, prof_total > 0 ? 100.0 * prof_sum[i] / prof_total : 0.0);
        }
    }
    int32_t *ids = nullptr, *tok = nullptr, *wts_i = nullptr;
    std::vector<int32_t> ids_host;   // the routed experts of the last forward (the host mirrors wts_i)
    std::vector<int32_t> rout_id;   // the id list of the last layer, downloaded for the expert-cache lookup
    // grouped experts: the rows of one expert gathered into contiguous q8_1 columns, one iq_mmvq for the group,
    // and the results scattered back to their (row, k) slots
    uint8_t* q8g = nullptr;
    float *tmp_e = nullptr, *tmp_d = nullptr;
    int32_t *d_cols = nullptr, *d_rows = nullptr;
    // the indirect expert path: one launch for every expert of a layer, weight pointers from the device table
    void** d_wp_g = nullptr;
    void** d_wp_d = nullptr;
    int32_t *d_xc_row = nullptr, *d_xc_col = nullptr;
    std::vector<const void*> hwp_g, hwp_d;
    std::vector<__half*> kc, vc;
    std::vector<HostMat> hq, hk, hv, ho, hg, hu, hd, hsg, hsu, hsd;
    struct Sess {
        std::vector<__half*> kc, vc;
        int64_t pos = 0;
        std::vector<int32_t> live;
        bool active = false;
        uint64_t used = 0;
    };
    std::vector<Sess> sess;
    int n_sess = 1, cur = 0;
    uint64_t use_clock = 0;
    std::vector<std::vector<float>> row_logits;   // host rows of the last step_rows() call
    std::vector<int> rows_pick;
    int n_rows_last = 0;
    bool use_graph = false;

    void alloc_session(Sess& ss) const {
        const auto& g = m->g;
        for (size_t l = 0; l < m->layers.size(); ++l) {
            ss.kc.push_back(dalloc_t<__half>((size_t) g.n_head_kv * ctx * g.head_dim));
            ss.vc.push_back(dalloc_t<__half>((size_t) g.n_head_kv * ctx * g.head_dim));
        }
    }
    void select(int i) {
        if (i == cur) return;
        sess[(size_t) cur].pos = pos;
        cur = i;
        load_cur();
    }
    void load_cur() {
        Sess& ss = sess[(size_t) cur];
        kc = ss.kc;
        vc = ss.vc;
        pos = ss.pos;
        ss.used = ++use_clock;
    }
    void copy_session(int src, int dst) {
        const auto& g = m->g;
        const size_t kb = (size_t) g.n_head_kv * ctx * g.head_dim * 2;   // __half
        for (size_t l = 0; l < m->layers.size(); ++l) {
            CK(cudaMemcpyAsync(sess[(size_t) dst].kc[l], sess[(size_t) src].kc[l], kb, cudaMemcpyDeviceToDevice, st));
            CK(cudaMemcpyAsync(sess[(size_t) dst].vc[l], sess[(size_t) src].vc[l], kb, cudaMemcpyDeviceToDevice, st));
        }
        sess[(size_t) dst].pos = sess[(size_t) src].pos;
        sess[(size_t) dst].live = sess[(size_t) src].live;
        CK(cudaStreamSynchronize(st));
    }

    void init(const Model& model, int ctx_, int tmax_, int n_sessions) {
        m = &model;
        ctx = ctx_;
        n_sess = std::max(1, n_sessions);
        tmax = std::max(1, tmax_);
        const auto& g = m->g;
        const size_t T = (size_t) tmax, N = (size_t) g.n_embd, Kp = (size_t) m->top_k, FF = (size_t) g.n_ff;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&st2, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming));
        // measured: giving the fill its own stream for the down third is a wash - the link is the same and each half
        // loses parallelism - so it is opt-in only (STRATA_GLM4_SPLITFETCH=1)
        fetch_split = std::getenv("STRATA_GLM4_SPLITFETCH") != nullptr;
        prefill_pipe = std::getenv("STRATA_GLM4_NO_PIPE_PREFETCH") == nullptr;
        ev_grp.resize(160);
        for (cudaEvent_t& e : ev_grp) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        CK(cudaMalloc(&d_pos, 8));
        CK(cudaMemcpy(d_pos, &pos_stage, 8, cudaMemcpyHostToDevice));
        att_tiles = (ctx + 127) / 128;
        graph_enabled = std::getenv("STRATA_GLM4_NOGRAPH") == nullptr;
        if (const char* v = std::getenv("STRATA_GLM4_ATTN_SKIP")) attn_skip = std::atoi(v);
        att_flash = std::getenv("STRATA_GLM4_NO_ATTN_FLASH") == nullptr;
        prof = std::getenv("STRATA_GLM4_PROF") != nullptr;
        if (const char* sp = std::getenv("STRATA_GLM4_ATTN_SPLITS")) att_splits = std::max(1, std::atoi(sp));
        if (const char* am = std::getenv("STRATA_GLM4_ATTN")) {
            const std::string v(am);
            if (v == "gemm") att_gemm = true;
            else if (v == "fast") att_thresh = 1 << 30;
        }

        if (prof) {
            pev.resize((size_t) (2 + 5 * m->layers.size() + 2));
            for (cudaEvent_t& e : pev) CK(cudaEventCreate(&e));   // timing events: no DisableTiming
            pev_f.resize(2 * m->layers.size());
            for (cudaEvent_t& e : pev_f) CK(cudaEventCreate(&e));
            for (cudaEvent_t& e : pev_f) CK(cudaEventRecord(e, st));
            pev_r.resize(2 * m->layers.size());
            for (cudaEvent_t& e : pev_r) CK(cudaEventCreate(&e));
            for (cudaEvent_t& e : pev_r) CK(cudaEventRecord(e, st));
            pev_a.resize(8 * m->layers.size());
            for (cudaEvent_t& e : pev_a) CK(cudaEventCreate(&e));
            for (cudaEvent_t& e : pev_a) CK(cudaEventRecord(e, st));   // the serial attention path leaves them alone
            pev_g.resize(6 * m->layers.size());
            for (cudaEvent_t& e : pev_g) CK(cudaEventCreate(&e));
            for (cudaEvent_t& e : pev_g) CK(cudaEventRecord(e, st));
            CK(cudaStreamSynchronize(st));
        }
        x = dalloc(T * N); h = dalloc(T * N); y = dalloc(T * N);
        qkf = dalloc(T * ((size_t) g.n_head + g.n_head_kv) * g.head_dim);   // q | k packed, one matmul
        vx = dalloc(T * g.n_head_kv * g.head_dim);
        att = dalloc(T * g.n_head * g.head_dim);
        router = dalloc(T * g.n_expert);
        gu = dalloc(T * Kp * 2 * FF); eh = dalloc(T * Kp * FF);   // [entry][gate (FF) | up (FF)]
        ey = dalloc(T * Kp * N);
        sg = dalloc(T * g.n_ff_shexp); su = dalloc(T * g.n_ff_shexp); shh = dalloc(T * g.n_ff_shexp);
        sy = dalloc(T * N);
        wts = dalloc(T * Kp);
        logits = dalloc((size_t) 8 * g.n_vocab);
        CK(cudaMalloc(&q8, q8_bytes((int64_t) (T * N + N)) + 4096));
        CK(cudaMalloc(&q8d, q8_bytes((int64_t) T * Kp * FF) + 4096));
        if (m->experts_host) {
            size_t gb = 0, ub = 0, db = 0;
            for (const Layer& L : m->layers) {
                gb = std::max(gb, L.hg.expert_bytes);
                ub = std::max(ub, L.hu.expert_bytes);
                db = std::max(db, L.hd.expert_bytes);
            }
            CK(cudaMalloc(&exp_g, gb));
            CK(cudaMalloc(&exp_u, ub));
            CK(cudaMalloc(&exp_d, db));
        }
        ids = dalloc_t<int32_t>(T * Kp);
        wts_i = dalloc_t<int32_t>(T * Kp);
        tok = dalloc_t<int32_t>(T);
        CK(cudaMalloc(&q8g, (size_t) (std::max<int64_t>(q8_bytes(N), q8_bytes(FF)) * (int64_t) (T * Kp)) + 256));
        tmp_e = dalloc(T * Kp * 2 * FF);
        tmp_d = dalloc(T * Kp * N);
        att_sc = dalloc((size_t) g.n_head * (size_t) ctx);
        // the value accumulation is split this many ways for the whole run, whatever the position: an empty split
        // contributes zeros, which is what the combine sums anyway
        att_nsplits = std::max(1, std::min(att_splits, att_tiles * 2));
        if (att_nsplits > att_splits) att_nsplits = att_splits;
        att_part = dalloc((size_t) g.n_head * (size_t) att_splits * (size_t) g.head_dim);

        d_cols = dalloc_t<int32_t>(T * Kp);
        d_rows = dalloc_t<int32_t>(T * Kp);
        CK(cudaMalloc(&d_wp_g, sizeof(void*) * (size_t) (T * Kp)));
        CK(cudaMalloc(&d_wp_d, sizeof(void*) * (size_t) (T * Kp)));
        d_xc_row = dalloc_t<int32_t>(T * Kp);   // zeros: every entry of a decode step reads row 0's activation
        d_xc_col = dalloc_t<int32_t>(T * Kp);
        {
            std::vector<int32_t> id((size_t) (T * Kp));
            for (size_t i = 0; i < id.size(); ++i) id[i] = (int32_t) i;
            CK(cudaMemcpy(d_xc_col, id.data(), id.size() * 4, cudaMemcpyHostToDevice));
        }
        hwp_g.resize((size_t) (T * Kp));
        hwp_d.resize((size_t) (T * Kp));
        if (std::getenv("STRATA_GLM4_SIZES")) {
            std::fprintf(stderr, "glm4 sizes: T=%zu N=%zu Kp=%zu FF=%zu FS=%zu\n", T, N, Kp, FF, (size_t) g.n_ff_shexp);
            std::fprintf(stderr, "glm4 sizes: q8=%zu q8d=%zu gu=%zu ey=%zu sg=%zu router=%zu\n",
                         q8_bytes((int64_t) (T * N + N)) + 4096, q8_bytes((int64_t) T * Kp * FF) + 4096, T * Kp * 2 * FF * 4,
                         T * Kp * N * 4, T * (size_t) g.n_ff_shexp * 4, T * (size_t) g.n_expert * 4);
        }
        for (const Layer& L : m->layers) {
            hq.push_back(L.hq); hk.push_back(L.hk); hv.push_back(L.hv); ho.push_back(L.ho);
            hg.push_back(L.hg); hu.push_back(L.hu); hd.push_back(L.hd);
            hsg.push_back(L.hsg); hsu.push_back(L.hsu); hsd.push_back(L.hsd);
        }
        if (m->experts_host && !std::getenv("STRATA_GLM4_NOPIN")) pin_experts();
        int max_smem = ctx * (int) sizeof(float);
        if (max_smem > 48 * 1024)
            CK(cudaFuncSetAttribute((const void*) k_attn_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));
        sess.resize((size_t) n_sess);
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const size_t per = (size_t) g.n_head_kv * g.head_dim * 2 * 2 * (size_t) ctx * m->layers.size();   // fp16 K/V
        if (n_sess > 1 && per * (size_t) n_sess + ((size_t) 1 << 30) > fr) {
            std::fprintf(stderr, "strata-glm4: not enough VRAM for %d slots at a context of %d (%.2f GiB each)\n", n_sess,
                         ctx, (double) per / (1 << 30));
            std::exit(1);
        }
        for (Sess& ss : sess) alloc_session(ss);
        // The expert cache, after every other allocation so its budget is the free VRAM left.  A slot holds one
        // (layer, expert)'s gate | up | down blobs; a decode step re-reads an expert from the mmap only on a miss.
        if (m->experts_host) {
            size_t gb = 0, ub = 0, db = 0;
            for (const Layer& L : m->layers) {
                gb = std::max(gb, L.hg.expert_bytes);
                ub = std::max(ub, L.hu.expert_bytes);
                db = std::max(db, L.hd.expert_bytes);
            }
            xc_g = (gb + 255) / 256 * 256;
            xc_u = (ub + 255) / 256 * 256;
            xc_d = (db + 255) / 256 * 256;
            xc_stride = xc_g + xc_u + xc_d;
            size_t fr2 = 0, tot2 = 0;
            CK(cudaMemGetInfo(&fr2, &tot2));
            size_t budget = 0;
            if (const char* ev = std::getenv("STRATA_GLM4_EXPERT_CACHE_MIB")) xc_mib = std::atoi(ev);
            const bool explicit_mib = xc_mib >= 0;
            if (explicit_mib) budget = (size_t) xc_mib << 20;
            else if (fr2 > ((size_t) 2 << 30)) budget = fr2 - ((size_t) 1 << 30);   // leave 1 GiB for the driver
            if (!explicit_mib && budget > ((size_t) 12 << 30)) budget = (size_t) 12 << 30;
            // never take the last 640 MiB of the card, whether the budget was asked for or derived
            const size_t kReserve = (size_t) 640 << 20;
            if (budget + kReserve > fr2) budget = fr2 > kReserve ? fr2 - kReserve : 0;
            xc_part = false;   // partitioned measured worse (equal slices vs demand): keep the global LRU
            xc_lfu = std::getenv("STRATA_GLM4_CACHE_LFU") != nullptr;
            while (budget >= xc_stride * 8) {
                const size_t slots = budget / xc_stride;
                if (cudaMalloc(&xcache, slots * xc_stride) == cudaSuccess) {
                    xc_slots = slots;
                    if (xc_part && !m->layers.empty()) {
                        xc_per = xc_slots / m->layers.size();
                        if (xc_per < 8) { xc_per = 0; xc_part = false; }   // too small to split: one global LRU
                    }
                    break;
                }
                (void) cudaGetLastError();
                budget = budget * 3 / 4;   // the driver refused it: back off and try again
            }
            if (xc_slots) {
                xc_layer.assign(xc_slots, -1);
                xc_expert.assign(xc_slots, -1);
                xc_used.assign(xc_slots, 0);
                xc_freq.assign(xc_slots, 0);
                // The device directory needs the expert bytes to be reachable from a kernel (i.e. the pinned region)
                // and every segment of a slot to be 16-byte clean for k_fetch's uint4 copy.
                // STRATA_GLM4_CACHE_POLICY=lfu (or the older STRATA_GLM4_CACHE_LFU) makes the victim the least
                // frequently used slot with a recency tie-break, with the counts halved every 8192 accesses.
                xc_lfu = (std::getenv("STRATA_GLM4_CACHE_LFU") != nullptr) ||
                         (std::getenv("STRATA_GLM4_CACHE_POLICY") &&
                          std::string(std::getenv("STRATA_GLM4_CACHE_POLICY")) == "lfu");
                bool ok = pinned;
                for (const Layer& L : m->layers)
                    ok = ok && L.hg.type == L.hu.type && L.hg.expert_bytes % 16 == 0 && L.hu.expert_bytes % 16 == 0 &&
                         L.hd.expert_bytes % 16 == 0;
                ok = ok && xc_stride % 16 == 0 && xc_g % (16 * kFetchChunks) == 0 && xc_u % (16 * kFetchChunks) == 0 &&
                     xc_d % (16 * kFetchChunks) == 0;
                dev_cache = ok && std::getenv("STRATA_GLM4_HOSTCACHE") == nullptr;
                if (dev_cache) {
                    const size_t npair = (size_t) m->layers.size() * (size_t) g.n_expert;
                    CK(cudaMalloc(&d_xc_layer, xc_slots * 4));
                    CK(cudaMalloc(&d_xc_expert, xc_slots * 4));
                    CK(cudaMalloc(&d_xc_used, xc_slots * 8));
                    CK(cudaMalloc(&d_xc_of, npair * 4));
                    CK(cudaMalloc(&d_miss_e, sizeof(int32_t) * 64));
                    CK(cudaMalloc(&d_miss_slot, sizeof(int32_t) * 64));
                    CK(cudaMalloc(&d_miss_n, 4));
                    CK(cudaMalloc(&d_xc_hits, 4));
                    CK(cudaMalloc(&d_xc_misses, 4));
                    CK(cudaMalloc(&d_xc_tick, 8));
                    if (xc_lfu) CK(cudaMalloc(&d_xc_freq, xc_slots * 4));
                    xc_tick = 0;
                    device_cache_from_host();   // all slots free, the map empty, the clock at 0
                    CK(cudaMemset(d_xc_hits, 0, 4));
                    CK(cudaMemset(d_xc_misses, 0, 4));
                    CK(cudaMemset(d_miss_n, 0, 4));
                }
                std::fprintf(stderr,
                             "strata-glm4: expert cache: %zu slots of %.2f MiB = %.2f GiB of VRAM (of %zu layer/expert "
                             "pairs)\n", xc_slots, (double) xc_stride / (1 << 20),
                             (double) (xc_slots * xc_stride) / (1 << 30), (size_t) m->layers.size() * (size_t) g.n_expert);
            }
        }
        cur = 0;
        load_cur();
        CK(cudaDeviceSynchronize());
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-glm4: context %d cells, prompt chunk %d, %.2f GiB of VRAM free\n", ctx, tmax,
                     (double) fr / (1 << 30));
    }

    void reset() {
        const size_t kb = (size_t) m->g.n_head_kv * ctx * m->g.head_dim * 2;   // __half
        for (size_t l = 0; l < kc.size(); ++l) {
            CK(cudaMemsetAsync(kc[l], 0, kb, st));
            CK(cudaMemsetAsync(vc[l], 0, kb, st));
        }
        CK(cudaStreamSynchronize(st));
        pos = 0;
        sess[(size_t) cur].pos = 0;
    }

    /// q8_1 of `cols` columns of `n` float values.
    size_t quant(const float* xin, int n, int cols, void* dst) {
        strata::kernels::quantize_q8_1_rows(xin, cols, n, dst, st);
        return (size_t) (n / 32) * 36;
    }

    /// out[col] = W . x[col] on iq_mmvq, `cols` columns of n_in values (packed).
    void mm(const Mat& w, const float* xin, float* out, int cols) {
        if (!strata::kernels::iq_supported(w.type)) {
            if (w.type == T_F32) {
                k_gemv_f32_rows<<<dim3(w.n_out, cols), kThreads, 0, st>>>((const float*) w.d, xin, out, w.n_in, w.n_out, w.n_in);
                return;
            }
            std::fprintf(stderr, "strata-glm4: matmul type %d is not supported\n", w.type);
            std::exit(1);
        }
        quant(xin, w.n_in, cols, q8);
        // ONE call for all the columns.  This used to loop iq_mmvq per column, which for a prompt of a few hundred
        // rows meant one launch per column per matrix - ~100k launches per chunk at ~9.7 us each, 1.4 s of the
        // attention bucket and 1.1 s of the shared expert.  The multi-column kernel is bitwise equal per column.
        strata::kernels::iq_mmvq(w.type, w.d, q8, out, w.n_in, w.n_out, cols, st);
    }

    /// One expert's blob of `w` copied into `slot` on the stream (gate / up / down staging).
    void stage_expert(const HostMat& w, uint8_t* slot, int e) {
        const auto t0 = Clock::now();
        CK(cudaMemcpyAsync(slot, w.p + (size_t) e * w.est(), w.expert_bytes, cudaMemcpyHostToDevice, st));
        if (prof) prof_copy_ms += ms_since(t0);   // pageable: the call itself stages the bytes on the host
    }

    /// The cache slot holding (layer, expert), filling it on a miss (LRU victim); -1 when the cache is off.  `s`
    /// lets the prompt pass issue the fill on a second stream so it overlaps the matvecs (prefill_pipe below).
    int cache_slot(int64_t layer, int e, cudaStream_t s = nullptr) {
        if (s == nullptr) s = st;
        if (!xc_slots) return -1;
        const uint64_t key = ((uint64_t) layer << 32) | (uint32_t) e;
        // aging: every 4096 lookups halve the frequencies, so a burst of old use does not pin a slot for ever
        if (xc_lfu && (++xc_calls & 4095) == 0)
            for (uint32_t& f : xc_freq) f >>= 1;
        auto it = xc_map.find(key);
        if (it != xc_map.end()) {
            ++xc_hits;
            const size_t s = (size_t) it->second;
            xc_used[s] = ++xc_tick;
            if (xc_lfu && xc_freq[s] < (1u << 30)) ++xc_freq[s];
            return it->second;
        }
        ++xc_misses;
        int slot = -1;
        uint64_t oldest = ~0ull;
        uint32_t least = ~0u;
        // partitioned: a layer evicts only its own slots, so another layer's burst cannot push this layer's hot experts
        // out; global (the default) lets any slot go.
        const size_t lo = xc_part ? (size_t) layer * xc_per : 0;
        const size_t hi = xc_part ? lo + xc_per : xc_slots;
        for (size_t i = lo; i < hi; ++i) {
            if (xc_layer[i] < 0) { slot = (int) i; break; }
            if (xc_lfu ? (xc_freq[i] < least || (xc_freq[i] == least && xc_used[i] < oldest))
                       : (xc_used[i] < oldest)) {
                least = xc_freq[i];
                oldest = xc_used[i];
                slot = (int) i;
            }
        }
        if (slot < 0) return -1;
        if (xc_layer[(size_t) slot] >= 0)
            xc_map.erase(((uint64_t) xc_layer[(size_t) slot] << 32) | (uint32_t) xc_expert[(size_t) slot]);
        xc_layer[(size_t) slot] = (int32_t) layer;
        xc_expert[(size_t) slot] = e;
        xc_used[(size_t) slot] = ++xc_tick;
        xc_freq[(size_t) slot] = 1;
        xc_map[key] = slot;
        uint8_t* dst = xcache + (size_t) slot * xc_stride;
        const auto tm0 = Clock::now();
        const HostMat& G = hg[(size_t) layer];
        const HostMat& U = hu[(size_t) layer];
        const HostMat& D = hd[(size_t) layer];
        if (G.stride) {
            // packed: one contiguous run covers gate | up | down
            CK(cudaMemcpyAsync(dst, G.p + (size_t) e * G.stride, xc_stride, cudaMemcpyHostToDevice, s));
        } else {
            CK(cudaMemcpyAsync(dst, G.p + (size_t) e * G.expert_bytes, G.expert_bytes, cudaMemcpyHostToDevice, s));
            CK(cudaMemcpyAsync(dst + xc_g, U.p + (size_t) e * U.expert_bytes, U.expert_bytes, cudaMemcpyHostToDevice, s));
            CK(cudaMemcpyAsync(dst + xc_g + xc_u, D.p + (size_t) e * D.expert_bytes, D.expert_bytes,
                               cudaMemcpyHostToDevice, s));
        }
        if (prof) prof_miss_ms += ms_since(tm0);   // pageable: the call stages the bytes on the host
        return slot;
    }

    /// Pull the device cache's hit/miss counters back (they are the authority for a decode-only run).
    void refresh_dev_counters() {
        if (!dev_cache) return;
        int32_t h = 0, ms = 0;
        CK(cudaMemcpyAsync(&h, d_xc_hits, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(&ms, d_xc_misses, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        dev_hits = h;
        dev_misses = ms;
    }

    /// The prefill path (n > 1) resolves slots on the host - it needs one contiguous weight pointer per expert group
    /// - so it takes the directory over for the length of a chunk: read it here, hand it back in
    /// device_cache_from_host().  Once per chunk, not once per layer.
    void host_cache_from_device() {
        if (!dev_cache) return;
        // `st` is a non-blocking stream, so the legacy default stream does not order against it: everything that
        // touches this state has to go through `st` and then synchronise.
        CK(cudaMemcpyAsync(xc_layer.data(), d_xc_layer, xc_slots * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(xc_expert.data(), d_xc_expert, xc_slots * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(xc_used.data(), d_xc_used, xc_slots * 8, cudaMemcpyDeviceToHost, st));
        if (d_xc_freq) CK(cudaMemcpyAsync(xc_freq.data(), d_xc_freq, xc_slots * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        xc_map.clear();
        uint64_t mx = 0;
        for (size_t i = 0; i < xc_slots; ++i) {
            if (xc_layer[i] < 0) continue;
            xc_map[((uint64_t) (uint32_t) xc_layer[i] << 32) | (uint32_t) xc_expert[i]] = (int) i;
            mx = std::max(mx, xc_used[i]);
        }
        xc_tick = mx;
        refresh_dev_counters();
    }

    void device_cache_from_host() {
        if (!dev_cache) return;
        const size_t ne = (size_t) m->g.n_expert;
        std::vector<int32_t> so((size_t) m->layers.size() * ne, -1);
        for (size_t i = 0; i < xc_slots; ++i)
            if (xc_layer[i] >= 0)
                so[(size_t) (uint32_t) xc_layer[i] * ne + (size_t) (uint32_t) xc_expert[i]] = (int32_t) i;
        CK(cudaMemcpyAsync(d_xc_layer, xc_layer.data(), xc_slots * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_xc_expert, xc_expert.data(), xc_slots * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_xc_used, xc_used.data(), xc_slots * 8, cudaMemcpyHostToDevice, st));
        if (d_xc_freq) CK(cudaMemcpyAsync(d_xc_freq, xc_freq.data(), xc_slots * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_xc_of, so.data(), so.size() * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_xc_tick, &xc_tick, 8, cudaMemcpyHostToDevice, st));
    }

    /// One chunk of `n` tokens at positions pos .. pos+n-1; the last row's logits into logits[0].
    /// One decode step through a captured graph; anything else goes straight to forward_body().
    void forward(const int32_t* toks, int n, int64_t pos0) {
        const bool g = graph_enabled && dev_cache && n == 1 && !capturing && std::getenv("STRATA_GLM4_PROF") == nullptr &&
                       std::getenv("STRATA_GLM4_DEBUG") == nullptr && std::getenv("STRATA_GLM4_ATTN") == nullptr;
        if (!g) {
            forward_body(toks, n, pos0, false);
            return;
        }
        tok_stage = toks[0];
        pos_stage = pos0;
        if (gexec == nullptr || graph_sess != cur) {
            if (gexec) {
                CK(cudaGraphExecDestroy(gexec));
                gexec = nullptr;
            }
            CK(cudaStreamSynchronize(st));
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            capturing = true;
            forward_body(&tok_stage, 1, pos0, true);
            capturing = false;
            cudaGraph_t gt = nullptr;
            CK(cudaStreamEndCapture(st, &gt));
            CK(cudaGraphInstantiate(&gexec, gt, 0));
            CK(cudaGraphDestroy(gt));
            graph_sess = cur;
        }
        CK(cudaGraphLaunch(gexec, st));
    }

    void forward_body(const int32_t* toks, int n, int64_t pos0, bool force_gemm) {
        pos_stage = pos0;   // the graph path sets it again, but every other caller relies on this
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
        const int H = (int) g.n_head, HKV = (int) g.n_head_kv, D = (int) g.head_dim;
        const bool dbg = std::getenv("STRATA_GLM4_DEBUG") != nullptr;
        if (dbg) std::fprintf(stderr, "glm4: forward n=%d pos0=%lld\n", n, (long long) pos0);
        // profiling of a decode step (n == 1 only): one event per stage boundary, deltas summed into buckets
        const bool pf = prof;   // TEMP: profile the prefill buckets too
        if (pf) pei = 0;
        auto mark = [&]() {
            if (pf) CK(cudaEventRecord(pev[(size_t) pei++], st));
        };
        if (pf && !prof_started) {
            prof_started = true;
            refresh_dev_counters();
            prof_misses0 = xc_misses + dev_misses;
        }
        if (m->experts_host && xc_slots && n > 1) host_cache_from_device();
        if (routecheck_left == 0 && std::getenv("STRATA_GLM4_ROUTECHECK")) routecheck_left = (int) m->layers.size();
        const bool need_ids = !(m->experts_host && n == 1 && dev_cache);
        mark();
        CK(cudaMemcpyAsync(tok, toks, (size_t) n * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_pos, &pos_stage, 8, cudaMemcpyHostToDevice, st));
        strata::kernels::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok, n, N, x, st);
        mark();
        for (int64_t l = 0; l < (int64_t) m->layers.size(); ++l) {
            if (dbg) std::fprintf(stderr, "glm4: layer %lld attn\n", (long long) l);
            const Layer& L = m->layers[(size_t) l];
            k_rms_rows<<<n, kThreads, 0, st>>>(x, L.attn_norm, h, N, m->eps);
            if (L.qk_merged) {
                mm(L.qk, h, qkf, n);   // q and k in one launch: the packed matrix is (H + HKV) * D rows
            } else {
                const size_t qb = quant(h, N, n, q8);
                const size_t qt = (size_t) (H + HKV) * D;   // per-row stride of the packed q|k buffer
                for (int c = 0; c < n; ++c) {
                    strata::kernels::iq_mmvq(L.wq.type, L.wq.d, (const char*) q8 + (size_t) c * qb,
                                             qkf + (size_t) c * qt, N, H * D, 1, st);
                    strata::kernels::iq_mmvq(L.wk.type, L.wk.d, (const char*) q8 + (size_t) c * qb,
                                             qkf + (size_t) c * qt + H * D, N, HKV * D, 1, st);
                }
            }
            mm(L.wv, h, vx, n);
            k_qk_prep_rows<<<(unsigned) (n * (H + HKV)), kThreads, 0, st>>>(qkf, vx, kc[(size_t) l], vc[(size_t) l],
                                                                           H, HKV, D, m->n_rot, d_pos, ctx,
                                                                           m->rope_base);
            if ((force_gemm || att_gemm || pos0 + 1 >= att_thresh) && n == 1 && D == 128 && HKV > 0 && H / HKV == 8) {
                // GEMM-style decode attention: K / V are read once for the 8 heads that share a KV head
                const int tiles = att_tiles, nsplits = att_nsplits;
                const float scale = 1.0f / sqrtf((float) D);
                cudaEvent_t* ea = pf ? &pev_a[(size_t) (8 * l)] : nullptr;
                if (pf) CK(cudaEventRecord(ea[0], st));
                k_attn_scores<<<dim3(HKV, tiles), 128, 0, st>>>(qkf, kc[(size_t) l], att_sc, H, HKV, D, ctx, d_pos,
                                                                scale);
                if (pf) CK(cudaEventRecord(ea[1], st));
                k_attn_softmax<<<H, kThreads, 0, st>>>(att_sc, H, ctx, d_pos);
                if (pf) CK(cudaEventRecord(ea[2], st));
                k_attn_values<<<dim3(HKV, nsplits), 128, 0, st>>>(att_sc, vc[(size_t) l], att_part, H, HKV, D, ctx,
                                                                  d_pos, nsplits);
                if (pf) CK(cudaEventRecord(ea[3], st));
                k_attn_combine<<<H, 128, 0, st>>>(att_part, att, H, D, nsplits, D);
                if (pf) CK(cudaEventRecord(ea[4], st));
            } else if (D == 128)
                if (att_flash && n > 1 && H / HKV == 8 && D == 128)
                    k_attn_flash<<<dim3(HKV * 2, n), kThreads, 0, st>>>(qkf, kc[(size_t) l], vc[(size_t) l], att, H,
                                                                        HKV, D, pos0, ctx, 1.0f / sqrtf((float) D));
                else
                    k_attn_rows_fast<<<dim3(H, n), kThreads, (size_t) (pos0 + n) * sizeof(float), st>>>(
                        qkf, kc[(size_t) l], vc[(size_t) l], att, H, HKV, D, pos0, ctx, 1.0f / sqrtf((float) D),
                        attn_skip);
            else
                k_attn_rows<<<dim3(H, n), kThreads, (size_t) (pos0 + n) * sizeof(float), st>>>(
                    qkf, kc[(size_t) l], vc[(size_t) l], att, H, HKV, D, pos0, ctx, 1.0f / sqrtf((float) D));
            if (l == 0 && std::getenv("STRATA_GLM4_ATTN_DUMP")) {
                std::vector<float> h((size_t) H * D);
                CK(cudaMemcpyAsync(h.data(), att, h.size() * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                double s = 0;
                for (float v : h) s += v;
                std::fprintf(stderr, "attn dump pos=%lld: sum %.6f first %.6f %.6f %.6f\n", (long long) pos0, s, h[0],
                             h[1], h[2]);
            }
            mm(L.wo, att, y, n);
            k_add_rows<<<(unsigned) ((n * N + 255) / 256), 256, 0, st>>>(x, y, n * N);
            mark();
            auto dbgstep = [&](const char* what) {
                if (!dbg) return;
                CK(cudaStreamSynchronize(st));
                std::fprintf(stderr, "glm4: layer %lld moe %s\n", (long long) l, what);
            };
            dbgstep("rms");
            cudaEvent_t* eg = pf ? &pev_g[(size_t) (6 * l)] : nullptr;
            if (pf) CK(cudaEventRecord(eg[0], st));
            k_rms_rows<<<n, kThreads, 0, st>>>(x, L.post_norm, h, N, m->eps);
            if (pf) CK(cudaEventRecord(eg[1], st));
            k_gemv_f32_rows<<<dim3(g.n_expert, n), kThreads, 0, st>>>(L.router, h, router, N, (int) g.n_expert, N);
            if (pf) CK(cudaEventRecord(eg[2], st));
            dbgstep("router");
            // The selection runs on the device (k_route_topk) and writes wts_i / wts directly, so the per-layer
            // round-trip is one download of the id list - which the expert cache needs - and nothing else.  It used to
            // be a logits download plus a host top-k plus two uploads, four driver submissions per layer.
            k_route_topk<<<n, 32, 0, st>>>(router, L.prob_bias, wts_i, wts, (int) g.n_expert, (int) Kp,
                                           m->route_scale, m->weights_norm ? 1 : 0);
            if (pf) CK(cudaEventRecord(eg[3], st));
            // STRATA_GLM4_ROUTECHECK: does the device selection agree with the host's (partial_sort with the same
            // comparator)?  Only for the first chunk of a run, and it synchronises.
            if (routecheck_left > 0) {
                {
                    std::vector<float> lg((size_t) n * g.n_expert), bb((size_t) g.n_expert);
                    std::vector<int32_t> dev((size_t) n * Kp), hidx((size_t) g.n_expert);
                    std::vector<float> hwt((size_t) n * Kp);
                    CK(cudaMemcpyAsync(lg.data(), router, lg.size() * 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaMemcpyAsync(bb.data(), L.prob_bias, bb.size() * 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaMemcpyAsync(dev.data(), wts_i, dev.size() * 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaStreamSynchronize(st));
                    int bad = 0, badw = 0;
                    float worst = 0.f;
                    for (int row = 0; row < n; ++row) {
                        float* rr = lg.data() + (size_t) row * g.n_expert;
                        std::vector<float> sel((size_t) g.n_expert);
                        for (int i = 0; i < (int) g.n_expert; ++i) {
                            rr[i] = 1.0f / (1.0f + std::exp(-rr[i]));
                            sel[(size_t) i] = rr[i] + bb[(size_t) i];
                            hidx[(size_t) i] = i;
                        }
                        std::partial_sort(hidx.begin(), hidx.begin() + Kp, hidx.end(), [&](int a, int c) {
                            return sel[(size_t) a] > sel[(size_t) c] ||
                                   (sel[(size_t) a] == sel[(size_t) c] && a < c);
                        });
                        float ws = 0;
                        for (int k = 0; k < Kp; ++k) ws += rr[hidx[(size_t) k]];
                        for (int k = 0; k < Kp; ++k) {
                            if (dev[(size_t) row * Kp + k] != hidx[(size_t) k]) ++bad;
                            const float w = m->weights_norm ? rr[hidx[(size_t) k]] / ws
                                                            : rr[hidx[(size_t) k]] * m->route_scale;
                            worst = std::max(worst, std::fabs(w - hwt[(size_t) row * Kp + k]));
                        }
                    }
                    if (bad)
                        std::fprintf(stderr, "glm4: route check (n=%d, layer %lld): %d of %d ids differ\n", n,
                                     (long long) l, bad, (int) (n * Kp));
                    routecheck_bad += bad;
                    routecheck_seen += (long long) (n * Kp);
                    if (--routecheck_left == 0)
                        std::fprintf(stderr, "glm4: route check over one chunk: %lld of %lld ids differ\n",
                                     routecheck_bad, routecheck_seen);
                }
            }
            const auto tr0 = Clock::now();
            if (need_ids) {
                const auto ts0 = Clock::now();
                std::vector<int32_t>& hi = rout_id;
                hi.resize((size_t) n * Kp);
                CK(cudaMemcpyAsync(hi.data(), wts_i, hi.size() * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                const auto ts1 = Clock::now();
                ids_host = hi;
                if (pf) {
                    prof_rsync += std::chrono::duration<double, std::milli>(ts1 - ts0).count();
                    prof_rup += std::chrono::duration<double, std::milli>(Clock::now() - ts1).count();
                }
            }
            if (pf) prof_router_ms += ms_since(tr0);
            mark();
            const size_t qb_row = quant(h, N, n, q8);
            dbgstep("route-host+quant");
            // The routed experts of this chunk grouped by expert: ONE blob copy per unique expert and layer, not one
            // per (row, expert).  A prompt chunk of 64 rows selects 512 entries per layer, but at most 128 distinct
            // experts - this is the difference between ~3 GB and ~0.4 GB of PCIe traffic per layer.
            std::map<int, std::vector<int>> groups;
            std::vector<int32_t> hcols, hrows, hstart;   // a group's slots, their rows, and where each group starts
            // Decode: ONE launch for every expert of the layer, its weight pointer read from a device table (the
            // cache slots; gate|up are back to back there, so n_out = 2*FF), plus a second for the down projection.
            bool ptrs_done = false;
            if (m->experts_host && n == 1 && dev_cache) {
                // The directory lives on the device, so nothing between the router kernel and the expert matvec needs
                // the host: k_resolve picks the slots (LRU) and lists the misses, k_fetch copies those blobs out of
                // the pinned region, and the matvec reads the pointer table k_resolve wrote.
                if (pf) CK(cudaEventRecord(pev_r[(size_t) (2 * l)], st));
                k_resolve<<<1, 128, 0, st>>>(wts_i, Kp, (int) l, (int) g.n_expert, (int) xc_slots, xc_stride,
                                           xc_g + xc_u, xcache, d_xc_layer, d_xc_expert, d_xc_of, d_xc_used,
                                           d_xc_tick, d_xc_freq, (uint8_t**) d_wp_g, (uint8_t**) d_wp_d, d_miss_e,
                                           d_miss_slot, d_miss_n, d_xc_hits, d_xc_misses);
                if (pf) CK(cudaEventRecord(pev_r[(size_t) (2 * l + 1)], st));
                const HostMat& G = hg[(size_t) l];
                const HostMat& U = hu[(size_t) l];
                const HostMat& Dd = hd[(size_t) l];
                if (pf) CK(cudaEventRecord(pev_f[(size_t) (2 * l)], st));
                if (!std::getenv("STRATA_GLM4_NOFETCH")) {   // timing probe: what does the fill cost?
                    const bool packed = hg[(size_t) l].stride != 0;
                    const bool split_here = fetch_split && !packed;
                    const dim3 fg((unsigned) (Kp * kFetchChunks), packed ? 1u : (split_here ? 2u : 3u));
                    if (split_here) {
                        // fork: the down third goes to its own stream and is joined before the down matvec
                        CK(cudaEventRecord(ev_fork, st));
                        CK(cudaStreamWaitEvent(st2, ev_fork, 0));
                    }
                    k_fetch<<<fg, 256, 0, st>>>(d_miss_e, d_miss_slot, d_miss_n, G.p,
                                                packed ? G.stride : G.expert_bytes, U.p, U.expert_bytes, Dd.p,
                                                Dd.expert_bytes, xc_g, xc_g + xc_u, xcache, xc_stride, kFetchChunks, 0);
                    if (split_here)
                        k_fetch<<<dim3(fg.x, 1u), 256, 0, st2>>>(d_miss_e, d_miss_slot, d_miss_n, G.p, G.expert_bytes,
                                                                 U.p, U.expert_bytes, Dd.p, Dd.expert_bytes, xc_g,
                                                                 xc_g + xc_u, xcache, xc_stride, kFetchChunks, 2);
                }
                if (pf) CK(cudaEventRecord(pev_f[(size_t) (2 * l + 1)], st));
                strata::kernels::iq_mmvq_ptrs(hg[(size_t) l].type, d_wp_g, d_xc_row, q8, gu, N, 2 * FF, n * Kp, st);
                ptrs_done = true;
            } else if (m->experts_host && n == 1 && xc_slots > 0 && hg[(size_t) l].type == hu[(size_t) l].type &&
                xc_g == hg[(size_t) l].row_bytes * (size_t) FF) {
                bool all = true;
                for (int col = 0; col < n * Kp; ++col) {
                    const int slot = cache_slot(l, ids_host[(size_t) col]);
                    if (slot < 0) { all = false; break; }
                    hwp_g[(size_t) col] = xcache + (size_t) slot * xc_stride;
                    hwp_d[(size_t) col] = (const uint8_t*) hwp_g[(size_t) col] + xc_g + xc_u;
                }
                if (all) {
                    CK(cudaMemcpyAsync(d_wp_g, hwp_g.data(), (size_t) n * Kp * sizeof(void*), cudaMemcpyHostToDevice,
                                       st));
                    CK(cudaMemcpyAsync(d_wp_d, hwp_d.data(), (size_t) n * Kp * sizeof(void*), cudaMemcpyHostToDevice,
                                       st));
                    strata::kernels::iq_mmvq_ptrs(hg[(size_t) l].type, d_wp_g, d_xc_row, q8, gu, N, 2 * FF, n * Kp, st);
                    ptrs_done = true;
                }
            }
            bool pipe = false;
            if (!ptrs_done && m->experts_host) {
                for (int col = 0; col < n * Kp; ++col) groups[ids_host[(size_t) col]].push_back(col);
                // Reserve every group's slot and issue its fill on st2 up front.  The fills of this layer can only
                // start once the previous layer's matvecs are done (they read the slots being overwritten), hence the
                // fork event; each group then has its own join event.
                if (prefill_pipe && groups.size() + 1 <= ev_grp.size() && n > 1) {
                    pipe = true;
                    CK(cudaEventRecord(ev_grp[0], st));
                    CK(cudaStreamWaitEvent(st2, ev_grp[0], 0));
                    size_t gi = 1;
                    for (const auto& grp : groups) {
                        cache_slot(l, grp.first, st2);
                        CK(cudaEventRecord(ev_grp[gi++], st2));
                    }
                }
                hstart.push_back(0);
                for (const auto& grp : groups) {
                    for (int col : grp.second) {
                        hcols.push_back(col);
                        hrows.push_back(col / Kp);
                    }
                    hstart.push_back((int32_t) hcols.size());
                }
                CK(cudaMemcpyAsync(d_cols, hcols.data(), hcols.size() * 4, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(d_rows, hrows.data(), hrows.size() * 4, cudaMemcpyHostToDevice, st));
            }
            if (!ptrs_done)
            if (m->experts_host) {
                const HostMat& G = hg[(size_t) l];
                const HostMat& U = hu[(size_t) l];
                const int qb_blocks = (int) ((qb_row + 255) / 256);
                int gi = 0;
                for (const auto& grp : groups) {
                    const int off = hstart[(size_t) gi], cnt = hstart[(size_t) gi + 1] - off;
                    if (pipe) CK(cudaStreamWaitEvent(st, ev_grp[(size_t) gi + 1], 0));
                    ++gi;
                    int slot = cache_slot(l, grp.first);
                    const uint8_t* gsrc;
                    const uint8_t* usrc;
                    if (slot >= 0) {
                        gsrc = xcache + (size_t) slot * xc_stride;
                        usrc = gsrc + xc_g;
                    } else {
                        stage_expert(G, exp_g, grp.first);
                        stage_expert(U, exp_u, grp.first);
                        gsrc = exp_g;
                        usrc = exp_u;
                    }
                    // gate and up are back to back in the cache slot (the same row width), so ONE launch covers both:
                    // n_out 2*FF rows starting at the gate.  (A shape the slot cannot guarantee falls back to two.)
                    const bool merged = G.type == U.type && slot >= 0 && xc_g == G.row_bytes * (size_t) FF;
                    const int nout = merged ? 2 * FF : FF;
                    if (cnt >= 2) {
                        // one weight read for the whole group: gather the rows' q8 columns, one matmul, scatter back
                        k_gather_cols<<<dim3(cnt, qb_blocks), 256, 0, st>>>(q8g, (const uint8_t*) q8, d_rows + off,
                                                                            (int) qb_row);
                        strata::kernels::iq_mmvq(G.type, gsrc, q8g, tmp_e, N, nout, cnt, st);
                        k_scatter_rows<<<dim3((nout + 255) / 256, cnt), 256, 0, st>>>(gu, tmp_e, d_cols + off, nout,
                                                                                    2 * FF);
                        if (!merged) {
                            k_gather_cols<<<dim3(cnt, qb_blocks), 256, 0, st>>>(q8g, (const uint8_t*) q8, d_rows + off,
                                                                                (int) qb_row);
                            strata::kernels::iq_mmvq(U.type, usrc, q8g, tmp_e, N, FF, cnt, st);
                            k_scatter_rows<<<dim3((FF + 255) / 256, cnt), 256, 0, st>>>(
                                gu + FF, tmp_e, d_cols + off, FF, 2 * FF);
                        }
                    } else {
                        for (int col : grp.second) {
                            const char* xq = (const char*) q8 + (size_t) (col / Kp) * qb_row;
                            strata::kernels::iq_mmvq(G.type, gsrc, xq, gu + (size_t) col * 2 * FF, N, nout, 1, st);
                            if (!merged)
                                strata::kernels::iq_mmvq(U.type, usrc, xq, gu + (size_t) col * 2 * FF + FF, N, FF, 1,
                                                         st);
                        }
                    }
                }
            } else {
                const size_t eg_step = (size_t) L.gate_exps.row_bytes * L.gate_exps.n_out;
                const size_t eu_step = (size_t) L.up_exps.row_bytes * L.up_exps.n_out;
                for (int row = 0; row < n; ++row)
                    for (int k = 0; k < Kp; ++k) {
                        const int col = row * Kp + k;
                        const int e = ids_host[(size_t) col];
                        const char* xq = (const char*) q8 + (size_t) row * qb_row;
                        strata::kernels::iq_mmvq(L.gate_exps.type, L.gate_exps.d + (size_t) e * eg_step, xq,
                                                 gu + (size_t) col * 2 * FF, N, FF, 1, st);
                        strata::kernels::iq_mmvq(L.up_exps.type, L.up_exps.d + (size_t) e * eu_step, xq,
                                                 gu + (size_t) col * 2 * FF + FF, N, FF, 1, st);
                    }
            }
            if (dbg) {
                std::vector<float> chk((size_t) n * N);
                CK(cudaMemcpyAsync(chk.data(), x, chk.size() * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                int bad = 0;
                float mx = 0;
                for (float v : chk) {
                    if (!std::isfinite(v)) ++bad;
                    else mx = std::max(mx, std::fabs(v));
                }
                std::fprintf(stderr, "glm4: layer %lld x rows=%d nonfinite=%d max=%g\n", (long long) l, n, bad, mx);
            }
            dbgstep("gate-up");
            mark();
            k_swiglu_gu<<<(unsigned) ((n * Kp * FF + 255) / 256), 256, 0, st>>>(gu, eh, n * Kp, FF);
            const size_t qb_ff = quant(eh, FF, n * Kp, q8d);
            dbgstep("swiglu+quant");
            if (ptrs_done) {
                if (fetch_split) {
                    CK(cudaEventRecord(ev_join, st2));
                    CK(cudaStreamWaitEvent(st, ev_join, 0));
                }
                strata::kernels::iq_mmvq_ptrs(hd[(size_t) l].type, d_wp_d, d_xc_col, q8d, ey, FF, N, n * Kp, st);
            } else if (m->experts_host) {
                const HostMat& D = hd[(size_t) l];
                const int qbff_blocks = (int) ((qb_ff + 255) / 256);
                int gi = 0;
                for (const auto& grp : groups) {
                    const int off = hstart[(size_t) gi], cnt = hstart[(size_t) gi + 1] - off;
                    ++gi;
                    const int slot = cache_slot(l, grp.first);
                    const uint8_t* dsrc = exp_d;
                    if (slot >= 0) dsrc = xcache + (size_t) slot * xc_stride + xc_g + xc_u;
                    else stage_expert(D, exp_d, grp.first);
                    if (cnt >= 2) {
                        k_gather_cols<<<dim3(cnt, qbff_blocks), 256, 0, st>>>(q8g, (const uint8_t*) q8d, d_cols + off,
                                                                             (int) qb_ff);
                        strata::kernels::iq_mmvq(D.type, dsrc, q8g, tmp_d, FF, N, cnt, st);
                        k_scatter_rows<<<dim3((N + 255) / 256, cnt), 256, 0, st>>>(ey, tmp_d, d_cols + off, N, N);
                    } else {
                        for (int col : grp.second)
                            strata::kernels::iq_mmvq(D.type, dsrc, (const char*) q8d + (size_t) col * qb_ff,
                                                     ey + (size_t) col * N, FF, N, 1, st);
                    }
                }
            } else {
                for (int row = 0; row < n; ++row)
                    for (int k = 0; k < Kp; ++k) {
                        const int col = row * Kp + k;
                        const int e = ids_host[(size_t) col];
                        const size_t ed = (size_t) L.down_exps.row_bytes * L.down_exps.n_out;
                        strata::kernels::iq_mmvq(L.down_exps.type, L.down_exps.d + (size_t) e * ed,
                                                 (const char*) q8d + (size_t) col * qb_ff, ey + (size_t) col * N,
                                                 FF, N, 1, st);
                    }
            }
            dbgstep("down");
            mark();
            mm(L.sh_gate, h, sg, n);
            mm(L.sh_up, h, su, n);
            k_swiglu<<<(unsigned) ((n * FS + 255) / 256), 256, 0, st>>>(sg, su, shh, n * FS);
            mm(L.sh_down, shh, sy, n);
            dbgstep("shared");
            k_moe_sum_rows<<<dim3((unsigned) ((N + 255) / 256), n), 256, 0, st>>>(x, ey, wts, Kp, sy, N);
            mark();
            dbgstep("sum");
        }
        k_rms_rows<<<n, kThreads, 0, st>>>(x, m->out_norm, h, N, m->eps);
        if (n <= 8) {
            const size_t qb = quant(h, N, n, q8);
            for (int c = 0; c < n; ++c)
                strata::kernels::iq_mmvq(m->head.type, m->head.d, (const char*) q8 + (size_t) c * qb,
                                         logits + (size_t) c * g.n_vocab, N, (int) g.n_vocab, 1, st);
            // the calls above leave one row per position; greedy() / host_logits() want the LAST position in row 0
            if (n > 1)
                CK(cudaMemcpyAsync(logits, logits + (size_t) (n - 1) * g.n_vocab, (size_t) g.n_vocab * 4,
                                   cudaMemcpyDeviceToDevice, st));
        } else {
            const size_t qb = quant(h, N, n, q8);
            strata::kernels::iq_mmvq(m->head.type, m->head.d, (const char*) q8 + (size_t) (n - 1) * qb, logits, N,
                                     (int) g.n_vocab, 1, st);
        }
        if (!capturing) CK(cudaGetLastError());
        if (m->experts_host && xc_slots && n > 1) device_cache_from_host();
        if (pf) {
            mark();
            CK(cudaEventSynchronize(pev[(size_t) (pei - 1)]));
            refresh_dev_counters();
            for (size_t i = 0; i + 1 < pev_f.size(); i += 2) {
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, pev_f[i], pev_f[i + 1]));
                prof_fetch_ms += ms;
                CK(cudaEventElapsedTime(&ms, pev_r[i], pev_r[i + 1]));
                prof_resolve_ms += ms;
            }
            for (size_t k = 0; k < (size_t) m->layers.size(); ++k)
                for (int j = 0; j < 3; ++j) {
                    float ms = 0;
                    CK(cudaEventElapsedTime(&ms, pev_g[6 * k + j], pev_g[6 * k + j + 1]));
                    prof_route_k[j] += ms;
                }
            for (size_t k = 0; k < (size_t) m->layers.size(); ++k) {
                static const int step[4] = {0, 1, 2, 3};
                for (int j = 0; j < 4; ++j) {
                    float ms = 0;
                    CK(cudaEventElapsedTime(&ms, pev_a[8 * k + step[j]], pev_a[8 * k + step[j] + 1]));
                    prof_attn_k[j] += ms;
                }
            }
            double sum = 0;
            for (int k = 1; k < pei; ++k) {
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, pev[(size_t) (k - 1)], pev[(size_t) k]));
                int b;
                if (k == 1) b = 0;                        // embed
                else if (k == pei - 1) b = 6;             // head
                else b = 1 + ((k - 2) % 5);               // attn / router / gate+up / down / shared
                prof_sum[b] += ms;
                sum += ms;
            }
            prof_total += sum;
            ++prof_tokens;
        }
    }

    /// Feed `n` tokens from the current session's position, in chunks; `last`'s logits end up in logits[0].
    int64_t process(const int32_t* toks, int64_t n, const std::function<bool(int64_t)>& progress) {
        int64_t done = 0;
        while (done < n) {
            const int take = (int) std::min<int64_t>(tmax, n - done);
            forward(toks + done, take, pos);
            pos += take;
            done += take;
            if (!progress(done)) return done;
        }
        return done;
    }

    void host_logits(std::vector<float>& lg) {
        lg.resize((size_t) m->g.n_vocab);
        CK(cudaMemcpyAsync(lg.data(), logits, lg.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
    void host_logits_row(int r, std::vector<float>& lg) {
        lg.resize((size_t) m->g.n_vocab);
        CK(cudaMemcpyAsync(lg.data(), logits + (size_t) r * m->g.n_vocab, lg.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
    int greedy() {
        std::vector<float> lg;
        host_logits(lg);
        return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    }

    /// One token per (slot, token) row, each in its own session: `n` forwards of one token.
    void step_rows(const int* slots, const int32_t* toks, int n) {
        row_logits.resize((size_t) n);
        rows_pick.resize((size_t) n);
        n_rows_last = n;
        for (int r = 0; r < n; ++r) {
            select(slots[r]);
            const int32_t one = toks[r];
            forward(&one, 1, pos);
            ++pos;
            session() .pos = pos;
            host_logits(row_logits[(size_t) r]);
            rows_pick[(size_t) r] =
                (int) (std::max_element(row_logits[(size_t) r].begin(), row_logits[(size_t) r].end()) -
                       row_logits[(size_t) r].begin());
        }
    }
    Sess& session() { return sess[(size_t) cur]; }

    void warmup() {
        reset();
        std::vector<int32_t> t(1, 1);
        forward(t.data(), 1, 0);
        reset();
    }

    /// The routed experts live in the read-only file mapping, and cudaHostRegister refuses a file view outright
    /// ("out of memory" at 1 GiB as readily as at 24, with or without Mapped/Portable - measured, see
    /// docs/SOLAR_PORT.md).  So every miss copy out of it is pageable, which measures 2.4 GB/s, and no kernel can
    /// read it at all.  Copying the blobs once into an anonymous buffer and pinning *that* makes the same
    /// cudaMemcpyAsync run at 22 GB/s (0.27 ms per 5.74 MiB blob) and lets a kernel read the region at 16-19 GB/s.
    /// Costs one sequential pass over the file at start-up - exactly the pass --prefetch would make - and the pinned
    /// pages are then no longer reclaimable, so it is off by default when VRAM is not the constraint (it is opt-in
    /// through the expert cache being on the host, i.e. always for this model).
    bool pinned = false;
    void pin_experts() {
        if (hg.empty() || !hg[0].expert_bytes) return;
        // Expert-major with the cache slot's own layout: one block holds gate | up | down of a single (layer, expert)
        // back to back.  A miss then copies one contiguous ~5.7 MiB run instead of three ~1.9 MiB ones, which is the
        // difference between 16.2 GB/s and the 20-21 GB/s the same volume reaches as one transfer.
        const size_t gpad = (hg[0].expert_bytes + 255) / 256 * 256;
        const size_t upad = (hu[0].expert_bytes + 255) / 256 * 256;
        const size_t dpad = (hd[0].expert_bytes + 255) / 256 * 256;
        const size_t stride = gpad + upad + dpad;
        const size_t ne0 = (size_t) hg[0].ne, nl = hg.size();
        const size_t total = nl * ne0 * stride;
        if (!total) return;
        const auto t0 = Clock::now();
        // cudaHostAlloc instead of malloc + cudaHostRegister: the driver commits and pins in one step, which took
        // 103 s for 34 GiB once the pages existed (it was faulting them back in under memory pressure), against a
        // second or two here.
        uint8_t* base = nullptr;
        cudaError_t e = cudaHostAlloc((void**) &base, total, cudaHostAllocMapped | cudaHostAllocPortable);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "strata-glm4: cudaHostAlloc(%.2f GiB) failed: %s; keeping the mmap\n",
                         (double) total / 1073741824.0, cudaGetErrorString(e));
            return;
        }
        // The blobs are scattered through the file, so the copy is a list of (source, destination) runs.  Eight
        // threads: at one thread it was page-fault bound (34 GiB in ~75 s).
        struct Run {
            const uint8_t* src;
            uint8_t* dst;
            size_t bytes;
        };
        std::vector<Run> runs;
        runs.reserve(nl * ne0 * 3);
        for (size_t l = 0; l < nl; ++l) {
            for (size_t e = 0; e < (size_t) hg[l].ne; ++e) {
                uint8_t* dst = base + (l * ne0 + e) * stride;
                runs.push_back({hg[l].p + e * hg[l].expert_bytes, dst, hg[l].expert_bytes});
                runs.push_back({hu[l].p + e * hu[l].expert_bytes, dst + gpad, hu[l].expert_bytes});
                runs.push_back({hd[l].p + e * hd[l].expert_bytes, dst + gpad + upad, hd[l].expert_bytes});
            }
            hg[l].p = base + l * ne0 * stride;
            hu[l].p = hg[l].p + gpad;
            hd[l].p = hg[l].p + gpad + upad;
            hg[l].stride = hu[l].stride = hd[l].stride = stride;
        }
        {
            const size_t nthreads = 8, per = (runs.size() + nthreads - 1) / nthreads;
            std::vector<std::thread> pool;
            for (size_t t = 0; t < nthreads; ++t) {
                const size_t lo = t * per, hi = std::min(runs.size(), lo + per);
                if (lo >= hi) break;
                pool.emplace_back([&runs, lo, hi] {
                    for (size_t i = lo; i < hi; ++i) std::memcpy(runs[i].dst, runs[i].src, runs[i].bytes);
                });
            }
            for (std::thread& th : pool) th.join();
        }
        pinned = true;
        std::fprintf(stderr, "strata-glm4: pinned %.2f GiB of routed experts in %.1f s (%.0f MB/s)\n",
                     (double) total / 1073741824.0, ms_since(t0) / 1000.0, (double) total / 1e6 / (ms_since(t0) / 1000.0));
    }

    /// Read every routed-expert byte once, so the file cache holds them before the first request.  Without this the
    /// first tokens pay a cold read per expert slot: ~45 ms each on the 5400 rpm D: volume, ~1 ms on the NVMe C:.
    /// The region is mmap'd, so this is plain memory traffic that the OS turns into sequential file reads.
    void prefetch_experts() {
        const auto t0 = Clock::now();
        std::vector<uint8_t> buf(8u << 20);
        uint64_t total = 0;
        for (size_t l = 0; l < hg.size(); ++l)
            total += ((uint64_t) hg[l].expert_bytes + (uint64_t) hu[l].expert_bytes + (uint64_t) hd[l].expert_bytes) *
                     (uint64_t) hg[l].ne;
        uint64_t done = 0, sink = 0;
        int pct = -1;
        std::fprintf(stderr, "strata-glm4: prefetch %.2f GiB of routed experts\n", (double) total / 1073741824.0);
        for (size_t l = 0; l < hg.size() && done < total; ++l) {
            for (const HostMat* M : {&hg[l], &hu[l], &hd[l]}) {
                const size_t bytes = M->expert_bytes * (size_t) M->ne;
                for (size_t off = 0; off < bytes; off += buf.size()) {
                    const size_t k = std::min(buf.size(), bytes - off);
                    std::memcpy(buf.data(), M->p + off, k);
                    sink += buf[0] + buf[k - 1];
                    done += k;
                }
            }
            if (const int p = (int) (done * 100 / total); p / 10 != pct / 10) {
                pct = p;
                std::fprintf(stderr, "strata-glm4: prefetch %3d%% (%.1f s)\n", p, ms_since(t0) / 1000.0);
            }
        }
        prefetch_sink = sink;
        std::fprintf(stderr, "strata-glm4: prefetch %.2f GiB in %.1f s (%.0f MB/s)\n", (double) total / 1073741824.0,
                     ms_since(t0) / 1000.0, (double) total / 1e6 / (ms_since(t0) / 1000.0));
    }
    uint64_t prefetch_sink = 0;
};

// ------------------------------------------------------------------------------------------------ sampling
using strata::qwen35::Sampling;

int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    strata::qwen35::apply_penalties(lg, s, hist);
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[(size_t) a] > lg[(size_t) b]; });
    idx.resize((size_t) kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i)
        if ((r -= p[i]) <= 0) return idx[i];
    return idx[keep - 1];
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t) x);
        s = e;
    }
    return v;
}

std::vector<int> eos_ids = {2};
bool is_eos(int t) { return std::find(eos_ids.begin(), eos_ids.end(), t) != eos_ids.end(); }

/// The end-of-generation ids a Solar file carries: eos_token_id plus the special text ends llama.cpp stops on.
void load_eos(const strata::GgufModel& gm) {
    eos_ids.clear();
    if (const strata::MetaValue* v = gm.meta().get("tokenizer.ggml.eos_token_id")) eos_ids.push_back((int) v->u);
    if (const strata::MetaValue* v = gm.meta().get("tokenizer.ggml.tokens");
        v && v->type == strata::MetaType::ARRAY) {
        static const char* const ends[] = {"<|end|>", "<|return|>", "<|call|>", "<|user|>", "<|endoftext|>"};
        for (size_t i = 0; i < v->items.size(); ++i)
            for (const char* e : ends)
                if (v->items[i].s == e) {
                    eos_ids.push_back((int) i);
                    break;
                }
    }
    std::sort(eos_ids.begin(), eos_ids.end());
    eos_ids.erase(std::unique(eos_ids.begin(), eos_ids.end()), eos_ids.end());
    if (eos_ids.empty()) eos_ids.push_back(2);
}

// ------------------------------------------------------------------------------------------------ serve
struct Slot {
    bool active = false, stop = false;
    int32_t x = 0;
    int64_t produced = 0, max_new = 0;
    Sampling smp;
    std::mt19937_64 rng;
    std::vector<int32_t> hist;
    Clock::time_point t0;
};

struct Serve {
    Engine& E;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::vector<Slot> slots;
    int n_active = 0;
    std::vector<float> lg;

    explicit Serve(Engine& e) : E(e), slots((size_t) e.n_sess) {}

    void poll_control() {
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = lines.begin(); it != lines.end();) {
            if (it->rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(it->c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
                it = lines.erase(it);
            } else if (it->rfind("BYIELD", 0) == 0) {
                it = lines.erase(it);
            } else {
                ++it;
            }
        }
    }

    void window_plain() {
        std::vector<int> rows;
        std::vector<int32_t> xs;
        for (int b = 0; b < E.n_sess; ++b)
            if (slots[(size_t) b].active) {
                rows.push_back(b);
                xs.push_back(slots[(size_t) b].x);
            }
        if (rows.empty()) return;
        E.step_rows(rows.data(), xs.data(), (int) rows.size());
        for (int r = 0; r < (int) rows.size(); ++r) {
            const int b = rows[(size_t) r];
            Slot& sl = slots[(size_t) b];
            Engine::Sess& ss = E.sess[(size_t) b];
            int y;
            if (sl.smp.needs_logits()) {
                y = sample(E.row_logits[(size_t) r], sl.smp, sl.hist, sl.rng);
            } else {
                y = E.rows_pick[(size_t) r];
            }
            std::printf("BT %d %d\n", b, y);
            ++sl.produced;
            sl.hist.push_back(y);
            ss.live.push_back(sl.x);
            const char* fin = is_eos(y) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                              : ss.pos + 1 > E.ctx ? "length" : nullptr;
            if (fin) {
                std::printf("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms_since(sl.t0));
                sl.active = false;
                ss.active = false;
                --n_active;
            } else {
                sl.x = y;
            }
        }
        std::fflush(stdout);
    }

    void window() {
        poll_control();
        window_plain();
    }

    size_t match_len(const Engine::Sess& ss, const std::vector<int32_t>& ids) const {
        const size_t L = ss.live.size();
        if (L == 0 || L >= ids.size() || !std::equal(ss.live.begin(), ss.live.end(), ids.begin())) return 0;
        return L;
    }

    int pick_session(const std::vector<int32_t>& ids) const {
        int best = -1, empty = -1, lru = -1;
        size_t bl = 0;
        for (int i = 0; i < E.n_sess; ++i) {
            const Engine::Sess& ss = E.sess[(size_t) i];
            if (ss.active) continue;
            const size_t l = match_len(ss, ids);
            if (l > bl) { bl = l; best = i; }
            if (ss.live.empty() && empty < 0) empty = i;
            if (lru < 0 || ss.used < E.sess[(size_t) lru].used) lru = i;
        }
        return best >= 0 ? best : empty >= 0 ? empty : lru;
    }

    void request(const std::string& line) {
        const bool adm = line[0] == 'B';
        if (!adm && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...> or BGEN <slot> <max_new> [keys] <ids>\n");
            std::fflush(stdout);
            return;
        }
        char* p = const_cast<char*>(line.c_str()) + (adm ? 4 : 4);
        int slot = -1;
        if (adm) {
            char* e1 = nullptr;
            const long b = std::strtol(p, &e1, 10);
            if (e1 == p) { std::printf("ERR BGEN: bad slot\n"); std::fflush(stdout); return; }
            p = e1;
            slot = (int) b;
            if (b < 0 || b >= (long) slots.size() || slots[(size_t) b].active || E.n_sess < 2) {
                std::printf("ERR BGEN: no such free slot (--batch %d)\n", E.n_sess < 2 ? 0 : E.n_sess);
                std::fflush(stdout);
                return;
            }
        }
        const long long max_new = std::strtoll(p, &p, 10);
        Sampling smp;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
            else { p = const_cast<char*>(st0); break; }
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty()) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            return;
        }
        if ((int64_t) ids.size() + max_new > E.ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, E.ctx);
            std::fflush(stdout);
            return;
        }
        for (int32_t t : ids)
            if (t < 0 || t >= (int32_t) E.m->g.n_vocab) {
                std::printf("ERR a token id is outside the vocabulary\n");
                std::fflush(stdout);
                return;
            }
        stop.store(false);
        const int s = adm ? slot : pick_session(ids);
        Engine::Sess& ss = E.sess[(size_t) s];
        size_t reuse = match_len(ss, ids);
        if (adm) {
            int src = -1;
            size_t best = reuse;
            for (int i = 0; i < E.n_sess; ++i) {
                if (i == s || E.sess[(size_t) i].active) continue;
                const size_t l = match_len(E.sess[(size_t) i], ids);
                if (l > best) { best = l; src = i; }
            }
            if (src >= 0) {
                E.copy_session(src, s);
                reuse = best;
            }
        }
        E.select(s);
        if (reuse == 0) {
            E.reset();
            ss.live.clear();
        }
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        bool cancelled = false;
        const int64_t got = E.process(ids.data() + reuse, fresh, [&](int64_t done) -> bool {
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) fresh, ms_since(t0),
                        (double) done / std::max(1e-9, ms_since(t0) / 1000.0));
            std::fflush(stdout);
            if (n_active > 0 && done < fresh) {
                poll_control();
                window();
            } else {
                poll_control();
            }
            if (stop.load() && done < fresh) { cancelled = true; return false; }
            return true;
        });
        ss.live.insert(ss.live.end(), ids.begin() + (long) reuse, ids.begin() + (long) reuse + (long) got);
        E.sess[(size_t) E.cur].pos = E.pos;
        const double prompt_ms = ms_since(t0);
        if (cancelled) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0\n", ids.size(), prompt_ms);
            if (adm) std::printf("BADM %d 0\n", slot);
            std::fflush(stdout);
            return;
        }
        std::mt19937_64 rng(smp.seed);
        static const bool pen_prompt = strata::kernels::penalty_counts_prompt(std::getenv("STRATA_PENALTY_PROMPT"));
        std::vector<int32_t> hist = strata::qwen35::penalty_history_start(ids, pen_prompt);
        const auto d0 = Clock::now();
        long long produced = 0;
        const char* finish = "length";
        int nx = smp.needs_logits() ? -1 : E.greedy();
        if (adm) {
            if (smp.needs_logits()) {
                E.host_logits(lg);
                nx = sample(lg, smp, hist, rng);
            }
            std::printf("T %d\n", nx);
            ++produced;
            hist.push_back(nx);
            const char* fin = is_eos(nx) ? "stop" : stop.load() ? "cancel" : max_new <= 1 ? "length"
                              : E.pos + 1 > E.ctx ? "length" : nullptr;
            const bool cont = fin == nullptr;
            std::printf("DONE %lld %zu %.1f %.1f %s 0\n", produced, ids.size(), prompt_ms, ms_since(d0),
                        cont ? "length" : fin);
            if (cont) {
                Slot& sl = slots[(size_t) slot];
                sl = Slot{};
                sl.active = true;
                sl.x = nx;
                sl.produced = 1;
                sl.max_new = max_new;
                sl.smp = smp;
                sl.rng = rng;
                sl.hist = hist;
                sl.t0 = Clock::now();
                ss.active = true;
                ++n_active;
            }
            std::printf("BADM %d %d\n", slot, cont ? 1 : 0);
            std::fflush(stdout);
            return;
        }
        for (;;) {
            if (smp.needs_logits()) {
                E.host_logits(lg);
                nx = sample(lg, smp, hist, rng);
            }
            auto emit = [&](int t) {
                std::printf("T %d\n", t);
                std::fflush(stdout);
                ++produced;
                hist.push_back(t);
                if (is_eos(t)) { finish = "stop"; return false; }
                if (produced >= max_new) return false;
                if (stop.load()) { finish = "cancel"; return false; }
                return true;
            };
            if (!emit(nx)) break;
            if (n_active > 0) window();
            if (E.pos + 1 > E.ctx) break;
            const int32_t one = nx;
            E.process(&one, 1, [](int64_t) { return true; });
            ss.live.push_back(nx);
            if (!smp.needs_logits()) nx = E.greedy();
        }
        E.sess[(size_t) E.cur].pos = E.pos;
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s 0\n", produced, ids.size(), prompt_ms, decode_ms, finish);
        std::fflush(stdout);
        std::fprintf(stderr, "strata-glm4: session %d: prompt %zu tokens = %zu reused + %lld read in %.0f ms, "
                     "%lld generated in %.0f ms (%.1f tok/s)\n", s, ids.size(), reuse, (long long) fresh, prompt_ms,
                     produced, decode_ms, decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0);
    }

    int run() {
        if (E.n_sess > 1) std::printf("INFO batch_slots=%d slot_cache=1\n", E.n_sess);
        std::printf("READY %d stop\n", E.ctx);
        std::fflush(stdout);
        std::thread reader([&] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop.store(true); continue; }
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        });
        reader.detach();
        for (;;) {
            std::string line;
            bool have = false;
            {
                std::unique_lock<std::mutex> lk(mu);
                if (n_active == 0) cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (!lines.empty()) {
                    line = lines.front();
                    lines.pop_front();
                    have = true;
                } else if (n_active == 0) {
                    return 0;
                }
            }
            if (!have) { window(); continue; }
            if (line.empty()) continue;
            if (line.rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(line.c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
            } else if (line.rfind("BYIELD", 0) == 0) {
            } else if (line.rfind("GEN", 0) == 0 || line.rfind("BGEN", 0) == 0) {
                request(line);
            } else {
                std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
                std::fflush(stdout);
            }
            if (n_active > 0) window();
        }
    }
};

}  // namespace

int strata::glm4::run(int argc, char** argv) {
    // One host round-trip per MoE layer sits on the critical path, so the CPU wake-up policy after a stream sync
    // matters: the default (Auto/Yield) can park the thread for tens of microseconds each time.
    if (const char* sw = std::getenv("STRATA_GLM4_SYNC")) {
        const std::string v(sw);
        if (v == "spin") cudaSetDeviceFlags(cudaDeviceScheduleSpin);
        else if (v == "yield") cudaSetDeviceFlags(cudaDeviceScheduleYield);
        else if (v == "block") cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    }
    std::string model, tokens, force, dump_full;
    int ctx = 8192, chunk = 512, n_gen = 64, batch = 0;
    int64_t expert_cache_mib = -1;
    bool serve_mode = false, warm = true, prefetch = false, ppl_mode = false;
    std::string ppl_dump;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-glm4: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--max-new") n_gen = std::atoi(next().c_str());
        else if (a == "--serve") serve_mode = true;
        else if (a == "--max-context" || a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--chunk" || a == "--prefill") { const std::string v = next(); chunk = v == "auto" ? 512 : std::atoi(v.c_str()); }
        else if (a == "--tokens") tokens = next();
        else if (a == "--tokens-file") {
            FILE* f = std::fopen(next().c_str(), "rb");
            if (!f) { std::fprintf(stderr, "strata-glm4: cannot read the tokens file\n"); return 2; }
            char buf[4096];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) tokens.append(buf, n);
            std::fclose(f);
        }
        else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--force") force = next();
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--batch" || a == "--slots" || a == "--parallel") batch = std::atoi(next().c_str());
        else if (a == "--expert-cache-mib") expert_cache_mib = std::atoll(next().c_str());
        else if (a == "--gpu" || a == "--pool-workers" || a == "--cpu-threads" || a == "--expert-vram" ||
                 a == "--expert-slots" || a == "--hybrid-rows" || a == "--draft-vocab" || a == "--mtp") {
            (void) next();   // accepted for strata's command line; no effect in this engine
        }
        else if (a == "--prefetch") prefetch = true;
        else if (a == "--ppl") ppl_mode = true;
        else if (a == "--ppl-dump") ppl_dump = next();
        else if (a == "--no-token-graph" || a == "--no-graph" || a == "--greedy" || a == "--no-warmup") {
            if (a == "--no-warmup") warm = false;
        }
        else if (serve_mode && a.rfind("--", 0) == 0) {
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-glm4: ignoring %s%s%s (not an option of this engine)\n", a.c_str(),
                         val ? " " : "", val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-glm4 --model GGUF [--max-context 8192] [--chunk 512] [--batch N]\n"
                         "       (--serve | --tokens IDS [--n 64] [--force IDS] [--dump-full-logits F])\n");
            return 2;
        }
    }
    if (model.empty()) { std::fprintf(stderr, "strata-glm4: --model is required\n"); return 2; }
    if (ctx < 256) { std::fprintf(stderr, "strata-glm4: bad --max-context\n"); return 2; }
    Model m;
    std::string err;
    try {
        if (!load(model, m, err)) { std::fprintf(stderr, "strata-glm4: %s\n", err.c_str()); return 1; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-glm4: %s\n", e.what());
        return 1;
    }
    load_eos(*m.file);
    if (chunk <= 0) chunk = 512;
    Engine E;
    E.xc_mib = expert_cache_mib;
    E.init(m, ctx, chunk, batch);
    if (prefetch && !E.pinned) E.prefetch_experts();
    if (warm) E.warmup();
    if (serve_mode) {
        Serve sv(E);
        return sv.run();
    }
    const std::vector<int32_t> prompt = parse_ids(tokens.c_str()), forced = parse_ids(force.c_str());
    if (prompt.empty()) { std::fprintf(stderr, "strata-glm4: --tokens or --serve is required\n"); return 2; }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    if (ppl_mode) {
        // Perplexity of the given token sequence - the model's own cross-entropy on it, at one token per step so the
        // head only has to run for the position being scored.  Same tokens in, same definition out, so two quants of
        // the same model are directly comparable.
        if (prompt.size() < 3) {
            std::fprintf(stderr, "strata-glm4: --ppl needs at least 3 tokens\n");
            return 2;
        }
        FILE* pd = ppl_dump.empty() ? nullptr : std::fopen(ppl_dump.c_str(), "w");
        E.reset();
        std::vector<float> lg;
        double nll = 0, kl = 0;
        int cnt = 0;
        const auto tp = Clock::now();
        for (size_t i = 0; i + 1 < prompt.size(); ++i) {
            const int32_t one = prompt[i];
            E.process(&one, 1, [](int64_t) { return true; });
            E.host_logits(lg);
            float mx = -INFINITY;
            for (float v : lg) mx = std::max(mx, v);
            double sum = 0;
            int top1 = 0;
            for (size_t j = 0; j < lg.size(); ++j) {
                sum += std::exp((double) (lg[j] - mx));
                if (lg[j] > lg[(size_t) top1]) top1 = (int) j;
            }
            const double lse = (double) mx + std::log(sum);
            const double lp = (double) lg[(size_t) prompt[i + 1]] - lse;
            nll -= lp;
            ++cnt;
            if (pd) std::fprintf(pd, "%d %d %.6f\n", (int) i, top1, -lp);
            if ((i & 127) == 0)
                std::fprintf(stderr, "strata-glm4: ppl %zu/%zu tokens, running ppl %.3f\n", i, prompt.size() - 1,
                             std::exp(nll / (double) cnt));
        }
        if (pd) std::fclose(pd);
        std::fprintf(stderr,
                     "strata-glm4: ppl: %d tokens in %.1f s, mean nll %.4f bits/token %.4f, **ppl %.3f**\n", cnt,
                     ms_since(tp) / 1000.0, nll / (double) cnt, nll / (double) cnt / std::log(2.0),
                     std::exp(nll / (double) cnt));
        (void) kl;
        return 0;
    }
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    E.reset();
    const auto t0 = Clock::now();
    E.process(prompt.data(), (int64_t) prompt.size(), [](int64_t) { return true; });
    const double pp_ms = ms_since(t0);
    const auto t1 = Clock::now();
    std::vector<float> lg;
    std::vector<int32_t> gen;
    for (int i = 0; i < n_gen; ++i) {
        const int a = E.greedy();
        gen.push_back(a);
        if (ff) {
            E.host_logits(lg);
            std::fwrite(lg.data(), 4, lg.size(), ff);
        }
        // Solar ends an assistant turn with <|end|> and this CLI path used to ignore it, so a finished translation was
        // followed by a second one until --n ran out (measured: a chapter came out two and a half times).
        if (forced.empty() && is_eos(a)) break;
        if (i + 1 == n_gen) break;
        const int32_t nx = i < (int) forced.size() ? forced[(size_t) i] : a;
        E.process(&nx, 1, [](int64_t) { return true; });
    }
    if (ff) std::fclose(ff);
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "strata-glm4: prompt %zu tokens in %.1f ms (%.2f tok/s), %zu generated in %.1f ms (%.2f tok/s)\n",
                 prompt.size(), pp_ms, prompt.size() / (pp_ms / 1000.0), gen.size(), ms_since(t1),
                 gen.size() / (ms_since(t1) / 1000.0));
    return 0;
}


































