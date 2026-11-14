// tools/glm4/expert_paths_bench.cpp - three ways to run one expert's prompt matmuls, timed on the real shapes.
//
// A 512-row prompt chunk spreads over at most 128 distinct experts, so one expert sees ~24 rows.  That is why the
// experts cannot be pre-transcoded (the routing changes every chunk) and why iq_mmvq's per-launch weight read
// dominates.  This tool times, per expert and for cnt = 8..32 rows:
//
//   mmvq   iq_mmvq (what the engine's prompt path uses today; q8_1 activations, dp4a)
//   wmma   gemm_iq_f16_grouped (the repo's unused Volta prompt-expert kernel: dequantizes into shared memory, fp16
//          tensor cores, no fp16 copy of the weights in VRAM)
//   deq    IQ3_XXS -> fp16 transcode (one warp per 256 weights, 16-byte stores) + cublasGemmEx
//
// Sixteen expert blobs are cycled so the weights come from HBM, not L2.  The last number is what a whole
// 40-layer chunk would cost (128 experts per layer per projection).
#include "strata/kernels/iq_kernels.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_));                                              \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;

namespace {
constexpr int TY = 18;      // GGML_TYPE_IQ3_XXS (the model's ffn_*_shexp)
constexpr int NNE = 16;     // expert blobs cycled, so nothing stays in L2
}  // namespace

// ---- the fast transcode: one warp per 256 weights, each lane 8 values as one 16-byte store ------------------------
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

static __device__ __forceinline__ float h2f_(ggml_half h) {
    const __half t = *reinterpret_cast<const __half*>(&h);
    return __half2float(t);
}

__global__ void deq_iq3_xxs_f16(const block_iq3_xxs* __restrict__ src, __half* __restrict__ dst, int n_in, int n_out) {
    const int per_row = n_in / QK_K;
    const int wid = (int) ((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int) (threadIdx.x & 31);
    if (wid >= n_out * per_row) return;
    const int row = wid / per_row, bk = wid % per_row;
    const block_iq3_xxs* b = src + (size_t) row * per_row + bk;
    const float d = h2f_(b->d);
    const int ib32 = lane >> 2, l = lane & 3;
    uint32_t aux32;
    memcpy(&aux32, b->qs + QK_K / 4 + 4 * ib32, sizeof(uint32_t));
    const float db = d * (0.5f + (float) (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
    const uint8_t* qs = b->qs + 8 * ib32;
    const uint8_t* g1 = (const uint8_t*) &iq3xxs_grid[qs[2 * l + 0]];
    const uint8_t* g2 = (const uint8_t*) &iq3xxs_grid[qs[2 * l + 1]];
    uint4 v;
    __half* hv = reinterpret_cast<__half*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        hv[j + 0] = __float2half(db * (float) g1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f));
        hv[j + 4] = __float2half(db * (float) g2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.f : 1.f));
    }
    *(uint4*) (dst + (size_t) row * n_in + (size_t) bk * QK_K + (size_t) lane * 8) = v;
}

/// fp32 rows -> fp16 rows (the engine's activations are fp32 between the matmuls).
__global__ void f32_to_f16(const float* __restrict__ src, __half* __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// ---- decode dot, vectorized: one warp per weight row, lane l covers 8 of every 32 values ------------------------
// Loads are the point: lane l reads 2 codebook bytes (8 weights) and one 4-byte scale/sign word per 32-value block,
// where the engine's row_dot_multi is shaped so that a warp's lanes mostly idle on 2-byte loads.
__global__ void row_dot_iq3_xxs(const uint8_t* __restrict__ W, size_t row_bytes, const uint8_t* __restrict__ xq8,
                                const __half* __restrict__ xd, float* __restrict__ y, int K, int n_out) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= n_out) return;
    const block_iq3_xxs* r = reinterpret_cast<const block_iq3_xxs*>(W + (size_t) row * row_bytes);
    const int ib32 = lane >> 2, j = lane & 3;
    float acc = 0.f;
    for (int b = 0; b < K / QK_K; ++b) {
        const block_iq3_xxs* blk = r + b;
        uint32_t aux;
        memcpy(&aux, blk->qs + QK_K / 4 + 4 * ib32, sizeof(uint32_t));
        const float db = h2f_(blk->d) * (0.5f + (float) (aux >> 28)) * 0.5f;
        const uint8_t signs = ksigns_iq2xs[(aux >> (7 * j)) & 127];
        const uint8_t* qs = blk->qs + 8 * ib32;
        const uint8_t* g1 = (const uint8_t*) &iq3xxs_grid[qs[2 * j + 0]];
        const uint8_t* g2 = (const uint8_t*) &iq3xxs_grid[qs[2 * j + 1]];
        const int xb = b * (QK_K / 32) + ib32;
        const int8_t* xa = reinterpret_cast<const int8_t*>(xq8 + (size_t) xb * 36 + 4) + 8 * j;
        float dot = 0.f;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float a1 = (float) xa[t + 0], a2 = (float) xa[t + 4];
            dot += ((signs & kmask_iq2xs[t + 0]) ? -(float) g1[t] : (float) g1[t]) * a1;
            dot += ((signs & kmask_iq2xs[t + 4]) ? -(float) g2[t] : (float) g2[t]) * a2;
        }
        acc += db * h2f_(xd[xb]) * dot;
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

int main() {
    const int N = 4096, FF = 1280;   // Solar-Open-100B: n_embd 4096, n_ff 1280
    const int GU = 2 * FF;           // gate|up merged, the layout the engine's cache slots use

    cublasHandle_t blas;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "no cuBLAS\n");
        return 1;
    }
    cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH);

    // expert blobs: gate|up (GU rows of N) and down (N rows of FF)
    const size_t gu_bytes = strata::kernels::iq_row_bytes(TY, N) * (size_t) GU;
    const size_t dn_bytes = strata::kernels::iq_row_bytes(TY, FF) * (size_t) N;
    std::vector<uint8_t*> hgu((size_t) NNE), hdn((size_t) NNE);
    {
        std::vector<uint8_t> blob(std::max(gu_bytes, dn_bytes));
        uint32_t seed = 20261114u;
        for (size_t i = 0; i < blob.size(); ++i) {
            seed = seed * 1664525u + 1013904223u;
            blob[i] = (uint8_t) (seed >> 24);
        }
        // a sane fp16 scale per 256-block, so the values are not denormal
        for (size_t off = 0; off + 2 <= blob.size(); off += 98) blob[off + 0] = 0x00, blob[off + 1] = 0x38;
        for (int e = 0; e < NNE; ++e) {
            CK(cudaMalloc(&hgu[(size_t) e], gu_bytes));
            CK(cudaMalloc(&hdn[(size_t) e], dn_bytes));
            CK(cudaMemcpy(hgu[(size_t) e], blob.data(), gu_bytes, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(hdn[(size_t) e], blob.data(), dn_bytes, cudaMemcpyHostToDevice));
        }
    }

    const int maxc = 32;
    float *x = nullptr, *yd = nullptr, *yg = nullptr, *yw = nullptr, *tmp = nullptr;
    __half *xh = nullptr, *gh = nullptr, *dh = nullptr;
    void *q8 = nullptr, *q8d = nullptr;
    CK(cudaMalloc(&x, (size_t) maxc * N * 4));
    CK(cudaMalloc(&yd, (size_t) maxc * GU * 4));
    CK(cudaMalloc(&yg, (size_t) maxc * GU * 4));
    CK(cudaMalloc(&yw, (size_t) maxc * GU * 4));
    CK(cudaMalloc(&tmp, (size_t) maxc * GU * 4));
    CK(cudaMalloc(&xh, (size_t) maxc * N * 2));
    CK(cudaMalloc(&gh, (size_t) GU * N * 2));
    CK(cudaMalloc(&dh, (size_t) N * FF * 2));
    CK(cudaMalloc(&q8, (size_t) maxc * (N / 32) * 36));
    CK(cudaMalloc(&q8d, (size_t) maxc * (FF / 32) * 36));
    std::vector<float> hx((size_t) maxc * N);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = std::sin((double) i * 0.017) * 1.5f;
    CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(yd, 0, (size_t) maxc * GU * 4));
    CK(cudaMemset(yg, 0, (size_t) maxc * GU * 4));
    CK(cudaMemset(yw, 0, (size_t) maxc * GU * 4));
    int32_t* bounds = nullptr;
    CK(cudaMalloc(&bounds, 2 * sizeof(int32_t)));
    CK(cudaMemset(bounds, 0, 2 * sizeof(int32_t)));

    const int reps = 30;
    std::printf("IQ3_XXS   n_embd %d   n_ff %d   gate|up %d rows   down %d rows   %d expert blobs cycled\n", N, FF, GU,
                N, NNE);
    std::printf("%5s | %-22s | %-22s | %-22s\n", "rows", "mmvq (today)", "wmma grouped", "deq+cuBLAS");

    for (int cnt : {8, 16, 24, 32}) {
        CK(cudaMemcpy(bounds + 1, &cnt, sizeof(int32_t), cudaMemcpyHostToDevice));
        // activations: q8_1 (for mmvq), fp16 (for wmma/deq)
        strata::kernels::quantize_q8_1_rows(x, cnt, N, q8, nullptr);
        strata::kernels::quantize_q8_1_rows(x, cnt, FF, q8d, nullptr);
        f32_to_f16<<<(unsigned) ((cnt * N + 255) / 256), 256>>>(x, xh, (int64_t) cnt * N);
        CK(cudaDeviceSynchronize());

        // ---- mmvq: gate|up then down, one launch each, per expert
        auto run_mmvq = [&]() {
            for (int e = 0; e < NNE; ++e) {
                strata::kernels::iq_mmvq(TY, hgu[(size_t) e], q8, yw, N, GU, cnt, nullptr);
                strata::kernels::iq_mmvq(TY, hdn[(size_t) e], q8d, yw, FF, N, cnt, nullptr);
            }
        };
        // ---- wmma: gemm_iq_f16_grouped with one expert per call
        auto run_wmma = [&]() {
            for (int e = 0; e < NNE; ++e) {
                strata::kernels::gemm_iq_f16_grouped(TY, hgu[(size_t) e], gu_bytes, GU, N, xh, N, bounds, 1, cnt, yg,
                                                     GU, nullptr);
                strata::kernels::gemm_iq_f16_grouped(TY, hdn[(size_t) e], dn_bytes, N, FF, xh, N, bounds, 1, cnt, yg,
                                                     GU, nullptr);
            }
        };
        // ---- deq: transcode this expert, then two cuBLAS GEMMs
        const int gblocks = (int) (((long long) GU * (N / 256) * 32 + 255) / 256);
        const int dblocks = (int) (((long long) N * (FF / 256) * 32 + 255) / 256);
        auto run_deq = [&]() {
            const float one = 1.f, zero = 0.f;
            for (int e = 0; e < NNE; ++e) {
                deq_iq3_xxs_f16<<<gblocks, 256>>>(reinterpret_cast<const block_iq3_xxs*>(hgu[(size_t) e]), gh, N, GU);
                cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, GU, cnt, N, &one, gh, CUDA_R_16F, N, xh, CUDA_R_16F, N,
                             &zero, yd, CUDA_R_32F, GU, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                deq_iq3_xxs_f16<<<dblocks, 256>>>(reinterpret_cast<const block_iq3_xxs*>(hdn[(size_t) e]), dh, FF, N);
                cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, cnt, FF, &one, dh, CUDA_R_16F, FF, xh, CUDA_R_16F, N,
                             &zero, yd, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            }
        };

        double t[3] = {0, 0, 0};
        int vi = 0;
        auto timeit = [&](auto&& f) {
            for (int i = 0; i < 3; ++i) f();
            CK(cudaDeviceSynchronize());
            const auto t0 = Clock::now();
            for (int r = 0; r < reps; ++r) f();
            CK(cudaDeviceSynchronize());
            t[vi++] = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / (reps * NNE);
        };
        timeit(run_mmvq);
        timeit(run_wmma);
        timeit(run_deq);
        std::printf("%5d | %8.3f ms %5.0f GB/s | %8.3f ms %5.0f GB/s | %8.3f ms (%.1fx / %.1fx)\n", cnt, t[0],
                    (gu_bytes + dn_bytes) / 1e9 / (t[0] * 1e-3), t[1],
                    (gu_bytes + dn_bytes) / 1e9 / (t[1] * 1e-3), t[2], t[0] / t[1], t[0] / t[2]);
        if (cnt == 24) {
            // sanity: wmma against the q8_1 dot (different rounding, same product)
            std::vector<float> b((size_t) GU), c((size_t) GU);
            const size_t off = (size_t) 24 * GU;
            // gate|up of expert 0, row 24? rows are 0..cnt-1 here, so compare row 0..2 of expert 0
            CK(cudaMemcpy(b.data(), yw, (size_t) GU * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(c.data(), yg, (size_t) GU * 4, cudaMemcpyDeviceToHost));
            double mx = 0, rel = 0;
            int nz = 0;
            for (int i = 0; i < GU; ++i) {
                if (!std::isfinite(c[(size_t) i])) { std::printf("  wmma produced a non-finite value\n"); break; }
                if (b[(size_t) i] != 0.f) ++nz;
                rel = std::max(rel, (double) std::fabs(c[(size_t) i] - b[(size_t) i]) / (1.0 + std::fabs(b[(size_t) i])));
                mx = std::max(mx, (double) std::fabs(c[(size_t) i]));
            }
            std::printf("        row 0 gate|up: mmvq max|y| %.4g, wmma max|y| %.4g, max rel diff %.3g (%d/%d nonzero)\n", mx,
                        mx, rel, nz, GU);
            (void) off;
            std::printf("        one 40-layer chunk (128 experts/layer, gate|up+down): mmvq %.1f s | wmma %.1f s | deq %.1f s\n",
                        128 * 40 * t[0] / 1e3, 128 * 40 * t[1] / 1e3, 128 * 40 * t[2] / 1e3);
            // what the same three cost at one row, i.e. decode
            const int c1 = 1;
            CK(cudaMemcpy(bounds + 1, &c1, sizeof(int32_t), cudaMemcpyHostToDevice));
            strata::kernels::quantize_q8_1_rows(x, c1, N, q8, nullptr);
            strata::kernels::quantize_q8_1_rows(x, c1, FF, q8d, nullptr);
            CK(cudaDeviceSynchronize());
            auto dec_mmvq = [&]() {
                for (int e = 0; e < NNE; ++e) {
                    strata::kernels::iq_mmvq(TY, hgu[(size_t) e], q8, yw, N, GU, 1, nullptr);
                    strata::kernels::iq_mmvq(TY, hdn[(size_t) e], q8d, yw, FF, N, 1, nullptr);
                }
            };
            auto dec_dot = [&]() {
                for (int e = 0; e < NNE; ++e) {
                    row_dot_iq3_xxs<<<(GU + 7) / 8, 256>>>(hgu[(size_t) e], gu_bytes / GU, (const uint8_t*) q8,
                                                           (const __half*) q8, yd, N, GU);
                    row_dot_iq3_xxs<<<(N + 7) / 8, 256>>>(hdn[(size_t) e], dn_bytes / N, (const uint8_t*) q8d,
                                                           (const __half*) q8d, yd, FF, N);
                }
            };
            double d[2] = {0, 0};
            vi = 0;
            timeit(dec_mmvq);
            d[0] = t[0];
            t[0] = t[1] = t[2] = 0;
            vi = 0;
            timeit(dec_dot);
            d[1] = t[0];
            const double gbd = (gu_bytes + dn_bytes) / 1e9;
            std::printf("        decode, one row: mmvq %.3f ms (%3.0f GB/s) | warp-per-row dot %.3f ms (%3.0f GB/s) -> %.1fx\n",
                        d[0], gbd / (d[0] * 1e-3), d[1], gbd / (d[1] * 1e-3), d[0] / d[1]);
            std::printf("        one token, 8 experts x 40 layers: mmvq %.1f ms (%.0f tok/s) | dot %.1f ms (%.0f tok/s)\n",
                        8 * 40 * d[0], 1e3 / (8 * 40 * d[0]), 8 * 40 * d[1], 1e3 / (8 * 40 * d[1]));
            // and the two answers must agree (fp32 sums, different order)
            std::vector<float> am((size_t) GU), ad((size_t) GU);
            dec_mmvq();
            CK(cudaMemcpy(am.data(), yw, (size_t) GU * 4, cudaMemcpyDeviceToHost));
            dec_dot();
            CK(cudaMemcpy(ad.data(), yd, (size_t) GU * 4, cudaMemcpyDeviceToHost));
            double worst = 0, mxv = 0;
            int nzd = 0;
            for (int i = 0; i < GU; ++i) {
                mxv = std::max(mxv, (double) std::fabs(am[(size_t) i]));
                if (ad[(size_t) i] != 0.f) ++nzd;
                worst = std::max(worst, (double) std::fabs(ad[(size_t) i] - am[(size_t) i]) / (1.0 + std::fabs(am[(size_t) i])));
            }
            std::printf("        gate|up row 0: max|y| %.4g, max rel diff %.3g, %d/%d nonzero\n", mxv, worst, nzd, GU);
        }
    }
    return 0;
}
