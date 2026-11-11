// tools/glm4/mmvq_bench.cpp - what does one i-quant matvec cost at Solar's expert shapes?
// Times iq_mmvq for 4096x1280 (gate/up) and 1280x4096 (down) at 1..8 columns, plus a tiny 32x32 call for the
// fixed per-launch cost, and prints the effective weight bandwidth.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
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

static double bench(int type, int n_in, int n_out, int ncols, int reps) {
    const size_t wb = strata::kernels::iq_row_bytes(type, n_in) * (size_t) n_out;
    uint8_t* w = nullptr;
    CK(cudaMalloc(&w, wb));
    CK(cudaMemset(w, 1, wb));
    const size_t x_bytes = (size_t) (n_in / 32) * 36 * (size_t) ncols;
    void* x = nullptr;
    float* y = nullptr;
    CK(cudaMalloc(&x, x_bytes));
    CK(cudaMemset(x, 0, x_bytes));
    CK(cudaMalloc(&y, (size_t) n_out * ncols * 4));
    CK(cudaMemset(y, 0, (size_t) n_out * ncols * 4));
    // warm up
    for (int i = 0; i < 3; ++i) strata::kernels::iq_mmvq(type, w, x, y, n_in, n_out, ncols, nullptr);
    CK(cudaDeviceSynchronize());
    const auto t0 = Clock::now();
    for (int i = 0; i < reps; ++i) strata::kernels::iq_mmvq(type, w, x, y, n_in, n_out, ncols, nullptr);
    CK(cudaDeviceSynchronize());
    const double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / reps;
    std::printf("  type %2d  %5d x %5d  cols %d: %8.1f us/call   %6.1f GB/s of weights\n", type, n_in, n_out, ncols,
                us, us > 0 ? (double) wb / (us * 1e3) : 0.0);
    CK(cudaFree(w));
    CK(cudaFree(x));
    CK(cudaFree(y));
    return us;
}

/// The indirect path as decode uses it: `ncols` DISTINCT expert matrices, one launch, weight pointers from a device
/// table.  STRATA_MMVQ_PTRS_RPB selects the rows per block (4 or 8).
static double bench_ptrs(int type, int n_in, int n_out, int ncols, int reps) {
    const size_t wb = strata::kernels::iq_row_bytes(type, n_in) * (size_t) n_out;
    std::vector<uint8_t*> mats((size_t) ncols);
    for (int c = 0; c < ncols; ++c) {
        CK(cudaMalloc(&mats[(size_t) c], wb));
        CK(cudaMemset(mats[(size_t) c], 1, wb));
    }
    void** tab = nullptr;
    CK(cudaMalloc(&tab, sizeof(void*) * (size_t) ncols));
    CK(cudaMemcpy(tab, mats.data(), sizeof(void*) * (size_t) ncols, cudaMemcpyHostToDevice));
    const size_t x_bytes = (size_t) (n_in / 32) * 36 * (size_t) ncols;
    void* x = nullptr;
    CK(cudaMalloc(&x, x_bytes));
    CK(cudaMemset(x, 0, x_bytes));
    int32_t* xcol = nullptr;
    CK(cudaMalloc(&xcol, sizeof(int32_t) * (size_t) ncols));
    {
        std::vector<int32_t> h((size_t) ncols);
        for (int c = 0; c < ncols; ++c) h[(size_t) c] = c;
        CK(cudaMemcpy(xcol, h.data(), sizeof(int32_t) * (size_t) ncols, cudaMemcpyHostToDevice));
    }
    float* y = nullptr;
    CK(cudaMalloc(&y, (size_t) n_out * ncols * 4));
    CK(cudaMemset(y, 0, (size_t) n_out * ncols * 4));
    for (int i = 0; i < 3; ++i)
        strata::kernels::iq_mmvq_ptrs(type, tab, xcol, x, y, n_in, n_out, ncols, nullptr);
    CK(cudaDeviceSynchronize());
    const auto t0 = Clock::now();
    for (int i = 0; i < reps; ++i)
        strata::kernels::iq_mmvq_ptrs(type, tab, xcol, x, y, n_in, n_out, ncols, nullptr);
    CK(cudaDeviceSynchronize());
    const double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / reps;
    std::printf("  ptrs %2d  %5d x %5d  cols %d: %8.1f us/call   %6.1f GB/s of weights (rpb=%s)\n", type, n_in,
                n_out, ncols, us, us > 0 ? (double) (wb * (size_t) ncols) / (us * 1e3) : 0.0,
                std::getenv("STRATA_MMVQ_PTRS_RPB") ? std::getenv("STRATA_MMVQ_PTRS_RPB") : "4");
    for (uint8_t* m : mats) CK(cudaFree(m));
    CK(cudaFree(tab));
    CK(cudaFree(x));
    CK(cudaFree(xcol));
    CK(cudaFree(y));
    return us;
}

/// The same launch, but the 8 matrices it reads are a different 8 out of `nmats`, so the working set
/// (nmats * matrix size) can be pushed past the 32 MB L2.  This is what decode actually does: every layer reads
/// fresh experts, so the effective bandwidth is the DRAM number, not the cache-resident one above.
static double bench_ptrs_rot(int type, int n_in, int n_out, int ncols, int nmats, int reps) {
    const size_t wb = strata::kernels::iq_row_bytes(type, n_in) * (size_t) n_out;
    std::vector<uint8_t*> mats((size_t) nmats);
    for (int c = 0; c < nmats; ++c) {
        CK(cudaMalloc(&mats[(size_t) c], wb));
        CK(cudaMemset(mats[(size_t) c], 1, wb));
    }
    void** tab = nullptr;
    CK(cudaMalloc(&tab, sizeof(void*) * (size_t) nmats));
    CK(cudaMemcpy(tab, mats.data(), sizeof(void*) * (size_t) nmats, cudaMemcpyHostToDevice));
    const size_t x_bytes = (size_t) (n_in / 32) * 36 * (size_t) ncols;
    void* x = nullptr;
    CK(cudaMalloc(&x, x_bytes));
    CK(cudaMemset(x, 0, x_bytes));
    int32_t* xcol = nullptr;
    CK(cudaMalloc(&xcol, sizeof(int32_t) * (size_t) ncols));
    {
        std::vector<int32_t> h((size_t) ncols);
        for (int c = 0; c < ncols; ++c) h[(size_t) c] = c;
        CK(cudaMemcpy(xcol, h.data(), sizeof(int32_t) * (size_t) ncols, cudaMemcpyHostToDevice));
    }
    float* y = nullptr;
    CK(cudaMalloc(&y, (size_t) n_out * ncols * 4));
    CK(cudaMemset(y, 0, (size_t) n_out * ncols * 4));
    for (int i = 0; i < 3; ++i)
        strata::kernels::iq_mmvq_ptrs(type, tab, xcol, x, y, n_in, n_out, ncols, nullptr);
    CK(cudaDeviceSynchronize());
    const auto t0 = Clock::now();
    int base = 0;
    for (int i = 0; i < reps; ++i) {
        strata::kernels::iq_mmvq_ptrs(type, tab + base, xcol, x, y, n_in, n_out, ncols, nullptr);
        base = (base + ncols) % (nmats - ncols);
    }
    CK(cudaDeviceSynchronize());
    const double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / reps;
    std::printf("  rot  %2d  %5d x %5d  cols %d over %d mats (%.0f MB working set): %8.1f us/call  %6.1f GB/s\n", type,
                n_in, n_out, ncols, nmats, (double) wb * nmats / 1e6, us,
                us > 0 ? (double) (wb * (size_t) ncols) / (us * 1e3) : 0.0);
    for (uint8_t* m : mats) CK(cudaFree(m));
    CK(cudaFree(tab));
    CK(cudaFree(x));
    CK(cudaFree(xcol));
    CK(cudaFree(y));
    return us;
}

/// The prefill shape: one expert matrix, many activation columns.  Throughput as effective TMAC/s tells whether the
/// prompt pass is limited by the dots themselves or by the per-call overhead.
static void bench_cols(int type, int n_in, int n_out, int ncols, int reps) {
    const size_t wb = strata::kernels::iq_row_bytes(type, n_in) * (size_t) n_out;
    uint8_t* w = nullptr;
    CK(cudaMalloc(&w, wb));
    CK(cudaMemset(w, 1, wb));
    const size_t xb = (size_t) (n_in / 32) * 36 * (size_t) ncols;
    void* x = nullptr;
    CK(cudaMalloc(&x, xb));
    CK(cudaMemset(x, 0, xb));
    float* y = nullptr;
    CK(cudaMalloc(&y, (size_t) n_out * ncols * 4));
    for (int i = 0; i < 2; ++i)
        strata::kernels::iq_mmvq(type, w, x, y, n_in, n_out, ncols, nullptr);
    CK(cudaDeviceSynchronize());
    const auto t0 = Clock::now();
    for (int i = 0; i < reps; ++i) strata::kernels::iq_mmvq(type, w, x, y, n_in, n_out, ncols, nullptr);
    CK(cudaDeviceSynchronize());
    const double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / reps;
    const double mac = (double) n_in * n_out * ncols;
    std::printf("  cols %4d  %5d x %5d: %9.1f us  -> %5.2f TMAC/s  (%6.1f GFLOP/s)\n", ncols, n_in, n_out, us,
                mac / (us * 1e6), 2.0 * mac / (us * 1e6));
    CK(cudaFree(w));
    CK(cudaFree(x));
    CK(cudaFree(y));
}

int main() {
    std::printf("prefill shape (one expert, many columns), IQ3_XXS:\n");
    for (int c : {8, 16, 32, 64, 128, 256}) bench_cols(18, 4096, 2560, c, 30);
    for (int c : {8, 16, 32, 64, 128, 256}) bench_cols(18, 1280, 4096, c, 30);
    std::printf("IQ2_XXS for comparison:\n");
    for (int c : {8, 32, 128}) bench_cols(16, 4096, 2560, c, 30);
    std::printf("DRAM-sized working set (what decode sees):\n");
    for (int type : {18, 16}) {
        bench_ptrs_rot(type, 4096, 2560, 8, 64, 100);   // 64 * 4 MB = 256 MB, far past L2
        bench_ptrs_rot(type, 1280, 4096, 8, 64, 100);
    }
    std::printf("iq_mmvq microbenchmark (weights read from VRAM, one call at a time)\n");
    std::printf("indirect expert path (8 distinct matrices, one launch):\n");
    for (int type : {18, 16}) {
        bench_ptrs(type, 4096, 2560, 8, 100);   // gate|up merged, 8 experts
        bench_ptrs(type, 1280, 4096, 8, 100);   // down, 8 experts
    }
    for (int type : {18, 16, 12}) {   // IQ3_XXS, IQ2_XXS, Q4_K
        std::printf("type %d (%s):\n", type, type == 18 ? "IQ3_XXS" : type == 16 ? "IQ2_XXS" : "Q4_K");
        for (int cols : {1, 2, 4, 8}) bench(type, 4096, 1280, cols, 200);
        for (int cols : {1, 8}) bench(type, 1280, 4096, cols, 200);
    }
    std::printf("fixed cost (32 x 32, cols 1):\n");
    bench(18, 256, 32, 1, 2000);
    return 0;
}
