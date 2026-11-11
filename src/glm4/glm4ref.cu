// src/glm4/glm4ref.cu - the reference path for Solar-Open-100B (GGUF architecture glm4moe), exactness first.
//
// One GPU, the whole model resident in VRAM (IQ3_XXS: 36.9 GB on a 16 GB card does NOT fit; see the placement note
// below), one token at a time, fp32 activations, every weight read in its GGUF form.  The quantized matmuls run on
// Strata's i-quant MMVQ (src/kernels/cuda/iq_kernels.cu: llama.cpp's vec_dot for IQ2_S / IQ3_S / IQ3_XXS / Q5_K,
// q8_1 activations), so a weight means what it means in llama.cpp.
//
// The forward pass, as llama.cpp's src/models/glm4-moe.cpp builds it:
//   x = token_embd[tok]
//   per layer:
//       h  = rms(x) * attn_norm
//       q  = W_q h (H * D), k = W_k h (HKV * D), v = W_v h (HKV * D)   -- no QK norm, no output gate
//       q, k: NEOX RoPE over all n_rot dims (base 1e6); GQA H / HKV; softmax(q k^T / sqrt(D)) v
//       x += W_o attn
//       h2 = rms(x) * post_attention_norm
//       logits = W_router h2 + exp_probs_b; p = sigmoid(logits); top-k; weights = p / sum(p_top)
//       routed = sum_i w_i * W_down_i (silu(W_gate_i h2) * W_up_i h2)
//       shared = W_down_s (silu(W_gate_s h2) * W_up_s h2)            -- always on, no scalar gate tensor
//       x += routed + shared
//   logits = W_output (rms(x) * output_norm)
//
// Usage: strata-glm4ref --model Solar-Open-100B.gguf --tokens ID,ID,... [--n 8] [--ctx 2048]
//                        [--dump-layers FILE] [--dump-logits FILE] [--dump-full-logits F] [--force IDS]
//   prints the greedy ids ("gen: id id ..."); --dump-logits writes each step's top-2 (id, logit) pairs,
//   --dump-full-logits every step's logits (f32, [step][vocab]) for a teacher-forced KL check.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "glm4ref: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

// ------------------------------------------------------------------------------------------------ plain dequantisation
// Only the formats a 1-D / f32 weight can have; every quantized 2-D weight goes through iq_mmvq.
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

constexpr int kThreads = 128;

/// y[sel][o] = W_e[o] . x[sel] for row o (grid.x), selection `sel` (grid.y): expert ids[sel] (or 0 without ids),
/// x of selection sel at x + sel * x_stride.
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

template <int T> __global__ void k_row(const uint8_t* W, size_t row_bytes, int row, int n, float* out) {
    const uint8_t* r = W + (size_t) row * row_bytes;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < n / 32; c += gridDim.x * blockDim.x) {
        float v[32];
        deq32<T>(r, c, v);
        for (int l = 0; l < 32; ++l) out[32 * c + l] = v[l];
    }
}

/// y = x / sqrt(mean(x^2) + eps) * w, per group of n (grid = groups); w (length n) may be null.
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xg = x + (size_t) blockIdx.x * n;
    float* yg = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xg[i] * xg[i];
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yg[i] = xg[i] * r * (w ? w[i] : 1.0f);
}

__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
inline float sigm_host(float v) { return 1.0f / (1.0f + std::exp(-v)); }

/// q / k preparation: NEOX RoPE over the first n_rot dims, k and v appended to the cache at `pos` (head < H: q).
__global__ void k_attn_prep(const float* qg, const float* kx, const float* vx, float* q, float* kc, float* vc, int H,
                            int /*HKV: the grid*/, int D, int n_rot, int pos, int ctx, float base) {
    __shared__ float buf[512];
    const int head = blockIdx.x;
    const bool isq = head < H;
    const float* src = isq ? qg + (size_t) head * D : kx + (size_t) (head - H) * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) buf[i] = src[i];
    __syncthreads();
    float* dst = isq ? q + (size_t) head * D : kc + ((size_t) (head - H) * ctx + pos) * D;
    const int half = n_rot / 2;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = buf[i];
        if (i < n_rot) {
            const int p = i < half ? i : i - half;
            const float theta = (float) pos * powf(base, -2.0f * (float) p / (float) n_rot);
            const float cs = cosf(theta), sn = sinf(theta);
            v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
        }
        dst[i] = v;
    }
    if (!isq)
        for (int i = threadIdx.x; i < D; i += blockDim.x)
            vc[((size_t) (head - H) * ctx + pos) * D + i] = vx[(size_t) (head - H) * D + i];
}

/// softmax(q k^T * scale) v for one query head per block (GQA: KV head = head / (H / HKV)).
__global__ void k_attn(const float* q, const float* kc, const float* vc, float* out, int H, int HKV, int D, int n_pos,
                       int ctx, float scale) {
    extern __shared__ float sc[];
    __shared__ float sh[kThreads];
    const int h = blockIdx.x, kvh = h / (H / HKV);
    const float* qh = q + (size_t) h * D;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) {
        const float* kp = kc + ((size_t) kvh * ctx + p) * D;
        float s = 0.f;
        for (int i = 0; i < D; ++i) s += qh[i] * kp[i];
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
        for (int p = 0; p < n_pos; ++p) acc += sc[p] * vc[((size_t) kvh * ctx + p) * D + i];
        out[(size_t) h * D + i] = acc / sum;
    }
}

__global__ void k_swiglu(const float* g, const float* u, float* h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
__global__ void k_add(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
/// x += sum_s w[s] * y[s] (the routed experts) + ys (the shared expert, always on).
__global__ void k_moe_sum(float* x, const float* y, const float* w, int k, const float* ys, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    for (int s = 0; s < k; ++s) acc += w[s] * y[(size_t) s * n + i];
    x[i] += acc + ys[i];
}

// ------------------------------------------------------------------------------------------------ the model
struct Tensor {
    const uint8_t* d = nullptr;   // device
    int type = -1;
    std::vector<int64_t> shape;
    size_t row_bytes = 0;
};

/// A tensor left in the GGUF's mmap (the routed experts of a model that does not fit in VRAM): each selected
/// expert's blob is copied to a device scratch before iq_mmvq, so only 2-3 MB crosses PCIe per expert.
struct HostTensor {
    const uint8_t* p = nullptr;
    int type = -1;
    std::vector<int64_t> shape;
    size_t row_bytes = 0;
    size_t expert_bytes = 0;
};

size_t row_bytes_of(int type, int64_t n) {
    if (type == T_F32) return (size_t) n * 4;
    if (type == T_F16 || type == T_BF16) return (size_t) n * 2;
    return strata::kernels::iq_row_bytes(type, n);
}

float* g_q8 = nullptr;       // q8_1 activations (one column of n_embd, or one per selected expert)
float* g_exp_raw = nullptr;  // device scratch for one host-resident expert blob
cudaStream_t g_st = nullptr;
std::vector<int> g_ids_host;

/// The q8_1 quantisation both gemv paths share: `cols` columns of n_in values (`x_stride` apart, or packed).
size_t quantize_columns(const float* x, int n_in, int cols, size_t x_stride) {
    if (x_stride && x_stride != (size_t) n_in) {
        std::fprintf(stderr, "glm4ref: the i-quant path wants contiguous input columns\n");
        std::exit(1);
    }
    strata::kernels::quantize_q8_1_rows(x, cols, n_in, g_q8, g_st);
    return (size_t) (n_in / 32) * 36;   // block_q8_1 = f16 d + f16 s + 32 int8
}

/// A routed-expert matrix from the GGUF's mmap: copy each selected expert's blob, then iq_mmvq it.
void gemv_host(const HostTensor& w, const float* x, float* y, const int* ids, int n_sel, size_t x_stride) {
    const int n_in = (int) w.shape[0], n_out = (int) w.shape[1];
    const int cols = x_stride ? n_sel : 1;
    const size_t qb = quantize_columns(x, n_in, cols, x_stride);
    for (int sel = 0; sel < n_sel; ++sel) {
        const size_t e = ids ? (size_t) g_ids_host[(size_t) sel] : 0;
        CK(cudaMemcpyAsync(g_exp_raw, w.p + e * w.expert_bytes, w.expert_bytes, cudaMemcpyHostToDevice, g_st));
        strata::kernels::iq_mmvq(w.type, g_exp_raw, (const char*) g_q8 + (x_stride ? (size_t) sel * qb : 0),
                                 y + (size_t) sel * n_out, n_in, n_out, 1, g_st);
    }
    CK(cudaGetLastError());
}

void gemv(const Tensor& w, const float* x, float* y, const int* ids = nullptr, int n_sel = 1, size_t x_stride = 0) {
    const int n_in = (int) w.shape[0], n_out = (int) w.shape[1];
    const size_t eb = w.shape.size() == 3 ? w.row_bytes * (size_t) w.shape[1] : 0;
    if (strata::kernels::iq_supported(w.type)) {
        const int cols = x_stride ? n_sel : 1;
        const size_t qb = quantize_columns(x, n_in, cols, x_stride);
        for (int sel = 0; sel < n_sel; ++sel) {
            const uint8_t* we = w.d + (ids ? (size_t) g_ids_host[(size_t) sel] * eb : 0);
            strata::kernels::iq_mmvq(w.type, we, (const char*) g_q8 + (x_stride ? (size_t) sel * qb : 0),
                                     y + (size_t) sel * n_out, n_in, n_out, 1, g_st);
        }
        CK(cudaGetLastError());
        return;
    }
    switch (w.type) {
    case T_F32: k_gemv<T_F32><<<dim3(n_out, n_sel), kThreads>>>(w.d, w.row_bytes, eb, ids, n_in, x, x_stride, y, n_out); break;
    case T_F16: k_gemv<T_F16><<<dim3(n_out, n_sel), kThreads>>>(w.d, w.row_bytes, eb, ids, n_in, x, x_stride, y, n_out); break;
    case T_BF16: k_gemv<T_BF16><<<dim3(n_out, n_sel), kThreads>>>(w.d, w.row_bytes, eb, ids, n_in, x, x_stride, y, n_out); break;
    default:
        std::fprintf(stderr, "glm4ref: no kernel for ggml type %d (%s)\n", w.type,
                     strata::ggml_type_name((uint32_t) w.type));
        std::exit(1);
    }
    CK(cudaGetLastError());
}

struct Model {
    strata::core::ModelGeometry g;
    int64_t top_k = 8;
    float eps = 1e-5f, rope_base = 1e6f, route_scale = 1.0f;
    int n_rot = 128;
    bool weights_norm = true;
    bool experts_host = false;           // the routed experts stay in the GGUF's mmap (they do not fit in VRAM)
    std::unique_ptr<strata::GgufModel> gm;   // keeps the mmap alive for the host tensors
    std::map<std::string, Tensor> t;
    std::map<std::string, HostTensor> h;
    std::map<std::string, float*> f32;   // the small vectors, f32

    const HostTensor& H(const std::string& n) const {
        auto it = h.find(n);
        if (it == h.end()) {
            std::fprintf(stderr, "glm4ref: missing host tensor %s\n", n.c_str());
            std::exit(1);
        }
        return it->second;
    }
    const Tensor& W(const std::string& n) const {
        auto it = t.find(n);
        if (it == t.end()) {
            std::fprintf(stderr, "glm4ref: missing tensor %s\n", n.c_str());
            std::exit(1);
        }
        return it->second;
    }
    const float* F(const std::string& n) const {
        auto it = f32.find(n);
        if (it == f32.end()) {
            std::fprintf(stderr, "glm4ref: missing vector %s\n", n.c_str());
            std::exit(1);
        }
        return it->second;
    }
};

float h2f_host(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) return (s ? -1.f : 1.f) * std::ldexp((float) m, -24);
    const uint32_t f = e == 31 ? (s | 0x7f800000u | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
    float r;
    std::memcpy(&r, &f, 4);
    return r;
}

bool load(const std::string& path, Model& m, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    m.gm = std::make_unique<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.gm;
    if (!strata::core::geometry_from_gguf(gm.meta(), m.g, m.top_k, err)) return false;
    if (m.g.family != strata::ModelFamily::Glm4Moe) {
        err = "this reference path is for Solar-Open-100B (glm4moe) only";
        return false;
    }
    size_t checked = 0;
    if (!strata::core::check_gguf_layout(gm, m.g, err, &checked)) return false;
    const std::string p = "glm4moe.";
    if (const strata::MetaValue* v = gm.meta().get(p + "attention.layer_norm_rms_epsilon")) m.eps = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.freq_base")) m.rope_base = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.dimension_count")) m.n_rot = (int) v->u;
    if (const strata::MetaValue* v = gm.meta().get(p + "expert_weights_scale")) m.route_scale = (float) v->num();
    m.weights_norm = m.g.expert_weights_norm;
    uint64_t total = 0;
    for (size_t s = 0; s < gm.size(); ++s)
        for (const strata::TensorInfo& ti : gm.shard(s).tensors()) total += strata::tensor_payload_bytes(ti);
    size_t free_vram = 0, total_vram = 0;
    CK(cudaMemGetInfo(&free_vram, &total_vram));
    m.experts_host = total + (uint64_t) 2 << 30 > (uint64_t) free_vram;
    if (m.experts_host)
        std::fprintf(stderr,
                     "glm4ref: %.2f GiB of weights do not fit in %.2f GiB of free VRAM: the routed experts stay in "
                     "the GGUF mmap (one expert blob crosses PCIe per expert per matrix)\n",
                     (double) total / (1 << 30), (double) free_vram / (1 << 30));
    uint64_t bytes = 0;
    for (size_t s = 0; s < gm.size(); ++s) {
        const strata::GgufFile& f = gm.shard(s);
        for (const strata::TensorInfo& ti : f.tensors()) {
            const uint64_t nb = strata::tensor_payload_bytes(ti);
            const uint8_t* src = f.tensor_data(ti);
            if (ti.shape.size() == 1) {
                // the small vectors: f32 on the device
                std::vector<float> h(ti.elements());
                if (ti.type == T_F32) std::memcpy(h.data(), src, h.size() * 4);
                else if (ti.type == T_BF16)
                    for (size_t i = 0; i < h.size(); ++i) {
                        const uint32_t u = (uint32_t) (src[2 * i] | (src[2 * i + 1] << 8)) << 16;
                        std::memcpy(&h[i], &u, 4);
                    }
                else if (ti.type == T_F16)
                    for (size_t i = 0; i < h.size(); ++i) h[i] = h2f_host((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
                else {
                    err = ti.name + ": a 1-D tensor of type " + ti.type_name();
                    return false;
                }
                float* d = nullptr;
                CK(cudaMalloc(&d, h.size() * 4));
                CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
                m.f32[ti.name] = d;
                bytes += h.size() * 4;
                continue;
            }
            Tensor w;
            w.type = (int) ti.type;
            for (uint64_t d : ti.shape) w.shape.push_back((int64_t) d);
            w.row_bytes = row_bytes_of(w.type, w.shape[0]);
            if (m.experts_host && ti.shape.size() == 3) {
                HostTensor hw;
                hw.p = src;
                hw.type = w.type;
                hw.shape = w.shape;
                hw.row_bytes = w.row_bytes;
                hw.expert_bytes = w.row_bytes * (size_t) w.shape[1];
                m.h[ti.name] = hw;
                bytes += nb;
                continue;
            }
            uint8_t* d = nullptr;
            CK(cudaMalloc(&d, nb));
            CK(cudaMemcpy(d, src, nb, cudaMemcpyHostToDevice));
            w.d = d;
            m.t[ti.name] = w;
            bytes += nb;
        }
    }
    std::fprintf(stderr, "glm4ref: %s: %zu tensors (layout checked), %.2f GiB on the GPU in %.1f s\n", path.c_str(),
                 checked, (double) bytes / (1 << 30),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

struct State {
    int ctx = 2048, pos = 0;
    float *x, *h, *y, *q, *k, *v, *att, *router, *eg, *eu, *eh, *ey, *sg, *su, *sh, *sy, *wts, *logits;
    int *ids, *tok;
    std::vector<float*> kc, vc;   // per layer
};

float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}

void init_state(const Model& m, State& s) {
    const auto& g = m.g;
    const int N = (int) g.n_embd, FF = (int) g.n_ff, K = (int) m.top_k;
    s.x = dalloc(N); s.h = dalloc(N); s.y = dalloc(N);
    s.q = dalloc(g.n_head * g.head_dim); s.k = dalloc(g.n_head_kv * g.head_dim);
    s.v = dalloc(g.n_head_kv * g.head_dim); s.att = dalloc(g.n_head * g.head_dim);
    s.router = dalloc(g.n_expert);
    s.eg = dalloc((size_t) K * FF); s.eu = dalloc((size_t) K * FF); s.eh = dalloc((size_t) K * FF); s.ey = dalloc((size_t) K * N);
    s.sg = dalloc(g.n_ff_shexp); s.su = dalloc(g.n_ff_shexp); s.sh = dalloc(g.n_ff_shexp); s.sy = dalloc(N);
    s.wts = dalloc(K);
    s.logits = dalloc(g.n_vocab);
    CK(cudaMalloc(&s.ids, K * sizeof(int)));
    CK(cudaMalloc(&s.tok, sizeof(int)));
    CK(cudaMalloc(&g_q8, (size_t) 1 << 20));   // q8_1 columns: up to K * FF values of one layer
    if (m.experts_host) {
        size_t eb = 0;
        for (const auto& kv : m.h) eb = std::max(eb, kv.second.expert_bytes);
        CK(cudaMalloc(&g_exp_raw, eb));
    }
    CK(cudaStreamCreate(&g_st));
    for (int64_t l = 0; l < g.n_layers; ++l) {
        s.kc.push_back(dalloc((size_t) g.n_head_kv * s.ctx * g.head_dim));
        s.vc.push_back(dalloc((size_t) g.n_head_kv * s.ctx * g.head_dim));
    }
}

std::string L(int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; }

void embed(const Model& m, State& s, int tok) {
    const Tensor& e = m.W("token_embd.weight");
    const int N = (int) m.g.n_embd;
    CK(cudaMemcpy(s.tok, &tok, 4, cudaMemcpyHostToDevice));
    if (strata::kernels::embed_type_supported(e.type)) {
        strata::kernels::iq_embed_rows(e.type, e.d, e.row_bytes, s.tok, 1, N, s.x, g_st);
        CK(cudaGetLastError());
        return;
    }
    switch (e.type) {
    case T_F32: k_row<T_F32><<<4, 128>>>(e.d, e.row_bytes, tok, N, s.x); break;
    case T_F16: k_row<T_F16><<<4, 128>>>(e.d, e.row_bytes, tok, N, s.x); break;
    case T_BF16: k_row<T_BF16><<<4, 128>>>(e.d, e.row_bytes, tok, N, s.x); break;
    default:
        std::fprintf(stderr, "glm4ref: token_embd type %d has no row reader\n", e.type);
        std::exit(1);
    }
    CK(cudaGetLastError());
}

/// One token through the model at position s.pos; the layer outputs into `layers` (host, n_layers x N) when given.
void forward(const Model& m, State& s, int tok, float* layers) {
    const auto& g = m.g;
    const int N = (int) g.n_embd, K = (int) m.top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
    const int B = 256;
    embed(m, s, tok);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const int H = (int) g.n_head, HKV = (int) g.n_head_kv, D = (int) g.head_dim;
        k_rms<<<1, kThreads>>>(s.x, m.F(L(l, "attn_norm.weight")), s.h, N, m.eps);
        gemv(m.W(L(l, "attn_q.weight")), s.h, s.q);
        gemv(m.W(L(l, "attn_k.weight")), s.h, s.k);
        gemv(m.W(L(l, "attn_v.weight")), s.h, s.v);
        k_attn_prep<<<H + HKV, kThreads>>>(s.q, s.k, s.v, s.q, s.kc[l], s.vc[l], H, HKV, D, m.n_rot, s.pos, s.ctx,
                                           m.rope_base);
        k_attn<<<H, kThreads, (size_t) (s.pos + 1) * sizeof(float)>>>(s.q, s.kc[l], s.vc[l], s.att, H, HKV, D,
                                                                      s.pos + 1, s.ctx, 1.0f / sqrtf((float) D));
        gemv(m.W(L(l, "attn_output.weight")), s.att, s.y);
        CK(cudaGetLastError());
        k_add<<<(N + B - 1) / B, B>>>(s.x, s.y, N);
        k_rms<<<1, kThreads>>>(s.x, m.F(L(l, "post_attention_norm.weight")), s.h, N, m.eps);
        // router: logits + exp_probs_b, sigmoid, top-k, renormalised (host side: exact and simple)
        gemv(m.W(L(l, "ffn_gate_inp.weight")), s.h, s.router);
        {
            // llama.cpp: probs = sigmoid(logits); selection = probs + exp_probs_b; top-k on the selection, but the
            // weights are the UNBIASED probs of the selected experts (build_moe_ffn: "leave probs unbiased").
            std::vector<float> r((size_t) g.n_expert), b((size_t) g.n_expert);
            CK(cudaMemcpy(r.data(), s.router, r.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), m.F(L(l, "exp_probs_b.bias")), b.size() * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < r.size(); ++i) r[i] = sigm_host(r[i]);
            std::vector<float> sel(r.size());
            for (size_t i = 0; i < r.size(); ++i) sel[i] = r[i] + b[i];
            std::vector<int> idx((size_t) g.n_expert);
            for (int i = 0; i < (int) idx.size(); ++i) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                              [&](int a, int c) { return sel[a] > sel[c] || (sel[a] == sel[c] && a < c); });
            std::vector<float> w(K);
            float ws = 0;
            for (int i = 0; i < K; ++i) ws += (w[i] = r[idx[i]]);
            for (float& v : w) v = m.weights_norm ? v / ws : v * m.route_scale;
            if (m.weights_norm && m.route_scale != 1.0f)
                for (float& v : w) v *= m.route_scale;
            g_ids_host.assign(idx.begin(), idx.begin() + K);
            CK(cudaMemcpy(s.ids, idx.data(), K * sizeof(int), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(s.wts, w.data(), K * 4, cudaMemcpyHostToDevice));
        }
        if (m.experts_host) {
            gemv_host(m.H(L(l, "ffn_gate_exps.weight")), s.h, s.eg, s.ids, K, 0);
            gemv_host(m.H(L(l, "ffn_up_exps.weight")), s.h, s.eu, s.ids, K, 0);
            k_swiglu<<<(K * FF + B - 1) / B, B>>>(s.eg, s.eu, s.eh, K * FF);
            gemv_host(m.H(L(l, "ffn_down_exps.weight")), s.eh, s.ey, s.ids, K, FF);
        } else {
            gemv(m.W(L(l, "ffn_gate_exps.weight")), s.h, s.eg, s.ids, K, 0);
            gemv(m.W(L(l, "ffn_up_exps.weight")), s.h, s.eu, s.ids, K, 0);
            k_swiglu<<<(K * FF + B - 1) / B, B>>>(s.eg, s.eu, s.eh, K * FF);
            gemv(m.W(L(l, "ffn_down_exps.weight")), s.eh, s.ey, s.ids, K, FF);
        }
        // the shared expert: always on, no scalar gate
        gemv(m.W(L(l, "ffn_gate_shexp.weight")), s.h, s.sg);
        gemv(m.W(L(l, "ffn_up_shexp.weight")), s.h, s.su);
        k_swiglu<<<(FS + B - 1) / B, B>>>(s.sg, s.su, s.sh, FS);
        gemv(m.W(L(l, "ffn_down_shexp.weight")), s.sh, s.sy);
        k_moe_sum<<<(N + B - 1) / B, B>>>(s.x, s.ey, s.wts, K, s.sy, N);
        CK(cudaGetLastError());
        if (layers) CK(cudaMemcpy(layers + (size_t) l * N, s.x, (size_t) N * 4, cudaMemcpyDeviceToHost));
    }
    k_rms<<<1, kThreads>>>(s.x, m.F("output_norm.weight"), s.h, N, m.eps);
    gemv(m.W("output.weight"), s.h, s.logits);
    CK(cudaDeviceSynchronize());
    ++s.pos;
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ',' || s[i] == ' ' || s[i] == '\n' || s[i] == '\r')) ++i;
        if (i >= s.size()) break;
        v.push_back(std::atoi(s.c_str() + i));
        while (i < s.size() && s[i] != ',' && s[i] != ' ' && s[i] != '\n') ++i;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, tokens, dump_layers, dump_logits, force, dump_full;
    int n_gen = 8, ctx = 2048;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "glm4ref: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--tokens") tokens = next();
        else if (a == "--tokens-file") {
            FILE* f = std::fopen(next().c_str(), "rb");
            if (!f) { std::fprintf(stderr, "glm4ref: cannot read the tokens file\n"); return 2; }
            char buf[4096];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) tokens.append(buf, n);
            std::fclose(f);
        } else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--dump-layers") dump_layers = next();
        else if (a == "--dump-logits") dump_logits = next();
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--force") force = next();
        else {
            std::fprintf(stderr, "usage: strata-glm4ref --model GGUF --tokens ID,ID,... | --tokens-file F [--n 8] "
                                 "[--ctx 2048] [--dump-layers FILE] [--dump-logits FILE] [--force IDS] "
                                 "[--dump-full-logits FILE]\n");
            return 2;
        }
    }
    const std::vector<int> prompt = parse_ids(tokens);
    const std::vector<int> forced = parse_ids(force);
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    if (model.empty() || prompt.empty()) {
        std::fprintf(stderr, "glm4ref: --model and --tokens are required\n");
        return 2;
    }
    Model m;
    std::string err;
    try {
        if (!load(model, m, err)) {
            std::fprintf(stderr, "glm4ref: %s\n", err.c_str());
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "glm4ref: %s\n", e.what());
        return 1;
    }
    State s;
    s.ctx = ctx;
    try {
        init_state(m, s);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "glm4ref: %s\n", e.what());
        return 1;
    }
    const int N = (int) m.g.n_embd, NL = (int) m.g.n_layers, V = (int) m.g.n_vocab;
    FILE* fl = dump_layers.empty() ? nullptr : std::fopen(dump_layers.c_str(), "wb");
    FILE* fg = dump_logits.empty() ? nullptr : std::fopen(dump_logits.c_str(), "wb");
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    std::vector<float> layers((size_t) NL * N), logits((size_t) V);
    const auto t0 = std::chrono::steady_clock::now();
    int next = -1;
    std::vector<int> gen;
    const int total = (int) prompt.size() + n_gen - 1;
    if (total > ctx) {
        std::fprintf(stderr, "glm4ref: %d positions do not fit --ctx %d\n", total, ctx);
        return 2;
    }
    for (int i = 0; i < total; ++i) {
        const int tok = i < (int) prompt.size() ? prompt[i] : next;
        forward(m, s, tok, fl ? layers.data() : nullptr);
        if (fl) std::fwrite(layers.data(), 4, layers.size(), fl);
        if (i + 1 >= (int) prompt.size()) {
            CK(cudaMemcpy(logits.data(), s.logits, (size_t) V * 4, cudaMemcpyDeviceToHost));
            int a = 0, b = -1;
            for (int j = 1; j < V; ++j)
                if (logits[j] > logits[a]) a = j;
            for (int j = 0; j < V; ++j)
                if (j != a && (b < 0 || logits[j] > logits[b])) b = j;
            next = gen.size() < forced.size() ? forced[gen.size()] : a;
            gen.push_back(a);
            if (ff) std::fwrite(logits.data(), 4, logits.size(), ff);
            if (fg) {
                const float rec[4] = {(float) a, logits[a], (float) b, logits[b]};
                std::fwrite(rec, 4, 4, fg);
            }
        }
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (fl) std::fclose(fl);
    if (fg) std::fclose(fg);
    if (ff) std::fclose(ff);
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "glm4ref: %zu prompt + %zu generated positions in %.2f s (%.2f tok/s, one token at a time)\n",
                 prompt.size(), gen.size(), sec, (double) total / sec);
    return 0;
}

