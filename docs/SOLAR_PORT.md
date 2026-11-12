# Solar-Open-100B on Strata — port notes

Status: **M0 done, M1 done** (2026-11-10). Source in `D:\Strata-solar` (git baseline `e2dfa7b`, toolchain
CMake 4.4.4 + MSVC 14.44 + CUDA 13.4, Ninja build in `build-ninja`).

## Toolchain (installed)

```text
cmake  4.4.4      user winget (Kitware.CMake), not on PATH: build_*.bat add it
cl     14.44.35207 C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
nvcc   13.4.59    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4
ninja  1.13.2     user winget (Ninja-build.Ninja)
```

The CUDA MSBuild integration cannot find the toolkit (empty `CudaToolkitDir`), so the build uses the **Ninja**
generator with `vcvars64.bat`; `cmake/backends/cuda.cmake` warns and builds for sm_120 (RTX 50) only.

```powershell
cmd /c D:\Strata-solar\build_solar.bat      # configure
cmd /c D:\Strata-solar\build_strata.bat     # strata.exe
cmd /c D:\Strata-solar\build_test.bat       # model_family_test
cmd /c D:\Strata-solar\build_glm4ref.bat    # strata-glm4ref
```

Run binaries with the CUDA runtime folder on PATH:

```powershell
cmd /c "set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;%PATH% && strata.exe --families"
```

## M0 — recognition (done, commit `04369bd`)

- `ModelFamily::Glm4Moe`, `model_family_from_arch("glm4moe")`, title `Solar-Open-100B`.
- `check_architecture_glm4moe`: every geometry key present, head/KV multiple, `key_length == value_length`,
  rope an even slice of the head, `expert_gating_func == 2` (sigmoid).
- `geometry_from_gguf`: glm4moe branch (no ssm, no hc, no indexer, no MTP; `n_ff_shexp = n_ff` because the files
  carry no shared width; `expert_weights_norm`, `expert_weights_scale`).
- `layer_tensors`: glm4moe layout (no `ffn_gate_inp_shexp`, no QK norm, no output gate, `exp_probs_b.bias`).
- Tests in `tests/core/model_family_test.cpp` (synthetic tiny glm4moe + the real file via `--real`).

Real file: `model_family_test --real D:\models\solar\Solar-Open-100B.i1-IQ3_XXS.gguf`

```text
architecture guard: ok
family Solar-Open-100B: layers 48 (+0 MTP), hidden 4096, attention 64/8 x 128 every 1, MoE 128 experts top-8
  FF 1280 shared 1280, hc 0, indexer 0, vocab 196608
layout: ok (675 tensors checked)
```

## M1 — reference forward (done, commit `15e244e`)

`src/glm4/glm4ref.cu` → `strata-glm4ref`. One token at a time, fp32 activations, llama.cpp's forward as
`src/models/glm4-moe.cpp` builds it, on Strata's **i-quant MMVQ** (`iq_mmvq`: IQ2_S / IQ3_S / IQ3_XXS / Q5_K, q8_1
activations) and `iq_embed_rows` for the IQ3_S embedding.

Placement: 36.95 GiB of weights do not fit in 14.8 GiB of free VRAM, so the routed experts stay in the GGUF **mmap**
and only the selected expert's blob (2.2 MB) crosses PCIe per expert per matrix; everything else (attention,
shared expert, embedding, head ≈ 3 GiB) is resident on the GPU. Decode: **2.5 tok/s** on the RTX 5060 Ti + 64 GB
(the engine path's LRU expert cache takes this to 8.8-10.2 tok/s, see M2)
RAM (llama.cpp with `-cmoe` does 4.6 tok/s).

Two llama.cpp details the first version got wrong (both now fixed):

1. the router bias `exp_probs_b` is added **after** the sigmoid and only for the **selection**; the weights are the
   unbiased `sigmoid(logits)` of the selected experts (`build_moe_ffn` in `src/llama-graph.cpp`);
2. the shared expert has **no** scalar gate tensor in these files: it is added at weight 1.

Validation (`tools/glm4/compare_logits.py`, prompt tokens 154820,113898 = `" Nate})),"`):

```text
llama.cpp greedy: 4414 4855 4186 4394 6097 4414 4855 4186
strata greedy   : 4414 4855 4186 4394 6097 4414 4855 4186 (and 4404 next)
top-1 mismatches over 9 steps: 0
per-token log-prob deltas: mostly < 0.1, worst 0.31 (llama.cpp evaluates the prompt as a batch with MMQ/fp16
accumulation; the reference decodes one token at a time with MMVQ/fp32)
```

Note: a **1-token** prompt cannot be compared directly — llama-server prepends a token (with `-b 1` it fails with
`failed to add token 0 at index 1`), so use two or more prompt tokens.

`gen:` and `--force` follow the q36ref CLI, so the same teacher-forced KL workflow applies.

## Test model

- Path: `D:\models\solar\Solar-Open-100B.i1-IQ3_XXS.gguf`
- Architecture: `glm4moe`
- Total: 102.6B, active 12B, IQ3_XXS 36.95 GB
- 48 layers, 4096 hidden, 64 query heads / 8 KV heads, head_dim 128
- 128 routed experts, top-8, 1 shared expert
- `expert_feed_forward_length = 1280`, `leading_dense_block_count = 0`
- `expert_gating_func = 2` (sigmoid), `expert_weights_norm = true`, scale 1.0
- No QK norm, no MTP block, no indexer, no hyper-connection
- Tokenizer: GPT-2 BPE, pre-tokenizer `solar-open`

## Actual GGUF tensor layout (675 tensors)

Global:

```text
token_embd.weight                 [4096, 196608]
output.weight                     [4096, 196608]
output_norm.weight                [4096]
```

Per layer (`blk.N.`):

```text
attn_norm.weight                  [4096]
attn_q.weight                     [4096, 8192]
attn_k.weight                     [4096, 1024]
attn_v.weight                     [4096, 1024]
attn_output.weight                [8192, 4096]
post_attention_norm.weight        [4096]
ffn_gate_inp.weight               [4096, 128]
exp_probs_b.bias                  [128]
ffn_gate_exps.weight              [4096, 1280, 128]
ffn_up_exps.weight                [4096, 1280, 128]
ffn_down_exps.weight              [1280, 4096, 128]
ffn_gate_shexp.weight             [4096, 1280]
ffn_up_shexp.weight               [4096, 1280]
ffn_down_shexp.weight             [1280, 4096]
```

Forward per layer:

```text
h = rmsnorm(x, attn_norm)
q = attn_q(h); k = attn_k(h); v = attn_v(h)
q/k = rope(..., base=1e6, n_rot=128)
x = x + attn_output(attention(q, k, v))
h = rmsnorm(x, post_attention_norm)
logits = ffn_gate_inp(h)
probs = sigmoid(logits)
selection = probs + exp_probs_b      # bias selects, does not weight
top8 = topk(selection, 8)
w = probs[top8] / sum(probs[top8])   # expert_weights_norm; wizard scale 1.0
x = x + shared_ffn(h) + sum(w_i * expert_i(h))
```

No QK norm, no output gate, no dense FFN layer (`leading_dense_block_count=0`).

## M2 — engine path (done, commit after `15e244e`)

`src/glm4/glm4_engine.cu` → `strata_glm4` → `strata-glm4`; `strata` dispatches a `glm4moe` file to it at the top of
`main`, and `strata --families` now prints `qwen4exp qwen35moe glm4moe`.

What it has:

- prompt chunks (`--chunk`, default 512; T rows share one i-quant matmul per matrix and one causal attention
  kernel `k_attn_rows`), a per-layer KV cache (8 KV heads x 128, fp32, 3.2 GiB at 8192 cells), and one session's
  cache swapped in by `select()`
- sessions (`--batch` / `--slots` / `--parallel N`), KV reuse across requests (`match_len`), `copy_session` for a
  batch admission that finds a longer prefix in another session, LRU pick
- the sampler of the qwen35 engine (`sampling.hpp`: temperature / top_k / top_p / min_p / penalties), greedy
- the serve protocol of `serve/server.py`: `READY`, `PP`, `T`, `DONE`, `GEN` / `GENI`-less, `BGEN` / `BT` /
  `BDONE` / `BADM`, `BSTOP` / `BYIELD` / `STOP`; EOS is `eos_token_id` plus the `<|end|>` / `<|return|>` /
  `<|call|>` / `<|user|>` ids the file carries
- expert placement decided at load: every expert in VRAM when it fits, else the routed experts stay in the GGUF
  mmap and one blob (2.0 MB) crosses PCIe per expert per matrix; attention, shared expert, embedding and head are
  always resident
- grouped experts: the rows routed to one expert are gathered into contiguous q8_1 columns, multiplied in ONE
  `iq_mmvq` (ncols = the rows), and scattered back; a 64-row prompt chunk drops from ~1500 launches per layer to
  ~180
- an **LRU expert cache** in VRAM (`--expert-cache-mib`, default = free VRAM - 2 GiB capped at 8 GiB): a slot holds
  one (layer, expert) gate|up|down blob (5.74 MiB); a hit reads nothing over PCIe.  At 2048 cells that is 1426 slots
  of 6144 pairs; a decode run of 32 tokens hits 93.9% of them

Not ported from q36: MTP (Solar's file has none), vision / GENI, the MMQ prompt path, the hybrid CPU expert pool,
CUDA graphs.  The prompt and decode matmuls are the same `iq_mmvq` calls the reference uses, so the numbers match
it token for token:

```text
strata-glm4 --tokens 154820,113898 --n 8
gen: 4414 4855 4186 4394 6097 4414 4855 4186      (llama.cpp's greedy continuation)
chunk 1 / 2 / 3 / 8 on an 8-token prompt: the same two tokens
serve: GEN 4 ... -> T 4414 4855 4186 4394; a second GEN reusing 3 tokens reads 1
serve --batch 2: BGEN 0 / BGEN 1 decode together (BT / BDONE)
```

Measured on the RTX 5060 Ti + 64 GB RAM (IQ3_XXS, experts in the mmap):

```text
prompt 64 tokens, chunk 256: 14.8 tok/s (4.3 s; 9.96 tok/s before the grouped matmuls, 2.57 tok/s before the
expert grouping)
prompt 128 tokens, chunk 64: 15.0 tok/s (8.5 s) — a prompt scales better than decode because its rows share experts
decode without the cache:    2.7 tok/s (one token picks 8 DISTINCT experts, so every one crosses PCIe)

cache size (varied 200-token continuation, forced random ids, so the experts spread out):

  2048 cells:  4 GiB (713 slots)   91.7% hits   8.53 tok/s
               8 GiB (1426)         97.4%        13.21 tok/s
              10.3 GiB (1839, max)  98.2%        14.32 tok/s
  8192 cells:  6 GiB (1069)         95.8%        11.62 tok/s
               8 GiB (1426)         97.4%        13.12 tok/s
               8.4 GiB (1505, max)  97.6%        13.54 tok/s

The default is now free VRAM - 1 GiB (capped at 12 GiB, never less than 640 MiB free);
`--expert-cache-mib N` overrides it, `0` disables the cache.  A repetitive text run (the greedy
`\\text` loop) hits 93.9% with only 1426 slots, so the table above is the pessimistic end.

VRAM:                        2.5 GiB weights + buffers, 3.2 GiB KV at 8192 cells

IQ2_XXS (25.3 GiB; the file keeps the token embedding and layers 0-2's down projections at Q2_K, which iq_mmvq did
not take - `vec_dot_q2_K_q8_1` and `dq_q2_k` were added to iq_kernels.cu and the greedy tokens match llama.cpp's
on the same file): decode 19.9 tok/s at 2048 cells (2586 slots, 97.4%) and 18.0 at 8192 (2040 slots, 96.2%),
prompt 64 tokens 16.1 tok/s.  The gain is sub-linear because the copies are no longer the only term: ~12B active
parameters of compute do not shrink with the file.

CUDA graphs are NOT in this path, and the reason is structural: the expert weights of a step live in LRU cache slots
whose pointers change with the routing, so a captured graph's fixed kernel arguments would be wrong.  Resolving them
from a device table inside one kernel per layer needs `native_expert_grouped` (the q36 path's indirect expert kernel),
and `native_expert_supported(IQ3_XXS, IQ3_XXS, 4096, 1280)` is **false** - those kernels are built for the 2560 / 640
layout.  Even with it, the decode step's ~1650 launches cost only ~3-5 ms of a 70 ms step, so a graph is worth ~5%,
not the missing 3x.

Stage profile (STRATA_GLM4_PROF=1, IQ2_XXS, 2048 cells, 97.4% hits, 52.0 ms/token): attn 10.3, router 7.2,
gate/up 20.5 (of which ~4.6 is the 6.9 cache misses at ~6.6 GB/s), down 7.5, shared 4.6, head 1.5, embed 0.2; the
host router round-trip is 17.3 ms of host wall time per token.

The microbenchmark (tools/glm4/mmvq_bench.cpp) settled the "1 ms per expert matvec": one 4096x1280 IQ3_XXS call is
10.2 us (196 GB/s of weights) and a trivial 32x32 call is 9.4 us - **~9.4 us is pure launch latency on this WDDM
box**, so the expert path is launch-bound, not compute-bound.  Merging gate and up into ONE call (they sit back to
back in an expert-cache slot, 2*FF rows of the same row width) took the decode from 19.9 to 27.5 tok/s
(51.9 -> 36.2 ms/token).  Then the indirect expert kernel (`iq_mmvq_ptrs`: column c reads its weight matrix from a
DEVICE pointer table, and the activation column from an index table) made it ONE launch for every expert of a layer,
gate|up and down: 16 -> 2 launches per layer, decode **35.5 tok/s** (28.0 ms/token).  Greedy tokens unchanged
throughout.  Profile now: attn 7.2, router 4.4, gate/up 10.5 (of which ~4.6 is the 6.9 misses), down 2.7, shared
1.8, head 1.4 ms; the host router round-trip is 11.9 ms of wall.

Two attention kernels were tried: k_attn_rows_fast (coalesced: a warp per position, 4 warps) and k_attn_gqa (one
block per KV head, the 8 query heads as warps over one shared K/V tile: 8x less cache traffic).  NEITHER moved the
bucket (10.28 / 10.34 ms against the original 10.61): it is latency-bound in the serial position loop, not by
coalescing or KV duplication.  The split-position flash-decoding version (`k_attn_split` + `k_attn_combine`, 256 positions per block, partial
softmax then a combine kernel) was then written and measured at 1500 positions (nsplit = 6): **16.62 ms against
16.60** for the coalesced kernel, and the GQA-shared one 16.77 - none of the three moves it.  With the attention
kernel skipped (STRATA_GLM4_NOATTN=1, diagnostics only) the bucket falls from 16.6 to 6.7 ms, so the kernel itself
is ~10 ms and the other 6.7 is the q/k/v/o matmuls + RoPE prep + residual add.  Two more wins: the KV cache is now **fp16** (half the traffic and VRAM: 3.2 -> 1.6 GiB at 8192 cells, ~110 more expert-cache
slots), and the attn_q / attn_k matrices are **packed into one (H + HKV) * D row matrix** so one launch fills both
(three uses of a launch per layer less).  Short-context decode 32.1 -> **36.2 tok/s** (27.4 ms), 1500 positions
26.8 -> **29.9**, prompt-1500 prefill 63 -> **72.7 tok/s**; greedy tokens unchanged.

Cache policy: a per-layer partition (each layer a fixed slice) measured WORSE (16.9 misses/token against 13.5 on a
19-token window), and an LFU eviction with aging identical to the LRU (3.8 misses/token, 26.8 tok/s at 1500
positions over 100 tokens - the warm-up window had inflated the miss count).  Global LRU stays.  At steady state the
1500-position step is 37.4 ms: attention 16.6, experts ~13.5 (3.8 misses), router 5.5, head 1.4.

Also tried and rejected: (1) multi-row batching across serve slots - the multi-row attention kernels work for one row
but a second session sharing the GPU produced wrong tokens (not resolved, and concurrency is not a goal); (2) an
expert kernel with one column per blockIdx.y instead of a serial column loop - 2x SLOWER (22.6 against 11.7 ms for
gate/up), the per-block IQ grid staging dominates when the 8 columns split into 8x the blocks.

The hand-written per-position dot
(one warp, one position, shfl reduction) is the limit: neither more blocks nor fewer KV reads help, because the
per-position reduction latency is serial.  A real fix is a GEMM-style attention (QK^T as a batched matmul, dp4a or
tensor cores), which is a new kernel, not a variation of this one.  Default stays k_attn_rows_fast;
STRATA_GLM4_ATTN=gqa|split selects the others.

Two ideas measured and NOT adopted: double-buffered expert copies (copy stream + events) and pinned staging - the
decode is copy-bound, compute is only 5-15% of a step, so overlapping saved nothing (2.64-2.75 tok/s against 2.64-2.80);
and a CPU fallback for cache misses: the CPU path does take these experts (IQ3_XXS at 4096/1280) once
kNativeActBytes / kNativeHBytes grow from 4096 / 1024 to 8192 / 2048, but one expert measured **1744 us** here against
**500-1000 us** for the 6.0 MiB PCIe copy, so a miss is cheaper to copy than to compute.
```

## Where the decode time actually goes (IQ3_XXS, 300-token greedy run)

The model used to live on `D:`, which is a **WD30PURX, a 5400 rpm HDD**.  Random 5.74 MiB reads (one expert blob)
measure **44.6 ms median** there (p90 70 ms, min 2 ms for a page-cache hit), so a 9.4-miss token spent 43 of its 56 ms
waiting for the platter - the engine was being benchmarked against a spinning disk.  `C:` is an NVMe (Samsung
MZVL8512HELU).  IQ3 now lives in `C:\models\solar`, and `--prefetch` reads the whole routed-expert region once at
start-up so the first request does not pay cold reads (34.45 GiB in 16.2 s = 2.28 GB/s; the very first run after the
copy managed only 805 MB/s because Windows was still writing the file back).

Steady state on the NVMe with `--prefetch` (IQ3_XXS, 300 greedy tokens from a 2-token prompt, 41.1 ms/token =
24 tok/s, 3.3 expert misses/token, 3.3 GiB/s of page-cache-fed copies):

| bucket | ms/token | what it is |
|---|---|---|
| attn | 11.5 | q/k/v/o matvecs + the serial per-position attention |
| gate/up | 11.3 | 8 experts, 4096x2560 IQ3, one `iq_mmvq_ptrs` launch per layer |
| router | 8.5 | host round-trip: D2H logits, sigmoid+bias+top-8, H2D ids/weights, x48 layers |
| down | 4.9 | 8 experts, 1280x4096 |
| shared | 3.1 | the shared expert, 3 matvecs |
| head | 1.5 | 128k-vocab matvec (Q6_K, ~400 MB) |
| embed | 0.2 | |

What was tried in this round:

- **`prob_bias` was copied back from the device once per layer, per token** - a pageable D2H that bought nothing.
  Mirroring it once at load took the router bucket from 8.85 to 8.06 ms; the per-layer `std::vector` scratch in the
  route step is reused as well.
- **Pinned staging for the router round-trip** (cudaHostAlloc, truly async D2H/H2D): the router bucket drops to
  5.4 ms but gate/up rises to 14.6, because the expert matvec then waits for the id upload on the device instead of
  on the host.  Net zero, so it was reverted.  The ~8.5 ms is the round-trip itself, not the copy.
- **`cudaSetDeviceFlags`** spin / yield / blocking-sync (`STRATA_GLM4_SYNC`): auto, spin and yield are identical,
  blocking costs +5 ms.  Not the lever.
- **GEMM attention below its 640-position threshold**: at 300 positions the serial kernel still wins (attn 11.5
  against 14.3 ms), so the threshold stays.

Isolated bound for the two expert matvecs, with a working set that actually exceeds the 32 MB of L2 (8 columns drawn
from 64 distinct matrices, see `glm4_mmvq_bench`):

| call | time | bandwidth |
|---|---|---|
| IQ3 4096x2560, 8 cols (gate\|up) | 97.5 us | 329 GB/s (73% of the 448 GB/s peak) |
| IQ3 1280x4096, 8 cols (down) | 64.3 us | 250 GB/s |
| IQ2 4096x2560, 8 cols | 64.4 us | 336 GB/s |

(The same bench with 8 matrices that stay resident reported 565 GB/s - that is L2, not DRAM, which is why the numbers
above matter.)

In the engine those same two launches cost ~200 us and ~100 us per layer, i.e. **about 2x the isolated number**.  The
remaining headroom is therefore in the gaps between launches (48 layers x ~20 launches per token, plus one host
round-trip per layer), not inside the kernels: a device-side router would recover at most the 8.5 ms router bucket,
and it needs the expert-cache lookup and the miss copy on the device too in order to remove the round-trip at all.

## Expert blobs out of a file mapping are pageable, so pin them (`pin_experts`)

`cudaHostRegister` will not pin a file view: it returns `out of memory` at 1 GiB as readily as at 24, with and
without `Mapped`/`Portable`.  Anonymous memory is fine (16 GiB registers in 0.09 s).  So while the experts live in the
GGUF mapping:

- every miss copy is a **pageable** `cudaMemcpyAsync`, measured **2.4 GB/s** (2.54 ms per 5.74 MiB blob);
- a kernel cannot read the region at all, which rules out a device-side fetch.

`pin_experts()` copies the blobs once into a `cudaHostAlloc(..., Mapped)` buffer at start-up and re-points the
`HostMat`s.  With that source:

| path | bandwidth | per 5.74 MiB blob |
|---|---|---|
| pageable (file mapping) | 2.4 GB/s | 2.54 ms |
| pinned, `cudaMemcpyAsync` H2D | 22.0 GB/s | 0.27 ms |
| device kernel reading it over PCIe (`read_sum`) | 16-19 GB/s | 0.31-0.38 ms |
| kernel copy pinned -> VRAM | 14-16 GB/s | 0.38-0.43 ms |

Cost is one sequential pass over the file at start-up (37-63 s for 34.45 GiB, 8 threads; the pass `--prefetch` would
have made anyway, so `--prefetch` is skipped when pinning is on) and 34.45 GiB of RAM that is no longer reclaimable.
`STRATA_GLM4_NOPIN=1` keeps the mapping instead.  First attempts used `malloc` + copy + register: 75 s to copy and
**103 s to register**, because the pages had been trimmed to the pagefile in the meantime - `cudaHostAlloc` does the
commit and the pinning in one step and takes seconds.

## Decode after pinning (IQ3_XXS, 300 greedy tokens, 2-token prompt, 28 cores)

| configuration | ms/token | tok/s |
|---|---|---|
| mmap + `--prefetch` (warm) | 29.6 | 33.6 |
| pinned experts | **27.2-27.4** | **36.0-36.4** |
| pinned + device-side top-k | 28.4-29.0 | 34.5-35.2 |

Buckets with pinned experts (27.4 ms): attn 8.76, gate/up 7.49, router 4.26, down 3.07, shared 2.20, head 1.42,
embed 0.10.  The miss copies drop to 0.13 ms of *host* time (3.3 misses x 0.27 ms now run as real DMA on the stream),
against 4.98-7.21 ms when the source was the mapping.

`k_route_topk` (sigmoid + selection bias + top-k + renormalise, 128 threads one expert each, then thread 0 sorts 128
shared floats) produces the same greedy tokens, but for a single decode row it costs ~1.1 ms/token more than the host
did: the extra kernel launch and its position on the critical path outweigh the two 32-byte uploads it saves.  It is
kept because the next step needs the selection on the device; `STRATA_GLM4_HOSTROUTE` was *not* wired up, so the
comparison is by commit (`be7395b`-era host route + pinning = 27.2 ms).

## Next: take the host out of the per-layer loop

The router bucket is ~89 us per layer of GPU idle: the host has the ids, does the cache lookup, builds the weight
pointer table and uploads it, and the expert matvec cannot start until that lands.  Now that the expert region is
pinned *and mapped*, the whole chain can live on the device:

1. a device-side `slot_of[layer][expert]` table plus the LRU state (1868 slots, one linear scan is fine at 3.3
   misses/token);
2. `k_resolve` - slot lookup, eviction, write `d_wp_g` / `d_wp_d`, and a work list of misses;
3. `k_fetch` - copy the missing blobs straight from the mapped host region at 16 GB/s (0.35 ms per blob);
4. the host then enqueues all 48 layers without a single synchronisation and samples once per token.

The two things it has to keep consistent: hit/miss counters (read once per token) and the host's mirror of the cache,
which the prefill path (n > 1) still uses for its grouped copy-per-unique-expert - replay the device's
insert/evict log into `xc_map` once per token.  Ceiling is the ~4.3 ms router bucket, so about 27.4 -> 23.5 ms
(42 tok/s).

## The expert-cache directory moved to the device (commit after `58270ae`)

The router bucket was the last thing on the per-layer critical path that needed the host: it had the ids, did the
slot lookup, built the weight-pointer table and uploaded it, and the expert matvec could not start until that landed.
Now the whole directory is device state - `slot_of[layer][expert]`, the slot tables, the LRU clock, the counters, a
miss list - and each layer is three launches with no synchronisation at all:

| kernel | what it does |
|---|---|
| `k_route_topk` (n blocks, 32 threads) | sigmoid, + selection bias, top-8, renormalise; 128 threads compute the sigmoid, thread 0 sorts 128 shared floats |
| `k_resolve` (1 thread) | per selected expert: slot lookup, or LRU victim + directory update + miss entry; writes `d_wp_g` / `d_wp_d` |
| `k_fetch` (Kp*4 x 3 blocks) | copies each missed blob's gate / up / down straight out of the pinned region, 16 bytes at a time |
| `iq_mmvq_ptrs` | the expert matvecs, reading the pointer table on the device |

The prefill path (`n > 1`) still resolves slots on the host, because it needs one contiguous weight pointer per expert
group for the gather/scatter matmul.  So the directory is handed over for the length of a chunk: read it back at the
start (`host_cache_from_device`), hand it back at the end (`device_cache_from_host`).  Once per chunk, not per layer.

Two things this exposed:

- **Stream ordering.** `st` is created with `cudaStreamNonBlocking`, so it does *not* order against the legacy default
  stream.  A plain `cudaMemcpy` in the hand-over raced the kernels on `st` and could read or write the directory out
  of order; everything that touches this state is now `cudaMemcpyAsync(..., st)` + `cudaStreamSynchronize(st)`.
- **Verification.** `STRATA_GLM4_ROUTECHECK=1` recomputes the selection on the host from the same logits and compares
  it with `wts_i` for a whole chunk: **0 of 1,726,464 ids differ** over five chunks (48 layers x 512/476 rows each),
  at 1, 2 and 512 rows per call.  The device selection is bit-identical to the old host `partial_sort`.

### Numbers (IQ3_XXS, single stream)

| workload | before | after |
|---|---|---|
| 300 tokens from a 2-token prompt | 27.2 ms, 36.0 tok/s (pinned, host directory) | **27.2 ms, 36.2 tok/s** |
| prefill 1500 tokens | 76.6 tok/s | **82.8-88.3 tok/s** |
| 1500-token prompt, 60 generated | 51.8 ms, 18.5 tok/s | 55.2 ms, 18.3 tok/s |

Buckets at short context (27.2 ms): attn 9.6, gate/up 6.8, router 4.0, down 3.2, shared 2.2, head 1.4, embed 0.1.
The router bucket is now pure GPU time (rms + the 128-row router GEMV + the selection); the round-trip, the copies and
the host top-k are all gone, and `miss copies` reports 0.05 ms of host time because `k_fetch` runs entirely on the
device.

### Why long context did not get faster

At 1500 positions a decode step misses **49.9 experts** (87% hit rate) - the text is novel, so a token genuinely walks
through a wide slice of the 6144 (layer, expert) pairs and the cache holds 1801 of them.  That is 286 MB of expert
traffic per token out of the pinned region at ~16 GB/s, which is most of the 55 ms.  The earlier "4.7 misses/token"
readings on the same prompt were a degenerate, repetitive generation (the model looped on two tokens, so the same
experts were reused).  This is inherent to the model size against a 16 GB card, not to the engine: the only levers
are cache capacity (already at the VRAM limit) and the replacement policy.

## Latency, not bandwidth: the 4096-wide reductions (same commit)

`k_gemv_f32_rows` (the router) and `k_rms_rows` both walk a 4096-float row with 128 threads, i.e. 32 *dependent*
strided loads per thread, and the router one runs at ~24 GB/s (2 MB per layer in ~60 us).  For Solar's only width the
32 loads are now hoisted into registers and issued together, and the sum is still taken in the same order, so the
logits stay **bit-identical** (verified: the greedy output is unchanged token for token, and `ROUTECHECK` still
reports 0 of 1.7M ids different).

Router bucket 3.96 -> 3.46 ms/token, total 27.2 -> **25.9 ms/token, 38.1 tok/s**.

### Final numbers, IQ3_XXS, single stream

| workload | ms/token | tok/s |
|---|---|---|
| 300 generated from a 2-token prompt | 25.9-26.1 | 37.8-38.1 |
| the same, contexts around 1500 (novel text, 50 misses/token) | 57.5 | 17.5 |
| prefill 1500 tokens | - | 80-88 |

Buckets at short context: attn 8.9, gate/up 6.8, router 3.5, down 3.2, shared 2.2, head 1.4.

## Replacement policy at long context: LRU is already about as good as it gets

With 1801 slots over 48 layers a layer owns ~37 of its 128 experts, and a decode step reads 8 per layer, so a *global*
LRU only reaches about five tokens back - every layer competes with the other 47.  Numbers, 1500-token prompt, 60
generated, IQ3:

| policy | decode misses/token | whole run (incl. prefill) |
|---|---|---|
| LRU (default) | 49.9 | 13426 misses, 69.2% hit |
| LFU + recency tie-break, counts halved every 8192 accesses (`STRATA_GLM4_CACHE_POLICY=lfu`) | 47.4 | 14748 misses, 66.2% hit |

So frequency buys 5% of the decode misses and loses on the prefill pass (which is a single streaming sweep, where
recency is the right signal).  Not adopted; the knob is there.

`k_fetch`'s geometry does not matter either - 4, 8, 16 and 32 blocks per segment measured 35.6, 36.7, 38.7 and
40.9 ms in the gate/up bucket, i.e. more blocks is slightly worse (`STRATA_GLM4_FETCH_CHUNKS`).

What that leaves: 50 misses x 3 x 1.91 MB = 286 MB per token, and the gate/up bucket says it moves at **~6 GB/s**,
against 16 GB/s for the same copies issued back to back in `pinprobe3` and 22 GB/s for `cudaMemcpyAsync` from pinned
memory.  Between two fetches there is ~30 ms of compute, so the PCIe link idles and (presumably) leaves its
high-throughput state, and a 1.9 MB transfer is too short to ramp it back.  Nothing in the engine can fix that - the
next layer's experts are not known before the layer runs - so the practical lever for long contexts is fewer bytes per
miss: IQ2_XXS experts are 4.22 MB against IQ3's 5.74, and its 2539 slots hold a larger fraction of the 6144 pairs.

### What the fill actually costs, and IQ2 against IQ3 on identical text

`STRATA_GLM4_NOFETCH=1` skips `k_fetch` (the output is then wrong, but the buckets are not): at 1500 positions the
gate/up bucket goes 37.65 -> 7.42 ms, so the fill is **30 ms/token for 290 MB, i.e. ~9.7 GB/s**.  The same shape in
isolation (`pinprobe5`: 1868 slots of 5.74 MiB in a 10.5 GiB VRAM buffer, 12 blocks, destination scattered over the
whole cache) runs at **21 GB/s**, and inserting a 6 ms compute-shaped gap between the copies does not change that
(`pinprobe4`).  So the gap is not the link dropping to a low power state; the difference is the 34 GiB pinned source
against the probe's 2 GiB one, which is not something the engine can change.  The cache itself is also maxed out:
1868 slots (10.48 GiB) is what the driver will hand out, and asking for more is refused and backed off.

The two quants, on the **same 20 forced tokens** after a 1500-token prompt (so the routing is identical and the
comparison is not confounded by one of them degenerating):

| | IQ3_XXS | IQ2_XXS |
|---|---|---|
| slots | 1801 x 5.74 MiB | 2539 x 4.22 MiB |
| decode misses/token | 68.5 | 46.8 |
| expert traffic/token | 393 MB | 197 MB |
| ms/token | 73.4 | 53.3 |
| tok/s | 14.2 | 19.5 |

At short context they are level (IQ3 25.9-26.1 ms against IQ2 24.6 ms).  IQ2 is genuinely ~1.4x faster once the
context is long, because it misses less *and* each miss is smaller - but on this prompt IQ2's free generation collapses
into a two-token loop (`8013 4108 8013 4294 ...`) where IQ3 keeps producing new tokens, and a loop is exactly what
makes a cache look fast.  Treat IQ2's long-context lead as real but weigh it against that.

## How much quality is IQ2_XXS losing against IQ3_XXS (`--ppl`)

`--ppl` (with `--tokens-file` and an optional `--ppl-dump pos top1 nll`) runs the sequence one token per step and
accumulates the model's own cross-entropy, so two quants of the same file are measured on identical tokens with an
identical definition.  Tokenisation came from `llama-tokenize --ids --no-bos`.

400 tokens of the novel's Chinese source:

| | IQ3_XXS | IQ2_XXS | delta |
|---|---|---|---|
| perplexity | 15.98 | 23.05 | **+44%** |
| bits/token | 3.998 | 4.527 | +0.53 |
| top-1 agreement | - | 68.7% of positions | 31% disagree |

400 tokens of the Korean translation:

| | IQ3_XXS | IQ2_XXS | delta |
|---|---|---|---|
| perplexity | 29.77 | 45.34 | **+52%** |
| bits/token | 4.896 | 5.503 | +0.61 |
| top-1 agreement | - | 65.7% of positions | 34% disagree |

Cross-check with llama.cpp's own `llama-perplexity` (`-c 512`, 2048 tokens of the same Chinese text): **13.64 for IQ3
against 19.27 for IQ2, a ratio of 1.41** - against 1.44 from this engine on the same language.  The absolute values
differ only because the window is longer (4 x 512 tokens against 400), which lowers perplexity; the ratio is the
quantity being compared.

Where the damage sits (Chinese source): half of the total excess comes from **20 positions out of 399**; 17.8% of
positions cost more than +1 nat and 1.3% more than +4 nats, while the median is +0.17.  So the bulk is a mild,
uniform loss of confidence - `e^-0.5` is ~70% of the probability on the right token - and the practical failures come
from a thin tail of positions where IQ2 essentially has no idea.  That tail is what turns into the two-token loops
seen in free generation.

The formats: IQ2_XXS is 66 bytes per 256 weights (**2.0625 bpw**) and IQ3_XXS 98 bytes (**3.0625 bpw**), so IQ2 keeps
67% of IQ3's bits, and the two are 13% and 19% of bf16.  Both files are the `i1` (importance-matrix) series.

## The LRU victim search was the long-context bottleneck nobody looked at

`k_resolve` ran on one thread, and on a miss it walked all ~1900 slots of the directory - two arrays in VRAM, read one
dependent load at a time.  An event pair around each kernel (`STRATA_GLM4_PROF`) put a number on it: **215 us per
layer**, for 1.05 misses per layer at 1500 positions, i.e. **10.3 ms/token**.  More than the router, more than the
shared expert, more than the head.

The search is now the whole block's job: 128 threads each take a strided slice, keep the `(freq, used, index)`
minimum, and a shared-memory tree reduces them; the slot update stays on thread 0.  A never-used slot has `used == 0`,
so it sorts below every live slot and the "first free" rule falls out of the same comparison.

| | before | after |
|---|---|---|
| `k_resolve` | 10.32 ms/token (215 us/layer) | **0.65 ms/token (13.5 us/layer)** |
| 1500 positions, 40 generated | 57.6 ms/token, 16.1 tok/s | **50.2-51.2 ms/token, 19.8-20.1 tok/s** |
| 300 generated from a 2-token prompt | 26.2 ms, 37.6 tok/s | 26.2 ms, 37.6 tok/s (0.43 ms of it is resolve) |

Greedy output unchanged.  The same timer shows `k_fetch` at **20.2 ms/token for 305 MB = 15.1 GB/s**, against 13-21
GB/s for the identical shape in isolation (`pinprobe5`/`pinprobe6`, including a 24 GiB source), so the fill is at the
hardware rate and the earlier "9.7 GB/s" was an artefact of the no-fetch comparison.

Budget at 1500 positions now (50.2 ms): fetch 20.2 (40%), attention 15.0 (30%), gate/up matvec ~5.7, router 3.4,
down 3.2, shared 2.2, head 1.4.  The GEMM attention path is worth 4 ms over the serial one at this length (14.9
against 18.9), and its arithmetic is still two orders of magnitude off the FLOP bound, so it is where the remaining
room is; everything else is either at the DRAM rate or a fixed number of launches.

## Attention, kernel by kernel (what the 15 ms was actually made of)

Four more event pairs inside the GEMM attention path split the bucket at 1500 positions.  On the way in:

| kernel | us/layer |
|---|---|
| scores | 51.2 |
| values | 67.2 |
| softmax | 6.7 |
| combine | 5.7 |
| **the four of them** | **131** |
| the whole attn bucket | 311 |

So only 131 of the 311 us was attention arithmetic; ~96 us is the q/k/v/o matvecs (28.9 MB of weights per layer, and
they are at the DRAM rate), and ~84 us is launch gaps between the ~9 kernels this bucket contains.  That changes what
is worth doing inside the attention itself.

Two changes, both keeping the arithmetic order so the greedy output is unchanged:

- **scores**: the eight query rows of the KV head are staged in shared next to the K tile, so the inner loop touches
  shared memory only, and it accumulates four elements per step with eight independent chains (one per head) instead of
  one dependent chain.  Every step is written as an explicit `fmaf` in the original order, so the dot is bit-identical.
  51.2 -> **35.1 us/layer**.
- **values**: the probabilities are head-major, so reading eight of them per position was eight scattered loads.  The
  split's slice is ~62 positions, so the block stages it in shared once and reads it as one block.
  67.2 -> **62.3 us/layer**.

A split sweep (4/8/16/24) gives 111/87/66/62 us for values, so the 24 splits the current formula lands on is right;
raising the cap does not help (`STRATA_GLM4_ATTN_SPLITS`).

| | before | after |
|---|---|---|
| 1500 positions, 60 generated | 50.2 ms/token, 20.0 tok/s | **45.7 ms/token, 21.9 tok/s** |
| 300 generated from a 2-token prompt | 26.2 ms, 37.6 tok/s | **24.3 ms, 41.1 tok/s** |

(Both unprofiled; `STRATA_GLM4_PROF=1` itself costs ~3 tok/s because of the event records.)

What is left in the attn bucket at 1500 positions: 4.6 ms of projection weights (bandwidth), 5.2 ms of attention
arithmetic (still ~5x off its own floor, but now only 10% of the token), and ~4 ms of launch gaps.  The gaps are the
general problem now - roughly a thousand kernel launches per token - and a CUDA graph of the decode step is the way to
attack them, not more kernel work.

## The decode step is a CUDA graph

A trivial-kernel probe puts this machine's launch cost at **9.67 us** per `<<<>>>` (0.53 us per node inside a graph).
A token is ~1000 launches, so the graph looked like a 9 ms win.  In the engine it is worth 3.5 ms, not 9: the host
runs ahead of the GPU in a single stream, so most of that 9.67 us is host enqueue time that never blocks the GPU, and
only the real inter-kernel gap (~3.5 us) shows up.

For the graph to be possible every kernel argument has to be fixed at capture time:

- the **token id** and the **position** live in fixed host buffers (`tok_stage`, `pos_stage`) and are copied from
  there by nodes inside the graph;
- the position reaches the kernels through a device counter `d_pos` (`k_qk_prep_rows`, `k_attn_scores`,
  `k_attn_softmax`, `k_attn_values` all take `const int64_t*` now), so no kernel argument varies;
- grids are sized for the whole context (`att_tiles = ctx/128`) and blocks past the live position return at once;
  the value-accumulation split count is derived from the position *inside* the kernel, so the split boundaries are the
  ones the position-dependent formula produced - the arithmetic, and therefore the greedy output, is unchanged;
- the graph is captured on the first decode step (`warmup`) and re-captured when the session changes; prefill
  (n > 1) is untouched, as is `STRATA_GLM4_NOGRAPH=1`, `STRATA_GLM4_PROF=1` or `STRATA_GLM4_DEBUG=1`.

| IQ3_XXS, single stream | no graph | graph |
|---|---|---|
| 300 generated from a 2-token prompt | 40.8 tok/s (24.5 ms) | **45.8 tok/s (21.9 ms)** |
| 1500-token prompt, 60 generated | 22.1 tok/s (45.2 ms) | **24.0 tok/s (41.7 ms)** |
| 1500-token prompt, 100 generated | 24.0 tok/s | **26.2 tok/s** |
| prefill 1500 tokens | 81 tok/s | 81 tok/s |

The 100-token sequence at 1500 positions is identical character for character between the two modes, and the 2-token
prompt still produces `4414 4855 4186 4394 6097 4414 4855 4186`, so the capture path is verified by the same check as
everything else.  `--serve` works through the graph too (`GEN 8 ...` returns the same 8 ids).

**A bug this found:** the position copy was added at the top of `forward` instead of the body, so the prefill path
(n > 1, which goes through the body directly) used a stale `pos_stage = 0` and wrote every prompt row's K/V at
position 0..n-1.  It showed up only as a different greedy sequence at 1500 positions - the 2-token check still passed -
which is exactly why that sequence is worth keeping in the tests.

## Two latency chains, 4.6 ms/token

With the graph in place the profiler's per-kernel timers point at two kernels whose cost had nothing to do with
arithmetic and everything to do with *dependent loads*:

- **`k_attn_values`**: the position loop was one global load of `V[p][d]` per iteration, and consecutive positions are
  256 bytes apart, so 62 positions in a row cost 62 memory latencies.  67 -> **21.6 us/layer** by keeping eight
  positions in flight (the accumulation order is unchanged, one `fmaf` at a time in ascending position order, so the
  output is bit-identical).
- **`k_route_topk`**: the best-eight selection lived in a dynamically indexed `int[32]`/`float[32]`, which the compiler
  places in **local memory**; a single thread doing ~1100 dependent local accesses cost **48.5 us/layer**, more than
  rms (`15`), the router GEMV (`10`) and the whole attention put together.  Slots are now registers filled with
  `-inf`/`INT_MAX`, so a new entry drops into the last slot and bubbles up with compile-time indices: 48.5 ->
  **12.7 us/layer**.

| IQ3_XXS, single stream, unprofiled | before | after |
|---|---|---|
| 300 generated from a 2-token prompt | 45.8 tok/s (21.9 ms) | **51.8 tok/s (19.3 ms)** |
| 1500-token prompt, 100 generated | 26.2 tok/s (38.2 ms) | **29.2 tok/s (34.3 ms)** |
| prefill 1500 tokens | 81 tok/s | 81 tok/s |

The 100-token sequence at 1500 positions is unchanged.

Note for future measurement: with 14 event records per layer the profiler itself costs ~5 us per record, so the
per-kernel numbers it reports are inflated by a few microseconds each and the profiled total is ~15% above the real
one.  Compare configurations with the profiler on (both inflated equally), quote absolute numbers with it off.

## Tried and rejected: overlapping the down third of the fill (STRATA_GLM4_SPLITFETCH)

The down projection is not needed until after the SwiGLU, so its third of the per-layer fill (~2 MB of the ~6) was
given its own stream, forked from the main one after `k_resolve` and joined just before the down matvec.  Measured at
1500 positions, 100 generated (ms/token):

| fetch | with graph | without graph |
|---|---|---|
| single stream | 34.3 | 38.7 |
| down third on its own stream | 34.4 | 38.9 |

A wash, in both modes.  Two reasons, and they are the same reason: the bytes are the same, so the link sets the floor,
and the fill is *latency*-bound per launch (one miss per layer means 12 working blocks), so halving the blocks per
stream slows each half down by about what the overlap saves.  There is no window to hide it in.  The code stays behind
`STRATA_GLM4_SPLITFETCH=1`.

## Where decode stands, and why it is done

At 1500 positions a decode step is 34.3 ms, and the budget is:

| | ms/token | share |
|---|---|---|
| expert fill (287 MB at 16.2 GB/s) | 17.7 | **52%** |
| attention (4 kernels) | 3.5 | 10% |
| projections (q/k/v/o + expert matvecs) | 4.5 | 13% |
| down / swiglu / quant | 3.0 | 9% |
| shared expert | 2.2 | 6% |
| head | 1.4 | 4% |
| router | 1.2 | 3% |
| cache resolve | 0.7 | 2% |

Everything except the fill is at the DRAM rate or a fixed amount of work, and the fill is at the PCIe rate for this
machine (16.2 GB/s in the engine, 20-21 GB/s for an isolated contiguous copy, 22 GB/s for `cudaMemcpyAsync` out of
pinned memory).  Two independent attempts to make it cheaper - more blocks per segment, and a second stream - both
came out neutral or worse.  The only remaining lever on the fill is *fewer bytes*: a 2.5 bpw quant would cut it 18%
and hold ~25% more slots, at a quality cost somewhere between IQ3_XXS and the IQ2_XXS that loops on this prompt.

So decode is at its practical floor: **51.8 tok/s at short context, 29.2 at 1500 positions** (IQ3_XXS, one stream).
The remaining work is the prompt pass, which is compute-bound at ~1.1 TFLOP/s and would need a real MMQ-style kernel
(block-tiled int8 dot products) rather than a tweak.

## The prompt pass was launching one kernel per column (`mm`)

Profiling the prefill (the profiler now covers `n > 1` as well) put the prompt pass at 750 tokens into perspective:
attention 36%, routed gate/up 32%, shared expert 14%, down 11%, embed 8%.  Not the expert matvecs at all.

The culprit was `mm()`:

```cpp
const size_t qb = quant(xin, w.n_in, cols, q8);
for (int c = 0; c < cols; ++c)
    iq_mmvq(w.type, w.d, (const char*) q8 + c * qb, out + c * w.n_out, w.n_in, w.n_out, 1, st);
```

One launch **per column**.  Every dense matmul in the prompt pass - q, k, v, o and the three shared-expert matrices -
went through it, so a 750-token chunk issued ~100k launches at ~9.7 us each.  `iq_mmvq` takes `ncols` and its
multi-column kernel is bitwise equal per column, so it is one call:

```cpp
quant(xin, w.n_in, cols, q8);
iq_mmvq(w.type, w.d, q8, out, w.n_in, w.n_out, cols, st);
```

| prefill, IQ3_XXS | before | after |
|---|---|---|
| 750 tokens, chunk 2048 | 8.88 s (84 tok/s) | **6.38 s (118 tok/s)** |
| 1500 tokens, chunk 512 | 19.25 s (78 tok/s) | **13.65 s (110 tok/s)** |
| 1500 tokens, chunk 2048 | 16.02 s (94 tok/s) | **11.47 s (131 tok/s)** |

attention bucket 3.23 -> 1.54 s, shared expert 1.28 -> 0.31 s, and the 100-token generation at 1500 positions is
unchanged character for character.  Decode is untouched (there `cols == 1`).

Chapter-sized job now: 11.5 s of prompt + 24 s for 700 generated tokens, ~35.5 s (was 42.5 s).

What is left in the prompt pass afterwards: routed gate/up 2.8 s (2.2x off the same kernel's isolated 2.4 TMAC/s),
attention 1.5 s, down 0.95 s, embed 0.72 s.  The embed is 0.95 ms per token for a 1.3 MB table read, which is three
orders of magnitude off and worth a look before any MMQ work.

## The prompt pass was copying and computing one after the other

With `mm()` fixed, the prompt pass at 750 tokens split into expert fill ~1.4 s and expert matvec ~1.3 s, strictly in
series: per group the code filled the cache slot and then ran the matvec, both on the same stream.

The fills are now issued for the whole layer up front on the second stream (`ev_grp`: one fork event after the
previous layer's matvecs - the fills overwrite slots those were reading - and one join event per group), so a group's
matvec starts as soon as *its* fill lands:

```text
for each group:  st2: fill slot, record ev_grp[g]
for each group:  st:  wait ev_grp[g], gather, matvec, scatter
```

| prefill, IQ3_XXS | serial | pipelined |
|---|---|---|
| 750 tokens, chunk 2048 | 6.27 s (120 tok/s) | **5.07 s (148 tok/s)** |
| 1500 tokens, chunk 2048 | 11.47 s (131 tok/s) | **9.87 s (152 tok/s)** |
| 1500 tokens, chunk 512 | 13.65 s (110 tok/s) | **10.20 s (147 tok/s)** |

gate/up bucket 2.79 -> 1.60 s.  The 100-token generation at 1500 positions is unchanged character for character.
With the fill hidden, chunk 512 now wins over 2048 again (its decode keeps 1816 slots against 1630), so the default
stays: 10.2 s of prompt + 24.5 s for 700 generated tokens, **~34.7 s a chapter** (42.5 s before this whole round,
47.6 s at the start of the session).

Remaining prompt-pass budget (750 tokens, 5.25 s): attention 1.53 s, gate/up 1.60 s (the matvec alone now), down
0.95 s, shared 0.31 s, and an embed bucket that swings 0.17-0.80 s between runs (cold start, not a real cost).

## The prompt attention is the QK dot, and it needs a tiled kernel

`STRATA_GLM4_ATTN_SKIP=1` stops the serial prompt-attention kernel right after the scores, which splits it:

| 750-token chunk | ms |
|---|---|
| scores (QK) | 1433 |
| softmax + values (AV) | 120 |
| the kernel | 1553 |

So 92% is the QK dot, and two attempts to speed it up did nothing at all:

- four positions per warp iteration, so four independent shuffle reductions overlap instead of one per position
  (was one dependent chain of five shuffles per position): 1558 -> 1557 ms;
- two positions in flight in the value loop (same structure that gave 3x in `k_attn_values` for decode): 1558 -> 1552.

Neither moved it, and the reason is scale: the QK dot here is 26 instructions per warp per position (4 half loads,
4 int->float converts, 4 fma, 5 shuffles, plus loop overhead) against 4 MACs of useful work, and it is *instruction*
bound rather than latency or bandwidth bound, so more in-flight work changes nothing.  The shapes say the fix is the
same one that already worked for the decode attention - a tile of query rows against a tile of positions, query in
registers and the K tile in shared, so each K element feeds 16+ fma instead of 4 - i.e. a real flash-style prompt
attention kernel, not a tweak.

It is worth doing for long prompts specifically, because the cost is quadratic: at 750 tokens the prompt attention is
1.55 s of a 4.9 s pass, and at 1500 it is ~6 s of the 10.2 s pass (57%).  A 4x kernel would take the chapter job from
34.7 s to about 30 s.

Current state, IQ3_XXS, one stream, unprofiled:

| | |
|---|---|
| prompt 1500 tokens | 10.2 s (147 tok/s) |
| decode, short context | 51.8 tok/s |
| decode, 1500 positions | 28.8 tok/s |
| chapter (1500 in, 700 out) | ~34.7 s |

### Third attempt on the QK loop: the clamp was costing 7% (but no more)

The 4-position groups were addressed through `p0 + j < n_pos ? p0 + j : n_pos - 1`, which keeps the compiler from
issuing the sixteen K loads together.  Addressing full groups unconditionally (with a scalar tail for the last one or
two positions per warp) is worth:

| 750-token prompt | before | after |
|---|---|---|
| prompt attention | 1553 ms | **1440 ms** |
| whole prompt pass | 4.87 s (154 tok/s) | **4.39 s (171 tok/s)** |
| 1500-token prompt pass | 10.21 s | **10.01 s (150 tok/s)** |

Both test sequences (the 100-token generation at 1500 positions and the 2-token one) are unchanged.

That is the third structural attempt on this loop - four positions per warp iteration, two positions in flight in the
value loop, and now conditional-free addressing - and the three together moved the QK step by 7%.  The arithmetic says
why more of this will not work: the prompt QK is 110 GMAC over a 1500-token prompt and costs ~6 s, i.e. **76 GMAC/s**,
while the card's 4608 lanes at 1.9 GHz can issue 8.75 G lane-cycles/s of work.  Every lane-iteration is therefore
waiting ~460 cycles, and with 32 warps resident an SM still cannot cover it - the K rows are being re-read eight times
(once per query head that shares a KV head) out of 3 MB per layer that the expert traffic keeps flushing from L2.

The fix that follows from that is the one that worked for decode: **one block per (KV head, query row) computing all
eight query heads together**, so K and V are read once per row instead of eight times, plus a query tile in registers
and the K/V tiles in shared (flash structure, online softmax).  4x less traffic on a stage that is 57% of the prompt
pass at 1500 tokens; it needs a new kernel of ~150-200 lines rather than another tweak of this one.

### Tried and rejected: all eight heads of a KV head in one block, scores in shared

The obvious next step from the analysis - one block per (KV head, row) computing all eight query heads, so K and V
are read once per row instead of eight times - was implemented and measured **2.3x slower**: prompt attention
1440 -> 3585 ms, prompt pass 4.39 -> 6.87 s at 750 tokens (output sequence unchanged, so it was functionally right).

The reason is what it costs in occupancy. Holding the row's scores is an `[8][n_pos]` shared array (24 KB at 750
positions, 48 KB at 1500) and the cross-warp value partials are another 16 KB, so the block asks for ~40 KB of shared
and the SM can host two of them; the kernel it replaced asks for 3 KB. For a stage that is latency-bound 32 warps of
occupancy matter more than 8x less traffic, and the three passes over the positions (scores+max, sum of exp, values)
did not help either.

So the traffic analysis still says the right structure is flash decoding's: 64-position tiles, online softmax, the
running max/sum/accumulators in registers only - **no per-position score array at all**, which keeps the shared
footprint near zero and the occupancy high. That is a standalone kernel, and it is the one piece of the prompt pass
left on the table (~10% of the chapter job).

Final state of the engine for this session, IQ3_XXS, one stream, unprofiled:

| | |
|---|---|
| prompt 1500 tokens | 10.0-10.5 s (142-150 tok/s) |
| prompt 750 tokens | 4.39 s (171 tok/s) |
| decode, short context | 51.8 tok/s |
| decode, 1500 positions | 28.8 tok/s |
| chapter (1500 in, 700 out) | ~34.5 s |
| decode with IQ2_XXS for comparison | same short, 1.4x faster at 1500 positions for +0.45 bits/token |

## Expert-major pinned layout: one transfer per miss instead of three

`pin_experts` used to reproduce the file's tensor-major layout (all gate blobs, then all up, then all down), so a
cache miss was three ~1.9 MiB `cudaMemcpyAsync` calls from three places.  It now packs gate | up | down of a
(layer, expert) back to back with the cache slot's own stride, so a miss is **one ~5.7 MiB contiguous run** - the shape
that measures 20-21 GB/s instead of 16.2.  `HostMat` grew a `stride` field (0 = the mmap layout) and the fill uses it.

| IQ3_XXS, unprofiled | before | after |
|---|---|---|
| decode, 1500 positions | 28.8 tok/s (34.7 ms) | **30.4 tok/s (32.9 ms)** |
| decode, short context | 51.8 tok/s | 52.1 tok/s |
| prompt 1500 tokens | 10.2 s | 10.3 s |
| chapter (1500 in, 700 out) | ~34.5 s | **~33.3 s** |

The fill itself went 17.7 -> 13.2 ms/token for the same volume, confirming the transfer-shape theory; the token only
gained 6% rather than the 13% the bucket numbers suggested, because the profiled buckets at 1500 positions are
inflated ~4x by their own 14 event records per layer - read the profiled numbers only against each other.

Remaining decode budget at 1500 positions (32.9 ms): fill ~13, attention ~3.5, projections ~4.5, down/misc ~3,
shared expert ~1.5, head 1.4, router ~1.2, resolve 0.7.  The fill is now at the link's measured rate; what is left is
the prompt attention (flash kernel) and the attention/projection bandwidth, all at their limits or needing new kernels.

## Flash prompt attention: 7-10% on the stage, a wash end to end

`k_attn_flash` is the flash form the analysis asked for - 64-position tiles, online softmax, the running max/sum/value
accumulators in registers and 9 KB of shared, **no per-position score array** - with four of the eight query heads per
block.  It lands, but barely:

| 750-token prompt | per-head kernel | flash |
|---|---|---|
| prompt attention | 1527-1582 ms | **1429 ms** |
| whole prompt pass | 4.95 s | 4.89 s |

and at 1500 tokens the pass is 9.93-10.22 s either way (the sequence `4372 4661 40820 ...` is unchanged, so the online
softmax is numerically equivalent in practice).  Eight heads per block was slightly *worse* than four (1461 against
1429) because the cross-warp partials double to 16 KB and the registers grow - the same occupancy trade-off that sank
the previous attempt, just less severely.

The lesson is the one the instruction count already predicted: the QK step is bound by issue slots (a warp needs ~17
instructions per position for 4 MACs), and sharing K across heads only takes that to ~13 MACs per instruction, a 1.7x
ceiling.  Measured 1.07x.  The other two items on the list - merging the shared expert's gate|up into one matvec and
fusing the router's three launches - are worth even less: both are bandwidth- or launch-bound at a scale where the
bytes and the launch count do not change (<= 0.5%), so they were not done.

Engine at the end of the session (IQ3_XXS, one stream, unprofiled):

| | |
|---|---|
| prompt 1500 tokens | 9.9-10.2 s (147-151 tok/s) |
| prompt 750 tokens | 4.89 s (153 tok/s) |
| decode, short context | 52.1 tok/s |
| decode, 1500 positions | 29.3-30.4 tok/s |
| chapter (1500 in, 700 out) | ~33.3 s |

## Running it as a service (the OpenAI API path)

The engine takes token ids and gives token ids back; `serve/server.py` is the HTTP, tokenizer, template and
stop-token layer around it.  Five things had to exist before a client could use Solar at all:

* **The tokenizer pack.** `python tools/strata_tokenizer.py --gguf <solar.gguf> --out <dir> --check` writes the five
  files (vocab, merges, token types, `tokenizer.json`, the GGUF's own chat template).  It refused the file before,
  because `tokenizer.ggml.pre` is `solar-open` and llama.cpp gives that case the QWEN35 pattern character for
  character (`LLAMA_VOCAB_PRE_TYPE_SOLAR_OPEN`, src/llama-vocab.cpp L393, next to QWEN35 at L396);
  `pre_tokenizer_kind` now maps it to the same rule.  Checked against `llama-tokenize` on Chinese, Korean, mixed
  CJK/emoji and code: 64/33/35/18 ids, every one equal, and a detokenise round trip of all four.
* **The template's globals.** Solar's template calls `strftime_now`, which llama.cpp provides and the server did
  not: every request failed with `'strftime_now' is undefined` until it was added next to `raise_exception`.
* **The message markers.** A reply opens with `<|content|>`, which the model writes itself; a control token's bytes
  are its literal spelling, so it came out in front of every answer.  `<|content|>`, `<|flush|>` and `<|think|>` are
  dropped from the text now.
* **The end of a turn.** With a one-line prompt the model was measured to finish and then open the next message
  (`<|flush|><|begin|>...`) instead of emitting `<|end|>`; with the real translation prompt it does emit `<|end|>`
  at the end.  The service stops on the family's end-of-turn ids and on the next `<|begin|>`, so both behaviours end
  the reply.
* **`reasoning_effort`.** The config's `"reasoning_effort": "minimal"` reaches the template as it is (the family has
  its own effort table; `minimal`/`low` also set the server's "the reply starts with the answer" flag, or the whole
  translation is reported as `reasoning_content` with an empty `content`).

Rebuild `strata` (not just `strata-glm4`) before pointing the server at it: the `strata.exe` on this machine was
three weeks old, predated the expert cache, and ran the chapter at 3.5 tok/s with no `pinned`/`expert cache` lines.

Measured through HTTP (`solar-iq3_xxs.json`, port 8848, `run-solar.bat`), one un-streamed request of chapter 21:

| | |
|---|---|
| prompt | 2,502 tokens in 16.4 s (152.3 tok/s) |
| generated | 1,587 tokens in 46.8 s (33.9 tok/s), `finish_reason: stop` |
| whole request | 63.3 s |
| special-token leaks in the text | none |

The text is the same as the CLI path's, including its two known prompt-level faults (a closing `---` /
`*번역 규칙 준수...` paragraph, and `즉시` where the glossary forbids it); those live in the system prompt, not in
the engine.

### Every request past 8K tokens died on its first generated token (fixed, `bdf11938`)

First real 16K use, in an agent loop: the engine exited with `cudaMemcpyAsync(...) failed: an illegal memory access
was encountered`, or (without the graph) `quantize_q8_1_rows` / `cudaGraphLaunch` instead - whichever async launch
the next sync caught. `STRATA_GLM4_DEBUG=1` (which syncs after each stage) put the fault in layer 0's attention on
the **first decode step**: `forward n=1 pos0=8268` printed, `layer 0 attn` printed, then the sync in the next
`dbgstep`.

The decode value pass (`k_attn_values`) stages a split's softmax row in `__shared__ float ps[8][256]`.  A split is
`ceil(n_pos / nsplits)` long and `nsplits` defaults to 32, so 256 slots hold exactly 8,192 positions; at 8,193 the
staging loop wrote `ps[7][257]` and the kernel took the fault.  `STRATA_GLM4_ATTN_SPLITS=64` (130 slots at 8,269)
made the crash disappear, which confirmed it.

The staging now runs in windows of 256 positions with the accumulation order unchanged (ascending positions, eight
in flight), so the arithmetic is bit-identical and shared use stays 8 KiB.  Tested on one engine: prompts of
8,268 / 9,948 / 12,468 tokens decode at 17.8 / 17.5 / 16.2 tok/s, and a repeated 8,268-token prompt still reuses
8,255 tokens of KV (0.3 s for the 13-token read).  The same shared tile is the only instance of the pattern in the
tree.

### What it costs: 16K against 32K (the deployment is 32K now)

An agent loop puts 3.4K of system prompt and tool definitions in front of every turn and reserves 4-8K for the
answer, so a 16K window leaves about 6K of conversation: OpenCode compacted three turns into a folder check.  32K
was measured on the same card (RTX 5060 Ti 16 GB, IQ3_XXS):

| | 16K | 32K |
|---|---|---|
| expert cache | 1,382 slots = 7.75 GiB | 846 slots = 4.74 GiB |
| KV cache | 3.2 GiB | 6.4 GiB |
| prefill | 150 tok/s (2.2K prompt) | 109 tok/s (12K prompt), 66 tok/s (6K increment past 12K) |
| decode at 12-24K positions | - | 12.9 / 14.5 / 13.5 tok/s |
| decode, short prompt | 23-26 tok/s | 19.4 tok/s |
| KV reuse | - | 17,915 of 17,928 tokens in 0.6 s |

The decode drop is the cache, not the context: doubling the KV takes 3.2 GiB away from the experts, and the hit rate
goes from about 90% to about 70%.  A turn that only adds a tool result still prefills a few hundred tokens (2-3 s),
which is what keeps the loop usable; the first turn of a session pays the whole 3.4K system prompt.

## Architectures and backends

`cmake/backends/cuda.cmake` defaults a hand build to `120` and warns about it: that is the RTX 50 card this engine
was developed on, not a support statement.  The release engine is built for `75;86;89;120`
(`packaging/build_cuda_linux.sh`, the `cuda13` variant).  A hand build names its own card:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
          -DCMAKE_CUDA_ARCHITECTURES="86;120"          # or "native", or 75 / 89

Solar-Open-100B (`glm4moe`) compiles for all four.  A build of `strata` with
`-DCMAKE_CUDA_ARCHITECTURES="75;86;89;120"` produced cubins for `sm_75`, `sm_86`, `sm_89` and `sm_120`
(`cuobjdump -lelf`), and that binary reproduces both verification strings on the RTX 5060 Ti.  The packaging
configuration builds the same way: `-DSTRATA_PORTABLE=ON` with
`"75-real;86-real;89-real;120-real;120-virtual"` (the shape `packaging/build_cuda_linux.sh` uses) links, gives the
same four cubins plus PTX for 120, and reproduces the 1,500-token string as well.

Both strings, from the release-parity binary:

| prompt | expected | got |
|---|---|---|
| 2 tokens | `4414 4855 4186 4394 6097 4414 4855 4186` | same |
| 1,500 tokens | `4372 4661 40820 8723 5548 50468 4316 4113` | same |

The 1,500-token prompt ran at 9.57 s (156.7 tok/s), the same range as the sm_120-only build, so the extra
architectures cost nothing at run time.  Nothing in the family's kernels is arch-gated: it uses warp shuffles and
half-precision math, both there since sm_70, and the loader converts BF16 tensors to F32 in software.

The family's targets are CUDA-only (`cmake/strata_gpu_engine.cmake`).  The decode path keeps the 34.45 GiB of routed
experts in mapped pinned host memory and copies a miss into a VRAM LRU from inside a captured graph; the HIP compat
layer maps the symbols, but that path has never been run on an AMD card, and a build that linked it would answer AMD
users with wrong experts.  A HIP build therefore does not link it: `strata --families` lists only the families that
build runs, and a glm4moe file is refused by name ("Solar-Open-100B (architecture 'glm4moe') is recognised, but
this build does not run it yet") instead of producing an answer.

What is verified, and what a third party should know before downloading 37 GB:

| | |
|---|---|
| GPU | sm_75+ compiles (75/86/89/120 in one binary).  Measured on 16 GB: 1,650-1,851 cache slots, 90-93% hit rate.  8 GB would run with roughly half the slots - estimate, not measured |
| RAM | 48 GB to pin the experts; below that `cudaHostAlloc` fails, `pin_experts` keeps the mmap and every miss becomes a disk read (logged, not fatal) |
| storage | the GGUF on an NVMe.  A 5.74 MiB random read measured 44.6 ms on a 5400 rpm HDD, which by itself makes the model unusable |
| OS | the build scripts here are Windows (MSVC + Ninja); the CMake targets are not Windows-only, but this port has never been compiled on Linux |
| whole chapter | 2,502 in / 1,587 out in 63.3 s through the HTTP server (16 GB card, 64 GB RAM, NVMe) |
