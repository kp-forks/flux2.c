# MPS Speed Optimization Log

## Standing Instructions
- **Continue tirelessly without asking for user prompt** — keep optimizing in a loop
- Commit each time a good improvement is reached
- Complexity increase must be justified by speed results
- Re-read this file at each context compaction
- Take notes on all results, what worked, what didn't

## Testing
- **Quick iteration**: use 256x256 with `--seed 42 -v` for timing measurements
- **Before committing**: run `make test` to verify no regressions
- **Benchmark command**:
  ```bash
  ./flux -d flux-klein-model -p "A woman wearing sunglasses" -o /tmp/bench.png -W 256 -H 256 -v --seed 42
  ./flux -d flux-klein-model -p "A woman wearing sunglasses" -o /tmp/bench.png -W 512 -H 512 -v --seed 42
  ```

## Pipeline
```
1. Text Encoding:    prompt -> Qwen3 4B (36 layers) -> [512, 7680] embeddings
2. Latent Init:      random noise [H/16, W/16, 128]
3. Denoising Loop (4 steps):
   per step: 5 double blocks -> 20 single blocks -> final layer -> velocity
4. VAE Decode:       latents -> VAE decoder -> RGB image
```

## Current Baseline (2026-02-06 / MacBook Pro M3 Max 40-core GPU, 128 GB, 400 GB/s)

### 256x256 (seq=256+512=768 tokens)
- Text encoding: 1.0s (Qwen3, GPU-resident forward, was 1.9s)
- Denoising total: 2073 ms (4 steps)
  - Step 1: ~570 ms, Steps 2-4: ~500 ms each
- VAE decode: 0.4s
- Transformer loading: 1.3s (includes bf16 weight cache warmup)
- **Total: ~5.4s**

### 512x512 (seq=1024+512=1536 tokens)
- Text encoding: 1.0s
- Denoising total: 4004 ms (4 steps)
  - Step 1: ~1058 ms, Steps 2-4: ~980 ms each
- VAE decode: 1.6s
- **Total: ~8.6s**

### Key observations
- Monolithic GPU batch: 1 command buffer per step (all 25 blocks + concat + slice + final)
- Step 1 ~15% slower than subsequent steps (residual MPS warmup)
- Matmul compute dominates (~4.5 TFLOPS for these dimensions)

## Already Optimized
- Batched GPU ops within each block (batch_begin/batch_end)
- Fused QKV+MLP projection in single blocks
- Fused bf16 attention kernel (seq <= 1024)
- bf16 MPS attention fallback (seq > 1024)
- Pre-warm bf16->f16 weight cache
- Persistent GPU tensors
- SwiGLU fused on GPU

## Optimization Attempts

### Attempt 1: Pre-warm bf16 weight buffer cache (SUCCESS)
- In mmap mode, first denoising step paid ~800ms overhead to copy ~7GB of bf16 weight data
  from mmap'd safetensors to Metal GPU buffers (via `get_cached_bf16_buffer`)
- Moved cache population to model loading (`warmup_mmap_bf16_buffers()`)
- Loads each block's bf16 mmap pointers, copies weight data to Metal buffers, frees f32 weights
- 113 cache entries: 5 double blocks × 14 weights + 20 single blocks × 2 weights + 3 input/output
- Loading time: 0.2s → 1.3s (+1.1s for weight cache warmup)
- **Result: 256x256 denoising 2822 → 2172ms (23% faster), 512x512 4420 → 4146ms (6% faster)**
- Step 1 overhead: 256x256 781ms → 124ms (84% less), 512x512 354ms → 123ms (65% less)

### Attempt 1b: MPSGraph JIT pre-warming (FAILED)
- Tried pre-warming MPSGraph JIT compilation by running dummy matmuls with all dimension tuples
- Created graphs for 9 linear ops + 3 SDPA ops per resolution, allocated dummy Metal buffers
- Total JIT warmup: only ~80ms (MPSGraph compiles fast on M3 Max)
- **Result: no improvement — JIT compilation was not the bottleneck. Reverted.**

### Attempt 1c: Pre-compute QK norm GPU tensors (FAILED)
- Tried pre-computing norm_q/norm_k bf16 GPU tensors at weight load time
- Eliminates 60 `bf16_tensor_from_f32` calls per step (20 single × 2 + 5 double × 4)
- Each call converts 128 floats (512 bytes) — tiny tensors
- **Result: no measurable improvement. Within batch, the per-dispatch overhead is negligible. Reverted.**

### Attempt 2: Monolithic GPU batch (SUCCESS)
- Previously: 5 separate batch_begin/batch_end per step (double blocks → concat → single blocks → slice → final)
- Each batch_end = [cmd commit] + [waitUntilCompleted] = GPU pipeline flush + CPU-GPU sync
- Key insight: ALL CPU modulation computations depend only on t_emb_silu + fixed weights, NOT on GPU results
- Precompute all modulation parameters at the start (double, single, final), pre-allocate all buffers
- Then run EVERYTHING in a single command buffer: double blocks + concat + single blocks + slice + final + bf16→f32
- Command buffer round-trips per step: 5 → 1 (eliminates 4 waitUntilCompleted syncs)
- Per-stage timing breakdown removed for bf16 path (all work in one batch, can't measure stages)
- **Result: 256x256 denoising 2172 → 2073ms (4.6% faster), 512x512 4146 → 4004ms (3.4% faster)**
- **Cumulative: 256x256 2822 → 2073ms (27% faster), 512x512 4420 → 4004ms (9% faster)**

### Attempt 3: Double block scratch pool (FAILED)
- Pre-allocated 20 GPU tensors per double block (norm, QKV, proj, FFN) as persistent scratch
- Reused across 5 double block iterations, avoiding alloc/free per block
- Added `flux_gpu_linear_bf16_native_into()` to write into pre-allocated buffers
- **Result: 2046-2069ms vs 2073ms baseline — within noise. Reverted.**
- Metal buffer pool already makes alloc/free fast; within monolithic batch, no overhead.

### Attempt 4: Merged QKV + gate/up weight matrices (FAILED)
- Merged Q+K+V into single [3*hidden, hidden] matmul, split with `split_3_bf16` shader
- Merged gate+up into single [2*mlp_hidden, hidden] matmul, fused `silu_mul_merged_bf16` shader
- Merged weights created at warmup time, stored persistently in `double_block_t`
- Reduces 103 → 73 MPS matmul encodes per step (30 fewer, 29% reduction)
- Tests pass (3/3), correct output
- **Result: 256x256 2064-2092ms (no improvement), 512x512 4042-4072ms (no improvement). Reverted.**
- Root cause: with monolithic GPU batch, GPU compute dominates (~500ms/step).
  CPU encoding of all 103 matmuls completes in ~30ms while GPU executes ~500ms.
  Reducing CPU encoding from 30ms to 22ms has zero observable effect.
- **Key insight**: Once a monolithic command buffer is in place, reducing the NUMBER
  of GPU operations doesn't help. Only reducing total GPU COMPUTE TIME matters.

### Analysis: Transformer at GPU compute limit
- 103 MPS matmul encodes per step, all in 1 command buffer
- CPU encoding time (~30ms) << GPU execution time (~500ms at 256x256)
- MPS matmul achieves ~4.5 TFLOPS (32% of peak f32)
- Scratch pool, merged weights, and encode reduction have no effect
- Further denoising optimization requires reducing FLOP count or increasing GPU utilization

### Attempt 5: Monolithic GPU Qwen3 text encoder (SUCCESS)
- Previous: each of 36 layers did CPU RMSnorm + GPU matmul + download to CPU = 72 GPU syncs
- New: keep hidden state on GPU (bf16) across all layers, 1 sync at the end
- Also skip layers 27-35 (only layers 0-26 needed for output extraction at 8, 17, 26)
- GPU blit copy saves hidden state at extraction layers within the command buffer
- bf16→f32 conversion enqueued before batch_end(), read after sync
- All existing GPU primitives reused (rms_norm_bf16, add_bf16, causal_attention_bf16, etc.)
- No new shaders needed — just rewired the data flow
- Syncs: 72 → 1, layers computed: 36 → 27 (25% compute reduction)
- **Result: text encoding 1.9s → 1.0s (47% faster)**
- End-to-end: 256x256 5.9s → 5.4s, 512x512 9.1s → 8.6s

### Next targets — beyond transformer denoising
- **Text encoder (Qwen3)**: now 1.0s (was 1.9s) — 1.0s remaining is mostly GPU compute
  - Theoretical minimum ~430ms (1.94 TFLOPS, assuming 4.5 TFLOPS achieved throughput)
  - Look at `./mlx/mlx/backend/metal/` for Apple Metal kernel insights (JITs, tiling strategies)
- **VAE decoder**: 0.4s (256x256), 1.6s (512x512) — scales with resolution
- **1024x1024 generation**: different dynamics — img_seq = 4096 tokens, attention-heavy
  - Attention becomes larger fraction of compute (O(n^2) scaling)
  - Custom fused attention kernel may help at larger seq lengths
- **img2img with `-i`**: adds reference image tokens, further increases sequence length
- **Step 1 warmup**: ~15% slower than subsequent steps (residual MPS JIT)
  - Already tried JIT pre-warming (Attempt 1b), didn't help

## Credits attribution rules
- Ideas / kernels / approaches should be only taken from BSD / MIT licensed code.
- If any optimization ideas or kernel code are taken from some other project,
  proper credits must be added to both the README and the relevant source file.
