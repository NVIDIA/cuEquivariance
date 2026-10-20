## Latest Changes

## 0.14.0 (unreleased)

### Added

- [Torch/JAX] Triangle Attention exposes a static `fast=False` precision option on supported Rubin GPUs. `fast=True` selects BF16 exponentials and is inference-only; use the default Accurate mode for training. Pair with the matching 0.14 ops packages.

### Notes

- [Torch/JAX] On supported Rubin GPUs, an empty Boolean-mask row in Triangle Attention returns zero output and LSE/MAX of `-1e9`, rather than the previous zero auxiliary values. Returned LSE uses natural logarithms in both precision modes. Torch length-only empty rows retain zero output/LSE/MAX.

## 0.13.0 (2026-10-15)

### Added

- [Torch] **Optimized Triangle Multiplicative Update kernels.** `cuet.triangle_multiplicative_update` now runs new prebuilt optimized kernels on the shapes each kernel was validated on. All other calls use the 0.12 implementation, unchanged. CUDA 13 wheels only. See "Optimized TriMul kernels" below.
  - Inference on H100 SXM, H200, GH200 480GB, B200 and RTX PRO 6000 Blackwell; training on H100 SXM, H200, GH200 480GB and B200.
  - Pair width C and hidden width D: 128/128, 256/256, 384/256 and 512/512, with and without projection bias. The exact supported shapes are listed in the GitHub release notes for 0.13.0.
- [JAX] `cuex.triangle_attention` accepts `mask=None` for unmasked attention, as `cuet.triangle_attention` does. The call then runs the unmasked kernels, without an all-True mask. This is supported through `jax.grad` and `jax.vmap`. With an older `cuequivariance-ops-jax`, the frontend makes an all-True mask and passes it. ([#307](https://github.com/NVIDIA/cuEquivariance/pull/307))

### Bug Fixes

- [Torch/JAX] Triangle Attention on RTX PRO 6000 Blackwell, CUDA 13 wheels: fixed a register spill in the masked forward kernels at head dimension 64 (BF16 and FP16) that made the masked forward slower than in older builds. Results are bit-identical.
- [JAX] Unmasked Triangle Attention calls no longer run the Boolean-mask kernels. Pass `mask=None` to get the unmasked forward and backward kernels.
- [Torch] Fixed a failure with PyTorch 2.14 in training: the fused gated dual GEMM backward returned too many gradients, and PyTorch 2.14 raises an error for this. Every training call that reached this kernel failed, including the 0.12 TriMul training route.

### Breaking Changes

- None.

### Notes

- **Supported PyTorch versions:** 2.11, 2.12 and 2.13. PyTorch 2.14 is not yet supported. A known PyTorch 2.14 training failure is fixed in this release (see Bug Fixes), but full 2.14 support comes in a future release.
- The optimized TriMul kernels run only with the CUDA 13 wheels (`cuequivariance-ops-torch-cu13`). The CUDA 12 wheels do not contain the optimized TriMul kernels and instead use the 0.12 implementation.
- The optimized TriMul kernels run only on these GPUs: H100 SXM (80 GB HBM3), H200, GH200 480GB, B200 and RTX PRO 6000 Blackwell (Server and Workstation Edition). All other GPUs use the 0.12 implementation. This includes H100 PCIe, H100 NVL, B300 / GB300, A100, L40(S) and Rubin.
- [JAX] The optimized TriMul kernels are PyTorch only. `cuex.triangle_multiplicative_update` is unchanged from 0.12.0. The supported PyTorch versions above do not apply to JAX.
- Recommended usage is unchanged: `torch.compile(model, mode="reduce-overhead")`. Warm up for a few iterations first.
- The optimized kernels were added only on shapes where they passed the accuracy release gate and were faster than the 0.12 implementation in our CI.
- [Torch] The optimized TriMul kernels run only on the exact listed shapes. See the GitHub release notes for 0.13.0 for the supported configurations, the precision rules and how to turn the kernels off.

### Known issues

- **Small-sequence training on B200.** A few B200 training cells at short sequence lengths (N=128) can run slightly slower than the 0.12 implementation. We are working to resolve this in a future release.
- **Exported models use the 0.12 implementation.** `torch.export` (strict and non-strict), ONNX export (and so TensorRT), and traces with symbolic sizes outside `torch.compile` (`make_fx`, `FakeTensorMode`) never record an optimized kernel.
- **Eager-mode routing cost.** In eager mode, calls that do not take an optimized kernel pay a small routing check on each call. The cost is most visible at small shapes, and most of all on RTX PRO 6000. Under `torch.compile`, the check costs nothing.
- **Second backward and double backward.** In training, a second backward through one forward (`retain_graph=True`):
  - is exact in eager mode, but needs about 1.9× the memory of the first backward;
  - gives NaN gradients when it is replayed from a CUDA graph. This is by design: the result is visibly wrong, never silently wrong. The first backward of every replay is exact.

  Double backward (`create_graph=True`) raises an error.
- **Exclusive GPU.** Some optimized kernels use software grid barriers and expect exclusive use of the GPU. Under MPS sharing without SM partitions, they can hang.
- **FP32 TriMul on GPUs with about 100 KB shared memory per block** (L40S and other Ada GPUs, RTX PRO 6000 and other sm_120 GPUs): with Triton newer than 3.6 (PyTorch 2.14 ships 3.8), the FP32 0.12 Triton kernel at C=128 fails with `OutOfResources`. This is not an issue of the optimized kernels; 0.12.0 has the same problem. BF16 calls are not affected.
- **Per-rank compile limit under DDP.** Under DDP or multi-node training, each rank applies the recompile limit separately. Ranks that see shapes in a different order can run the optimized kernels on different shapes. Results are not affected, but a small speed difference between ranks is possible.

## 0.12.0 (2026-09-24)

### Added

- [Performance] Improvements to triangle multiplicative update under `torch.compile` (inference/training, geometric mean improvement across all supported shapes in v0.11.1): 1.2x/1.1x on A100, 1.2x/1.1x on H100/H200, 1.3x/1.15x on B200 / B300, on RTX Pro 6000 Blackwell Server Edition.
- [Torch] Expanded `cuet.triangle_multiplicative_update` kernel coverage for pair width 384 and hidden width 256: inference and training on H100 / H200, and inference on B200 / B300 and RTX PRO 6000 Blackwell. The H100 / H200 inference path also accepts sequence lengths that are not multiples of 128.
- [Torch] `cuet.attention_pair_bias` accepts `norm_kind="layer_norm"` (default) or `"rms_norm"`. RMSNorm requires every normalization bias to be `None`. (#299)
- [Torch] Expanded Attention Pair Bias training kernel coverage on H100 / H200 for RMSNorm with single width 768, pair width 384, 12 heads, and head dimension 64 under BF16 autocast and `torch.compile`.
- [ops-torch] Added an RMSNorm pair-update / Attention Pair Bias inference handoff on H100 / H200, B200 / B300, and RTX PRO 6000 Blackwell for supported shapes, including pair width 384 with 12 heads and head dimension 64. `pair_update_finalize_masked_attention_bias` returns the updated pair representation and an optional masked attention-bias cache; pass that cache to `attention_pair_bias_masked_cache` with `is_cached_z_proj=True`.
- [ops-torch] Added module-based Attention Pair Bias entry points, packed Q/K/V projection parameters, and `attention_pair_bias_shape_is_supported`. Module-based entry points preserve normalization for non-affine LayerNorm and RMSNorm modules.
- [ops-torch] Added split-parameter Triangle Multiplicative Update entry points, preparation helpers, and shape-support and inference-route queries for callers that manage separate projection parameters or select kernels before allocating tensors.
- [Torch/JAX] Expanded Triangle Attention layout support on RTX PRO 6000 Blackwell (sm120f) to include non-contiguous Q/K/V layouts with a contiguous head dimension and 16-byte-aligned outer strides. Updated the head-dimension-64 forward kernel.
- [Torch] Triangle Attention with `kv_lengths` on B200 / B300 now handles zero-length rows through CUDA postprocessing in eager inference. Under `torch.compile`, row zeroing remains visible to the compiler.
- [Torch/JAX] Added Vera Rubin (sm_107) to the `cuequivariance-ops-*-cu13` wheel builds, using the existing package names. cuEquivariance is currently functional on Vera Rubin for the purpose of ecosystem integration, and performance improvements will be released in the near future.
  - Note: Users must bring their own Rubin compatible container and/or PyTorch version and install the wheel inside of it.

### Bug Fixes

- [Torch/JAX] Fixed a warp-reduction bug in fused tensor products that could silently produce incorrect results in segmented polynomials, tensor products, and indexed linear layers. The fix applies across GPU architectures.
- [Torch] Fixed the Triangle Multiplicative Update training performance regression at batch sizes greater than 1 for the covered BF16 shapes.
- [Torch] `cuet.triangle_multiplicative_update` now applies `eps` to both LayerNorms in the reference implementation. Fixed an assertion failure in the generic path when the pair and hidden widths differ.
- [Torch] Fixed graph breaks caused by dispatch warnings under `torch.compile`.
- [Torch] Fixed Triangle Attention backward handling of sliced inputs and K/V tensors with different layouts on B200 / B300. Misaligned input views are copied to aligned storage before the SM100f / SM120f kernels run.
- [JAX] Fixed the `custom_vjp` output-structure mismatch in `cuex.triangle_attention` that caused `jax.grad` and `jax.value_and_grad` to fail. Added coverage for differentiation through checkpointed scans. (#295, #300)

### Breaking Changes

- [Torch] Triangle Attention output on SM100f / SM120f can now preserve the query tensor's non-contiguous layout. Call `.contiguous()` before operations that require contiguous storage. Auxiliary `lse` and `max` outputs remain contiguous.
- [Torch] `cuet.attention_pair_bias` raises `ValueError` when `w_proj_z` is `None` unless `is_cached_z_proj=True`. (#299)
- [JAX] cuequivariance-jax now requires JAX >= 0.9.1, matching the XLA FFI API used by the kernel bindings.

### Notes

- CUDA 12 wheel builds now use CUDA 12.9.1 (previously CUDA 12.8). The cu12 builds now include the SM100f and SM120f Triangle Attention kernels.
- CUDA 13 wheel builds now use CUDA 13.4.1 (previously CUDA 13.0), which now supports Rubin.
- [Torch] Under `torch.compile`, Triangle Multiplicative Update calls outside the specialized kernels' supported shapes use a compiler-visible reference formula for hidden width 256 and pair widths 256–512, with BF16 inputs or FP32 inputs under BF16 autocast at default precision. Other shapes and precision modes retain their existing dispatch rules.
- [Torch] Shape support also depends on dtype, layout, gradient mode, and the kernels available in the installed build. H100 / H200 TriMul training kernels still require sequence lengths divisible by 128. The APB RMSNorm inference handoff is exposed through `cuequivariance_ops_torch`; setting `norm_kind="rms_norm"` on `cuet.attention_pair_bias` alone does not select it. If the finalizer returns no cache, use the ordinary attention call.

## 0.11.1 (2026-08-07)

Patch release focused on PyTorch compatibility. No API or behavior changes — the pure-Python frontend is unchanged from 0.11.0. This is a coordinated version bump across the frontend and the compiled `cuequivariance-ops-*` wheels.

### Changed
- [ops-torch] Widened PyTorch version support. The `cuequivariance-ops-torch-cu12` / `-cu13` wheels now import and run across PyTorch **2.11, 2.12, and 2.13**. Previously the compiled extension was ABI-locked to the exact PyTorch minor it was built against, so any other version failed with an undefined-symbol error at import.

### Bug Fixes
- [ops-torch] `torch>=2.11` is now a real runtime dependency. It was previously declared only in an optional `test` extra, so `pip` did not enforce it — a user on PyTorch < 2.11 could install the wheel and then hit a cryptic import failure. `pip` now blocks the incompatible install up front.

## 0.11.0 (2026-08-04)

### Added

- [Torch] `cuet.triangle_attention` accepts `kv_lengths`, an int32 per-row key/value length tensor (shape `[B, N, 1, 1, 1]`) for right-padded sequences.
- Passing `kv_lengths` selects the SM100f (B200 / B300) or sm120f (RTX 6000 Pro Blackwell) length fast path when available; on other GPUs it is converted to the equivalent dense prefix mask. `kv_lengths` and `mask` are mutually exclusive (passing both raises). Added `cuet.mask_to_kv_lengths` to convert and validate prefix masks.
- [Torch/JAX] Native `sm120f` Triangle Attention kernel for RTX 6000 Pro Blackwell, replacing the generic kernel used on that SKU in 0.10.
- [Torch] `cuet.attention_pair_bias` exposes the optional generalized projection parameters `b_proj_k`, `b_proj_v`, `w_ln_q`/`b_ln_q`, `w_ln_k`/`b_ln_k`, and `b_proj_g` as keyword-only arguments (default `None`), preserving the strict contract on `single_repr`; `w_proj_z` may be omitted when `is_cached_z_proj=True`. ([#293](https://github.com/NVIDIA/cuEquivariance/pull/293))
- [JAX] Added a dictionary-based interface for organizing and processing features by representation type. ([#271](https://github.com/NVIDIA/cuEquivariance/pull/271))

### Bug Fixes

- [Torch] Triangle Attention no longer silently mis-handles dense masks on Blackwell. In 0.10, passing any `mask=` on Blackwell (B200 / B300) took the `SM100f` fast path. `SM100f` is a length/prefix kernel — it attends `[0, kv_length)` per row — so it collapsed any mask into a single per-row length. A full or right-padded (prefix) mask was fast and correct, but an arbitrary (non-prefix) mask was fast and silently incorrect: interior gaps were dropped, with no warning and no error. In 0.11, `SM100f` rejects any dense `mask=` (including an all-ones mask) and routes it to the correct fallback kernel. Arbitrary masks are now always correct.
- [Torch] Fixed TorchScript and Pytorch 2.2 compatibility issues in `cuet.ChannelWiseTensorProduct` and `cuet.SegmentedPolynomial`. ([#277](https://github.com/NVIDIA/cuEquivariance/pull/277))

### Breaking Changes

- [Torch] `cuet.attention_pair_bias` now takes the single representation `single_repr` plus the node-LayerNorm (`w_ln_a`/`b_ln_a`) and Q/K/V/gate projection weights instead of pre-projected `q`/`k`/`v` and a separate `s`, and performs the projections internally. The pre-projected `q`/`k`/`v` and `s` are removed, and the op is now self-attention only (square `N × N`) — rectangular `U × V` (`U != V`) cross-attention is no longer supported. ([#290](https://github.com/NVIDIA/cuEquivariance/pull/290))
- [Torch] On Blackwell (B200 / B300), a dense `mask=` passed to `cuet.triangle_attention` (including an all-ones mask) no longer takes the SM100f fast path — it routes to the correct fallback, which is slower than the fast path. Build `kv_lengths` once upstream, then pass `kv_lengths=cuet.mask_to_kv_lengths(mask)` for the fast path. A one-time warning points to this whenever a mask would have reached `SM100f`.
- [Torch] PyTorch 2.11.X is now required.

### Notes

- Which masking choice accelerates what (Torch, B200/B300):
  - *Fully-valid sequences* → pass no `mask` and no `kv_lengths`. This is the fastest path, and it routes exactly as it did in 0.10 (unaffected by the mask change).
  - *Right-padded / prefix sequences* → pass `kv_lengths`. The kernel skips the padded keys entirely, while native PyTorch computes them and masks afterwards — so cuEquivariance's advantage over native PyTorch is largest here.
  - *Arbitrary (non-prefix) masks* → pass `mask=`. Correct in 0.11, but runs the fallback.
- Build `kv_lengths` once, upstream. `cuet.mask_to_kv_lengths` does a one-time host sync (prefix-shape validation; it raises on a non-prefix mask), so compute it in the collator / data-processing stage. The op itself is host-sync-free with `kv_lengths`: the only range-check `.item()` is opt-in (`CUEQ_TRIATTN_VALIDATE_KV_LENGTHS`) and skipped under `torch.compile`; the clamp is device-side.
- [JAX] The `mask` → `kv_lengths` changes are Torch-only. The JAX path auto-selects the `SM100f` fast path from its mask and is unaffected.
- Recommended usage. For best performance, run cuEquivariance ops under `torch.compile(model, mode="reduce-overhead")` (CUDA graphs), which removes per-call launch/dispatch overhead — most impactful at small shapes and low batch, where the kernels are launch-bound. Warm up first: the first few iterations (≈3) go to tracing and CUDA-graph capture and are slower; the speedups appear only once the graph is set up. CUDA graphs assume static shapes and re-capture when shapes change.
- Hardware support. Triangle Attention: `SM100f` fast path on B200 / B300 (via `kv_lengths`); native `sm120f` on RTX 6000 Pro Blackwell; generic kernel on H100 / H200 / A100 / L40. APB inference: H100 / H200, B200, B300, RTX 6000 Pro Blackwell, A100 (not L40). APB training: H100 / H200, B200, B300 (not A100, RTX 6000 Pro Blackwell, or L40).
- Data type support. Triangle Attention supports BF16, FP16, and FP32/TF32 — the FP32/TF32 kernel requires `hidden_dim <= 32` and `hidden_dim % 4 == 0`; larger head dimensions must use BF16 or FP16. The specialized `sm100f` and `sm120f` kernels are BF16/FP16 only. APB's optimized paths for inference and training use BF16 and, for select routes, FP16.
- APB shape support. The optimized `attention_pair_bias` path supports pair width 128 and 256, head dimension 8-64, and num heads 4-16.
- Triangle Multiplication remains unchanged in 0.11.
- Known issue. A minor per-call overhead affects the Triangle Attention `kv_lengths` (padded-sequence) inference path, most visible at short sequences. It does not affect correctness, training, or the no-mask path. A fix is in progress.

### Masked and Right-Padded Sequences on B200 / B300

On B200 / B300, supplying `kv_lengths` selects the `SM100f` length fast path, and the advantage over native PyTorch grows when sequences are masked or right-padded. Native PyTorch materializes the dense mask and computes the padded keys before masking them; cuEquivariance passes a compact per-row length and skips them in the kernel entirely.

### Known issues

- A minor per-call overhead affects the Triangle Attention `kv_lengths` (padded-sequence) inference path, most visible at short sequences. It does not affect correctness, training, or the no-mask path. A fix is in progress.
- CUDA 12.8 wheels do not support Blackwell's SM100F fast path. In the next release we will upgrade our CUDA 12.8 wheels to use CUDA 12.9 which will support TriAttn’s SM100F fast path on Blackwell.
- TriMul training is slower than native PyTorch (with `torch.compile`) at batch sizes > 1 due to a slowdown in the backwards pass – we will work to resolve this in our next release.
- There is an issue using torch version 2.11+ which we will fix in a minor update (v0.11.1) coming soon

### Documentation

- *[Torch/JAX]* Expose generalized projection params in attention_pair_bias wrapper ([#293](https://github.com/NVIDIA/cuEquivariance/pull/293))
- *[Torch]* Triangle Attention kv_lengths documentation ([#287](https://github.com/NVIDIA/cuEquivariance/pull/287))
- [JAX] Added an example showing how to implement MACE with Equinox. ([#281](https://github.com/NVIDIA/cuEquivariance/pull/281))

## 0.10.0 (2026-04-21)

### Added
- Python 3.14 support finalized, including a fix for stale tuple hashes in `SegmentedTensorProduct` after in-place operand mutation, and updated CI matrix ([#272](https://github.com/NVIDIA/cuEquivariance/pull/272))
- [Torch/JAX] `cuet.triangle_attention`/`cuex.triangle_attention`: new faster sm100f (CC 10.0/10.3) forward kernel for hidden_dim ≤ 256, bwd hidden_dim ≤ 128; `bias` is cast to q/k/v dtype (instead of always float32) under sm100f; non-contiguous input tensors are handled internally — no manual contiguity assertion is required as long as shape requirements are met; updated docstrings. Only available on cu13 builds ([#260](https://github.com/NVIDIA/cuEquivariance/pull/260))
- [JAX] MACE `flax.nnx` example restructured to use `nnx.split` + `@jax.jit` on `(graphdef, state)` instead of `@nnx.jit` on the module, removing the Python-side nnx graph traversal overhead from each training/inference step ([#261](https://github.com/NVIDIA/cuEquivariance/pull/261))
- [JAX] NVTX markers added to the MACE examples to make step boundaries visible in `nsys` profiles ([#266](https://github.com/NVIDIA/cuEquivariance/pull/266))

### Bug fix
- [Torch] `SegmentedPolynomial` checkpoint portability: GPU-saved models now load correctly on CPU. Implemented via `__reduce__` on `SegmentedPolynomialFromUniform1dJit`, `SegmentedPolynomialFusedTP`, `SegmentedPolynomialIndexedLinear`, and `SegmentedPolynomial`, plus graceful fallback when specific `cuequivariance_ops_torch` extensions (e.g. `uniform_1d`) are unavailable ([#270](https://github.com/NVIDIA/cuEquivariance/pull/270))
- [Torch] Replaced deprecated `is_fx_tracing` with `is_fx_symbolic_tracing` ([#270](https://github.com/NVIDIA/cuEquivariance/pull/270))
- [JAX] Restrict PTX 88 to sm_121 for CUDA 12.9+, avoiding breakage on other architectures (addresses the known issue noted in the 0.9.0 release) ([#250](https://github.com/NVIDIA/cuEquivariance/pull/250))
- [Torch] `cuet.attention_pair_bias`: fixed incorrect results when the hidden dimension is not a multiple of 32; the previous torch fallback for these cases is removed as the kernel now handles them correctly

### Notes
- [Torch] The `CUEQ_TORCH_COMPILE` environment variable (experimental) enables `torch.compile` for `cuet.triangle_attention`; useful for non-contiguous tensor inputs on Ampere/Hopper architectures

### Documentation
- Fixed tutorial format issues ([#274](https://github.com/NVIDIA/cuEquivariance/pull/274))

## 0.9.1 (2026-03-16)

### Bug fix
- [Torch/JAX] Fixed a rare overflow in the uniform 1d kernel when indices are large, by casting index arithmetic from 32-bit to 64-bit
- [Torch/JAX] Disabled parallel compilation of uniform 1d kernels by default (`CUEQUIVARIANCE_OPS_PARALLEL_COMPILE` now defaults to `0`). Parallel compilation caused issues in multi-GPU setups. It will be re-enabled by default in a future release once the underlying bugs are resolved.

## 0.9.0 (2026-02-17)

### Added
- GB10 (DGX Spark) support
- Support for Python 3.13 and 3.14
- [JAX] Support for Triton 3.6.0
- [JAX] `flax.nnx` MACE example
- [Torch/JAX] Deterministic indexing mode for uniform 1d kernels
- [Torch/JAX] Parallel JIT compilation for uniform_1d kernels with per-kernel caching, significantly reducing compilation time. New optional environment variable `CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR` allows setting a directory for caching compiled kernels.
- Documentation: new tutorials for JAX and PyTorch segmented polynomials

### Bug fix
- [JAX] Fixed Triton tuning issue for triangular multiplicative update
- [JAX] Compatibility with JAX 0.8.2: fixed FFI interface and dtype casting issues when x64 mode is not enabled
- [JAX] Improved triangle attention error messages
- [Torch/JAX] Fixed `yx_rotation` descriptor
- [Torch] TensorRT QDP plugin workaround

### Breaking Changes
- [Torch/JAX] The environment variable `CUEQUIVARIANCE_OPS_USE_JIT` no longer exists. JIT compilation is now the default behavior for uniform_1d kernels (already since few releases).
- [Torch/JAX] Renamed `filter_drop_unsued_operands` to `filter_drop_unused_operands` (typo fix)
- [Torch/JAX] Removed `nvfatbin` optional dependency
- [Torch] Removed deprecated primitive classes: `TensorProduct`, `EquivariantTensorProduct`, `SymmetricTensorProduct`, and `IWeightedSymmetricTensorProduct`. Use `cuet.SegmentedPolynomial` with `method='uniform_1d'` instead, or the high-level APIs (`cuet.ChannelWiseTensorProduct`, `cuet.FullyConnectedTensorProduct`, `cuet.SymmetricContraction`). Attempting to import these classes will raise an `ImportError` with migration instructions.
- [Torch] Removed deprecated low-level wrapper classes: `TensorProductUniform1d`, `TensorProductUniform4x1d`, `TensorProductUniform3x1dIndexed`, `TensorProductUniform4x1dIndexed`, and `SymmetricTensorContraction` from `cuequivariance_ops_torch`. Use `torch.ops.cuequivariance.uniform_1d` or `cuet.SegmentedPolynomial` instead.

### Notes
- [JAX] DGX Spark/GB10 (sm_121) with CUDA 12.9: This release uses PTX 87, which works correctly for most architectures but is not compatible with DGX Spark/GB10 on CUDA 12.9. To enable DGX Spark/GB10 support with CUDA 12.9, refer to [#250](https://github.com/NVIDIA/cuEquivariance/pull/250) for a simple frontend integration tweak that restricts PTX 88 to sm_121 only. This fix will be merged after the 0.9.0 release.

## 0.8.1 (2026-01-09)

### Bug fix
- [Torch] Fixed `torch.compile` compatibility for non-contiguous tensors in backward pass for `cuet.triangle_attention`, `cuet.triangle_multiplicative_update`, and `cuet.attention_pair_bias`. This resolves stride mismatch errors in TorchInductor when compiling models using these operations ([#223](https://github.com/NVIDIA/cuEquivariance/issues/223))

## 0.8.0 (2025-12-05)

### Added
- [Torch/JAX] Blackwell-optimized BF16/FP16 forward and backward kernels for `cuet.triangle_attention` (runs on compute capabilities 10.0 and 10.3). These kernels provide superior performance especially for long sequences and higher head dimensions. This is only supported on cu13 builds as of this release.

### Bug fix
- [Torch/JAX] Fixed index overflow and out of bound issues leading to illegal memory access in `cuet.triangle_attention` 

### Notes
- [Torch/JAX] Blackwell-optimized kernels require the sequence length N to be a multiple of 8 for the forward pass; pad the sequence if necessary
- [Torch/JAX] Blackwell-optimized kernels are currently supported only for CUDA 13 builds

## 0.7.0 (2025-10-13)

### Added
- Support for CUDA 13
- Support for Python 3.13
- [Torch] MACE example in the documentation
- [JAX] MACE and NequIP examples in the folder `cuequivariance_jax/examples`
- [Torch/JAX] The Segmented Polynomial operation's `math_dtype` argument now accepts method-specific string values, with each method supporting different options
- [Torch] ONNX export and TensorRT runtime plugin support for triangle attention, triangle multiplication and attention with pairwise bias
- [Torch] Support for caching the "bias tensor z" calculated from the proj_z linear layer in the Attention with pairwise bias kernel (as implemented in the Boltz code)

### Bug fix
- Correct the documentation `CUEQ_TRITON_TUNING_MODE` -> `CUEQ_TRITON_TUNING`
- [JAX] Make `triangle_multiplicative_update` use the same tuning cache files as its PyTorch counterpart

## 0.6.1 (2025-09-04)

### Added
- [Torch/JAX] Support for variable leading batch dimensions in triangle multiplicative update
- [Torch/JAX] Triangle attention kernel support for additional input configs: all hidden_dim<=32 and divisible by 4 for tf32/fp32, and for all hidden_dim<=128 and divisible by 8 for bf16/fp16. In the rare instance that the kernel does not support an input config, fallback to torch is enabled instead of erroring out.
- [Torch/JAX] Tuned config for RTX PRO 6000 GPUs for triangle multiplicative update.
- [JAX] vmap support for triangle multiplicative update and triangle attention
- [Torch] Improved error reporting on import failure with traceback information for stacktrace

### Bug fix
- [Torch/JAX] Fixed illegal memory access issue stemming from int32 indexing for longer sequences in triangle multiplicative update and attention with pair bias.
- [JAX] Moved to using nondiff_argnums instead of nondiff_argnames to be compatible with older JAX versions

## 0.6.0 (2025-08-11)

### Added
- [Torch] New feature: Added `cuet.attention_pair_bias` (support for caching the pair bias tensor & further kernel acceleration coming up soon. There maybe API related changes for this in the next release)
- [Torch/JAX] Added `method` argument to `cuet.SegmentedPolynomial`/`cuex.segmented_polynomial` to give users control over which backend solution is used (naive, uniform_1d, fused_tp, indexed_linear).
- [Torch/JAX] Added torch fallback option based on sequence lengths for triangle kernels and attention pair bias. The user may control this by setting env vars: `CUEQ_TRIMUL_FALLBACK_THRESHOLD`, `CUEQ_TRIATTN_FALLBACK_THRESHOLD`, `CUEQ_ATTENTION_PAIR_BIAS_FALLBACK_THRESHOLD`. The corresponding APIs default to torch fallback for seq_lens < 100 for optimal performances.
- [Torch/JAX] Added support for the optional projection and gating biases in the input and output of `cuex.triangle_multiplicative_update`
- [JAX] Added JAX bindings for triangle operations with `cuex.triangle_attention` and `cuex.triangle_multiplicative_update`

### Bug fix
- [Torch] Added `cueuivariance_ops_torch.init_triton_cache()` for users to initialize triton cache before calling torch compiled triangular multiplicative update. If not used, Torch compile would break if directly applied on `cuex.triangle_multiplicative_update`.
- [Torch/JAX] Fixed the illegal memory access error for long sequences in triangle attention. This increases the usable limits on sequence lengths.


### Breaking Changes
- Dropped support for CUDA 11. Only CUDA 12 is now supported (`cuequivariance-ops-torch-cu12`, `cuequivariance-ops-jax-cu12`).
- [Torch/JAX] Simplified precision arg of triangular multiplicative update to just two: None (defaults to triton language dot's default for non-32b input and for 32b input, tf32/tf32x3 based on 1/0 value set in torch.backends.cuda.matmul.allow_tf32) and IEEE-754.
- [Torch/JAX] We have moved away from the default round-towards-zero (RZ) implementation to round-nearest (RN) for better tf32 accuracy in cuex.triangle_multiplicative_update. In rare circumstances, this may cause minor differences in results observed. 

### Known Issues
- [JAX] The function `cuex.triangle_multiplicative_update` requires `triton<=3.3.1`. We are waiting for an update of the package `jax-triton`.
- [PyTorch] The function `cuet.triangle_multiplicative_update` requires `triton>=3.4.0` on Blackwell GPUs.
- As a consequence of the two last point, `cuex.triangle_multiplicative_update` can't run on Blackwell GPUs.
- `cuet.attention_pair_bias` does not support caching of projected pairwise tensor. We are working on adding support for this.

## 0.5.1 (2025-06-18)

This release includes improvements to triangle multiplicative update with torch.compile support and enhanced tuning configuration options.

### Added
- [Torch] `torch.compile` support for `cuet.triangle_multiplicative_update`
- [Torch] Optional precision argument for `cuet.triangle_multiplicative_update`:
  - `precision (Precision, optional)`: Precision mode for matrix multiplications. If None, uses TF32 if enabled in PyTorch using `torch.backends.cuda.matmul.allow_tf32`, otherwise uses DEFAULT precision.
  - Available options:
    - `DEFAULT`: Use default precision setting of `triton.language.dot`
    - `TF32`: Use TensorFloat-32 precision
    - `TF32x3`: Use TensorFloat-32 precision with 3x accumulation
    - `IEEE`: Use IEEE 754 precision

### Improved
- [Torch] Enhanced tuning configuration for `cuet.triangle_multiplicative_update` with support for multi-process tuning. Our tuning modes:
  - **Quick testing**: Default configuration where tuning configs, if existent, are looked-up. If not, then falls back to default kernel parameters. No tuning is performed.
  - **On-Demand tuning**: Set `CUEQ_TRITON_TUNING_MODE = "ONDEMAND"` to auto-tune for new shapes encountered on first run (may take several minutes)
  - **AOT tuning**: Set `CUEQ_TRITON_TUNING_MODE = "AOT"` to perform full ahead-of-time tuning for optimal performance **(may take several hours)**
  - **Ignore cache**: Set `CUEQ_TRITON_IGNORE_EXISTING_CACHE` to ignore both the default settings that come with the package and any user-local settings previously saved with AOT/ONDEMAND tuning. May be used to regenerate optimal settings for a particular setup.
  - **Cache directory**: Set `CUEQ_TRITON_CACHE_DIR` to specify where tuning configurations are stored. Default location is `${HOME}/.cache/cuequivariance-triton`. **Note**: When running in containers where `$HOME` is inside the container (typically `/root`), tuning changes may be lost on container restart unless the container is committed or a persistent cache directory is specified.

### Fixed
- [Torch] Fixed torch.compile compatibility issues with triangle multiplicative update
- [Torch] Tuning issues for `cuet.triangle_multiplicative_update` with multiple processes.

### Limitations
- PyTorch does not currently bundle the latest Triton version as pytorch-triton. As a result, Blackwell GPU users may occasionally experience hangs or instability during model execution. Users may attempt installation with the latest Triton from source at their own risk. We are monitoring this issue and will remedy as soon as possible.
- [Torch] Tuning for `cuet.triangle_multiplicative_update` is always performed for GPU-0 and may not be the optimal setting for all GPUs in a heterogenous multi-GPU setting

## 0.5.0 (2025-06-10)

This release introduces `triangle_attention` and `triangle_multiplicative_update`.
This is the last release with cuda11 support. In the next release we will drop cuda11.

### Added
- [Torch] Add `cuet.triangle_attention`
- [Torch] Add `cuet.triangle_multiplicative_update`
- [JAX] Add `cuex.experimental.indexed_linear`. Note that this function is not working with cuda11 because it requires cuBLAS 12.5.
- [Torch/JAX] Add argument `simplify_irreps3: bool = False` to `cue.descriptors.channelwise_tensor_product`
- [Torch/JAX] Add method `permute_inputs` to `SegmentedPolynomial`

### Improved
- [Torch/JAX] In some settings, accelerate the CUDA kernel for uniform 1d segmented polynomials (like symmetric contraction and channelwise tensor product). While most operation speeds are unchanged, we observe up to 2x speedup in some cases.

### Limitations
- PyTorch does not currently bundle the latest Triton version as pytorch-triton. As a result, Blackwell GPU users may occasionally experience hangs or instability during model execution. Users may attempt installation with the latest Triton from source at their own risk. We are monitoring this issue and will remedy as soon as possible.

### Documentation
- `cuet.triangle_multiplicative_update`: Auto-tuning behavior can be controlled through environment variables:
  - Default: Full Ahead-of-Time (AOT) auto-tuning enabled for optimal performance **(may take several hours)**
  - Quick testing: Set `CUEQ_DISABLE_AOT_TUNING = 1` and `CUEQ_DEFAULT_CONFIG = 1` to disable all tuning
  - On-Demand tuning: `CUEQ_DISABLE_AOT_TUNING = 1`, auto-tunes for new shapes encountered on first run. (may take several minutes)
  - Note: When using Docker with default or on-demand tuning enabled, commit the container to persist tuning changes
  - Note: When running in a multi-GPU setup, we recommend setting `CUEQ_DISABLE_AOT_TUNING = 1` and `CUEQ_DEFAULT_CONFIG = 1`.

## 0.4.0 (2025-04-25)

This release introduces some changes to the API, it introduce the class `cue.SegmentedPolynomial` (and corresponding counterparts) which generalizes the notion of segmented tensor product by allowing to construct non-homogeneous polynomials.

### Added
- [Torch] `cuet.SegmentedPolynomial` module giving access to the indexing features of the uniform 1d kernel
- [Torch/JAX] Add full support for float16 and bfloat16
- [Torch/JAX] Class `cue.SegmentedOperand`
- [Torch/JAX] Class `cue.SegmentedPolynomial`
- [Torch/JAX] Class `cue.EquivariantPolynomial` that contains a `cue.SegmentedPolynomial` and the `cue.Rep` of its inputs and outputs
- [Torch/JAX] Add caching for `cue.descriptor.symmetric_contraction`
- [Torch/JAX] Add caching for `cue.SegmentedTensorProduct.symmetrize_operands`
- [JAX] ARM config support
- [JAX] `cuex.segmented_polynomial` and `cuex.equivariant_polynomial`
- [JAX] Advanced Batching capabilities, each input/output of a segmented polynomial can have multiple axes and any of those can be indexed.
- [JAX] Implementation of the Dead Code Elimination rule for the primitive `cuex.segmented_polynomial`

### Breaking Changes
- [Torch/JAX] Rename `SegmentedTensorProduct.flop_cost` to `flop`
- [Torch/JAX] Rename `SegmentedTensorProduct.memory_cost` to `memory`
- [Torch/JAX] Removed `IrrepsArray` in favor of `RepArray`
- [Torch/JAX] Change folder structure of cuequivariance and cuequivariance-jax. Now the main subfolders are `segmented_polynomials` and `group_theory`
- [Torch/JAX] Deprecate `cue.EquivariantTensorProduct` in favor of `cue.EquivariantPolynomial`. The later will have a limited list of features compared to `cue.EquivariantTensorProduct`. It does not contain `change_layout` and the methods to move the operands. Please open an issue if you need any of the missing methods.
- [Torch/JAX] The descriptors return `cue.EquivariantPolynomial` instead of `cue.EquivariantTensorProduct`
- [Torch/JAX] Change `cue.SegmentedPolynomial.canonicalize_subscripts` behavior for coefficient subscripts. It transposes the coefficients to be ordered the same way as the rest of the subscripts.
- [Torch] To reduce the size of the so library, we removed support of math dtype fp32 when using IO dtype fp64 in the case of the fully connected tensor product. (It concerns `cuet.FullyConnectedTensorProduct` and `cuet.FullyConnectedTensorProductConv`). Please open an issue if you need this feature.

### Fixed
- [Torch/JAX] `cue.SegmentedTensorProduct.sort_indices_for_identical_operands` was silently operating on STP with non scalar coefficient, now it will raise an error to say that this case is not implemented. We should implement it at some point.


## 0.3.0 (2025-03-05)

The main changes are:
1. [JAX] New JIT Uniform 1d kernel with JAX bindings
   1. Computes any polynomial based on 1d uniform STPs
   2. Supports arbitrary derivatives
   3. Provides optional fused scatter/gather for the inputs and outputs
   4. 🎉 We observed a ~3x speedup for MACE with cuEquivariance-JAX v0.3.0 compared to cuEquivariance-Torch v0.2.0 🎉
2. [Torch] Adds torch.compile support
3. [Torch] Beta limited Torch bindings to the new JIT Uniform 1d kernel 
   1. enable the new kernel by setting the environement variable `CUEQUIVARIANCE_OPS_USE_JIT=1`
4. [Torch] Implements scatter/gather fusion through a beta API for Uniform 1d 
   1. this is a temporary API that will change, `cuequivariance_torch.primitives.tensor_product.TensorProductUniform4x1dIndexed`

### Breaking Changes
- [Torch/JAX] Removed `cue.TensorProductExecution` and added `cue.Operation` which is more lightweight and better aligned with the backend.
- [JAX] In `cuex.equivariant_tensor_product`, the arguments `dtype_math` and `dtype_output` are renamed to `math_dtype` and `output_dtype` respectively. This change adds consistency with the rest of the library.
- [JAX] In `cuex.equivariant_tensor_product`, the arguments `algorithm`, `precision`, `use_custom_primitive` and `use_custom_kernels` have been removed. This change avoids a proliferation of arguments that are not used in all implementations. An argument `impl: str` has been added instead to select the implementation.
- [JAX] Removed `cuex.symmetric_tensor_product`. The `cuex.tensor_product` function now handles any non-homogeneous polynomials.
- [JAX] The batching support (`jax.vmap`) of `cuex.equivariant_tensor_product` is now limited to specific use cases.
- [JAX] The interface of `cuex.tensor_product` has changed. It now takes a list of `tuple[cue.Operation, cue.SegmentedTensorProduct]` instead of a single `cue.SegmentedTensorProduct`. This change allows `cuex.tensor_product` to execute any type of non-homogeneous polynomials.
- [JAX] Removed `cuex.flax_linen.Linear` to reduce maintenance burden. Use `cue.descriptor.linear` together with `cuex.equivariant_tensor_product` instead.
```python
e = cue.descriptors.linear(input.irreps, output_irreps)
w = self.param(name, jax.random.normal, (e.inputs[0].dim,), input.dtype)
output = cuex.equivariant_tensor_product(e, w, input)
```

### Fixed
- [Torch/JAX] Fixed `cue.descriptor.full_tensor_product` which was ignoring the `irreps3_filter` argument.
- [Torch/JAX] Fixed a rare bug with `np.bincount` when using an old version of numpy. The input is now flattened to make it work with all versions.
- [Torch] Identified a bug in the CUDA kernel and disabled CUDA kernel for `cuet.TransposeSegments` and `cuet.TransposeIrrepsLayout`.

### Added
- [Torch/JAX] Added `__mul__` to `cue.EquivariantTensorProduct` to allow rescaling the equivariant tensor product.
- [JAX] Added JAX Bindings to the uniform 1d JIT kernel. This kernel handles any kind of non-homogeneous polynomials as long as the contraction pattern (subscripts) has only one mode. It handles batched/shared/indexed input/output. The indexed input/output is processed through atomic operations.
- [JAX] Added an `indices` argument to `cuex.equivariant_tensor_product` and `cuex.tensor_product` to handle the scatter/gather fusion.
- [Torch] Beta limited Torch bindings to the new JIT Uniform 1d kernel (enable the new kernel by setting the environement variable `CUEQUIVARIANCE_OPS_USE_JIT=1`)
- [Torch] Implements scatter/gather fusion through a beta API for Uniform 1d (this is a temporary API that will change, `cuequivariance_torch.primitives.tensor_product.TensorProductUniform4x1dIndexed`)


## 0.2.0 (2025-01-24)

### Breaking Changes

- Minimal Python version is now 3.10 in all packages.
- `cuet.TensorProduct` and `cuet.EquivariantTensorProduct` now require inputs to be of shape `(batch_size, dim)` or `(1, dim)`. Inputs of dimension `(dim,)` are no longer allowed.
- `cuex.IrrepsArray` is now an alias for `cuex.RepArray`.
- `cuex.RepArray.irreps` and `cuex.RepArray.segments` are no longer functions. They are now properties.
- `cuex.IrrepsArray.is_simple` has been replaced by `cuex.RepArray.is_irreps_array`.
- The function `cuet.spherical_harmonics` has been replaced by the Torch Module `cuet.SphericalHarmonics`. This change enables the use of `torch.jit.script` and `torch.compile`.

### Added

- Added experimental support for `torch.compile`. Known issue: the export in C++ is not working.
- Added `cue.IrrepsAndLayout`: A simple class that inherits from `cue.Rep` and contains a `cue.Irreps` and a `cue.IrrepsLayout`.
- Added `cuex.RepArray` for representing an array of any kind of representations (not only irreps as was previously possible with `cuex.IrrepsArray`).

### Fixed

- Added support for empty batch dimension in `cuet` (`cuequivariance_torch`).
- Moved `README.md` and `LICENSE` into the source distribution.
- Fixed `cue.SegmentedTensorProduct.flop_cost` for the special case of 1 operand.

### Improved

- Removed special case handling for degree 0 in `cuet.SymmetricTensorProduct`.

## 0.1.0 (2024-11-18)

- Beta version of cuEquivariance released.
