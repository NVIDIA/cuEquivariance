## Latest Changes

### Breaking Changes
- [Torch] Removed deprecated primitive classes: `TensorProduct`, `EquivariantTensorProduct`, `SymmetricTensorProduct`, and `IWeightedSymmetricTensorProduct`. Use `cuet.SegmentedPolynomial` with `method='uniform_1d'` instead, or the high-level APIs (`cuet.ChannelWiseTensorProduct`, `cuet.FullyConnectedTensorProduct`, `cuet.SymmetricContraction`). Attempting to import these classes will raise an `ImportError` with migration instructions.
- [Torch] Removed deprecated low-level wrapper classes: `TensorProductUniform1d`, `TensorProductUniform4x1d`, `TensorProductUniform3x1dIndexed`, `TensorProductUniform4x1dIndexed`, and `SymmetricTensorContraction` from `cuequivariance_ops_torch`. Use `torch.ops.cuequivariance.uniform_1d` or `cuet.SegmentedPolynomial` instead.

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