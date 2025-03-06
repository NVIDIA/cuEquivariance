## Latest Changes

### Breaking Changes
- Rename `SegmentedTensorProduct.flop_cost` in `flops`
- Rename `SegmentedTensorProduct.memory_cost` in `memory`
- Removed `IrrepsArray` in favor of `RepArray`


## 0.3.0 (2025-03-05)

The main changes are:
1. [JAX] New JIT Uniform 1d kernel with JAX bindings
   1. Computes any polynomial based on 1d uniform STPs
   2. Supports arbitrary derivatives
   3. Provides optional fused scatter/gather for the inputs and outputs
   4. 🎉 We observed a ~3x speedup for MACE with cuEquivariance-JAX v0.3.0 compared to cuEquivariance-Torch v0.2.0 🎉
2. [Torch] Adds torch.compile support
3. [Torch] Beta limited Torch bindings to the new JIT Uniform 1d kernel (see tutorial in the documentation)
4. [Torch] Implements scatter/gather fusion through a beta API for Uniform 1d (see tutorial in the documentation)

### Breaking Changes
- In `cuex.equivariant_tensor_product`, the arguments `dtype_math` and `dtype_output` are renamed to `math_dtype` and `output_dtype` respectively. This change adds consistency with the rest of the library.
- In `cuex.equivariant_tensor_product`, the arguments `algorithm`, `precision`, `use_custom_primitive` and `use_custom_kernels` have been removed. This change avoids a proliferation of arguments that are not used in all implementations. An argument `impl: str` has been added instead to select the implementation.
- Removed `cue.TensorProductExecution` and added `cue.Operation` which is more lightweight and better aligned with the backend.
- Removed `cuex.symmetric_tensor_product`. The `cuex.tensor_product` function now handles any non-homogeneous polynomials.
- Removed `cuex.flax_linen.Linear` to reduce maintenance burden. Use `cue.descriptor.linear` together with `cuex.equivariant_tensor_product` instead.
- The batching support (`jax.vmap`) of `cuex.equivariant_tensor_product` is now limited to specific use cases.
- The interface of `cuex.tensor_product` has changed. It now takes a list of `tuple[cue.Operation, cue.SegmentedTensorProduct]` instead of a single `cue.SegmentedTensorProduct`. This change allows `cuex.tensor_product` to execute any type of non-homogeneous polynomials.

### Fixed
- Identified a bug in the CUDA kernel and disabled CUDA kernel for `cuet.TransposeSegments` and `cuet.TransposeIrrepsLayout`.
- Fixed `cue.descriptor.full_tensor_product` which was ignoring the `irreps3_filter` argument.
- Fixed a rare bug with `np.bincount` when using an old version of numpy. The input is now flattened to make it work with all versions.

### Added
- Added JAX Bindings to the uniform 1d JIT kernel. This kernel handles any kind of non-homogeneous polynomials as long as the contraction pattern (subscripts) has only one mode. It handles batched/shared/indexed input/output. The indexed input/output is processed through atomic operations.
- Added an `indices` argument to `cuex.equivariant_tensor_product` and `cuex.tensor_product` to handle the scatter/gather fusion.
- Added `__mul__` to `cue.EquivariantTensorProduct` to allow rescaling the equivariant tensor product.
- Added a uniform 1d kernel with scatter/gather fusion under `cuet.primitives.tensor_product.TensorProductUniform4x1dIndexed` and `cuet.primitives.tensor_product.TensorProductUniform3x1dIndexed`.


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
