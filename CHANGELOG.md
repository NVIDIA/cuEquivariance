## Latest Changes

### Breaking Changes
- Rename `SegmentedTensorProduct.flop_cost` in `flops`
- Rename `SegmentedTensorProduct.memory_cost` in `memory`


## 0.3.0-rc1

### Breaking Changes
- In `cuex.equivariant_tensor_product`, the arguments `dtype_math` and `dtype_output` are renamed `math_dtype` and `output_dtype` respectively. Adding consistency with the rest of the library.
- In `cuex.equivariant_tensor_product`, the arguments `algorithm`, `precision`, `use_custom_primitive` and `use_custom_kernels` are removed. This is to avoid a proliferation of arguments that are not used in all the implementations. An argument `impl: str` is added instead to select the implementation.
- Removed `cue.TensorProductExecution` and added instead `cue.Operation` that is more lightweight and aligned with the backend.
- Removed `cuex.symmetric_tensor_product`. `cuex.tensor_product` is now able to handle any non-homogeneous polynomials.
- Removed `cuex.flax_linen.Linear` to reduce maintenance burden. Use `cue.descriptor.linear` together with `cuex.equivariant_tensor_product` instead.
- The batching support (`jax.vmap`) of `cuex.equivariant_tensor_product` is currently limited.
- The interface of `cuex.tensor_product` is changed. Now it takes a list of `tuple[cue.Operation, cue.SegmentedTensorProduct]` instead of a single `cue.SegmentedTensorProduct`. This allows `cuex.tensor_product` to execute any sort of non-homogeneous polynomials.

### Fixed
- Identified bug in CUDA kernel, disable CUDA kernel for `cuet.TransposeSegments` and `cuet.TransposeIrrepsLayout`.
- `cue.descriptor.full_tensor_product` was ignoring the `irreps3_filter` argument.

### Added
- JAX Bindings to the uniform 1d JIT kernel. This kernel handles any kind of non-homogeneous polynomials as long as the contraction pattern (subscripts) have only one index. It handles batched/shared/indexed input/output. The indexed input/output are handled by atomic operations.
- Added `__mul__` to `cue.EquivariantTensorProduct` to allow rescaling the equivariant tensor product.
- Added a uniform 1d kernel with scatter/gather fusion under `cuet.primitives.tensor_product.TensorProductUniform4x1dIndexed` and `cuet.primitives.tensor_product.TensorProductUniform3x1dIndexed`.


## 0.2.0 (2025-01-24)

### Breaking Changes

- Minimal python version is now 3.10 in all packages.
- `cuet.TensorProduct` and `cuet.EquivariantTensorProduct` now require inputs to be of shape `(batch_size, dim)` or `(1, dim)`. Inputs of dimension `(dim,)` are no more allowed.
- `cuex.IrrepsArray` is an alias for `cuex.RepArray`.
- `cuex.RepArray.irreps` and `cuex.RepArray.segments` are not functions anymore. They are now properties.
- `cuex.IrrepsArray.is_simple` is replaced by `cuex.RepArray.is_irreps_array`.
- The function `cuet.spherical_harmonics` is replaced by the Torch Module `cuet.SphericalHarmonics`. This was done to allow the use of `torch.jit.script` and `torch.compile`.

### Added

- Add an experimental support for `torch.compile`. Known issue: the export in c++ is not working.
- Add `cue.IrrepsAndLayout`: A simple class that inherits from `cue.Rep` and contains a `cue.Irreps` and a `cue.IrrepsLayout`.
- Add `cuex.RepArray` for representing an array of any kind of representations (not only irreps like before with `cuex.IrrepsArray`).

### Fixed

- Add support for empty batch dimension in `cuet` (`cuequivariance_torch`).
- Move `README.md` and `LICENSE` into the source distribution.
- Fix `cue.SegmentedTensorProduct.flop_cost` for the special case of 1 operand.

### Improved

- No more special case for degree 0 in `cuet.SymmetricTensorProduct`.

## 0.1.0 (2024-11-18)

- Beta version of cuEquivariance released.
