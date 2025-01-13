## Latest Changes

### Breaking Changes

- `cuet.TensorProduct` and `cuet.EquivariantTensorProduct` now require inputs to be of shape `(batch_size, dim)` or `(1, dim)`. Inputs of dimension `(dim,)` are no more allowed.
- `cuet.TensorProduct` and `cuet.EquivariantTensorProduct` are no more variadic functions. They now require a list of `torch.Tensor` as input.
- `cuex.IrrepsArray` is an alias for `cuex.RepArray`.
- `cuex.RepArray.irreps` and `cuex.RepArray.segments` are not functions anymore. They are now properties.
- `cuex.IrrepsArray.is_simple` is replaced by `cuex.RepArray.is_irreps_array`.
- The function `cuet.spherical_harmonics` is replaced by the Torch Module `cuet.SphericalHarmonics`. This was done to allow the use of `torch.jit.script` and `torch.compile`.

### Added

- Add an experimental support for `torch.jit.script` and `torch.compile`. Known issue: the export in c++ is not working.
- Add `cue.IrrepsAndLayout`: A simple class that inherits from `cue.Rep` and contains a `cue.Irreps` and a `cue.IrrepsLayout`.
- Add `cuex.RepArray` for representing an array of any kind of representations (not only irreps like before with `cuex.IrrepsArray`).

### Fixed

- Add support for empty batch dimension in `cuet` (`cuequivariance_torch`).

## 0.1.0 (2024-11-18)

- Beta version of cuEquivariance released.
