## Latest Changes

### Added

- Partial support of `torch.jit.script` and `torch.compile`
- Added `cuex.RepArray` for representing an array of any kind of representations (not only irreps like before with `IrrepsArray`).

### Changed

- `cuequivariance_torch.TensorProduct` and `cuequivariance_torch.EquivariantTensorProduct` now require lists of `torch.Tensor` as input.
- `cuex.IrrepsArray` is now an alias for `cuex.RepArray` and its `.irreps` attribute and `.segments` are not functions anymore but properties.

## Removed

- `cuex.IrrepsArray.is_simple` is replaced by `cuex.RepArray.is_irreps_array`.

### Fixed

- Add support for empty batch dimension in `cuequivariance-torch`.

## 0.1.0 (2024-11-18)

- Beta version of cuEquivariance released.
