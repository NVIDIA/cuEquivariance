---
name: cuequivariance-jax
description: Execute equivariant polynomials in JAX using segmented_polynomial (naive/uniform_1d), equivariant_polynomial with RepArray, ir_dict with dict[Irrep, Array], and Flax NNX layers (IrrepsLinear, SphericalHarmonics). Use when writing JAX code with cuequivariance.
---

# cuequivariance_jax: Executing Equivariant Polynomials in JAX

## Overview

`cuequivariance_jax` (imported as `cuex`) executes `cuequivariance` polynomials on GPU via JAX. It provides:

1. **Core primitive**: `cuex.segmented_polynomial()` -- JAX primitive with full AD/vmap/JIT support
2. **Two data representations** (both built on `segmented_polynomial`):
   - `cuex.equivariant_polynomial()` + `RepArray` -- the original interface, a single contiguous array with representation metadata
   - `cuex.ir_dict` module -- `dict[Irrep, Array]` interface, conceptually simpler, works naturally with `jax.tree` operations
3. **NNX layers**: `cuex.nnx` module -- Flax NNX `Module` wrappers using `dict[Irrep, Array]`

## Execution methods

| Method | Backend | Requirements |
|--------|---------|-------------|
| `"naive"` | Pure JAX | Always works, any platform |
| `"uniform_1d"` | CUDA kernel | GPU, all segments uniform shape within each operand, single mode |
| `"indexed_linear"` | CUDA kernel | GPU, linear operations with `cuex.Repeats` indexing |

## Core primitive: segmented_polynomial

```python
import jax
import jax.numpy as jnp
import cuequivariance as cue
import cuequivariance_jax as cuex

# Build a descriptor
e = cue.descriptors.channelwise_tensor_product(
    32 * cue.Irreps("SO3", "0 + 1"),
    cue.Irreps("SO3", "0 + 1"),
    cue.Irreps("SO3", "0 + 1"),
)
poly = e.polynomial

batch = 64
w = jnp.ones((poly.inputs[0].size,))            # weights (shared across batch)
x = jax.random.normal(key, (batch, poly.inputs[1].size))  # batched input 1
y = jax.random.normal(key, (batch, poly.inputs[2].size))  # batched input 2

# Execute with naive method
[out] = cuex.segmented_polynomial(
    poly,
    [w, x, y],                                              # inputs
    [jax.ShapeDtypeStruct((batch, poly.outputs[0].size), jnp.float32)],  # output spec
    method="naive",
)

# Execute with uniform_1d (GPU, requires uniform segments)
[out] = cuex.segmented_polynomial(
    poly, [w, x, y],
    [jax.ShapeDtypeStruct((batch, poly.outputs[0].size), jnp.float32)],
    method="uniform_1d",
)
```

### Multiple batch axes with broadcasting

Inputs can have any number of batch axes (everything before the last axis). Standard NumPy broadcasting applies: each batch axis is either size-1 or a common size. Inputs with fewer batch dimensions are implicitly prepended with size-1 axes:

```python
# 2 batch axes with size-1 broadcasting
w = jnp.ones((1, 10, poly.inputs[0].size))        # shared across axis 0
x = jnp.ones((5, 10, poly.inputs[1].size))         # 5 along axis 0
y = jnp.ones((5, 1, poly.inputs[2].size))          # shared across axis 1

[out] = cuex.segmented_polynomial(
    poly, [w, x, y],
    [jax.ShapeDtypeStruct((5, 10, poly.outputs[0].size), jnp.float32)],
    method="uniform_1d",
)
# out.shape == (5, 10, ...)

# Fewer batch dims: weights with no batch axis broadcast across all
w = jnp.ones((poly.inputs[0].size,))              # 0 batch axes -> prepended as (1, 1, ...)
x = jnp.ones((5, 10, poly.inputs[1].size))
y = jnp.ones((5, 10, poly.inputs[2].size))

[out] = cuex.segmented_polynomial(
    poly, [w, x, y],
    [jax.ShapeDtypeStruct((5, 10, poly.outputs[0].size), jnp.float32)],
    method="uniform_1d",
)
```

### Indexing (gather/scatter)

Index arrays provide gather (for inputs) and scatter (for outputs). One index per operand (inputs + outputs), `None` means no indexing. Index arrays decouple input/output batch shapes -- the output shape is determined by the index ranges, not by the input shapes:

```python
a = jnp.ones((1, 50, poly.inputs[0].size))
b = jnp.ones((10, 50, poly.inputs[1].size))
c = jnp.ones((100, 1, poly.inputs[2].size))

i = jax.random.randint(key, (100, 50), 0, 10)     # gather b along axis 0
j1 = jax.random.randint(key, (100, 50), 0, 11)    # scatter output axis 0
j2 = jax.random.randint(key, (100, 1), 0, 12)     # scatter output axis 1

[out] = cuex.segmented_polynomial(
    poly, [a, b, c],
    [jax.ShapeDtypeStruct((11, 12, poly.outputs[0].size), jnp.float32)],
    indices=[None, np.s_[i, :], None, np.s_[j1, j2]],
    method="uniform_1d",
)
# out.shape == (11, 12, ...) -- determined by index ranges, not input shapes
```

### Gradients

Fully differentiable -- supports `jax.grad`, `jax.jacobian`, `jax.jvp`, `jax.vmap`:

```python
def loss(w, x, y):
    [out] = cuex.segmented_polynomial(
        poly, [w, x, y],
        [jax.ShapeDtypeStruct((batch, poly.outputs[0].size), jnp.float32)],
        method="naive",
    )
    return jnp.sum(out ** 2)

grad_w = jax.grad(loss, 0)(w, x, y)
```

## RepArray interface: equivariant_polynomial

The original interface. Wraps `segmented_polynomial` with `RepArray` -- a single contiguous array with representation metadata:

```python
e = cue.descriptors.fully_connected_tensor_product(
    4 * cue.Irreps("SO3", "0 + 1"),
    cue.Irreps("SO3", "0 + 1"),
    4 * cue.Irreps("SO3", "0 + 1"),
)

inputs = [
    cuex.randn(jax.random.key(i), rep, (batch,), jnp.float32)
    for i, rep in enumerate(e.inputs)
]

# Returns a RepArray with representation metadata
out = cuex.equivariant_polynomial(e, inputs, method="naive")
out.array   # the raw jax.Array
out.reps    # dict mapping axes to Rep objects
```

## ir_dict interface

An alternative to `RepArray`. Uses `dict[Irrep, Array]` where each value has shape `(..., multiplicity, irrep_dim)`. Conceptually simpler: works naturally with `jax.tree` operations and is the standard representation for NNX layers.

### Preparing a polynomial for ir_dict

Descriptors produce `EquivariantPolynomial` with dense operands. To use `ir_dict`, split operands by irrep:

```python
e = cue.descriptors.channelwise_tensor_product(
    32 * cue.Irreps("SO3", "0 + 1"),
    cue.Irreps("SO3", "0 + 1"),
    cue.Irreps("SO3", "0 + 1"),
    simplify_irreps3=True,
)

# Split irreps-typed operands into per-irrep pieces
# Order matters: split inner operands first to preserve operand indices
poly = (
    e.split_operand_by_irrep(2)      # split input 2
     .split_operand_by_irrep(1)      # split input 1
     .split_operand_by_irrep(-1)     # split output
     .polynomial
)
# After split: all operands have uniform segment shapes (required for uniform_1d)
```

### Executing with segmented_polynomial_uniform_1d

```python
from einops import rearrange

num_edges, num_nodes = 100, 30

# Weights: reshape to (batch, num_segments, segment_size)
w_flat = jax.random.normal(key, (num_edges, poly.inputs[0].size))
w = rearrange(w_flat, "e (s m) -> e s m", s=poly.inputs[0].num_segments)

# Node features: dict[Irrep, Array] reshaped to (nodes, ir.dim, mul) for ir_mul layout
node_feats = {
    cue.SO3(0): jnp.ones((num_nodes, 32, 1)),   # 32x scalar
    cue.SO3(1): jnp.ones((num_nodes, 32, 3)),    # 32x vector
}
x1 = jax.tree.map(lambda v: rearrange(v, "n m i -> n i m"), node_feats)

# Spherical harmonics: (edges, ir.dim) -- no multiplicity dimension
sph = {
    cue.SO3(0): jnp.ones((num_edges, 1)),
    cue.SO3(1): jnp.ones((num_edges, 3)),
}

# Build output template
senders = jax.random.randint(key, (num_edges,), 0, num_nodes)
receivers = jax.random.randint(key, (num_edges,), 0, num_nodes)
irreps_out = e.outputs[0].irreps
out_template = {
    ir: jax.ShapeDtypeStruct(
        (num_nodes, desc.num_segments) + desc.segment_shape, w.dtype
    )
    for (_, ir), desc in zip(irreps_out, poly.outputs)
}

# Execute with gather (senders) and scatter (receivers)
y = cuex.ir_dict.segmented_polynomial_uniform_1d(
    poly,
    [w, x1, sph],
    out_template,
    input_indices=[None, senders, None],
    output_indices=receivers,
    name="tensor_product",
)
# y is dict[Irrep, Array] with accumulated results at receiver nodes
```

### ir_dict utility functions

```python
# Validate dict matches irreps
cuex.ir_dict.assert_mul_ir_dict(irreps, x)  # asserts shape (..., mul, ir.dim)

# Convert flat array <-> dict
d = cuex.ir_dict.flat_to_dict(irreps, flat_array)           # layout="mul_ir" default
d = cuex.ir_dict.flat_to_dict(irreps, flat_array, layout="ir_mul")
flat = cuex.ir_dict.dict_to_flat(irreps, d)

# Arithmetic
z = cuex.ir_dict.irreps_add(x, y)
z = cuex.ir_dict.irreps_zeros_like(x)

# Create template dict
template = cuex.ir_dict.mul_ir_dict(irreps, jax.ShapeDtypeStruct(shape, dtype))
```

## NNX layers

### IrrepsLinear

Equivariant linear layer using `dict[Irrep, Array]`:

```python
from flax import nnx

linear = cuex.nnx.IrrepsLinear(
    irreps_in=cue.Irreps(cue.SO3, "4x0 + 2x1").regroup(),   # must be regrouped
    irreps_out=cue.Irreps(cue.SO3, "3x0 + 5x1").regroup(),
    scale=1.0,
    dtype=jnp.float32,
    rngs=nnx.Rngs(0),
)

# Input/output: dict[Irrep, Array] with shape (batch, mul, ir.dim)
x = {
    cue.SO3(0): jnp.ones((batch, 4, 1)),
    cue.SO3(1): jnp.ones((batch, 2, 3)),
}
y = linear(x)
# y[cue.SO3(0)].shape == (batch, 3, 1)
# y[cue.SO3(1)].shape == (batch, 5, 3)
```

Implementation uses `jnp.einsum("uv,...ui->...vi", w, x[ir])` per irrep with `1/sqrt(mul_in)` normalization.

### SphericalHarmonics

```python
sh = cuex.nnx.SphericalHarmonics(max_degree=3, eps=0.0)

vectors = jax.random.normal(key, (batch, 3))  # 3D vectors
y = sh(vectors)
# y[cue.O3(0, 1)].shape == (batch, 1, 1)   # L=0
# y[cue.O3(1, -1)].shape == (batch, 1, 3)  # L=1
# y[cue.O3(2, 1)].shape == (batch, 1, 5)   # L=2
# y[cue.O3(3, -1)].shape == (batch, 1, 7)  # L=3
```

### IrrepsNormalize

```python
norm = cuex.nnx.IrrepsNormalize(eps=1e-6, scale=1.0, skip_scalars=True)
y = norm(x)  # normalizes non-scalar irreps by RMS over ir.dim, averaged over mul
```

### MLP (scalar only)

```python
mlp = cuex.nnx.MLP(
    layer_sizes=[64, 128, 64],
    activation=jax.nn.silu,
    output_activation=False,
    dtype=jnp.float32,
    rngs=nnx.Rngs(0),
)
y = mlp(x_scalar)  # standard dense MLP with 1/sqrt(fan_in) normalization
```

### IrrepsIndexedLinear

For species-indexed linear layers (different weights per atom type):

```python
indexed_linear = cuex.nnx.IrrepsIndexedLinear(
    irreps_in=cue.Irreps(cue.O3, "8x0e").regroup(),
    irreps_out=cue.Irreps(cue.O3, "16x0e").regroup(),
    num_indices=50,   # number of species
    scale=1.0,
    dtype=jnp.float32,
    rngs=nnx.Rngs(0),
)

# num_index_counts: how many atoms of each species
species_counts = jnp.array([3, 4, 3, ...])  # sum = batch_size
y = indexed_linear(x, species_counts)
```

Uses `method="indexed_linear"` internally with `cuex.Repeats`.

## Preparing polynomials for uniform_1d

The `uniform_1d` CUDA kernel requires:
1. All segments within each operand have **the same shape**
2. A **single mode** in the subscripts (after preprocessing)

### From EquivariantPolynomial to uniform_1d-ready

For `equivariant_polynomial()` (RepArray interface):

```python
e = cue.descriptors.channelwise_tensor_product(...)
e = e.squeeze_modes().flatten_coefficient_modes()
# If still >1 mode: e = e.flatten_modes(["u", "w"])
out = cuex.equivariant_polynomial(e, inputs, method="uniform_1d")
```

For `ir_dict` (dict[Irrep, Array] interface):

```python
e = cue.descriptors.channelwise_tensor_product(..., simplify_irreps3=True)
poly = (
    e.split_operand_by_irrep(2)   # split input 2
     .split_operand_by_irrep(1)   # split input 1
     .split_operand_by_irrep(-1)  # split output
     .polynomial
)
# Now each operand has uniform segments -> use ir_dict.segmented_polynomial_uniform_1d
```

### Why split_operand_by_irrep works

Before splitting, a polynomial has dense operands like `32x0+32x1` with segments `((1,32), (3,32))` -- non-uniform shapes. After `split_operand_by_irrep`, each piece has a single irrep type, so all segments within that operand are the same shape. The pytree structure of `dict[Irrep, Array]` maps naturally to the split operands.

## RepArray

Representation-aware JAX array:

```python
rep = cue.IrrepsAndLayout(cue.Irreps("SO3", "4x0 + 2x1"), cue.ir_mul)
x = cuex.RepArray(rep, jnp.ones((batch, rep.dim)))
x = cuex.randn(jax.random.key(0), rep, (batch,), jnp.float32)

x.array   # raw jax.Array
x.reps    # {axis: Rep}
x.irreps  # Irreps (if last axis is IrrepsAndLayout)
```

## Complete GNN message-passing example

This pattern is used in NequIP, MACE, and similar equivariant GNN models:

```python
class MessagePassing(nnx.Module):
    def __init__(self, irreps_in, irreps_sh, irreps_out, epsilon, *, name, dtype, rngs):
        e = (
            cue.descriptors.channelwise_tensor_product(
                irreps_in, irreps_sh, irreps_out, True
            )
            * epsilon
        )
        self.weight_numel = e.inputs[0].dim
        self.irreps_out = e.outputs[0].irreps
        self.poly = (
            e.split_operand_by_irrep(2)
             .split_operand_by_irrep(1)
             .split_operand_by_irrep(-1)
             .polynomial
        )

    def __call__(self, weights, node_feats, sph, senders, receivers, num_nodes):
        # weights: (num_edges, weight_numel)
        w = rearrange(weights, "e (s m) -> e s m", s=self.poly.inputs[0].num_segments)
        # node_feats: dict[Irrep, Array] with (nodes, mul, ir.dim)
        x1 = jax.tree.map(lambda v: rearrange(v, "n m i -> n i m"), node_feats)
        # sph: dict[Irrep, Array] with (edges, 1, ir.dim) or (edges, ir.dim)
        x2 = jax.tree.map(lambda v: rearrange(v, "e 1 i -> e i"), sph)

        out_template = {
            ir: jax.ShapeDtypeStruct(
                (num_nodes, desc.num_segments) + desc.segment_shape, w.dtype
            )
            for (_, ir), desc in zip(self.irreps_out, self.poly.outputs)
        }

        y = cuex.ir_dict.segmented_polynomial_uniform_1d(
            self.poly, [w, x1, x2], out_template,
            input_indices=[None, senders, None],
            output_indices=receivers,
            name="tensor_product",
        )
        # Rearrange output back to (nodes, mul, ir.dim) for downstream layers
        return {
            ir: rearrange(v, "n (i s) m -> n (s m) i", i=ir.dim)
            for ir, v in y.items()
        }
```

## Key file locations

| Component | Path |
|-----------|------|
| `segmented_polynomial` primitive | `cuequivariance_jax/segmented_polynomials/segmented_polynomial.py` |
| `uniform_1d` backend | `cuequivariance_jax/segmented_polynomials/segmented_polynomial_uniform_1d.py` |
| `naive` backend | `cuequivariance_jax/segmented_polynomials/segmented_polynomial_naive.py` |
| `indexed_linear` backend | `cuequivariance_jax/segmented_polynomials/segmented_polynomial_indexed_linear.py` |
| `equivariant_polynomial` | `cuequivariance_jax/equivariant_polynomial.py` |
| `ir_dict` module | `cuequivariance_jax/ir_dict.py` |
| `nnx` module | `cuequivariance_jax/nnx.py` |
| `RepArray` | `cuequivariance_jax/rep_array/rep_array_.py` |
| `Repeats` / utilities | `cuequivariance_jax/segmented_polynomials/utils.py` |
| NequIP example | `cuequivariance_jax/examples/nequip_nnx.py` |
| MACE example | `cuequivariance_jax/examples/mace_nnx.py` |
