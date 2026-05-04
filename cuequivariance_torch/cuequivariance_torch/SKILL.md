---
name: cuequivariance-torch
description: Execute equivariant tensor products in PyTorch using SegmentedPolynomial (naive/uniform_1d/fused_tp/indexed_linear), high-level operations (ChannelWiseTensorProduct, FullyConnectedTensorProduct, Linear, SymmetricContraction, SphericalHarmonics, Rotation), and layers (BatchNorm, FullyConnectedTensorProductConv). Use when writing PyTorch code with cuequivariance.
---

# cuequivariance_torch: Executing Equivariant Polynomials in PyTorch

## Overview

`cuequivariance_torch` (imported as `cuet`) executes `cuequivariance` polynomials on GPU via PyTorch. It provides:

1. **Core primitive**: `cuet.SegmentedPolynomial` — `torch.nn.Module` with multiple CUDA backends
2. **High-level operations** (`torch.nn.Module`): `ChannelWiseTensorProduct`, `FullyConnectedTensorProduct`, `Linear`, `SymmetricContraction`, `SphericalHarmonics`, `Rotation`, `Inversion`
3. **Layers**: `cuet.layers.BatchNorm`, `cuet.layers.FullyConnectedTensorProductConv` (message passing)
4. **Utilities**: `triangle_attention`, `triangle_multiplicative_update`, `attention_pair_bias` (AlphaFold2-style)
5. **Export support**: `onnx_custom_translation_table()`, `register_tensorrt_plugins()`

## Execution methods

| Method | Backend | Requirements |
|--------|---------|-------------|
| `"naive"` | Pure PyTorch (einsum) | Always works, any platform |
| `"uniform_1d"` | CUDA kernel | GPU, all segments uniform shape within each operand, single mode |
| `"fused_tp"` | CUDA kernel | GPU, 3- or 4-operand contractions, float32/float64 |
| `"indexed_linear"` | CUDA kernel | GPU, linear with indexed weights, sorted indices |

## Core primitive: SegmentedPolynomial

```python
import torch
import cuequivariance as cue
import cuequivariance_torch as cuet

# Build a descriptor
e = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2])
poly = e.polynomial

# Create the module
sp = cuet.SegmentedPolynomial(poly, method="uniform_1d")

# Forward pass
x = torch.randn(batch, 3, device="cuda")
[output] = sp([x])
# output.shape == (batch, 9)  -- 1 + 3 + 5
```

### Inputs, indexing, and scatter

```python
e = cue.descriptors.channelwise_tensor_product(
    16 * cue.Irreps("SO3", "0 + 1"),
    cue.Irreps("SO3", "0 + 1"),
    cue.Irreps("SO3", "0 + 1"),
)
poly = e.polynomial

sp = cuet.SegmentedPolynomial(poly, method="uniform_1d")

w = torch.randn(1, poly.inputs[0].size, device="cuda")            # shared weights
x1 = torch.randn(batch, poly.inputs[1].size, device="cuda")       # batched input 1
x2 = torch.randn(batch, poly.inputs[2].size, device="cuda")       # batched input 2

# Basic forward
[out] = sp([w, x1, x2])

# With input gathering (e.g., gather x1 by node index)
senders = torch.randint(0, num_nodes, (num_edges,), device="cuda")
[out] = sp([w, x1, x2], input_indices={1: senders})

# With output scattering (accumulate into target nodes)
receivers = torch.randint(0, num_nodes, (num_edges,), device="cuda")
[out] = sp(
    [w, x1, x2],
    input_indices={1: senders},
    output_indices={0: receivers},
    output_shapes={0: torch.empty(num_nodes, 1, device="cuda")},
)
```

### Math dtype control

```python
# Compute in float32 regardless of input dtype
sp = cuet.SegmentedPolynomial(poly, method="fused_tp", math_dtype=torch.float32)

# For fused_tp, math_dtype must be float32 or float64
# For naive, any torch.dtype works
# For uniform_1d, float32 or float64 (auto-selects float32 if input is e.g. float16)
```

## High-level operations

All operations are `torch.nn.Module` subclasses. They wrap `SegmentedPolynomial` and handle layout transposition automatically.

### Memory layout

`IrrepsLayout` controls memory order within each `(mul, ir)` block:

- `cue.mul_ir`: data ordered as `(mul, ir.dim)` — **default, compatible with e3nn**
- `cue.ir_mul`: data ordered as `(ir.dim, mul)` — **used internally by descriptors**

Operations accept `layout` (applies to all), or per-operand `layout_in1`, `layout_in2`, `layout_out`.

### ChannelWiseTensorProduct

Channel-wise tensor product: pairs channels of `x1` with channels of `x2`.

```python
# With internal weights (default: shared_weights=True, internal_weights=True)
tp = cuet.ChannelWiseTensorProduct(
    cue.Irreps("SO3", "32x0 + 32x1"),   # irreps_in1
    cue.Irreps("SO3", "0 + 1"),          # irreps_in2
    layout=cue.mul_ir,
    device="cuda",
    dtype=torch.float32,
)
# tp.weight_numel -- number of weight parameters
# tp.irreps_out -- output irreps (auto-computed)

x1 = torch.randn(batch, tp.irreps_in1.dim, device="cuda")
x2 = torch.randn(batch, tp.irreps_in2.dim, device="cuda")

out = tp(x1, x2)  # uses internal weight parameter
# out.shape == (batch, tp.irreps_out.dim)

# With external weights (shared_weights=False)
tp = cuet.ChannelWiseTensorProduct(
    cue.Irreps("SO3", "32x0 + 32x1"),
    cue.Irreps("SO3", "0 + 1"),
    layout=cue.mul_ir,
    shared_weights=False,
    device="cuda",
)
w = torch.randn(batch, tp.weight_numel, device="cuda")
out = tp(x1, x2, weight=w)

# With gather/scatter for graph neural networks
out = tp(x1, x2, weight=w, indices_1=senders, indices_out=receivers, size_out=num_nodes)
```

Default method: `"uniform_1d"` if segments are uniform, else `"naive"`.

### FullyConnectedTensorProduct

All input irrep pairs contribute to all output irreps (dense contraction).

```python
tp = cuet.FullyConnectedTensorProduct(
    cue.Irreps("O3", "4x0e + 4x1o"),    # irreps_in1
    cue.Irreps("O3", "0e + 1o"),         # irreps_in2
    cue.Irreps("O3", "4x0e + 4x1o"),    # irreps_out
    layout=cue.mul_ir,
    internal_weights=True,
    device="cuda",
)

out = tp(x1, x2)  # uses internal weights
# or: out = tp(x1, x2, weight=w)  # external weights
```

Default method: `"fused_tp"`.

### Linear

Equivariant linear layer (weight-only, no second input).

```python
linear = cuet.Linear(
    cue.Irreps("SO3", "4x0 + 2x1"),     # irreps_in
    cue.Irreps("SO3", "3x0 + 5x1"),     # irreps_out
    layout=cue.mul_ir,
    internal_weights=True,
    device="cuda",
)

out = linear(x)

# Species-indexed weights (different weights per atom type)
linear = cuet.Linear(
    irreps_in, irreps_out,
    weight_classes=50,   # 50 different weight sets
    internal_weights=True,
    device="cuda",
)
out = linear(x, weight_indices=species_indices)  # species_indices: (batch,) int tensor
```

Default method: `"naive"`. Use `method="fused_tp"` for CUDA acceleration.

### SymmetricContraction

MACE-style symmetric contraction with element-indexed weights.

```python
sc = cuet.SymmetricContraction(
    cue.Irreps("O3", "32x0e + 32x1o"),  # irreps_in (uniform mul required)
    cue.Irreps("O3", "32x0e"),           # irreps_out (uniform mul required)
    contraction_degree=3,
    num_elements=95,                      # number of chemical elements
    layout=cue.ir_mul,
    dtype=torch.float32,
    device="cuda",
)

# indices: (batch,) int tensor selecting which element weights to use
out = sc(x, indices)
# out.shape == (batch, irreps_out.dim)
```

Default method: `"uniform_1d"` if segments are uniform, else `"naive"`.

### SphericalHarmonics

```python
sh = cuet.SphericalHarmonics(
    ls=[0, 1, 2, 3],
    normalize=True,
    device="cuda",
)

vectors = torch.randn(batch, 3, device="cuda")
out = sh(vectors)
# out.shape == (batch, 1 + 3 + 5 + 7)  -- sum of 2l+1
```

Default method: `"uniform_1d"`.

### Rotation and Inversion

```python
# Rotation (SO3 or O3 irreps)
rot = cuet.Rotation(
    cue.Irreps("SO3", "4x0 + 2x1 + 1x2"),
    layout=cue.ir_mul,
    device="cuda",
)

# Euler angles (YXY convention)
gamma = torch.tensor([0.1], device="cuda")
beta = torch.tensor([0.2], device="cuda")
alpha = torch.tensor([0.3], device="cuda")
out = rot(gamma, beta, alpha, x)

# Helper: encode angle for rotation
encoded = cuet.encode_rotation_angle(angle, ell=3)  # cos/sin encoding

# Helper: 3D vector to Euler angles
beta, alpha = cuet.vector_to_euler_angles(vector)

# Inversion (O3 irreps only)
inv = cuet.Inversion(
    cue.Irreps("O3", "4x0e + 2x1o"),
    layout=cue.ir_mul,
    device="cuda",
)
out = inv(x)
```

## Layers

### BatchNorm

Batch normalization for equivariant representations (adapted from e3nn).

```python
bn = cuet.layers.BatchNorm(
    cue.Irreps("O3", "4x0e + 4x1o"),
    layout=cue.mul_ir,
    eps=1e-5,
    momentum=0.1,
    affine=True,
)

out = bn(x)  # x.shape == (batch, irreps.dim)
```

### FullyConnectedTensorProductConv

Message passing layer for equivariant GNNs (DiffDock-style).

```python
conv = cuet.layers.FullyConnectedTensorProductConv(
    in_irreps=cue.Irreps("O3", "4x0e + 4x1o"),
    sh_irreps=cue.Irreps("O3", "0e + 1o"),
    out_irreps=cue.Irreps("O3", "4x0e + 4x1o"),
    mlp_channels=[16, 32, 32],
    mlp_activation=torch.nn.ReLU(),
    batch_norm=True,
    layout=cue.ir_mul,
)

# graph = ((src, dst), (num_src_nodes, num_dst_nodes))
graph = ((src, dst), (num_src_nodes, num_dst_nodes))

out = conv(src_features, edge_sh, edge_emb, graph, reduce="mean")
# out.shape == (num_dst_nodes, out_irreps.dim)

# Optional: separate scalar features for efficient first-layer GEMM
out = conv(src_features, edge_sh, edge_emb, graph,
           src_scalars=src_scalars, dst_scalars=dst_scalars)
```

## Triangle operations (AlphaFold2-style)

Require `cuequivariance_ops_torch`.

```python
# Triangle attention with pair bias
out = cuet.triangle_attention(q, k, v, bias, mask=mask, scale=scale)
# q, k, v: (B, N, H, Q/K, D), bias: (B, 1, H, Q, K)

# Triangle multiplicative update
out = cuet.triangle_multiplicative_update(
    x,         # (B, I, J, C)
    mask=mask, # (B, I, J)
    precision=cuet.TriMulPrecision.DEFAULT,
)

# Attention with pair bias (diffusion models)
out = cuet.attention_pair_bias(q, k, v, bias, mask=mask)
```

## ONNX and TensorRT export

```python
# ONNX export
table = cuet.onnx_custom_translation_table()
onnx_program = torch.onnx.export(model, inputs, custom_translation_table=table)

# TensorRT plugin registration
cuet.register_tensorrt_plugins()
```

## Complete GNN example

```python
import torch
import cuequivariance as cue
import cuequivariance_torch as cuet

class SimpleGNN(torch.nn.Module):
    def __init__(self, irreps_in, irreps_sh, irreps_out):
        super().__init__()
        self.tp = cuet.ChannelWiseTensorProduct(
            irreps_in, irreps_sh, layout=cue.mul_ir,
            shared_weights=False, device="cuda",
        )
        self.linear = cuet.Linear(
            self.tp.irreps_out, irreps_out,
            layout=cue.mul_ir, internal_weights=True, device="cuda",
        )
        self.sh = cuet.SphericalHarmonics(
            ls=[ir.l for _, ir in irreps_sh], normalize=True, device="cuda",
        )

    def forward(self, node_feats, edge_vec, edge_index, num_nodes):
        src, dst = edge_index
        edge_sh = self.sh(edge_vec)
        w = torch.randn(1, self.tp.weight_numel, device=node_feats.device)

        # Message: tensor product on edges with scatter to destination nodes
        messages = self.tp(
            node_feats, edge_sh, weight=w,
            indices_1=src, indices_2=None,
            indices_out=dst, size_out=num_nodes,
        )
        return self.linear(messages)
```

## Key file locations

| Component | Path |
|-----------|------|
| `SegmentedPolynomial` | `cuequivariance_torch/primitives/segmented_polynomial.py` |
| `uniform_1d` backend | `cuequivariance_torch/primitives/segmented_polynomial_uniform_1d.py` |
| `naive` backend | `cuequivariance_torch/primitives/segmented_polynomial_naive.py` |
| `fused_tp` backend | `cuequivariance_torch/primitives/segmented_polynomial_fused_tp.py` |
| `indexed_linear` backend | `cuequivariance_torch/primitives/segmented_polynomial_indexed_linear.py` |
| `ChannelWiseTensorProduct` | `cuequivariance_torch/operations/tp_channel_wise.py` |
| `FullyConnectedTensorProduct` | `cuequivariance_torch/operations/tp_fully_connected.py` |
| `Linear` | `cuequivariance_torch/operations/linear.py` |
| `SymmetricContraction` | `cuequivariance_torch/operations/symmetric_contraction.py` |
| `SphericalHarmonics` | `cuequivariance_torch/operations/spherical_harmonics.py` |
| `Rotation` / `Inversion` | `cuequivariance_torch/operations/rotation.py` |
| `BatchNorm` | `cuequivariance_torch/layers/batchnorm.py` |
| `FullyConnectedTensorProductConv` | `cuequivariance_torch/layers/tp_conv_fully_connected.py` |
| Triangle operations | `cuequivariance_torch/primitives/triangle.py` |
| Layout transposition | `cuequivariance_torch/primitives/transpose.py` |
