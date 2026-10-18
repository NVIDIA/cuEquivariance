---
name: cuequivariance-torch
description: Use when building or debugging cuEquivariance PyTorch tensor products or equivariant GNNs, especially CPU references, layouts, indexing, and gradients.
license: Apache-2.0
metadata:
  author: Mario Geiger <mgeiger@nvidia.com>
  tags: [pytorch, equivariance, tensor-products]
---

# cuEquivariance in PyTorch

## Purpose

Use `cuequivariance_torch` (`cuet`) to execute `cuequivariance` (`cue`) descriptors with PyTorch autograd. This guide covers CPU reference calculations, CUDA backend selection, and equivariant layers. Use the project's existing framework and environment.

## Inputs

Required for the computation: operation, SO3/O3 irreps, layout, device, dtype, and weight ownership. Derive these from the request or existing code and state any defaults. Graph operations also require source/destination counts and node/edge indexing. Optional inputs include a preferred backend or performance/export targets. Preserve the user's project constraints and installed API.

## Instructions

1. Check the selected interpreter and imports once using the probe below. Reuse a working environment. A missing CUDA backend is usually a method-selection issue, not a reason to reinstall PyTorch.
2. Set `method="naive"` for CPU calculations. Match module parameters, features, weights, and reference computations in device and dtype. Use float64 for small numerical and gradient checks.
3. Specify layout and weight ownership explicitly. Derive sizes and output order from the module/descriptor; do not guess multiplicities from a filter.
4. Adapt the relevant example to the task and run its requested checks together. Report the interpreter, shapes, numerical error, and gradient results actually observed. Investigate a failed check before broadening the implementation.
5. If an API differs from this guide, inspect that installed object's signature and source once. Adapt to the installed version or explain the required version; do not silently replace cuEquivariance with another library.

## Prerequisites and environment reuse

Run this with the project's Python interpreter (including its existing virtual environment, if any). Check imports independently so one missing package does not hide a working CPU PyTorch:

```python
import importlib
import sys

print("python:", sys.executable)
available, import_errors = {}, {}
for name in ("torch", "cuequivariance", "cuequivariance_torch"):
    try:
        module = importlib.import_module(name)
        available[name] = module
        print(name, getattr(module, "__version__", "unknown"), module.__file__)
    except (ImportError, OSError) as error:
        import_errors[name] = f"{type(error).__name__}: {error}"
        print(name, import_errors[name])
if "torch" in available:
    torch = available["torch"]
    print("CUDA build:", torch.version.cuda, "available:", torch.cuda.is_available())
```

Choose the next step from the probe:

- All imports work: continue with this interpreter and `method="naive"` for CPU; install nothing.
- PyTorch works but cuEquivariance is missing: use another existing project interpreter if it has both. Otherwise install only the missing cuEquivariance packages into the intended environment, honoring its version constraints. `cuequivariance-torch` is the distribution name; `cuequivariance_torch` is the import name.
- An import reports a missing shared library or binary mismatch: use the reported package/path to diagnose compatibility. This differs from an absent package; repeated installations or a backend change may not fix it.

Use the same interpreter for pip and execution. Installing cuEquivariance does not require replacing an already working CPU PyTorch. Optional CUDA/optimization warnings alone are not installation failures.

If no usable project environment is available, or it cannot be modified normally, create a **new** environment when installation is appropriate. Preserve existing environments and system-package protection. On Linux/macOS, this fallback also works when `venv` cannot bootstrap pip through `ensurepip` (the host needs pip 22.3+):

```bash
cueq_env=$(mktemp -d "${TMPDIR:-/tmp}/cueq-cpu.XXXXXX")
python3 -m venv --without-pip "$cueq_env"
python3 -m pip --python "$cueq_env" install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip --python "$cueq_env" install cuequivariance-torch
"$cueq_env/bin/python" -c 'import torch, cuequivariance_torch; print(torch.__version__)'
```

Honor project version/source constraints and keep using the new environment's Python explicitly. Stop at the first failed setup command and diagnose its error; preserve the failure output. If pip is unavailable, use an existing manager such as `uv` or report the missing prerequisite. See [pip's interpreter selection](https://pip.pypa.io/en/stable/topics/python-option/).

## Limitations and backends

| Method | Use and constraints |
| --- | --- |
| `naive` | Pure PyTorch reference implementation; supported CPU path and autograd. |
| `uniform_1d` | CUDA; uniform segment shapes within each operand and a single mode after descriptor transformations. |
| `fused_tp` | CUDA contractions with 3 or 4 operands; math dtype float32/float64. Some versions fall back on CPU, but select `naive` explicitly for a CPU reference. |
| `indexed_linear` | CUDA indexed linear operations; indices must be sorted. |

High-level defaults vary: `SphericalHarmonics` selects `uniform_1d`, `FullyConnectedTensorProduct` selects `fused_tp`, `Linear` selects `naive`, and `ChannelWiseTensorProduct`/`SymmetricContraction` select according to descriptor compatibility. Moving tensors to CPU does not change every constructor's backend choice. CUDA acceleration requires compatible PyTorch, GPU, and `cuequivariance_ops_torch` builds; CPU checks do not establish GPU performance or export compatibility.

## Layouts and weights

- `cue.mul_ir` is the high-level default, compatible with e3nn: each block is `(multiplicity, irrep_dimension)`. Core descriptors commonly use `cue.ir_mul`: `(irrep_dimension, multiplicity)`. Check each descriptor operand's `.layout`.
- Convert **each irrep block**, not the whole flattened tensor with a reshape. `cuet.TransposeIrrepsLayout(irreps, source=..., target=..., use_fallback=True)` performs the blockwise conversion using PyTorch.
- For externally supplied weights set `internal_weights=False`. Shared weights have shape `(1, module.weight_numel)`; distinct weights per item have shape `(batch, module.weight_numel)` and require `shared_weights=False`. A gathered graph operation uses the **edge count** for this batch dimension.
- `filter_irreps_out` selects allowed irrep types, not a requested output representation with prescribed multiplicities. Read `module.irreps_out`; preserve its actual order when constructing the next layer or reference.
- For training, create learned parameters or an edge-weight network in `__init__`. Do not draw new random weights inside `forward()`.

## Examples

The CPU examples below share these imports. Replace the small representations and tensors with the task's inputs; keep the layout, indexing, and gradient contracts.

```python
import numpy as np
import torch
import cuequivariance as cue
import cuequivariance_torch as cuet

torch.manual_seed(0)
dtype = torch.float64
```

### Spherical harmonics and a NumPy reference

`normalize=False` evaluates the raw polynomial; `normalize=True` first normalizes the input vectors. Match that choice in the reference and use nonzero vectors for gradient checks.

```python
ls = [0, 1, 2]
sh = cuet.SphericalHarmonics(ls, normalize=False, method="naive")
vectors = torch.randn(2, 3, dtype=dtype, requires_grad=True)
values = sh(vectors)
descriptor = cue.descriptors.spherical_harmonics(cue.SO3(1), ls)
np.testing.assert_allclose(
    values.detach().numpy(), descriptor(vectors.detach().numpy())[0], atol=1e-12
)
assert values.shape == (2, sum(2 * l + 1 for l in ls))
assert torch.autograd.gradcheck(sh, (vectors,))
```

### Linear: layout conversion and rotations

Use identical explicit weights for a layout comparison. For row-vector features, apply the representation matrix as `x @ D.T`.

```python
irreps_in = cue.Irreps("SO3", "3x0 + 2x1")
irreps_out = cue.Irreps("SO3", "2x0 + 4x1")
options = dict(internal_weights=False, method="naive", dtype=dtype)
linear = cuet.Linear(irreps_in, irreps_out, layout=cue.mul_ir, **options)
linear_ir = cuet.Linear(irreps_in, irreps_out, layout=cue.ir_mul, **options)
to_ir = cuet.TransposeIrrepsLayout(
    irreps_in, source=cue.mul_ir, target=cue.ir_mul, use_fallback=True
)
to_mul = cuet.TransposeIrrepsLayout(
    irreps_out, source=cue.ir_mul, target=cue.mul_ir, use_fallback=True
)
x = torch.randn(2, irreps_in.dim, dtype=dtype, requires_grad=True)
w = torch.randn(1, linear.weight_numel, dtype=dtype, requires_grad=True)
y = linear(x, w)
torch.testing.assert_close(y, to_mul(linear_ir(to_ir(x), w)))
rotation = np.array([0.2, -0.3, 0.1])
din, dout = [torch.from_numpy(cue.IrrepsAndLayout(r, cue.mul_ir).exp_map(rotation, []))
             for r in (irreps_in, irreps_out)]
torch.testing.assert_close(linear(x @ din.T, w), y @ dout.T)
for g in torch.autograd.grad(y.square().sum(), (x, w)):
    assert torch.isfinite(g).all() and g.abs().sum() > 0
```

### Channelwise product with external per-example weights

The core descriptor takes `(weights, input1, input2)` and returns a list of outputs. The high-level module takes `(input1, input2, weight=...)` and returns a tensor. Matching `ir_mul` layouts avoids an extra transpose in this reference.

```python
a, b = cue.Irreps("SO3", "3x0 + 3x1"), cue.Irreps("SO3", "0 + 1")
keep = [cue.SO3(0), cue.SO3(1)]
tp = cuet.ChannelWiseTensorProduct(
    a, b, keep, layout=cue.ir_mul, shared_weights=False,
    internal_weights=False, method="naive", dtype=dtype,
)
x1, x2, w = [torch.randn(2, n, dtype=dtype, requires_grad=True)
             for n in (a.dim, b.dim, tp.weight_numel)]
y = tp(x1, x2, weight=w)
descriptor = cue.descriptors.channelwise_tensor_product(a, b, keep)
reference = descriptor(*(t.detach().numpy() for t in (w, x1, x2)))[0]
np.testing.assert_allclose(y.detach().numpy(), reference, atol=1e-12)
assert y.shape == (2, tp.irreps_out.dim)
for g in torch.autograd.grad(y.square().sum(), (x1, x2, w)):
    assert torch.isfinite(g).all() and g.abs().sum() > 0
```

For a fully connected product, specify the desired output irreps in `cuet.FullyConnectedTensorProduct(a, b, irreps_out, method="naive", dtype=dtype)`. Its default weights are internal trainable parameters. Use the same external-weight flags when the caller or an edge network owns the weights.

### SegmentedPolynomial: gather and scatter

Input/output index keys refer to **operand positions**, not feature channels. Supply `torch.int32` or `torch.int64` indices on the computation's device. `output_shapes` takes a two-dimensional **tensor**, whose first dimension determines the destination count; it does not take a shape tuple. Repeated receivers sum, and destinations without incoming edges remain zero.

```python
descriptor = cue.descriptors.linear(cue.Irreps("SO3", "3x0"), cue.Irreps("SO3", "2x0"))
sp = cuet.SegmentedPolynomial(descriptor.polynomial, method="naive")
nodes = torch.randn(5, 3, dtype=dtype, requires_grad=True)
w = torch.randn(1, descriptor.inputs[0].dim, dtype=dtype, requires_grad=True)
senders = torch.tensor([0, 2, 1, 0], dtype=torch.int64)
receivers = torch.tensor([1, 0, 1, 2], dtype=torch.int64)
[actual] = sp([w, nodes], input_indices={1: senders},
              output_indices={0: receivers}, output_shapes={0: nodes.new_empty(4, 1)})
[edges] = sp([w, nodes[senders]])
reference = nodes.new_zeros(4, descriptor.outputs[0].dim).index_add_(0, receivers, edges)
torch.testing.assert_close(actual, reference)
assert torch.count_nonzero(actual[3]) == 0
grads = torch.autograd.grad(actual.square().sum(), (nodes, w))
ref_grads = torch.autograd.grad(reference.square().sum(), (nodes, w))
for g, ref in zip(grads, ref_grads):
    torch.testing.assert_close(g, ref)
```

Reuse the descriptor to build a numerical reference: its coefficients include normalization factors, so a hand-written matrix product without those factors can be wrong. For high-level channelwise gathering/scattering, use `indices_1`, `indices_2`, `indices_out`, and `size_out` instead of the primitive's dictionaries.

### Trainable message passing

This CPU reference uses a radial MLP to produce one weight vector per edge. Its parameters persist across calls, and `irreps_sh` matches the single copy of each spherical harmonic degree.

```python
class SimpleGNN(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.tp = cuet.ChannelWiseTensorProduct(
            irreps_in, cue.Irreps("SO3", "0 + 1 + 2"), layout=cue.mul_ir,
            shared_weights=False, internal_weights=False, method="naive",
        )
        self.radial = torch.nn.Sequential(
            torch.nn.Linear(1, 16), torch.nn.SiLU(),
            torch.nn.Linear(16, self.tp.weight_numel),
        )
        self.sh = cuet.SphericalHarmonics([0, 1, 2], normalize=True, method="naive")
        self.linear = cuet.Linear(
            self.tp.irreps_out, irreps_out, layout=cue.mul_ir, method="naive"
        )

    def forward(self, node_features, edge_vectors, senders, receivers, num_dst):
        weights = self.radial(edge_vectors.norm(dim=-1, keepdim=True))
        messages = self.tp(
            node_features, self.sh(edge_vectors), weight=weights,
            indices_1=senders, indices_out=receivers, size_out=num_dst,
        )
        return self.linear(messages)
```

Move the whole module and inputs to the same dtype/device (for example `model.double()` for float64 checks). Use nonzero edge vectors when normalizing spherical harmonics. After a scalar loss/backward pass, check gradients on the radial network as well as node features; repeated forwards should agree when no stochastic layers are present.

## Other operations and layers

Inspect the requested operation's installed signature/source via `inspect.signature` and `inspect.getfile`, rather than loading the entire API. These are the non-obvious contracts:

| Operation | Contract |
| --- | --- |
| Species-indexed `Linear` | Set `weight_classes`; pass `weight_indices` of shape `(batch,)` and weights of shape `(weight_classes, weight_numel)`. |
| `SymmetricContraction` | MACE-style; uniform multiplicities, `contraction_degree`, `num_elements`, and integer element indices in `sc(x, indices)`. |
| `Rotation` / `Inversion` | `rot(gamma, beta, alpha, x)` uses YXY Euler angles; `inv(x)` uses O3 parity. Specify layout. Angle helpers: `encode_rotation_angle`, `vector_to_euler_angles`. |
| `layers.BatchNorm` | Equivariant normalization; configure irreps and layout. |
| `layers.FullyConnectedTensorProductConv` | Configure input/SH/output irreps and edge MLP; call with `(src_features, edge_sh, edge_emb, ((src, dst), (num_src, num_dst)))`. Supports `reduce="mean"` and optional `src_scalars`/`dst_scalars`. |

### Triangle operations and export

These paths require compatible optional GPU packages and their own shape checks. Do not install them for a CPU tensor-product request. Locate the installed `primitives/triangle.py` via `cuet.__file__`:

- `triangle_attention`: Q/K/V `(B,N,H,Q_or_K,D)`, bias `(B,1,H,Q,K)`. `mask_to_kv_lengths(prefix_mask)` supplies `kv_lengths`; arbitrary dense masks use `mask=...` and the fallback.
- `triangle_multiplicative_update`: input `(B,I,J,C)`, mask `(B,I,J)`, precision enum `cuet.TriMulPrecision`.
- `attention_pair_bias`: OpenFold3/Boltz contract, returns `(out, auxiliary)`; projection biases and query/key normalization keywords vary by version. Inspect its signature before assembling the weights.
- ONNX: pass `custom_translation_table=cuet.onnx_custom_translation_table()` to `torch.onnx.export`. TensorRT: `cuet.register_tensorrt_plugins()`. Validate the actual exported runtime.

## Troubleshooting

| Symptom | Check and recovery |
| --- | --- |
| Missing CUDA kernels on CPU | Pass `method="naive"` at module construction; retain the installed CPU PyTorch. |
| `opt_einsum_fx not available` warning | Optional optimization; continue the naive reference calculation without installing it. |
| Missing import after installation | Compare `sys.executable`, module paths, and the interpreter used with pip. Reuse the working interpreter. |
| `ensurepip` unavailable / externally managed Python | Use a fresh environment and host `pip --python` as above; preserve existing environments and system packages. |
| Unexpected keyword or missing descriptor method | Inspect the installed signature/version. Use its documented equivalent or explain the mismatch; avoid repeated blind API guesses. |
| Correct shapes but wrong numerical values | Check per-block layout, output irrep order, normalization, and the shared/per-item weight convention. |
| Scatter shape/error or wrong destination sums | Use operand index keys, integral indices, a tensor in `output_shapes`, and accumulation for repeated receivers. |
| Missing/nonfinite gradients | Keep tensor operations in autograd; detach only the NumPy reference. Use nonzero vectors and a nonconstant loss; check every requested differentiable operand. |
