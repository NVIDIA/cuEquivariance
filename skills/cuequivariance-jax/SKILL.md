---
name: cuequivariance-jax
description: Use when writing or debugging cuEquivariance JAX polynomials, irrep dictionaries, Flax NNX layers, or CPU/CUDA execution and layouts.
license: Apache-2.0
metadata:
  author: Mario Geiger <mgeiger@nvidia.com>
  tags: [jax, equivariance, tensor-products]
---

# cuEquivariance in JAX

## Purpose

Execute `cuequivariance` (`cue`) descriptors with `cuequivariance_jax` (`cuex`), including autodiff, JIT, irrep dictionaries, and Flax NNX layers. Preserve the user's framework, environment, and device constraints. Custom representations and Clebsch–Gordan coefficients belong to the core `cuequivariance` API; PyTorch modules use `cuequivariance_torch`.

## Inputs

Required inputs are the operation, SO3/O3 irreps, layout, batch shapes, dtype, device, and weight ownership; derive them from the request or existing code and state defaults. Graph operations also need sender/receiver indices and destination count. Optional inputs include a preferred backend or performance target. Derive operand sizes and output irreps from descriptors rather than guessing multiplicities.

## Instructions

1. Probe the selected interpreter once; reuse working packages. Choose `method="naive"` for CPU. Install only missing requirements, with Flax needed only for NNX layers.
2. Choose dense `RepArray` or `dict[Irrep, Array]`. High-level dictionary blocks are `(..., mul, ir.dim)`; the low-level uniform CUDA wrapper has a different shape contract.
3. Build the descriptor outside transformed functions. Derive output specs from its outputs and the broadcast batch shape. Enable x64 **before creating arrays** for float64 numerical checks.
4. Run the relevant small example with the selected interpreter, then adapt it. Compare against NumPy with identical operands; check JIT or derivatives when relevant. Report executed shapes/devices and any checks that could not run.

## Prerequisites and environment recovery

CPU polynomial execution needs JAX, `cuequivariance`, and `cuequivariance-jax`; it needs no CUDA kernel package. Check imports independently so one missing package does not hide a working JAX installation:

```python
import importlib
import sys

print("python:", sys.executable)
available, import_errors = {}, {}
for name in ("jax", "cuequivariance", "cuequivariance_jax"):
    try:
        module = importlib.import_module(name)
        available[name] = module
        print(name, getattr(module, "__version__", "unknown"), module.__file__)
    except (ImportError, OSError) as error:
        import_errors[name] = f"{type(error).__name__}: {error}"
print("import errors:", import_errors)
if "jax" in available:
    print("devices:", available["jax"].devices())
```

If these imports work, continue without reinstalling. For NNX, additionally check `from flax import nnx`. A binary/version import error is different from an absent package: inspect that error and the project's dependency constraints before changing versions. In a checkout, follow its installation instructions; keep core and frontend source revisions compatible. When supplied, honor `scripts/requirements.txt` beside this skill or the project's lockfile instead of substituting latest releases.

If an isolated environment is needed, create a **new unique directory**, preserving existing environments and project files. On Linux, this recipe bypasses `ensurepip` and uses host pip 22.3+:

```bash
set -e
cueq_jax_env=$(mktemp -d "${TMPDIR:-/tmp}/cueq-jax.XXXXXX")
python3 -m venv --without-pip "$cueq_jax_env"
python3 -m pip --python "$cueq_jax_env" install "jax[cpu]" cuequivariance-jax
JAX_PLATFORMS=cpu "$cueq_jax_env/bin/python" -c 'import jax, cuequivariance_jax; print(jax.devices())'
```

Replace the install package list with `-r` and the supplied requirements path when versions are prescribed. Add Flax only for an NNX task unless those requirements already include it. Keep using the chosen interpreter explicitly. Stop on a setup failure and retain its error; do not delete `.venv` or change system packages to repair `ensurepip`. If host pip is unavailable, use an existing manager such as `uv`. See [pip's interpreter selection](https://pip.pypa.io/en/stable/topics/python-option/) and [JAX installation](https://docs.jax.dev/en/latest/installation.html).

## Limitations and backend selection

| Route | Supported use |
| --- | --- |
| `segmented_polynomial(..., method="naive")` | Portable JAX execution, including CPU, JIT, gradients, and vmap. |
| `cuex.nnx.IrrepsLinear` | CPU/GPU channel mixing within matching irreps using JAX einsum. |
| `cuex.nnx.SphericalHarmonics` | CPU/GPU; normalizes vectors and uses the naive polynomial implementation. |
| `method="uniform_1d"` | CUDA; uniform segment shapes within each operand and a single mode after preparation. |
| `method="indexed_linear"`, `cuex.nnx.IrrepsIndexedLinear` | CUDA indexed linear route, using `cuex.Repeats`; not the ordinary CPU linear layer. |

A CPU-only task should use the first three routes without installing GPU dependencies. CUDA execution requires compatible JAX, GPU, and kernel builds; CPU checks do not establish CUDA compatibility or performance. Do not infer a CPU fallback from moving arrays to CPU. Avoid zero input vectors with NNX spherical harmonics at `eps=0`, since normalization divides by their norm; choose a positive `eps` if zero vectors are possible.

## Examples

Each example below runs with this common preamble; the NNX example adds Flax. For a CPU check, launch the script with `JAX_PLATFORMS=cpu`. The examples use small nonconstant arrays so layout errors remain visible.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import cuequivariance as cue
import cuequivariance_jax as cuex
from cuequivariance_jax import ir_dict

dtype = jnp.float64
print("devices:", jax.devices())
```

### Spherical harmonics without Flax

The descriptor takes an input irrep and a list of degrees. This evaluates the **raw polynomial**; normalize vectors first when unit-direction harmonics are required. O3 keys retain parity, unlike SO3 keys.

```python
desc = cue.descriptors.spherical_harmonics_ir_dict(cue.O3(1, -1), [0, 1, 2])
vectors = jnp.array([[0.3, -0.7, 1.1], [1.2, 0.4, -0.2]], dtype=dtype)
values = cuex.segmented_polynomial(
    desc.polynomial, [vectors],
    [jax.ShapeDtypeStruct(vectors.shape[:-1] + (op.size,), dtype)
     for op in desc.polynomial.outputs],
    method="naive",
)
(irreps_out,) = desc.output_irreps
harmonics = {
    ir: value.reshape(vectors.shape[:-1] + (mul, ir.dim))
    for (mul, ir), value in zip(irreps_out, values)
}
for actual, expected in zip(values, desc.polynomial(np.asarray(vectors))):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
assert {ir.l: v.shape for ir, v in harmonics.items()} == {
    0: (2, 1, 1), 1: (2, 1, 3), 2: (2, 1, 5),
}
print({str(ir): (v.shape, v.devices()) for ir, v in harmonics.items()})
```

### Dictionary layouts and exact round trips

`flat_to_dict(..., layout="ir_mul")` transposes each block to `(..., mul, ir.dim)`. **`dict_to_flat` has no layout argument and always emits `mul_ir`.** Returning to `ir_mul` requires a transpose within every block; a whole-array reshape changes the meaning of the data.

**Regroup duplicate irreps before using irrep keys**, or later blocks overwrite earlier ones. Regroup the corresponding feature data too: changing only the irreps metadata does not reorder an existing buffer. A `RepArray` can regroup metadata and data together. Use descriptor/irreps order when assembling dictionaries and operands.

```python
def dict_to_ir_mul_flat(irreps, blocks):
    return jnp.concatenate([
        blocks[ir].swapaxes(-1, -2).reshape(
            blocks[ir].shape[:-2] + (mul * ir.dim,)
        )
        for mul, ir in irreps
    ], axis=-1)

irreps = cue.Irreps("SO3", "3x0 + 2x1").regroup()
flat_ir = jnp.arange(2 * irreps.dim).reshape(2, irreps.dim)
blocks = ir_dict.flat_to_dict(irreps, flat_ir, layout="ir_mul")
offset = 0
for mul, ir in irreps:
    size = mul * ir.dim
    expected = np.asarray(flat_ir[..., offset:offset + size]).reshape(2, ir.dim, mul)
    assert blocks[ir].shape == (2, mul, ir.dim)
    np.testing.assert_array_equal(blocks[ir], expected.swapaxes(-1, -2))
    offset += size
flat_mul = ir_dict.dict_to_flat(irreps, blocks)
restored_blocks = ir_dict.flat_to_dict(irreps, flat_mul, layout="mul_ir")
for ir in blocks:
    np.testing.assert_array_equal(restored_blocks[ir], blocks[ir])
np.testing.assert_array_equal(dict_to_ir_mul_flat(irreps, restored_blocks), flat_ir)
```

`ir_dict.assert_mul_ir_dict` checks block conventions; `irreps_add` and `irreps_zeros_like` operate on matching dictionaries. `mul_ir_dict(irreps, data)` broadcasts a template object to each key; it does not infer feature dimensions.

### Dense polynomial: broadcasting, JIT, and a gradient

`segmented_polynomial` takes a list of inputs and returns a list of outputs. Channelwise products put weights first. Shared weights need **no batch axis**; broadcast all input shapes except the final feature axis. `filter_irreps_out` selects irrep types, not output multiplicities. Dense descriptor operands carry their layout in `e.inputs` and `e.outputs`.

```python
e = cue.descriptors.channelwise_tensor_product(
    cue.Irreps("SO3", "3x0 + 3x1"), cue.Irreps("SO3", "0 + 1"),
    [cue.SO3(0), cue.SO3(1)],
)
poly = e.polynomial
rng = np.random.default_rng(7)
w = jnp.asarray(rng.normal(size=poly.inputs[0].size))
x = jnp.asarray(rng.normal(size=(3, 1, poly.inputs[1].size)))
y = jnp.asarray(rng.normal(size=(1, 2, poly.inputs[2].size)))

def run(weights, a, b):
    batch = np.broadcast_shapes(weights.shape[:-1], a.shape[:-1], b.shape[:-1])
    return cuex.segmented_polynomial(
        poly, [weights, a, b],
        [jax.ShapeDtypeStruct(batch + (op.size,), jnp.result_type(weights, a, b))
         for op in poly.outputs],
        method="naive",
    )[0]

reference = e(*(np.asarray(v) for v in (w, x, y)))[0]
np.testing.assert_allclose(run(w, x, y), reference, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(jax.jit(run)(w, x, y), reference, rtol=1e-12, atol=1e-12)
assert reference.shape == (3, 2, poly.outputs[0].size)
gradient = jax.grad(lambda v: jnp.sum(run(v, x, y) ** 2))(w)
index = int(jnp.argmax(jnp.abs(gradient)))
assert abs(gradient[index]) > 1e-8
epsilon = 1e-5
loss_plus = np.sum(e(np.asarray(w.at[index].add(epsilon)), np.asarray(x), np.asarray(y))[0] ** 2)
loss_minus = np.sum(e(np.asarray(w.at[index].add(-epsilon)), np.asarray(x), np.asarray(y))[0] ** 2)
np.testing.assert_allclose(gradient[index], (loss_plus - loss_minus) / (2 * epsilon),
                           rtol=1e-6, atol=1e-7)
```

For gather/scatter, supply one `indices` entry per polynomial operand, inputs then outputs; `None` leaves an operand unindexed. Output indices require an explicit destination shape. Repeated destinations accumulate. Compare against an edgewise calculation with explicit summation before optimizing a graph contraction.

### Split irrep polynomials on CPU

`channelwise_tensor_product_ir_dict` returns an `IrDictPolynomial`. Its `input_irreps` groups are weights, features, and the second input; each `(mul, ir)` becomes one polynomial operand. Its `output_irreps` gives the split output order. These split descriptors use `ir_mul`: transpose the high-level blocks before flattening, then reverse that conversion on outputs. There is no `ir_dict.equivariant_polynomial` entrypoint.

```python
def split_naive(desc, groups):
    assert len(groups) == len(desc.input_irreps)
    operands = [
        group[ir].swapaxes(-1, -2).reshape(group[ir].shape[:-2] + (mul * ir.dim,))
        for irreps, group in zip(desc.input_irreps, groups)
        for mul, ir in irreps
    ]
    batch = np.broadcast_shapes(*(v.shape[:-1] for v in operands))
    outputs = cuex.segmented_polynomial(
        desc.polynomial, operands,
        [jax.ShapeDtypeStruct(batch + (op.size,), jnp.result_type(*operands))
         for op in desc.polynomial.outputs],
        method="naive",
    )
    values = iter(outputs)
    return tuple({
        ir: next(values).reshape(batch + (ir.dim, mul)).swapaxes(-1, -2)
        for mul, ir in irreps
    } for irreps in desc.output_irreps)

desc = cue.descriptors.channelwise_tensor_product_ir_dict(
    cue.Irreps("SO3", "3x0 + 3x1"), cue.Irreps("SO3", "0 + 1"),
    [cue.SO3(0), cue.SO3(1)],
)
rng = np.random.default_rng(8)
groups = [
    {ir: jnp.asarray(rng.normal(size=batch + (mul, ir.dim))) for mul, ir in irreps}
    for irreps, batch in zip(desc.input_irreps, [(), (3, 1), (1, 2)])
]
(result,) = split_naive(desc, groups)
numpy_inputs = [
    np.asarray(group[ir]).swapaxes(-1, -2).reshape(group[ir].shape[:-2] + (mul * ir.dim,))
    for irreps, group in zip(desc.input_irreps, groups) for mul, ir in irreps
]
expected = desc.polynomial(*numpy_inputs)
for (mul, ir), flat in zip(desc.output_irreps[0], expected):
    assert result[ir].shape == (3, 2, mul, ir.dim)
    np.testing.assert_allclose(result[ir], flat.reshape(3, 2, ir.dim, mul).swapaxes(-1, -2),
                               rtol=1e-12, atol=1e-12)
```

### Flax NNX layers

`IrrepsLinear` requires regrouped irreps and explicit `nnx.Rngs`. It mixes channels only within the same irrep, with `scale / sqrt(mul_in)` normalization. An output irrep absent from the input receives zeros, not a learned cross-irrep mapping. `nnx.Param` weights can be read with `[...]` for an independent reference. Row-vector features rotate as `x @ D.T`.

```python
from flax import nnx
from cuequivariance_jax import nnx as cue_nnx

irreps_in = cue.Irreps("SO3", "3x0 + 2x1").regroup()
irreps_out = cue.Irreps("SO3", "2x0 + 4x1 + 2").regroup()
linear = cue_nnx.IrrepsLinear(irreps_in, irreps_out, scale=0.7, dtype=dtype, rngs=nnx.Rngs(9))
rng = np.random.default_rng(10)
features = {ir: jnp.asarray(rng.normal(size=(2, mul, ir.dim))) for mul, ir in irreps_in}
output = linear(features)
for mul, ir in irreps_out:
    assert output[ir].shape == (2, mul, ir.dim)
np.testing.assert_array_equal(output[cue.SO3(2)], 0)
for mul, ir in irreps_in:
    expected = np.einsum("uv,...ui->...vi", np.asarray(linear.w[str(ir)][...]), features[ir])
    np.testing.assert_allclose(output[ir], expected * linear.scale / np.sqrt(mul),
                               rtol=1e-12, atol=1e-12)
rotation = np.array([0.3, -0.2, 0.1])
rotated_output = linear({ir: v @ ir.exp_map(rotation, []).T for ir, v in features.items()})
for ir, value in output.items():
    np.testing.assert_allclose(rotated_output[ir], value @ ir.exp_map(rotation, []).T,
                               rtol=1e-12, atol=1e-12)

vectors = jnp.array([[0.3, -0.7, 1.1], [1.2, 0.4, -0.2]], dtype=dtype)
sh = cue_nnx.SphericalHarmonics(max_degree=2)
harmonics = sh(vectors)
for ir, value in harmonics.items():
    assert value.shape == (2, 1, ir.dim)
    assert bool(jnp.isfinite(value).all())
print({str(ir): (v.shape, v.devices()) for ir, v in harmonics.items()})
```

`IrrepsNormalize(eps=..., scale=..., skip_scalars=...)` normalizes using the sum of squared components averaged over multiplicities. `MLP(layer_sizes=..., activation=..., dtype=..., rngs=...)` operates on scalar features with fan-in normalization. Use `IrrepsIndexedLinear` only for the supported CUDA route: species counts must match grouped input rows and their sum must match the batch size.

### Dense RepArray interface

Use representation metadata from the descriptor to avoid a layout mismatch. A single output normally returns one `RepArray`; its `.array` is the JAX array and `.reps` maps axes to representations.

```python
e = cue.descriptors.fully_connected_tensor_product(
    cue.Irreps("SO3", "2x0 + 1"), cue.Irreps("SO3", "0 + 1"),
    cue.Irreps("SO3", "0 + 2x1"),
)
keys = jax.random.split(jax.random.key(11), len(e.inputs))
inputs = [cuex.randn(key, rep, (2,), dtype) for key, rep in zip(keys, e.inputs)]
output = cuex.equivariant_polynomial(e, inputs, method="naive")
np.testing.assert_allclose(output.array, e(*(np.asarray(v.array) for v in inputs))[0],
                           rtol=1e-12, atol=1e-12)
```

## CUDA preparation and message passing

For a compatible dense descriptor, `e.squeeze_modes().flatten_coefficient_modes()` prepares its modes before `equivariant_polynomial(..., method="uniform_1d")`. Check uniform segment shapes within each operand. Splitting by irrep separates buffers; it does **not** remove the need to convert their layouts.

`ir_dict.segmented_polynomial_uniform_1d(poly, inputs, outputs, input_indices=..., output_indices=...)` expects pytree leaves shaped `(..., op.num_segments, *op.segment_shape)`. This is not the high-level `(..., mul, ir.dim)` convention. For each `_ir_dict` operand, transpose and flatten as in `split_naive`, then reshape using that operand's segment metadata. Ensure pytree leaf order matches polynomial operand order. Output specs need the same segment suffix. Reverse both reshapes and the transpose before passing results into `IrrepsLinear`.

For GNN message passing, weight arrays belong to edges, features to source nodes, and spherical harmonics to edge directions. Supply gather indices for node features (senders) and scatter indices for outputs (receivers); destination count determines output batch size. Construct learned weights or an edge-weight network once in the NNX module initializer. Use explicit split random keys for independent inputs. The source examples `cuequivariance_jax/examples/nequip_nnx.py` and `mace_nnx.py` show complete architectures and CUDA execution.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `ensurepip` unavailable or a partial environment exists | Preserve it; use a new directory with `--without-pip` and host pip `--python`, or an installed environment manager. |
| Missing CUDA kernels on CPU | Select `naive` or ordinary NNX layers. Changing the device alone does not change the method. |
| Float64 truncated or numerical comparison fails | Enable x64 before array creation, use the same dtype/normalization in both computations, then inspect layout and operand order. |
| Dictionary shape/value mismatch | Regroup duplicate irreps with their data, preserve irrep order, and transpose per block. `dict_to_flat` emits `mul_ir`. |
| Missing output or incorrect broadcast | One output spec per polynomial output; broadcast leading axes of all inputs, including weights. |
| An API is missing in the installed version | Inspect that symbol's signature/source and installed versions; do not invent keyword arguments or reinstall the entire stack blindly. |

Locate installed source via `cuex.__file__`. In a checkout, relevant files under `cuequivariance_jax/cuequivariance_jax/` are `ir_dict.py`, `nnx.py`, `equivariant_polynomial.py`, `rep_array/rep_array_.py`, and `segmented_polynomials/segmented_polynomial*.py`. Package API documentation is at https://docs.nvidia.com/cuda/cuequivariance/.
