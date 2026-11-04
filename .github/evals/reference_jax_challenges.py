# SPDX-License-Identifier: Apache-2.0
"""Maintainer reference only. Never copy this file into an agent environment."""

import cuequivariance_jax.ir_dict as ir_dict
import jax
import jax.numpy as jnp

import cuequivariance as cue
import cuequivariance_jax as cuex

jax.config.update("jax_enable_x64", True)


def to_blocks(irreps, flat):
    data = cuex.RepArray(cue.IrrepsAndLayout(irreps, cue.ir_mul), flat).regroup()
    return ir_dict.flat_to_dict(data.irreps, data.array, layout="ir_mul")


def from_blocks(irreps, blocks):
    offsets, parts = {}, []
    for mul, ir in irreps:
        start = offsets.get(ir, 0)
        block = blocks[ir][..., start : start + mul, :]
        parts.append(block.swapaxes(-1, -2).reshape(block.shape[:-2] + (mul * ir.dim,)))
        offsets[ir] = start + mul
    return jnp.concatenate(parts, axis=-1)


INPUT = cue.Irreps("SO3", "2x0 + 2x1")
OUTPUT = cue.Irreps("SO3", "0 + 1")
PRODUCT = cue.descriptors.channelwise_tensor_product(
    INPUT, cue.Irreps("SO3", "0 + 1 + 2"), [cue.SO3(0), cue.SO3(1)]
)
PROJECTION = cue.descriptors.linear(PRODUCT.outputs[0].irreps, OUTPUT)
HARMONICS = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2])


def transpose(flat, irreps, *, source):
    parts, offset = [], 0
    for mul, ir in irreps:
        suffix = (mul, ir.dim) if source == "mul_ir" else (ir.dim, mul)
        block = flat[..., offset : offset + mul * ir.dim]
        parts.append(
            block.reshape(flat.shape[:-1] + suffix)
            .swapaxes(-1, -2)
            .reshape(flat.shape[:-1] + (mul * ir.dim,))
        )
        offset += mul * ir.dim
    return jnp.concatenate(parts, axis=-1)


def run(descriptor, operands, rows):
    return cuex.segmented_polynomial(
        descriptor.polynomial,
        operands,
        [jax.ShapeDtypeStruct((rows, descriptor.outputs[0].dim), jnp.float64)],
        method="naive",
    )[0]


def apply(params, features, vectors, senders, receivers, num_dst):
    lengths = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    weights = jnp.tanh(lengths @ params["radial_in"]) @ params["radial_out"]
    sh = run(HARMONICS, [vectors / lengths], vectors.shape[0])
    inputs = transpose(features, INPUT, source="mul_ir")
    messages = run(PRODUCT, [weights, inputs[senders], sh], vectors.shape[0])
    summed = (
        jnp.zeros((num_dst, messages.shape[-1]), messages.dtype)
        .at[receivers]
        .add(messages)
    )
    output = run(PROJECTION, [params["projection"], summed], num_dst)
    return transpose(output, OUTPUT, source="ir_mul")
