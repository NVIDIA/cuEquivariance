# SPDX-License-Identifier: Apache-2.0
"""Maintainer reference only. Never copy this file into an agent environment."""

import torch

import cuequivariance as cue
import cuequivariance_torch as cuet


def to_blocks(irreps, flat):
    parts, offset = {}, 0
    for mul, ir in irreps:
        block = flat[..., offset : offset + mul * ir.dim]
        block = block.reshape(flat.shape[:-1] + (ir.dim, mul)).transpose(-1, -2)
        parts.setdefault(ir, []).append(block)
        offset += mul * ir.dim
    return {ir: torch.cat(values, dim=-2) for ir, values in parts.items()}


def from_blocks(irreps, blocks):
    offsets, parts = {}, []
    for mul, ir in irreps:
        start = offsets.get(ir, 0)
        block = blocks[ir][..., start : start + mul, :]
        parts.append(
            block.transpose(-1, -2).reshape(block.shape[:-2] + (mul * ir.dim,))
        )
        offsets[ir] = start + mul
    return torch.cat(parts, dim=-1)


class GraphLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.product = cuet.ChannelWiseTensorProduct(
            cue.Irreps("SO3", "2x0 + 2x1"),
            cue.Irreps("SO3", "0 + 1 + 2"),
            [cue.SO3(0), cue.SO3(1)],
            shared_weights=False,
            internal_weights=False,
            layout=cue.mul_ir,
            method="naive",
            dtype=torch.float64,
        )
        self.readout = cuet.Linear(
            self.product.irreps_out,
            cue.Irreps("SO3", "0 + 1"),
            internal_weights=False,
            layout=cue.mul_ir,
            method="naive",
            dtype=torch.float64,
        )
        self.sh = cuet.SphericalHarmonics([0, 1, 2], normalize=True, method="naive")
        self.radial_in = torch.nn.Parameter(torch.zeros(1, 4, dtype=torch.float64))
        self.radial_out = torch.nn.Parameter(
            torch.zeros(4, self.product.weight_numel, dtype=torch.float64)
        )
        self.projection = torch.nn.Parameter(
            torch.zeros(self.readout.weight_numel, dtype=torch.float64)
        )

    def forward(self, features, vectors, senders, receivers, num_dst):
        lengths = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        weights = torch.tanh(lengths @ self.radial_in) @ self.radial_out
        messages = self.product(
            features,
            self.sh(vectors),
            weight=weights,
            indices_1=senders,
            indices_out=receivers,
            size_out=num_dst,
        )
        return self.readout(messages, self.projection[None, :])
