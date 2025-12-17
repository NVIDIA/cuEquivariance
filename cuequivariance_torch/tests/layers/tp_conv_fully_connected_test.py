# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pytest
import torch
from cuequivariance_torch.layers.tp_conv_fully_connected import scatter_reduce
from torch import nn

import cuequivariance as cue
import cuequivariance_torch as cuet

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize(
    "mlp_channels, mlp_activation, scalar_sizes",
    [
        [(30, 8, 8), nn.Sequential(nn.Dropout(0.3), nn.ReLU()), (15, 15, 0)],
        [(7,), nn.GELU(), (2, 3, 2)],
        [None, None, None],
    ],
)
def test_tensor_product_conv_equivariance(
    mlp_channels, mlp_activation, scalar_sizes, batch_norm, layout
):
    # Skip redundant layout combinations for speed
    if layout == cue.ir_mul:
        pytest.skip("Skipping redundant layout test")

    # Skip complex mlp configurations with batch_norm=False for speed
    if not batch_norm and mlp_channels is not None and len(mlp_channels) > 1:
        pytest.skip("Skipping complex MLP test without batch_norm")

    torch.manual_seed(12345)

    in_irreps = cue.Irreps("O3", "10x0e + 10x1o + 5x2e")
    out_irreps = cue.Irreps("O3", "20x0e + 5x1o + 5x2e")
    sh_irreps = cue.Irreps("O3", "0e + 1o")

    tp_conv = cuet.layers.FullyConnectedTensorProductConv(
        in_irreps=in_irreps,
        sh_irreps=sh_irreps,
        out_irreps=out_irreps,
        mlp_channels=mlp_channels,
        mlp_activation=mlp_activation,
        batch_norm=batch_norm,
        layout=layout,
        use_fallback=not torch.cuda.is_available(),
    ).to(device)

    num_src_nodes, num_dst_nodes = 9, 7
    num_edges = 40
    src = torch.randint(num_src_nodes, (num_edges,), device=device)
    dst = torch.randint(num_dst_nodes, (num_edges,), device=device)
    edge_index = torch.vstack((src, dst))

    src_pos = torch.randn(num_src_nodes, 3, device=device)
    dst_pos = torch.randn(num_dst_nodes, 3, device=device)
    edge_vec = dst_pos[dst] - src_pos[src]
    edge_sh = torch.concatenate(
        [
            torch.ones(num_edges, 1, device=device),
            edge_vec / edge_vec.norm(dim=1, keepdim=True),
        ],
        dim=1,
    )
    src_features = torch.randn(num_src_nodes, in_irreps.dim, device=device)

    def D(irreps, axis, angle):
        return torch.block_diag(
            *[
                torch.from_numpy(ir.rotation(axis, angle)).to(device, torch.float32)
                for mul, ir in irreps
                for _ in range(mul)
            ]
        )

    axis, angle = np.array([0.6, 0.3, -0.1]), 0.52
    D_in = D(in_irreps, axis, angle)
    D_sh = D(sh_irreps, axis, angle)
    D_out = D(out_irreps, axis, angle)

    if mlp_channels is None:
        edge_emb = torch.randn(num_edges, tp_conv.tp.weight_numel, device=device)
        src_scalars = dst_scalars = None
    else:
        if scalar_sizes:
            edge_emb = torch.randn(num_edges, scalar_sizes[0], device=device)
            src_scalars = (
                None
                if scalar_sizes[1] == 0
                else torch.randn(num_src_nodes, scalar_sizes[1], device=device)
            )
            dst_scalars = (
                None
                if scalar_sizes[2] == 0
                else torch.randn(num_dst_nodes, scalar_sizes[2], device=device)
            )
        else:
            edge_emb = torch.randn(num_edges, tp_conv.mlp[0].in_features, device=device)
            src_scalars = dst_scalars = None

    # rotate before
    out_before = tp_conv(
        src_features=src_features @ D_in.T,
        edge_sh=edge_sh @ D_sh.T,
        edge_emb=edge_emb,
        graph=(edge_index, (num_src_nodes, num_dst_nodes)),
        src_scalars=src_scalars,
        dst_scalars=dst_scalars,
    )

    # rotate after
    out_after = (
        tp_conv(
            src_features=src_features,
            edge_sh=edge_sh,
            edge_emb=edge_emb,
            graph=(edge_index, (num_src_nodes, num_dst_nodes)),
            src_scalars=src_scalars,
            dst_scalars=dst_scalars,
        )
        @ D_out.T
    )

    torch.allclose(out_before, out_after, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("reduce", ["sum", "mean", "prod", "amax", "amin"])
def test_scatter_reduce(reduce: str):
    src = torch.Tensor([3, 1, 0, 1, 1, 2])
    index = torch.Tensor([0, 1, 2, 2, 3, 1])

    src = src.to(device)
    index = index.to(device)

    out = scatter_reduce(src, index, dim=0, dim_size=None, reduce=reduce)

    out_true = {
        "sum": torch.Tensor([3.0, 3.0, 1.0, 1.0]),
        "mean": torch.Tensor([3.0, 1.5, 0.5, 1.0]),
        "prod": torch.Tensor([3.0, 2.0, 0.0, 1.0]),
        "amax": torch.Tensor([3.0, 2.0, 1.0, 1.0]),
        "amin": torch.Tensor([3.0, 1.0, 0.0, 1.0]),
    }
    assert torch.allclose(out.cpu(), out_true[reduce])


def test_scatter_reduce_empty():
    src, index = torch.empty((0, 41)), torch.empty((0,))
    src = src.to(device)
    index = index.to(device)

    out = scatter_reduce(src, index, dim=0, dim_size=None)

    assert out.numel() == 0
    assert out.size(1) == src.size(1)
