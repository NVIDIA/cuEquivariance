# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import numpy as np
import pytest
import torch
from torch import nn

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance_torch.layers.tp_conv_fully_connected import scatter_reduce

device = torch.device("cuda:0")


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
    device = torch.device("cuda")
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
    device = torch.device("cuda")
    src, index = torch.empty((0, 41)), torch.empty((0,))
    src = src.to(device)
    index = index.to(device)

    out = scatter_reduce(src, index, dim=0, dim_size=None)

    assert out.numel() == 0
    assert out.size(1) == src.size(1)


@pytest.mark.parametrize(
    "layout, batch_norm", [(cue.mul_ir, False), (cue.mul_ir, True), (cue.ir_mul, False)]
)
@pytest.mark.parametrize(
    "in_irreps, out_irreps",
    [
        ("0e", "0e"),
        ("3x2e + 1e", "3x2e"),
        ("10x0e + 10x1o + 5x2e", "20x0e + 5x1o + 5x2e"),
    ],
)
def test_compare_with_cugraph(
    batch_norm: bool, layout: cue.IrrepsLayout, in_irreps: str, out_irreps: str
):
    try:
        from cugraph_equivariant.nn import FullyConnectedTensorProductConv
        import e3nn
    except ImportError:
        pytest.skip("cugraph_equivariant and e3nn are not installed")

    torch.manual_seed(12345)

    in_irreps = cue.Irreps("O3", in_irreps)
    out_irreps = cue.Irreps("O3", out_irreps)

    tp_conv = cuet.layers.FullyConnectedTensorProductConv(
        in_irreps=in_irreps,
        sh_irreps=cue.Irreps("O3", "0e + 1o"),
        out_irreps=out_irreps,
        mlp_channels=None,
        mlp_activation=None,
        batch_norm=batch_norm,
        layout=layout,
    ).to(device)
    tp_conv_cugraph = FullyConnectedTensorProductConv(
        in_irreps=e3nn.o3.Irreps(str(in_irreps)),
        sh_irreps=e3nn.o3.Irreps("0e + 1o"),
        out_irreps=e3nn.o3.Irreps(str(out_irreps)),
        mlp_channels=None,
        mlp_activation=None,
        batch_norm=batch_norm,
        e3nn_compat_mode=(layout == cue.mul_ir),
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

    edge_emb = torch.randn(num_edges, tp_conv.tp.weight_numel, device=device)

    A = tp_conv(
        src_features=src_features,
        edge_sh=edge_sh,
        edge_emb=edge_emb,
        graph=(edge_index, (num_src_nodes, num_dst_nodes)),
    )
    B = tp_conv_cugraph(
        src_features=src_features,
        edge_sh=edge_sh,
        edge_emb=edge_emb,
        graph=(edge_index, (num_src_nodes, num_dst_nodes)),
    )

    torch.testing.assert_close(A, B, rtol=1e-4, atol=1e-4)
