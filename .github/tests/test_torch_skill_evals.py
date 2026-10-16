# SPDX-License-Identifier: Apache-2.0
"""CPU reference and autograd contracts for the PyTorch skill evals."""

import numpy as np
import torch

import cuequivariance as cue
import cuequivariance_torch as cuet


def test_spherical_harmonics_numpy_and_gradcheck():
    torch.manual_seed(30)
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    module = cuet.SphericalHarmonics([0, 1, 2], normalize=False, method="naive")
    e = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2])
    y = module(x)
    assert y.shape == (2, 9)
    np.testing.assert_allclose(y.detach().numpy(), e(x.detach().numpy())[0], atol=1e-12)
    assert torch.autograd.gradcheck(module, (x,))


def transpose_blocks(data, irreps, source_layout):
    pieces, offset = [], 0
    for mul, ir in irreps:
        shape = (mul, ir.dim) if source_layout == cue.mul_ir else (ir.dim, mul)
        block = data[..., offset : offset + mul * ir.dim]
        pieces.append(
            block.reshape(*data.shape[:-1], *shape)
            .transpose(-1, -2)
            .reshape(*data.shape[:-1], -1)
        )
        offset += mul * ir.dim
    return torch.cat(pieces, dim=-1)


def test_linear_layouts_rotation_and_gradients():
    torch.manual_seed(31)
    a, b = cue.Irreps("SO3", "2x0+3x1"), cue.Irreps("SO3", "3x0+2x1")
    kwargs = dict(internal_weights=False, method="naive", dtype=torch.float64)
    mul = cuet.Linear(a, b, layout=cue.mul_ir, **kwargs)
    ir = cuet.Linear(a, b, layout=cue.ir_mul, **kwargs)
    x = torch.randn(2, a.dim, dtype=torch.float64, requires_grad=True)
    w = torch.randn(1, mul.weight_numel, dtype=torch.float64, requires_grad=True)
    y = mul(x, w)
    y_ir = ir(transpose_blocks(x, a, cue.mul_ir), w)
    torch.testing.assert_close(y, transpose_blocks(y_ir, b, cue.ir_mul))
    rotation = np.array([0.3, -0.1, 0.4])
    da, db = [
        torch.from_numpy(cue.IrrepsAndLayout(rep, cue.mul_ir).exp_map(rotation, []))
        for rep in (a, b)
    ]
    torch.testing.assert_close(mul(x @ da.T, w), y @ db.T)
    for grad in torch.autograd.grad(y.square().sum(), (x, w)):
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_channelwise_external_unshared_weights():
    torch.manual_seed(32)
    a, b = cue.Irreps("SO3", "2x0+2x1"), cue.Irreps("SO3", "0+1")
    keep = [cue.SO3(0), cue.SO3(1)]
    module = cuet.ChannelWiseTensorProduct(
        a,
        b,
        keep,
        shared_weights=False,
        internal_weights=False,
        layout=cue.ir_mul,
        method="naive",
        dtype=torch.float64,
    )
    e = cue.descriptors.channelwise_tensor_product(a, b, keep)
    x, y, w = [
        torch.randn(3, size, dtype=torch.float64, requires_grad=True)
        for size in (a.dim, b.dim, module.weight_numel)
    ]
    actual = module(x, y, w)
    expected = e(*(v.detach().numpy() for v in (w, x, y)))[0]
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-12)
    for grad in torch.autograd.grad(actual.square().sum(), (x, y, w)):
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_gather_scatter_repeated_receivers_and_gradients():
    torch.manual_seed(33)
    e = cue.descriptors.linear(cue.Irreps("SO3", "2x0"), cue.Irreps("SO3", "3x0"))
    module = cuet.SegmentedPolynomial(e.polynomial, method="naive")
    x = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    w = torch.randn(1, e.inputs[0].dim, dtype=torch.float64, requires_grad=True)
    senders = torch.tensor([0, 1, 0, 3, 1])
    receivers = torch.tensor([2, 0, 2, 1, 2])
    shape = torch.empty(4, 1, dtype=torch.float64)
    [actual] = module(
        [w, x],
        input_indices={1: senders},
        output_indices={0: receivers},
        output_shapes={0: shape},
    )
    [edges] = module([w, x[senders]])
    expected = torch.zeros(4, 3, dtype=torch.float64).index_add_(0, receivers, edges)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[3], torch.zeros(3, dtype=torch.float64))
    actual_grads = torch.autograd.grad(actual.square().sum(), (x, w))
    expected_grads = torch.autograd.grad(expected.square().sum(), (x, w))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_fully_connected_tensor_product_cpu():
    torch.manual_seed(34)
    irreps = cue.Irreps("SO3", "0+1")
    module = cuet.FullyConnectedTensorProduct(
        irreps, irreps, irreps, method="naive", dtype=torch.float64
    )
    x = torch.randn(2, irreps.dim, dtype=torch.float64, requires_grad=True)
    result = module(x, x)
    assert result.shape == x.shape
    result.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
