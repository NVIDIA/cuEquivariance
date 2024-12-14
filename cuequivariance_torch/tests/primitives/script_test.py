import pytest
import torch

import cuequivariance as cue
from cuequivariance_torch.primitives.symmetric_tensor_product import (
    CUDAKernel as SymmetricTensorProduct,
)
from cuequivariance_torch.primitives.tensor_product import (
    FusedTensorProductOp3,
    FusedTensorProductOp4,
    TensorProductUniform3x1d,
    TensorProductUniform4x1d,
)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def test_script_symmetric_contraction():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    ds = cue.descriptors.symmetric_contraction(
        32 * cue.Irreps("SO3", "0 + 1"), 32 * cue.Irreps("SO3", "0 + 1"), [1, 2, 3]
    ).ds

    batch = 12
    x0 = torch.randn(3, ds[0].operands[0].size, device=device, dtype=torch.float32)
    i0 = torch.zeros(batch, device=device, dtype=torch.int32)
    x1 = torch.randn(batch, ds[0].operands[1].size, device=device, dtype=torch.float32)

    module = SymmetricTensorProduct(ds, device, torch.float32)
    module = torch.jit.script(module)

    assert module(x0, i0, x1).shape == (batch, ds[0].operands[-1].size)


def test_script_fused_tp_3():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    d = (
        cue.descriptors.full_tensor_product(
            cue.Irreps("SO3", "32x1"), cue.Irreps("SO3", "1")
        )
        .d.flatten_coefficient_modes()
        .squeeze_modes("v")
    )

    batch = 12
    x0 = torch.randn(batch, d.operands[0].size, device=device, dtype=torch.float32)
    x1 = torch.randn(batch, d.operands[1].size, device=device, dtype=torch.float32)

    module = FusedTensorProductOp3(d, (0, 1), device, torch.float32)
    module = torch.jit.script(module)

    assert module([x0, x1]).shape == (batch, d.operands[2].size)


def test_script_fused_tp_4():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    d = (
        cue.descriptors.fully_connected_tensor_product(
            cue.Irreps("SO3", "32x1"), cue.Irreps("SO3", "1"), cue.Irreps("SO3", "32x1")
        )
        .d.flatten_coefficient_modes()
        .squeeze_modes("v")
        .permute_operands([1, 2, 0, 3])
    )

    batch = 12
    x0 = torch.randn(batch, d.operands[0].size, device=device, dtype=torch.float32)
    x1 = torch.randn(batch, d.operands[1].size, device=device, dtype=torch.float32)
    x2 = torch.randn(batch, d.operands[2].size, device=device, dtype=torch.float32)

    module = FusedTensorProductOp4(d, (0, 1, 2), device, torch.float32)
    module = torch.jit.script(module)

    assert module([x0, x1, x2]).shape == (batch, d.operands[3].size)


def test_script_uniform_tp_3():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    d = (
        cue.descriptors.full_tensor_product(
            cue.Irreps("SO3", "32x1"), cue.Irreps("SO3", "1")
        )
        .d.flatten_coefficient_modes()
        .squeeze_modes("v")
    )

    batch = 12
    x0 = torch.randn(batch, d.operands[0].size, device=device, dtype=torch.float32)
    x1 = torch.randn(batch, d.operands[1].size, device=device, dtype=torch.float32)

    module = TensorProductUniform3x1d(d, device, torch.float32)
    module = torch.jit.script(module)

    assert module([x0, x1]).shape == (batch, d.operands[2].size)


def test_script_uniform_tp_4():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    d = (
        cue.descriptors.channelwise_tensor_product(
            cue.Irreps("SO3", "32x1"), cue.Irreps("SO3", "1"), cue.Irreps("SO3", "32x1")
        )
        .d.flatten_coefficient_modes()
        .squeeze_modes("v")
    )

    batch = 12
    x0 = torch.randn(batch, d.operands[0].size, device=device, dtype=torch.float32)
    x1 = torch.randn(batch, d.operands[1].size, device=device, dtype=torch.float32)
    x2 = torch.randn(batch, d.operands[2].size, device=device, dtype=torch.float32)

    module = TensorProductUniform4x1d(d, device, torch.float32)
    module = torch.jit.script(module)

    assert module([x0, x1, x2]).shape == (batch, d.operands[3].size)
