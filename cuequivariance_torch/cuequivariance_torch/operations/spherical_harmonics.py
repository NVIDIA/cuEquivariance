# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import *

import torch

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import equivariant_tensor_product as etp


def spherical_harmonics(
    ls: list[int],
    vectors: torch.Tensor,
    normalize: bool = True,
    optimize_fallback: Optional[bool] = None,
) -> torch.Tensor:
    r"""Compute the spherical harmonics of the input vectors.

    Parameters
    ----------
    ls : list[int]
        List of spherical harmonic degrees.
    vectors : torch.Tensor
        Input vectors of shape (..., 3).
    normalize : bool
        Whether to normalize the input vectors.

    Returns
    -------
    torch.Tensor
        The spherical harmonics of the input vectors of shape (..., dim)
        where dim is the sum of 2*l+1 for l in ls.
    """
    if isinstance(ls, int):
        ls = [ls]
    assert ls == sorted(set(ls))
    assert vectors.shape[-1] == 3

    if normalize:
        vectors = torch.nn.functional.normalize(vectors, dim=-1)

    x = vectors.reshape(-1, 3)
    m = cuet.EquivariantTensorProduct(
        etp.spherical_harmonics(cue.SO3(1), ls),
        device=x.device,
        math_dtype=x.dtype,
        optimize_fallback=optimize_fallback,
    )
    y = m(x)
    y = y.reshape(vectors.shape[:-1] + (y.shape[-1],))
    return y
