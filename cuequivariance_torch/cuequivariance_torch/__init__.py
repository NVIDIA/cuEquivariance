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
import importlib.resources

__version__ = (
    importlib.resources.files(__package__).joinpath("VERSION").read_text().strip()
)

from .primitives.tensor_product import TensorProduct, _Wrapper
from .primitives.symmetric_tensor_product import (
    SymmetricTensorProduct,
    IWeightedSymmetricTensorProduct,
)
from .primitives.transpose import TransposeSegments, TransposeIrrepsLayout

from .primitives.equivariant_tensor_product import EquivariantTensorProduct
from .primitives.segmented_polynomial import SegmentedPolynomial
from .operations.tp_channel_wise import ChannelWiseTensorProduct
from .operations.tp_fully_connected import FullyConnectedTensorProduct
from .operations.linear import Linear
from .operations.symmetric_contraction import SymmetricContraction
from .operations.rotation import (
    Rotation,
    encode_rotation_angle,
    vector_to_euler_angles,
    Inversion,
)
from .operations.spherical_harmonics import SphericalHarmonics
from .primitives.triangle import (
    triangle_attention,
    triangle_multiplicative_update,
    attention_pair_bias,
    TriMulPrecision,
)

from cuequivariance_torch import layers


def onnx_custom_translation_table():
    r"""
     Returns ONNX translation table for custom operations from cuequivariance_ops_torch
     to be passed to torch.onnx.export as 'custom_translation_table' argument.
     
     Example:
     >>> import cuequivariance_torch
     >>> # cueq_custon_onnx_table = cuequivariance_torch.onnx_custom_translation_table()
     >>> # onnx_program = torch.onnx.export(module, inputs, custom_translation_table=cueq_custom_onnx_table)
    """
    from cuequivariance_ops_torch.onnx import op_table
    return op_table

def register_tensorrt_plugins():
    r"""
    Registers TensorRT plugins for custom operations from cuequivariance_ops_torch

    Example:
    >>> import cuequivariance_torch
    >>> # cuequivariance_torch.register_plugins()
    """
    from cuequivariance_ops_torch.tensorrt import register_plugins
    register_plugins()


__all__ = [
    "TensorProduct",
    "_Wrapper",
    "SymmetricTensorProduct",
    "IWeightedSymmetricTensorProduct",
    "TransposeSegments",
    "TransposeIrrepsLayout",
    "EquivariantTensorProduct",
    "SegmentedPolynomial",
    "ChannelWiseTensorProduct",
    "FullyConnectedTensorProduct",
    "Linear",
    "SymmetricContraction",
    "Rotation",
    "encode_rotation_angle",
    "vector_to_euler_angles",
    "Inversion",
    "SphericalHarmonics",
    "triangle_attention",
    "triangle_multiplicative_update",
    "attention_pair_bias",
    "TriMulPrecision",
    "layers",
    "onnx_custom_translation_table",
    "register_tensorrt_plugins",
]
