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

"""
Deprecated API stubs with migration guidance.

This module contains stubs for deprecated classes that have been removed
from the public API. Each class raises an ImportError with detailed
migration instructions to help users update their code.
"""


class _DeprecatedClass:
    """Base class for deprecated API stubs."""

    def __init__(self, *args, **kwargs):
        raise ImportError(
            f"{self.__class__.__name__} has been removed.\n{self._migration_message}"
        )


class TensorProduct(_DeprecatedClass):
    _migration_message = (
        "TensorProduct has been removed and replaced by SegmentedPolynomial.\n"
        "\n"
        "Migration path:\n"
        "  The TensorProduct class has been unified into SegmentedPolynomial,\n"
        "  which provides a cleaner API with multiple backend methods.\n"
        "  \n"
        "  OLD:\n"
        "    from cuequivariance_torch import TensorProduct\n"
        "    import cuequivariance as cue\n"
        "    descriptor = cue.SegmentedTensorProduct(...)\n"
        "    tp = TensorProduct(descriptor, device=device, math_dtype=dtype)\n"
        "  \n"
        "  NEW:\n"
        "    import cuequivariance_torch as cuet\n"
        "    import cuequivariance as cue\n"
        "    \n"
        "    # Create a polynomial for evaluating the last operand\n"
        "    descriptor = cue.SegmentedTensorProduct(...)\n"
        "    poly = cue.SegmentedPolynomial.eval_last_operand(descriptor)\n"
        "    \n"
        "    # Use SegmentedPolynomial with the desired method\n"
        "    tp = cuet.SegmentedPolynomial(poly, method='uniform_1d', math_dtype=dtype, device=device)\n"
        "  \n"
        "  Or use the high-level operation APIs:\n"
        "    - cuet.ChannelWiseTensorProduct\n"
        "    - cuet.FullyConnectedTensorProduct\n"
        "  \n"
        "  Available methods: 'naive', 'uniform_1d', 'fused_tp', 'indexed_linear'\n"
    )


class EquivariantTensorProduct(_DeprecatedClass):
    _migration_message = (
        "EquivariantTensorProduct has been removed and replaced by SegmentedPolynomial.\n"
        "\n"
        "Migration path:\n"
        "  The EquivariantTensorProduct class has been unified into SegmentedPolynomial,\n"
        "  which provides a cleaner API with multiple backend methods.\n"
        "  \n"
        "  OLD:\n"
        "    from cuequivariance_torch import EquivariantTensorProduct\n"
        "    import cuequivariance as cue\n"
        "    descriptor = cue.SegmentedTensorProduct(...)\n"
        "    etp = EquivariantTensorProduct(descriptor, device=device, math_dtype=dtype)\n"
        "  \n"
        "  NEW:\n"
        "    import cuequivariance_torch as cuet\n"
        "    import cuequivariance as cue\n"
        "    \n"
        "    # Create a polynomial for evaluating the last operand\n"
        "    descriptor = cue.SegmentedTensorProduct(...)\n"
        "    poly = cue.SegmentedPolynomial.eval_last_operand(descriptor)\n"
        "    \n"
        "    # Use SegmentedPolynomial with the desired method\n"
        "    etp = cuet.SegmentedPolynomial(poly, method='uniform_1d', math_dtype=dtype, device=device)\n"
        "  \n"
        "  Or use the high-level operation API:\n"
        "    - cuet.FullyConnectedTensorProduct\n"
        "  \n"
        "  Available methods: 'naive', 'uniform_1d', 'fused_tp', 'indexed_linear'\n"
    )


class SymmetricTensorProduct(_DeprecatedClass):
    _migration_message = (
        "SymmetricTensorProduct has been removed and replaced by SegmentedPolynomial.\n"
        "\n"
        "Migration path:\n"
        "  The SymmetricTensorProduct class has been unified into SegmentedPolynomial,\n"
        "  which provides a cleaner API with multiple backend methods.\n"
        "  \n"
        "  OLD:\n"
        "    from cuequivariance_torch import SymmetricTensorProduct\n"
        "    import cuequivariance as cue\n"
        "    descriptor = cue.SegmentedTensorProduct(...)\n"
        "    stp = SymmetricTensorProduct(descriptor, device=device, math_dtype=dtype)\n"
        "  \n"
        "  NEW:\n"
        "    import cuequivariance_torch as cuet\n"
        "    import cuequivariance as cue\n"
        "    \n"
        "    # Create a polynomial for evaluating the last operand\n"
        "    descriptor = cue.SegmentedTensorProduct(...)\n"
        "    poly = cue.SegmentedPolynomial.eval_last_operand(descriptor)\n"
        "    \n"
        "    # Use SegmentedPolynomial with the desired method\n"
        "    stp = cuet.SegmentedPolynomial(poly, method='uniform_1d', math_dtype=dtype, device=device)\n"
        "  \n"
        "  Or use the high-level operation API:\n"
        "    - cuet.SymmetricContraction (for symmetric contraction operations)\n"
        "  \n"
        "  Available methods: 'naive', 'uniform_1d', 'fused_tp', 'indexed_linear'\n"
    )


class IWeightedSymmetricTensorProduct(_DeprecatedClass):
    _migration_message = (
        "IWeightedSymmetricTensorProduct has been removed and replaced by SegmentedPolynomial.\n"
        "\n"
        "Migration path:\n"
        "  The IWeightedSymmetricTensorProduct class has been unified into SegmentedPolynomial,\n"
        "  which provides a cleaner API with multiple backend methods.\n"
        "  \n"
        "  OLD:\n"
        "    from cuequivariance_torch import IWeightedSymmetricTensorProduct\n"
        "    import cuequivariance as cue\n"
        "    descriptor = cue.SegmentedTensorProduct(...)\n"
        "    iwstp = IWeightedSymmetricTensorProduct(descriptor, device=device, math_dtype=dtype)\n"
        "  \n"
        "  NEW:\n"
        "    import cuequivariance_torch as cuet\n"
        "    import cuequivariance as cue\n"
        "    \n"
        "    # Create a polynomial for evaluating the last operand\n"
        "    descriptor = cue.SegmentedTensorProduct(...)\n"
        "    poly = cue.SegmentedPolynomial.eval_last_operand(descriptor)\n"
        "    \n"
        "    # Use SegmentedPolynomial with the desired method\n"
        "    iwstp = cuet.SegmentedPolynomial(poly, method='uniform_1d', math_dtype=dtype, device=device)\n"
        "  \n"
        "  Or use the high-level operation API:\n"
        "    - cuet.SymmetricContraction (with internal_weights=True for weighted operations)\n"
        "  \n"
        "  Available methods: 'naive', 'uniform_1d', 'fused_tp', 'indexed_linear'\n"
    )


__all__ = [
    "TensorProduct",
    "EquivariantTensorProduct",
    "SymmetricTensorProduct",
    "IWeightedSymmetricTensorProduct",
]
