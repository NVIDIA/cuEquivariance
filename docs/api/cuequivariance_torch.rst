.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. module:: cuequivariance_torch
.. currentmodule:: cuequivariance_torch

cuequivariance-torch
====================

Tensor Products
---------------

.. autosummary::
   :toctree: generated/
   :template: pytorch_module_template.rst

   EquivariantTensorProduct
   TensorProduct

Special Cases of Tensor Products
--------------------------------

.. autosummary::
   :toctree: generated/
   :template: pytorch_module_template.rst

   ChannelWiseTensorProduct
   FullyConnectedTensorProduct
   Linear
   SymmetricContraction
   TransposeIrrepsLayout

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   spherical_harmonics

Euclidean Operations
--------------------

.. autosummary::
   :toctree: generated/
   :template: pytorch_module_template.rst

   Rotation
   Inversion

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   encode_rotation_angle
   vector_to_euler_angles

Extra Modules
-------------

.. autosummary::
   :toctree: generated/
   :template: pytorch_module_template.rst

   layers.BatchNorm
   layers.FullyConnectedTensorProductConv
