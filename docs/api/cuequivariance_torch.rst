.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
   SPDX-License-Identifier: Apache-2.0

.. module:: cuequivariance_torch

cuequivariance-torch
====================

.. currentmodule:: cuequivariance_torch

Tensor Products
---------------

.. autosummary::
   :toctree: generated/
   :template: pytorch_module_template.rst

   EquivariantTensorProduct
   SymmetricTensorProduct
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
