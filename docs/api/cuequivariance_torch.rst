.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
   SPDX-License-Identifier: Apache-2.0

.. module:: cuequivariance_torch

:code:`import cuequivariance_torch as cuet`
===========================================

.. currentmodule:: cuequivariance_torch

The PyTorch modules

.. autosummary::
   :toctree: generated/
   :template: pytorch_module_template.rst

   EquivariantTensorProduct
   TensorProduct
   SymmetricTensorProduct
   IWeightedSymmetricTensorProduct
   TransposeIrrepsLayout
   ChannelWiseTensorProduct
   FullyConnectedTensorProduct
   Linear
   SymmetricContraction
   Rotation
   Inversion
   layers.BatchNorm
   layers.FullyConnectedTensorProductConv

The functions

.. autosummary::
   :toctree: generated/

   spherical_harmonics
   encode_rotation_angle
   vector_to_euler_angles
