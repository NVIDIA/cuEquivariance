.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: LicenseRef-NvidiaProprietary

   NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
   property and proprietary rights in and to this material, related
   documentation and any modifications thereto. Any use, reproduction,
   disclosure or distribution of this material and related documentation
   without an express license agreement from NVIDIA CORPORATION or
   its affiliates is strictly prohibited.

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
