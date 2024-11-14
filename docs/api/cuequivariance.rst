.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
   SPDX-License-Identifier: Apache-2.0

.. module:: cuequivariance

cuequivariance
==============

.. currentmodule:: cuequivariance

Group Representations
---------------------

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   Rep
   Irrep
   SO3
   O3
   SU2

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   clebsch_gordan

Equivariant Tensor Products
---------------------------

These classes represent tensor products.

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   Irreps
   IrrepsLayout
   SegmentedTensorProduct
   EquivariantTensorProduct

Descriptors
-----------

These functions create specific instances of tensor products.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   descriptors.spherical_harmonics
   descriptors.fully_connected_tensor_product
   descriptors.channelwise_tensor_product
   descriptors.linear
   descriptors.symmetric_contraction
   descriptors.x_rotation
   descriptors.y_rotation
   descriptors.escn_tp
   descriptors.gatr_linear
   descriptors.gatr_geometric_product
   descriptors.gatr_outer_product
