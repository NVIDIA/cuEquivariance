.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
   SPDX-License-Identifier: Apache-2.0

.. module:: cuequivariance

:code:`import cuequivariance as cue`
====================================

.. currentmodule:: cuequivariance

Here are the functions and classes concerning group theory.

.. autosummary::
   :toctree: generated/

   Rep
   Irrep
   SO3
   O3
   SU2
   clebsch_gordan

The following classes are used to define equivariant tensor products.

.. autosummary::
   :toctree: generated/

   Irreps
   IrrepsLayout
   SegmentedTensorProduct
   EquivariantTensorProduct

These functions create specific instances of tensor products.

.. autosummary::
   :toctree: generated/

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
