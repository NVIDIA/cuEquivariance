.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: LicenseRef-NvidiaProprietary

   NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
   property and proprietary rights in and to this material, related
   documentation and any modifications thereto. Any use, reproduction,
   disclosure or distribution of this material and related documentation
   without an express license agreement from NVIDIA CORPORATION or
   its affiliates is strictly prohibited.

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
