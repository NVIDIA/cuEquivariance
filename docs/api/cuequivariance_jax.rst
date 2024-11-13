.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: LicenseRef-NvidiaProprietary

   NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
   property and proprietary rights in and to this material, related
   documentation and any modifications thereto. Any use, reproduction,
   disclosure or distribution of this material and related documentation
   without an express license agreement from NVIDIA CORPORATION or
   its affiliates is strictly prohibited.

.. module:: cuequivariance_jax

:code:`import cuequivariance_jax as cuex`
=========================================

.. currentmodule:: cuequivariance_jax

The data objects

.. autosummary::
   :toctree: generated/

   IrrepsArray
   from_segments
   as_irreps_array
   concatenate
   randn

The functions to compute the tensor products

.. autosummary::
   :toctree: generated/

   equivariant_tensor_product
   symmetric_tensor_product
   tensor_product

Some high-level modules

.. autosummary::
   :toctree: generated/

   flax_linen.Linear
   flax_linen.LayerNorm
