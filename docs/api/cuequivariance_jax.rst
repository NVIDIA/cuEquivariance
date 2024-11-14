.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
   SPDX-License-Identifier: Apache-2.0

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
