.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
   SPDX-License-Identifier: Apache-2.0

.. module:: cuequivariance
.. currentmodule:: cuequivariance

cuequivariance
==============

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

:doc:`List of Descriptors <cuequivariance.descriptors>`

.. toctree::
   :hidden:

   cuequivariance.descriptors
