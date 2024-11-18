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
