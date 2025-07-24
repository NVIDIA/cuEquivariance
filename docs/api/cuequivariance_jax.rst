.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. module:: cuequivariance_jax
.. currentmodule:: cuequivariance_jax

cuequivariance-jax
==================

RepArray
--------

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   RepArray

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   from_segments
   as_irreps_array
   concatenate
   randn

Tensor Products
---------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   equivariant_polynomial
   segmented_polynomial

Extra Modules
-------------

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   flax_linen.LayerNorm

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   spherical_harmonics

Triangle
--------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   triangle_multiplicative_update
   triangle_attention

Experimental
------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   experimental.indexed_linear
