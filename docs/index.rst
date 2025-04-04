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

NVIDIA cuEquivariance Documentation
===================================

cuEquivariance is a Python library designed to facilitate the construction of high-performance equivariant neural networks using segmented tensor products. cuEquivariance provides a comprehensive API for describing segmented tensor products and optimized CUDA kernels for their execution. Additionally, cuEquivariance offers bindings for both PyTorch and JAX, ensuring broad compatibility and ease of integration.

Equivariance is the mathematical formalization of the concept of "respecting symmetries." Robust physical models exhibit equivariance with respect to rotations and translations in three-dimensional space. Artificial intelligence models that incorporate equivariance are often more data-efficient.

An introduction to group representations can be found in the page :doc:`tutorials/irreps`.

Open Source
-----------

cuEquivariance frontend is open-source and available on `GitHub <https://github.com/NVIDIA/cuEquivariance>`_ under the Apache 2.0 license.

Installation
------------

The easiest way to install cuEquivariance is from `PyPi <https://pypi.org/>`_ using `pip <https://pip.pypa.io/en/stable/>`_.

.. code-block:: bash

   # Choose the frontend you want to use
   pip install cuequivariance-jax
   pip install cuequivariance-torch
   pip install cuequivariance  # Installs only the core non-ML components

   # CUDA kernels
   pip install cuequivariance-ops-torch-cu11
   pip install cuequivariance-ops-torch-cu12
   pip install cuequivariance-ops-jax-cu12

Requirements
------------

 - ``cuequivariance-ops-torch-*`` packages are available for Linux x86_64/aarch64 and require PyTorch 2.4.0 or later. aarch64 is only available for Python 3.12.
 - ``cuequivariance-ops-jax-cu12`` package is available for Linux x86_64/aarch64 and requires JAX 0.5.0 or later.

Organization
------------

cuEquivariance is split into three packages:

.. jupyter-execute::

   import cuequivariance as cue
   # All the non-ML components

   import cuequivariance_jax as cuex
   # For the JAX implementations

   import cuequivariance_torch as cuet
   # For the PyTorch implementations


.. image:: _static/main_components.png
   :alt: Main components of cuEquivariance
   :align: center

Most tensor products are defined using a hierarchy of components:

* :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>` - Encapsulates the :class:`cue.Rep <cuequivariance.Rep>` for each input and the output
* :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>` - Included in the EquivariantPolynomial, composed of one or several tensor products
* :class:`cue.SegmentedTensorProduct <cuequivariance.SegmentedTensorProduct>` - Defines the specific tensor product operations

This descriptor can then be used in two ways:

* In PyTorch: Create a :class:`cuet.SegmentedPolynomial <cuequivariance_torch.SegmentedPolynomial>` module for use in models
* In JAX: Execute using :class:`cuex.equivariant_polynomial <cuequivariance_jax.equivariant_polynomial>` or :class:`cuex.segmented_polynomial <cuequivariance_jax.segmented_polynomial>`

Tutorials
---------

| :doc:`tutorials/irreps`
| :doc:`tutorials/layout`
| :doc:`tutorials/poly`
| :doc:`tutorials/stp`

.. toctree::
   :hidden:

   tutorials/index

API Reference
-------------

.. toctree::
   :maxdepth: 1

   api/cuequivariance
   api/cuequivariance_jax
   api/cuequivariance_torch

What's New
----------

.. toctree::
   :maxdepth: 2

   changelog
