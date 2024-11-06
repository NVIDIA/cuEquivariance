.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: LicenseRef-NvidiaProprietary

   NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
   property and proprietary rights in and to this material, related
   documentation and any modifications thereto. Any use, reproduction,
   disclosure or distribution of this material and related documentation
   without an express license agreement from NVIDIA CORPORATION or
   its affiliates is strictly prohibited.

.. cuEquivariance documentation master file, created by
   sphinx-quickstart on Tue Apr 30 02:19:35 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NVIDIA cuEquivariance Documentation
============================

cuEquivariance is a Python library designed to facilitate the construction of high-performance equivariant neural networks using segmented tensor products. cuEquivariance provides a comprehensive API for describing segmented tensor products and optimized CUDA kernels for their execution. Additionally, cuEquivariance offers bindings for both PyTorch and JAX, ensuring broad compatibility and ease of integration.

Equivariance is the mathematical formalization of the concept of "respecting symmetries." Robust physical models exhibit equivariance with respect to rotations and translations in three-dimensional space. Artificial intelligence models that incorporate equivariance are often more data-efficient.

Installation
------------

The easiest way to install cuEquivariance is from PyPi with:

.. code-block:: bash

   pip install cuequivariance

   # CUDA kernels for different CUDA versions
   pip install cuequivariance-ops-torch-cu11
   pip install cuequivariance-ops-torch-cu12

   # Frontend for different ML frameworks
   pip install cuequivariance-jax
   pip install cuequivariance-torch


.. toctree::
   :maxdepth: 1
   :caption: FAQ

   tutorials/irreps
   tutorials/stp
   tutorials/mace

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/cuequivariance
   api/cuequivariance_jax
   api/cuequivariance_torch

.. toctree::
   :maxdepth: 1
   :caption: Change Log

   changelog
