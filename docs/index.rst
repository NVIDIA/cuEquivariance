.. SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cuEquivariance is an NVIDIA Python library of GPU-accelerated building blocks for
geometric deep learning. It covers two areas: **equivariant tensor products** for
models that respect the symmetries of three-dimensional space, and **triangle
operations** for the pair-representation blocks of biomolecular structure
prediction. Both have PyTorch and JAX frontends and share one set of prebuilt CUDA
kernel wheels.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Equivariant tensor products
      :link: equivariant-tensor-products
      :link-type: ref

      A descriptor API for segmented polynomials and segmented tensor products,
      with CUDA kernels that execute them. The operations behind equivariant
      interatomic potentials such as MACE, NequIP and Allegro.

      * Irreps, layouts and group representations
      * Segmented tensor products and polynomials
      * Linear, channel-wise and fully connected tensor product layers,
        symmetric contraction, spherical harmonics

      +++
      Describe the polynomial once, execute it in PyTorch or JAX.

   .. grid-item-card:: Triangle operations
      :link: triangle-operations
      :link-type: ref

      Fused kernels for the pair-representation blocks of structure prediction
      models such as AlphaFold 3, Boltz-2 and RFdiffusion.

      * Triangle multiplicative update
      * Triangle attention
      * Attention with pair bias

      +++
      Drop-in replacements for the PyTorch and JAX reference blocks.

The frontends are open source under the Apache 2.0 license on `GitHub
<https://github.com/NVIDIA/cuEquivariance>`_. The CUDA kernels ship as prebuilt
``cuequivariance-ops-*`` wheels for CUDA 12 and CUDA 13.

What's new in 0.13
------------------

Released 2026-10-15. Highlights:

* **Optimized triangle multiplicative update kernels** for PyTorch on H100 SXM, H200,
  GH200, B200 and RTX PRO 6000 Blackwell, for inference and training, at the pair
  and hidden widths used by production structure prediction models (128/128,
  256/256, 384/256 and 512/512). CUDA 13 wheels only.
* ``cuex.triangle_attention`` accepts ``mask=None`` for unmasked attention, as the
  PyTorch frontend does.
* From 0.12: attention pair bias with RMSNorm, expanded H100 and H200 training
  coverage, and the pair-update to attention-bias inference handoff.
* From 0.11: a dictionary-based JAX interface for organizing features by
  representation type, and MACE examples for Flax NNX and Equinox.

The full list of changes, the exact supported shapes and the known issues are in
the :doc:`changelog` and the `GitHub release notes
<https://github.com/NVIDIA/cuEquivariance/releases>`_.

Installation
------------

Install from `PyPI <https://pypi.org/>`_ with pip. Pick one frontend and the kernel
wheel that matches your CUDA version. The ``-cu13`` wheels contain kernels that the
``-cu12`` wheels do not; use ``-cu13`` when your driver allows it.

.. tab-set::

   .. tab-item:: PyTorch

      .. code-block:: bash

         pip install cuequivariance-torch cuequivariance-ops-torch-cu13   # or -cu12

   .. tab-item:: JAX

      .. code-block:: bash

         pip install cuequivariance-jax cuequivariance-ops-jax-cu13       # or -cu12

   .. tab-item:: Core only

      .. code-block:: bash

         # Descriptors and NumPy execution, no GPU kernels
         pip install cuequivariance

**Requirements.** Linux x86_64 or aarch64 for the ``cuequivariance-ops-*`` wheels.
Python 3.10 to 3.14. PyTorch 2.11, 2.12 or 2.13 for the ``cuequivariance-ops-torch``
wheels (the pure-Python parts of ``cuequivariance-torch`` work from PyTorch 2.4).
JAX 0.8.1 or newer for ``cuequivariance-jax``.

Supported operations and hardware
---------------------------------

Every operation has a generic implementation that runs on any supported NVIDIA GPU.
The table lists where an operation also has a specialized kernel in 0.13.0. Details
per shape and data type are in the :doc:`changelog`.

.. list-table::
   :header-rows: 1
   :widths: 24 12 32 32

   * - Operation
     - Frontends
     - Specialized kernels
     - Notes
   * - Segmented tensor products and polynomials
     - PyTorch, JAX
     - All supported GPUs: descriptor-driven CUDA kernels, including the fused
       tensor product and the uniform 1d kernels.
     - Kernel methods ``uniform_1d``, ``fused_tp`` and ``indexed_linear``, with
       the math precision selectable through ``math_dtype`` (FP32 or FP64).
       NumPy execution in the core package.
   * - Tensor product layers
     - PyTorch, JAX
     - Built on the kernels above: linear, channel-wise and fully connected tensor
       products, symmetric contraction, spherical harmonics, layout transposes.
     - PyTorch ``nn.Module`` classes; JAX functions and Flax NNX layers.
   * - Triangle multiplicative update
     - PyTorch, JAX
     - PyTorch: inference on H100 SXM, H200, GH200, B200, RTX PRO 6000 Blackwell;
       training on H100 SXM, H200, GH200, B200. Pair/hidden widths 128/128,
       256/256, 384/256, 512/512.
     - BF16. CUDA 13 wheels. Other GPUs and JAX use the generic implementation.
   * - Triangle attention
     - PyTorch, JAX
     - Native kernels on B200 / B300 (SM100f) and RTX PRO 6000 Blackwell (sm120f);
       generic kernel on H100, H200, A100, L40.
     - BF16, FP16; FP32/TF32 for small head dimensions. ``kv_lengths`` selects the
       right-padded fast path on Blackwell. Forward and backward.
   * - Attention pair bias
     - PyTorch
     - Inference on H100, H200, B200, B300, RTX PRO 6000 Blackwell, A100; training
       on H100, H200.
     - LayerNorm or RMSNorm. Pair widths 128, 256 and 384; 4 to 16 heads; head
       dimension 8 to 64.

Getting started
---------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Tensor products in PyTorch
      :link: tutorials/pytorch/index
      :link-type: doc

      :class:`cuet.SegmentedPolynomial <cuequivariance_torch.SegmentedPolynomial>`
      and the layer modules, with a complete :doc:`MACE model <tutorials/pytorch/MACE>`.

   .. grid-item-card:: Tensor products in JAX
      :link: tutorials/jax/index
      :link-type: doc

      :func:`cuex.equivariant_polynomial <cuequivariance_jax.equivariant_polynomial>`,
      :func:`cuex.segmented_polynomial <cuequivariance_jax.segmented_polynomial>` and
      the dictionary-based :doc:`feature interface <tutorials/jax/ir_dict>`.

   .. grid-item-card:: Triangle operations in PyTorch
      :link: api/cuequivariance_torch
      :link-type: doc

      :func:`cuet.triangle_multiplicative_update <cuequivariance_torch.triangle_multiplicative_update>`,
      :func:`cuet.triangle_attention <cuequivariance_torch.triangle_attention>` and
      :func:`cuet.attention_pair_bias <cuequivariance_torch.attention_pair_bias>`.

   .. grid-item-card:: Triangle operations in JAX
      :link: api/cuequivariance_jax
      :link-type: doc

      :func:`cuex.triangle_multiplicative_update <cuequivariance_jax.triangle_multiplicative_update>` and
      :func:`cuex.triangle_attention <cuequivariance_jax.triangle_attention>`.

.. _equivariant-tensor-products:

Equivariant tensor products
---------------------------

Equivariance is the mathematical formalization of "respecting symmetries". Models
that are equivariant to rotations and translations in three-dimensional space are
often more data-efficient. An introduction to group representations is on the page
:doc:`tutorials/irreps`.

cuEquivariance splits this area into three packages. The core package describes the
mathematics; the two frontends execute it.

.. jupyter-execute::

   import cuequivariance as cue
   # Group representations, descriptors and NumPy execution

   import cuequivariance_jax as cuex
   # JAX execution and Flax NNX layers

   import cuequivariance_torch as cuet
   # PyTorch operations and layers

.. image:: _static/main_components.png
   :alt: Main components of cuEquivariance
   :align: center

A tensor product is a hierarchy of descriptors:

* :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>` holds
  the :class:`cue.Rep <cuequivariance.Rep>` of each input and of the output.
* :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>` is the
  polynomial inside it, made of one or several tensor products.
* :class:`cue.SegmentedTensorProduct <cuequivariance.SegmentedTensorProduct>`
  defines one tensor product.

The same descriptor is executed with :class:`cuet.SegmentedPolynomial
<cuequivariance_torch.SegmentedPolynomial>` in PyTorch, or with
:func:`cuex.equivariant_polynomial <cuequivariance_jax.equivariant_polynomial>` and
:func:`cuex.segmented_polynomial <cuequivariance_jax.segmented_polynomial>` in JAX.
The tutorials :doc:`tutorials/stp` and :doc:`tutorials/poly` build descriptors step
by step.

On a MACE model, cuEquivariance trains and runs faster than `e3nn
<https://github.com/e3nn/e3nn>`_ and reaches larger systems before running out of
memory. Crosses mark the largest number of atoms that fit on one device.

.. image:: _static/mace_benchmark.png
   :alt: Performance comparison between cuEquivariance and e3nn on a MACE model
   :align: center
   :width: 80%

.. _triangle-operations:

Triangle operations
-------------------

Structure prediction models keep a pair representation, a tensor with one row per
pair of residues or atoms, and update it with three kinds of blocks. cuEquivariance
provides each one as a single fused call with the same inputs and outputs as the
reference PyTorch or JAX implementation:

* **Triangle multiplicative update**
  (:func:`cuet.triangle_multiplicative_update <cuequivariance_torch.triangle_multiplicative_update>`,
  :func:`cuex.triangle_multiplicative_update <cuequivariance_jax.triangle_multiplicative_update>`):
  input LayerNorm, gated projections, the outgoing or incoming triangle
  contraction, output LayerNorm and gate.
* **Triangle attention**
  (:func:`cuet.triangle_attention <cuequivariance_torch.triangle_attention>`,
  :func:`cuex.triangle_attention <cuequivariance_jax.triangle_attention>`): attention
  along the rows of the pair representation with a per-head bias, a Boolean mask or
  right-padded ``kv_lengths``.
* **Attention pair bias**
  (:func:`cuet.attention_pair_bias <cuequivariance_torch.attention_pair_bias>`):
  attention over the single representation biased by a projection of the pair
  representation, with LayerNorm or RMSNorm, as in the AlphaFold 3 diffusion
  module.

Forward and backward are supported, so the operations serve inference and
training. The recommended way to run them is ``torch.compile(model,
mode="reduce-overhead")`` after a few warm-up iterations; under compile, the
kernel selection costs nothing per call. Calls on shapes or GPUs without a
specialized kernel run the generic implementation, so every supported shape works
on every supported GPU. The supported shapes, precision rules and switches for
each release are listed in the :doc:`changelog`.

Agent skills
------------

The repository ships three agent skills that teach coding assistants how to use
each package: `cuequivariance
<https://github.com/NVIDIA/cuEquivariance/blob/main/skills/cuequivariance/SKILL.md>`_,
`cuequivariance-jax
<https://github.com/NVIDIA/cuEquivariance/blob/main/skills/cuequivariance-jax/SKILL.md>`_
and `cuequivariance-torch
<https://github.com/NVIDIA/cuEquivariance/blob/main/skills/cuequivariance-torch/SKILL.md>`_.
Installed packages expose the same text through ``python -m cuequivariance skill``,
``python -m cuequivariance_jax skill`` and ``python -m cuequivariance_torch skill``.

Tutorials
---------

| :doc:`tutorials/irreps`
| :doc:`tutorials/layout`
| :doc:`tutorials/stp`
| :doc:`tutorials/poly`
| :doc:`tutorials/pytorch/index`
| :doc:`tutorials/jax/index`

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

Release Notes
-------------

.. toctree::
   :maxdepth: 2

   changelog

Status
------

cuEquivariance is in a Beta state. Beta products may not be fully functional, may
contain errors or design flaws, and may be changed at any time without notice.
Please report issues and feedback on `GitHub
<https://github.com/NVIDIA/cuEquivariance/issues>`_.
