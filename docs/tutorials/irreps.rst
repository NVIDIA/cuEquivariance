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

.. _tuto_irreps:

Groups and Representations
==========================

What Is a Group?
----------------

Imagine a set of operations that can be performed on an object—such as rotating a 3D model, flipping bits in a binary string, or shuffling elements in a list. These operations can be combined: performing one after another is equivalent to some single operation. In mathematics, especially in abstract algebra, such a set (of operations) with a composition law is called a *group*.

A **group** is a collection of elements (which could be numbers, functions, transformations, etc.) combined with a composition law (like addition or multiplication) that satisfies certain rules:

1. **Closure**: Combining any two elements produces another element in the group.
2. **Associativity**: The way operations are grouped does not change the result.
3. **Identity Element**: There exists an element that does not change other elements when combined with them.
4. **Inverse Element**: For every element, there exists another that reverses its effect.


What Is a Group Representation?
-------------------------------

*The action of a group on a vector space.*

A **group representation** is a way to map or "represent" each element of this abstract group to a concrete object, typically matrices or linear transformations acting on vector spaces. Essentially, it expresses abstract group operations in terms of matrix multiplication.

Why do this? Matrices and linear algebra are powerful tools with well-established methods for calculations and problem-solving. By representing group elements as matrices, one can leverage linear algebra to study and work with the group.

Irreducible Representations
---------------------------

A group representation can be decomposed into simpler, irreducible parts. An **irreducible representation** (irrep) is a representation that cannot be further decomposed into smaller representations.
In mathematical terms, an irrep is a representation that has no nontrivial invariant subspaces.

As a consequence, any representation can be expressed as a direct sum of irreducible representations. This decomposition is known as the **irreducible decomposition** of the representation.

.. _irreps-of-so3:

Irreps of :math:`SO(3)`
-----------------------

The group :math:`SO(3)` is the group of rotations in 3D space. It has a countable number of irreducible representations, each labeled by a non-negative integer. The irreps of :math:`SO(3)` are indexed by the non-negative integers :math:`l = 0, 1, 2, \ldots`. The dimension of the :math:`l`-th irrep is :math:`2l + 1`.
Some of the irreps of :math:`SO(3)` are well-known and have special names:

- The trivial representation (:math:`l = 0`) is one-dimensional and corresponds to scalar quantities that do not transform under rotations (e.g., mass, charge, etc.).
- The vector representation (:math:`l = 1`) is three-dimensional and corresponds to vectors in 3D space (e.g., position, velocity, force, etc.).

The higher-dimensional irreps are less common but are still important in physics and mathematics. They appear when we consider tensor products of vector representations.
For instance the :math:`l = 2` irrep is a five-dimensional representation that corresponds to rank-2 symmetric traceless tensors. The remaining degrees of freedom in a rank-2 tensor are captured by the :math:`l = 0` (the trace) and :math:`l = 1` (the antisymmetric part) irreps.


Irreps of :math:`O(3)`
----------------------

The group :math:`O(3)` is the group of rotations and reflections in 3D space. It can equivalently be described as the direct product of :math:`SO(3)` and :math:`Z_2`.
The group :math:`Z_2` is the group of two elements, the identity and the inversion. It's the smallest non-trivial group. It has two irreducible representations, both of dimension 1, called the even and odd representations.
The even representation corresponds to the trivial representation, and the odd representation corresponds to the sign: the identity is mapped to 1, and the inversion is mapped to -1.
The irreps of :math:`O(3)` are labeled by a pair of integers :math:`(l, p)`, where :math:`l` is a non-negative integer and :math:`p` is either 1 or -1. The dimension of the :math:`(l, p)`-th irrep is :math:`2l + 1`.


Example: Decomposing the Stress Tensor into Irreducible Representations
-----------------------------------------------------------------------

Let's explore the example of the stress tensor in the context of group representations. The stress tensor :math:`\sigma` is a :math:`3 \times 3` matrix that represents the internal forces acting within a material. Each element of this matrix describes how forces are transmitted in different directions at a point inside the material.
Why a Matrix? Materials can experience various types of mechanical stresses: tension, compression, shear, etc. The stress tensor captures all these forces in a single object.

Under a rotation of the coordinate system, the stress tensor transforms as:

.. math::

   \sigma \longrightarrow R \sigma R^T

where :math:`\sigma` is the stress tensor and :math:`R` is a rotation matrix.
It can be decomposed into irreducible components corresponding to different irreducible representations (irreps) of :math:`SO(3)`.

- Trace part: :math:`l = 0` irrep (scalar)
- Antisymmetric part: :math:`l = 1` irrep (vector)
- Symmetric traceless part: :math:`l = 2` irrep

Let see how representations are encoded in cuEquivariance.

The class :code:`Irreps`
------------------------

The :class:`Irreps <cuequivariance.Irreps>` class is designed to describe which irreducible representations and in which quantities are present in a given group representation.

Imagine you have a vector of dimension 80 and you know that the first 32 components are unaffected by rotations (scalar) and the next 48 components can be regrouped into 16 triplets that transform like vectors under rotations. You can describe this as follows:

.. jupyter-execute::

   import cuequivariance as cue

   cue.Irreps("SO3", "32x0 + 16x1")

The object above represents a group representation of the group :math:`SO(3)` (rotations in 3D space).
This example has two "segments". The first segment ``32x0`` indicates 32 copies of the trivial representation (``0``) and the second segment ``16x1`` indicates 16 copies of the vector representation (``1``).

The segments are separated by a ``+`` sign. Each segment consists of a number followed by ``x`` and then the irrep label (``0`` and ``1`` in this example). The number before ``x`` indicates how many copies of the irrep are present in the representation. The interpretation of the irrep label depends on the group.

As a convenience, a multiplicity of 1 can be omitted: ``1x2`` can be written as ``2``.

cuEquivariance provides irreps for the following groups: :math:`SO(3)`, :math:`O(3)` and :math:`SU(2)`.
The first argument to the :class:`Irreps <cuequivariance.Irreps>` constructor is the group name, which is a shorthand for :class:`cue.SO3 <cuequivariance.SO3>`, :class:`cue.O3 <cuequivariance.O3>` and :class:`cue.SU2 <cuequivariance.SU2>` respectively. Here is an example for the group :math:`SU(2)`:

.. jupyter-execute::

   cue.Irreps("SU2", "6x1/2")



The order is important
----------------------

The :class:`Irreps <cuequivariance.Irreps>` class is most of the time used to "tag" the data. So typically, you will have an ``Irreps`` object associated with a PyTorch tensor or a NumPy array. The order in which you declare the irreps will be the order in which the data is stored in the tensor or array.
For example, let say you have a tensor of 4 numbers ``[1.0, 2.0, 3.0, 4.0]``.
If you declare the irreps as ``"1x0 + 1x1"`` the first number (``1.0``) will be associated with the scalar representation and last numbers (``[2.0, 3.0, 4.0]``) will be associated with the vector representation. But if you declare the irreps as ``"1x1 + 1x0"`` the first three numbers (``[1.0, 2.0, 3.0]``) will be associated with the vector representation and the last number (``4.0``) will be associated with the scalar representation.

If you input data in the wrong order, transformations will misinterpret it.
Downstream tasks (e.g., equivariant layers in neural networks) rely on the specific structure.


Set a default group
-------------------

You can use the :func:`cue.assume <cuequivariance.assume>` to set a default group for all the irreps you create. This is useful when you are working with a single group and you don't want to specify it every time.

.. jupyter-execute::

   with cue.assume(cue.SU2):
      irreps = cue.Irreps("6x1/2")
      print(irreps)

