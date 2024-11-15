.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
   SPDX-License-Identifier: Apache-2.0

.. _tuto_irreps:

Group representations
=====================

What Is a Group Representation?
-------------------------------

Imagine a set of operations that can be performed on an object—such as rotating a 3D model, flipping bits in a binary string, or shuffling elements in a list. These operations can be combined: performing one after another is equivalent to some single operation. In mathematics, especially in abstract algebra, such a set (of operations) with a composition law is called a *group*.

A **group** is a collection of elements (which could be numbers, functions, transformations, etc.) combined with a composition law (like addition or multiplication) that satisfies certain rules:

1. **Closure**: Combining any two elements produces another element in the group.
2. **Associativity**: The way operations are grouped does not change the result.
3. **Identity Element**: There exists an element that does not change other elements when combined with them.
4. **Inverse Element**: For every element, there exists another that reverses its effect.

A **group representation** is a way to map or "represent" each element of this abstract group to a concrete object, typically matrices or linear transformations acting on vector spaces. Essentially, it expresses abstract group operations in terms of matrix multiplication.

Why do this? Matrices and linear algebra are powerful tools with well-established methods for calculations and problem-solving. By representing group elements as matrices, one can leverage linear algebra to study and work with the group.

Irreducible Representations
---------------------------

A group representation can be decomposed into simpler, irreducible parts. An **irreducible representation** (irrep) is a representation that cannot be further decomposed into smaller, nontrivial representations. In other words, an irrep is a representation that has no nontrivial invariant subspaces.

As a consequence, any representation can be expressed as a direct sum of irreducible representations. This decomposition is known as the **irreducible decomposition** of the representation.

:code:`Irreps`
--------------

The :class:`Irreps <cuequivariance.Irreps>` class is designed to describe which irreducible representations and in which quantities are present in a given group representation.

.. jupyter-execute::

   import cuequivariance as cue

   cue.Irreps("SO3", "32x0 + 16x1")

The object above represents a group representation of the group :math:`SO(3)` (rotations in 3D space).
This example has two "segments". The first segment ``32x0`` indicates 32 copies of the trivial representation (0) and the second segment ``16x1`` indicates 16 copies of the vector representation (1).

The segments are separated by a ``+`` sign. Each segment consists of a number followed by ``x`` and then the irrep label. The number indicates how many copies of the irrep are present in the representation. The interpretation of the irrep label depends on the group.

As a convenience, a multiplicity of 1 can be omitted: ``1x2`` can be written as ``2``.

cuEquivariance provides irreps for the following groups: :math:`SO(3)`, :math:`O(3)` and :math:`SU(2)`.

.. jupyter-execute::

   cue.Irreps("SU2", "6x1/2")

The first argument to the :class:`Irreps <cuequivariance.Irreps>` constructor is the group name, it is a shorthand for :class:`cue.SO3 <cuequivariance.SO3>`, :class:`cue.O3 <cuequivariance.O3>` and :class:`cue.SU2 <cuequivariance.SU2>` respectively.
If needed, you can also create custom irreps, see :ref:`custom-irreps` below.

.. jupyter-execute::

   irreps = cue.Irreps("O3", "10x0e + 2x1o")
   irreps

Here are some useful properties of the :class:`Irreps <cuequivariance.Irreps>` object:

.. jupyter-execute::

   irreps.dim

.. jupyter-execute::

   irreps.filter(drop="0e")

The order is important
----------------------

The ordering of the representations is (often) meaningful, for example, these two :class:`Irreps <cuequivariance.Irreps>` objects are not equal:

.. jupyter-execute::

   assert cue.Irreps("SO3", "32x0 + 16x1") != cue.Irreps("SO3", "16x1 + 32x0")

``32x0 + 16x1``: First 32 components correspond to scalar (0). Next 48 components (16 vector representations × 3 components each) correspond to the vector representations.

``16x1 + 32x0``: First 48 components are for the vector representations. Last 32 components are scalars.

Thus, the ordering affects how you interpret and operate on the data. For example:
If you input data in the wrong order, transformations will misinterpret it.
Downstream tasks (e.g., equivariant layers in neural networks) rely on the specific structure.

.. _irreps-of-so3:

Irreps of :math:`SO(3)`
-----------------------

The group :math:`SO(3)` is the group of rotations in 3D space. It has a countable number of irreducible representations, each labeled by a non-negative integer. The irreps of :math:`SO(3)` are indexed by the non-negative integers :math:`l = 0, 1, 2, \ldots`. The dimension of the :math:`l`-th irrep is :math:`2l + 1`.
Some of the irreps of :math:`SO(3)` are well-known and have special names:

- The trivial representation (0) is one-dimensional and corresponds to scalar quantities that do not transform under rotations (e.g., mass, charge, etc.).
- The vector representation (1) is three-dimensional and corresponds to vectors in 3D space (e.g., position, velocity, force, etc.).

The higher-dimensional irreps are less common but are still important in physics and mathematics. They appear when we consider tensor products of vector representations.
For instance the :math:`l = 2` irrep is a five-dimensional representation that corresponds to rank-2 symmetric traceless tensors. The remaining degrees of freedom in a rank-2 tensor are captured by the :math:`l = 0` (the trace) and :math:`l = 1` (the antisymmetric part) irreps.


Irreps of :math:`O(3)`
----------------------

The group :math:`O(3)` is the group of rotations and reflections in 3D space. It can equivalently be described as the direct product of :math:`SO(3)` and :math:`Z_2`.
:math:`Z_2` is the group of two elements, the identity and the inversion. It's the smallest non-trivial group. It has two irreducible representations, both of dimension 1, called the even and odd representations.
The even representation corresponds to the trivial representation, and the odd representation corresponds to the sign: the identity is mapped to 1, and the inversion is mapped to -1.
The irreps of :math:`O(3)` are labeled by a pair of integers :math:`(l, p)`, where :math:`l` is a non-negative integer and :math:`p` is either 1 or -1. The dimension of the :math:`(l, p)`-th irrep is :math:`2l + 1`.


Set a default group
-------------------

You can use the :func:`cue.assume <cuequivariance.assume>` context manager to fix the group.

.. jupyter-execute::

   with cue.assume(cue.SU2):
      irreps = cue.Irreps("6x1/2")
      print(irreps)


.. _custom-irreps:

Custom Irreps
-------------

In some cases, you may want to define a custom set of irreducible representations of a group.
Here is a simple example of how to define the irreps of the group :math:`Z_2`. For this we need to define a class that inherits from :class:`cue.Irrep <cuequivariance.Irrep>` and implement the required methods.

.. jupyter-execute::

   from __future__ import annotations

   import re
   from typing import Iterator

   import numpy as np


   class Z2(cue.Irrep):
      odd: bool

      def __init__(rep: Z2, odd: bool):
         rep.odd = odd

      @classmethod
      def regexp_pattern(cls) -> re.Pattern:
         return re.compile(r"(odd|even)")

      @classmethod
      def from_string(cls, string: str) -> Z2:
         return cls(odd=string == "odd")

      def __repr__(rep: Z2) -> str:
         return "odd" if rep.odd else "even"

      def __mul__(rep1: Z2, rep2: Z2) -> Iterator[Z2]:
         return [Z2(odd=rep1.odd ^ rep2.odd)]

      @classmethod
      def clebsch_gordan(cls, rep1: Z2, rep2: Z2, rep3: Z2) -> np.ndarray:
         if rep3 in rep1 * rep2:
               return np.array(
                  [[[[1]]]]
               )  # (number_of_paths, rep1.dim, rep2.dim, rep3.dim)
         else:
               return np.zeros((0, 1, 1, 1))

      @property
      def dim(rep: Z2) -> int:
         return 1

      def __lt__(rep1: Z2, rep2: Z2) -> bool:
         # False < True
         return rep1.odd < rep2.odd

      @classmethod
      def iterator(cls) -> Iterator[Z2]:
         for odd in [False, True]:
               yield Z2(odd=odd)

      def discrete_generators(rep: Z2) -> np.ndarray:
         if rep.odd:
               return -np.ones((1, 1, 1))  # (number_of_generators, rep.dim, rep.dim)
         else:
               return np.ones((1, 1, 1))

      def continuous_generators(rep: Z2) -> np.ndarray:
         return np.zeros((0, rep.dim, rep.dim))  # (lie_dim, rep.dim, rep.dim)

      def algebra(self) -> np.ndarray:
         return np.zeros((0, 0, 0))  # (lie_dim, lie_dim, lie_dim)


   cue.Irreps(Z2, "13x odd + 6x even")
