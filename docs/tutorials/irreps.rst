.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
   SPDX-License-Identifier: Apache-2.0

.. _tuto_irreps:

Group representations
=====================

What Is a Group Representation?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Imagine a set of operations that can be performed on an object—such as rotating a 3D model, flipping bits in a binary string, or shuffling elements in a list. These operations can be combined: performing one after another is equivalent to some single operation. In mathematics, especially in abstract algebra, such a set (of operations) with a composition law is called a *group*.

A **group** is a collection of elements (which could be numbers, functions, transformations, etc.) combined with a composition law (like addition or multiplication) that satisfies certain rules:

1. **Closure**: Combining any two elements produces another element in the group.
2. **Associativity**: The way operations are grouped does not change the result.
3. **Identity Element**: There exists an element that does not change other elements when combined with them.
4. **Inverse Element**: For every element, there exists another that reverses its effect.

A **group representation** is a way to map or "represent" each element of this abstract group to a concrete object, typically matrices or linear transformations acting on vector spaces. Essentially, it expresses abstract group operations in terms of matrix multiplication.

Why do this? Matrices and linear algebra are powerful tools with well-established methods for calculations and problem-solving. By representing group elements as matrices, one can leverage linear algebra to study and work with the group.

Irreducible Representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A group representation can be decomposed into simpler, irreducible parts. An **irreducible representation** (irrep) is a representation that cannot be further decomposed into smaller, nontrivial representations. In other words, an irrep is a representation that has no nontrivial invariant subspaces.

As a consequence, any representation can be expressed as a direct sum of irreducible representations. This decomposition is known as the **irreducible decomposition** of the representation.

:code:`Irreps` in cuEquivariance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :code:`Irreps` class in cuEquivariance is designed to describe which irreducible representations and in which quantities are present in a given group representation.

.. jupyter-execute::

   import cuequivariance as cue

   cue.Irreps("SO3", "32x0 + 16x1")

The object above represents a group representation of the group :math:`SO(3)` (rotations in 3D space) with 32 copies of the trivial representation (0) and 16 copies of the first nontrivial representation (1).
Note that the ordering of the representations is (often) meaningful, for example, these two :code:`Irreps` objects are not equal:

.. jupyter-execute::

   cue.Irreps("SO3", "32x0 + 16x1") != cue.Irreps("SO3", "16x1 + 32x0")

because the order of the representations is different. In the first case we have 32 copies of the trivial representation followed by 16 copies of vector representation, while in the second case we have 16 copies of the vector representation followed by 32 copies of the trivial representation.

cuEquivariance provides irreps for the following groups: :math:`SO(3)`, :math:`O(3)` and :math:`SU(2)`.

.. jupyter-execute::

   cue.Irreps("SU2", "6x1/2")

The first argument to the :code:`Irreps` constructor is the group name, it is a shorthand for :code:`cue.SO3`, :code:`cue.O3` and :code:`cue.SU2` respectively.
If needed, you can also create custom irreps, see :ref:`custom-irreps`.

.. jupyter-execute::

   irreps = cue.Irreps("O3", "10x0e + 2x1o")
   irreps

Here are some useful properties of the :code:`Irreps` object:

.. jupyter-execute::

   irreps.dim

.. jupyter-execute::

   irreps.filter(drop="0e")


.. _irreps-of-so3

Irreps of :math:`SO(3)`
^^^^^^^^^^^^^^^^^^^^^^^

The group :math:`SO(3)` is the group of rotations in 3D space. It has a countable number of irreducible representations, each labeled by a non-negative integer. The irreps of :math:`SO(3)` are indexed by the non-negative integers :math:`l = 0, 1, 2, \ldots`. The dimension of the :math:`l`-th irrep is :math:`2l + 1`.
Some of the irreps of :math:`SO(3)` are well-known and have special names:

- The trivial representation (0) is one-dimensional and corresponds to scalar quantities that do not transform under rotations (e.g., mass, charge, etc.).
- The vector representation (1) is three-dimensional and corresponds to vectors in 3D space (e.g., position, velocity, force, etc.).

The higher-dimensional irreps are less common but are still important in physics and mathematics. They appear when we consider tensor products of vector representations.
For instance the :math:`l = 2` irrep is a five-dimensional representation that corresponds to rank-2 symmetric traceless tensors. The remaining degrees of freedom in a rank-2 tensor are captured by the :math:`l = 0` (the trace) and :math:`l = 1` (the antisymmetric part) irreps.


Set a default group
^^^^^^^^^^^^^^^^^^^

You can use the :code:`assume` context manager to fix the group.

.. jupyter-execute::

   with cue.assume(cue.SU2):
      irreps = cue.Irreps("6x1/2")
      print(irreps)


.. _custom-irreps:

Custom Irreps
^^^^^^^^^^^^^

In some cases, you may want to define a custom set of irreducible representations of a group.
Here is a simple example of how to define the irreps of the group :math:`Z_2`. For this we need to define a class that inherits from :code:`cue.Irrep` and implement the required methods.

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
