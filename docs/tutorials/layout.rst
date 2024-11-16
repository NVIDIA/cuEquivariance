.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
   SPDX-License-Identifier: Apache-2.0

.. _tuto_layout:

Data Layouts
============

When representing a collection of irreps with multiplicities there are two ways to organize the data in memory:

   * **(ir, mul)** - Irreps are the outermost dimension.
   * **(mul, ir)** - Multiplicities are the outermost dimension. This is the layout used by `e3nn <https://github.com/e3nn/e3nn>`_.

.. image:: /_static/layout.png
   :alt: Illustration of data layouts
   :align: center

In the example above, all the blocks have a multiplicity of 4. Given the dimension of the irreps it could correspond to the irreps "4x0e + 4x1e + 4x2e".