.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
   SPDX-License-Identifier: Apache-2.0

Data Layouts
============

When representing a collection of irreps with multiplicities there is two ways to organize the data in memory:

   * **(ir, mul)** - Irreps are the outermost dimension.
   * **(mul, ir)** - Multiplicities are the outermost dimension. This is the layout used by ``e3nn``.

.. image:: /_static/layout.png
   :alt: Illustration of data layouts
   :align: center