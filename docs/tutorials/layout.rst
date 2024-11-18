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