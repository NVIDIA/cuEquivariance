# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

import cuequivariance as cue


def test_compute_last_operand_1():
    d = cue.SegmentedTensorProduct.from_subscripts("uv_vw_uw")
    d.add_path(None, None, None, c=1.0, dims={"u": 2, "v": 3, "w": 4})

    x0 = np.random.randn(d.operands[0].size)
    x1 = np.random.randn(d.operands[1].size)
    x2 = cue.segmented_polynomials.compute_last_operand(d, x0, x1)

    x2_ = (x0.reshape(2, 3) @ x1.reshape(3, 4)).reshape(-1)
    np.testing.assert_allclose(x2_, x2)


def test_compute_last_operand_2():
    d = cue.SegmentedTensorProduct.from_subscripts("uv,vw,uw")
    d.add_path(None, None, None, c=1.0, dims={"u": 2, "v": 3, "w": 4})

    x0 = np.random.randn(10, d.operands[0].size)
    x1 = np.random.randn(10, d.operands[1].size)
    x2 = cue.segmented_polynomials.compute_last_operand(d, x0, x1)

    x2_ = (x0.reshape(10, 2, 3) @ x1.reshape(10, 3, 4)).reshape(10, -1)
    np.testing.assert_allclose(x2_, x2)


def test_compute_last_operand_3():
    d = cue.SegmentedTensorProduct.from_subscripts("uv,vw,uw")
    d.add_path(None, None, None, c=1.0, dims={"u": 2, "v": 3, "w": 4})

    x0 = np.random.randn(1, d.operands[0].size, 5)
    x1 = np.random.randn(10, d.operands[1].size, 1)
    x2 = cue.segmented_polynomials.compute_last_operand(
        d, x0, x1, segment_axes=[1, 1, 1]
    )

    x2_ = np.einsum(
        "ZuvA,ZvwA->ZuwA", x0.reshape(1, 2, 3, 5), x1.reshape(10, 3, 4, 1)
    ).reshape(10, -1, 5)
    np.testing.assert_allclose(x2_, x2)


def test_compute_last_operand_4():
    d = cue.SegmentedTensorProduct.from_subscripts("iuv_jvw_kuw+ijk")
    c = np.random.randn(2, 3, 4)
    d.add_path(None, None, None, c=c, dims={"u": 2, "v": 3, "w": 4})

    x0 = np.random.randn(d.operands[0].size)
    x1 = np.random.randn(d.operands[1].size)
    x2 = cue.segmented_polynomials.compute_last_operand(d, x0, x1)

    x2_ = np.einsum(
        "ijk,iuv,jvw->kuw", c, x0.reshape(2, 2, 3), x1.reshape(3, 3, 4)
    ).reshape(-1)
    np.testing.assert_allclose(x2_, x2)


def test_primitive_compute_last_operand():
    d = cue.SegmentedTensorProduct.from_subscripts("iuv_jvw_kuw+ijk")
    c = np.random.randn(2, 3, 4)
    d.add_path(None, None, None, c=c, dims={"u": 2, "v": 3, "w": 4})

    x0 = np.random.randn(d.operands[0].size)
    x1 = np.random.randn(d.operands[1].size)

    x2_ = np.einsum(
        "ijk,iuv,jvw->kuw", c, x0.reshape(2, 2, 3), x1.reshape(3, 3, 4)
    ).reshape(-1)

    d = d.to_dict(True)

    x2 = cue.segmented_polynomials.primitive_compute_last_operand(
        [ope["subscripts"] for ope in d["operands"]],
        d["coefficient_subscripts"],
        [ope["segments"] for ope in d["operands"]],
        (),
        [ope["segment_offsets"] for ope in d["operands"]],
        [ope["segment_sizes"] for ope in d["operands"]],
        d["paths"]["indices"],
        [np.array(c) for c in d["paths"]["coefficients"]],
        [0] * len(d["operands"]),
        np.result_type(x0, x1),
        x0,
        x1,
    )

    np.testing.assert_allclose(x2_, x2)
