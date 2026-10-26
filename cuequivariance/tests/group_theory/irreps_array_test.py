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
import pytest
from cuequivariance.group_theory.irreps_array import numpy_irreps_array

import cuequivariance as cue


def test_init():
    irreps = cue.Irreps("SU2", "1/2 + 16x2")
    array = np.zeros((128, irreps.dim))

    cue.NumpyIrrepsArray(irreps, array, cue.mul_ir)
    cue.NumpyIrrepsArray(irreps, array, cue.ir_mul)
    cue.NumpyIrrepsArray(irreps, array, "mul_ir")
    cue.NumpyIrrepsArray(irreps, array, "ir_mul")

    with pytest.raises(ValueError):
        cue.NumpyIrrepsArray(irreps, array, "invalid")
    with pytest.raises(ValueError):
        cue.NumpyIrrepsArray(irreps, np.zeros((128, 128, 128)), cue.mul_ir)


def test_sort():
    x = cue.NumpyIrrepsArray(cue.Irreps("SU2", "1/2 + 0 + 1"), np.arange(6), cue.mul_ir)
    y = x.sort()

    assert y.irreps == cue.Irreps("SU2", "0 + 1/2 + 1")
    np.testing.assert_array_equal(x.segments[0], y.segments[1])
    np.testing.assert_array_equal(x.segments[1], y.segments[0])
    np.testing.assert_array_equal(x.segments[2], y.segments[2])


def test_array_properties_repr_equality_and_reshape():
    irreps = cue.Irreps("SU2", "0 + 1/2 + 1")
    compact = cue.NumpyIrrepsArray(irreps, np.arange(6), cue.mul_ir)
    batched = cue.NumpyIrrepsArray(
        irreps, np.arange(12, dtype=np.float32).reshape(2, 6), cue.mul_ir
    )

    assert compact.dtype == np.dtype(np.int64)
    assert batched.shape == (2, 6)
    assert batched.ndim == 2
    assert len(batched) == 2
    assert repr(compact) == f"{irreps} [0 1 2 3 4 5]"
    assert repr(batched).startswith(f"{irreps}\n")
    assert compact == cue.NumpyIrrepsArray(irreps, np.arange(6), cue.mul_ir)
    assert compact != cue.NumpyIrrepsArray(irreps, np.arange(6) + 1, cue.mul_ir)
    assert compact.__eq__(object()) is NotImplemented
    np.testing.assert_array_equal((-compact).array, -compact.array)

    reshaped = batched.reshape((1, 2, -1))
    assert reshaped.shape == (1, 2, irreps.dim)
    np.testing.assert_array_equal(reshaped.array, batched.array.reshape(1, 2, 6))
    with pytest.raises(ValueError, match="Last dimension of shape"):
        batched.reshape((2, 3))
    with pytest.raises(TypeError, match="Expected array with shape"):
        cue.NumpyIrrepsArray(irreps, object(), cue.mul_ir)


def test_array_layout_simplify_merge_sort_and_filter():
    irreps = cue.Irreps("SU2", "0 + 1/2 + 1")
    values = np.arange(12).reshape(2, 6)
    x = cue.NumpyIrrepsArray(irreps, values, cue.mul_ir)

    assert x.filter() is x
    assert x.merge_consecutive().irreps == irreps
    kept = x.filter(keep=lambda mul_ir: mul_ir.ir != irreps[0].ir)
    assert kept.irreps == cue.Irreps("SU2", "1/2 + 1")
    np.testing.assert_array_equal(kept.array, values[..., 1:])
    dropped = x.filter(drop=lambda mul_ir: mul_ir.ir == irreps[-1].ir)
    assert dropped.irreps == cue.Irreps("SU2", "0 + 1/2")
    np.testing.assert_array_equal(dropped.array, values[..., :3])
    assert x.filter(keep="1").irreps == cue.Irreps("SU2", "1")
    assert x.filter(drop="0").irreps == cue.Irreps("SU2", "1/2 + 1")
    with pytest.raises(ValueError, match="Only one of `keep` or `drop`"):
        x.filter(keep="0", drop="1")

    unsorted = cue.NumpyIrrepsArray(
        cue.Irreps("SU2", "1/2 + 0 + 1"), values, cue.mul_ir
    )
    sorted_x = unsorted.sort()
    np.testing.assert_array_equal(
        sorted_x.array,
        np.concatenate([values[..., 2:3], values[..., :2], values[..., 3:]], axis=-1),
    )

    split_irreps = cue.Irreps("SU2", "2x1 + 1x1")
    ir_mul = cue.NumpyIrrepsArray(split_irreps, np.arange(9), cue.ir_mul)
    simplified = ir_mul.simplify()
    assert simplified.irreps == cue.Irreps("SU2", "3x1")
    np.testing.assert_array_equal(
        simplified.segments[0], np.concatenate(ir_mul.segments, axis=-1)
    )
    assert ir_mul.merge_consecutive().irreps == cue.Irreps("SU2", "3x1")

    layout_irreps = cue.Irreps("SU2", "2x1")
    layout_array = cue.NumpyIrrepsArray(layout_irreps, np.arange(6), cue.ir_mul)
    assert layout_array.change_layout(cue.ir_mul) is layout_array
    mul_ir = layout_array.change_layout(cue.mul_ir)
    assert mul_ir.layout == cue.mul_ir
    np.testing.assert_array_equal(mul_ir.array, np.array([0, 2, 4, 1, 3, 5]))
    np.testing.assert_array_equal(
        mul_ir.change_layout(cue.ir_mul).array, layout_array.array
    )


@pytest.mark.parametrize("operation", ["keep", "drop"])
def test_filter_rejects_invalid_normalized_irreps(monkeypatch, operation):
    array = cue.NumpyIrrepsArray(cue.Irreps("SU2", "0 + 1"), np.arange(4), cue.mul_ir)
    with pytest.raises(ValueError):
        array.filter(**{operation: "not-an-irrep"})
    monkeypatch.setattr(numpy_irreps_array, "into_list_of_irrep", lambda *_: [object()])

    with pytest.raises(ValueError, match=f"Invalid `{operation}`"):
        array.filter(**{operation: "0"})


def test_from_segments_validates_and_preserves_segment_values():
    irreps = cue.Irreps("SU2", "2x1 + 0")
    vector_segment = np.arange(12).reshape(2, 2, 3)
    scalar_segment = np.arange(2).reshape(2, 1, 1)
    result = cue.from_segments(
        irreps, [vector_segment, scalar_segment], cue.mul_ir, (2,)
    )

    np.testing.assert_array_equal(
        result.array,
        np.concatenate(
            [vector_segment.reshape(2, 6), scalar_segment.reshape(2, 1)], axis=-1
        ),
    )
    np.testing.assert_array_equal(result.segments[0], vector_segment)
    np.testing.assert_array_equal(result.segments[1], scalar_segment)

    with pytest.raises(ValueError, match="Expected 2 segments, got 1"):
        cue.from_segments(irreps, [vector_segment], cue.mul_ir, (2,))
    with pytest.raises(ValueError, match=r"Expected shape\[:-2\]"):
        cue.from_segments(
            irreps,
            [vector_segment, scalar_segment],
            cue.mul_ir,
            (3,),
        )
    with pytest.raises(ValueError, match=r"Expected shape\[-2:\]"):
        cue.from_segments(
            irreps,
            [np.zeros((2, 2, 2)), scalar_segment],
            cue.mul_ir,
            (2,),
        )


def test_concatenate_irreps_layouts_and_arrays():
    first_irreps = cue.Irreps("SU2", "0")
    second_irreps = cue.Irreps("SU2", "1")
    assert cue.concatenate([first_irreps, second_irreps]) == cue.Irreps("SU2", "0 + 1")
    assert cue.concatenate(
        [
            cue.IrrepsAndLayout(first_irreps, cue.mul_ir),
            cue.IrrepsAndLayout(second_irreps, cue.mul_ir),
        ]
    ) == cue.IrrepsAndLayout(cue.Irreps("SU2", "0 + 1"), cue.mul_ir)

    values = np.arange(2, dtype=np.float32).reshape(2, 1)
    first = cue.NumpyIrrepsArray(first_irreps, values, cue.mul_ir)
    second = cue.NumpyIrrepsArray(first_irreps, values + 2, cue.mul_ir)
    joined = cue.concatenate([first, second])
    assert joined.irreps == cue.Irreps("SU2", "0 + 0")
    np.testing.assert_array_equal(
        joined.array, np.concatenate([values, values + 2], axis=-1)
    )

    with pytest.raises(ValueError, match="Expected at least one input"):
        cue.concatenate([])
    with pytest.raises(TypeError, match="Expected a list of IrrepsArray"):
        cue.concatenate([first, first_irreps])
    with pytest.raises(ValueError, match="same irrep class"):
        cue.concatenate(
            [
                first,
                cue.NumpyIrrepsArray(cue.Irreps("SO3", "0"), values, cue.mul_ir),
            ]
        )
    with pytest.raises(ValueError, match="same layout"):
        cue.concatenate(
            [
                first,
                cue.NumpyIrrepsArray(first_irreps, values, cue.ir_mul),
            ]
        )
