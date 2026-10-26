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
import jax.numpy as jnp
import numpy as np
import pytest

import cuequivariance as cue
import cuequivariance_jax as cuex


@cue.assume("SO3", cue.ir_mul)
def test_segments():
    x = cuex.RepArray("2x0 + 1", jnp.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    x0, x1 = x.segments
    assert x0.shape == (1, 2)
    assert x1.shape == (3, 1)
    y = cuex.from_segments("2x0 + 1", [x0, x1], x.shape)
    assert x.irreps == y.irreps
    assert x.layout == y.layout
    assert jnp.allclose(x.array, y.array)


@cue.assume("SO3", cue.ir_mul)
def test_slice_by_mul():
    x = cuex.RepArray("2x0 + 1", jnp.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    x = x.slice_by_mul[1:]
    assert x.irreps == cue.Irreps("0 + 1")
    assert x.layout == cue.ir_mul
    assert jnp.allclose(x.array, jnp.array([1.0, 0.0, 0.0, 0.0]))


@cue.assume("SO3", cue.ir_mul)
def test_rep_array_constructor_indexing_and_addition_edges():
    with pytest.raises(ValueError, match="Array shape"):
        cuex.RepArray({0: cue.SO3(1)}, jnp.zeros((2,)))

    array = cuex.RepArray({1: "1"}, jnp.arange(6.0).reshape(2, 3))
    with pytest.raises(ValueError, match="another RepArray"):
        cuex.RepArray({}, array)

    assert array.is_irreps_array()
    assert array[None].shape == (1, 2, 3)
    assert array[None].reps == {2: array.rep(-1)}
    assert jnp.array_equal(array[:1].array, array.array[:1])
    indices = jnp.array([1, 0])
    assert jnp.array_equal(array[indices].array, array.array[indices])

    non_irreps = cuex.RepArray({0: cue.SO3(1)}, jnp.zeros((3, 2)))
    multi_axis = cuex.RepArray({0: cue.SO3(1), 1: cue.SO3(1)}, jnp.zeros((3, 3)))
    assert not non_irreps.is_irreps_array()
    assert not multi_axis.is_irreps_array()
    with pytest.raises(ValueError, match="No Rep for axis 1"):
        non_irreps.rep(1)

    scalars = cuex.RepArray("2x0", jnp.array([2.0, 3.0]))
    other = cuex.RepArray("2x0", jnp.array([5.0, -1.0]))
    summed = scalars + other
    assert summed.reps == scalars.reps and summed.layout == scalars.layout
    np.testing.assert_allclose(summed.array, [7.0, 2.0])
    assert scalars + 0 is scalars
    assert 0 + scalars is scalars
    assert jnp.array_equal((-scalars).array, jnp.array([-2.0, -3.0]))
    assert jnp.array_equal((scalars - scalars).array, jnp.zeros((2,)))
    with pytest.raises(AssertionError):
        _ = scalars + 1
    with pytest.raises(ValueError, match="not a RepArray"):
        _ = scalars + object()
    with pytest.raises(ValueError, match="different reps"):
        _ = scalars + cuex.RepArray("0 + 0", jnp.array([1.0, 1.0]))


@cue.assume("SO3", cue.ir_mul)
def test_irreps_array_layout_and_empty_operation_edges():
    values = jnp.arange(5.0)
    mixed = cuex.RepArray("2x0 + 1", values)

    assert mixed.filter(mask=[True, True]) is mixed
    empty = mixed.filter(mask=[False, False])
    assert empty.shape == (0,)
    assert empty.irreps.dim == 0
    assert jnp.array_equal(empty.array, jnp.zeros((0,)))
    kept = mixed.filter(keep="0")
    assert kept.irreps == cue.Irreps("2x0")
    assert jnp.array_equal(kept.array, values[:2])

    mul_ir = cuex.RepArray("0 + 0", jnp.array([1.0, 2.0]), cue.mul_ir)
    simplified = mul_ir.simplify()
    assert simplified.irreps == cue.Irreps("2x0")
    assert jnp.array_equal(simplified.array, mul_ir.array)

    for layout in (cue.ir_mul, cue.mul_ir):
        vector = cuex.RepArray("1", jnp.array([1.0, 2.0, 3.0]), layout)
        transformed = vector.transform(jnp.zeros((3,)))
        assert transformed.layout == layout
        assert jnp.allclose(transformed.array, vector.array)
        rotated = vector.transform(jnp.array([np.pi, 0.0, 0.0]))
        np.testing.assert_allclose(rotated.array, [1.0, -2.0, -3.0], atol=1e-6)

    tensor = cuex.RepArray({0: cue.SO3(1)}, jnp.eye(3))
    assert jnp.allclose(tensor.transform(jnp.zeros((3,))).array, tensor.array)

    original = cuex.RepArray("2x0 + 1", values, cue.ir_mul)
    assert original.change_layout(cue.ir_mul) is original
    round_trip = original.change_layout(cue.mul_ir).change_layout(cue.ir_mul)
    assert round_trip.layout == cue.ir_mul
    assert jnp.array_equal(round_trip.array, original.array)

    batched_values = jnp.arange(10.0).reshape(2, 5)
    for layout in (cue.ir_mul, cue.mul_ir):
        batched = cuex.RepArray("2x0 + 1", batched_values, layout)
        moved = batched.move_axis_to_mul(-2)
        assert moved.shape == (10,)
        assert moved.irreps == 2 * batched.irreps
        if layout == cue.ir_mul:
            expected = jnp.moveaxis(batched_values, 0, -1).reshape((10,))
        else:
            expected = jnp.concatenate(
                [segment.reshape((-1,)) for segment in batched.segments]
            )
        assert jnp.array_equal(moved.array, expected)


@cue.assume("SO3", cue.ir_mul)
def test_from_segments_and_slice_by_mul_validation_edges():
    with pytest.raises(ValueError, match="segments must have ndim"):
        cuex.from_segments("0", [jnp.ones((1,))], (1,))
    with pytest.raises(ValueError, match="number of segments"):
        cuex.from_segments("0", [], (1,))

    empty = cuex.from_segments("", [], (2, 0), cue.ir_mul, jnp.float32)
    assert empty.shape == (2, 0)
    assert empty.dtype == jnp.float32
    assert jnp.array_equal(empty.array, jnp.zeros((2, 0), dtype=jnp.float32))

    values = jnp.arange(5.0)
    for layout in (cue.ir_mul, cue.mul_ir):
        array = cuex.RepArray("2x0 + 1", values, layout)
        selected = array.slice_by_mul[1:3]
        assert selected.irreps == cue.Irreps("0 + 1")
        assert selected.layout == layout
        assert jnp.array_equal(selected.array, values[1:])

        with pytest.raises(IndexError, match="only supports one slices"):
            _ = array.slice_by_mul[1]
        with pytest.raises(NotImplementedError, match="does not support strides"):
            _ = array.slice_by_mul[::2]


@cue.assume("SO3", cue.ir_mul)
def test_rep_array_non_scalar_layout_and_simplification_values():
    array = cuex.RepArray("2x1", jnp.arange(1.0, 7.0))
    np.testing.assert_array_equal(
        array.change_layout(cue.mul_ir).array, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    )
    split = cuex.RepArray("1 + 1", jnp.arange(1.0, 7.0))
    merged = split.simplify()
    assert merged.irreps == cue.Irreps("2x1")
    np.testing.assert_array_equal(merged.array, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
