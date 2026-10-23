# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import pytest

import cuequivariance as cue


def _one_path_stp(subscripts, *segment_shapes, coefficients=1.0):
    d = cue.SegmentedTensorProduct.from_subscripts(subscripts)
    for oid, shape in enumerate(segment_shapes):
        d.add_segment(oid, shape)
    d.add_path(*([0] * len(segment_shapes)), c=coefficients)
    return d


def _make_etp(d, operands, *, symmetrize=True):
    # The deprecated wrapper warns on every construction, including transforms.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return cue.EquivariantTensorProduct(d, operands, symmetrize=symmetrize)


def _scalar_rep(multiplicity=1):
    return cue.IrrepsAndLayout(cue.Irreps("SO3", f"{multiplicity}x0"), cue.ir_mul)


def test_sequence_constructor_and_repeated_input_helpers():
    scalar = _scalar_rep()
    operands = [scalar, scalar, scalar]
    ds = [
        _one_path_stp("i,i,i,i", (1,), (1,), (1,), (1,)),
        _one_path_stp("i,i,i", (1,), (1,), (1,)),
        _one_path_stp("i,i", (1,), (1,)),
    ]

    with pytest.warns(DeprecationWarning, match="deprecated"):
        etp = cue.EquivariantTensorProduct(ds, operands)

    assert etp.operands == tuple(operands)
    assert etp.num_operands == 3
    assert etp.num_inputs == 2
    assert etp.inputs == tuple(operands[:-1])
    assert etp.output == operands[-1]
    assert etp._degrees(0) == {1}
    assert etp._degrees(1) == {0, 1, 2}
    assert "0..2" in repr(etp)
    assert hash(etp) == hash(_make_etp(ds, operands))
    with pytest.raises(AssertionError):
        _ = etp.d

    assert etp.map_operands(2) == [0, 2]
    assert etp.map_operands(3) == [0, 1, 2]
    assert etp.map_operands(4) == [0, 1, 1, 2]
    assert etp.stp_operand(0) == ds[0].operands[0]
    assert etp.stp_operand(1) == ds[0].operands[1]
    assert etp.stp_operand(2) == ds[0].operands[-1]
    assert etp.all_same_segment_shape()
    assert etp.memory_cost((2, 3, 4), (4, 8, 16)) == 96
    assert etp.flop_cost(3) == 3 * etp.flop_cost(1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        scaled = etp * 2
        assert scaled.ds == [d * 2 for d in ds]
        assert (2 * etp).ds == scaled.ds
        assert etp.squeeze_modes().num_operands == 3
        assert etp.consolidate_paths().ds == etp.ds
        assert etp.canonicalize_subscripts().num_operands == 3
        assert etp.flatten_modes("i").num_operands == 3
        assert etp.flatten_coefficient_modes().ds == etp.ds
        assert etp.symmetrize_operands().num_operands == 3
        assert etp.sort_indices_for_identical_operands().num_operands == 3

        backward, order = etp.backward(0)
        assert order == (2, 1, 0)
        assert backward.operands == tuple(operands[i] for i in order)
        backward, order = etp.backward(1)
        assert order == (2, 0, 1, 1)
        assert backward.operands == tuple(operands[i] for i in order)


def test_single_descriptor_permutation_and_stacking():
    scalar = _scalar_rep()
    d = _one_path_stp("i,i,i", (1,), (1,), (1,))
    etp = _make_etp(d, [scalar, scalar, scalar])
    assert etp.d == d

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        permuted = etp.permute_operands((1, 0, 2))
        assert permuted.operands == (scalar, scalar, scalar)
        assert permuted.d == d.permute_operands((1, 0, 2))

        distinct_reps = [cue.O3(0, 1), cue.O3(0, -1), cue.SO3(0)]
        distinct = _make_etp(d, distinct_reps)
        moved = distinct.move_operand_first(-1)
        assert moved.operands == (distinct_reps[2], distinct_reps[1], distinct_reps[0])
        moved = distinct.move_operand_last(0)
        assert moved.operands == (distinct_reps[2], distinct_reps[1], distinct_reps[0])

        stacked = cue.EquivariantTensorProduct.stack(
            [etp, etp], stacked=[True, False, True]
        )
        assert [operand.dim for operand in stacked.operands] == [2, 1, 2]
        assert stacked.d.num_paths == 2
        assert {(path.indices[0], path.indices[-1]) for path in stacked.d.paths} == {
            (0, 0),
            (1, 1),
        }


def test_change_layout_transposes_non_scalar_modes():
    irreps = cue.Irreps("SO3", "2x1")
    rep = cue.IrrepsAndLayout(irreps, cue.ir_mul)
    d = _one_path_stp("ia,ia", (3, 2), (3, 2))
    etp = _make_etp(d, [rep, rep], symmetrize=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mul_ir = etp.change_layout(cue.mul_ir)
        assert mul_ir.d.subscripts == "ai,ai"
        assert all(operand.layout == cue.mul_ir for operand in mul_ir.operands)
        ir_mul = mul_ir.change_layout(cue.ir_mul)
        assert ir_mul.d.subscripts == "ia,ia"
        assert all(operand.layout == cue.ir_mul for operand in ir_mul.operands)


def test_constructor_rejects_incompatible_sizes_and_operand_segments():
    scalar = _scalar_rep()
    with pytest.raises(AssertionError):
        _make_etp(
            _one_path_stp("i,j,ij", (1,), (1,), (1, 1)),
            [scalar, scalar, _scalar_rep(2)],
        )

    with pytest.raises(ValueError, match="Input 0 size mismatch"):
        _make_etp(
            [_one_path_stp("i,i,i", (2,), (2,), (2,))],
            [scalar, scalar, scalar],
        )

    with pytest.raises(ValueError, match="Last input size mismatch"):
        _make_etp(
            [_one_path_stp("i,j,j,i", (1,), (2,), (2,), (1,))],
            [scalar, scalar, scalar],
        )

    with pytest.raises(ValueError, match="Output size mismatch"):
        _make_etp(
            [
                _one_path_stp(
                    "i,j,j,ik+k",
                    (1,),
                    (1,),
                    (1,),
                    (1, 2),
                    coefficients=np.ones(2),
                )
            ],
            [scalar, scalar, scalar],
        )


def test_sequence_constructor_requires_shared_segment_definitions():
    scalar = _scalar_rep()
    two_scalars = _scalar_rep(2)

    one_input_layout = _one_path_stp("i,j,ij", (2,), (1,), (2, 1))
    split_input_layout = cue.SegmentedTensorProduct.from_subscripts("i,j,ij")
    split_input_layout.add_segments(0, [(1,), (1,)])
    split_input_layout.add_segment(1, (1,))
    split_input_layout.add_segments(2, [(1, 1), (1, 1)])
    split_input_layout.add_path(0, 0, 0, c=1.0)
    split_input_layout.add_path(1, 0, 1, c=1.0)
    with pytest.raises(AssertionError):
        _make_etp(
            [one_input_layout, split_input_layout],
            [two_scalars, scalar, two_scalars],
        )

    repeated_once = _one_path_stp("i,j,ij", (1,), (2,), (1, 2))
    repeated_twice = _one_path_stp("i,jk,jk,ijk", (1,), (1, 2), (1, 2), (1, 1, 2))
    with pytest.raises(ValueError, match="Different operands for the last input"):
        _make_etp(
            [repeated_once, repeated_twice],
            [scalar, two_scalars, two_scalars],
        )

    same_inputs_different_outputs = _one_path_stp(
        "ij,j,j,ijk+k",
        (1, 2),
        (2,),
        (2,),
        (1, 2, 1),
        coefficients=np.ones(1),
    )
    repeated_common = _one_path_stp("ij,j,ij", (1, 2), (2,), (1, 2))
    with pytest.raises(AssertionError):
        _make_etp(
            [repeated_common, same_inputs_different_outputs],
            [two_scalars, two_scalars, two_scalars],
        )
