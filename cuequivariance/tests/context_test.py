# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
import pytest

import cuequivariance as cue
import numpy as np


def test_rep_collection_context():
    explicit = cue.Irreps("O3", "0e + 1o + 2e")
    with cue.assume("O3"):
        implicit = cue.Irreps("0e + 1o + 2e")

    assert explicit == implicit

    with pytest.raises(ValueError):
        cue.Irreps("0e + 1o + 2e")


def test_layout_context():
    explicit = cue.NumpyIrrepsArray(cue.Irreps("O3", "0e + 1o"), np.ones(4), cue.ir_mul)

    with cue.assume(cue.ir_mul):
        implicit = cue.NumpyIrrepsArray(cue.Irreps("O3", "0e + 1o"), np.ones(4))

    assert explicit == implicit

    with pytest.raises(ValueError):
        cue.NumpyIrrepsArray(cue.Irreps("O3", "0e + 1o"), np.ones(4))


def test_rep_collection_and_layout_context():
    explicit = cue.NumpyIrrepsArray(cue.Irreps("O3", "0e + 1o"), np.ones(4), cue.ir_mul)

    with cue.assume("O3", cue.ir_mul):
        implicit = cue.NumpyIrrepsArray("0e + 1o", np.ones(4))

    assert explicit == implicit


def test_decorator():
    @cue.assume("O3", cue.ir_mul)
    def func():
        assert cue.get_irrep_scope() == cue.O3
        assert cue.get_layout_scope() == cue.ir_mul

    assert cue.get_irrep_scope(False) is None
    assert cue.get_layout_scope(False) is None
    func()
    assert cue.get_irrep_scope(False) is None
    assert cue.get_layout_scope(False) is None
