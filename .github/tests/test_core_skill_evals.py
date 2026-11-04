# SPDX-License-Identifier: Apache-2.0
"""Numerical contracts behind the core skill evals; not agent benchmark scores."""

import dataclasses
import re

import numpy as np
import pytest

import cuequivariance as cue


@dataclasses.dataclass(frozen=True)
class Z3(cue.Irrep):
    k: int

    def __post_init__(self):
        if type(self.k) is not int or self.k not in (0, 1, 2):
            raise ValueError("Z3 labels are integers 0, 1, 2")

    @classmethod
    def regexp_pattern(cls):
        return re.compile(r"([012])")

    @classmethod
    def from_string(cls, string):
        if string not in ("0", "1", "2"):
            raise ValueError(string)
        return cls(int(string))

    def __repr__(self):
        return str(self.k)

    def __lt__(self, other):
        return self.k < other.k

    def __mul__(self, other):
        return [Z3((self.k + other.k) % 3)]

    @classmethod
    def iterator(cls):
        return iter([cls(0), cls(1), cls(2)])

    @classmethod
    def clebsch_gordan(cls, a, b, c):
        return np.ones((1, 1, 1, 1)) if c in a * b else np.zeros((0, 1, 1, 1))

    @property
    def dim(self):
        return 1

    def discrete_generators(self):
        return np.array([[[np.exp(2j * np.pi * self.k / 3)]]])

    def continuous_generators(self):
        return np.zeros((0, 1, 1))

    def algebra(self):
        return np.zeros((0, 0, 0))


def test_z3_group_and_cg_covariance():
    irreps = list(Z3.iterator())
    assert irreps[0] == Z3(0)
    assert len(set(irreps)) == 3
    for invalid in (-1, 3, 1.5, True):
        with pytest.raises(ValueError):
            Z3(invalid)
    for invalid in ("-1", "3", "01", "bad"):
        with pytest.raises(ValueError):
            Z3.from_string(invalid)
    for a in irreps:
        assert Z3.from_string(str(a)) == a
        np.testing.assert_allclose(a.discrete_generators() ** 3, 1, atol=1e-14)
        assert a.continuous_generators().shape == (0, 1, 1)
        assert a.algebra().shape == (0, 0, 0)
        for b in irreps:
            assert a * b == [Z3((a.k + b.k) % 3)]
            for c in irreps:
                cg = cue.clebsch_gordan(a, b, c)
                allowed = (a.k + b.k) % 3 == c.k
                assert cg.shape == (int(allowed), 1, 1, 1)
                if allowed:
                    np.testing.assert_allclose(
                        a.discrete_generators() * b.discrete_generators() * cg,
                        cg * c.discrete_generators(),
                        atol=1e-14,
                    )


def test_full_tensor_product_dimension_and_equivariance():
    e = cue.descriptors.full_tensor_product(
        cue.Irreps("SO3", "2x0 + 1 + 2"), cue.Irreps("SO3", "0 + 1")
    )
    assert len(e.inputs) == 2
    assert e.outputs[0].irreps.regroup() == cue.Irreps("SO3", "3x0+5x1+3x2+3")
    assert e.outputs[0].dim == 40
    rng = np.random.default_rng(12)
    x = [rng.normal(size=(4, rep.dim)) for rep in e.inputs]
    rotation = np.array([0.31, -0.19, 0.43])
    matrices = [rep.exp_map(rotation, []) for rep in (*e.inputs, *e.outputs)]
    np.testing.assert_allclose(
        e(*(v @ d.T for v, d in zip(x, matrices)))[0],
        e(*x)[0] @ matrices[-1].T,
        atol=1e-12,
    )


def test_ir_dict_operand_contract_and_dense_reference():
    a = cue.Irreps("SO3", "2x0+2x1+2x2")
    b = cue.Irreps("SO3", "0+1+2+3")
    keep = [cue.SO3(degree) for degree in range(3)]
    split = cue.descriptors.channelwise_tensor_product_ir_dict(a, b, keep)
    # The ir_dict descriptor simplifies outputs automatically. Request the
    # same representation ordering in the dense reference as well.
    dense = cue.descriptors.channelwise_tensor_product(
        a, b, keep, simplify_irreps3=True
    )
    rng = np.random.default_rng(14)
    blocks = [
        [rng.normal(size=(3, mul, ir.dim)) for mul, ir in group]
        for group in split.input_irreps
    ]
    operands = [[v.swapaxes(-1, -2).reshape(3, -1) for v in group] for group in blocks]
    actual = split.polynomial(*(v for group in operands for v in group))
    expected = dense(*(np.concatenate(group, axis=-1) for group in operands))[0]
    assert split.output_irreps[0] == dense.outputs[0].irreps
    np.testing.assert_allclose(np.concatenate(actual, axis=-1), expected, atol=1e-12)


def test_layout_conversion_and_linear_equivariance():
    irreps = cue.Irreps("SO3", "2x0+3x1")
    x = np.arange(2 * irreps.dim, dtype=float).reshape(2, -1) / 7

    def convert(data, source_layout):
        pieces, offset = [], 0
        for mul, ir in irreps:
            shape = (mul, ir.dim) if source_layout == cue.mul_ir else (ir.dim, mul)
            block = data[..., offset : offset + mul * ir.dim]
            pieces.append(
                block.reshape(*data.shape[:-1], *shape)
                .swapaxes(-1, -2)
                .reshape(*data.shape[:-1], -1)
            )
            offset += mul * ir.dim
        return np.concatenate(pieces, axis=-1)

    flat = convert(x, cue.mul_ir)
    np.testing.assert_array_equal(convert(flat, cue.ir_mul), x)
    assert not np.array_equal(flat, x)
    e = cue.descriptors.linear(irreps, irreps)
    w = np.random.default_rng(9).normal(size=e.inputs[0].dim)
    rotation = np.array([0.2, 0.3, -0.4])
    d = e.inputs[1].exp_map(rotation, [])
    np.testing.assert_allclose(e(w, flat @ d.T)[0], e(w, flat)[0] @ d.T, atol=1e-12)
