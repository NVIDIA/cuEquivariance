# SPDX-License-Identifier: Apache-2.0
"""Numerical acceptance and defect rejection for challenge-v1 JAX submissions."""

import pytest
from challenge_checks import check_routing, grader, reference, run_mutation


def test_layout_reference():
    grader.check_layout(reference("jax"), "jax")


def test_graph_reference():
    grader.check_graph(reference("jax"), "jax")


@pytest.mark.parametrize(
    "defect",
    [
        "overwrite_duplicate",
        "reshape_only",
        "merge_parity",
        "shared_radial",
        "detached_vectors",
        "mean_reduction",
        "wrong_layout",
    ],
)
def test_rejects_wrong_solution(defect):
    run_mutation("jax", defect)


def test_nearby_negative_and_broadcast_defect(tmp_path):
    check_routing("jax", tmp_path)
