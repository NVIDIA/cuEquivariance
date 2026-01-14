# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import graphviz

    import cuequivariance as cue


def visualize_polynomial(
    poly: "cue.SegmentedPolynomial",
    input_names: list[str],
    output_names: list[str],
) -> "graphviz.Digraph":
    """
    Create a graphviz diagram showing the dataflow from inputs through STPs to outputs.

    Args:
        poly: The SegmentedPolynomial to visualize.
        input_names: Names for each input operand (length must match poly.num_inputs).
        output_names: Names for each output operand (length must match poly.num_outputs).

    Returns:
        A graphviz.Digraph object that can be rendered, saved, or displayed.

    Example:
        >>> import cuequivariance as cue
        >>> from cuequivariance.segmented_polynomials.visualization import visualize_polynomial
        >>> poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [1, 2, 3]).polynomial
        >>> graph = visualize_polynomial(poly, ["x"], ["Y"])
        >>> graph.render("spherical_harmonics", format="png", cleanup=True)  # doctest: +SKIP
        >>> # Or in Jupyter:
        >>> # graph  # Displays inline

    Raises:
        ValueError: If the number of names doesn't match the number of inputs/outputs.
        ImportError: If graphviz is not installed.
    """
    # Validate parameters first
    if len(input_names) != poly.num_inputs:
        raise ValueError(
            f"Expected {poly.num_inputs} input names, got {len(input_names)}"
        )
    if len(output_names) != poly.num_outputs:
        raise ValueError(
            f"Expected {poly.num_outputs} output names, got {len(output_names)}"
        )

    # Import graphviz (checked after parameter validation)
    try:
        import graphviz
    except ImportError as e:
        raise ImportError(
            "graphviz is required for visualization. Install it with: pip install graphviz"
        ) from e

    # Create directed graph
    dot = graphviz.Digraph(comment="Segmented Polynomial Flow")
    dot.attr(rankdir="LR")  # Left to right layout
    dot.attr("node", shape="box")

    # Create input nodes
    for i, (name, operand) in enumerate(zip(input_names, poly.inputs)):
        label = f"{name}\\n{operand.num_segments} segments\\nsize={operand.size}"
        dot.node(f"input_{i}", label, style="filled", fillcolor="lightblue")

    # Create output nodes
    for i, (name, operand) in enumerate(zip(output_names, poly.outputs)):
        label = f"{name}\\n{operand.num_segments} segments\\nsize={operand.size}"
        dot.node(f"output_{i}", label, style="filled", fillcolor="lightgreen")

    # Create STP nodes and edges
    for stp_idx, (operation, stp) in enumerate(poly.operations):
        # Create STP node
        stp_label = f"{stp.subscripts}\\n{stp.num_paths} paths"
        dot.node(f"stp_{stp_idx}", stp_label, style="filled", fillcolor="lightyellow")

        # Create edges from inputs to this STP
        for operand_idx in operation.input_buffers(poly.num_inputs):
            dot.edge(f"input_{operand_idx}", f"stp_{stp_idx}")

        # Create edge from this STP to output
        output_buffer = operation.output_buffer(poly.num_inputs)
        output_idx = output_buffer - poly.num_inputs
        dot.edge(f"stp_{stp_idx}", f"output_{output_idx}")

    return dot
