#!/usr/bin/env python3
"""Demo script showing the visualize_polynomial function output."""

import cuequivariance as cue
from cuequivariance.segmented_polynomials import visualize_polynomial

# Example 1: Simple spherical harmonics
print("=" * 70)
print("Example 1: Spherical Harmonics (degree 1, 2, 3)")
print("=" * 70)
sh_poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [1, 2, 3]).polynomial
print(sh_poly)
print()

graph = visualize_polynomial(sh_poly, ["x"], ["Y"])
print("Generated Graphviz DOT source:")
print("-" * 70)
print(graph.source)
print("-" * 70)
print()

# Example 2: Linear layer with multiple operations
print("=" * 70)
print("Example 2: Linear Layer")
print("=" * 70)
irreps_in = cue.Irreps("O3", "8x0e + 8x1o")
irreps_out = cue.Irreps("O3", "4x0e + 4x1o")
linear_poly = cue.descriptors.linear(irreps_in, irreps_out).polynomial
print(linear_poly)
print()

graph = visualize_polynomial(linear_poly, ["weights", "input"], ["output"])
print("Generated Graphviz DOT source:")
print("-" * 70)
print(graph.source)
print("-" * 70)
print()

# Example 3: Tensor product with two inputs
print("=" * 70)
print("Example 3: Channel-wise Tensor Product")
print("=" * 70)
irreps = cue.Irreps("O3", "0e + 1o")
tp_poly = cue.descriptors.channelwise_tensor_product(irreps, irreps, irreps).polynomial
print(tp_poly)
print()

graph = visualize_polynomial(tp_poly, ["weights", "x1", "x2"], ["y"])
print("Generated Graphviz DOT source:")
print("-" * 70)
print(graph.source)
print("-" * 70)
print()

print("\n" + "=" * 70)
print("To save diagrams to files, use:")
print("  graph.render('output_filename', format='png', cleanup=True)")
print("To view directly, use:")
print("  graph.view()")
print("=" * 70)
