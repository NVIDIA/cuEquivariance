#!/usr/bin/env python3
"""Example script demonstrating the visualize_polynomial function."""

import cuequivariance as cue
from cuequivariance.segmented_polynomials import visualize_polynomial

# Example 1: Spherical harmonics
print("Creating spherical harmonics polynomial...")
sh_poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [1, 2, 3]).polynomial
print(sh_poly)
print()

graph = visualize_polynomial(sh_poly, ["x"], ["Y"])
print("Saving spherical_harmonics.png...")
graph.render("spherical_harmonics", format="png", cleanup=True)
print("✓ Saved spherical_harmonics.png")
print()

# Example 2: Linear layer
print("Creating linear layer polynomial...")
irreps_in = cue.Irreps("O3", "32x0e + 32x1o")
irreps_out = cue.Irreps("O3", "16x0e + 48x1o")
linear_poly = cue.descriptors.linear(irreps_in, irreps_out).polynomial
print(linear_poly)
print()

graph = visualize_polynomial(linear_poly, ["weights", "input"], ["output"])
print("Saving linear_layer.png...")
graph.render("linear_layer", format="png", cleanup=True)
print("✓ Saved linear_layer.png")
print()

# Example 3: Tensor product
print("Creating tensor product polynomial...")
irreps = cue.Irreps("O3", "0e + 1o + 2e")
tp_poly = cue.descriptors.channelwise_tensor_product(irreps, irreps, irreps).polynomial
print(tp_poly)
print()

graph = visualize_polynomial(tp_poly, ["x1", "x2"], ["y"])
print("Saving tensor_product.png...")
graph.render("tensor_product", format="png", cleanup=True)
print("✓ Saved tensor_product.png")
print()

print("All visualizations generated successfully!")
print("Note: You can also use graph.view() to open the image directly,")
print("or in Jupyter notebooks, just return the graph object to display it inline.")
