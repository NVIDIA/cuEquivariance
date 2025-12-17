# SegmentedPolynomial Visualization

This module provides a function to visualize `SegmentedPolynomial` objects as flow diagrams using Graphviz.

## Installation

First, install the required dependency:

```bash
pip install graphviz
```

Note: You may also need to install the Graphviz system package. See https://graphviz.org/download/

## Usage

```python
import cuequivariance as cue
from cuequivariance.segmented_polynomials import visualize_polynomial

# Create a polynomial (e.g., spherical harmonics)
poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [1, 2, 3]).polynomial

# Generate the visualization
graph = visualize_polynomial(
    poly,
    input_names=["x"],      # One name per input
    output_names=["Y"]      # One name per output
)

# Save to file
graph.render("my_diagram", format="png", cleanup=True)

# Or view directly (opens in default viewer)
graph.view()

# Or in Jupyter notebooks, just display it:
# graph  # This will render inline
```

## Features

The visualization shows:

- **Input nodes** (blue): Display the input name, number of segments, and total size
- **STP nodes** (yellow): Display the subscripts and number of computation paths
- **Output nodes** (green): Display the output name, number of segments, and total size
- **Edges**: Show the dataflow from inputs through STPs to outputs
  - Multiple edges are drawn when an input is used multiple times in an STP

## Examples

### Example 1: Spherical Harmonics
```python
poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [1, 2]).polynomial
graph = visualize_polynomial(poly, ["position"], ["harmonics"])
```
Shows how a position vector flows through multiple degree polynomials.

### Example 2: Linear Layer
```python
irreps_in = cue.Irreps("O3", "32x0e + 32x1o")
irreps_out = cue.Irreps("O3", "16x0e + 48x1o")
poly = cue.descriptors.linear(irreps_in, irreps_out).polynomial
graph = visualize_polynomial(poly, ["weights", "input"], ["output"])
```
Shows how weights and input combine to produce output.

### Example 3: Tensor Product
```python
irreps = cue.Irreps("O3", "0e + 1o + 2e")
poly = cue.descriptors.channelwise_tensor_product(irreps, irreps, irreps).polynomial
graph = visualize_polynomial(poly, ["weights", "x1", "x2"], ["output"])
```
Shows how two inputs are combined via a tensor product.

## Customization

The returned `graphviz.Digraph` object can be further customized:

```python
graph = visualize_polynomial(poly, input_names, output_names)

# Change graph attributes
graph.attr(rankdir="TB")  # Top to bottom instead of left to right
graph.attr(bgcolor="white")

# Add custom styling
graph.node_attr.update(fontname="Arial", fontsize="12")
graph.edge_attr.update(color="gray")

# Render with custom options
graph.render("output", format="svg", cleanup=True)
```

## Documentation

For complete examples with rendered output, see the [Segmented Polynomials tutorial](docs/tutorials/poly.rst) in the documentation.

For API testing without requiring Graphviz installed, see `test_visualization.py`.
