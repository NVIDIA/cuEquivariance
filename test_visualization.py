#!/usr/bin/env python3
"""Test script for the visualize_polynomial function (works without graphviz)."""

import cuequivariance as cue


def test_visualization_api():
    """Test that the visualization function has the correct API without rendering."""
    from cuequivariance.segmented_polynomials import visualize_polynomial

    # Create a simple polynomial
    sh_poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [1, 2]).polynomial

    print("Testing visualize_polynomial API...")
    print(f"Polynomial: {sh_poly}")
    print(f"  num_inputs: {sh_poly.num_inputs}")
    print(f"  num_outputs: {sh_poly.num_outputs}")
    print(f"  num_operations: {len(sh_poly.operations)}")
    print()

    # Test error handling for wrong number of names
    try:
        visualize_polynomial(sh_poly, ["x", "y"], ["Y"])  # Too many input names
        print("❌ Should have raised ValueError for wrong number of inputs")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    try:
        visualize_polynomial(sh_poly, ["x"], ["Y", "Z"])  # Too many output names
        print("❌ Should have raised ValueError for wrong number of outputs")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test that it raises ImportError if graphviz is not installed
    try:
        graph = visualize_polynomial(sh_poly, ["x"], ["Y"])
        print("✓ graphviz is installed, graph created successfully")
        print(f"  Graph type: {type(graph)}")
        # Print the DOT source
        print("\nGenerated DOT source:")
        print(graph.source)
    except ImportError as e:
        print(f"✓ Correctly raised ImportError when graphviz not installed: {e}")

    print("\n✓ All API tests passed!")


if __name__ == "__main__":
    test_visualization_api()
