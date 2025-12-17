# cuEquivariance

cuEquivariance is an NVIDIA Python library designed to facilitate the construction of high-performance geometric neural networks using segmented polynomials and triangular operations. cuEquivariance provides a comprehensive API for describing segmented polynomials made out of segmented tensor products and optimized CUDA kernels for their execution. Additionally, cuEquivariance offers bindings for both PyTorch and JAX, ensuring broad compatibility and ease of integration.

Equivariance is the mathematical formalization of the concept of "respecting symmetries." Robust physical models exhibit equivariance with respect to rotations and translations in three-dimensional space. Artificial intelligence models that incorporate equivariance are often more data-efficient.

## Documentation

Please refer to the project documentation for more information [https://docs.nvidia.com/cuda/cuequivariance/](https://docs.nvidia.com/cuda/cuequivariance/).

## Installation

```bash
# Choose the frontend you want to use
pip install cuequivariance-jax
pip install cuequivariance-torch
pip install cuequivariance  # Installs only the core non-ML components

# CUDA kernels
pip install cuequivariance-ops-jax-cu12   # or -cu13
pip install cuequivariance-ops-torch-cu12 # or -cu13
```

## License

All files hosted in this repository are subject to the Apache 2.0 license.

## Disclaimer

cuEquivariance is in a Beta state. Beta products may not be fully functional, may contain errors or design flaws, and may be changed at any time without notice. We appreciate your feedback to improve and iterate on our Beta products.

