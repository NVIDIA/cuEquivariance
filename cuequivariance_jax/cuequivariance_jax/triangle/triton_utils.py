# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Minimal Triton utilities for JAX primitives.

This module provides a lightweight helper for integrating Triton kernels into JAX.

This implementation was inspired by NVIDIA TransformerEngine
(https://github.com/NVIDIA/TransformerEngine).
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import zlib
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.extend.core as jex_core
import jax.numpy as jnp
from jax import core
from jax._src.lib import gpu_triton
from jax.interpreters import mlir
from packaging import version
from triton.backends.nvidia import compiler as cb
from triton.compiler import compiler as tc
from triton.compiler.errors import CompilationError
from triton.runtime import cache as triton_cache

try:
    import triton

    TRITON_VERSION = version.parse(triton.__version__)
except (ImportError, AttributeError):
    TRITON_VERSION = None

# Configure Triton cache directory
try:
    cache_dir = (
        triton_cache.knobs.cache.dir
        if hasattr(triton_cache, "knobs")
        else triton_cache.default_cache_dir()
    )
    if not cache_dir:
        cache_dir = os.path.join(os.path.expanduser("~"), ".triton", "cache")
        if hasattr(triton_cache, "knobs"):
            triton_cache.knobs.cache.dir = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
except Exception:
    cache_dir = os.path.join(os.path.expanduser("~"), ".triton", "cache")
    os.makedirs(cache_dir, exist_ok=True)

__all__ = ["triton_call_lowering", "triton_call"]

_TRITON_KERNEL_CACHE = {}

_DTYPE_MAP = {
    jnp.dtype("bfloat16"): "bf16",
    jnp.dtype("float64"): "fp64",
    jnp.dtype("float32"): "fp32",
    jnp.dtype("float16"): "fp16",
    jnp.dtype("int64"): "i64",
    jnp.dtype("int32"): "i32",
    jnp.dtype("int16"): "i16",
    jnp.dtype("int8"): "i8",
    jnp.dtype("uint64"): "u64",
    jnp.dtype("uint32"): "u32",
    jnp.dtype("uint16"): "u16",
    jnp.dtype("uint8"): "u8",
    jnp.dtype("bool"): "i1",
}


_PTXAS_VERSION_CACHE = None


def _get_max_ptx_version():
    """Detects the maximum PTX version supported by the available ptxas."""
    global _PTXAS_VERSION_CACHE
    if _PTXAS_VERSION_CACHE is not None:
        return _PTXAS_VERSION_CACHE

    try:
        # Check ptxas version
        result = subprocess.run(
            ["ptxas", "--version"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            # Example output: "Cuda compilation tools, release 12.6, V12.6.68"
            match = re.search(r"release (\d+)\.(\d+)", result.stdout)
            if match:
                major, minor = int(match.group(1)), int(match.group(2))
                # Map CUDA version to PTX version
                if major == 12:
                    if minor >= 8:
                        version = 87
                    elif minor >= 5:
                        version = 85
                    else:
                        version = 80 + minor
                elif major == 11:
                    if minor >= 8:
                        version = 78
                    else:
                        version = 70 + minor
                else:
                    version = None
                _PTXAS_VERSION_CACHE = version
    except Exception:
        pass

    return _PTXAS_VERSION_CACHE


def _get_triton_dtype(aval: core.ShapedArray) -> str:
    """Convert JAX dtype to Triton type string."""
    return f"*{_DTYPE_MAP[aval.dtype]}"


def _compile_triton(
    kernel_fn: Callable,
    signature: Mapping[str, str],
    constants: Mapping[str, Any],
    num_warps: int,
    num_stages: int,
    compute_capability: int,
):
    """Compile a Triton kernel to PTX with caching."""
    # Include source code in cache key to handle edits
    cache_key = hashlib.md5(
        str(
            (
                kernel_fn.__name__,
                getattr(kernel_fn, "src", ""),
                tuple(sorted(signature.items())),
                tuple(sorted(constants.items())),
                num_warps,
                num_stages,
                compute_capability,
            )
        ).encode()
    ).hexdigest()

    if cache_key in _TRITON_KERNEL_CACHE:
        return _TRITON_KERNEL_CACHE[cache_key]

    # Detect maximum supported PTX version
    max_ptx_version = _get_max_ptx_version()

    cuda_options_kwargs = {
        "num_warps": num_warps,
        "num_stages": num_stages,
        "num_ctas": 1,
        "cluster_dims": (1, 1, 1),
        "debug": False,
        "enable_fp_fusion": False,
    }

    # Try adding ptx_version if detected and supported by Triton
    if max_ptx_version is not None:
        try:
            # Check if CUDAOptions accepts ptx_version
            cb.CUDAOptions(**cuda_options_kwargs, ptx_version=max_ptx_version)
            cuda_options_kwargs["ptx_version"] = max_ptx_version
        except TypeError:
            # Argument not supported by this Triton version
            pass

    options = cb.CUDAOptions(**cuda_options_kwargs)

    # Handle different Triton API versions
    compiled = None

    # Triton 3.3.x is known to be incompatible due to constexpr handling bugs
    if TRITON_VERSION is not None and (
        TRITON_VERSION.major == 3 and TRITON_VERSION.minor == 3
    ):
        raise ImportError(
            f"Triton version {TRITON_VERSION} is not supported due to known issues. "
            "Please upgrade to Triton 3.4+ or downgrade to Triton 3.2.x."
        )

    # Helper to try compilation with specific arguments
    def try_compile(signature_dict, **kwargs):
        try:
            src = tc.ASTSource(fn=kernel_fn, signature=signature_dict, **kwargs)
            return tc.compile(
                src,
                target=tc.GPUTarget("cuda", compute_capability, 32),
                options=options.__dict__,
            )
        except (TypeError, AttributeError, CompilationError):
            return None

    # 1. Try Triton 3.4.0+ API: constexprs should not be in signature
    compiled = try_compile(signature, constexprs=constants)

    # 2. Try Triton 3.1.0-3.2.0: constexprs should be in signature as "constexpr"
    if compiled is None:
        signature_with_constexpr = {**signature, **{k: "constexpr" for k in constants}}
        compiled = try_compile(signature_with_constexpr, constexprs=constants)

    # 3. Try Triton 3.0.0: uses 'constants' instead of 'constexprs'
    if compiled is None:
        # Re-create signature_with_constexpr just to be safe/clear
        signature_with_constexpr = {**signature, **{k: "constexpr" for k in constants}}
        compiled = try_compile(signature_with_constexpr, constants=constants)

    if compiled is None:
        raise RuntimeError("Failed to compile Triton kernel with any API version")

    args = (
        (
            compiled.name,
            num_warps,
            1,
            compiled.metadata.shared,
            compiled.asm["ptx"],
            "",
            compute_capability,
        )
        if version.parse(jax.__version__) >= version.parse("0.8.2")
        else (
            compiled.name,
            num_warps,
            compiled.metadata.shared,
            compiled.asm["ptx"],
            "",
            compute_capability,
            1,
            1,
            1,
        )
    )
    kernel = gpu_triton.TritonKernel(*args)

    _TRITON_KERNEL_CACHE[cache_key] = kernel
    return kernel


def triton_call_lowering(
    ctx,
    kernel_fn: Callable,
    *array_args,
    grid,
    num_warps: int = 4,
    num_stages: int = 3,
    input_output_aliases: Mapping[int, int] | None = None,
    constexprs: Mapping[str, Any] | None = None,
):
    """Helper for MLIR lowering that calls a Triton kernel."""
    compute_capability = gpu_triton.get_compute_capability(0)

    all_avals = list(ctx.avals_in) + list(ctx.avals_out)
    constexprs = constexprs or {}
    tensor_arg_names = [n for n in kernel_fn.arg_names if n not in constexprs]

    signature = {n: _get_triton_dtype(a) for n, a in zip(tensor_arg_names, all_avals)}

    # Normalize grid to tuple of 3 ints
    g = tuple(grid) if isinstance(grid, (tuple, list)) else (grid,)
    grid_tuple = g[:3] + (1,) * (3 - len(g[:3]))

    kernel = _compile_triton(
        kernel_fn,
        signature,
        constexprs,
        num_warps,
        num_stages,
        compute_capability,
    )

    kernel_params = [gpu_triton.create_array_parameter(0, 16) for _ in all_avals]

    # WARNING: Must use explicit indexing (grid_tuple[0], grid_tuple[1], grid_tuple[2])
    # instead of unpacking (*grid_tuple). Unpacking causes UnexpectedTracerError.
    call_proto = gpu_triton.TritonKernelCall(
        kernel, grid_tuple[0], grid_tuple[1], grid_tuple[2], kernel_params
    ).to_proto(kernel_fn.__name__, b"")

    return jax.ffi.ffi_lowering(
        "triton_kernel_call",
        api_version=2,
        backend_config=zlib.compress(call_proto),
        operand_output_aliases=input_output_aliases or {},
    )(ctx, *array_args)


# Define global Triton kernel call primitive
_triton_kernel_call_p = jex_core.Primitive("triton_kernel_call")
_triton_kernel_call_p.multiple_results = True


def _triton_abstract_eval(*avals, out_shape, **unused_kwargs):
    """Abstract evaluation for Triton kernel call."""
    return tuple(core.ShapedArray(s.shape, s.dtype) for s in out_shape)


def _triton_lowering_rule(
    ctx, *mlir_args, kernel, grid, num_warps, num_stages, constexprs, out_shape
):
    """Lowering rule for Triton kernel call."""
    return triton_call_lowering(
        ctx,
        kernel,
        *mlir_args,
        grid=grid,
        num_warps=num_warps,
        num_stages=num_stages,
        constexprs=dict(constexprs),
    )


# Register primitive
_triton_kernel_call_p.def_abstract_eval(_triton_abstract_eval)
mlir.register_lowering(_triton_kernel_call_p, _triton_lowering_rule, platform="gpu")


def triton_call(
    *args,
    kernel: Callable,
    out_shape: Sequence[jax.ShapeDtypeStruct],
    grid,
    num_warps: int = 4,
    num_stages: int = 3,
    **kwargs,
):
    """High-level API to call a Triton kernel from JAX."""
    # WARNING: Do NOT inline these variables into the bind() call below.
    # Inlining causes UnexpectedTracerError because JAX tracers leak into
    # primitive parameters. These intermediate variables must be created
    # separately to avoid tracer leaks during JAX transformations.
    constexprs_tuple = tuple(sorted(kwargs.items()))
    out_shape_tuple = tuple(out_shape)
    grid_tuple = tuple(grid) if isinstance(grid, (list, tuple)) else (grid,)

    result = _triton_kernel_call_p.bind(
        *args,
        kernel=kernel,
        grid=grid_tuple,
        num_warps=num_warps,
        num_stages=num_stages,
        constexprs=constexprs_tuple,
        out_shape=out_shape_tuple,
    )

    num_outputs = len(out_shape)
    if num_outputs == 1:
        return result[0] if isinstance(result, tuple) else result
    return result
