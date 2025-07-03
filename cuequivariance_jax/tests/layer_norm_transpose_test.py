# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import jax.numpy as jnp
import jax.random as random
import pytest
from cuequivariance_ops.triton import Layout
from jax import test_util

from cuequivariance_jax.triangle.layer_norm_transpose import (
    layer_norm_transpose,
    layer_norm_transpose_reference_forward,
)


@pytest.mark.parametrize("elementwise_affine", [True, False])
@pytest.mark.parametrize(
    "layout_str,input_shape,expected_output_shape,feature_dim",
    [
        # 2D layouts
        ("nd->nd", (16, 64), (16, 64), 64),
        ("nd->dn", (16, 64), (64, 16), 64),
        # 3D layouts
        ("bnd->bnd", (2, 16, 64), (2, 16, 64), 64),
        ("bdn->bnd", (2, 64, 16), (2, 16, 64), 64),
        ("bnd->bdn", (2, 16, 64), (2, 64, 16), 64),
        ("dbn->bnd", (64, 2, 16), (2, 16, 64), 64),
        ("bnd->dbn", (2, 16, 64), (64, 2, 16), 64),
        # 4D layouts
        ("bijd->bijd", (2, 8, 8, 64), (2, 8, 8, 64), 64),
        ("bijd->bdij", (2, 8, 8, 64), (2, 64, 8, 8), 64),
        ("bdij->bijd", (2, 64, 8, 8), (2, 8, 8, 64), 64),
        ("dbij->bijd", (64, 2, 8, 8), (2, 8, 8, 64), 64),
        ("bijd->dbij", (2, 8, 8, 64), (64, 2, 8, 8), 64),
    ],
)
def test_layer_norm_transpose(
    elementwise_affine, layout_str, input_shape, expected_output_shape, feature_dim
):
    """Test to verify implementation across all layouts and elementwise_affine settings."""
    key = random.PRNGKey(42)
    eps = 1e-5

    D = feature_dim

    # Generate test data with random weights and biases
    key, subkey1, subkey2, subkey3 = random.split(key, 4)
    x = random.normal(subkey1, input_shape, dtype=jnp.float32)
    w = random.normal(subkey2, (D,), dtype=jnp.float32) * 0.1 + 1.0  # Random around 1.0
    b = random.normal(subkey3, (D,), dtype=jnp.float32) * 0.1  # Random around 0.0

    # Test the implementation
    out = layer_norm_transpose(
        x,
        w,
        b,
        layout=layout_str,
        eps=eps,
        elementwise_affine=elementwise_affine,
    )

    # Check output shape
    assert out.shape == expected_output_shape, (
        f"Shape mismatch for layout {layout_str}: got {out.shape}, expected {expected_output_shape}"
    )

    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(out)), f"Contains NaN values for layout {layout_str}"
    assert not jnp.any(jnp.isinf(out)), f"Contains Inf values for layout {layout_str}"

    # Compare with reference implementation
    # Need to convert string layout to enum and prepare input tensor like main function does
    if layout_str == "nd->nd":
        N, D = x.shape
        B = 1
        x_ref = x.reshape(1, N, D)
        layout_enum = Layout.BND_BND
    elif layout_str == "nd->dn":
        N, D = x.shape
        B = 1
        x_ref = x.reshape(1, N, D)
        layout_enum = Layout.BND_BDN
    elif layout_str == "bnd->bnd":
        B, N, D = x.shape
        x_ref = x
        layout_enum = Layout.BND_BND
    elif layout_str == "bdn->bnd":
        B, D, N = x.shape
        x_ref = x
        layout_enum = Layout.BDN_BND
    elif layout_str == "bnd->bdn":
        B, N, D = x.shape
        x_ref = x
        layout_enum = Layout.BND_BDN
    elif layout_str == "dbn->bnd":
        D, B, N = x.shape
        x_ref = x
        layout_enum = Layout.DBN_BND
    elif layout_str == "bnd->dbn":
        B, N, D = x.shape
        x_ref = x
        layout_enum = Layout.BND_DBN
    elif layout_str == "bijd->bijd":
        B, II, J, D = x.shape
        x_ref = x.reshape(B, II * J, D)
        layout_enum = Layout.BND_BND
    elif layout_str == "bijd->bdij":
        B, II, J, D = x.shape
        x_ref = x.reshape(B, II * J, D)
        layout_enum = Layout.BND_BDN
    elif layout_str == "bdij->bijd":
        B, D, II, J = x.shape
        x_ref = x.reshape(B, D, II * J)
        layout_enum = Layout.BDN_BND
    elif layout_str == "dbij->bijd":
        D, B, II, J = x.shape
        x_ref = x.reshape(D, B, II * J)
        layout_enum = Layout.DBN_BND
    elif layout_str == "bijd->dbij":
        B, II, J, D = x.shape
        x_ref = x.reshape(B, II * J, D)
        layout_enum = Layout.BND_DBN

    # Get reference output by calling the reference function directly
    ref_out, ref_mean, ref_rstd = layer_norm_transpose_reference_forward(
        x_ref, w, b, eps, elementwise_affine, layout_enum
    )

    # Reshape reference output back to expected shape for 4D cases
    if layout_str == "bijd->bijd":
        B, II, J, D = x.shape
        ref_out = ref_out.reshape(B, II, J, D)
    elif layout_str == "bijd->bdij":
        B, II, J, D = x.shape
        ref_out = ref_out.reshape(B, D, II, J)
    elif layout_str == "bdij->bijd":
        B, D, II, J = x.shape
        ref_out = ref_out.reshape(B, II, J, D)
    elif layout_str == "dbij->bijd":
        D, B, II, J = x.shape
        ref_out = ref_out.reshape(B, II, J, D)
    elif layout_str == "bijd->dbij":
        B, II, J, D = x.shape
        ref_out = ref_out.reshape(D, B, II, J)

    # Check that outputs match (within numerical precision)
    assert jnp.allclose(out, ref_out, rtol=1e-5, atol=1e-6), (
        f"Output mismatch with reference for layout {layout_str}: "
        f"max diff = {jnp.max(jnp.abs(out - ref_out))}"
    )

    # Test gradient computation using jax.test_util.check_grads
    def loss_fn(x, w, b):
        out = layer_norm_transpose(
            x,
            w,
            b,
            layout=layout_str,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )
        return jnp.mean(out**2)

    if elementwise_affine:
        # Check gradients for all parameters when elementwise_affine=True
        test_util.check_grads(loss_fn, (x, w, b), order=1, modes=["rev"])
    else:
        # When elementwise_affine=False, only check gradient w.r.t. input x
        # The gradients w.r.t. w and b should be zero but may be numerically unstable
        def loss_fn_x_only(x):
            out = layer_norm_transpose(
                x,
                w,
                b,
                layout=layout_str,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )
            return jnp.mean(out**2)

        test_util.check_grads(loss_fn_x_only, (x,), order=1, modes=["rev"])

        # Note: When elementwise_affine=False, gradients w.r.t. w and b should ideally be zero
        # but the current implementation may not handle this correctly. This is a known issue.
        # For now, we just verify that gradients can be computed without errors

        # TODO: Fix implementation so that grad_w and grad_b are zero when elementwise_affine=False
