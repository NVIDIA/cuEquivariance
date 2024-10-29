# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import pytest

import cuequivariance.segmented_tensor_product as stp


def test_subscripts():
    with pytest.raises(ValueError):
        stp.Subscripts("#$%@")

    with pytest.raises(ValueError):
        stp.Subscripts("Zu")  # uppercase not supported anymore

    with pytest.raises(ValueError):
        stp.Subscripts("uZ")  # uppercase after lowercase

    with pytest.raises(ValueError):
        stp.Subscripts("uZ+ij+kl")  # multiple + signs

    subscripts = stp.Subscripts("ui,vj,uvk+ijk")
    assert subscripts.canonicalize() == "ui,vj,uvk+ijk"

    assert subscripts.coefficients == "ijk"

    assert subscripts.num_operands == 3
    assert subscripts.operands[0] == "ui"
    assert subscripts.operands[1] == "vj"
    assert subscripts.operands[2] == "uvk"

    assert subscripts.is_subset_of("uwi,vwj,uvk+ijk")  # using w=1
    assert subscripts.is_equivalent("ui_vj_uvk+ijk")


def test_canonicalize():
    assert stp.Subscripts("ui").canonicalize() == "uv"
    assert stp.Subscripts("ab,ad+bd").canonicalize() == "ui,uj+ij"
