# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0


def test_import():
    import cuequivariance

    assert cuequivariance.__version__ is not None
    assert cuequivariance.__version__ != "0.0.0"
