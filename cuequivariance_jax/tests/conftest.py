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
import jax
import pytest


@pytest.fixture(autouse=True)
def check_jax_memory():
    yield
    try:
        gpu = jax.local_devices(backend="gpu")[0]
        usage_gib = gpu.memory_stats()["peak_bytes_in_use"] / (1024**3)
        limit = 2.0
        assert usage_gib <= limit, (
            f"JAX peak memory usage {usage_gib:.2f}GiB exceeds {limit}GiB limit!"
        )
    except (IndexError, KeyError, RuntimeError):
        pass  # No GPU available
