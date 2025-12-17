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
import os
from typing import Sequence

import pytest
import torch
import torch._dynamo

torch._dynamo.config.cache_size_limit = 100

try:
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401
    import onnxscript  # noqa: F401

    from cuequivariance_torch import (
        onnx_custom_translation_table,
        register_tensorrt_plugins,
    )

    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False


try:
    import torch_tensorrt

    TORCH_TRT_AVAILABLE = True
except Exception:
    TORCH_TRT_AVAILABLE = False


def verify_onnx(module, onnx_module, inputs, dtype):
    if dtype != torch.float32:
        pytest.skip("onnxrt only checked for float32")
    from onnxruntime import SessionOptions
    from onnxruntime_extensions import get_library_path
    from torch.onnx.verification import (
        VerificationOptions,
        _compare_onnx_pytorch_model,
    )

    original_init = SessionOptions.__init__

    def new_init(self):
        original_init(self)
        try:
            self.register_custom_ops_library(get_library_path())
        except Exception:
            pass

    SessionOptions.__init__ = new_init
    _compare_onnx_pytorch_model(
        module, onnx_module, tuple(inputs), None, None, VerificationOptions()
    )
    SessionOptions.__init__ = original_init
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def verify_trt(module, onnx_module, inputs, dtype):
    import tensorrt
    from pkg_resources import parse_version

    if parse_version(tensorrt.__version__) < parse_version("10.3.0"):
        pytest.skip("TRT < 10.3.0 is not supported!")
    if dtype == torch.float64:
        pytest.skip("TRT does not support float64")

    from onnxruntime import InferenceSession, SessionOptions
    from onnxruntime_extensions import get_library_path
    from polygraphy.backend.onnxrt import OnnxrtRunner
    from polygraphy.backend.trt import (
        CreateConfig,
        TrtRunner,
        engine_from_network,
        network_from_onnx_path,
    )
    from polygraphy.comparator import Comparator, DataLoader

    register_tensorrt_plugins()

    network = network_from_onnx_path(onnx_module)
    trt_engine = engine_from_network(network, config=CreateConfig())

    if dtype != torch.float32:
        pytest.skip("Comparator only supports float32")

    # Create runners for ONNX and TRT models
    trt_runner = TrtRunner(trt_engine)

    options = SessionOptions()
    options.register_custom_ops_library(get_library_path())
    onnx_runner = OnnxrtRunner(InferenceSession(onnx_module, sess_options=options))

    results = Comparator.run([trt_runner, onnx_runner], data_loader=DataLoader())
    Comparator.compare_accuracy(results)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def find_dtype(inputs, fallback_dtype):
    if isinstance(inputs, torch.Tensor):
        return inputs.dtype
    if isinstance(inputs, list) and len(inputs) > 0:
        return find_dtype(inputs[0], fallback_dtype)
    if isinstance(inputs, tuple) and len(inputs) > 0:
        return find_dtype(inputs[0], fallback_dtype)
    if isinstance(inputs, dict) and "inputs" in inputs:
        return find_dtype(inputs["inputs"], fallback_dtype)
    return fallback_dtype


def module_with_mode(
    mode: str,
    module: torch.nn.Module,
    inputs: list[torch.Tensor] | list[list[torch.Tensor]],
    math_dtype: torch.dtype,
    tmp_path: str,
    grad_modes: list[str] = ["eager", "compile", "jit", "export"],
) -> torch.nn.Module:
    dtype = find_dtype(inputs, math_dtype)
    if mode in ["trt", "torch_trt", "onnx", "onnx_dynamo"]:
        if not ONNX_AVAILABLE:
            pytest.skip("ONNX not available!")
        if dtype == torch.float64 or math_dtype == torch.float64:
            pytest.skip("TRT/ORT do not support float64")
    torch._dynamo.reset()
    with torch.set_grad_enabled(mode in grad_modes):
        if mode == "compile":
            torch._dynamo.allow_in_graph(torch.autograd.grad)
            # mfx = make_fx(module)(*inputs)
            module = torch.compile(module, fullgraph=True)
        elif mode == "fx":
            module = torch.fx.symbolic_trace(module)
        elif mode == "script":
            module = torch.jit.script(module)
            fname = os.path.join(tmp_path, "test.ts")
            torch.jit.save(module, fname)
            module = torch.jit.load(fname)
        elif mode == "jit":
            if isinstance(inputs, dict):
                module = torch.jit.trace(module, example_kwarg_inputs=inputs)
            else:
                module = torch.jit.trace(module, inputs)
            fname = os.path.join(tmp_path, "test.ts")
            torch.jit.save(module, fname)
            module = torch.jit.load(fname)
        elif mode == "export":
            if isinstance(inputs, dict):
                exp_program = torch.export.export(module, tuple(), inputs)
            else:
                exp_program = torch.export.export(module, tuple(inputs))
            fname = os.path.join(tmp_path, "test.pt2")
            torch.export.save(exp_program, fname)
            del exp_program
            module = torch.export.load(fname).module()
        elif mode == "torch_trt":
            if not TORCH_TRT_AVAILABLE:
                pytest.skip("torch_tensorrt is not installed!")
            register_tensorrt_plugins()
            exp_program = torch.export.export(module, tuple(inputs))
            module = torch_tensorrt.dynamo.compile(
                exp_program,
                inputs=inputs,
                require_full_compilation=True,
                min_block_size=1,
                enabled_precisions={torch.float32, dtype},
                # dryrun=True
            )
        elif mode == "onnx" or mode == "trt":
            try:
                onnx_path = os.path.join(tmp_path, "test.onnx")
                torch.onnx.export(
                    module,
                    tuple(inputs),
                    onnx_path,
                    dynamo=True,
                    custom_translation_table=onnx_custom_translation_table(),
                )
                if mode == "trt":
                    verify_trt(module, onnx_path, inputs, dtype)
                else:
                    verify_onnx(module, onnx_path, inputs, dtype)
            except ImportError:
                pytest.skip("ONNX/TRT is not available")
        elif mode == "eager":
            pass
        else:
            raise ValueError(f"No such mode: {mode}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return module


def create_random_tensor_2d(batch_size, stride, requires_grad, dtype, is_shared):
    data = torch.randn(
        (stride,) if is_shared else (batch_size, stride),
        dtype=dtype,
        device="cuda",
    ).requires_grad_(requires_grad)

    return data


def maybe_detach_and_to(tensor, *args, **kwargs):
    if tensor is not None:
        return tensor.clone().detach().to(*args, **kwargs)
    return None


def run_fwd_test(module, x: Sequence):
    with torch.no_grad():
        out = module(*x)
        test_output = [maybe_detach_and_to(out, dtype=torch.float32)]
        return test_output


def run_fwd_bwd_test(module, x: Sequence):
    out = module(*x)

    loss = out.sum()
    loss.backward()

    test_output = [maybe_detach_and_to(out, dtype=torch.float32)]
    test_output.extend([maybe_detach_and_to(t.grad, dtype=torch.float32) for t in x])

    return test_output


def run_bwd_bwd_test(module, x: Sequence):
    test_outputs = []
    out = module(*x)
    grads = torch.autograd.grad(out.pow(2).sum(), x, create_graph=True)
    test_outputs.extend([maybe_detach_and_to(g, dtype=torch.float32) for g in grads])
    loss = sum([g.sum() for g in grads])
    loss.backward()
    test_outputs.extend([maybe_detach_and_to(t.grad, dtype=torch.float32) for t in x])
    return test_outputs


def assert_close_modules(m_test, m_ref, inputs_test, procedure, tol_dict):
    outs_test = procedure(m_test, inputs_test)

    inputs_ref = [
        x.clone()
        .detach()
        .to(device="cuda", dtype=torch.float32)
        .requires_grad_(x.requires_grad)
        for x in inputs_test
    ]
    outs_ref = procedure(m_ref, inputs_ref)
    for out_test, out_ref in zip(outs_test, outs_ref):
        torch.testing.assert_close(out_test, out_ref, **tol_dict)


tol_dict = {
    # we compare against double for precision reasons
    # hence FP64 and FP32 threshold are the same
    (torch.float64, torch.float64): {"atol": 1e-9, "rtol": 1e-5},
    (torch.float32, torch.float64): {"atol": 1e-4, "rtol": 1e-5},
    (torch.float64, torch.float32): {"atol": 1e-4, "rtol": 1e-5},
    (torch.float32, torch.float32): {"atol": 1e-4, "rtol": 1e-5},
    (torch.float32, "float32"): {"atol": 1e-4, "rtol": 1e-5},
    (torch.bfloat16, torch.float32): {"atol": 4.0, "rtol": 1e-2},
    (torch.float16, torch.float32): {"atol": 0.25, "rtol": 1e-2},
}
