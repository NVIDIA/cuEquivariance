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
import os
import timeit

import pytest
import torch
import torch._dynamo

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors

torch._dynamo.config.cache_size_limit = 100


try:
    import cuequivariance_ops_torch.onnx  # noqa: F401
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401
    import onnxscript  # noqa: F401
    from cuequivariance_ops_torch.tensorrt import register_plugins

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

    register_plugins()

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


def module_with_mode(
    mode,
    module,
    inputs,
    math_dtype,
    tmp_path,
    grad_modes=["eager", "compile", "jit", "export"],
):
    if isinstance(inputs[0], list):
        dtype = inputs[0][0].dtype
    else:
        dtype = inputs[0].dtype
    if mode in ["trt", "torch_trt", "onnx", "onnx_dynamo", "export"]:
        if not ONNX_AVAILABLE:
            pytest.skip("ONNX not available!")
        if dtype == torch.float64 or math_dtype == torch.float64:
            pytest.skip("TRT/ORT do not support float64")

    with torch.set_grad_enabled(mode in grad_modes):
        if mode == "compile":
            import sys

            if sys.version_info.major == 3 and sys.version_info.minor >= 12:
                pytest.skip("torch dynamo needs cpy <= 3.11")
                module = torch.compile(module)
        elif mode == "fx":
            module = torch.fx.symbolic_trace(module)
        elif mode == "jit":
            module = torch.jit.trace(module, inputs)
            fname = os.path.join(tmp_path, "test.ts")
            torch.jit.save(module, fname)
            module = torch.jit.load(fname)
        elif mode == "export":
            exp_program = torch.export.export(module, tuple(inputs))
            fname = os.path.join(tmp_path, "test.pt2")
            torch.export.save(exp_program, fname)
            del exp_program
            module = torch.export.load(fname).module()
        elif mode == "torch_trt":
            if not TORCH_TRT_AVAILABLE:
                pytest.skip("torch_tensorrt is not installed!")
            register_plugins()
            exp_program = torch_tensorrt.dynamo.trace(module, inputs)
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
                    module, tuple(inputs), onnx_path, opset_version=17, verbose=False
                )
                if mode == "trt":
                    verify_trt(module, onnx_path, inputs, dtype)
                else:
                    verify_onnx(module, onnx_path, inputs, dtype)
            except ImportError:
                pytest.skip("ONNX/TRT is not available")

        elif mode == "onnx_dynamo":
            try:
                from cuequivariance_ops_torch.onnx import (
                    cuequivariance_ops_torch_onnx_registry,
                )

                export_options = torch.onnx.ExportOptions(
                    onnx_registry=cuequivariance_ops_torch_onnx_registry
                )
                onnx_program = torch.onnx.dynamo_export(
                    module, *inputs, export_options=export_options
                )
                onnx_path = os.path.join(tmp_path, "test.onnx")
                onnx_program.save(onnx_path)
                verify_onnx(module, onnx_path, inputs, dtype)
            except ImportError:
                pytest.skip("ONNX is not available")
        elif mode == "eager":
            pass
        else:
            raise ValueError(f"No such mode: {mode}")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return module


def maybe_detach_and_to(tensor, *args, **kwargs):
    if tensor is not None:
        return tensor.clone().detach().to(*args, **kwargs)
    return None


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def make_descriptors():
    # This ETP will trigger the fusedTP kernel
    yield descriptors.fully_connected_tensor_product(
        cue.Irreps("O3", "32x0e + 32x1o"),
        cue.Irreps("O3", "0e + 1o + 2e"),
        cue.Irreps("O3", "32x0e + 32x1o"),
    ).flatten_coefficient_modes()

    # This ETP will trigger the uniform1dx4 kernel
    yield (
        descriptors.channelwise_tensor_product(
            cue.Irreps("O3", "32x0e + 32x1o"),
            cue.Irreps("O3", "0e + 1o + 2e"),
            cue.Irreps("O3", "0e + 1o"),
        )
        .flatten_coefficient_modes()
        .squeeze_modes()
    )

    # These ETPs will trigger the symmetricContraction kernel
    yield descriptors.spherical_harmonics(cue.SO3(1), [1, 2, 3])
    yield descriptors.symmetric_contraction(
        cue.Irreps("O3", "32x0e + 32x1o"), cue.Irreps("O3", "32x0e + 32x1o"), [1, 2, 3]
    )


settings1 = [
    (torch.float32, torch.float64),
    (torch.float64, torch.float32),
    (torch.float32, torch.float32),
    (torch.float64, torch.float64),
]
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    settings1 += [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
    ]


@pytest.mark.parametrize("e", make_descriptors())
@pytest.mark.parametrize("dtype, math_dtype", settings1)
def test_performance_cuda_vs_fx(
    e: cue.EquivariantTensorProduct,
    dtype: torch.dtype,
    math_dtype: torch.dtype,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    m = cuet.EquivariantTensorProduct(
        e,
        layout=cue.ir_mul,
        device=device,
        math_dtype=math_dtype,
        use_fallback=False,
    )

    m1 = cuet.EquivariantTensorProduct(
        e,
        layout=cue.ir_mul,
        device=device,
        math_dtype=math_dtype,
        use_fallback=True,
        optimize_fallback=True,
    )

    inputs = [
        torch.randn((1024, inp.irreps.dim), device=device, dtype=dtype)
        for inp in e.inputs
    ]

    for _ in range(10):
        m(inputs)
        m1(inputs)
    torch.cuda.synchronize()

    def f():
        ret = m(inputs)
        ret = torch.sum(ret)
        return ret

    def f1():
        ret = m1(inputs)
        ret = torch.sum(ret)
        return ret

    t0 = timeit.Timer(f).timeit(number=10)
    t1 = timeit.Timer(f1).timeit(number=10)
    assert t0 < t1


settings2 = [
    (torch.float32, torch.float32, 1e-4, 1e-6),
    (torch.float32, torch.float64, 1e-5, 1e-6),
    (torch.float64, torch.float32, 1e-5, 1e-6),
    (torch.float64, torch.float64, 1e-12, 0),
]
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    settings2 += [
        (torch.float16, torch.float32, 1, 0.2),
        (torch.bfloat16, torch.float32, 1, 0.2),
    ]


@pytest.mark.parametrize("e", make_descriptors())
@pytest.mark.parametrize("dtype, math_dtype, atol, rtol", settings2)
def test_precision_cuda_vs_fx(
    e: cue.EquivariantTensorProduct,
    dtype: torch.dtype,
    math_dtype: torch.dtype,
    atol: float,
    rtol: float,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    inputs = [
        torch.randn((1024, inp.irreps.dim), device=device, dtype=dtype)
        for inp in e.inputs
    ]
    m = cuet.EquivariantTensorProduct(
        e, layout=cue.ir_mul, device=device, math_dtype=math_dtype, use_fallback=False
    )
    y0 = m(inputs)

    m = cuet.EquivariantTensorProduct(
        e,
        layout=cue.ir_mul,
        device=device,
        math_dtype=torch.float64,
        use_fallback=True,
        optimize_fallback=True,
    )
    inputs = [x.to(torch.float64) for x in inputs]
    y1 = m(inputs).to(dtype)

    torch.testing.assert_close(y0, y1, atol=atol, rtol=rtol)


@pytest.mark.parametrize("e", make_descriptors())
@pytest.mark.parametrize("dtype, math_dtype, atol, rtol", settings2)
def test_compile(
    e: cue.EquivariantTensorProduct,
    dtype: torch.dtype,
    math_dtype: torch.dtype,
    atol: float,
    rtol: float,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    m = cuet.EquivariantTensorProduct(
        e, layout=cue.mul_ir, use_fallback=False, device=device, math_dtype=math_dtype
    )
    inputs = [
        torch.randn((1024, inp.irreps.dim), device=device, dtype=dtype)
        for inp in e.inputs
    ]
    res = m(inputs)
    m_compile = torch.compile(m, fullgraph=True)
    res_script = m_compile(inputs)
    torch.testing.assert_close(res, res_script, atol=atol, rtol=rtol)


@pytest.mark.parametrize("e", make_descriptors())
@pytest.mark.parametrize("dtype, math_dtype, atol, rtol", settings2)
def test_script(
    e: cue.EquivariantTensorProduct,
    dtype: torch.dtype,
    math_dtype: torch.dtype,
    atol: float,
    rtol: float,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    m = cuet.EquivariantTensorProduct(
        e, layout=cue.mul_ir, use_fallback=False, device=device, math_dtype=math_dtype
    )
    inputs = [
        torch.randn((1024, inp.irreps.dim), device=device, dtype=dtype)
        for inp in e.inputs
    ]
    res = m(inputs)
    m_script = torch.jit.script(m)
    res_script = m_script(inputs)
    torch.testing.assert_close(res, res_script, atol=atol, rtol=rtol)


# export_modes = ["onnx", "onnx_dynamo", "trt", "torch_trt", "jit"]
export_modes = ["trt", "onnx"]


@pytest.mark.parametrize("e", make_descriptors())
@pytest.mark.parametrize("dtype, math_dtype, atol, rtol", settings2)
@pytest.mark.parametrize("mode", export_modes)
def test_export(
    e: cue.EquivariantTensorProduct,
    dtype: torch.dtype,
    math_dtype: torch.dtype,
    atol: float,
    rtol: float,
    mode: str,
    tmp_path,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    m = cuet.EquivariantTensorProduct(
        e, layout=cue.mul_ir, math_dtype=math_dtype, use_fallback=False, device=device
    )
    inputs = [
        torch.randn((1024, inp.irreps.dim), device=device, dtype=dtype)
        for inp in e.inputs
    ]
    res = m(inputs)
    m_script = module_with_mode(mode, m, [inputs], math_dtype, tmp_path)
    res_script = m_script(inputs)
    torch.testing.assert_close(res, res_script, atol=atol, rtol=rtol)
