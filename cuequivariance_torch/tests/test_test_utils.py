# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
from types import ModuleType

import pytest
import torch
from cuequivariance_torch._tests import utils


def _install_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    parent_name, _, child_name = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        monkeypatch.setattr(parent, child_name, module, raising=False)
    return module


def test_find_dtype_recurses_through_supported_containers():
    fallback = torch.float16
    tensor = torch.ones(2, dtype=torch.float64)

    assert utils.find_dtype(tensor, fallback) == torch.float64
    assert utils.find_dtype([tensor], fallback) == torch.float64
    assert utils.find_dtype((tensor,), fallback) == torch.float64
    assert utils.find_dtype({"inputs": (tensor,)}, fallback) == torch.float64
    assert utils.find_dtype({"other": tensor}, fallback) == fallback
    assert utils.find_dtype([], fallback) == fallback


@pytest.mark.parametrize("mode", ["eager", "fx", "script", "jit", "export"])
def test_module_with_mode_preserves_relu_result(mode, tmp_path):
    module = torch.nn.ReLU()
    inputs = [torch.tensor([-2.0, 0.5])]

    converted = utils.module_with_mode(mode, module, inputs, torch.float32, tmp_path)

    torch.testing.assert_close(converted(*inputs), module(*inputs))


@pytest.mark.parametrize("mode", ["jit", "export"])
def test_module_with_mode_supports_keyword_inputs(mode, tmp_path):
    module = torch.nn.ReLU()
    inputs = {"input": torch.tensor([-1.0, 3.0])}

    converted = utils.module_with_mode(mode, module, inputs, torch.float32, tmp_path)

    torch.testing.assert_close(converted(**inputs), module(**inputs))


@pytest.mark.parametrize("mode", ["onnx", "trt", "torch_trt", "onnx_dynamo"])
def test_module_with_mode_skips_backend_modes_without_onnx(mode, monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "ONNX_AVAILABLE", False)

    with pytest.raises(pytest.skip.Exception, match="ONNX not available"):
        utils.module_with_mode(
            mode, torch.nn.ReLU(), [torch.ones(1)], torch.float32, tmp_path
        )


@pytest.mark.parametrize(
    ("input_dtype", "math_dtype"),
    [(torch.float64, torch.float32), (torch.float32, torch.float64)],
)
def test_module_with_mode_skips_float64_onnx_inputs(
    input_dtype, math_dtype, monkeypatch, tmp_path
):
    monkeypatch.setattr(utils, "ONNX_AVAILABLE", True)

    with pytest.raises(pytest.skip.Exception, match="float64"):
        utils.module_with_mode(
            "onnx",
            torch.nn.ReLU(),
            [torch.ones(1, dtype=input_dtype)],
            math_dtype,
            tmp_path,
        )


def test_module_with_mode_skips_torch_trt_when_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "ONNX_AVAILABLE", True)
    monkeypatch.setattr(utils, "TORCH_TRT_AVAILABLE", False)

    with pytest.raises(pytest.skip.Exception, match="torch_tensorrt is not installed"):
        utils.module_with_mode(
            "torch_trt", torch.nn.ReLU(), [torch.ones(1)], torch.float32, tmp_path
        )


def test_module_with_mode_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="No such mode: invalid"):
        utils.module_with_mode(
            "invalid", torch.nn.ReLU(), [torch.ones(1)], torch.float32, tmp_path
        )


@pytest.mark.parametrize("mode", ["onnx", "trt"])
def test_module_with_mode_exports_and_selects_backend_verifier(
    mode, monkeypatch, tmp_path
):
    monkeypatch.setattr(utils, "ONNX_AVAILABLE", True)
    monkeypatch.setattr(
        utils, "onnx_custom_translation_table", lambda: "table", raising=False
    )
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)

    export_calls = []
    verifier_calls = []

    def fake_export(module, inputs, path, **kwargs):
        export_calls.append((module, inputs, path, kwargs))

    def fake_verify(module, path, inputs, dtype):
        verifier_calls.append(
            ("trt" if mode == "trt" else "onnx", module, path, inputs, dtype)
        )

    monkeypatch.setattr(utils.torch.onnx, "export", fake_export)
    monkeypatch.setattr(utils, "verify_onnx", fake_verify)
    monkeypatch.setattr(utils, "verify_trt", fake_verify)
    module = torch.nn.ReLU()
    inputs = [torch.tensor([-1.0, 2.0])]

    converted = utils.module_with_mode(mode, module, inputs, torch.float32, tmp_path)

    assert converted is module
    assert len(export_calls) == 1
    exported_module, exported_inputs, path, kwargs = export_calls[0]
    assert exported_module is module
    assert exported_inputs == tuple(inputs)
    assert path.endswith("test.onnx")
    assert kwargs == {"dynamo": True, "custom_translation_table": "table"}
    assert verifier_calls == [
        ("trt" if mode == "trt" else "onnx", module, path, inputs, torch.float32)
    ]


def test_module_with_mode_skips_export_import_error(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "ONNX_AVAILABLE", True)
    monkeypatch.setattr(
        utils, "onnx_custom_translation_table", lambda: None, raising=False
    )
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)

    def unavailable_export(*args, **kwargs):
        raise ImportError("optional ONNX dependency unavailable")

    monkeypatch.setattr(utils.torch.onnx, "export", unavailable_export)

    with pytest.raises(pytest.skip.Exception, match="ONNX/TRT is not available"):
        utils.module_with_mode(
            "onnx", torch.nn.ReLU(), [torch.ones(1)], torch.float32, tmp_path
        )


@pytest.mark.parametrize("library_path_fails", [False, True])
def test_verify_onnx_registers_custom_ops_when_available_and_compares(
    library_path_fails, monkeypatch
):
    calls = []

    class SessionOptions:
        def __init__(self):
            self.registered_libraries = []

        def register_custom_ops_library(self, path):
            self.registered_libraries.append(path)

    onnxruntime = ModuleType("onnxruntime")
    onnxruntime.SessionOptions = SessionOptions
    extensions = ModuleType("onnxruntime_extensions")

    def get_library_path():
        if library_path_fails:
            raise RuntimeError("custom ops library is unavailable")
        return "custom_ops.so"

    extensions.get_library_path = get_library_path
    verification = ModuleType("torch.onnx.verification")

    class VerificationOptions:
        pass

    def compare(module, onnx_module, inputs, *args):
        options = SessionOptions()
        expected_libraries = [] if library_path_fails else ["custom_ops.so"]
        assert options.registered_libraries == expected_libraries
        calls.append((module, onnx_module, inputs, args))

    verification.VerificationOptions = VerificationOptions
    verification._compare_onnx_pytorch_model = compare
    monkeypatch.setitem(sys.modules, "onnxruntime", onnxruntime)
    monkeypatch.setitem(sys.modules, "onnxruntime_extensions", extensions)
    monkeypatch.setitem(sys.modules, "torch.onnx.verification", verification)
    monkeypatch.setattr(utils.torch.onnx, "verification", verification, raising=False)
    monkeypatch.setattr(utils.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(utils.torch.cuda, "empty_cache", lambda: None)
    original_init = SessionOptions.__init__
    module, onnx_module, inputs = object(), "model.onnx", [torch.ones(1)]

    utils.verify_onnx(module, onnx_module, inputs, torch.float32)

    assert len(calls) == 1
    assert calls[0][:3] == (module, onnx_module, tuple(inputs))
    assert calls[0][3][:2] == (None, None)
    assert isinstance(calls[0][3][2], VerificationOptions)
    assert SessionOptions.__init__ is original_init


def test_verify_onnx_skips_non_float32():
    with pytest.raises(pytest.skip.Exception, match="only checked for float32"):
        utils.verify_onnx(None, None, [], torch.float64)


@pytest.mark.parametrize(
    ("version", "dtype", "skip_reason"),
    [
        ("10.2.0", torch.float32, "TRT < 10.3.0"),
        ("10.3.0", torch.float64, "does not support float64"),
    ],
)
def test_verify_trt_rejects_unsupported_version_or_dtype(
    version, dtype, skip_reason, monkeypatch
):
    _install_module(monkeypatch, "tensorrt", __version__=version)
    _install_module(
        monkeypatch,
        "pkg_resources",
        parse_version=lambda value: tuple(int(part) for part in value.split(".")),
    )

    with pytest.raises(pytest.skip.Exception, match=skip_reason):
        utils.verify_trt(None, None, [], dtype)


@pytest.mark.parametrize(
    ("dtype", "comparator_supported"),
    [(torch.float32, True), (torch.float16, False)],
)
def test_verify_trt_builds_runners_and_checks_comparator_dtype(
    dtype, comparator_supported, monkeypatch
):
    events = []

    class SessionOptions:
        def __init__(self):
            self.registered_libraries = []

        def register_custom_ops_library(self, path):
            self.registered_libraries.append(path)

    def inference_session(path, sess_options):
        events.append(("inference_session", path, sess_options))
        return ("onnx_session", path)

    class Comparator:
        @staticmethod
        def run(runners, data_loader):
            events.append(("run", runners, data_loader))
            return "results"

        @staticmethod
        def compare_accuracy(results):
            events.append(("compare_accuracy", results))

    _install_module(monkeypatch, "tensorrt", __version__="10.3.0")
    _install_module(
        monkeypatch,
        "pkg_resources",
        parse_version=lambda value: tuple(int(part) for part in value.split(".")),
    )
    _install_module(
        monkeypatch,
        "onnxruntime",
        SessionOptions=SessionOptions,
        InferenceSession=inference_session,
    )
    _install_module(
        monkeypatch,
        "onnxruntime_extensions",
        get_library_path=lambda: "custom_ops.so",
    )
    _install_module(monkeypatch, "polygraphy", __path__=[])
    _install_module(monkeypatch, "polygraphy.backend", __path__=[])
    _install_module(
        monkeypatch,
        "polygraphy.backend.onnxrt",
        OnnxrtRunner=lambda session: ("onnx_runner", session),
    )
    _install_module(
        monkeypatch,
        "polygraphy.backend.trt",
        CreateConfig=lambda: "config",
        TrtRunner=lambda engine: ("trt_runner", engine),
        engine_from_network=lambda network, config: ("engine", network, config),
        network_from_onnx_path=lambda path: ("network", path),
    )
    _install_module(
        monkeypatch,
        "polygraphy.comparator",
        Comparator=Comparator,
        DataLoader=lambda: "loader",
    )
    monkeypatch.setattr(
        utils,
        "register_tensorrt_plugins",
        lambda: events.append(("plugins",)),
        raising=False,
    )
    monkeypatch.setattr(utils.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(utils.torch.cuda, "empty_cache", lambda: None)

    if not comparator_supported:
        with pytest.raises(
            pytest.skip.Exception, match="Comparator only supports float32"
        ):
            utils.verify_trt(object(), "model.onnx", [torch.ones(1)], dtype)
        assert events[0] == ("plugins",)
        return

    utils.verify_trt(object(), "model.onnx", [torch.ones(1)], dtype)

    assert events[0] == ("plugins",)
    inference_event = events[1]
    assert inference_event[:2] == ("inference_session", "model.onnx")
    assert inference_event[2].registered_libraries == ["custom_ops.so"]
    assert events[2] == (
        "run",
        [
            ("trt_runner", ("engine", ("network", "model.onnx"), "config")),
            ("onnx_runner", ("onnx_session", "model.onnx")),
        ],
        "loader",
    )
    assert events[3] == ("compare_accuracy", "results")


@pytest.mark.parametrize(
    ("is_shared", "expected_shape"), [(True, (3,)), (False, (2, 3))]
)
def test_create_random_tensor_2d_builds_expected_shape(
    is_shared, expected_shape, monkeypatch
):
    calls = []

    def fake_randn(shape, *, dtype, device):
        calls.append((shape, dtype, device))
        return torch.zeros(shape, dtype=dtype)

    monkeypatch.setattr(utils.torch, "randn", fake_randn)

    tensor = utils.create_random_tensor_2d(
        batch_size=2,
        stride=3,
        requires_grad=True,
        dtype=torch.float64,
        is_shared=is_shared,
    )

    assert tensor.shape == expected_shape
    assert tensor.dtype == torch.float64
    assert tensor.requires_grad
    assert calls == [(expected_shape, torch.float64, "cuda")]


def test_run_fwd_test_returns_detached_float32_output():
    inputs = [torch.tensor([1.5, -2.0], dtype=torch.float64, requires_grad=True)]

    outputs = utils.run_fwd_test(torch.nn.Identity(), inputs)

    assert len(outputs) == 1
    assert outputs[0].dtype == torch.float32
    assert not outputs[0].requires_grad
    torch.testing.assert_close(outputs[0], torch.tensor([1.5, -2.0]))


def test_run_fwd_bwd_test_returns_output_and_input_gradient():
    inputs = [torch.tensor([2.0, -3.0], requires_grad=True)]

    outputs = utils.run_fwd_bwd_test(lambda x: x.square(), inputs)

    torch.testing.assert_close(outputs[0], torch.tensor([4.0, 9.0]))
    torch.testing.assert_close(outputs[1], torch.tensor([4.0, -6.0]))


def test_run_bwd_bwd_test_differentiates_squared_output_loss():
    inputs = [torch.tensor([2.0], requires_grad=True)]

    outputs = utils.run_bwd_bwd_test(lambda x: x.pow(3), inputs)

    # The helper differentiates loss=(x**3)**2=x**6: 6*x**5, then 30*x**4.
    torch.testing.assert_close(outputs[0], torch.tensor([192.0]))
    torch.testing.assert_close(outputs[1], torch.tensor([480.0]))


def test_maybe_detach_and_to_preserves_none():
    assert utils.maybe_detach_and_to(None, dtype=torch.float32) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="helper requires CUDA")
def test_assert_close_modules_compares_outputs_and_gradients():
    module = torch.nn.Linear(2, 1, bias=False, device="cuda")
    inputs = [torch.tensor([[1.0, -2.0]], device="cuda", requires_grad=True)]

    utils.assert_close_modules(
        module,
        module,
        inputs,
        utils.run_fwd_bwd_test,
        {"atol": 1e-6, "rtol": 1e-6},
    )
