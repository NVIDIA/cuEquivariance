# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import util
from types import SimpleNamespace

import pytest

triton_utils = pytest.importorskip("cuequivariance_jax.triangle.triton_utils")


@pytest.mark.parametrize(
    ("release", "expected_ptx", "cuda_version"),
    [
        ("12.4", 84, (12, 4)),
        ("12.5", 85, (12, 5)),
        ("12.8", 87, (12, 8)),
        ("12.9", 87, (12, 9)),
        ("11.7", 77, (11, 7)),
        ("11.8", 78, (11, 8)),
        ("10.2", None, (10, 2)),
    ],
)
def test_get_max_ptx_version_maps_ptxas_releases(
    monkeypatch, release, expected_ptx, cuda_version
):
    monkeypatch.setattr(triton_utils, "_PTXAS_VERSION_CACHE", None)
    monkeypatch.setattr(triton_utils, "_CUDA_VERSION_CACHE", None)
    monkeypatch.setattr(
        triton_utils.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"Cuda compilation tools, release {release}, V{release}.68",
        ),
    )

    assert triton_utils._get_max_ptx_version() == expected_ptx
    assert triton_utils._CUDA_VERSION_CACHE == cuda_version


def test_get_max_ptx_version_reuses_cache_and_handles_unavailable_ptxas(monkeypatch):
    monkeypatch.setattr(triton_utils, "_PTXAS_VERSION_CACHE", 85)
    monkeypatch.setattr(triton_utils, "_CUDA_VERSION_CACHE", (12, 5))
    monkeypatch.setattr(
        triton_utils.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("cached result should avoid ptxas"),
    )
    assert triton_utils._get_max_ptx_version() == 85

    for result in (
        SimpleNamespace(returncode=1, stdout=""),
        SimpleNamespace(returncode=0, stdout="unexpected ptxas output"),
    ):
        monkeypatch.setattr(triton_utils, "_PTXAS_VERSION_CACHE", None)
        monkeypatch.setattr(triton_utils, "_CUDA_VERSION_CACHE", None)
        monkeypatch.setattr(
            triton_utils.subprocess,
            "run",
            lambda *args, result=result, **kwargs: result,
        )
        assert triton_utils._get_max_ptx_version() is None

    monkeypatch.setattr(triton_utils, "_PTXAS_VERSION_CACHE", None)

    def ptxas_missing(*args, **kwargs):
        raise OSError("ptxas is not on PATH")

    monkeypatch.setattr(triton_utils.subprocess, "run", ptxas_missing)
    assert triton_utils._get_max_ptx_version() is None


def _load_isolated_triton_utils():
    spec = util.spec_from_file_location(
        "_cuequivariance_jax_triton_utils_init_test", triton_utils.__file__
    )
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_triton_utils_cache_directory_fallbacks(monkeypatch, tmp_path):
    cache_dir = tmp_path / ".triton" / "cache"
    monkeypatch.setattr(triton_utils.os.path, "expanduser", lambda _: str(tmp_path))

    empty_cache = SimpleNamespace(cache=SimpleNamespace(dir=""))
    monkeypatch.setattr(triton_utils.triton_cache, "knobs", empty_cache, raising=False)
    module = _load_isolated_triton_utils()
    assert module.cache_dir == str(cache_dir)
    assert empty_cache.cache.dir == str(cache_dir)
    assert cache_dir.is_dir()
    cache_dir.rmdir()

    class BrokenCacheKnobs:
        @property
        def cache(self):
            raise RuntimeError("cache configuration unavailable")

    monkeypatch.setattr(
        triton_utils.triton_cache, "knobs", BrokenCacheKnobs(), raising=False
    )
    module = _load_isolated_triton_utils()
    assert module.cache_dir == str(cache_dir)
    assert cache_dir.is_dir()


def test_triton_utils_tolerates_missing_triton_version_attribute(monkeypatch):
    monkeypatch.setattr(triton_utils.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.delattr(triton_utils.triton, "__version__")

    module = _load_isolated_triton_utils()

    assert module.TRITON_VERSION is None


class _OptionsWithClusterAndPtx:
    def __init__(
        self,
        num_warps,
        num_stages,
        num_ctas,
        debug,
        enable_fp_fusion,
        cluster_dims=None,
        ptx_version=None,
    ):
        self.__dict__.update(locals())
        del self.__dict__["self"]


class _OptionsWithoutOptionalFields:
    def __init__(self, num_warps, num_stages, num_ctas, debug, enable_fp_fusion):
        self.__dict__.update(locals())
        del self.__dict__["self"]


def _compiled_kernel():
    return SimpleNamespace(
        name="mock_kernel",
        metadata=SimpleNamespace(shared=32),
        asm={"ptx": "mock ptx"},
    )


def _install_fake_compiler(
    monkeypatch,
    options_type,
    compile_impl,
    compute_versions=(12, 9),
    max_ptx_version=85,
):
    observed = {"sources": [], "compiles": []}

    def make_source(**kwargs):
        observed["sources"].append(kwargs)
        return kwargs

    def compile_source(source, target, options):
        observed["compiles"].append(
            {"source": source, "target": target, "options": options}
        )
        return compile_impl(source, target, options)

    monkeypatch.setattr(triton_utils.cb, "CUDAOptions", options_type)
    monkeypatch.setattr(triton_utils.tc, "ASTSource", make_source)
    monkeypatch.setattr(
        triton_utils.tc, "GPUTarget", lambda backend, cc, warp: (backend, cc, warp)
    )
    monkeypatch.setattr(triton_utils.tc, "compile", compile_source)
    monkeypatch.setattr(
        triton_utils.gpu_triton,
        "TritonKernel",
        lambda *args: ("kernel", args),
        raising=False,
    )
    monkeypatch.setattr(triton_utils, "_get_max_ptx_version", lambda: max_ptx_version)
    monkeypatch.setattr(triton_utils, "_CUDA_VERSION_CACHE", compute_versions)
    monkeypatch.setattr(
        triton_utils, "TRITON_VERSION", triton_utils.version.parse("3.4.0")
    )
    monkeypatch.setattr(triton_utils, "_TRITON_KERNEL_CACHE", {})
    monkeypatch.setattr(triton_utils.jax, "__version__", "0.8.2")
    return observed


def _run_fake_compile(compute_capability=121):
    def kernel_fn(x, BLOCK):
        del x, BLOCK

    kernel_fn.src = "mock kernel source"
    return triton_utils._compile_triton(
        kernel_fn,
        {"x": "*fp32"},
        {"BLOCK": 16},
        num_warps=4,
        num_stages=2,
        compute_capability=compute_capability,
        enable_fp_fusion=True,
    )


@pytest.mark.parametrize(
    ("options_type", "compute_capability", "max_ptx_version", "optional_options"),
    [
        (
            _OptionsWithClusterAndPtx,
            121,
            85,
            {"cluster_dims": (1, 1, 1), "ptx_version": 88},
        ),
        (_OptionsWithoutOptionalFields, 90, 85, {}),
        (
            _OptionsWithClusterAndPtx,
            90,
            None,
            {"cluster_dims": (1, 1, 1), "ptx_version": None},
        ),
    ],
)
def test_compile_triton_passes_only_supported_cuda_options(
    monkeypatch, options_type, compute_capability, max_ptx_version, optional_options
):
    observed = _install_fake_compiler(
        monkeypatch,
        options_type,
        lambda *args: _compiled_kernel(),
        max_ptx_version=max_ptx_version,
    )

    kernel = _run_fake_compile(compute_capability)

    expected_options = {
        "num_warps": 4,
        "num_stages": 2,
        "num_ctas": 1,
        "debug": False,
        "enable_fp_fusion": True,
        **optional_options,
    }
    assert observed["compiles"][0]["options"] == expected_options
    assert observed["compiles"][0]["target"] == ("cuda", compute_capability, 32)
    assert kernel == (
        "kernel",
        ("mock_kernel", 4, 1, 32, "mock ptx", "", compute_capability),
    )


@pytest.mark.parametrize("success_on", [3, None])
def test_compile_triton_retries_legacy_constexpr_apis(monkeypatch, success_on):
    calls = []

    def compile_api(*args):
        calls.append(args)
        if len(calls) == success_on:
            return _compiled_kernel()
        if len(calls) == 1:
            raise TypeError("unsupported Triton compiler API")
        raise AttributeError("legacy constexpr API required")

    observed = _install_fake_compiler(
        monkeypatch, _OptionsWithClusterAndPtx, compile_api
    )

    if success_on is None:
        with pytest.raises(RuntimeError, match="Failed to compile Triton kernel"):
            _run_fake_compile(90)
    else:
        _run_fake_compile(90)

    assert len(calls) == 3
    assert observed["sources"][0]["signature"] == {"x": "*fp32"}
    assert observed["sources"][0]["constexprs"] == {"BLOCK": 16}
    assert observed["sources"][1]["signature"] == {
        "x": "*fp32",
        "BLOCK": "constexpr",
    }
    assert observed["sources"][1]["constexprs"] == {"BLOCK": 16}
    assert observed["sources"][2]["signature"] == {
        "x": "*fp32",
        "BLOCK": "constexpr",
    }
    assert observed["sources"][2]["constants"] == {"BLOCK": 16}


def test_compile_triton_rejects_triton_33_before_compilation(monkeypatch):
    observed = _install_fake_compiler(
        monkeypatch,
        _OptionsWithClusterAndPtx,
        lambda *args: pytest.fail("unsupported Triton must be rejected first"),
    )
    monkeypatch.setattr(
        triton_utils, "TRITON_VERSION", triton_utils.version.parse("3.3.0")
    )

    with pytest.raises(ImportError, match="Triton version 3.3.0 is not supported"):
        _run_fake_compile()

    assert observed["compiles"] == []
