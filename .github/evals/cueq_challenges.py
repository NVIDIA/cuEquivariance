# SPDX-License-Identifier: Apache-2.0
"""Execute challenge submissions. This file is a verifier, never a skill resource.

Standalone: python cueq_challenges.py --backend jax --case layout --solution FILE
Harbor: read entry.json and grade /workspace/solution.py in the prepared CPU venv.
Custom scores supplement ACES; they do not overwrite its metrics or pass rule.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

PARAMETERS = ("radial_in", "radial_out", "projection")


def load_solution(path):
    # Match normal execution of a saved Python module, including sibling helpers
    # and decorators that resolve annotations through sys.modules.
    sys.path.insert(0, str(path.resolve().parent))
    spec = importlib.util.spec_from_file_location("challenge_solution", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def array(value):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def close(actual, expected, *, exact=False):
    import numpy as np

    actual, expected = array(actual), array(expected)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    assert np.isfinite(actual).all(), "nonfinite result"
    if exact:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-9)


def numpy_blocks(irreps, flat):
    """Independent oracle: slice original blocks and concatenate equal irreps."""
    import numpy as np

    parts, offset = {}, 0
    for mul, ir in irreps:
        block = flat[..., offset : offset + mul * ir.dim]
        block = block.reshape(flat.shape[:-1] + (ir.dim, mul)).swapaxes(-1, -2)
        parts.setdefault(ir, []).append(block)
        offset += mul * ir.dim
    return {ir: np.concatenate(values, axis=-2) for ir, values in parts.items()}


def change_layout(flat, irreps, *, source):
    import numpy as np

    parts, offset = [], 0
    for mul, ir in irreps:
        suffix = (mul, ir.dim) if source == "mul_ir" else (ir.dim, mul)
        block = flat[..., offset : offset + mul * ir.dim]
        parts.append(
            block.reshape(flat.shape[:-1] + suffix)
            .swapaxes(-1, -2)
            .reshape(flat.shape[:-1] + (mul * ir.dim,))
        )
        offset += mul * ir.dim
    return np.concatenate(parts, axis=-1)


def check_layout(solution, backend):
    import numpy as np

    import cuequivariance as cue

    if backend == "jax":
        import jax
        import jax.numpy as xp

        jax.config.update("jax_enable_x64", True)
    else:
        import torch

    for text in ("1o + 2x0e + 2x1o + 1e + 0o", "2x1e + 0o + 1o + 1e + 2x0e + 1o"):
        irreps = cue.Irreps("O3", text)
        for batch in ((), (4,), (2, 3)):
            raw = np.arange(int(np.prod(batch)) * irreps.dim, dtype=np.float64)
            raw = (raw.reshape(batch + (irreps.dim,)) + 0.25) / 8
            if backend == "jax":
                x = xp.asarray(raw)
            else:
                # Exercise a noncontiguous tensor with the same values.
                x = torch.tensor(np.stack((raw, -raw), -1))[..., 0]
                x.requires_grad_()
                assert not x.is_contiguous()
            expected = numpy_blocks(irreps, raw)
            blocks = solution.to_blocks(irreps, x)
            assert set(blocks) == set(expected), (
                "keys must include both degree and parity"
            )
            for ir, value in expected.items():
                close(blocks[ir], value, exact=True)
            restored = solution.from_blocks(irreps, blocks)
            close(restored, raw, exact=True)

            # Use a nonuniform linear loss so inverse permutations cannot hide.
            cotangent = np.linspace(0.2, 1.3, raw.size).reshape(raw.shape)
            if backend == "jax":

                def convert(z):
                    return solution.to_blocks(irreps, z)

                def round_trip(z):
                    return solution.from_blocks(irreps, convert(z))

                close(jax.jit(round_trip)(x), raw, exact=True)
                for ir, value in jax.jit(convert)(x).items():
                    close(value, expected[ir], exact=True)
                gradient = jax.grad(
                    lambda z: xp.sum(round_trip(z) * xp.asarray(cotangent))
                )(x)
            else:
                gradient = torch.autograd.grad(
                    (restored * torch.tensor(cotangent)).sum(), x
                )[0]
            close(gradient, cotangent)

            rep = cue.IrrepsAndLayout(irreps, cue.ir_mul)
            for angles, parity in (([0.21, -0.32, 0.17], [0]), ([0, 0, 0], [1])):
                matrix = rep.exp_map(angles, parity)
                transformed = raw @ matrix.T
                z = (
                    xp.asarray(transformed)
                    if backend == "jax"
                    else torch.tensor(transformed)
                )
                transformed_blocks = solution.to_blocks(irreps, z)
                for ir in expected:
                    close(
                        transformed_blocks[ir],
                        expected[ir] @ ir.exp_map(angles, parity).T,
                    )


def graph_descriptors():
    import cuequivariance as cue

    irreps_in = cue.Irreps("SO3", "2x0 + 2x1")
    irreps_sh = cue.Irreps("SO3", "0 + 1 + 2")
    irreps_out = cue.Irreps("SO3", "0 + 1")
    product = cue.descriptors.channelwise_tensor_product(
        irreps_in, irreps_sh, [cue.SO3(0), cue.SO3(1)]
    )
    projection = cue.descriptors.linear(product.outputs[0].irreps, irreps_out)
    harmonics = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2])
    return irreps_in, irreps_out, product, projection, harmonics


def graph_reference(params, x, vectors, senders, receivers, num_dst):
    """Core NumPy descriptors plus an explicit edge loop; no frontend forward."""
    import numpy as np

    irreps_in, irreps_out, product, projection, harmonics = graph_descriptors()
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    weights = np.tanh(lengths @ params["radial_in"]) @ params["radial_out"]
    sh = harmonics(vectors / lengths)[0]
    flat = change_layout(x, irreps_in, source="mul_ir")
    result = np.zeros((num_dst, irreps_out.dim), dtype=np.float64)
    for edge, (src, dst) in enumerate(zip(senders, receivers)):
        message = product(weights[edge], flat[src], sh[edge])[0]
        projected = projection(params["projection"], message)[0]
        result[dst] += change_layout(projected, irreps_out, source="ir_mul")
    return result


def graph_inputs(seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    irreps_in, _, product, projection, _ = graph_descriptors()
    params = {
        "radial_in": rng.normal(scale=0.4, size=(1, 4)),
        "radial_out": rng.normal(scale=0.4, size=(4, product.inputs[0].dim)),
        "projection": rng.normal(scale=0.4, size=(projection.inputs[0].dim,)),
    }
    x = rng.normal(size=(5, irreps_in.dim))
    vectors = rng.normal(size=(7, 3)) + np.array([0.4, -0.2, 0.7])
    senders = np.array([0, 1, 3, 0, 1, 4, 2], dtype=np.int64)
    receivers = np.array([2, 0, 2, 1, 2, 0, 1], dtype=np.int64)
    return params, x, vectors, senders, receivers


def check_graph(solution, backend):
    import numpy as np

    import cuequivariance as cue

    if backend == "jax":
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
    else:
        import torch

    irreps_in, irreps_out, _, _, _ = graph_descriptors()
    for seed in (41, 83):
        params, x, vectors, senders, receivers = graph_inputs(seed)
        expected = graph_reference(params, x, vectors, senders, receivers, 4)
        if backend == "jax":
            p = {k: jnp.asarray(v) for k, v in params.items()}
            a, b, s, r = map(jnp.asarray, (x, vectors, senders, receivers))

            def apply(pp, xx, vv, ss, rr):
                return solution.apply(pp, xx, vv, ss, rr, 4)

            actual = apply(p, a, b, s, r)
            close(jax.jit(apply)(p, a, b, s, r), expected)

            def loss(pp, xx, vv):
                return jnp.sum(apply(pp, xx, vv, s, r) ** 2)

            derivatives = jax.grad(loss, argnums=(0, 1, 2))(p, a, b)
            close(
                jax.jit(jax.grad(loss))(p, a, b)["radial_in"],
                derivatives[0]["radial_in"],
            )
            grad_params, grad_x, grad_vectors = derivatives
        else:
            model = solution.GraphLayer().double()
            assert isinstance(model, torch.nn.Module)
            registered = dict(model.named_parameters())
            assert set(registered) == set(PARAMETERS), (
                "register exactly the three specified parameters"
            )
            with torch.no_grad():
                for name, value in params.items():
                    assert registered[name].shape == value.shape, name
                    registered[name].copy_(torch.tensor(value))
            a = torch.tensor(x, requires_grad=True)
            b = torch.tensor(vectors, requires_grad=True)
            s, r = torch.tensor(senders), torch.tensor(receivers)

            def apply(pp, xx, vv, ss, rr):
                return model(xx, vv, ss, rr, 4)

            p = registered
            actual = apply(p, a, b, s, r)
            grads = torch.autograd.grad(
                actual.square().sum(), [*(registered[k] for k in PARAMETERS), a, b]
            )
            grad_params = dict(zip(PARAMETERS, grads[:3]))
            grad_x, grad_vectors = grads[-2:]
        close(actual, expected)
        close(apply(p, a, b, s, r), expected)
        close(actual[3], np.zeros(4), exact=True)

        # Relabel source nodes and reorder edges independently.
        order = np.array([3, 0, 4, 1, 2])
        inverse = np.argsort(order)
        edge_order = np.array([6, 1, 4, 0, 3, 5, 2])
        if backend == "jax":
            new_s, new_r = map(
                jnp.asarray, (inverse[senders][edge_order], receivers[edge_order])
            )
        else:
            new_s, new_r = map(
                torch.tensor, (inverse[senders][edge_order], receivers[edge_order])
            )
        close(apply(p, a[order], b[edge_order], new_s, new_r), expected)
        destination_order = np.array([3, 1, 0, 2])
        destination_inverse = np.argsort(destination_order)
        cast = jnp.asarray if backend == "jax" else torch.tensor
        close(
            apply(p, a, b, s, cast(destination_inverse[receivers])),
            expected[destination_order],
        )
        if backend == "jax":
            resized = solution.apply(p, a[:4], b[:4], s[:4], r[:4], 6)
        else:
            resized = model(a[:4], b[:4], s[:4], r[:4], 6)
        close(
            resized,
            graph_reference(params, x[:4], vectors[:4], senders[:4], receivers[:4], 6),
        )
        angles = [0.31, -0.23, 0.41]
        din = cue.IrrepsAndLayout(irreps_in, cue.mul_ir).exp_map(angles, [])
        dout = cue.IrrepsAndLayout(irreps_out, cue.mul_ir).exp_map(angles, [])
        dv = cue.SO3(1).exp_map(angles, [])
        close(apply(p, a @ cast(din.T), b @ cast(dv.T), s, r), expected @ dout.T)

        # Check one informative derivative in every parameter/input against NumPy.
        gradients = {**grad_params, "features": grad_x, "vectors": grad_vectors}
        values = {**params, "features": x, "vectors": vectors}
        for name, gradient in gradients.items():
            gradient = array(gradient)
            assert np.isfinite(gradient).all() and np.max(np.abs(gradient)) > 1e-9, name
            index = np.unravel_index(np.argmax(np.abs(gradient)), gradient.shape)
            losses = []
            for sign in (-1, 1):
                shifted = {k: v.copy() for k, v in values.items()}
                shifted[name][index] += sign * 1e-5
                y = graph_reference(
                    {k: shifted[k] for k in PARAMETERS},
                    shifted["features"],
                    shifted["vectors"],
                    senders,
                    receivers,
                    4,
                )
                losses.append(np.sum(y**2))
            np.testing.assert_allclose(
                gradient[index], (losses[1] - losses[0]) / 2e-5, rtol=2e-5, atol=2e-7
            )
        # A real parameter update must retain the reference semantics.
        updated = {k: params[k] - 1e-3 * array(grad_params[k]) for k in PARAMETERS}
        if backend == "jax":
            after = apply({k: jnp.asarray(v) for k, v in updated.items()}, a, b, s, r)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer.zero_grad()
            for name in PARAMETERS:
                p[name].grad = grad_params[name].detach().clone()
            optimizer.step()
            after = apply(p, a, b, s, r)
            assert {k: id(v) for k, v in model.named_parameters()} == {
                k: id(v) for k, v in registered.items()
            }
        close(after, graph_reference(updated, x, vectors, senders, receivers, 4))
        assert not np.allclose(array(after), array(actual)), (
            "update did not affect output"
        )


def check_routing(solution, backend, path):
    import numpy as np

    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("cuequivariance") for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("cuequivariance")
    for count in (1, 3, 7):
        predictions = np.linspace(-0.7, 1.1, count).reshape(count, 1)
        targets = np.linspace(1.2, -0.1, count)
        expected = np.mean((predictions[:, 0] - targets) ** 2)
        expected_grad = 2 * (predictions - targets[:, None]) / count
        if backend == "jax":
            import jax
            import jax.numpy as jnp

            jax.config.update("jax_enable_x64", True)
            p, t = jnp.asarray(predictions), jnp.asarray(targets)
            value = jax.jit(solution.mse_loss)(p, t)
            gradient = jax.grad(solution.mse_loss)(p, t)
        else:
            import torch

            p = torch.tensor(predictions, requires_grad=True)
            value = solution.mse_loss(p, torch.tensor(targets))
            gradient = torch.autograd.grad(value, p)[0]
        close(value, expected)
        close(gradient, expected_grad)


def worker(backend, case, solution):
    try:
        module = load_solution(solution)
        if case == "layout":
            check_layout(module, backend)
        elif case == "graph":
            check_graph(module, backend)
        else:
            check_routing(module, backend, solution)
    except Exception as error:
        return {
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    return {"passed": True}


def evaluate(backend, case, solution, *, timeout=180, python=None):
    command = [
        python or sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        backend,
        "--case",
        case,
        "--solution",
        str(solution.resolve()),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        marker = "CUEQ_RESULT="
        if result.returncode or marker not in result.stdout:
            return {
                "passed": False,
                "error": f"verifier exit {result.returncode}",
                "output": (result.stdout + result.stderr)[-6000:],
            }
        return json.loads(result.stdout.rsplit(marker, 1)[1].splitlines()[0])
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"submission exceeded {timeout} seconds"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "torch"))
    parser.add_argument("--case", choices=("layout", "graph", "routing"))
    parser.add_argument("--solution", type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    harbor = args.backend is None
    if harbor:
        entry_path = Path(os.environ.get("HARBOR_ENTRY_JSON", "/tests/entry.json"))
        entry = json.loads(entry_path.read_text())
        backend, _, case, version = entry["id"].split("-")
        assert version == "v1" and backend in ("jax", "torch")
        assert case in ("layout", "graph", "routing")
        args.backend, args.case = backend, case
        args.solution = Path("/workspace/solution.py")
    elif args.case is None or args.solution is None:
        parser.error("--backend requires --case and --solution")
    if args.worker:
        print(
            "CUEQ_RESULT=" + json.dumps(worker(args.backend, args.case, args.solution))
        )
        return
    interpreter = (
        Path("/workspace/.venv/bin/python") if harbor else Path(sys.executable)
    )
    result = evaluate(
        args.backend,
        args.case,
        args.solution,
        timeout=args.timeout,
        python=str(interpreter),
    )
    print(json.dumps(result, indent=2))
    if harbor:
        reward = Path(
            os.environ.get("HARBOR_REWARD_JSON", "/logs/verifier/reward.json")
        )
        reward.parent.mkdir(parents=True, exist_ok=True)
        reward.write_text(
            json.dumps(
                {
                    "custom_metrics": {"numerical_success": float(result["passed"])},
                    "details": {"numerical_success": result},
                },
                indent=2,
            )
        )
    elif not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
