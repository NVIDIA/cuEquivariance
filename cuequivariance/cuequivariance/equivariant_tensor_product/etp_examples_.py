# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import itertools

import cuequivariance as cue
import cuequivariance.segmented_tensor_product as stp
import cuequivariance.equivariant_tensor_product as etp
from typing import *


def etp_examples(reduced_examples: bool = False) -> dict[str, dict[str, Any]]:
    output = {}

    def add_etp(
        name: str,
        batch: list[int],  # one per operand
        dtype: str,
        e: etp.EquivariantTensorProduct,
    ):
        assert len(batch) == e.num_operands

        def format_int(i: int) -> str:
            if i >= 1_000_000:
                return f"{i // 1_000_000}M"
            if i >= 1_000:
                return f"{i // 1000}k"
            return f"{i}"

        name = f"{name}-{format_int(max(batch))}-{dtype}"

        output[name] = {
            "ETP": e,
            "batch": batch,
            "dtypes": ((dtype,) * e.num_operands, dtype),
        }

    for lmax in [2, 3, 4]:
        for dtype in ["fp32", "fp64"]:
            add_etp(
                f"spherical-harmonics-{lmax}",
                [10_000, 10_000],
                dtype,
                etp.spherical_harmonics(cue.O3(1, -1), range(1, lmax + 1)),
                # TODO: where we start at 1 because our kernel doesn't support l=0, we should change this for simplicity
            )

    def execute(fn):
        fn()

    @execute
    def diffdock():
        def f(ns, nv):
            irreps_feat = cue.Irreps("O3", f"{ns}x0e+{nv}x1o+{nv}x1e+{ns}x0o")
            irreps_sh = cue.Irreps("O3", "0e + 1o + 2e")

            e = (
                etp.fully_connected_tensor_product(irreps_feat, irreps_sh, irreps_feat)
                .squeeze_modes()
                .canonicalize_subscripts()
            )
            yield "irmul", e
            assert e.d.subscripts == "uv,iu,j,kv+ijk"
            yield "mulir", e.change_layout(cue.mul_ir)
            yield "irmul-flatten", e.flatten_coefficient_modes()

        batch_sizes = (
            [10_000] if reduced_examples else [10, 100, 1_000, 5_000, 10_000, 25_000]
        )
        for lb in batch_sizes:
            dtype = "fp32"
            batch = [lb] * 4

            for layout, e in f(16, 4):
                add_etp(
                    f"diffdock-tensor-product-16-4-{layout}",
                    batch,
                    dtype,
                    e,
                )
            for layout, e in f(48, 10):
                add_etp(
                    f"diffdock-tensor-product-48-10-{layout}",
                    batch,
                    dtype,
                    e,
                )

    @execute
    def mace_off23():
        def add(
            name: str,
            batches: list[int],
            e: etp.EquivariantTensorProduct,
            shared_weights: bool = False,
        ):
            for batch in batches:
                Zs = [batch] * e.num_operands
                if shared_weights:
                    Zs[0] = 1

                for dtype in ["fp32", "fp64"]:
                    add_etp(name, Zs, dtype, e)

        # let's keep num_nodes and num_edges similar for now for easier benchmarking
        if reduced_examples:
            num_nodes = [10_000]
            num_edges = [10_000]
        else:
            num_nodes = [10, 100, 1_000, 5_000, 10_000, 25_000]
            num_edges = [10, 100, 1_000, 5_000, 10_000, 25_000]

        sizes = (
            ["small", "medium", "large"]
            if reduced_examples
            else ["xxsmall", "xsmall", "small", "medium", "large"]
        )

        for size in sizes:
            mul = {
                "xxsmall": 32,
                "xsmall": 64,
                "small": 96,
                "medium": 128,
                "large": 192,
            }[size]

            lmax = {
                "xxsmall": 0,
                "xsmall": 0,
                "small": 0,
                "medium": 1,
                "large": 2,
            }[size]

            irreps_hid = cue.Irreps("O3", "0e + 1o + 2e")[: lmax + 1]
            irreps_int = cue.Irreps("O3", "0e + 1o + 2e + 3o")

            e = etp.channelwise_tensor_product(mul * irreps_hid, irreps_int, irreps_int)
            e = e.squeeze_modes("v")
            assert e.d.subscripts == "u,iu,j,ku+ijk"
            add(f"mace-off23-tensor-product-{size}-irmul", num_edges, e)
            add(
                f"mace-off23-tensor-product-{size}-irmul-flatten",
                num_edges,
                e.flatten_coefficient_modes(),
            )
            add(
                f"mace-off23-tensor-product-{size}-mulir",
                num_edges,
                e.change_layout(cue.mul_ir),
            )

            irreps_out = e.output.irreps
            e = etp.linear(irreps_out, mul * irreps_int)
            add(f"mace-off23-linear-{size}-irmul", num_nodes, e, True)
            add(
                f"mace-off23-linear-{size}-mulir",
                num_nodes,
                e.change_layout(cue.mul_ir),
                True,
            )
            add(
                f"mace-off23-linear-{size}-irmul-flatten",
                num_nodes,
                e.flatten_modes("i"),
                True,
            )

            e = etp.linear(mul * irreps_hid, mul * irreps_hid)
            add(
                f"mace-off23-skip-linear-{size}-irmul",
                num_nodes,
                e.squeeze_modes(),
                True,
            )
            add(
                f"mace-off23-skip-linear-{size}-mulir",
                num_nodes,
                e.change_layout(cue.mul_ir).squeeze_modes(),
                True,
            )
            add(
                f"mace-off23-skip-linear-{size}-irmul-flatten",
                num_nodes,
                e.flatten_modes("i").squeeze_modes(),
                True,
            )

            e = etp.symmetric_contraction(mul * irreps_int, mul * irreps_hid, [1, 2, 3])
            add(
                f"mace-off23-symmetric-contraction-{size}-irmul-flatten",
                num_nodes,
                e,
                True,  # indexed weights ~ shared weights
            )

    @execute
    def escn():
        # baseline super slow, don't choose batchsize too large for now
        muls = [64] if reduced_examples else [32, 64, 128]
        lmaxs = [6] if reduced_examples else [6, 8]
        batch_sizes = [10_000] if reduced_examples else [10, 100, 1_000, 5_000, 10_000]
        for mul in muls:
            for lmax in lmaxs:
                irreps = mul * cue.Irreps(
                    "SO3", itertools.islice(cue.SO3.iterator(), lmax + 1)
                )
                e_tp = etp.escn_tp(irreps, irreps, m_max=2)

                for batch in batch_sizes:
                    add_etp(
                        f"escn-tensor-product-{lmax}-{mul}-irmul-flatten",
                        [batch, batch, batch],
                        "fp32",
                        e_tp,
                    )

    output = {k: output[k] for k in sorted(output)}
    return output
