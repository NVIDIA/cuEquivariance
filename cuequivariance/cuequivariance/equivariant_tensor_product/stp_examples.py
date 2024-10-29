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


def examples(reduced_examples: bool = False) -> dict[str, dict[str, Any]]:
    output = {}

    def dump_descriptor(
        variant_name: str,
        batch: list[int],  # one per operand
        buffers: Optional[list[int]],
        dtype: str,
        d: stp.SegmentedTensorProduct,
    ):
        assert len(batch) == d.num_operands

        if buffers is None:
            buffers = list(range(d.num_operands))

        def format_int(i: int) -> str:
            if i >= 1_000_000:
                return f"{i // 1_000_000}M"
            if i >= 1_000:
                return f"{i // 1000}k"
            return f"{i}"

        name = f"{variant_name}-{format_int(max(batch))}-{dtype}"

        output[name] = {
            "STP": d,
            "batch": batch,
            "buffer_index": buffers,
            "dtypes": ((dtype,) * d.num_operands, dtype),
        }

    def execute(fn):
        fn()

    @execute
    def diffdock():
        def f(ns, nv):
            irreps_feat = cue.Irreps("O3", f"{ns}x0e+{nv}x1o+{nv}x1e+{ns}x0o")
            irreps_sh = cue.Irreps("O3", "0e + 1o + 2e")

            d = (
                etp.fully_connected_tensor_product(irreps_feat, irreps_sh, irreps_feat)
                .d.squeeze_modes()
                .canonicalize_subscripts()
            )
            yield "irmul", d
            assert d.subscripts == "uv,iu,j,kv+ijk"
            yield "mulir", d.add_or_transpose_modes("uv,ui,j,vk+ijk")
            yield "irmul-flatten", d.flatten_coefficient_modes()

        batch_sizes = (
            [10_000] if reduced_examples else [10, 100, 1_000, 5_000, 10_000, 25_000]
        )
        for lb in batch_sizes:
            dtype = "fp32"
            batch = [lb] * 4

            for layout, d in f(16, 4):
                dump_descriptor(
                    f"diffdock-tensor-product-16-4-{layout}",
                    batch,
                    None,
                    dtype,
                    d,
                )
            for layout, d in f(48, 10):
                dump_descriptor(
                    f"diffdock-tensor-product-48-10-{layout}",
                    batch,
                    None,
                    dtype,
                    d,
                )

    @execute
    def mace_off23():
        def dump(
            variant_name: str,
            batches: list[int],
            d: stp.SegmentedTensorProduct,
            buffers: Optional[list[int]] = None,
            shared_weights: bool = False,
        ):
            for batch in batches:
                Zs = [batch] * d.num_operands
                if shared_weights:
                    Zs[0] = 1

                for dtype in ["fp32", "fp64"]:
                    dump_descriptor(
                        variant_name,
                        Zs,
                        buffers,
                        dtype,
                        d,
                    )

        # let's keep num_nodes and num_edges similar for now for easier benchmarking
        if reduced_examples:
            num_nodes = [5_000]
            num_edges = [5_000]
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
            d, irreps_out = e.d, e.operands[-1].irreps
            d = d.squeeze_modes("v")
            assert d.subscripts == "u,iu,j,ku+ijk"
            dump(f"mace-off23-tensor-product-{size}-irmul", num_edges, d)
            dump(
                f"mace-off23-tensor-product-{size}-irmul-flatten",
                num_edges,
                d.flatten_coefficient_modes(),
            )
            d = d.add_or_transpose_modes("u,ui,j,uk+ijk")
            dump(f"mace-off23-tensor-product-{size}-mulir", num_edges, d)

            irreps_out = irreps_out.simplify()
            d = etp.linear(irreps_out, mul * irreps_int).d
            dump(f"mace-off23-linear-{size}-irmul", num_nodes, d, None, True)
            dump(
                f"mace-off23-linear-{size}-mulir",
                num_nodes,
                d.add_or_transpose_modes("uv,ui,vi"),
                None,
                True,
            )
            dump(
                f"mace-off23-linear-{size}-irmul-flatten",
                num_nodes,
                d.flatten_modes("i"),
                None,
                True,
            )

            d = etp.fully_connected_tensor_product(
                mul * irreps_hid, cue.Irreps("O3", "10x0e"), mul * irreps_hid
            ).d
            dump(
                f"mace-off23-skip-fctp-{size}-irmul",
                num_nodes,
                d.squeeze_modes(),
                None,
                True,
            )
            dump(
                f"mace-off23-skip-fctp-{size}-mulir",
                num_nodes,
                d.add_or_transpose_modes("uvw,ui,vj,kw+ijk").squeeze_modes(),
                None,
                True,
            )
            dump(
                f"mace-off23-skip-fctp-{size}-irmul-flatten",
                num_nodes,
                d.flatten_coefficient_modes().squeeze_modes(),
                None,
                True,
            )

            for nu in range(1, 3 + 1):
                [d] = etp.symmetric_contraction(
                    mul * irreps_int, mul * irreps_hid, [nu]
                ).ds
                d = d.sort_indices_for_identical_operands(range(1, nu + 1))

                dump(
                    f"mace-off23-symmetric-contraction-{size}-{nu}-irmul-flatten",
                    num_nodes,
                    d,
                    [0] + [1] * nu + [2],
                )

    @execute
    def escn():
        # baseline super slow, don't choose batchsize too large for now
        muls = [64] if reduced_examples else [32, 64, 128]
        lmaxs = [6] if reduced_examples else [6, 8]
        batch_sizes = [1_000] if reduced_examples else [10, 100, 1_000, 5_000, 10_000]
        for mul in muls:
            for lmax in lmaxs:
                irreps = mul * cue.Irreps(
                    "SO3", itertools.islice(cue.SO3.iterator(), lmax + 1)
                )
                d_tp = etp.escn_tp(irreps, irreps, 2).d
                d_rot = etp.yx_rotation(irreps).d

                for batch in batch_sizes:
                    dump_descriptor(
                        f"escn-tensor-product-{lmax}-{mul}-irmul-flatten",
                        [batch, batch, batch],
                        None,
                        "fp32",
                        d_tp,
                    )
                    dump_descriptor(
                        f"escn-rotation-{lmax}-{mul}-irmul-flatten",
                        [batch, batch, batch, batch],
                        None,
                        "fp32",
                        d_rot,
                    )

    output = {k: output[k] for k in sorted(output)}
    return output
