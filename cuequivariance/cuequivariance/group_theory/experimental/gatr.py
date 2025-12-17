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
import cuequivariance as cue


def gatr_linear(mul_in: int, mul_out: int) -> cue.SegmentedTensorProduct:
    """
    subsrcipts: ``weights[uv],input[iu],output[iv]``

    References:
      - https://arxiv.org/pdf/2305.18415
      - `Source code <https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/blob/3f967c978445648ef83d87190d32176f7fd91565/gatr/primitives/linear.py#L33-L43>`_

    Examples:
        >>> d = gatr_linear(32, 32)
        >>> d
        uv,iu,iv sizes=9216,512,512 num_segments=9,8,8 num_paths=12 i={1, 3} u=32 v=32
    """
    d = cue.SegmentedTensorProduct.from_subscripts("uv,iu,iv")

    for ope, mul in [(1, mul_in), (2, mul_out)]:
        one = d.add_segment(ope, (1, mul))
        e0 = d.add_segment(ope, (1, mul))
        ei = d.add_segment(ope, (3, mul))
        e0i = d.add_segment(ope, (3, mul))
        eij = d.add_segment(ope, (3, mul))
        e0ij = d.add_segment(ope, (3, mul))
        e123 = d.add_segment(ope, (1, mul))
        e0123 = d.add_segment(ope, (1, mul))

    for xs in [[one], [e0, ei], [e0i, eij], [e0ij, e123], [e0123]]:
        x, xs = xs[0], xs[1:]
        d.add_path(None, x, x, c=1)
        for y in xs:
            d.add_path(-1, y, y, c=1)

    d.add_path(None, one, e0, c=1)
    d.add_path(None, ei, e0i, c=1)
    d.add_path(None, eij, e0ij, c=1)
    d.add_path(None, e123, e0123, c=1)
    return d


def gatr_geometric_product() -> cue.SegmentedTensorProduct:
    """
    subsrcipts: ``input1[i],input2[j],output[k]+ijk``

    References:
      - https://arxiv.org/pdf/2305.18415
      - `Source code <https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/blob/3f967c978445648ef83d87190d32176f7fd91565/gatr/primitives/bilinear.py#L63>`_

    Examples:
        >>> d = gatr_geometric_product(); d
        i,j,k+ijk sizes=16,16,16 num_segments=8,8,8 num_paths=60 i={1, 3} j={1, 3} k={1, 3}

        >>> d = d.append_modes_to_all_operands("u", dict(u=32)); d
        iu,ju,ku+ijk sizes=512,512,512 num_segments=8,8,8 num_paths=60 i={1, 3} j={1, 3} k={1, 3} u=32

        >>> d = d.normalize_paths_for_operand(2); d.paths[-1]
        op0[7]*op1[6]*op2[1]*[[[-0.25]]]

        >>> d = d.flatten_coefficient_modes(); d
        u,u,u sizes=512,512,512 num_segments=16,16,16 num_paths=192 u=32
    """
    return cue.SegmentedTensorProduct.from_base64(
        """
    eJztV9FuozAQ/BXE6+ETNsaW7leiPPRSekerS6OmfYry72eWcTcxNimlTdWm0gpGZne8s7uYZJdvn35v
    Vw/t5nGb/8rytrgt7n60t3d5keX3m+bhan3dPVjs8m3z51+zJrfFQi6LrL9UkQueLvdFdt44d883V49/
    u6Bd3q6v21XTE5RF1lnn6O7SGaByBlg5A9TOAGtngMYZoHVGuxKvBFRgkMRQAxr4KvJVgBK+isJKhhqw
    Am8Pa0DNDBrpKEqyYuh3M0xmwVBRDhWgwhY99A6aHTTIKiIjB00MGlDyqq9kDw2gL2oPLaCGYs2l1pS6
    ZOh5DfNakNWUQw2oeFVhi5p4JUPva+BriMEASoQZboDhqhuuuuFSGyIrAS12s8RrAX1RLRfVUpgbVjeq
    q/vm5qZdtc9j7gb9Z7nsng2g48WFJrDDMlzwq58t7CT8sHQOGGjtmIOXQhaRihWpYBHk0Ofnb96ZTpT4
    8sFT0JRhBU6UhKJE3Cm+IBPajpyEjPEcFGpEqxgXK8bUJpQEHZgiVxzrTc9SSu3ksJkdmVaO5xzDcRpr
    SKR/B1qnjcTMfkwZ9ZhWEYoNtcaGNRy+1CsRETeITc1XbAzmn1EXdMpc3Hv3ylGcN02vDR229UTlA6p4
    6wc08fYkaM53Kn5dre/4bT+pZDhssaK94GfaCzp0iWqDlKZ9tGYFfx803y/fm0/UUdWiH1K/80hy8VwG
    5YlIf7PAaZmG/zmH8Ow1QErL/f4/i15xrQ==
        """
    )


def gatr_outer_product() -> cue.SegmentedTensorProduct:
    """
    subsrcipts: ``input1[i],input2[j],output[k]+ijk``

    References:
      - https://arxiv.org/pdf/2305.18415
      - `Source code <https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/blob/3f967c978445648ef83d87190d32176f7fd91565/gatr/primitives/bilinear.py#L88>`_

    Examples:
        >>> d = gatr_outer_product(); d
        i,j,k+ijk sizes=16,16,16 num_segments=8,8,8 num_paths=30 i={1, 3} j={1, 3} k={1, 3}
    """
    return cue.SegmentedTensorProduct.from_base64(
        """
    eJzVVMtugzAQ/BXka3GFbR5SfwVxSAlpnagJCukJ8e+1N0MKBjeqSBtFWpnReGZYLxYtaz5fm/Ko61PD
    XgKmw224e9LbHQsDdqir42q/tht5y5rq7aPakyzPRREG50XNLNgtujD4X595snp1eremlun9WpfVOSAK
    A1tWaJ7CFKA0BahMAcamABNTgKkpwMwUvZVyBaBEgqCEBDCFVpJWAgpoJdliQAWbpIQUMEGCogQFKKFV
    pCVBTIIYUEAQkzYFVNAmpE0AJdiU2BRQgM2IzQo74PJQbTa61JePYz7Pc1TYvQk0Niw0N4uFS/Tso9mu
    wru1M0ggbpzxTbkp3OflPjMfuCO39StnIRefF80TwtPUSEQtTXJ8PXre7xz4N01yMT/2yUl+6BF7GOIl
    rW+43+57G8i9OdwNcnO4N2jBjXqQO3GTeS8Y07IJ/8FtucsUbtjC8I82pouu+wK/DgJI
        """
    )
