.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

MACE Operations
===============

In this notebook, we will go through the blocks that make up the `MACE <https://github.com/ACEsuit/mace/tree/main>`_ architecture, and how they can be accelerated through the use of cuEquivariance

There are 4 operations in MACE that act on irreps:

- Channelwise tensor product
- Symmetric contraction
- Linear layers
- Indexed linear layers

Before we go through them one by one, a small remark about the structure of the irreps is necessary.

A note on layouts
-----------------

Let us consider a collection of 10 :math:`l=1` objects, or vectors.

In `e3nn`, this would be stored as a :math:`10\times3` tensor with the :math:`(x,y,z)` components of each vector contiguous. This is what we refer to as the `mul_ir` layout.

For performance reasons, in cuEquivariance we adopt the transpose of this layout, i.e. the same object would correspond to a :math:`3\times10` object, with all the :math:`x` terms contiguous.

In the following, we will use the preferred layout for each library, but it must be noted that the transposition operation can be quite expensive, so that adhering to the correct layout throughout your code will result in the best performance.

Since any `e3nn` operation can be performed in cuEquivariance, this should in general be always possible.

Let us now start by importing a few useful packages

.. jupyter-execute::

    import torch
    import numpy as np
    from typing import Tuple, List
    import time
    from e3nn import o3
    import cuequivariance as cue
    import cuequivariance_torch as cuet
    from cuequivariance.group_theory.experimental.e3nn import O3_e3nn
    
.. jupyter-execute:: 
    :hide-code:
    
    # CODE ADAPTED FROM https://github.com/ACEsuit/mace to compute reference
    def tp_out_irreps_with_instructions(
        irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
    ) -> Tuple[o3.Irreps, List]:
        trainable = True
    
        # Collect possible irreps and their instructions
        irreps_out_list: List[Tuple[int, o3.Irreps]] = []
        instructions = []
        for i, (mul, ir_in) in enumerate(irreps1):
            for j, (_, ir_edge) in enumerate(irreps2):
                for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                    if ir_out in target_irreps:
                        k = len(irreps_out_list)  # instruction index
                        irreps_out_list.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", trainable))
    
        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_out = o3.Irreps(irreps_out_list)
        irreps_out, permut, _ = irreps_out.sort()
    
        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, permut[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]
    
        instructions = sorted(instructions, key=lambda x: x[2])
    
        return irreps_out, instructions
    
    from typing import Dict, Optional, Union
    
    import opt_einsum_fx
    import torch
    import torch.fx
    from e3nn import o3
    from e3nn.util.codegen import CodeGenMixin
    from e3nn.util.jit import compile_mode
        
    BATCH_EXAMPLE = 10
    ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]
    
    import collections
    import itertools
    
    _TP = collections.namedtuple("_TP", "op, args")
    _INPUT = collections.namedtuple("_INPUT", "tensor, start, stop")
    
    
    def _wigner_nj(
        irrepss: List[o3.Irreps],
        normalization: str = "component",
        filter_ir_mid=None,
        dtype=None,
    ):
        irrepss = [o3.Irreps(irreps) for irreps in irrepss]
        if filter_ir_mid is not None:
            filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]
    
        if len(irrepss) == 1:
            (irreps,) = irrepss
            ret = []
            e = torch.eye(irreps.dim, dtype=dtype)
            i = 0
            for mul, ir in irreps:
                for _ in range(mul):
                    sl = slice(i, i + ir.dim)
                    ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                    i += ir.dim
            return ret
    
        *irrepss_left, irreps_right = irrepss
        ret = []
        for ir_left, path_left, C_left in _wigner_nj(
            irrepss_left,
            normalization=normalization,
            filter_ir_mid=filter_ir_mid,
            dtype=dtype,
        ):
            i = 0
            for mul, ir in irreps_right:
                for ir_out in ir_left * ir:
                    if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                        continue
    
                    C = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                    if normalization == "component":
                        C *= ir_out.dim**0.5
                    if normalization == "norm":
                        C *= ir_left.dim**0.5 * ir.dim**0.5
    
                    C = torch.einsum("jk,ijl->ikl", C_left.flatten(1), C)
                    C = C.reshape(
                        ir_out.dim, *(irreps.dim for irreps in irrepss_left), ir.dim
                    )
                    for u in range(mul):
                        E = torch.zeros(
                            ir_out.dim,
                            *(irreps.dim for irreps in irrepss_left),
                            irreps_right.dim,
                            dtype=dtype,
                        )
                        sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                        E[..., sl] = C
                        ret += [
                            (
                                ir_out,
                                _TP(
                                    op=(ir_left, ir, ir_out),
                                    args=(
                                        path_left,
                                        _INPUT(len(irrepss_left), sl.start, sl.stop),
                                    ),
                                ),
                                E,
                            )
                        ]
                i += mul * ir.dim
        return sorted(ret, key=lambda x: x[0])
    
    
    def U_matrix_real(
        irreps_in: Union[str, o3.Irreps],
        irreps_out: Union[str, o3.Irreps],
        correlation: int,
        normalization: str = "component",
        filter_ir_mid=None,
        dtype=None,
        use_nonsymmetric_product=False,
    ):
        irreps_out = o3.Irreps(irreps_out)
        irrepss = [o3.Irreps(irreps_in)] * correlation
    
        if correlation == 4:
            filter_ir_mid = [(i, 1 if i % 2 == 0 else -1) for i in range(12)]
        try:
            wigners = _wigner_nj(irrepss, normalization, filter_ir_mid, dtype)
        except NotImplementedError as e:
            raise NotImplementedError(
                "The requested Clebsch-Gordan coefficients are not implemented, please install cuequivariance; pip install cuequivariance"
            ) from e
    
        current_ir = wigners[0][0]
        out = []
        stack = torch.tensor([])
    
        for ir, _, base_o3 in wigners:
            if ir in irreps_out and ir == current_ir:
                stack = torch.cat((stack, base_o3.squeeze().unsqueeze(-1)), dim=-1)
                last_ir = current_ir
            elif ir in irreps_out and ir != current_ir:
                if len(stack) != 0:
                    out += [last_ir, stack]
                stack = base_o3.squeeze().unsqueeze(-1)
                current_ir, last_ir = ir, ir
            else:
                current_ir = ir
        try:
            out += [last_ir, stack]
        except:  # pylint: disable=bare-except
            first_dim = irreps_out.dim
            if first_dim != 1:
                size = [first_dim] + [o3.Irreps(irreps_in).dim] * correlation + [1]
            else:
                size = [o3.Irreps(irreps_in).dim] * correlation + [1]
            out = [str(irreps_out)[:-2], torch.zeros(size, dtype=dtype)]
        return out
    
    class SymmetricContraction(CodeGenMixin, torch.nn.Module):
        def __init__(
            self,
            irreps_in: o3.Irreps,
            irreps_out: o3.Irreps,
            correlation: Union[int, Dict[str, int]],
            irrep_normalization: str = "component",
            path_normalization: str = "element",
            use_reduced_cg: bool = False,
            internal_weights: Optional[bool] = None,
            shared_weights: Optional[bool] = None,
            num_elements: Optional[int] = None,
        ) -> None:
            super().__init__()
    
            if irrep_normalization is None:
                irrep_normalization = "component"
    
            if path_normalization is None:
                path_normalization = "element"
    
            assert irrep_normalization in ["component", "norm", "none"]
            assert path_normalization in ["element", "path", "none"]
    
            self.irreps_in = o3.Irreps(irreps_in)
            self.irreps_out = o3.Irreps(irreps_out)
    
            del irreps_in, irreps_out
    
            if not isinstance(correlation, tuple):
                corr = correlation
                correlation = {}
                for irrep_out in self.irreps_out:
                    correlation[irrep_out] = corr
    
            assert shared_weights or not internal_weights
    
            if internal_weights is None:
                internal_weights = True
    
            self.internal_weights = internal_weights
            self.shared_weights = shared_weights
    
            del internal_weights, shared_weights
    
            self.contractions = torch.nn.ModuleList()
            for irrep_out in self.irreps_out:
                self.contractions.append(
                    Contraction(
                        irreps_in=self.irreps_in,
                        irrep_out=o3.Irreps(str(irrep_out.ir)),
                        correlation=correlation[irrep_out],
                        internal_weights=self.internal_weights,
                        num_elements=num_elements,
                        weights=self.shared_weights,
                        use_reduced_cg=use_reduced_cg,
                    )
                )
    
        def forward(self, x: torch.Tensor, y: torch.Tensor):
            outs = [contraction(x, y) for contraction in self.contractions]
            return torch.cat(outs, dim=-1)
    
    
    class Contraction(torch.nn.Module):
        def __init__(
            self,
            irreps_in: o3.Irreps,
            irrep_out: o3.Irreps,
            correlation: int,
            internal_weights: bool = True,
            use_reduced_cg: bool = False,
            num_elements: Optional[int] = None,
            weights: Optional[torch.Tensor] = None,
        ) -> None:
            super().__init__()
    
            self.num_features = irreps_in.count((0, 1))
            self.coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
            self.correlation = correlation
            dtype = torch.get_default_dtype()
    
            path_weight = []
            for nu in range(1, correlation + 1):
                U_matrix = U_matrix_real(
                    irreps_in=self.coupling_irreps,
                    irreps_out=irrep_out,
                    correlation=nu,
                    dtype=dtype,
                )[-1]
                path_weight.append(not torch.equal(U_matrix, torch.zeros_like(U_matrix)))
                self.register_buffer(f"U_matrix_{nu}", U_matrix)
    
            # Tensor contraction equations
            self.contractions_weighting = torch.nn.ModuleList()
            self.contractions_features = torch.nn.ModuleList()
    
            # Create weight for product basis
            self.weights = torch.nn.ParameterList([])
    
            for i in range(correlation, 0, -1):
                # Shapes definying
                num_params = self.U_tensors(i).size()[-1]
                num_equivariance = 2 * irrep_out.lmax + 1
                num_ell = self.U_tensors(i).size()[-2]
    
                if i == correlation:
                    parse_subscript_main = (
                        [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                        + ["ik,ekc,bci,be -> bc"]
                        + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                    )
                    graph_module_main = torch.fx.symbolic_trace(
                        lambda x, y, w, z: torch.einsum(
                            "".join(parse_subscript_main), x, y, w, z
                        )
                    )
    
                    # Optimizing the contractions
                    self.graph_opt_main = opt_einsum_fx.optimize_einsums_full(
                        model=graph_module_main,
                        example_inputs=(
                            torch.randn(
                                [num_equivariance] + [num_ell] * i + [num_params]
                            ).squeeze(0),
                            torch.randn((num_elements, num_params, self.num_features)),
                            torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                            torch.randn((BATCH_EXAMPLE, num_elements)),
                        ),
                    )
                    # Parameters for the product basis
                    w = torch.nn.Parameter(
                        torch.randn((num_elements, num_params, self.num_features))
                        / num_params
                    )
                    self.weights_max = w
                else:
                    # Generate optimized contractions equations
                    parse_subscript_weighting = (
                        [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                        + ["k,ekc,be->bc"]
                        + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                    )
                    parse_subscript_features = (
                        ["bc"]
                        + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                        + ["i,bci->bc"]
                        + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                    )
    
                    # Symbolic tracing of contractions
                    graph_module_weighting = torch.fx.symbolic_trace(
                        lambda x, y, z: torch.einsum(
                            "".join(parse_subscript_weighting), x, y, z
                        )
                    )
                    graph_module_features = torch.fx.symbolic_trace(
                        lambda x, y: torch.einsum("".join(parse_subscript_features), x, y)
                    )
    
                    # Optimizing the contractions
                    graph_opt_weighting = opt_einsum_fx.optimize_einsums_full(
                        model=graph_module_weighting,
                        example_inputs=(
                            torch.randn(
                                [num_equivariance] + [num_ell] * i + [num_params]
                            ).squeeze(0),
                            torch.randn((num_elements, num_params, self.num_features)),
                            torch.randn((BATCH_EXAMPLE, num_elements)),
                        ),
                    )
                    graph_opt_features = opt_einsum_fx.optimize_einsums_full(
                        model=graph_module_features,
                        example_inputs=(
                            torch.randn(
                                [BATCH_EXAMPLE, self.num_features, num_equivariance]
                                + [num_ell] * i
                            ).squeeze(2),
                            torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                        ),
                    )
                    self.contractions_weighting.append(graph_opt_weighting)
                    self.contractions_features.append(graph_opt_features)
                    # Parameters for the product basis
                    w = torch.nn.Parameter(
                        torch.randn((num_elements, num_params, self.num_features))
                        / num_params
                    )
                    self.weights.append(w)
    
            for idx, keep in enumerate(path_weight):
                zero_flag = not keep
                if idx < correlation - 1:
                    if zero_flag:
                        self.weights[idx] = EmptyParam(self.weights[idx])
                    self.register_buffer(
                        f"weights_{idx}_zeroed",
                        torch.tensor(zero_flag, dtype=torch.bool),
                    )
                else:
                    if zero_flag:
                        self.weights_max = EmptyParam(self.weights_max)
                    self.register_buffer(
                        "weights_max_zeroed",
                        torch.tensor(zero_flag, dtype=torch.bool),
                    )
    
            if not internal_weights:
                self.weights = weights[:-1]
                self.weights_max = weights[-1]
    
        def forward(self, x: torch.Tensor, y: torch.Tensor):
    
            out = self.graph_opt_main(
                self.U_tensors(self.correlation),
                self.weights_max,
                x,
                y,
            )
            for i, (weight, contract_weights, contract_features) in enumerate(
                zip(self.weights, self.contractions_weighting, self.contractions_features)
            ):
                c_tensor = contract_weights(
                    self.U_tensors(self.correlation - i - 1),
                    weight,
                    y,
                )
                c_tensor = c_tensor + out
                out = contract_features(c_tensor, x)
    
            return out.view(out.shape[0], -1)
    
        def U_tensors(self, nu: int):
            return dict(self.named_buffers())[f"U_matrix_{nu}"]
    
    
    class EmptyParam(torch.nn.Parameter):
        def __new__(cls, data):  # pylint: disable=signature-differs
            zero = torch.zeros_like(data)
            return super().__new__(cls, zero, requires_grad=False)
    
        def requires_grad_(self):
            return self
           
           
Channelwise tensor product
--------------------------

This is the main operation performed on the edges in a MACE model, typically found in the `InteractionBlock` modules.

It consists in the tensor product between the features of each neighbor and the spherical harmonics representing the edge, but it is computed in a _"channel-wise"_ fashion, in the sense that the neighbor's channels are not mixed.

The original implementation in `e3nn` makes use of a custom tensor product (the following code is adapted from the MACE repository):

.. jupyter-execute::

    # Parameters
    multiplicity = 128
    num_nodes = 1000
    num_edges = 10000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    irreps_node_input = o3.Irreps(f"{multiplicity}x0e + {multiplicity}x1o")
    irreps_edge_attr = o3.Irreps("1x0e + 1x1o")
    target_irreps = irreps_edge_attr
    
    # Create the instructions
    irreps_mid, instructions = tp_out_irreps_with_instructions(
        irreps_node_input,
        irreps_edge_attr,
        target_irreps,
    )
    
    # Create the TP module
    conv_tp = o3.TensorProduct(
        irreps_node_input,
        irreps_edge_attr,
        irreps_mid,
        instructions=instructions,
        shared_weights=False,
        internal_weights=False
    ).to(device)
    
    # Create input tensors
    node_feats = torch.randn(num_nodes, irreps_node_input.dim, device=device, dtype=dtype)
    senders = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)
    receivers = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)
    edge_attrs = torch.randn(num_edges, irreps_edge_attr.dim, device=device, dtype=dtype)
    weights = torch.randn(num_edges, conv_tp.weight_numel, device=device, dtype=dtype)
    
    # Perform TP
    mji = conv_tp(
        node_feats[senders], edge_attrs, weights
    )  # [num_nodes, irreps]
    # Perform scatter
    m_tmp = torch.zeros(num_nodes, irreps_mid.dim, device=device, dtype=dtype)
    message = m_tmp.scatter_add(0, receivers.unsqueeze(-1).expand_as(mji), mji)
    # Output shape
    print("Output shape:", message.shape)
    
As you can see, besides the TensorProduct itself, this requires gathering all node features corresponding to the edges (`node_feats[senders]`), and scattering the output back to the correct nodes.

In cuEquivariance, not only we can perform the TP, but we can also perform the gather/scatter operations in a single call.
For this operation, we will use our `uniform_1d` kernel, since there is a single set of irreps in the `channelwise` structure.

Let's do this explicitly, then we will show a premade module just for this operation.

For more information abou buildingt the descriptor itself, you can refer to the definition of `cue.descriptors.channelwise_tensor_product`.

.. jupyter-execute::

    # Cue version of the irreps
    irreps_in1 = cue.Irreps("O3", irreps_node_input)
    irreps_in2 = cue.Irreps("O3", irreps_edge_attr)
    irreps_out = cue.Irreps("O3", target_irreps)
    # Defining the operation
    e = cue.descriptors.channelwise_tensor_product(
        irreps_in1, irreps_in2, irreps_out
    )
    # The TP itself:
    cue_tp = cuet.SegmentedPolynomial(
        e.polynomial,
        method="uniform_1d"
    ).to(device)
    
    # Transposing inputs layout:
    cue_node_feats = cuet.TransposeIrrepsLayout(
        irreps_in1,
        source=cue.mul_ir,
        target=cue.ir_mul,
        device=device,
        use_fallback=device=="cpu",
    )(node_feats)
    cue_edge_attrs = cuet.TransposeIrrepsLayout(
        irreps_in2,
        source=cue.mul_ir,
        target=cue.ir_mul,
        device=device,
        use_fallback=device=="cpu",
    )(edge_attrs)
    
    # Performing the TP
    cue_message = cue_tp(
        [weights, cue_node_feats, cue_edge_attrs],
        input_indices={1: senders}, # indices for cue_node_feats
        output_shapes={0: cue_node_feats}, # We only care about the first dimension being num_nodes
        output_indices={0: receivers}, # Indices for the output
    )
    print("Output shape:", cue_message[0].shape)
    
    # Transposing the output
    cue_message_transp = cuet.TransposeIrrepsLayout(
        e.outputs[0].irreps,
        source=cue.ir_mul,
        target=cue.mul_ir,
        device=device,
        use_fallback=device=="cpu",
    )(cue_message[0])
    # Comparing the result
    print("Results match:", torch.allclose(message, cue_message_transp, atol=1e-5))
    
Alternatively, we can use the premade function for this particular tensor product:

.. jupyter-execute::

    # Defining TP through the premade block
    cue_cw = cuet.ChannelWiseTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        layout=cue.ir_mul,
        shared_weights=False,
        internal_weights=False,
        device=device
    )
    
    # Performing the TP
    cue_cw_message = cue_cw(
        cue_node_feats,
        cue_edge_attrs,
        weights,
        indices_1=senders,
        indices_out=receivers,
        size_out=num_nodes
    )
    # Transposing
    cue_cw_message_transp = cuet.TransposeIrrepsLayout(
        e.outputs[0].irreps,
        source=cue.ir_mul,
        target=cue.mul_ir,
        device=device,
        use_fallback=device=="cpu",
    )(cue_cw_message)
    
    # Comparing the results
    print("Results match:", torch.allclose(message, cue_cw_message_transp, atol=1e-5))
    
We can also compare the speed of the two approaches (in their respective layouts):

.. jupyter-execute::

    throwaway = 10
    repetitions = 1000 if device=="cuda" else 10
    
    e3nn_times = []
    for _ in range(throwaway):
        mji = conv_tp(node_feats[senders], edge_attrs, weights)
        m_tmp = torch.zeros(num_nodes, irreps_mid.dim, device=device, dtype=dtype)
        message = m_tmp.scatter_add(0, receivers.unsqueeze(-1).expand_as(mji), mji)
    for _ in range(repetitions):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        mji = conv_tp(node_feats[senders], edge_attrs, weights)
        m_tmp = torch.zeros(num_nodes, irreps_mid.dim, device=device, dtype=dtype)
        message = m_tmp.scatter_add(0, receivers.unsqueeze(-1).expand_as(mji), mji)
        torch.cuda.synchronize()
        e3nn_times.append(time.perf_counter()-t1)
    
    cuet_times = []
    for _ in range(throwaway):
        cue_message = cue_tp(
            [weights, cue_node_feats, cue_edge_attrs],
            input_indices={1: senders},
            output_shapes={0: cue_node_feats},
            output_indices={0: receivers},
        )
    for _ in range(repetitions):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        cue_message = cue_tp(
            [weights, cue_node_feats, cue_edge_attrs],
            input_indices={1: senders},
            output_shapes={0: cue_node_feats},
            output_indices={0: receivers},
        )
        torch.cuda.synchronize()
        cuet_times.append(time.perf_counter()-t1)
    e3nn_avg = 1000*np.mean(e3nn_times)
    cuet_avg = 1000*np.mean(cuet_times)
    print(f"e3nn time: {e3nn_avg:.2} ms")
    print(f"Cuequivariance time: {cuet_avg:.2} ms")
    print(f"Speedup: {e3nn_avg/cuet_avg:.2}x")
    
Of course a true comparison would require to see the module used in a real model, and our kernels tend to have best performance for very large input sizes, but even from this simple example it is clear that cuEquivariance offers a very good speedup for this operation.

Of course the backwards and double-backward pass are also supported and accelerated, but they will not be shown in this example.

Symmetric Contraction
---------------------

The Symmetric Contraction is the most distinctive TP in MACE.
It consists of a tensor product with a single input that gets contracted with itself multiple times. It is typically used in the `EquivariantProductBasisBlock`.

As in the previous case, we will first consider the original MACE implementation:

.. jupyter-execute::

    # Parameters
    num_species = 10
    multiplicity = 128
    correlation = 3
    num_nodes = 1000
    dtype = torch.float32
    irreps_in = o3.Irreps(f"{multiplicity}x0e + {multiplicity}x1o + {multiplicity}x2e + {multiplicity}x3o")
    irreps_out = o3.Irreps(f"{multiplicity}x0e + {multiplicity}x1o")
    
    # Define operation
    sc = SymmetricContraction(
        irreps_in,
        irreps_out,
        correlation=correlation,
        num_elements=num_species
    ).to(dtype).to(device)
    
    # Create inputs
    node_feats = torch.randn(num_nodes, multiplicity, irreps_in.dim // multiplicity, device=device, dtype=dtype)
    species = torch.randint(0, num_species, (num_nodes,), device=device, dtype=torch.int64)
    species_1hot = torch.nn.functional.one_hot(species, num_species).to(dtype).to(device)
    
    # Perform operation
    out_feats = sc(node_feats, species_1hot)
    
    # Output shape
    print("Output shape:", out_feats.shape)
    
We can now perform the same operation using the corresponding cuEquivariance module (you can check the module definition to see the descriptor utilized inside).

While the original module needs a 1-hot version of the atomic species, we use the species index directly and can perform more efficient operations.

Please note that in order to match the weights used in the previous implementation we will need to manually manipulate the internal weights of the system.
In a native scenario, however, the weights can of course be used as they are.
We also need to use the `O3_e3nn` group for compatibility, but the standard `"O3"` would work for the general case.
    
.. jupyter-execute::

    cue_irreps_in = cue.Irreps(O3_e3nn, irreps_in)
    cue_irreps_out = cue.Irreps(O3_e3nn, irreps_out)
    
    # The SC module
    cue_sc = cuet.SymmetricContraction(
        cue_irreps_in,
        cue_irreps_out,
        contraction_degree=correlation,
        num_elements=num_species,
        layout_in=cue.ir_mul,
        layout_out=cue.ir_mul,
        original_mace=True,
        device=device,
        dtype=dtype,
    )
    # Modifying the weights by hand
    cue_sc.weight.data = torch.concatenate([x for x in sc.parameters()], dim=1)
    
    # The input in this case is close to the needed shape:
    cue_node_feats = torch.transpose(node_feats, 1, 2).flatten(1)
    
    cue_out_feats = cue_sc(cue_node_feats, species)
    
    print("Output shape:", cue_out_feats.shape)
    
    # Transposing the output
    cue_out_feats_transp = cuet.TransposeIrrepsLayout(
        cue_irreps_out,
        source=cue.ir_mul,
        target=cue.mul_ir,
        device=device,
        use_fallback=device=="cpu",
    )(cue_out_feats)
    # Comparing the result
    print("Results match:", torch.allclose(out_feats, cue_out_feats_transp, atol=1e-5))
    
Here too we can compare the speed of the two approaches:

.. jupyter-execute::

    throwaway = 10 if device=="cuda" else 1
    repetitions = 100 if device=="cuda" else 2
    
    e3nn_times = []
    for _ in range(throwaway):
        out_feats = sc(node_feats, species_1hot)
    for _ in range(repetitions):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out_feats = sc(node_feats, species_1hot)
        torch.cuda.synchronize()
        e3nn_times.append(time.perf_counter()-t1)
    
    cuet_times = []
    for _ in range(throwaway):
        cue_out_feats = cue_sc(cue_node_feats, species)
    for _ in range(repetitions):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        cue_out_feats = cue_sc(cue_node_feats, species)
        torch.cuda.synchronize()
        cuet_times.append(time.perf_counter()-t1)
    e3nn_avg = 1000*np.mean(e3nn_times)
    cuet_avg = 1000*np.mean(cuet_times)
    print(f"e3nn time: {e3nn_avg:.3} ms")
    print(f"Cuequivariance time: {cuet_avg:.3} ms")
    print(f"Speedup: {e3nn_avg/cuet_avg:.3}x")
    
Linear layers
-------------

The linear layers are the most basic `e3nn` operation, used in several blocks in MACE.

While we do not provide a large speedup for this operation, we can perform natively in the `ir_mul` layout, for use in a complete cuEquivariance pipeline.

Let us start again from the original implementation:

.. jupyter-execute::

    # Parameters
    multiplicity = 128
    num_nodes = 10000
    dtype = torch.float32
    irreps_in = o3.Irreps(f"{multiplicity}x0e + {multiplicity}x1o")
    irreps_out = o3.Irreps(f"{multiplicity}x0e + {multiplicity}x1o")
    
    # Define operation
    lin = o3.Linear(
        irreps_in,
        irreps_out,
    ).to(dtype).to(device)
    
    # Create inputs
    in_feats = torch.randn(num_nodes, irreps_in.dim, device=device, dtype=dtype)
    
    # Perform operation
    out_feats = lin(in_feats)
    
    # Output shape
    print("Output shape:", out_feats.shape)

And the equivalent cuEquivariance code:

.. jupyter-execute::

    cue_irreps_in = cue.Irreps("O3", irreps_in)
    cue_irreps_out = cue.Irreps("O3", irreps_out)
    
    # The linear module
    cue_lin = cuet.Linear(
        cue_irreps_in,
        cue_irreps_out,
        internal_weights=False,
        layout=cue.ir_mul,
        device=device,
        dtype=dtype,
    )
    
    # Transposing the input
    cue_in_feats = cuet.TransposeIrrepsLayout(
        cue_irreps_out,
        source=cue.mul_ir,
        target=cue.ir_mul,
        device=device,
        use_fallback=device=="cpu",
    )(in_feats)
    
    cue_out_feats = cue_lin(cue_in_feats, weight=lin.weight.unsqueeze(0))
    
    print("Output shape:", cue_out_feats.shape)
    
    # Transposing the output
    cue_out_feats_transp = cuet.TransposeIrrepsLayout(
        cue_irreps_out,
        source=cue.ir_mul,
        target=cue.mul_ir,
        device=device,
        use_fallback=device=="cpu",
    )(cue_out_feats)
    # Comparing the result
    print("Results match:", torch.allclose(out_feats, cue_out_feats_transp, atol=1e-5))
    
Here too the results match.

We can compare the speed, although the difference will not be large in this case.

.. jupyter-execute::

    throwaway = 10
    repetitions = 1000 if device=="cuda" else 10
    
    e3nn_times = []
    for _ in range(throwaway):
        out_feats = lin(in_feats)
    for _ in range(repetitions):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out_feats = lin(in_feats)
        torch.cuda.synchronize()
        e3nn_times.append(time.perf_counter()-t1)
    
    cuet_times = []
    for _ in range(throwaway):
        cue_lin(cue_in_feats, weight=lin.weight.unsqueeze(0))
    for _ in range(repetitions):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        cue_lin(cue_in_feats, weight=lin.weight.unsqueeze(0))
        torch.cuda.synchronize()
        cuet_times.append(time.perf_counter()-t1)
    e3nn_avg = 1000*np.mean(e3nn_times)
    cuet_avg = 1000*np.mean(cuet_times)
    print(f"e3nn time: {e3nn_avg:.3} ms")
    print(f"Cuequivariance time: {cuet_avg:.3} ms")
    print(f"Speedup: {e3nn_avg/cuet_avg:.3}x")
    
Skip_tp or Indexed Linear
-------------------------

The last operation is an operation used in MACE in the `InteractionBlock` and typically called `skip_tp`, as it is used as a skip connection.

However, in the context of cuEquivariance we will typically refer to this operation as *indexed linear*, as it consists of a linear operation where the weight matrix is indexed on the species of each input.

We will first present the original implementation, which makes use of an expensive `FullyConnectedTensorProduct`.

.. jupyter-execute::

    # Parameters
    num_species = 20
    multiplicity = 128
    num_nodes = 10000
    dtype = torch.float32
    irreps_in = o3.Irreps(f"{multiplicity}x0e + {multiplicity}x1o")
    attr_irreps = o3.Irreps(f"{num_species}x0e")
    irreps_out = o3.Irreps(f"{multiplicity}x0e + {multiplicity}x1o")
    
    # Define operation
    skip_tp = o3.FullyConnectedTensorProduct(
        irreps_in,
        attr_irreps,
        irreps_out,
    ).to(dtype).to(device)
    
    # Create inputs
    in_feats = torch.randn(num_nodes, irreps_in.dim, device=device, dtype=dtype)
    species = torch.randint(0, num_species, (num_nodes,), device=device, dtype=torch.int64)
    species, _ = torch.sort(species)
    species_1hot = torch.nn.functional.one_hot(species, num_species).to(dtype).to(device)
    
    # Perform operation
    out_feats = skip_tp(in_feats, species_1hot)
    
    # Output shape
    print("Output shape:", out_feats.shape)
    
We will now show the equivalent cuEquivariance implementation that makes use of a `Linear` block and its indexing capabilities.

We will show the use of two different backends: `naive` and `indexed_linear`.
While the first can work in any setting, the second can only be used when the atomic species are sorted. However, it offers much better performance.

.. jupyter-execute::

    cue_irreps_in = cue.Irreps("O3", irreps_in)
    cue_irreps_out = cue.Irreps("O3", irreps_out)
    
    # The linear module
    cue_lin = cuet.Linear(
        cue_irreps_in,
        cue_irreps_out,
        internal_weights=False,
        weight_classes=num_species,
        layout=cue.ir_mul,
        device=device,
        dtype=dtype,
        method='naive'
    )
    # The faster linear module
    cue_indexed_lin = cuet.Linear(
        cue_irreps_in,
        cue_irreps_out,
        internal_weights=False,
        weight_classes=num_species,
        layout=cue.ir_mul,
        device=device,
        dtype=dtype,
        method='indexed_linear' if device=="cuda" else "naive"
    )
    
    # Transposing the input
    cue_in_feats = cuet.TransposeIrrepsLayout(
        cue_irreps_out,
        source=cue.mul_ir,
        target=cue.ir_mul,
        device=device,
        use_fallback=device=="cpu",
    )(in_feats)
    
    # Rearranging the weights by hand
    cue_weight = skip_tp.weight.reshape(2*multiplicity, num_species, multiplicity
                    ).transpose(0,1).reshape(num_species, -1)/np.sqrt(num_species)
    # Performing the operation
    cue_out_feats = cue_lin(cue_in_feats, weight=cue_weight, weight_indices=species)
    
    print("Output shape:", cue_out_feats.shape)
    
    # Transposing the output
    cue_out_feats_transp = cuet.TransposeIrrepsLayout(
        cue_irreps_out,
        source=cue.ir_mul,
        target=cue.mul_ir,
        device=device,
        use_fallback=device=="cpu",
    )(cue_out_feats)
    # Comparing the result
    print("Results match:", torch.allclose(out_feats, cue_out_feats_transp, atol=1e-3))
    
    # Performing the operation with the other backend
    cue_out_feats = cue_indexed_lin(cue_in_feats, weight=cue_weight, weight_indices=species)
    
    print("Output shape:", cue_out_feats.shape)
    
    # Transposing the output
    cue_out_feats_transp = cuet.TransposeIrrepsLayout(
        cue_irreps_out,
        source=cue.ir_mul,
        target=cue.mul_ir,
        device=device,
        use_fallback=device=="cpu",
    )(cue_out_feats)
    # Comparing the result
    print("Results match:", torch.allclose(out_feats, cue_out_feats_transp, atol=1e-3))
    
And we can compare the speed for the two implementations:

.. jupyter-execute::

    throwaway = 10
    repetitions = 100 if device=="cuda" else 10
    
    e3nn_times = []
    for _ in range(throwaway):
        out_feats = skip_tp(in_feats, species_1hot)
    for _ in range(repetitions):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out_feats = skip_tp(in_feats, species_1hot)
        torch.cuda.synchronize()
        e3nn_times.append(time.perf_counter()-t1)
    
    cuet_times = []
    for _ in range(throwaway):
        cue_out_feats = cue_lin(cue_in_feats, weight=cue_weight, weight_indices=species)
    for _ in range(repetitions):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        cue_out_feats = cue_lin(cue_in_feats, weight=cue_weight, weight_indices=species)
        torch.cuda.synchronize()
        cuet_times.append(time.perf_counter()-t1)
    
    cuet_v2_times = []
    for _ in range(throwaway):
        cue_out_feats = cue_indexed_lin(cue_in_feats, weight=cue_weight, weight_indices=species)
    for _ in range(repetitions):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        cue_out_feats = cue_indexed_lin(cue_in_feats, weight=cue_weight, weight_indices=species)
        torch.cuda.synchronize()
        cuet_v2_times.append(time.perf_counter()-t1)
    
    e3nn_avg = 1000*np.mean(e3nn_times)
    cuet_avg = 1000*np.mean(cuet_times)
    cuet_v2_avg = 1000*np.mean(cuet_v2_times)
    print(f"e3nn time: {e3nn_avg:.3} ms")
    print(f"Cuequivariance naive time: {cuet_avg:.3} ms")
    print(f"Speedup: {e3nn_avg/cuet_avg:.3}x")
    print(f"Cuequivariance indexed linear time: {cuet_v2_avg:.3} ms")
    print(f"Speedup: {e3nn_avg/cuet_v2_avg:.3}x")
    
As you can see, by using the best kernel we can achieve a very good speedup also in this case.

By using all of these modules, it is possible to accelerate a model like MACE up to 10 times, depending on the model and input size.

These operations are supported by the `official implementation of MACE <https://github.com/ACEsuit/mace/tree/main>`_.
