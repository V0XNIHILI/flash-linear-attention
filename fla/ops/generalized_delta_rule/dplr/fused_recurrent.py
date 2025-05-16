# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import div, exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard, use_cuda_graph

logger = logging.getLogger(__name__)


@triton.jit
def fused_dplr_step(
    p_q,
    p_k,
    p_v,
    p_a,
    p_b,
    p_gk,
    p_o,
    b_h,
    mask_k,
    mask_v,
    scale: tl.constexpr,
    REVERSE: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    SAVE_TMP: tl.constexpr,
    sa,
):
    b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
    b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
    b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
    b_b = tl.load(p_b, mask=mask_k, other=0).to(tl.float32)
    b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

    # b_h [BV, BK], b_a[None, :] [:, BK] -> tmp [BV]
    b_sa = tl.sum(b_h * b_a[None, :], axis=1)
    if SAVE_TMP:
        tl.store(sa, b_sa.to(sa.dtype.element_ty), mask=mask_v)
    b_h = exp(b_gk)[None, :] * b_h + (b_sa[:, None] * b_b[None, :] + b_k[None, :] * b_v[:, None])
    b_o = tl.sum(b_h * b_q[None, :], axis=1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
    p_q += (-1 if REVERSE else 1) * H*K
    p_k += (-1 if REVERSE else 1) * H*K
    p_a += (-1 if REVERSE else 1) * H*K
    p_b += (-1 if REVERSE else 1) * H*K
    p_gk += (-1 if REVERSE else 1) * H*K
    p_v += (-1 if REVERSE else 1) * H*V
    p_o += (-1 if REVERSE else 1) * H*V
    return p_q, p_k, p_v, p_a, p_b, p_gk, p_o, b_h


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [16, 32, 64]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=['BK'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_dplr_delta_rule_fwd_kernel(
    q,
    k,
    v,
    a,
    b,
    gk,
    o,
    h0,
    ht,
    hckpt,
    cu_seqlens,
    scale,
    T,
    NUM_CKPT,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SAVE_CKPT: tl.constexpr,
    SAVE_CKPT_T: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0).to(tl.int32), tl.program_id(1).to(tl.int32)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_a = a + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_b = b + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_gk = gk + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    p_o = o + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[None, :] & mask_v[:, None]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[None, :] * V + o_v[:, None]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    if SAVE_CKPT:
        p_sa = sa + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
        for ckpt_idx in range(0, NUM_CKPT):
            for _ in range(0, SAVE_CKPT_T):
                p_q, p_k, p_v, p_a, p_b, p_gk, p_o, b_h = fused_dplr_step(
                    p_q, p_k, p_v, p_a, p_b, p_gk,
                    p_o, b_h, mask_k, mask_v, scale, REVERSE, H, K, V, SAVE_CKPT, p_sa
                )
                p_sa += (-1 if REVERSE else 1) * H*V
            p_hckpt = hckpt + i_n * H * NUM_CKPT * K*V + i_h * NUM_CKPT * \
                K*V + ckpt_idx * K*V + o_k[None, :] * V + o_v[:, None]
            tl.store(p_hckpt, b_h.to(p_hckpt.dtype.element_ty), mask=mask_h)
        # deal with leftover steps
        for _ in range(NUM_CKPT*SAVE_CKPT_T, T):
            p_q, p_k, p_v, p_a, p_b, p_gk, p_o, b_h = fused_dplr_step(
                p_q, p_k, p_v, p_a, p_b, p_gk,
                p_o, b_h, mask_k, mask_v, scale, REVERSE, H, K, V, SAVE_CKPT, p_sa
            )
        p_hckpt = hckpt + i_n * H * NUM_CKPT * K*V + i_h * NUM_CKPT * \
            K*V + ckpt_idx * K*V + o_k[None, :] * V + o_v[:, None]
    else:
        for _ in range(0, T):
            p_q, p_k, p_v, p_a, p_b, p_gk, p_o, b_h = fused_dplr_step(
                p_q, p_k, p_v, p_a, p_b, p_gk,
                p_o, b_h, mask_k, mask_v, scale, REVERSE, H, K, V, False, None
            )

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + o_k[None, :] * V + o_v[:, None]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_dplr_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: Optional[float] = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    SAVE_CKPT: bool = False,
    SAVE_CKPT_T: int = 16,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)

    h0 = initial_state
    if output_final_state:
        ht = q.new_empty(N, H, K, V, dtype=torch.float32)
    else:
        ht = None

    if SAVE_CKPT:
        # calculate the number of checkpoints
        # Save checkpoints at timesteps: SAVE_CKPT_T, 2*SAVE_CKPT_T, ...,
        # and do not save the last timestep
        num_ckpt = triton.cdiv(T, SAVE_CKPT_T) - 1
        hckpt = q.new_empty(N, H, num_ckpt, K, V, dtype=torch.float32)
        sa = q.new_empty(N, T, H, V, dtype=torch.float32)
    else:
        num_ckpt = 0
        hckpt = None
        sa = None

    o = torch.empty_like(v)

    def grid(meta): return (triton.cdiv(V, meta['BV']), N * H)
    fused_recurrent_dplr_delta_rule_fwd_kernel[grid](
        q,
        k,
        v,
        a,
        b,
        gk,
        o,
        h0,
        ht,
        sa,
        hckpt,
        cu_seqlens,
        scale,
        T=T,
        NUM_CKPT=num_ckpt,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        REVERSE=reverse,
        SAVE_CKPT=SAVE_CKPT,
        SAVE_CKPT_T=SAVE_CKPT_T,
    )
    return o, ht, sa, hckpt, num_ckpt


class FusedRecurrentDPLRDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        gk: torch.Tensor,
        scale: Optional[float] = 1.0,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        reverse: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        training: bool = False,
        ckpt_steps: int = 16,
    ):
        o, ht, sa, hckpt, num_ckpt = fused_recurrent_dplr_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            gk=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            SAVE_CKPT=training,
            SAVE_CKPT_T=ckpt_steps,
        )
        if training:
            ctx.save_for_backward(q, k, v, a, b, gk, ht, sa, hckpt)
            ctx.scale = scale
            ctx.reverse = reverse
            ctx.cu_seqlens = cu_seqlens
            ctx.has_initial_state = initial_state is not None
            ctx.has_final_state = output_final_state
            ctx.num_ckpt = num_ckpt
            ctx.ckpt_steps = ckpt_steps
        return o, ht

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Backward pass for fused_recurrent_dplr_delta_rule is not implemented and will not be supported. "
            "This kernel is only for inference. "
            "For training, please use `chunk_dplr_delta_rule`."
        )


def fused_recurrent_dplr_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: Optional[float] = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    training: bool = False,
    ckpt_steps: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    This function computes the recurrence S_t = S_t @ (D_t + a_t b_t^T) + v_t k_t^T in a recurrent manner.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        a (torch.Tensor):
            a of shape `[B, T, H, K]`.
        b (torch.Tensor):
            b of shape `[B, T, H, K]`.
        gk (torch.Tensor):
            gk of shape `[B, T, H, K]`. decay term in log space!
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: 1.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        reverse (Optional[bool]):
            If `True`, process the state passing in reverse order. Default: `False`.
        cu_seqlens (Optional[torch.Tensor]):
            Cumulative sequence lengths of shape `[N + 1]` used for variable-length training,
            consistent with the FlashAttention API.
        training (Optional[bool]):
            Whether to use the training mode. Default: `False`.
        ckpt_steps (Optional[int]):
            Number of steps to checkpoint the intermediate states.
            This is only used when `training=True`. Default: 16.
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = q.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    if training and output_final_state == False:
        logger.warning("output_final_state must be True during training")
        output_final_state = True

    o, final_state = FusedRecurrentDPLRDeltaRuleFunction.apply(
        q,
        k,
        v,
        a,
        b,
        gk,
        scale,
        initial_state,
        output_final_state,
        reverse,
        cu_seqlens,
        training,
        ckpt_steps,
    )
    return o, final_state
