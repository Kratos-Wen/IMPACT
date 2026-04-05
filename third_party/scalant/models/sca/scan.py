# Copyright Zeyun Zhong (zeyun.zhong@kit.edu)

import triton
import triton.language as tl
import torch
from torch.autograd import Function
import functools


@triton.jit
def fwd_sequential_scan(
        gk,  # gating tensor [B, H, T, K] or [B, H, T, K, V]
        kv,  # key/value tensor [B, H, T, K, V]
        states,  # output states [B, H, T, K, V]
        B,  # batch
        H,  # head
        T,  # sequence length
        K,  # key dimension
        V,  # value dimension
        BK: tl.constexpr,  # Size of each block of computation along K
        BV: tl.constexpr,  # Size of each block of computation along V
        HAS_V_DIM: tl.constexpr,  # Flag for gk's V dimension
):
    # Compute indices for BH, K, V
    i_v, i_k, i_bh = tl.program_id(2).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(0).to(tl.int64)

    p_k = tl.arange(0, BK) + i_k * BK
    p_v = tl.arange(0, BV) + i_v * BV

    # Mask handling
    mask_k = p_k < K
    mask_v = p_v < V
    mask = mask_k[:, None] & mask_v[None, :]

    s = tl.zeros([BK, BV], dtype=tl.float32)

    # Sequentially update state across the sequence length
    for i in range(T):
        offset = i * K * V
        i_ptr = i_bh * T * K * V + offset + (p_k[:, None] * V) + p_v[None, :]

        # Load gk with shape-dependent indexing
        if HAS_V_DIM:
            gk_ptr = i_ptr  # [B, H, T, K, V] case
            gk_val = tl.load(gk + gk_ptr, mask=mask, other=0.0).to(tl.float32)
            gk_val = tl.exp(gk_val)
        else:
            gk_ptr = i_bh * T * K + i * K + p_k  # [B, H, T, K] case
            gk_val = tl.load(gk + gk_ptr, mask=mask_k, other=0.0).to(tl.float32)
            gk_val = tl.exp(gk_val[:, None])

        kv_val = tl.load(kv + i_ptr, mask=mask, other=0.0).to(tl.float32)

        # Update the state
        s = gk_val * s + kv_val

        # Store the state
        tl.store(states + i_ptr, s, mask=mask)


@triton.jit
def bwd_sequential_scan(
    grad_states,  # gradient of the output states [B, H, T, K, V]
    gk,           # gating tensor from forward pass [B, H, T, K]
    states,       # States [B, H, T, K, V]
    grad_gk,      # gradient w.r.t gating tensor [B, H, T, K]
    grad_kv,      # gradient w.r.t key/value tensor [B, H, T, K, V]
    B, H, T, K, V,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_V_DIM: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(2).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(0).to(tl.int64)

    p_k = tl.arange(0, BK) + i_k * BK
    p_v = tl.arange(0, BV) + i_v * BV

    mask_k = p_k < K
    mask_v = p_v < V
    mask = mask_k[:, None] & mask_v[None, :]

    grad_s = tl.zeros([BK, BV], dtype=tl.float32)  # Initialize the cumulative gradient of states

    # Iterate backwards through the sequence
    for i in range(T-1, -1, -1):
        offset = i * K * V
        i_ptr = i_bh * T * K * V + offset + (p_k[:, None] * V) + p_v[None, :]

        grad_cur = tl.load(grad_states + i_ptr, mask=mask, other=0.0).to(tl.float32)
        grad_s += grad_cur
        tl.store(grad_kv + i_ptr, grad_s, mask=mask)

        # Load gk with shape-dependent indexing
        if HAS_V_DIM:
            gk_ptr = i_ptr
            gk_val = tl.load(gk + gk_ptr, mask=mask, other=0.0).to(tl.float32)
            gk_val = tl.exp(gk_val)
        else:
            gk_ptr = i_bh * T * K + i * K + p_k
            gk_val = tl.load(gk + gk_ptr, mask=mask_k, other=0.0).to(tl.float32)
            gk_val = tl.exp(gk_val[:, None])

        # Update gradients for gk
        if i > 0:
            prev_ptr = i_ptr - K * V
            state_last_step = tl.load(states + prev_ptr, mask=mask, other=0.0).to(tl.float32)
        else:
            state_last_step = tl.zeros([BK, BV], dtype=tl.float32)

        grad_gk_val = grad_s * state_last_step * gk_val
        if HAS_V_DIM:
            tl.store(grad_gk + i_ptr, grad_gk_val, mask=mask)
        else:
            # Sum over V dimension for 4D case
            grad_gk_ptr = i_bh * T * K + i * K + p_k
            tl.store(grad_gk + grad_gk_ptr, tl.sum(grad_gk_val, axis=1), mask=mask_k)

        grad_s *= gk_val


def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx,
                  *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                  **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper


class SequentialScan(Function):
    @staticmethod
    @contiguous
    @torch.cuda.amp.custom_fwd
    def forward(ctx, gk, kv):
        B, H, T, K, V = kv.shape
        num_warps = 8
        has_v_dim = gk.ndim == 5
        if has_v_dim:
            assert gk.shape[-1] == V, f"Linear Recurrence only accepts last dimension of {V}, while you have {gk.shape[-1]}"

        BK, BV = min(K, 64), min(V, 64)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

        states = kv.new_empty(B, H, T, K, V, dtype=torch.float32)

        grid = (B * H, NK, NV)
        fwd_sequential_scan[grid](
            gk, kv, states, B, H, T, K, V, BK, BV, has_v_dim,
            num_warps=num_warps
        )

        ctx.save_for_backward(gk, states)
        ctx.has_v_dim = has_v_dim
        return states.to(kv.dtype)

    @staticmethod
    @contiguous
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        gk, states = ctx.saved_tensors
        B, H, T, K, V = states.shape
        num_warps = 8
        has_v_dim = ctx.has_v_dim

        BK, BV = min(K, 64), min(V, 64)
        if not has_v_dim:
            BV = V

        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

        grad_gk = gk.new_empty(*gk.shape, dtype=torch.float32)
        grad_kv = gk.new_empty(B, H, T, K, V, dtype=torch.float32)

        grid = (B * H, NK, NV)
        bwd_sequential_scan[grid](
            grad_output, gk, states, grad_gk, grad_kv, B, H, T, K, V, BK, BV,
            has_v_dim,
            num_warps=num_warps
        )
        return grad_gk.to(gk.dtype), grad_kv.to(gk.dtype)


def sequential_scan(
        gk,  # gating tensor [B, H, T, K] or [B, H, T, K, V]
        kv,
):
    return SequentialScan.apply(gk, kv)