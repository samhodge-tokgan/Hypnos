"""Export-friendly reimplementation of MEMatte's adaptive token routing.

Derived from MEMatte (https://github.com/linyiheng123/MEMatte), MIT licensed --
see the NOTICE file. `reference_routing` is a transcription of upstream's eval
path; the rest is a rewrite of it for ONNX export.

Why this file exists
--------------------
MEMatte's *inference* routing path cannot be exported to ONNX. In upstream
`modeling/backbone/vit.py` the eval branch does three untraceable things:

  1. ``hard_keep_decision.sum().item() > self.max_number_token``
     — a host synchronisation plus a data-dependent Python branch.
  2. ``x.reshape(B, -1, C)[selected_indices]``
     — boolean-mask indexing, whose output length depends on the DATA, so the
       traced graph would bake in whatever count the example input happened to
       produce.
  3. ``msa.masked_scatter_(...)``
     — no ONNX equivalent.

The fix and the feature are the same thing. Upstream *already* truncates to the
top-K tokens by ``pred_score[...,0] - pred_score[...,1]`` whenever more than
``max_number_token`` tokens pass the router (vit.py:590-597). Making that
truncation unconditional turns the variable-length gather into a fixed-length
``topk -> gather -> attend -> scatter``, which exports cleanly *and* is exactly
the user-facing token cap that bounds memory.

Equivalence argument
--------------------
Let S be the set the reference selects (``score > 0``) and K the cap.

* |S| > K — upstream itself keeps only the top-K of S by score. Identical.
* |S| <= K — we gather K tokens (S plus the next-highest scorers) but mask the
  surplus out of the attention, so every genuinely-selected token attends over
  exactly S, and only genuinely-selected tokens are scattered back. Identical.

The mask keeps each token's own diagonal entry live (the same trick upstream's
``softmax_with_policy`` uses with its identity term), so no row can be fully
masked and the softmax can never produce NaN.

Two details matter for exactness:

* The global blocks — and only they are routed — are built with
  ``use_rel_pos=False`` and ``window_size=0`` (vit.py:521-523), so their
  attention is plain scaled dot-product with no positional term and no window
  partitioning. That is what ``masked_attention`` reproduces.
* We sort the selected indices into ascending order, matching the order boolean
  mask indexing produces upstream, so the reduction order inside the softmax and
  the ``attn @ v`` matmul lines up and the two paths agree to float noise rather
  than to a looser tolerance.

``tools/routing_equivalence_test.py`` checks all of this on random data and then
exports the rewrite to ONNX and re-runs it under ONNX Runtime.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_attention(attn, x, mask):
    """Plain multi-head attention with an additive boolean column mask.

    Mirrors ``Attention.forward`` for the case the routed (global) blocks
    actually use: ``x.ndim == 3``, ``use_rel_pos=False``, no window partition.

    Args:
        attn: the upstream ``Attention`` module (for qkv/proj/scale/num_heads).
        x: ``[B, N, C]``.
        mask: ``[B, 1, 1, N]`` or broadcastable; nonzero = attendable column.
    Returns:
        ``[B, N, C]``
    """
    B, N, _ = x.shape
    heads = attn.num_heads

    qkv = attn.qkv(x).reshape(B, N, 3, heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, B * heads, N, -1).unbind(0)

    a = (q * attn.scale) @ k.transpose(-2, -1)          # [B*heads, N, N]
    a = a.reshape(B, heads, N, N)
    # Masked columns get -inf so exp() gives an exact zero: a genuinely-selected
    # row's softmax is then bit-for-bit the softmax over S alone.
    a = a.masked_fill(mask == 0, float("-inf"))
    a = a.softmax(dim=-1).reshape(B * heads, N, N)

    out = (a @ v).view(B, heads, N, -1).permute(0, 2, 1, 3).reshape(B, N, -1)
    return attn.proj(out)


def build_mask(policy):
    """Column mask that keeps the diagonal live.

    ``policy`` is ``[B, K]`` with 1 for genuinely-selected tokens. Returns
    ``[B, 1, K, K]``: column j is attendable if it was selected, or if it is the
    row's own position. The diagonal term is what makes a fully-unselected row
    safe (it is upstream's ``eye`` trick in ``softmax_with_policy``).
    """
    B, K = policy.shape
    col = policy.reshape(B, 1, 1, K)
    eye = torch.eye(K, dtype=policy.dtype, device=policy.device).reshape(1, 1, K, K)
    return col + (1.0 - col) * eye


def export_routing(x_flat, pred_score, max_tokens, attn, fast_out):
    """Export-friendly routed global attention.

    Args:
        x_flat: ``[B, N, C]`` normalised tokens (i.e. after ``norm1``).
        pred_score: ``[B, N, 2]`` router logits.
        max_tokens: 0-d or 1-element int64 tensor — the runtime token cap.
        attn: the upstream ``Attention`` module.
        fast_out: ``[B, N, C]`` the LTRM (cheap path) result for every token.
    Returns:
        ``[B, N, C]`` with genuinely-selected positions replaced by the global
        attention result and all others left as ``fast_out``.
    """
    B, N, C = x_flat.shape
    score = pred_score[..., 0] - pred_score[..., 1]                    # [B, N]

    # The cap must be clamped to the ACTUAL token count, and N varies with the
    # plate resolution. Deriving it from the traced shape (a plain Python int)
    # would bake the trace-time resolution into the graph, so read it back as a
    # tensor: this becomes an ONNX Shape node and stays symbolic.
    n = torch._shape_as_tensor(x_flat)[1].to(torch.int64)
    k = torch.clamp(max_tokens.reshape(-1)[0].to(torch.int64), min=1)
    k = torch.minimum(k, n)

    _, idx = torch.topk(score, k, dim=1)                               # [B, k]
    # Ascending index order matches upstream's boolean-mask gather, keeping the
    # reduction order (and therefore the float result) aligned.
    idx, _ = torch.sort(idx, dim=1)

    sel_score = torch.gather(score, 1, idx)                            # [B, k]
    policy = (sel_score > 0).to(x_flat.dtype)                          # [B, k]

    idx_c = idx.unsqueeze(-1).expand(-1, -1, C)                        # [B, k, C]
    sel = torch.gather(x_flat, 1, idx_c)                               # [B, k, C]

    slow = masked_attention(attn, sel, build_mask(policy))             # [B, k, C]

    # Write back only where the router genuinely selected the token; elsewhere
    # keep the cheap path. torch.where (not arithmetic blending) so a masked
    # row's value can never contaminate the result.
    keep = policy.unsqueeze(-1) > 0                                    # [B, k, 1]
    fast_sel = torch.gather(fast_out, 1, idx_c)
    blended = torch.where(keep, slow, fast_sel)
    return fast_out.scatter(1, idx_c, blended)


def router_forward_export(self, x, policy=None):
    """Shape-explicit equivalent of ``Router.forward`` (backbone/utils.py:253).

    Upstream reads ``B, N, C = x.size()`` and then uses N arithmetically — to
    average the global branch (``.sum(dim=1) / N``) and to broadcast it
    (``global_x.expand(B, N, C // 2)``). That LOOKS like it would freeze the
    traced token count, but ``tools/shape_dynamics_test.py`` confirms it does
    not: the TorchScript tracer keeps values derived from ``aten::size``
    symbolic, so the stock module already exports correctly.

    This version makes the dependency explicit (a Shape node plus ``expand_as``)
    and is kept as belt-and-braces for the resolution-dynamic export. It is
    verified equivalent to the stock module in the same test. Use it or not; do
    not assume the stock one is broken.

    ``policy`` is accepted and ignored, exactly as upstream does.
    """
    del policy
    x = self.in_conv(x)
    C = x.shape[2]
    local_x = x[:, :, : C // 2]
    n = torch._shape_as_tensor(x)[1].to(x.dtype)
    global_x = x[:, :, C // 2:].sum(dim=1, keepdim=True) / n
    x = torch.cat([local_x, global_x.expand_as(local_x)], dim=-1)
    return self.out_conv(x)


# ---------------------------------------------------------------------------
# Reference implementation, transcribed from upstream's eval path. Used only by
# the equivalence test — it is deliberately NOT export-friendly.
# ---------------------------------------------------------------------------
def reference_routing(x_flat, pred_score, max_tokens, attn, fast_out):
    """Upstream ``ViT.forward`` + ``Block.forward`` eval behaviour."""
    B, N, C = x_flat.shape
    assert B == 1, "upstream's eval path assumes batch size 1"

    # vit.py:587-597 — argmax decision, then top-K truncation if it overflows.
    hard = torch.zeros_like(pred_score[..., :1])                       # [B, N, 1]
    hard[pred_score[..., 0] > pred_score[..., 1]] = 1.0
    if hard.sum().item() > max_tokens:
        _, sort_index = torch.sort(pred_score[..., 0] - pred_score[..., 1],
                                   descending=True)
        sort_index = sort_index[:, :max_tokens]
        hard = torch.zeros_like(pred_score[..., :1])
        hard[:, sort_index.squeeze(0), :] = 1.0

    # vit.py:400-407 — cheap path everywhere, global attention on the selected.
    out = fast_out.clone()
    selected = hard.squeeze(-1).bool()                                 # [B, N]
    if torch.any(selected):
        sel = x_flat.reshape(B, -1, C)[selected].unsqueeze(0)          # [1, s, C]
        # policy=None -> plain softmax (Attention.forward line 103-104).
        slow = _plain_attention(attn, sel)
        out.masked_scatter_(selected.unsqueeze(-1), slow)
    return out


def _plain_attention(attn, x):
    """``Attention.forward`` with policy=None, ndim==3, use_rel_pos=False."""
    B, N, _ = x.shape
    heads = attn.num_heads
    qkv = attn.qkv(x).reshape(B, N, 3, heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, B * heads, N, -1).unbind(0)
    a = (q * attn.scale) @ k.transpose(-2, -1)
    a = a.softmax(dim=-1)
    out = (a @ v).view(B, heads, N, -1).permute(0, 2, 1, 3).reshape(B, N, -1)
    return attn.proj(out)
