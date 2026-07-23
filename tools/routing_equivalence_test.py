#!/usr/bin/env python3
"""Verify the ONNX-export rewrite of MEMatte's token routing.

This is the test that substantiates the central design claim: replacing
upstream's data-dependent boolean-mask gather with a fixed-length
``topk -> gather -> attend -> scatter`` changes nothing about the result, while
making the graph exportable and making the token cap a real memory control.

Two things are checked:

  1. EQUIVALENCE — ``tools/mematte_routing.export_routing`` matches a direct
     transcription of upstream's eval path across every regime that matters:
     the cap above, at, and below the number of tokens the router selects, plus
     the degenerate all-selected / none-selected cases.

  2. EXPORTABILITY — the rewrite exports to ONNX with a *runtime* token cap and
     reproduces the PyTorch result under ONNX Runtime at token caps it was not
     traced with (which is the whole point of the dynamic-K design).

No MEMatte checkpoint or detectron2 install is needed: the routing logic is
exercised on a real ``Attention`` module with random weights, which is what the
rewrite actually touches.

Run:  python3 tools/routing_equivalence_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mematte_routing import export_routing, reference_routing  # noqa: E402

TOL = 1e-5


class Attention(nn.Module):
    """The upstream ``Attention`` module's parameters and scale.

    Reproduced rather than imported so the test needs no detectron2. Only the
    pieces the routed path uses exist here; the routed (global) blocks are built
    with ``use_rel_pos=False``, so there are no positional parameters.
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)


class RoutedBlock(nn.Module):
    """Minimal module wrapping the rewrite, so it can be exported to ONNX."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = Attention(dim, num_heads)

    def forward(self, x_flat, pred_score, max_tokens, fast_out):
        return export_routing(x_flat, pred_score, max_tokens, self.attn, fast_out)


def make_case(N, C, heads, n_selected, seed):
    """Build a case where exactly `n_selected` tokens have score > 0."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, N, C, generator=g)
    fast = torch.randn(1, N, C, generator=g)

    # Distinct, well-separated scores so the top-K choice is unambiguous and the
    # comparison is not testing tie-breaking behaviour.
    order = torch.randperm(N, generator=g)
    score = torch.empty(N)
    score[order] = torch.linspace(1.0, -1.0, N)
    if n_selected == 0:
        score = score.abs().neg() - 0.1
    elif n_selected == N:
        score = score.abs() + 0.1
    else:
        # Shift so exactly n_selected entries are strictly positive.
        s, _ = torch.sort(score, descending=True)
        score = score - (s[n_selected - 1] + s[n_selected]) / 2.0

    pred = torch.zeros(1, N, 2)
    pred[0, :, 0] = score
    assert int((pred[..., 0] > pred[..., 1]).sum()) == n_selected
    return x, pred, fast


def run_equivalence():
    torch.manual_seed(0)
    C, heads = 64, 4
    failures = 0
    print("--- equivalence: rewrite vs upstream eval path ---")
    for N, n_sel in [
        (64, 40),    # cap above selection (the common case)
        (64, 64),    # everything selected
        (64, 0),     # nothing selected (upstream skips attention entirely)
        (64, 1),     # a single token selected
        (128, 90),
        (100, 55),   # N not a power of two
    ]:
        for cap in [8, 16, n_sel if n_sel else 1, N, N * 2]:
            blk = RoutedBlock(C, heads).eval()
            x, pred, fast = make_case(N, C, heads, n_sel, seed=N * 1000 + cap)
            with torch.no_grad():
                got = export_routing(x, pred, torch.tensor([cap]), blk.attn, fast)
                want = reference_routing(x, pred, cap, blk.attn, fast)
            diff = (got - want).abs().max().item()
            ok = diff <= TOL
            failures += 0 if ok else 1
            print(f"  N={N:4d} selected={n_sel:4d} cap={cap:5d}  "
                  f"max|diff|={diff:.3e}  {'ok' if ok else 'FAIL'}")
    return failures


def run_onnx():
    """Export with a runtime token cap and check ORT reproduces PyTorch."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        print(f"--- onnx export: SKIPPED ({e}) ---")
        return 0

    print("--- onnx export: dynamic token cap ---")
    C, heads, N = 64, 4, 96
    blk = RoutedBlock(C, heads).eval()
    x, pred, fast = make_case(N, C, heads, 60, seed=7)
    trace_cap = torch.tensor([32], dtype=torch.int64)

    path = Path(__file__).resolve().parent.parent / "build" / "routing_test.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        blk,
        (x, pred, trace_cap, fast),
        str(path),
        input_names=["x", "pred_score", "max_tokens", "fast_out"],
        output_names=["out"],
        # N dynamic so one graph serves any resolution; max_tokens is a real
        # runtime input, which is what makes the cap a live control.
        dynamic_axes={"x": {1: "N"}, "pred_score": {1: "N"},
                      "fast_out": {1: "N"}, "out": {1: "N"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"  exported {path.name} ({path.stat().st_size / 1024:.0f} KB)")

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    failures = 0
    # Caps the graph was NOT traced with, including one above N.
    for cap in [8, 32, 60, 96, 200]:
        with torch.no_grad():
            want = export_routing(x, pred, torch.tensor([cap]), blk.attn, fast).numpy()
        got = sess.run(["out"], {
            "x": x.numpy(),
            "pred_score": pred.numpy(),
            "max_tokens": np.array([cap], dtype=np.int64),
            "fast_out": fast.numpy(),
        })[0]
        diff = float(np.abs(got - want).max())
        ok = diff <= TOL
        failures += 0 if ok else 1
        print(f"  cap={cap:4d} (traced at 32)  max|diff|={diff:.3e}  "
              f"{'ok' if ok else 'FAIL'}")

    # A DIFFERENT token count must also work: N is a dynamic axis, so the graph
    # has to derive it at runtime rather than bake in the traced 96. This is the
    # check that catches the token count being frozen at trace time.
    for n2, cap in [(48, 16), (160, 64), (160, 4096)]:
        x2, pred2, fast2 = make_case(n2, C, heads, n2 // 2, seed=n2 + cap)
        with torch.no_grad():
            want = reference_routing(x2, pred2, cap, blk.attn, fast2).numpy()
        got = sess.run(["out"], {
            "x": x2.numpy(), "pred_score": pred2.numpy(),
            "max_tokens": np.array([cap], dtype=np.int64), "fast_out": fast2.numpy(),
        })[0]
        diff = float(np.abs(got - want).max())
        ok = got.shape == want.shape and diff <= TOL
        failures += 0 if ok else 1
        print(f"  N={n2:4d} cap={cap:5d} (traced N=96)  max|diff|={diff:.3e}  "
              f"{'ok' if ok else 'FAIL'}")

    # And the ONNX graph must still agree with the UPSTREAM reference.
    for cap in [16, 64]:
        with torch.no_grad():
            want = reference_routing(x, pred, cap, blk.attn, fast).numpy()
        got = sess.run(["out"], {
            "x": x.numpy(), "pred_score": pred.numpy(),
            "max_tokens": np.array([cap], dtype=np.int64), "fast_out": fast.numpy(),
        })[0]
        diff = float(np.abs(got - want).max())
        ok = diff <= TOL
        failures += 0 if ok else 1
        print(f"  cap={cap:4d} onnx vs UPSTREAM  max|diff|={diff:.3e}  "
              f"{'ok' if ok else 'FAIL'}")
    return failures


def main():
    failures = run_equivalence() + run_onnx()
    print()
    if failures:
        print(f"routing_equivalence_test: {failures} failure(s)")
        return 1
    print("routing_equivalence_test: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
