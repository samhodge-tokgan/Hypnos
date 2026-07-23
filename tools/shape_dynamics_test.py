#!/usr/bin/env python3
"""Check that MEMatte's shape-sensitive helpers export at a dynamic resolution.

The helpers under test are transcribed from MEMatte
(https://github.com/linyiheng123/MEMatte), MIT licensed -- see the NOTICE file.

The backbone must run at whatever size the plate is, from ONE ONNX graph, or the
plugin cannot honour "input and output at the same resolution". Two upstream
helpers in ``modeling/backbone/utils.py`` look like they might freeze the traced
resolution into the graph:

  * ``get_abs_pos`` — bicubic ``F.interpolate`` of the positional embedding to
    the current token grid, with the target size read from ``x.shape``.
  * ``window_partition`` / ``window_unpartition`` — padding amounts and ``view``
    shapes derived from H and W (window size 14 rarely divides the grid).

humbaba's Depth Anything 3 export needed the dynamo exporter plus two
monkeypatches for exactly this class of problem. This test establishes that
MEMatte does NOT: the legacy TorchScript exporter traces both helpers into
genuine Shape/Resize/Reshape nodes, so one graph serves every resolution.

That matters because MEMatte cannot use the dynamo exporter — ``torch.export``
rejects the tensor-valued ``topk`` the runtime token cap depends on (it needs a
concrete k). Legacy + ``dynamic_axes`` is therefore the export path, and this
test guards the assumption that makes it viable.

The helpers are transcribed here rather than imported so the test needs no
detectron2 install; keep them in sync with upstream.

Run:  python3 tools/shape_dynamics_test.py
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TOL = 1e-4


# --- transcribed from MEMatte modeling/backbone/utils.py --------------------
def get_abs_pos(abs_pos, has_cls_token, hw):
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num
    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w), mode="bicubic", align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    return abs_pos.reshape(1, h, w, -1)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class ShapeSensitiveBlock(nn.Module):
    """The two helpers wired together the way ViT.forward uses them."""

    def __init__(self, dim=8, pretrain_grid=32, window_size=14):
        super().__init__()
        self.window_size = window_size
        self.pos_embed = nn.Parameter(torch.randn(1, pretrain_grid * pretrain_grid + 1, dim))
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):  # x: [1, H, W, C]
        x = x + get_abs_pos(self.pos_embed, True, (x.shape[1], x.shape[2]))
        H, W = x.shape[1], x.shape[2]
        win, pad_hw = window_partition(x, self.window_size)
        win = self.lin(win)
        return window_unpartition(win, self.window_size, pad_hw, (H, W))


# --- transcribed from MEMatte modeling/backbone/utils.py:235 ---------------
class Router(nn.Module):
    """Upstream Router. Its forward bakes the token count into a trace."""

    def __init__(self, embed_dim=64):
        super().__init__()
        self.in_conv = nn.Sequential(nn.LayerNorm(embed_dim),
                                     nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4), nn.GELU(),
            nn.Linear(embed_dim // 4, 2), nn.LogSoftmax(dim=-1))

    def forward(self, x, policy=None):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:, :, : C // 2]
        global_x = (x[:, :, C // 2:]).sum(dim=1, keepdim=True) / N
        x = torch.cat([local_x, global_x.expand(B, N, C // 2)], dim=-1)
        return self.out_conv(x)


def check_router(np, ort, out_dir):
    """Both the stock and the shape-safe Router must hold up at a new resolution.

    Upstream's ``Router.forward`` reads ``B, N, C = x.size()`` and then uses N
    arithmetically (``.sum(dim=1) / N`` and ``expand(B, N, C // 2)``), which
    looks like it would freeze the traced token count. It does not: the
    TorchScript tracer records ``aten::size`` and keeps values derived from it
    symbolic, so the exported graph divides by the real N. Only an explicit
    ``torch.tensor(N)`` would bake a constant.

    That is worth pinning down with a test rather than assuming, because the
    failure mode would be silent — a wrong divisor yields wrong router scores
    and therefore wrong token selection, with no shape error to notice.

    ``router_forward_export`` makes the same thing explicit via a Shape node. It
    is kept as belt-and-braces and is checked here to be equivalent, not because
    the stock version is broken.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mematte_routing import router_forward_export

    torch.manual_seed(1)
    failures = 0
    trace_n, other_n = 96, 240

    for label, patched in (("stock", False), ("patched", True)):
        model = Router(64).eval()
        if patched:
            model.forward = router_forward_export.__get__(model, Router)
        path = out_dir / f"router_{label}.onnx"
        torch.onnx.export(
            model, (torch.randn(1, trace_n, 64),), str(path),
            input_names=["x"], output_names=["score"],
            dynamic_axes={"x": {1: "N"}, "score": {1: "N"}},
            opset_version=17, dynamo=False,
        )
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        # Compare against the STOCK eager module, which is ground truth at any N.
        truth = Router(64).eval()
        truth.load_state_dict(model.state_dict())

        agrees = {}
        for n in (trace_n, other_n):
            xt = torch.randn(1, n, 64)
            with torch.no_grad():
                want = truth(xt).numpy()
            try:
                got = sess.run(["score"], {"x": xt.numpy()})[0]
                agrees[n] = got.shape == want.shape and float(np.abs(got - want).max()) <= TOL
            except Exception:  # noqa: BLE001
                agrees[n] = False

        ok = agrees[trace_n] and agrees[other_n]
        print(f"  router {label:8s}: N={trace_n} {'ok' if agrees[trace_n] else 'FAIL'}, "
              f"N={other_n} {'ok' if agrees[other_n] else 'FAIL'}")
        failures += 0 if ok else 1
    return failures


def main():
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        print(f"shape_dynamics_test: SKIPPED ({e})")
        return 0

    torch.manual_seed(0)
    model = ShapeSensitiveBlock().eval()
    trace_h, trace_w = 64, 64
    example = torch.randn(1, trace_h, trace_w, 8)

    out = Path(__file__).resolve().parent.parent / "build" / "shape_dynamics_test.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (example,), str(out),
        input_names=["x"], output_names=["y"],
        dynamic_axes={"x": {1: "H", 2: "W"}, "y": {1: "H", 2: "W"}},
        opset_version=17, dynamo=False,
    )
    print(f"exported {out.name}, traced at {trace_h}x{trace_w}")

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    failures = 0
    # Sizes deliberately NOT equal to the traced one, not square, not multiples
    # of the window size, and both larger and smaller than the pretrain grid.
    for H, W in [(64, 64), (96, 96), (32, 48), (128, 80), (224, 128), (160, 352)]:
        xt = torch.randn(1, H, W, 8)
        with torch.no_grad():
            want = model(xt).numpy()
        try:
            got = sess.run(["y"], {"x": xt.numpy()})[0]
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  {H}x{W}: RUNTIME FAIL {str(e)[:160]}")
            continue
        same = got.shape == want.shape
        diff = float(np.abs(got - want).max()) if same else float("nan")
        ok = same and diff <= TOL
        failures += 0 if ok else 1
        print(f"  {H}x{W}: shape {got.shape} max|diff|={diff:.2e} {'ok' if ok else 'FAIL'}")

    failures += check_router(np, ort, out.parent)

    if failures:
        print(f"shape_dynamics_test: {failures} failure(s)")
        return 1
    print("shape_dynamics_test: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
