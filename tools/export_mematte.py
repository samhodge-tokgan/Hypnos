#!/usr/bin/env python3
"""Export MEMatte to two ONNX graphs for the Hypnos OpenFX plugin.

The forward-pass replacements here are derived from MEMatte
(https://github.com/linyiheng123/MEMatte), MIT licensed -- see the NOTICE file.

Produces, per backbone variant:

  mematte_<s|b>_backbone.onnx
      inputs   image       [1, 4, H, W] float32 — ImageNet-normalised RGB with
                                                  the raw trimap as channel 3,
                                                  H and W multiples of 32
               max_tokens  [1] int64          — the global-attention token cap
      output   features    [1, C, H/16, W/16]

  mematte_<s|b>_decoder.onnx
      inputs   features    [1, C, h, w]
               image       [1, 4, H, W]        — the matching image tile
      output   alpha       [1, 1, H, W]        — sigmoid, [0, 1]

Why two graphs
--------------
Upstream runs its decoder over 512-pixel tiles with a 64-pixel halo
(``MEMatte.patch_inference``). Exporting that Python loop would bake in the tile
count for one plate size. Splitting the export lets the C++ engine drive the
tiling, so decoder memory is bounded by ONE tile at any resolution
(see src/MatteEngine.cpp).

Why the legacy exporter
-----------------------
``torch.export`` / ``dynamo=True`` cannot trace a tensor-valued ``topk`` — it
demands a concrete k — and the runtime token cap is exactly that. The legacy
TorchScript exporter handles it, and (verified by tools/shape_dynamics_test.py)
also keeps MEMatte's positional-embedding interpolation and window partitioning
resolution-dynamic. So: ``dynamo=False`` plus ``dynamic_axes``.

The routing rewrite
-------------------
The inference-time router is not exportable as written; see the module docstring
of ``tools/mematte_routing.py`` for what is replaced and why the replacement is
equivalent. ``tools/routing_equivalence_test.py`` checks that claim against a
transcription of upstream's own eval path.

Prerequisites
-------------
    git clone https://github.com/linyiheng123/MEMatte
    pip install -r tools/requirements.txt
    # plus detectron2 (CPU is fine for export):
    pip install 'git+https://github.com/facebookresearch/detectron2.git'
    # and the MEMatte checkpoints from its README (MIT licence)

Usage
-----
    python3 tools/export_mematte.py \
        --mematte-repo ../MEMatte \
        --checkpoint ../MEMatte/checkpoints/MEMatte_ViTS_DIM.pth \
        --variant s \
        --out-dir build/models
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mematte_routing import export_routing  # noqa: E402

CONFIGS = {
    "s": "configs/MEMatte_S_topk0.25_win_global_long.py",
    "b": "configs/MEMatte_B_topk0.25_win_global_long.py",
}


# ---------------------------------------------------------------------------
# Export-time forward replacements
# ---------------------------------------------------------------------------
def routed_block_forward(blk, x, pred_score, max_tokens):
    """``Block.forward`` for a routed (global-attention) block, export-friendly.

    Replaces the eval branch at vit.py:400-407. Upstream gathers with a boolean
    mask (variable length) and writes back with ``masked_scatter_``; this uses a
    fixed-length topk gather/scatter instead.

    ``norm1`` is applied to all tokens and then gathered, rather than gathered
    and then normalised as upstream does. LayerNorm is per-token, so this is
    identical; it just normalises N tokens instead of K.

    ``drop_path`` is omitted: it is a no-op in eval mode (and drop_path_rate is 0
    for these checkpoints), so it cannot change the exported result.
    """
    B, H, W, C = x.shape
    shortcut = x

    if not blk.use_efficient_block:
        raise RuntimeError("routed_block_forward called on a non-routed block")

    fast = blk.efficient_block(x).reshape(B, -1, C)     # LTRM, cheap path
    xn = blk.norm1(x).reshape(B, -1, C)
    msa = export_routing(xn, pred_score, max_tokens, blk.attn, fast)
    msa = msa.reshape(B, H, W, C)

    x = shortcut + msa
    x = x + blk.mlp_forward(x)
    if blk.use_residual_block:
        x = blk.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    if blk.use_convnext_block:
        x = blk.convnext(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return x


def make_vit_forward(get_abs_pos):
    """Build the export-time ``ViT.forward`` (replaces vit.py:558-613)."""

    def vit_forward(vit, x, max_tokens):
        x = vit.patch_embed(x)
        if vit.pos_embed is not None:
            x = x + get_abs_pos(vit.pos_embed, vit.pretrain_use_cls_token,
                                (x.shape[1], x.shape[2]))
        B, _, _, C = x.shape
        count = 0
        for i, blk in enumerate(vit.blocks):
            if i in vit.window_block_indexes or i == 0:
                # Windowed blocks are unrouted and export as-is.
                x = blk(x, policy=None)
                continue
            # Upstream passes prev_decision here, but Router.forward ignores its
            # second argument entirely (backbone/utils.py:253), so it is dropped.
            pred_score = vit.routers[count](x.reshape(B, -1, C), None).reshape(B, -1, 2)
            count += 1
            x = routed_block_forward(blk, x, pred_score, max_tokens)
        return x.permute(0, 3, 1, 2)

    return vit_forward


class BackboneWrapper(nn.Module):
    def __init__(self, vit, vit_forward):
        super().__init__()
        self.vit = vit
        self._fwd = vit_forward

    def forward(self, image, max_tokens):
        return self._fwd(self.vit, image, max_tokens)


class DecoderWrapper(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, features, image):
        return self.decoder(features, image)["phas"]


# ---------------------------------------------------------------------------
def load_model(repo: Path, variant: str, checkpoint: Path):
    sys.path.insert(0, str(repo))
    try:
        from detectron2.config import LazyConfig, instantiate
    except ImportError as e:
        raise SystemExit(
            "detectron2 is required to instantiate MEMatte's LazyConfig model.\n"
            "  pip install 'git+https://github.com/facebookresearch/detectron2.git'\n"
            f"(import error: {e})"
        )
    from modeling.backbone.utils import get_abs_pos  # noqa: F401  (MEMatte repo)

    cfg_path = repo / CONFIGS[variant]
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")
    cfg = LazyConfig.load(str(cfg_path))

    # The distillation teacher is training-only; dropping it avoids
    # instantiating (and having to load weights for) a second backbone.
    cfg.model.distill = False
    if hasattr(cfg.model, "teacher_backbone"):
        cfg.model.teacher_backbone = None

    model = instantiate(cfg.model)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = state.get("model", state)
    state = {k: v for k, v in state.items() if not k.startswith("teacher_backbone.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("teacher_backbone.")]
    if missing:
        print(f"  WARNING: {len(missing)} missing key(s), e.g. {missing[:5]}")
    if unexpected:
        print(f"  note: {len(unexpected)} unexpected key(s), e.g. {unexpected[:5]}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, get_abs_pos


def export(args):
    repo = Path(args.mematte_repo).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading MEMatte ViT-{args.variant.upper()} from {repo}")
    model, get_abs_pos = load_model(repo, args.variant, Path(args.checkpoint))

    backbone = BackboneWrapper(model.backbone, make_vit_forward(get_abs_pos)).eval()
    decoder = DecoderWrapper(model.decoder).eval()

    H = W = args.trace_size
    image = torch.randn(1, 4, H, W)
    image[:, 3] = 0.5  # a plausible trimap channel
    max_tokens = torch.tensor([args.trace_tokens], dtype=torch.int64)

    with torch.no_grad():
        features = backbone(image, max_tokens)
    print(f"  backbone output: {tuple(features.shape)}")

    bb_path = out_dir / f"mematte_{args.variant}_backbone.onnx"
    torch.onnx.export(
        backbone, (image, max_tokens), str(bb_path),
        input_names=["image", "max_tokens"], output_names=["features"],
        dynamic_axes={"image": {2: "H", 3: "W"},
                      "features": {2: "Hf", 3: "Wf"}},
        opset_version=args.opset, dynamo=False,
    )
    print(f"  wrote {bb_path.name} ({bb_path.stat().st_size / 1e6:.0f} MB)")

    dec_path = out_dir / f"mematte_{args.variant}_decoder.onnx"
    torch.onnx.export(
        decoder, (features, image), str(dec_path),
        input_names=["features", "image"], output_names=["alpha"],
        dynamic_axes={"features": {2: "Hf", 3: "Wf"},
                      "image": {2: "H", 3: "W"},
                      "alpha": {2: "H", 3: "W"}},
        opset_version=args.opset, dynamo=False,
    )
    print(f"  wrote {dec_path.name} ({dec_path.stat().st_size / 1e6:.0f} MB)")

    if not args.no_validate:
        validate(backbone, decoder, bb_path, dec_path, args)
    return 0


def validate(backbone, decoder, bb_path, dec_path, args):
    """Check ONNX vs PyTorch at sizes and token caps other than the traced ones.

    This is the check that the dynamic-resolution / dynamic-token-cap design
    actually holds for the real model, not just the isolated routing block.
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        print(f"  validation SKIPPED ({e})")
        return

    print("validating ONNX against PyTorch (sizes/caps not traced):")
    bb = ort.InferenceSession(str(bb_path), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(dec_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for H, W, K in [(args.trace_size, args.trace_size, args.trace_tokens),
                    (320, 448, 4096), (512, 512, 1024), (640, 384, 65536)]:
        img = torch.randn(1, 4, H, W)
        img[:, 3] = torch.randint(0, 3, (1, H, W)).float() / 2.0
        kt = torch.tensor([K], dtype=torch.int64)
        with torch.no_grad():
            f_t = backbone(img, kt)
            a_t = decoder(f_t, img)
        f_o = bb.run(["features"], {"image": img.numpy(),
                                    "max_tokens": kt.numpy()})[0]
        a_o = dec.run(["alpha"], {"features": f_o, "image": img.numpy()})[0]
        df = float(np.abs(f_o - f_t.numpy()).max())
        da = float(np.abs(a_o - a_t.numpy()).max())
        worst = max(worst, da)
        ok = da <= args.tol
        print(f"  {H}x{W} K={K:6d}: features {df:.2e}  alpha {da:.2e} "
              f"{'ok' if ok else 'FAIL'}")
    print(f"  worst alpha deviation: {worst:.2e} (tolerance {args.tol:.0e})")
    if worst > args.tol:
        raise SystemExit("validation FAILED - do not ship these models")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mematte-repo", required=True, help="path to a MEMatte checkout")
    ap.add_argument("--checkpoint", required=True, help="MEMatte .pth checkpoint")
    ap.add_argument("--variant", choices=sorted(CONFIGS), default="s")
    ap.add_argument("--out-dir", default="build/models")
    ap.add_argument("--trace-size", type=int, default=512,
                    help="resolution to trace at (multiple of 32); the graph is "
                         "resolution-dynamic, this only picks the example input")
    ap.add_argument("--trace-tokens", type=int, default=18500)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()
    if args.trace_size % 32:
        raise SystemExit("--trace-size must be a multiple of 32 (size_divisibility)")
    return export(args)


if __name__ == "__main__":
    raise SystemExit(main())
