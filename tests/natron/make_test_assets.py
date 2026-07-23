#!/usr/bin/env python3
"""Generate a synthetic plate + trimap in several encodings.

Deterministic fixtures for the colour/trimap matrix check: the SAME scene is
written display-referred (8-bit sRGB) and scene-referred (linear ACEScg EXR),
and the SAME trimap is written as 8-bit 0/128/255 and as float 0/0.5/1. Feeding
any combination through the plugin must produce the same matte — that is the
regression test for the trimap-encoding and colour-space handling.

The subject has a soft, semi-transparent edge so the unknown band is doing real
work rather than just relabelling hard pixels.

Usage:  python3 tests/natron/make_test_assets.py test-assets
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Rec.709 (linear) -> ACEScg AP1. Inverse of the AP1->Rec.709 matrix the plugin
# applies in src/Color.h, so a round trip through the plugin is the identity.
REC709_TO_AP1 = np.array([
    [0.61309732, 0.33952285, 0.04737983],
    [0.07019422, 0.91635557, 0.01345022],
    [0.02061560, 0.10956983, 0.86981457],
], dtype=np.float32)


def srgb_to_linear(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def make_scene(w=512, h=384):
    """Return (rgb_srgb [h,w,3] in [0,1], alpha [h,w] in [0,1])."""
    ys, xs = np.mgrid[0:h, 0:w]
    cx, cy, r = w * 0.5, h * 0.5, min(w, h) * 0.30

    # A disc with a soft rim, plus a few fine 'hair' strands crossing the rim —
    # the kind of detail that a downscaled inference would visibly lose.
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    alpha = np.clip((r + 6.0 - d) / 12.0, 0.0, 1.0).astype(np.float32)
    for k in range(14):
        ang = k * (2 * np.pi / 14)
        t = np.linspace(0, 1, 800)
        px = (cx + np.cos(ang) * (r + t * r * 0.7)).astype(int)
        py = (cy + np.sin(ang) * (r + t * r * 0.7)).astype(int)
        ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        alpha[py[ok], px[ok]] = np.maximum(alpha[py[ok], px[ok]], 0.85)

    fg = np.stack([np.full((h, w), 0.85), np.full((h, w), 0.55),
                   np.full((h, w), 0.25)], -1).astype(np.float32)
    bg = np.stack([xs / w * 0.3 + 0.1, ys / h * 0.3 + 0.15,
                   np.full((h, w), 0.45)], -1).astype(np.float32)
    rgb = fg * alpha[..., None] + bg * (1.0 - alpha[..., None])
    return rgb.astype(np.float32), alpha


def make_trimap(alpha, band=14):
    """0 where certainly background, 1 where certainly foreground, 0.5 between.

    The unknown band is grown around every partial-alpha pixel so the model has
    a genuine region to solve, matching how a trimap is authored in practice.
    """
    h, w = alpha.shape
    known_fg = alpha >= 0.99
    known_bg = alpha <= 0.01
    unknown = ~(known_fg | known_bg)
    # Cheap box dilation of the unknown region (no scipy dependency).
    acc = np.zeros((h, w), dtype=np.int32)
    pad = np.pad(unknown.astype(np.int32), band)
    for dy in range(2 * band + 1):
        for dx in range(2 * band + 1):
            acc += pad[dy:dy + h, dx:dx + w]
    unknown = acc > 0

    tri = np.full((h, w), 0.5, dtype=np.float32)
    tri[known_bg & ~unknown] = 0.0
    tri[known_fg & ~unknown] = 1.0
    return tri


def main(out_dir="test-assets", w=512, h=384, suffix=""):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rgb_srgb, alpha = make_scene(w, h)
    # Scale the unknown band with the plate so the matting problem stays
    # comparable across resolutions rather than becoming a thin sliver at 4K.
    tri = make_trimap(alpha, band=max(4, round(14 * min(w, h) / 384)))

    written = []

    try:
        from PIL import Image
    except ImportError:
        print("Pillow is required: pip install pillow", file=sys.stderr)
        return 1

    # --- display-referred: 8-bit sRGB ------------------------------------
    p = out / f"plate_srgb8{suffix}.png"
    Image.fromarray((rgb_srgb * 255).round().astype(np.uint8), "RGB").save(p)
    written.append(p)

    # 8-bit trimap in the classic 0 / 128 / 255 encoding. Written by explicit
    # selection rather than arithmetic so the mid value is exactly 128 — the
    # whole point of these fixtures is that the encoding is unambiguous.
    tri8 = np.select([tri <= 0.0, tri >= 1.0], [0, 255], default=128).astype(np.uint8)
    p = out / f"trimap_int8{suffix}.png"
    Image.fromarray(tri8, "L").save(p)
    written.append(p)

    # Ground-truth alpha, for scoring a matte if you want a quality number.
    p = out / f"alpha_gt{suffix}.png"
    Image.fromarray((alpha * 255).round().astype(np.uint8), "L").save(p)
    written.append(p)

    # --- scene-referred: linear ACEScg EXR --------------------------------
    try:
        # imageio's OpenCV backend ships with the EXR codec compiled in but
        # disabled behind this flag; without it EXR writing raises.
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import imageio.v3 as iio

        lin709 = srgb_to_linear(rgb_srgb).astype(np.float32)
        acescg = lin709 @ REC709_TO_AP1.T
        p = out / f"plate_acescg{suffix}.exr"
        iio.imwrite(p, acescg.astype(np.float32))
        written.append(p)

        # Float trimap with a literal 0 / 0.5 / 1 encoding, written as DATA — it
        # must not be colour-converted on the way in or out.
        p = out / f"trimap_float{suffix}.exr"
        iio.imwrite(p, np.repeat(tri[..., None], 3, axis=2).astype(np.float32))
        written.append(p)
    except Exception as e:  # noqa: BLE001
        print(f"note: EXR outputs skipped ({e}); "
              f"install imageio + OpenEXR support for the scene-referred half")

    for p in written:
        print(f"wrote {p}")
    print(f"unknown band covers {(tri == 0.5).mean() * 100:.1f}% of the frame")
    return 0


if __name__ == "__main__":
    # usage: make_test_assets.py [out_dir] [WIDTH HEIGHT]
    a = sys.argv[1:]
    d = a[0] if a else "test-assets"
    if len(a) >= 3:
        W, Hh = int(a[1]), int(a[2])
        raise SystemExit(main(d, W, Hh, suffix=f"_{W}x{Hh}"))
    raise SystemExit(main(d))
