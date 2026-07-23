# Development

## Layout

```
src/
  MEMattePlugin.cpp   OFX plugin: clips, params, colour + trimap handling, output modes
  MatteEngine.{h,cpp} two ORT sessions (backbone + tiled decoder), OOM retry ladder
  MemProbe.{h,cpp}    runtime VRAM / RAM probing (dlopen NVML, no build-time CUDA dep)
  Sizing.{h,cpp}      memory model: tokens and resolution that fit a budget
  Trimap.{h,cpp}      trimap value -> 0 / 0.5 / 1
  Color.h             ACEScg <-> sRGB, bit-depth-aware pixel IO
  OrtAccel.h          CoreML / CUDA EP selection (+ gpu_mem_limit)
  WinLoader.cpp       Windows delay-load hook for the privately-named ORT
tools/
  mematte_routing.py  the export-friendly routing rewrite (+ upstream reference)
  export_mematte.py   ONNX export of both graphs
  routing_equivalence_test.py, shape_dynamics_test.py
tests/
  trimap_test.cpp, sizing_test.cpp   ctest unit tests (no ORT, no host needed)
  ort_check.cpp                      providers + memory + one real inference
  natron/                            headless host tests and fixture generation
```

## Pinned versions

| Component | Version |
|---|---|
| OpenFX SDK | `OFX_Release_1.5.1` |
| ONNX Runtime | 1.27.1 (CoreML arm64 / CUDA 12 Linux / static-CRT CUDA Windows) |
| ONNX opset | 17 |
| MEMatte | upstream `main`, MIT |

## The export, and why it is shaped this way

Read `tools/mematte_routing.py`'s module docstring first — it is the design document for the
whole export.

Three constraints interact:

1. **The inference router is not exportable.** Upstream's eval path uses `.item()`, boolean-mask
   indexing (data-dependent output length) and `masked_scatter_`.
2. **The token cap must be a runtime input**, so one model file serves every cap.
3. **One graph must serve every resolution**, because the plugin infers at the plate's own size.

(1) and (2) are solved by the same change: make upstream's own top-K truncation unconditional, so
the gather is fixed-length and K comes in as a tensor. `tools/routing_equivalence_test.py` proves
this changes nothing (max deviation `3e-7`, usually exactly `0`).

(3) forces the **legacy** exporter. `torch.export` / `dynamo=True` cannot trace a tensor-valued
`topk` — it demands a concrete k — so the dynamo path is closed. Fortunately the legacy
TorchScript exporter keeps MEMatte's shape-sensitive helpers dynamic, which
`tools/shape_dynamics_test.py` verifies at six unseen resolutions. This differs from humbaba,
where DA3 *required* dynamo plus two monkeypatches; MEMatte needs neither.

A note on a non-bug: `Router.forward` reads `B, N, C = x.size()` and then divides by `N`, which
looks like it would bake the traced token count. It does not — the tracer keeps values derived
from `aten::size` symbolic. This is tested rather than assumed, because the failure mode would be
silent (wrong divisor → wrong scores → wrong tokens, no shape error).

### Running the export

```sh
git clone https://github.com/linyiheng123/MEMatte
pip install -r tools/requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git'

python3 tools/export_mematte.py --mematte-repo ../MEMatte \
    --checkpoint ../MEMatte/checkpoints/MEMatte_ViTS_DIM.pth \
    --variant s --out-dir build/models
```

It validates ONNX against PyTorch at resolutions and token caps other than the traced ones and
exits non-zero if the deviation exceeds `--tol`. Both variants have been exported this way and
agree to 3.8e-5 on alpha.

Two install notes that cost time:

- detectron2 must be installed with `--no-build-isolation` (its `setup.py` imports torch, which
  pip's isolated build environment hides), and on macOS with `KMP_DUPLICATE_LIB_OK=TRUE` (its
  metadata step aborts on the duplicate OpenMP runtime).
- MEMatte's config imports its training dataloader, so `easydict` and `opencv-python-headless`
  are needed even though the export never touches them.

## Execution providers are chosen PER GRAPH

CoreML's MIL compiler rejects the backbone: the runtime token cap leaves the TopK/Gather
dimensions unbounded and it reports `has unbounded dimension which is not supported`. ONNX Runtime
then falls back partition by partition, which is slower than never offering CoreML. Measured on
Apple Silicon, ViT-S at 512×512:

| graph | CPU | CoreML offered |
|---|---|---|
| backbone (dynamic K) | **858 ms** | 2923 ms |
| decoder (plain convs) | 546 ms | **259 ms** |

`MatteEngine::Configure` therefore offers the accelerator only to the decoder on macOS, and to
both graphs elsewhere. **Re-measure this on CUDA before assuming it transfers** — CUDA handles
dynamic shapes natively and should want the GPU for both.

Remaining follow-up: the decoder still emits four `unbounded dimension` complaints from its
dynamic tile height/width. Padding every decoder tile to a fixed 640×640 would make its shapes
static and should silence them; static-K preset exports would do the same for the backbone, at the
cost of one model file per cap.

## Model contract

From `configs/common/model.py`, `meta_arch/mematte.py` and `backbone/vit.py` upstream:

- Input `[1, 4, H, W]`: RGB normalised with ImageNet stats **on a [0,1] scale**
  (`mean = [123.675, 116.280, 103.530] / 255`, `std = [58.395, 57.120, 57.375] / 255`),
  concatenated with the **raw** trimap (0 / 0.5 / 1) as channel 3.
- `size_divisibility = 32`. Upstream normalises **first**, then zero-pads bottom-right, so the
  padding sits at the normalised origin rather than at black. `MatteEngine::RunOnce` replicates
  that ordering.
- Patch size 16 → features at H/16.
- Windowed blocks at `0,1,3,4,6,7,9,10` (window 14); **routing happens only at blocks 2, 5, 8, 11**,
  which are built with `use_rel_pos=False` and `window_size=0` — plain attention, which is what
  makes the exact rewrite possible.
- Output alpha `[1,1,H,W]` in [0,1], then composited: `trimap == 0 → 0`, `trimap == 1 → 1`.

## Calibrating the memory model

`src/Sizing.cpp` estimates peak bytes as weights + linear activations + the quadratic attention
term + one decoder tile. It reproduces MEMatte's published 1024² figures to within ~8% (ViT-S) and
~1% (ViT-B), but it has **not** been fitted to a real allocator trace. To calibrate:

```sh
build/ort_check                                        # providers + free VRAM
build/ort_check build/models/mematte_s_backbone.onnx 2048 2048 18500
```

`ort_check` re-probes after the run and prints the VRAM the session actually consumed. Sweep
resolution and token cap, fit the constants, and update `EstimatePeakBytes` — then relax the
tolerance bounds in `tests/sizing_test.cpp` only if the structure genuinely changed.

Until that is done, the safety net is the retry ladder in `MatteEngine::Run` and the headroom
parameter, not the estimate.

## Host quirks inherited from humbaba

These cost real debugging time there; do not undo them.

- **Linux arch dir is `Linux-x86-64`** (hyphens). With an underscore the binary loads fine but OFX
  hosts silently skip it.
- **DaVinci Resolve's render context throws** on the OFX message suite. `SafeSetMessage` /
  `SafeClearMessage` swallow that, or the exception unwinds out of the render action and the host
  reports the whole render as failed.
- **Windows ships its own `onnxruntime.dll`** in System32 (Windows ML). Because the loader keys
  modules by base name, the bundled runtime is renamed `onnxruntime_hyp.dll` and loaded by full
  path from `WinLoader.cpp`'s delay-load hook. The name also differs from humbaba's
  `onnxruntime_da3.dll` so both bundles can coexist in one host process.
- **Nuke 16.1 forces MSVCP140 14.36** process-wide; Microsoft's prebuilt ORT 1.27 needs ≥14.40 and
  fails with LoadLibrary error 1114. Windows therefore uses a static-CRT ORT build.
- **Only the OFX entry points are exported** (`packaging/ofx_exports.*`), so nothing interposes on
  the host or a sibling plugin.
