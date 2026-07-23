# Linux (x86-64, CUDA) build, deployment and testing

Target for this project: an internal Linux workstation, deliberately not named here — put it in
your `~/.ssh/config` and refer to it by alias.

**Validated on:** Ubuntu 24.04.4, NVIDIA RTX A6000 (48 GB, driver 570.211.01), CUDA 12,
gcc 12.4, CMake 4.3, Natron 2.5.0. Everything in this document has been run there.

## Measured results

| | |
|---|---|
| ViT-S backbone, 512², CUDA | 12.8 ms (macOS CPU: 858 ms) |
| 4K matte (3840×2160), cap 2048 | 2.6 s, 6.9 GB VRAM, MAE 0.0003 |
| Cross-platform parity vs macOS | max 4.9e-4, MAE 1.3e-6 |
| NVML probe | 43.8 / 48.0 GB free, correct |

Token cap versus cost, ViT-S at 2048² — the same resolution throughout:

| cap | VRAM | time |
|---|---|---|
| 1024 | 1.3 GB | 111 ms |
| 4096 | 2.3 GB | 131 ms |
| 16384 | 35.3 GB | 651 ms |
| 24576 (3072×2048) | **OOM at 48 GB** | — |

That last row is why the sizing model in `src/Sizing.cpp` is deliberately conservative: it must
predict the OOM rather than admit the job. `tests/sizing_test.cpp` pins all of these numbers.

## Build prerequisites

- A C++17 compiler with `GLIBCXX_3.4.21+`. On Rocky/RHEL 8 use **gcc-toolset-12**
  (`source /opt/rh/gcc-toolset-12/enable`); the stock gcc 8.5 works for the plugin itself but is
  older than what the dependencies expect.
- CMake ≥ 3.20.
- **patchelf** (`dnf install patchelf` / `apt-get install patchelf`) — required for the private
  soname step; CMake fails with a clear error if it is missing.
- **CUDA Toolkit 12.x** and **cuDNN 9**, for the ONNX Runtime CUDA package the build fetches.

```bash
source /opt/rh/gcc-toolset-12/enable      # Rocky 8 only
cmake -S . -B build -DHYP_WITH_ONNX=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ort_check MEMatte -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

Bundle layout produced:

```
MEMatte.ofx.bundle/
  Contents/
    Linux-x86-64/                     # <-- OFX spec arch dir: HYPHENS, not "Linux-x86_64"
      MEMatte.ofx                     # RUNPATH=$ORIGIN
      libonnxruntime_hyp.so.1         # privately renamed (see docs/DEVELOPMENT.md)
      libonnxruntime_providers_*.so
    Resources/                        # ONNX models (fetched or exported)
```

> **Gotcha inherited from humbaba:** the OFX packaging spec names the 64-bit Linux arch directory
> `Linux-x86-64` (hyphens). With `Linux-x86_64` the binary is valid and `dlopen`s fine, but OFX
> hosts silently skip it.

## Runtime requirements

The bundle ships ONNX Runtime and its CUDA provider libraries (found via `$ORIGIN`) but **not** the
CUDA runtime itself. The host must provide, and the loader must find:

- an NVIDIA driver (`libcuda.so.1`),
- CUDA 12.x: `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`, `libcufft.so.11`,
  `libcurand.so.10`, `libnvrtc.so.12`,
- cuDNN 9: `libcudnn.so.9`.

```bash
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:/usr/lib64:$LD_LIBRARY_PATH"
```

If they are missing the plugin still loads and runs on **CPU**, which for matting is slow but
correct — and is an explicit, supported mode (`Device = CPU`).

`MemProbe` `dlopen`s `libnvidia-ml.so.1` (falling back to `libcudart`) purely to read free VRAM;
neither is a link-time dependency, so a machine with no NVIDIA stack loads the plugin fine.

## Install

```bash
cp -r build/MEMatte.ofx.bundle ~/OFX/Plugins/          # per-user
sudo cp -r build/MEMatte.ofx.bundle /usr/OFX/Plugins/  # system-wide
export OFX_PLUGIN_PATH="$HOME/OFX/Plugins"
```

Then fetch or export the models (see the README) into `Contents/Resources`, or point
`MEMATTE_MODEL_DIR` at them.

## Testing on the Linux GPU workstation

### 1. Providers and memory

```bash
build/ort_check
# expect: CUDAExecutionProvider listed, "CUDA EP available: YES",
#         and a GPU line from NVML with real free/total bytes
```

### 2. Headless Natron — discovery, then a render

```bash
python3 tests/natron/make_test_assets.py test-assets

OFX_PLUGIN_PATH="$HOME/OFX/Plugins" \
  NatronRenderer --clear-openfx-cache -t tests/natron/check_plugin.py < /dev/null
# expect: RESULT: PASS, and INPUTS listing a Trimap input

HYP_INPUT=$PWD/test-assets/plate_srgb8.png \
HYP_TRIMAP=$PWD/test-assets/trimap_int8.png \
HYP_OUTPUT=$PWD/build/test/matte.exr \
MEMATTE_MODEL_DIR=$PWD/build/models \
HYP_ACESCG=0 HYP_TOKENS=18500 \
OFX_PLUGIN_PATH="$HOME/OFX/Plugins" \
  NatronRenderer --clear-openfx-cache -t tests/natron/render_matte.py < /dev/null
```

### 3. Nuke

```bash
OFX_PLUGIN_PATH="$HOME/OFX/Plugins" nuke -t
# >>> import nuke; n = nuke.createNode('MEMatte'); print(n.name())
```
Check the node has two inputs, that a Read → MEMatte render produces a matte, and that the
*Status* knob reports the measured VRAM and the planned token count.

### 4. Memory behaviour

Sample VRAM at 10 Hz around a render (1 Hz misses the peak on a fast frame):

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > /tmp/v.log &
SMI=$!; ... NatronRenderer ... ; kill $SMI; sort -n /tmp/v.log | tail -1
```

Verified degradation ladder on the 4K plate, with the budget forced small:

| `HYP_BUDGET_MB` | what happened |
|---|---|
| 8000 | token cap lowered to 5091 |
| 3000 | cap lowered **and** resolution reduced to 960×540 |
| 1200 | resolution reduced to 608 px long side |

All three rendered; none failed. The resolution step costs real quality (MAE 0.0019 versus
0.0003 at full resolution), which is why it is the last resort.

> **Always check the matte, not just the exit code.** On failure the plugin falls through to
> passthrough, which still writes a file and still reports `RESULT: PASS`. Score the output
> against `alpha_gt*.png`: a passthrough scores ~0.23 MAE, a real matte ~0.002. An unrecognised
> out-of-memory string caused exactly this silent failure once — see `tests/oom_test.cpp`.

Confirm with `nvidia-smi` that the VRAM is attributed to the **host process** (in-process
execution, no forked worker), as humbaba verified for Flame on Rocky 8.
