# Windows (x64, CUDA) build, deployment and testing

**Validated on:** Windows 11 Pro for Workstations build 26200 (a libvirt/QEMU guest with two
RTX 3090s passed through, driver 591.86), Visual Studio 2022 Build Tools MSVC 14.44, CMake 3.27,
CUDA 12.8, Natron 2.6. The plugin builds, loads and renders there.

## Build

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DHYP_WITH_ONNX=ON
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

No patchelf equivalent is needed: the private DLL name is applied by copying the runtime under a
new filename, and the delay-load hook in `src/WinLoader.cpp` resolves it.

Bundle layout produced:

```
MEMatte.ofx.bundle\
  Contents\
    Win64\
      MEMatte.ofx
      onnxruntime_hyp.dll               <-- privately renamed
      onnxruntime_providers_cuda.dll
      onnxruntime_providers_shared.dll
    Resources\                          <-- ONNX models (fetched or exported)
```

## The two Windows-specific hazards

Both are inherited from humbaba, and both are now verified closed on this machine.

### 1. Windows ships its own `onnxruntime.dll`

Windows ML installs one in System32, and hosts such as Nuke make it resident. The loader keys
modules by **base name**, so loading our copy by full path would silently return the OS module —
a different, ABI-incompatible ORT.

The bundled runtime is therefore renamed `onnxruntime_hyp.dll`, the `.ofx` **delay-loads**
`onnxruntime.dll`, and `src/WinLoader.cpp`'s hook redirects that import to the private name by
full path. Verified:

```powershell
dumpbin /DEPENDENTS build\MEMatte.ofx.bundle\Contents\Win64\MEMatte.ofx
#   Image has the following delay load dependencies:
#     onnxruntime.dll
```

The name also differs from humbaba's `onnxruntime_da3.dll`, so both bundles can be installed at
once. Confirmed: headless Natron loaded `com.tokgan.openfx.MEMatte` alongside
`DepthAnything3`, `LensDistortion` and `MoGeFocal` in one process, all with their own ORT copies.

### 2. The host's Visual C++ runtime wins

A host ships its own VC++ runtime in its application directory, which takes precedence on the
loader path. **Nuke 16.1v3 ships `MSVCP140.dll` 14.36** — measured on this machine, where
System32 has 14.44. A plugin that dynamically imports MSVCP140 gets the host's version, and if it
needs a symbol only present in a newer one, it fails to load. That is precisely what broke
humbaba (prebuilt ORT 1.27 needing ≥14.40, failing with `LoadLibrary` error 1114).

Two layers of defence:

- The bundled ONNX Runtime is a **static-CRT build** (see `ORT_URL` in `CMakeLists.txt`).
  Verified: `onnxruntime_hyp.dll` and `onnxruntime_providers_cuda.dll` import **no**
  MSVCP/VCRUNTIME at all.
- The **plugin itself** is linked against the static CRT (`CMAKE_MSVC_RUNTIME_LIBRARY`).
  Before this, `MEMatte.ofx` imported 38 MSVCP140 symbols and carried the same latent risk.

The `.ofx` now has exactly two dependencies:

```
KERNEL32.dll
onnxruntime.dll     (delay-loaded -> onnxruntime_hyp.dll)
```

Static linking is safe here because the OFX boundary is a C ABI — no CRT objects (`FILE*`,
allocations, C++ types) cross between plugin and host, so separate heaps cannot bite.

## Install

```powershell
Copy-Item -Recurse -Force build\MEMatte.ofx.bundle "C:\Program Files\Common Files\OFX\Plugins\"
```

Then fetch or export the models into `Contents\Resources`, or set `$env:MEMATTE_MODEL_DIR`.

## Measured results

```powershell
.\build\Release\ort_check.exe
#   CUDAExecutionProvider listed, CUDA EP available: YES
#   Memory: GPU 23.8 GB/24.0 GB free (NVML), RAM 112.5 GB/120.0 GB available

.\build\Release\ort_check.exe models\mematte_s_backbone.onnx 512 512 18500
#   inference OK: 48.9 ms, VRAM consumed 789 MB
```

The 789 MB matches the 799 MB measured for the same configuration on Linux/A6000, so the sizing
model in `src/Sizing.cpp` carries over.

End-to-end in headless Natron, against the synthetic fixture:

| platform | MAE vs ground truth | mean alpha |
|---|---|---|
| Windows / CUDA | 0.0018 | 0.219110 |
| Linux / CUDA | 0.0018 | 0.219111 |
| macOS / CPU+CoreML | 0.0018 | 0.219110 |

Cross-platform agreement is `max|diff| = 4.9e-4`, `MAE ~1e-6`.

## Not tested

**Nuke could not be run**: this machine has Nuke 16.1v3 installed but its RLM licence server was
unreachable (`ENT_STATUS_RLM_LICENSE_COMM_ERROR`), so the plugin has never actually been loaded
into Nuke. The CRT analysis above is a strong proxy — the `.ofx` now has no CRT dependency for
Nuke's runtime to conflict with — but it is not a substitute for running it.

DaVinci Resolve and Fusion are installed here and also untested.
