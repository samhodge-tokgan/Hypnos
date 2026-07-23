// Copyright the Hypnos authors.
// SPDX-License-Identifier: Apache-2.0
//
// Cross-platform ONNX Runtime accelerator execution-provider selection:
// CoreML on macOS, CUDA on Linux/Windows (with the caller's CPU fallback).
//
// Adapted from the humbaba DA3 plugin, extended with an explicit VRAM cap so the
// matting session can be bounded by a measured memory budget (see MemProbe.h).
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>

#include <onnxruntime_cxx_api.h>

namespace hyp {

// Ort::Session takes a path of type const ORTCHAR_T* — wchar_t on Windows, char on
// POSIX. std::filesystem::path::c_str() yields exactly that native type, so this
// converts a UTF-8/native std::string path for the Session constructor. Keep the
// returned object alive across the c_str() use (it is, within a full expression).
inline std::filesystem::path OrtPath(const std::string& s) {
  return std::filesystem::path(s);
}

inline const char* AcceleratorSubstr() {
#ifdef __APPLE__
  return "CoreML";
#else
  return "CUDA";
#endif
}

// True if the platform accelerator EP is compiled into this ONNX Runtime build.
inline bool AcceleratorAvailable() {
  const std::string want = AcceleratorSubstr();
  for (const auto& p : Ort::GetAvailableProviders())
    if (p.find(want) != std::string::npos) return true;
  return false;
}

// Append the platform accelerator EP to `so`. On macOS this is CoreML (using the
// given CoreML options); on Linux/Windows it is CUDA (device 0). Throws
// Ort::Exception on failure — callers catch and fall back to CPU.
//
// `gpu_mem_limit` (bytes, 0 = unlimited) caps the CUDA EP's arena. This is a real
// ONNX Runtime option but it is CUDA-ONLY: CoreML on Apple Silicon draws from
// unified memory and exposes no byte-limit API, so on macOS the memory lever is
// the token cap and the processing resolution instead.
//
// `coreml_static` requests RequireStaticInputShapes. The matting backbone takes a
// runtime token count (dynamic TopK), so callers pass false — demanding static
// shapes would make CoreML reject the whole subgraph rather than partially
// accelerate it.
inline void AppendAccelerator(Ort::SessionOptions& so, const char* coreml_units,
                              bool coreml_static, bool coreml_mlprogram = true,
                              uint64_t gpu_mem_limit = 0) {
#ifdef __APPLE__
  (void)gpu_mem_limit;
  std::unordered_map<std::string, std::string> opts;
  if (coreml_units) opts["MLComputeUnits"] = coreml_units;
  if (coreml_mlprogram) opts["ModelFormat"] = "MLProgram";
  if (coreml_static) opts["RequireStaticInputShapes"] = "1";
  so.AppendExecutionProvider("CoreML", opts);
#else
  (void)coreml_units;
  (void)coreml_static;
  (void)coreml_mlprogram;
  OrtCUDAProviderOptions cuda{};
  cuda.device_id = 0;
  if (gpu_mem_limit > 0) {
    cuda.gpu_mem_limit = static_cast<size_t>(gpu_mem_limit);
    // kSameAsRequested (1) instead of the default kNextPowerOfTwo (0): the
    // power-of-two growth can double the arena well past what a run actually
    // needs, which is exactly the over-reservation the budget is meant to stop.
    cuda.arena_extend_strategy = 1;
  }
  so.AppendExecutionProvider_CUDA(cuda);
#endif
}

}  // namespace hyp
