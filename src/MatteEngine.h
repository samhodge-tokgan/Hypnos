// Copyright the Hypnos authors.
// SPDX-License-Identifier: Apache-2.0
//
// MEMatte inference: two ONNX Runtime sessions (ViT backbone + detail decoder)
// wrapped so an OFX plugin can extract an alpha matte at the source resolution
// without risking an out-of-memory failure.
//
// Two graphs, not one, for a reason. Upstream MEMatte runs its decoder over
// 512-pixel tiles with a 64-pixel halo (MEMatte.patch_inference); exporting that
// Python loop into ONNX would bake in the tile count. Splitting the export lets
// C++ drive the tiling, so the decoder's peak memory is bounded by ONE tile
// regardless of plate size, and only the backbone scales with resolution.
//
// Memory is controlled on three axes, applied in this order of preference:
//   1. max_tokens  — MEMatte's own cap on how many tokens reach global attention.
//                    Bounds the quadratic term while keeping FULL resolution, so
//                    it costs no matte detail. This is the preferred lever.
//   2. proc_long_side — downscale before inference, upsample the alpha after.
//                    Bounds everything linearly but softens fine detail, so it is
//                    a fallback, not a default.
//   3. CPU         — correct and unbounded by VRAM, just slower.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Sizing.h"

namespace hyp {

enum class Device { Auto = 0, GPU, CPU };

// CoreML MLComputeUnits selection (macOS). Ignored by the CUDA EP.
enum class ComputeUnits { All = 0, CPUAndGPU, CPUAndNeuralEngine, CPUOnly };

struct MatteConfig {
  std::string backbone_path;
  std::string decoder_path;
  Device device = Device::Auto;
  ComputeUnits compute_units = ComputeUnits::All;
  int intra_threads = 0;        // <=0 leaves the ORT default
  uint64_t gpu_mem_limit = 0;   // bytes; 0 = uncapped. CUDA EP only.
};

struct MatteRequest {
  int max_tokens = 18500;       // MEMatte's global-attention token cap
  int proc_long_side = 0;       // 0 = process at native resolution
};

struct MatteResult {
  std::vector<float> alpha;     // width*height, row-major, [0,1]
  int width = 0;
  int height = 0;

  // What was actually used (may differ from the request after the retry ladder).
  int used_tokens = 0;
  int proc_width = 0;
  int proc_height = 0;
  bool used_cpu = false;
  // Non-empty when the engine had to degrade; surface this to the user.
  std::string degraded;

  bool ok() const { return !alpha.empty(); }
};

class MatteEngine {
 public:
  MatteEngine();
  ~MatteEngine();

  MatteEngine(const MatteEngine&) = delete;
  MatteEngine& operator=(const MatteEngine&) = delete;

  // Create both sessions. Returns false with last_error() set on failure.
  // Falls back to CPU automatically if the accelerator cannot be used, which is
  // reported through accelerator_active() rather than as an error.
  bool Configure(const MatteConfig& cfg);

  // True once Configure has produced usable sessions.
  bool ready() const;

  // True if the sessions actually placed nodes on the platform accelerator.
  bool accelerator_active() const;

  // Run matting.
  //   rgb    : in_w*in_h*3 interleaved, sRGB-encoded [0,1] (see Color.h)
  //   trimap : in_w*in_h, values exactly 0.0 / 0.5 / 1.0 (see Trimap.h)
  // Returns alpha at in_w x in_h. On an accelerator out-of-memory the engine
  // retries with a smaller token cap, then a reduced processing resolution, then
  // on CPU, recording what it did in MatteResult::degraded. An empty result means
  // every attempt failed; see last_error().
  MatteResult Run(const float* rgb, const float* trimap, int in_w, int in_h,
                  const MatteRequest& req);

  const std::string& last_error() const { return last_error_; }
  const MatteConfig& config() const { return cfg_; }

  // Whether the platform accelerator EP is compiled into this ORT build.
  static bool AcceleratorAvailable();

 private:
  // One attempt at the given processing size and token cap. Returns an empty
  // vector on failure (with last_error_ set); throws nothing.
  bool RunOnce(const float* rgb, const float* trimap, int in_w, int in_h, int proc_w,
               int proc_h, int max_tokens, std::vector<float>* alpha_proc);

  struct Impl;
  std::unique_ptr<Impl> impl_;
  MatteConfig cfg_;
  std::string last_error_;
};

}  // namespace hyp
