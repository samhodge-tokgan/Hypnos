// Copyright the Hypnos authors.
// SPDX-License-Identifier: Apache-2.0
//
// The decoder tiling, input normalisation constants and padding order follow
// MEMatte (https://github.com/linyiheng123/MEMatte), MIT licensed - see NOTICE.
#include "MatteEngine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include <onnxruntime_cxx_api.h>

#include "OomDetect.h"
#include "OrtAccel.h"

namespace hyp {

// ImageNet normalisation on a [0,1] scale — MEMatte's configs/common/model.py
// divides the usual 0-255 constants by 255. Applied to RGB only; the trimap is
// concatenated as a RAW 4th channel and must NOT be normalised.
static const float kPixelMean[3] = {123.675f / 255.f, 116.280f / 255.f, 103.530f / 255.f};
static const float kPixelStd[3] = {58.395f / 255.f, 57.120f / 255.f, 57.375f / 255.f};

namespace {

const char* ComputeUnitsString(ComputeUnits u) {
  switch (u) {
    case ComputeUnits::All: return "ALL";
    case ComputeUnits::CPUAndGPU: return "CPUAndGPU";
    case ComputeUnits::CPUAndNeuralEngine: return "CPUAndNeuralEngine";
    case ComputeUnits::CPUOnly: return "CPUOnly";
  }
  return "ALL";
}

// Bilinear resample of an interleaved image with `c` channels from (sw x sh) to
// (dw x dh). Row-major, channel-interleaved.
void ResampleBilinear(const float* src, int sw, int sh, float* dst, int dw, int dh, int c) {
  const float sx = sw > 1 && dw > 1 ? static_cast<float>(sw - 1) / (dw - 1) : 0.f;
  const float sy = sh > 1 && dh > 1 ? static_cast<float>(sh - 1) / (dh - 1) : 0.f;
  for (int y = 0; y < dh; ++y) {
    const float fy = y * sy;
    const int y0 = static_cast<int>(fy);
    const int y1 = std::min(y0 + 1, sh - 1);
    const float wy = fy - y0;
    for (int x = 0; x < dw; ++x) {
      const float fx = x * sx;
      const int x0 = static_cast<int>(fx);
      const int x1 = std::min(x0 + 1, sw - 1);
      const float wx = fx - x0;
      const float* p00 = src + (static_cast<size_t>(y0) * sw + x0) * c;
      const float* p01 = src + (static_cast<size_t>(y0) * sw + x1) * c;
      const float* p10 = src + (static_cast<size_t>(y1) * sw + x0) * c;
      const float* p11 = src + (static_cast<size_t>(y1) * sw + x1) * c;
      float* d = dst + (static_cast<size_t>(y) * dw + x) * c;
      for (int k = 0; k < c; ++k) {
        const float top = p00[k] * (1 - wx) + p01[k] * wx;
        const float bot = p10[k] * (1 - wx) + p11[k] * wx;
        d[k] = top * (1 - wy) + bot * wy;
      }
    }
  }
}

// Nearest-neighbour resample, single channel. Used for the TRIMAP: bilinear
// would invent values between 0, 0.5 and 1 and blur the label boundaries the
// model relies on.
void ResampleNearest(const float* src, int sw, int sh, float* dst, int dw, int dh) {
  for (int y = 0; y < dh; ++y) {
    const int sy = sh > 0 && dh > 0
                       ? std::min(sh - 1, static_cast<int>((static_cast<int64_t>(y) * sh) / dh))
                       : 0;
    for (int x = 0; x < dw; ++x) {
      const int sx = sw > 0 && dw > 0
                         ? std::min(sw - 1, static_cast<int>((static_cast<int64_t>(x) * sw) / dw))
                         : 0;
      dst[static_cast<size_t>(y) * dw + x] = src[static_cast<size_t>(sy) * sw + sx];
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------

struct MatteEngine::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "MEMatte"};
  Ort::AllocatorWithDefaultOptions alloc;

  std::unique_ptr<Ort::Session> backbone;
  std::unique_ptr<Ort::Session> decoder;

  // Resolved I/O names (the graphs are ours, but resolve by name rather than
  // trusting the exporter's ordering).
  std::string bb_in_image, bb_in_tokens, bb_out_features;
  std::string dec_in_features, dec_in_image, dec_out_alpha;

  bool accel_active = false;

  static std::string FindName(Ort::Session& s, bool input, size_t index,
                              Ort::AllocatorWithDefaultOptions& a) {
    Ort::AllocatedStringPtr p = input ? s.GetInputNameAllocated(index, a)
                                      : s.GetOutputNameAllocated(index, a);
    return std::string(p.get());
  }
};

MatteEngine::MatteEngine() : impl_(std::make_unique<Impl>()) {}
MatteEngine::~MatteEngine() = default;

bool MatteEngine::ready() const {
  return impl_ && impl_->backbone && impl_->decoder;
}
bool MatteEngine::accelerator_active() const { return impl_ && impl_->accel_active; }
bool MatteEngine::AcceleratorAvailable() { return hyp::AcceleratorAvailable(); }

bool MatteEngine::Configure(const MatteConfig& cfg) {
  cfg_ = cfg;
  last_error_.clear();
  impl_->backbone.reset();
  impl_->decoder.reset();
  impl_->accel_active = false;

  if (cfg.backbone_path.empty() || cfg.decoder_path.empty()) {
    last_error_ = "Model paths not set (backbone and decoder ONNX files required).";
    return false;
  }

  auto make_options = [&](bool with_accel) {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (cfg.intra_threads > 0) so.SetIntraOpNumThreads(cfg.intra_threads);
    if (with_accel) {
      // RequireStaticInputShapes is deliberately FALSE: the backbone takes the
      // token cap as a runtime input, so its TopK/Gather shapes are dynamic.
      // Demanding static shapes makes CoreML reject the subgraph outright
      // instead of accelerating the parts it can.
      hyp::AppendAccelerator(so, ComputeUnitsString(cfg.compute_units),
                             /*coreml_static=*/false, /*coreml_mlprogram=*/true,
                             cfg.gpu_mem_limit);
    }
    return so;
  };

  const bool want_accel =
      cfg.device != Device::CPU && hyp::AcceleratorAvailable();

  // Offer the accelerator PER GRAPH, because the answer differs by graph.
  //
  // On macOS the backbone's token cap is a runtime input, so its TopK/Gather
  // dimensions are unbounded. CoreML's MIL compiler rejects those outright
  // ("has unbounded dimension which is not supported"); ONNX Runtime then falls
  // back partition by partition, which is both noisy in the host log and
  // measurably SLOWER than simply using CPU. Measured on Apple Silicon,
  // ViT-S at 512x512:
  //     backbone   858 ms CPU   vs  2923 ms with CoreML offered  (3.4x worse)
  //     decoder    546 ms CPU   vs   259 ms with CoreML offered  (2.1x better)
  // The decoder is plain convolutions with no unbounded dims, so it compiles and
  // genuinely benefits. Hence: backbone on CPU, decoder on CoreML.
  //
  // CUDA handles dynamic shapes natively, so on Linux/Windows both graphs are
  // offered the GPU. (Re-measure there before assuming; see docs/DEVELOPMENT.md.)
  auto offer_accelerator_to = [&](bool is_backbone) {
    if (!want_accel) return false;
#ifdef __APPLE__
    return !is_backbone;
#else
    (void)is_backbone;
    return true;
#endif
  };

  bool backbone_accel = false, decoder_accel = false;
  auto open = [&](const std::string& path, bool is_backbone,
                  std::unique_ptr<Ort::Session>* out, bool* on_accel) -> bool {
    const bool offer = offer_accelerator_to(is_backbone);
    if (offer) {
      try {
        Ort::SessionOptions so = make_options(true);
        *out = std::make_unique<Ort::Session>(impl_->env, hyp::OrtPath(path).c_str(), so);
        *on_accel = true;
        return true;
      } catch (const Ort::Exception& e) {
        // The accelerator EP can fail at session-creation time (not at append)
        // if its runtime libraries are missing/incompatible — e.g. on Linux with
        // no CUDA toolkit / cuDNN. Fall through to CPU so the plugin still works.
        last_error_ = std::string(hyp::AcceleratorSubstr()) +
                      " session create failed (falling back to CPU): " + e.what();
      }
    }
    try {
      Ort::SessionOptions so = make_options(false);
      *out = std::make_unique<Ort::Session>(impl_->env, hyp::OrtPath(path).c_str(), so);
      *on_accel = false;
      return true;
    } catch (const Ort::Exception& e) {
      last_error_ = std::string("session create failed: ") + e.what();
      return false;
    }
  };

  if (!open(cfg.backbone_path, /*is_backbone=*/true, &impl_->backbone, &backbone_accel) ||
      !open(cfg.decoder_path, /*is_backbone=*/false, &impl_->decoder, &decoder_accel)) {
    impl_->backbone.reset();
    impl_->decoder.reset();
    return false;
  }

  if (cfg.device == Device::GPU && !hyp::AcceleratorAvailable()) {
    last_error_ = std::string(hyp::AcceleratorSubstr()) +
                  " execution provider is not available in this build; using CPU.";
  }
  // "Accelerated" means at least one graph is on the device. The retry ladder
  // uses this to decide whether a CPU fallback is still available to try.
  impl_->accel_active = backbone_accel || decoder_accel;

  // Resolve I/O names. The backbone takes (image, max_tokens); whichever input is
  // not the image is the token cap.
  try {
    const size_t nin = impl_->backbone->GetInputCount();
    for (size_t i = 0; i < nin; ++i) {
      const std::string n = Impl::FindName(*impl_->backbone, true, i, impl_->alloc);
      if (n == "max_tokens") impl_->bb_in_tokens = n;
      else if (impl_->bb_in_image.empty()) impl_->bb_in_image = n;
    }
    impl_->bb_out_features = Impl::FindName(*impl_->backbone, false, 0, impl_->alloc);

    const size_t dnin = impl_->decoder->GetInputCount();
    for (size_t i = 0; i < dnin; ++i) {
      const std::string n = Impl::FindName(*impl_->decoder, true, i, impl_->alloc);
      if (n == "features") impl_->dec_in_features = n;
      else if (n == "image") impl_->dec_in_image = n;
      else if (impl_->dec_in_features.empty()) impl_->dec_in_features = n;
      else if (impl_->dec_in_image.empty()) impl_->dec_in_image = n;
    }
    impl_->dec_out_alpha = Impl::FindName(*impl_->decoder, false, 0, impl_->alloc);
  } catch (const Ort::Exception& e) {
    last_error_ = std::string("failed to read model I/O names: ") + e.what();
    impl_->backbone.reset();
    impl_->decoder.reset();
    return false;
  }

  if (impl_->bb_in_image.empty() || impl_->bb_in_tokens.empty()) {
    last_error_ =
        "Backbone model does not expose the expected inputs (image, max_tokens). "
        "Re-export with tools/export_mematte.py.";
    impl_->backbone.reset();
    impl_->decoder.reset();
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------

bool MatteEngine::RunOnce(const float* rgb, const float* trimap, int in_w, int in_h,
                          int proc_w, int proc_h, int max_tokens,
                          std::vector<float>* alpha_proc) {
  alpha_proc->clear();
  if (!ready()) {
    last_error_ = "engine not configured";
    return false;
  }

  // 1. Resample to the processing size if it differs from the source.
  std::vector<float> rgb_proc, tri_proc;
  const float* rgb_use = rgb;
  const float* tri_use = trimap;
  if (proc_w != in_w || proc_h != in_h) {
    rgb_proc.resize(static_cast<size_t>(proc_w) * proc_h * 3);
    tri_proc.resize(static_cast<size_t>(proc_w) * proc_h);
    ResampleBilinear(rgb, in_w, in_h, rgb_proc.data(), proc_w, proc_h, 3);
    ResampleNearest(trimap, in_w, in_h, tri_proc.data(), proc_w, proc_h);
    rgb_use = rgb_proc.data();
    tri_use = tri_proc.data();
  }

  // 2. Pad to MEMatte's size divisibility. Upstream normalises FIRST and then
  //    zero-pads, so the padding sits at the normalised origin (the ImageNet
  //    mean colour), not at black. Replicate that ordering exactly.
  const int pw = RoundUpToMultiple(proc_w, kSizeDivisor);
  const int ph = RoundUpToMultiple(proc_h, kSizeDivisor);
  const size_t plane = static_cast<size_t>(pw) * ph;
  std::vector<float> nchw(plane * 4, 0.0f);
  for (int y = 0; y < proc_h; ++y) {
    for (int x = 0; x < proc_w; ++x) {
      const float* s = rgb_use + (static_cast<size_t>(y) * proc_w + x) * 3;
      const size_t idx = static_cast<size_t>(y) * pw + x;
      for (int c = 0; c < 3; ++c)
        nchw[c * plane + idx] = (s[c] - kPixelMean[c]) / kPixelStd[c];
      nchw[3 * plane + idx] = tri_use[static_cast<size_t>(y) * proc_w + x];
    }
  }

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // 3. Backbone -> feature map at 1/16.
  std::vector<Ort::Value> bb_out;
  try {
    std::array<int64_t, 4> shape{1, 4, ph, pw};
    Ort::Value image = Ort::Value::CreateTensor<float>(mem, nchw.data(), nchw.size(),
                                                       shape.data(), shape.size());
    std::array<int64_t, 1> kshape{1};
    std::array<int64_t, 1> kdata{static_cast<int64_t>(std::max(1, max_tokens))};
    Ort::Value tokens = Ort::Value::CreateTensor<int64_t>(mem, kdata.data(), kdata.size(),
                                                          kshape.data(), kshape.size());
    const char* in_names[] = {impl_->bb_in_image.c_str(), impl_->bb_in_tokens.c_str()};
    const char* out_names[] = {impl_->bb_out_features.c_str()};
    Ort::Value ins[] = {std::move(image), std::move(tokens)};
    bb_out = impl_->backbone->Run(Ort::RunOptions{nullptr}, in_names, ins, 2, out_names, 1);
  } catch (const Ort::Exception& e) {
    last_error_ = std::string("backbone inference failed: ") + e.what();
    return false;
  }
  if (bb_out.empty()) {
    last_error_ = "backbone produced no output";
    return false;
  }

  const auto fshape = bb_out[0].GetTensorTypeAndShapeInfo().GetShape();
  if (fshape.size() != 4) {
    last_error_ = "unexpected backbone output rank";
    return false;
  }
  const int fc = static_cast<int>(fshape[1]);
  const int fh = static_cast<int>(fshape[2]);
  const int fw = static_cast<int>(fshape[3]);
  const float* feat = bb_out[0].GetTensorData<float>();

  // 4. Decoder over 512-px tiles with a 64-px halo, mirroring upstream
  //    MEMatte.patch_inference. Peak decoder memory is one tile, independent of
  //    plate size. Tile geometry is computed against the canvas padded up to a
  //    multiple of the tile size; out-of-range reads are zero-filled rather than
  //    materialising a padded copy of the whole image.
  // Tile origins are multiples of 512 and the halo is 64, so every tile bound is
  // a multiple of 16 and maps exactly onto the 1/16 feature grid.
  const int pad_h = RoundUpToMultiple(ph, kDecoderTile);
  const int pad_w = RoundUpToMultiple(pw, kDecoderTile);

  alpha_proc->assign(static_cast<size_t>(proc_w) * proc_h, 0.0f);

  std::vector<float> tile_img, tile_fea;
  for (int ti = 0; ti * kDecoderTile < pad_h; ++ti) {
    const int start_top = ti * kDecoderTile;
    if (start_top >= proc_h) break;  // this tile's output is entirely padding
    const int end_bottom = start_top + kDecoderTile;
    const int top = start_top - kDecoderHalo < 0 ? start_top : start_top - kDecoderHalo;
    const int bottom = end_bottom + kDecoderHalo > pad_h ? end_bottom : end_bottom + kDecoderHalo;
    const int th = bottom - top;

    for (int tj = 0; tj * kDecoderTile < pad_w; ++tj) {
      const int start_left = tj * kDecoderTile;
      if (start_left >= proc_w) break;
      const int end_right = start_left + kDecoderTile;
      const int left = start_left - kDecoderHalo < 0 ? start_left : start_left - kDecoderHalo;
      const int right = end_right + kDecoderHalo > pad_w ? end_right : end_right + kDecoderHalo;
      const int tw = right - left;

      // Image tile [1,4,th,tw], zero-filled outside the padded source.
      tile_img.assign(static_cast<size_t>(th) * tw * 4, 0.0f);
      const size_t tplane = static_cast<size_t>(th) * tw;
      for (int y = 0; y < th; ++y) {
        const int sy = top + y;
        if (sy < 0 || sy >= ph) continue;
        for (int x = 0; x < tw; ++x) {
          const int sx = left + x;
          if (sx < 0 || sx >= pw) continue;
          const size_t src_idx = static_cast<size_t>(sy) * pw + sx;
          const size_t dst_idx = static_cast<size_t>(y) * tw + x;
          for (int c = 0; c < 4; ++c)
            tile_img[c * tplane + dst_idx] = nchw[c * plane + src_idx];
        }
      }

      // Feature tile [1,fc,th/16,tw/16] over the matching region.
      const int ftop = top / kPatchSize, fleft = left / kPatchSize;
      const int fth = th / kPatchSize, ftw = tw / kPatchSize;
      tile_fea.assign(static_cast<size_t>(fth) * ftw * fc, 0.0f);
      const size_t fplane_t = static_cast<size_t>(fth) * ftw;
      const size_t fplane_s = static_cast<size_t>(fh) * fw;
      for (int y = 0; y < fth; ++y) {
        const int sy = ftop + y;
        if (sy < 0 || sy >= fh) continue;
        for (int x = 0; x < ftw; ++x) {
          const int sx = fleft + x;
          if (sx < 0 || sx >= fw) continue;
          for (int c = 0; c < fc; ++c)
            tile_fea[c * fplane_t + static_cast<size_t>(y) * ftw + x] =
                feat[c * fplane_s + static_cast<size_t>(sy) * fw + sx];
        }
      }

      std::vector<Ort::Value> dec_out;
      try {
        std::array<int64_t, 4> ishape{1, 4, th, tw};
        std::array<int64_t, 4> fshape_t{1, fc, fth, ftw};
        Ort::Value vi = Ort::Value::CreateTensor<float>(mem, tile_img.data(),
                                                        tile_img.size(), ishape.data(), 4);
        Ort::Value vf = Ort::Value::CreateTensor<float>(mem, tile_fea.data(),
                                                        tile_fea.size(), fshape_t.data(), 4);
        const char* in_names[] = {impl_->dec_in_features.c_str(),
                                  impl_->dec_in_image.c_str()};
        const char* out_names[] = {impl_->dec_out_alpha.c_str()};
        Ort::Value ins[] = {std::move(vf), std::move(vi)};
        dec_out = impl_->decoder->Run(Ort::RunOptions{nullptr}, in_names, ins, 2, out_names, 1);
      } catch (const Ort::Exception& e) {
        last_error_ = std::string("decoder inference failed: ") + e.what();
        alpha_proc->clear();
        return false;
      }
      if (dec_out.empty()) {
        last_error_ = "decoder produced no output";
        alpha_proc->clear();
        return false;
      }
      const float* a = dec_out[0].GetTensorData<float>();

      // Copy the tile's interior (the halo is context only) into the result.
      const int out_top = (start_top - kDecoderHalo < 0) ? 0 : kDecoderHalo;
      const int out_left = (start_left - kDecoderHalo < 0) ? 0 : kDecoderHalo;
      for (int y = 0; y < kDecoderTile; ++y) {
        const int dy = start_top + y;
        if (dy >= proc_h) break;
        const int sy = out_top + y;
        if (sy >= th) break;
        for (int x = 0; x < kDecoderTile; ++x) {
          const int dx = start_left + x;
          if (dx >= proc_w) break;
          const int sx = out_left + x;
          if (sx >= tw) break;
          (*alpha_proc)[static_cast<size_t>(dy) * proc_w + dx] =
              a[static_cast<size_t>(sy) * tw + sx];
        }
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------

MatteResult MatteEngine::Run(const float* rgb, const float* trimap, int in_w, int in_h,
                             const MatteRequest& req) {
  MatteResult out;
  out.width = in_w;
  out.height = in_h;
  if (in_w <= 0 || in_h <= 0 || !rgb || !trimap) {
    last_error_ = "invalid input buffer";
    return out;
  }

  // Starting processing size: native unless the caller asked for a smaller one.
  const int src_long = std::max(in_w, in_h);
  int proc_w = in_w, proc_h = in_h;
  if (req.proc_long_side > 0 && req.proc_long_side < src_long) {
    const double s = static_cast<double>(req.proc_long_side) / src_long;
    proc_w = std::max(kSizeDivisor, static_cast<int>(std::lround(in_w * s)));
    proc_h = std::max(kSizeDivisor, static_cast<int>(std::lround(in_h * s)));
  }
  int tokens = std::min(std::max(req.max_tokens, kMinTokens), kMaxTokens);

  std::vector<float> alpha_proc;
  std::string degraded;
  bool ok = false;

  // Retry ladder. ORT surfaces an accelerator OOM as an Ort::Exception, which
  // RunOnce turns into a failed return with last_error_ set. Degrade in the order
  // that costs the least quality first: token cap, then resolution, then device.
  for (int attempt = 0; attempt < 12 && !ok; ++attempt) {
    ok = RunOnce(rgb, trimap, in_w, in_h, proc_w, proc_h, tokens, &alpha_proc);
    if (ok) break;

    const bool oom = IsOutOfMemoryError(last_error_);
    if (!oom) break;  // a real error, not memory pressure — do not thrash

    if (tokens > kMinTokens) {
      tokens = std::max(kMinTokens, tokens / 2);
      degraded = "reduced the token cap to " + std::to_string(tokens) +
                 " to fit available memory";
      continue;
    }
    if (std::max(proc_w, proc_h) > 512) {
      proc_w = std::max(kSizeDivisor, proc_w / 2);
      proc_h = std::max(kSizeDivisor, proc_h / 2);
      degraded = "reduced the processing resolution to " + std::to_string(proc_w) + "x" +
                 std::to_string(proc_h) + " to fit available memory (matte detail is "
                 "softened; give the GPU more headroom to avoid this)";
      continue;
    }
    if (impl_->accel_active) {
      MatteConfig cpu = cfg_;
      cpu.device = Device::CPU;
      if (!Configure(cpu)) break;
      tokens = std::min(std::max(req.max_tokens, kMinTokens), kMaxTokens);
      proc_w = in_w;
      proc_h = in_h;
      degraded = "fell back to CPU inference (slower) because the GPU ran out of memory";
      continue;
    }
    break;
  }

  if (!ok) return out;

  // Upsample the alpha back to the source resolution if we processed smaller.
  std::vector<float> alpha_full;
  if (proc_w != in_w || proc_h != in_h) {
    alpha_full.resize(static_cast<size_t>(in_w) * in_h);
    ResampleBilinear(alpha_proc.data(), proc_w, proc_h, alpha_full.data(), in_w, in_h, 1);
  } else {
    alpha_full = std::move(alpha_proc);
  }

  // Composite against the FULL-RESOLUTION trimap. Upstream forces alpha to 0/1 in
  // the known regions; doing it at source resolution means a downscaled inference
  // only ever softens the unknown band, never the hard foreground/background edges.
  for (size_t i = 0, n = alpha_full.size(); i < n; ++i) {
    const float t = trimap[i];
    if (t <= 0.0f) alpha_full[i] = 0.0f;
    else if (t >= 1.0f) alpha_full[i] = 1.0f;
    else alpha_full[i] = std::min(std::max(alpha_full[i], 0.0f), 1.0f);
  }

  out.alpha = std::move(alpha_full);
  out.used_tokens = tokens;
  out.proc_width = proc_w;
  out.proc_height = proc_h;
  out.used_cpu = !impl_->accel_active;
  out.degraded = degraded;
  if (!degraded.empty()) last_error_.clear();
  return out;
}

}  // namespace hyp
