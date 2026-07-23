// Copyright the Hypnos authors.
// SPDX-License-Identifier: Apache-2.0
//
// MEMatte OpenFX plugin.
//
// Trimap-guided alpha matting via MEMatte (AAAI 2025) on ONNX Runtime, with the
// platform accelerator (CoreML on macOS, CUDA on Linux/Windows) and CPU fallback.
//
// Input and output are always at the SAME resolution. Inference cost is governed
// by MEMatte's global-attention token cap, which bounds memory without discarding
// image detail; a reduced processing resolution is used only as a last resort
// when the measured free memory cannot fit the plate even at the minimum cap.
//
// If built without ONNX Runtime (HYP_WITH_ONNX undefined), render() is a passthrough.

#ifdef _WIN32
// CMake already defines NOMINMAX for this target; guard so a direct
// compile also works without warning C4005 (macro redefinition).
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if HYP_WITH_ONNX && (defined(__APPLE__) || defined(__linux__))
#include <dlfcn.h>
#include <sys/stat.h>
#endif

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
// Lower-case .h: that is the real filename in the OpenFX SDK. The historical
// "ofxsProcessing.H" spelling only works on case-insensitive filesystems.
#include "ofxsProcessing.h"

#include "Color.h"
#if HYP_WITH_ONNX
#include "MatteEngine.h"
#include "MemProbe.h"
#include "Trimap.h"
#endif

#define kPluginName "MEMatte"
#define kPluginGrouping "Tokgan"
#define kPluginDescription \
  "Extract an alpha matte from an image and a trimap using MEMatte via ONNX " \
  "Runtime with hardware acceleration. Output matches the input resolution; " \
  "memory use is capped by the global-attention token limit."
#define kPluginIdentifier "com.tokgan.openfx.MEMatte"
#define kPluginVersionMajor 0
#define kPluginVersionMinor 1

#define kClipTrimap "Trimap"

#define kParamModelVariant "modelVariant"
#define kParamBackboneFile "backboneFile"
#define kParamDecoderFile "decoderFile"
#define kParamInputACEScg "inputIsACEScg"
#define kParamTrimapSource "trimapSource"
#define kParamTrimapEncoding "trimapEncoding"
#define kParamTrimapTolerance "trimapTolerance"
#define kParamTrimapInvert "trimapInvert"
#define kParamOutputMode "outputMode"
#define kParamMaxTokens "maxTokens"
#define kParamAutoBudget "autoBudget"
#define kParamMemBudget "memBudgetMB"
#define kParamHeadroom "memHeadroom"
#define kParamDevice "device"
#define kParamComputeUnits "computeUnits"
#define kParamThreads "intraThreads"
#define kParamStatus "status"
#define kParamRefresh "refreshStatus"

enum OutputMode { eOutputMatte = 0, eOutputRGBAUnpremult, eOutputRGBAPremult };
enum TrimapSource { eTrimapFromClip = 0, eTrimapFromSourceAlpha };
enum ModelVariant { eModelViTS = 0, eModelViTB };

static int EmbedDimFor(int variant) { return variant == eModelViTB ? 768 : 384; }

////////////////////////////////////////////////////////////////////////////////
// Passthrough copy processor (used when ONNX is unavailable or a render fails).

class CopyBase : public OFX::ImageProcessor {
 protected:
  OFX::Image* _srcImg;
 public:
  explicit CopyBase(OFX::ImageEffect& e) : OFX::ImageProcessor(e), _srcImg(nullptr) {}
  void setSrcImg(OFX::Image* v) { _srcImg = v; }
};

template <class PIX, int nComponents>
class CopyProcessor : public CopyBase {
 public:
  explicit CopyProcessor(OFX::ImageEffect& e) : CopyBase(e) {}
  void multiThreadProcessImages(OfxRectI w) override {
    for (int y = w.y1; y < w.y2; ++y) {
      if (_effect.abort()) break;
      PIX* dst = static_cast<PIX*>(_dstImg->getPixelAddress(w.x1, y));
      if (!dst) continue;
      for (int x = w.x1; x < w.x2; ++x) {
        const PIX* src =
            static_cast<const PIX*>(_srcImg ? _srcImg->getPixelAddress(x, y) : nullptr);
        if (src)
          for (int c = 0; c < nComponents; ++c) dst[c] = src[c];
        else
          for (int c = 0; c < nComponents; ++c) dst[c] = PIX(0);
        dst += nComponents;
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Best-effort user messages. DaVinci Resolve's render context does not support the
// OFX persistent-message suite; calling set/clearPersistentMessage there THROWS,
// which unwinds out of the render action and the host reports the whole render as
// failed. Wrap the calls so a missing/unsupported message suite can never fail a
// render (the message is simply dropped on such hosts).
static void SafeSetMessage(OFX::ImageEffect& e, OFX::Message::MessageTypeEnum type,
                           const std::string& msg) {
  try { e.setPersistentMessage(type, "", msg); } catch (...) {}
}
static void SafeClearMessage(OFX::ImageEffect& e) {
  try { e.clearPersistentMessage(); } catch (...) {}
}

////////////////////////////////////////////////////////////////////////////////

class MEMattePlugin : public OFX::ImageEffect {
 protected:
  OFX::Clip* _dstClip;
  OFX::Clip* _srcClip;
  OFX::Clip* _trimapClip;

  OFX::ChoiceParam* _modelVariant;
  OFX::StringParam* _backboneFile;
  OFX::StringParam* _decoderFile;
  OFX::BooleanParam* _inputACEScg;
  OFX::ChoiceParam* _trimapSource;
  OFX::ChoiceParam* _trimapEncoding;
  OFX::DoubleParam* _trimapTolerance;
  OFX::BooleanParam* _trimapInvert;
  OFX::ChoiceParam* _outputMode;
  OFX::IntParam* _maxTokens;
  OFX::BooleanParam* _autoBudget;
  OFX::IntParam* _memBudget;
  OFX::DoubleParam* _headroom;
  OFX::ChoiceParam* _device;
  OFX::ChoiceParam* _computeUnits;
  OFX::IntParam* _threads;
  OFX::StringParam* _status;

 public:
  explicit MEMattePlugin(OfxImageEffectHandle handle) : OFX::ImageEffect(handle) {
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    _trimapClip = fetchClip(kClipTrimap);

    _modelVariant = fetchChoiceParam(kParamModelVariant);
    _backboneFile = fetchStringParam(kParamBackboneFile);
    _decoderFile = fetchStringParam(kParamDecoderFile);
    _inputACEScg = fetchBooleanParam(kParamInputACEScg);
    _trimapSource = fetchChoiceParam(kParamTrimapSource);
    _trimapEncoding = fetchChoiceParam(kParamTrimapEncoding);
    _trimapTolerance = fetchDoubleParam(kParamTrimapTolerance);
    _trimapInvert = fetchBooleanParam(kParamTrimapInvert);
    _outputMode = fetchChoiceParam(kParamOutputMode);
    _maxTokens = fetchIntParam(kParamMaxTokens);
    _autoBudget = fetchBooleanParam(kParamAutoBudget);
    _memBudget = fetchIntParam(kParamMemBudget);
    _headroom = fetchDoubleParam(kParamHeadroom);
    _device = fetchChoiceParam(kParamDevice);
    _computeUnits = fetchChoiceParam(kParamComputeUnits);
    _threads = fetchIntParam(kParamThreads);
    _status = fetchStringParam(kParamStatus);
  }

  void render(const OFX::RenderArguments& args) override;
  void getClipPreferences(OFX::ClipPreferencesSetter& clipPreferences) override;
  void getRegionsOfInterest(const OFX::RegionsOfInterestArguments& args,
                            OFX::RegionOfInterestSetter& rois) override;
  void changedParam(const OFX::InstanceChangedArgs& args,
                    const std::string& paramName) override;

 private:
  void renderPassthrough(const OFX::RenderArguments& args);
#if HYP_WITH_ONNX
  bool renderMatte(const OFX::RenderArguments& args);
  bool resolveModelPaths(std::string* backbone, std::string* decoder, std::string* err);
  std::string buildStatusText();

  std::mutex _engineMutex;
  std::unique_ptr<hyp::MatteEngine> _engine;
  std::string _engineKey;
#endif
};

////////////////////////////////////////////////////////////////////////////////

#if HYP_WITH_ONNX

// Locate the plugin bundle's Contents/Resources directory by finding this
// binary's own path (dladdr / GetModuleHandleEx). Empty if it cannot be found.
static std::string bundleResourceDir() {
#if defined(_WIN32)
  HMODULE hm = nullptr;
  if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCSTR>(&bundleResourceDir), &hm)) {
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(hm, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
      std::string p(buf, n);  // ...\Contents\Win64\MEMatte.ofx
      // Match "\Contents\" CASE-INSENSITIVELY: hosts load the bundle with whatever
      // case they built the path in (Nuke uses lowercase "...\Contents\win64\"),
      // and GetModuleFileNameA returns that load-time case. A case-sensitive rfind
      // would miss and the bundled models would appear absent.
      std::string lower(p);
      for (char& c : lower)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
      const auto pos = lower.rfind("\\contents\\");
      if (pos != std::string::npos) return p.substr(0, pos) + "\\Contents\\Resources\\";
    }
  }
#elif defined(__APPLE__) || defined(__linux__)
#if defined(__APPLE__)
  const std::string marker = "/Contents/MacOS/";
#else
  const std::string marker = "/Contents/Linux-x86-64/";
#endif
  Dl_info info;
  if (dladdr(reinterpret_cast<const void*>(&bundleResourceDir), &info) && info.dli_fname) {
    std::string p(info.dli_fname);
    const auto pos = p.rfind(marker);
    if (pos != std::string::npos) return p.substr(0, pos) + "/Contents/Resources/";
  }
#endif
  return std::string();
}

static bool fileExists(const std::string& p) {
  if (p.empty()) return false;
#if defined(_WIN32)
  return GetFileAttributesA(p.c_str()) != INVALID_FILE_ATTRIBUTES;
#else
  struct stat st;
  return stat(p.c_str(), &st) == 0;
#endif
}

bool MEMattePlugin::resolveModelPaths(std::string* backbone, std::string* decoder,
                                      std::string* err) {
  int variant = eModelViTS;
  _modelVariant->getValue(variant);
  const std::string tag = (variant == eModelViTB) ? "b" : "s";

  _backboneFile->getValue(*backbone);
  _decoderFile->getValue(*decoder);
  if (!backbone->empty() && !decoder->empty()) return true;

  // Directory override, then the bundle's own Resources.
  std::string dir;
  if (const char* env = std::getenv("MEMATTE_MODEL_DIR")) {
    dir = env;
    if (!dir.empty() && dir.back() != '/' && dir.back() != '\\') dir += "/";
  }
  if (dir.empty() || !fileExists(dir + "mematte_" + tag + "_backbone.onnx"))
    dir = bundleResourceDir();

  if (backbone->empty()) *backbone = dir + "mematte_" + tag + "_backbone.onnx";
  if (decoder->empty()) *decoder = dir + "mematte_" + tag + "_decoder.onnx";

  if (!fileExists(*backbone) || !fileExists(*decoder)) {
    *err = "MEMatte ONNX models not found (looked for mematte_" + tag +
           "_{backbone,decoder}.onnx). Run the bundle's fetch_models script, set "
           "$MEMATTE_MODEL_DIR, or set the Backbone/Decoder file parameters.";
    return false;
  }
  return true;
}

// Decide the token cap and processing resolution for this frame, given the
// memory budget. Prefers the token cap (no quality cost) over downscaling.
struct Plan {
  int tokens = 0;
  int long_side = 0;  // 0 = native resolution
  std::string note;
};

static Plan PlanInference(int W, int H, int embed_dim, int user_tokens,
                          uint64_t budget_bytes) {
  Plan p;
  p.tokens = user_tokens;
  if (budget_bytes == 0) return p;  // uncapped: honour the user's request as-is

  // A cap above the number of tokens that exist is not a cap at all, so compare
  // against the effective request. Without this, a small plate reports "cap
  // lowered" every frame even though nothing was actually constrained.
  const int total_tokens =
      std::max(1, W / hyp::kPatchSize) * std::max(1, H / hyp::kPatchSize);
  const int effective_request = std::min(user_tokens, total_tokens);

  const int fit = hyp::TokensForBudget(W, H, embed_dim, budget_bytes);
  if (fit >= effective_request) return p;  // the request already fits

  if (fit > 0) {
    p.tokens = fit;
    p.note = "token cap lowered to " + std::to_string(fit) + " to fit the memory budget";
    return p;
  }

  // Even the minimum cap does not fit at native resolution: downscale. This is
  // the only step that costs matte detail, so it is last and it is announced.
  p.tokens = hyp::kMinTokens;
  const int ls = hyp::LongSideForBudget(W, H, embed_dim, budget_bytes);
  if (ls > 0 && ls < std::max(W, H)) {
    p.long_side = ls;
    p.note = "processing resolution reduced to " + std::to_string(ls) +
             " px long side (matte detail softened) - the memory budget cannot fit "
             "this plate at full resolution";
  } else {
    p.note = "memory budget is very small; running at the minimum token cap";
  }
  return p;
}

std::string MEMattePlugin::buildStatusText() {
  const hyp::MemInfo mi = hyp::ProbeMemory();
  std::string s = hyp::DescribeMemory(mi);

  int variant = eModelViTS, tokens = 18500, dev = 0, budget_mb = 4096;
  bool autoBudget = true;
  double headroom = 20.0;
  _modelVariant->getValue(variant);
  _maxTokens->getValue(tokens);
  _device->getValue(dev);
  _autoBudget->getValue(autoBudget);
  _memBudget->getValue(budget_mb);
  _headroom->getValue(headroom);

  uint64_t budget = 0;
  const bool gpu = mi.gpu_valid && dev != 2 /*CPU*/;
  if (autoBudget) {
    const uint64_t avail = gpu ? mi.gpu_free : mi.sys_avail;
    const double keep = 1.0 - std::min(std::max(headroom, 0.0), 90.0) / 100.0;
    budget = static_cast<uint64_t>(avail * keep);
  } else {
    budget = static_cast<uint64_t>(budget_mb) * 1024ull * 1024ull;
  }
  s += "\nBudget: " + hyp::FormatBytes(budget) + (gpu ? " (GPU)" : " (system RAM)");

  if (_srcClip && _srcClip->isConnected()) {
    // getRegionOfDefinition returns CANONICAL coordinates; convert to pixels by
    // dividing x by the pixel aspect ratio (render scale is 1 outside a render).
    const OfxRectD rod = _srcClip->getRegionOfDefinition(0.0);
    const double par = _srcClip->getPixelAspectRatio();
    const int W = static_cast<int>(std::lround((rod.x2 - rod.x1) / (par > 0.0 ? par : 1.0)));
    const int H = static_cast<int>(std::lround(rod.y2 - rod.y1));
    if (W > 0 && H > 0) {
      const Plan p = PlanInference(W, H, EmbedDimFor(variant), tokens, budget);
      s += "\nPlan for " + std::to_string(W) + "x" + std::to_string(H) + ": " +
           std::to_string(p.tokens) + " tokens, " +
           (p.long_side ? std::to_string(p.long_side) + " px long side"
                        : std::string("native resolution"));
      s += "\nEstimated peak: " +
           hyp::FormatBytes(hyp::EstimatePeakBytes(W, H, p.tokens, EmbedDimFor(variant)));
      if (!p.note.empty()) s += "\nNote: " + p.note;
    }
  }
  return s;
}

bool MEMattePlugin::renderMatte(const OFX::RenderArguments& args) {
  std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
  std::unique_ptr<OFX::Image> src(_srcClip->fetchImage(args.time));
  if (!dst.get() || !src.get()) return false;

  const OFX::BitDepthEnum sbd = src->getPixelDepth();
  const OFX::BitDepthEnum dbd = dst->getPixelDepth();
  const int sComps = hyp::ComponentCount(src->getPixelComponents());
  const int dComps = hyp::ComponentCount(dst->getPixelComponents());
  if (sComps <= 0 || dComps <= 0) return false;
  const bool srcInteger = hyp::IsIntegerDepth(sbd);

  const OfxRectI rod = src->getRegionOfDefinition();
  const int W = rod.x2 - rod.x1;
  const int H = rod.y2 - rod.y1;
  if (W <= 0 || H <= 0) return false;

  std::string backbone, decoder, err;
  if (!resolveModelPaths(&backbone, &decoder, &err)) {
    SafeSetMessage(*this, OFX::Message::eMessageError, err);
    return false;
  }

  // ---- parameters -------------------------------------------------------
  int variant = eModelViTS, outMode = eOutputMatte, triSrc = eTrimapFromClip;
  int triEnc = 0, dev = 0, cu = 0, threads = 0, tokens = 18500, budget_mb = 4096;
  bool acescg = true, triInvert = false, autoBudget = true;
  double triTol = 0.05, headroom = 20.0;
  _modelVariant->getValue(variant);
  _outputMode->getValue(outMode);
  _trimapSource->getValue(triSrc);
  _trimapEncoding->getValue(triEnc);
  _trimapTolerance->getValue(triTol);
  _trimapInvert->getValue(triInvert);
  _inputACEScg->getValue(acescg);
  _device->getValue(dev);
  _computeUnits->getValue(cu);
  _threads->getValue(threads);
  _maxTokens->getValue(tokens);
  _autoBudget->getValue(autoBudget);
  _memBudget->getValue(budget_mb);
  _headroom->getValue(headroom);

  // ---- memory budget ----------------------------------------------------
  const hyp::MemInfo mi = hyp::ProbeMemory();
  const bool gpu_path = mi.gpu_valid && dev != 2;
  uint64_t budget = 0;
  if (autoBudget) {
    const uint64_t avail = gpu_path ? mi.gpu_free : mi.sys_avail;
    const double keep = 1.0 - std::min(std::max(headroom, 0.0), 90.0) / 100.0;
    budget = static_cast<uint64_t>(avail * keep);
  } else {
    budget = static_cast<uint64_t>(budget_mb) * 1024ull * 1024ull;
  }
  const Plan plan = PlanInference(W, H, EmbedDimFor(variant), tokens, budget);

  // ---- engine (cached across frames) ------------------------------------
  std::lock_guard<std::mutex> lock(_engineMutex);

  hyp::MatteConfig cfg;
  cfg.backbone_path = backbone;
  cfg.decoder_path = decoder;
  cfg.device = dev == 1 ? hyp::Device::GPU : (dev == 2 ? hyp::Device::CPU : hyp::Device::Auto);
  cfg.compute_units = static_cast<hyp::ComputeUnits>(cu);
  cfg.intra_threads = threads;
  // Only the CUDA EP honours a byte cap; harmless elsewhere.
  cfg.gpu_mem_limit = gpu_path ? budget : 0;

  const std::string key = backbone + "|" + decoder + "|" + std::to_string(dev) + "|" +
                          std::to_string(cu) + "|" + std::to_string(threads) + "|" +
                          std::to_string(cfg.gpu_mem_limit);
  if (!_engine || key != _engineKey) {
    _engine = std::make_unique<hyp::MatteEngine>();
    if (!_engine->Configure(cfg)) {
      SafeSetMessage(*this, OFX::Message::eMessageError, _engine->last_error());
      _engine.reset();
      _engineKey.clear();
      return false;
    }
    _engineKey = key;
  }

  // ---- build model inputs ----------------------------------------------
  std::unique_ptr<OFX::Image> tri;
  const bool useTrimapClip =
      triSrc == eTrimapFromClip && _trimapClip && _trimapClip->isConnected();
  if (useTrimapClip) tri.reset(_trimapClip->fetchImage(args.time));
  const OFX::BitDepthEnum tbd = tri.get() ? tri->getPixelDepth() : sbd;
  const int tComps = tri.get() ? hyp::ComponentCount(tri->getPixelComponents()) : 0;

  const hyp::TrimapEncoding enc = static_cast<hyp::TrimapEncoding>(triEnc);

  std::vector<float> rgb(static_cast<size_t>(W) * H * 3);
  std::vector<float> trimap(static_cast<size_t>(W) * H);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const size_t i = static_cast<size_t>(y) * W + x;
      float* d = rgb.data() + i * 3;

      const void* s = src->getPixelAddress(rod.x1 + x, rod.y1 + y);
      float px[4];
      hyp::ReadPixel(s, sbd, sComps, px);
      // ACEScg -> sRGB applies only to linear float input; integer input is
      // already display-referred sRGB (the 8-bit 0-255 Rec.709 assumption).
      if (acescg && !srcInteger) {
        hyp::AcescgToSrgb(px[0], px[1], px[2], d);
      } else {
        d[0] = hyp::Clamp01(px[0]);
        d[1] = hyp::Clamp01(px[1]);
        d[2] = hyp::Clamp01(px[2]);
      }

      // Trimap is DATA: read the raw value with no colour conversion, then
      // classify. When there is no trimap clip we use the source alpha.
      float raw;
      if (tri.get()) {
        // Indexed in the SOURCE RoD. Where the trimap clip is smaller,
        // getPixelAddress returns null and ReadPixel yields 0, so anything
        // outside the trimap's own extent is treated as background - which is
        // the sane reading of "not covered by the trimap".
        const void* tp = tri->getPixelAddress(rod.x1 + x, rod.y1 + y);
        float tpx[4];
        hyp::ReadPixel(tp, tbd, tComps, tpx);
        // ReadPixel puts a 1-component value in every slot, so R is correct for
        // both a single-channel and an RGB(A) trimap.
        raw = tpx[0];
      } else {
        raw = px[3];
      }
      trimap[i] = hyp::DecodeTrimap(raw, enc, static_cast<float>(triTol), triInvert);
    }
  }

  // ---- run --------------------------------------------------------------
  hyp::MatteRequest req;
  req.max_tokens = plan.tokens;
  req.proc_long_side = plan.long_side;
  hyp::MatteResult res = _engine->Run(rgb.data(), trimap.data(), W, H, req);
  if (!res.ok()) {
    SafeSetMessage(*this, OFX::Message::eMessageError,
                   _engine->last_error().empty() ? "Matting inference failed"
                                                 : _engine->last_error());
    return false;
  }

  // Report degradation, but never fail the render for it.
  std::string note = plan.note;
  if (!res.degraded.empty()) note = note.empty() ? res.degraded : note + "; " + res.degraded;
  if (!note.empty()) {
    SafeSetMessage(*this, OFX::Message::eMessageWarning, "MEMatte: " + note);
  } else {
    SafeClearMessage(*this);
  }

  // ---- write ------------------------------------------------------------
  const OfxRectI w = args.renderWindow;
  for (int y = w.y1; y < w.y2; ++y) {
    if (abort()) break;
    for (int x = w.x1; x < w.x2; ++x) {
      void* dpix = dst->getPixelAddress(x, y);
      if (!dpix) continue;
      const int lx = x - rod.x1, ly = y - rod.y1;
      float a = 0.0f;
      if (lx >= 0 && lx < W && ly >= 0 && ly < H)
        a = res.alpha[static_cast<size_t>(ly) * W + lx];

      float outpx[4];
      if (outMode == eOutputMatte) {
        // R=G=B=A so the matte reads correctly whether the host gives us a
        // single-channel or an RGBA destination.
        outpx[0] = outpx[1] = outpx[2] = outpx[3] = a;
      } else {
        // Source RGB in its ORIGINAL colour space (not the sRGB fed to the model).
        float px[4] = {0.f, 0.f, 0.f, 1.f};
        hyp::ReadPixel(src->getPixelAddress(x, y), sbd, sComps, px);
        const float m = (outMode == eOutputRGBAPremult) ? a : 1.0f;
        outpx[0] = px[0] * m;
        outpx[1] = px[1] * m;
        outpx[2] = px[2] * m;
        outpx[3] = a;
      }
      hyp::WritePixel(dpix, dbd, dComps, outpx);
    }
  }
  return true;
}

#endif  // HYP_WITH_ONNX

void MEMattePlugin::renderPassthrough(const OFX::RenderArguments& args) {
  std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
  if (!dst.get()) OFX::throwSuiteStatusException(kOfxStatFailed);
  const OFX::BitDepthEnum bd = dst->getPixelDepth();
  const OFX::PixelComponentEnum comp = dst->getPixelComponents();
  std::unique_ptr<OFX::Image> src(_srcClip->fetchImage(args.time));

  auto run = [&](CopyBase& p) {
    p.setSrcImg(src.get());
    p.setDstImg(dst.get());
    p.setRenderWindow(args.renderWindow);
    p.process();
  };
  if (comp == OFX::ePixelComponentRGBA) {
    if (bd == OFX::eBitDepthFloat) { CopyProcessor<float, 4> p(*this); run(p); }
    else if (bd == OFX::eBitDepthUShort) { CopyProcessor<unsigned short, 4> p(*this); run(p); }
    else { CopyProcessor<unsigned char, 4> p(*this); run(p); }
  } else if (comp == OFX::ePixelComponentRGB) {
    if (bd == OFX::eBitDepthFloat) { CopyProcessor<float, 3> p(*this); run(p); }
    else if (bd == OFX::eBitDepthUShort) { CopyProcessor<unsigned short, 3> p(*this); run(p); }
    else { CopyProcessor<unsigned char, 3> p(*this); run(p); }
  } else {
    if (bd == OFX::eBitDepthFloat) { CopyProcessor<float, 1> p(*this); run(p); }
    else if (bd == OFX::eBitDepthUShort) { CopyProcessor<unsigned short, 1> p(*this); run(p); }
    else { CopyProcessor<unsigned char, 1> p(*this); run(p); }
  }
}

void MEMattePlugin::render(const OFX::RenderArguments& args) {
#if HYP_WITH_ONNX
  if (renderMatte(args)) return;
  // fall through to passthrough on failure so the graph still renders
#endif
  renderPassthrough(args);
}

void MEMattePlugin::getClipPreferences(OFX::ClipPreferencesSetter& clipPreferences) {
  int outMode = eOutputMatte;
  _outputMode->getValue(outMode);
  if (outMode == eOutputMatte) {
    // A single-channel matte is the default output. Hosts that cannot carry an
    // Alpha-only clip will hand us RGBA instead; render() writes R=G=B=A so the
    // result is correct either way.
    clipPreferences.setClipComponents(*_dstClip, OFX::ePixelComponentAlpha);
  } else {
    clipPreferences.setClipComponents(*_dstClip, OFX::ePixelComponentRGBA);
    clipPreferences.setOutputPremultiplication(outMode == eOutputRGBAPremult
                                                   ? OFX::eImagePreMultiplied
                                                   : OFX::eImageUnPreMultiplied);
  }
}

void MEMattePlugin::getRegionsOfInterest(const OFX::RegionsOfInterestArguments& args,
                                         OFX::RegionOfInterestSetter& rois) {
  // Matting needs the whole frame (global attention spans the image), so ask for
  // the full RoD of both inputs no matter how small the requested output region.
  if (_srcClip && _srcClip->isConnected())
    rois.setRegionOfInterest(*_srcClip, _srcClip->getRegionOfDefinition(args.time));
  if (_trimapClip && _trimapClip->isConnected())
    rois.setRegionOfInterest(*_trimapClip, _trimapClip->getRegionOfDefinition(args.time));
}

void MEMattePlugin::changedParam(const OFX::InstanceChangedArgs& /*args*/,
                                 const std::string& paramName) {
#if HYP_WITH_ONNX
  // The status text is refreshed on any user edit (and on demand). It is never
  // written from render(): OFX forbids setting parameters during a render action.
  if (paramName == kParamStatus) return;
  if (_status) {
    try { _status->setValue(buildStatusText()); } catch (...) {}
  }
#else
  (void)paramName;
#endif
}

////////////////////////////////////////////////////////////////////////////////

mDeclarePluginFactory(MEMatteFactory, {}, {});

using namespace OFX;

void MEMatteFactory::describe(OFX::ImageEffectDescriptor& desc) {
  desc.setLabels(kPluginName, kPluginName, kPluginName);
  desc.setPluginGrouping(kPluginGrouping);
  desc.setPluginDescription(kPluginDescription);

  desc.addSupportedContext(eContextFilter);
  desc.addSupportedContext(eContextGeneral);
  desc.addSupportedBitDepth(eBitDepthFloat);
  desc.addSupportedBitDepth(eBitDepthUByte);
  desc.addSupportedBitDepth(eBitDepthUShort);

  desc.setSingleInstance(false);
  desc.setHostFrameThreading(false);
  desc.setSupportsMultiResolution(true);
  // Matting inference needs the whole image: no tiling.
  desc.setSupportsTiles(false);
  desc.setTemporalClipAccess(false);
  desc.setRenderTwiceAlways(false);
  desc.setSupportsMultipleClipPARs(false);
}

void MEMatteFactory::describeInContext(OFX::ImageEffectDescriptor& desc,
                                       OFX::ContextEnum /*context*/) {
  ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
  src->addSupportedComponent(ePixelComponentRGBA);
  src->addSupportedComponent(ePixelComponentRGB);
  src->setTemporalClipAccess(false);
  src->setSupportsTiles(false);
  src->setIsMask(false);

  ClipDescriptor* tri = desc.defineClip(kClipTrimap);
  tri->addSupportedComponent(ePixelComponentAlpha);
  tri->addSupportedComponent(ePixelComponentRGBA);
  tri->addSupportedComponent(ePixelComponentRGB);
  tri->setTemporalClipAccess(false);
  tri->setSupportsTiles(false);
  tri->setIsMask(false);
  // Optional so the node still works from the source alpha alone.
  tri->setOptional(true);

  ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
  dst->addSupportedComponent(ePixelComponentRGBA);
  dst->addSupportedComponent(ePixelComponentAlpha);
  dst->setSupportsTiles(false);

  PageParamDescriptor* page = desc.definePageParam("Controls");

  {
    ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamModelVariant);
    p->setLabel("Model");
    p->setHint("MEMatte backbone. ViT-S is faster and lighter; ViT-B is more accurate "
               "and roughly twice the memory.");
    p->appendOption("MEMatte ViT-S (fast)");
    p->appendOption("MEMatte ViT-B (accurate)");
    p->setDefault(eModelViTS);
    page->addChild(*p);
  }
  {
    ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamOutputMode);
    p->setLabel("Output");
    p->setHint("Matte: a single-channel alpha. The RGBA modes attach the matte to the "
               "source RGB, premultiplied or not.");
    p->appendOption("Matte (single channel)");
    p->appendOption("RGBA (unpremultiplied)");
    p->appendOption("RGBA (premultiplied)");
    p->setDefault(eOutputMatte);
    page->addChild(*p);
  }

  // ---- Trimap -----------------------------------------------------------
  {
    ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamTrimapSource);
    p->setLabel("Trimap from");
    p->setHint("Where the trimap comes from. The Trimap input is used when connected; "
               "otherwise the source alpha can act as the trimap.");
    p->appendOption("Trimap input");
    p->appendOption("Source alpha");
    p->setDefault(eTrimapFromClip);
    page->addChild(*p);
  }
  {
    ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamTrimapEncoding);
    p->setLabel("Trimap encoding");
    p->setHint("How trimap values map to background / unknown / foreground. "
               "Auto only needs to recognise near-black and near-white, so it does not "
               "matter whether mid grey arrived as 0.5, 128/255, or an sRGB-decoded "
               "0.214 - anything that is not near an extreme is treated as unknown.");
    p->appendOption("Auto (near-black / near-white)");
    p->appendOption("Float 0 / 0.5 / 1");
    p->appendOption("Integer 0 / 128 / 255");
    p->appendOption("sRGB mid grey (0.214 linear)");
    p->setDefault(0);
    page->addChild(*p);
  }
  {
    DoubleParamDescriptor* p = desc.defineDoubleParam(kParamTrimapTolerance);
    p->setLabel("Trimap tolerance");
    p->setHint("Half-width of the band accepted around pure black and pure white in "
               "Auto mode. Larger values classify more pixels as known.");
    p->setRange(0.0, 0.49);
    p->setDisplayRange(0.0, 0.25);
    p->setDefault(0.05);
    page->addChild(*p);
  }
  {
    BooleanParamDescriptor* p = desc.defineBooleanParam(kParamTrimapInvert);
    p->setLabel("Invert trimap");
    p->setHint("Swap the foreground and background senses (unknown is unaffected).");
    p->setDefault(false);
    page->addChild(*p);
  }

  // ---- Colour -----------------------------------------------------------
  {
    BooleanParamDescriptor* p = desc.defineBooleanParam(kParamInputACEScg);
    p->setLabel("Input is ACEScg");
    p->setHint("Convert ACEScg (AP1 linear) to sRGB before inference. Applies to float "
               "clips only; 8/16-bit clips are already display-referred sRGB.");
    p->setDefault(true);
    page->addChild(*p);
  }

  // ---- Resources --------------------------------------------------------
  {
    IntParamDescriptor* p = desc.defineIntParam(kParamMaxTokens);
    p->setLabel("Max tokens");
    p->setHint("MEMatte's cap on how many tokens reach global attention. This is the "
               "primary memory lever and costs NO resolution: the matte is still "
               "computed at full size. Lower = less memory and faster.");
    p->setRange(hyp::kMinTokens, hyp::kMaxTokens);
    p->setDisplayRange(2048, 65536);
    p->setDefault(18500);
    page->addChild(*p);
  }
  {
    BooleanParamDescriptor* p = desc.defineBooleanParam(kParamAutoBudget);
    p->setLabel("Auto from available memory");
    p->setHint("Measure free VRAM (or system RAM on CPU / Apple Silicon) and cap the "
               "token count to fit, instead of using a fixed budget.");
    p->setDefault(true);
    page->addChild(*p);
  }
  {
    IntParamDescriptor* p = desc.defineIntParam(kParamMemBudget);
    p->setLabel("Memory budget (MB)");
    p->setHint("Fixed inference budget, used when Auto is off.");
    p->setRange(256, 262144);
    p->setDisplayRange(1024, 24576);
    p->setDefault(4096);
    page->addChild(*p);
  }
  {
    DoubleParamDescriptor* p = desc.defineDoubleParam(kParamHeadroom);
    p->setLabel("Memory headroom (%)");
    p->setHint("Fraction of the measured free memory left for the host application "
               "and other processes. Raise this if the host competes for VRAM.");
    p->setRange(0.0, 90.0);
    p->setDisplayRange(0.0, 50.0);
    p->setDefault(20.0);
    page->addChild(*p);
  }
  {
    ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamDevice);
    p->setLabel("Device");
    p->setHint("Auto uses the platform accelerator when available and falls back to CPU. "
               "CPU is slower but is not limited by VRAM - a valid workaround on a busy "
               "or small GPU.");
    p->appendOption("Auto");
    p->appendOption("GPU");
    p->appendOption("CPU");
    p->setDefault(0);
    page->addChild(*p);
  }
  {
    ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamComputeUnits);
    p->setLabel("Compute units (macOS)");
    p->setHint("CoreML hardware selection on Apple Silicon. Ignored by the CUDA EP.");
    p->appendOption("All (ANE/GPU/CPU)");
    p->appendOption("CPU + GPU");
    p->appendOption("CPU + Neural Engine");
    p->appendOption("CPU only");
    p->setDefault(0);
    page->addChild(*p);
  }
  {
    IntParamDescriptor* p = desc.defineIntParam(kParamThreads);
    p->setLabel("Max threads");
    p->setHint("Intra-op thread cap (0 = ONNX Runtime default).");
    p->setRange(0, 64);
    p->setDisplayRange(0, 16);
    p->setDefault(0);
    page->addChild(*p);
  }

  // ---- Models / status --------------------------------------------------
  {
    StringParamDescriptor* p = desc.defineStringParam(kParamBackboneFile);
    p->setLabel("Backbone file");
    p->setHint("Optional override for the MEMatte backbone ONNX. Otherwise "
               "$MEMATTE_MODEL_DIR, then the plugin bundle's Resources.");
    p->setStringType(eStringTypeFilePath);
    page->addChild(*p);
  }
  {
    StringParamDescriptor* p = desc.defineStringParam(kParamDecoderFile);
    p->setLabel("Decoder file");
    p->setHint("Optional override for the MEMatte decoder ONNX.");
    p->setStringType(eStringTypeFilePath);
    page->addChild(*p);
  }
  {
    StringParamDescriptor* p = desc.defineStringParam(kParamStatus);
    p->setLabel("Status");
    p->setHint("Measured memory and the planned token count / resolution. Updates when "
               "you change a parameter.");
    p->setStringType(eStringTypeMultiLine);
    p->setEnabled(false);
    p->setDefault("(change a parameter to refresh)");
    page->addChild(*p);
  }
}

OFX::ImageEffect* MEMatteFactory::createInstance(OfxImageEffectHandle handle,
                                                 OFX::ContextEnum /*context*/) {
  return new MEMattePlugin(handle);
}

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids) {
  static MEMatteFactory p(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
  ids.push_back(&p);
}
}  // namespace Plugin
}  // namespace OFX
