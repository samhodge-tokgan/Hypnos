// Prints the available ONNX Runtime execution providers, the measured GPU/system
// memory, and optionally runs the MEMatte backbone once on the accelerator.
//
// Usage:
//   ort_check                          -> providers + memory; exit 0 if accelerator present
//   ort_check <backbone.onnx> [W H K]  -> also load and run one inference, reporting
//                                         timing, output stats and estimated vs. real cost
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "MemProbe.h"

#if defined(__APPLE__)
static const char* kAccel = "CoreML";
#else
static const char* kAccel = "CUDA";
#endif

static void AppendAccel(Ort::SessionOptions& so) {
#if defined(__APPLE__)
  // No RequireStaticInputShapes: the MEMatte backbone takes a runtime token cap,
  // so its TopK/Gather shapes are dynamic and demanding static shapes would make
  // CoreML reject the subgraph outright instead of accelerating what it can.
  std::unordered_map<std::string, std::string> opts = {{"MLComputeUnits", "ALL"},
                                                       {"ModelFormat", "MLProgram"}};
  so.AppendExecutionProvider("CoreML", opts);
#else
  OrtCUDAProviderOptions cuda{};
  cuda.device_id = 0;
  so.AppendExecutionProvider_CUDA(cuda);
#endif
}

int main(int argc, char** argv) {
  bool accel = false;
  std::cout << "ONNX Runtime available execution providers:\n";
  for (const auto& p : Ort::GetAvailableProviders()) {
    std::cout << "  - " << p << "\n";
    if (p.find(kAccel) != std::string::npos) accel = true;
  }
  std::cout << kAccel << " EP available: " << (accel ? "YES" : "NO") << "\n";

  const hyp::MemInfo mi = hyp::ProbeMemory();
  std::cout << "Memory: " << hyp::DescribeMemory(mi) << "\n";

  if (argc < 2) return accel ? 0 : 1;

  const std::string model = argv[1];
  const int W = argc > 2 ? std::atoi(argv[2]) : 1024;
  const int H = argc > 3 ? std::atoi(argv[3]) : 1024;
  const int K = argc > 4 ? std::atoi(argv[4]) : 18500;

  std::cout << "\nLoading model on " << kAccel << ": " << model << "\n";
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort_check");
  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  bool on_accel = false;
  try {
    AppendAccel(so);
    on_accel = true;
  } catch (const Ort::Exception& e) {
    std::cout << "  accelerator append failed, using CPU: " << e.what() << "\n";
  }

  std::unique_ptr<Ort::Session> sess;
  try {
    std::filesystem::path model_path(model);
    sess = std::make_unique<Ort::Session>(env, model_path.c_str(), so);
  } catch (const Ort::Exception& e) {
    std::cout << "  session create FAILED: " << e.what() << "\n";
    return 2;
  }
  std::cout << "  session created (" << (on_accel ? kAccel : "CPU") << ")\n";

  Ort::AllocatorWithDefaultOptions alloc;

  // The backbone takes (image [1,4,H,W], max_tokens [1] int64).
  std::vector<Ort::AllocatedStringPtr> in_holders;
  std::vector<std::string> in_names_s;
  for (size_t i = 0; i < sess->GetInputCount(); ++i) {
    in_holders.push_back(sess->GetInputNameAllocated(i, alloc));
    in_names_s.emplace_back(in_holders.back().get());
  }

  const size_t n = static_cast<size_t>(W) * H * 4;
  std::vector<float> image(n);
  for (size_t i = 0; i < n; ++i) image[i] = 0.3f + 0.4f * float(i % 97) / 97.0f;
  std::vector<int64_t> tokens{K};

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> ishape{1, 4, H, W};
  std::vector<int64_t> kshape{1};

  std::vector<Ort::Value> ins;
  std::vector<const char*> in_names;
  for (const std::string& nm : in_names_s) {
    in_names.push_back(nm.c_str());
    if (nm == "max_tokens") {
      ins.push_back(Ort::Value::CreateTensor<int64_t>(mem, tokens.data(), tokens.size(),
                                                      kshape.data(), kshape.size()));
    } else {
      ins.push_back(Ort::Value::CreateTensor<float>(mem, image.data(), image.size(),
                                                    ishape.data(), ishape.size()));
    }
  }

  std::vector<Ort::AllocatedStringPtr> out_holders;
  std::vector<const char*> out_names;
  const size_t nout = sess->GetOutputCount();
  for (size_t i = 0; i < nout; ++i) {
    out_holders.push_back(sess->GetOutputNameAllocated(i, alloc));
    out_names.push_back(out_holders.back().get());
  }

  try {
    sess->Run(Ort::RunOptions{nullptr}, in_names.data(), ins.data(), ins.size(),
              out_names.data(), nout);
  } catch (const Ort::Exception& e) {
    std::cout << "  warm-up run FAILED: " << e.what() << "\n";
    return 3;
  }
  const auto t0 = std::chrono::high_resolution_clock::now();
  auto outs = sess->Run(Ort::RunOptions{nullptr}, in_names.data(), ins.data(), ins.size(),
                        out_names.data(), nout);
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  auto& d = outs[0];
  const auto os = d.GetTensorTypeAndShapeInfo().GetShape();
  size_t on = 1;
  for (auto v : os) on *= static_cast<size_t>(v);
  const float* od = d.GetTensorData<float>();
  float mn = od[0], mx = od[0];
  double sum = 0;
  for (size_t i = 0; i < on; ++i) { mn = std::min(mn, od[i]); mx = std::max(mx, od[i]); sum += od[i]; }

  std::cout << "  inference OK: " << ms << " ms  (" << W << "x" << H << ", K=" << K << ")\n";
  std::cout << "  output[0] shape: [";
  for (size_t i = 0; i < os.size(); ++i) std::cout << os[i] << (i + 1 < os.size() ? "," : "");
  std::cout << "]  min=" << mn << " max=" << mx << " mean=" << (sum / on) << "\n";

  // Compare the sizing model against what the device actually consumed, so the
  // constants in MatteEngine.cpp can be recalibrated from real measurements.
  const hyp::MemInfo after = hyp::ProbeMemory();
  if (mi.gpu_valid && after.gpu_valid && mi.gpu_free > after.gpu_free) {
    std::cout << "  VRAM consumed by this session: "
              << hyp::FormatBytes(mi.gpu_free - after.gpu_free) << "\n";
  }
  return 0;
}
