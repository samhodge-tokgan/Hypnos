// Copyright the Hypnos authors.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for OOM detection.
//
// This exists because the failure mode is SILENT. If an out-of-memory error is
// not recognised, MatteEngine's retry ladder abandons the render on the first
// attempt and the plugin falls through to passthrough — writing the source image
// into the matte output. It looks like a successful render and is not.
//
// The strings below are verbatim from real runs. The BFC-arena one is the case
// that actually shipped broken: a 4K matte with a capped CUDA arena silently
// became a passthrough (MAE 0.23 against ground truth instead of 0.0015).
#include <cstdio>
#include <string>

#include "OomDetect.h"

static int g_failures = 0;

static void ExpectOom(const std::string& msg, bool want, const std::string& what) {
  if (hyp::IsOutOfMemoryError(msg) != want) {
    std::printf("FAIL: %s\n  message: %s\n", what.c_str(), msg.c_str());
    ++g_failures;
  }
}

int main() {
  // --- must be recognised as OOM ------------------------------------------

  // Verbatim from an RTX A6000 run with gpu_mem_limit set (the regression).
  ExpectOom("Non-zero status code returned while running Conv node. "
            "Name:'/decoder/fusion_blks.1/conv/conv/Conv' Status Message: "
            "/onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:358 void* "
            "onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, "
            "onnxruntime::Stream*) Available memory of 189612800 is smaller than "
            "requested bytes of 221179904",
            true, "capped CUDA arena exhaustion");

  ExpectOom("CUDA failure 2: out of memory ; GPU=0", true, "classic CUDA OOM");
  ExpectOom("cudaErrorMemoryAllocation", true, "CUDA allocation error code");
  ExpectOom("std::bad_alloc", true, "host allocation failure");
  ExpectOom("Failed to allocate memory for requested buffer of size 1073741824",
            true, "ORT allocation failure");
  ExpectOom("onnxruntime::BFCArena::AllocateRawInternal", true, "BFC arena by name");
  ExpectOom("Out of memory", true, "capitalised spelling");

  // --- must NOT be treated as OOM -----------------------------------------
  // These are real failures the ladder must not paper over by retrying smaller.
  ExpectOom("", false, "empty message");
  ExpectOom("Load model from mematte_s_backbone.onnx failed: No such file or directory",
            false, "missing model file");
  ExpectOom("Invalid rank for input: image Got: 3 Expected: 4", false, "shape mismatch");
  ExpectOom("This is an invalid model. Error: Duplicate definition of name (x).",
            false, "malformed model");
  ExpectOom("Backbone model does not expose the expected inputs (image, max_tokens).",
            false, "wrong export");

  if (g_failures == 0) {
    std::printf("oom_test: all checks passed\n");
    return 0;
  }
  std::printf("oom_test: %d failure(s)\n", g_failures);
  return 1;
}
