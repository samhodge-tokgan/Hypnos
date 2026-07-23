// Copyright the Hypnos authors.
// SPDX-License-Identifier: Apache-2.0
//
// Recognising an out-of-memory failure in an ONNX Runtime error string.
//
// This lives in its own header, free of any ONNX Runtime dependency, so it can
// be unit-tested directly (tests/oom_test.cpp). That matters more than it looks:
// ONNX Runtime does not use consistent wording for allocation failures, and an
// unrecognised OOM fails SILENTLY — the engine's retry ladder gives up on the
// first attempt and the plugin falls through to passthrough, writing the source
// image where a matte should be. That is a wrong result presented as a good one.
//
// It happened for real on an RTX A6000 with a capped CUDA arena, which reports
//   "BFCArena::AllocateRawInternal ... Available memory of 189612800 is smaller
//    than requested bytes of 221179904"
// matching none of the obvious "out of memory" spellings. The 4K matte silently
// became a passthrough (MAE 0.23 instead of 0.0015).
//
// When in doubt, ADD a pattern: a false positive costs one wasted retry at a
// smaller size, a false negative costs a silently wrong render.
#pragma once

#include <string>

namespace hyp {

inline bool IsOutOfMemoryError(const std::string& what) {
  static const char* kNeedles[] = {
      // Common spellings across execution providers.
      "out of memory", "Out of memory", "OOM", "cudaErrorMemoryAllocation",
      "failed to allocate", "Failed to allocate", "bad_alloc",
      // ONNX Runtime's BFC arena, including the capped-arena wording that this
      // list originally missed.
      "BFCArena", "Available memory of", "smaller than requested",
  };
  for (const char* n : kNeedles)
    if (what.find(n) != std::string::npos) return true;
  return false;
}

}  // namespace hyp
