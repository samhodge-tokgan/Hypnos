// Copyright the Hypnos authors.
// SPDX-License-Identifier: Apache-2.0
//
// Inference sizing: how big a plate, and how many global-attention tokens, fit in
// a given memory budget. Kept free of any ONNX Runtime dependency so it can be
// unit-tested on its own (tests/sizing_test.cpp).
#pragma once

#include <cstdint>

namespace hyp {

// Bounds used by the sizing search and the UI.
constexpr int kMinTokens = 1024;
constexpr int kMaxTokens = 262144;
constexpr int kSizeDivisor = 32;   // MEMatte size_divisibility
constexpr int kPatchSize = 16;     // ViT patch -> feature stride
constexpr int kDecoderTile = 512;  // upstream patch_inference tile
constexpr int kDecoderHalo = 64;   // upstream patch_inference overlap

// Embedding width per MEMatte backbone variant.
constexpr int kEmbedDimViTS = 384;
constexpr int kEmbedDimViTB = 768;

// Round `v` UP to the next multiple of `m`. MEMatte requires input dimensions to
// be multiples of 32 (its size_divisibility); the model is padded, not resized.
int RoundUpToMultiple(int v, int m);

// Estimated peak inference bytes for a plate of proc_w x proc_h at `max_tokens`,
// for the given embedding width.
//
// CALIBRATION STATUS: fitted to measured VRAM on an NVIDIA RTX A6000 (48 GB,
// driver 570, CUDA 12, ONNX Runtime 1.27.1) for ViT-S, over a sweep of
// 512^2..3072x2048 and token caps 1024..65536. The model is deliberately
// CONSERVATIVE — it reads 1.2x to 1.8x above measured across that sweep —
// because under-estimating is the dangerous direction: it admits a job that then
// runs out of memory. It correctly predicts the one configuration that actually
// OOMed (3072x2048 at 24576 tokens).
//
// Do NOT compare this against the MEMatte paper's 0.71/1.49 GB figures: those are
// PyTorch peaks for a different attention implementation, and are much lower than
// what ONNX Runtime actually allocates.
//
// Still unfitted: macOS/CoreML unified memory, and the ViT-B head scaling beyond
// a spot check. Always pair the estimate with a headroom margin and rely on the
// engine's OOM retry ladder as the real safety net. See docs/DEVELOPMENT.md.
uint64_t EstimatePeakBytes(int proc_w, int proc_h, int max_tokens, int embed_dim);

// Largest token cap that fits `budget_bytes` at the given processing size, or 0
// if even kMinTokens does not fit.
int TokensForBudget(int proc_w, int proc_h, int embed_dim, uint64_t budget_bytes);

// Longest side that fits `budget_bytes` at the minimum token cap, preserving the
// source aspect ratio. 0 if nothing fits.
int LongSideForBudget(int src_w, int src_h, int embed_dim, uint64_t budget_bytes);

}  // namespace hyp
