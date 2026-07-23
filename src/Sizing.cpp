// Copyright the Hypnos authors.
// SPDX-License-Identifier: Apache-2.0
#include "Sizing.h"

#include <algorithm>
#include <cmath>

namespace hyp {

// Bytes of live K x K tensor per element PER HEAD in a routed block. A block
// materialises several at once (identity, broadcast mask, raw scores, masked
// scores, softmax result), so this is far above the naive 4 bytes of a single
// fp32 score matrix. Measured ~21-27 on an RTX A6000 at heads=6; 32 is
// deliberately conservative. EstimatePeakBytes and TokensForBudget MUST use the
// same value — TokensForBudget inverts the quadratic that EstimatePeakBytes
// builds, and letting the two drift silently breaks the budget planner.
static constexpr uint64_t kAttnBytesPerElemPerHead = 32ull;

// Global safety margin on the whole estimate, as a percentage.
//
// The fitted terms track measured VRAM well but not perfectly, and the residual
// error is not symmetric in consequence: over-estimating costs a slightly lower
// token cap, while under-estimating costs a failed render. This margin buys back
// the few percent where the fit dips under (e.g. ViT-B at 1536^2 with a low cap)
// without needing a coefficient tuned to every bucket the CUDA arena happens to
// land on.
static constexpr uint64_t kSafetyMarginPct = 115ull;

int RoundUpToMultiple(int v, int m) {
  if (m <= 0) return v;
  if (v <= 0) return m;
  return ((v + m - 1) / m) * m;
}

namespace {
// Everything in EstimatePeakBytes EXCEPT the quadratic attention term, before the
// safety margin. TokensForBudget inverts the quadratic, so it must work in the
// same un-margined space; sharing this helper is what stops the two drifting.
uint64_t FixedBytesRaw(uint64_t tokens, uint64_t d) {
  const uint64_t weights = 144ull * d * d * 4ull;
  const uint64_t activations = 28ull * tokens * d * 4ull;
  const uint64_t decoder = 200ull << 20;
  const uint64_t runtime = 600ull << 20;
  return weights + activations + decoder + runtime;
}
uint64_t TokenCount(int proc_w, int proc_h) {
  return static_cast<uint64_t>(std::max(1, proc_w / kPatchSize)) *
         static_cast<uint64_t>(std::max(1, proc_h / kPatchSize));
}
}  // namespace

uint64_t EstimatePeakBytes(int proc_w, int proc_h, int max_tokens, int embed_dim) {
  const uint64_t tokens = TokenCount(proc_w, proc_h);
  const uint64_t k =
      std::min<uint64_t>(static_cast<uint64_t>(std::max(1, max_tokens)), tokens);
  const uint64_t d = static_cast<uint64_t>(std::max(1, embed_dim));
  const uint64_t heads = std::max<uint64_t>(1, d / 64);

  // See kAttnBytesPerElemPerHead: several K x K tensors are live at once, so this
  // is far above the naive 4 bytes of a single fp32 score matrix. This is the
  // term the token cap exists to bound.
  const uint64_t attention = k * k * heads * kAttnBytesPerElemPerHead;

  const uint64_t raw = FixedBytesRaw(tokens, d) + attention;
  return raw / 100ull * kSafetyMarginPct;
}

int TokensForBudget(int proc_w, int proc_h, int embed_dim, uint64_t budget_bytes) {
  const uint64_t tokens = TokenCount(proc_w, proc_h);
  const uint64_t d = static_cast<uint64_t>(std::max(1, embed_dim));
  const uint64_t heads = std::max<uint64_t>(1, d / 64);

  // Work in raw (un-margined) bytes: convert the budget in, solve the quadratic,
  // and the result is directly comparable to what EstimatePeakBytes will report.
  const uint64_t budget_raw = budget_bytes / kSafetyMarginPct * 100ull;
  const uint64_t fixed_raw = FixedBytesRaw(tokens, d);
  if (budget_raw <= fixed_raw) return 0;

  const double room = static_cast<double>(budget_raw - fixed_raw);
  const double k = std::sqrt(
      room / (static_cast<double>(heads) * static_cast<double>(kAttnBytesPerElemPerHead)));

  int result = static_cast<int>(std::min<double>(k, static_cast<double>(tokens)));
  result = std::min(result, kMaxTokens);

  // The floor is the smaller of kMinTokens and the number of tokens that
  // actually EXIST at this resolution. A plate below ~512x512 has fewer than
  // kMinTokens tokens in total, and capping at "all of them" is a complete
  // success, not a failure to fit. Comparing against kMinTokens unconditionally
  // made every small plate report that nothing fits, which pushed the planner
  // into the downscale branch for images that fit with room to spare.
  const int floor_tokens = std::min<int>(kMinTokens, static_cast<int>(tokens));
  if (result < floor_tokens) return 0;

  // Rounding in the conversions above can leave the result a hair over budget;
  // step down until it genuinely fits, so the caller's invariant always holds.
  while (result > floor_tokens &&
         EstimatePeakBytes(proc_w, proc_h, result, embed_dim) > budget_bytes) {
    result = result * 99 / 100;
  }
  return EstimatePeakBytes(proc_w, proc_h, result, embed_dim) > budget_bytes ? 0 : result;
}

int LongSideForBudget(int src_w, int src_h, int embed_dim, uint64_t budget_bytes) {
  const int src_long = std::max(src_w, src_h);
  if (src_long <= 0) return 0;
  // Binary search the largest long side (in multiples of 32) that fits at the
  // minimum token cap. The estimate is monotonic in size, so this is exact for it.
  int lo = kSizeDivisor, hi = RoundUpToMultiple(src_long, kSizeDivisor), best = 0;
  while (lo <= hi) {
    int mid = RoundUpToMultiple((lo + hi) / 2, kSizeDivisor);
    if (mid > hi) mid = hi;
    const double s = static_cast<double>(mid) / static_cast<double>(src_long);
    const int w = RoundUpToMultiple(
        std::max(kSizeDivisor, static_cast<int>(std::lround(src_w * s))), kSizeDivisor);
    const int h = RoundUpToMultiple(
        std::max(kSizeDivisor, static_cast<int>(std::lround(src_h * s))), kSizeDivisor);
    if (EstimatePeakBytes(w, h, kMinTokens, embed_dim) <= budget_bytes) {
      best = mid;
      lo = mid + kSizeDivisor;
    } else {
      hi = mid - kSizeDivisor;
    }
  }
  return best;
}

}  // namespace hyp
