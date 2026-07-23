// Copyright the Hypnos authors.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the memory-sizing model and the live memory probe.
//
// These check INTERNAL CONSISTENCY and the sanity of the estimate against the
// two figures MEMatte's paper reports (0.71 GB for ViT-S and 1.49 GB for ViT-B at
// 1024x1024 on Composition-1K). They do not, and cannot, prove the estimate
// matches a real allocator on a given GPU — that requires measuring on hardware
// with tests/ort_check.cpp. See the calibration note in src/Sizing.h.
#include <cstdint>
#include <cstdio>
#include <string>

#include "MemProbe.h"
#include "Sizing.h"

static int g_failures = 0;

static void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

static constexpr uint64_t GB = 1024ull * 1024ull * 1024ull;
static constexpr uint64_t MB = 1024ull * 1024ull;

int main() {
  using namespace hyp;

  // --- RoundUpToMultiple ---------------------------------------------------
  Check(RoundUpToMultiple(1, 32) == 32, "round up 1 -> 32");
  Check(RoundUpToMultiple(32, 32) == 32, "round up 32 -> 32 (already aligned)");
  Check(RoundUpToMultiple(33, 32) == 64, "round up 33 -> 64");
  Check(RoundUpToMultiple(1920, 32) == 1920, "round up 1920 -> 1920");
  Check(RoundUpToMultiple(1080, 32) == 1088, "round up 1080 -> 1088");
  Check(RoundUpToMultiple(0, 32) == 32, "round up 0 -> 32 (never zero-sized)");

  // --- Estimate vs MEASURED VRAM on an RTX A6000 ---------------------------
  // Ground truth from tests/ort_check.cpp on real hardware (CUDA 12, ORT 1.27.1,
  // ViT-S). The estimate must never sit BELOW measured — that would admit a job
  // that then OOMs — and should not be absurdly above it either.
  struct Measured { int w, h, k, dim; uint64_t vram; const char* note; };
  const Measured kA6000[] = {
      // ViT-S
      {512, 512, 18500, kEmbedDimViTS, 799 * MB, "S 512^2 cap>N"},
      {1024, 1024, 4096, kEmbedDimViTS, 2355 * MB, "S 1024^2 k=4096"},
      {1536, 1536, 16384, kEmbedDimViTS, 14643 * MB, "S 1536^2 k=9216"},
      {2048, 2048, 1024, kEmbedDimViTS, 1331 * MB, "S 2048^2 k=1024"},
      {2048, 2048, 16384, kEmbedDimViTS, 36147 * MB, "S 2048^2 k=16384"},
      // ViT-B — validates that the cost really does scale with head count.
      {512, 512, 16384, kEmbedDimViTB, 1331 * MB, "B 512^2 cap>N"},
      {1024, 1024, 4096, kEmbedDimViTB, 5427 * MB, "B 1024^2 k=4096"},
      {1536, 1536, 1024, kEmbedDimViTB, 2355 * MB, "B 1536^2 k=1024"},
      {1536, 1536, 16384, kEmbedDimViTB, 19763 * MB, "B 1536^2 k=9216"},
  };
  for (const auto& m : kA6000) {
    const uint64_t est = EstimatePeakBytes(m.w, m.h, m.k, m.dim);
    const double ratio = static_cast<double>(est) / static_cast<double>(m.vram);
    std::printf("  %-16s measured %-9s estimate %-9s (%.2fx)\n", m.note,
                FormatBytes(m.vram).c_str(), FormatBytes(est).c_str(), ratio);
    Check(est >= m.vram, std::string("estimate is not below measured: ") + m.note);
    Check(ratio <= 3.0, std::string("estimate is not wildly above measured: ") + m.note);
  }

  // The one configuration that actually ran out of memory on a 48 GB card:
  // 3072x2048 with the cap above the token count (24576 tokens). The estimate
  // MUST exceed the card's capacity, or the planner would have admitted it.
  const uint64_t oom = EstimatePeakBytes(3072, 2048, 65536, kEmbedDimViTS);
  std::printf("  %-16s measured OOM@48GB estimate %s\n", "3072x2048 k=N",
              FormatBytes(oom).c_str());
  Check(oom > 48ull * GB, "the observed OOM configuration is predicted to not fit");

  Check(EstimatePeakBytes(1024, 1024, 18500, kEmbedDimViTB) >
            EstimatePeakBytes(1024, 1024, 18500, kEmbedDimViTS),
        "ViT-B costs more than ViT-S");

  // --- Monotonicity (the search in LongSideForBudget depends on it) --------
  Check(EstimatePeakBytes(2048, 2048, 8192, kEmbedDimViTS) >
            EstimatePeakBytes(1024, 1024, 8192, kEmbedDimViTS),
        "cost grows with resolution");
  Check(EstimatePeakBytes(2048, 2048, 16384, kEmbedDimViTS) >
            EstimatePeakBytes(2048, 2048, 8192, kEmbedDimViTS),
        "cost grows with the token cap");

  // The token cap must actually bound the quadratic term: at a fixed large
  // resolution, halving the cap must cut cost materially.
  const uint64_t hi = EstimatePeakBytes(4096, 4096, 32768, kEmbedDimViTS);
  const uint64_t lo = EstimatePeakBytes(4096, 4096, 8192, kEmbedDimViTS);
  Check(lo < hi / 2, "lowering the token cap substantially lowers peak memory");

  // Capping above the available token count is a no-op (k = min(K, N)).
  Check(EstimatePeakBytes(512, 512, 100000, kEmbedDimViTS) ==
            EstimatePeakBytes(512, 512, (512 / 16) * (512 / 16), kEmbedDimViTS),
        "token cap saturates at the actual token count");

  // --- TokensForBudget -----------------------------------------------------
  for (uint64_t budget : {2ull * GB, 6ull * GB, 12ull * GB, 24ull * GB}) {
    const int k = TokensForBudget(2048, 2048, kEmbedDimViTS, budget);
    if (k > 0) {
      Check(EstimatePeakBytes(2048, 2048, k, kEmbedDimViTS) <= budget,
            "TokensForBudget result fits the budget (" + FormatBytes(budget) + ")");
      Check(k >= kMinTokens, "TokensForBudget respects the floor");
      Check(k <= kMaxTokens, "TokensForBudget respects the ceiling");
    }
  }
  // REGRESSION: a plate small enough to have fewer than kMinTokens tokens in
  // total must still report a workable cap when it comfortably fits. This first
  // showed up on a real 512x384 render (768 tokens): TokensForBudget compared
  // against kMinTokens unconditionally, returned 0, and the planner wrongly
  // announced "memory budget is very small" and dropped to the floor.
  for (auto wh : {std::pair<int, int>{512, 384}, {320, 256}, {256, 256}}) {
    const int w = wh.first, h = wh.second;
    const int n = (w / kPatchSize) * (h / kPatchSize);
    const int k = TokensForBudget(w, h, kEmbedDimViTS, 4 * GB);
    Check(n < kMinTokens, "this fixture really is below the token floor");
    Check(k > 0, "a small plate that fits must not report zero tokens");
    Check(k == n, "a comfortably-fitting small plate is capped by its token count, "
                  "not by the floor");
    Check(EstimatePeakBytes(w, h, k, kEmbedDimViTS) <= 4 * GB,
          "the small-plate result still fits the budget");
  }

  // A budget that cannot even hold the weights must refuse rather than return junk.
  Check(TokensForBudget(4096, 4096, kEmbedDimViTB, 16 * MB) == 0,
        "an impossible budget yields 0 tokens");
  // A generous budget should allow more tokens than a tight one.
  Check(TokensForBudget(2048, 2048, kEmbedDimViTS, 24 * GB) >=
            TokensForBudget(2048, 2048, kEmbedDimViTS, 4 * GB),
        "a bigger budget allows at least as many tokens");

  // --- LongSideForBudget ---------------------------------------------------
  // 4K plate, small budget: must recommend a reduced long side that really fits.
  const int ls = LongSideForBudget(3840, 2160, kEmbedDimViTS, 2 * GB);
  std::printf("4K plate in 2 GB -> long side %d\n", ls);
  if (ls > 0) {
    Check(ls % kSizeDivisor == 0, "long side is a multiple of 32");
    Check(ls <= 3840, "long side never exceeds the source");
    const double s = static_cast<double>(ls) / 3840.0;
    const int w = RoundUpToMultiple(static_cast<int>(3840 * s), kSizeDivisor);
    const int h = RoundUpToMultiple(static_cast<int>(2160 * s), kSizeDivisor);
    Check(EstimatePeakBytes(w, h, kMinTokens, kEmbedDimViTS) <= 2 * GB,
          "the recommended long side actually fits the budget");
  }
  // With plenty of memory the native resolution should be reachable.
  Check(LongSideForBudget(3840, 2160, kEmbedDimViTS, 64 * GB) == 3840,
        "a large budget allows native 4K");
  // A hopeless budget must return 0 rather than a bogus size.
  Check(LongSideForBudget(3840, 2160, kEmbedDimViTB, 8 * MB) == 0,
        "an impossible budget yields no workable size");

  // --- Live probe sanity ---------------------------------------------------
  const MemInfo mi = ProbeMemory();
  std::printf("probe: %s\n", DescribeMemory(mi).c_str());
  Check(mi.sys_total > 0, "system total memory is reported");
  Check(mi.sys_avail > 0 && mi.sys_avail <= mi.sys_total,
        "system available memory is within total");
  if (mi.gpu_valid) {
    Check(mi.gpu_total > 0, "GPU total is reported when a GPU is detected");
    Check(mi.gpu_free <= mi.gpu_total, "GPU free is within total");
    Check(!mi.gpu_source.empty(), "GPU source is named");
  }

  if (g_failures == 0) {
    std::printf("sizing_test: all checks passed\n");
    return 0;
  }
  std::printf("sizing_test: %d failure(s)\n", g_failures);
  return 1;
}
