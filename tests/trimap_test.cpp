// Copyright the Hypnos authors.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the trimap decoder. The central claim under test is that the
// default (Auto) mode produces the SAME labels whether a trimap was authored as
// float 0/0.5/1, as 8-bit 0/128/255, or as an 8-bit file that has been
// sRGB-decoded to linear on the way in (the ACEScg case) — because Auto only
// classifies the extremes and calls everything else unknown.
#include <cmath>
#include <cstdio>
#include <string>

#include "Trimap.h"

static int g_failures = 0;

static void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

static void CheckEq(float got, float want, const std::string& what) {
  if (std::fabs(got - want) > 1e-6f) {
    std::printf("FAIL: %s (got %g, want %g)\n", what.c_str(), got, want);
    ++g_failures;
  }
}

int main() {
  using hyp::DecodeTrimap;
  using hyp::TrimapEncoding;
  const float tol = 0.05f;

  // --- The headline claim: encoding-independence in Auto mode. --------------
  // Background is 0 and foreground is 1 in every encoding; only the mid-grey
  // representation differs, and all three must land on "unknown".
  const float kFloatGrey = 0.5f;
  const float kIntGrey = 128.0f / 255.0f;          // 0.50196
  const float kSrgbDecodedGrey = hyp::kSrgbMidGreyLinear;  // 0.21404

  for (float grey : {kFloatGrey, kIntGrey, kSrgbDecodedGrey}) {
    CheckEq(DecodeTrimap(0.0f, TrimapEncoding::Auto, tol, false), 0.0f, "auto: black -> bg");
    CheckEq(DecodeTrimap(1.0f, TrimapEncoding::Auto, tol, false), 1.0f, "auto: white -> fg");
    CheckEq(DecodeTrimap(grey, TrimapEncoding::Auto, tol, false), 0.5f,
            "auto: mid grey -> unknown (grey=" + std::to_string(grey) + ")");
  }

  // Near-extremes inside the tolerance band still classify as known.
  CheckEq(DecodeTrimap(0.02f, TrimapEncoding::Auto, tol, false), 0.0f, "auto: near-black -> bg");
  CheckEq(DecodeTrimap(0.98f, TrimapEncoding::Auto, tol, false), 1.0f, "auto: near-white -> fg");
  // Just outside the band is unknown, which is the safe default.
  CheckEq(DecodeTrimap(0.10f, TrimapEncoding::Auto, tol, false), 0.5f, "auto: 0.10 -> unknown");
  CheckEq(DecodeTrimap(0.90f, TrimapEncoding::Auto, tol, false), 0.5f, "auto: 0.90 -> unknown");

  // Scene-referred super-white must still read as foreground, not be clipped
  // into the unknown band.
  CheckEq(DecodeTrimap(4.0f, TrimapEncoding::Auto, tol, false), 1.0f,
          "auto: scene-referred super-white -> fg");

  // Tolerance widens the known bands.
  CheckEq(DecodeTrimap(0.10f, TrimapEncoding::Auto, 0.2f, false), 0.0f,
          "auto: 0.10 with wide tolerance -> bg");

  // A degenerate tolerance must not make a value both black and white.
  const float both = DecodeTrimap(0.5f, TrimapEncoding::Auto, 0.99f, false);
  Check(both == 0.0f || both == 1.0f || both == 0.5f, "auto: clamped tolerance stays valid");

  // --- Explicit encodings --------------------------------------------------
  CheckEq(DecodeTrimap(0.0f, TrimapEncoding::Float0_Half_1, tol, false), 0.0f, "float: 0");
  CheckEq(DecodeTrimap(0.5f, TrimapEncoding::Float0_Half_1, tol, false), 0.5f, "float: 0.5");
  CheckEq(DecodeTrimap(1.0f, TrimapEncoding::Float0_Half_1, tol, false), 1.0f, "float: 1");
  CheckEq(DecodeTrimap(kIntGrey, TrimapEncoding::Float0_Half_1, tol, false), 0.5f,
          "float: 128/255 is nearest 0.5");

  // Upstream MEMatte's own 8-bit rule: <85 -> 0, >=170 -> 1, else 0.5.
  CheckEq(DecodeTrimap(84.0f / 255.0f, TrimapEncoding::Integer0_128_255, tol, false), 0.0f,
          "int: 84 -> bg");
  CheckEq(DecodeTrimap(85.0f / 255.0f, TrimapEncoding::Integer0_128_255, tol, false), 0.5f,
          "int: 85 -> unknown");
  CheckEq(DecodeTrimap(169.0f / 255.0f, TrimapEncoding::Integer0_128_255, tol, false), 0.5f,
          "int: 169 -> unknown");
  CheckEq(DecodeTrimap(170.0f / 255.0f, TrimapEncoding::Integer0_128_255, tol, false), 1.0f,
          "int: 170 -> fg");
  CheckEq(DecodeTrimap(128.0f / 255.0f, TrimapEncoding::Integer0_128_255, tol, false), 0.5f,
          "int: 128 -> unknown");

  // sRGB-decoded mid grey sits much closer to black than to 0.5, so this mode
  // exists to stop it being mistaken for background.
  CheckEq(DecodeTrimap(kSrgbDecodedGrey, TrimapEncoding::SrgbMidGrey, tol, false), 0.5f,
          "srgb: 0.214 -> unknown");
  CheckEq(DecodeTrimap(0.0f, TrimapEncoding::SrgbMidGrey, tol, false), 0.0f, "srgb: 0 -> bg");
  CheckEq(DecodeTrimap(1.0f, TrimapEncoding::SrgbMidGrey, tol, false), 1.0f, "srgb: 1 -> fg");

  // --- Invert --------------------------------------------------------------
  CheckEq(DecodeTrimap(0.0f, TrimapEncoding::Auto, tol, true), 1.0f, "invert: black -> fg");
  CheckEq(DecodeTrimap(1.0f, TrimapEncoding::Auto, tol, true), 0.0f, "invert: white -> bg");
  CheckEq(DecodeTrimap(0.5f, TrimapEncoding::Auto, tol, true), 0.5f,
          "invert: unknown is unaffected");

  // --- Output domain -------------------------------------------------------
  // Every path must yield exactly one of the three labels the model expects.
  for (int i = -50; i <= 350; ++i) {
    const float v = i / 255.0f;
    for (int e = 0; e <= 3; ++e) {
      const float r = DecodeTrimap(v, static_cast<TrimapEncoding>(e), tol, false);
      Check(r == 0.0f || r == 0.5f || r == 1.0f, "output is always 0, 0.5 or 1");
    }
  }

  if (g_failures == 0) {
    std::printf("trimap_test: all checks passed\n");
    return 0;
  }
  std::printf("trimap_test: %d failure(s)\n", g_failures);
  return 1;
}
