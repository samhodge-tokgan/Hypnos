// Copyright the Hypnos authors.
// SPDX-License-Identifier: Apache-2.0
#include "Trimap.h"

#include <algorithm>
#include <cmath>

namespace hyp {
namespace {

// Return the anchor value (0, 0.5 or 1) whose position is nearest to v.
// `grey` is where "unknown" sits for this encoding.
float NearestOfThree(float v, float grey) {
  const float d0 = std::fabs(v - 0.0f);
  const float dg = std::fabs(v - grey);
  const float d1 = std::fabs(v - 1.0f);
  if (d0 <= dg && d0 <= d1) return 0.0f;
  if (d1 <= dg) return 1.0f;
  return 0.5f;
}

}  // namespace

float DecodeTrimap(float v, TrimapEncoding enc, float tolerance, bool invert) {
  // Guard against a nonsensical tolerance from the UI; 0.5 would make every
  // value "near" both extremes at once.
  const float tol = std::min(std::max(tolerance, 0.0f), 0.49f);

  float out;
  switch (enc) {
    case TrimapEncoding::Float0_Half_1:
      out = NearestOfThree(std::min(std::max(v, 0.0f), 1.0f), 0.5f);
      break;

    case TrimapEncoding::Integer0_128_255:
      // Upstream MEMatte (data/dim_dataset.py and meta_arch/mematte.py):
      // <85 -> 0, >=170 -> 1, else 0.5, on the 0-255 scale.
      if (v < 85.0f / 255.0f) out = 0.0f;
      else if (v >= 170.0f / 255.0f) out = 1.0f;
      else out = 0.5f;
      break;

    case TrimapEncoding::SrgbMidGrey:
      out = NearestOfThree(std::min(std::max(v, 0.0f), 1.0f), kSrgbMidGreyLinear);
      break;

    case TrimapEncoding::Auto:
    default:
      // Only the extremes are classified; anything in between is unknown, so no
      // assumption about where mid grey landed is needed. Note v is deliberately
      // NOT clamped above 1: scene-referred super-white is still foreground.
      if (v <= tol) out = 0.0f;
      else if (v >= 1.0f - tol) out = 1.0f;
      else out = 0.5f;
      break;
  }

  if (invert && out != 0.5f) out = 1.0f - out;
  return out;
}

}  // namespace hyp
