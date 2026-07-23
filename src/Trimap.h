// Copyright the Hypnos authors.
// SPDX-License-Identifier: Apache-2.0
//
// Trimap decoding: turn whatever the host hands us into MEMatte's expected
// 0 / 0.5 / 1 (background / unknown / foreground) 4th input channel.
//
// Why this needs care. A trimap is DATA, not an image, but it travels through an
// image pipeline that may or may not colour-manage it. The same "mid grey" pixel
// can arrive as:
//   * 0.5      — authored as float data and passed through untouched
//   * 0.50196  — authored as 8-bit 128 and normalised (128/255)
//   * 0.21404  — authored as 8-bit 128, then sRGB-decoded to linear on read
//                (this is what happens in an ACEScg scene-referred comp)
// Neutral greys are near-identical in AP1 and Rec.709 primaries, so the ACEScg
// case lands on the same ~0.214 as linear Rec.709.
//
// The KEY observation: we never actually need to recognise mid grey. Only the
// two extremes matter — a pixel is background if it is near black, foreground if
// it is near white, and UNKNOWN otherwise. Unknown is also the safe default,
// because it just means "let the model decide" rather than forcing a wrong hard
// edge. That makes the default mode immune to the whole encoding question, which
// is the practical answer to "it is difficult to see a mid grey".
//
// The explicit modes exist for users who want exact, documented behaviour.
#pragma once

namespace hyp {

enum class TrimapEncoding {
  // Near-black -> 0, near-white -> 1, everything else -> 0.5. Encoding-agnostic.
  Auto = 0,
  // Nearest of {0, 0.5, 1}: float data authored with a literal 0.5 mid grey.
  Float0_Half_1,
  // Upstream MEMatte's rule on 8-bit values: <85 -> 0, >=170 -> 1, else 0.5.
  Integer0_128_255,
  // Nearest of {0, 0.21404, 1}: 8-bit mid grey that has been sRGB-decoded to
  // linear (the ACEScg / scene-referred case).
  SrgbMidGrey,
};

// 8-bit mid grey (128/255 = 0.50196) decoded through the sRGB EOTF.
constexpr float kSrgbMidGreyLinear = 0.21404114f;

// Decode one normalised sample to exactly 0.0f, 0.5f or 1.0f.
//
// `v` is the raw sample already normalised to a nominal [0,1] by the pixel
// reader, but NOT clamped: scene-referred float can legitimately exceed 1.0, and
// a super-white trimap pixel still means foreground.
// `tolerance` is the half-width of the band accepted around an anchor, in
// normalised units. `invert` swaps the foreground/background sense for pipelines
// that author trimaps the other way round.
float DecodeTrimap(float v, TrimapEncoding enc, float tolerance, bool invert);

}  // namespace hyp
