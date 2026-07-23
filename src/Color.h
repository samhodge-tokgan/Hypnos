// Copyright the Hypnos authors.
// SPDX-License-Identifier: Apache-2.0
//
// Colour conversion and bit-depth-aware pixel IO shared by the plugin.
//
// The working assumption (same as the humbaba DA3 plugin): a FLOAT clip is
// linear scene-referred, ACEScg (AP1 primaries) by default, and must be
// converted to display-referred sRGB before inference because that is what the
// network was trained on. An INTEGER clip (8- or 16-bit) is already
// display-referred sRGB/Rec.709, so normalising 0-255 to [0,1] is all that is
// needed and the ACEScg step is skipped.
#pragma once

#include <algorithm>
#include <cmath>

#include "ofxsImageEffect.h"

namespace hyp {

inline float Clamp01(float v) { return std::min(std::max(v, 0.0f), 1.0f); }

// Linear -> sRGB transfer function.
inline float SrgbEncode(float c) {
  c = Clamp01(c);
  return c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

// sRGB -> linear transfer function.
inline float SrgbDecode(float c) {
  c = Clamp01(c);
  return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

// ACEScg (AP1, linear) -> sRGB-encoded Rec.709, per channel.
inline void AcescgToSrgb(float r, float g, float b, float* out) {
  // AP1 -> Rec.709 (linear) matrix.
  const float lr = 1.70505f * r - 0.62179f * g - 0.08326f * b;
  const float lg = -0.13026f * r + 1.14080f * g - 0.01055f * b;
  const float lb = -0.02400f * r - 0.12897f * g + 1.15297f * b;
  out[0] = SrgbEncode(lr);
  out[1] = SrgbEncode(lg);
  out[2] = SrgbEncode(lb);
}

// ---------------------------------------------------------------------------
// Pixel IO. Hosts hand us float, 8-bit or 16-bit buffers with 1 (Alpha), 3 (RGB)
// or 4 (RGBA) components. Everything is normalised to a float RGBA quad on read
// and written back in the destination's own depth/component layout.
// ---------------------------------------------------------------------------

inline int ComponentCount(OFX::PixelComponentEnum c) {
  switch (c) {
    case OFX::ePixelComponentRGBA: return 4;
    case OFX::ePixelComponentRGB: return 3;
    case OFX::ePixelComponentAlpha: return 1;
    default: return 0;
  }
}

inline bool IsIntegerDepth(OFX::BitDepthEnum bd) {
  return bd == OFX::eBitDepthUByte || bd == OFX::eBitDepthUShort;
}

// Read one pixel into rgba[4]. A 1-component (Alpha) source fills all of R,G,B
// with the value AND sets alpha to it, so a single-channel trimap clip reads
// correctly whichever channel the caller looks at. Missing alpha defaults to 1.
inline void ReadPixel(const void* p, OFX::BitDepthEnum bd, int nComps, float* rgba) {
  rgba[0] = rgba[1] = rgba[2] = 0.0f;
  rgba[3] = 1.0f;
  if (!p || nComps <= 0) return;

  float v[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  switch (bd) {
    case OFX::eBitDepthUByte: {
      const unsigned char* s = static_cast<const unsigned char*>(p);
      for (int i = 0; i < nComps; ++i) v[i] = s[i] * (1.0f / 255.0f);
      break;
    }
    case OFX::eBitDepthUShort: {
      const unsigned short* s = static_cast<const unsigned short*>(p);
      for (int i = 0; i < nComps; ++i) v[i] = s[i] * (1.0f / 65535.0f);
      break;
    }
    default: {  // eBitDepthFloat
      const float* s = static_cast<const float*>(p);
      for (int i = 0; i < nComps; ++i) v[i] = s[i];
      break;
    }
  }

  if (nComps == 1) {
    rgba[0] = rgba[1] = rgba[2] = rgba[3] = v[0];
  } else if (nComps == 3) {
    rgba[0] = v[0]; rgba[1] = v[1]; rgba[2] = v[2]; rgba[3] = 1.0f;
  } else {
    rgba[0] = v[0]; rgba[1] = v[1]; rgba[2] = v[2]; rgba[3] = v[3];
  }
}

// Write rgba[4] in the destination layout. A 1-component destination receives
// the ALPHA value (the matte), which is what a single-channel output clip wants.
// Integer destinations clamp and quantize; float destinations are written
// verbatim so scene-referred values above 1.0 survive.
inline void WritePixel(void* p, OFX::BitDepthEnum bd, int nComps, const float* rgba) {
  if (!p || nComps <= 0) return;
  const float src[4] = {rgba[0], rgba[1], rgba[2], rgba[3]};

  switch (bd) {
    case OFX::eBitDepthUByte: {
      unsigned char* d = static_cast<unsigned char*>(p);
      if (nComps == 1) {
        d[0] = static_cast<unsigned char>(std::lround(Clamp01(src[3]) * 255.0f));
      } else {
        for (int i = 0; i < nComps; ++i)
          d[i] = static_cast<unsigned char>(std::lround(Clamp01(src[i]) * 255.0f));
      }
      break;
    }
    case OFX::eBitDepthUShort: {
      unsigned short* d = static_cast<unsigned short*>(p);
      if (nComps == 1) {
        d[0] = static_cast<unsigned short>(std::lround(Clamp01(src[3]) * 65535.0f));
      } else {
        for (int i = 0; i < nComps; ++i)
          d[i] = static_cast<unsigned short>(std::lround(Clamp01(src[i]) * 65535.0f));
      }
      break;
    }
    default: {  // eBitDepthFloat
      float* d = static_cast<float*>(p);
      if (nComps == 1) {
        d[0] = src[3];
      } else {
        for (int i = 0; i < nComps; ++i) d[i] = src[i];
      }
      break;
    }
  }
}

}  // namespace hyp
