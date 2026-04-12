layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_FragColor;

layout(push_constant) uniform PresentPushConstants {
  uint sourceTexId;
  uint sourceSamplerId;
  uint acesLutTexId;
  uint agxLutTexId;
  uint lutSamplerId;
  uint flags;
  float acesExposureScale;
  float agxExposureScale;
  float compareSplit;
  float shaperMinLog2;
  float shaperInvRange;
}
pcPresent;

const uint kPresentFlagManualSrgbEncode = 1u << 0u;
const uint kPresentFlagPrimaryUseAgx = 1u << 1u;
const uint kPresentFlagCompareEnabled = 1u << 2u;
const uint kPresentFlagGrayCardDebug = 1u << 3u;
const uint kPresentFlagAcesLutAvailable = 1u << 4u;
const uint kPresentFlagAgxLutAvailable = 1u << 5u;
const float kCompareSplitMin = 0.1;
const float kCompareSplitMax = 0.9;
const vec2 kLabelGlyphSize = vec2(0.050, 0.086);
const float kLabelGlyphSpacing = 0.014;
const vec2 kLabelPadding = vec2(0.020, 0.018);
const vec2 kGrayCardPanelOrigin = vec2(0.18, 0.38);
const vec2 kGrayCardPanelSize = vec2(0.64, 0.24);
const float kGrayCardOuterBorder = 0.035;
const float kGrayCardInnerBorder = 0.07;
const float kLutSize = 64.0;
const vec2 kLutTileGrid = vec2(8.0, 8.0);
const vec2 kLutAtlasSize = vec2(512.0, 512.0);

// IEC 61966-2-1 sRGB piecewise OETF for swapchain presentation.
vec3 linearToSrgb(vec3 c) {
  const bvec3 useLinear = lessThanEqual(c, vec3(0.0031308));
  const vec3 linear = c * 12.92;
  const vec3 nonlinear =
      1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
  return mix(nonlinear, linear, useLinear);
}

vec3 srgbToLinear(vec3 c) {
  const bvec3 useLinear = lessThanEqual(c, vec3(0.04045));
  const vec3 linear = c / 12.92;
  const vec3 nonlinear = pow(max((c + 0.055) / 1.055, vec3(0.0)), vec3(2.4));
  return mix(nonlinear, linear, useLinear);
}

vec3 acesFitted(vec3 x) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 shapeForToneMapLut(vec3 color) {
  return clamp(
      (log2(max(color, vec3(1.0e-6))) - vec3(pcPresent.shaperMinLog2)) *
          vec3(pcPresent.shaperInvRange),
      0.0, 1.0);
}

vec3 samplePackedToneMapLut(uint lutTexId, vec3 color) {
  vec3 shaped = shapeForToneMapLut(color);
  float slice = shaped.b * (kLutSize - 1.0);
  float sliceFloor = floor(slice);
  float sliceCeil = min(sliceFloor + 1.0, kLutSize - 1.0);
  float sliceMix = slice - sliceFloor;
  vec2 uvWithinSlice =
      ((shaped.rg * (kLutSize - 1.0)) + vec2(0.5)) / kLutAtlasSize;
  vec3 low = textureBindless2D(lutTexId, pcPresent.lutSamplerId,
                               (vec2(mod(sliceFloor, kLutTileGrid.x),
                                     floor(sliceFloor / kLutTileGrid.x)) /
                                kLutTileGrid) +
                                   uvWithinSlice)
                 .rgb;
  vec3 high = textureBindless2D(lutTexId, pcPresent.lutSamplerId,
                                (vec2(mod(sliceCeil, kLutTileGrid.x),
                                      floor(sliceCeil / kLutTileGrid.x)) /
                                 kLutTileGrid) +
                                    uvWithinSlice)
                  .rgb;
  return mix(low, high, sliceMix);
}

bool usesManualSrgbEncode() {
  return (pcPresent.flags & kPresentFlagManualSrgbEncode) != 0u;
}

bool isPrimaryAgx() {
  return (pcPresent.flags & kPresentFlagPrimaryUseAgx) != 0u;
}

bool isCompareEnabled() {
  return (pcPresent.flags & kPresentFlagCompareEnabled) != 0u;
}

bool isGrayCardDebugEnabled() {
  return (pcPresent.flags & kPresentFlagGrayCardDebug) != 0u;
}

bool hasAcesLut() {
  return (pcPresent.flags & kPresentFlagAcesLutAvailable) != 0u;
}

bool hasAgxLut() {
  return (pcPresent.flags & kPresentFlagAgxLutAvailable) != 0u;
}

float exposureScaleForToneMapper(bool useAgx) {
  return useAgx ? pcPresent.agxExposureScale : pcPresent.acesExposureScale;
}

vec3 toneMapExposedColor(vec3 exposed, bool useAgx) {
  bool useLut = useAgx ? hasAgxLut() : hasAcesLut();
  vec3 mapped = vec3(0.0);
  if (useLut) {
    uint lutTexId = useAgx ? pcPresent.agxLutTexId : pcPresent.acesLutTexId;
    mapped = samplePackedToneMapLut(lutTexId, exposed);
    if (!usesManualSrgbEncode()) {
      mapped = srgbToLinear(mapped);
    }
    return mapped;
  }

  mapped = acesFitted(exposed);
  if (usesManualSrgbEncode()) {
    mapped = linearToSrgb(mapped);
  }
  return mapped;
}

uint glyphRowBits(int glyphId, int row) {
  if (glyphId == 0) { // A
    const uint rows[7] =
        uint[7](0x0Eu, 0x11u, 0x11u, 0x1Fu, 0x11u, 0x11u, 0x11u);
    return rows[row];
  }
  if (glyphId == 1) { // C
    const uint rows[7] =
        uint[7](0x0Fu, 0x10u, 0x10u, 0x10u, 0x10u, 0x10u, 0x0Fu);
    return rows[row];
  }
  if (glyphId == 2) { // E
    const uint rows[7] =
        uint[7](0x1Fu, 0x10u, 0x10u, 0x1Eu, 0x10u, 0x10u, 0x1Fu);
    return rows[row];
  }
  if (glyphId == 3) { // S
    const uint rows[7] =
        uint[7](0x0Fu, 0x10u, 0x10u, 0x0Eu, 0x01u, 0x01u, 0x1Eu);
    return rows[row];
  }
  if (glyphId == 4) { // G
    const uint rows[7] =
        uint[7](0x0Fu, 0x10u, 0x10u, 0x13u, 0x11u, 0x11u, 0x0Fu);
    return rows[row];
  }
  const uint rows[7] = uint[7](0x11u, 0x0Au, 0x04u, 0x04u, 0x04u, 0x0Au,
                               0x11u); // X
  return rows[row];
}

bool sampleGlyphMask(vec2 glyphUv, int glyphId) {
  if (any(lessThan(glyphUv, vec2(0.0))) ||
      any(greaterThanEqual(glyphUv, vec2(1.0)))) {
    return false;
  }

  ivec2 cell = ivec2(floor(glyphUv * vec2(5.0, 7.0)));
  int row = clamp(cell.y, 0, 6);
  int col = clamp(cell.x, 0, 4);
  uint bits = glyphRowBits(glyphId, row);
  return ((bits >> uint(4 - col)) & 1u) != 0u;
}

int labelGlyphId(bool useAgx, int glyphIndex) {
  if (useAgx) {
    if (glyphIndex == 0) {
      return 0;
    }
    if (glyphIndex == 1) {
      return 4;
    }
    return 5;
  }

  if (glyphIndex == 0) {
    return 0;
  }
  if (glyphIndex == 1) {
    return 1;
  }
  if (glyphIndex == 2) {
    return 2;
  }
  return 3;
}

bool sampleCompareLabelHdr(vec2 localUv, bool useAgx, out vec3 hdrOverride) {
  if (!isCompareEnabled()) {
    return false;
  }

  int glyphCount = useAgx ? 3 : 4;
  float textWidth = float(glyphCount) * kLabelGlyphSize.x +
                    float(glyphCount - 1) * kLabelGlyphSpacing;
  vec2 labelSize = vec2(textWidth, kLabelGlyphSize.y) + 2.0 * kLabelPadding;
  vec2 labelOrigin = vec2(0.5 - 0.5 * labelSize.x,
                          kGrayCardPanelOrigin.y - labelSize.y - 0.05);
  vec2 labelLocal = localUv - labelOrigin;
  vec2 labelUv = labelLocal / labelSize;
  if (any(lessThan(labelUv, vec2(0.0))) ||
      any(greaterThanEqual(labelUv, vec2(1.0)))) {
    return false;
  }

  const float border = 0.06;
  if (labelUv.x < border || labelUv.x >= 1.0 - border || labelUv.y < border ||
      labelUv.y >= 1.0 - border) {
    hdrOverride = vec3(1.0);
    return true;
  }

  hdrOverride = vec3(0.0);
  vec2 contentPos = labelLocal - kLabelPadding;
  if (contentPos.x < 0.0 || contentPos.x >= textWidth || contentPos.y < 0.0 ||
      contentPos.y >= kLabelGlyphSize.y) {
    return true;
  }

  float glyphAdvance = kLabelGlyphSize.x + kLabelGlyphSpacing;
  int glyphIndex = int(floor(contentPos.x / glyphAdvance));
  if (glyphIndex < 0 || glyphIndex >= glyphCount) {
    return true;
  }
  float glyphLocalX = contentPos.x - float(glyphIndex) * glyphAdvance;
  if (glyphLocalX >= kLabelGlyphSize.x) {
    return true;
  }
  vec2 glyphUv =
      vec2(glyphLocalX / kLabelGlyphSize.x, contentPos.y / kLabelGlyphSize.y);
  if (sampleGlyphMask(glyphUv, labelGlyphId(useAgx, glyphIndex))) {
    hdrOverride = vec3(1.0);
  }
  return true;
}

vec2 currentScreenUv() {
  ivec2 sourceSize = textureBindlessSize2D(pcPresent.sourceTexId);
  vec2 safeSize = max(vec2(sourceSize), vec2(1.0));
  return clamp(gl_FragCoord.xy / safeSize, vec2(0.0), vec2(1.0));
}

vec2 sideLocalUv(vec2 screenUv, bool compareSide) {
  if (!isCompareEnabled()) {
    return screenUv;
  }
  float split =
      clamp(pcPresent.compareSplit, kCompareSplitMin, kCompareSplitMax);
  if (compareSide) {
    return vec2((screenUv.x - split) / max(1.0 - split, 1.0e-5), screenUv.y);
  }
  return vec2(screenUv.x / max(split, 1.0e-5), screenUv.y);
}

bool sampleGrayCardHdr(vec2 localUv, out vec3 hdrOverride) {
  if (!isGrayCardDebugEnabled()) {
    return false;
  }
  vec2 panelUv = (localUv - kGrayCardPanelOrigin) / kGrayCardPanelSize;
  if (any(lessThan(panelUv, vec2(0.0))) ||
      any(greaterThanEqual(panelUv, vec2(1.0)))) {
    return false;
  }

  if (panelUv.x < kGrayCardOuterBorder ||
      panelUv.x >= 1.0 - kGrayCardOuterBorder ||
      panelUv.y < kGrayCardOuterBorder ||
      panelUv.y >= 1.0 - kGrayCardOuterBorder) {
    hdrOverride = vec3(0.0);
    return true;
  }
  if (panelUv.x < kGrayCardInnerBorder ||
      panelUv.x >= 1.0 - kGrayCardInnerBorder ||
      panelUv.y < kGrayCardInnerBorder ||
      panelUv.y >= 1.0 - kGrayCardInnerBorder) {
    hdrOverride = vec3(1.0);
    return true;
  }

  vec2 innerUv = (panelUv - vec2(kGrayCardInnerBorder)) /
                 (1.0 - 2.0 * kGrayCardInnerBorder);
  if (innerUv.x < 0.25) {
    hdrOverride = vec3(0.0);
  } else if (innerUv.x < 0.50) {
    hdrOverride = vec3(0.18);
  } else if (innerUv.x < 0.75) {
    hdrOverride = vec3(1.0);
  } else {
    hdrOverride = vec3(4.0);
  }
  if (abs(innerUv.x - 0.25) < 0.01 || abs(innerUv.x - 0.50) < 0.01 ||
      abs(innerUv.x - 0.75) < 0.01) {
    hdrOverride = vec3(0.0);
  }
  return true;
}

bool sampleToneMapOverlayHdr(vec2 localUv, bool useAgx, out vec3 hdrOverride) {
  vec3 overlay = vec3(0.0);
  if (sampleCompareLabelHdr(localUv, useAgx, overlay)) {
    hdrOverride = overlay;
    return true;
  }
  if (sampleGrayCardHdr(localUv, overlay)) {
    hdrOverride = overlay;
    return true;
  }
  return false;
}

void main() {
  vec4 hdr =
      textureBindless2D(pcPresent.sourceTexId, pcPresent.sourceSamplerId, uv);
  vec2 screenUv = currentScreenUv();
  float split =
      clamp(pcPresent.compareSplit, kCompareSplitMin, kCompareSplitMax);
  bool compareSide = isCompareEnabled() && screenUv.x >= split;
  vec2 localUv = sideLocalUv(screenUv, compareSide);
  vec3 hdrColor = hdr.rgb;
  bool useAgx = compareSide ? !isPrimaryAgx() : isPrimaryAgx();
  vec3 overlayHdr = vec3(0.0);
  if (sampleToneMapOverlayHdr(localUv, useAgx, overlayHdr)) {
    hdrColor = overlayHdr;
  }

  vec3 exposed = max(hdrColor * exposureScaleForToneMapper(useAgx), vec3(0.0));
  vec3 mapped = toneMapExposedColor(exposed, useAgx);
  if (isCompareEnabled()) {
    float dividerWidth = max(fwidth(screenUv.x) * 2.0, 1.0 / 4096.0);
    float dividerMask =
        1.0 - smoothstep(0.0, dividerWidth, abs(screenUv.x - split));
    mapped = mix(mapped, vec3(1.0), dividerMask);
  }
  out_FragColor = vec4(mapped, hdr.a);
}
