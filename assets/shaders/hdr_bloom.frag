#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_FragColor;

layout(push_constant) uniform HDRBloomPushConstants {
  uint sourceTexId;
  uint secondaryTexId;
  uint exposureTexId;
  uint sourceSamplerId;
  uint mode;
  uint flags;
  float threshold;
  float softKnee;
  float scatter;
  float manualExposureEv;
  float adaptationTargetGray;
  float adaptationMinEv;
  float adaptationMaxEv;
}
pc;

const uint kHDRBloomModePrefilterDownsample = 0u;
const uint kHDRBloomModeDownsample = 1u;
const uint kHDRBloomModeUpsample = 2u;
const uint kHDRBloomModeCopy = 3u;
const uint kHDRPostFlagAdaptationEnabled = 1u << 1u;
const uint kInvalidTextureBindlessIndex = 0xffffffffu;

float luminance(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }

float exposureEv() {
  float ev = pc.manualExposureEv;
  if ((pc.flags & kHDRPostFlagAdaptationEnabled) != 0u &&
      pc.exposureTexId != kInvalidTextureBindlessIndex) {
    float adaptedLuminance = max(
        textureBindless2D(pc.exposureTexId, pc.sourceSamplerId, vec2(0.5)).r,
        1.0e-4);
    ev += clamp(log2(max(pc.adaptationTargetGray, 1.0e-4) / adaptedLuminance),
                pc.adaptationMinEv, pc.adaptationMaxEv);
  }
  return ev;
}

vec3 bloomPrefilter(vec3 color) {
  float exposureScale = exp2(exposureEv());
  float luma = luminance(color * exposureScale);
  float knee = max(pc.softKnee, 1.0e-4);
  float soft = clamp((luma - pc.threshold + knee) / (2.0 * knee), 0.0, 1.0);
  float contribution = max(luma - pc.threshold, 0.0) + soft * soft * knee;
  return color * clamp(contribution / max(luma, 1.0e-4), 0.0, 16.0);
}

// fullscreen_copy.vert emits framebuffer-space UVs; wrap them to screen space.
vec2 screenUv() { return fract(uv); }

vec3 sampleBloom(uint texId, vec2 sampleUv, vec2 texel) {
  vec2 clampedUv = clamp(sampleUv, texel * 0.5, vec2(1.0) - texel * 0.5);
  return textureBindless2D(texId, pc.sourceSamplerId, clampedUv).rgb;
}

float karisWeight(vec3 color) { return 1.0 / (1.0 + luminance(color)); }

vec3 sampleKarisAverage(uint texId, vec2 texel, bool prefilter) {
  vec2 baseUv = screenUv();
  vec3 a = sampleBloom(texId, baseUv + texel * vec2(-2.0, -2.0), texel);
  vec3 b = sampleBloom(texId, baseUv + texel * vec2(0.0, -2.0), texel);
  vec3 c = sampleBloom(texId, baseUv + texel * vec2(2.0, -2.0), texel);
  vec3 d = sampleBloom(texId, baseUv + texel * vec2(-2.0, 0.0), texel);
  vec3 e = sampleBloom(texId, baseUv, texel);
  vec3 f = sampleBloom(texId, baseUv + texel * vec2(2.0, 0.0), texel);
  vec3 g = sampleBloom(texId, baseUv + texel * vec2(-2.0, 2.0), texel);
  vec3 h = sampleBloom(texId, baseUv + texel * vec2(0.0, 2.0), texel);
  vec3 i = sampleBloom(texId, baseUv + texel * vec2(2.0, 2.0), texel);
  vec3 j = sampleBloom(texId, baseUv + texel * vec2(-1.0, -1.0), texel);
  vec3 k = sampleBloom(texId, baseUv + texel * vec2(1.0, -1.0), texel);
  vec3 l = sampleBloom(texId, baseUv + texel * vec2(-1.0, 1.0), texel);
  vec3 m = sampleBloom(texId, baseUv + texel * vec2(1.0, 1.0), texel);
  if (prefilter) {
    a = bloomPrefilter(a);
    b = bloomPrefilter(b);
    c = bloomPrefilter(c);
    d = bloomPrefilter(d);
    e = bloomPrefilter(e);
    f = bloomPrefilter(f);
    g = bloomPrefilter(g);
    h = bloomPrefilter(h);
    i = bloomPrefilter(i);
    j = bloomPrefilter(j);
    k = bloomPrefilter(k);
    l = bloomPrefilter(l);
    m = bloomPrefilter(m);
  }

  vec3 wide = (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + e * 0.125;
  vec3 close = (j + k + l + m) * 0.125;
  if (!prefilter) {
    return wide + close;
  }

  float wa = karisWeight(a);
  float wb = karisWeight(b);
  float wc = karisWeight(c);
  float wd = karisWeight(d);
  float we = karisWeight(e);
  float wf = karisWeight(f);
  float wg = karisWeight(g);
  float wh = karisWeight(h);
  float wi = karisWeight(i);
  float wj = karisWeight(j);
  float wk = karisWeight(k);
  float wl = karisWeight(l);
  float wm = karisWeight(m);
  vec3 weighted = (a * wa + c * wc + g * wg + i * wi) * 0.03125 +
                  (b * wb + d * wd + f * wf + h * wh) * 0.0625 +
                  e * we * 0.125 + (j * wj + k * wk + l * wl + m * wm) * 0.125;
  float weightSum = (wa + wc + wg + wi) * 0.03125 +
                    (wb + wd + wf + wh) * 0.0625 + we * 0.125 +
                    (wj + wk + wl + wm) * 0.125;
  return weighted / max(weightSum, 1.0e-4);
}

vec3 sampleTent(uint texId, vec2 texel) {
  vec2 baseUv = screenUv();
  vec3 sum = vec3(0.0);
  sum += sampleBloom(texId, baseUv + texel * vec2(-1.0, -1.0), texel) * 1.0;
  sum += sampleBloom(texId, baseUv + texel * vec2(0.0, -1.0), texel) * 2.0;
  sum += sampleBloom(texId, baseUv + texel * vec2(1.0, -1.0), texel) * 1.0;
  sum += sampleBloom(texId, baseUv + texel * vec2(-1.0, 0.0), texel) * 2.0;
  sum += sampleBloom(texId, baseUv, texel) * 4.0;
  sum += sampleBloom(texId, baseUv + texel * vec2(1.0, 0.0), texel) * 2.0;
  sum += sampleBloom(texId, baseUv + texel * vec2(-1.0, 1.0), texel) * 1.0;
  sum += sampleBloom(texId, baseUv + texel * vec2(0.0, 1.0), texel) * 2.0;
  sum += sampleBloom(texId, baseUv + texel * vec2(1.0, 1.0), texel) * 1.0;
  return sum * (1.0 / 16.0);
}

void main() {
  ivec2 sourceSize = textureBindlessSize2D(pc.sourceTexId);
  vec2 sourceTexel = 1.0 / max(vec2(sourceSize), vec2(1.0));

  if (pc.mode == kHDRBloomModePrefilterDownsample) {
    out_FragColor =
        vec4(sampleKarisAverage(pc.sourceTexId, sourceTexel, true), 1.0);
    return;
  }
  if (pc.mode == kHDRBloomModeDownsample) {
    out_FragColor =
        vec4(sampleKarisAverage(pc.sourceTexId, sourceTexel, false), 1.0);
    return;
  }

  vec3 low = sampleTent(pc.sourceTexId, sourceTexel);
  if (pc.mode == kHDRBloomModeUpsample &&
      pc.secondaryTexId != kInvalidTextureBindlessIndex) {
    ivec2 secondarySize = textureBindlessSize2D(pc.secondaryTexId);
    vec2 secondaryTexel = 1.0 / max(vec2(secondarySize), vec2(1.0));
    vec3 high = sampleBloom(pc.secondaryTexId, screenUv(), secondaryTexel);
    out_FragColor = vec4(high + low * pc.scatter, 1.0);
    return;
  }

  out_FragColor = vec4(low, 1.0);
}
