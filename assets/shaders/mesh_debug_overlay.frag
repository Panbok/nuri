#version 460
#include "common.sp"

layout(location = 0) in PerVertex vtx;

layout(location = 0) out vec4 out_FragColor;

float edgeCoverage(vec3 bary, float edgeWidthPx) {
  const vec3 clampedBary = max(bary, vec3(0.0));
  const vec3 derivative = fwidth(clampedBary);
  const vec3 aa = smoothstep(vec3(0.0), derivative * edgeWidthPx, clampedBary);
  return 1.0 - min(min(aa.x, aa.y), aa.z);
}

vec3 heatColor(float t) {
  const vec3 cold = vec3(0.12, 0.35, 0.95);
  const vec3 mid = vec3(0.15, 0.90, 0.30);
  const vec3 hot = vec3(1.00, 0.25, 0.10);
  return t < 0.5 ? mix(cold, mid, t * 2.0) : mix(mid, hot, (t - 0.5) * 2.0);
}

void main() {
  const bool isTessellated = vtx.tessellatedFlag >= 0.5;
  const float triEdge = edgeCoverage(vtx.triBarycentric, 1.1);
  const float patchEdge = edgeCoverage(vtx.patchBarycentric, 2.8);
  if (pc.debugVisualizationMode == kDebugVisualizationNone) {
    discard;
  }

  if (pc.debugVisualizationMode == kDebugVisualizationWireOverlay) {
    // Tessellated draws: show only coarse patch boundaries.
    const float edge = isTessellated ? patchEdge : triEdge;
    const float alpha = smoothstep(0.10, 0.95, edge) * 0.70;
    if (alpha <= 1.0e-4) {
      discard;
    }
    out_FragColor = vec4(0.18, 0.82, 0.92, alpha);
    return;
  }
  if (pc.debugVisualizationMode == kDebugVisualizationWireframeOnly) {
    discard;
  }

  if (!isTessellated) {
    const float alpha = smoothstep(0.10, 0.95, triEdge) * 0.70;
    if (alpha <= 1.0e-4) {
      discard;
    }
    out_FragColor = vec4(0.18, 0.82, 0.92, alpha);
    return;
  }

  const float avgTessFactor =
      (vtx.patchOuterFactors.x + vtx.patchOuterFactors.y +
       vtx.patchOuterFactors.z + vtx.patchInnerFactor) *
      0.25;
  const float heatT = clamp((avgTessFactor - 1.0) * (1.0 / 63.0), 0.0, 1.0);
  const float patchMask = smoothstep(0.04, 0.98, patchEdge);
  const float alpha = patchMask * 0.95;

  if (alpha <= 1.0e-4) {
    discard;
  }

  const vec3 patchColor = mix(heatColor(heatT), vec3(1.00, 0.30, 0.12), 0.45);
  out_FragColor = vec4(patchColor, alpha);
}
