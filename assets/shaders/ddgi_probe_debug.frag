#version 460 core

layout(location = 0) in vec2 inLocal;
layout(location = 1) in vec4 inColor;
layout(location = 0) out vec4 outColor;

void main() {
  const float radiusSquared = dot(inLocal, inLocal);
  if (radiusSquared > 1.0)
    discard;
  const float rim = smoothstep(1.0, 0.72, radiusSquared);
  outColor = vec4(inColor.rgb * mix(0.55, 1.0, rim), inColor.a);
}
