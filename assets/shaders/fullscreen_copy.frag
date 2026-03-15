#include "common.sp"

layout(location = 0) in vec2 outUv;
layout(location = 0) out vec4 out_FragColor;

void main() {
  out_FragColor = textureBindless2D(pc.frameData.sceneColorTexId,
                                    pc.frameData.sceneColorSamplerId, outUv);
}
