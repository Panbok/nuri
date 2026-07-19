#pragma once
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_render_types.h"
namespace nuri {

[[nodiscard]] inline RenderPipelineDesc
fullscreenPipelineDesc(Format colorFormat, ShaderHandle vertex,
                       ShaderHandle fragment) {
  return RenderPipelineDesc{
      .vertexInput = {},
      .vertexShader = vertex,
      .fragmentShader = fragment,
      .colorFormats = {colorFormat},
      .depthFormat = Format::Count,
      .cullMode = CullMode::None,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Triangle,
      .blendEnabled = false,
  };
}

[[nodiscard]] inline DrawItem
makeFullscreenDraw(RenderPipelineHandle pipeline,
                   std::span<const std::byte> pushConstants,
                   std::string_view label, uint32_t debugColor) {
  return DrawItem{.pipeline = pipeline,
                  .vertexCount = 3u,
                  .instanceCount = 1u,
                  .pushConstants = pushConstants,
                  .debugLabel = label,
                  .debugColor = debugColor};
}

}
