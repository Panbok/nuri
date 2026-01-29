#pragma once

#include "nuri/gfx/gpu_types.h"
#include "nuri/pch.h"

namespace nuri {

struct BufferDesc {
  BufferUsage usage = BufferUsage::Vertex;
  Storage storage = Storage::Device;
  size_t size = 0;
  std::span<const std::byte> data{};
};

struct TextureDesc {
  TextureType type = TextureType::Texture2D;
  Format format = Format::RGBA8_UNORM;
  TextureDimensions dimensions{1, 1, 1};
  TextureUsage usage = TextureUsage::Sampled;
  Storage storage = Storage::Device;
  uint32_t numLayers = 1;
  uint32_t numSamples = 1;
  uint32_t numMipLevels = 1;
  bool generateMipmaps = false;
};

struct ShaderDesc {
  std::string_view moduleName{};
  std::string_view source{}; // GLSL source code
  ShaderStage stage = ShaderStage::Vertex;
};

struct RenderPipelineDesc {
  VertexInput vertexInput{};
  ShaderHandle vertexShader{};
  ShaderHandle fragmentShader{};
  std::array<Format, 1> colorFormats{Format::RGBA8_UNORM};
  Format depthFormat = Format::D32_FLOAT;
  CullMode cullMode = CullMode::Back;
  PolygonMode polygonMode = PolygonMode::Fill;
  SpecializationInfo specInfo{};
};

struct ComputePipelineDesc {
  ShaderHandle computeShader{};
  SpecializationInfo specInfo{};
};

} // namespace nuri
