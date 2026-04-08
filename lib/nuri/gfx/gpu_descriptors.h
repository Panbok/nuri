#pragma once

#include "nuri/gfx/gpu_types.h"

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
  std::span<const std::byte> data{};
  uint32_t dataNumMipLevels = 1;
  bool generateMipmaps = false;
};

struct SamplerDesc {
  SamplerFilter minFilter = SamplerFilter::Linear;
  SamplerFilter magFilter = SamplerFilter::Linear;
  SamplerMipMode mipMode = SamplerMipMode::Disabled;
  SamplerWrapMode wrapU = SamplerWrapMode::Repeat;
  SamplerWrapMode wrapV = SamplerWrapMode::Repeat;
  SamplerWrapMode wrapW = SamplerWrapMode::Repeat;
  float mipLodMin = 0.0f;
  float mipLodMax = 15.0f;
  uint8_t maxAnisotropy = 1;
};

struct ShaderDesc {
  std::string_view moduleName{};
  std::string_view source{}; // GLSL source code
  ShaderStage stage = ShaderStage::Vertex;
};

struct RenderPipelineDesc {
  VertexInput vertexInput{};
  ShaderHandle vertexShader{};
  ShaderHandle tessControlShader{};
  ShaderHandle tessEvalShader{};
  ShaderHandle geometryShader{};
  ShaderHandle fragmentShader{};
  std::array<Format, 1> colorFormats{Format::RGBA8_UNORM};
  uint32_t colorAttachmentCount = 1;
  Format depthFormat = Format::Count;
  CullMode cullMode = CullMode::Back;
  PolygonMode polygonMode = PolygonMode::Fill;
  Topology topology = Topology::Triangle;
  uint32_t patchControlPoints = 0;
  bool blendEnabled = false;
  SpecializationInfo specInfo{};
};

struct ComputePipelineDesc {
  ShaderHandle computeShader{};
  SpecializationInfo specInfo{};
};

} // namespace nuri
