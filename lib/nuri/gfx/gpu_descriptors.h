#pragma once
#include "nuri/gfx/gpu_types.h"
#include <vector>
namespace nuri {

struct BufferDesc {
  BufferUsage usage = BufferUsage::Vertex;
  Storage storage = Storage::Device;
  size_t size = 0;
  std::span<const std::byte> data{};
  bool immutable = false;
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
  float mipLodBias = 0.0f;
  uint8_t maxAnisotropy = 1;
  bool depthCompareEnabled = false;
  CompareOp depthCompareOp = CompareOp::LessEqual;
  constexpr bool operator==(const SamplerDesc &) const noexcept = default;
};

struct ShaderDesc {
  std::string_view moduleName{};
  std::string_view source{};
  ShaderStage stage = ShaderStage::Vertex;
};

struct RenderPipelineDesc {
  VertexInput vertexInput{};
  ShaderHandle vertexShader{};
  ShaderHandle tessControlShader{};
  ShaderHandle tessEvalShader{};
  ShaderHandle geometryShader{};
  ShaderHandle fragmentShader{};
  std::vector<Format> colorFormats{Format::RGBA8_UNORM};
  uint32_t colorAttachmentCount = 1;
  Format depthFormat = Format::Count;
  CullMode cullMode = CullMode::Back;
  PolygonMode polygonMode = PolygonMode::Fill;
  Topology topology = Topology::Triangle;
  uint32_t patchControlPoints = 0;
  uint32_t numSamples = 1u;
  float minSampleShading = 0.0f;
  bool alphaToCoverageEnabled = false;
  bool blendEnabled = false;
  SpecializationInfo specInfo{};
  RasterPipelineState rasterState{};
  std::span<const RasterPipelineState> prewarmRasterStates{};
};

struct ComputePipelineDesc {
  ShaderHandle computeShader{};
  SpecializationInfo specInfo{};
};

struct MeshletPipelineDesc {
  ShaderHandle taskShader{};
  ShaderHandle meshShader{};
  ShaderHandle fragmentShader{};
  std::vector<Format> colorFormats{Format::RGBA8_UNORM};
  uint32_t colorAttachmentCount = 1;
  Format depthFormat = Format::Count;
  CullMode cullMode = CullMode::Back;
  PolygonMode polygonMode = PolygonMode::Fill;
  uint32_t numSamples = 1u;
  float minSampleShading = 0.0f;
  bool alphaToCoverageEnabled = false;
  bool blendEnabled = false;
  SpecializationInfo specInfo{};
  RasterPipelineState rasterState{};
  std::span<const RasterPipelineState> prewarmRasterStates{};
};

} // namespace nuri
