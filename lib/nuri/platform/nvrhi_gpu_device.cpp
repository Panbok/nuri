#include "nuri/pch.h"

#include "nuri/platform/nvrhi_gpu_device.h"

#include "nuri/core/containers/slot_pool.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/core/window.h"
#include "nuri/gfx/recording_retirement_tracker.h"
#include "nuri/resources/gpu/geometry_pool.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/utils/env_utils.h"

#if !defined(VK_NO_PROTOTYPES)
#define VK_NO_PROTOTYPES 1
#endif
#include <volk.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glslang/Include/glslang_c_interface.h>
#include <nvrhi/nvrhi.h>
#include <nvrhi/vulkan.h>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstring>
#include <deque>
#include <format>
#include <functional>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace nuri {
namespace {

constexpr uint32_t kMaxGraphicsRecordingContexts = 1u;
constexpr uint32_t kSwapchainFramesInFlight = 2u;
constexpr uint32_t kPreferredSwapchainImageCount = 3u;
constexpr uint32_t kBindlessCapacity = 4096u;
constexpr uint32_t kMaxNvrhiGpuTimerQueries = 2048u;
constexpr uint32_t kWholeFrameTimingSlotCount = kSwapchainFramesInFlight + 1u;
constexpr size_t kPushConstantByteSize = 128u;
constexpr uint32_t kDefaultMeshletMaxVertices = 64u;
constexpr uint32_t kDefaultMeshletMaxPrimitives = 124u;
constexpr uint32_t kDefaultMeshletWorkGroupSize = 32u;
constexpr std::array<uint8_t, 4> kSupportedAnisotropyLevels = {2u, 4u, 8u, 16u};
constexpr uint16_t kSpvOpExecutionMode = 16u;
constexpr uint16_t kSpvOpConstant = 43u;
constexpr uint16_t kSpvOpSpecConstant = 50u;
constexpr uint16_t kSpvOpExecutionModeId = 331u;
constexpr uint32_t kSpvExecutionModeLocalSize = 17u;
constexpr uint32_t kSpvExecutionModeLocalSizeId = 38u;
constexpr size_t kSpvHeaderWordCount = 5u;

[[nodiscard]] bool normalizeSpirvLocalSizeId(std::vector<uint8_t> &spirv) {
  if (spirv.size() % sizeof(uint32_t) != 0u ||
      spirv.size() < kSpvHeaderWordCount * sizeof(uint32_t)) {
    return false;
  }

  std::vector<uint32_t> words(spirv.size() / sizeof(uint32_t));
  std::memcpy(words.data(), spirv.data(), spirv.size());

  std::unordered_map<uint32_t, uint32_t> scalarConstants;
  for (size_t wordIndex = kSpvHeaderWordCount; wordIndex < words.size();) {
    const uint32_t instruction = words[wordIndex];
    const uint16_t wordCount = static_cast<uint16_t>(instruction >> 16u);
    const uint16_t opcode = static_cast<uint16_t>(instruction & 0xffffu);
    if (wordCount == 0u || wordIndex + wordCount > words.size()) {
      return false;
    }

    if ((opcode == kSpvOpConstant || opcode == kSpvOpSpecConstant) &&
        wordCount >= 4u) {
      scalarConstants[words[wordIndex + 2u]] = words[wordIndex + 3u];
    }
    wordIndex += wordCount;
  }

  bool changed = false;
  for (size_t wordIndex = kSpvHeaderWordCount; wordIndex < words.size();) {
    const uint32_t instruction = words[wordIndex];
    const uint16_t wordCount = static_cast<uint16_t>(instruction >> 16u);
    const uint16_t opcode = static_cast<uint16_t>(instruction & 0xffffu);
    if (wordCount == 0u || wordIndex + wordCount > words.size()) {
      return false;
    }

    if (opcode == kSpvOpExecutionModeId && wordCount == 6u &&
        words[wordIndex + 2u] == kSpvExecutionModeLocalSizeId) {
      const auto x = scalarConstants.find(words[wordIndex + 3u]);
      const auto y = scalarConstants.find(words[wordIndex + 4u]);
      const auto z = scalarConstants.find(words[wordIndex + 5u]);
      if (x != scalarConstants.end() && y != scalarConstants.end() &&
          z != scalarConstants.end()) {
        words[wordIndex] =
            (static_cast<uint32_t>(wordCount) << 16u) | kSpvOpExecutionMode;
        words[wordIndex + 2u] = kSpvExecutionModeLocalSize;
        words[wordIndex + 3u] = x->second;
        words[wordIndex + 4u] = y->second;
        words[wordIndex + 5u] = z->second;
        changed = true;
      }
    }
    wordIndex += wordCount;
  }

  if (changed) {
    std::memcpy(spirv.data(), words.data(), spirv.size());
  }
  return changed;
}

template <typename HandleType>
[[nodiscard]] constexpr bool areSameHandle(HandleType a, HandleType b) {
  return a.index == b.index && a.generation == b.generation;
}

[[nodiscard]] bool isRenderDocAttached() {
#if defined(_WIN32)
  if (GetModuleHandleA("renderdoc.dll") != nullptr ||
      GetModuleHandleA("renderdoccmd.dll") != nullptr) {
    return true;
  }
#endif
  return readEnvVar("RENDERDOC_CAPFILE").has_value();
}

[[nodiscard]] bool resolveValidationEnabled(bool renderDocAttached) {
  if (const std::optional<bool> override =
          readEnvBoolOverride("NURI_VK_VALIDATION");
      override.has_value()) {
    return *override;
  }
#if defined(NURI_DEBUG)
  return true;
#else
  return renderDocAttached;
#endif
}

[[nodiscard]] std::string formatVkVersion(uint32_t version) {
  std::array<char, 32> text{};
  const int written =
      std::snprintf(text.data(), text.size(), "%u.%u.%u",
                    static_cast<unsigned>(VK_API_VERSION_MAJOR(version)),
                    static_cast<unsigned>(VK_API_VERSION_MINOR(version)),
                    static_cast<unsigned>(VK_API_VERSION_PATCH(version)));
  if (written <= 0 || static_cast<size_t>(written) >= text.size()) {
    return std::to_string(version);
  }
  return std::string(text.data(), static_cast<size_t>(written));
}

[[nodiscard]] nvrhi::Format toNvrhiFormat(Format format) {
  switch (format) {
  case Format::R32_UINT:
    return nvrhi::Format::R32_UINT;
  case Format::R8_UNORM:
    return nvrhi::Format::R8_UNORM;
  case Format::R16_UNORM:
    return nvrhi::Format::R16_UNORM;
  case Format::R32_FLOAT:
    return nvrhi::Format::R32_FLOAT;
  case Format::RG32_FLOAT:
    return nvrhi::Format::RG32_FLOAT;
  case Format::RG16_FLOAT:
    return nvrhi::Format::RG16_FLOAT;
  case Format::RGBA8_UNORM:
    return nvrhi::Format::RGBA8_UNORM;
  case Format::RGBA8_SRGB:
    return nvrhi::Format::SRGBA8_UNORM;
  case Format::RGBA8_UINT:
    return nvrhi::Format::RGBA8_UINT;
  case Format::RGBA16_FLOAT:
    return nvrhi::Format::RGBA16_FLOAT;
  case Format::RGBA32_FLOAT:
    return nvrhi::Format::RGBA32_FLOAT;
  case Format::D16_UNORM:
    return nvrhi::Format::D16;
  case Format::D32_FLOAT:
    return nvrhi::Format::D32;
  case Format::BC7_RGBA_UNORM:
    return nvrhi::Format::BC7_UNORM;
  case Format::BC7_RGBA_SRGB:
    return nvrhi::Format::BC7_UNORM_SRGB;
  case Format::ETC2_RGB8_UNORM:
  case Format::ETC2_RGB8_SRGB:
  case Format::Count:
    break;
  }
  return nvrhi::Format::UNKNOWN;
}

[[nodiscard]] VkFormat toVkFormat(Format format) {
  switch (format) {
  case Format::R32_UINT:
    return VK_FORMAT_R32_UINT;
  case Format::R8_UNORM:
    return VK_FORMAT_R8_UNORM;
  case Format::R16_UNORM:
    return VK_FORMAT_R16_UNORM;
  case Format::R32_FLOAT:
    return VK_FORMAT_R32_SFLOAT;
  case Format::RG32_FLOAT:
    return VK_FORMAT_R32G32_SFLOAT;
  case Format::RG16_FLOAT:
    return VK_FORMAT_R16G16_SFLOAT;
  case Format::RGBA8_UNORM:
    return VK_FORMAT_R8G8B8A8_UNORM;
  case Format::RGBA8_SRGB:
    return VK_FORMAT_R8G8B8A8_SRGB;
  case Format::RGBA8_UINT:
    return VK_FORMAT_R8G8B8A8_UINT;
  case Format::RGBA16_FLOAT:
    return VK_FORMAT_R16G16B16A16_SFLOAT;
  case Format::RGBA32_FLOAT:
    return VK_FORMAT_R32G32B32A32_SFLOAT;
  case Format::D16_UNORM:
    return VK_FORMAT_D16_UNORM;
  case Format::D32_FLOAT:
    return VK_FORMAT_D32_SFLOAT;
  case Format::BC7_RGBA_UNORM:
    return VK_FORMAT_BC7_UNORM_BLOCK;
  case Format::BC7_RGBA_SRGB:
    return VK_FORMAT_BC7_SRGB_BLOCK;
  case Format::ETC2_RGB8_UNORM:
    return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
  case Format::ETC2_RGB8_SRGB:
    return VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK;
  case Format::Count:
    break;
  }
  return VK_FORMAT_UNDEFINED;
}

[[nodiscard]] nvrhi::Format toNvrhiSwapchainFormat(VkFormat format) {
  switch (format) {
  case VK_FORMAT_B8G8R8A8_UNORM:
    return nvrhi::Format::BGRA8_UNORM;
  case VK_FORMAT_B8G8R8A8_SRGB:
    return nvrhi::Format::SBGRA8_UNORM;
  case VK_FORMAT_R8G8B8A8_SRGB:
    return nvrhi::Format::SRGBA8_UNORM;
  case VK_FORMAT_R8G8B8A8_UNORM:
  default:
    return nvrhi::Format::RGBA8_UNORM;
  }
}

[[nodiscard]] Format toNuriSwapchainFormat(VkFormat format) {
  switch (format) {
  case VK_FORMAT_R8G8B8A8_SRGB:
  case VK_FORMAT_B8G8R8A8_SRGB:
    return Format::RGBA8_SRGB;
  case VK_FORMAT_R8G8B8A8_UNORM:
  case VK_FORMAT_B8G8R8A8_UNORM:
  default:
    return Format::RGBA8_UNORM;
  }
}

[[nodiscard]] nvrhi::TextureDimension
toNvrhiTextureDimension(TextureType type, uint32_t samples, uint32_t layers) {
  switch (type) {
  case TextureType::Texture3D:
    return nvrhi::TextureDimension::Texture3D;
  case TextureType::TextureCube:
    return layers > 1u ? nvrhi::TextureDimension::TextureCubeArray
                       : nvrhi::TextureDimension::TextureCube;
  case TextureType::Texture2D:
  case TextureType::Count:
  default:
    if (samples > 1u) {
      return layers > 1u ? nvrhi::TextureDimension::Texture2DMSArray
                         : nvrhi::TextureDimension::Texture2DMS;
    }
    return layers > 1u ? nvrhi::TextureDimension::Texture2DArray
                       : nvrhi::TextureDimension::Texture2D;
  }
}

[[nodiscard]] nvrhi::SamplerAddressMode
toNvrhiAddressMode(SamplerWrapMode mode) {
  switch (mode) {
  case SamplerWrapMode::Repeat:
    return nvrhi::SamplerAddressMode::Repeat;
  case SamplerWrapMode::MirrorRepeat:
    return nvrhi::SamplerAddressMode::MirroredRepeat;
  case SamplerWrapMode::Clamp:
  case SamplerWrapMode::Count:
  default:
    return nvrhi::SamplerAddressMode::ClampToEdge;
  }
}

[[nodiscard]] nvrhi::ComparisonFunc toNvrhiCompareOp(CompareOp op) {
  switch (op) {
  case CompareOp::Less:
    return nvrhi::ComparisonFunc::Less;
  case CompareOp::LessEqual:
    return nvrhi::ComparisonFunc::LessOrEqual;
  case CompareOp::Greater:
    return nvrhi::ComparisonFunc::Greater;
  case CompareOp::GreaterEqual:
    return nvrhi::ComparisonFunc::GreaterOrEqual;
  case CompareOp::Equal:
    return nvrhi::ComparisonFunc::Equal;
  case CompareOp::NotEqual:
    return nvrhi::ComparisonFunc::NotEqual;
  case CompareOp::Always:
    return nvrhi::ComparisonFunc::Always;
  case CompareOp::Never:
  case CompareOp::Count:
  default:
    return nvrhi::ComparisonFunc::Never;
  }
}

[[nodiscard]] nvrhi::PrimitiveType toNvrhiPrimitiveType(Topology topology) {
  switch (topology) {
  case Topology::Point:
    return nvrhi::PrimitiveType::PointList;
  case Topology::Line:
    return nvrhi::PrimitiveType::LineList;
  case Topology::LineStrip:
    return nvrhi::PrimitiveType::LineStrip;
  case Topology::TriangleStrip:
    return nvrhi::PrimitiveType::TriangleStrip;
  case Topology::Patch:
    return nvrhi::PrimitiveType::PatchList;
  case Topology::Triangle:
  case Topology::Count:
  default:
    return nvrhi::PrimitiveType::TriangleList;
  }
}

[[nodiscard]] nvrhi::RasterCullMode toNvrhiCullMode(CullMode mode) {
  switch (mode) {
  case CullMode::None:
    return nvrhi::RasterCullMode::None;
  case CullMode::Front:
    return nvrhi::RasterCullMode::Front;
  case CullMode::Back:
  case CullMode::Count:
  default:
    return nvrhi::RasterCullMode::Back;
  }
}

[[nodiscard]] nvrhi::RasterFillMode toNvrhiFillMode(PolygonMode mode) {
  return mode == PolygonMode::Line ? nvrhi::RasterFillMode::Wireframe
                                   : nvrhi::RasterFillMode::Solid;
}

[[nodiscard]] nvrhi::ResolveMode toNvrhiResolveMode(ResolveMode mode) {
  switch (mode) {
  case ResolveMode::SampleZero:
    return nvrhi::ResolveMode::SampleZero;
  case ResolveMode::Min:
    return nvrhi::ResolveMode::Min;
  case ResolveMode::Max:
    return nvrhi::ResolveMode::Max;
  case ResolveMode::Average:
  case ResolveMode::None:
  case ResolveMode::Count:
  default:
    return nvrhi::ResolveMode::Average;
  }
}

[[nodiscard]] nvrhi::Format toNvrhiVertexFormat(VertexFormat format) {
  switch (format) {
  case VertexFormat::Float1:
    return nvrhi::Format::R32_FLOAT;
  case VertexFormat::Float2:
    return nvrhi::Format::RG32_FLOAT;
  case VertexFormat::Float3:
    return nvrhi::Format::RGB32_FLOAT;
  case VertexFormat::Float4:
    return nvrhi::Format::RGBA32_FLOAT;
  case VertexFormat::Int1:
    return nvrhi::Format::R32_SINT;
  case VertexFormat::Int2:
    return nvrhi::Format::RG32_SINT;
  case VertexFormat::Int3:
    return nvrhi::Format::RGB32_SINT;
  case VertexFormat::Int4:
    return nvrhi::Format::RGBA32_SINT;
  case VertexFormat::UInt1:
    return nvrhi::Format::R32_UINT;
  case VertexFormat::UInt2:
    return nvrhi::Format::RG32_UINT;
  case VertexFormat::UInt3:
    return nvrhi::Format::RGB32_UINT;
  case VertexFormat::UInt4:
    return nvrhi::Format::RGBA32_UINT;
  case VertexFormat::Byte4_Norm:
    return nvrhi::Format::RGBA8_SNORM;
  case VertexFormat::UByte4_Norm:
    return nvrhi::Format::RGBA8_UNORM;
  case VertexFormat::Short2:
    return nvrhi::Format::RG16_SINT;
  case VertexFormat::Short2_Norm:
    return nvrhi::Format::RG16_SNORM;
  case VertexFormat::Count:
    break;
  }
  return nvrhi::Format::UNKNOWN;
}

[[nodiscard]] nvrhi::ShaderType toNvrhiShaderType(ShaderStage stage) {
  switch (stage) {
  case ShaderStage::Vertex:
    return nvrhi::ShaderType::Vertex;
  case ShaderStage::TessControl:
    return nvrhi::ShaderType::Hull;
  case ShaderStage::TessEval:
    return nvrhi::ShaderType::Domain;
  case ShaderStage::Geometry:
    return nvrhi::ShaderType::Geometry;
  case ShaderStage::Fragment:
    return nvrhi::ShaderType::Pixel;
  case ShaderStage::Compute:
    return nvrhi::ShaderType::Compute;
  case ShaderStage::Task:
    return nvrhi::ShaderType::Amplification;
  case ShaderStage::Mesh:
    return nvrhi::ShaderType::Mesh;
  case ShaderStage::RayGen:
    return nvrhi::ShaderType::RayGeneration;
  case ShaderStage::AnyHit:
    return nvrhi::ShaderType::AnyHit;
  case ShaderStage::ClosestHit:
    return nvrhi::ShaderType::ClosestHit;
  case ShaderStage::Miss:
    return nvrhi::ShaderType::Miss;
  case ShaderStage::Intersection:
    return nvrhi::ShaderType::Intersection;
  case ShaderStage::Callable:
    return nvrhi::ShaderType::Callable;
  case ShaderStage::Count:
    break;
  }
  return nvrhi::ShaderType::None;
}

[[nodiscard]] glslang_stage_t toGlslangStage(ShaderStage stage) {
  switch (stage) {
  case ShaderStage::Vertex:
    return GLSLANG_STAGE_VERTEX;
  case ShaderStage::TessControl:
    return GLSLANG_STAGE_TESSCONTROL;
  case ShaderStage::TessEval:
    return GLSLANG_STAGE_TESSEVALUATION;
  case ShaderStage::Geometry:
    return GLSLANG_STAGE_GEOMETRY;
  case ShaderStage::Fragment:
    return GLSLANG_STAGE_FRAGMENT;
  case ShaderStage::Compute:
    return GLSLANG_STAGE_COMPUTE;
  case ShaderStage::Task:
    return GLSLANG_STAGE_TASK;
  case ShaderStage::Mesh:
    return GLSLANG_STAGE_MESH;
  case ShaderStage::RayGen:
    return GLSLANG_STAGE_RAYGEN;
  case ShaderStage::AnyHit:
    return GLSLANG_STAGE_ANYHIT;
  case ShaderStage::ClosestHit:
    return GLSLANG_STAGE_CLOSESTHIT;
  case ShaderStage::Miss:
    return GLSLANG_STAGE_MISS;
  case ShaderStage::Intersection:
    return GLSLANG_STAGE_INTERSECT;
  case ShaderStage::Callable:
    return GLSLANG_STAGE_CALLABLE;
  case ShaderStage::Count:
    break;
  }
  return GLSLANG_STAGE_VERTEX;
}

[[nodiscard]] glslang_resource_t
makeGlslangResource(const VkPhysicalDeviceLimits &limits,
                    const MeshletLimits &meshletLimits) {
  const int maxMeshOutputVertices =
      static_cast<int>(meshletLimits.maxMeshOutputVertices != 0u
                           ? meshletLimits.maxMeshOutputVertices
                           : kDefaultMeshletMaxVertices);
  const int maxMeshOutputPrimitives =
      static_cast<int>(meshletLimits.maxMeshOutputPrimitives != 0u
                           ? meshletLimits.maxMeshOutputPrimitives
                           : kDefaultMeshletMaxPrimitives);
  const int maxMeshWorkGroupSizeX =
      static_cast<int>(meshletLimits.maxMeshWorkGroupSizeX != 0u
                           ? meshletLimits.maxMeshWorkGroupSizeX
                           : kDefaultMeshletWorkGroupSize);
  const int maxTaskWorkGroupSizeX =
      static_cast<int>(meshletLimits.maxTaskWorkGroupSizeX != 0u
                           ? meshletLimits.maxTaskWorkGroupSizeX
                           : kDefaultMeshletWorkGroupSize);
  const glslang_resource_t resource{
      .max_lights = 32,
      .max_clip_planes = static_cast<int>(limits.maxClipDistances),
      .max_texture_units = 32,
      .max_texture_coords = 32,
      .max_vertex_attribs = static_cast<int>(limits.maxVertexInputAttributes),
      .max_vertex_uniform_components =
          static_cast<int>(limits.maxUniformBufferRange / 4u),
      .max_varying_floats = static_cast<int>(std::min(
          limits.maxVertexOutputComponents, limits.maxFragmentInputComponents)),
      .max_vertex_texture_image_units = 32,
      .max_combined_texture_image_units = 80,
      .max_texture_image_units = 32,
      .max_fragment_uniform_components = 4096,
      .max_draw_buffers = 32,
      .max_vertex_uniform_vectors = 128,
      .max_varying_vectors = 8,
      .max_fragment_uniform_vectors = 16,
      .max_vertex_output_vectors =
          static_cast<int>(limits.maxVertexOutputComponents / 4u),
      .max_fragment_input_vectors =
          static_cast<int>(limits.maxFragmentInputComponents / 4u),
      .min_program_texel_offset = limits.minTexelOffset,
      .max_program_texel_offset = static_cast<int>(limits.maxTexelOffset),
      .max_clip_distances = static_cast<int>(limits.maxClipDistances),
      .max_compute_work_group_count_x =
          static_cast<int>(limits.maxComputeWorkGroupCount[0]),
      .max_compute_work_group_count_y =
          static_cast<int>(limits.maxComputeWorkGroupCount[1]),
      .max_compute_work_group_count_z =
          static_cast<int>(limits.maxComputeWorkGroupCount[2]),
      .max_compute_work_group_size_x =
          static_cast<int>(limits.maxComputeWorkGroupSize[0]),
      .max_compute_work_group_size_y =
          static_cast<int>(limits.maxComputeWorkGroupSize[1]),
      .max_compute_work_group_size_z =
          static_cast<int>(limits.maxComputeWorkGroupSize[2]),
      .max_compute_uniform_components = 1024,
      .max_compute_texture_image_units = 16,
      .max_compute_image_uniforms = 8,
      .max_compute_atomic_counters = 8,
      .max_compute_atomic_counter_buffers = 1,
      .max_varying_components = 60,
      .max_vertex_output_components =
          static_cast<int>(limits.maxVertexOutputComponents),
      .max_geometry_input_components =
          static_cast<int>(limits.maxGeometryInputComponents),
      .max_geometry_output_components =
          static_cast<int>(limits.maxGeometryOutputComponents),
      .max_fragment_input_components =
          static_cast<int>(limits.maxFragmentInputComponents),
      .max_image_units = 8,
      .max_combined_image_units_and_fragment_outputs = 8,
      .max_combined_shader_output_resources = 8,
      .max_image_samples = 0,
      .max_vertex_image_uniforms = 0,
      .max_tess_control_image_uniforms = 0,
      .max_tess_evaluation_image_uniforms = 0,
      .max_geometry_image_uniforms = 0,
      .max_fragment_image_uniforms = 8,
      .max_combined_image_uniforms = 8,
      .max_geometry_texture_image_units = 16,
      .max_geometry_output_vertices =
          static_cast<int>(limits.maxGeometryOutputVertices),
      .max_geometry_total_output_components =
          static_cast<int>(limits.maxGeometryTotalOutputComponents),
      .max_geometry_uniform_components = 1024,
      .max_geometry_varying_components = 64,
      .max_tess_control_input_components = static_cast<int>(
          limits.maxTessellationControlPerVertexInputComponents),
      .max_tess_control_output_components = static_cast<int>(
          limits.maxTessellationControlPerVertexOutputComponents),
      .max_tess_control_texture_image_units = 16,
      .max_tess_control_uniform_components = 1024,
      .max_tess_control_total_output_components = 4096,
      .max_tess_evaluation_input_components =
          static_cast<int>(limits.maxTessellationEvaluationInputComponents),
      .max_tess_evaluation_output_components =
          static_cast<int>(limits.maxTessellationEvaluationOutputComponents),
      .max_tess_evaluation_texture_image_units = 16,
      .max_tess_evaluation_uniform_components = 1024,
      .max_tess_patch_components = 120,
      .max_patch_vertices = 32,
      .max_tess_gen_level = 64,
      .max_viewports = static_cast<int>(limits.maxViewports),
      .max_vertex_atomic_counters = 0,
      .max_tess_control_atomic_counters = 0,
      .max_tess_evaluation_atomic_counters = 0,
      .max_geometry_atomic_counters = 0,
      .max_fragment_atomic_counters = 8,
      .max_combined_atomic_counters = 8,
      .max_atomic_counter_bindings = 1,
      .max_vertex_atomic_counter_buffers = 0,
      .max_tess_control_atomic_counter_buffers = 0,
      .max_tess_evaluation_atomic_counter_buffers = 0,
      .max_geometry_atomic_counter_buffers = 0,
      .max_fragment_atomic_counter_buffers = 1,
      .max_combined_atomic_counter_buffers = 1,
      .max_atomic_counter_buffer_size = 16384,
      .max_transform_feedback_buffers = 4,
      .max_transform_feedback_interleaved_components = 64,
      .max_cull_distances = static_cast<int>(limits.maxCullDistances),
      .max_combined_clip_and_cull_distances =
          static_cast<int>(limits.maxCombinedClipAndCullDistances),
      .max_samples = 4,
      .max_mesh_output_vertices_nv = maxMeshOutputVertices,
      .max_mesh_output_primitives_nv = maxMeshOutputPrimitives,
      .max_mesh_work_group_size_x_nv = maxMeshWorkGroupSizeX,
      .max_mesh_work_group_size_y_nv = 1,
      .max_mesh_work_group_size_z_nv = 1,
      .max_task_work_group_size_x_nv = maxTaskWorkGroupSizeX,
      .max_task_work_group_size_y_nv = 1,
      .max_task_work_group_size_z_nv = 1,
      .max_mesh_view_count_nv = 4,
      .max_mesh_output_vertices_ext = maxMeshOutputVertices,
      .max_mesh_output_primitives_ext = maxMeshOutputPrimitives,
      .max_mesh_work_group_size_x_ext = maxMeshWorkGroupSizeX,
      .max_mesh_work_group_size_y_ext = 1,
      .max_mesh_work_group_size_z_ext = 1,
      .max_task_work_group_size_x_ext = maxTaskWorkGroupSizeX,
      .max_task_work_group_size_y_ext = 1,
      .max_task_work_group_size_z_ext = 1,
      .max_mesh_view_count_ext = 4,
      .maxDualSourceDrawBuffersEXT = 1,
      .limits =
          {
              .non_inductive_for_loops = true,
              .while_loops = true,
              .do_while_loops = true,
              .general_uniform_indexing = true,
              .general_attribute_matrix_vector_indexing = true,
              .general_varying_indexing = true,
              .general_sampler_indexing = true,
              .general_variable_indexing = true,
              .general_constant_matrix_vector_indexing = true,
          },
  };
  return resource;
}

void appendGlslangLog(std::string &message, std::string_view label,
                      const char *log) {
  if (log == nullptr || log[0] == '\0') {
    return;
  }
  message += "\n";
  message += label;
  message += ": ";
  message += log;
}

[[nodiscard]] Result<std::vector<uint8_t>, std::string>
compileGlslToSpirv(ShaderStage stage, const char *code,
                   const glslang_resource_t &resource) {
  const glslang_input_t input{
      .language = GLSLANG_SOURCE_GLSL,
      .stage = toGlslangStage(stage),
      .client = GLSLANG_CLIENT_VULKAN,
      .client_version = GLSLANG_TARGET_VULKAN_1_3,
      .target_language = GLSLANG_TARGET_SPV,
      .target_language_version = GLSLANG_TARGET_SPV_1_6,
      .code = code,
      .default_version = 100,
      .default_profile = GLSLANG_NO_PROFILE,
      .force_default_version_and_profile = false,
      .forward_compatible = false,
      .messages = GLSLANG_MSG_DEFAULT_BIT,
      .resource = &resource,
  };

  glslang_shader_t *shader = glslang_shader_create(&input);
  if (shader == nullptr) {
    return Result<std::vector<uint8_t>, std::string>::makeError(
        "glslang_shader_create() failed");
  }
  std::unique_ptr<glslang_shader_t, decltype(&glslang_shader_delete)>
      shaderGuard(shader, glslang_shader_delete);

  if (!glslang_shader_preprocess(shader, &input)) {
    std::string message = "glslang_shader_preprocess() failed";
    appendGlslangLog(message, "info", glslang_shader_get_info_log(shader));
    appendGlslangLog(message, "debug",
                     glslang_shader_get_info_debug_log(shader));
    return Result<std::vector<uint8_t>, std::string>::makeError(
        std::move(message));
  }

  if (!glslang_shader_parse(shader, &input)) {
    std::string message = "glslang_shader_parse() failed";
    appendGlslangLog(message, "info", glslang_shader_get_info_log(shader));
    appendGlslangLog(message, "debug",
                     glslang_shader_get_info_debug_log(shader));
    return Result<std::vector<uint8_t>, std::string>::makeError(
        std::move(message));
  }

  glslang_program_t *program = glslang_program_create();
  if (program == nullptr) {
    return Result<std::vector<uint8_t>, std::string>::makeError(
        "glslang_program_create() failed");
  }
  std::unique_ptr<glslang_program_t, decltype(&glslang_program_delete)>
      programGuard(program, glslang_program_delete);

  glslang_program_add_shader(program, shader);
  if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT |
                                         GLSLANG_MSG_VULKAN_RULES_BIT)) {
    std::string message = "glslang_program_link() failed";
    appendGlslangLog(message, "info", glslang_program_get_info_log(program));
    appendGlslangLog(message, "debug",
                     glslang_program_get_info_debug_log(program));
    return Result<std::vector<uint8_t>, std::string>::makeError(
        std::move(message));
  }

  glslang_spv_options_t options{
      .generate_debug_info = true,
      .strip_debug_info = false,
      .disable_optimizer = false,
      .optimize_size = true,
      .disassemble = false,
      .validate = true,
      .emit_nonsemantic_shader_debug_info = false,
      .emit_nonsemantic_shader_debug_source = false,
  };
  glslang_program_SPIRV_generate_with_options(program, input.stage, &options);
  if (const char *messages = glslang_program_SPIRV_get_messages(program);
      messages != nullptr && messages[0] != '\0') {
    NURI_LOG_WARNING("glslang SPIR-V generation: %s", messages);
  }

  const auto *spirv =
      reinterpret_cast<const uint8_t *>(glslang_program_SPIRV_get_ptr(program));
  const size_t numBytes =
      glslang_program_SPIRV_get_size(program) * sizeof(uint32_t);
  if (spirv == nullptr || numBytes == 0u) {
    return Result<std::vector<uint8_t>, std::string>::makeError(
        "glslang produced an empty SPIR-V module");
  }

  std::vector<uint8_t> bytes(spirv, spirv + numBytes);
  return Result<std::vector<uint8_t>, std::string>::makeResult(
      std::move(bytes));
}

[[nodiscard]] nvrhi::Format toNvrhiIndexFormat(IndexFormat format) {
  return format == IndexFormat::U16 ? nvrhi::Format::R16_UINT
                                    : nvrhi::Format::R32_UINT;
}

[[nodiscard]] bool isDepthFormat(Format format) {
  return format == Format::D16_UNORM || format == Format::D32_FLOAT;
}

[[nodiscard]] size_t bytesPerBlock(Format format) {
  switch (format) {
  case Format::R8_UNORM:
    return 1u;
  case Format::R16_UNORM:
    return 2u;
  case Format::R32_UINT:
  case Format::R32_FLOAT:
    return 4u;
  case Format::RG16_FLOAT:
  case Format::RG32_FLOAT:
    return format == Format::RG16_FLOAT ? 4u : 8u;
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
  case Format::RGBA8_UINT:
    return 4u;
  case Format::RGBA16_FLOAT:
    return 8u;
  case Format::RGBA32_FLOAT:
    return 16u;
  case Format::D16_UNORM:
    return 2u;
  case Format::D32_FLOAT:
    return 4u;
  case Format::BC7_RGBA_UNORM:
  case Format::BC7_RGBA_SRGB:
    return 16u;
  case Format::ETC2_RGB8_UNORM:
  case Format::ETC2_RGB8_SRGB:
    return 8u;
  case Format::Count:
    break;
  }
  return 0u;
}

[[nodiscard]] uint32_t blockExtent(Format format) {
  switch (format) {
  case Format::BC7_RGBA_UNORM:
  case Format::BC7_RGBA_SRGB:
  case Format::ETC2_RGB8_UNORM:
  case Format::ETC2_RGB8_SRGB:
    return 4u;
  default:
    return 1u;
  }
}

[[nodiscard]] size_t textureReadbackBytesPerPixel(Format format) {
  switch (format) {
  case Format::R8_UNORM:
    return 1u;
  case Format::R16_UNORM:
    return sizeof(uint16_t);
  case Format::R32_UINT:
  case Format::R32_FLOAT:
    return 4u;
  case Format::RG16_FLOAT:
    return sizeof(uint16_t) * 2u;
  case Format::RG32_FLOAT:
    return sizeof(float) * 2u;
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
  case Format::RGBA8_UINT:
    return 4u;
  case Format::RGBA16_FLOAT:
    return sizeof(uint16_t) * 4u;
  case Format::RGBA32_FLOAT:
    return sizeof(float) * 4u;
  case Format::D16_UNORM:
    return sizeof(uint16_t);
  case Format::D32_FLOAT:
    return sizeof(float);
  case Format::BC7_RGBA_UNORM:
  case Format::BC7_RGBA_SRGB:
  case Format::ETC2_RGB8_UNORM:
  case Format::ETC2_RGB8_SRGB:
  case Format::Count:
    break;
  }
  return 0u;
}

[[nodiscard]] uint32_t mipSize(uint32_t base, uint32_t mip) {
  return std::max(1u, base >> std::min(mip, 31u));
}

[[nodiscard]] size_t textureMipByteSize(Format format, uint32_t width,
                                        uint32_t height, uint32_t depth) {
  const size_t blockBytes = bytesPerBlock(format);
  const uint32_t block = blockExtent(format);
  if (blockBytes == 0u) {
    return 0u;
  }
  const uint64_t blocksX = (static_cast<uint64_t>(width) + block - 1u) / block;
  const uint64_t blocksY = (static_cast<uint64_t>(height) + block - 1u) / block;
  return static_cast<size_t>(blocksX * blocksY * std::max(1u, depth) *
                             blockBytes);
}

[[nodiscard]] nvrhi::ResourceStates
toNvrhiTextureState(GraphicsBarrierState state,
                    GraphicsBarrierAccessMode access, bool isDepthTexture) {
  switch (state) {
  case GraphicsBarrierState::Read:
    return isDepthTexture ? (nvrhi::ResourceStates::ShaderResource |
                             nvrhi::ResourceStates::DepthRead)
                          : nvrhi::ResourceStates::ShaderResource;
  case GraphicsBarrierState::Write:
    return hasGraphicsBarrierAccessFlag(access,
                                        GraphicsBarrierAccessMode::Write)
               ? nvrhi::ResourceStates::UnorderedAccess
               : (isDepthTexture ? (nvrhi::ResourceStates::ShaderResource |
                                    nvrhi::ResourceStates::DepthRead)
                                 : nvrhi::ResourceStates::ShaderResource);
  case GraphicsBarrierState::Attachment:
    return isDepthTexture ? nvrhi::ResourceStates::DepthWrite
                          : nvrhi::ResourceStates::RenderTarget;
  case GraphicsBarrierState::Present:
    return nvrhi::ResourceStates::Present;
  case GraphicsBarrierState::Unknown:
  default:
    return nvrhi::ResourceStates::Common;
  }
}

[[nodiscard]] nvrhi::ResourceStates
toNvrhiBufferState(GraphicsBarrierState state,
                   GraphicsBarrierAccessMode access) {
  switch (state) {
  case GraphicsBarrierState::Read:
    return nvrhi::ResourceStates::ShaderResource |
           nvrhi::ResourceStates::IndirectArgument |
           nvrhi::ResourceStates::IndexBuffer |
           nvrhi::ResourceStates::VertexBuffer;
  case GraphicsBarrierState::Write:
    return hasGraphicsBarrierAccessFlag(access,
                                        GraphicsBarrierAccessMode::Write)
               ? nvrhi::ResourceStates::UnorderedAccess
               : nvrhi::ResourceStates::ShaderResource;
  case GraphicsBarrierState::Attachment:
  case GraphicsBarrierState::Present:
  case GraphicsBarrierState::Unknown:
  default:
    return nvrhi::ResourceStates::Common;
  }
}

[[nodiscard]] nvrhi::ResourceStates
permanentReadBufferState(BufferUsage usage) {
  nvrhi::ResourceStates state = nvrhi::ResourceStates::ShaderResource |
                                nvrhi::ResourceStates::VertexBuffer |
                                nvrhi::ResourceStates::IndexBuffer |
                                nvrhi::ResourceStates::IndirectArgument;
  if (hasBufferUsage(usage, BufferUsage::Uniform)) {
    state = state | nvrhi::ResourceStates::ConstantBuffer;
  }
  return state;
}

[[nodiscard]] bool usesPermanentReadState(const BufferDesc &desc) {
  return desc.immutable && desc.storage == Storage::Device;
}

[[nodiscard]] nvrhi::Color toNvrhiColor(const ClearColor &color) {
  return nvrhi::Color(color.r, color.g, color.b, color.a);
}

[[nodiscard]] nvrhi::Viewport toNvrhiViewport(const Viewport &viewport) {
  return nvrhi::Viewport(viewport.x, viewport.x + viewport.width, viewport.y,
                         viewport.y + viewport.height, viewport.minDepth,
                         viewport.maxDepth);
}

[[nodiscard]] nvrhi::Rect toNvrhiRect(const RectU32 &rect) {
  return nvrhi::Rect(
      static_cast<int>(rect.x), static_cast<int>(rect.x + rect.width),
      static_cast<int>(rect.y), static_cast<int>(rect.y + rect.height));
}

[[nodiscard]] nvrhi::Rect viewportRect(const Viewport &viewport) {
  return nvrhi::Rect(static_cast<int>(viewport.x),
                     static_cast<int>(viewport.x + viewport.width),
                     static_cast<int>(viewport.y),
                     static_cast<int>(viewport.y + viewport.height));
}

[[nodiscard]] std::string patchGlslPrelude(ShaderStage stage,
                                           std::string_view source) {
  if (source.find("#version ") != std::string_view::npos) {
    return std::string(source);
  }

  auto contains = [source](std::string_view token) {
    return source.find(token) != std::string_view::npos;
  };
  std::string patched;
  patched.reserve(source.size() + 4096u);

  switch (stage) {
  case ShaderStage::Task:
  case ShaderStage::Mesh:
    patched +=
        "#version 460\n"
        "#extension GL_EXT_buffer_reference : require\n"
        "#extension GL_EXT_buffer_reference_uvec2 : require\n"
        "#extension GL_EXT_debug_printf : enable\n"
        "#extension GL_EXT_nonuniform_qualifier : require\n"
        "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n"
        "#extension GL_EXT_mesh_shader : require\n";
    break;
  case ShaderStage::Vertex:
  case ShaderStage::Compute:
  case ShaderStage::TessControl:
  case ShaderStage::TessEval:
    patched += "#version 460\n"
               "#extension GL_EXT_buffer_reference : require\n"
               "#extension GL_EXT_buffer_reference_uvec2 : require\n"
               "#extension GL_EXT_debug_printf : enable\n"
               "#extension GL_EXT_nonuniform_qualifier : require\n"
               "#extension GL_EXT_samplerless_texture_functions : require\n"
               "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : "
               "require\n";
    if (stage == ShaderStage::Compute) {
      patched +=
          "layout (set = 0, binding = 0) uniform texture2D   kTextures2D[];\n"
          "layout (set = 0, binding = 1) uniform sampler     kSamplers[];\n"
          "vec4 textureBindless2D(uint textureid, uint samplerid, vec2 uv) {\n"
          "  return texture(nonuniformEXT(sampler2D(kTextures2D[textureid], "
          "kSamplers[samplerid])), uv);\n"
          "}\n"
          "vec4 textureBindless2DLod(uint textureid, uint samplerid, vec2 uv, "
          "float lod) {\n"
          "  return textureLod(nonuniformEXT(sampler2D(kTextures2D[textureid], "
          "kSamplers[samplerid])), uv, lod);\n"
          "}\n"
          "ivec2 textureBindlessSize2D(uint textureid) {\n"
          "  return textureSize(nonuniformEXT(kTextures2D[textureid]), 0);\n"
          "}\n";
    }
    break;
  case ShaderStage::Fragment:
    patched += "#version 460\n"
               "#extension GL_EXT_buffer_reference_uvec2 : require\n"
               "#extension GL_EXT_debug_printf : enable\n"
               "#extension GL_EXT_nonuniform_qualifier : require\n"
               "#extension GL_EXT_samplerless_texture_functions : require\n"
               "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : "
               "require\n";
    if (contains("kTLAS[")) {
      patched +=
          "#extension GL_EXT_buffer_reference : require\n"
          "#extension GL_EXT_ray_query : require\n"
          "layout(set = 0, binding = 4) uniform accelerationStructureEXT "
          "kTLAS[];\n";
    }
    patched +=
        "layout (set = 0, binding = 0) uniform texture2D   kTextures2D[];\n"
        "layout (set = 1, binding = 0) uniform texture3D   kTextures3D[];\n"
        "layout (set = 2, binding = 0) uniform textureCube kTexturesCube[];\n"
        "layout (set = 3, binding = 0) uniform texture2D   "
        "kTextures2DShadow[];\n"
        "layout (set = 0, binding = 1) uniform sampler       kSamplers[];\n"
        "layout (set = 3, binding = 1) uniform samplerShadow "
        "kSamplersShadow[];\n";
    if (contains("textureBindless2D(")) {
      patched +=
          "vec4 textureBindless2D(uint textureid, uint samplerid, vec2 uv) {\n"
          "  return texture(nonuniformEXT(sampler2D(kTextures2D[textureid], "
          "kSamplers[samplerid])), uv);\n"
          "}\n";
    }
    if (contains("textureBindless2DLod(")) {
      patched +=
          "vec4 textureBindless2DLod(uint textureid, uint samplerid, vec2 uv, "
          "float lod) {\n"
          "  return textureLod(nonuniformEXT(sampler2D(kTextures2D[textureid], "
          "kSamplers[samplerid])), uv, lod);\n"
          "}\n";
    }
    if (contains("textureBindless2DShadow(")) {
      patched +=
          "float textureBindless2DShadow(uint textureid, uint samplerid, vec3 "
          "uvw) {"
          "  return texture(nonuniformEXT(sampler2DShadow("
          "kTextures2DShadow[textureid], kSamplersShadow[samplerid])), uvw);\n"
          "}\n";
    }
    if (contains("textureBindlessSize2D(")) {
      patched +=
          "ivec2 textureBindlessSize2D(uint textureid) {\n"
          "  return textureSize(nonuniformEXT(kTextures2D[textureid]), 0);\n"
          "}\n";
    }
    if (contains("textureBindlessCube(")) {
      patched +=
          "vec4 textureBindlessCube(uint textureid, uint samplerid, vec3 uvw) "
          "{\n"
          "  return "
          "texture(nonuniformEXT(samplerCube(kTexturesCube[textureid], "
          "kSamplers[samplerid])), uvw);\n"
          "}\n";
    }
    if (contains("textureBindlessCubeLod(")) {
      patched +=
          "vec4 textureBindlessCubeLod(uint textureid, uint samplerid, vec3 "
          "uvw, float lod) {\n"
          "  return textureLod(nonuniformEXT(samplerCube(kTexturesCube["
          "textureid], kSamplers[samplerid])), uvw, lod);\n"
          "}\n";
    }
    if (contains("textureBindlessQueryLevels2D(")) {
      patched +=
          "int textureBindlessQueryLevels2D(uint textureid) {\n"
          "  return textureQueryLevels(nonuniformEXT(kTextures2D[textureid]));"
          "\n"
          "}\n";
    }
    if (contains("textureBindlessQueryLevelsCube(")) {
      patched += "int textureBindlessQueryLevelsCube(uint textureid) {\n"
                 "  return "
                 "textureQueryLevels(nonuniformEXT(kTexturesCube[textureid]));"
                 "\n"
                 "}\n";
    }
    break;
  default:
    patched += "#version 460\n";
    break;
  }

  patched.append(source.data(), source.size());
  return patched;
}

class NvrhiLogCallback final : public nvrhi::IMessageCallback {
public:
  void message(nvrhi::MessageSeverity severity,
               const char *messageText) override {
    const char *text = messageText != nullptr ? messageText : "";
    switch (severity) {
    case nvrhi::MessageSeverity::Info:
      NURI_LOG_DEBUG("NVRHI: %s", text);
      break;
    case nvrhi::MessageSeverity::Warning:
      NURI_LOG_WARNING("NVRHI: %s", text);
      break;
    case nvrhi::MessageSeverity::Error:
      NURI_LOG_ERROR("NVRHI: %s", text);
      break;
    case nvrhi::MessageSeverity::Fatal:
      NURI_LOG_FATAL("NVRHI: %s", text);
      break;
    }
  }
};

template <typename Resource> struct ResourceSlot {
  Resource resource{};
  std::string debugName;
  Format format = Format::RGBA8_UNORM;
};

struct BufferResource {
  nvrhi::BufferHandle buffer{};
  std::string debugName;
  size_t byteSize = 0u;
  std::byte *mapped = nullptr;
  VkDeviceMemory mappedMemory = VK_NULL_HANDLE;
  bool hostVisible = false;
  bool immutable = false;
};

class NvrhiPreparedGpuBuffer final : public PreparedGpuBuffer {
public:
  NvrhiPreparedGpuBuffer(nvrhi::DeviceHandle device, BufferResource resource)
      : device_(std::move(device)), resource_(std::move(resource)) {}

  ~NvrhiPreparedGpuBuffer() override {
    if (resource_.mapped != nullptr && resource_.buffer && device_) {
      device_->unmapBuffer(resource_.buffer.Get());
    }
  }

  [[nodiscard]] BufferResource take() {
    BufferResource result = std::move(resource_);
    resource_ = {};
    return result;
  }

private:
  nvrhi::DeviceHandle device_{};
  BufferResource resource_{};
};

struct TextureResource {
  nvrhi::TextureHandle texture{};
  std::string debugName;
  Format format = Format::RGBA8_UNORM;
  TextureDesc desc{};
  nvrhi::TextureDimension dimension = nvrhi::TextureDimension::Unknown;
};

class NvrhiPreparedGpuTexture final : public PreparedGpuTexture {
public:
  explicit NvrhiPreparedGpuTexture(TextureResource resource)
      : resource_(std::move(resource)) {}

  [[nodiscard]] TextureResource take() { return std::move(resource_); }

private:
  TextureResource resource_{};
};

struct ShaderResource {
  nvrhi::ShaderHandle shader{};
  std::string debugName;
  ShaderStage stage = ShaderStage::Vertex;
};

using PipelineVariantKey = RasterPipelineState;

struct PipelineVariantKeyHasher {
  [[nodiscard]] size_t
  operator()(const PipelineVariantKey &key) const noexcept {
    const PipelineVariantKey canonical = canonicalRasterPipelineState(key);
    size_t hash = static_cast<size_t>(canonical.compareOp);
    auto mix = [&hash](uint64_t value) {
      hash ^= static_cast<size_t>(value) + 0x9e3779b97f4a7c15ull +
              (hash << 6u) + (hash >> 2u);
    };
    mix(canonical.depthWrite ? 1u : 0u);
    mix(canonical.depthBiasEnable ? 1u : 0u);
    mix(static_cast<uint32_t>(canonical.depthBiasConstant));
    mix(std::bit_cast<uint32_t>(canonical.depthBiasSlope));
    mix(std::bit_cast<uint32_t>(canonical.depthBiasClamp));
    return hash;
  }
};

struct RenderPipelineResource {
  nvrhi::GraphicsPipelineDesc baseDesc{};
  nvrhi::FramebufferInfo framebufferInfo{};
  nvrhi::InputLayoutHandle inputLayout{};
  std::vector<nvrhi::ShaderHandle> specializedShaders;
  std::unordered_map<PipelineVariantKey, nvrhi::GraphicsPipelineHandle,
                     PipelineVariantKeyHasher>
      variants;
  std::string debugName;
};

struct ComputePipelineResource {
  nvrhi::ComputePipelineHandle pipeline{};
  nvrhi::ShaderHandle specializedShader{};
  std::string debugName;
};

struct MeshletPipelineResource {
  nvrhi::MeshletPipelineDesc baseDesc{};
  nvrhi::FramebufferInfo framebufferInfo{};
  std::vector<nvrhi::ShaderHandle> specializedShaders;
  std::unordered_map<PipelineVariantKey, nvrhi::MeshletPipelineHandle,
                     PipelineVariantKeyHasher>
      variants;
  std::string debugName;
};

template <typename NuriHandle, typename Resource> class ResourceTable {
public:
  struct DebugSlotInfo {
    uint32_t generation = 0u;
    bool inRange = false;
    bool live = false;
    bool retired = false;
    std::string_view debugName{};
  };

  struct ReservedSlot {
    NuriHandle handle{};
    Resource *resource = nullptr;
  };

  struct RetiredSlot {
    uint32_t index = 0u;
    Resource resource{};
  };

  ReservedSlot reserve(std::string debugName,
                       Format format = Format::RGBA8_UNORM) {
    const SlotReservation reservation = slotsMeta_.acquire();
    if (reservation.appended) {
      slots_.emplace_back();
    }
    Resource &slot = slots_[reservation.index];
    slot = Resource{};
    slot.debugName = std::move(debugName);
    if constexpr (requires { slot.format; }) {
      slot.format = format;
    }
    return ReservedSlot{
        .handle = NuriHandle{reservation.index, reservation.generation},
        .resource = &slot,
    };
  }

  NuriHandle allocate(Resource resource) {
    const SlotReservation reservation = slotsMeta_.acquire();
    if (reservation.appended) {
      slots_.emplace_back();
    }
    slots_[reservation.index] = std::move(resource);
    return NuriHandle{reservation.index, reservation.generation};
  }

  void deallocate(NuriHandle handle) {
    if (!isValid(handle)) {
      return;
    }
    slots_[handle.index] = Resource{};
    slotsMeta_.release(handle.index);
  }

  [[nodiscard]] std::optional<RetiredSlot> take(NuriHandle handle) {
    if (!isValid(handle)) {
      return std::nullopt;
    }
    Resource resource = std::move(slots_[handle.index]);
    slots_[handle.index] = Resource{};
    slotsMeta_.retire(handle.index);
    return RetiredSlot{
        .index = handle.index,
        .resource = std::move(resource),
    };
  }

  void recycleRetired(uint32_t index) {
    NURI_ASSERT(index < slots_.size(),
                "ResourceTable::recycleRetired: index out of range");
    NURI_ASSERT(slotsMeta_.isRetired(index),
                "ResourceTable::recycleRetired: slot is not retired");
    slotsMeta_.recycle(index);
  }

  [[nodiscard]] bool replace(NuriHandle handle, Resource resource) {
    if (!isValid(handle)) {
      return false;
    }
    slots_[handle.index] = std::move(resource);
    return true;
  }

  void clear() {
    slots_.clear();
    slotsMeta_.clear();
  }

  [[nodiscard]] bool isValid(NuriHandle handle) const {
    return handle.index < slots_.size() &&
           slotsMeta_.isValid(handle.index, handle.generation);
  }

  [[nodiscard]] Resource *get(NuriHandle handle) {
    return isValid(handle) ? &slots_[handle.index] : nullptr;
  }

  [[nodiscard]] const Resource *get(NuriHandle handle) const {
    return isValid(handle) ? &slots_[handle.index] : nullptr;
  }

  [[nodiscard]] Format getFormat(NuriHandle handle) const {
    if (const Resource *slot = get(handle); slot != nullptr) {
      if constexpr (requires { slot->format; }) {
        return slot->format;
      }
    }
    return Format::RGBA8_UNORM;
  }

  [[nodiscard]] DebugSlotInfo debugSlotInfo(uint32_t index) const {
    if (index >= slots_.size()) {
      return {};
    }
    return DebugSlotInfo{
        .generation = slotsMeta_.generation(index),
        .inRange = true,
        .live = slotsMeta_.isLive(index),
        .retired = slotsMeta_.isRetired(index),
        .debugName = slots_[index].debugName,
    };
  }

private:
  std::deque<Resource> slots_;
  SlotPool<UnmaskedNonZeroGenerationPolicy> slotsMeta_;
};

struct FramebufferTexture {
  TextureHandle handle{};
  TextureDesc desc{};
  std::string debugName;
};

struct SwapchainImage {
  VkImage image = VK_NULL_HANDLE;
  nvrhi::TextureHandle texture{};
};

struct Swapchain {
  VkSwapchainKHR handle = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
  VkColorSpaceKHR colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  VkExtent2D extent{};
  std::vector<SwapchainImage> images;
};

struct FramebufferCacheKey {
  TextureHandle colorTexture{};
  TextureHandle colorResolveTexture{};
  TextureHandle depthTexture{};
  TextureHandle depthResolveTexture{};
  uint64_t swapchainGeneration = 0u;
  uint32_t swapchainImageIndex = 0u;
  ResolveMode colorResolveMode = ResolveMode::None;
  ResolveMode depthResolveMode = ResolveMode::None;
  bool hasColorAttachment = false;
  bool colorIsSwapchain = false;
  bool colorResolve = false;
  bool depthResolve = false;

  [[nodiscard]] bool references(TextureHandle texture) const noexcept {
    return colorTexture == texture || colorResolveTexture == texture ||
           depthTexture == texture || depthResolveTexture == texture;
  }

  constexpr bool
  operator==(const FramebufferCacheKey &) const noexcept = default;
};

struct CachedFramebuffer {
  FramebufferCacheKey key{};
  nvrhi::FramebufferHandle framebuffer{};
};

struct NvrhiTimingQuery {
  GpuTimingScope scope = GpuTimingScope::None;
  std::string debugLabel;
  nvrhi::TimerQueryHandle query{};
};

struct NvrhiWholeFrameTimingSlot {
  VkQueryPool queryPool = VK_NULL_HANDLE;
  std::array<nvrhi::CommandListHandle, 2> commandLists{};
};

struct ActiveGraphicsRecordingContext {
  RecordingContextHandle handle{};
  nvrhi::CommandListHandle commandList{};
  std::vector<nvrhi::FramebufferHandle> framebuffers;
  std::vector<NvrhiTimingQuery> timingQueries;
  uint64_t recordingSerial = 0u;
  uint32_t workerIndex = 0u;
};

struct RecordedGraphicsCommandBuffer {
  RecordedCommandBufferHandle handle{};
  nvrhi::CommandListHandle commandList{};
  std::vector<nvrhi::FramebufferHandle> framebuffers;
  std::vector<NvrhiTimingQuery> timingQueries;
  uint64_t recordingSerial = 0u;
};

struct PendingGraphicsCommandList {
  uint64_t submissionInstance = 0u;
  nvrhi::CommandListHandle commandList{};
  std::vector<nvrhi::FramebufferHandle> framebuffers;
};

struct PendingGpuTimingSubmission {
  uint64_t frameIndex = 0u;
  uint64_t submissionInstance = 0u;
  std::vector<NvrhiTimingQuery> timingQueries;
  NvrhiWholeFrameTimingSlot wholeFrameTiming{};
};

struct PendingAsyncUploadSubmission {
  uint64_t submissionInstance = 0u;
  nvrhi::CommandListHandle commandList{};
  bool containsTextureData = false;
  nvrhi::CommandQueue queue = nvrhi::CommandQueue::Graphics;
};

struct SubmissionRecord {
  SubmissionHandle handle{};
  uint64_t graphicsInstance = 0u;
  uint64_t copyInstance = 0u;
  uint64_t requiredRecordingSerial = 0u;
  bool requiresGraphicsVisibility = false;
  bool graphicsVisibilityQueued = false;
};

struct QueueFamilySelection {
  uint32_t graphics = std::numeric_limits<uint32_t>::max();
  uint32_t graphicsQueueCount = 0u;
  bool hasGraphics = false;
  bool hasDedicatedCopyQueue = false;
};

[[nodiscard]] GpuMultisampleCapabilities
queryMultisampleCapabilities(VkPhysicalDevice device,
                             bool sampleRateShadingEnabled) {
  if (device == VK_NULL_HANDLE) {
    return {};
  }
  const auto supports4x = [device](VkFormat format, VkImageUsageFlags usage) {
    VkImageFormatProperties properties{};
    return vkGetPhysicalDeviceImageFormatProperties(
               device, format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, usage,
               0u, &properties) == VK_SUCCESS &&
           (properties.sampleCounts & VK_SAMPLE_COUNT_4_BIT) != 0u;
  };
  VkPhysicalDeviceDepthStencilResolveProperties resolveProperties{
      .sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES,
  };
  VkPhysicalDeviceProperties2 properties{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &resolveProperties,
  };
  vkGetPhysicalDeviceProperties2(device, &properties);

  const bool sample4Color = supports4x(VK_FORMAT_R16G16B16A16_SFLOAT,
                                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
  return GpuMultisampleCapabilities{
      .sample4Color = sample4Color,
      .sample4Depth = supports4x(VK_FORMAT_D32_SFLOAT,
                                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT),
      .depthResolveMin = (resolveProperties.supportedDepthResolveModes &
                          VK_RESOLVE_MODE_MIN_BIT) != 0u,
      .alphaToCoverage = sample4Color,
      .sampleRateShading = sampleRateShadingEnabled,
  };
}

} // namespace

struct NvrhiGPUDeviceImpl {
  Window *window = nullptr;
  GLFWwindow *glfwWindow = nullptr;
  NvrhiLogCallback nvrhiLog{};
  bool renderDocAttached = false;
  bool validationEnabled = false;
  bool glslangInitialized = false;
  uint64_t currentFrameIndex = 0u;
  uint64_t submittedFrameCount = 0u;
  uint32_t preparedSwapchainImageIndex = 0u;
  bool hasPreparedSwapchainImage = false;

  VkInstance instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue graphicsQueue = VK_NULL_HANDLE;
  VkQueue assetCopyQueue = VK_NULL_HANDLE;
  uint32_t graphicsQueueFamily = 0u;
  bool hasDedicatedAssetCopyQueue = false;
  bool graphicsQueueTimestampsSupported = false;
  VkPhysicalDeviceProperties physicalDeviceProperties{};
  GPUAdapterInfo adapterInfo{};
  VkPhysicalDeviceMemoryProperties memoryProperties{};
  std::vector<const char *> instanceExtensions;
  std::vector<const char *> deviceExtensions;
  Swapchain swapchain{};
  std::vector<VkSemaphore> imageAvailableSemaphores;
  std::vector<VkSemaphore> renderFinishedSemaphores;
  std::vector<uint64_t> frameSemaphoreReuseWaitInstances;
  std::vector<uint64_t> frameResourceReuseWaitInstances;
  std::vector<uint64_t> swapchainImageReuseWaitInstances;
  uint32_t semaphoreFrameIndex = 0u;
  uint64_t preparedSwapchainImageWaitInstance = 0u;
  SwapchainPresentMode requestedPresentMode = SwapchainPresentMode::Mailbox;
  SwapchainPresentMode activePresentMode = SwapchainPresentMode::Unknown;

  nvrhi::vulkan::DeviceHandle nvrhiDevice{};
  nvrhi::CommandListHandle immediateCommandList{};
  nvrhi::BindingLayoutHandle bindless2DLayout{};
  nvrhi::BindingLayoutHandle bindless3DLayout{};
  nvrhi::BindingLayoutHandle bindlessCubeLayout{};
  nvrhi::BindingLayoutHandle bindlessShadowLayout{};
  nvrhi::BindingLayoutHandle pushConstantsLayout{};
  nvrhi::DescriptorTableHandle bindless2DTable{};
  nvrhi::DescriptorTableHandle bindless3DTable{};
  nvrhi::DescriptorTableHandle bindlessCubeTable{};
  nvrhi::DescriptorTableHandle bindlessShadowTable{};
  nvrhi::BindingSetHandle pushConstantsSet{};

  TextureCompressionCaps compressionCaps{};
  GpuMultisampleCapabilities multisampleCapabilities{};
  uint8_t maxSamplerAnisotropy = 1u;
  float maxSamplerLodBias = 0.0f;
  bool supportsDrawIndirectCount = false;
  bool meshletExtensionEnabled = false;
  bool meshletFeaturesEnabled = false;
  bool meshletsSupported = false;
  MeshletLimits meshletLimits{};

  ResourceTable<SamplerHandle, ResourceSlot<nvrhi::SamplerHandle>> samplers;
  SamplerHandle cubemapSampler{};
  SamplerHandle bilinearSampler{};
  SamplerHandle trilinearSampler{};
  std::array<SamplerHandle, 4> anisotropicSamplers{};

  ResourceTable<BufferHandle, BufferResource> buffers;
  ResourceTable<TextureHandle, TextureResource> textures;
  ResourceTable<ShaderHandle, ShaderResource> shaders;
  ResourceTable<RenderPipelineHandle, RenderPipelineResource> renderPipelines;
  ResourceTable<ComputePipelineHandle, ComputePipelineResource>
      computePipelines;
  ResourceTable<MeshletPipelineHandle, MeshletPipelineResource>
      meshletPipelines;
  std::vector<FramebufferTexture> framebufferTextures;
  std::vector<CachedFramebuffer> cachedFramebuffers;
  uint64_t swapchainGeneration = 1u;
  using RetiredNativeResource =
      std::variant<BufferResource, TextureResource,
                   ResourceSlot<nvrhi::SamplerHandle>, ShaderResource,
                   RenderPipelineResource, ComputePipelineResource,
                   MeshletPipelineResource>;
  enum class RetiredResourceTable : uint8_t {
    Buffer,
    Texture,
    Sampler,
    Shader,
    RenderPipeline,
    ComputePipeline,
    MeshletPipeline,
  };
  struct RetiredResource {
    RetiredNativeResource resource{};
    RetiredResourceTable table = RetiredResourceTable::Buffer;
    uint32_t slotIndex = 0u;
    uint64_t requiredRecordingSerial = 0u;
    uint64_t lastSubmittedAtDestruction = 0u;
    uint64_t lastUseInstance = 0u;
    bool lastUseResolved = false;
  };
  std::vector<RetiredResource> retiredResources;
  RecordingRetirementTracker recordingRetirement;
  uint64_t latestSubmittedInstance = 0u;
  uint64_t latestAsyncUploadGraphicsInstance = 0u;
  uint64_t latestAsyncUploadCopyInstance = 0u;

  mutable std::mutex immediateMutex;
  // Serializes driver-side native resource allocation performed by asset
  // workers. Prepared resources do not enter Nuri handle tables until the
  // render thread publishes them.
  std::mutex resourcePreparationMutex;
  std::mutex graphicsContextMutex;
  std::vector<ActiveGraphicsRecordingContext> activeGraphicsContexts;
  std::vector<RecordedGraphicsCommandBuffer> recordedGraphicsCommandBuffers;
  std::vector<PendingGraphicsCommandList> pendingGraphicsCommandLists;
  std::vector<nvrhi::CommandListHandle> availableGraphicsCommandLists;
  std::vector<PendingGpuTimingSubmission> pendingGpuTimingSubmissions;
  std::vector<NvrhiWholeFrameTimingSlot> availableWholeFrameTimingSlots;
  nvrhi::CommandListHandle pendingAsyncUploadCommandList{};
  uint64_t pendingAsyncUploadBytes = 0u;
  uint32_t pendingAsyncUploadTextureCount = 0u;
  uint64_t textureUploadTexturesRecorded = 0u;
  uint64_t textureUploadBytesRecorded = 0u;
  uint64_t textureUploadBatchesSubmitted = 0u;
  uint64_t textureUploadBoundedBatchFlushes = 0u;
  uint64_t textureUploadCompletionWaits = 0u;
  bool trimAsyncUploadCommandListPoolAfterTextureUploads = false;
  std::vector<PendingAsyncUploadSubmission> pendingAsyncUploadSubmissions;
  std::vector<nvrhi::CommandListHandle> availableAsyncUploadCommandLists;
  uint64_t undeclaredGraphicsPipelineVariantMisses = 0u;
  uint64_t undeclaredMeshletPipelineVariantMisses = 0u;
  std::deque<GpuTimingReport> completedGpuTimingReports;
  GpuTimingReport latestCompletedGpuTimingReport{};
  uint64_t droppedGpuTimingReports = 0u;
  SlotPool<UnmaskedNonZeroGenerationPolicy> recordingContextSlots;
  SlotPool<UnmaskedNonZeroGenerationPolicy> recordedCommandBufferSlots;
  SlotPool<UnmaskedNonZeroGenerationPolicy> submissionSlots;
  std::vector<SubmissionRecord> submissions;
  std::unique_ptr<GeometryPool> geometryPool;
  bool loggedGpuTimingQueryWarning = false;
  bool loggedWholeFrameTimingQueryWarning = false;
  bool loggedShadowSdsmTimingCollectionDiagnostic = false;
};

namespace {

using Impl = NvrhiGPUDeviceImpl;

template <typename Resource>
void retireNativeResource(Impl &impl, Impl::RetiredResourceTable table,
                          uint32_t slotIndex, Resource resource) {
  impl.retiredResources.push_back(Impl::RetiredResource{
      .resource = Impl::RetiredNativeResource{std::move(resource)},
      .table = table,
      .slotIndex = slotIndex,
      .requiredRecordingSerial = impl.recordingRetirement.latestIssuedSerial(),
      .lastSubmittedAtDestruction = impl.latestSubmittedInstance,
  });
}

void recycleRetiredResourceSlot(Impl &impl,
                                const Impl::RetiredResource &retired) {
  switch (retired.table) {
  case Impl::RetiredResourceTable::Buffer:
    impl.buffers.recycleRetired(retired.slotIndex);
    break;
  case Impl::RetiredResourceTable::Texture:
    impl.textures.recycleRetired(retired.slotIndex);
    break;
  case Impl::RetiredResourceTable::Sampler:
    impl.samplers.recycleRetired(retired.slotIndex);
    break;
  case Impl::RetiredResourceTable::Shader:
    impl.shaders.recycleRetired(retired.slotIndex);
    break;
  case Impl::RetiredResourceTable::RenderPipeline:
    impl.renderPipelines.recycleRetired(retired.slotIndex);
    break;
  case Impl::RetiredResourceTable::ComputePipeline:
    impl.computePipelines.recycleRetired(retired.slotIndex);
    break;
  case Impl::RetiredResourceTable::MeshletPipeline:
    impl.meshletPipelines.recycleRetired(retired.slotIndex);
    break;
  }
}

void collectRetiredResources(Impl &impl, uint64_t completedInstance) {
  size_t writeIndex = 0u;
  for (size_t readIndex = 0u; readIndex < impl.retiredResources.size();
       ++readIndex) {
    Impl::RetiredResource &retired = impl.retiredResources[readIndex];
    if (!retired.lastUseResolved) {
      const std::optional<uint64_t> lastUse =
          impl.recordingRetirement.tryResolveLastUse(
              retired.requiredRecordingSerial,
              retired.lastSubmittedAtDestruction);
      if (lastUse.has_value()) {
        retired.lastUseInstance = *lastUse;
        retired.lastUseResolved = true;
      }
    }
    if (retired.lastUseResolved &&
        retired.lastUseInstance <= completedInstance) {
      recycleRetiredResourceSlot(impl, retired);
      continue;
    }
    if (writeIndex != readIndex) {
      impl.retiredResources[writeIndex] = std::move(retired);
    }
    ++writeIndex;
  }
  impl.retiredResources.resize(writeIndex);
}

void releaseAllRetiredResourcesAfterIdle(Impl &impl) {
  for (const Impl::RetiredResource &retired : impl.retiredResources) {
    recycleRetiredResourceSlot(impl, retired);
  }
  impl.retiredResources.clear();
}

[[nodiscard]] const BufferResource *
retiredBufferAtSlot(const Impl &impl, uint32_t slotIndex,
                    const Impl::RetiredResource **outRetirement = nullptr) {
  for (auto it = impl.retiredResources.rbegin();
       it != impl.retiredResources.rend(); ++it) {
    if (it->table != Impl::RetiredResourceTable::Buffer ||
        it->slotIndex != slotIndex) {
      continue;
    }
    if (outRetirement != nullptr) {
      *outRetirement = &*it;
    }
    return std::get_if<BufferResource>(&it->resource);
  }
  return nullptr;
}

void invalidateCachedFramebuffers(Impl &impl, TextureHandle texture) {
  if (!nuri::isValid(texture)) {
    return;
  }
  std::erase_if(impl.cachedFramebuffers,
                [texture](const CachedFramebuffer &entry) {
                  return entry.key.references(texture);
                });
}

void invalidateSwapchainFramebuffers(Impl &impl) {
  std::erase_if(impl.cachedFramebuffers, [](const CachedFramebuffer &entry) {
    return entry.key.colorIsSwapchain;
  });
  ++impl.swapchainGeneration;
  if (impl.swapchainGeneration == 0u) {
    impl.swapchainGeneration = 1u;
  }
}

void flushMappedBufferRange(Impl &impl, BufferResource &buffer, size_t offset,
                            size_t size) {
  if (size == 0u || buffer.mapped == nullptr ||
      buffer.mappedMemory == VK_NULL_HANDLE || impl.device == VK_NULL_HANDLE) {
    return;
  }
  if (offset > buffer.byteSize || size > buffer.byteSize - offset) {
    return;
  }

  const VkDeviceSize atomSize = std::max<VkDeviceSize>(
      1u, impl.physicalDeviceProperties.limits.nonCoherentAtomSize);
  const VkDeviceSize writeOffset = static_cast<VkDeviceSize>(offset);
  const VkDeviceSize alignedOffset =
      atomSize > 1u ? (writeOffset / atomSize) * atomSize : writeOffset;

  VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
  range.memory = buffer.mappedMemory;
  range.offset = alignedOffset;
  range.size = VK_WHOLE_SIZE;

  const VkResult result = vkFlushMappedMemoryRanges(impl.device, 1u, &range);
  if (result != VK_SUCCESS) {
    NURI_LOG_WARNING("NvrhiGPUDevice::flushMappedBuffer: flush failed for "
                     "buffer '%s' with VkResult %d",
                     buffer.debugName.c_str(), static_cast<int>(result));
  }
}

void accumulateGpuTimingScope(GpuTimingReport &report, GpuTimingScope scope,
                              float timeMs, uint64_t frameIndex) {
  for (const GpuTimingScopeMergeDesc desc : kGpuTimingScopeDescs) {
    if (desc.scope != scope) {
      continue;
    }
    report.*desc.timeMs += timeMs;
    report.*desc.sourceFrameIndex = frameIndex;
    report.availableScopeMask |= desc.bit;
    break;
  }

  const GpuTimingScope parent = gpuTimingParentScope(scope);
  if (parent != GpuTimingScope::None) {
    accumulateGpuTimingScope(report, parent, timeMs, frameIndex);
  }
}

nvrhi::TimerQueryHandle createGpuTimerQuery(Impl &impl) {
  if (!impl.nvrhiDevice) {
    return {};
  }
  nvrhi::TimerQueryHandle query = impl.nvrhiDevice->createTimerQuery();
  if (!query && !impl.loggedGpuTimingQueryWarning) {
    impl.loggedGpuTimingQueryWarning = true;
    NURI_LOG_WARNING(
        "NvrhiGPUDevice: failed to create a GPU timer query; timing for "
        "some passes will be unavailable");
  }
  return query;
}

void destroyWholeFrameTimingSlot(Impl &impl, NvrhiWholeFrameTimingSlot &slot) {
  slot.commandLists = {};
  if (slot.queryPool != VK_NULL_HANDLE && impl.device != VK_NULL_HANDLE) {
    vkDestroyQueryPool(impl.device, slot.queryPool, nullptr);
  }
  slot.queryPool = VK_NULL_HANDLE;
}

[[nodiscard]] bool initializeWholeFrameTimingSlots(Impl &impl) {
  if (!impl.graphicsQueueTimestampsSupported || impl.device == VK_NULL_HANDLE ||
      !impl.nvrhiDevice) {
    return false;
  }
  impl.availableWholeFrameTimingSlots.reserve(kWholeFrameTimingSlotCount);
  const auto fail = [&impl]() {
    for (NvrhiWholeFrameTimingSlot &created :
         impl.availableWholeFrameTimingSlots) {
      destroyWholeFrameTimingSlot(impl, created);
    }
    impl.availableWholeFrameTimingSlots.clear();
    return false;
  };
  const VkQueryPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = 2u,
  };
  for (uint32_t i = 0u; i < kWholeFrameTimingSlotCount; ++i) {
    NvrhiWholeFrameTimingSlot slot{};
    if (vkCreateQueryPool(impl.device, &poolInfo, nullptr, &slot.queryPool) !=
        VK_SUCCESS) {
      destroyWholeFrameTimingSlot(impl, slot);
      return fail();
    }
    slot.commandLists[0] = impl.nvrhiDevice->createCommandList();
    slot.commandLists[1] = impl.nvrhiDevice->createCommandList();
    if (!slot.commandLists[0] || !slot.commandLists[1]) {
      destroyWholeFrameTimingSlot(impl, slot);
      return fail();
    }
    impl.availableWholeFrameTimingSlots.push_back(std::move(slot));
  }
  return true;
}

[[nodiscard]] NvrhiWholeFrameTimingSlot
acquireWholeFrameTimingSlot(Impl &impl) {
  if (impl.availableWholeFrameTimingSlots.empty()) {
    return {};
  }
  NvrhiWholeFrameTimingSlot result =
      std::move(impl.availableWholeFrameTimingSlots.back());
  impl.availableWholeFrameTimingSlots.pop_back();

  result.commandLists[0]->open();
  const nvrhi::Object beginNative = result.commandLists[0]->getNativeObject(
      nvrhi::ObjectTypes::VK_CommandBuffer);
  const VkCommandBuffer beginCommandBuffer =
      static_cast<VkCommandBuffer>(beginNative);
  if (beginCommandBuffer == VK_NULL_HANDLE) {
    result.commandLists[0]->close();
    impl.availableWholeFrameTimingSlots.push_back(std::move(result));
    return {};
  }
  vkCmdResetQueryPool(beginCommandBuffer, result.queryPool, 0u, 2u);
  vkCmdWriteTimestamp2(beginCommandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                       result.queryPool, 0u);
  result.commandLists[0]->close();

  result.commandLists[1]->open();
  const nvrhi::Object endNative = result.commandLists[1]->getNativeObject(
      nvrhi::ObjectTypes::VK_CommandBuffer);
  const VkCommandBuffer endCommandBuffer =
      static_cast<VkCommandBuffer>(endNative);
  if (endCommandBuffer == VK_NULL_HANDLE) {
    result.commandLists[1]->close();
    impl.availableWholeFrameTimingSlots.push_back(std::move(result));
    return {};
  }
  vkCmdWriteTimestamp2(endCommandBuffer, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                       result.queryPool, 1u);
  result.commandLists[1]->close();
  return result;
}

class ScopedNvrhiPassTiming final {
public:
  ScopedNvrhiPassTiming(Impl &impl, nvrhi::ICommandList &commandList,
                        std::vector<NvrhiTimingQuery> *outQueries,
                        GpuTimingScope scope, std::string_view debugLabel)
      : impl_(impl), commandList_(commandList), outQueries_(outQueries),
        scope_(scope), debugLabel_(debugLabel) {
    if (outQueries_ == nullptr) {
      return;
    }
    query_ = createGpuTimerQuery(impl_);
    if (!query_) {
      return;
    }
    commandList_.beginTimerQuery(query_.Get());
    active_ = true;
  }

  ScopedNvrhiPassTiming(const ScopedNvrhiPassTiming &) = delete;
  ScopedNvrhiPassTiming &operator=(const ScopedNvrhiPassTiming &) = delete;

  ~ScopedNvrhiPassTiming() { cancel(); }

  void commit() {
    if (!active_) {
      return;
    }
    commandList_.endTimerQuery(query_.Get());
    if (outQueries_ != nullptr) {
      outQueries_->push_back(NvrhiTimingQuery{
          .scope = scope_,
          .debugLabel = std::string(debugLabel_),
          .query = std::move(query_),
      });
    }
    active_ = false;
  }

  void cancel() {
    if (!active_) {
      return;
    }
    commandList_.endTimerQuery(query_.Get());
    query_ = nullptr;
    active_ = false;
  }

private:
  Impl &impl_;
  nvrhi::ICommandList &commandList_;
  std::vector<NvrhiTimingQuery> *outQueries_ = nullptr;
  GpuTimingScope scope_ = GpuTimingScope::None;
  std::string_view debugLabel_{};
  nvrhi::TimerQueryHandle query_{};
  bool active_ = false;
};

void collectCompletedGpuTimingSubmissions(Impl &impl) {
  if (!impl.nvrhiDevice || impl.pendingGpuTimingSubmissions.empty()) {
    return;
  }

  const uint64_t completedInstance =
      impl.nvrhiDevice->queueGetCompletedInstance(
          nvrhi::CommandQueue::Graphics);
  size_t writeIndex = 0u;
  for (size_t readIndex = 0u;
       readIndex < impl.pendingGpuTimingSubmissions.size(); ++readIndex) {
    PendingGpuTimingSubmission &pending =
        impl.pendingGpuTimingSubmissions[readIndex];
    // Timer-query slots can contain available results from an older owner until
    // this submission executes its reset commands. Never poll them before the
    // submission has completed, even when no whole-frame timing slot was
    // available for this frame.
    bool allQueriesReady = pending.submissionInstance <= completedInstance;
    if (allQueriesReady) {
      for (const NvrhiTimingQuery &timingQuery : pending.timingQueries) {
        if (!timingQuery.query ||
            !impl.nvrhiDevice->pollTimerQuery(timingQuery.query.Get())) {
          allQueriesReady = false;
          break;
        }
      }
    }
    if (!allQueriesReady) {
      if (writeIndex != readIndex) {
        impl.pendingGpuTimingSubmissions[writeIndex] = std::move(pending);
      }
      ++writeIndex;
      continue;
    }

    GpuTimingReport completedReport{};
    completedReport.passTimings.reserve(pending.timingQueries.size());
    bool collectedShadowSdsm = false;
    float collectedShadowSdsmTimeMs = 0.0f;
    for (const NvrhiTimingQuery &timingQuery : pending.timingQueries) {
      if (!timingQuery.query) {
        continue;
      }
      const float timeMs =
          impl.nvrhiDevice->getTimerQueryTime(timingQuery.query.Get()) *
          1000.0f;
      accumulateGpuTimingScope(completedReport, timingQuery.scope, timeMs,
                               pending.frameIndex);
      completedReport.passTimings.push_back(GpuTimingReport::PassTiming{
          .debugName = timingQuery.debugLabel.empty()
                           ? std::string("unnamed_pass")
                           : timingQuery.debugLabel,
          .sourceFrameIndex = pending.frameIndex,
          .timeMs = timeMs,
      });
      if (timingQuery.scope == GpuTimingScope::ShadowSdsm) {
        collectedShadowSdsm = true;
        collectedShadowSdsmTimeMs += timeMs;
      }
    }
    if (pending.wholeFrameTiming.queryPool != VK_NULL_HANDLE) {
      std::array<uint64_t, 2> timestamps{};
      const VkResult timingResult =
          vkGetQueryPoolResults(impl.device, pending.wholeFrameTiming.queryPool,
                                0u, 2u, sizeof(timestamps), timestamps.data(),
                                sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
      if (timingResult == VK_SUCCESS && timestamps[1] >= timestamps[0]) {
        const double elapsedMs =
            static_cast<double>(timestamps[1] - timestamps[0]) *
            static_cast<double>(
                impl.physicalDeviceProperties.limits.timestampPeriod) /
            1'000'000.0;
        completedReport.wholeFrameTimeMs = static_cast<float>(elapsedMs);
        completedReport.wholeFrameSourceFrameIndex = pending.frameIndex;
        completedReport.availableScopeMask |= kGpuTimingScopeWholeFrameBit;
      } else if (!impl.loggedWholeFrameTimingQueryWarning) {
        impl.loggedWholeFrameTimingQueryWarning = true;
        NURI_LOG_WARNING(
            "NvrhiGPUDevice: whole-frame timestamp query readback failed");
      }
      NvrhiWholeFrameTimingSlot completedSlot =
          std::move(pending.wholeFrameTiming);
      pending.wholeFrameTiming = {};
      impl.availableWholeFrameTimingSlots.push_back(std::move(completedSlot));
    }
    if (collectedShadowSdsm &&
        !impl.loggedShadowSdsmTimingCollectionDiagnostic) {
      impl.loggedShadowSdsmTimingCollectionDiagnostic = true;
      NURI_LOG_INFO(
          "NvrhiGPUDevice: collected shadow SDSM timing result frame=%llu "
          "timeMs=%.3f",
          static_cast<unsigned long long>(pending.frameIndex),
          collectedShadowSdsmTimeMs);
    }
    mergeGpuTimingReportScopes(impl.latestCompletedGpuTimingReport,
                               completedReport);
    impl.latestCompletedGpuTimingReport.passTimings =
        completedReport.passTimings;
    if (impl.completedGpuTimingReports.size() >= 256u) {
      impl.completedGpuTimingReports.pop_front();
      ++impl.droppedGpuTimingReports;
    }
    impl.completedGpuTimingReports.push_back(completedReport);
  }

  impl.pendingGpuTimingSubmissions.resize(writeIndex);
}

[[nodiscard]] std::string vkError(std::string_view context, VkResult result) {
  std::string message(context);
  message += ": ";
  message += nvrhi::vulkan::resultToString(result);
  return message;
}

[[nodiscard]] Result<bool, std::string>
waitForGraphicsQueueInstance(Impl &impl, uint64_t instance,
                             std::string_view context) {
  if (instance == 0u) {
    return Result<bool, std::string>::makeResult(true);
  }
  const VkSemaphore queueSemaphore =
      impl.nvrhiDevice->getQueueSemaphore(nvrhi::CommandQueue::Graphics);
  const VkSemaphoreWaitInfo waitInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1u,
      .pSemaphores = &queueSemaphore,
      .pValues = &instance,
  };
  const VkResult result = vkWaitSemaphores(impl.device, &waitInfo, UINT64_MAX);
  if (result != VK_SUCCESS) {
    return Result<bool, std::string>::makeError(vkError(context, result));
  }
  return Result<bool, std::string>::makeResult(true);
}

void collectCompletedAsyncUploadSubmissions(Impl &impl) {
  if (!impl.nvrhiDevice || impl.pendingAsyncUploadSubmissions.empty()) {
    return;
  }

  const uint64_t completedGraphics =
      impl.nvrhiDevice->queueGetCompletedInstance(
          nvrhi::CommandQueue::Graphics);
  const uint64_t completedCopy =
      impl.hasDedicatedAssetCopyQueue
          ? impl.nvrhiDevice->queueGetCompletedInstance(
                nvrhi::CommandQueue::Copy)
          : completedGraphics;
  size_t writeIndex = 0u;
  for (size_t readIndex = 0u;
       readIndex < impl.pendingAsyncUploadSubmissions.size(); ++readIndex) {
    PendingAsyncUploadSubmission &pending =
        impl.pendingAsyncUploadSubmissions[readIndex];
    const uint64_t completed = pending.queue == nvrhi::CommandQueue::Copy
                                   ? completedCopy
                                   : completedGraphics;
    if (pending.submissionInstance <= completed) {
      impl.availableAsyncUploadCommandLists.push_back(
          std::move(pending.commandList));
      continue;
    }
    if (pending.submissionInstance > completed) {
      if (writeIndex != readIndex) {
        impl.pendingAsyncUploadSubmissions[writeIndex] = std::move(pending);
      }
      ++writeIndex;
    }
  }
  impl.pendingAsyncUploadSubmissions.resize(writeIndex);
}

[[nodiscard]] nvrhi::CommandListHandle
acquireAsyncUploadCommandList(Impl &impl) {
  if (!impl.availableAsyncUploadCommandLists.empty()) {
    nvrhi::CommandListHandle commandList =
        std::move(impl.availableAsyncUploadCommandLists.back());
    impl.availableAsyncUploadCommandLists.pop_back();
    return commandList;
  }
  return impl.nvrhiDevice->createCommandList(
      nvrhi::CommandListParameters{}.setQueueType(
          impl.hasDedicatedAssetCopyQueue ? nvrhi::CommandQueue::Copy
                                          : nvrhi::CommandQueue::Graphics));
}

[[nodiscard]] nvrhi::CommandListHandle &
ensurePendingAsyncUploadCommandList(Impl &impl) {
  if (!impl.pendingAsyncUploadCommandList) {
    impl.pendingAsyncUploadCommandList = acquireAsyncUploadCommandList(impl);
    if (impl.pendingAsyncUploadCommandList) {
      impl.pendingAsyncUploadCommandList->open();
    }
  }
  return impl.pendingAsyncUploadCommandList;
}

[[nodiscard]] nvrhi::CommandListHandle
takePendingAsyncUploadCommandList(Impl &impl) {
  if (!impl.pendingAsyncUploadCommandList) {
    return {};
  }
  impl.pendingAsyncUploadCommandList->close();
  if (impl.pendingAsyncUploadTextureCount != 0u) {
    ++impl.textureUploadBatchesSubmitted;
  }
  impl.pendingAsyncUploadBytes = 0u;
  impl.pendingAsyncUploadTextureCount = 0u;
  return std::exchange(impl.pendingAsyncUploadCommandList, {});
}

uint64_t
submitAsyncUploadCommandList(Impl &impl,
                             const nvrhi::CommandListHandle &commandList,
                             bool containsTextureData) {
  const nvrhi::CommandQueue queue = impl.hasDedicatedAssetCopyQueue
                                        ? nvrhi::CommandQueue::Copy
                                        : nvrhi::CommandQueue::Graphics;
  nvrhi::ICommandList *raw = commandList.Get();
  const uint64_t instance =
      impl.nvrhiDevice->executeCommandLists(&raw, 1u, queue);
  if (queue == nvrhi::CommandQueue::Copy) {
    impl.latestAsyncUploadCopyInstance = instance;
  } else {
    impl.latestSubmittedInstance = instance;
    impl.latestAsyncUploadGraphicsInstance = instance;
  }
  impl.pendingAsyncUploadSubmissions.push_back(
      PendingAsyncUploadSubmission{.submissionInstance = instance,
                                   .commandList = commandList,
                                   .containsTextureData = containsTextureData,
                                   .queue = queue});
  return instance;
}

void flushPendingAsyncUploadCommandList(Impl &impl,
                                        bool queueGraphicsVisibility = true) {
  const bool containsTextureData = impl.pendingAsyncUploadTextureCount != 0u;
  nvrhi::CommandListHandle commandList =
      takePendingAsyncUploadCommandList(impl);
  if (commandList) {
    const uint64_t instance =
        submitAsyncUploadCommandList(impl, commandList, containsTextureData);
    if (queueGraphicsVisibility && impl.hasDedicatedAssetCopyQueue) {
      impl.nvrhiDevice->queueWaitForSemaphore(
          nvrhi::CommandQueue::Graphics,
          impl.nvrhiDevice->getQueueSemaphore(nvrhi::CommandQueue::Copy),
          instance);
    }
  }
}

[[nodiscard]] bool hasInstanceLayer(const char *layerName) {
  uint32_t count = 0u;
  vkEnumerateInstanceLayerProperties(&count, nullptr);
  std::vector<VkLayerProperties> layers(count);
  if (count != 0u) {
    vkEnumerateInstanceLayerProperties(&count, layers.data());
  }
  return std::any_of(layers.begin(), layers.end(),
                     [layerName](const VkLayerProperties &layer) {
                       return std::strcmp(layer.layerName, layerName) == 0;
                     });
}

[[nodiscard]] bool hasInstanceExtension(const char *extensionName) {
  uint32_t count = 0u;
  vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> extensions(count);
  if (count != 0u) {
    vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data());
  }
  return std::any_of(extensions.begin(), extensions.end(),
                     [extensionName](const VkExtensionProperties &extension) {
                       return std::strcmp(extension.extensionName,
                                          extensionName) == 0;
                     });
}

[[nodiscard]] bool hasDeviceExtension(VkPhysicalDevice device,
                                      const char *extensionName) {
  uint32_t count = 0u;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> extensions(count);
  if (count != 0u) {
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count,
                                         extensions.data());
  }
  return std::any_of(extensions.begin(), extensions.end(),
                     [extensionName](const VkExtensionProperties &extension) {
                       return std::strcmp(extension.extensionName,
                                          extensionName) == 0;
                     });
}

[[nodiscard]] MeshletLimits
toMeshletLimits(const VkPhysicalDeviceMeshShaderPropertiesEXT &props) {
  MeshletLimits limits{};
  limits.maxTaskWorkGroupTotalCount = props.maxTaskWorkGroupTotalCount;
  limits.maxTaskWorkGroupInvocations = props.maxTaskWorkGroupInvocations;
  limits.maxTaskWorkGroupSizeX = props.maxTaskWorkGroupSize[0];
  limits.maxTaskWorkGroupSizeY = props.maxTaskWorkGroupSize[1];
  limits.maxTaskWorkGroupSizeZ = props.maxTaskWorkGroupSize[2];
  limits.maxTaskWorkGroupCountX = props.maxTaskWorkGroupCount[0];
  limits.maxTaskWorkGroupCountY = props.maxTaskWorkGroupCount[1];
  limits.maxTaskWorkGroupCountZ = props.maxTaskWorkGroupCount[2];
  limits.maxTaskPayloadBytes = props.maxTaskPayloadSize;
  limits.maxTaskSharedMemoryBytes = props.maxTaskSharedMemorySize;
  limits.maxTaskPayloadAndSharedMemoryBytes =
      props.maxTaskPayloadAndSharedMemorySize;
  limits.maxMeshWorkGroupTotalCount = props.maxMeshWorkGroupTotalCount;
  limits.maxMeshWorkGroupInvocations = props.maxMeshWorkGroupInvocations;
  limits.maxMeshWorkGroupSizeX = props.maxMeshWorkGroupSize[0];
  limits.maxMeshWorkGroupSizeY = props.maxMeshWorkGroupSize[1];
  limits.maxMeshWorkGroupSizeZ = props.maxMeshWorkGroupSize[2];
  limits.maxMeshWorkGroupCountX = props.maxMeshWorkGroupCount[0];
  limits.maxMeshWorkGroupCountY = props.maxMeshWorkGroupCount[1];
  limits.maxMeshWorkGroupCountZ = props.maxMeshWorkGroupCount[2];
  limits.maxMeshSharedMemoryBytes = props.maxMeshSharedMemorySize;
  limits.maxMeshPayloadAndSharedMemoryBytes =
      props.maxMeshPayloadAndSharedMemorySize;
  limits.maxMeshOutputMemoryBytes = props.maxMeshOutputMemorySize;
  limits.maxMeshPayloadAndOutputMemoryBytes =
      props.maxMeshPayloadAndOutputMemorySize;
  limits.maxMeshOutputComponents = props.maxMeshOutputComponents;
  limits.meshOutputPerVertexGranularity = props.meshOutputPerVertexGranularity;
  limits.meshOutputPerPrimitiveGranularity =
      props.meshOutputPerPrimitiveGranularity;
  limits.maxMeshOutputVertices = props.maxMeshOutputVertices;
  limits.maxMeshOutputPrimitives = props.maxMeshOutputPrimitives;
  limits.maxMeshOutputLayers = props.maxMeshOutputLayers;
  limits.maxPreferredTaskWorkGroupInvocations =
      props.maxPreferredTaskWorkGroupInvocations;
  limits.maxPreferredMeshWorkGroupInvocations =
      props.maxPreferredMeshWorkGroupInvocations;
  limits.prefersLocalInvocationVertexOutput =
      props.prefersLocalInvocationVertexOutput == VK_TRUE;
  limits.prefersLocalInvocationPrimitiveOutput =
      props.prefersLocalInvocationPrimitiveOutput == VK_TRUE;
  limits.prefersCompactVertexOutput =
      props.prefersCompactVertexOutput == VK_TRUE;
  limits.prefersCompactPrimitiveOutput =
      props.prefersCompactPrimitiveOutput == VK_TRUE;
  return limits;
}

[[nodiscard]] bool meshletLimitsSupportDefaults(const MeshletLimits &limits) {
  return limits.maxMeshOutputVertices >= kDefaultMeshletMaxVertices &&
         limits.maxMeshOutputPrimitives >= kDefaultMeshletMaxPrimitives &&
         limits.maxMeshWorkGroupInvocations != 0u &&
         limits.maxTaskWorkGroupInvocations != 0u &&
         limits.maxMeshWorkGroupSizeX != 0u &&
         limits.maxTaskWorkGroupSizeX != 0u && limits.maxTaskPayloadBytes != 0u;
}

VKAPI_ATTR VkBool32 VKAPI_CALL nvrhiDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT *callbackData, void *) {
  const char *message =
      callbackData != nullptr && callbackData->pMessage != nullptr
          ? callbackData->pMessage
          : "";
  if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0u) {
    NURI_LOG_ERROR("Vulkan validation: %s", message);
  } else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) !=
             0u) {
    NURI_LOG_WARNING("Vulkan validation: %s", message);
  } else {
    NURI_LOG_DEBUG("Vulkan validation: %s", message);
  }
  return VK_FALSE;
}

void destroyDebugMessenger(Impl &impl) {
  if (impl.debugMessenger == VK_NULL_HANDLE ||
      impl.instance == VK_NULL_HANDLE) {
    return;
  }
  auto *destroyMessenger =
      reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
          vkGetInstanceProcAddr(impl.instance,
                                "vkDestroyDebugUtilsMessengerEXT"));
  if (destroyMessenger != nullptr) {
    destroyMessenger(impl.instance, impl.debugMessenger, nullptr);
  }
  impl.debugMessenger = VK_NULL_HANDLE;
}

[[nodiscard]] Result<bool, std::string> createDebugMessenger(Impl &impl) {
  if (!impl.validationEnabled ||
      !hasInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto *createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(impl.instance, "vkCreateDebugUtilsMessengerEXT"));
  if (createMessenger == nullptr) {
    return Result<bool, std::string>::makeResult(true);
  }
  const VkDebugUtilsMessengerCreateInfoEXT createInfo{
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
      .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
      .pfnUserCallback = nvrhiDebugCallback,
  };
  const VkResult result = createMessenger(impl.instance, &createInfo, nullptr,
                                          &impl.debugMessenger);
  if (result != VK_SUCCESS) {
    return Result<bool, std::string>::makeError(
        vkError("NvrhiGPUDevice::createDebugMessenger", result));
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string> createInstance(Impl &impl) {
  const VkResult volkResult = volkInitialize();
  if (volkResult != VK_SUCCESS) {
    return Result<bool, std::string>::makeError(
        vkError("NvrhiGPUDevice::createInstance volkInitialize", volkResult));
  }
  if (glslang_initialize_process() == 0) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::createInstance glslang_initialize_process failed");
  }
  impl.glslangInitialized = true;

  uint32_t glfwExtensionCount = 0u;
  const char **glfwExtensions =
      glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
  if (glfwExtensions == nullptr || glfwExtensionCount == 0u) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::createInstance: GLFW did not provide Vulkan "
        "instance extensions");
  }

  std::unordered_set<std::string_view> seenExtensions;
  impl.instanceExtensions.clear();
  for (uint32_t i = 0u; i < glfwExtensionCount; ++i) {
    if (seenExtensions.emplace(glfwExtensions[i]).second) {
      impl.instanceExtensions.push_back(glfwExtensions[i]);
    }
  }
  if (impl.validationEnabled &&
      hasInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME) &&
      seenExtensions.emplace(VK_EXT_DEBUG_UTILS_EXTENSION_NAME).second) {
    impl.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  std::vector<const char *> layers;
  if (impl.validationEnabled &&
      hasInstanceLayer("VK_LAYER_KHRONOS_validation")) {
    layers.push_back("VK_LAYER_KHRONOS_validation");
  }

  const VkApplicationInfo appInfo{
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "Nuri",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "Nuri",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_API_VERSION_1_3,
  };
  const VkInstanceCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = static_cast<uint32_t>(layers.size()),
      .ppEnabledLayerNames = layers.empty() ? nullptr : layers.data(),
      .enabledExtensionCount =
          static_cast<uint32_t>(impl.instanceExtensions.size()),
      .ppEnabledExtensionNames = impl.instanceExtensions.data(),
  };
  const VkResult result =
      vkCreateInstance(&createInfo, nullptr, &impl.instance);
  if (result != VK_SUCCESS) {
    return Result<bool, std::string>::makeError(
        vkError("NvrhiGPUDevice::createInstance", result));
  }
  volkLoadInstance(impl.instance);
  return createDebugMessenger(impl);
}

[[nodiscard]] QueueFamilySelection findQueueFamilies(Impl &impl,
                                                     VkPhysicalDevice device) {
  QueueFamilySelection selection{};
  uint32_t queueFamilyCount = 0u;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> families(queueFamilyCount);
  if (queueFamilyCount != 0u) {
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             families.data());
  }
  for (uint32_t i = 0u; i < queueFamilyCount; ++i) {
    VkBool32 presentSupported = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, impl.surface,
                                         &presentSupported);
    if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0u &&
        presentSupported == VK_TRUE) {
      selection.graphics = i;
      selection.graphicsQueueCount = families[i].queueCount;
      selection.hasGraphics = true;
      // NVRHI does not currently emit Vulkan queue-family ownership
      // transfers. A second queue from the graphics-capable family is the
      // safe asynchronous copy path: it supports the final shader-readable
      // barriers and needs no ownership transfer. Transfer-only families
      // deliberately remain on the graphics fallback.
      selection.hasDedicatedCopyQueue = families[i].queueCount >= 2u;
      return selection;
    }
  }
  return selection;
}

[[nodiscard]] bool hasRequiredVulkanFeatures(VkPhysicalDevice device,
                                             std::string &reason) {
  VkPhysicalDeviceVulkan13Features features13{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
  };
  VkPhysicalDeviceVulkan12Features features12{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &features13,
  };
  VkPhysicalDeviceFeatures2 features2{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &features12,
  };
  vkGetPhysicalDeviceFeatures2(device, &features2);

  if (features12.bufferDeviceAddress != VK_TRUE) {
    reason = "buffer device address is not supported";
    return false;
  }
  if (features12.timelineSemaphore != VK_TRUE) {
    reason = "timeline semaphores are not supported";
    return false;
  }
  if (features12.runtimeDescriptorArray != VK_TRUE ||
      features12.descriptorBindingPartiallyBound != VK_TRUE ||
      features12.shaderSampledImageArrayNonUniformIndexing != VK_TRUE) {
    reason = "descriptor indexing features are not supported";
    return false;
  }
  if (features12.shaderFloat16 != VK_TRUE) {
    reason = "float16 shader arithmetic is not supported";
    return false;
  }
  if (features2.features.tessellationShader != VK_TRUE ||
      features2.features.geometryShader != VK_TRUE) {
    reason = "tessellation/geometry shader stages are not supported";
    return false;
  }
  if (features13.dynamicRendering != VK_TRUE ||
      features13.synchronization2 != VK_TRUE) {
    reason = "Vulkan 1.3 dynamic rendering/synchronization2 is unsupported";
    return false;
  }
  if (features13.shaderDemoteToHelperInvocation != VK_TRUE) {
    reason = "shader demote to helper invocation is not supported";
    return false;
  }
  return true;
}

[[nodiscard]] bool isDeviceSuitable(Impl &impl, VkPhysicalDevice device,
                                    QueueFamilySelection &queues,
                                    std::string &reason) {
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(device, &properties);
  if (properties.apiVersion < VK_API_VERSION_1_3) {
    reason = "Vulkan 1.3 is required, device exposes " +
             formatVkVersion(properties.apiVersion);
    return false;
  }
  if (!hasDeviceExtension(device, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
    reason = "VK_KHR_swapchain is not supported";
    return false;
  }
  queues = findQueueFamilies(impl, device);
  if (!queues.hasGraphics) {
    reason = "no graphics+present queue family";
    return false;
  }
  uint32_t surfaceFormatCount = 0u;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, impl.surface,
                                       &surfaceFormatCount, nullptr);
  uint32_t presentModeCount = 0u;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, impl.surface,
                                            &presentModeCount, nullptr);
  if (surfaceFormatCount == 0u || presentModeCount == 0u) {
    reason = "swapchain surface has no supported formats or present modes";
    return false;
  }
  return hasRequiredVulkanFeatures(device, reason);
}

[[nodiscard]] Result<bool, std::string> selectPhysicalDevice(Impl &impl) {
  uint32_t deviceCount = 0u;
  VkResult result =
      vkEnumeratePhysicalDevices(impl.instance, &deviceCount, nullptr);
  if (result != VK_SUCCESS) {
    return Result<bool, std::string>::makeError(
        vkError("NvrhiGPUDevice::selectPhysicalDevice", result));
  }
  if (deviceCount == 0u) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::selectPhysicalDevice: no Vulkan devices");
  }
  std::vector<VkPhysicalDevice> devices(deviceCount);
  result =
      vkEnumeratePhysicalDevices(impl.instance, &deviceCount, devices.data());
  if (result != VK_SUCCESS) {
    return Result<bool, std::string>::makeError(
        vkError("NvrhiGPUDevice::selectPhysicalDevice", result));
  }

  VkPhysicalDevice fallback = VK_NULL_HANDLE;
  QueueFamilySelection fallbackQueues{};
  std::string lastRejectReason;
  for (VkPhysicalDevice device : devices) {
    QueueFamilySelection queues{};
    std::string reason;
    if (!isDeviceSuitable(impl, device, queues, reason)) {
      lastRejectReason = reason;
      continue;
    }
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device, &properties);
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      impl.physicalDevice = device;
      impl.graphicsQueueFamily = queues.graphics;
      impl.hasDedicatedAssetCopyQueue = queues.hasDedicatedCopyQueue;
      break;
    }
    if (fallback == VK_NULL_HANDLE) {
      fallback = device;
      fallbackQueues = queues;
    }
  }
  if (impl.physicalDevice == VK_NULL_HANDLE && fallback != VK_NULL_HANDLE) {
    impl.physicalDevice = fallback;
    impl.graphicsQueueFamily = fallbackQueues.graphics;
    impl.hasDedicatedAssetCopyQueue = fallbackQueues.hasDedicatedCopyQueue;
  }
  if (impl.physicalDevice == VK_NULL_HANDLE) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::selectPhysicalDevice: no suitable Vulkan device (" +
        lastRejectReason + ")");
  }

  vkGetPhysicalDeviceProperties(impl.physicalDevice,
                                &impl.physicalDeviceProperties);
  uint32_t queueFamilyCount = 0u;
  vkGetPhysicalDeviceQueueFamilyProperties(impl.physicalDevice,
                                           &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(
      impl.physicalDevice, &queueFamilyCount, queueFamilies.data());
  impl.graphicsQueueTimestampsSupported =
      impl.graphicsQueueFamily < queueFamilies.size() &&
      queueFamilies[impl.graphicsQueueFamily].timestampValidBits != 0u;
  impl.multisampleCapabilities =
      queryMultisampleCapabilities(impl.physicalDevice, false);
  impl.adapterInfo = GPUAdapterInfo{
      .name = impl.physicalDeviceProperties.deviceName,
      .vendorId = impl.physicalDeviceProperties.vendorID,
      .deviceId = impl.physicalDeviceProperties.deviceID,
      .driverVersion =
          std::to_string(impl.physicalDeviceProperties.driverVersion),
  };
  vkGetPhysicalDeviceMemoryProperties(impl.physicalDevice,
                                      &impl.memoryProperties);
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string> createLogicalDevice(Impl &impl) {
  const std::array queuePriorities{1.0f, 0.75f};
  const VkDeviceQueueCreateInfo queueCreateInfo{
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = impl.graphicsQueueFamily,
      .queueCount = impl.hasDedicatedAssetCopyQueue ? 2u : 1u,
      .pQueuePriorities = queuePriorities.data(),
  };

  VkPhysicalDeviceVulkan13Features supported13{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
  };
  VkPhysicalDeviceVulkan12Features supported12{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &supported13,
  };
  VkPhysicalDeviceMeshShaderFeaturesEXT supportedMesh{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
      .pNext = &supported12,
  };
  VkPhysicalDeviceFeatures2 supported2{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &supportedMesh,
  };
  vkGetPhysicalDeviceFeatures2(impl.physicalDevice, &supported2);

  impl.meshletExtensionEnabled = false;
  impl.meshletFeaturesEnabled = false;
  impl.meshletsSupported = false;
  impl.meshletLimits = {};
  const bool meshShaderExtensionPresent = hasDeviceExtension(
      impl.physicalDevice, VK_EXT_MESH_SHADER_EXTENSION_NAME);
  const bool meshShaderFeaturesPresent = supportedMesh.taskShader == VK_TRUE &&
                                         supportedMesh.meshShader == VK_TRUE &&
                                         supported13.maintenance4 == VK_TRUE;
  const bool enableMeshShaderExtension =
      meshShaderExtensionPresent && meshShaderFeaturesPresent;
  if (enableMeshShaderExtension) {
    VkPhysicalDeviceMeshShaderPropertiesEXT meshProps{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT,
    };
    VkPhysicalDeviceProperties2 props2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &meshProps,
    };
    vkGetPhysicalDeviceProperties2(impl.physicalDevice, &props2);
    impl.meshletLimits = toMeshletLimits(meshProps);
  }

  VkPhysicalDeviceVulkan13Features enabled13{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
  };
  enabled13.synchronization2 = VK_TRUE;
  enabled13.dynamicRendering = VK_TRUE;
  enabled13.shaderDemoteToHelperInvocation = VK_TRUE;
  enabled13.maintenance4 = supported13.maintenance4;

  VkPhysicalDeviceVulkan12Features enabled12{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &enabled13,
  };
  enabled12.drawIndirectCount = supported12.drawIndirectCount;
  enabled12.shaderFloat16 = VK_TRUE;
  enabled12.descriptorIndexing = VK_TRUE;
  enabled12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
  enabled12.shaderStorageImageArrayNonUniformIndexing =
      supported12.shaderStorageImageArrayNonUniformIndexing;
  enabled12.descriptorBindingSampledImageUpdateAfterBind =
      supported12.descriptorBindingSampledImageUpdateAfterBind;
  enabled12.descriptorBindingStorageImageUpdateAfterBind =
      supported12.descriptorBindingStorageImageUpdateAfterBind;
  enabled12.descriptorBindingPartiallyBound = VK_TRUE;
  enabled12.runtimeDescriptorArray = VK_TRUE;
  enabled12.timelineSemaphore = VK_TRUE;
  enabled12.bufferDeviceAddress = VK_TRUE;
  VkPhysicalDeviceMeshShaderFeaturesEXT enabledMesh{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
      .pNext = &enabled12,
  };
  if (enableMeshShaderExtension) {
    enabledMesh.taskShader = VK_TRUE;
    enabledMesh.meshShader = VK_TRUE;
  }
  VkPhysicalDeviceFeatures2 enabled2{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = enableMeshShaderExtension ? static_cast<void *>(&enabledMesh)
                                         : static_cast<void *>(&enabled12),
  };
  enabled2.features.samplerAnisotropy = supported2.features.samplerAnisotropy;
  enabled2.features.fillModeNonSolid = supported2.features.fillModeNonSolid;
  enabled2.features.multiDrawIndirect = supported2.features.multiDrawIndirect;
  enabled2.features.shaderInt64 = supported2.features.shaderInt64;
  enabled2.features.tessellationShader = VK_TRUE;
  enabled2.features.geometryShader = VK_TRUE;

  impl.deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  if (enableMeshShaderExtension) {
    impl.deviceExtensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
    impl.meshletExtensionEnabled = true;
    impl.meshletFeaturesEnabled = true;
  }
  const VkDeviceCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &enabled2,
      .queueCreateInfoCount = 1u,
      .pQueueCreateInfos = &queueCreateInfo,
      .enabledExtensionCount =
          static_cast<uint32_t>(impl.deviceExtensions.size()),
      .ppEnabledExtensionNames = impl.deviceExtensions.data(),
  };
  const VkResult result =
      vkCreateDevice(impl.physicalDevice, &createInfo, nullptr, &impl.device);
  if (result != VK_SUCCESS) {
    return Result<bool, std::string>::makeError(
        vkError("NvrhiGPUDevice::createLogicalDevice", result));
  }
  vkGetDeviceQueue(impl.device, impl.graphicsQueueFamily, 0u,
                   &impl.graphicsQueue);
  if (impl.hasDedicatedAssetCopyQueue) {
    vkGetDeviceQueue(impl.device, impl.graphicsQueueFamily, 1u,
                     &impl.assetCopyQueue);
  }
  volkLoadDevice(impl.device);

  NURI_LOG_INFO("NvrhiGPUDevice: async asset uploads use %s",
                impl.hasDedicatedAssetCopyQueue
                    ? "a dedicated same-family copy queue"
                    : "the ordered graphics-queue fallback");

  impl.maxSamplerAnisotropy =
      supported2.features.samplerAnisotropy == VK_TRUE
          ? static_cast<uint8_t>(std::clamp(
                static_cast<uint32_t>(
                    impl.physicalDeviceProperties.limits.maxSamplerAnisotropy),
                1u, 16u))
          : 1u;
  impl.maxSamplerLodBias =
      impl.physicalDeviceProperties.limits.maxSamplerLodBias;
  impl.compressionCaps.bc7 =
      supported2.features.textureCompressionBC == VK_TRUE;
  // The NVRHI format enum used here exposes BC formats but not ETC2/ASTC, so
  // report only formats this backend can actually create.
  impl.compressionCaps.etc2 = false;
  impl.compressionCaps.astc = false;
  impl.supportsDrawIndirectCount = supported12.drawIndirectCount == VK_TRUE;

  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] nvrhi::BindingLayoutVector globalBindingLayouts(Impl &impl) {
  nvrhi::BindingLayoutVector layouts;
  layouts.push_back(impl.bindless2DLayout);
  layouts.push_back(impl.bindless3DLayout);
  layouts.push_back(impl.bindlessCubeLayout);
  layouts.push_back(impl.bindlessShadowLayout);
  layouts.push_back(impl.pushConstantsLayout);
  return layouts;
}

[[nodiscard]] nvrhi::BindingSetVector globalBindingSets(Impl &impl) {
  nvrhi::BindingSetVector sets;
  sets.push_back(impl.bindless2DTable);
  sets.push_back(impl.bindless3DTable);
  sets.push_back(impl.bindlessCubeTable);
  sets.push_back(impl.bindlessShadowTable);
  sets.push_back(impl.pushConstantsSet);
  return sets;
}

[[nodiscard]] Result<bool, std::string> createNvrhiDevice(Impl &impl) {
  nvrhi::vulkan::DeviceDesc desc{};
  desc.errorCB = &impl.nvrhiLog;
  desc.instance = impl.instance;
  desc.physicalDevice = impl.physicalDevice;
  desc.device = impl.device;
  desc.graphicsQueue = impl.graphicsQueue;
  desc.graphicsQueueIndex = static_cast<int>(impl.graphicsQueueFamily);
  if (impl.hasDedicatedAssetCopyQueue) {
    desc.transferQueue = impl.assetCopyQueue;
    desc.transferQueueIndex = static_cast<int>(impl.graphicsQueueFamily);
  }
  desc.deviceExtensions = impl.deviceExtensions.data();
  desc.numDeviceExtensions = impl.deviceExtensions.size();
  desc.instanceExtensions = impl.instanceExtensions.data();
  desc.numInstanceExtensions = impl.instanceExtensions.size();
  desc.bufferDeviceAddressSupported = true;
  desc.maxTimerQueries = kMaxNvrhiGpuTimerQueries;

#if VK_HEADER_VERSION >= 301
  vk::detail::DynamicLoader dynamicLoader(desc.vulkanLibraryName);
#else
  vk::DynamicLoader dynamicLoader(desc.vulkanLibraryName);
#endif
  const PFN_vkGetInstanceProcAddr getInstanceProcAddr =
      dynamicLoader.getProcAddress<PFN_vkGetInstanceProcAddr>(
          "vkGetInstanceProcAddr");
  VULKAN_HPP_DEFAULT_DISPATCHER.init(desc.instance, getInstanceProcAddr,
                                     desc.device);
  impl.nvrhiDevice = nvrhi::vulkan::createDevice(desc);
  if (!impl.nvrhiDevice) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::createNvrhiDevice: nvrhi::vulkan::createDevice "
        "failed");
  }
  const bool meshDispatchEntryPointsPresent =
      vkCmdDrawMeshTasksEXT != nullptr &&
      vkCmdDrawMeshTasksIndirectEXT != nullptr &&
      vkCmdDrawMeshTasksIndirectCountEXT != nullptr;
  impl.meshletsSupported =
      impl.meshletExtensionEnabled && impl.meshletFeaturesEnabled &&
      meshletLimitsSupportDefaults(impl.meshletLimits) &&
      meshDispatchEntryPointsPresent &&
      impl.nvrhiDevice->queryFeatureSupport(nvrhi::Feature::Meshlets);
  if (impl.meshletExtensionEnabled && impl.meshletFeaturesEnabled &&
      !meshDispatchEntryPointsPresent) {
    NURI_LOG_WARNING(
        "NvrhiGPUDevice::createNvrhiDevice: VK_EXT_mesh_shader is enabled, "
        "but one or more mesh dispatch entry points are unavailable");
  }
  impl.meshletLimits.supportsMeshDispatchIndirect = impl.meshletsSupported;
  impl.meshletLimits.supportsMeshDispatchIndirectCount =
      impl.meshletsSupported && impl.supportsDrawIndirectCount;
  impl.immediateCommandList = impl.nvrhiDevice->createCommandList();
  if (!impl.immediateCommandList) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::createNvrhiDevice: failed to create immediate "
        "command list");
  }
  if (impl.graphicsQueueTimestampsSupported &&
      !initializeWholeFrameTimingSlots(impl)) {
    impl.loggedWholeFrameTimingQueryWarning = true;
    NURI_LOG_WARNING(
        "NvrhiGPUDevice: whole-frame GPU timing disabled because its fixed "
        "query ring could not be created");
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string> createGlobalBindingLayouts(Impl &impl) {
  nvrhi::BindlessLayoutDesc bindless2D{};
  bindless2D.setVisibility(nvrhi::ShaderType::All)
      .setFirstSlot(0u)
      .setMaxCapacity(kBindlessCapacity)
      .setLayoutType(nvrhi::BindlessLayoutDesc::LayoutType::Immutable)
      .addRegisterSpace(nvrhi::BindingLayoutItem::Texture_SRV(0u))
      .addRegisterSpace(nvrhi::BindingLayoutItem::Sampler(1u))
      .addRegisterSpace(nvrhi::BindingLayoutItem::Texture_UAV(2u));
  impl.bindless2DLayout = impl.nvrhiDevice->createBindlessLayout(bindless2D);

  nvrhi::BindlessLayoutDesc bindless3D{};
  bindless3D.setVisibility(nvrhi::ShaderType::All)
      .setFirstSlot(0u)
      .setMaxCapacity(kBindlessCapacity)
      .setLayoutType(nvrhi::BindlessLayoutDesc::LayoutType::Immutable)
      .addRegisterSpace(nvrhi::BindingLayoutItem::Texture_SRV(0u));
  impl.bindless3DLayout = impl.nvrhiDevice->createBindlessLayout(bindless3D);

  nvrhi::BindlessLayoutDesc bindlessCube{};
  bindlessCube.setVisibility(nvrhi::ShaderType::All)
      .setFirstSlot(0u)
      .setMaxCapacity(kBindlessCapacity)
      .setLayoutType(nvrhi::BindlessLayoutDesc::LayoutType::Immutable)
      .addRegisterSpace(nvrhi::BindingLayoutItem::Texture_SRV(0u));
  impl.bindlessCubeLayout =
      impl.nvrhiDevice->createBindlessLayout(bindlessCube);

  nvrhi::BindlessLayoutDesc bindlessShadow{};
  bindlessShadow.setVisibility(nvrhi::ShaderType::All)
      .setFirstSlot(0u)
      .setMaxCapacity(kBindlessCapacity)
      .setLayoutType(nvrhi::BindlessLayoutDesc::LayoutType::Immutable)
      .addRegisterSpace(nvrhi::BindingLayoutItem::Texture_SRV(0u))
      .addRegisterSpace(nvrhi::BindingLayoutItem::Sampler(1u));
  impl.bindlessShadowLayout =
      impl.nvrhiDevice->createBindlessLayout(bindlessShadow);

  nvrhi::BindingLayoutDesc pushConstants{};
  pushConstants.setVisibility(nvrhi::ShaderType::All)
      .addItem(nvrhi::BindingLayoutItem::PushConstants(
          0u, static_cast<uint32_t>(kPushConstantByteSize)));
  impl.pushConstantsLayout =
      impl.nvrhiDevice->createBindingLayout(pushConstants);

  if (!impl.bindless2DLayout || !impl.bindless3DLayout ||
      !impl.bindlessCubeLayout || !impl.bindlessShadowLayout ||
      !impl.pushConstantsLayout) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::createGlobalBindingLayouts: failed to create binding "
        "layouts");
  }

  impl.bindless2DTable =
      impl.nvrhiDevice->createDescriptorTable(impl.bindless2DLayout);
  impl.bindless3DTable =
      impl.nvrhiDevice->createDescriptorTable(impl.bindless3DLayout);
  impl.bindlessCubeTable =
      impl.nvrhiDevice->createDescriptorTable(impl.bindlessCubeLayout);
  impl.bindlessShadowTable =
      impl.nvrhiDevice->createDescriptorTable(impl.bindlessShadowLayout);
  if (!impl.bindless2DTable || !impl.bindless3DTable ||
      !impl.bindlessCubeTable || !impl.bindlessShadowTable) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::createGlobalBindingLayouts: failed to create "
        "descriptor tables");
  }

  nvrhi::BindingSetDesc pushConstantsSet{};
  pushConstantsSet.addItem(nvrhi::BindingSetItem::PushConstants(
      0u, static_cast<uint32_t>(kPushConstantByteSize)));
  impl.pushConstantsSet = impl.nvrhiDevice->createBindingSet(
      pushConstantsSet, impl.pushConstantsLayout);
  if (!impl.pushConstantsSet) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::createGlobalBindingLayouts: failed to create push "
        "constant binding set");
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] VkSurfaceFormatKHR
chooseSurfaceFormat(std::span<const VkSurfaceFormatKHR> formats) {
  for (const VkSurfaceFormatKHR &format : formats) {
    if (format.format == VK_FORMAT_R8G8B8A8_UNORM &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return format;
    }
  }
  for (const VkSurfaceFormatKHR &format : formats) {
    if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return format;
    }
  }
  return formats.empty() ? VkSurfaceFormatKHR{} : formats.front();
}

[[nodiscard]] SwapchainPresentMode requestedPresentModeFromEnvironment() {
  std::string modeName;
  if (std::optional<std::string> override = readEnvVar("NURI_PRESENT_MODE");
      override.has_value()) {
    modeName = *override;
    std::transform(
        modeName.begin(), modeName.end(), modeName.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  }
  if (modeName == "immediate") {
    return SwapchainPresentMode::Immediate;
  }
  if (modeName == "mailbox") {
    return SwapchainPresentMode::Mailbox;
  }
  if (modeName.empty() || modeName == "default") {
    return SwapchainPresentMode::Mailbox;
  }
  return SwapchainPresentMode::Fifo;
}

[[nodiscard]] VkPresentModeKHR
toVkPresentMode(SwapchainPresentMode mode) noexcept {
  switch (mode) {
  case SwapchainPresentMode::Immediate:
    return VK_PRESENT_MODE_IMMEDIATE_KHR;
  case SwapchainPresentMode::Mailbox:
    return VK_PRESENT_MODE_MAILBOX_KHR;
  case SwapchainPresentMode::Fifo:
  case SwapchainPresentMode::Unknown:
  default:
    return VK_PRESENT_MODE_FIFO_KHR;
  }
}

[[nodiscard]] SwapchainPresentMode
fromVkPresentMode(VkPresentModeKHR mode) noexcept {
  switch (mode) {
  case VK_PRESENT_MODE_IMMEDIATE_KHR:
    return SwapchainPresentMode::Immediate;
  case VK_PRESENT_MODE_MAILBOX_KHR:
    return SwapchainPresentMode::Mailbox;
  case VK_PRESENT_MODE_FIFO_KHR:
    return SwapchainPresentMode::Fifo;
  default:
    return SwapchainPresentMode::Unknown;
  }
}

[[nodiscard]] VkPresentModeKHR
choosePresentMode(std::span<const VkPresentModeKHR> modes,
                  SwapchainPresentMode requestedMode) {
  const VkPresentModeKHR requested = toVkPresentMode(requestedMode);
  if (std::find(modes.begin(), modes.end(), requested) != modes.end()) {
    return requested;
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

[[nodiscard]] VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &caps,
                                          Window &window) {
  if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    return caps.currentExtent;
  }
  int32_t width = 0;
  int32_t height = 0;
  window.getFramebufferSize(width, height);
  return VkExtent2D{
      .width = std::clamp(static_cast<uint32_t>(std::max(width, 1)),
                          caps.minImageExtent.width, caps.maxImageExtent.width),
      .height =
          std::clamp(static_cast<uint32_t>(std::max(height, 1)),
                     caps.minImageExtent.height, caps.maxImageExtent.height),
  };
}

void destroySwapchain(Impl &impl) {
  if (impl.nvrhiDevice) {
    impl.nvrhiDevice->waitForIdle();
  }
  invalidateSwapchainFramebuffers(impl);
  impl.swapchain.images.clear();
  if (impl.swapchain.handle != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(impl.device, impl.swapchain.handle, nullptr);
    impl.swapchain.handle = VK_NULL_HANDLE;
  }
  impl.frameSemaphoreReuseWaitInstances.assign(kSwapchainFramesInFlight, 0u);
  impl.frameResourceReuseWaitInstances.clear();
  impl.swapchainImageReuseWaitInstances.clear();
  impl.preparedSwapchainImageWaitInstance = 0u;
  impl.hasPreparedSwapchainImage = false;
}

[[nodiscard]] Result<bool, std::string> createSwapchain(Impl &impl) {
  if (impl.window == nullptr) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::createSwapchain: window is null");
  }

  VkSurfaceCapabilitiesKHR caps{};
  VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
      impl.physicalDevice, impl.surface, &caps);
  if (result != VK_SUCCESS) {
    return Result<bool, std::string>::makeError(
        vkError("NvrhiGPUDevice::createSwapchain capabilities", result));
  }
  uint32_t formatCount = 0u;
  vkGetPhysicalDeviceSurfaceFormatsKHR(impl.physicalDevice, impl.surface,
                                       &formatCount, nullptr);
  std::vector<VkSurfaceFormatKHR> formats(formatCount);
  if (formatCount != 0u) {
    vkGetPhysicalDeviceSurfaceFormatsKHR(impl.physicalDevice, impl.surface,
                                         &formatCount, formats.data());
  }
  uint32_t presentModeCount = 0u;
  vkGetPhysicalDeviceSurfacePresentModesKHR(impl.physicalDevice, impl.surface,
                                            &presentModeCount, nullptr);
  std::vector<VkPresentModeKHR> modes(presentModeCount);
  if (presentModeCount != 0u) {
    vkGetPhysicalDeviceSurfacePresentModesKHR(impl.physicalDevice, impl.surface,
                                              &presentModeCount, modes.data());
  }

  const VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(formats);
  const VkPresentModeKHR presentMode =
      choosePresentMode(modes, impl.requestedPresentMode);
  const VkExtent2D extent = chooseSwapExtent(caps, *impl.window);
  uint32_t imageCount =
      std::max(caps.minImageCount, kPreferredSwapchainImageCount);
  if (caps.maxImageCount > 0u) {
    imageCount = std::min(imageCount, caps.maxImageCount);
  }

  const VkSwapchainKHR oldSwapchain = impl.swapchain.handle;
  if (oldSwapchain != VK_NULL_HANDLE && impl.nvrhiDevice) {
    impl.nvrhiDevice->waitForIdle();
    collectCompletedAsyncUploadSubmissions(impl);
    releaseAllRetiredResourcesAfterIdle(impl);
    impl.nvrhiDevice->runGarbageCollection();
    collectCompletedGpuTimingSubmissions(impl);
  }
  const VkSwapchainCreateInfoKHR createInfo{
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = impl.surface,
      .minImageCount = imageCount,
      .imageFormat = surfaceFormat.format,
      .imageColorSpace = surfaceFormat.colorSpace,
      .imageExtent = extent,
      .imageArrayLayers = 1u,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .preTransform = caps.currentTransform,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = presentMode,
      .clipped = VK_TRUE,
      .oldSwapchain = oldSwapchain,
  };
  VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
  result =
      vkCreateSwapchainKHR(impl.device, &createInfo, nullptr, &newSwapchain);
  if (result != VK_SUCCESS) {
    return Result<bool, std::string>::makeError(
        vkError("NvrhiGPUDevice::createSwapchain", result));
  }
  if (oldSwapchain != VK_NULL_HANDLE) {
    invalidateSwapchainFramebuffers(impl);
    impl.swapchain.images.clear();
    vkDestroySwapchainKHR(impl.device, oldSwapchain, nullptr);
  }

  uint32_t actualImageCount = 0u;
  result = vkGetSwapchainImagesKHR(impl.device, newSwapchain, &actualImageCount,
                                   nullptr);
  if (result != VK_SUCCESS || actualImageCount == 0u) {
    vkDestroySwapchainKHR(impl.device, newSwapchain, nullptr);
    return Result<bool, std::string>::makeError(
        result != VK_SUCCESS
            ? vkError("NvrhiGPUDevice::createSwapchain get images", result)
            : std::string("NvrhiGPUDevice::createSwapchain: no images"));
  }
  std::vector<VkImage> images(actualImageCount);
  result = vkGetSwapchainImagesKHR(impl.device, newSwapchain, &actualImageCount,
                                   images.data());
  if (result != VK_SUCCESS) {
    vkDestroySwapchainKHR(impl.device, newSwapchain, nullptr);
    return Result<bool, std::string>::makeError(
        vkError("NvrhiGPUDevice::createSwapchain get image list", result));
  }

  impl.swapchain.handle = newSwapchain;
  impl.swapchain.format = surfaceFormat.format;
  impl.swapchain.colorSpace = surfaceFormat.colorSpace;
  impl.swapchain.extent = extent;
  impl.swapchain.images.clear();
  impl.swapchain.images.reserve(actualImageCount);
  impl.frameSemaphoreReuseWaitInstances.assign(kSwapchainFramesInFlight, 0u);
  impl.frameResourceReuseWaitInstances.assign(actualImageCount, 0u);
  impl.swapchainImageReuseWaitInstances.assign(actualImageCount, 0u);
  impl.preparedSwapchainImageWaitInstance = 0u;

  nvrhi::TextureDesc textureDesc{};
  textureDesc.setDimension(nvrhi::TextureDimension::Texture2D)
      .setFormat(toNvrhiSwapchainFormat(surfaceFormat.format))
      .setWidth(extent.width)
      .setHeight(extent.height)
      .setIsRenderTarget(true)
      .setInitialState(nvrhi::ResourceStates::Present)
      .setKeepInitialState(true)
      .setDebugName("Swapchain image");
  for (VkImage image : images) {
    nvrhi::TextureHandle texture =
        impl.nvrhiDevice->createHandleForNativeTexture(
            nvrhi::ObjectTypes::VK_Image,
            nvrhi::Object(reinterpret_cast<uintptr_t>(image)), textureDesc);
    if (!texture) {
      destroySwapchain(impl);
      return Result<bool, std::string>::makeError(
          "NvrhiGPUDevice::createSwapchain: failed to wrap swapchain image");
    }
    impl.swapchain.images.push_back(
        SwapchainImage{.image = image, .texture = texture});
  }

  impl.imageAvailableSemaphores.resize(kSwapchainFramesInFlight,
                                       VK_NULL_HANDLE);
  impl.renderFinishedSemaphores.resize(kSwapchainFramesInFlight,
                                       VK_NULL_HANDLE);
  for (uint32_t i = 0u; i < kSwapchainFramesInFlight; ++i) {
    const VkSemaphoreCreateInfo semaphoreInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    if (impl.imageAvailableSemaphores[i] == VK_NULL_HANDLE) {
      result = vkCreateSemaphore(impl.device, &semaphoreInfo, nullptr,
                                 &impl.imageAvailableSemaphores[i]);
      if (result != VK_SUCCESS) {
        return Result<bool, std::string>::makeError(
            vkError("NvrhiGPUDevice::createSwapchain image semaphore", result));
      }
    }
    if (impl.renderFinishedSemaphores[i] == VK_NULL_HANDLE) {
      result = vkCreateSemaphore(impl.device, &semaphoreInfo, nullptr,
                                 &impl.renderFinishedSemaphores[i]);
      if (result != VK_SUCCESS) {
        return Result<bool, std::string>::makeError(vkError(
            "NvrhiGPUDevice::createSwapchain render semaphore", result));
      }
    }
  }

  const bool mailboxAvailable =
      std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) !=
      modes.end();
  NURI_LOG_INFO(
      "NvrhiGPUDevice: swapchain %ux%u images=%u requestedPresentMode=%u "
      "presentMode=%u mailboxAvailable=%s",
      extent.width, extent.height, actualImageCount,
      static_cast<unsigned>(toVkPresentMode(impl.requestedPresentMode)),
      static_cast<unsigned>(presentMode), mailboxAvailable ? "true" : "false");
  impl.activePresentMode = fromVkPresentMode(presentMode);
  return Result<bool, std::string>::makeResult(true);
}

void destroyVulkan(Impl &impl) {
  if (impl.nvrhiDevice) {
    flushPendingAsyncUploadCommandList(impl);
    impl.nvrhiDevice->waitForIdle();
    collectCompletedAsyncUploadSubmissions(impl);
    releaseAllRetiredResourcesAfterIdle(impl);
    impl.nvrhiDevice->runGarbageCollection();
    collectCompletedGpuTimingSubmissions(impl);
  }
  for (PendingGpuTimingSubmission &pending : impl.pendingGpuTimingSubmissions) {
    destroyWholeFrameTimingSlot(impl, pending.wholeFrameTiming);
  }
  for (NvrhiWholeFrameTimingSlot &slot : impl.availableWholeFrameTimingSlots) {
    destroyWholeFrameTimingSlot(impl, slot);
  }
  impl.availableWholeFrameTimingSlots.clear();
  impl.pendingGpuTimingSubmissions.clear();
  impl.pendingAsyncUploadCommandList = nullptr;
  impl.pendingAsyncUploadSubmissions.clear();
  impl.availableAsyncUploadCommandLists.clear();
  impl.activeGraphicsContexts.clear();
  impl.recordedGraphicsCommandBuffers.clear();
  impl.pendingGraphicsCommandLists.clear();
  impl.availableGraphicsCommandLists.clear();
  impl.submissions.clear();
  impl.recordingContextSlots.clear();
  impl.recordedCommandBufferSlots.clear();
  impl.submissionSlots.clear();
  // Cached framebuffer handles retain their attachment textures. Release them
  // before either the texture table or swapchain image wrappers are torn down.
  impl.cachedFramebuffers.clear();
  impl.samplers.clear();
  impl.buffers.clear();
  impl.textures.clear();
  impl.shaders.clear();
  impl.renderPipelines.clear();
  impl.computePipelines.clear();
  impl.meshletPipelines.clear();
  impl.framebufferTextures.clear();
  destroySwapchain(impl);
  impl.immediateCommandList = nullptr;
  impl.bindless2DTable = nullptr;
  impl.bindless3DTable = nullptr;
  impl.bindlessCubeTable = nullptr;
  impl.bindlessShadowTable = nullptr;
  impl.pushConstantsSet = nullptr;
  impl.bindless2DLayout = nullptr;
  impl.bindless3DLayout = nullptr;
  impl.bindlessCubeLayout = nullptr;
  impl.bindlessShadowLayout = nullptr;
  impl.pushConstantsLayout = nullptr;
  impl.nvrhiDevice = nullptr;
  for (VkSemaphore semaphore : impl.imageAvailableSemaphores) {
    if (semaphore != VK_NULL_HANDLE) {
      vkDestroySemaphore(impl.device, semaphore, nullptr);
    }
  }
  for (VkSemaphore semaphore : impl.renderFinishedSemaphores) {
    if (semaphore != VK_NULL_HANDLE) {
      vkDestroySemaphore(impl.device, semaphore, nullptr);
    }
  }
  impl.imageAvailableSemaphores.clear();
  impl.renderFinishedSemaphores.clear();
  impl.frameSemaphoreReuseWaitInstances.clear();
  impl.frameResourceReuseWaitInstances.clear();
  impl.swapchainImageReuseWaitInstances.clear();
  impl.preparedSwapchainImageWaitInstance = 0u;
  if (impl.device != VK_NULL_HANDLE) {
    vkDestroyDevice(impl.device, nullptr);
    impl.device = VK_NULL_HANDLE;
  }
  if (impl.surface != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(impl.instance, impl.surface, nullptr);
    impl.surface = VK_NULL_HANDLE;
  }
  destroyDebugMessenger(impl);
  if (impl.instance != VK_NULL_HANDLE) {
    vkDestroyInstance(impl.instance, nullptr);
    impl.instance = VK_NULL_HANDLE;
  }
  if (impl.glslangInitialized) {
    glslang_finalize_process();
    impl.glslangInitialized = false;
  }
}

template <typename ImplType>
using ActiveContextPtr =
    std::conditional_t<std::is_const_v<std::remove_reference_t<ImplType>>,
                       const ActiveGraphicsRecordingContext *,
                       ActiveGraphicsRecordingContext *>;

template <typename ImplType>
[[nodiscard]] ActiveContextPtr<ImplType>
findActiveGraphicsContextSlot(ImplType &impl, RecordingContextHandle handle) {
  if (handle.index >= impl.activeGraphicsContexts.size() ||
      !impl.recordingContextSlots.isValid(handle.index, handle.generation)) {
    return nullptr;
  }
  auto &entry = impl.activeGraphicsContexts[handle.index];
  if (!areSameHandle(entry.handle, handle)) {
    return nullptr;
  }
  return &entry;
}

[[nodiscard]] Result<SubmissionHandle, std::string>
allocateSubmissionHandle(Impl &impl, uint64_t graphicsInstance,
                         uint64_t copyInstance = 0u,
                         uint64_t requiredRecordingSerial = 0u,
                         bool requiresGraphicsVisibility = false) {
  const SlotReservation slot = impl.submissionSlots.acquire();
  SubmissionHandle handle{
      .index = slot.index,
      .generation = slot.generation,
  };
  impl.submissions.push_back(SubmissionRecord{
      .handle = handle,
      .graphicsInstance = graphicsInstance,
      .copyInstance = copyInstance,
      .requiredRecordingSerial = requiredRecordingSerial,
      .requiresGraphicsVisibility = requiresGraphicsVisibility,
  });
  return Result<SubmissionHandle, std::string>::makeResult(handle);
}

[[nodiscard]] std::optional<uint64_t>
resolveSubmissionInstance(const Impl &impl, const SubmissionRecord &record) {
  if (record.requiredRecordingSerial == 0u) {
    return record.graphicsInstance;
  }
  return impl.recordingRetirement.tryResolveLastUse(
      record.requiredRecordingSerial, record.graphicsInstance);
}

void collectCompletedSubmissionRecords(Impl &impl, uint64_t completedGraphics,
                                       uint64_t completedCopy) {
  if (impl.submissions.empty()) {
    return;
  }
  size_t writeIndex = 0u;
  for (size_t readIndex = 0u; readIndex < impl.submissions.size();
       ++readIndex) {
    SubmissionRecord &record = impl.submissions[readIndex];
    const std::optional<uint64_t> resolvedGraphics =
        resolveSubmissionInstance(impl, record);
    if (resolvedGraphics.has_value() &&
        *resolvedGraphics <= completedGraphics &&
        record.copyInstance <= completedCopy &&
        (record.copyInstance == 0u || !record.requiresGraphicsVisibility ||
         record.graphicsVisibilityQueued)) {
      impl.submissionSlots.release(record.handle.index);
      continue;
    }
    if (writeIndex != readIndex) {
      impl.submissions[writeIndex] = std::move(record);
    }
    ++writeIndex;
  }
  impl.submissions.resize(writeIndex);
}

void collectCompletedGraphicsCommandLists(Impl &impl, uint64_t completed) {
  size_t writeIndex = 0u;
  for (size_t readIndex = 0u;
       readIndex < impl.pendingGraphicsCommandLists.size(); ++readIndex) {
    PendingGraphicsCommandList &pending =
        impl.pendingGraphicsCommandLists[readIndex];
    if (pending.submissionInstance <= completed) {
      pending.framebuffers.clear();
      impl.availableGraphicsCommandLists.push_back(
          std::move(pending.commandList));
      continue;
    }
    if (writeIndex != readIndex) {
      impl.pendingGraphicsCommandLists[writeIndex] = std::move(pending);
    }
    ++writeIndex;
  }
  impl.pendingGraphicsCommandLists.resize(writeIndex);
}

[[nodiscard]] bool textureUsageIsUAV(TextureUsage usage) {
  return usage == TextureUsage::Storage ||
         usage == TextureUsage::StorageSampled;
}

[[nodiscard]] bool textureUsageIsShaderResource(TextureUsage usage,
                                                Format format) {
  return usage == TextureUsage::Sampled ||
         usage == TextureUsage::AttachmentSampled ||
         usage == TextureUsage::InputAttachment ||
         usage == TextureUsage::StorageSampled || isDepthFormat(format);
}

[[nodiscard]] bool textureUsageNeedsDescriptor(TextureUsage usage,
                                               Format format) {
  return textureUsageIsShaderResource(usage, format) ||
         textureUsageIsUAV(usage);
}

[[nodiscard]] nvrhi::DescriptorTableHandle
descriptorTableForTexture(Impl &impl, const TextureResource &texture) {
  if (texture.desc.type == TextureType::Texture3D) {
    return impl.bindless3DTable;
  }
  if (texture.desc.type == TextureType::TextureCube) {
    return impl.bindlessCubeTable;
  }
  if (isDepthFormat(texture.format)) {
    return impl.bindlessShadowTable;
  }
  return impl.bindless2DTable;
}

[[nodiscard]] nvrhi::BindingSetItem
descriptorItemForTexture(uint32_t slot, const TextureResource &texture) {
  if (texture.desc.usage == TextureUsage::Storage ||
      texture.desc.usage == TextureUsage::StorageSampled) {
    return nvrhi::BindingSetItem::Texture_UAV(
        slot, texture.texture, toNvrhiFormat(texture.format),
        nvrhi::TextureSubresourceSet(
            0u, 1u, 0u, nvrhi::TextureSubresourceSet::AllArraySlices),
        texture.dimension);
  }
  return nvrhi::BindingSetItem::Texture_SRV(
      slot, texture.texture, toNvrhiFormat(texture.format),
      nvrhi::AllSubresources, texture.dimension);
}

[[nodiscard]] bool writeTextureDescriptor(Impl &impl, uint32_t slot,
                                          const TextureResource &texture) {
  if (!textureUsageNeedsDescriptor(texture.desc.usage, texture.format)) {
    return true;
  }
  if (isDepthFormat(texture.format)) {
    const nvrhi::BindingSetItem item = descriptorItemForTexture(slot, texture);
    return impl.nvrhiDevice->writeDescriptorTable(impl.bindlessShadowTable,
                                                  item) &&
           impl.nvrhiDevice->writeDescriptorTable(impl.bindless2DTable, item);
  }
  if (texture.desc.usage == TextureUsage::StorageSampled) {
    const nvrhi::BindingSetItem srv = nvrhi::BindingSetItem::Texture_SRV(
        slot, texture.texture, toNvrhiFormat(texture.format),
        nvrhi::AllSubresources, texture.dimension);
    const nvrhi::BindingSetItem uav = nvrhi::BindingSetItem::Texture_UAV(
        slot, texture.texture, toNvrhiFormat(texture.format),
        nvrhi::TextureSubresourceSet(
            0u, 1u, 0u, nvrhi::TextureSubresourceSet::AllArraySlices),
        texture.dimension);
    return impl.nvrhiDevice->writeDescriptorTable(impl.bindless2DTable, srv) &&
           impl.nvrhiDevice->writeDescriptorTable(impl.bindless2DTable, uav);
  }
  nvrhi::DescriptorTableHandle table = descriptorTableForTexture(impl, texture);
  if (!table) {
    return false;
  }
  return impl.nvrhiDevice->writeDescriptorTable(
      table, descriptorItemForTexture(slot, texture));
}

[[nodiscard]] bool writeSamplerDescriptor(Impl &impl, uint32_t slot,
                                          nvrhi::SamplerHandle sampler,
                                          bool shadow) {
  if (shadow) {
    return impl.nvrhiDevice->writeDescriptorTable(
        impl.bindlessShadowTable,
        nvrhi::BindingSetItem::Sampler(slot, sampler));
  }
  return impl.nvrhiDevice->writeDescriptorTable(
      impl.bindless2DTable, nvrhi::BindingSetItem::Sampler(slot, sampler));
}

[[nodiscard]] SamplerHandle
findBestAnisotropicSampler(std::span<const SamplerHandle> samplers,
                           uint8_t maxAnisotropy) {
  SamplerHandle best{};
  uint8_t bestLevel = 0u;
  for (size_t i = 0u;
       i < samplers.size() && i < kSupportedAnisotropyLevels.size(); ++i) {
    const uint8_t level = kSupportedAnisotropyLevels[i];
    if (level <= maxAnisotropy && level >= bestLevel &&
        nuri::isValid(samplers[i])) {
      best = samplers[i];
      bestLevel = level;
    }
  }
  return best;
}

[[nodiscard]] nvrhi::SamplerDesc toNvrhiSamplerDesc(const SamplerDesc &desc,
                                                    const Impl &impl) {
  nvrhi::SamplerDesc sampler{};
  sampler.setMinFilter(desc.minFilter == SamplerFilter::Linear)
      .setMagFilter(desc.magFilter == SamplerFilter::Linear)
      .setMipFilter(desc.mipMode == SamplerMipMode::Linear)
      .setAddressU(toNvrhiAddressMode(desc.wrapU))
      .setAddressV(toNvrhiAddressMode(desc.wrapV))
      .setAddressW(toNvrhiAddressMode(desc.wrapW))
      .setMipBias(std::clamp(desc.mipLodBias, -impl.maxSamplerLodBias,
                             impl.maxSamplerLodBias))
      .setMaxAnisotropy(static_cast<float>(std::max(
          1u, std::min(static_cast<uint32_t>(desc.maxAnisotropy),
                       static_cast<uint32_t>(impl.maxSamplerAnisotropy)))));
  if (desc.depthCompareEnabled) {
    sampler.setReductionType(nvrhi::SamplerReductionType::Comparison);
  }
  return sampler;
}

[[nodiscard]] bool textureUsageIsRenderTarget(TextureUsage usage,
                                              Format format) {
  return usage == TextureUsage::Attachment ||
         usage == TextureUsage::AttachmentSampled ||
         usage == TextureUsage::InputAttachment || isDepthFormat(format);
}

[[nodiscard]] nvrhi::ResourceStates
initialTextureState(const TextureDesc &desc) {
  if (isDepthFormat(desc.format)) {
    return nvrhi::ResourceStates::DepthRead |
           nvrhi::ResourceStates::ShaderResource;
  }
  if (desc.usage == TextureUsage::Attachment) {
    return nvrhi::ResourceStates::RenderTarget;
  }
  if (desc.usage == TextureUsage::Storage) {
    return nvrhi::ResourceStates::UnorderedAccess;
  }
  return nvrhi::ResourceStates::ShaderResource;
}

[[nodiscard]] nvrhi::TextureDesc
toNvrhiTextureDesc(const TextureDesc &desc, std::string_view debugName) {
  const uint32_t samples = std::max(desc.numSamples, 1u);
  const uint32_t layers = std::max(desc.numLayers, 1u);
  const uint32_t arraySize =
      desc.type == TextureType::TextureCube ? layers * 6u : layers;
  nvrhi::TextureDesc nvrhiDesc{};
  nvrhiDesc.setDimension(toNvrhiTextureDimension(desc.type, samples, layers))
      .setFormat(toNvrhiFormat(desc.format))
      .setWidth(std::max(desc.dimensions.width, 1u))
      .setHeight(std::max(desc.dimensions.height, 1u))
      .setDepth(std::max(desc.dimensions.depth, 1u))
      .setArraySize(arraySize)
      .setSampleCount(samples)
      .setMipLevels(std::max(desc.numMipLevels, 1u))
      .setIsRenderTarget(textureUsageIsRenderTarget(desc.usage, desc.format))
      .setIsUAV(textureUsageIsUAV(desc.usage))
      .setInitialState(initialTextureState(desc))
      .setKeepInitialState(true)
      .setDebugName(std::string(debugName));
  nvrhiDesc.isShaderResource =
      textureUsageIsShaderResource(desc.usage, desc.format);
  return nvrhiDesc;
}

[[nodiscard]] size_t textureRowPitch(Format format, uint32_t width) {
  const uint32_t block = blockExtent(format);
  const size_t blockBytes = bytesPerBlock(format);
  return ((static_cast<size_t>(width) + block - 1u) / block) * blockBytes;
}

[[nodiscard]] size_t textureDepthPitch(Format format, uint32_t width,
                                       uint32_t height) {
  const uint32_t block = blockExtent(format);
  const size_t rows = (static_cast<size_t>(height) + block - 1u) / block;
  return textureRowPitch(format, width) * rows;
}

[[nodiscard]] bool canGenerateCpuRgbaMipmaps(const TextureDesc &desc) {
  return desc.type == TextureType::Texture2D &&
         (desc.format == Format::RGBA8_UNORM ||
          desc.format == Format::RGBA8_SRGB ||
          desc.format == Format::RGBA16_FLOAT) &&
         desc.dataNumMipLevels == 1u && desc.numMipLevels > 1u &&
         desc.generateMipmaps;
}

[[nodiscard]] std::vector<std::byte>
generateNextRgba8Mip(std::span<const std::byte> src, uint32_t srcWidth,
                     uint32_t srcHeight) {
  const uint32_t dstWidth = std::max(1u, srcWidth >> 1u);
  const uint32_t dstHeight = std::max(1u, srcHeight >> 1u);
  std::vector<std::byte> dst(static_cast<size_t>(dstWidth) * dstHeight * 4u);
  for (uint32_t y = 0u; y < dstHeight; ++y) {
    for (uint32_t x = 0u; x < dstWidth; ++x) {
      std::array<uint32_t, 4> sum{};
      uint32_t count = 0u;
      for (uint32_t dy = 0u; dy < 2u; ++dy) {
        const uint32_t sy = y * 2u + dy;
        if (sy >= srcHeight) {
          continue;
        }
        for (uint32_t dx = 0u; dx < 2u; ++dx) {
          const uint32_t sx = x * 2u + dx;
          if (sx >= srcWidth) {
            continue;
          }
          const size_t srcOffset =
              (static_cast<size_t>(sy) * srcWidth + sx) * 4u;
          for (uint32_t c = 0u; c < 4u; ++c) {
            sum[c] += std::to_integer<uint32_t>(src[srcOffset + c]);
          }
          ++count;
        }
      }
      const size_t dstOffset = (static_cast<size_t>(y) * dstWidth + x) * 4u;
      for (uint32_t c = 0u; c < 4u; ++c) {
        dst[dstOffset + c] =
            static_cast<std::byte>((sum[c] + count / 2u) / count);
      }
    }
  }
  return dst;
}

[[nodiscard]] uint16_t loadU16(std::span<const std::byte> src, size_t offset) {
  uint16_t value = 0u;
  std::memcpy(&value, src.data() + offset, sizeof(value));
  return value;
}

void storeU16(std::vector<std::byte> &dst, size_t offset, uint16_t value) {
  std::memcpy(dst.data() + offset, &value, sizeof(value));
}

[[nodiscard]] std::vector<std::byte>
generateNextRgba16FMip(std::span<const std::byte> src, uint32_t srcWidth,
                       uint32_t srcHeight) {
  const uint32_t dstWidth = std::max(1u, srcWidth >> 1u);
  const uint32_t dstHeight = std::max(1u, srcHeight >> 1u);
  std::vector<std::byte> dst(static_cast<size_t>(dstWidth) * dstHeight * 8u);
  for (uint32_t y = 0u; y < dstHeight; ++y) {
    for (uint32_t x = 0u; x < dstWidth; ++x) {
      std::array<float, 4> sum{};
      uint32_t count = 0u;
      for (uint32_t dy = 0u; dy < 2u; ++dy) {
        const uint32_t sy = y * 2u + dy;
        if (sy >= srcHeight) {
          continue;
        }
        for (uint32_t dx = 0u; dx < 2u; ++dx) {
          const uint32_t sx = x * 2u + dx;
          if (sx >= srcWidth) {
            continue;
          }
          const size_t srcOffset =
              (static_cast<size_t>(sy) * srcWidth + sx) * 8u;
          for (uint32_t c = 0u; c < 4u; ++c) {
            sum[c] += glm::unpackHalf1x16(
                loadU16(src, srcOffset + c * sizeof(uint16_t)));
          }
          ++count;
        }
      }
      const float invCount = 1.0f / static_cast<float>(count);
      const size_t dstOffset = (static_cast<size_t>(y) * dstWidth + x) * 8u;
      for (uint32_t c = 0u; c < 4u; ++c) {
        storeU16(dst, dstOffset + c * sizeof(uint16_t),
                 glm::packHalf1x16(sum[c] * invCount));
      }
    }
  }
  return dst;
}

[[nodiscard]] Result<std::vector<std::byte>, std::string>
generateRgbaMipChain(const TextureDesc &desc, uint32_t layers, uint32_t faces) {
  const uint32_t width = std::max(desc.dimensions.width, 1u);
  const uint32_t height = std::max(desc.dimensions.height, 1u);
  const uint32_t sliceCount = layers * faces;
  const size_t baseSliceBytes =
      textureMipByteSize(desc.format, width, height, 1u);
  const size_t requiredBaseBytes = baseSliceBytes * sliceCount;
  if (desc.data.size() < requiredBaseBytes) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "NvrhiGPUDevice::generateRgbaMipChain: base mip data is truncated");
  }

  std::vector<std::vector<std::byte>> currentSlices(sliceCount);
  std::vector<std::byte> mipChain;
  size_t totalBytes = 0u;
  for (uint32_t mip = 0u; mip < desc.numMipLevels; ++mip) {
    totalBytes += textureMipByteSize(desc.format, mipSize(width, mip),
                                     mipSize(height, mip), 1u) *
                  sliceCount;
  }
  mipChain.reserve(totalBytes);

  for (uint32_t slice = 0u; slice < sliceCount; ++slice) {
    const std::byte *src = desc.data.data() + slice * baseSliceBytes;
    currentSlices[slice].assign(src, src + baseSliceBytes);
  }

  uint32_t srcWidth = width;
  uint32_t srcHeight = height;
  for (uint32_t mip = 0u; mip < desc.numMipLevels; ++mip) {
    for (const std::vector<std::byte> &slice : currentSlices) {
      mipChain.insert(mipChain.end(), slice.begin(), slice.end());
    }
    if (mip + 1u == desc.numMipLevels) {
      break;
    }
    std::vector<std::vector<std::byte>> nextSlices(sliceCount);
    for (uint32_t slice = 0u; slice < sliceCount; ++slice) {
      if (desc.format == Format::RGBA16_FLOAT) {
        nextSlices[slice] =
            generateNextRgba16FMip(currentSlices[slice], srcWidth, srcHeight);
      } else {
        nextSlices[slice] =
            generateNextRgba8Mip(currentSlices[slice], srcWidth, srcHeight);
      }
    }
    currentSlices = std::move(nextSlices);
    srcWidth = std::max(1u, srcWidth >> 1u);
    srcHeight = std::max(1u, srcHeight >> 1u);
  }

  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(mipChain));
}

void executeCommandListAndWait(Impl &impl,
                               const nvrhi::CommandListHandle &commandList) {
  nvrhi::ICommandList *raw = commandList.Get();
  impl.nvrhiDevice->executeCommandLists(&raw, 1u,
                                        nvrhi::CommandQueue::Graphics);
  impl.nvrhiDevice->waitForIdle();
  impl.nvrhiDevice->runGarbageCollection();
}

constexpr uint64_t kMaxAsyncTextureUploadBatchBytes =
    256ull * 1024ull * 1024ull;
constexpr uint32_t kMaxAsyncTextureUploadBatchTextures = 64u;

[[nodiscard]] Result<bool, std::string>
uploadTextureData(Impl &impl, const TextureDesc &desc,
                  nvrhi::ITexture *texture) {
  if (desc.data.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (texture == nullptr) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::uploadTextureData: texture is null");
  }

  const uint32_t mipCount = std::min(std::max(desc.dataNumMipLevels, 1u),
                                     std::max(desc.numMipLevels, 1u));
  const uint32_t layers =
      desc.type == TextureType::Texture3D ? 1u : std::max(desc.numLayers, 1u);
  const uint32_t faces = desc.type == TextureType::TextureCube ? 6u : 1u;
  std::vector<std::byte> generatedMipData;
  std::span<const std::byte> uploadData = desc.data;
  uint32_t uploadMipCount = mipCount;
  if (canGenerateCpuRgbaMipmaps(desc)) {
    auto generated = generateRgbaMipChain(desc, layers, faces);
    if (generated.hasError()) {
      return Result<bool, std::string>::makeError(generated.error());
    }
    generatedMipData = std::move(generated).value();
    uploadData = generatedMipData;
    uploadMipCount = std::max(desc.numMipLevels, 1u);
  }

  uint64_t requiredUploadBytes = 0u;
  for (uint32_t mip = 0u; mip < uploadMipCount; ++mip) {
    const uint32_t width = mipSize(desc.dimensions.width, mip);
    const uint32_t height = mipSize(desc.dimensions.height, mip);
    const uint32_t depth = desc.type == TextureType::Texture3D
                               ? mipSize(desc.dimensions.depth, mip)
                               : 1u;
    const size_t mipBytes =
        textureMipByteSize(desc.format, width, height, depth);
    if (mipBytes == 0u) {
      return Result<bool, std::string>::makeError(
          "NvrhiGPUDevice::uploadTextureData: texture format has no upload "
          "size");
    }
    requiredUploadBytes += static_cast<uint64_t>(mipBytes) *
                           static_cast<uint64_t>(layers) *
                           static_cast<uint64_t>(faces);
  }
  if (requiredUploadBytes > uploadData.size()) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::uploadTextureData: texture data is truncated");
  }

  std::lock_guard lock(impl.immediateMutex);
  collectCompletedAsyncUploadSubmissions(impl);
  const bool batchWouldOverflowBytes =
      impl.pendingAsyncUploadBytes != 0u &&
      (requiredUploadBytes > kMaxAsyncTextureUploadBatchBytes ||
       impl.pendingAsyncUploadBytes >
           kMaxAsyncTextureUploadBatchBytes -
               std::min(requiredUploadBytes, kMaxAsyncTextureUploadBatchBytes));
  if (impl.pendingAsyncUploadCommandList &&
      (batchWouldOverflowBytes || impl.pendingAsyncUploadTextureCount >=
                                      kMaxAsyncTextureUploadBatchTextures)) {
    ++impl.textureUploadBoundedBatchFlushes;
    flushPendingAsyncUploadCommandList(impl);
  }
  nvrhi::CommandListHandle &commandList =
      ensurePendingAsyncUploadCommandList(impl);
  if (!commandList) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::uploadTextureData: failed to create upload command "
        "list");
  }

  size_t offset = 0u;
  for (uint32_t mip = 0u; mip < uploadMipCount; ++mip) {
    for (uint32_t layer = 0u; layer < layers; ++layer) {
      for (uint32_t face = 0u; face < faces; ++face) {
        const uint32_t width = mipSize(desc.dimensions.width, mip);
        const uint32_t height = mipSize(desc.dimensions.height, mip);
        const uint32_t depth = desc.type == TextureType::Texture3D
                                   ? mipSize(desc.dimensions.depth, mip)
                                   : 1u;
        const size_t mipBytes =
            textureMipByteSize(desc.format, width, height, depth);
        const uint32_t arraySlice = layer * faces + face;
        commandList->writeTexture(
            texture, arraySlice, mip, uploadData.data() + offset,
            textureRowPitch(desc.format, width),
            textureDepthPitch(desc.format, width, height));
        offset += mipBytes;
      }
    }
  }
  impl.pendingAsyncUploadBytes += requiredUploadBytes;
  ++impl.pendingAsyncUploadTextureCount;
  impl.textureUploadBytesRecorded += requiredUploadBytes;
  ++impl.textureUploadTexturesRecorded;
  impl.trimAsyncUploadCommandListPoolAfterTextureUploads = true;

  if (desc.generateMipmaps && desc.numMipLevels > uploadMipCount) {
    NURI_LOG_WARNING(
        "NvrhiGPUDevice::uploadTextureData: generateMipmaps is not "
        "implemented yet; uploaded %u/%u mip levels",
        uploadMipCount, std::max(desc.numMipLevels, 1u));
  }

  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<TextureResource, std::string>
createTextureResource(Impl &impl, const TextureDesc &desc,
                      std::string_view debugName) {
  if (desc.dimensions.width == 0u || desc.dimensions.height == 0u) {
    return Result<TextureResource, std::string>::makeError(
        "Texture dimensions cannot be zero");
  }
  if (toNvrhiFormat(desc.format) == nvrhi::Format::UNKNOWN) {
    return Result<TextureResource, std::string>::makeError(
        "Unsupported texture format");
  }

  nvrhi::TextureDesc nvrhiDesc = toNvrhiTextureDesc(desc, debugName);
  nvrhi::TextureHandle texture = impl.nvrhiDevice->createTexture(nvrhiDesc);
  if (!texture) {
    return Result<TextureResource, std::string>::makeError(
        "Failed to create NVRHI texture");
  }

  auto uploadResult = uploadTextureData(impl, desc, texture.Get());
  if (uploadResult.hasError()) {
    return Result<TextureResource, std::string>::makeError(
        uploadResult.error());
  }

  TextureDesc storedDesc = desc;
  storedDesc.data = {};
  return Result<TextureResource, std::string>::makeResult(TextureResource{
      .texture = texture,
      .debugName = std::string(debugName),
      .format = desc.format,
      .desc = storedDesc,
      .dimension = nvrhiDesc.dimension,
  });
}

[[nodiscard]] Result<nvrhi::ShaderHandle, std::string>
specializeShader(Impl &impl, const nvrhi::ShaderHandle &shader,
                 const SpecializationInfo &specInfo, std::string_view context) {
  if (specInfo.entries.empty()) {
    return Result<nvrhi::ShaderHandle, std::string>::makeResult(shader);
  }
  if (specInfo.data == nullptr || specInfo.dataSize == 0u) {
    return Result<nvrhi::ShaderHandle, std::string>::makeError(
        std::string(context) + ": specialization data is empty");
  }

  std::vector<nvrhi::ShaderSpecialization> constants;
  constants.reserve(specInfo.entries.size());
  for (const SpecializationEntry &entry : specInfo.entries) {
    if (entry.size != sizeof(uint32_t) ||
        static_cast<size_t>(entry.offset) + entry.size > specInfo.dataSize) {
      return Result<nvrhi::ShaderHandle, std::string>::makeError(
          std::string(context) +
          ": specialization entries must be 4-byte constants within data");
    }
    uint32_t value = 0u;
    std::memcpy(&value,
                static_cast<const std::byte *>(specInfo.data) + entry.offset,
                sizeof(value));
    constants.push_back(
        nvrhi::ShaderSpecialization::UInt32(entry.constantId, value));
  }
  nvrhi::ShaderHandle specialized =
      impl.nvrhiDevice->createShaderSpecialization(
          shader.Get(), constants.data(),
          static_cast<uint32_t>(constants.size()));
  if (!specialized) {
    return Result<nvrhi::ShaderHandle, std::string>::makeError(
        std::string(context) + ": failed to create shader specialization");
  }
  return Result<nvrhi::ShaderHandle, std::string>::makeResult(specialized);
}

[[nodiscard]] bool
makePaddedPushConstants(std::span<const std::byte> source,
                        std::array<std::byte, kPushConstantByteSize> &out) {
  if (source.size() > out.size()) {
    return false;
  }
  out.fill(std::byte{0});
  if (!source.empty()) {
    std::memcpy(out.data(), source.data(), source.size());
  }
  return true;
}

void bindGlobalSets(nvrhi::GraphicsState &state, Impl &impl) {
  state.bindings = globalBindingSets(impl);
}

void bindGlobalSets(nvrhi::ComputeState &state, Impl &impl) {
  state.bindings = globalBindingSets(impl);
}

void bindGlobalSets(nvrhi::MeshletState &state, Impl &impl) {
  state.bindings = globalBindingSets(impl);
}

[[nodiscard]] PipelineVariantKey makePipelineVariantKey(const DrawItem &draw) {
  return makeRasterPipelineState(
      draw.useDepthState ? draw.depthState : DepthState{}, draw.depthBiasEnable,
      draw.depthBiasConstant, draw.depthBiasSlope, draw.depthBiasClamp);
}

[[nodiscard]] PipelineVariantKey
makePipelineVariantKey(RasterPipelineState rasterState) {
  return canonicalRasterPipelineState(rasterState);
}

[[nodiscard]] PipelineVariantKey
makePipelineVariantKey(const MeshDispatchItem &dispatch) {
  return makeRasterPipelineState(
      dispatch.useDepthState ? dispatch.depthState : DepthState{},
      dispatch.depthBiasEnable, dispatch.depthBiasConstant,
      dispatch.depthBiasSlope, dispatch.depthBiasClamp);
}

[[nodiscard]] std::optional<std::string>
createGraphicsPipelineVariant(Impl &impl, RenderPipelineResource &pipeline,
                              PipelineVariantKey key,
                              nvrhi::IGraphicsPipeline *&outPipeline) {
  key = canonicalRasterPipelineState(key);
  if (auto found = pipeline.variants.find(key);
      found != pipeline.variants.end()) {
    outPipeline = found->second.Get();
    return std::nullopt;
  }
  nvrhi::GraphicsPipelineDesc desc = pipeline.baseDesc;
  desc.renderState.depthStencilState.setDepthFunc(
      toNvrhiCompareOp(key.compareOp));
  desc.renderState.depthStencilState.setDepthWriteEnable(key.depthWrite);
  desc.renderState.rasterState.setDepthBias(
      key.depthBiasEnable ? key.depthBiasConstant : 0);
  desc.renderState.rasterState.setSlopeScaleDepthBias(
      key.depthBiasEnable ? key.depthBiasSlope : 0.0f);
  desc.renderState.rasterState.setDepthBiasClamp(
      key.depthBiasEnable ? key.depthBiasClamp : 0.0f);
  nvrhi::GraphicsPipelineHandle handle =
      impl.nvrhiDevice->createGraphicsPipeline(desc, pipeline.framebufferInfo);
  if (!handle) {
    return std::string("Failed to create graphics pipeline variant");
  }
  auto [it, _] = pipeline.variants.emplace(key, std::move(handle));
  outPipeline = it->second.Get();
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string>
findGraphicsPipelineVariant(Impl &impl, RenderPipelineResource &pipeline,
                            const DrawItem &draw,
                            nvrhi::IGraphicsPipeline *&outPipeline) {
  const PipelineVariantKey key = makePipelineVariantKey(draw);
  if (auto found = pipeline.variants.find(key);
      found != pipeline.variants.end()) {
    outPipeline = found->second.Get();
    return std::nullopt;
  }
  ++impl.undeclaredGraphicsPipelineVariantMisses;
  return std::string("Graphics pipeline raster variant was not declared during "
                     "pipeline preparation: ") +
         pipeline.debugName;
}

[[nodiscard]] std::optional<std::string>
createMeshletPipelineVariant(Impl &impl, MeshletPipelineResource &pipeline,
                             PipelineVariantKey key,
                             nvrhi::IMeshletPipeline *&outPipeline) {
  key = canonicalRasterPipelineState(key);
  if (auto found = pipeline.variants.find(key);
      found != pipeline.variants.end()) {
    outPipeline = found->second.Get();
    return std::nullopt;
  }
  nvrhi::MeshletPipelineDesc desc = pipeline.baseDesc;
  desc.renderState.depthStencilState.setDepthFunc(
      toNvrhiCompareOp(key.compareOp));
  desc.renderState.depthStencilState.setDepthWriteEnable(key.depthWrite);
  desc.renderState.rasterState.setDepthBias(
      key.depthBiasEnable ? key.depthBiasConstant : 0);
  desc.renderState.rasterState.setSlopeScaleDepthBias(
      key.depthBiasEnable ? key.depthBiasSlope : 0.0f);
  desc.renderState.rasterState.setDepthBiasClamp(
      key.depthBiasEnable ? key.depthBiasClamp : 0.0f);
  nvrhi::MeshletPipelineHandle handle =
      impl.nvrhiDevice->createMeshletPipeline(desc, pipeline.framebufferInfo);
  if (!handle) {
    return std::string("Failed to create meshlet pipeline variant");
  }
  auto [it, _] = pipeline.variants.emplace(key, std::move(handle));
  outPipeline = it->second.Get();
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string>
findMeshletPipelineVariant(Impl &impl, MeshletPipelineResource &pipeline,
                           const MeshDispatchItem &dispatch,
                           nvrhi::IMeshletPipeline *&outPipeline) {
  const PipelineVariantKey key = makePipelineVariantKey(dispatch);
  if (auto found = pipeline.variants.find(key);
      found != pipeline.variants.end()) {
    outPipeline = found->second.Get();
    return std::nullopt;
  }
  ++impl.undeclaredMeshletPipelineVariantMisses;
  return std::string("Meshlet pipeline raster variant was not declared during "
                     "pipeline preparation: ") +
         pipeline.debugName;
}

[[nodiscard]] nvrhi::FramebufferInfo
makeFramebufferInfo(const RenderPipelineDesc &desc) {
  nvrhi::FramebufferInfo info{};
  for (uint32_t i = 0u; i < desc.colorAttachmentCount; ++i) {
    info.addColorFormat(toNvrhiFormat(desc.colorFormats[i]));
  }
  if (desc.depthFormat != Format::Count) {
    info.setDepthFormat(toNvrhiFormat(desc.depthFormat));
  }
  info.setSampleCount(std::max(desc.numSamples, 1u));
  return info;
}

[[nodiscard]] nvrhi::FramebufferInfo
makeFramebufferInfo(const MeshletPipelineDesc &desc) {
  nvrhi::FramebufferInfo info{};
  for (uint32_t i = 0u; i < desc.colorAttachmentCount; ++i) {
    info.addColorFormat(toNvrhiFormat(desc.colorFormats[i]));
  }
  if (desc.depthFormat != Format::Count) {
    info.setDepthFormat(toNvrhiFormat(desc.depthFormat));
  }
  info.setSampleCount(std::max(desc.numSamples, 1u));
  return info;
}

[[nodiscard]] nvrhi::ViewportState makeViewportState(const Viewport &viewport,
                                                     const RectU32 *scissor) {
  nvrhi::ViewportState state{};
  state.addViewport(toNvrhiViewport(viewport));
  if (scissor != nullptr) {
    state.addScissorRect(toNvrhiRect(*scissor));
  } else {
    state.addScissorRect(viewportRect(viewport));
  }
  return state;
}

[[nodiscard]] bool areSameRect(const RectU32 &lhs,
                               const RectU32 &rhs) noexcept {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.width == rhs.width &&
         lhs.height == rhs.height;
}

[[nodiscard]] nvrhi::TextureHandle
passColorTextureHandle(Impl &impl, const RenderPass &pass, bool &outSwapchain) {
  outSwapchain = false;
  if (nuri::isValid(pass.colorTexture)) {
    if (TextureResource *texture = impl.textures.get(pass.colorTexture);
        texture != nullptr) {
      return texture->texture;
    }
    return {};
  }
  if (!impl.hasPreparedSwapchainImage ||
      impl.preparedSwapchainImageIndex >= impl.swapchain.images.size()) {
    return {};
  }
  outSwapchain = true;
  return impl.swapchain.images[impl.preparedSwapchainImageIndex].texture;
}

[[nodiscard]] nvrhi::TextureHandle passTextureHandle(Impl &impl,
                                                     TextureHandle handle) {
  TextureResource *texture = impl.textures.get(handle);
  return texture != nullptr ? texture->texture : nvrhi::TextureHandle{};
}

[[nodiscard]] nvrhi::BufferHandle passBufferHandle(Impl &impl,
                                                   BufferHandle handle) {
  BufferResource *buffer = impl.buffers.get(handle);
  return buffer != nullptr ? buffer->buffer : nvrhi::BufferHandle{};
}

[[nodiscard]] nvrhi::ResourceStates
textureShaderReadState(const TextureResource &texture) {
  nvrhi::ResourceStates state = nvrhi::ResourceStates::ShaderResource;
  if (isDepthFormat(texture.format)) {
    state = state | nvrhi::ResourceStates::DepthRead;
  }
  return state;
}

[[nodiscard]] nvrhi::ResourceStates
textureComputeDependencyState(const TextureResource &texture) {
  return texture.desc.usage == TextureUsage::Storage
             ? nvrhi::ResourceStates::UnorderedAccess
             : textureShaderReadState(texture);
}

[[nodiscard]] Result<bool, std::string>
requestTextureDependencyStates(Impl &impl, nvrhi::ICommandList &commandList,
                               std::span<const TextureHandle> textures,
                               bool computeDependency,
                               std::string_view context) {
  for (const TextureHandle handle : textures) {
    if (!nuri::isValid(handle)) {
      continue;
    }
    TextureResource *texture = impl.textures.get(handle);
    if (texture == nullptr || !texture->texture) {
      return Result<bool, std::string>::makeError(
          std::string(context) + ": invalid dependency texture");
    }
    const nvrhi::ResourceStates state =
        computeDependency ? textureComputeDependencyState(*texture)
                          : textureShaderReadState(*texture);
    commandList.setTextureState(texture->texture.Get(), nvrhi::AllSubresources,
                                state);
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
requestBufferDependencyStates(Impl &impl, nvrhi::ICommandList &commandList,
                              std::span<const BufferHandle> buffers,
                              bool computeDependency,
                              std::string_view context) {
  const nvrhi::ResourceStates state =
      computeDependency ? (nvrhi::ResourceStates::ShaderResource |
                           nvrhi::ResourceStates::UnorderedAccess)
                        : nvrhi::ResourceStates::ShaderResource;
  for (const BufferHandle handle : buffers) {
    if (!nuri::isValid(handle)) {
      continue;
    }
    BufferResource *buffer = impl.buffers.get(handle);
    if (buffer == nullptr || !buffer->buffer) {
      return Result<bool, std::string>::makeError(
          std::string(context) + ": invalid dependency buffer");
    }
    if (buffer->immutable && !computeDependency) {
      continue;
    }
    commandList.setBufferState(buffer->buffer.Get(), state);
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] TextureDimensions
textureDimensionsFromHandle(Impl &impl, const nvrhi::TextureHandle &texture) {
  if (!texture) {
    return TextureDimensions{};
  }
  const nvrhi::TextureDesc &desc = texture->getDesc();
  return TextureDimensions{
      .width = desc.width, .height = desc.height, .depth = desc.depth};
}

[[nodiscard]] Result<bool, std::string>
recordComputeDispatches(Impl &impl, nvrhi::ICommandList &commandList,
                        std::span<const ComputeDispatchItem> dispatches,
                        std::string_view context,
                        bool dependencyStatesPreplanned) {
  for (const ComputeDispatchItem &dispatch : dispatches) {
    if (!impl.computePipelines.isValid(dispatch.pipeline)) {
      return Result<bool, std::string>::makeError(
          std::string(context) + ": invalid compute pipeline handle");
    }
    if (dispatch.dispatch.x == 0u || dispatch.dispatch.y == 0u ||
        dispatch.dispatch.z == 0u) {
      return Result<bool, std::string>::makeError(
          std::string(context) + ": invalid compute dispatch size");
    }
    ComputePipelineResource *pipeline =
        impl.computePipelines.get(dispatch.pipeline);
    if (!dependencyStatesPreplanned) {
      auto dependencyBufferResult = requestBufferDependencyStates(
          impl, commandList, dispatch.dependencyBuffers, true, context);
      if (dependencyBufferResult.hasError()) {
        return dependencyBufferResult;
      }
      auto dependencyTextureResult = requestTextureDependencyStates(
          impl, commandList, dispatch.dependencyTextures, true, context);
      if (dependencyTextureResult.hasError()) {
        return dependencyTextureResult;
      }
      commandList.commitBarriers();
    }

    nvrhi::ComputeState state{};
    state.setPipeline(pipeline->pipeline.Get());
    bindGlobalSets(state, impl);
    commandList.setComputeState(state);

    std::array<std::byte, kPushConstantByteSize> push{};
    if (!makePaddedPushConstants(dispatch.pushConstants, push)) {
      return Result<bool, std::string>::makeError(
          std::string(context) + ": push constants exceed 128 bytes (" +
          std::to_string(dispatch.pushConstants.size()) + " bytes)");
    }
    commandList.setPushConstants(push.data(), push.size());
    commandList.dispatch(dispatch.dispatch.x, dispatch.dispatch.y,
                         dispatch.dispatch.z);
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
recordMeshDispatches(Impl &impl, nvrhi::ICommandList &commandList,
                     nvrhi::IFramebuffer *framebuffer, const Viewport &viewport,
                     std::span<const MeshDispatchItem> dispatches,
                     std::string_view context) {
  if (dispatches.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!impl.meshletsSupported) {
    return Result<bool, std::string>::makeError(
        std::string(context) + ": meshlets are unsupported by this device");
  }

  struct BoundMeshletState {
    MeshletPipelineHandle pipeline{};
    PipelineVariantKey variantKey{};
    nvrhi::IMeshletPipeline *variant = nullptr;
    MeshDispatchCommandType command = MeshDispatchCommandType::Direct;
    BufferHandle indirectBuffer{};
    BufferHandle indirectCountBuffer{};
    bool useScissor = false;
    RectU32 scissor{};
    bool valid = false;
  };
  BoundMeshletState boundState{};

  for (const MeshDispatchItem &dispatch : dispatches) {
    NURI_PROFILER_ZONE("NvrhiGPUDevice.mesh_dispatch_submission",
                       NURI_PROFILER_COLOR_CMD_DISPATCH);
    MeshletPipelineResource *pipeline =
        impl.meshletPipelines.get(dispatch.pipeline);
    if (pipeline == nullptr) {
      return Result<bool, std::string>::makeError(
          std::string(context) + ": invalid meshlet pipeline handle");
    }
    if (dispatch.command == MeshDispatchCommandType::Direct &&
        (dispatch.groupsX == 0u || dispatch.groupsY == 0u ||
         dispatch.groupsZ == 0u)) {
      return Result<bool, std::string>::makeError(
          std::string(context) + ": invalid direct mesh dispatch size");
    }

    auto dependencyBufferResult = requestBufferDependencyStates(
        impl, commandList, dispatch.dependencyBuffers, false, context);
    if (dependencyBufferResult.hasError()) {
      return dependencyBufferResult;
    }
    auto dependencyTextureResult = requestTextureDependencyStates(
        impl, commandList, dispatch.dependencyTextures, false, context);
    if (dependencyTextureResult.hasError()) {
      return dependencyTextureResult;
    }
    commandList.commitBarriers();

    nvrhi::IMeshletPipeline *variant = nullptr;
    auto variantError =
        findMeshletPipelineVariant(impl, *pipeline, dispatch, variant);
    if (variantError) {
      return Result<bool, std::string>::makeError(*variantError);
    }

    const PipelineVariantKey variantKey = makePipelineVariantKey(dispatch);
    const bool needsStateBind =
        !boundState.valid ||
        !areSameHandle(boundState.pipeline, dispatch.pipeline) ||
        !(boundState.variantKey == variantKey) ||
        boundState.variant != variant ||
        boundState.command != dispatch.command ||
        !areSameHandle(boundState.indirectBuffer, dispatch.indirectBuffer) ||
        !areSameHandle(boundState.indirectCountBuffer,
                       dispatch.indirectCountBuffer) ||
        boundState.useScissor != dispatch.useScissor ||
        (dispatch.useScissor &&
         !areSameRect(boundState.scissor, dispatch.scissor));
    nvrhi::MeshletState state{};
    if (dispatch.command != MeshDispatchCommandType::Direct) {
      nvrhi::BufferHandle indirectBuffer =
          passBufferHandle(impl, dispatch.indirectBuffer);
      if (!indirectBuffer) {
        return Result<bool, std::string>::makeError(
            std::string(context) + ": invalid mesh dispatch indirect buffer");
      }
      state.setIndirectParams(indirectBuffer.Get());
      if (dispatch.command == MeshDispatchCommandType::IndirectCount) {
        if (!impl.meshletLimits.supportsMeshDispatchIndirectCount) {
          return Result<bool, std::string>::makeError(
              std::string(context) +
              ": mesh dispatch indirect-count is unsupported by this device");
        }
        nvrhi::BufferHandle countBuffer =
            passBufferHandle(impl, dispatch.indirectCountBuffer);
        if (!countBuffer) {
          return Result<bool, std::string>::makeError(
              std::string(context) +
              ": invalid mesh dispatch indirect count buffer");
        }
        state.setIndirectCountBuffer(countBuffer.Get());
      }
    }

    if (needsStateBind) {
      state.setPipeline(variant)
          .setFramebuffer(framebuffer)
          .setViewport(makeViewportState(
              viewport, dispatch.useScissor ? &dispatch.scissor : nullptr));
      bindGlobalSets(state, impl);
      commandList.setMeshletState(state);
      boundState = BoundMeshletState{
          .pipeline = dispatch.pipeline,
          .variantKey = variantKey,
          .variant = variant,
          .command = dispatch.command,
          .indirectBuffer = dispatch.indirectBuffer,
          .indirectCountBuffer = dispatch.indirectCountBuffer,
          .useScissor = dispatch.useScissor,
          .scissor = dispatch.scissor,
          .valid = true,
      };
    }

    std::array<std::byte, kPushConstantByteSize> push{};
    if (!makePaddedPushConstants(dispatch.pushConstants, push)) {
      return Result<bool, std::string>::makeError(
          std::string(context) + ": push constants exceed 128 bytes (" +
          std::to_string(dispatch.pushConstants.size()) + " bytes)");
    }
    commandList.setPushConstants(push.data(), push.size());

    switch (dispatch.command) {
    case MeshDispatchCommandType::Indirect:
      commandList.dispatchMeshIndirect(
          static_cast<uint32_t>(dispatch.indirectBufferOffset),
          std::max(dispatch.indirectDispatchCount, 1u));
      break;
    case MeshDispatchCommandType::IndirectCount:
      commandList.dispatchMeshIndirectCount(
          static_cast<uint32_t>(dispatch.indirectBufferOffset),
          static_cast<uint32_t>(dispatch.indirectCountBufferOffset),
          std::max(dispatch.indirectDispatchCount, 1u));
      break;
    case MeshDispatchCommandType::Direct:
    default:
      commandList.dispatchMesh(dispatch.groupsX, dispatch.groupsY,
                               dispatch.groupsZ);
      break;
    }
    NURI_PROFILER_ZONE_END();
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
recordRenderPass(Impl &impl,
                 std::vector<nvrhi::FramebufferHandle> *framebuffers,
                 std::vector<NvrhiTimingQuery> *timingQueries,
                 nvrhi::ICommandList &commandList, const RenderPass &pass) {
  ScopedNvrhiPassTiming passTiming(impl, commandList, timingQueries,
                                   pass.gpuTimingScope, pass.debugLabel);
  const bool copyOnly = pass.executionMode == RenderPassExecutionMode::CopyOnly;
  if (copyOnly) {
    if (pass.hasColorAttachment || nuri::isValid(pass.colorTexture) ||
        nuri::isValid(pass.colorResolveTexture) ||
        nuri::isValid(pass.depthTexture) ||
        nuri::isValid(pass.depthResolveTexture) ||
        !pass.preDispatches.empty() || !pass.draws.empty() ||
        !pass.meshDispatches.empty()) {
      return Result<bool, std::string>::makeError(
          "NvrhiGPUDevice::recordRenderPass: invalid copy-only pass payload");
    }
    if (pass.textureCopies.empty()) {
      return Result<bool, std::string>::makeError(
          "NvrhiGPUDevice::recordRenderPass: copy-only pass has no copies");
    }

    for (const TextureCopyItem &copy : pass.textureCopies) {
      TextureResource *source = impl.textures.get(copy.sourceTexture);
      TextureResource *destination = impl.textures.get(copy.destinationTexture);
      if (source == nullptr || destination == nullptr || !source->texture ||
          !destination->texture) {
        return Result<bool, std::string>::makeError(
            "NvrhiGPUDevice::recordRenderPass: invalid texture copy handle");
      }
      if (copy.sourceTexture.index == copy.destinationTexture.index &&
          copy.sourceTexture.generation == copy.destinationTexture.generation) {
        return Result<bool, std::string>::makeError(
            "NvrhiGPUDevice::recordRenderPass: texture copy source and "
            "destination match");
      }
      if (source->format != destination->format ||
          source->desc.numSamples != 1u || destination->desc.numSamples != 1u ||
          source->desc.type != TextureType::Texture2D ||
          destination->desc.type != TextureType::Texture2D) {
        return Result<bool, std::string>::makeError(
            "NvrhiGPUDevice::recordRenderPass: incompatible texture copy "
            "resources");
      }
      if (copy.width == 0u || copy.height == 0u ||
          copy.sourceMipLevel >= source->desc.numMipLevels ||
          copy.destinationMipLevel >= destination->desc.numMipLevels ||
          copy.sourceLayer >= source->desc.numLayers ||
          copy.destinationLayer >= destination->desc.numLayers) {
        return Result<bool, std::string>::makeError(
            "NvrhiGPUDevice::recordRenderPass: invalid texture copy region");
      }
      const uint32_t sourceWidth =
          mipSize(source->desc.dimensions.width, copy.sourceMipLevel);
      const uint32_t sourceHeight =
          mipSize(source->desc.dimensions.height, copy.sourceMipLevel);
      const uint32_t destinationWidth =
          mipSize(destination->desc.dimensions.width, copy.destinationMipLevel);
      const uint32_t destinationHeight = mipSize(
          destination->desc.dimensions.height, copy.destinationMipLevel);
      if (copy.sourceX >= sourceWidth || copy.sourceY >= sourceHeight ||
          copy.width > sourceWidth - copy.sourceX ||
          copy.height > sourceHeight - copy.sourceY ||
          copy.destinationX >= destinationWidth ||
          copy.destinationY >= destinationHeight ||
          copy.width > destinationWidth - copy.destinationX ||
          copy.height > destinationHeight - copy.destinationY) {
        return Result<bool, std::string>::makeError(
            "NvrhiGPUDevice::recordRenderPass: texture copy region out of "
            "bounds");
      }

      commandList.setTextureState(source->texture.Get(), nvrhi::AllSubresources,
                                  nvrhi::ResourceStates::CopySource);
      commandList.setTextureState(destination->texture.Get(),
                                  nvrhi::AllSubresources,
                                  nvrhi::ResourceStates::CopyDest);
      commandList.commitBarriers();

      nvrhi::TextureSlice sourceSlice{};
      sourceSlice.setOrigin(copy.sourceX, copy.sourceY, 0u)
          .setSize(copy.width, copy.height, 1u)
          .setMipLevel(copy.sourceMipLevel)
          .setArraySlice(copy.sourceLayer);
      nvrhi::TextureSlice destinationSlice{};
      destinationSlice.setOrigin(copy.destinationX, copy.destinationY, 0u)
          .setSize(copy.width, copy.height, 1u)
          .setMipLevel(copy.destinationMipLevel)
          .setArraySlice(copy.destinationLayer);
      commandList.copyTexture(destination->texture.Get(), destinationSlice,
                              source->texture.Get(), sourceSlice);
    }

    passTiming.commit();
    return Result<bool, std::string>::makeResult(true);
  }
  if (!pass.textureCopies.empty()) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::recordRenderPass: non-copy pass contains texture "
        "copies");
  }
  auto preDispatchResult =
      recordComputeDispatches(impl, commandList, pass.preDispatches,
                              "NvrhiGPUDevice::recordRenderPass preDispatch",
                              /*dependencyStatesPreplanned=*/true);
  if (preDispatchResult.hasError()) {
    return preDispatchResult;
  }
  if (pass.executionMode == RenderPassExecutionMode::ComputeOnly) {
    if (!pass.meshDispatches.empty()) {
      return Result<bool, std::string>::makeError(
          "NvrhiGPUDevice::recordRenderPass: compute-only pass contains mesh "
          "dispatches");
    }
    passTiming.commit();
    return Result<bool, std::string>::makeResult(true);
  }

  bool colorIsSwapchain = false;
  nvrhi::TextureHandle colorTexture{};
  if (pass.hasColorAttachment) {
    colorTexture = passColorTextureHandle(impl, pass, colorIsSwapchain);
    if (!colorTexture) {
      return Result<bool, std::string>::makeError(
          "NvrhiGPUDevice::recordRenderPass: invalid color texture");
    }
  }

  nvrhi::TextureHandle colorResolveTexture{};
  if (nuri::isValid(pass.colorResolveTexture)) {
    colorResolveTexture = passTextureHandle(impl, pass.colorResolveTexture);
    if (!colorResolveTexture) {
      return Result<bool, std::string>::makeError(
          "NvrhiGPUDevice::recordRenderPass: invalid color resolve texture");
    }
  }

  nvrhi::TextureHandle depthTexture{};
  if (nuri::isValid(pass.depthTexture)) {
    depthTexture = passTextureHandle(impl, pass.depthTexture);
    if (!depthTexture) {
      return Result<bool, std::string>::makeError(
          "NvrhiGPUDevice::recordRenderPass: invalid depth texture");
    }
  }

  nvrhi::TextureHandle depthResolveTexture{};
  if (nuri::isValid(pass.depthResolveTexture)) {
    depthResolveTexture = passTextureHandle(impl, pass.depthResolveTexture);
    if (!depthResolveTexture) {
      return Result<bool, std::string>::makeError(
          "NvrhiGPUDevice::recordRenderPass: invalid depth resolve texture");
    }
  }

  const bool colorFramebufferResolve =
      colorTexture && colorResolveTexture &&
      pass.color.storeOp == StoreOp::MsaaResolve;
  const bool depthFramebufferResolve =
      depthTexture && depthResolveTexture &&
      pass.depth.storeOp == StoreOp::MsaaResolve;

  nvrhi::FramebufferDesc framebufferDesc{};
  if (colorTexture) {
    nvrhi::FramebufferAttachment colorAttachment{};
    colorAttachment.setTexture(colorTexture.Get());
    if (colorFramebufferResolve) {
      colorAttachment.setResolveTexture(colorResolveTexture.Get())
          .setResolveMode(toNvrhiResolveMode(pass.color.resolveMode));
    }
    framebufferDesc.addColorAttachment(colorAttachment);
  }
  if (depthTexture) {
    nvrhi::FramebufferAttachment depthAttachment{};
    depthAttachment.setTexture(depthTexture.Get());
    if (depthFramebufferResolve) {
      depthAttachment.setResolveTexture(depthResolveTexture.Get())
          .setResolveMode(toNvrhiResolveMode(pass.depth.resolveMode));
    }
    framebufferDesc.setDepthAttachment(depthAttachment);
  }

  const FramebufferCacheKey framebufferKey{
      .colorTexture = pass.hasColorAttachment && !colorIsSwapchain
                          ? pass.colorTexture
                          : TextureHandle{},
      .colorResolveTexture =
          colorFramebufferResolve ? pass.colorResolveTexture : TextureHandle{},
      .depthTexture = pass.depthTexture,
      .depthResolveTexture =
          depthFramebufferResolve ? pass.depthResolveTexture : TextureHandle{},
      .swapchainGeneration = colorIsSwapchain ? impl.swapchainGeneration : 0u,
      .swapchainImageIndex =
          colorIsSwapchain ? impl.preparedSwapchainImageIndex : 0u,
      .colorResolveMode =
          colorFramebufferResolve ? pass.color.resolveMode : ResolveMode::None,
      .depthResolveMode =
          depthFramebufferResolve ? pass.depth.resolveMode : ResolveMode::None,
      .hasColorAttachment = pass.hasColorAttachment,
      .colorIsSwapchain = colorIsSwapchain,
      .colorResolve = colorFramebufferResolve,
      .depthResolve = depthFramebufferResolve,
  };
  auto cachedFramebuffer = std::find_if(
      impl.cachedFramebuffers.begin(), impl.cachedFramebuffers.end(),
      [&framebufferKey](const CachedFramebuffer &entry) {
        return entry.key == framebufferKey;
      });
  nvrhi::FramebufferHandle framebuffer{};
  if (cachedFramebuffer != impl.cachedFramebuffers.end()) {
    framebuffer = cachedFramebuffer->framebuffer;
  } else {
    NURI_PROFILER_ZONE("NvrhiGPUDevice.create_framebuffer",
                       NURI_PROFILER_COLOR_CREATE);
    framebuffer = impl.nvrhiDevice->createFramebuffer(framebufferDesc);
    NURI_PROFILER_ZONE_END();
    if (framebuffer) {
      impl.cachedFramebuffers.push_back(CachedFramebuffer{
          .key = framebufferKey,
          .framebuffer = framebuffer,
      });
    }
  }
  if (!framebuffer) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::recordRenderPass: failed to create framebuffer");
  }
  if (framebuffers != nullptr) {
    framebuffers->push_back(framebuffer);
  }

  if (colorTexture) {
    commandList.setTextureState(colorTexture.Get(), nvrhi::AllSubresources,
                                nvrhi::ResourceStates::RenderTarget);
  }
  if (depthTexture) {
    commandList.setTextureState(depthTexture.Get(), nvrhi::AllSubresources,
                                nvrhi::ResourceStates::DepthWrite);
  }
  commandList.commitBarriers();

  if (colorTexture && pass.color.loadOp == LoadOp::Clear) {
    commandList.clearTextureFloat(colorTexture.Get(), nvrhi::AllSubresources,
                                  toNvrhiColor(pass.color.clearColor));
  }
  if (depthTexture && pass.depth.loadOp == LoadOp::Clear) {
    commandList.clearDepthStencilTexture(
        depthTexture.Get(), nvrhi::AllSubresources, true, pass.depth.clearDepth,
        false, static_cast<uint8_t>(pass.depth.clearStencil));
  }

  Viewport viewport = pass.viewport;
  if (!pass.useViewport) {
    const nvrhi::TextureHandle viewportTexture =
        colorTexture ? colorTexture : depthTexture;
    const TextureDimensions dimensions =
        textureDimensionsFromHandle(impl, viewportTexture);
    viewport = {.x = 0.0f,
                .y = 0.0f,
                .width = static_cast<float>(dimensions.width),
                .height = static_cast<float>(dimensions.height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f};
  }

  {
    NURI_PROFILER_ZONE("NvrhiGPUDevice.record_pass_dependencies",
                       NURI_PROFILER_COLOR_BARRIER);
    auto dependencyBufferResult = requestBufferDependencyStates(
        impl, commandList, pass.dependencyBuffers, false,
        "NvrhiGPUDevice::recordRenderPass");
    if (dependencyBufferResult.hasError()) {
      return dependencyBufferResult;
    }
    auto dependencyTextureResult = requestTextureDependencyStates(
        impl, commandList, pass.dependencyTextures, false,
        "NvrhiGPUDevice::recordRenderPass");
    if (dependencyTextureResult.hasError()) {
      return dependencyTextureResult;
    }
    commandList.commitBarriers();
    NURI_PROFILER_ZONE_END();
  }

  {
    NURI_PROFILER_ZONE("NvrhiGPUDevice.record_draws",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    for (const DrawItem &draw : pass.draws) {
      RenderPipelineResource *pipeline =
          impl.renderPipelines.get(draw.pipeline);
      if (pipeline == nullptr) {
        return Result<bool, std::string>::makeError(
            "NvrhiGPUDevice::recordRenderPass: invalid render pipeline handle");
      }
      nvrhi::IGraphicsPipeline *variant = nullptr;
      auto variantError =
          findGraphicsPipelineVariant(impl, *pipeline, draw, variant);
      if (variantError) {
        return Result<bool, std::string>::makeError(*variantError);
      }

      nvrhi::GraphicsState state{};
      state.setPipeline(variant)
          .setFramebuffer(framebuffer.Get())
          .setViewport(makeViewportState(
              viewport, draw.useScissor ? &draw.scissor : nullptr));
      bindGlobalSets(state, impl);

      if (nuri::isValid(draw.vertexBuffer)) {
        nvrhi::BufferHandle vertexBuffer =
            passBufferHandle(impl, draw.vertexBuffer);
        if (!vertexBuffer) {
          return Result<bool, std::string>::makeError(
              "NvrhiGPUDevice::recordRenderPass: invalid vertex buffer handle");
        }
        state.addVertexBuffer(nvrhi::VertexBufferBinding()
                                  .setBuffer(vertexBuffer.Get())
                                  .setSlot(0u)
                                  .setOffset(draw.vertexBufferOffset));
      }
      if (draw.indexCount > 0u || draw.command != DrawCommandType::Direct) {
        nvrhi::BufferHandle indexBuffer =
            passBufferHandle(impl, draw.indexBuffer);
        if (!indexBuffer) {
          return Result<bool, std::string>::makeError(
              "NvrhiGPUDevice::recordRenderPass: invalid index buffer handle");
        }
        state.setIndexBuffer(
            nvrhi::IndexBufferBinding()
                .setBuffer(indexBuffer.Get())
                .setFormat(toNvrhiIndexFormat(draw.indexFormat))
                .setOffset(static_cast<uint32_t>(draw.indexBufferOffset)));
      }
      if (draw.command != DrawCommandType::Direct) {
        nvrhi::BufferHandle indirectBuffer =
            passBufferHandle(impl, draw.indirectBuffer);
        if (!indirectBuffer) {
          return Result<bool, std::string>::makeError(
              "NvrhiGPUDevice::recordRenderPass: invalid indirect buffer "
              "handle");
        }
        state.setIndirectParams(indirectBuffer.Get());
        if (draw.command == DrawCommandType::IndexedIndirectCount) {
          nvrhi::BufferHandle countBuffer =
              passBufferHandle(impl, draw.indirectCountBuffer);
          if (!countBuffer) {
            return Result<bool, std::string>::makeError(
                "NvrhiGPUDevice::recordRenderPass: invalid indirect count "
                "buffer handle");
          }
          state.setIndirectCountBuffer(countBuffer.Get());
        }
      }

      commandList.setGraphicsState(state);
      std::array<std::byte, kPushConstantByteSize> push{};
      if (!makePaddedPushConstants(draw.pushConstants, push)) {
        return Result<bool, std::string>::makeError(
            "NvrhiGPUDevice::recordRenderPass: push constants exceed 128 "
            "bytes");
      }
      commandList.setPushConstants(push.data(), push.size());

      switch (draw.command) {
      case DrawCommandType::IndexedIndirect:
        commandList.drawIndexedIndirect(
            static_cast<uint32_t>(draw.indirectBufferOffset),
            std::max(draw.indirectDrawCount, 1u));
        break;
      case DrawCommandType::IndexedIndirectCount:
        if (impl.supportsDrawIndirectCount) {
          commandList.drawIndexedIndirectCount(
              static_cast<uint32_t>(draw.indirectBufferOffset),
              static_cast<uint32_t>(draw.indirectCountBufferOffset),
              std::max(draw.indirectDrawCount, 1u));
        } else {
          commandList.drawIndexedIndirect(
              static_cast<uint32_t>(draw.indirectBufferOffset),
              std::max(draw.indirectDrawCount, 1u));
        }
        break;
      case DrawCommandType::Direct:
      default:
        if (draw.indexCount > 0u) {
          commandList.drawIndexed(
              nvrhi::DrawArguments()
                  .setVertexCount(draw.indexCount)
                  .setInstanceCount(draw.instanceCount)
                  .setStartIndexLocation(draw.firstIndex)
                  .setStartVertexLocation(
                      static_cast<uint32_t>(draw.vertexOffset))
                  .setStartInstanceLocation(draw.firstInstance));
        } else {
          commandList.draw(nvrhi::DrawArguments()
                               .setVertexCount(draw.vertexCount)
                               .setInstanceCount(draw.instanceCount)
                               .setStartVertexLocation(draw.firstVertex)
                               .setStartInstanceLocation(draw.firstInstance));
        }
        break;
      }
    }
    NURI_PROFILER_ZONE_END();
  }

  {
    NURI_PROFILER_ZONE("NvrhiGPUDevice.record_mesh_dispatches",
                       NURI_PROFILER_COLOR_CMD_DISPATCH);
    auto meshDispatchResult = recordMeshDispatches(
        impl, commandList, framebuffer.Get(), viewport, pass.meshDispatches,
        "NvrhiGPUDevice::recordRenderPass meshDispatch");
    if (meshDispatchResult.hasError()) {
      return meshDispatchResult;
    }
    NURI_PROFILER_ZONE_END();
  }

  if ((colorFramebufferResolve || depthFramebufferResolve) &&
      pass.draws.empty() && pass.meshDispatches.empty()) {
    commandList.resolveFramebuffer(framebuffer.Get());
  }
  if (colorResolveTexture && colorTexture && !colorFramebufferResolve) {
    commandList.resolveTexture(colorResolveTexture.Get(),
                               nvrhi::AllSubresources, colorTexture.Get(),
                               nvrhi::AllSubresources);
  }
  if (depthResolveTexture && depthTexture && !depthFramebufferResolve) {
    NURI_LOG_WARNING("NvrhiGPUDevice::recordRenderPass: depth resolve requires "
                     "StoreOp::MsaaResolve");
  }
  if (colorIsSwapchain && colorTexture) {
    commandList.setTextureState(colorTexture.Get(), nvrhi::AllSubresources,
                                nvrhi::ResourceStates::Present);
    commandList.commitBarriers();
  }

  passTiming.commit();
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
createBuiltinSamplers(NvrhiGPUDevice &device, Impl &impl) {
  const auto createBuiltin =
      [&device](
          const SamplerDesc &desc,
          std::string_view debugName) -> Result<SamplerHandle, std::string> {
    return device.createSampler(desc, debugName);
  };

  auto bilinear = createBuiltin(SamplerDesc{.minFilter = SamplerFilter::Linear,
                                            .magFilter = SamplerFilter::Linear,
                                            .mipMode = SamplerMipMode::Disabled,
                                            .wrapU = SamplerWrapMode::Repeat,
                                            .wrapV = SamplerWrapMode::Repeat,
                                            .wrapW = SamplerWrapMode::Repeat},
                                "nuri_sampler_bilinear");
  if (bilinear.hasError()) {
    return Result<bool, std::string>::makeError(bilinear.error());
  }
  impl.bilinearSampler = bilinear.value();

  auto trilinear = createBuiltin(SamplerDesc{.minFilter = SamplerFilter::Linear,
                                             .magFilter = SamplerFilter::Linear,
                                             .mipMode = SamplerMipMode::Linear,
                                             .wrapU = SamplerWrapMode::Repeat,
                                             .wrapV = SamplerWrapMode::Repeat,
                                             .wrapW = SamplerWrapMode::Repeat},
                                 "nuri_sampler_trilinear");
  if (trilinear.hasError()) {
    return Result<bool, std::string>::makeError(trilinear.error());
  }
  impl.trilinearSampler = trilinear.value();

  for (size_t i = 0u; i < kSupportedAnisotropyLevels.size(); ++i) {
    const uint8_t requested = kSupportedAnisotropyLevels[i];
    if (impl.maxSamplerAnisotropy < requested) {
      continue;
    }
    auto sampler =
        createBuiltin(SamplerDesc{.minFilter = SamplerFilter::Linear,
                                  .magFilter = SamplerFilter::Linear,
                                  .mipMode = SamplerMipMode::Linear,
                                  .wrapU = SamplerWrapMode::Repeat,
                                  .wrapV = SamplerWrapMode::Repeat,
                                  .wrapW = SamplerWrapMode::Repeat,
                                  .maxAnisotropy = requested},
                      std::format("nuri_sampler_aniso_{}x", requested));
    if (sampler.hasError()) {
      return Result<bool, std::string>::makeError(sampler.error());
    }
    impl.anisotropicSamplers[i] = sampler.value();
  }

  auto cubemap = createBuiltin(SamplerDesc{.minFilter = SamplerFilter::Linear,
                                           .magFilter = SamplerFilter::Linear,
                                           .mipMode = SamplerMipMode::Linear,
                                           .wrapU = SamplerWrapMode::Clamp,
                                           .wrapV = SamplerWrapMode::Clamp,
                                           .wrapW = SamplerWrapMode::Clamp},
                               "nuri_cubemap_sampler");
  if (cubemap.hasError()) {
    return Result<bool, std::string>::makeError(cubemap.error());
  }
  impl.cubemapSampler = cubemap.value();
  return Result<bool, std::string>::makeResult(true);
}

} // namespace

NvrhiGPUDevice::NvrhiGPUDevice() = default;

NvrhiGPUDevice::~NvrhiGPUDevice() {
  if (impl_) {
    destroyVulkan(*impl_);
  }
}

std::unique_ptr<NvrhiGPUDevice>
NvrhiGPUDevice::create(Window &window, const GPUDeviceCreateDesc &desc) {
  auto device = std::unique_ptr<NvrhiGPUDevice>(new NvrhiGPUDevice());
  device->impl_ = std::make_unique<Impl>();
  Impl &impl = *device->impl_;
  impl.recordingContextSlots.reserve(kMaxGraphicsRecordingContexts);
  impl.recordedCommandBufferSlots.reserve(kMaxGraphicsRecordingContexts);
  impl.submissionSlots.reserve(kSwapchainFramesInFlight + 1u);
  impl.activeGraphicsContexts.reserve(kMaxGraphicsRecordingContexts);
  impl.recordedGraphicsCommandBuffers.reserve(kMaxGraphicsRecordingContexts);
  impl.pendingGraphicsCommandLists.reserve(kSwapchainFramesInFlight + 1u);
  impl.availableGraphicsCommandLists.reserve(kSwapchainFramesInFlight + 1u);
  impl.pendingAsyncUploadSubmissions.reserve(kSwapchainFramesInFlight + 1u);
  impl.availableAsyncUploadCommandLists.reserve(kSwapchainFramesInFlight + 1u);
  impl.window = &window;
  impl.glfwWindow = static_cast<GLFWwindow *>(window.nativeHandle());
  impl.requestedPresentMode = requestedPresentModeFromEnvironment();
  impl.renderDocAttached = isRenderDocAttached();
  impl.validationEnabled = resolveValidationEnabled(impl.renderDocAttached);

  NURI_LOG_INFO("NvrhiGPUDevice::create: RenderDoc attached=%s validation=%s",
                impl.renderDocAttached ? "true" : "false",
                impl.validationEnabled ? "true" : "false");
  auto instanceResult = createInstance(impl);
  if (instanceResult.hasError()) {
    NURI_LOG_ERROR("%s", instanceResult.error().c_str());
    destroyVulkan(impl);
    return nullptr;
  }
  VkResult result = glfwCreateWindowSurface(impl.instance, impl.glfwWindow,
                                            nullptr, &impl.surface);
  if (result != VK_SUCCESS) {
    NURI_LOG_ERROR("%s",
                   vkError("NvrhiGPUDevice::create surface", result).c_str());
    destroyVulkan(impl);
    return nullptr;
  }
  auto physicalResult = selectPhysicalDevice(impl);
  if (physicalResult.hasError()) {
    NURI_LOG_ERROR("%s", physicalResult.error().c_str());
    destroyVulkan(impl);
    return nullptr;
  }
  auto logicalResult = createLogicalDevice(impl);
  if (logicalResult.hasError()) {
    NURI_LOG_ERROR("%s", logicalResult.error().c_str());
    destroyVulkan(impl);
    return nullptr;
  }
  NURI_LOG_INFO(
      "NvrhiGPUDevice::create: Vulkan device='%s' vendor=0x%04x device=0x%04x "
      "api=%s",
      impl.physicalDeviceProperties.deviceName,
      impl.physicalDeviceProperties.vendorID,
      impl.physicalDeviceProperties.deviceID,
      formatVkVersion(impl.physicalDeviceProperties.apiVersion).c_str());

  auto nvrhiResult = createNvrhiDevice(impl);
  if (nvrhiResult.hasError()) {
    NURI_LOG_ERROR("%s", nvrhiResult.error().c_str());
    destroyVulkan(impl);
    return nullptr;
  }
  auto bindingResult = createGlobalBindingLayouts(impl);
  if (bindingResult.hasError()) {
    NURI_LOG_ERROR("%s", bindingResult.error().c_str());
    destroyVulkan(impl);
    return nullptr;
  }
  auto swapchainResult = createSwapchain(impl);
  if (swapchainResult.hasError()) {
    NURI_LOG_ERROR("%s", swapchainResult.error().c_str());
    destroyVulkan(impl);
    return nullptr;
  }
  auto samplerResult = createBuiltinSamplers(*device, impl);
  if (samplerResult.hasError()) {
    NURI_LOG_ERROR("NvrhiGPUDevice::create: %s", samplerResult.error().c_str());
    destroyVulkan(impl);
    return nullptr;
  }
  impl.geometryPool =
      std::make_unique<GeometryPool>(*device, desc.geometryPool);
  return device;
}

bool NvrhiGPUDevice::shouldClose() const {
  return impl_ == nullptr || impl_->window == nullptr ||
         impl_->window->shouldClose();
}

void NvrhiGPUDevice::getWindowSize(int32_t &outWidth,
                                   int32_t &outHeight) const {
  if (impl_ == nullptr || impl_->window == nullptr) {
    outWidth = 0;
    outHeight = 0;
    return;
  }
  impl_->window->getWindowSize(outWidth, outHeight);
}

void NvrhiGPUDevice::getFramebufferSize(int32_t &outWidth,
                                        int32_t &outHeight) const {
  if (impl_ == nullptr || impl_->window == nullptr) {
    outWidth = 0;
    outHeight = 0;
    return;
  }
  impl_->window->getFramebufferSize(outWidth, outHeight);
}

void NvrhiGPUDevice::resizeSwapchain(int32_t width, int32_t height) {
  if (impl_ == nullptr || width <= 0 || height <= 0) {
    return;
  }
  impl_->nvrhiDevice->waitForIdle();
  auto swapchainResult = createSwapchain(*impl_);
  if (swapchainResult.hasError()) {
    NURI_LOG_WARNING("NvrhiGPUDevice::resizeSwapchain: %s",
                     swapchainResult.error().c_str());
    return;
  }

  impl_->framebufferTextures.erase(
      std::remove_if(impl_->framebufferTextures.begin(),
                     impl_->framebufferTextures.end(),
                     [this](const FramebufferTexture &entry) {
                       return !impl_->textures.isValid(entry.handle);
                     }),
      impl_->framebufferTextures.end());
  for (const FramebufferTexture &entry : impl_->framebufferTextures) {
    TextureDesc resized = entry.desc;
    resized.dimensions.width = static_cast<uint32_t>(width);
    resized.dimensions.height = static_cast<uint32_t>(height);
    auto resource = createTextureResource(*impl_, resized, entry.debugName);
    if (resource.hasError()) {
      NURI_LOG_WARNING("NvrhiGPUDevice::resizeSwapchain: failed to resize "
                       "texture '%s': %s",
                       entry.debugName.c_str(), resource.error().c_str());
      continue;
    }
    if (!impl_->textures.replace(entry.handle, std::move(resource.value()))) {
      NURI_LOG_WARNING("NvrhiGPUDevice::resizeSwapchain: failed to replace "
                       "texture '%s'",
                       entry.debugName.c_str());
      continue;
    }
    if (TextureResource *slot = impl_->textures.get(entry.handle);
        slot != nullptr &&
        !writeTextureDescriptor(*impl_, entry.handle.index, *slot)) {
      NURI_LOG_WARNING("NvrhiGPUDevice::resizeSwapchain: failed to rewrite "
                       "texture descriptor for '%s'",
                       entry.debugName.c_str());
    }
  }
}

Format NvrhiGPUDevice::getSwapchainFormat() const {
  return impl_ != nullptr ? toNuriSwapchainFormat(impl_->swapchain.format)
                          : Format::RGBA8_UNORM;
}

uint32_t NvrhiGPUDevice::getSwapchainImageIndex() const {
  return impl_ != nullptr ? impl_->preparedSwapchainImageIndex : 0u;
}

uint32_t NvrhiGPUDevice::getSwapchainImageCount() const {
  return impl_ != nullptr
             ? static_cast<uint32_t>(impl_->swapchain.images.size())
             : 0u;
}

double NvrhiGPUDevice::getTime() const {
  return impl_ != nullptr && impl_->window != nullptr ? impl_->window->getTime()
                                                      : 0.0;
}

bool NvrhiGPUDevice::supportsSwapchainPresentModeChange() const noexcept {
  return impl_ != nullptr && impl_->device != VK_NULL_HANDLE &&
         impl_->surface != VK_NULL_HANDLE;
}

SwapchainPresentMode NvrhiGPUDevice::getSwapchainPresentMode() const noexcept {
  return impl_ != nullptr ? impl_->activePresentMode
                          : SwapchainPresentMode::Unknown;
}

Result<SwapchainPresentMode, std::string>
NvrhiGPUDevice::setSwapchainPresentMode(SwapchainPresentMode mode) {
  if (!supportsSwapchainPresentModeChange()) {
    return Result<SwapchainPresentMode, std::string>::makeError(
        "NVRHI swapchain is not initialized");
  }
  if (mode == SwapchainPresentMode::Unknown) {
    return Result<SwapchainPresentMode, std::string>::makeError(
        "Unknown is not a selectable swapchain present mode");
  }
  if (mode == impl_->requestedPresentMode) {
    return Result<SwapchainPresentMode, std::string>::makeResult(
        impl_->activePresentMode);
  }

  const SwapchainPresentMode previousMode = impl_->requestedPresentMode;
  impl_->requestedPresentMode = mode;
  auto swapchainResult = createSwapchain(*impl_);
  if (swapchainResult.hasError()) {
    impl_->requestedPresentMode = previousMode;
    return Result<SwapchainPresentMode, std::string>::makeError(
        swapchainResult.error());
  }
  return Result<SwapchainPresentMode, std::string>::makeResult(
      impl_->activePresentMode);
}

namespace {

Result<BufferResource, std::string>
prepareBufferResource(Impl &impl, const BufferDesc &desc,
                      std::string_view debugName,
                      bool immediateMutexAlreadyLocked = false) {
  const size_t resolvedSize = desc.size != 0u ? desc.size : desc.data.size();
  if (resolvedSize == 0u) {
    return Result<BufferResource, std::string>::makeError(
        "Buffer size is zero");
  }
  if (!desc.data.empty() && desc.size != 0u && desc.data.size() != desc.size) {
    return Result<BufferResource, std::string>::makeError(
        "Buffer data size must match buffer size");
  }
  if (desc.usage == BufferUsage::None) {
    return Result<BufferResource, std::string>::makeError(
        "Buffer usage is empty");
  }
  if (desc.immutable && desc.storage != Storage::Device) {
    return Result<BufferResource, std::string>::makeError(
        "Immutable buffers must use device-local storage");
  }
  if (desc.immutable && desc.data.empty()) {
    return Result<BufferResource, std::string>::makeError(
        "Immutable buffers require initial data");
  }

  const bool permanentReadState = usesPermanentReadState(desc);
  nvrhi::BufferDesc bufferDesc{};
  bufferDesc.setByteSize(static_cast<uint64_t>(resolvedSize))
      .setDebugName(std::string(debugName))
      .setIsVertexBuffer(hasBufferUsage(desc.usage, BufferUsage::Vertex))
      .setIsIndexBuffer(hasBufferUsage(desc.usage, BufferUsage::Index))
      .setIsConstantBuffer(hasBufferUsage(desc.usage, BufferUsage::Uniform))
      .setCanHaveUAVs(hasBufferUsage(desc.usage, BufferUsage::Storage))
      .setCanHaveRawViews(hasBufferUsage(desc.usage, BufferUsage::Storage))
      .setIsDrawIndirectArgs(hasBufferUsage(desc.usage, BufferUsage::Indirect))
      .setInitialState(nvrhi::ResourceStates::Common)
      .setKeepInitialState(true);
  if (desc.storage == Storage::HostVisible) {
    bufferDesc.setCpuAccess(nvrhi::CpuAccessMode::Write);
  }

  nvrhi::BufferHandle buffer = impl.nvrhiDevice->createBuffer(bufferDesc);
  if (!buffer) {
    return Result<BufferResource, std::string>::makeError(
        "Failed to create NVRHI buffer");
  }

  BufferResource resource{.buffer = buffer,
                          .debugName = std::string(debugName),
                          .byteSize = resolvedSize,
                          .mapped = nullptr,
                          .mappedMemory = VK_NULL_HANDLE,
                          .hostVisible = desc.storage == Storage::HostVisible,
                          .immutable = desc.immutable};
  if (resource.hostVisible) {
    const nvrhi::Object nativeMemory =
        buffer->getNativeObject(nvrhi::ObjectTypes::VK_DeviceMemory);
    resource.mappedMemory = VkDeviceMemory(nativeMemory.integer);
    resource.mapped = static_cast<std::byte *>(
        impl.nvrhiDevice->mapBuffer(buffer.Get(), nvrhi::CpuAccessMode::Write));
    if (!resource.mapped) {
      return Result<BufferResource, std::string>::makeError(
          "Failed to map host-visible buffer");
    }
    if (!desc.data.empty()) {
      std::memcpy(resource.mapped, desc.data.data(), desc.data.size());
      flushMappedBufferRange(impl, resource, 0u, desc.data.size());
    }
  }

  if (!desc.data.empty() && desc.storage == Storage::Device) {
    std::unique_lock immediateLock(impl.immediateMutex, std::defer_lock);
    if (!immediateMutexAlreadyLocked) {
      immediateLock.lock();
    }
    collectCompletedAsyncUploadSubmissions(impl);
    nvrhi::CommandListHandle &commandList =
        ensurePendingAsyncUploadCommandList(impl);
    if (!commandList) {
      return Result<BufferResource, std::string>::makeError(
          "Failed to create buffer upload command list");
    }
    commandList->writeBuffer(buffer.Get(), desc.data.data(), desc.data.size(),
                             0u);
    if (permanentReadState) {
      commandList->setPermanentBufferState(
          buffer.Get(), permanentReadBufferState(desc.usage));
    }
    impl.pendingAsyncUploadBytes += static_cast<uint64_t>(desc.data.size());
  }

  return Result<BufferResource, std::string>::makeResult(std::move(resource));
}

} // namespace

Result<BufferHandle, std::string>
NvrhiGPUDevice::createBuffer(const BufferDesc &desc,
                             std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto prepared = prepareBuffer(desc, debugName);
  if (prepared.hasError()) {
    return Result<BufferHandle, std::string>::makeError(prepared.error());
  }
  return publishPreparedBuffer(std::move(prepared.value()));
}

Result<std::unique_ptr<PreparedGpuBuffer>, std::string>
NvrhiGPUDevice::prepareBuffer(const BufferDesc &desc,
                              std::string_view debugName) {
  if (impl_ == nullptr || !impl_->nvrhiDevice) {
    return Result<std::unique_ptr<PreparedGpuBuffer>, std::string>::makeError(
        "NVRHI device is not initialized");
  }
  std::lock_guard preparationLock(impl_->resourcePreparationMutex);
  auto resource = prepareBufferResource(*impl_, desc, debugName);
  if (resource.hasError()) {
    return Result<std::unique_ptr<PreparedGpuBuffer>, std::string>::makeError(
        resource.error());
  }
  std::unique_ptr<PreparedGpuBuffer> prepared =
      std::make_unique<NvrhiPreparedGpuBuffer>(impl_->nvrhiDevice,
                                               std::move(resource.value()));
  return Result<std::unique_ptr<PreparedGpuBuffer>, std::string>::makeResult(
      std::move(prepared));
}

Result<BufferHandle, std::string> NvrhiGPUDevice::publishPreparedBuffer(
    std::unique_ptr<PreparedGpuBuffer> prepared) {
  if (impl_ == nullptr || prepared == nullptr) {
    return Result<BufferHandle, std::string>::makeError(
        "Prepared buffer is null");
  }
  auto *typed = dynamic_cast<NvrhiPreparedGpuBuffer *>(prepared.get());
  if (typed == nullptr) {
    return Result<BufferHandle, std::string>::makeError(
        "Prepared buffer belongs to a different GPU backend");
  }
  return Result<BufferHandle, std::string>::makeResult(
      impl_->buffers.allocate(typed->take()));
}

Result<std::vector<std::unique_ptr<PreparedGpuBuffer>>, std::string>
NvrhiGPUDevice::prepareBufferBatch(
    std::span<const PreparedBufferRequest> requests) {
  if (impl_ == nullptr || !impl_->nvrhiDevice) {
    return Result<std::vector<std::unique_ptr<PreparedGpuBuffer>>,
                  std::string>::makeError("NVRHI device is not initialized");
  }
  std::scoped_lock lock(impl_->resourcePreparationMutex, impl_->immediateMutex);
  std::vector<std::unique_ptr<PreparedGpuBuffer>> prepared{};
  prepared.reserve(requests.size());
  for (const PreparedBufferRequest &request : requests) {
    auto resource =
        prepareBufferResource(*impl_, request.desc, request.debugName, true);
    if (resource.hasError()) {
      return Result<std::vector<std::unique_ptr<PreparedGpuBuffer>>,
                    std::string>::makeError(resource.error());
    }
    prepared.push_back(std::make_unique<NvrhiPreparedGpuBuffer>(
        impl_->nvrhiDevice, std::move(resource.value())));
  }
  return Result<std::vector<std::unique_ptr<PreparedGpuBuffer>>,
                std::string>::makeResult(std::move(prepared));
}

Result<TextureHandle, std::string>
NvrhiGPUDevice::createTexture(const TextureDesc &desc,
                              std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto prepared = prepareTexture(desc, debugName);
  if (prepared.hasError()) {
    return Result<TextureHandle, std::string>::makeError(prepared.error());
  }
  return publishPreparedTexture(std::move(prepared.value()));
}

Result<std::unique_ptr<PreparedGpuTexture>, std::string>
NvrhiGPUDevice::prepareTexture(const TextureDesc &desc,
                               std::string_view debugName) {
  if (impl_ == nullptr || !impl_->nvrhiDevice) {
    return Result<std::unique_ptr<PreparedGpuTexture>, std::string>::makeError(
        "NVRHI device is not initialized");
  }
  std::lock_guard preparationLock(impl_->resourcePreparationMutex);
  auto resource = createTextureResource(*impl_, desc, debugName);
  if (resource.hasError()) {
    return Result<std::unique_ptr<PreparedGpuTexture>, std::string>::makeError(
        resource.error());
  }
  std::unique_ptr<PreparedGpuTexture> prepared =
      std::make_unique<NvrhiPreparedGpuTexture>(std::move(resource.value()));
  return Result<std::unique_ptr<PreparedGpuTexture>, std::string>::makeResult(
      std::move(prepared));
}

Result<TextureHandle, std::string> NvrhiGPUDevice::publishPreparedTexture(
    std::unique_ptr<PreparedGpuTexture> prepared) {
  if (impl_ == nullptr || prepared == nullptr) {
    return Result<TextureHandle, std::string>::makeError(
        "Prepared texture is null");
  }
  auto *typed = dynamic_cast<NvrhiPreparedGpuTexture *>(prepared.get());
  if (typed == nullptr) {
    return Result<TextureHandle, std::string>::makeError(
        "Prepared texture belongs to a different GPU backend");
  }
  TextureResource resource = typed->take();
  const Format format = resource.format;
  const std::string debugName = resource.debugName;
  const auto reserved = impl_->textures.reserve(debugName, format);
  *reserved.resource = std::move(resource);
  if (!writeTextureDescriptor(*impl_, reserved.handle.index,
                              *reserved.resource)) {
    impl_->textures.deallocate(reserved.handle);
    return Result<TextureHandle, std::string>::makeError(
        "Failed to write texture descriptor");
  }
  return Result<TextureHandle, std::string>::makeResult(reserved.handle);
}

Result<TextureHandle, std::string>
NvrhiGPUDevice::createFramebufferTexture(const TextureDesc &desc,
                                         std::string_view debugName) {
  if (impl_ == nullptr || impl_->window == nullptr) {
    return Result<TextureHandle, std::string>::makeError(
        "No window available to get framebuffer size");
  }
  int32_t width = 0;
  int32_t height = 0;
  impl_->window->getFramebufferSize(width, height);
  if (width <= 0 || height <= 0) {
    return Result<TextureHandle, std::string>::makeError(
        "Failed to get framebuffer size");
  }
  TextureDesc resized = desc;
  resized.dimensions.width = static_cast<uint32_t>(width);
  resized.dimensions.height = static_cast<uint32_t>(height);
  auto result = createTexture(resized, debugName);
  if (result.hasError()) {
    return result;
  }
  impl_->framebufferTextures.push_back(FramebufferTexture{
      .handle = result.value(),
      .desc = desc,
      .debugName = std::string(debugName),
  });
  return result;
}

Result<SamplerHandle, std::string>
NvrhiGPUDevice::createSampler(const SamplerDesc &desc,
                              std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  nvrhi::SamplerHandle sampler =
      impl_->nvrhiDevice->createSampler(toNvrhiSamplerDesc(desc, *impl_));
  if (!sampler) {
    return Result<SamplerHandle, std::string>::makeError(
        "Failed to create sampler");
  }
  ResourceSlot<nvrhi::SamplerHandle> resource{
      .resource = sampler,
      .debugName = std::string(debugName),
      .format = Format::RGBA8_UNORM,
  };
  SamplerHandle handle = impl_->samplers.allocate(std::move(resource));
  if (!writeSamplerDescriptor(*impl_, handle.index, sampler, false) ||
      (desc.depthCompareEnabled &&
       !writeSamplerDescriptor(*impl_, handle.index, sampler, true))) {
    impl_->samplers.deallocate(handle);
    return Result<SamplerHandle, std::string>::makeError(
        "Failed to write sampler descriptor");
  }
  return Result<SamplerHandle, std::string>::makeResult(handle);
}

Result<TextureHandle, std::string> NvrhiGPUDevice::createDepthBuffer() {
  TextureDesc desc{.type = TextureType::Texture2D,
                   .format = Format::D32_FLOAT,
                   .dimensions = {1u, 1u, 1u},
                   .usage = TextureUsage::AttachmentSampled};
  return createFramebufferTexture(desc, "Depth buffer");
}

Result<ShaderHandle, std::string>
NvrhiGPUDevice::createShaderModule(const ShaderDesc &desc) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (desc.source.empty()) {
    return Result<ShaderHandle, std::string>::makeError(
        "Shader source is empty");
  }
  const std::string moduleName(desc.moduleName);
  const std::string patched = patchGlslPrelude(desc.stage, desc.source);
  const glslang_resource_t resource = makeGlslangResource(
      impl_->physicalDeviceProperties.limits, impl_->meshletLimits);
  auto compileResult =
      compileGlslToSpirv(desc.stage, patched.c_str(), resource);
  if (compileResult.hasError()) {
    return Result<ShaderHandle, std::string>::makeError(compileResult.error());
  }
  std::vector<uint8_t> spirv = std::move(compileResult.value());
  if (desc.stage == ShaderStage::Task || desc.stage == ShaderStage::Mesh) {
    [[maybe_unused]] const bool normalized = normalizeSpirvLocalSizeId(spirv);
  }

  nvrhi::ShaderDesc shaderDesc{};
  shaderDesc.setShaderType(toNvrhiShaderType(desc.stage))
      .setDebugName(moduleName)
      .setEntryName("main");
  nvrhi::ShaderHandle shader =
      impl_->nvrhiDevice->createShader(shaderDesc, spirv.data(), spirv.size());
  if (!shader) {
    return Result<ShaderHandle, std::string>::makeError(
        "Failed to create NVRHI shader");
  }

  ShaderResource resourceSlot{
      .shader = shader, .debugName = moduleName, .stage = desc.stage};
  ShaderHandle handle = impl_->shaders.allocate(std::move(resourceSlot));
  return Result<ShaderHandle, std::string>::makeResult(handle);
}

Result<RenderPipelineHandle, std::string>
NvrhiGPUDevice::createRenderPipeline(const RenderPipelineDesc &desc,
                                     std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  ShaderResource *vertexShader = impl_->shaders.get(desc.vertexShader);
  ShaderResource *fragmentShader = impl_->shaders.get(desc.fragmentShader);
  if (vertexShader == nullptr || fragmentShader == nullptr) {
    return Result<RenderPipelineHandle, std::string>::makeError(
        "Invalid render pipeline shader handle");
  }
  if (desc.colorAttachmentCount > desc.colorFormats.size()) {
    return Result<RenderPipelineHandle, std::string>::makeError(
        "Color attachment count exceeds provided color formats");
  }

  const auto reserved = impl_->renderPipelines.reserve(std::string(debugName));
  RenderPipelineResource &resource = *reserved.resource;
  resource.debugName = std::string(debugName);
  resource.framebufferInfo = makeFramebufferInfo(desc);

  std::vector<VertexAttribute> sortedAttributes(
      desc.vertexInput.attributes.begin(), desc.vertexInput.attributes.end());
  std::sort(sortedAttributes.begin(), sortedAttributes.end(),
            [](const VertexAttribute &lhs, const VertexAttribute &rhs) {
              return lhs.location < rhs.location;
            });
  std::vector<nvrhi::VertexAttributeDesc> attributes;
  attributes.reserve(sortedAttributes.size());
  for (const VertexAttribute &attr : sortedAttributes) {
    if (attr.binding >= desc.vertexInput.bindings.size()) {
      impl_->renderPipelines.deallocate(reserved.handle);
      return Result<RenderPipelineHandle, std::string>::makeError(
          "Vertex attribute binding index is out of range");
    }
    attributes.push_back(
        nvrhi::VertexAttributeDesc()
            .setName("ATTR" + std::to_string(attr.location))
            .setFormat(toNvrhiVertexFormat(attr.format))
            .setBufferIndex(attr.binding)
            .setOffset(attr.offset)
            .setElementStride(desc.vertexInput.bindings[attr.binding].stride));
  }
  resource.inputLayout = impl_->nvrhiDevice->createInputLayout(
      attributes.empty() ? nullptr : attributes.data(),
      static_cast<uint32_t>(attributes.size()), vertexShader->shader.Get());
  if (!resource.inputLayout && !attributes.empty()) {
    impl_->renderPipelines.deallocate(reserved.handle);
    return Result<RenderPipelineHandle, std::string>::makeError(
        "Failed to create input layout");
  }

  auto specialize =
      [&](ShaderHandle handle, const char *stageName,
          nvrhi::IShader *&outShader) -> std::optional<std::string> {
    ShaderResource *shader = impl_->shaders.get(handle);
    if (shader == nullptr) {
      outShader = nullptr;
      return std::nullopt;
    }
    if (desc.specInfo.entries.empty()) {
      outShader = shader->shader.Get();
      return std::nullopt;
    }
    auto result =
        specializeShader(*impl_, shader->shader, desc.specInfo, stageName);
    if (result.hasError()) {
      return result.error();
    }
    nvrhi::ShaderHandle specialized = std::move(result).value();
    outShader = specialized.Get();
    resource.specializedShaders.push_back(std::move(specialized));
    return std::nullopt;
  };

  nvrhi::IShader *vsShader = nullptr;
  nvrhi::IShader *fsShader = nullptr;
  nvrhi::IShader *hsShader = nullptr;
  nvrhi::IShader *dsShader = nullptr;
  nvrhi::IShader *gsShader = nullptr;
  auto vsError = specialize(
      desc.vertexShader, "NvrhiGPUDevice::createRenderPipeline VS", vsShader);
  auto fsError = specialize(
      desc.fragmentShader, "NvrhiGPUDevice::createRenderPipeline FS", fsShader);
  auto hsError =
      specialize(desc.tessControlShader,
                 "NvrhiGPUDevice::createRenderPipeline HS", hsShader);
  auto dsError = specialize(
      desc.tessEvalShader, "NvrhiGPUDevice::createRenderPipeline DS", dsShader);
  auto gsError = specialize(
      desc.geometryShader, "NvrhiGPUDevice::createRenderPipeline GS", gsShader);
  if (vsError || fsError || hsError || dsError || gsError) {
    std::string error = vsError   ? *vsError
                        : fsError ? *fsError
                        : hsError ? *hsError
                        : dsError ? *dsError
                                  : *gsError;
    impl_->renderPipelines.deallocate(reserved.handle);
    return Result<RenderPipelineHandle, std::string>::makeError(error);
  }

  nvrhi::RenderState renderState{};
  renderState.rasterState.setCullMode(toNvrhiCullMode(desc.cullMode))
      .setFillMode(toNvrhiFillMode(desc.polygonMode))
      .setFrontCounterClockwise(true)
      .setScissorEnable(true)
      .setMultisampleEnable(desc.numSamples > 1u);
  renderState.depthStencilState
      .setDepthTestEnable(desc.depthFormat != Format::Count)
      .setDepthWriteEnable(desc.depthFormat != Format::Count)
      .setDepthFunc(nvrhi::ComparisonFunc::Less);
  renderState.blendState.setAlphaToCoverageEnable(desc.alphaToCoverageEnabled);
  for (uint32_t i = 0u; i < desc.colorAttachmentCount; ++i) {
    nvrhi::BlendState::RenderTarget target{};
    target.setBlendEnable(desc.blendEnabled)
        .setSrcBlend(desc.blendEnabled ? nvrhi::BlendFactor::SrcAlpha
                                       : nvrhi::BlendFactor::One)
        .setDestBlend(desc.blendEnabled ? nvrhi::BlendFactor::InvSrcAlpha
                                        : nvrhi::BlendFactor::Zero)
        .setSrcBlendAlpha(nvrhi::BlendFactor::One)
        .setDestBlendAlpha(desc.blendEnabled ? nvrhi::BlendFactor::InvSrcAlpha
                                             : nvrhi::BlendFactor::Zero);
    renderState.blendState.setRenderTarget(i, target);
  }

  resource.baseDesc.setPrimType(toNvrhiPrimitiveType(desc.topology))
      .setPatchControlPoints(desc.patchControlPoints)
      .setInputLayout(resource.inputLayout.Get())
      .setVertexShader(vsShader)
      .setFragmentShader(fsShader)
      .setTessellationControlShader(hsShader)
      .setTessellationEvaluationShader(dsShader)
      .setGeometryShader(gsShader)
      .setRenderState(renderState);
  resource.baseDesc.bindingLayouts = globalBindingLayouts(*impl_);

  if (readEnvBoolOverride("NURI_DEBUG_PIPELINE_CREATE_LOG").value_or(false)) {
    const auto shaderName = [this](ShaderHandle handle) -> const char * {
      const ShaderResource *shader = impl_->shaders.get(handle);
      return shader != nullptr ? shader->debugName.c_str() : "-";
    };
    NURI_LOG_INFO(
        "NvrhiGPUDevice::createRenderPipeline: %.*s VS=%s HS=%s DS=%s GS=%s "
        "FS=%s",
        static_cast<int>(debugName.size()), debugName.data(),
        shaderName(desc.vertexShader), shaderName(desc.tessControlShader),
        shaderName(desc.tessEvalShader), shaderName(desc.geometryShader),
        shaderName(desc.fragmentShader));
  }

  nvrhi::IGraphicsPipeline *pipeline = nullptr;
  auto pipelineError = createGraphicsPipelineVariant(
      *impl_, resource, makePipelineVariantKey(desc.rasterState), pipeline);
  if (pipelineError) {
    impl_->renderPipelines.deallocate(reserved.handle);
    return Result<RenderPipelineHandle, std::string>::makeError(*pipelineError);
  }
  for (const RasterPipelineState rasterState : desc.prewarmRasterStates) {
    pipelineError = createGraphicsPipelineVariant(
        *impl_, resource, makePipelineVariantKey(rasterState), pipeline);
    if (pipelineError) {
      impl_->renderPipelines.deallocate(reserved.handle);
      return Result<RenderPipelineHandle, std::string>::makeError(
          *pipelineError);
    }
  }
  return Result<RenderPipelineHandle, std::string>::makeResult(reserved.handle);
}

Result<ComputePipelineHandle, std::string>
NvrhiGPUDevice::createComputePipeline(const ComputePipelineDesc &desc,
                                      std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  ShaderResource *shader = impl_->shaders.get(desc.computeShader);
  if (shader == nullptr) {
    return Result<ComputePipelineHandle, std::string>::makeError(
        "Invalid compute shader handle");
  }
  const auto reserved = impl_->computePipelines.reserve(std::string(debugName));
  ComputePipelineResource &resource = *reserved.resource;
  resource.debugName = std::string(debugName);

  nvrhi::IShader *computeShader = shader->shader.Get();
  if (!desc.specInfo.entries.empty()) {
    auto specialized =
        specializeShader(*impl_, shader->shader, desc.specInfo,
                         "NvrhiGPUDevice::createComputePipeline");
    if (specialized.hasError()) {
      impl_->computePipelines.deallocate(reserved.handle);
      return Result<ComputePipelineHandle, std::string>::makeError(
          specialized.error());
    }
    resource.specializedShader = std::move(specialized).value();
    computeShader = resource.specializedShader.Get();
  }

  nvrhi::ComputePipelineDesc pipelineDesc{};
  pipelineDesc.setComputeShader(computeShader);
  pipelineDesc.bindingLayouts = globalBindingLayouts(*impl_);
  resource.pipeline = impl_->nvrhiDevice->createComputePipeline(pipelineDesc);
  if (!resource.pipeline) {
    impl_->computePipelines.deallocate(reserved.handle);
    return Result<ComputePipelineHandle, std::string>::makeError(
        "Failed to create compute pipeline");
  }
  return Result<ComputePipelineHandle, std::string>::makeResult(
      reserved.handle);
}

Result<MeshletPipelineHandle, std::string>
NvrhiGPUDevice::createMeshletPipeline(const MeshletPipelineDesc &desc,
                                      std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (impl_ == nullptr || !impl_->meshletsSupported) {
    return Result<MeshletPipelineHandle, std::string>::makeError(
        "NvrhiGPUDevice::createMeshletPipeline: meshlets are unsupported by "
        "this device");
  }
  if (desc.colorAttachmentCount > desc.colorFormats.size()) {
    return Result<MeshletPipelineHandle, std::string>::makeError(
        "Color attachment count exceeds provided color formats");
  }

  ShaderResource *taskShader = nuri::isValid(desc.taskShader)
                                   ? impl_->shaders.get(desc.taskShader)
                                   : nullptr;
  ShaderResource *meshShader = impl_->shaders.get(desc.meshShader);
  ShaderResource *fragmentShader = nuri::isValid(desc.fragmentShader)
                                       ? impl_->shaders.get(desc.fragmentShader)
                                       : nullptr;
  if (meshShader == nullptr ||
      (nuri::isValid(desc.fragmentShader) && fragmentShader == nullptr)) {
    return Result<MeshletPipelineHandle, std::string>::makeError(
        "Invalid meshlet pipeline shader handle");
  }
  if (taskShader != nullptr && taskShader->stage != ShaderStage::Task) {
    return Result<MeshletPipelineHandle, std::string>::makeError(
        "Meshlet task shader handle has the wrong shader stage");
  }
  if (meshShader->stage != ShaderStage::Mesh) {
    return Result<MeshletPipelineHandle, std::string>::makeError(
        "Meshlet mesh shader handle has the wrong shader stage");
  }
  if (fragmentShader != nullptr &&
      fragmentShader->stage != ShaderStage::Fragment) {
    return Result<MeshletPipelineHandle, std::string>::makeError(
        "Meshlet fragment shader handle has the wrong shader stage");
  }

  const auto reserved = impl_->meshletPipelines.reserve(std::string(debugName));
  MeshletPipelineResource &resource = *reserved.resource;
  resource.debugName = std::string(debugName);
  resource.framebufferInfo = makeFramebufferInfo(desc);

  auto specialize =
      [&](ShaderResource *shader, const char *stageName,
          nvrhi::IShader *&outShader) -> std::optional<std::string> {
    if (shader == nullptr) {
      outShader = nullptr;
      return std::nullopt;
    }
    if (desc.specInfo.entries.empty()) {
      outShader = shader->shader.Get();
      return std::nullopt;
    }
    auto result =
        specializeShader(*impl_, shader->shader, desc.specInfo, stageName);
    if (result.hasError()) {
      return result.error();
    }
    nvrhi::ShaderHandle specialized = std::move(result).value();
    outShader = specialized.Get();
    resource.specializedShaders.push_back(std::move(specialized));
    return std::nullopt;
  };

  nvrhi::IShader *asShader = nullptr;
  nvrhi::IShader *msShader = nullptr;
  nvrhi::IShader *fsShader = nullptr;
  auto asError = specialize(
      taskShader, "NvrhiGPUDevice::createMeshletPipeline AS", asShader);
  auto msError = specialize(
      meshShader, "NvrhiGPUDevice::createMeshletPipeline MS", msShader);
  auto fsError = specialize(
      fragmentShader, "NvrhiGPUDevice::createMeshletPipeline FS", fsShader);
  if (asError || msError || fsError) {
    std::string error = asError ? *asError : msError ? *msError : *fsError;
    impl_->meshletPipelines.deallocate(reserved.handle);
    return Result<MeshletPipelineHandle, std::string>::makeError(error);
  }

  nvrhi::RenderState renderState{};
  renderState.rasterState.setCullMode(toNvrhiCullMode(desc.cullMode))
      .setFillMode(toNvrhiFillMode(desc.polygonMode))
      .setFrontCounterClockwise(true)
      .setScissorEnable(true)
      .setMultisampleEnable(desc.numSamples > 1u);
  renderState.depthStencilState
      .setDepthTestEnable(desc.depthFormat != Format::Count)
      .setDepthWriteEnable(desc.depthFormat != Format::Count)
      .setDepthFunc(nvrhi::ComparisonFunc::Less);
  renderState.blendState.setAlphaToCoverageEnable(desc.alphaToCoverageEnabled);
  for (uint32_t i = 0u; i < desc.colorAttachmentCount; ++i) {
    nvrhi::BlendState::RenderTarget target{};
    target.setBlendEnable(desc.blendEnabled)
        .setSrcBlend(desc.blendEnabled ? nvrhi::BlendFactor::SrcAlpha
                                       : nvrhi::BlendFactor::One)
        .setDestBlend(desc.blendEnabled ? nvrhi::BlendFactor::InvSrcAlpha
                                        : nvrhi::BlendFactor::Zero)
        .setSrcBlendAlpha(nvrhi::BlendFactor::One)
        .setDestBlendAlpha(desc.blendEnabled ? nvrhi::BlendFactor::InvSrcAlpha
                                             : nvrhi::BlendFactor::Zero);
    renderState.blendState.setRenderTarget(i, target);
  }

  resource.baseDesc.setPrimType(nvrhi::PrimitiveType::TriangleList)
      .setTaskShader(asShader)
      .setMeshShader(msShader)
      .setFragmentShader(fsShader)
      .setRenderState(renderState);
  resource.baseDesc.bindingLayouts = globalBindingLayouts(*impl_);

  nvrhi::IMeshletPipeline *pipeline = nullptr;
  auto pipelineError = createMeshletPipelineVariant(
      *impl_, resource, makePipelineVariantKey(desc.rasterState), pipeline);
  if (pipelineError) {
    impl_->meshletPipelines.deallocate(reserved.handle);
    return Result<MeshletPipelineHandle, std::string>::makeError(
        *pipelineError);
  }
  for (const RasterPipelineState rasterState : desc.prewarmRasterStates) {
    pipelineError = createMeshletPipelineVariant(
        *impl_, resource, makePipelineVariantKey(rasterState), pipeline);
    if (pipelineError) {
      impl_->meshletPipelines.deallocate(reserved.handle);
      return Result<MeshletPipelineHandle, std::string>::makeError(
          *pipelineError);
    }
  }
  return Result<MeshletPipelineHandle, std::string>::makeResult(
      reserved.handle);
}

void NvrhiGPUDevice::destroyRenderPipeline(RenderPipelineHandle pipeline) {
  if (impl_ != nullptr) {
    std::scoped_lock lock(impl_->immediateMutex, impl_->graphicsContextMutex);
    flushPendingAsyncUploadCommandList(*impl_);
    if (auto retired = impl_->renderPipelines.take(pipeline)) {
      retireNativeResource(*impl_, Impl::RetiredResourceTable::RenderPipeline,
                           retired->index, std::move(retired->resource));
    }
  }
}

void NvrhiGPUDevice::destroyComputePipeline(ComputePipelineHandle pipeline) {
  if (impl_ != nullptr) {
    std::scoped_lock lock(impl_->immediateMutex, impl_->graphicsContextMutex);
    flushPendingAsyncUploadCommandList(*impl_);
    if (auto retired = impl_->computePipelines.take(pipeline)) {
      retireNativeResource(*impl_, Impl::RetiredResourceTable::ComputePipeline,
                           retired->index, std::move(retired->resource));
    }
  }
}

void NvrhiGPUDevice::destroyMeshletPipeline(MeshletPipelineHandle pipeline) {
  if (impl_ != nullptr) {
    std::scoped_lock lock(impl_->immediateMutex, impl_->graphicsContextMutex);
    flushPendingAsyncUploadCommandList(*impl_);
    if (auto retired = impl_->meshletPipelines.take(pipeline)) {
      retireNativeResource(*impl_, Impl::RetiredResourceTable::MeshletPipeline,
                           retired->index, std::move(retired->resource));
    }
  }
}

void NvrhiGPUDevice::destroyBuffer(BufferHandle buffer) {
  if (impl_ == nullptr) {
    return;
  }
  std::scoped_lock lock(impl_->immediateMutex, impl_->graphicsContextMutex);
  flushPendingAsyncUploadCommandList(*impl_, false);
  auto retired = impl_->buffers.take(buffer);
  if (!retired) {
    return;
  }
  if (retired->resource.mapped != nullptr) {
    impl_->nvrhiDevice->unmapBuffer(retired->resource.buffer.Get());
    retired->resource.mapped = nullptr;
    retired->resource.mappedMemory = VK_NULL_HANDLE;
  }
  retireNativeResource(*impl_, Impl::RetiredResourceTable::Buffer,
                       retired->index, std::move(retired->resource));
}

void NvrhiGPUDevice::destroyTexture(TextureHandle texture) {
  if (!impl_) {
    return;
  }
  std::scoped_lock lock(impl_->immediateMutex, impl_->graphicsContextMutex);
  flushPendingAsyncUploadCommandList(*impl_);
  invalidateCachedFramebuffers(*impl_, texture);
  impl_->framebufferTextures.erase(
      std::remove_if(impl_->framebufferTextures.begin(),
                     impl_->framebufferTextures.end(),
                     [texture](const FramebufferTexture &entry) {
                       return areSameHandle(entry.handle, texture);
                     }),
      impl_->framebufferTextures.end());
  if (auto retired = impl_->textures.take(texture)) {
    retireNativeResource(*impl_, Impl::RetiredResourceTable::Texture,
                         retired->index, std::move(retired->resource));
  }
}

void NvrhiGPUDevice::destroySampler(SamplerHandle sampler) {
  if (impl_ != nullptr) {
    std::scoped_lock lock(impl_->immediateMutex, impl_->graphicsContextMutex);
    flushPendingAsyncUploadCommandList(*impl_);
    if (auto retired = impl_->samplers.take(sampler)) {
      retireNativeResource(*impl_, Impl::RetiredResourceTable::Sampler,
                           retired->index, std::move(retired->resource));
    }
  }
}

void NvrhiGPUDevice::destroyShaderModule(ShaderHandle shader) {
  if (impl_ != nullptr) {
    std::scoped_lock lock(impl_->immediateMutex, impl_->graphicsContextMutex);
    flushPendingAsyncUploadCommandList(*impl_);
    if (auto retired = impl_->shaders.take(shader)) {
      retireNativeResource(*impl_, Impl::RetiredResourceTable::Shader,
                           retired->index, std::move(retired->resource));
    }
  }
}

bool NvrhiGPUDevice::isValid(BufferHandle h) const {
  return impl_ != nullptr && impl_->buffers.isValid(h);
}

bool NvrhiGPUDevice::isValid(TextureHandle h) const {
  return impl_ != nullptr && impl_->textures.isValid(h);
}

bool NvrhiGPUDevice::isValid(SamplerHandle h) const {
  return impl_ != nullptr && impl_->samplers.isValid(h);
}

bool NvrhiGPUDevice::isValid(ShaderHandle h) const {
  return impl_ != nullptr && impl_->shaders.isValid(h);
}

bool NvrhiGPUDevice::isValid(RenderPipelineHandle h) const {
  return impl_ != nullptr && impl_->renderPipelines.isValid(h);
}

bool NvrhiGPUDevice::isValid(ComputePipelineHandle h) const {
  return impl_ != nullptr && impl_->computePipelines.isValid(h);
}

bool NvrhiGPUDevice::isValid(MeshletPipelineHandle h) const {
  return impl_ != nullptr && impl_->meshletPipelines.isValid(h);
}

Format NvrhiGPUDevice::getTextureFormat(TextureHandle h) const {
  return impl_ != nullptr ? impl_->textures.getFormat(h) : Format::RGBA8_UNORM;
}

TextureDimensions NvrhiGPUDevice::getTextureDimensions(TextureHandle h) const {
  if (impl_ == nullptr) {
    return {};
  }
  const TextureResource *texture = impl_->textures.get(h);
  return texture != nullptr ? texture->desc.dimensions : TextureDimensions{};
}

TextureCompressionCaps NvrhiGPUDevice::getTextureCompressionCaps() const {
  return impl_ != nullptr ? impl_->compressionCaps : TextureCompressionCaps{};
}

TextureUploadTelemetry NvrhiGPUDevice::getTextureUploadTelemetry() const {
  if (impl_ == nullptr) {
    return {};
  }
  std::lock_guard lock(impl_->immediateMutex);
  return TextureUploadTelemetry{
      .texturesRecorded = impl_->textureUploadTexturesRecorded,
      .bytesRecorded = impl_->textureUploadBytesRecorded,
      .batchesSubmitted = impl_->textureUploadBatchesSubmitted,
      .boundedBatchFlushes = impl_->textureUploadBoundedBatchFlushes,
      .completionWaits = impl_->textureUploadCompletionWaits,
      .pendingBytes = impl_->pendingAsyncUploadBytes,
      .pendingTextures = impl_->pendingAsyncUploadTextureCount,
  };
}

GPUAdapterInfo NvrhiGPUDevice::getAdapterInfo() const {
  return impl_ != nullptr ? impl_->adapterInfo : GPUAdapterInfo{};
}

GpuMultisampleCapabilities NvrhiGPUDevice::getMultisampleCapabilities() const {
  return impl_ != nullptr ? impl_->multisampleCapabilities
                          : GpuMultisampleCapabilities{};
}

bool NvrhiGPUDevice::supportsFeature(GPUFeature feature) const {
  if (impl_ == nullptr || !impl_->nvrhiDevice) {
    return false;
  }
  switch (feature) {
  case GPUFeature::Meshlets:
    return impl_->meshletsSupported;
  case GPUFeature::RayTracingClusters:
    return impl_->nvrhiDevice->queryFeatureSupport(
        nvrhi::Feature::RayTracingClusters);
  }
  return false;
}

MeshletLimits NvrhiGPUDevice::getMeshletLimits() const {
  return impl_ != nullptr ? impl_->meshletLimits : MeshletLimits{};
}

bool NvrhiGPUDevice::supportsSampledImageLinearFiltering(Format format) const {
  if (impl_ == nullptr || impl_->physicalDevice == VK_NULL_HANDLE) {
    return false;
  }
  const VkFormat vkFormat = toVkFormat(format);
  if (vkFormat == VK_FORMAT_UNDEFINED) {
    return false;
  }
  VkFormatProperties2 props{.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
  vkGetPhysicalDeviceFormatProperties2(impl_->physicalDevice, vkFormat, &props);
  return (props.formatProperties.optimalTilingFeatures &
          VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0u;
}

uint32_t NvrhiGPUDevice::getTextureBindlessIndex(TextureHandle h) const {
  return impl_ != nullptr && impl_->textures.isValid(h)
             ? h.index
             : kInvalidTextureBindlessIndex;
}

uint32_t NvrhiGPUDevice::getSamplerBindlessIndex(SamplerHandle h) const {
  return impl_ != nullptr && impl_->samplers.isValid(h) ? h.index : 0u;
}

uint8_t NvrhiGPUDevice::getMaxSamplerAnisotropy() const {
  return impl_ != nullptr ? impl_->maxSamplerAnisotropy : 1u;
}

uint32_t NvrhiGPUDevice::getLinearRepeatSamplerBindlessIndex(
    bool useMipmaps, uint8_t maxAnisotropy) const {
  if (impl_ == nullptr) {
    return 0u;
  }
  if (!useMipmaps) {
    return getSamplerBindlessIndex(impl_->bilinearSampler);
  }
  if (maxAnisotropy <= 1u || impl_->maxSamplerAnisotropy <= 1u) {
    return getSamplerBindlessIndex(impl_->trilinearSampler);
  }
  const uint8_t clamped = std::min(maxAnisotropy, impl_->maxSamplerAnisotropy);
  if (SamplerHandle handle =
          findBestAnisotropicSampler(impl_->anisotropicSamplers, clamped);
      nuri::isValid(handle)) {
    return getSamplerBindlessIndex(handle);
  }
  return getSamplerBindlessIndex(impl_->trilinearSampler);
}

uint32_t NvrhiGPUDevice::getDefaultSamplerBindlessIndex() const {
  return impl_ != nullptr ? getSamplerBindlessIndex(impl_->bilinearSampler)
                          : 0u;
}

uint32_t NvrhiGPUDevice::getCubemapSamplerBindlessIndex() const {
  return impl_ != nullptr ? getSamplerBindlessIndex(impl_->cubemapSampler) : 0u;
}

uint64_t NvrhiGPUDevice::getBufferDeviceAddress(BufferHandle h,
                                                size_t offset) const {
  if (impl_ == nullptr || (offset & 7u) != 0u) {
    return 0u;
  }
  const BufferResource *buffer = impl_->buffers.get(h);
  if (buffer == nullptr || !buffer->buffer) {
    return 0u;
  }
  return buffer->buffer->getGpuVirtualAddress() + offset;
}

bool NvrhiGPUDevice::resolveGeometry(GeometryAllocationHandle h,
                                     GeometryAllocationView &out) const {
  return impl_ != nullptr && impl_->geometryPool != nullptr
             ? impl_->geometryPool->resolve(h, out)
             : false;
}

uint64_t NvrhiGPUDevice::geometryMutationVersion() const {
  return impl_ != nullptr && impl_->geometryPool != nullptr
             ? impl_->geometryPool->mutationVersion()
             : 0u;
}

GpuTimingReport NvrhiGPUDevice::getLatestCompletedGpuTimingReport() const {
  if (impl_ == nullptr) {
    return {};
  }
  std::lock_guard lock(impl_->immediateMutex);
  return impl_->latestCompletedGpuTimingReport;
}

size_t NvrhiGPUDevice::drainCompletedGpuTimingReports(
    std::span<GpuTimingReport> outReports) {
  if (impl_ == nullptr || outReports.empty()) {
    return 0u;
  }
  std::lock_guard lock(impl_->immediateMutex);
  size_t count = 0u;
  while (count < outReports.size() &&
         !impl_->completedGpuTimingReports.empty()) {
    outReports[count++] = impl_->completedGpuTimingReports.front();
    impl_->completedGpuTimingReports.pop_front();
  }
  return count;
}

uint64_t NvrhiGPUDevice::droppedGpuTimingReportCount() const {
  if (impl_ == nullptr) {
    return 0u;
  }
  std::lock_guard lock(impl_->immediateMutex);
  return impl_->droppedGpuTimingReports;
}

Result<bool, std::string> NvrhiGPUDevice::beginFrame(uint64_t frameIndex) {
  if (impl_ == nullptr) {
    return Result<bool, std::string>::makeError("NVRHI device is null");
  }
  uint64_t frameResourceWaitInstance = 0u;
  size_t frameResourceSlot = std::numeric_limits<size_t>::max();
  {
    std::scoped_lock lock(impl_->immediateMutex, impl_->graphicsContextMutex);
    impl_->currentFrameIndex = frameIndex;
    impl_->hasPreparedSwapchainImage = false;
    impl_->preparedSwapchainImageWaitInstance = 0u;
    if (!impl_->frameResourceReuseWaitInstances.empty()) {
      frameResourceSlot = static_cast<size_t>(
          frameIndex % impl_->frameResourceReuseWaitInstances.size());
      frameResourceWaitInstance =
          impl_->frameResourceReuseWaitInstances[frameResourceSlot];
    }
  }
  Result<bool, std::string> waitResult =
      Result<bool, std::string>::makeResult(true);
  NURI_PROFILER_ZONE("NvrhiGPUDevice.frame_resource_reuse_wait",
                     NURI_PROFILER_COLOR_WAIT);
  waitResult = waitForGraphicsQueueInstance(
      *impl_, frameResourceWaitInstance,
      "NvrhiGPUDevice::beginFrame frame resource reuse");
  NURI_PROFILER_ZONE_END();
  if (waitResult.hasError()) {
    return waitResult;
  }
  {
    std::scoped_lock lock(impl_->immediateMutex, impl_->graphicsContextMutex);
    if (frameResourceSlot < impl_->frameResourceReuseWaitInstances.size() &&
        impl_->frameResourceReuseWaitInstances[frameResourceSlot] ==
            frameResourceWaitInstance) {
      impl_->frameResourceReuseWaitInstances[frameResourceSlot] = 0u;
    }
    NURI_PROFILER_ZONE("NvrhiGPUDevice.collect_background_copy_submissions",
                       NURI_PROFILER_COLOR_CMD_COPY);
    collectCompletedAsyncUploadSubmissions(*impl_);
    if (impl_->trimAsyncUploadCommandListPoolAfterTextureUploads) {
      impl_->availableAsyncUploadCommandLists.clear();
      bool textureUploadPending = impl_->pendingAsyncUploadTextureCount != 0u;
      for (const PendingAsyncUploadSubmission &pending :
           impl_->pendingAsyncUploadSubmissions) {
        textureUploadPending =
            textureUploadPending || pending.containsTextureData;
      }
      if (!textureUploadPending) {
        impl_->trimAsyncUploadCommandListPoolAfterTextureUploads = false;
      }
    }
    NURI_PROFILER_ZONE_END();
    const uint64_t completed = impl_->nvrhiDevice->queueGetCompletedInstance(
        nvrhi::CommandQueue::Graphics);
    const uint64_t completedCopy =
        impl_->hasDedicatedAssetCopyQueue
            ? impl_->nvrhiDevice->queueGetCompletedInstance(
                  nvrhi::CommandQueue::Copy)
            : completed;
    NURI_PROFILER_ZONE("NvrhiGPUDevice.collect_retired_resources",
                       NURI_PROFILER_COLOR_DESTROY);
    collectRetiredResources(*impl_, completed);
    collectCompletedSubmissionRecords(*impl_, completed, completedCopy);
    collectCompletedGraphicsCommandLists(*impl_, completed);
    impl_->nvrhiDevice->runGarbageCollection();
    NURI_PROFILER_ZONE_END();
    NURI_PROFILER_ZONE("NvrhiGPUDevice.collect_gpu_timing_submissions",
                       NURI_PROFILER_COLOR_CMD_COPY);
    collectCompletedGpuTimingSubmissions(*impl_);
    NURI_PROFILER_ZONE_END();
  }
  if (impl_->geometryPool != nullptr) {
    Result<bool, std::string> geometryPoolResult =
        Result<bool, std::string>::makeResult(true);
    NURI_PROFILER_ZONE("NvrhiGPUDevice.geometry_pool_begin_frame",
                       NURI_PROFILER_COLOR_WAIT);
    geometryPoolResult = impl_->geometryPool->beginFrame(frameIndex);
    NURI_PROFILER_ZONE_END();
    return geometryPoolResult;
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> NvrhiGPUDevice::prepareFrameOutput() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_WAIT);
  if (impl_ == nullptr || impl_->swapchain.handle == VK_NULL_HANDLE) {
    return Result<bool, std::string>::makeError("Swapchain is not initialized");
  }
  if (impl_->hasPreparedSwapchainImage) {
    return Result<bool, std::string>::makeResult(true);
  }

  impl_->semaphoreFrameIndex = static_cast<uint32_t>(impl_->currentFrameIndex %
                                                     kSwapchainFramesInFlight);
  if (impl_->semaphoreFrameIndex <
      impl_->frameSemaphoreReuseWaitInstances.size()) {
    const uint64_t waitInstance =
        impl_->frameSemaphoreReuseWaitInstances[impl_->semaphoreFrameIndex];
    Result<bool, std::string> waitResult =
        Result<bool, std::string>::makeResult(true);
    NURI_PROFILER_ZONE("NvrhiGPUDevice.frame_semaphore_reuse_wait",
                       NURI_PROFILER_COLOR_WAIT);
    waitResult =
        waitForGraphicsQueueInstance(*impl_, waitInstance,
                                     "NvrhiGPUDevice::prepareFrameOutput "
                                     "frame semaphore reuse");
    NURI_PROFILER_ZONE_END();
    if (waitResult.hasError()) {
      return waitResult;
    }
    impl_->frameSemaphoreReuseWaitInstances[impl_->semaphoreFrameIndex] = 0u;
  }

  VkResult result = VK_SUCCESS;
  {
    NURI_PROFILER_ZONE("NvrhiGPUDevice.acquire_swapchain_image",
                       NURI_PROFILER_COLOR_WAIT);
    result = vkAcquireNextImageKHR(
        impl_->device, impl_->swapchain.handle, UINT64_MAX,
        impl_->imageAvailableSemaphores[impl_->semaphoreFrameIndex],
        VK_NULL_HANDLE, &impl_->preparedSwapchainImageIndex);
    NURI_PROFILER_ZONE_END();
  }
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    Result<bool, std::string> resizeResult =
        Result<bool, std::string>::makeResult(true);
    NURI_PROFILER_ZONE("NvrhiGPUDevice.recreate_swapchain",
                       NURI_PROFILER_COLOR_WAIT);
    resizeResult = createSwapchain(*impl_);
    NURI_PROFILER_ZONE_END();
    if (resizeResult.hasError()) {
      return Result<bool, std::string>::makeError(resizeResult.error());
    }
    NURI_PROFILER_ZONE("NvrhiGPUDevice.acquire_swapchain_image_after_recreate",
                       NURI_PROFILER_COLOR_WAIT);
    result = vkAcquireNextImageKHR(
        impl_->device, impl_->swapchain.handle, UINT64_MAX,
        impl_->imageAvailableSemaphores[impl_->semaphoreFrameIndex],
        VK_NULL_HANDLE, &impl_->preparedSwapchainImageIndex);
    NURI_PROFILER_ZONE_END();
  }
  if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    return Result<bool, std::string>::makeError(
        vkError("NvrhiGPUDevice::prepareFrameOutput", result));
  }
  impl_->preparedSwapchainImageWaitInstance =
      impl_->preparedSwapchainImageIndex <
              impl_->swapchainImageReuseWaitInstances.size()
          ? impl_->swapchainImageReuseWaitInstances
                [impl_->preparedSwapchainImageIndex]
          : 0u;
  impl_->hasPreparedSwapchainImage = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<RecordingContextHandle, std::string>
NvrhiGPUDevice::acquireGraphicsRecordingContext(uint32_t workerIndex) {
  if (impl_ == nullptr || !impl_->nvrhiDevice) {
    return Result<RecordingContextHandle, std::string>::makeError(
        "NVRHI device is null");
  }
  std::lock_guard lock(impl_->graphicsContextMutex);
  const SlotReservation slot = impl_->recordingContextSlots.acquire();
  const uint32_t index = slot.index;
  RecordingContextHandle handle{
      .index = index,
      .generation = slot.generation,
  };
  if (impl_->activeGraphicsContexts.size() <= index) {
    impl_->activeGraphicsContexts.resize(static_cast<size_t>(index) + 1u);
  }

  nvrhi::CommandListHandle commandList{};
  if (!impl_->availableGraphicsCommandLists.empty()) {
    commandList = std::move(impl_->availableGraphicsCommandLists.back());
    impl_->availableGraphicsCommandLists.pop_back();
  } else {
    NURI_PROFILER_ZONE("NvrhiGPUDevice.create_command_list",
                       NURI_PROFILER_COLOR_CREATE);
    commandList = impl_->nvrhiDevice->createCommandList();
    NURI_PROFILER_ZONE_END();
  }
  if (!commandList) {
    impl_->recordingContextSlots.release(index);
    return Result<RecordingContextHandle, std::string>::makeError(
        "Failed to create command list");
  }
  {
    NURI_PROFILER_ZONE("NvrhiGPUDevice.open_command_list",
                       NURI_PROFILER_COLOR_CMD_COPY);
    commandList->open();
    NURI_PROFILER_ZONE_END();
  }
  impl_->activeGraphicsContexts[index] = ActiveGraphicsRecordingContext{
      .handle = handle,
      .commandList = commandList,
      .framebuffers = {},
      .timingQueries = {},
      .recordingSerial = impl_->recordingRetirement.beginRecording(),
      .workerIndex = workerIndex,
  };
  return Result<RecordingContextHandle, std::string>::makeResult(handle);
}

Result<bool, std::string> NvrhiGPUDevice::recordGraphicsBarriers(
    RecordingContextHandle ctx,
    std::span<const GraphicsBarrierRecord> barriers) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_BARRIER);
  if (barriers.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }
  nvrhi::CommandListHandle commandList{};
  {
    std::lock_guard lock(impl_->graphicsContextMutex);
    if (ActiveGraphicsRecordingContext *entry =
            findActiveGraphicsContextSlot(*impl_, ctx);
        entry != nullptr) {
      commandList = entry->commandList;
    }
  }
  if (!commandList) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::recordGraphicsBarriers: unknown recording context");
  }

  for (const GraphicsBarrierRecord &barrier : barriers) {
    if (barrier.isTexture()) {
      if (barrier.afterState == GraphicsBarrierState::Unknown) {
        continue;
      }
      TextureResource *texture = impl_->textures.get(barrier.textureHandle());
      if (texture == nullptr) {
        return Result<bool, std::string>::makeError(
            "NvrhiGPUDevice::recordGraphicsBarriers: invalid texture handle");
      }
      commandList->setTextureState(
          texture->texture.Get(), nvrhi::AllSubresources,
          toNvrhiTextureState(barrier.afterState, barrier.afterAccess,
                              isDepthFormat(texture->format)));
      continue;
    }
    BufferResource *buffer = impl_->buffers.get(barrier.bufferHandle());
    if (buffer == nullptr) {
      const BufferHandle invalidHandle = barrier.bufferHandle();
      const auto slot = impl_->buffers.debugSlotInfo(invalidHandle.index);
      const Impl::RetiredResource *retirement = nullptr;
      const BufferResource *retired =
          retiredBufferAtSlot(*impl_, invalidHandle.index, &retirement);
      const std::string_view slotState =
          !slot.inRange  ? std::string_view("out_of_range")
          : slot.live    ? std::string_view("live")
          : slot.retired ? std::string_view("retired")
                         : std::string_view("free");
      return Result<bool, std::string>::makeError(std::format(
          "NvrhiGPUDevice::recordGraphicsBarriers: "
          "[DEBUG-buflife-7c2a] invalid buffer handle index={} "
          "generation={} slot_state={} current_generation={} "
          "current_debug_name='{}' retired_debug_name='{}' "
          "retirement_recording_serial={} "
          "retirement_last_submitted={} retirement_last_use={} "
          "retirement_last_use_resolved={} latest_recording_serial={} "
          "latest_submitted={}",
          invalidHandle.index, invalidHandle.generation, slotState,
          slot.generation, slot.debugName,
          retired != nullptr ? std::string_view(retired->debugName)
                             : std::string_view{},
          retirement != nullptr ? retirement->requiredRecordingSerial : 0u,
          retirement != nullptr ? retirement->lastSubmittedAtDestruction : 0u,
          retirement != nullptr ? retirement->lastUseInstance : 0u,
          retirement != nullptr && retirement->lastUseResolved,
          impl_->recordingRetirement.latestIssuedSerial(),
          impl_->latestSubmittedInstance));
    }
    if (buffer->immutable) {
      if (barrier.afterState == GraphicsBarrierState::Read) {
        continue;
      }
      return Result<bool, std::string>::makeError(
          "NvrhiGPUDevice::recordGraphicsBarriers: immutable buffer cannot "
          "transition to a write or non-read state");
    }
    commandList->setBufferState(
        buffer->buffer.Get(),
        toNvrhiBufferState(barrier.afterState, barrier.afterAccess));
  }
  commandList->commitBarriers();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
NvrhiGPUDevice::recordGraphicsPass(RecordingContextHandle ctx,
                                   const RenderPass &pass) {
  std::lock_guard lock(impl_->graphicsContextMutex);
  ActiveGraphicsRecordingContext *entry =
      findActiveGraphicsContextSlot(*impl_, ctx);
  if (entry == nullptr || !entry->commandList) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::recordGraphicsPass: unknown recording context");
  }
  return recordRenderPass(*impl_, &entry->framebuffers, &entry->timingQueries,
                          *entry->commandList, pass);
}

Result<RecordedCommandBufferHandle, std::string>
NvrhiGPUDevice::finishGraphicsRecordingContext(RecordingContextHandle ctx) {
  std::lock_guard lock(impl_->graphicsContextMutex);
  ActiveGraphicsRecordingContext *active =
      findActiveGraphicsContextSlot(*impl_, ctx);
  if (active == nullptr) {
    return Result<RecordedCommandBufferHandle, std::string>::makeError(
        "NvrhiGPUDevice::finishGraphicsRecordingContext: unknown recording "
        "context");
  }

  {
    NURI_PROFILER_ZONE("NvrhiGPUDevice.close_command_list",
                       NURI_PROFILER_COLOR_CMD_COPY);
    active->commandList->close();
    NURI_PROFILER_ZONE_END();
  }
  const SlotReservation slot = impl_->recordedCommandBufferSlots.acquire();
  const uint32_t index = slot.index;
  RecordedCommandBufferHandle handle{
      .index = index,
      .generation = slot.generation,
  };
  impl_->recordedGraphicsCommandBuffers.push_back(RecordedGraphicsCommandBuffer{
      .handle = handle,
      .commandList = active->commandList,
      .framebuffers = std::move(active->framebuffers),
      .timingQueries = std::move(active->timingQueries),
      .recordingSerial = active->recordingSerial});
  impl_->activeGraphicsContexts[ctx.index] = ActiveGraphicsRecordingContext{};
  impl_->recordingContextSlots.release(ctx.index);
  return Result<RecordedCommandBufferHandle, std::string>::makeResult(handle);
}

Result<bool, std::string>
NvrhiGPUDevice::discardGraphicsRecordingContext(RecordingContextHandle ctx) {
  std::lock_guard lock(impl_->graphicsContextMutex);
  ActiveGraphicsRecordingContext *active =
      findActiveGraphicsContextSlot(*impl_, ctx);
  if (active == nullptr) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::discardGraphicsRecordingContext: unknown recording "
        "context");
  }
  NURI_ASSERT(
      impl_->recordingRetirement.resolveRecording(active->recordingSerial, 0u),
      "NvrhiGPUDevice::discardGraphicsRecordingContext: recording serial "
      "was already resolved");
  impl_->activeGraphicsContexts[ctx.index] = ActiveGraphicsRecordingContext{};
  impl_->recordingContextSlots.release(ctx.index);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> NvrhiGPUDevice::discardRecordedGraphicsCommandBuffer(
    RecordedCommandBufferHandle commandBuffer) {
  std::lock_guard lock(impl_->graphicsContextMutex);
  for (size_t i = 0u; i < impl_->recordedGraphicsCommandBuffers.size(); ++i) {
    if (!areSameHandle(impl_->recordedGraphicsCommandBuffers[i].handle,
                       commandBuffer)) {
      continue;
    }
    const uint64_t recordingSerial =
        impl_->recordedGraphicsCommandBuffers[i].recordingSerial;
    nvrhi::CommandListHandle reusableCommandList =
        std::move(impl_->recordedGraphicsCommandBuffers[i].commandList);
    if (i + 1u != impl_->recordedGraphicsCommandBuffers.size()) {
      impl_->recordedGraphicsCommandBuffers[i] =
          std::move(impl_->recordedGraphicsCommandBuffers.back());
    }
    impl_->availableGraphicsCommandLists.push_back(
        std::move(reusableCommandList));
    impl_->recordedCommandBufferSlots.release(commandBuffer.index);
    impl_->recordedGraphicsCommandBuffers.pop_back();
    NURI_ASSERT(
        impl_->recordingRetirement.resolveRecording(recordingSerial, 0u),
        "NvrhiGPUDevice::discardRecordedGraphicsCommandBuffer: recording "
        "serial was already resolved");
    return Result<bool, std::string>::makeResult(true);
  }
  return Result<bool, std::string>::makeError(
      "NvrhiGPUDevice::discardRecordedGraphicsCommandBuffer: unknown command "
      "buffer");
}

Result<SubmissionHandle, std::string>
NvrhiGPUDevice::submitRecordedGraphicsFrame(
    std::span<const RecordedCommandBufferHandle> commandBuffers,
    std::span<const SubmitBatchMeta> batches) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_SUBMIT);
  if (commandBuffers.empty()) {
    return Result<SubmissionHandle, std::string>::makeResult(
        SubmissionHandle{});
  }

  std::scoped_lock lock(impl_->immediateMutex, impl_->graphicsContextMutex);
  std::vector<uint8_t> presentFlags(commandBuffers.size(), 0u);
  for (const SubmitBatchMeta &batch : batches) {
    if (batch.commandBufferOffset > commandBuffers.size() ||
        batch.commandBufferCount >
            commandBuffers.size() - batch.commandBufferOffset) {
      return Result<SubmissionHandle, std::string>::makeError(
          "NvrhiGPUDevice::submitRecordedGraphicsFrame: submit batch range is "
          "out of bounds");
    }
    if (batch.presentsFrameOutput && batch.commandBufferCount > 0u) {
      presentFlags[batch.commandBufferOffset + batch.commandBufferCount - 1u] =
          1u;
    }
  }
  const bool wantsPresent = std::find(presentFlags.begin(), presentFlags.end(),
                                      1u) != presentFlags.end();
  if (wantsPresent && !impl_->hasPreparedSwapchainImage) {
    return Result<SubmissionHandle, std::string>::makeError(
        "NvrhiGPUDevice::submitRecordedGraphicsFrame: frame output was not "
        "prepared");
  }

  const auto foldHandle = [](RecordedCommandBufferHandle handle) -> uint64_t {
    return (static_cast<uint64_t>(handle.index) << 32u) | handle.generation;
  };
  std::unordered_map<uint64_t, size_t> recordedIndexByHandle;
  recordedIndexByHandle.reserve(impl_->recordedGraphicsCommandBuffers.size());
  for (size_t i = 0u; i < impl_->recordedGraphicsCommandBuffers.size(); ++i) {
    recordedIndexByHandle.emplace(
        foldHandle(impl_->recordedGraphicsCommandBuffers[i].handle), i);
  }

  std::vector<size_t> matchedIndices(commandBuffers.size());
  std::vector<nvrhi::ICommandList *> nvrhiCommandLists;
  nvrhiCommandLists.reserve(commandBuffers.size() + 3u);
  for (size_t i = 0u; i < commandBuffers.size(); ++i) {
    const auto found =
        recordedIndexByHandle.find(foldHandle(commandBuffers[i]));
    if (found == recordedIndexByHandle.end()) {
      return Result<SubmissionHandle, std::string>::makeError(
          "NvrhiGPUDevice::submitRecordedGraphicsFrame: unknown recorded "
          "command buffer");
    }
    matchedIndices[i] = found->second;
    nvrhiCommandLists.push_back(
        impl_->recordedGraphicsCommandBuffers[found->second].commandList.Get());
  }

  size_t timingQueryCount = 0u;
  for (const size_t matchedIndex : matchedIndices) {
    timingQueryCount += impl_->recordedGraphicsCommandBuffers[matchedIndex]
                            .timingQueries.size();
  }
  std::vector<NvrhiTimingQuery> timingQueries;
  timingQueries.reserve(timingQueryCount);
  for (const size_t matchedIndex : matchedIndices) {
    RecordedGraphicsCommandBuffer &recorded =
        impl_->recordedGraphicsCommandBuffers[matchedIndex];
    if (!recorded.timingQueries.empty()) {
      timingQueries.insert(
          timingQueries.end(),
          std::make_move_iterator(recorded.timingQueries.begin()),
          std::make_move_iterator(recorded.timingQueries.end()));
      recorded.timingQueries.clear();
    }
  }

  NvrhiWholeFrameTimingSlot wholeFrameTiming =
      acquireWholeFrameTimingSlot(*impl_);
  if (wholeFrameTiming.queryPool != VK_NULL_HANDLE) {
    nvrhiCommandLists.insert(nvrhiCommandLists.begin(),
                             wholeFrameTiming.commandLists[0].Get());
    nvrhiCommandLists.push_back(wholeFrameTiming.commandLists[1].Get());
  }

  if (wantsPresent) {
    if (impl_->semaphoreFrameIndex >= impl_->imageAvailableSemaphores.size() ||
        impl_->semaphoreFrameIndex >= impl_->renderFinishedSemaphores.size()) {
      return Result<SubmissionHandle, std::string>::makeError(
          "NvrhiGPUDevice::submitRecordedGraphicsFrame: invalid frame "
          "semaphore slot");
    }
    if (impl_->preparedSwapchainImageWaitInstance != 0u) {
      impl_->nvrhiDevice->queueWaitForSemaphore(
          nvrhi::CommandQueue::Graphics,
          impl_->nvrhiDevice->getQueueSemaphore(nvrhi::CommandQueue::Graphics),
          impl_->preparedSwapchainImageWaitInstance);
    }
    impl_->nvrhiDevice->queueWaitForSemaphore(
        nvrhi::CommandQueue::Graphics,
        impl_->imageAvailableSemaphores[impl_->semaphoreFrameIndex], 0u);
    impl_->nvrhiDevice->queueSignalSemaphore(
        nvrhi::CommandQueue::Graphics,
        impl_->renderFinishedSemaphores[impl_->semaphoreFrameIndex], 0u);
  }
  const bool frameUploadContainsTextureData =
      impl_->pendingAsyncUploadTextureCount != 0u;
  nvrhi::CommandListHandle frameUploadCommandList =
      takePendingAsyncUploadCommandList(*impl_);
  uint64_t frameCopyInstance = 0u;
  if (frameUploadCommandList) {
    if (impl_->hasDedicatedAssetCopyQueue) {
      frameCopyInstance = submitAsyncUploadCommandList(
          *impl_, frameUploadCommandList, frameUploadContainsTextureData);
      impl_->nvrhiDevice->queueWaitForSemaphore(
          nvrhi::CommandQueue::Graphics,
          impl_->nvrhiDevice->getQueueSemaphore(nvrhi::CommandQueue::Copy),
          frameCopyInstance);
    } else {
      const size_t insertIndex =
          wholeFrameTiming.queryPool != VK_NULL_HANDLE ? 1u : 0u;
      nvrhiCommandLists.insert(nvrhiCommandLists.begin() + insertIndex,
                               frameUploadCommandList.Get());
    }
  }
  uint64_t instance = 0u;
  {
    NURI_PROFILER_ZONE("NvrhiGPUDevice.execute_command_lists",
                       NURI_PROFILER_COLOR_SUBMIT);
    instance = impl_->nvrhiDevice->executeCommandLists(
        nvrhiCommandLists.data(), nvrhiCommandLists.size(),
        nvrhi::CommandQueue::Graphics);
    NURI_PROFILER_ZONE_END();
  }
  impl_->latestSubmittedInstance = instance;
  for (const size_t matchedIndex : matchedIndices) {
    const uint64_t recordingSerial =
        impl_->recordedGraphicsCommandBuffers[matchedIndex].recordingSerial;
    NURI_ASSERT(
        impl_->recordingRetirement.resolveRecording(recordingSerial, instance),
        "NvrhiGPUDevice::submitRecordedGraphicsFrame: recording serial was "
        "already resolved");
  }
  if (frameUploadCommandList && !impl_->hasDedicatedAssetCopyQueue) {
    impl_->latestAsyncUploadGraphicsInstance = instance;
    impl_->pendingAsyncUploadSubmissions.push_back(PendingAsyncUploadSubmission{
        .submissionInstance = instance,
        .commandList = std::move(frameUploadCommandList),
        .containsTextureData = frameUploadContainsTextureData,
        .queue = nvrhi::CommandQueue::Graphics,
    });
  }
  ++impl_->submittedFrameCount;
  if (!impl_->frameResourceReuseWaitInstances.empty()) {
    const size_t frameResourceSlot =
        static_cast<size_t>(impl_->currentFrameIndex %
                            impl_->frameResourceReuseWaitInstances.size());
    impl_->frameResourceReuseWaitInstances[frameResourceSlot] = instance;
  }
  if (!timingQueries.empty() || wholeFrameTiming.queryPool != VK_NULL_HANDLE) {
    impl_->pendingGpuTimingSubmissions.push_back(PendingGpuTimingSubmission{
        .frameIndex = impl_->currentFrameIndex,
        .submissionInstance = instance,
        .timingQueries = std::move(timingQueries),
        .wholeFrameTiming = std::move(wholeFrameTiming),
    });
  }

  if (wantsPresent) {
    const VkSwapchainKHR swapchain = impl_->swapchain.handle;
    const uint32_t imageIndex = impl_->preparedSwapchainImageIndex;
    if (imageIndex < impl_->swapchainImageReuseWaitInstances.size()) {
      impl_->swapchainImageReuseWaitInstances[imageIndex] = instance;
    }
    if (impl_->semaphoreFrameIndex <
        impl_->frameSemaphoreReuseWaitInstances.size()) {
      impl_->frameSemaphoreReuseWaitInstances[impl_->semaphoreFrameIndex] =
          instance;
    }
    const VkSemaphore renderFinished =
        impl_->renderFinishedSemaphores[impl_->semaphoreFrameIndex];
    const VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1u,
        .pWaitSemaphores = &renderFinished,
        .swapchainCount = 1u,
        .pSwapchains = &swapchain,
        .pImageIndices = &imageIndex,
    };
    VkResult result = VK_SUCCESS;
    {
      NURI_PROFILER_ZONE("NvrhiGPUDevice.queue_present",
                         NURI_PROFILER_COLOR_PRESENT);
      result = vkQueuePresentKHR(impl_->graphicsQueue, &presentInfo);
      NURI_PROFILER_ZONE_END();
    }
    impl_->hasPreparedSwapchainImage = false;
    impl_->preparedSwapchainImageWaitInstance = 0u;
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
      auto resizeResult = createSwapchain(*impl_);
      if (resizeResult.hasError()) {
        return Result<SubmissionHandle, std::string>::makeError(
            resizeResult.error());
      }
    } else if (result != VK_SUCCESS) {
      return Result<SubmissionHandle, std::string>::makeError(vkError(
          "NvrhiGPUDevice::submitRecordedGraphicsFrame present", result));
    }
  }

  std::sort(matchedIndices.begin(), matchedIndices.end(), std::greater<>());
  for (const size_t index : matchedIndices) {
    if (index >= impl_->recordedGraphicsCommandBuffers.size()) {
      continue;
    }
    RecordedGraphicsCommandBuffer &recorded =
        impl_->recordedGraphicsCommandBuffers[index];
    const uint32_t handleIndex = recorded.handle.index;
    impl_->pendingGraphicsCommandLists.push_back(PendingGraphicsCommandList{
        .submissionInstance = instance,
        .commandList = std::move(recorded.commandList),
        .framebuffers = std::move(recorded.framebuffers),
    });
    if (index + 1u != impl_->recordedGraphicsCommandBuffers.size()) {
      impl_->recordedGraphicsCommandBuffers[index] =
          std::move(impl_->recordedGraphicsCommandBuffers.back());
    }
    impl_->recordedGraphicsCommandBuffers.pop_back();
    impl_->recordedCommandBufferSlots.release(handleIndex);
  }

  return allocateSubmissionHandle(*impl_, instance, frameCopyInstance);
}

bool NvrhiGPUDevice::isSubmissionComplete(SubmissionHandle handle) const {
  if (impl_ == nullptr || handle.generation == 0u) {
    return true;
  }
  for (const SubmissionRecord &record : impl_->submissions) {
    if (areSameHandle(record.handle, handle)) {
      const std::optional<uint64_t> resolvedGraphics =
          resolveSubmissionInstance(*impl_, record);
      if (!resolvedGraphics.has_value() ||
          impl_->nvrhiDevice->queueGetCompletedInstance(
              nvrhi::CommandQueue::Graphics) < *resolvedGraphics) {
        return false;
      }
      return record.copyInstance == 0u ||
             (impl_->hasDedicatedAssetCopyQueue &&
              impl_->nvrhiDevice->queueGetCompletedInstance(
                  nvrhi::CommandQueue::Copy) >= record.copyInstance);
    }
  }
  return true;
}

Result<bool, std::string>
NvrhiGPUDevice::makeSubmissionVisibleToGraphics(SubmissionHandle handle) {
  if (impl_ == nullptr || handle.generation == 0u) {
    return Result<bool, std::string>::makeResult(true);
  }
  std::lock_guard lock(impl_->immediateMutex);
  for (SubmissionRecord &record : impl_->submissions) {
    if (!areSameHandle(record.handle, handle)) {
      continue;
    }
    if (!record.requiresGraphicsVisibility || record.copyInstance == 0u ||
        record.graphicsVisibilityQueued) {
      return Result<bool, std::string>::makeResult(true);
    }
    if (!impl_->hasDedicatedAssetCopyQueue) {
      return Result<bool, std::string>::makeError(
          "NvrhiGPUDevice::makeSubmissionVisibleToGraphics: copy submission "
          "has no copy queue");
    }
    const uint64_t completedCopy =
        impl_->nvrhiDevice->queueGetCompletedInstance(
            nvrhi::CommandQueue::Copy);
    if (completedCopy < record.copyInstance) {
      return Result<bool, std::string>::makeResult(false);
    }
    impl_->nvrhiDevice->queueWaitForSemaphore(
        nvrhi::CommandQueue::Graphics,
        impl_->nvrhiDevice->getQueueSemaphore(nvrhi::CommandQueue::Copy),
        record.copyInstance);
    record.graphicsVisibilityQueued = true;
    return Result<bool, std::string>::makeResult(true);
  }
  // Completed graphics-only records may already have been collected.
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> NvrhiGPUDevice::submitComputeDispatches(
    std::span<const ComputeDispatchItem> dispatches) {
  if (dispatches.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }
  std::lock_guard lock(impl_->immediateMutex);
  flushPendingAsyncUploadCommandList(*impl_);
  impl_->immediateCommandList->open();
  auto result =
      recordComputeDispatches(*impl_, *impl_->immediateCommandList, dispatches,
                              "NvrhiGPUDevice::submitComputeDispatches",
                              /*dependencyStatesPreplanned=*/false);
  impl_->immediateCommandList->close();
  if (result.hasError()) {
    return result;
  }
  executeCommandListAndWait(*impl_, impl_->immediateCommandList);
  return Result<bool, std::string>::makeResult(true);
}

Result<GeometryAllocationHandle, std::string> NvrhiGPUDevice::allocateGeometry(
    std::span<const std::byte> vertexBytes, uint32_t vertexCount,
    std::span<const std::byte> indexBytes, uint32_t indexCount,
    std::string_view debugName) {
  if (impl_ == nullptr || impl_->geometryPool == nullptr) {
    return Result<GeometryAllocationHandle, std::string>::makeError(
        "Geometry pool is not initialized");
  }
  return impl_->geometryPool->allocate(vertexBytes, vertexCount, indexBytes,
                                       indexCount, debugName);
}

Result<GeometryAllocationHandle, std::string>
NvrhiGPUDevice::adoptPreparedGeometry(BufferHandle vertexBuffer,
                                      size_t vertexBytes, uint32_t vertexCount,
                                      BufferHandle indexBuffer,
                                      size_t indexBytes, uint32_t indexCount,
                                      std::string_view debugName) {
  if (impl_ == nullptr || impl_->geometryPool == nullptr) {
    return Result<GeometryAllocationHandle, std::string>::makeError(
        "Geometry pool is not initialized");
  }
  return impl_->geometryPool->adoptPrepared(vertexBuffer, vertexBytes,
                                            vertexCount, indexBuffer,
                                            indexBytes, indexCount, debugName);
}

void NvrhiGPUDevice::releaseGeometry(GeometryAllocationHandle h) {
  if (impl_ != nullptr && impl_->geometryPool != nullptr) {
    impl_->geometryPool->release(h);
  }
}

Result<SubmissionHandle, std::string>
NvrhiGPUDevice::submitBackgroundBufferCopies(
    std::span<const BufferCopyRegion> regions, std::string_view) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
  if (regions.empty()) {
    return Result<SubmissionHandle, std::string>::makeResult(
        SubmissionHandle{});
  }

  size_t copyCount = 0u;
  for (const BufferCopyRegion &region : regions) {
    if (region.size == 0u) {
      continue;
    }
    const BufferResource *src = impl_->buffers.get(region.srcBuffer);
    const BufferResource *dst = impl_->buffers.get(region.dstBuffer);
    if (src == nullptr || dst == nullptr) {
      return Result<SubmissionHandle, std::string>::makeError(
          "NvrhiGPUDevice::submitBackgroundBufferCopies: invalid buffer "
          "handle");
    }
    if (region.srcOffset > src->byteSize ||
        region.size > src->byteSize - region.srcOffset) {
      return Result<SubmissionHandle, std::string>::makeError(
          "NvrhiGPUDevice::submitBackgroundBufferCopies: source copy range is "
          "out of bounds");
    }
    if (region.dstOffset > dst->byteSize ||
        region.size > dst->byteSize - region.dstOffset) {
      return Result<SubmissionHandle, std::string>::makeError(
          "NvrhiGPUDevice::submitBackgroundBufferCopies: destination copy "
          "range is out of bounds");
    }
    if (dst->immutable) {
      return Result<SubmissionHandle, std::string>::makeError(
          "NvrhiGPUDevice::submitBackgroundBufferCopies: destination buffer "
          "is immutable");
    }
    ++copyCount;
  }
  if (copyCount == 0u) {
    return Result<SubmissionHandle, std::string>::makeResult(
        SubmissionHandle{});
  }

  std::lock_guard lock(impl_->immediateMutex);
  collectCompletedAsyncUploadSubmissions(*impl_);
  flushPendingAsyncUploadCommandList(*impl_);
  impl_->nvrhiDevice->runGarbageCollection();
  nvrhi::CommandListHandle commandList = acquireAsyncUploadCommandList(*impl_);
  if (!commandList) {
    return Result<SubmissionHandle, std::string>::makeError(
        "NvrhiGPUDevice::submitBackgroundBufferCopies: failed to create "
        "command list");
  }

  commandList->open();
  for (const BufferCopyRegion &region : regions) {
    if (region.size == 0u) {
      continue;
    }
    BufferResource *src = impl_->buffers.get(region.srcBuffer);
    BufferResource *dst = impl_->buffers.get(region.dstBuffer);
    NURI_ASSERT(src != nullptr && dst != nullptr,
                "NvrhiGPUDevice::submitBackgroundBufferCopies: validated "
                "buffer disappeared");
    commandList->copyBuffer(dst->buffer.Get(), region.dstOffset,
                            src->buffer.Get(), region.srcOffset, region.size);
  }
  commandList->close();
  const uint64_t instance =
      submitAsyncUploadCommandList(*impl_, commandList, false);
  return impl_->hasDedicatedAssetCopyQueue
             ? allocateSubmissionHandle(*impl_, 0u, instance, 0u, true)
             : allocateSubmissionHandle(*impl_, instance);
}

Result<SubmissionHandle, std::string> NvrhiGPUDevice::submitPendingUploads() {
  if (impl_ == nullptr || !impl_->nvrhiDevice) {
    return Result<SubmissionHandle, std::string>::makeError(
        "NVRHI device is not initialized");
  }
  std::lock_guard lock(impl_->immediateMutex);
  collectCompletedAsyncUploadSubmissions(*impl_);
  const bool containsTextureData = impl_->pendingAsyncUploadTextureCount != 0u;
  nvrhi::CommandListHandle commandList =
      takePendingAsyncUploadCommandList(*impl_);
  if (!commandList) {
    if (impl_->latestAsyncUploadGraphicsInstance == 0u &&
        impl_->latestAsyncUploadCopyInstance == 0u) {
      return Result<SubmissionHandle, std::string>::makeResult(
          SubmissionHandle{});
    }
    return allocateSubmissionHandle(
        *impl_, impl_->latestAsyncUploadGraphicsInstance,
        impl_->latestAsyncUploadCopyInstance, 0u, true);
  }
  const uint64_t instance =
      submitAsyncUploadCommandList(*impl_, commandList, containsTextureData);
  return impl_->hasDedicatedAssetCopyQueue
             ? allocateSubmissionHandle(*impl_, 0u, instance, 0u, true)
             : allocateSubmissionHandle(*impl_, instance);
}

bool NvrhiGPUDevice::assetUploadsUseDedicatedCopyQueue() const noexcept {
  return impl_ != nullptr && impl_->hasDedicatedAssetCopyQueue;
}

Result<SubmissionHandle, std::string> NvrhiGPUDevice::captureWorkCompletion() {
  if (impl_ == nullptr || !impl_->nvrhiDevice) {
    return Result<SubmissionHandle, std::string>::makeError(
        "NVRHI device is not initialized");
  }
  std::scoped_lock lock(impl_->immediateMutex, impl_->graphicsContextMutex);
  flushPendingAsyncUploadCommandList(*impl_, false);
  return allocateSubmissionHandle(
      *impl_, impl_->latestSubmittedInstance,
      impl_->latestAsyncUploadCopyInstance,
      impl_->recordingRetirement.latestIssuedSerial());
}

Result<bool, std::string>
NvrhiGPUDevice::updateBuffer(BufferHandle bufferHandle,
                             std::span<const std::byte> data, size_t offset) {
  const BufferUpdate update{
      .buffer = bufferHandle,
      .data = data,
      .offset = offset,
  };
  return updateBuffers(std::span<const BufferUpdate>(&update, 1u));
}

Result<bool, std::string>
NvrhiGPUDevice::updateBuffers(std::span<const BufferUpdate> updates) {
  size_t deviceLocalUpdateCount = 0u;
  for (const BufferUpdate &update : updates) {
    if (update.data.empty()) {
      continue;
    }
    BufferResource *buffer = impl_->buffers.get(update.buffer);
    if (buffer == nullptr || update.offset > buffer->byteSize ||
        update.data.size() > buffer->byteSize - update.offset) {
      return Result<bool, std::string>::makeError(
          "NvrhiGPUDevice::updateBuffers: invalid buffer range");
    }
    if (buffer->immutable) {
      return Result<bool, std::string>::makeError(
          "NvrhiGPUDevice::updateBuffers: buffer is immutable");
    }
    deviceLocalUpdateCount += buffer->mapped == nullptr ? 1u : 0u;
  }

  if (deviceLocalUpdateCount == 0u) {
    for (const BufferUpdate &update : updates) {
      if (update.data.empty()) {
        continue;
      }
      BufferResource *buffer = impl_->buffers.get(update.buffer);
      NURI_ASSERT(
          buffer != nullptr && buffer->mapped != nullptr,
          "NvrhiGPUDevice::updateBuffers: validated buffer disappeared");
      std::memcpy(buffer->mapped + update.offset, update.data.data(),
                  update.data.size());
      flushMappedBufferRange(*impl_, *buffer, update.offset,
                             update.data.size());
    }
    return Result<bool, std::string>::makeResult(true);
  }

  std::lock_guard lock(impl_->immediateMutex);
  collectCompletedAsyncUploadSubmissions(*impl_);
  nvrhi::CommandListHandle &commandList =
      ensurePendingAsyncUploadCommandList(*impl_);
  if (!commandList) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::updateBuffers: failed to create command list");
  }

  for (const BufferUpdate &update : updates) {
    if (update.data.empty()) {
      continue;
    }
    BufferResource *buffer = impl_->buffers.get(update.buffer);
    NURI_ASSERT(buffer != nullptr,
                "NvrhiGPUDevice::updateBuffers: validated buffer disappeared");
    if (buffer->mapped != nullptr) {
      std::memcpy(buffer->mapped + update.offset, update.data.data(),
                  update.data.size());
      flushMappedBufferRange(*impl_, *buffer, update.offset,
                             update.data.size());
      continue;
    }
    commandList->writeBuffer(buffer->buffer.Get(), update.data.data(),
                             update.data.size(), update.offset);
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
NvrhiGPUDevice::readBuffer(BufferHandle bufferHandle, size_t offset,
                           std::span<std::byte> outBytes) {
  if (outBytes.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }
  BufferResource *buffer = impl_->buffers.get(bufferHandle);
  if (buffer == nullptr || offset + outBytes.size() > buffer->byteSize) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::readBuffer: invalid buffer range");
  }
  if (buffer->mapped != nullptr) {
    std::memcpy(outBytes.data(), buffer->mapped + offset, outBytes.size());
    return Result<bool, std::string>::makeResult(true);
  }

  nvrhi::BufferDesc readbackDesc{};
  readbackDesc.setByteSize(static_cast<uint64_t>(outBytes.size()))
      .setCpuAccess(nvrhi::CpuAccessMode::Read)
      .setDebugName("NVRHI buffer readback");
  nvrhi::BufferHandle readback = impl_->nvrhiDevice->createBuffer(readbackDesc);
  if (!readback) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::readBuffer: failed to create readback buffer");
  }

  std::lock_guard lock(impl_->immediateMutex);
  flushPendingAsyncUploadCommandList(*impl_);
  impl_->immediateCommandList->open();
  impl_->immediateCommandList->copyBuffer(
      readback.Get(), 0u, buffer->buffer.Get(), offset, outBytes.size());
  impl_->immediateCommandList->close();
  executeCommandListAndWait(*impl_, impl_->immediateCommandList);

  void *mapped =
      impl_->nvrhiDevice->mapBuffer(readback.Get(), nvrhi::CpuAccessMode::Read);
  if (mapped == nullptr) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::readBuffer: failed to map readback buffer");
  }
  std::memcpy(outBytes.data(), mapped, outBytes.size());
  impl_->nvrhiDevice->unmapBuffer(readback.Get());
  return Result<bool, std::string>::makeResult(true);
}

std::byte *NvrhiGPUDevice::getMappedBufferPtr(BufferHandle bufferHandle) {
  BufferResource *buffer =
      impl_ != nullptr ? impl_->buffers.get(bufferHandle) : nullptr;
  return buffer != nullptr ? buffer->mapped : nullptr;
}

void NvrhiGPUDevice::flushMappedBuffer(BufferHandle bufferHandle, size_t offset,
                                       size_t size) {
  if (impl_ == nullptr) {
    return;
  }
  BufferResource *buffer = impl_->buffers.get(bufferHandle);
  if (buffer == nullptr) {
    return;
  }
  flushMappedBufferRange(*impl_, *buffer, offset, size);
}

Result<bool, std::string>
NvrhiGPUDevice::readTexture(TextureHandle textureHandle,
                            const TextureReadbackRegion &region,
                            std::span<std::byte> outBytes) {
  TextureResource *texture = impl_->textures.get(textureHandle);
  if (texture == nullptr || !texture->texture) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::readTexture: invalid texture handle");
  }
  if (region.width == 0u || region.height == 0u) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::readTexture: readback region size must be non-zero");
  }
  const size_t bytesPerPixel = textureReadbackBytesPerPixel(texture->format);
  if (bytesPerPixel == 0u) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::readTexture: unsupported texture format for "
        "readback");
  }
  const nvrhi::TextureDesc &textureDesc = texture->texture->getDesc();
  if (region.mipLevel >= textureDesc.mipLevels ||
      region.layer >= textureDesc.arraySize) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::readTexture: invalid subresource");
  }
  const uint32_t mipWidth =
      mipSize(texture->desc.dimensions.width, region.mipLevel);
  const uint32_t mipHeight =
      mipSize(texture->desc.dimensions.height, region.mipLevel);
  if (region.x >= mipWidth || region.y >= mipHeight) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::readTexture: readback origin is out of bounds");
  }
  if (region.width > mipWidth - region.x ||
      region.height > mipHeight - region.y) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::readTexture: readback region is out of bounds");
  }
  const uint64_t requiredBytes64 = static_cast<uint64_t>(region.width) *
                                   static_cast<uint64_t>(region.height) *
                                   static_cast<uint64_t>(bytesPerPixel);
  if (requiredBytes64 > std::numeric_limits<size_t>::max()) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::readTexture: readback size overflows size_t");
  }
  const size_t rowPitch = static_cast<size_t>(region.width) * bytesPerPixel;
  const size_t requiredBytes = static_cast<size_t>(requiredBytes64);
  if (outBytes.size() < requiredBytes) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::readTexture: output buffer is too small");
  }

  nvrhi::TextureDesc stagingDesc = textureDesc;
  stagingDesc.setWidth(region.width)
      .setHeight(region.height)
      .setDepth(1u)
      .setArraySize(1u)
      .setMipLevels(1u)
      .setSampleCount(1u)
      .setDebugName("NVRHI texture readback staging");
  nvrhi::StagingTextureHandle staging =
      impl_->nvrhiDevice->createStagingTexture(stagingDesc,
                                               nvrhi::CpuAccessMode::Read);
  if (!staging) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::readTexture: failed to create staging texture");
  }

  nvrhi::TextureSlice srcSlice{};
  srcSlice.setOrigin(region.x, region.y, 0u)
      .setSize(region.width, region.height, 1u)
      .setMipLevel(region.mipLevel)
      .setArraySlice(region.layer);
  nvrhi::TextureSlice dstSlice{};
  dstSlice.setSize(region.width, region.height, 1u);

  std::lock_guard lock(impl_->immediateMutex);
  flushPendingAsyncUploadCommandList(*impl_);
  impl_->immediateCommandList->open();
  impl_->immediateCommandList->copyTexture(staging.Get(), dstSlice,
                                           texture->texture.Get(), srcSlice);
  impl_->immediateCommandList->close();
  executeCommandListAndWait(*impl_, impl_->immediateCommandList);

  size_t mappedRowPitch = 0u;
  void *mapped = impl_->nvrhiDevice->mapStagingTexture(
      staging.Get(), dstSlice, nvrhi::CpuAccessMode::Read, &mappedRowPitch);
  if (mapped == nullptr) {
    return Result<bool, std::string>::makeError(
        "NvrhiGPUDevice::readTexture: failed to map staging texture");
  }
  const auto *src = static_cast<const std::byte *>(mapped);
  for (uint32_t y = 0u; y < region.height; ++y) {
    std::memcpy(outBytes.data() + y * rowPitch, src + y * mappedRowPitch,
                rowPitch);
  }
  impl_->nvrhiDevice->unmapStagingTexture(staging.Get());
  return Result<bool, std::string>::makeResult(true);
}

void NvrhiGPUDevice::waitIdle() {
  if (impl_ != nullptr && impl_->nvrhiDevice) {
    std::scoped_lock lock(impl_->immediateMutex, impl_->graphicsContextMutex);
    flushPendingAsyncUploadCommandList(*impl_);
    impl_->nvrhiDevice->waitForIdle();
    collectCompletedAsyncUploadSubmissions(*impl_);
    releaseAllRetiredResourcesAfterIdle(*impl_);
    const uint64_t completed = impl_->nvrhiDevice->queueGetCompletedInstance(
        nvrhi::CommandQueue::Graphics);
    const uint64_t completedCopy =
        impl_->hasDedicatedAssetCopyQueue
            ? impl_->nvrhiDevice->queueGetCompletedInstance(
                  nvrhi::CommandQueue::Copy)
            : completed;
    collectCompletedSubmissionRecords(*impl_, completed, completedCopy);
    collectCompletedGraphicsCommandLists(*impl_, completed);
    impl_->nvrhiDevice->runGarbageCollection();
    collectCompletedGpuTimingSubmissions(*impl_);
  }
}

} // namespace nuri
