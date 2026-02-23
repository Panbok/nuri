#pragma once

#include "nuri/gfx/gpu_types.h"

#include <cstddef>
#include <span>
#include <string_view>

namespace nuri {

// Public render-graph contract: a submit item may depend on up to 4 buffers.
// Backends can impose stricter limits, but higher-level layers should respect
// this cap to avoid backend-specific failures.
constexpr size_t kMaxDependencyBuffers = 4;

struct Viewport {
  float x = 0.0f;
  float y = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  float minDepth = 0.0f;
  float maxDepth = 1.0f;
};

struct RectU32 {
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t width = 0;
  uint32_t height = 0;
};

struct ClearColor {
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  float a = 1.0f;
};

struct AttachmentColor {
  LoadOp loadOp = LoadOp::Clear;
  StoreOp storeOp = StoreOp::Store;
  ClearColor clearColor{};
};

struct AttachmentDepth {
  LoadOp loadOp = LoadOp::Clear;
  StoreOp storeOp = StoreOp::Store;
  float clearDepth = 1.0f;
  uint32_t clearStencil = 0;
};

struct DispatchSize {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

struct ComputeDispatchItem {
  ComputePipelineHandle pipeline{};
  DispatchSize dispatch{};
  std::span<const std::byte> pushConstants{};
  std::span<const BufferHandle> dependencyBuffers{};
  std::string_view debugLabel{};
  uint32_t debugColor = 0xffffffffu;
};

enum class DrawCommandType : uint8_t {
  Direct,
  IndexedIndirect,
  IndexedIndirectCount,
};

struct DrawItem {
  DrawCommandType command = DrawCommandType::Direct;
  RenderPipelineHandle pipeline{};
  BufferHandle vertexBuffer{};
  uint64_t vertexBufferOffset = 0;
  BufferHandle indexBuffer{};
  uint64_t indexBufferOffset = 0;
  BufferHandle indirectBuffer{};
  uint64_t indirectBufferOffset = 0;
  BufferHandle indirectCountBuffer{};
  uint64_t indirectCountBufferOffset = 0;
  uint32_t indirectDrawCount = 0;
  uint32_t indirectStride = 0;
  IndexFormat indexFormat = IndexFormat::U32;
  uint32_t vertexCount = 0;
  uint32_t indexCount = 0;
  uint32_t instanceCount = 1;
  uint32_t firstVertex = 0;
  uint32_t firstIndex = 0;
  int32_t vertexOffset = 0;
  uint32_t firstInstance = 0;
  bool useScissor = false;
  RectU32 scissor{};
  bool useDepthState = false;
  DepthState depthState{};
  bool depthBiasEnable = false;
  float depthBiasConstant = 0.0f;
  float depthBiasSlope = 0.0f;
  float depthBiasClamp = 0.0f;
  std::span<const std::byte> pushConstants{};
  std::string_view debugLabel{};
  uint32_t debugColor = 0xffffffffu;
};

struct RenderPass {
  AttachmentColor color;
  TextureHandle colorTexture{};
  AttachmentDepth depth;
  TextureHandle depthTexture{};
  bool useViewport = false;
  Viewport viewport{};
  std::span<const ComputeDispatchItem> preDispatches{};
  std::span<const BufferHandle> dependencyBuffers{};
  std::span<const DrawItem> draws{};
  std::string_view debugLabel{};
  uint32_t debugColor = 0xffffffffu;
};

struct RenderFrame {
  std::span<const RenderPass> passes{};
};

} // namespace nuri
