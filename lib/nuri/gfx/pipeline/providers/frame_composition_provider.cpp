#include "nuri/pch.h"

#include "nuri/gfx/pipeline/providers/frame_composition_provider.h"

#include "nuri/core/profiling.h"

namespace nuri {
namespace {

struct SceneColorMipRingSpec {
  uint32_t mipLevel = 0u;
  std::string_view debugNameBase{};
};

constexpr std::array<SceneColorMipRingSpec, kFrameCompositionSceneColorMipCount>
    kSceneColorMipRingSpecs{{
        {.mipLevel = 0u, .debugNameBase = "frame_scene_color"},
        {.mipLevel = 1u, .debugNameBase = "frame_scene_color_half"},
        {.mipLevel = 2u, .debugNameBase = "frame_scene_color_quarter"},
    }};

uint32_t levelDimensions(uint32_t base, uint32_t mipLevel) {
  return std::max(1u, base >> std::min(mipLevel, 31u));
}

TextureDesc makeTextureDesc(Format format, uint32_t width, uint32_t height,
                            TextureUsage usage) {
  return TextureDesc{
      .type = TextureType::Texture2D,
      .format = format,
      .dimensions = {.width = width, .height = height, .depth = 1u},
      .usage = usage,
      .storage = Storage::Device,
      .numLayers = 1u,
      .numSamples = 1u,
      .numMipLevels = 1u,
      .data = {},
      .dataNumMipLevels = 1u,
      .generateMipmaps = false,
  };
}

uint32_t textureBytesPerPixel(Format format) {
  switch (format) {
  case Format::RG16_FLOAT:
    return sizeof(uint16_t) * 2u;
  case Format::RG32_FLOAT:
    return sizeof(float) * 2u;
  case Format::R32_UINT:
  case Format::R32_FLOAT:
  case Format::D32_FLOAT:
    return sizeof(uint32_t);
  case Format::D16_UNORM:
    return sizeof(uint16_t);
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
  case Format::RGBA8_UINT:
    return 4u;
  case Format::RGBA16_FLOAT:
    return 8u;
  case Format::RGBA32_FLOAT:
    return 16u;
  case Format::BC7_RGBA_UNORM:
  case Format::BC7_RGBA_SRGB:
  case Format::ETC2_RGB8_UNORM:
  case Format::ETC2_RGB8_SRGB:
  case Format::Count:
    break;
  }
  return 0u;
}

} // namespace

FrameCompositionProvider::FrameCompositionProvider(
    GPUDevice &gpu, std::pmr::memory_resource *memory)
    : gpu_(gpu),
      memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      sceneColorMipTextures_{
          TextureRing(memory_),
          TextureRing(memory_),
          TextureRing(memory_),
      },
      frameColorTextures_(memory_), sceneDepthTextures_(memory_),
      motionVectorTextures_(memory_) {}

FrameCompositionProvider::~FrameCompositionProvider() {
  for (auto &textures : sceneColorMipTextures_) {
    destroyTextures(textures);
  }
  destroyTextures(frameColorTextures_);
  destroyTextures(sceneDepthTextures_);
  destroyHistoryTextures();
  destroyMotionVectorTextures();
}

Result<bool, std::string>
FrameCompositionProvider::prepare(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  ctx.shared.textureRequirements |= kBaselineFrameTextureRequirements;
  ctx.shared.sceneColorGraphTexture = {};
  ctx.shared.frameColorGraphTexture = {};
  ctx.shared.sceneDepthGraphTexture = {};
  ctx.shared.sceneDepthPyramidGraphTextures = {};
  ctx.shared.motionVectorGraphTexture = {};
  ctx.shared.previousMotionVectorGraphTexture = {};

  auto ensureResult = ensureTextures(ctx.shared.textureRequirements);
  if (ensureResult.hasError()) {
    return ensureResult;
  }

  ctx.shared.sceneColorTexture =
      currentRingTexture(sceneColorMipTextures_[0], ctx.frame.frameIndex);
  ctx.shared.sceneColorHalfResTexture =
      currentRingTexture(sceneColorMipTextures_[1], ctx.frame.frameIndex);
  ctx.shared.sceneColorQuarterResTexture =
      currentRingTexture(sceneColorMipTextures_[2], ctx.frame.frameIndex);
  ctx.shared.frameColorTexture =
      currentRingTexture(frameColorTextures_, ctx.frame.frameIndex);
  ctx.shared.sceneDepthTexture =
      currentRingTexture(sceneDepthTextures_, ctx.frame.frameIndex);
  ctx.shared.historyColorReadTexture =
      historyColorTextures_[(ctx.frame.frameIndex + 1u) & 1u];
  ctx.shared.historyColorWriteTexture =
      historyColorTextures_[ctx.frame.frameIndex & 1u];
  ctx.shared.motionVectorTexture =
      currentRingTexture(motionVectorTextures_, ctx.frame.frameIndex);
  ctx.shared.previousMotionVectorTexture =
      ctx.frame.camera.historyValid
          ? previousRingTexture(motionVectorTextures_, ctx.frame.frameIndex)
          : TextureHandle{};
  AntiAliasingFrameMetrics &aaMetrics = ctx.frame.metrics.antiAliasing;
  aaMetrics.motionVectorFormat = kFrameCompositionMotionVectorFormat;
  aaMetrics.motionVectorWidth = framebufferWidth_;
  aaMetrics.motionVectorHeight = framebufferHeight_;
  aaMetrics.motionVectorTextureCount =
      static_cast<uint32_t>(motionVectorTextures_.size());
  aaMetrics.motionVectorAllocationCount = motionVectorAllocationCount_;
  aaMetrics.motionVectorReallocationCount = motionVectorReallocationCount_;
  aaMetrics.motionVectorRg32FallbackCount = 0u;
  aaMetrics.motionVectorAllocated =
      nuri::isValid(ctx.shared.motionVectorTexture);
  aaMetrics.previousMotionVectorValid =
      nuri::isValid(ctx.shared.previousMotionVectorTexture);
  aaMetrics.motionVectorFormatSupported =
      kFrameCompositionMotionVectorFormat == Format::RG16_FLOAT;
  const uint64_t bytesPerTexture = static_cast<uint64_t>(framebufferWidth_) *
                                   static_cast<uint64_t>(framebufferHeight_) *
                                   static_cast<uint64_t>(textureBytesPerPixel(
                                       kFrameCompositionMotionVectorFormat));
  aaMetrics.motionVectorTextureBytes =
      aaMetrics.motionVectorAllocated ? bytesPerTexture : 0u;
  aaMetrics.previousMotionVectorTextureBytes =
      !motionVectorTextures_.empty() ? bytesPerTexture : 0u;
  aaMetrics.motionVectorTotalBytes =
      bytesPerTexture * static_cast<uint64_t>(motionVectorTextures_.size());
  aaMetrics.motionVectorClearBytes = 0u;
  if (ctx.shared.sceneDepthSamplerId == 0u) {
    ctx.shared.sceneDepthSamplerId = gpu_.getDefaultSamplerBindlessIndex();
  }
  ctx.frame.sharedDepthTexture = ctx.shared.sceneDepthTexture;

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> FrameCompositionProvider::ensureTextures(
    FrameTextureRequirementFlags requirements) {
  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
  const uint32_t safeWidth =
      static_cast<uint32_t>(std::max(framebufferWidth, 1));
  const uint32_t safeHeight =
      static_cast<uint32_t>(std::max(framebufferHeight, 1));
  const uint32_t ringCount = std::max(1u, gpu_.getSwapchainImageCount());

  const bool dimensionsChanged =
      framebufferWidth_ != safeWidth || framebufferHeight_ != safeHeight;
  const bool ringChanged = textureRingCount_ != ringCount;
  const bool requirementsChanged = allocatedRequirements_ != requirements;
  if (!dimensionsChanged && !ringChanged && !requirementsChanged) {
    return Result<bool, std::string>::makeResult(true);
  }

  const FrameTextureRequirementFlags previousRequirements =
      allocatedRequirements_;
  const bool fullRecreate = dimensionsChanged || ringChanged;
  framebufferWidth_ = safeWidth;
  framebufferHeight_ = safeHeight;
  textureRingCount_ = ringCount;
  allocatedRequirements_ = requirements;

  const bool needsSceneColor = hasFrameTextureRequirementFlag(
      requirements, FrameTextureRequirementFlags::SceneColor);
  const bool hadSceneColor = hasFrameTextureRequirementFlag(
      previousRequirements, FrameTextureRequirementFlags::SceneColor);
  if (needsSceneColor && (fullRecreate || !hadSceneColor)) {
    auto recreateResult = recreateMipTextureRing(sceneColorMipTextures_[0], 0u,
                                                 "frame_scene_color");
    if (recreateResult.hasError()) {
      invalidateAllocationState();
      return recreateResult;
    }
  } else if (!needsSceneColor && hadSceneColor) {
    destroyTextures(sceneColorMipTextures_[0]);
  }

  const bool needsFrameColor = hasFrameTextureRequirementFlag(
      requirements, FrameTextureRequirementFlags::FrameColor);
  const bool hadFrameColor = hasFrameTextureRequirementFlag(
      previousRequirements, FrameTextureRequirementFlags::FrameColor);
  if (needsFrameColor && (fullRecreate || !hadFrameColor)) {
    auto recreateResult = recreateFullResTextureRing(
        frameColorTextures_, kFrameCompositionFrameColorFormat,
        TextureUsage::AttachmentSampled, "frame_output_color");
    if (recreateResult.hasError()) {
      invalidateAllocationState();
      return recreateResult;
    }
  } else if (!needsFrameColor && hadFrameColor) {
    destroyTextures(frameColorTextures_);
  }

  const bool needsSceneDepth = hasFrameTextureRequirementFlag(
      requirements, FrameTextureRequirementFlags::SceneDepth);
  const bool hadSceneDepth = hasFrameTextureRequirementFlag(
      previousRequirements, FrameTextureRequirementFlags::SceneDepth);
  if (needsSceneDepth && (fullRecreate || !hadSceneDepth)) {
    auto recreateResult = recreateFullResTextureRing(
        sceneDepthTextures_, kFrameCompositionDepthFormat,
        TextureUsage::AttachmentSampled, "frame_scene_depth");
    if (recreateResult.hasError()) {
      invalidateAllocationState();
      return recreateResult;
    }
  } else if (!needsSceneDepth && hadSceneDepth) {
    destroyTextures(sceneDepthTextures_);
  }

  const bool needsSceneColorMipChain = hasFrameTextureRequirementFlag(
      requirements, FrameTextureRequirementFlags::SceneColorMipChain);
  const bool hadSceneColorMipChain = hasFrameTextureRequirementFlag(
      previousRequirements, FrameTextureRequirementFlags::SceneColorMipChain);
  if (needsSceneColorMipChain && (fullRecreate || !hadSceneColorMipChain)) {
    for (size_t i = 1; i < kSceneColorMipRingSpecs.size(); ++i) {
      const SceneColorMipRingSpec &spec = kSceneColorMipRingSpecs[i];
      auto recreateResult = recreateMipTextureRing(
          sceneColorMipTextures_[i], spec.mipLevel, spec.debugNameBase);
      if (recreateResult.hasError()) {
        invalidateAllocationState();
        return recreateResult;
      }
    }
  } else if (!needsSceneColorMipChain && hadSceneColorMipChain) {
    for (size_t i = 1; i < sceneColorMipTextures_.size(); ++i) {
      destroyTextures(sceneColorMipTextures_[i]);
    }
  }

  const bool needsHistoryColor = hasFrameTextureRequirementFlag(
      requirements, FrameTextureRequirementFlags::HistoryColor);
  const bool hadHistoryColor = hasFrameTextureRequirementFlag(
      previousRequirements, FrameTextureRequirementFlags::HistoryColor);
  if (needsHistoryColor && (fullRecreate || !hadHistoryColor)) {
    auto historyResult = recreateHistoryTextures();
    if (historyResult.hasError()) {
      invalidateAllocationState();
      return historyResult;
    }
  } else if (!needsHistoryColor && hadHistoryColor) {
    destroyHistoryTextures();
  }

  const bool needsMotionVectors = hasFrameTextureRequirementFlag(
      requirements, FrameTextureRequirementFlags::MotionVectors);
  const bool hadMotionVectors = hasFrameTextureRequirementFlag(
      previousRequirements, FrameTextureRequirementFlags::MotionVectors);
  if (needsMotionVectors && (fullRecreate || !hadMotionVectors)) {
    auto motionVectorResult = recreateMotionVectorTextures();
    if (motionVectorResult.hasError()) {
      invalidateAllocationState();
      return motionVectorResult;
    }
  } else if (!needsMotionVectors && hadMotionVectors) {
    destroyMotionVectorTextures();
  }

  return Result<bool, std::string>::makeResult(true);
}

void FrameCompositionProvider::invalidateAllocationState() noexcept {
  framebufferWidth_ = 0u;
  framebufferHeight_ = 0u;
  textureRingCount_ = 0u;
  allocatedRequirements_ = FrameTextureRequirementFlags::None;
}

Result<bool, std::string> FrameCompositionProvider::recreateFullResTextureRing(
    TextureRing &textures, Format format, TextureUsage usage,
    std::string_view debugNameBase) {
  destroyTextures(textures);
  textures.resize(textureRingCount_);

  const TextureDesc desc =
      makeTextureDesc(format, framebufferWidth_, framebufferHeight_, usage);
  for (uint32_t i = 0; i < textureRingCount_; ++i) {
    const std::string debugName =
        std::string(debugNameBase) + "_" + std::to_string(i);
    auto createResult = gpu_.createTexture(desc, debugName);
    if (createResult.hasError()) {
      destroyTextures(textures);
      return Result<bool, std::string>::makeError(createResult.error());
    }
    textures[i] = createResult.value();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> FrameCompositionProvider::recreateMipTextureRing(
    TextureRing &textures, uint32_t mipLevel, std::string_view debugNameBase) {
  destroyTextures(textures);
  textures.resize(textureRingCount_);

  const TextureDesc desc =
      makeTextureDesc(kFrameCompositionSceneColorFormat,
                      levelDimensions(framebufferWidth_, mipLevel),
                      levelDimensions(framebufferHeight_, mipLevel),
                      TextureUsage::AttachmentSampled);
  for (uint32_t i = 0; i < textureRingCount_; ++i) {
    const std::string debugName =
        std::string(debugNameBase) + "_" + std::to_string(i);
    auto createResult = gpu_.createTexture(desc, debugName);
    if (createResult.hasError()) {
      destroyTextures(textures);
      return Result<bool, std::string>::makeError(createResult.error());
    }
    textures[i] = createResult.value();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> FrameCompositionProvider::recreateHistoryTextures() {
  destroyHistoryTextures();

  const TextureDesc desc =
      makeTextureDesc(kFrameCompositionFrameColorFormat, framebufferWidth_,
                      framebufferHeight_, TextureUsage::AttachmentSampled);
  for (size_t i = 0; i < historyColorTextures_.size(); ++i) {
    const std::string debugName = "frame_history_color_" + std::to_string(i);
    auto createResult = gpu_.createTexture(desc, debugName);
    if (createResult.hasError()) {
      destroyHistoryTextures();
      return Result<bool, std::string>::makeError(createResult.error());
    }
    historyColorTextures_[i] = createResult.value();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
FrameCompositionProvider::recreateMotionVectorTextures() {
  const bool replacingExistingTextures = !motionVectorTextures_.empty();
  destroyMotionVectorTextures();

  const uint32_t motionVectorRingCount = std::max(2u, textureRingCount_);
  motionVectorTextures_.resize(motionVectorRingCount);
  const TextureDesc desc =
      makeTextureDesc(kFrameCompositionMotionVectorFormat, framebufferWidth_,
                      framebufferHeight_, TextureUsage::AttachmentSampled);
  for (uint32_t i = 0; i < motionVectorRingCount; ++i) {
    const std::string debugName = "frame_motion_vectors_" + std::to_string(i);
    auto createResult = gpu_.createTexture(desc, debugName);
    if (createResult.hasError()) {
      destroyMotionVectorTextures();
      return Result<bool, std::string>::makeError(createResult.error());
    }
    motionVectorTextures_[i] = createResult.value();
  }
  if (replacingExistingTextures) {
    ++motionVectorReallocationCount_;
  }
  motionVectorAllocationCount_ += motionVectorRingCount;
  return Result<bool, std::string>::makeResult(true);
}

void FrameCompositionProvider::destroyTextures(TextureRing &textures) {
  for (TextureHandle &texture : textures) {
    if (!nuri::isValid(texture)) {
      continue;
    }
    gpu_.destroyTexture(texture);
    texture = {};
  }
  textures.clear();
}

void FrameCompositionProvider::destroyHistoryTextures() {
  for (TextureHandle &texture : historyColorTextures_) {
    if (!nuri::isValid(texture)) {
      continue;
    }
    gpu_.destroyTexture(texture);
    texture = {};
  }
}

void FrameCompositionProvider::destroyMotionVectorTextures() {
  destroyTextures(motionVectorTextures_);
}

TextureHandle FrameCompositionProvider::currentRingTexture(
    const TextureRing &textures, uint64_t frameIndex) const noexcept {
  if (textures.empty()) {
    return {};
  }
  return textures[static_cast<size_t>(frameIndex % textures.size())];
}

TextureHandle FrameCompositionProvider::previousRingTexture(
    const TextureRing &textures, uint64_t frameIndex) const noexcept {
  if (textures.empty()) {
    return {};
  }
  const uint64_t previousIndex =
      (frameIndex + static_cast<uint64_t>(textures.size()) - 1u) %
      static_cast<uint64_t>(textures.size());
  return textures[static_cast<size_t>(previousIndex)];
}

} // namespace nuri
