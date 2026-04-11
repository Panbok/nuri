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
      frameColorTextures_(memory_), sceneDepthTextures_(memory_) {}

FrameCompositionProvider::~FrameCompositionProvider() {
  for (auto &textures : sceneColorMipTextures_) {
    destroyTextures(textures);
  }
  destroyTextures(frameColorTextures_);
  destroyTextures(sceneDepthTextures_);
  destroyHistoryTextures();
}

Result<bool, std::string>
FrameCompositionProvider::prepare(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  ctx.shared.textureRequirements |= kBaselineFrameTextureRequirements;
  ctx.shared.sceneColorGraphTexture = {};
  ctx.shared.frameColorGraphTexture = {};
  ctx.shared.sceneDepthGraphTexture = {};
  ctx.shared.sceneDepthPyramidGraphTextures = {};

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
  ctx.shared.sceneDepthSamplerId = gpu_.getDefaultSamplerBindlessIndex();
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

  framebufferWidth_ = safeWidth;
  framebufferHeight_ = safeHeight;
  textureRingCount_ = ringCount;
  allocatedRequirements_ = requirements;

  if (hasFrameTextureRequirementFlag(
          requirements, FrameTextureRequirementFlags::SceneColor)) {
    auto recreateResult = recreateMipTextureRing(sceneColorMipTextures_[0], 0u,
                                                 "frame_scene_color");
    if (recreateResult.hasError()) {
      return recreateResult;
    }
  } else {
    destroyTextures(sceneColorMipTextures_[0]);
  }

  if (hasFrameTextureRequirementFlag(
          requirements, FrameTextureRequirementFlags::FrameColor)) {
    auto recreateResult = recreateFullResTextureRing(
        frameColorTextures_, kFrameCompositionFrameColorFormat,
        TextureUsage::AttachmentSampled, "frame_output_color");
    if (recreateResult.hasError()) {
      return recreateResult;
    }
  } else {
    destroyTextures(frameColorTextures_);
  }

  if (hasFrameTextureRequirementFlag(
          requirements, FrameTextureRequirementFlags::SceneDepth)) {
    auto recreateResult = recreateFullResTextureRing(
        sceneDepthTextures_, kFrameCompositionDepthFormat,
        TextureUsage::AttachmentSampled, "frame_scene_depth");
    if (recreateResult.hasError()) {
      return recreateResult;
    }
  } else {
    destroyTextures(sceneDepthTextures_);
  }

  if (hasFrameTextureRequirementFlag(
          requirements, FrameTextureRequirementFlags::SceneColorMipChain)) {
    for (size_t i = 1; i < kSceneColorMipRingSpecs.size(); ++i) {
      const SceneColorMipRingSpec &spec = kSceneColorMipRingSpecs[i];
      auto recreateResult = recreateMipTextureRing(
          sceneColorMipTextures_[i], spec.mipLevel, spec.debugNameBase);
      if (recreateResult.hasError()) {
        return recreateResult;
      }
    }
  } else {
    for (size_t i = 1; i < sceneColorMipTextures_.size(); ++i) {
      destroyTextures(sceneColorMipTextures_[i]);
    }
  }

  if (hasFrameTextureRequirementFlag(
          requirements, FrameTextureRequirementFlags::HistoryColor)) {
    auto historyResult = recreateHistoryTextures();
    if (historyResult.hasError()) {
      return historyResult;
    }
  } else {
    destroyHistoryTextures();
  }

  return Result<bool, std::string>::makeResult(true);
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

TextureHandle FrameCompositionProvider::currentRingTexture(
    const TextureRing &textures, uint64_t frameIndex) const noexcept {
  if (textures.empty()) {
    return {};
  }
  return textures[static_cast<size_t>(frameIndex % textures.size())];
}

} // namespace nuri
