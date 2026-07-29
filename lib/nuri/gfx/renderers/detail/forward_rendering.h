#pragma once
#include "nuri/gfx/dynamic_buffer.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/renderers/detail/instance_data.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"
#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
namespace nuri {

struct ForwardSceneDrawCache {
  explicit ForwardSceneDrawCache(std::pmr::memory_resource *memory)
      : instanceMatrices(memory), instanceRemap(memory),
        materialTextureAccessHandles(memory),
        environmentTextureAccessHandles(memory) {}

  void reset() {
    scene = nullptr;
    topologyVersion = std::numeric_limits<uint64_t>::max();
    materialVersion = std::numeric_limits<uint64_t>::max();
    modelMaterialBindingVersion = std::numeric_limits<uint64_t>::max();
    transformVersion = std::numeric_limits<uint64_t>::max();
    geometryMutationVersion = std::numeric_limits<uint64_t>::max();
    instanceMatrices.clear();
    instanceRemap.clear();
    materialTextureAccessHandles.clear();
    environmentTextureAccessHandles.clear();
  }

  const RenderScene *scene = nullptr;
  uint64_t topologyVersion = std::numeric_limits<uint64_t>::max();
  uint64_t materialVersion = std::numeric_limits<uint64_t>::max();
  uint64_t modelMaterialBindingVersion = std::numeric_limits<uint64_t>::max();
  uint64_t transformVersion = std::numeric_limits<uint64_t>::max();
  uint64_t geometryMutationVersion = std::numeric_limits<uint64_t>::max();
  std::pmr::vector<InstanceData> instanceMatrices;
  std::pmr::vector<uint32_t> instanceRemap;
  std::pmr::vector<TextureHandle> materialTextureAccessHandles;
  std::pmr::vector<TextureHandle> environmentTextureAccessHandles;
};

template <typename Handle>
void appendUniqueForwardHandle(std::pmr::vector<Handle> &handles,
                               Handle handle) {
  if (nuri::isValid(handle) &&
      std::ranges::find(handles, handle) == handles.end()) {
    handles.push_back(handle);
  }
}

[[nodiscard]] inline std::optional<SubmeshLod>
resolveForwardLod(const Submesh &submesh, int32_t forcedLod,
                  bool preferExplicitLod0 = false) {
  if (forcedLod < 0) {
    const uint32_t lodCount =
        std::clamp(submesh.lodCount, 0u, Submesh::kMaxLodCount);
    if (preferExplicitLod0 && lodCount > 0u && submesh.lods[0].indexCount > 0u)
      return submesh.lods[0];
    if (submesh.indexCount > 0u) {
      return SubmeshLod{.indexOffset = submesh.indexOffset,
                        .indexCount = submesh.indexCount};
    }
    for (uint32_t lod = preferExplicitLod0 ? 1u : 0u; lod < lodCount; ++lod) {
      if (submesh.lods[lod].indexCount > 0u) {
        return submesh.lods[lod];
      }
    }
    return std::nullopt;
  }
  const uint32_t lodCount =
      std::clamp(submesh.lodCount, 1u, Submesh::kMaxLodCount);
  uint32_t candidate =
      std::min(static_cast<uint32_t>(forcedLod), lodCount - 1u);
  while (candidate > 0u && submesh.lods[candidate].indexCount == 0u) {
    --candidate;
  }
  if (submesh.lods[candidate].indexCount > 0u) {
    return submesh.lods[candidate];
  }
  return submesh.indexCount > 0u ? std::optional<SubmeshLod>(SubmeshLod{
                                       .indexOffset = submesh.indexOffset,
                                       .indexCount = submesh.indexCount})
                                 : std::nullopt;
}

inline void rebuildForwardInstances(std::span<const Renderable> renderables,
                                    std::pmr::vector<InstanceData> &matrices,
                                    std::pmr::vector<uint32_t> &remap) {
  matrices.clear();
  remap.clear();
  matrices.reserve(renderables.size());
  remap.reserve(renderables.size());
  for (uint32_t i = 0; i < renderables.size(); ++i) {
    matrices.push_back(makeInstanceData(renderables[i].modelMatrix));
    remap.push_back(i);
  }
}

struct ForwardInstanceBufferView {
  BufferHandle matrices{};
  BufferHandle remap{};
};

class ForwardInstanceBuffers {
public:
  ForwardInstanceBuffers(GPUDevice &gpu, std::string_view debugNamePrefix,
                         std::pmr::memory_resource *memory)
      : gpu_(gpu),
        matrices_(gpu,
                  BufferDesc{.usage = BufferUsage::Storage,
                             .storage = Storage::Device},
                  std::string(debugNamePrefix) + "_instance_matrices", memory),
        remap_(gpu,
               BufferDesc{.usage = BufferUsage::Storage,
                          .storage = Storage::Device},
               std::string(debugNamePrefix) + "_instance_remap", memory),
        matrixUploadVersions_(memory), remapUploadVersions_(memory) {}

  [[nodiscard]] Result<ForwardInstanceBufferView, std::string>
  prepare(uint64_t frameIndex, size_t minimumLaneCount,
          std::span<const InstanceData> matrices,
          std::span<const uint32_t> remap, uint64_t version,
          bool externallyAnimated) {
    std::optional<DynamicBufferAcquisition> matrix;
    if (!externallyAnimated) {
      auto acquired = matrices_.acquire(
          frameIndex, std::max(matrices.size_bytes(), sizeof(InstanceData)),
          minimumLaneCount);
      if (acquired.hasError())
        return Result<ForwardInstanceBufferView, std::string>::makeError(
            acquired.error());
      matrix = acquired.value();
      resizeVersions(matrixUploadVersions_, matrices_.laneCount());
      if (matrix->replaced)
        matrixUploadVersions_[matrix->lane] = kInvalidVersion;
    }
    auto remapAcquired = remap_.acquire(
        frameIndex, std::max(remap.size_bytes(), sizeof(uint32_t)),
        minimumLaneCount);
    if (remapAcquired.hasError()) {
      matrices_.abandonPrepared();
      return Result<ForwardInstanceBufferView, std::string>::makeError(
          remapAcquired.error());
    }
    const DynamicBufferAcquisition remapLane = remapAcquired.value();
    resizeVersions(remapUploadVersions_, remap_.laneCount());
    if (remapLane.replaced)
      remapUploadVersions_[remapLane.lane] = kInvalidVersion;

    std::array<BufferUpdate, 2> updates{};
    size_t count = 0u;
    if (matrix && matrixUploadVersions_[matrix->lane] != version &&
        !matrices.empty()) {
      updates[count++] = {.buffer = matrix->buffer,
                          .data = std::as_bytes(matrices)};
    }
    if (remapUploadVersions_[remapLane.lane] != version && !remap.empty()) {
      updates[count++] = {.buffer = remapLane.buffer,
                          .data = std::as_bytes(remap)};
    }
    if (count != 0u) {
      auto update = gpu_.updateBuffers({updates.data(), count});
      if (update.hasError()) {
        abandonPrepared();
        return Result<ForwardInstanceBufferView, std::string>::makeError(
            update.error());
      }
    }
    if (matrix)
      matrixUploadVersions_[matrix->lane] = version;
    remapUploadVersions_[remapLane.lane] = version;
    return Result<ForwardInstanceBufferView, std::string>::makeResult(
        {.matrices = matrix ? matrix->buffer : BufferHandle{},
         .remap = remapLane.buffer});
  }

  void onFrameSubmitted(const RenderFrameContext &frame) noexcept {
    matrices_.submitPrepared(frame.submission);
    remap_.submitPrepared(frame.submission);
  }

  void onFrameAbandoned(const RenderFrameContext &) noexcept {
    abandonPrepared();
  }

  void abandonPrepared() noexcept {
    matrices_.abandonPrepared();
    remap_.abandonPrepared();
  }

  void reset() noexcept {
    matrices_.reset();
    remap_.reset();
    matrixUploadVersions_.clear();
    remapUploadVersions_.clear();
  }

private:
  static constexpr uint64_t kInvalidVersion =
      std::numeric_limits<uint64_t>::max();
  static void resizeVersions(std::pmr::vector<uint64_t> &versions,
                             size_t count) {
    versions.resize(count, kInvalidVersion);
  }
  GPUDevice &gpu_;
  DynamicBufferRing matrices_;
  DynamicBufferRing remap_;
  std::pmr::vector<uint64_t> matrixUploadVersions_;
  std::pmr::vector<uint64_t> remapUploadVersions_;
};

template <typename DrawRange>
void collectForwardMaterialTextures(const ResourceManager &resources,
                                    const DrawRange &draws,
                                    std::pmr::vector<TextureHandle> &textures) {
  textures.clear();
  for (const auto &draw : draws) {
    const MaterialRecord *material = resources.tryGet(draw.material);
    if (material == nullptr) {
      continue;
    }
    forEachMaterialTextureRef(material->textureRefs, [&](TextureRef ref) {
      const TextureRecord *texture = resources.tryGet(ref);
      if (texture != nullptr) {
        appendUniqueForwardHandle(textures, texture->texture);
      }
    });
  }
}

inline void
collectForwardEnvironmentTextures(const RenderScene &scene,
                                  const ResourceManager &resources,
                                  std::pmr::vector<TextureHandle> &textures) {
  textures.clear();
  const EnvironmentHandles environment = scene.environment();
  const std::array refs{environment.cubemap, environment.irradiance,
                        environment.prefilteredGgx,
                        environment.prefilteredCharlie, environment.brdfLut};
  for (TextureRef ref : refs) {
    if (const TextureRecord *texture = resources.tryGet(ref)) {
      appendUniqueForwardHandle(textures, texture->texture);
    }
  }
}

} // namespace nuri
