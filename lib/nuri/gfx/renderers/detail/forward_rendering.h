#pragma once
#include "nuri/gfx/dynamic_buffer.h"
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

inline void
rebuildForwardInstances(std::span<const Renderable> renderables,
                        std::pmr::vector<InstanceData> &matrices,
                        std::pmr::vector<uint32_t> &remap,
                        std::pmr::vector<uint64_t> &uploadVersions) {
  matrices.clear();
  remap.clear();
  matrices.reserve(renderables.size());
  remap.reserve(renderables.size());
  for (uint32_t i = 0; i < renderables.size(); ++i) {
    matrices.push_back(makeInstanceData(renderables[i].modelMatrix));
    remap.push_back(i);
  }
  std::ranges::fill(uploadVersions, std::numeric_limits<uint64_t>::max());
}

inline Result<bool, std::string> uploadForwardInstances(
    GPUDevice &gpu, std::span<const DynamicBufferSlot> matrixRing,
    std::span<const DynamicBufferSlot> remapRing,
    std::span<const InstanceData> matrices, std::span<const uint32_t> remap,
    std::span<uint64_t> uploadVersions, uint32_t slot, uint64_t version,
    bool externallyAnimated) {
  if (externallyAnimated || uploadVersions[slot] == version) {
    return Result<bool, std::string>::makeResult(true);
  }
  std::array<BufferUpdate, 2> updates{};
  size_t count = 0;
  if (!matrices.empty()) {
    updates[count++] = {.buffer = matrixRing[slot].buffer->handle(),
                        .data = std::as_bytes(matrices)};
  }
  if (!remap.empty()) {
    updates[count++] = {.buffer = remapRing[slot].buffer->handle(),
                        .data = std::as_bytes(remap)};
  }
  if (count > 0u) {
    auto result = gpu.updateBuffers({updates.data(), count});
    if (result.hasError()) {
      return result;
    }
  }
  uploadVersions[slot] = version;
  return Result<bool, std::string>::makeResult(true);
}

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
