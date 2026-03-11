#pragma once

#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/resources/gpu/resource_handles.h"
#include "nuri/resources/gpu/texture.h"
#include "nuri/resources/storage/mesh/mesh_cache_utils.h"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace nuri {

enum class TextureRequestKind : uint8_t {
  Texture2D = 0,
  Ktx2Texture2D = 1,
  Ktx2Cubemap = 2,
  EquirectHdrCubemap = 3,
};

struct TextureKey {
  std::string canonicalPath{};
  uint64_t optionsHash = 0;
  TextureRequestKind kind = TextureRequestKind::Texture2D;

  bool operator==(const TextureKey &rhs) const noexcept {
    return optionsHash == rhs.optionsHash && kind == rhs.kind &&
           canonicalPath == rhs.canonicalPath;
  }
};

struct TextureKeyHash {
  size_t operator()(const TextureKey &key) const noexcept {
    const size_t h1 = std::hash<std::string>{}(key.canonicalPath);
    const size_t h2 = std::hash<uint64_t>{}(key.optionsHash);
    const size_t h3 = std::hash<uint8_t>{}(static_cast<uint8_t>(key.kind));
    return h1 ^ (h2 + 0x9e3779b9u + (h1 << 6u) + (h1 >> 2u)) ^
           (h3 + 0x9e3779b9u + (h2 << 6u) + (h2 >> 2u));
  }
};

struct ModelKey {
  std::string canonicalPath{};
  uint64_t importOptionsHash = 0;

  bool operator==(const ModelKey &rhs) const noexcept {
    return importOptionsHash == rhs.importOptionsHash &&
           canonicalPath == rhs.canonicalPath;
  }
};

struct ModelKeyHash {
  size_t operator()(const ModelKey &key) const noexcept {
    const size_t h1 = std::hash<std::string>{}(key.canonicalPath);
    const size_t h2 = std::hash<uint64_t>{}(key.importOptionsHash);
    return h1 ^ (h2 + 0x9e3779b9u + (h1 << 6u) + (h1 >> 2u));
  }
};

struct MaterialKey {
  uint64_t descHash = 0;
  std::string sourceIdentity{};

  bool operator==(const MaterialKey &rhs) const noexcept {
    return descHash == rhs.descHash && sourceIdentity == rhs.sourceIdentity;
  }
};

struct MaterialKeyHash {
  size_t operator()(const MaterialKey &key) const noexcept {
    const size_t h1 = std::hash<uint64_t>{}(key.descHash);
    const size_t h2 = std::hash<std::string>{}(key.sourceIdentity);
    return h1 ^ (h2 + 0x9e3779b9u + (h1 << 6u) + (h1 >> 2u));
  }
};

[[nodiscard]] inline std::string
canonicalizeResourcePath(std::string_view inputPath) {
  if (inputPath.empty()) {
    return {};
  }

  const std::filesystem::path rawPath{std::string(inputPath)};
  std::error_code ec;
  std::filesystem::path canonical =
      std::filesystem::weakly_canonical(rawPath, ec);
  if (ec) {
    canonical = rawPath.lexically_normal();
  }

  std::string path = canonical.string();
#if defined(_WIN32)
  std::transform(path.begin(), path.end(), path.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
#endif
  return path;
}

[[nodiscard]] inline uint64_t
hashTextureLoadOptions(const TextureLoadOptions &options) {
  uint64_t hash = 1469598103934665603ull;
  const auto mix = [&hash](uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ull;
  };
  mix(options.srgb ? 1ull : 0ull);
  mix(options.generateMipmaps ? 1ull : 0ull);
  return hash;
}

[[nodiscard]] inline uint64_t
hashModelImportOptions(const MeshImportOptions &options) {
  return hashMeshImportOptions(options);
}

[[nodiscard]] inline uint64_t hashMaterialDesc(const MaterialDesc &desc) {
  uint64_t hash = 1469598103934665603ull;
  const auto mix = [&hash](uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ull;
  };
  const auto mixFloat = [&mix](float value) {
    mix(static_cast<uint64_t>(std::bit_cast<uint32_t>(value)));
  };

  mixFloat(desc.baseColorFactor.x);
  mixFloat(desc.baseColorFactor.y);
  mixFloat(desc.baseColorFactor.z);
  mixFloat(desc.baseColorFactor.w);

  mixFloat(desc.emissiveFactor.x);
  mixFloat(desc.emissiveFactor.y);
  mixFloat(desc.emissiveFactor.z);

  mixFloat(desc.metallicFactor);
  mixFloat(desc.roughnessFactor);
  mixFloat(desc.sheenColorFactor.x);
  mixFloat(desc.sheenColorFactor.y);
  mixFloat(desc.sheenColorFactor.z);
  mixFloat(desc.sheenWeight);
  mixFloat(desc.sheenRoughnessFactor);
  mixFloat(desc.clearcoatFactor);
  mixFloat(desc.clearcoatRoughnessFactor);
  mixFloat(desc.clearcoatNormalScale);
  mixFloat(desc.normalScale);
  mixFloat(desc.occlusionStrength);
  mixFloat(desc.alphaCutoff);

  mix(desc.doubleSided ? 1ull : 0ull);
  mix(static_cast<uint64_t>(desc.alphaMode));
  mix(desc.featureMask);

  mix(desc.textures.baseColor.index);
  mix(desc.textures.baseColor.generation);
  mix(desc.textures.metallicRoughness.index);
  mix(desc.textures.metallicRoughness.generation);
  mix(desc.textures.normal.index);
  mix(desc.textures.normal.generation);
  mix(desc.textures.occlusion.index);
  mix(desc.textures.occlusion.generation);
  mix(desc.textures.emissive.index);
  mix(desc.textures.emissive.generation);
  mix(desc.textures.clearcoat.index);
  mix(desc.textures.clearcoat.generation);
  mix(desc.textures.clearcoatRoughness.index);
  mix(desc.textures.clearcoatRoughness.generation);
  mix(desc.textures.clearcoatNormal.index);
  mix(desc.textures.clearcoatNormal.generation);
  mix(desc.textures.sheenColor.index);
  mix(desc.textures.sheenColor.generation);
  mix(desc.textures.sheenRoughness.index);
  mix(desc.textures.sheenRoughness.generation);

  mix(desc.uvSets.baseColor);
  mix(desc.uvSets.metallicRoughness);
  mix(desc.uvSets.normal);
  mix(desc.uvSets.occlusion);
  mix(desc.uvSets.emissive);
  mix(desc.uvSets.clearcoat);
  mix(desc.uvSets.clearcoatRoughness);
  mix(desc.uvSets.clearcoatNormal);
  mix(desc.uvSets.sheenColor);
  mix(desc.uvSets.sheenRoughness);

  mix(desc.samplers.baseColor);
  mix(desc.samplers.metallicRoughness);
  mix(desc.samplers.normal);
  mix(desc.samplers.occlusion);
  mix(desc.samplers.emissive);
  mix(desc.samplers.clearcoat);
  mix(desc.samplers.clearcoatRoughness);
  mix(desc.samplers.clearcoatNormal);
  mix(desc.samplers.sheenColor);
  mix(desc.samplers.sheenRoughness);

  for (uint32_t slotIndex = 0; slotIndex < kMaterialTextureSlotCount;
       ++slotIndex) {
    const MaterialTextureTransformData &transform =
        desc.transforms.slots[slotIndex];
    mixFloat(transform.offset.x);
    mixFloat(transform.offset.y);
    mixFloat(transform.scale.x);
    mixFloat(transform.scale.y);
    mixFloat(transform.rotationRadians);
  }

  return hash;
}

} // namespace nuri
