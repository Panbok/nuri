#pragma once

#include "nuri/resources/cpu/material_data.h"
#include "nuri/resources/storage/texture/texture_artifact_builder.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace nuri {

struct MaterialTextureArtifactSpec {
  MaterialTextureSlotData MaterialData::*slot = nullptr;
  bool srgb = false;
  TextureMipSemantic mipSemantic = TextureMipSemantic::Generic;
  TextureArtifactEncoding encoding = TextureArtifactEncoding::Etc1s;
};

inline constexpr std::array<MaterialTextureArtifactSpec,
                            kMaterialTextureSlotCount>
    kMaterialTextureArtifactSpecs{{
        {&MaterialData::baseColor, true, TextureMipSemantic::Generic,
         TextureArtifactEncoding::Uastc},
        {&MaterialData::metallicRoughness, false,
         TextureMipSemantic::RoughnessG, TextureArtifactEncoding::Etc1s},
        {&MaterialData::normal, false, TextureMipSemantic::NormalMap,
         TextureArtifactEncoding::Uastc},
        {&MaterialData::occlusion, false, TextureMipSemantic::Generic,
         TextureArtifactEncoding::Etc1s},
        {&MaterialData::emissive, true, TextureMipSemantic::Generic,
         TextureArtifactEncoding::Uastc},
        {&MaterialData::clearcoat, false, TextureMipSemantic::Generic,
         TextureArtifactEncoding::Etc1s},
        {&MaterialData::clearcoatRoughness, false,
         TextureMipSemantic::RoughnessG, TextureArtifactEncoding::Etc1s},
        {&MaterialData::clearcoatNormal, false, TextureMipSemantic::NormalMap,
         TextureArtifactEncoding::Uastc},
        {&MaterialData::specular, false, TextureMipSemantic::Generic,
         TextureArtifactEncoding::Etc1s},
        {&MaterialData::specularColor, true, TextureMipSemantic::Generic,
         TextureArtifactEncoding::Uastc},
        {&MaterialData::sheenColor, true, TextureMipSemantic::Generic,
         TextureArtifactEncoding::Uastc},
        {&MaterialData::sheenRoughness, false, TextureMipSemantic::RoughnessA,
         TextureArtifactEncoding::Etc1s},
        {&MaterialData::transmission, false, TextureMipSemantic::Generic,
         TextureArtifactEncoding::Etc1s},
        {&MaterialData::thickness, false, TextureMipSemantic::Generic,
         TextureArtifactEncoding::Etc1s},
    }};

static_assert(kMaterialTextureArtifactSpecs.size() ==
              kMaterialTextureSlotCount);

[[nodiscard]] inline TextureArtifactBuildOptions
makeMaterialTextureArtifactBuildOptions(const MaterialData &material,
                                        size_t slotIndex) noexcept {
  const MaterialTextureArtifactSpec &spec =
      kMaterialTextureArtifactSpecs[slotIndex];
  TextureMipSemantic semantic = spec.mipSemantic;
  if (slotIndex == 0u && material.alphaMode == MaterialAlphaMode::Mask) {
    semantic = TextureMipSemantic::AlphaCoverage;
  }
  return TextureArtifactBuildOptions{
      .loadOptions =
          {
              .srgb = spec.srgb,
              .generateMipmaps = true,
              .mipSemantic = semantic,
              .alphaCoverageCutoff = material.alphaCutoff,
          },
      .encoding = spec.encoding,
  };
}

} // namespace nuri
