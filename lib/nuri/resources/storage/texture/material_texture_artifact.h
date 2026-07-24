#pragma once
#include "nuri/resources/cpu/material_data.h"
#include "nuri/resources/storage/texture/texture_artifact_builder.h"
#include <array>
namespace nuri {
struct MaterialTextureArtifactSpec {
  bool srgb = false;
  TextureMipSemantic mipSemantic = TextureMipSemantic::Generic;
  TextureArtifactEncoding encoding = TextureArtifactEncoding::Etc1s;
};
inline constexpr std::array<MaterialTextureArtifactSpec,
                            kMaterialTextureSlotCount>
    kMaterialTextureArtifactSpecs{{
        {true, TextureMipSemantic::Generic, TextureArtifactEncoding::Uastc},
        {false, TextureMipSemantic::RoughnessG, TextureArtifactEncoding::Etc1s},
        {false, TextureMipSemantic::NormalMap, TextureArtifactEncoding::Uastc},
        {false, TextureMipSemantic::Generic, TextureArtifactEncoding::Etc1s},
        {true, TextureMipSemantic::Generic, TextureArtifactEncoding::Uastc},
        {false, TextureMipSemantic::Generic, TextureArtifactEncoding::Etc1s},
        {false, TextureMipSemantic::RoughnessG, TextureArtifactEncoding::Etc1s},
        {false, TextureMipSemantic::NormalMap, TextureArtifactEncoding::Uastc},
        {false, TextureMipSemantic::Generic, TextureArtifactEncoding::Etc1s},
        {true, TextureMipSemantic::Generic, TextureArtifactEncoding::Uastc},
        {true, TextureMipSemantic::Generic, TextureArtifactEncoding::Uastc},
        {false, TextureMipSemantic::RoughnessA, TextureArtifactEncoding::Etc1s},
        {false, TextureMipSemantic::Generic, TextureArtifactEncoding::Etc1s},
        {false, TextureMipSemantic::Generic, TextureArtifactEncoding::Etc1s},
    }};
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
      .contentContract = semantic == TextureMipSemantic::NormalMap
                             ? TextureContentContract::NormalRgbCleanVarianceA
                             : TextureContentContract::Generic,
  };
}
} // namespace nuri
