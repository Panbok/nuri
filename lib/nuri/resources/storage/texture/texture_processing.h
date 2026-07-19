#pragma once
#include "nuri/core/result.h"
#include <cstdint>
#include <span>
#include <string>
#include <vector>
namespace nuri {

enum class TextureMipSemantic : uint8_t {
  Generic = 0,
  AlphaCoverage = 1,
  NormalMap = 2,
  RoughnessG = 3,
  RoughnessA = 4,
};

struct TextureLoadOptions {
  bool srgb = false;
  bool generateMipmaps = false;
  TextureMipSemantic mipSemantic = TextureMipSemantic::Generic;
  float alphaCoverageCutoff = 0.5f;
};

[[nodiscard]] uint32_t textureMipLevelCount(uint32_t width,
                                            uint32_t height) noexcept;
[[nodiscard]] uint32_t textureMipDimension(uint32_t base,
                                           uint32_t mip) noexcept;
[[nodiscard]] Result<std::vector<std::byte>, std::string>
generateRgba8Mip(std::span<const std::byte> source, uint32_t width,
                 uint32_t height, const TextureLoadOptions &options);
[[nodiscard]] Result<std::vector<std::byte>, std::string>
generateSemanticRgba8MipChain(std::span<const std::byte> baseData,
                              uint32_t width, uint32_t height,
                              uint32_t mipLevels,
                              const TextureLoadOptions &options);
[[nodiscard]] bool
shouldGenerateSemanticRgba8MipChain(const TextureLoadOptions &options,
                                    uint32_t mipLevels) noexcept;

} // namespace nuri
