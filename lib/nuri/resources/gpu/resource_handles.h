#pragma once

#include "nuri/defines.h"

#include <cstdint>

namespace nuri {

inline constexpr uint32_t kResourceHandleIndexBits = 20u;
inline constexpr uint32_t kResourceHandleGenerationBits = 12u;
inline constexpr uint32_t kResourceHandleIndexMask =
    (1u << kResourceHandleIndexBits) - 1u;
inline constexpr uint32_t kResourceHandleGenerationMask =
    (1u << kResourceHandleGenerationBits) - 1u;

struct NURI_API TextureRef {
  uint32_t value = 0;
};

struct NURI_API MaterialRef {
  uint32_t value = 0;
};

struct NURI_API ModelRef {
  uint32_t value = 0;
};

inline constexpr TextureRef kInvalidTextureRef{};
inline constexpr MaterialRef kInvalidMaterialRef{};
inline constexpr ModelRef kInvalidModelRef{};

struct NURI_API ResourceHandleParts {
  uint32_t index = 0;
  uint32_t generation = 0;
};

[[nodiscard]] constexpr uint32_t packResourceHandle(uint32_t index,
                                                    uint32_t generation) {
  if (index > kResourceHandleIndexMask || generation == 0u ||
      generation > kResourceHandleGenerationMask) {
    return 0u;
  }
  return (generation << kResourceHandleIndexBits) | index;
}

[[nodiscard]] constexpr ResourceHandleParts
unpackResourceHandle(uint32_t packed) {
  if (packed == 0u) {
    return ResourceHandleParts{};
  }
  return ResourceHandleParts{
      .index = packed & kResourceHandleIndexMask,
      .generation =
          (packed >> kResourceHandleIndexBits) & kResourceHandleGenerationMask,
  };
}

[[nodiscard]] constexpr bool isValid(TextureRef ref) { return ref.value != 0u; }
[[nodiscard]] constexpr bool isValid(MaterialRef ref) {
  return ref.value != 0u;
}
[[nodiscard]] constexpr bool isValid(ModelRef ref) { return ref.value != 0u; }

[[nodiscard]] constexpr uint32_t indexOf(TextureRef ref) {
  return unpackResourceHandle(ref.value).index;
}
[[nodiscard]] constexpr uint32_t generationOf(TextureRef ref) {
  return unpackResourceHandle(ref.value).generation;
}

[[nodiscard]] constexpr uint32_t indexOf(MaterialRef ref) {
  return unpackResourceHandle(ref.value).index;
}
[[nodiscard]] constexpr uint32_t generationOf(MaterialRef ref) {
  return unpackResourceHandle(ref.value).generation;
}

[[nodiscard]] constexpr uint32_t indexOf(ModelRef ref) {
  return unpackResourceHandle(ref.value).index;
}
[[nodiscard]] constexpr uint32_t generationOf(ModelRef ref) {
  return unpackResourceHandle(ref.value).generation;
}

[[nodiscard]] constexpr TextureRef makeTextureRef(uint32_t index,
                                                  uint32_t generation) {
  return TextureRef{packResourceHandle(index, generation)};
}
[[nodiscard]] constexpr MaterialRef makeMaterialRef(uint32_t index,
                                                    uint32_t generation) {
  return MaterialRef{packResourceHandle(index, generation)};
}
[[nodiscard]] constexpr ModelRef makeModelRef(uint32_t index,
                                              uint32_t generation) {
  return ModelRef{packResourceHandle(index, generation)};
}

[[nodiscard]] constexpr uint32_t nextResourceGeneration(uint32_t generation) {
  const uint32_t next = (generation + 1u) & kResourceHandleGenerationMask;
  return next == 0u ? 1u : next;
}

} // namespace nuri
