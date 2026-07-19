#pragma once
#include <cstdint>
namespace nuri {

inline constexpr uint32_t kResourceHandleIndexBits = 20u,
                          kResourceHandleGenerationBits = 12u;
static_assert(kResourceHandleIndexBits + kResourceHandleGenerationBits == 32u);
inline constexpr uint32_t kResourceHandleIndexMask =
                              (1u << kResourceHandleIndexBits) - 1u,
                          kResourceHandleGenerationMask =
                              (1u << kResourceHandleGenerationBits) - 1u;

template <typename Tag> struct PackedHandle {
  uint32_t value = 0;
  constexpr bool operator==(const PackedHandle &) const noexcept = default;
};

using TextureRef = PackedHandle<struct TextureRefTag>;
using MaterialRef = PackedHandle<struct MaterialRefTag>;
using ModelRef = PackedHandle<struct ModelRefTag>;

inline constexpr TextureRef kInvalidTextureRef{};
inline constexpr MaterialRef kInvalidMaterialRef{};
inline constexpr ModelRef kInvalidModelRef{};

[[nodiscard]] constexpr uint32_t packResourceHandle(uint32_t index,
                                                    uint32_t generation) {
  if (index > kResourceHandleIndexMask || generation == 0u ||
      generation > kResourceHandleGenerationMask) {
    return 0u;
  }
  return (generation << kResourceHandleIndexBits) | index;
}

template <typename Tag>
[[nodiscard]] constexpr bool isValid(PackedHandle<Tag> handle) {
  return handle.value != 0u;
}
template <typename Tag>
[[nodiscard]] constexpr uint32_t indexOf(PackedHandle<Tag> handle) {
  return handle.value & kResourceHandleIndexMask;
}
template <typename Tag>
[[nodiscard]] constexpr uint32_t generationOf(PackedHandle<Tag> handle) {
  return handle.value >> kResourceHandleIndexBits;
}
template <typename Tag>
[[nodiscard]] constexpr PackedHandle<Tag>
makePackedHandle(uint32_t index, uint32_t generation) {
  return {packResourceHandle(index, generation)};
}
} // namespace nuri
