#pragma once

#include <cstddef>
#include <cstdint>

namespace nuri::smaa_lut {

inline constexpr char kAreaFilename[] = "smaa_area_rgba8.bin";
inline constexpr char kSearchFilename[] = "smaa_search_rgba8.bin";
inline constexpr uint32_t kAreaWidth = 160u;
inline constexpr uint32_t kAreaHeight = 560u;
inline constexpr uint32_t kSearchWidth = 64u;
inline constexpr uint32_t kSearchHeight = 16u;
inline constexpr size_t kAreaByteCount =
    static_cast<size_t>(kAreaWidth) * kAreaHeight * 4u;
inline constexpr size_t kSearchByteCount =
    static_cast<size_t>(kSearchWidth) * kSearchHeight * 4u;

} // namespace nuri::smaa_lut
