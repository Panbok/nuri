#pragma once

#include "nuri/bakery/bakery_types.h"
#include "nuri/core/result.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace nuri::bakery::detail {

struct ScenePortableBakePlan {
  bool shouldBake = false;
  std::filesystem::path scenePath;
  std::filesystem::path materialCachePath;
  std::vector<ScenePortableTextureTarget> prebuildNativeTargets{};
};

struct ScenePortableBakeStats {
  bool wroteAnyFiles = false;
  uint32_t portableTexturesWritten = 0u;
  uint32_t nativeTexturesWritten = 0u;
  uint64_t portableBytesWritten = 0u;
  uint64_t nativeBytesWritten = 0u;
  bool wroteMaterialCache = false;
};

[[nodiscard]] Result<ScenePortableBakePlan, std::string>
planScenePortableAssetsBake(const ScenePortableAssetsBakeRequest &request);

[[nodiscard]] Result<ScenePortableBakeStats, std::string>
bakeScenePortableAssetsToDisk(const ScenePortableBakePlan &plan);

} // namespace nuri::bakery::detail
