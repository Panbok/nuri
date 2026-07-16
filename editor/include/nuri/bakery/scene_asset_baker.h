#pragma once

#include "nuri/bakery/bakery_types.h"
#include "nuri/core/result.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace nuri::bakery::detail {

struct SceneTextureArtifactBakePlan {
  bool shouldBake = false;
  std::filesystem::path scenePath;
  std::filesystem::path materialCachePath;
  std::vector<SceneTextureArtifactTarget> prebuildNativeTargets{};
  bool forceRebuild = false;
};

struct SceneTextureArtifactBakeStats {
  bool wroteAnyFiles = false;
  uint32_t artifactsWritten = 0u;
  uint64_t artifactBytesWritten = 0u;
  uint32_t ddsPackEntries = 0u;
  uint64_t ddsPackBytes = 0u;
  bool wroteDdsPack = false;
  bool wroteMaterialCache = false;
};

[[nodiscard]] Result<SceneTextureArtifactBakePlan, std::string>
planSceneTextureArtifactsBake(const SceneTextureArtifactsBakeRequest &request);

[[nodiscard]] Result<SceneTextureArtifactBakeStats, std::string>
bakeSceneTextureArtifactsToDisk(const SceneTextureArtifactBakePlan &plan);

} // namespace nuri::bakery::detail
