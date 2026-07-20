#pragma once

#include "nuri/core/result.h"
#include "nuri/core/runtime_config.h"

#include <cstddef>
#include <filesystem>
#include <vector>

namespace nuri::bakery::detail {

struct SmaaLutBakePlan {
  bool shouldBake = false;
  std::filesystem::path areaOutputPath;
  std::filesystem::path searchOutputPath;
};

struct SmaaLutArtifacts {
  std::vector<std::byte> areaRgba8;
  std::vector<std::byte> searchRgba8;
};

[[nodiscard]] SmaaLutBakePlan planSmaaLutBake(const RuntimeConfig &config,
                                              bool forceRebuild);

[[nodiscard]] SmaaLutArtifacts generateSmaaLutArtifacts();

[[nodiscard]] Result<bool, std::string>
bakeSmaaLutsToDisk(const SmaaLutBakePlan &plan);

} // namespace nuri::bakery::detail
