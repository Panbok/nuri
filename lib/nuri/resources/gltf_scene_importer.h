#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/scene/light.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace nuri {

struct NURI_API ImportedLightInfo {
  LightDesc desc{};
  std::string sourceName{};
  int32_t sourceNodeIndex = -1;
};

using ImportedLightSet = std::vector<ImportedLightInfo>;

class NURI_API GltfSceneImporter {
public:
  [[nodiscard]] static Result<ImportedLightSet, std::string>
  loadLightsFromFile(std::string_view path);
};

} // namespace nuri
