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

/// Reads a `.gltf` or `.glb` file and extracts `KHR_lights_punctual` node
/// lights into an `ImportedLightSet`; unsupported light extensions and scenes
/// without punctual lights return an empty result. Callers must inspect the
/// `[[nodiscard]]` Result for file I/O failures, JSON/GLB parse failures,
/// unsupported extensions, missing or invalid required node/light data, and
/// hierarchy or validation failures such as out-of-range indices or cycles.
class NURI_API GltfSceneImporter {
public:
  [[nodiscard]] static Result<ImportedLightSet, std::string>
  loadLightsFromFile(std::string_view path);
};

} // namespace nuri
