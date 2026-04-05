#pragma once

#include "nuri/app/editor_scene_spec.h"
#include "nuri/core/result.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace nuri {

class EditorRuntime;

class EditorSceneCatalog {
public:
  [[nodiscard]] Result<void, std::string> append(EditorSceneSpec spec);
  [[nodiscard]] const EditorSceneSpec *find(std::string_view id) const;
  [[nodiscard]] std::span<const EditorSceneEntry> entries() const noexcept;
  [[nodiscard]] bool requestActivate(std::string_view id);
  [[nodiscard]] Result<bool, std::string>
  applyPendingActivation(EditorRuntime &runtime);
  [[nodiscard]] std::string_view initialSceneId() const noexcept;
  [[nodiscard]] std::string_view activeSceneId() const noexcept;
  [[nodiscard]] uint64_t version() const noexcept { return version_; }

  void updateActive(EditorRuntime &runtime, double deltaTime);
  void shutdown(EditorRuntime &runtime);

private:
  struct StringViewHash {
    using is_transparent = void;

    [[nodiscard]] size_t operator()(std::string_view value) const noexcept {
      return std::hash<std::string_view>{}(value);
    }

    [[nodiscard]] size_t operator()(const std::string &value) const noexcept {
      return std::hash<std::string_view>{}(value);
    }
  };

  [[nodiscard]] std::optional<size_t> findIndex(std::string_view id) const;

  std::vector<EditorSceneEntry> entries_{};
  std::vector<EditorSceneSpec> specs_{};
  std::unordered_map<std::string, size_t, StringViewHash, std::equal_to<>>
      indexById_{};
  std::optional<size_t> initialIndex_{};
  std::optional<size_t> activeIndex_{};
  std::optional<size_t> pendingIndex_{};
  uint64_t version_ = 0;
};

} // namespace nuri
