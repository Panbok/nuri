#include "nuri/editor_pch.h"

#include "nuri/app/editor_runtime.h"
#include "nuri/app/editor_scene_catalog.h"
#include "nuri/core/profiling.h"

namespace nuri {

[[nodiscard]] Result<void, std::string>
EditorSceneCatalog::append(EditorSceneSpec spec) {
  if (spec.info.id.empty()) {
    return Result<void, std::string>::makeError(
        "EditorSceneCatalog::append: scene id must not be empty");
  }
  if (findIndex(spec.info.id).has_value()) {
    return Result<void, std::string>::makeError(
        std::string("EditorSceneCatalog::append: duplicate scene id '") +
        spec.info.id + "'");
  }
  if (spec.info.initiallySelected && initialIndex_.has_value()) {
    return Result<void, std::string>::makeError(
        std::string("EditorSceneCatalog::append: multiple initial scenes: '") +
        entries_[*initialIndex_].info.id + "' and '" + spec.info.id + "'");
  }

  const size_t index = entries_.size();
  entries_.push_back(EditorSceneEntry{.info = spec.info});
  specs_.push_back(std::move(spec));
  indexById_.emplace(entries_.back().info.id, index);
  if (entries_.back().info.initiallySelected) {
    initialIndex_ = index;
  }
  ++version_;
  return Result<void, std::string>::makeResult();
}

const EditorSceneSpec *EditorSceneCatalog::find(std::string_view id) const {
  const auto index = findIndex(id);
  return index.has_value() ? &specs_[*index] : nullptr;
}

std::span<const EditorSceneEntry> EditorSceneCatalog::entries() const noexcept {
  return entries_;
}

bool EditorSceneCatalog::requestActivate(std::string_view id) {
  const auto index = findIndex(id);
  if (!index.has_value()) {
    return false;
  }
  pendingIndex_ = *index;
  return true;
}

Result<bool, std::string>
EditorSceneCatalog::applyPendingActivation(EditorRuntime &runtime) {
  NURI_PROFILER_FUNCTION();
  if (!pendingIndex_.has_value()) {
    return Result<bool, std::string>::makeResult(false);
  }

  const size_t targetIndex = *pendingIndex_;
  pendingIndex_.reset();
  if (activeIndex_ == targetIndex) {
    return Result<bool, std::string>::makeResult(false);
  }

  EditorSceneEntry &targetEntry = entries_[targetIndex];
  EditorSceneSpec &targetSpec = specs_[targetIndex];
  if (!targetEntry.prepared && targetSpec.prepare) {
    EditorScenePrepareContext prepareContext{.runtime = runtime};
    auto prepareResult = targetSpec.prepare(prepareContext);
    if (prepareResult.hasError()) {
      return Result<bool, std::string>::makeError(prepareResult.error());
    }
    targetEntry.prepared = true;
  }

  const std::optional<size_t> previousActiveIndex = activeIndex_;
  if (activeIndex_.has_value()) {
    EditorSceneEntry &activeEntry = entries_[*activeIndex_];
    EditorSceneSpec &activeSpec = specs_[*activeIndex_];
    if (activeSpec.deactivate) {
      EditorSceneDeactivateContext deactivateContext{.runtime = runtime};
      activeSpec.deactivate(deactivateContext);
    }
    activeEntry.active = false;
  }

  runtime.resetSceneState();
  if (targetSpec.activate) {
    EditorSceneActivateContext activateContext{.runtime = runtime};
    auto activateResult = targetSpec.activate(activateContext);
    if (activateResult.hasError()) {
      activeIndex_.reset();
      if (previousActiveIndex.has_value()) {
        EditorSceneEntry &previousEntry = entries_[*previousActiveIndex];
        EditorSceneSpec &previousSpec = specs_[*previousActiveIndex];
        runtime.resetSceneState();
        if (previousSpec.activate &&
            !previousSpec.activate(activateContext).hasError()) {
          activeIndex_ = *previousActiveIndex;
          previousEntry.active = true;
        }
      }
      return Result<bool, std::string>::makeError(activateResult.error());
    }
  }

  activeIndex_ = targetIndex;
  targetEntry.active = true;
  return Result<bool, std::string>::makeResult(true);
}

std::string_view EditorSceneCatalog::initialSceneId() const noexcept {
  if (initialIndex_.has_value()) {
    return entries_[*initialIndex_].info.id;
  }
  return !entries_.empty() ? std::string_view(entries_.front().info.id)
                           : std::string_view{};
}

std::string_view EditorSceneCatalog::activeSceneId() const noexcept {
  return activeIndex_.has_value() ? entries_[*activeIndex_].info.id
                                  : std::string_view{};
}

void EditorSceneCatalog::updateActive(EditorRuntime &runtime,
                                      double deltaTime) {
  NURI_PROFILER_FUNCTION();
  if (!activeIndex_.has_value()) {
    return;
  }
  EditorSceneSpec &activeSpec = specs_[*activeIndex_];
  if (activeSpec.update) {
    EditorSceneUpdateContext updateContext{
        .runtime = runtime,
        .deltaTime = deltaTime,
    };
    activeSpec.update(updateContext);
  }
}

void EditorSceneCatalog::shutdown(EditorRuntime &runtime) {
  NURI_PROFILER_FUNCTION();
  if (!activeIndex_.has_value()) {
    return;
  }
  EditorSceneEntry &activeEntry = entries_[*activeIndex_];
  EditorSceneSpec &activeSpec = specs_[*activeIndex_];
  if (activeSpec.deactivate) {
    EditorSceneDeactivateContext deactivateContext{.runtime = runtime};
    activeSpec.deactivate(deactivateContext);
  }
  runtime.resetSceneState();
  activeEntry.active = false;
  activeIndex_.reset();
}

std::optional<size_t> EditorSceneCatalog::findIndex(std::string_view id) const {
  const auto it = indexById_.find(id);
  return it != indexById_.end() ? std::optional<size_t>(it->second)
                                : std::nullopt;
}

} // namespace nuri
