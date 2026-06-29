#pragma once

#include "nuri/gfx/frame/render_frame_context.h"

#include <span>
#include <string>
#include <string_view>

namespace nuri::tools::snapshot {

enum class SnapshotCaptureAvailability {
  FirstSlice,
  KnownNotCapturable,
};

struct SnapshotCaptureCatalogEntry {
  std::string_view name{};
  RenderCaptureValueKind kind = RenderCaptureValueKind::Color;
  RenderCaptureLifetimeClass lifetime =
      RenderCaptureLifetimeClass::FrameSharedRingTexture;
  std::string_view defaultCompareProfile{};
  std::string_view producer{};
  uint32_t version = 1u;
  SnapshotCaptureAvailability availability =
      SnapshotCaptureAvailability::FirstSlice;
  std::string_view coverageArea{};
  std::string_view diagnosticWork{};
};

[[nodiscard]] std::span<const SnapshotCaptureCatalogEntry>
snapshotCaptureCatalog() noexcept;
[[nodiscard]] const SnapshotCaptureCatalogEntry *
findSnapshotCaptureCatalogEntry(std::string_view name) noexcept;
[[nodiscard]] std::string_view
renderCaptureValueKindName(RenderCaptureValueKind kind) noexcept;
[[nodiscard]] std::string_view
renderCaptureLifetimeName(RenderCaptureLifetimeClass lifetime) noexcept;

} // namespace nuri::tools::snapshot

