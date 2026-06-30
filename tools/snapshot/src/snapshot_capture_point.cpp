#include "nuri/tools/snapshot/snapshot_capture_point.h"

#include <array>

namespace nuri::tools::snapshot {
namespace {

constexpr std::array kCatalog{
    SnapshotCaptureCatalogEntry{
        .name = "final_color",
        .kind = RenderCaptureValueKind::Color,
        .lifetime = RenderCaptureLifetimeClass::ToolCaptureTexture,
        .defaultCompareProfile = "ldr_color",
        .producer = "Present ToneMap Capture Pass",
        .coverageArea = "tonemapped present output",
    },
    SnapshotCaptureCatalogEntry{
        .name = "scene_color_hdr",
        .kind = RenderCaptureValueKind::LinearHdrColor,
        .defaultCompareProfile = "hdr_color",
        .producer = "opaque/composition scene color",
        .coverageArea = "HDR scene color before frame composition",
    },
    SnapshotCaptureCatalogEntry{
        .name = "frame_color_hdr",
        .kind = RenderCaptureValueKind::LinearHdrColor,
        .defaultCompareProfile = "hdr_color",
        .producer = "final HDR frame color before present",
        .coverageArea = "HDR frame color after transparent/HDR passes",
    },
    SnapshotCaptureCatalogEntry{
        .name = "scene_depth",
        .kind = RenderCaptureValueKind::Depth,
        .defaultCompareProfile = "depth",
        .producer = "OpaqueRenderer",
        .coverageArea = "scene depth",
    },
    SnapshotCaptureCatalogEntry{
        .name = "material_normals",
        .kind = RenderCaptureValueKind::Normal,
        .defaultCompareProfile = "normal",
        .producer = "OpaqueRenderer normal pre-pass",
        .coverageArea = "material/world normals",
    },
    SnapshotCaptureCatalogEntry{
        .name = "motion_vectors",
        .kind = RenderCaptureValueKind::Velocity,
        .defaultCompareProfile = "velocity",
        .producer = "TAA velocity path",
        .coverageArea = "motion vectors",
    },
    SnapshotCaptureCatalogEntry{
        .name = "reactive_mask",
        .kind = RenderCaptureValueKind::Mask,
        .defaultCompareProfile = "mask",
        .producer = "TAA reactive-mask path",
        .coverageArea = "reactive mask",
    },
    SnapshotCaptureCatalogEntry{
        .name = "ambient_occlusion",
        .kind = RenderCaptureValueKind::Scalar,
        .defaultCompareProfile = "scalar",
        .producer = "GTAO",
        .coverageArea = "final GTAO visibility",
    },
    SnapshotCaptureCatalogEntry{
        .name = "shadow_cascade_0",
        .kind = RenderCaptureValueKind::ShadowDepth,
        .lifetime = RenderCaptureLifetimeClass::FeaturePersistentTexture,
        .defaultCompareProfile = "shadow_depth",
        .producer = "ShadowRenderer",
        .coverageArea = "shadow cascade 0 depth",
    },
    SnapshotCaptureCatalogEntry{
        .name = "shadow_cascade_1",
        .kind = RenderCaptureValueKind::ShadowDepth,
        .lifetime = RenderCaptureLifetimeClass::FeaturePersistentTexture,
        .defaultCompareProfile = "shadow_depth",
        .producer = "ShadowRenderer",
        .coverageArea = "shadow cascade 1 depth",
    },
    SnapshotCaptureCatalogEntry{
        .name = "shadow_cascade_2",
        .kind = RenderCaptureValueKind::ShadowDepth,
        .lifetime = RenderCaptureLifetimeClass::FeaturePersistentTexture,
        .defaultCompareProfile = "shadow_depth",
        .producer = "ShadowRenderer",
        .coverageArea = "shadow cascade 2 depth",
    },
    SnapshotCaptureCatalogEntry{
        .name = "shadow_cascade_3",
        .kind = RenderCaptureValueKind::ShadowDepth,
        .lifetime = RenderCaptureLifetimeClass::FeaturePersistentTexture,
        .defaultCompareProfile = "shadow_depth",
        .producer = "ShadowRenderer",
        .coverageArea = "shadow cascade 3 depth",
    },
    SnapshotCaptureCatalogEntry{
        .name = "shadow_preview",
        .kind = RenderCaptureValueKind::DebugPreview,
        .lifetime = RenderCaptureLifetimeClass::FeaturePersistentTexture,
        .defaultCompareProfile = "debug_preview",
        .producer = "ShadowRenderer",
        .coverageArea = "shadow depth preview",
    },
    SnapshotCaptureCatalogEntry{
        .name = "transmission_visibility_depth",
        .kind = RenderCaptureValueKind::Depth,
        .lifetime = RenderCaptureLifetimeClass::FeaturePersistentTexture,
        .defaultCompareProfile = "depth",
        .producer = "OpaqueRenderer",
        .coverageArea = "transmission visibility depth",
    },
    SnapshotCaptureCatalogEntry{
        .name = "transmission_feedback",
        .kind = RenderCaptureValueKind::LinearHdrColor,
        .lifetime = RenderCaptureLifetimeClass::FeaturePersistentTexture,
        .defaultCompareProfile = "hdr_color",
        .producer = "TransparentRenderer",
        .coverageArea = "full-res transmission feedback",
    },
    SnapshotCaptureCatalogEntry{
        .name = "transmission_feedback_half",
        .kind = RenderCaptureValueKind::LinearHdrColor,
        .lifetime = RenderCaptureLifetimeClass::FeaturePersistentTexture,
        .defaultCompareProfile = "hdr_color",
        .producer = "TransparentRenderer",
        .coverageArea = "half-res transmission feedback",
    },
    SnapshotCaptureCatalogEntry{
        .name = "transmission_feedback_quarter",
        .kind = RenderCaptureValueKind::LinearHdrColor,
        .lifetime = RenderCaptureLifetimeClass::FeaturePersistentTexture,
        .defaultCompareProfile = "hdr_color",
        .producer = "TransparentRenderer",
        .coverageArea = "quarter-res transmission feedback",
    },
    SnapshotCaptureCatalogEntry{
        .name = "gtao_edges",
        .kind = RenderCaptureValueKind::Mask,
        .defaultCompareProfile = "mask",
        .producer = "GTAO",
        .availability = SnapshotCaptureAvailability::KnownNotCapturable,
        .coverageArea = "GTAO edge mask",
        .diagnosticWork = "requires explicit transient capture copy",
    },
    SnapshotCaptureCatalogEntry{
        .name = "gtao_raw",
        .kind = RenderCaptureValueKind::Scalar,
        .defaultCompareProfile = "scalar",
        .producer = "GTAO",
        .availability = SnapshotCaptureAvailability::KnownNotCapturable,
        .coverageArea = "raw GTAO visibility",
        .diagnosticWork = "requires explicit transient capture copy",
    },
};

} // namespace

std::span<const SnapshotCaptureCatalogEntry> snapshotCaptureCatalog() noexcept {
  return kCatalog;
}

const SnapshotCaptureCatalogEntry *
findSnapshotCaptureCatalogEntry(std::string_view name) noexcept {
  for (const SnapshotCaptureCatalogEntry &entry : kCatalog) {
    if (entry.name == name) {
      return &entry;
    }
  }
  return nullptr;
}

std::string_view
renderCaptureValueKindName(RenderCaptureValueKind kind) noexcept {
  switch (kind) {
  case RenderCaptureValueKind::Color:
    return "color";
  case RenderCaptureValueKind::LinearHdrColor:
    return "linear_hdr_color";
  case RenderCaptureValueKind::Depth:
    return "depth";
  case RenderCaptureValueKind::ShadowDepth:
    return "shadow_depth";
  case RenderCaptureValueKind::Normal:
    return "normal";
  case RenderCaptureValueKind::Velocity:
    return "velocity";
  case RenderCaptureValueKind::Mask:
    return "mask";
  case RenderCaptureValueKind::Scalar:
    return "scalar";
  case RenderCaptureValueKind::DebugPreview:
    return "debug_preview";
  }
  return "unknown";
}

std::string_view
renderCaptureLifetimeName(RenderCaptureLifetimeClass lifetime) noexcept {
  switch (lifetime) {
  case RenderCaptureLifetimeClass::FrameSharedRingTexture:
    return "frame_shared_ring_texture";
  case RenderCaptureLifetimeClass::FeaturePersistentTexture:
    return "feature_persistent_texture";
  case RenderCaptureLifetimeClass::ToolCaptureTexture:
    return "tool_capture_texture";
  case RenderCaptureLifetimeClass::CaptureCopyTexture:
    return "capture_copy_texture";
  }
  return "unknown";
}

} // namespace nuri::tools::snapshot
