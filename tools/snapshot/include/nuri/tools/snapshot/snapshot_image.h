#pragma once

#include "nuri/core/result.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/tools/snapshot/snapshot_capture_point.h"

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nuri::tools::snapshot {

struct SnapshotImage {
  uint32_t width = 0u;
  uint32_t height = 0u;
  uint32_t channelCount = 0u;
  std::vector<float> values{};
};

struct SnapshotReadbackImage {
  RenderCapturePoint point{};
  std::vector<std::byte> bytes{};
  size_t rowStride = 0u;
  std::string hash{};
};

struct SnapshotArtifactPaths {
  std::filesystem::path raw{};
  std::filesystem::path metadata{};
  std::filesystem::path preview{};
};

struct SnapshotArtifactMetadata {
  std::string target{};
  uint32_t capturePointVersion = 0u;
  std::string kind{};
  std::string format{};
  uint32_t width = 0u;
  uint32_t height = 0u;
  uint32_t mip = 0u;
  uint32_t layer = 0u;
  std::string colorSpace{};
  std::string origin{};
  std::string profile{};
  std::string payload{};
  std::string hash{};
  DDGICaptureMetadata ddgiMetadata{};
};

[[nodiscard]] size_t snapshotFormatBytesPerPixel(Format format) noexcept;
[[nodiscard]] Result<SnapshotReadbackImage, std::string>
readSnapshotCapture(GPUDevice &gpu, const RenderCapturePoint &point);
[[nodiscard]] Result<SnapshotImage, std::string>
decodeSnapshotImage(const SnapshotReadbackImage &image);
[[nodiscard]] std::string snapshotHashBytes(std::span<const std::byte> bytes);
[[nodiscard]] Result<bool, std::string>
writeSnapshotArtifacts(const SnapshotReadbackImage &image,
                       const std::filesystem::path &artifactStem,
                       SnapshotArtifactPaths &outPaths,
                       std::string_view compareProfile = {});
[[nodiscard]] Result<SnapshotImage, std::string>
readSnapshotImageFile(const std::filesystem::path &path);
[[nodiscard]] Result<SnapshotArtifactMetadata, std::string>
readSnapshotArtifactMetadata(const std::filesystem::path &path);
[[nodiscard]] Result<bool, std::string>
writeSnapshotPreviewPng(const SnapshotImage &image,
                        const std::filesystem::path &path);
[[nodiscard]] Result<bool, std::string> writeSnapshotComparisonPreviews(
    const SnapshotImage &actual, const SnapshotImage &expected,
    std::string_view profile, const std::filesystem::path &actualPath,
    const std::filesystem::path &expectedPath);

} // namespace nuri::tools::snapshot
