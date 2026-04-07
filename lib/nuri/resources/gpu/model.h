#pragma once

#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <memory_resource>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "nuri/core/result.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/math/types.h"
#include "nuri/resources/cpu/mesh_data.h"
#include "nuri/resources/gpu/buffer.h"
#include "nuri/resources/mesh_importer.h"

namespace nuri {

class Model;
struct ModelAnimationPackedData;

enum class PackedVertexFormat : uint8_t {
  StaticQuantized20 = 0,
  AnimatedFloat24 = 1,
};

struct StaticVertexDecodeGpuData {
  glm::vec4 offset{0.0f, 0.0f, 0.0f, 0.0f};
  glm::vec4 scale{1.0f, 1.0f, 1.0f, 0.0f};
};

// Handle for asynchronous model loading. Uses std::shared_future for warmup so
// that the destructor does not block (std::future's destructor would wait for
// the async task); the shared state is released when the last reference goes
// away.
class NURI_API ModelAsyncLoad final {
public:
  ModelAsyncLoad() = default;
  ~ModelAsyncLoad() = default;

  ModelAsyncLoad(const ModelAsyncLoad &) = delete;
  ModelAsyncLoad &operator=(const ModelAsyncLoad &) = delete;
  ModelAsyncLoad(ModelAsyncLoad &&) noexcept = default;
  ModelAsyncLoad &operator=(ModelAsyncLoad &&) noexcept = default;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] bool isInFlight() const noexcept;
  [[nodiscard]] bool isReady() const;
  [[nodiscard]] bool isFinalized() const noexcept { return finalized_; }
  [[nodiscard]] std::optional<bool> cacheHit() const noexcept;
  [[nodiscard]] std::string_view warmupError() const noexcept;

  // Non-blocking: returns an error while warmup is still in progress.
  // Returns true when cache was hit, false when cache was rebuilt.
  [[nodiscard]] Result<bool, std::string> resolveWarmup();

  // Final GPU model creation step. Must be called on a thread that is valid
  // for GPUDevice usage.
  [[nodiscard]] Result<std::unique_ptr<Model>, std::string>
  finalize(GPUDevice &gpu,
           std::pmr::memory_resource *mem = std::pmr::get_default_resource(),
           std::string_view debugName = {});

private:
  friend class Model;
  explicit ModelAsyncLoad(std::string sourcePath, MeshImportOptions options,
                          std::future<Result<bool, std::string>> warmupFuture)
      : sourcePath_(std::move(sourcePath)), options_(std::move(options)),
        warmupFuture_(std::move(warmupFuture)) {}

  std::string sourcePath_{};
  MeshImportOptions options_{};
  std::shared_future<Result<bool, std::string>> warmupFuture_{};
  bool warmupCompleted_ = false;
  bool warmupCacheHit_ = false;
  std::string warmupError_{};
  bool finalized_ = false;
};

class NURI_API Model final {
public:
  static constexpr uint32_t kInvalidMaterialIndex =
      std::numeric_limits<uint32_t>::max();

  struct ModelAnimationGpuView {
    BufferHandle skinInfluenceBuffer{};
    BufferHandle morphMetaBuffer{};
    BufferHandle morphDeltaBuffer{};
    uint32_t skinInfluenceCount = 0;
    uint32_t morphTargetCount = 0;
    uint32_t morphVertexCount = 0;
  };

  ~Model();

  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;
  Model(Model &&) = delete;
  Model &operator=(Model &&) = delete;

  [[nodiscard]] static Result<std::unique_ptr<Model>, std::string>
  create(GPUDevice &gpu, const MeshData &data, std::string_view debugName = {});

  [[nodiscard]] static Result<std::unique_ptr<Model>, std::string>
  createFromFile(
      GPUDevice &gpu, std::string_view path,
      const MeshImportOptions &options = {},
      // Used only for transient import/cache-read allocations.
      // Model-owned data is allocated on stable default PMR storage.
      std::pmr::memory_resource *mem = std::pmr::get_default_resource(),
      std::string_view debugName = {});

  // Async-friendly path:
  // 1) Start background CPU cache warmup/import work.
  // 2) Poll ModelAsyncLoad and finalize on the GPU thread when ready.
  [[nodiscard]] static Result<ModelAsyncLoad, std::string>
  createFromFileAsync(std::string_view path,
                      const MeshImportOptions &options = {});

  [[nodiscard]] GeometryAllocationHandle geometryHandle() const noexcept {
    return geometry_;
  }
  [[nodiscard]] std::span<const Submesh> submeshes() const noexcept {
    return submeshes_;
  }
  [[nodiscard]] uint32_t vertexCount() const noexcept { return vertexCount_; }
  [[nodiscard]] uint32_t indexCount() const noexcept { return indexCount_; }
  [[nodiscard]] const BoundingBox &bounds() const noexcept { return bounds_; }
  [[nodiscard]] const ModelAnimationGpuView &animationGpuView() const noexcept {
    return animationGpuView_;
  }
  [[nodiscard]] PackedVertexFormat drawVertexFormat() const noexcept {
    return drawVertexFormat_;
  }
  [[nodiscard]] BufferHandle vertexDecodeBuffer() const noexcept {
    return vertexDecodeBuffer_ != nullptr ? vertexDecodeBuffer_->handle()
                                          : BufferHandle{};
  }
  [[nodiscard]] uint64_t vertexDecodeBufferAddress() const noexcept {
    return vertexDecodeBufferAddress_;
  }
  [[nodiscard]] bool hasAnimationData() const noexcept {
    return nuri::isValid(animationGpuView_.skinInfluenceBuffer) ||
           nuri::isValid(animationGpuView_.morphDeltaBuffer);
  }
  [[nodiscard]] uint32_t sourceMaterialCount() const noexcept {
    return static_cast<uint32_t>(sourceMaterialToRuntime_.size());
  }
  [[nodiscard]] uint32_t
  materialIndexForSource(uint32_t sourceMaterialIndex) const noexcept;
  [[nodiscard]] uint32_t
  materialIndexForSubmesh(uint32_t submeshIndex) const noexcept;
  [[nodiscard]] bool setMaterialIndexForSource(uint32_t sourceMaterialIndex,
                                               uint32_t materialIndex) noexcept;
  void setMaterialIndexForAllSources(uint32_t materialIndex) noexcept;

private:
  [[nodiscard]] static Result<std::unique_ptr<Model>, std::string>
  createFromPackedVertices(GPUDevice &gpu, const MeshData &data,
                           std::span<const std::byte> packedVertexBytes,
                           PackedVertexFormat packedVertexFormat,
                           std::span<const std::byte> staticDecodeBytes,
                           uint32_t staticDecodeCount,
                           const ModelAnimationPackedData *animationPackedData,
                           std::string_view debugName);

  // CPU-only path that ensures an up-to-date mesh cache file exists.
  // Returns true when a valid cache was already present, false when rebuilt.
  [[nodiscard]] static Result<bool, std::string> warmFileCache(
      std::string_view path, const MeshImportOptions &options = {},
      // Used for transient import allocations during warmup only.
      std::pmr::memory_resource *mem = std::pmr::get_default_resource());

  Model(GPUDevice &gpu, GeometryAllocationHandle geometry,
        std::pmr::vector<Submesh> submeshes, uint32_t vertexCount,
        uint32_t indexCount, BoundingBox bounds,
        PackedVertexFormat drawVertexFormat,
        ModelAnimationGpuView animationGpuView,
        uint64_t vertexDecodeBufferAddress,
        std::unique_ptr<Buffer> vertexDecodeBuffer,
        std::unique_ptr<Buffer> skinInfluenceBuffer,
        std::unique_ptr<Buffer> morphMetaBuffer,
        std::unique_ptr<Buffer> morphDeltaBuffer,
        std::pmr::vector<uint32_t> sourceMaterialToRuntime)
      : gpu_(&gpu), geometry_(geometry), submeshes_(std::move(submeshes)),
        vertexCount_(vertexCount), indexCount_(indexCount), bounds_(bounds),
        drawVertexFormat_(drawVertexFormat),
        animationGpuView_(animationGpuView),
        vertexDecodeBufferAddress_(vertexDecodeBufferAddress),
        vertexDecodeBuffer_(std::move(vertexDecodeBuffer)),
        skinInfluenceBuffer_(std::move(skinInfluenceBuffer)),
        morphMetaBuffer_(std::move(morphMetaBuffer)),
        morphDeltaBuffer_(std::move(morphDeltaBuffer)),
        sourceMaterialToRuntime_(std::move(sourceMaterialToRuntime)) {}

  GPUDevice *gpu_ = nullptr;
  GeometryAllocationHandle geometry_{};
  std::pmr::vector<Submesh> submeshes_;
  uint32_t vertexCount_ = 0;
  uint32_t indexCount_ = 0;
  BoundingBox bounds_{};
  PackedVertexFormat drawVertexFormat_ = PackedVertexFormat::StaticQuantized20;
  ModelAnimationGpuView animationGpuView_{};
  uint64_t vertexDecodeBufferAddress_ = 0u;
  std::unique_ptr<Buffer> vertexDecodeBuffer_;
  std::unique_ptr<Buffer> skinInfluenceBuffer_;
  std::unique_ptr<Buffer> morphMetaBuffer_;
  std::unique_ptr<Buffer> morphDeltaBuffer_;
  std::pmr::vector<uint32_t> sourceMaterialToRuntime_;
};

using Mesh = Model;

} // namespace nuri
