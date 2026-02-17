#pragma once

#include <cstdint>
#include <memory>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "nuri/core/result.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/math/types.h"
#include "nuri/resources/cpu/mesh_data.h"
#include "nuri/resources/mesh_importer.h"

namespace nuri {

class NURI_API Model final {
public:
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
      std::pmr::memory_resource *mem = std::pmr::get_default_resource(),
      std::string_view debugName = {});

  [[nodiscard]] GeometryAllocationHandle geometryHandle() const noexcept {
    return geometry_;
  }
  [[nodiscard]] std::span<const Submesh> submeshes() const noexcept {
    return submeshes_;
  }
  [[nodiscard]] uint32_t vertexCount() const noexcept { return vertexCount_; }
  [[nodiscard]] uint32_t indexCount() const noexcept { return indexCount_; }
  [[nodiscard]] const BoundingBox &bounds() const noexcept { return bounds_; }

private:
  Model(GPUDevice &gpu, GeometryAllocationHandle geometry,
        std::vector<Submesh> submeshes, uint32_t vertexCount,
        uint32_t indexCount, BoundingBox bounds)
      : gpu_(&gpu), geometry_(geometry), submeshes_(std::move(submeshes)),
        vertexCount_(vertexCount), indexCount_(indexCount),
        bounds_(std::move(bounds)) {}

  GPUDevice *gpu_ = nullptr;
  GeometryAllocationHandle geometry_{};
  std::vector<Submesh> submeshes_;
  uint32_t vertexCount_ = 0;
  uint32_t indexCount_ = 0;
  BoundingBox bounds_{};
};

using Mesh = Model;

} // namespace nuri
