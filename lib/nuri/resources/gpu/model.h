#pragma once

#include "nuri/core/result.h"
#include "nuri/resources/cpu/mesh_data.h"
#include "nuri/resources/gpu/buffer.h"
#include "nuri/resources/mesh_importer.h"

namespace nuri {

class GPUDevice;

class NURI_API Model final {
public:
  ~Model() = default;

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

  [[nodiscard]] const Buffer *vertexBuffer() const noexcept {
    return vertexBuffer_.get();
  }
  [[nodiscard]] const Buffer *indexBuffer() const noexcept {
    return indexBuffer_.get();
  }
  [[nodiscard]] std::span<const Submesh> submeshes() const noexcept {
    return submeshes_;
  }
  [[nodiscard]] uint32_t vertexCount() const noexcept { return vertexCount_; }
  [[nodiscard]] uint32_t indexCount() const noexcept { return indexCount_; }

private:
  Model(std::unique_ptr<Buffer> &&vertexBuffer,
        std::unique_ptr<Buffer> &&indexBuffer, std::vector<Submesh> submeshes,
        uint32_t vertexCount, uint32_t indexCount)
      : vertexBuffer_(std::move(vertexBuffer)),
        indexBuffer_(std::move(indexBuffer)), submeshes_(std::move(submeshes)),
        vertexCount_(vertexCount), indexCount_(indexCount) {}

  std::unique_ptr<Buffer> vertexBuffer_;
  std::unique_ptr<Buffer> indexBuffer_;
  std::vector<Submesh> submeshes_;
  uint32_t vertexCount_ = 0;
  uint32_t indexCount_ = 0;
};

using Mesh = Model;

} // namespace nuri
