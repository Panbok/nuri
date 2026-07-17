#pragma once

#include "nuri/core/result.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/owned_gpu_resource.h"

namespace nuri {

class NURI_API Buffer final {
public:
  ~Buffer() = default;

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&) = delete;
  Buffer &operator=(Buffer &&) = delete;

  [[nodiscard]] static Result<std::unique_ptr<Buffer>, std::string>
  create(GPUDevice &gpu, const BufferDesc &desc,
         std::string_view debugName = {}) {
    auto result = gpu.createBuffer(desc, debugName);
    if (result.hasError()) {
      return Result<std::unique_ptr<Buffer>, std::string>::makeError(
          result.error());
    }

    return Result<std::unique_ptr<Buffer>, std::string>::makeResult(
        std::unique_ptr<Buffer>(new Buffer(
            gpu, result.value(), desc.size != 0 ? desc.size : desc.data.size(),
            desc.usage, desc.storage, std::string(debugName))));
  }

  [[nodiscard]] static Result<std::unique_ptr<Buffer>, std::string>
  publishPrepared(GPUDevice &gpu, std::unique_ptr<PreparedGpuBuffer> prepared,
                  const BufferDesc &desc, std::string_view debugName = {}) {
    auto result = gpu.publishPreparedBuffer(std::move(prepared));
    if (result.hasError()) {
      return Result<std::unique_ptr<Buffer>, std::string>::makeError(
          result.error());
    }
    return Result<std::unique_ptr<Buffer>, std::string>::makeResult(
        std::unique_ptr<Buffer>(new Buffer(
            gpu, result.value(), desc.size != 0 ? desc.size : desc.data.size(),
            desc.usage, desc.storage, std::string(debugName))));
  }

  [[nodiscard]] BufferHandle handle() const noexcept { return resource_.get(); }
  [[nodiscard]] size_t size() const noexcept { return size_; }
  [[nodiscard]] BufferUsage usage() const noexcept { return usage_; }
  [[nodiscard]] Storage storage() const noexcept { return storage_; }
  [[nodiscard]] std::string_view debugName() const noexcept {
    return debugName_;
  }
  [[nodiscard]] bool valid() const noexcept { return resource_.valid(); }

private:
  Buffer(GPUDevice &gpu, BufferHandle handle, size_t size, BufferUsage usage,
         Storage storage, std::string debugName)
      : resource_(gpu, handle), size_(size), usage_(usage), storage_(storage),
        debugName_(std::move(debugName)) {}

  OwnedBufferHandle resource_;
  size_t size_ = 0;
  BufferUsage usage_ = BufferUsage::Vertex;
  Storage storage_ = Storage::Device;
  std::string debugName_;
};

} // namespace nuri
