#include "nuri/pch.h"

#include "nuri/gfx/pipeline/providers/material_table_gpu_provider.h"

#include "nuri/core/profiling.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/resource_manager.h"

namespace nuri {
namespace {

template <typename T>
std::span<const std::byte> tableBytes(std::span<const T> values,
                                      const T &fallback) {
  if (values.empty()) {
    return std::as_bytes(std::span<const T>(&fallback, size_t{1u}));
  }
  return std::as_bytes(values);
}

template <typename T> size_t requiredTableBytes(std::span<const T> values) {
  return std::max(values.size() * sizeof(T), sizeof(T));
}

} // namespace

MaterialTableGpuProvider::MaterialTableGpuProvider(GPUDevice &gpu)
    : gpu_(gpu) {}

MaterialTableGpuProvider::~MaterialTableGpuProvider() { destroyBuffers(); }

Result<bool, std::string> MaterialTableGpuProvider::ensureBufferCapacity(
    std::unique_ptr<Buffer> &buffer, size_t &capacityBytes,
    size_t requiredBytes, std::string_view debugName) {
  if (buffer && buffer->valid() && capacityBytes >= requiredBytes) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (buffer && buffer->valid()) {
    gpu_.destroyBuffer(buffer->handle());
  }
  buffer.reset();
  auto createResult = Buffer::create(gpu_,
                                     BufferDesc{.usage = BufferUsage::Storage,
                                                .storage = Storage::Device,
                                                .size = requiredBytes},
                                     debugName);
  if (createResult.hasError()) {
    return Result<bool, std::string>::makeError(createResult.error());
  }
  buffer = std::move(createResult.value());
  capacityBytes = requiredBytes;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
MaterialTableGpuProvider::prepare(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  if (ctx.frame.resources == nullptr) {
    return Result<bool, std::string>::makeError(
        "MaterialTableGpuProvider::prepare: frame resources are null");
  }

  const MaterialTableSnapshot snapshot =
      ctx.frame.resources->materialSnapshot();
  auto ensureHeaderResult = ensureBufferCapacity(
      headerBuffer_, headerCapacityBytes_, requiredTableBytes(snapshot.headers),
      "material_header_table");
  if (ensureHeaderResult.hasError()) {
    return ensureHeaderResult;
  }
  auto ensureClearcoatResult = ensureBufferCapacity(
      clearcoatBuffer_, clearcoatCapacityBytes_,
      requiredTableBytes(snapshot.clearcoat), "material_clearcoat_table");
  if (ensureClearcoatResult.hasError()) {
    return ensureClearcoatResult;
  }
  auto ensureSheenResult = ensureBufferCapacity(
      sheenBuffer_, sheenCapacityBytes_, requiredTableBytes(snapshot.sheen),
      "material_sheen_table");
  if (ensureSheenResult.hasError()) {
    return ensureSheenResult;
  }
  auto ensureTransmissionResult = ensureBufferCapacity(
      transmissionBuffer_, transmissionCapacityBytes_,
      requiredTableBytes(snapshot.transmission), "material_transmission_table");
  if (ensureTransmissionResult.hasError()) {
    return ensureTransmissionResult;
  }
  auto ensureSpecularResult = ensureBufferCapacity(
      specularBuffer_, specularCapacityBytes_,
      requiredTableBytes(snapshot.specular), "material_specular_table");
  if (ensureSpecularResult.hasError()) {
    return ensureSpecularResult;
  }

  if (uploadedVersion_ != snapshot.version) {
    const MaterialHeaderGpuData defaultHeader{};
    const MaterialClearcoatGpuData defaultClearcoat{};
    const MaterialSheenGpuData defaultSheen{};
    const MaterialTransmissionGpuData defaultTransmission{};
    const MaterialSpecularGpuData defaultSpecular{};

    auto updateHeaderResult =
        gpu_.updateBuffer(headerBuffer_->handle(),
                          tableBytes(snapshot.headers, defaultHeader), 0u);
    if (updateHeaderResult.hasError()) {
      return updateHeaderResult;
    }
    auto updateClearcoatResult =
        gpu_.updateBuffer(clearcoatBuffer_->handle(),
                          tableBytes(snapshot.clearcoat, defaultClearcoat), 0u);
    if (updateClearcoatResult.hasError()) {
      return updateClearcoatResult;
    }
    auto updateSheenResult = gpu_.updateBuffer(
        sheenBuffer_->handle(), tableBytes(snapshot.sheen, defaultSheen), 0u);
    if (updateSheenResult.hasError()) {
      return updateSheenResult;
    }
    auto updateTransmissionResult = gpu_.updateBuffer(
        transmissionBuffer_->handle(),
        tableBytes(snapshot.transmission, defaultTransmission), 0u);
    if (updateTransmissionResult.hasError()) {
      return updateTransmissionResult;
    }
    auto updateSpecularResult =
        gpu_.updateBuffer(specularBuffer_->handle(),
                          tableBytes(snapshot.specular, defaultSpecular), 0u);
    if (updateSpecularResult.hasError()) {
      return updateSpecularResult;
    }
    uploadedVersion_ = snapshot.version;
  }

  ctx.shared.materialTableGpuData = MaterialTableGpuData{
      .headerBuffer = headerBuffer_->handle(),
      .clearcoatBuffer = clearcoatBuffer_->handle(),
      .sheenBuffer = sheenBuffer_->handle(),
      .transmissionBuffer = transmissionBuffer_->handle(),
      .specularBuffer = specularBuffer_->handle(),
      .headerBufferAddress =
          gpu_.getBufferDeviceAddress(headerBuffer_->handle()),
      .clearcoatBufferAddress =
          gpu_.getBufferDeviceAddress(clearcoatBuffer_->handle()),
      .sheenBufferAddress = gpu_.getBufferDeviceAddress(sheenBuffer_->handle()),
      .transmissionBufferAddress =
          gpu_.getBufferDeviceAddress(transmissionBuffer_->handle()),
      .specularBufferAddress =
          gpu_.getBufferDeviceAddress(specularBuffer_->handle()),
      .version = snapshot.version,
  };

  if (ctx.shared.materialTableGpuData->headerBufferAddress == 0u ||
      ctx.shared.materialTableGpuData->clearcoatBufferAddress == 0u ||
      ctx.shared.materialTableGpuData->sheenBufferAddress == 0u ||
      ctx.shared.materialTableGpuData->transmissionBufferAddress == 0u ||
      ctx.shared.materialTableGpuData->specularBufferAddress == 0u) {
    return Result<bool, std::string>::makeError(
        "MaterialTableGpuProvider::prepare: invalid buffer address");
  }

  return Result<bool, std::string>::makeResult(true);
}

void MaterialTableGpuProvider::destroyBuffers() {
  const auto destroy = [this](std::unique_ptr<Buffer> &buffer) {
    if (buffer && buffer->valid()) {
      gpu_.destroyBuffer(buffer->handle());
    }
    buffer.reset();
  };
  destroy(headerBuffer_);
  destroy(clearcoatBuffer_);
  destroy(sheenBuffer_);
  destroy(transmissionBuffer_);
  destroy(specularBuffer_);
  headerCapacityBytes_ = 0u;
  clearcoatCapacityBytes_ = 0u;
  sheenCapacityBytes_ = 0u;
  transmissionCapacityBytes_ = 0u;
  specularCapacityBytes_ = 0u;
  uploadedVersion_ = std::numeric_limits<uint64_t>::max();
}

} // namespace nuri
