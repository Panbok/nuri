#include "nuri/pch.h"

#include "nuri/gfx/pipeline/providers/material_table_gpu_provider.h"

#include "nuri/core/profiling.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/resource_manager.h"

namespace nuri {
namespace {

size_t grownBufferCapacity(size_t currentCapacity, size_t requiredBytes) {
  if (currentCapacity == 0u) {
    return requiredBytes;
  }
  const size_t maxCapacity = std::numeric_limits<size_t>::max();
  const size_t grownCapacity =
      currentCapacity > (maxCapacity / 2u) ? maxCapacity : currentCapacity * 2u;
  return std::max(requiredBytes, grownCapacity);
}

// The returned span must be consumed immediately by call sites such as
// gpu_.updateBuffer(). Non-empty values point at the caller-owned table and are
// safe while that span outlives the call; the fallback branch creates a
// single-element span pointing at fallback and dangles once fallback goes away.
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

Result<bool, std::string>
MaterialTableGpuProvider::ensureBufferCapacity(ManagedBuffer &managedBuffer,
                                               size_t requiredBytes,
                                               std::string_view debugName) {
  if (managedBuffer.buffer && managedBuffer.buffer->valid() &&
      managedBuffer.capacityBytes >= requiredBytes) {
    return Result<bool, std::string>::makeResult(true);
  }
  const size_t newCapacity =
      grownBufferCapacity(managedBuffer.capacityBytes, requiredBytes);
  if (managedBuffer.buffer && managedBuffer.buffer->valid()) {
    gpu_.destroyBuffer(managedBuffer.buffer->handle());
  }
  managedBuffer.buffer.reset();
  auto createResult = Buffer::create(gpu_,
                                     BufferDesc{.usage = BufferUsage::Storage,
                                                .storage = Storage::Device,
                                                .size = newCapacity},
                                     debugName);
  if (createResult.hasError()) {
    return Result<bool, std::string>::makeError(createResult.error());
  }
  managedBuffer.buffer = std::move(createResult.value());
  managedBuffer.capacityBytes = newCapacity;
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
  auto ensureHeaderResult =
      ensureBufferCapacity(headerBuffer_, requiredTableBytes(snapshot.headers),
                           "material_header_table");
  if (ensureHeaderResult.hasError()) {
    return ensureHeaderResult;
  }
  auto ensureClearcoatResult = ensureBufferCapacity(
      clearcoatBuffer_, requiredTableBytes(snapshot.clearcoat),
      "material_clearcoat_table");
  if (ensureClearcoatResult.hasError()) {
    return ensureClearcoatResult;
  }
  auto ensureSheenResult = ensureBufferCapacity(
      sheenBuffer_, requiredTableBytes(snapshot.sheen), "material_sheen_table");
  if (ensureSheenResult.hasError()) {
    return ensureSheenResult;
  }
  auto ensureTransmissionResult = ensureBufferCapacity(
      transmissionBuffer_, requiredTableBytes(snapshot.transmission),
      "material_transmission_table");
  if (ensureTransmissionResult.hasError()) {
    return ensureTransmissionResult;
  }
  auto ensureSpecularResult = ensureBufferCapacity(
      specularBuffer_, requiredTableBytes(snapshot.specular),
      "material_specular_table");
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
        gpu_.updateBuffer(headerBuffer_.buffer->handle(),
                          tableBytes(snapshot.headers, defaultHeader), 0u);
    if (updateHeaderResult.hasError()) {
      return updateHeaderResult;
    }
    auto updateClearcoatResult =
        gpu_.updateBuffer(clearcoatBuffer_.buffer->handle(),
                          tableBytes(snapshot.clearcoat, defaultClearcoat), 0u);
    if (updateClearcoatResult.hasError()) {
      return updateClearcoatResult;
    }
    auto updateSheenResult =
        gpu_.updateBuffer(sheenBuffer_.buffer->handle(),
                          tableBytes(snapshot.sheen, defaultSheen), 0u);
    if (updateSheenResult.hasError()) {
      return updateSheenResult;
    }
    auto updateTransmissionResult = gpu_.updateBuffer(
        transmissionBuffer_.buffer->handle(),
        tableBytes(snapshot.transmission, defaultTransmission), 0u);
    if (updateTransmissionResult.hasError()) {
      return updateTransmissionResult;
    }
    auto updateSpecularResult =
        gpu_.updateBuffer(specularBuffer_.buffer->handle(),
                          tableBytes(snapshot.specular, defaultSpecular), 0u);
    if (updateSpecularResult.hasError()) {
      return updateSpecularResult;
    }
    uploadedVersion_ = snapshot.version;
  }

  ctx.shared.materialTableGpuData = MaterialTableGpuData{
      .headerBuffer = headerBuffer_.buffer->handle(),
      .clearcoatBuffer = clearcoatBuffer_.buffer->handle(),
      .sheenBuffer = sheenBuffer_.buffer->handle(),
      .transmissionBuffer = transmissionBuffer_.buffer->handle(),
      .specularBuffer = specularBuffer_.buffer->handle(),
      .headerBufferAddress =
          gpu_.getBufferDeviceAddress(headerBuffer_.buffer->handle()),
      .clearcoatBufferAddress =
          gpu_.getBufferDeviceAddress(clearcoatBuffer_.buffer->handle()),
      .sheenBufferAddress =
          gpu_.getBufferDeviceAddress(sheenBuffer_.buffer->handle()),
      .transmissionBufferAddress =
          gpu_.getBufferDeviceAddress(transmissionBuffer_.buffer->handle()),
      .specularBufferAddress =
          gpu_.getBufferDeviceAddress(specularBuffer_.buffer->handle()),
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
  const auto destroy = [this](ManagedBuffer &managedBuffer) {
    if (managedBuffer.buffer && managedBuffer.buffer->valid()) {
      gpu_.destroyBuffer(managedBuffer.buffer->handle());
    }
    managedBuffer.buffer.reset();
    managedBuffer.capacityBytes = 0u;
  };
  destroy(headerBuffer_);
  destroy(clearcoatBuffer_);
  destroy(sheenBuffer_);
  destroy(transmissionBuffer_);
  destroy(specularBuffer_);
  uploadedVersion_ = kNoVersionUploaded;
}

} // namespace nuri
