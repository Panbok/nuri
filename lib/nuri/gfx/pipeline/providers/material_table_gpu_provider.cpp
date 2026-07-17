#include "nuri/pch.h"

#include "nuri/gfx/pipeline/providers/material_table_gpu_provider.h"

#include "nuri/core/profiling.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/resource_manager.h"

namespace nuri {
namespace {

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
  return ensureDynamicBufferCapacity(gpu_, managedBuffer,
                                     BufferDesc{.usage = BufferUsage::Storage,
                                                .storage = Storage::Device,
                                                .size = requiredBytes},
                                     debugName);
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
  for (size_t i = 0; i < snapshot.headers.size(); ++i) {
    const MaterialHeaderGpuData &header = snapshot.headers[i];
    if (header.clearcoatExtensionIndex != kInvalidMaterialExtensionIndex &&
        header.clearcoatExtensionIndex >= snapshot.clearcoat.size()) {
      return Result<bool, std::string>::makeError(
          "MaterialTableGpuProvider::prepare: clearcoat extension index is "
          "out of range for material " +
          std::to_string(i) + " (clearcoatExtensionIndex=" +
          std::to_string(header.clearcoatExtensionIndex) +
          ", snapshot.clearcoat.size=" +
          std::to_string(snapshot.clearcoat.size()) + ")");
    }
    if (header.sheenExtensionIndex != kInvalidMaterialExtensionIndex &&
        header.sheenExtensionIndex >= snapshot.sheen.size()) {
      return Result<bool, std::string>::makeError(
          "MaterialTableGpuProvider::prepare: sheen extension index is out of "
          "range for material " +
          std::to_string(i) + " (sheenExtensionIndex=" +
          std::to_string(header.sheenExtensionIndex) +
          ", snapshot.sheen.size=" + std::to_string(snapshot.sheen.size()) +
          ")");
    }
    if (header.transmissionExtensionIndex != kInvalidMaterialExtensionIndex &&
        header.transmissionExtensionIndex >= snapshot.transmission.size()) {
      return Result<bool, std::string>::makeError(
          "MaterialTableGpuProvider::prepare: transmission extension index is "
          "out of range for material " +
          std::to_string(i) + " (transmissionExtensionIndex=" +
          std::to_string(header.transmissionExtensionIndex) +
          ", snapshot.transmission.size=" +
          std::to_string(snapshot.transmission.size()) + ")");
    }
    if (header.specularExtensionIndex != kInvalidMaterialExtensionIndex &&
        header.specularExtensionIndex >= snapshot.specular.size()) {
      return Result<bool, std::string>::makeError(
          "MaterialTableGpuProvider::prepare: specular extension index is out "
          "of range for material " +
          std::to_string(i) + " (specularExtensionIndex=" +
          std::to_string(header.specularExtensionIndex) +
          ", snapshot.specular.size=" +
          std::to_string(snapshot.specular.size()) + ")");
    }
  }
  auto ensureHeaderResult =
      ensureBufferCapacity(headerBuffer_, requiredTableBytes(snapshot.headers),
                           "material_header_table");
  if (ensureHeaderResult.hasError()) {
    return ensureHeaderResult;
  }
  if (ensureHeaderResult.value()) {
    uploadedVersion_ = kNoVersionUploaded;
  }
  auto ensureClearcoatResult = ensureBufferCapacity(
      clearcoatBuffer_, requiredTableBytes(snapshot.clearcoat),
      "material_clearcoat_table");
  if (ensureClearcoatResult.hasError()) {
    return ensureClearcoatResult;
  }
  if (ensureClearcoatResult.value()) {
    uploadedVersion_ = kNoVersionUploaded;
  }
  auto ensureSheenResult = ensureBufferCapacity(
      sheenBuffer_, requiredTableBytes(snapshot.sheen), "material_sheen_table");
  if (ensureSheenResult.hasError()) {
    return ensureSheenResult;
  }
  if (ensureSheenResult.value()) {
    uploadedVersion_ = kNoVersionUploaded;
  }
  auto ensureTransmissionResult = ensureBufferCapacity(
      transmissionBuffer_, requiredTableBytes(snapshot.transmission),
      "material_transmission_table");
  if (ensureTransmissionResult.hasError()) {
    return ensureTransmissionResult;
  }
  if (ensureTransmissionResult.value()) {
    uploadedVersion_ = kNoVersionUploaded;
  }
  auto ensureSpecularResult = ensureBufferCapacity(
      specularBuffer_, requiredTableBytes(snapshot.specular),
      "material_specular_table");
  if (ensureSpecularResult.hasError()) {
    return ensureSpecularResult;
  }
  if (ensureSpecularResult.value()) {
    uploadedVersion_ = kNoVersionUploaded;
  }

  if (uploadedVersion_ != snapshot.version) {
    const MaterialHeaderGpuData defaultHeader{};
    const MaterialClearcoatGpuData defaultClearcoat{};
    const MaterialSheenGpuData defaultSheen{};
    const MaterialTransmissionGpuData defaultTransmission{};
    const MaterialSpecularGpuData defaultSpecular{};
    const bool fullUpload = uploadedVersion_ == kNoVersionUploaded ||
                            uploadedVersion_ != snapshot.dirtyBaseVersion;
    std::vector<BufferUpdate> updates;
    updates.reserve(5u);
    const auto appendUpdate = [&updates, fullUpload]<typename T>(
                                  BufferHandle buffer,
                                  std::span<const T> values, const T &fallback,
                                  MaterialTableDirtyRange dirty) -> bool {
      if (fullUpload) {
        updates.push_back(BufferUpdate{
            .buffer = buffer,
            .data = tableBytes(values, fallback),
        });
        return true;
      }
      if (dirty.empty()) {
        return true;
      }
      if (dirty.first > values.size() ||
          dirty.count > values.size() - dirty.first) {
        return false;
      }
      const std::span<const T> changed =
          values.subspan(dirty.first, dirty.count);
      updates.push_back(BufferUpdate{
          .buffer = buffer,
          .data = std::as_bytes(changed),
          .offset = static_cast<size_t>(dirty.first) * sizeof(T),
      });
      return true;
    };
    const bool rangesValid =
        appendUpdate(headerBuffer_.buffer->handle(), snapshot.headers,
                     defaultHeader, snapshot.dirtyHeaders) &&
        appendUpdate(clearcoatBuffer_.buffer->handle(), snapshot.clearcoat,
                     defaultClearcoat, snapshot.dirtyClearcoat) &&
        appendUpdate(sheenBuffer_.buffer->handle(), snapshot.sheen,
                     defaultSheen, snapshot.dirtySheen) &&
        appendUpdate(transmissionBuffer_.buffer->handle(),
                     snapshot.transmission, defaultTransmission,
                     snapshot.dirtyTransmission) &&
        appendUpdate(specularBuffer_.buffer->handle(), snapshot.specular,
                     defaultSpecular, snapshot.dirtySpecular);
    if (!rangesValid) {
      return Result<bool, std::string>::makeError(
          "MaterialTableGpuProvider::prepare: material dirty range is out "
          "of bounds");
    }
    if (!updates.empty()) {
      auto updateResult = gpu_.updateBuffers(updates);
      if (updateResult.hasError()) {
        return updateResult;
      }
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
  retireDynamicBuffer(gpu_, headerBuffer_);
  retireDynamicBuffer(gpu_, clearcoatBuffer_);
  retireDynamicBuffer(gpu_, sheenBuffer_);
  retireDynamicBuffer(gpu_, transmissionBuffer_);
  retireDynamicBuffer(gpu_, specularBuffer_);
  uploadedVersion_ = kNoVersionUploaded;
}

} // namespace nuri
