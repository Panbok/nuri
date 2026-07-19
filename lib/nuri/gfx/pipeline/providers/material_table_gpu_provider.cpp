#include "nuri/gfx/pipeline/providers/material_table_gpu_provider.h"
#include "nuri/core/profiling.h"
#include "nuri/pch.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/resource_manager.h"
namespace nuri {
namespace {
struct TableView {
  std::span<const std::byte> bytes{};
  MaterialTableDirtyRange dirty{};
  size_t elementSize = 0u;
  std::string_view name{};
};
template <typename T>
TableView tableView(std::span<const T> values, MaterialTableDirtyRange dirty,
                    std::string_view name) {
  static const T empty{};
  return {.bytes = std::as_bytes(values.empty() ? std::span<const T>(&empty, 1u)
                                                : values),
          .dirty = dirty,
          .elementSize = sizeof(T),
          .name = name};
}
} // namespace

MaterialTableGpuProvider::MaterialTableGpuProvider(GPUDevice &gpu)
    : gpu_(gpu) {}

MaterialTableGpuProvider::~MaterialTableGpuProvider() { destroyBuffers(); }

Result<bool, std::string>
MaterialTableGpuProvider::prepare(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  const MaterialTableSnapshot snapshot = ctx.resources.materialSnapshot();
  const std::array tables{
      tableView(snapshot.headers, snapshot.dirtyHeaders,
                "material_header_table"),
      tableView(snapshot.clearcoat, snapshot.dirtyClearcoat,
                "material_clearcoat_table"),
      tableView(snapshot.sheen, snapshot.dirtySheen, "material_sheen_table"),
      tableView(snapshot.transmission, snapshot.dirtyTransmission,
                "material_transmission_table"),
      tableView(snapshot.specular, snapshot.dirtySpecular,
                "material_specular_table"),
  };
  for (size_t i = 0; i < tables.size(); ++i) {
    auto ensured =
        ensureDynamicBufferCapacity(gpu_, buffers_[i],
                                    BufferDesc{.usage = BufferUsage::Storage,
                                               .storage = Storage::Device,
                                               .size = tables[i].bytes.size()},
                                    tables[i].name);
    if (ensured.hasError()) {
      return ensured;
    }
    if (ensured.value()) {
      uploadedVersion_ = kNoVersionUploaded;
    }
  }
  if (uploadedVersion_ != snapshot.version) {
    const bool fullUpload = uploadedVersion_ == kNoVersionUploaded ||
                            uploadedVersion_ != snapshot.dirtyBaseVersion;
    std::vector<BufferUpdate> updates;
    updates.reserve(tables.size());
    for (size_t i = 0; i < tables.size(); ++i) {
      const TableView &table = tables[i];
      if (!fullUpload && table.dirty.empty()) {
        continue;
      }
      const size_t offset =
          fullUpload ? 0u : table.dirty.first * table.elementSize;
      const std::span<const std::byte> bytes =
          fullUpload ? table.bytes
                     : table.bytes.subspan(offset, table.dirty.count *
                                                       table.elementSize);
      updates.push_back(BufferUpdate{
          .buffer = buffers_[i].buffer->handle(),
          .data = bytes,
          .offset = offset,
      });
    }
    if (!updates.empty()) {
      auto updateResult = gpu_.updateBuffers(updates);
      if (updateResult.hasError()) {
        return updateResult;
      }
    }
    uploadedVersion_ = snapshot.version;
  }
  std::array<BufferHandle, 5> handles{};
  std::array<uint64_t, 5> addresses{};
  for (size_t i = 0; i < buffers_.size(); ++i) {
    handles[i] = buffers_[i].buffer->handle();
    addresses[i] = gpu_.getBufferDeviceAddress(handles[i]);
  }
  ctx.shared.materialTableGpuData = MaterialTableGpuData{
      .headerBuffer = handles[0],
      .clearcoatBuffer = handles[1],
      .sheenBuffer = handles[2],
      .transmissionBuffer = handles[3],
      .specularBuffer = handles[4],
      .headerBufferAddress = addresses[0],
      .clearcoatBufferAddress = addresses[1],
      .sheenBufferAddress = addresses[2],
      .transmissionBufferAddress = addresses[3],
      .specularBufferAddress = addresses[4],
      .version = snapshot.version,
  };
  return Result<bool, std::string>::makeResult(true);
}

void MaterialTableGpuProvider::destroyBuffers() {
  for (DynamicBufferSlot &buffer : buffers_) {
    retireDynamicBuffer(buffer);
  }
  uploadedVersion_ = kNoVersionUploaded;
}

} // namespace nuri
