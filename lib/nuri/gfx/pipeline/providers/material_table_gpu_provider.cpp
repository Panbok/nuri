#include "nuri/gfx/pipeline/providers/material_table_gpu_provider.h"
#include "nuri/core/profiling.h"
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

MaterialTableGpuProvider::MaterialTableGpuProvider(GPUDevice &gpu) : gpu_(gpu) {
  constexpr std::array names{"material_header_table",
                             "material_clearcoat_table", "material_sheen_table",
                             "material_transmission_table",
                             "material_specular_table"};
  const BufferDesc policy{.usage = BufferUsage::Storage,
                          .storage = Storage::Device};
  for (size_t i = 0; i < tables_.size(); ++i) {
    tables_[i].ring =
        std::make_unique<DynamicBufferRing>(gpu_, policy, names[i]);
  }
}

MaterialTableGpuProvider::~MaterialTableGpuProvider() = default;

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
  std::array<DynamicBufferAcquisition, 5> acquisitions{};
  const size_t laneCount = std::max(1u, gpu_.getSwapchainImageCount());
  for (size_t i = 0; i < tables.size(); ++i) {
    auto acquired = tables_[i].ring->acquire(ctx.frame.frameIndex,
                                             tables[i].bytes.size(), laneCount);
    if (acquired.hasError()) {
      abandonPrepared();
      return Result<bool, std::string>::makeError(acquired.error());
    }
    acquisitions[i] = acquired.value();
    tables_[i].laneVersions.resize(acquisitions[i].lane + 1u,
                                   kNoVersionUploaded);
    if (acquisitions[i].replaced) {
      tables_[i].laneVersions[acquisitions[i].lane] = kNoVersionUploaded;
    }
  }
  std::vector<BufferUpdate> updates;
  updates.reserve(tables.size());
  for (size_t i = 0; i < tables.size(); ++i) {
    const TableView &table = tables[i];
    const uint64_t laneVersion = tables_[i].laneVersions[acquisitions[i].lane];
    if (laneVersion == snapshot.version) {
      continue;
    }
    const bool fullUpload = laneVersion == kNoVersionUploaded ||
                            laneVersion != snapshot.dirtyBaseVersion;
    if (!fullUpload && table.dirty.empty()) {
      continue;
    }
    const size_t offset =
        fullUpload ? 0u : table.dirty.first * table.elementSize;
    updates.push_back(BufferUpdate{
        .buffer = acquisitions[i].buffer,
        .data = fullUpload ? table.bytes
                           : table.bytes.subspan(offset, table.dirty.count *
                                                             table.elementSize),
        .offset = offset,
    });
  }
  if (!updates.empty()) {
    auto updateResult = gpu_.updateBuffers(updates);
    if (updateResult.hasError()) {
      abandonPrepared();
      return updateResult;
    }
  }
  MaterialTableGpuData gpuData{};
  for (size_t i = 0; i < tables_.size(); ++i) {
    MaterialTableGpuRegion &region = gpuData.regions[i];
    region.buffer = acquisitions[i].buffer;
    region.address = gpu_.getBufferDeviceAddress(region.buffer);
    tables_[i].laneVersions[acquisitions[i].lane] = snapshot.version;
  }
  gpuData.version = snapshot.version;
  ctx.shared.materialTableGpuData = gpuData;
  return Result<bool, std::string>::makeResult(true);
}

void MaterialTableGpuProvider::onFrameSubmitted(
    const RenderFrameContext &frame) noexcept {
  for (TableState &table : tables_) {
    table.ring->submitPrepared(frame.submission);
  }
}

void MaterialTableGpuProvider::onFrameAbandoned(
    const RenderFrameContext &) noexcept {
  abandonPrepared();
}

void MaterialTableGpuProvider::abandonPrepared() noexcept {
  for (TableState &table : tables_) {
    table.ring->abandonPrepared();
  }
}

} // namespace nuri
