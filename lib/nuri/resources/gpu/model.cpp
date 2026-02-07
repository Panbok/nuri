#include "nuri/resources/gpu/model.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/resources/mesh_importer.h"

namespace nuri {

Result<std::unique_ptr<Model>, std::string>
Model::create(GPUDevice &gpu, const MeshData &data,
              std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const std::span<const std::byte> vertexBytes{
      reinterpret_cast<const std::byte *>(data.vertices.data()),
      data.vertices.size() * sizeof(Vertex)};
  const std::span<const std::byte> indexBytes{
      reinterpret_cast<const std::byte *>(data.indices.data()),
      data.indices.size() * sizeof(uint32_t)};

  const std::string vertexDebugName =
      debugName.empty() ? std::string() : std::string(debugName) + "_vb";
  const std::string indexDebugName =
      debugName.empty() ? std::string() : std::string(debugName) + "_ib";

  BufferDesc vertexDesc{
      .usage = BufferUsage::Vertex,
      .storage = Storage::Device,
      .size = vertexBytes.size(),
      .data = vertexBytes,
  };

  BufferDesc indexDesc{
      .usage = BufferUsage::Index,
      .storage = Storage::Device,
      .size = indexBytes.size(),
      .data = indexBytes,
  };

  auto vertexBufferResult = Buffer::create(gpu, vertexDesc, vertexDebugName);
  if (vertexBufferResult.hasError()) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        vertexBufferResult.error());
  }

  auto indexBufferResult = Buffer::create(gpu, indexDesc, indexDebugName);
  if (indexBufferResult.hasError()) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        indexBufferResult.error());
  }

  std::vector<Submesh> submeshes(data.submeshes.begin(), data.submeshes.end());
  return Result<std::unique_ptr<Model>, std::string>::makeResult(
      std::unique_ptr<Model>(
          new Model(std::move(vertexBufferResult.value()),
                    std::move(indexBufferResult.value()), std::move(submeshes),
                    static_cast<uint32_t>(data.vertices.size()),
                    static_cast<uint32_t>(data.indices.size()))));
}

Result<std::unique_ptr<Model>, std::string> Model::createFromFile(
    GPUDevice &gpu, std::string_view path, const MeshImportOptions &options,
    std::pmr::memory_resource *mem, std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto meshDataResult = MeshImporter::loadFromFile(path, options, mem);
  if (meshDataResult.hasError()) {
    const std::string pathStr{path};
    NURI_LOG_WARNING("Model::createFromFile: Failed to load mesh '%s': %s",
                     pathStr.c_str(), meshDataResult.error().c_str());
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        meshDataResult.error());
  }

  auto modelResult = create(gpu, meshDataResult.value(), debugName);
  if (modelResult.hasError()) {
    const std::string pathStr{path};
    NURI_LOG_WARNING(
        "Model::createFromFile: Failed to create model from '%s': %s",
        pathStr.c_str(), modelResult.error().c_str());
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        modelResult.error());
  }

  NURI_LOG_DEBUG("Model::createFromFile: Created model from file '%.*s'",
                 static_cast<int>(path.size()), path.data());
  return modelResult;
}

} // namespace nuri
