#include "nuri/pch.h"
#include "nuri/gfx/renderers/scene_draw_database.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"
namespace nuri {
namespace {
MaterialRef resolveMaterial(const Renderable &renderable,
                            const ModelRecord &model, uint32_t submesh) {
  if (isValid(renderable.materialOverride)) {
    return renderable.materialOverride;
  }
  MaterialRef material = model.materialForSubmesh(submesh);
  if (!isValid(material)) {
    material = model.materialForSource(submesh);
  }
  return isValid(material) ? material : renderable.material;
}
}

SceneDrawDatabase::SceneDrawDatabase(GPUDevice &gpu,
                                     std::pmr::memory_resource *memory)
    : gpu_(gpu), instances_(memory ? memory
                                   : std::pmr::get_default_resource()),
      draws_(memory ? memory : std::pmr::get_default_resource()) {}

Result<bool, std::string> SceneDrawDatabase::prepare(FrameBuildContext &ctx) {
  if (!ctx.frame.scene) {
    ctx.shared.sceneDrawDatabase = nullptr;
    return Result<bool, std::string>::makeResult(true);
  }
  auto result = update(*ctx.frame.scene, ctx.resources);
  if (result.hasError()) {
    return result;
  }
  ctx.shared.sceneDrawDatabase = this;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
SceneDrawDatabase::prepareScene(RenderScenePreparationContext &ctx) {
  auto result = update(ctx.scene, ctx.resources);
  if (result.hasError()) {
    return result;
  }
  ctx.sceneDrawDatabase = this;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
SceneDrawDatabase::update(const RenderScene &scene,
                          const ResourceManager &resources) {
  const MaterialTableSnapshot materials = resources.materialSnapshot();
  const uint64_t geometryVersion = gpu_.geometryMutationVersion();
  if (geometryVersion != 0 && scene_ == &scene &&
      topologyVersion_ == scene.topologyVersion() &&
      materialVersion_ == materials.version &&
      materialBindingVersion_ == resources.modelMaterialBindingVersion() &&
      geometryVersion_ == geometryVersion) {
    return Result<bool, std::string>::makeResult(false);
  }
  instances_.clear();
  draws_.clear();
  const std::span<const Renderable> renderables = scene.renderables();
  for (uint32_t instanceIndex = 0;
       instanceIndex < static_cast<uint32_t>(renderables.size());
       ++instanceIndex) {
    const Renderable &renderable = renderables[instanceIndex];
    const ModelRecord *modelRecord = resources.tryGet(renderable.model);
    if (!modelRecord || !modelRecord->model) {
      return Result<bool, std::string>::makeError(
          "SceneDrawDatabase: failed to resolve model");
    }
    const Model &model = *modelRecord->model;
    instances_.push_back(SceneInstanceRecord{&renderable, &model});
    GeometryAllocationView geometry{};
    if (!gpu_.resolveGeometry(model.geometryHandle(), geometry)) {
      return Result<bool, std::string>::makeError(
          "SceneDrawDatabase: failed to resolve geometry");
    }
    const uint64_t vertexAddress = gpu_.getBufferDeviceAddress(
        geometry.vertexBuffer, geometry.vertexByteOffset);
    if (vertexAddress == 0) {
      return Result<bool, std::string>::makeError(
          "SceneDrawDatabase: geometry has no vertex address");
    }
    const IndexFormat indexFormat =
        geometry.indexCount != 0 &&
                geometry.indexByteSize / geometry.indexCount == sizeof(uint16_t)
            ? IndexFormat::U16
            : IndexFormat::U32;
    const bool dynamicCaster =
        !renderable.morphWeights.empty() || !renderable.skinPalette.empty();
    const std::span<const Submesh> submeshes = model.submeshes();
    for (uint32_t submeshIndex = 0;
         submeshIndex < static_cast<uint32_t>(submeshes.size());
         ++submeshIndex) {
      const MaterialRef material =
          resolveMaterial(renderable, *modelRecord, submeshIndex);
      const MaterialRecord *materialRecord = resources.tryGet(material);
      const bool alphaMasked = materialRecord &&
                               materialRecord->desc.alphaMode ==
                                   MaterialAlphaMode::Mask;
      const bool alphaBlended = materialRecord &&
                                materialRecord->desc.alphaMode ==
                                    MaterialAlphaMode::Blend;
      const bool transmission =
          materialRecord &&
          (materialRecord->desc.featureMask & kMaterialFeatureTransmission) != 0;
      const TextureRecord *baseColor =
          materialRecord
              ? resources.tryGet(materialRecord->textureRefs[
                    kMaterialTextureSlotBaseColor])
              : nullptr;
      uint32_t materialIndex = resources.materialTableIndex(material);
      if (materials.headers.empty() ||
          materialIndex >= materials.headers.size()) {
        materialIndex = 0;
      }
      draws_.push_back(SceneDrawRecord{
          .renderable = &renderable,
          .model = &model,
          .submesh = &submeshes[submeshIndex],
          .meshletView = model.hasMeshlets() ? &model.meshletGpuView() : nullptr,
          .submeshIndex = submeshIndex,
          .instanceIndex = instanceIndex,
          .geometryHandle = model.geometryHandle(),
          .indexBuffer = geometry.indexBuffer,
          .indexBufferOffset = geometry.indexByteOffset,
          .indexFormat = indexFormat,
          .baseVertexBuffer = geometry.vertexBuffer,
          .vertexBuffer = geometry.vertexBuffer,
          .baseVertexDecodeBuffer = model.vertexDecodeBuffer(),
          .vertexDecodeBuffer = model.vertexDecodeBuffer(),
          .vertexBufferByteOffset = geometry.vertexByteOffset,
          .baseVertexBufferAddress = vertexAddress,
          .baseVertexDecodeBufferAddress = model.vertexDecodeBufferAddress(),
          .vertexBufferAddress = vertexAddress,
          .vertexDecodeBufferAddress = model.vertexDecodeBufferAddress(),
          .basePackedVertexFormat = static_cast<uint32_t>(model.drawVertexFormat()),
          .vertexDecodeIndex = submeshIndex,
          .packedVertexFormat = static_cast<uint32_t>(model.drawVertexFormat()),
          .material = material,
          .materialIndex = materialIndex,
          .baseColorTexture = baseColor ? baseColor->texture : TextureHandle{},
          .doubleSided = materialRecord && materialRecord->desc.doubleSided,
          .alphaMasked = alphaMasked,
          .alphaBlended = alphaBlended,
          .transmission = transmission,
          .sortedTransmissionFeedback = transmission && !alphaMasked,
          .materialNormalRequired =
              alphaMasked ||
              (materialRecord &&
               (isValid(materialRecord->textureRefs[
                    kMaterialTextureSlotNormal]) ||
                isValid(materialRecord->desc.textures[
                    kMaterialTextureSlotNormal]))),
          .dynamicCaster = dynamicCaster,
      });
    }
  }
  scene_ = &scene;
  topologyVersion_ = scene.topologyVersion();
  materialVersion_ = materials.version;
  materialBindingVersion_ = resources.modelMaterialBindingVersion();
  geometryVersion_ = geometryVersion;
  return Result<bool, std::string>::makeResult(true);
}

}
