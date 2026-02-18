#include "nuri/pch.h"

#include "nuri/scene/render_scene.h"

namespace nuri {

RenderScene::RenderScene(std::pmr::memory_resource *memory)
    : opaqueRenderables_(memory ? memory : std::pmr::get_default_resource()) {}

Result<uint32_t, std::string>
RenderScene::addOpaqueRenderable(std::unique_ptr<Model> model,
                                 std::unique_ptr<Texture> albedoTexture,
                                 const glm::mat4 &modelMatrix) {
  if (!model) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addOpaqueRenderable: model is null");
  }
  if (!albedoTexture) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addOpaqueRenderable: albedo texture is null");
  }

  return addOpaqueRenderable(std::shared_ptr<Model>(std::move(model)),
                             std::shared_ptr<Texture>(std::move(albedoTexture)),
                             modelMatrix);
}

Result<uint32_t, std::string>
RenderScene::addOpaqueRenderable(std::shared_ptr<Model> model,
                                 std::shared_ptr<Texture> albedoTexture,
                                 const glm::mat4 &modelMatrix) {
  if (!model) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addOpaqueRenderable: model is null");
  }

  if (!albedoTexture) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addOpaqueRenderable: albedo texture is null");
  }

  OpaqueRenderable renderable{};
  renderable.model = std::move(model);
  renderable.albedoTexture = std::move(albedoTexture);
  renderable.modelMatrix = modelMatrix;

  opaqueRenderables_.push_back(std::move(renderable));
  ++topologyVersion_;
  ++transformVersion_;
  return Result<uint32_t, std::string>::makeResult(
      static_cast<uint32_t>(opaqueRenderables_.size() - 1));
}

Result<uint32_t, std::string> RenderScene::addOpaqueRenderablesInstanced(
    std::shared_ptr<Model> model, std::shared_ptr<Texture> albedoTexture,
    std::span<const glm::mat4> modelMatrices) {
  if (!model) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addOpaqueRenderablesInstanced: model is null");
  }
  if (!albedoTexture) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addOpaqueRenderablesInstanced: albedo texture is null");
  }
  if (modelMatrices.empty()) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addOpaqueRenderablesInstanced: modelMatrices is empty");
  }
  if (modelMatrices.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addOpaqueRenderablesInstanced: instance count exceeds "
        "UINT32_MAX");
  }

  const size_t startIndex = opaqueRenderables_.size();
  const size_t requiredSize = startIndex + modelMatrices.size();
  if (requiredSize >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addOpaqueRenderablesInstanced: total renderable count "
        "exceeds UINT32_MAX");
  }

  opaqueRenderables_.reserve(requiredSize);
  for (const glm::mat4 &modelMatrix : modelMatrices) {
    OpaqueRenderable renderable{};
    renderable.model = model;
    renderable.albedoTexture = albedoTexture;
    renderable.modelMatrix = modelMatrix;
    opaqueRenderables_.push_back(std::move(renderable));
  }
  ++topologyVersion_;
  ++transformVersion_;
  return Result<uint32_t, std::string>::makeResult(
      static_cast<uint32_t>(startIndex));
}

bool RenderScene::setOpaqueRenderableTransform(uint32_t index,
                                               const glm::mat4 &modelMatrix) {
  if (index >= opaqueRenderables_.size()) {
    return false;
  }
  opaqueRenderables_[index].modelMatrix = modelMatrix;
  ++transformVersion_;
  return true;
}

const OpaqueRenderable *RenderScene::opaqueRenderable(uint32_t index) const {
  if (index >= opaqueRenderables_.size()) {
    return nullptr;
  }
  return &opaqueRenderables_[index];
}

void RenderScene::clearOpaqueRenderables() {
  if (opaqueRenderables_.empty()) {
    return;
  }
  opaqueRenderables_.clear();
  ++topologyVersion_;
  ++transformVersion_;
}

void RenderScene::setEnvironmentCubemap(std::unique_ptr<Texture> cubemap) {
  environmentCubemap_ = std::move(cubemap);
}

} // namespace nuri
