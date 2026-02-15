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

  OpaqueRenderable renderable{};
  renderable.model = std::move(model);
  renderable.albedoTexture = std::move(albedoTexture);
  renderable.modelMatrix = modelMatrix;

  opaqueRenderables_.push_back(std::move(renderable));
  ++topologyVersion_;
  return Result<uint32_t, std::string>::makeResult(
      static_cast<uint32_t>(opaqueRenderables_.size() - 1));
}

bool RenderScene::setOpaqueRenderableTransform(uint32_t index,
                                               const glm::mat4 &modelMatrix) {
  if (index >= opaqueRenderables_.size()) {
    return false;
  }
  opaqueRenderables_[index].modelMatrix = modelMatrix;
  return true;
}

const OpaqueRenderable *RenderScene::opaqueRenderable(uint32_t index) const {
  if (index >= opaqueRenderables_.size()) {
    return nullptr;
  }
  return &opaqueRenderables_[index];
}

void RenderScene::setEnvironmentCubemap(std::unique_ptr<Texture> cubemap) {
  environmentCubemap_ = std::move(cubemap);
}

} // namespace nuri
