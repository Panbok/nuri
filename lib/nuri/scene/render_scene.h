#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/resources/gpu/texture.h"

#include <cstdint>
#include <memory>
#include <memory_resource>
#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {

struct NURI_API OpaqueRenderable {
  std::shared_ptr<Model> model{};
  std::shared_ptr<Texture> albedoTexture{};
  glm::mat4 modelMatrix{1.0f};
};

class NURI_API RenderScene {
public:
  explicit RenderScene(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~RenderScene() = default;

  RenderScene(const RenderScene &) = delete;
  RenderScene &operator=(const RenderScene &) = delete;
  RenderScene(RenderScene &&) = delete;
  RenderScene &operator=(RenderScene &&) = delete;

  [[nodiscard]] Result<uint32_t, std::string>
  addOpaqueRenderable(std::unique_ptr<Model> model,
                      std::unique_ptr<Texture> albedoTexture,
                      const glm::mat4 &modelMatrix = glm::mat4(1.0f));
  [[nodiscard]] Result<uint32_t, std::string>
  addOpaqueRenderable(std::shared_ptr<Model> model,
                      std::shared_ptr<Texture> albedoTexture,
                      const glm::mat4 &modelMatrix = glm::mat4(1.0f));
  [[nodiscard]] Result<uint32_t, std::string> addOpaqueRenderablesInstanced(
      std::shared_ptr<Model> model, std::shared_ptr<Texture> albedoTexture,
      std::span<const glm::mat4> modelMatrices);
  [[nodiscard]] bool setOpaqueRenderableTransform(uint32_t index,
                                                  const glm::mat4 &modelMatrix);

  [[nodiscard]] const OpaqueRenderable *opaqueRenderable(uint32_t index) const;
  [[nodiscard]] uint64_t topologyVersion() const noexcept {
    return topologyVersion_;
  }
  [[nodiscard]] uint64_t transformVersion() const noexcept {
    return transformVersion_;
  }
  [[nodiscard]] std::span<const OpaqueRenderable> opaqueRenderables() const {
    return opaqueRenderables_;
  }
  void clearOpaqueRenderables();

  void setEnvironmentCubemap(std::unique_ptr<Texture> cubemap);
  [[nodiscard]] Texture *environmentCubemap() {
    return environmentCubemap_.get();
  }
  [[nodiscard]] const Texture *environmentCubemap() const {
    return environmentCubemap_.get();
  }

private:
  std::pmr::vector<OpaqueRenderable> opaqueRenderables_;
  std::unique_ptr<Texture> environmentCubemap_;
  uint64_t topologyVersion_ = 0;
  uint64_t transformVersion_ = 0;
};

} // namespace nuri
