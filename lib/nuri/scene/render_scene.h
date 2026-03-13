#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"

#include <cstdint>
#include <memory_resource>
#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {
class ResourceManager;

struct NURI_API Renderable {
  ModelRef model = kInvalidModelRef;
  MaterialRef material = kInvalidMaterialRef;
  glm::mat4 modelMatrix{1.0f};
};

struct NURI_API EnvironmentHandles {
  TextureRef cubemap = kInvalidTextureRef;
  TextureRef irradiance = kInvalidTextureRef;
  TextureRef prefilteredGgx = kInvalidTextureRef;
  TextureRef prefilteredCharlie = kInvalidTextureRef;
  TextureRef brdfLut = kInvalidTextureRef;
};

class NURI_API RenderScene {
public:
  explicit RenderScene(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~RenderScene();

  RenderScene(const RenderScene &) = delete;
  RenderScene &operator=(const RenderScene &) = delete;
  RenderScene(RenderScene &&) = delete;
  RenderScene &operator=(RenderScene &&) = delete;

  [[nodiscard]] Result<uint32_t, std::string>
  addRenderable(ModelRef model, MaterialRef material,
                const glm::mat4 &modelMatrix = glm::mat4(1.0f));
  [[nodiscard]] Result<uint32_t, std::string>
  addRenderablesInstanced(ModelRef model, MaterialRef material,
                          std::span<const glm::mat4> modelMatrices);
  [[nodiscard]] bool setRenderableTransform(uint32_t index,
                                            const glm::mat4 &modelMatrix);

  [[nodiscard]] const Renderable *renderable(uint32_t index) const;
  [[nodiscard]] std::span<const Renderable> renderables() const {
    return renderables_;
  }
  void clearRenderables();
  [[nodiscard]] uint64_t topologyVersion() const noexcept {
    return topologyVersion_;
  }
  [[nodiscard]] uint64_t transformVersion() const noexcept {
    return transformVersion_;
  }
  void bindResources(ResourceManager *resources);

  void setEnvironment(EnvironmentHandles handles);
  [[nodiscard]] const EnvironmentHandles &environment() const noexcept {
    return environment_;
  }

private:
  void retainRenderable(const Renderable &renderable);
  void releaseRenderable(const Renderable &renderable);
  void retainEnvironment(const EnvironmentHandles &handles);
  void releaseEnvironment(const EnvironmentHandles &handles);

  std::pmr::vector<Renderable> renderables_;
  ResourceManager *resources_ = nullptr;
  EnvironmentHandles environment_{};
  uint64_t topologyVersion_ = 0;
  uint64_t transformVersion_ = 0;
};

} // namespace nuri
