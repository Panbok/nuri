#include "nuri/pch.h"

#include "nuri/scene/render_scene.h"

#include "nuri/core/profiling.h"
#include "nuri/resources/gpu/resource_manager.h"

namespace nuri {
namespace {

template <typename Fn>
void forEachEnvironmentTextureRef(const EnvironmentHandles &handles, Fn &&fn) {
  fn(handles.cubemap);
  fn(handles.irradiance);
  fn(handles.prefilteredGgx);
  fn(handles.prefilteredCharlie);
  fn(handles.brdfLut);
}

} // namespace

RenderScene::RenderScene(std::pmr::memory_resource *memory)
    : opaqueRenderables_(memory ? memory : std::pmr::get_default_resource()) {}

RenderScene::~RenderScene() {
  clearOpaqueRenderables();
  setEnvironment(EnvironmentHandles{});
}

Result<uint32_t, std::string>
RenderScene::addOpaqueRenderable(ModelRef model, MaterialRef material,
                                 const glm::mat4 &modelMatrix) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!isValid(model)) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addOpaqueRenderable: model handle is invalid");
  }
  if (!isValid(material)) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addOpaqueRenderable: material handle is invalid");
  }
  if (resources_ != nullptr) {
    if (resources_->tryGet(model) == nullptr) {
      return Result<uint32_t, std::string>::makeError(
          "RenderScene::addOpaqueRenderable: model handle is stale");
    }
    if (resources_->tryGet(material) == nullptr) {
      return Result<uint32_t, std::string>::makeError(
          "RenderScene::addOpaqueRenderable: material handle is stale");
    }
  }
  if (opaqueRenderables_.size() >=
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addOpaqueRenderable: renderable count exceeds "
        "UINT32_MAX");
  }

  OpaqueRenderable renderable{};
  renderable.model = model;
  renderable.material = material;
  renderable.modelMatrix = modelMatrix;
  retainRenderable(renderable);

  opaqueRenderables_.push_back(std::move(renderable));
  ++topologyVersion_;
  ++transformVersion_;
  return Result<uint32_t, std::string>::makeResult(
      static_cast<uint32_t>(opaqueRenderables_.size() - 1));
}

Result<uint32_t, std::string> RenderScene::addOpaqueRenderablesInstanced(
    ModelRef model, MaterialRef material,
    std::span<const glm::mat4> modelMatrices) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!isValid(model)) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addOpaqueRenderablesInstanced: model handle is invalid");
  }
  if (!isValid(material)) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addOpaqueRenderablesInstanced: material handle is "
        "invalid");
  }
  if (resources_ != nullptr) {
    if (resources_->tryGet(model) == nullptr) {
      return Result<uint32_t, std::string>::makeError(
          "RenderScene::addOpaqueRenderablesInstanced: model handle is stale");
    }
    if (resources_->tryGet(material) == nullptr) {
      return Result<uint32_t, std::string>::makeError(
          "RenderScene::addOpaqueRenderablesInstanced: material handle is "
          "stale");
    }
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
    renderable.material = material;
    renderable.modelMatrix = modelMatrix;
    retainRenderable(renderable);
    opaqueRenderables_.push_back(std::move(renderable));
  }
  ++topologyVersion_;
  ++transformVersion_;
  return Result<uint32_t, std::string>::makeResult(
      static_cast<uint32_t>(startIndex));
}

bool RenderScene::setOpaqueRenderableTransform(uint32_t index,
                                               const glm::mat4 &modelMatrix) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
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
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (opaqueRenderables_.empty()) {
    return;
  }
  for (const OpaqueRenderable &renderable : opaqueRenderables_) {
    releaseRenderable(renderable);
  }
  opaqueRenderables_.clear();
  ++topologyVersion_;
  ++transformVersion_;
}

void RenderScene::bindResources(ResourceManager *resources) {
  if (resources_ == resources) {
    return;
  }

  if (resources_ != nullptr) {
    for (const OpaqueRenderable &renderable : opaqueRenderables_) {
      releaseRenderable(renderable);
    }
    releaseEnvironment(environment_);
  }

  resources_ = resources;

  if (resources_ != nullptr) {
    for (const OpaqueRenderable &renderable : opaqueRenderables_) {
      retainRenderable(renderable);
    }
    retainEnvironment(environment_);
  }
}

void RenderScene::setEnvironment(EnvironmentHandles handles) {
  if (resources_ == nullptr) {
    environment_ = handles;
    return;
  }

  const auto updateTextureRef = [this](TextureRef &currentRef,
                                       TextureRef nextRef) {
    if (currentRef.value == nextRef.value) {
      return;
    }
    if (isValid(currentRef)) {
      resources_->release(currentRef);
    }
    if (isValid(nextRef)) {
      if (resources_->tryGet(nextRef) == nullptr) {
        NURI_ASSERT(false, "RenderScene::setEnvironment: stale texture handle");
        nextRef = kInvalidTextureRef;
      } else {
        resources_->retain(nextRef);
      }
    }
    currentRef = nextRef;
  };

  updateTextureRef(environment_.cubemap, handles.cubemap);
  updateTextureRef(environment_.irradiance, handles.irradiance);
  updateTextureRef(environment_.prefilteredGgx, handles.prefilteredGgx);
  updateTextureRef(environment_.prefilteredCharlie, handles.prefilteredCharlie);
  updateTextureRef(environment_.brdfLut, handles.brdfLut);
}

void RenderScene::retainRenderable(const OpaqueRenderable &renderable) {
  if (resources_ == nullptr) {
    return;
  }
  resources_->retain(renderable.model);
  resources_->retain(renderable.material);
}

void RenderScene::releaseRenderable(const OpaqueRenderable &renderable) {
  if (resources_ == nullptr) {
    return;
  }
  resources_->release(renderable.model);
  resources_->release(renderable.material);
}

void RenderScene::retainEnvironment(const EnvironmentHandles &handles) {
  if (resources_ == nullptr) {
    return;
  }
  forEachEnvironmentTextureRef(handles, [this](TextureRef textureRef) {
    if (isValid(textureRef)) {
      resources_->retain(textureRef);
    }
  });
}

void RenderScene::releaseEnvironment(const EnvironmentHandles &handles) {
  if (resources_ == nullptr) {
    return;
  }
  forEachEnvironmentTextureRef(handles, [this](TextureRef textureRef) {
    if (isValid(textureRef)) {
      resources_->release(textureRef);
    }
  });
}

} // namespace nuri
