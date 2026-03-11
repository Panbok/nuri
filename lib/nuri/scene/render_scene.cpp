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
    : renderables_(memory ? memory : std::pmr::get_default_resource()) {}

RenderScene::~RenderScene() {
  clearOpaqueRenderables();
  setEnvironment(EnvironmentHandles{});
}

Result<uint32_t, std::string>
RenderScene::addRenderable(ModelRef model, MaterialRef material,
                           const glm::mat4 &modelMatrix) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!isValid(model)) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addRenderable: model handle is invalid");
  }
  if (!isValid(material)) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addRenderable: material handle is invalid");
  }
  if (resources_ != nullptr) {
    if (resources_->tryGet(model) == nullptr) {
      return Result<uint32_t, std::string>::makeError(
          "RenderScene::addRenderable: model handle is stale");
    }
    if (resources_->tryGet(material) == nullptr) {
      return Result<uint32_t, std::string>::makeError(
          "RenderScene::addRenderable: material handle is stale");
    }
  }
  if (renderables_.size() >=
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addRenderable: renderable count exceeds UINT32_MAX");
  }

  Renderable renderable{};
  renderable.model = model;
  renderable.material = material;
  renderable.modelMatrix = modelMatrix;

  renderables_.emplace_back(renderable);
  retainRenderable(renderable);
  ++topologyVersion_;
  ++transformVersion_;
  return Result<uint32_t, std::string>::makeResult(
      static_cast<uint32_t>(renderables_.size() - 1));
}

Result<uint32_t, std::string>
RenderScene::addRenderablesInstanced(ModelRef model, MaterialRef material,
                                     std::span<const glm::mat4> modelMatrices) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!isValid(model)) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addRenderablesInstanced: model handle is invalid");
  }
  if (!isValid(material)) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addRenderablesInstanced: material handle is invalid");
  }
  if (resources_ != nullptr) {
    if (resources_->tryGet(model) == nullptr) {
      return Result<uint32_t, std::string>::makeError(
          "RenderScene::addRenderablesInstanced: model handle is stale");
    }
    if (resources_->tryGet(material) == nullptr) {
      return Result<uint32_t, std::string>::makeError(
          "RenderScene::addRenderablesInstanced: material handle is stale");
    }
  }
  if (modelMatrices.empty()) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addRenderablesInstanced: modelMatrices is empty");
  }
  if (modelMatrices.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addRenderablesInstanced: instance count exceeds "
        "UINT32_MAX");
  }

  const size_t startIndex = renderables_.size();
  const size_t requiredSize = startIndex + modelMatrices.size();
  if (requiredSize >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addRenderablesInstanced: total renderable count exceeds "
        "UINT32_MAX");
  }

  renderables_.reserve(requiredSize);
  for (const glm::mat4 &modelMatrix : modelMatrices) {
    Renderable renderable{};
    renderable.model = model;
    renderable.material = material;
    renderable.modelMatrix = modelMatrix;
    retainRenderable(renderable);
    renderables_.push_back(renderable);
  }
  ++topologyVersion_;
  ++transformVersion_;
  return Result<uint32_t, std::string>::makeResult(
      static_cast<uint32_t>(startIndex));
}

bool RenderScene::setRenderableTransform(uint32_t index,
                                         const glm::mat4 &modelMatrix) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (index >= renderables_.size()) {
    return false;
  }
  renderables_[index].modelMatrix = modelMatrix;
  ++transformVersion_;
  return true;
}

const Renderable *RenderScene::renderable(uint32_t index) const {
  if (index >= renderables_.size()) {
    return nullptr;
  }
  return &renderables_[index];
}

void RenderScene::clearRenderables() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (renderables_.empty()) {
    return;
  }
  for (const Renderable &renderable : renderables_) {
    releaseRenderable(renderable);
  }
  renderables_.clear();
  ++topologyVersion_;
  ++transformVersion_;
}

Result<uint32_t, std::string>
RenderScene::addOpaqueRenderable(ModelRef model, MaterialRef material,
                                 const glm::mat4 &modelMatrix) {
  return addRenderable(model, material, modelMatrix);
}

Result<uint32_t, std::string> RenderScene::addOpaqueRenderablesInstanced(
    ModelRef model, MaterialRef material,
    std::span<const glm::mat4> modelMatrices) {
  return addRenderablesInstanced(model, material, modelMatrices);
}

bool RenderScene::setOpaqueRenderableTransform(uint32_t index,
                                               const glm::mat4 &modelMatrix) {
  return setRenderableTransform(index, modelMatrix);
}

const OpaqueRenderable *RenderScene::opaqueRenderable(uint32_t index) const {
  return renderable(index);
}

void RenderScene::clearOpaqueRenderables() { clearRenderables(); }

void RenderScene::bindResources(ResourceManager *resources) {
  if (resources_ == resources) {
    return;
  }

  if (resources_ != nullptr) {
    for (const Renderable &renderable : renderables_) {
      releaseRenderable(renderable);
    }
    releaseEnvironment(environment_);
  }

  resources_ = resources;

  if (resources_ == nullptr) {
    return;
  }

  size_t writeIndex = 0;
  for (size_t readIndex = 0; readIndex < renderables_.size(); ++readIndex) {
    const Renderable renderable = renderables_[readIndex];
    if (!resources_->owns(renderable.model) ||
        !resources_->owns(renderable.material)) {
      continue;
    }
    renderables_[writeIndex] = renderable;
    retainRenderable(renderables_[writeIndex]);
    ++writeIndex;
  }

  if (writeIndex != renderables_.size()) {
    renderables_.resize(writeIndex);
    ++topologyVersion_;
    ++transformVersion_;
  }

  const auto sanitizeTextureRef = [this](TextureRef &ref) {
    if (isValid(ref) && !resources_->owns(ref)) {
      ref = kInvalidTextureRef;
    }
  };
  sanitizeTextureRef(environment_.cubemap);
  sanitizeTextureRef(environment_.irradiance);
  sanitizeTextureRef(environment_.prefilteredGgx);
  sanitizeTextureRef(environment_.prefilteredCharlie);
  sanitizeTextureRef(environment_.brdfLut);
  retainEnvironment(environment_);
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

void RenderScene::retainRenderable(const Renderable &renderable) {
  if (resources_ == nullptr) {
    return;
  }
  resources_->retain(renderable.model);
  resources_->retain(renderable.material);
}

void RenderScene::releaseRenderable(const Renderable &renderable) {
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
