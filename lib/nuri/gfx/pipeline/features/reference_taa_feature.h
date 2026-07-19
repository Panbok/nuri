#pragma once
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/shader.h"
#include <array>
#include <filesystem>
#include <memory>
namespace nuri {

class RenderPipeline;

class NURI_API ReferenceTAAResolvePass final {
public:
  ReferenceTAAResolvePass(GPUDevice &gpu, RuntimeCompositeConfig config);
  ~ReferenceTAAResolvePass();
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> publishFrameData(FrameBuildContext &ctx);
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);

private:
  GPUDevice &gpu_;
  std::unique_ptr<Shader> shader_{};
  ShaderHandle vertexShader_{};
  ShaderHandle fragmentShader_{};
  SamplerHandle linearClampSampler_{};
  RenderPipelineHandle pipeline_{};
  std::filesystem::path vertexPath_{};
  std::filesystem::path fragmentPath_{};
  bool initialized_ = false;
};

NURI_API void registerReferenceTAAStage(RenderPipeline &pipeline,
                                        GPUDevice &gpu,
                                        RuntimeCompositeConfig config);

} // namespace nuri
