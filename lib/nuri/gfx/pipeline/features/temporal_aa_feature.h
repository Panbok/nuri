#pragma once
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
namespace nuri {

class GPUDevice;
class RenderPipeline;

NURI_API void registerTemporalInputStages(RenderPipeline &pipeline,
                                          GPUDevice &gpu,
                                          RuntimeCompositeConfig config);
NURI_API void registerTemporalAAStages(RenderPipeline &pipeline, GPUDevice &gpu,
                                       RuntimeCompositeConfig config);

} // namespace nuri
