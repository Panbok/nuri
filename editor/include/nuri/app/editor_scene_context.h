#pragma once

#include "nuri/core/result.h"

namespace nuri {

class EditorRuntime;

struct EditorScenePrepareContext {
  EditorRuntime &runtime;
};

struct EditorSceneActivateContext {
  EditorRuntime &runtime;
};

struct EditorSceneDeactivateContext {
  EditorRuntime &runtime;
};

struct EditorSceneUpdateContext {
  EditorRuntime &runtime;
  double deltaTime = 0.0;
};

} // namespace nuri
