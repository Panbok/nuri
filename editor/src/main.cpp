#include "nuri/editor_pch.h"

#include "nuri/app/editor_application.h"
#include "nuri/core/profiling.h"
#include "nuri/core/runtime_config.h"

int main() {
  NURI_PROFILER_THREAD("Main");
  auto configResult = nuri::loadRuntimeConfigFromEnvOrDefault();
  NURI_ASSERT(!configResult.hasError(), "Failed to load app config: %s",
              configResult.error().c_str());

  nuri::EditorApplication app{std::move(configResult.value())};
  app.run();
  return 0;
}
