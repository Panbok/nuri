#pragma once

#include "nuri/defines.h"

namespace nuri {

// Marks the calling thread as background work relative to the editor/event
// thread. Unsupported platforms retain their default scheduler priority.
NURI_API void setCurrentThreadBackgroundPriority() noexcept;
NURI_API void setCurrentThreadInteractivePriority() noexcept;
NURI_API void setCurrentThreadNormalPriority() noexcept;

} // namespace nuri
