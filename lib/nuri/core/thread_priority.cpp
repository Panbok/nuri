#include "nuri/pch.h"

#include "nuri/core/thread_priority.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace nuri {

void setCurrentThreadBackgroundPriority() noexcept {
#if defined(_WIN32)
  (void)::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#elif defined(__linux__)
  const auto threadId = static_cast<id_t>(::syscall(SYS_gettid));
  (void)::setpriority(PRIO_PROCESS, threadId, 5);
#endif
}

void setCurrentThreadInteractivePriority() noexcept {
#if defined(_WIN32)
  (void)::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif
}

void setCurrentThreadNormalPriority() noexcept {
#if defined(_WIN32)
  (void)::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_NORMAL);
#endif
}

} // namespace nuri
