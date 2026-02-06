#pragma once

#if defined(NURI_PROFILING)
#include <cstring>
#include "tracy/Tracy.hpp"
// predefined RGB colors for "heavy" point-of-interest operations
#define NURI_PROFILER_COLOR_WAIT 0xff0000
#define NURI_PROFILER_COLOR_SUBMIT 0x0000ff
#define NURI_PROFILER_COLOR_PRESENT 0x00ff00
#define NURI_PROFILER_COLOR_CREATE 0xff6600
#define NURI_PROFILER_COLOR_DESTROY 0xffa500
#define NURI_PROFILER_COLOR_BARRIER 0xffffff
#define NURI_PROFILER_COLOR_CMD_DRAW 0x8b0000
#define NURI_PROFILER_COLOR_CMD_COPY 0x8b0a50
#define NURI_PROFILER_COLOR_CMD_RTX 0x8b0000
#define NURI_PROFILER_COLOR_CMD_DISPATCH 0x8b0000
//
#define NURI_PROFILER_FUNCTION() ZoneScoped
#define NURI_PROFILER_FUNCTION_COLOR(color) ZoneScopedC(color)
#define NURI_PROFILER_ZONE(name, color)                                        \
  {                                                                            \
    ZoneScopedC(color);                                                        \
    ZoneName(name, strlen(name))
#define NURI_PROFILER_ZONE_END() }
#define NURI_PROFILER_THREAD(name) tracy::SetThreadName(name)
#define NURI_PROFILER_FRAME(name) FrameMarkNamed(name)
#else
#define NURI_PROFILER_FUNCTION()
#define NURI_PROFILER_FUNCTION_COLOR(color)
#define NURI_PROFILER_ZONE(name, color) {
#define NURI_PROFILER_ZONE_END() }
#define NURI_PROFILER_THREAD(name)
#define NURI_PROFILER_FRAME(name)
#endif // NURI_PROFILING