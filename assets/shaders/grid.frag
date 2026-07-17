//
#version 460 core

// clang-format off
#include "grid_params.sp"
#include "grid_calculation.sp"
// clang-format on

layout(location = 0) in vec2 uv;
layout(location = 1) in vec2 camPos;
layout(location = 0) out vec4 out_FragColor;

void main() { out_FragColor = gridColor(uv, camPos); }
