//
#version 460 core

#include "grid_calculation.sp"
#include "grid_params.sp"

layout(location = 0) in vec2 uv;
layout(location = 1) in vec2 camPos;
layout(location = 0) out vec4 out_FragColor;

void main() { out_FragColor = gridColor(uv, camPos); }
