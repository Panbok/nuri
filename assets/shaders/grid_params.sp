// Extents of grid in world coordinates
const float gridSize = 100.0;

// Size of one cell
const float gridCellSize = 0.025;

// Color of thin lines
const vec4 gridColorThin = vec4(0.5, 0.5, 0.5, 1.0);

// Color of thick lines (every tenth line)
const vec4 gridColorThick = vec4(0.0, 0.0, 0.0, 1.0);

// Minimum number of pixels between cell lines before LOD switch should occur.
const float gridMinPixelsBetweenCells = 2.0;