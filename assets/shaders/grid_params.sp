// Extents of grid in world coordinates
float gridSize = 100.0;

// Size of one cell
float gridCellSize = 0.025;

// Color of thin lines
vec4 gridColorThin = vec4(0.5, 0.5, 0.5, 1.0);

// Color of thick lines (every tenth line)
vec4 gridColorThick = vec4(0.0, 0.0, 0.0, 1.0);

// Minimum number of pixels between cell lines before LOD switch should occur. 
const float gridMinPixelsBetweenCells = 2.0;