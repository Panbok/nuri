#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glm/common.hpp>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/gtc/quaternion.hpp>

#include <imgui.h>

#if __has_include(<implot.h>)
#include <implot.h>
#else
#include <implot/implot.h>
#endif

#if __has_include(<meshoptimizer.h>)
#include <meshoptimizer.h>
#elif __has_include(<meshoptimizer/meshoptimizer.h>)
#include <meshoptimizer/meshoptimizer.h>
#endif
