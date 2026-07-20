#include "nuri/pch.h"

#include "nuri/bakery/smaa_lut_baker.h"
#include "nuri/gfx/smaa_lut_contract.h"

#include <array>
#include <cmath>
#include <fstream>

// The LUT construction below is derived from SMAA's AreaTex.py and
// SearchTex.py reference generators.
//
// Copyright (C) 2013 Jorge Jimenez, Jose I. Echevarria, Belen Masia,
// Fernando Navarro, and Diego Gutierrez.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

namespace nuri::bakery::detail {
namespace {

constexpr uint32_t kOrthoSize = 16u;
constexpr uint32_t kDiagSize = 20u;
constexpr uint32_t kDiagSamples = 30u;
constexpr double kSmoothMaxDistance = 32.0;

struct Vec2 {
  double x = 0.0;
  double y = 0.0;
};

[[nodiscard]] constexpr Vec2 operator+(Vec2 lhs, Vec2 rhs) {
  return {lhs.x + rhs.x, lhs.y + rhs.y};
}

[[nodiscard]] constexpr Vec2 operator-(Vec2 lhs, Vec2 rhs) {
  return {lhs.x - rhs.x, lhs.y - rhs.y};
}

[[nodiscard]] constexpr Vec2 operator*(Vec2 value, double scale) {
  return {value.x * scale, value.y * scale};
}

[[nodiscard]] constexpr Vec2 operator/(Vec2 value, double scale) {
  return {value.x / scale, value.y / scale};
}

struct Area {
  double first = 0.0;
  double second = 0.0;
};

[[nodiscard]] constexpr Area operator+(Area lhs, Area rhs) {
  return {lhs.first + rhs.first, lhs.second + rhs.second};
}

[[nodiscard]] constexpr Area operator*(Area value, double scale) {
  return {value.first * scale, value.second * scale};
}

[[nodiscard]] constexpr Area lerp(Area from, Area to, double amount) {
  return from + (to + from * -1.0) * amount;
}

[[nodiscard]] double rounded(double value) {
  volatile double result = value;
  return result;
}

[[nodiscard]] Area segmentArea(Vec2 p1, Vec2 p2, uint32_t pixelX) {
  const Vec2 delta = p2 - p1;
  const double x1 = static_cast<double>(pixelX);
  const double x2 = x1 + 1.0;
  const double y1 = p1.y + delta.y * (x1 - p1.x) / delta.x;
  const double y2 = p1.y + delta.y * (x2 - p1.x) / delta.x;
  const bool inside = (x1 >= p1.x && x1 < p2.x) || (x2 > p1.x && x2 <= p2.x);
  if (!inside) {
    return {};
  }

  const bool trapezoid = std::copysign(1.0, y1) == std::copysign(1.0, y2) ||
                         std::abs(y1) < 1.0e-4 || std::abs(y2) < 1.0e-4;
  if (trapezoid) {
    const double area = (y1 + y2) * 0.5;
    return area < 0.0 ? Area{std::abs(area), 0.0} : Area{0.0, std::abs(area)};
  }

  const double crossingX = -p1.y * delta.x / delta.y + p1.x;
  double integralPart = 0.0;
  const double fractionalPart = std::modf(crossingX, &integralPart);
  const double a1 = crossingX > p1.x ? y1 * fractionalPart * 0.5 : 0.0;
  const double a2 = crossingX < p2.x ? y2 * (1.0 - fractionalPart) * 0.5 : 0.0;
  const double selected = std::abs(a1) > std::abs(a2) ? a1 : -a2;
  return selected < 0.0 ? Area{std::abs(a1), std::abs(a2)}
                        : Area{std::abs(a2), std::abs(a1)};
}

[[nodiscard]] std::pair<Area, Area> smoothArea(double distance, Area a1,
                                               Area a2) {
  const Area b1{std::sqrt(a1.first * 2.0) * 0.5,
                std::sqrt(a1.second * 2.0) * 0.5};
  const Area b2{std::sqrt(a2.first * 2.0) * 0.5,
                std::sqrt(a2.second * 2.0) * 0.5};
  const double amount = std::clamp(distance / kSmoothMaxDistance, 0.0, 1.0);
  return {lerp(b1, a1, amount), lerp(b2, a2, amount)};
}

[[nodiscard]] Area orthoArea(uint32_t pattern, uint32_t left, uint32_t right,
                             double offset) {
  const double distance = static_cast<double>(left + right + 1u);
  const double upper = 0.5 + offset;
  const double lower = upper - 1.0;
  const Vec2 startLower{0.0, lower};
  const Vec2 startUpper{0.0, upper};
  const Vec2 center{distance * 0.5, 0.0};
  const Vec2 endLower{distance, lower};
  const Vec2 endUpper{distance, upper};

  switch (pattern) {
  case 0u:
  case 5u:
  case 10u:
  case 15u:
    return {};
  case 1u:
    return left <= right ? segmentArea(startLower, center, left) : Area{};
  case 2u:
    return left >= right ? segmentArea(center, endLower, left) : Area{};
  case 3u: {
    auto [a1, a2] = smoothArea(distance, segmentArea(startLower, center, left),
                               segmentArea(center, endLower, left));
    return a1 + a2;
  }
  case 4u:
    return left <= right ? segmentArea(startUpper, center, left) : Area{};
  case 6u: {
    const Area full = segmentArea(startUpper, endLower, left);
    if (std::abs(offset) == 0.0) {
      return full;
    }
    const Area halves = segmentArea(startUpper, center, left) +
                        segmentArea(center, endLower, left);
    return (full + halves) * 0.5;
  }
  case 7u:
    return segmentArea(startUpper, endLower, left);
  case 8u:
    return left >= right ? segmentArea(center, endUpper, left) : Area{};
  case 9u: {
    const Area full = segmentArea(startLower, endUpper, left);
    if (std::abs(offset) == 0.0) {
      return full;
    }
    const Area halves = segmentArea(startLower, center, left) +
                        segmentArea(center, endUpper, left);
    return (full + halves) * 0.5;
  }
  case 11u:
  case 13u:
    return segmentArea(startLower, endUpper, left);
  case 12u: {
    auto [a1, a2] = smoothArea(distance, segmentArea(startUpper, center, left),
                               segmentArea(center, endUpper, left));
    return a1 + a2;
  }
  case 14u:
    return segmentArea(startUpper, endLower, left);
  default:
    return {};
  }
}

constexpr std::array<std::array<uint32_t, 2>, 16> kDiagEdges = {
    std::array<uint32_t, 2>{0u, 0u},
    {1u, 0u},
    {0u, 2u},
    {1u, 2u},
    {2u, 0u},
    {3u, 0u},
    {2u, 2u},
    {3u, 2u},
    {0u, 1u},
    {1u, 1u},
    {0u, 3u},
    {1u, 3u},
    {2u, 1u},
    {3u, 1u},
    {2u, 3u},
    {3u, 3u},
};

[[nodiscard]] double sampledDiagonalArea(Vec2 p1, Vec2 p2, Vec2 pixel) {
  if (p1.x == p2.x && p1.y == p2.y) {
    return 1.0;
  }
  const double midpointX = rounded(rounded(p1.x + p2.x) / 2.0);
  const double midpointY = rounded(rounded(p1.y + p2.y) / 2.0);
  const double a = rounded(p2.y - p1.y);
  const double b = rounded(p1.x - p2.x);
  uint32_t insideCount = 0u;
  for (uint32_t x = 0u; x < kDiagSamples; ++x) {
    for (uint32_t y = 0u; y < kDiagSamples; ++y) {
      const double offsetX = rounded(static_cast<double>(x) /
                                     static_cast<double>(kDiagSamples - 1u));
      const double offsetY = rounded(static_cast<double>(y) /
                                     static_cast<double>(kDiagSamples - 1u));
      const double sampleX = rounded(pixel.x + offsetX);
      const double sampleY = rounded(pixel.y + offsetY);
      const double xTerm = rounded(a * rounded(sampleX - midpointX));
      const double yTerm = rounded(b * rounded(sampleY - midpointY));
      const double side = rounded(xTerm + yTerm);
      insideCount += side > 0.0 ? 1u : 0u;
    }
  }
  return static_cast<double>(insideCount) /
         static_cast<double>(kDiagSamples * kDiagSamples);
}

[[nodiscard]] Area diagonalLineArea(uint32_t pattern, Vec2 p1, Vec2 p2,
                                    uint32_t left, Vec2 offset) {
  if (kDiagEdges[pattern][0] > 0u) {
    p1 = p1 + offset;
  }
  if (kDiagEdges[pattern][1] > 0u) {
    p2 = p2 + offset;
  }
  const double leftValue = static_cast<double>(left);
  const double a1 =
      sampledDiagonalArea(p1, p2, Vec2{1.0 + leftValue, leftValue});
  const double a2 =
      sampledDiagonalArea(p1, p2, Vec2{1.0 + leftValue, 1.0 + leftValue});
  return {1.0 - a1, a2};
}

[[nodiscard]] Area diagonalArea(uint32_t pattern, uint32_t left, uint32_t right,
                                Vec2 offset) {
  const double distance = static_cast<double>(left + right + 1u);
  const Vec2 diagonal{distance, distance};
  const auto area = [&](Vec2 p1, Vec2 p2) {
    return diagonalLineArea(pattern, p1, p2, left, offset);
  };
  const Vec2 p00{0.0, 0.0};
  const Vec2 p10{1.0, 0.0};
  const Vec2 p11{1.0, 1.0};

  switch (pattern) {
  case 0u:
    return (area(p11, p11 + diagonal) + area(p10, p10 + diagonal)) * 0.5;
  case 1u:
    return (area(p10, p00 + diagonal) + area(p10, p10 + diagonal)) * 0.5;
  case 2u:
    return (area(p00, p10 + diagonal) + area(p10, p10 + diagonal)) * 0.5;
  case 3u:
    return area(p10, p10 + diagonal);
  case 4u:
    return (area(p11, p00 + diagonal) + area(p11, p10 + diagonal)) * 0.5;
  case 5u:
    return (area(p11, p00 + diagonal) + area(p10, p10 + diagonal)) * 0.5;
  case 6u:
    return area(p11, p10 + diagonal);
  case 7u:
    return (area(p11, p10 + diagonal) + area(p10, p10 + diagonal)) * 0.5;
  case 8u:
    return (area(p00, p11 + diagonal) + area(p10, p11 + diagonal)) * 0.5;
  case 9u:
    return area(p10, p11 + diagonal);
  case 10u:
    return (area(p00, p11 + diagonal) + area(p10, p10 + diagonal)) * 0.5;
  case 11u:
    return (area(p10, p11 + diagonal) + area(p10, p10 + diagonal)) * 0.5;
  case 12u:
    return area(p11, p11 + diagonal);
  case 13u:
    return (area(p11, p11 + diagonal) + area(p10, p11 + diagonal)) * 0.5;
  case 14u:
    return (area(p11, p11 + diagonal) + area(p11, p10 + diagonal)) * 0.5;
  case 15u:
    return (area(p11, p11 + diagonal) + area(p10, p10 + diagonal)) * 0.5;
  default:
    return {};
  }
}

[[nodiscard]] std::byte quantize(double value) {
  return static_cast<std::byte>(static_cast<uint8_t>(255.0 * value));
}

void setAreaPixel(std::vector<std::byte> &bytes, uint32_t x, uint32_t y,
                  Area area) {
  const size_t offset =
      (static_cast<size_t>(y) * smaa_lut::kAreaWidth + x) * 4u;
  bytes[offset] = quantize(area.first);
  bytes[offset + 1u] = quantize(area.second);
}

[[nodiscard]] std::vector<std::byte> generateAreaLut() {
  std::vector<std::byte> bytes(smaa_lut::kAreaByteCount, std::byte{0});
  for (size_t i = 3u; i < bytes.size(); i += 4u) {
    bytes[i] = std::byte{255};
  }

  constexpr std::array<double, 7> orthoOffsets = {0.0,   -0.25,  0.25, -0.125,
                                                  0.125, -0.375, 0.375};
  constexpr std::array<Vec2, 5> diagOffsets = {
      Vec2{0.0, 0.0}, Vec2{0.25, -0.25}, Vec2{-0.25, 0.25}, Vec2{0.125, -0.125},
      Vec2{-0.125, 0.125}};
  constexpr std::array<std::array<uint32_t, 2>, 16> orthoEdges = {
      std::array<uint32_t, 2>{0u, 0u},
      {3u, 0u},
      {0u, 3u},
      {3u, 3u},
      {1u, 0u},
      {4u, 0u},
      {1u, 3u},
      {4u, 3u},
      {0u, 1u},
      {3u, 1u},
      {0u, 4u},
      {3u, 4u},
      {1u, 1u},
      {4u, 1u},
      {1u, 4u},
      {4u, 4u},
  };

  for (uint32_t subsample = 0u; subsample < orthoOffsets.size(); ++subsample) {
    for (uint32_t pattern = 0u; pattern < orthoEdges.size(); ++pattern) {
      for (uint32_t left = 0u; left < kOrthoSize; ++left) {
        for (uint32_t right = 0u; right < kOrthoSize; ++right) {
          const uint32_t x = left + kOrthoSize * orthoEdges[pattern][0];
          const uint32_t y = right + kOrthoSize * orthoEdges[pattern][1] +
                             5u * kOrthoSize * subsample;
          setAreaPixel(bytes, x, y,
                       orthoArea(pattern, left * left, right * right,
                                 orthoOffsets[subsample]));
        }
      }
    }
  }

  for (uint32_t subsample = 0u; subsample < diagOffsets.size(); ++subsample) {
    for (uint32_t pattern = 0u; pattern < kDiagEdges.size(); ++pattern) {
      for (uint32_t left = 0u; left < kDiagSize; ++left) {
        for (uint32_t right = 0u; right < kDiagSize; ++right) {
          const uint32_t x =
              5u * kOrthoSize + left + kDiagSize * kDiagEdges[pattern][0];
          const uint32_t y = right + kDiagSize * kDiagEdges[pattern][1] +
                             4u * kDiagSize * subsample;
          setAreaPixel(
              bytes, x, y,
              diagonalArea(pattern, left, right, diagOffsets[subsample]));
        }
      }
    }
  }
  return bytes;
}

using EdgeQuad = std::array<uint8_t, 4>;

[[nodiscard]] constexpr uint32_t bilinearIndex(const EdgeQuad &edges) {
  const uint32_t a = 8u * edges[0] + 24u * edges[1];
  const uint32_t b = 8u * edges[2] + 24u * edges[3];
  return (4u * a + 28u * b) / 32u;
}

[[nodiscard]] uint8_t searchDeltaLeft(const EdgeQuad &left,
                                      const EdgeQuad &top) {
  uint8_t distance = top[3] == 1u ? 1u : 0u;
  if (distance == 1u && top[2] == 1u && left[1] != 1u && left[3] != 1u) {
    ++distance;
  }
  return distance;
}

[[nodiscard]] uint8_t searchDeltaRight(const EdgeQuad &left,
                                       const EdgeQuad &top) {
  uint8_t distance = top[3] == 1u && left[1] != 1u && left[3] != 1u ? 1u : 0u;
  if (distance == 1u && top[2] == 1u && left[0] != 1u && left[2] != 1u) {
    ++distance;
  }
  return distance;
}

[[nodiscard]] std::vector<std::byte> generateSearchLut() {
  std::array<EdgeQuad, 33> edgeLookup{};
  std::array<bool, 33> hasEdge{};
  for (uint32_t bits = 0u; bits < 16u; ++bits) {
    const EdgeQuad edges = {
        static_cast<uint8_t>((bits >> 3u) & 1u),
        static_cast<uint8_t>((bits >> 2u) & 1u),
        static_cast<uint8_t>((bits >> 1u) & 1u),
        static_cast<uint8_t>(bits & 1u),
    };
    const uint32_t index = bilinearIndex(edges);
    edgeLookup[index] = edges;
    hasEdge[index] = true;
  }

  constexpr uint32_t sourceWidth = 66u;
  constexpr uint32_t sourceHeight = 33u;
  std::array<uint8_t, sourceWidth * sourceHeight> source{};
  for (uint32_t x = 0u; x < 33u; ++x) {
    for (uint32_t y = 0u; y < 33u; ++y) {
      if (!hasEdge[x] || !hasEdge[y]) {
        continue;
      }
      const EdgeQuad &left = edgeLookup[x];
      const EdgeQuad &top = edgeLookup[y];
      source[y * sourceWidth + x] =
          static_cast<uint8_t>(127u * searchDeltaLeft(left, top));
      source[y * sourceWidth + 33u + x] =
          static_cast<uint8_t>(127u * searchDeltaRight(left, top));
    }
  }

  std::vector<std::byte> bytes(smaa_lut::kSearchByteCount, std::byte{0});
  for (uint32_t y = 0u; y < smaa_lut::kSearchHeight; ++y) {
    const uint32_t sourceY = 32u - y;
    for (uint32_t x = 0u; x < smaa_lut::kSearchWidth; ++x) {
      const size_t offset =
          (static_cast<size_t>(y) * smaa_lut::kSearchWidth + x) * 4u;
      bytes[offset] = static_cast<std::byte>(source[sourceY * sourceWidth + x]);
      bytes[offset + 3u] = std::byte{255};
    }
  }
  return bytes;
}

[[nodiscard]] bool hasExpectedSize(const std::filesystem::path &path,
                                   uintmax_t expectedSize) {
  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec) && !ec &&
         std::filesystem::file_size(path, ec) == expectedSize && !ec;
}

[[nodiscard]] Result<bool, std::string>
writeBytes(std::span<const std::byte> bytes,
           const std::filesystem::path &path) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output || !output.write(reinterpret_cast<const char *>(bytes.data()),
                               static_cast<std::streamsize>(bytes.size()))) {
    return Result<bool, std::string>::makeError(
        "SMAA LUT baker: failed to write '" + path.string() + "'");
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace

SmaaLutBakePlan planSmaaLutBake(const RuntimeConfig &config,
                                bool forceRebuild) {
  SmaaLutBakePlan plan{
      .areaOutputPath = config.roots.shaders / smaa_lut::kAreaFilename,
      .searchOutputPath = config.roots.shaders / smaa_lut::kSearchFilename,
  };
  plan.shouldBake =
      forceRebuild ||
      !hasExpectedSize(plan.areaOutputPath, smaa_lut::kAreaByteCount) ||
      !hasExpectedSize(plan.searchOutputPath, smaa_lut::kSearchByteCount);
  return plan;
}

SmaaLutArtifacts generateSmaaLutArtifacts() {
  return SmaaLutArtifacts{
      .areaRgba8 = generateAreaLut(),
      .searchRgba8 = generateSearchLut(),
  };
}

Result<bool, std::string> bakeSmaaLutsToDisk(const SmaaLutBakePlan &plan) {
  const std::filesystem::path outputDirectory =
      plan.areaOutputPath.parent_path();
  std::error_code ec;
  std::filesystem::create_directories(outputDirectory, ec);
  if (ec) {
    return Result<bool, std::string>::makeError(
        "SMAA LUT baker: failed to create output directory '" +
        outputDirectory.string() + "': " + ec.message());
  }
  if (plan.searchOutputPath.parent_path() != outputDirectory) {
    return Result<bool, std::string>::makeError(
        "SMAA LUT baker: area and search outputs must share a directory");
  }

  SmaaLutArtifacts artifacts = generateSmaaLutArtifacts();
  auto areaResult = writeBytes(artifacts.areaRgba8, plan.areaOutputPath);
  if (areaResult.hasError()) {
    return areaResult;
  }
  return writeBytes(artifacts.searchRgba8, plan.searchOutputPath);
}

} // namespace nuri::bakery::detail
