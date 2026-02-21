#include "nuri/pch.h"

#include "nuri/resources/storage/mesh/mesh_binary_codec.h"

namespace nuri {
namespace {

template <typename T>
[[nodiscard]] Result<std::vector<std::byte>, std::string>
makeCodecError(T &&message) {
  return Result<std::vector<std::byte>, std::string>::makeError(
      std::forward<T>(message));
}

[[nodiscard]] bool checkedMulToSize(size_t a, size_t b, size_t &out) {
  if (a == 0 || b == 0) {
    out = 0;
    return true;
  }
  if (a > (std::numeric_limits<size_t>::max() / b)) {
    return false;
  }
  out = a * b;
  return true;
}

} // namespace

Result<std::vector<std::byte>, std::string>
meshBinaryEncodeVertexBuffer(std::span<const std::byte> vertexBytes,
                             uint32_t vertexStrideBytes) {
  if (vertexStrideBytes == 0) {
    return makeCodecError("meshBinaryEncodeVertexBuffer: vertex stride is 0");
  }
  if ((vertexBytes.size() % vertexStrideBytes) != 0u) {
    return makeCodecError(
        "meshBinaryEncodeVertexBuffer: vertex byte size is not aligned to "
        "vertex stride");
  }
  const size_t vertexCount = vertexBytes.size() / vertexStrideBytes;
  if (vertexCount == 0) {
    return Result<std::vector<std::byte>, std::string>::makeResult({});
  }

  const size_t encodedBound =
      meshopt_encodeVertexBufferBound(vertexCount, vertexStrideBytes);
  std::vector<std::byte> encoded(encodedBound);
  const size_t encodedSize = meshopt_encodeVertexBuffer(
      reinterpret_cast<unsigned char *>(encoded.data()), encoded.size(),
      vertexBytes.data(), vertexCount, vertexStrideBytes);
  if (encodedSize == 0) {
    return makeCodecError("meshBinaryEncodeVertexBuffer: meshopt encode failed");
  }
  encoded.resize(encodedSize);
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(encoded));
}

Result<std::vector<std::byte>, std::string>
meshBinaryDecodeVertexBuffer(std::span<const std::byte> encodedBytes,
                             uint32_t vertexCount,
                             uint32_t vertexStrideBytes) {
  if (vertexStrideBytes == 0) {
    return makeCodecError("meshBinaryDecodeVertexBuffer: vertex stride is 0");
  }
  if (vertexCount == 0) {
    return Result<std::vector<std::byte>, std::string>::makeResult({});
  }
  if (encodedBytes.empty()) {
    return makeCodecError(
        "meshBinaryDecodeVertexBuffer: encoded vertex bytes are empty");
  }

  size_t decodedSize = 0;
  if (!checkedMulToSize(static_cast<size_t>(vertexCount),
                        static_cast<size_t>(vertexStrideBytes), decodedSize)) {
    return makeCodecError(
        "meshBinaryDecodeVertexBuffer: decoded vertex buffer size overflow");
  }

  std::vector<std::byte> decoded(decodedSize);
  const int decodeResult = meshopt_decodeVertexBuffer(
      decoded.data(), vertexCount, vertexStrideBytes,
      reinterpret_cast<const unsigned char *>(encodedBytes.data()),
      encodedBytes.size());
  if (decodeResult != 0) {
    return makeCodecError("meshBinaryDecodeVertexBuffer: meshopt decode failed");
  }

  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(decoded));
}

Result<std::vector<std::byte>, std::string>
meshBinaryEncodeIndexBuffer(std::span<const uint32_t> indices,
                            uint32_t vertexCount) {
  if (indices.empty()) {
    return Result<std::vector<std::byte>, std::string>::makeResult({});
  }
  const size_t encodedBound =
      meshopt_encodeIndexBufferBound(indices.size(), vertexCount);
  std::vector<std::byte> encoded(encodedBound);
  const size_t encodedSize = meshopt_encodeIndexBuffer(
      reinterpret_cast<unsigned char *>(encoded.data()), encoded.size(),
      indices.data(), indices.size());
  if (encodedSize == 0) {
    return makeCodecError("meshBinaryEncodeIndexBuffer: meshopt encode failed");
  }
  encoded.resize(encodedSize);
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(encoded));
}

Result<std::vector<std::byte>, std::string>
meshBinaryDecodeIndexBuffer(std::span<const std::byte> encodedBytes,
                            uint32_t indexCount, uint32_t indexStrideBytes) {
  if (indexStrideBytes != sizeof(uint32_t)) {
    return makeCodecError(
        "meshBinaryDecodeIndexBuffer: unsupported index stride");
  }
  if (indexCount == 0) {
    return Result<std::vector<std::byte>, std::string>::makeResult({});
  }
  if (encodedBytes.empty()) {
    return makeCodecError(
        "meshBinaryDecodeIndexBuffer: encoded index bytes are empty");
  }

  size_t decodedSize = 0;
  if (!checkedMulToSize(static_cast<size_t>(indexCount),
                        static_cast<size_t>(indexStrideBytes), decodedSize)) {
    return makeCodecError(
        "meshBinaryDecodeIndexBuffer: decoded index buffer size overflow");
  }

  std::vector<std::byte> decoded(decodedSize);
  const int decodeResult = meshopt_decodeIndexBuffer(
      decoded.data(), indexCount, indexStrideBytes,
      reinterpret_cast<const unsigned char *>(encodedBytes.data()),
      encodedBytes.size());
  if (decodeResult != 0) {
    return makeCodecError("meshBinaryDecodeIndexBuffer: meshopt decode failed");
  }

  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(decoded));
}

} // namespace nuri
