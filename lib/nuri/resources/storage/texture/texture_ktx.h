#pragma once
#include "nuri/core/result.h"
#include "nuri/gfx/gpu_types.h"
#include <array>
#include <cstdio>
#include <filesystem>
#include <ktx.h>
#include <memory>
#include <span>
namespace nuri::detail {

struct KtxTextureDeleter {
  void operator()(ktxTexture *texture) const noexcept {
    if (texture) {
      ktxTexture_Destroy(texture);
    }
  }
  void operator()(ktxTexture2 *texture) const noexcept {
    operator()(ktxTexture(texture));
  }
};

using KtxTexturePtr = std::unique_ptr<ktxTexture, KtxTextureDeleter>;
using KtxTexture2Ptr = std::unique_ptr<ktxTexture2, KtxTextureDeleter>;

struct FileCloser {
  void operator()(std::FILE *file) const noexcept {
    if (file) {
      std::fclose(file);
    }
  }
};

using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

[[nodiscard]] inline FilePtr openFile(const std::filesystem::path &path,
                                      bool write = false) {
  std::FILE *file = nullptr;
#ifdef _WIN32
  _wfopen_s(&file, path.c_str(), write ? L"wb" : L"rb");
#else
  file = std::fopen(path.string().c_str(), write ? "wb" : "rb");
#endif
  return FilePtr(file);
}

[[nodiscard]] inline Result<KtxTexturePtr, std::string> loadKtxTexture(
    const std::filesystem::path &path, std::string_view context,
    ktxTextureCreateFlags flags = KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT) {
  FilePtr file = openFile(path);
  ktxTexture *texture = nullptr;
  const KTX_error_code error =
      file ? ktxTexture_CreateFromStdioStream(file.get(), flags, &texture)
           : KTX_FILE_OPEN_FAILED;
  if (error != KTX_SUCCESS || !texture) {
    return Result<KtxTexturePtr, std::string>::makeError(
        std::string(context) + ": failed to read KTX file '" + path.string() +
        "' (error " + std::to_string(static_cast<int>(error)) + ")");
  }
  return Result<KtxTexturePtr, std::string>::makeResult(KtxTexturePtr(texture));
}

[[nodiscard]] inline Result<std::span<const std::byte>, std::string>
viewKtxImage(ktxTexture &texture, uint32_t level, uint32_t layer, uint32_t face,
             std::string_view context) {
  ktx_size_t offset = 0u;
  if (ktxTexture_GetImageOffset(&texture, level, layer, face, &offset) !=
      KTX_SUCCESS) {
    return Result<std::span<const std::byte>, std::string>::makeError(
        std::string(context) + ": failed to resolve KTX image offset");
  }
  const auto size = ktxTexture_GetImageSize(&texture, level);
  const auto dataSize = ktxTexture_GetDataSize(&texture);
  const auto *data = ktxTexture_GetData(&texture);
  if (data == nullptr || offset > dataSize || size > dataSize - offset) {
    return Result<std::span<const std::byte>, std::string>::makeError(
        std::string(context) + ": KTX image payload is out of bounds");
  }
  return Result<std::span<const std::byte>, std::string>::makeResult(
      {reinterpret_cast<const std::byte *>(data) + offset,
       static_cast<size_t>(size)});
}

inline constexpr ktx_uint32_t kVkRgba8Unorm = 37u;
inline constexpr ktx_uint32_t kVkRgba8Srgb = 43u;
inline constexpr ktx_uint32_t kVkRgba16Float = 97u;
inline constexpr ktx_uint32_t kVkRgba32Float = 109u;
inline constexpr ktx_uint32_t kVkBc7Unorm = 145u;
inline constexpr ktx_uint32_t kVkBc7Srgb = 146u;
inline constexpr ktx_uint32_t kVkEtc2Unorm = 147u;
inline constexpr ktx_uint32_t kVkEtc2Srgb = 148u;

[[nodiscard]] inline Result<Format, std::string>
resolveKtxFormat(const ktxTexture &texture, std::string_view context) {
  using Entry = std::pair<ktx_uint32_t, Format>;
  constexpr std::array kVkFormats{
      Entry{kVkRgba8Unorm, Format::RGBA8_UNORM},
      Entry{kVkRgba8Srgb, Format::RGBA8_SRGB},
      Entry{kVkRgba16Float, Format::RGBA16_FLOAT},
      Entry{kVkRgba32Float, Format::RGBA32_FLOAT},
      Entry{kVkBc7Unorm, Format::BC7_RGBA_UNORM},
      Entry{kVkBc7Srgb, Format::BC7_RGBA_SRGB},
      Entry{kVkEtc2Unorm, Format::ETC2_RGB8_UNORM},
      Entry{kVkEtc2Srgb, Format::ETC2_RGB8_SRGB},
  };
  constexpr std::array kGlFormats{
      Entry{0x8058u, Format::RGBA8_UNORM},
      Entry{0x8C43u, Format::RGBA8_SRGB},
      Entry{0x881Au, Format::RGBA16_FLOAT},
      Entry{0x8814u, Format::RGBA32_FLOAT},
  };
  const auto formats = texture.classId == ktxTexture2_c
                           ? std::span<const Entry>(kVkFormats)
                           : std::span<const Entry>(kGlFormats);
  const ktx_uint32_t encoded =
      texture.classId == ktxTexture2_c
          ? reinterpret_cast<const ktxTexture2 &>(texture).vkFormat
          : reinterpret_cast<const ktxTexture1 &>(texture).glInternalformat;
  for (const auto &[value, format] : formats) {
    if (value == encoded) {
      return Result<Format, std::string>::makeResult(format);
    }
  }
  return Result<Format, std::string>::makeError(std::string(context) +
                                                ": unsupported KTX format " +
                                                std::to_string(encoded));
}

} // namespace nuri::detail
