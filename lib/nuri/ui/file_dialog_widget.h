#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>

#include "nuri/defines.h"

namespace nuri {

struct FileDialogFilter {
  std::string_view displayName;
  std::string_view pattern;
};

struct OpenFileRequest {
  std::string_view title;
  std::span<const FileDialogFilter> filters;
  std::string_view defaultExtension;
};

class NURI_API FileDialogWidget {
public:
#if defined(_WIN32)
  [[nodiscard]] std::optional<std::filesystem::path>
  openFile(const OpenFileRequest &request) const;
#else
  [[nodiscard]] std::optional<std::filesystem::path>
  openFile(const OpenFileRequest &request) const {
    (void)request;
    return std::nullopt;
  }
#endif

  [[nodiscard]] std::optional<std::filesystem::path> openFontFile() const {
    static constexpr std::array<FileDialogFilter, 4> kFontFilters = {
        FileDialogFilter{"Font Files (*.ttf;*.otf)", "*.ttf;*.otf"},
        FileDialogFilter{"TrueType (*.ttf)", "*.ttf"},
        FileDialogFilter{"OpenType (*.otf)", "*.otf"},
        FileDialogFilter{"All Files (*.*)", "*.*"},
    };

    OpenFileRequest request{};
    request.title = "Select Font File";
    request.filters = kFontFilters;
    request.defaultExtension = "ttf";
    return openFile(request);
  }
};

} // namespace nuri
