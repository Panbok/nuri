#include "nuri/pch.h"

#include "nuri/core/log.h"
#include "nuri/ui/file_dialog_widget.h"

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <GLFW/glfw3.h>
#include <windows.h>
// clang-format off
#include <commdlg.h>
#pragma comment(lib, "Comdlg32.lib")
// clang-format on
#ifndef GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#include <GLFW/glfw3native.h>

namespace nuri {

namespace {

std::wstring utf8ToWide(std::string_view text) {
  if (text.empty()) {
    return {};
  }

  const int textSize = static_cast<int>(text.size());
  int requiredSize = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                         text.data(), textSize, nullptr, 0);
  if (requiredSize <= 0) {
    requiredSize =
        MultiByteToWideChar(CP_ACP, 0, text.data(), textSize, nullptr, 0);
    if (requiredSize <= 0) {
      return {};
    }

    std::wstring converted(static_cast<size_t>(requiredSize), L'\0');
    if (MultiByteToWideChar(CP_ACP, 0, text.data(), textSize, converted.data(),
                            requiredSize) <= 0) {
      return {};
    }
    if (!converted.empty() && converted.back() == L'\0') {
      converted.pop_back();
    }
    return converted;
  }

  std::wstring converted(static_cast<size_t>(requiredSize), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), textSize,
                          converted.data(), requiredSize) <= 0) {
    return {};
  }
  if (!converted.empty() && converted.back() == L'\0') {
    converted.pop_back();
  }
  return converted;
}

void appendWindowsFilterEntry(std::wstring &buffer, std::wstring_view text) {
  buffer.append(text);
  buffer.push_back(L'\0');
}

std::wstring
buildWindowsFilterBuffer(std::span<const FileDialogFilter> filters) {
  std::wstring buffer;

  for (const FileDialogFilter &filter : filters) {
    if (filter.displayName.empty() || filter.pattern.empty()) {
      continue;
    }

    appendWindowsFilterEntry(buffer, utf8ToWide(filter.displayName));
    appendWindowsFilterEntry(buffer, utf8ToWide(filter.pattern));
  }

  if (buffer.empty()) {
    appendWindowsFilterEntry(buffer, L"All Files (*.*)");
    appendWindowsFilterEntry(buffer, L"*.*");
  }

  // OPENFILENAMEW expects the filter list to end with an extra null terminator.
  buffer.push_back(L'\0');
  return buffer;
}

HWND resolveDialogOwnerWindow(const OpenFileRequest &request) {
  if (request.ownerWindowHandle != nullptr) {
    auto *const glfwWindow =
        static_cast<GLFWwindow *>(request.ownerWindowHandle);
    if (glfwWindow != nullptr) {
      HWND hwnd = glfwGetWin32Window(glfwWindow);
      if (hwnd != nullptr && IsWindow(hwnd) != FALSE) {
        return hwnd;
      }
    }
  }

  HWND activeWindow = GetActiveWindow();
  if (activeWindow != nullptr && IsWindow(activeWindow) != FALSE) {
    return activeWindow;
  }
  return nullptr;
}

} // namespace

std::optional<std::filesystem::path>
FileDialogWidget::openFile(const OpenFileRequest &request) const {
  std::array<wchar_t, 4096> selectedPath = {};
  std::wstring windowsTitle = utf8ToWide(request.title);
  std::wstring windowsDefaultExtension = utf8ToWide(request.defaultExtension);
  std::wstring windowsFilters = buildWindowsFilterBuffer(request.filters);

  OPENFILENAMEW ofn{};
  ofn.lStructSize = sizeof(ofn);
  ofn.hwndOwner = resolveDialogOwnerWindow(request);
  ofn.lpstrFile = selectedPath.data();
  ofn.nMaxFile = static_cast<DWORD>(selectedPath.size());
  ofn.lpstrFilter = windowsFilters.c_str();
  ofn.nFilterIndex = 1;
  ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
  ofn.lpstrTitle = windowsTitle.empty() ? nullptr : windowsTitle.c_str();
  ofn.lpstrDefExt = windowsDefaultExtension.empty()
                        ? nullptr
                        : windowsDefaultExtension.c_str();

  if (GetOpenFileNameW(&ofn) == FALSE) {
    const DWORD dialogError = CommDlgExtendedError();
    if (dialogError == 0) {
      return std::nullopt;
    }

    NURI_LOG_WARNING("FileDialogWidget::openFile: GetOpenFileNameW failed "
                     "(CommDlgExtendedError=%lu)",
                     static_cast<unsigned long>(dialogError));
    return std::nullopt;
  }
  return std::filesystem::path(selectedPath.data());
}

} // namespace nuri

#endif // defined(_WIN32)
