#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

namespace {

void writeArgument(std::ostream &stream, const int index,
                   const std::string &argument) {
  stream << index << ':' << argument.size() << ':' << argument << '\n';
}

int runHelper(const std::vector<std::string> &arguments) {
  const int argc = static_cast<int>(arguments.size());
  if (argc < 2) {
    return 64;
  }
  const std::string &mode = arguments[1];
  if (mode == "echo") {
    for (int i = 2; i < argc; ++i) {
      const std::string &argument = arguments[static_cast<size_t>(i)];
      writeArgument(std::cout, i - 2, argument);
      writeArgument(std::cerr, i - 2, argument);
    }
    return 0;
  }
  if (mode == "exit") {
    if (argc < 3) {
      return 64;
    }
    std::cout << "before-exit\n";
    std::cerr << "exit-diagnostic\n";
    return std::atoi(arguments[2].c_str());
  }
  if (mode == "sleep") {
    if (argc < 3) {
      return 64;
    }
    std::cout << "before-sleep\n" << std::flush;
    std::cerr << "sleep-diagnostic\n" << std::flush;
    std::this_thread::sleep_for(
        std::chrono::milliseconds(std::atoi(arguments[2].c_str())));
    return 0;
  }
  if (mode == "cwd") {
    std::cout << std::filesystem::current_path().string();
    return 0;
  }
  if (mode == "bulk") {
    if (argc < 3) {
      return 64;
    }
    const size_t bytes =
        static_cast<size_t>(std::strtoull(arguments[2].c_str(), nullptr, 10));
    std::cout << std::string(bytes, 'o');
    std::cerr << std::string(bytes, 'e');
    return 0;
  }
  return 64;
}

#if defined(_WIN32)
[[nodiscard]] std::string toUtf8(const std::wstring &wide) {
  if (wide.empty()) {
    return {};
  }
  const int size = WideCharToMultiByte(CP_UTF8, 0u, wide.data(),
                                       static_cast<int>(wide.size()), nullptr,
                                       0, nullptr, nullptr);
  std::string utf8(static_cast<size_t>(size), '\0');
  WideCharToMultiByte(CP_UTF8, 0u, wide.data(), static_cast<int>(wide.size()),
                      utf8.data(), size, nullptr, nullptr);
  return utf8;
}
#endif

} // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t **argv) {
  _setmode(_fileno(stdout), _O_BINARY);
  _setmode(_fileno(stderr), _O_BINARY);
  std::vector<std::string> arguments;
  arguments.reserve(static_cast<size_t>(argc));
  for (int i = 0; i < argc; ++i) {
    arguments.push_back(toUtf8(argv[i] != nullptr ? argv[i] : L""));
  }
  return runHelper(arguments);
}
#else
int main(int argc, char **argv) {
  std::vector<std::string> arguments;
  arguments.reserve(static_cast<size_t>(argc));
  for (int i = 0; i < argc; ++i) {
    arguments.emplace_back(argv[i] != nullptr ? argv[i] : "");
  }
  return runHelper(arguments);
}
#endif
