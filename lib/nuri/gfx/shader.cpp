#include "nuri/gfx/shader.h"
#include "nuri/gfx/gpu_device.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"

#include <algorithm>
#include <sstream>
#include <vector>

namespace nuri {

namespace {
[[nodiscard]] std::filesystem::path
normalizePath(const std::filesystem::path &filePath) {
  std::error_code ec;
  auto normalized = std::filesystem::weakly_canonical(filePath, ec);
  if (ec) {
    normalized = std::filesystem::absolute(filePath, ec);
  }
  if (ec) {
    return filePath.lexically_normal();
  }
  return normalized.lexically_normal();
}

[[nodiscard]] std::string
readFileToString(const std::filesystem::path &filePath, std::string &errorMsg) {
  NURI_PROFILER_FUNCTION();
  errorMsg.clear();

  std::ifstream file(filePath, std::ios::binary);
  if (!file.is_open()) {
    errorMsg = "Failed to open file: " + filePath.string();
    return {};
  }

  std::error_code ec;
  const auto fileSize = std::filesystem::file_size(filePath, ec);

  std::string content;

  if (!ec && fileSize > 0) {
    content.resize(fileSize);
    file.read(content.data(), static_cast<std::streamsize>(fileSize));

    const auto bytesRead = file.gcount();

    if (file.bad()) {
      errorMsg = "Failed to read file: " + filePath.string();
      return {};
    }

    if (bytesRead != static_cast<std::streamsize>(fileSize)) {
      content.resize(static_cast<size_t>(bytesRead));
    }
  } else {
    try {
      content.assign(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());

      if (file.bad()) {
        errorMsg = "Failed to read file: " + filePath.string();
        return {};
      }
    } catch (const std::exception &e) {
      errorMsg =
          "Failed to read file: " + filePath.string() + " (" + e.what() + ")";
      return {};
    }
  }

  return content;
}

[[nodiscard]] std::string_view trimLeft(std::string_view value) {
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
    value.remove_prefix(1);
  }
  return value;
}

struct IncludeDirective {
  enum class Kind : uint8_t { NotInclude, Include, Invalid };
  Kind kind = Kind::NotInclude;
  std::string includePath;
  std::string error;
};

[[nodiscard]] IncludeDirective parseIncludeDirective(std::string_view line) {
  IncludeDirective directive{};
  std::string_view sv = trimLeft(line);
  if (!sv.starts_with("#include")) {
    return directive;
  }

  sv.remove_prefix(std::string_view("#include").size());
  sv = trimLeft(sv);
  if (sv.empty() || sv.front() != '"') {
    directive.kind = IncludeDirective::Kind::Invalid;
    directive.error = "Malformed include directive";
    return directive;
  }

  sv.remove_prefix(1);
  const size_t closingQuote = sv.find('"');
  if (closingQuote == std::string_view::npos) {
    directive.kind = IncludeDirective::Kind::Invalid;
    directive.error = "Include path must be wrapped in double quotes";
    return directive;
  }

  directive.includePath = std::string(sv.substr(0, closingQuote));
  std::string_view trailing = trimLeft(sv.substr(closingQuote + 1));
  if (!trailing.empty() && !trailing.starts_with("//")) {
    directive.kind = IncludeDirective::Kind::Invalid;
    directive.error = "Unexpected trailing tokens after include directive";
    return directive;
  }

  directive.kind = IncludeDirective::Kind::Include;
  return directive;
}

[[nodiscard]] bool expandShaderIncludesRecursive(
    const std::filesystem::path &filePath,
    std::vector<std::filesystem::path> &includeStack, std::string &outCode,
    std::string &errorMsg) {
  const std::filesystem::path normalizedPath = normalizePath(filePath);

  const auto cycleIt = std::find(includeStack.begin(), includeStack.end(),
                                 normalizedPath);
  if (cycleIt != includeStack.end()) {
    std::ostringstream oss;
    oss << "Shader include cycle detected: ";
    for (auto it = cycleIt; it != includeStack.end(); ++it) {
      if (it != cycleIt) {
        oss << " -> ";
      }
      oss << it->string();
    }
    oss << " -> " << normalizedPath.string();
    errorMsg = oss.str();
    return false;
  }

  std::string source = readFileToString(normalizedPath, errorMsg);
  if (!errorMsg.empty()) {
    return false;
  }

  includeStack.push_back(normalizedPath);

  std::istringstream sourceStream(source);
  std::string line;
  size_t lineNumber = 0;

  while (std::getline(sourceStream, line)) {
    ++lineNumber;
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }

    const IncludeDirective directive = parseIncludeDirective(line);
    if (directive.kind == IncludeDirective::Kind::NotInclude) {
      outCode += line;
      outCode.push_back('\n');
      continue;
    }

    if (directive.kind == IncludeDirective::Kind::Invalid) {
      errorMsg = std::string("Invalid include in '") + normalizedPath.string() +
                 "' at line " + std::to_string(lineNumber) + ": " +
                 directive.error;
      includeStack.pop_back();
      return false;
    }

    const std::filesystem::path includePath =
        normalizePath(normalizedPath.parent_path() / directive.includePath);

    std::string includeCode;
    if (!expandShaderIncludesRecursive(includePath, includeStack, includeCode,
                                       errorMsg)) {
      errorMsg = std::string("While expanding include '") +
                 directive.includePath + "' in '" + normalizedPath.string() +
                 "' at line " + std::to_string(lineNumber) + ": " + errorMsg;
      includeStack.pop_back();
      return false;
    }

    outCode += includeCode;
    if (!includeCode.empty() && includeCode.back() != '\n') {
      outCode.push_back('\n');
    }
  }

  includeStack.pop_back();
  return true;
}
} // namespace

Shader::Shader(std::string_view moduleName, GPUDevice &gpu)
    : moduleName_(moduleName), gpu_(gpu), shaderHandles_{} {}

Shader::~Shader() = default;

Result<std::string, std::string> Shader::load(std::string_view path) {
  NURI_PROFILER_FUNCTION();
  std::string errorMsg;
  std::string content = readFileToString(std::filesystem::path(path), errorMsg);

  if (!errorMsg.empty()) {
    return Result<std::string, std::string>::makeError(std::move(errorMsg));
  }

  return Result<std::string, std::string>::makeResult(std::move(content));
}

Result<ShaderHandle, std::string> Shader::compile(const std::string &code,
                                                  ShaderStage stage) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (code.empty()) {
    return Result<ShaderHandle, std::string>::makeError(
        "Shader code is empty for stage " +
        std::to_string(static_cast<int>(stage)));
  }

  const auto stageIndex = static_cast<size_t>(stage);
  if (stageIndex >= static_cast<size_t>(ShaderStage::Count)) {
    return Result<ShaderHandle, std::string>::makeError(
        "Invalid shader stage: " + std::to_string(stageIndex));
  }

  ShaderDesc desc{
      .moduleName = moduleName_,
      .source = code,
      .stage = stage,
  };

  auto result = gpu_.createShaderModule(desc);
  if (result.hasError()) {
    return Result<ShaderHandle, std::string>::makeError(result.error());
  }

  shaderHandles_[stageIndex] = result.value();

  if (nuri::isValid(shaderHandles_[stageIndex])) {
    debug_glsl_source_code_[stage] = code;
  }

  return Result<ShaderHandle, std::string>::makeResult(
      shaderHandles_[stageIndex]);
}

Result<ShaderHandle, std::string> Shader::compileFromFile(std::string_view path,
                                                          ShaderStage stage) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  std::vector<std::filesystem::path> includeStack;
  std::string expandedCode;
  std::string expandError;
  if (!expandShaderIncludesRecursive(std::filesystem::path(path), includeStack,
                                     expandedCode, expandError)) {
    const std::string pathStr{path};
    NURI_LOG_WARNING(
        "Shader::compileFromFile: Failed to load shader file '%s': %s",
        pathStr.c_str(), expandError.c_str());
    return Result<ShaderHandle, std::string>::makeError(
        "Failed to load shader file '" + pathStr + "': " + expandError);
  }

  auto compileResult = compile(expandedCode, stage);
  if (compileResult.hasError()) {
    const std::string pathStr{path};
    NURI_LOG_WARNING(
        "Shader::compileFromFile: Failed to compile shader file '%s': %s",
        pathStr.c_str(), compileResult.error().c_str());
    return Result<ShaderHandle, std::string>::makeError(
        "Failed to compile shader file '" + pathStr +
        "': " + compileResult.error());
  }

  NURI_LOG_DEBUG(
      "Shader::compileFromFile: Compiled shader file '%.*s' for stage %s",
      static_cast<int>(path.size()), path.data(), ShaderStageToString(stage));

  return compileResult;
}

ShaderHandle Shader::getHandle(ShaderStage stage) const {
  const auto stageIndex = static_cast<size_t>(stage);
  if (stageIndex >= static_cast<size_t>(ShaderStage::Count)) {
    return {};
  }

  return shaderHandles_[stageIndex];
}

} // namespace nuri
