#include "nuri/gfx/shader.h"
#include "nuri/gfx/gpu_device.h"

#include "nuri/pch.h"

namespace nuri {

namespace {
[[nodiscard]] std::string
readFileToString(const std::filesystem::path &filePath, std::string &errorMsg) {
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
} // namespace

Shader::Shader(std::string_view moduleName, GPUDevice &gpu)
    : moduleName_(moduleName), gpu_(gpu), shaderHandles_{} {}

Shader::~Shader() = default;

Result<std::string, std::string> Shader::load(std::string_view path) {
  std::string errorMsg;
  std::string content = readFileToString(std::filesystem::path(path), errorMsg);

  if (!errorMsg.empty()) {
    return Result<std::string, std::string>::makeError(std::move(errorMsg));
  }

  return Result<std::string, std::string>::makeResult(content);
}

Result<ShaderHandle, std::string> Shader::compile(const std::string &code,
                                                  ShaderStage stage) {
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
  auto codeResult = load(path);
  if (codeResult.hasError()) {
    const std::string pathStr{path};
    return Result<ShaderHandle, std::string>::makeError(
        "Failed to load shader file '" + pathStr + "': " + codeResult.error());
  }

  auto compileResult = compile(codeResult.value(), stage);
  if (compileResult.hasError()) {
    const std::string pathStr{path};
    return Result<ShaderHandle, std::string>::makeError(
        "Failed to compile shader file '" + pathStr +
        "': " + compileResult.error());
  }

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
