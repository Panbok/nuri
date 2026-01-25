#include "shader.h"

namespace nuri {

[[nodiscard]] std::string
readFileToString(const std::filesystem::path &filePath, std::string &errorMsg) {
  errorMsg.clear();

  if (!std::filesystem::exists(filePath)) {
    errorMsg = "File does not exist: " + filePath.string();
    return {};
  }

  std::ifstream file(filePath, std::ios::binary);
  if (!file.is_open()) {
    errorMsg = "Failed to open file: " + filePath.string();
    return {};
  }

  const auto fileSize = std::filesystem::file_size(filePath);
  std::string content;
  content.resize(fileSize);

  file.read(content.data(), static_cast<std::streamsize>(fileSize));

  if (!file) {
    errorMsg = "Failed to read file: " + filePath.string();
    return {};
  }

  return content;
}

Shader::Shader(const std::string &moduleName,
               const std::unique_ptr<lvk::IContext> &ctx)
    : moduleName_(moduleName), ctx_(ctx), shaderHandles_{} {}

Shader::~Shader() = default;

nuri::Result<std::string, std::string> Shader::load(const std::string &path) {
  std::string errorMsg;
  std::string content = readFileToString(std::filesystem::path(path), errorMsg);

  if (!errorMsg.empty()) {
    return nuri::Result<std::string, std::string>::makeError(errorMsg);
  }

  return nuri::Result<std::string, std::string>::makeResult(content);
}

nuri::Result<lvk::Holder<lvk::ShaderModuleHandle> *, std::string>
Shader::compile(const std::string &code, const nuri::ShaderStage stage) {
  if (code.empty()) {
    return nuri::Result<lvk::Holder<lvk::ShaderModuleHandle> *, std::string>::
        makeError("Shader code is empty for stage " +
                  std::to_string(static_cast<int>(stage)));
  }

  const auto stageIndex = static_cast<size_t>(stage);
  if (stageIndex >= Stage_Count) {
    return nuri::Result<lvk::Holder<lvk::ShaderModuleHandle> *,
                        std::string>::makeError("Invalid shader stage: " +
                                                std::to_string(stageIndex));
  }

  lvk::Result res;
  shaderHandles_[stageIndex] = ctx_->createShaderModule(
      {code.c_str(), getLvkShaderStage(stage), moduleName_.c_str()}, &res);

  if (!res.isOk()) {
    return nuri::Result<lvk::Holder<lvk::ShaderModuleHandle> *,
                        std::string>::makeError(res.message);
  }

  if (shaderHandles_[stageIndex].valid()) {
    debugGLSLSourceCode[shaderHandles_[stageIndex].index()] = code;
  }

  return nuri::Result<lvk::Holder<lvk::ShaderModuleHandle> *,
                      std::string>::makeResult(&shaderHandles_[stageIndex]);
}

lvk::ShaderModuleHandle Shader::getHandle(const ShaderStage stage) const {
  const auto stageIndex = static_cast<size_t>(stage);
  if (stageIndex >= Stage_Count || !shaderHandles_[stageIndex].valid()) {
    return {};
  }

  return shaderHandles_[stageIndex];
}

} // namespace nuri