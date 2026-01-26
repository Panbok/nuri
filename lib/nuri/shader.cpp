#include "shader.h"

namespace nuri {

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

Shader::Shader(const std::string_view &moduleName, lvk::IContext &ctx)
    : moduleName_(moduleName), ctx_(ctx), shaderHandles_{} {}

Shader::~Shader() = default;

nuri::Result<std::string, std::string_view>
Shader::load(const std::string_view &path) {
  std::string errorMsg;
  std::string content = readFileToString(std::filesystem::path(path), errorMsg);

  if (!errorMsg.empty()) {
    return nuri::Result<std::string, std::string_view>::makeError(errorMsg);
  }

  return nuri::Result<std::string, std::string_view>::makeResult(content);
}

nuri::Result<std::reference_wrapper<lvk::Holder<lvk::ShaderModuleHandle>>,
             std::string_view>
Shader::compile(const std::string &code, const nuri::ShaderStage stage) {
  if (code.empty()) {
    return nuri::Result<
        std::reference_wrapper<lvk::Holder<lvk::ShaderModuleHandle>>,
        std::string_view>::makeError("Shader code is empty for stage " +
                                     std::to_string(static_cast<int>(stage)));
  }

  const auto stageIndex = static_cast<size_t>(stage);
  if (stageIndex >= Stage_Count) {
    return nuri::Result<
        std::reference_wrapper<lvk::Holder<lvk::ShaderModuleHandle>>,
        std::string_view>::makeError("Invalid shader stage: " +
                                     std::to_string(stageIndex));
  }

  lvk::Result res;
  shaderHandles_[stageIndex] = ctx_.createShaderModule(
      {code.c_str(), getLvkShaderStage(stage), moduleName_.data()}, &res);

  if (!res.isOk()) {
    return nuri::Result<
        std::reference_wrapper<lvk::Holder<lvk::ShaderModuleHandle>>,
        std::string_view>::makeError(res.message);
  }

  if (shaderHandles_[stageIndex].valid()) {
    debug_glsl_source_code_[stage] = code;
  }

  return nuri::Result<
      std::reference_wrapper<lvk::Holder<lvk::ShaderModuleHandle>>,
      std::string_view>::makeResult(std::ref(shaderHandles_[stageIndex]));
}

lvk::ShaderModuleHandle Shader::getHandle(const ShaderStage stage) const {
  const auto stageIndex = static_cast<size_t>(stage);
  if (stageIndex >= Stage_Count || !shaderHandles_[stageIndex].valid()) {
    return {};
  }

  return shaderHandles_[stageIndex];
}

} // namespace nuri