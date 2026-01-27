#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"


namespace nuri {
enum ShaderStage : uint8_t {
  Stage_Vert,
  Stage_Tesc,
  Stage_Tese,
  Stage_Geom,
  Stage_Frag,
  Stage_Comp,
  Stage_Task,
  Stage_Mesh,
  Stage_RayGen,
  Stage_AnyHit,
  Stage_ClosestHit,
  Stage_Miss,
  Stage_Intersection,
  Stage_Callable,
  Stage_Count,
};

static_assert(Stage_Count ==
                  static_cast<uint8_t>(lvk::ShaderStage::Stage_Callable) + 1,
              "Stage_Count must be equal to the number of lvk::ShaderStage");

class NURI_API Shader {
public:
  Shader(std::string_view moduleName, lvk::IContext &ctx);
  ~Shader();

  Shader(const Shader &) = delete;
  Shader &operator=(const Shader &) = delete;
  Shader(Shader &&) = delete;
  Shader &operator=(Shader &&) = delete;

  static std::unique_ptr<Shader> create(std::string_view moduleName,
                                        lvk::IContext &ctx) {
    return std::make_unique<Shader>(moduleName, ctx);
  }

  nuri::Result<std::string, std::string> load(std::string_view path);

  nuri::Result<std::reference_wrapper<lvk::Holder<lvk::ShaderModuleHandle>>,
               std::string>
  compile(const std::string &code, const nuri::ShaderStage stage);

  [[nodiscard]] lvk::ShaderModuleHandle getHandle(ShaderStage stage) const;

  [[nodiscard]] inline nuri::ShaderStage
  getNuriShaderStage(const lvk::ShaderStage lvkStage) const {
    switch (lvkStage) {
    case lvk::ShaderStage::Stage_Vert:
      return ShaderStage::Stage_Vert;
    case lvk::ShaderStage::Stage_Frag:
      return ShaderStage::Stage_Frag;
    case lvk::ShaderStage::Stage_Comp:
      return ShaderStage::Stage_Comp;
    case lvk::ShaderStage::Stage_Task:
      return ShaderStage::Stage_Task;
    case lvk::ShaderStage::Stage_Mesh:
      return ShaderStage::Stage_Mesh;
    case lvk::ShaderStage::Stage_RayGen:
      return ShaderStage::Stage_RayGen;
    case lvk::ShaderStage::Stage_AnyHit:
      return ShaderStage::Stage_AnyHit;
    case lvk::ShaderStage::Stage_ClosestHit:
      return ShaderStage::Stage_ClosestHit;
    case lvk::ShaderStage::Stage_Miss:
      return ShaderStage::Stage_Miss;
    case lvk::ShaderStage::Stage_Intersection:
      return ShaderStage::Stage_Intersection;
    case lvk::ShaderStage::Stage_Callable:
      return ShaderStage::Stage_Callable;
    case lvk::ShaderStage::Stage_Tesc:
      return ShaderStage::Stage_Tesc;
    case lvk::ShaderStage::Stage_Tese:
      return ShaderStage::Stage_Tese;
    case lvk::ShaderStage::Stage_Geom:
      return ShaderStage::Stage_Geom;
    default:
      break;
    }

    NURI_ASSERT(false, "Invalid Nuri shader stage");
    return ShaderStage::Stage_Count; // should not happen
  }

  [[nodiscard]] inline lvk::ShaderStage
  getLvkShaderStage(const nuri::ShaderStage nuriStage) const {
    switch (nuriStage) {
    case nuri::ShaderStage::Stage_Vert:
      return lvk::ShaderStage::Stage_Vert;
    case nuri::ShaderStage::Stage_Frag:
      return lvk::ShaderStage::Stage_Frag;
    case Stage_Tesc:
      return lvk::ShaderStage::Stage_Tesc;
    case Stage_Tese:
      return lvk::ShaderStage::Stage_Tese;
    case Stage_Geom:
      return lvk::ShaderStage::Stage_Geom;
    case Stage_Comp:
      return lvk::ShaderStage::Stage_Comp;
    case Stage_Task:
      return lvk::ShaderStage::Stage_Task;
    case Stage_Mesh:
      return lvk::ShaderStage::Stage_Mesh;
    case Stage_RayGen:
      return lvk::ShaderStage::Stage_RayGen;
    case Stage_AnyHit:
      return lvk::ShaderStage::Stage_AnyHit;
    case Stage_ClosestHit:
      return lvk::ShaderStage::Stage_ClosestHit;
    case Stage_Miss:
      return lvk::ShaderStage::Stage_Miss;
    case Stage_Intersection:
      return lvk::ShaderStage::Stage_Intersection;
    case Stage_Callable:
      return lvk::ShaderStage::Stage_Callable;
    case Stage_Count:
      break;
    }

    NURI_ASSERT(false, "Invalid LVK shader stage");
    return lvk::ShaderStage::Stage_Callable; // should not happen
  }

private:
  std::string moduleName_;
  lvk::IContext &ctx_;

  std::array<lvk::Holder<lvk::ShaderModuleHandle>, Stage_Count> shaderHandles_;

  std::unordered_map<ShaderStage, std::string> debug_glsl_source_code_;
};
} // namespace nuri