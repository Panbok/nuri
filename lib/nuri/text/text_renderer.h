#pragma once

#include "nuri/core/layer.h"
#include "nuri/core/result.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/text/text_layouter.h"

#include <filesystem>
#include <limits>
#include <memory_resource>
#include <string>
#include <string_view>

namespace nuri {

class GPUDevice;

class NURI_API TextRenderer {
public:
  struct ShaderPaths {
    std::filesystem::path uiVertex;
    std::filesystem::path uiFragment;
    std::filesystem::path worldVertex;
    std::filesystem::path worldFragment;
  };

  struct CreateDesc {
    GPUDevice &gpu;
    FontManager &fonts;
    TextLayouter &layouter;
    std::pmr::memory_resource &memory;
    ShaderPaths shaderPaths;
  };

  explicit TextRenderer(const CreateDesc &desc);
  ~TextRenderer();

  TextRenderer(const TextRenderer &) = delete;
  TextRenderer &operator=(const TextRenderer &) = delete;
  TextRenderer(TextRenderer &&) = delete;
  TextRenderer &operator=(TextRenderer &&) = delete;

  Result<bool, std::string> beginFrame(uint64_t frameIndex);
  Result<TextBounds, std::string> enqueue2D(const Text2DDesc &desc,
                                            std::pmr::memory_resource &scratch);
  Result<TextBounds, std::string> enqueue3D(const Text3DDesc &desc,
                                            std::pmr::memory_resource &scratch);
  Result<bool, std::string> append3DPasses(RenderFrameContext &frame,
                                           RenderPassList &out);
  Result<bool, std::string> append2DPasses(RenderFrameContext &frame,
                                           RenderPassList &out);
  void setWorldAlphaDiscardThreshold(float threshold);
  [[nodiscard]] float worldAlphaDiscardThreshold() const;
  void clear();

private:
  struct FrameBuffers {
    BufferHandle vb{};
    BufferHandle ib{};
    size_t vbBytes = 0;
    size_t ibBytes = 0;
    size_t ibQuadCapacity = 0;
  };

  struct UiQuad {
    float minX = 0.0f, minY = 0.0f, maxX = 0.0f, maxY = 0.0f;
    float uvMinX = 0.0f, uvMinY = 0.0f, uvMaxX = 0.0f, uvMaxY = 0.0f;
    float pxRange = 4.0f;
    uint32_t color = 0xffffffffu;
    uint32_t atlas = 0;
  };

  struct WorldQuad {
    float minX = 0.0f, minY = 0.0f, maxX = 0.0f, maxY = 0.0f;
    float uvMinX = 0.0f, uvMinY = 0.0f, uvMaxX = 0.0f, uvMaxY = 0.0f;
    float pxRange = 4.0f;
    uint32_t color = 0xffffffffu;
    uint32_t atlas = 0;
    uint32_t transformId = 0;
  };

  struct WorldTransform {
    glm::mat4 worldFromText{1.0f};
    TextBillboardMode billboard = TextBillboardMode::None;
  };

  struct ResolvedWorldTransform {
    glm::vec4 basisX{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec4 basisY{0.0f, 1.0f, 0.0f, 0.0f};
    glm::vec4 translation{0.0f, 0.0f, 0.0f, 0.0f};
  };

  struct WorldGlyphInstance {
    glm::vec4 rectMinMax{0.0f};
    glm::vec4 uvMinMax{0.0f};
    uint32_t color = 0xffffffffu;
    uint32_t transformIndex = 0;
    uint32_t _pad0 = 0;
    uint32_t _pad1 = 0;
  };

  struct UiVertex {
    glm::vec2 pos{0.0f};
    glm::vec2 uv{0.0f};
    uint32_t color = 0xffffffffu;
  };

  struct UiBatch {
    uint32_t atlas = 0;
    float pxRange = 4.0f;
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
  };

  struct WorldBatch {
    uint32_t atlas = 0;
    float pxRange = 4.0f;
    uint32_t firstInstance = 0;
    uint32_t instanceCount = 0;
  };

  struct UiPC {
    glm::mat4 proj{1.0f};
    uint32_t atlas = 0;
    float pxRange = 4.0f;
    float pad0 = 0.0f;
    float pad1 = 0.0f;
  };

  struct WorldPC {
    glm::mat4 viewProj{1.0f};
    uint64_t glyphBufferAddress = 0;
    uint64_t transformBufferAddress = 0;
    uint32_t atlas = 0;
    float pxRange = 4.0f;
    float alphaDiscardThreshold = 1.0e-3f;
    float pad1 = 0.0f;
  };

  struct PerfCounters {
    uint32_t uiGlyphs = 0;
    uint32_t uiBatches = 0;
    uint32_t worldGlyphs = 0;
    uint32_t worldBatches = 0;
    size_t uiVertexUploadBytes = 0;
    size_t uiIndexUploadBytes = 0;
    size_t worldVertexUploadBytes = 0;
    size_t worldIndexUploadBytes = 0;
  };

  struct BillboardFrameBasis {
    glm::vec3 sphericalRight{1.0f, 0.0f, 0.0f};
    glm::vec3 sphericalUp{0.0f, -1.0f, 0.0f};
    glm::vec3 sphericalForward{0.0f, 0.0f, -1.0f};
    glm::vec3 cameraPos{0.0f};
  };

  static void hashWorldTransform(uint64_t &hash,
                                 const WorldTransform &transform);
  static void hashWorldQuad(uint64_t &hash, const WorldQuad &quad);
  static uint64_t hashCameraFrameState(const CameraFrameState &camera);
  static bool uiBatchLess(const UiQuad &a, const UiQuad &b);
  static bool worldBatchLess(const WorldQuad &a, const WorldQuad &b);
  static BillboardFrameBasis
  buildBillboardFrameBasis(const CameraFrameState &camera);
  static glm::mat4 resolveWorldFromBillboard(const WorldTransform &transform,
                                             const BillboardFrameBasis &basis);

  Result<bool, std::string> compileUiShaders();
  Result<bool, std::string> compileWorldShaders();
  Result<bool, std::string> ensureUiPipeline(Format colorFormat);
  Result<bool, std::string> ensureWorldPipeline(Format colorFormat,
                                                Format depthFormat);
  void syncFrames(std::pmr::vector<FrameBuffers> &frames);
  uint32_t frameSlot(std::pmr::vector<FrameBuffers> &frames);
  Result<bool, std::string> ensureFrameBuffers(FrameBuffers &frame,
                                               size_t vbBytes, size_t ibQuads,
                                               std::string_view debugPrefix);
  Result<bool, std::string>
  ensureWorldInstanceBuffer(FrameBuffers &frame, size_t bytes,
                            std::string_view debugPrefix);
  Result<bool, std::string> uploadUi(uint32_t slot);
  Result<bool, std::string> uploadWorld(uint32_t slot);
  void buildUiGeometry();
  void buildWorldGeometry(const CameraFrameState &camera);
  void resetPerfCounters();
  void emitPerfValidation(uint64_t frameIndex);
  void destroyGpu();

  static constexpr uint64_t kHashSeed = 1469598103934665603ull;

  GPUDevice &gpu_;
  FontManager &fonts_;
  TextLayouter &layouter_;
  std::pmr::memory_resource &memory_;
  ShaderPaths shaderPaths_{};

  uint64_t frameIndex_ = std::numeric_limits<uint64_t>::max();
  bool uiAppended_ = false;
  bool worldAppended_ = false;
  bool uiQueueNeedsSort_ = false;
  bool worldQueueNeedsSort_ = false;
  uint64_t lastPerfValidationFrame_ = 0;
  PerfCounters perf_{};

  ShaderHandle uiVs_{};
  ShaderHandle uiFs_{};
  ShaderHandle worldVs_{};
  ShaderHandle worldFs_{};
  RenderPipelineHandle uiPipeline_{};
  RenderPipelineHandle worldPipeline_{};
  Format uiPipelineColor_ = Format::Count;
  Format worldPipelineColor_ = Format::Count;
  Format worldPipelineDepth_ = Format::Count;

  std::pmr::vector<UiQuad> uiQueue_;
  std::pmr::vector<WorldQuad> worldQueue_;
  std::pmr::vector<WorldTransform> worldTransforms_;
  std::pmr::vector<ResolvedWorldTransform> resolvedWorldTransforms_;
  std::pmr::vector<WorldGlyphInstance> worldInstances_;
  std::pmr::vector<UiVertex> uiVerts_;
  std::pmr::vector<UiBatch> uiBatches_;
  std::pmr::vector<WorldBatch> worldBatches_;
  std::pmr::vector<DrawItem> uiDraws_;
  std::pmr::vector<DrawItem> worldDraws_;
  std::pmr::vector<UiPC> uiPcs_;
  std::pmr::vector<WorldPC> worldPcs_;
  std::pmr::vector<FrameBuffers> uiFrames_;
  std::pmr::vector<FrameBuffers> worldFrames_;

  uint64_t worldQueueHash_ = kHashSeed;
  uint64_t lastBuiltWorldQueueHash_ = 0;
  uint64_t lastBuiltWorldCameraHash_ = 0;
  bool worldGeometryValid_ = false;
  bool worldHasBillboards_ = false;
  uint64_t worldGlyphBufferAddress_ = 0;
  uint64_t worldTransformBufferAddress_ = 0;
  float worldAlphaDiscardThreshold_ = 1.0e-3f;
  BufferHandle worldDependencyBuffer_{};
};

} // namespace nuri
