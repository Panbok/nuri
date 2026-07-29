#pragma once
#include "nuri/core/result.h"
#include "nuri/gfx/dynamic_buffer.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/render_graph/render_graph.h"
#include "nuri/text/detail/text_layouter.h"
#include "nuri/text/text_system.h"
#include <filesystem>
#include <limits>
#include <memory_resource>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
namespace nuri {

class GPUDevice;
class RenderPipeline;
class TextSystem;

class TextRenderer {
  friend class TextSystem;
  friend struct std::default_delete<TextRenderer>;
  friend void registerText3DStage(RenderPipeline &, TextSystem &);
  friend void registerText2DStage(RenderPipeline &, TextSystem &);

private:
  struct CreateDesc {
    GPUDevice &gpu;
    FontManager &fonts;
    TextLayouter &layouter;
    std::pmr::memory_resource &memory;
    TextShaderPaths shaderPaths;
  };
  explicit TextRenderer(const CreateDesc &desc);
  ~TextRenderer();
  TextRenderer(const TextRenderer &) = delete;
  TextRenderer &operator=(const TextRenderer &) = delete;
  TextRenderer(TextRenderer &&) = delete;
  TextRenderer &operator=(TextRenderer &&) = delete;
  void beginFrame(uint64_t frameIndex);
  Result<TextBounds, std::string> enqueue2D(const Text2DDesc &desc,
                                            std::pmr::memory_resource &scratch);
  Result<TextBounds, std::string> enqueue3D(const Text3DDesc &desc,
                                            std::pmr::memory_resource &scratch);
  Result<bool, std::string>
  append3DGraphPass(RenderFrameContext &frame, RenderGraphBuilder &graph,
                    RenderGraphTextureId sceneDepthGraphTexture = {},
                    bool hasPriorColorPass = false);
  Result<bool, std::string>
  buildTransparentStageContribution(RenderFrameContext &frame,
                                    TransparentStageContribution &out);
  Result<bool, std::string> append2DGraphPass(RenderFrameContext &frame,
                                              RenderGraphBuilder &graph,
                                              bool hasPriorColorPass = false);
  void onFrameSubmitted(SubmissionHandle submission) noexcept;
  void onFrameAbandoned() noexcept;
  void clear();
  enum class TextDomain : uint8_t { Ui, World };
  struct TextCommonDesc {
    std::string_view utf8{};
    const TextStyle *style = nullptr;
    const TextLayoutParams *layout = nullptr;
    TextColor fillColor{};
  };
  struct Text2DPayload {
    glm::vec2 anchor{0.0f};
  };
  struct Text3DPayload {
    uint32_t transformId = 0u;
  };
  using TextPayload = std::variant<Text2DPayload, Text3DPayload>;
  struct GlyphPacket {
    float minX = 0.0f, minY = 0.0f, maxX = 0.0f, maxY = 0.0f;
    float uvMinX = 0.0f, uvMinY = 0.0f, uvMaxX = 0.0f, uvMaxY = 0.0f;
    float pxRange = 4.0f;
    uint32_t color = 0xffffffffu;
    uint32_t atlas = 0;
    TextDomain domain = TextDomain::Ui;
    uint32_t transformId = 0;
    TextureHandle atlasTexture{};
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
  struct GlyphInstance {
    glm::vec4 rectMinMax{0.0f};
    glm::vec4 uvMinMax{0.0f};
    uint32_t color = 0xffffffffu;
    uint32_t transformIndex = 0;
    uint32_t _pad0 = 0;
    uint32_t _pad1 = 0;
  };
  struct GlyphBatch {
    TextDomain domain = TextDomain::Ui;
    uint32_t atlas = 0;
    float pxRange = 4.0f;
    uint32_t firstInstance = 0;
    uint32_t instanceCount = 0;
    TextureHandle atlasTexture{};
    float sortDepth = 0.0f;
  };
  struct UiPC {
    glm::mat4 proj{1.0f};
    uint64_t glyphBufferAddress = 0;
    uint32_t atlas = 0;
    float pxRange = 4.0f;
  };
  Result<TextBounds, std::string>
  enqueueCommon(const TextCommonDesc &desc, const TextPayload &payload,
                std::pmr::memory_resource &scratch);
  struct WorldPC {
    glm::mat4 viewProj{1.0f};
    uint64_t glyphBufferAddress = 0;
    uint64_t transformBufferAddress = 0;
    uint32_t atlas = 0;
    float pxRange = 4.0f;
    float alphaDiscardThreshold = 1.0e-3f;
    float pad1 = 0.0f;
  };
  struct BillboardFrameBasis {
    glm::vec3 sphericalRight{1.0f, 0.0f, 0.0f};
    glm::vec3 sphericalUp{0.0f, -1.0f, 0.0f};
    glm::vec3 sphericalForward{0.0f, 0.0f, -1.0f};
    glm::vec3 cameraPos{0.0f};
  };
  static void hashWorldTransform(uint64_t &hash,
                                 const WorldTransform &transform);
  static void hashGlyphPacket(uint64_t &hash, const GlyphPacket &packet);
  static uint64_t hashCameraFrameState(const CameraFrameState &camera);
  static bool glyphBatchLess(const GlyphPacket &a, const GlyphPacket &b);
  static BillboardFrameBasis
  buildBillboardFrameBasis(const CameraFrameState &camera);
  static glm::mat4 resolveWorldFromBillboard(const WorldTransform &transform,
                                             const BillboardFrameBasis &basis);
  Result<bool, std::string>
  compileShaders(std::string_view name, const std::filesystem::path &vertexPath,
                 const std::filesystem::path &fragmentPath,
                 ShaderHandle &vertex, ShaderHandle &fragment);
  Result<bool, std::string> ensureUiPipeline(Format colorFormat);
  Result<bool, std::string> ensureWorldPipeline(Format colorFormat,
                                                Format depthFormat);
  Result<bool, std::string> uploadGlyphPackets();
  Result<bool, std::string> prepareGlyphPackets(const CameraFrameState &camera);
  Result<bool, std::string> prepareWorldRenderState(RenderFrameContext &frame,
                                                    TextureHandle &outDepth,
                                                    Format &outDepthFormat);
  void buildGlyphPackets(const CameraFrameState &camera);
  void appendWorldTransparentTextureRead(TextureHandle texture);
  void destroyGpu();
  static constexpr uint64_t kHashSeed = 1469598103934665603ull;
  GPUDevice &gpu_;
  FontManager &fonts_;
  TextLayouter &layouter_;
  std::pmr::memory_resource &memory_;
  TextShaderPaths shaderPaths_{};
  uint64_t frameIndex_ = std::numeric_limits<uint64_t>::max();
  bool uiAppended_ = false;
  bool worldAppended_ = false;
  bool glyphQueueNeedsSort_ = false;
  ShaderHandle uiVs_{};
  ShaderHandle uiFs_{};
  ShaderHandle worldVs_{};
  ShaderHandle worldFs_{};
  RenderPipelineHandle uiPipeline_{};
  RenderPipelineHandle worldPipeline_{};
  Format uiPipelineColor_ = Format::Count;
  Format worldPipelineColor_ = Format::Count;
  Format worldPipelineDepth_ = Format::Count;
  std::pmr::vector<GlyphPacket> glyphQueue_;
  size_t uiGlyphCount_ = 0u;
  size_t worldGlyphCount_ = 0u;
  std::pmr::vector<WorldTransform> worldTransforms_;
  std::pmr::vector<ResolvedWorldTransform> resolvedWorldTransforms_;
  std::pmr::vector<GlyphInstance> glyphInstances_;
  std::pmr::vector<GlyphBatch> glyphBatches_;
  std::pmr::vector<DrawItem> uiDraws_;
  std::pmr::vector<DrawItem> worldDraws_;
  std::pmr::vector<float> worldSortDepths_;
  std::pmr::vector<TransparentStageSortableDraw> worldTransparentDraws_;
  std::pmr::vector<UiPC> uiPcs_;
  std::pmr::vector<WorldPC> worldPcs_;
  DynamicBufferRing glyphBuffers_;
  std::pmr::vector<TextureHandle> worldTransparentTextureReadList_;
  uint64_t glyphQueueHash_ = kHashSeed;
  uint64_t lastBuiltGlyphQueueHash_ = 0;
  uint64_t lastBuiltWorldCameraHash_ = 0;
  bool glyphGeometryValid_ = false;
  bool glyphPrepared_ = false;
  bool worldHasBillboards_ = false;
  uint64_t glyphBufferAddress_ = 0;
  uint64_t worldTransformBufferAddress_ = 0;
  BufferHandle glyphDependencyBuffer_{};
};

} // namespace nuri
