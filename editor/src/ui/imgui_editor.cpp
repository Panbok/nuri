#include "nuri/editor_pch.h"

#include "nuri/ui/imgui_editor.h"

#include "nuri/bakery/bakery_system.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/core/runtime_config.h"
#include "nuri/core/window.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/imgui_gpu_renderer.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/render_graph/render_graph_telemetry.h"
#include "nuri/platform/imgui_glfw_platform.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/resources/storage/font/nfont_compiler.h"
#include "nuri/scene/render_scene.h"
#include "nuri/text/text_system.h"
#include "nuri/ui/camera_controller_widget.h"
#include "nuri/ui/file_dialog_widget.h"
#include "nuri/ui/linear_graph.h"
#include "nuri/utils/fsp_counter.h"
#include "scene_light_editor.h"

#include <ImGuizmo.h>

namespace nuri {

namespace {

constexpr size_t kMaxLogLines = 2000;
constexpr float kLogFilterWidth = 200.0f;
constexpr float kPassListWidth = 140.0f;
constexpr double kMetricGraphUpdateIntervalSeconds = 0.04;
constexpr double kLogUpdateIntervalSeconds = 0.10;
constexpr float kMetricGraphWindowWidth = 300.0f;
constexpr float kMetricGraphWindowHeight = 280.0f;
constexpr double kMetricSampleMinDeltaSeconds = 1.0e-6;
constexpr std::size_t kMetricGraphSampleCount = 240;
constexpr std::size_t kHierarchyBatchSize = 30;
constexpr uint32_t kUiMaxTessInstances = 65536u;
constexpr const char *kDockspaceWindowName = "NuriDockspace";
constexpr const char *kDockspaceRootId = "NuriDockspace##Root";
constexpr const char *kLogWindowName = "Log";
constexpr const char *kRenderGraphTelemetryWindowName =
    "Render Graph Telemetry";
constexpr const char *kFontCompilerWindowName = "Font Compiler";
constexpr const char *kBakeryWindowName = "Bakery";
constexpr const char *kLightsWindowName = "Lights";
constexpr const char *kRenderPassesWindowName = "Render Passes";
constexpr const char *kHierarchyWindowName = "Hierarchy";
constexpr const char *kInspectorWindowName = "Inspector";
constexpr const char *kTextureFilteringWindowName = "Texture Filtering";
constexpr const char *kCameraControllerWindowName = "Camera Controller";
constexpr const char *kCameraHelpWindowName = "Camera Help";
constexpr const char *kGizmoControlsWindowName = "Gizmo Controls";
constexpr const char *kTelemetryWindowName = "Telemetry";
constexpr std::array<uint8_t, 4> kTextureFilterAnisotropyLevels = {2u, 4u, 8u,
                                                                   16u};
constexpr std::array<const char *, 4> kTextureFilterAnisotropyLabels = {
    "2x", "4x", "8x", "16x"};
constexpr std::array<const char *, 3> kTextureFilterModeLabels = {
    "Bilinear", "Trilinear", "Anisotropic"};

enum class PassInspectorKind : uint8_t {
  Skybox,
  Opaque,
  Transmission,
  Transparent,
  Composite,
  Debug,
  Generic,
};

PassInspectorKind classifyPassInspector(std::string_view featureName,
                                        std::string_view passName) {
  if (featureName == "SkyboxFeature" || passName == "SkyboxPass") {
    return PassInspectorKind::Skybox;
  }
  if (featureName == "OpaqueFeature" || passName == "OpaqueMainPass" ||
      passName == "OpaquePickPass") {
    return PassInspectorKind::Opaque;
  }
  if (featureName == "TransmissionFeature" ||
      passName == "TransmissionDownsamplePass" ||
      passName == "TransmissionCopyPass" ||
      passName == "TransmissionMainPass") {
    return PassInspectorKind::Transmission;
  }
  if (featureName == "TransparentFeature" ||
      passName == "TransparentMainPass" || passName == "TransparentPickPass") {
    return PassInspectorKind::Transparent;
  }
  if (featureName == "CompositeFeature" || passName == "CompositePass") {
    return PassInspectorKind::Composite;
  }
  if (featureName == "DebugFeature" || passName == "DebugGridPass" ||
      passName == "DebugSceneOverlayPass") {
    return PassInspectorKind::Debug;
  }
  return PassInspectorKind::Generic;
}

const char *formatDisplayName(Format format) {
  switch (format) {
  case Format::R32_UINT:
    return "R32_UINT";
  case Format::R32_FLOAT:
    return "R32_FLOAT";
  case Format::RG32_FLOAT:
    return "RG32_FLOAT";
  case Format::RGBA8_UNORM:
    return "RGBA8_UNORM";
  case Format::RGBA8_SRGB:
    return "RGBA8_SRGB";
  case Format::RGBA8_UINT:
    return "RGBA8_UINT";
  case Format::RGBA16_FLOAT:
    return "RGBA16_FLOAT";
  case Format::RGBA32_FLOAT:
    return "RGBA32_FLOAT";
  case Format::BC7_RGBA_UNORM:
    return "BC7_RGBA_UNORM";
  case Format::BC7_RGBA_SRGB:
    return "BC7_RGBA_SRGB";
  case Format::ETC2_RGB8_UNORM:
    return "ETC2_RGB8_UNORM";
  case Format::ETC2_RGB8_SRGB:
    return "ETC2_RGB8_SRGB";
  case Format::D32_FLOAT:
    return "D32_FLOAT";
  case Format::Count:
    return "Invalid";
  }
  return "Unknown";
}

const char *textureFilterModeDisplayName(TextureFilterMode mode) {
  switch (sanitizeTextureFilterMode(mode)) {
  case TextureFilterMode::Bilinear:
    return "Bilinear";
  case TextureFilterMode::Trilinear:
    return "Trilinear";
  case TextureFilterMode::Anisotropic:
    return "Anisotropic";
  }
  return "Unknown";
}

const char *bakeJobKindName(bakery::BakeJobKind kind) {
  switch (kind) {
  case bakery::BakeJobKind::BrdfLut:
    return "BRDF LUT";
  case bakery::BakeJobKind::EnvmapPrefilter:
    return "Envmap Prefilter";
  case bakery::BakeJobKind::ScenePortableAssets:
    return "Scene Portable Assets";
  }
  return "Unknown";
}

const char *bakeJobStateName(bakery::BakeJobState state) {
  switch (state) {
  case bakery::BakeJobState::Queued:
    return "Queued";
  case bakery::BakeJobState::CacheCheck:
    return "CacheCheck";
  case bakery::BakeJobState::GpuSetup:
    return "GpuSetup";
  case bakery::BakeJobState::GpuStep:
    return "GpuStep";
  case bakery::BakeJobState::WriteQueued:
    return "WriteQueued";
  case bakery::BakeJobState::WriteInFlight:
    return "WriteInFlight";
  case bakery::BakeJobState::Succeeded:
    return "Succeeded";
  case bakery::BakeJobState::Skipped:
    return "Skipped";
  case bakery::BakeJobState::Failed:
    return "Failed";
  case bakery::BakeJobState::Canceled:
    return "Canceled";
  }
  return "Unknown";
}

struct LogLevelMeta {
  LogLevel level;
  std::string_view tag;
};

constexpr LogLevelMeta kLogLevels[] = {
    {LogLevel::Trace, "[Trace]"}, {LogLevel::Debug, "[Debug]"},
    {LogLevel::Info, "[Info]"},   {LogLevel::Warning, "[Warn]"},
    {LogLevel::Error, "[Error]"}, {LogLevel::Fatal, "[Fatal]"},
};

float sanitizeSample(float value) {
  return std::isfinite(value) ? value : 0.0f;
}

struct LogLine {
  LogLevel level = LogLevel::Info;
  std::string message;
};

struct LogFilterState {
  bool autoScroll = true;
  bool requestScroll = false;
  ImGuiTextFilter textFilter;
  bool showTrace = true;
  bool showDebug = true;
  bool showInfo = true;
  bool showWarning = true;
  bool showFatal = true;

  bool levelEnabled(LogLevel level) const {
    switch (level) {
    case LogLevel::Trace:
      return showTrace;
    case LogLevel::Debug:
      return showDebug;
    case LogLevel::Info:
      return showInfo;
    case LogLevel::Warning:
      return showWarning;
    case LogLevel::Error:
      return true;
    case LogLevel::Fatal:
      return showFatal;
    }
    return true;
  }
};

struct FontCompilerUiState {
  std::filesystem::path outputDirectory;
  std::vector<std::filesystem::path> availableNfonts;
  std::array<char, 512> sourcePath = {};
  std::array<char, 512> outputPath = {};
  std::array<char, 512> selectedNfontPath = {};
  bool autoOutputName = true;
  int charsetPreset = 0;
  float minimumEmSize = 40.0f;
  float pxRange = 4.0f;
  float outerPixelPadding = 2.0f;
  int atlasSpacing = 1;
  bool useRgba16fAtlas = true;
  int atlasWidthPreset = 1;
  int atlasHeightPreset = 1;
  int maxAtlasWidth = 2048;
  int maxAtlasHeight = 2048;
  int threadCount = 0;
  int selectedNfontIndex = -1;
  float globalFontSizePx = 42.0f;
  std::shared_future<Result<NFontCompileReport, std::string>> compileFuture;
  bool compileInFlight = false;
  bool nfontListInitialized = false;
  std::string status;
  std::string error;
  std::string globalStatus;
  std::string globalError;
  NFontCompileReport lastReport{};
  FileDialogWidget fileDialog{};

  FontCompilerUiState() {
    auto runtimeConfigResult = loadRuntimeConfigFromEnvOrDefault();
    if (runtimeConfigResult.hasError()) {
      outputDirectory = std::filesystem::path("assets") / "fonts";
    } else {
      outputDirectory = runtimeConfigResult.value().roots.fonts;
    }
    outputDirectory = outputDirectory.lexically_normal();

    const std::string defaultOutput = (outputDirectory / "generated_ui.nfont")
                                          .lexically_normal()
                                          .generic_string();
    std::memcpy(outputPath.data(), defaultOutput.c_str(),
                std::min(outputPath.size() - 1, defaultOutput.size()));
  }
};

struct BakeryUiState {
  std::array<char, 512> envHdrPath = {};
  std::array<char, 512> scenePath = {};
  bool forceRebuild = false;
  bool prebuildBc7 = false;
  bool prebuildEtc2 = false;
  bool prebuildRgba8 = false;
  std::string status{};
  std::string error{};
  FileDialogWidget fileDialog{};

  BakeryUiState() {
    constexpr std::string_view kDefaultEnvHdr = "piazza_bologni_1k.hdr";
    const size_t copyCount =
        std::min(envHdrPath.size() - 1u, kDefaultEnvHdr.size());
    if (copyCount > 0) {
      std::memcpy(envHdrPath.data(), kDefaultEnvHdr.data(), copyCount);
    }
    envHdrPath[copyCount] = '\0';
  }
};

struct RenderGraphTelemetryUiState {
  std::array<char, 512> outputPath = {};
  std::string status{};
  std::string error{};
  std::string lastSuggestedPath{};
  FileDialogWidget fileDialog{};
  bool initializedOutputPath = false;
};

struct TelemetryOverlayUiState {
  bool overlayEnabled = true;
  bool showFpsMs = true;
  bool showInstanceStats = true;
  bool showDrawTessStats = true;
  bool showIndirectStats = true;
  bool showDebugDrawStats = true;
  bool showPatchHeatmap = true;
  bool showDispatchStats = true;
  bool showGraphs = true;
  bool showImGuiMetricsWindow = false;
};

struct SceneSelectionUiState {
  std::vector<std::string> ids{};
  std::vector<std::string> names{};
  std::vector<const char *> nameViews{};
  std::string hotkeyHint = "Toggle Editor: F6";
  int selectedIndex = 0;
  std::optional<std::string> pendingSelectionRequest{};
  uint64_t version = 0;

  void set(std::span<const EditorSceneSelectionOption> scenes,
           std::string_view selectedSceneId, uint64_t newVersion,
           std::string_view hotkeyHintIn) {
    if (version != newVersion) {
      version = newVersion;
      ids.clear();
      names.clear();
      ids.reserve(scenes.size());
      names.reserve(scenes.size());
      for (const EditorSceneSelectionOption &scene : scenes) {
        ids.emplace_back(scene.id);
        names.emplace_back(scene.label);
      }
      nameViews.clear();
      nameViews.reserve(names.size());
      for (const std::string &name : names) {
        nameViews.push_back(name.c_str());
      }
    }
    if (hotkeyHintIn.empty()) {
      hotkeyHint = "Toggle Editor: F6";
    } else {
      hotkeyHint.assign(hotkeyHintIn.data(), hotkeyHintIn.size());
    }
    if (nameViews.empty()) {
      selectedIndex = 0;
      pendingSelectionRequest.reset();
      return;
    }
    auto it = std::find(ids.begin(), ids.end(), selectedSceneId);
    selectedIndex =
        it == ids.end() ? 0 : static_cast<int>(std::distance(ids.begin(), it));
  }
};

struct RenderableInspectorState {
  RenderableId renderableId = kInvalidRenderableId;
  MaterialRef ownedOverride = kInvalidMaterialRef;
  int baselineSlotIndex = 0;
  int selectedTextureIndex = 0;
};

struct HierarchyNodeStats {
  uint32_t renderableCount = 0u;
  uint32_t lightCount = 0u;
};

struct HierarchyNodeTopology {
  std::string labelName{};
  std::vector<NodeId> children{};
};

enum class HierarchyRowKind : uint8_t {
  SceneRoot,
  Node,
  Batch,
};

struct HierarchyVisibleRow {
  HierarchyRowKind kind = HierarchyRowKind::Node;
  int depth = 0;
  NodeId node = kInvalidNodeId;
  size_t beginIndex = 0u;
  size_t endIndex = 0u;
};

[[nodiscard]] constexpr size_t hierarchyNodeSlot(NodeId node) {
  return static_cast<size_t>(indexOf(node));
}

struct MaterialSourceEntry {
  MaterialRef ref = kInvalidMaterialRef;
  std::string label{};
};

struct MaterialTextureEntry {
  const char *label = "";
  TextureRef ref = kInvalidTextureRef;
};

template <typename T = ImTextureID>
inline T toImTextureId(uint32_t bindlessIndex) {
  if constexpr (std::is_pointer_v<T>) {
    return static_cast<T>(
        reinterpret_cast<void *>(static_cast<uintptr_t>(bindlessIndex)));
  } else {
    return static_cast<T>(static_cast<uintptr_t>(bindlessIndex));
  }
}

[[nodiscard]] std::string nodeDisplayName(const SceneGraph &graph,
                                          NodeId node) {
  std::string_view name{};
  if (graph.getNodeName(node, name) && !name.empty()) {
    return std::string(name);
  }
  return "Node #" + std::to_string(indexOf(node));
}

[[nodiscard]] bool selectionNodeStillValid(const SceneGraph &graph,
                                           const SceneEditorSelectionState &s) {
  if (!isValid(s.node)) {
    return false;
  }
  glm::mat4 dummy(1.0f);
  return graph.getNodeLocalTransform(s.node, dummy);
}

[[nodiscard]] std::optional<uint32_t>
findRenderableIndexById(const RenderScene &scene, RenderableId id) {
  return scene.findRenderableIndex(id);
}

void applyNodeSelection(const RenderScene &scene, NodeId node,
                        SceneEditorSelectionState &selection) {
  const RenderableId previousRenderable = selection.renderableId;
  selection.clear();
  if (!isValid(node)) {
    return;
  }

  selection.node = node;
  RenderableId firstRenderable = kInvalidRenderableId;
  RenderableId matchedRenderable = kInvalidRenderableId;
  scene.graph().forEachRenderableOnNode(node, [&](RenderableId renderableId) {
    if (!isValid(firstRenderable)) {
      firstRenderable = renderableId;
    }
    if (renderableId == previousRenderable) {
      matchedRenderable = renderableId;
    }
  });
  if (!isValid(matchedRenderable)) {
    matchedRenderable = firstRenderable;
  }
  if (isValid(matchedRenderable)) {
    const auto renderableIndex =
        findRenderableIndexById(scene, matchedRenderable);
    if (renderableIndex.has_value()) {
      selection.kind = SceneSelectionKind::NodeRenderable;
      selection.renderableId = matchedRenderable;
      selection.renderableIndex = *renderableIndex;
      return;
    }
  }

  LightId firstLight = kInvalidLightId;
  scene.graph().forEachLightOnNode(node, [&](LightId lightId) {
    if (!isValid(firstLight)) {
      firstLight = lightId;
    }
  });
  if (isValid(firstLight)) {
    selection.kind = SceneSelectionKind::Light;
    selection.lightId = firstLight;
    return;
  }
  selection.kind = SceneSelectionKind::Node;
}

[[nodiscard]] std::vector<NodeId> collectChildNodes(const SceneGraph &graph,
                                                    NodeId node) {
  std::vector<NodeId> out;
  NodeId child = kInvalidNodeId;
  if (!graph.getNodeFirstChild(node, child)) {
    return out;
  }
  while (isValid(child)) {
    out.push_back(child);
    NodeId next = kInvalidNodeId;
    if (!graph.getNodeNextSibling(child, next)) {
      break;
    }
    child = next;
  }
  std::reverse(out.begin(), out.end());
  return out;
}

[[nodiscard]] std::vector<MaterialSourceEntry>
buildMaterialSourceEntries(const Renderable &renderable,
                           const ResourceManager *resources) {
  std::vector<MaterialSourceEntry> entries;
  if (resources == nullptr) {
    entries.push_back(
        MaterialSourceEntry{.ref = renderable.material, .label = "Fallback"});
    return entries;
  }

  const ModelRecord *modelRecord = resources->tryGet(renderable.model);
  if (modelRecord == nullptr || modelRecord->model == nullptr ||
      modelRecord->model->sourceMaterialCount() == 0u) {
    entries.push_back(
        MaterialSourceEntry{.ref = renderable.material, .label = "Fallback"});
    return entries;
  }

  entries.reserve(modelRecord->model->sourceMaterialCount());
  for (uint32_t sourceIndex = 0;
       sourceIndex < modelRecord->model->sourceMaterialCount(); ++sourceIndex) {
    const MaterialRef mapped = modelRecord->materialForSource(sourceIndex);
    const MaterialRef resolved = isValid(mapped) ? mapped : renderable.material;
    MaterialSourceEntry entry{};
    entry.ref = resolved;
    entry.label = "Source Slot " + std::to_string(sourceIndex);
    if (const MaterialRecord *record = resources->tryGet(resolved);
        record != nullptr && !record->debugName.empty()) {
      entry.label += " - " + std::string(record->debugName);
    }
    entries.push_back(std::move(entry));
  }

  return entries;
}

[[nodiscard]] std::vector<MaterialTextureEntry>
buildMaterialTextureEntries(const MaterialRecord &record) {
  struct Spec {
    const char *label;
    TextureRef MaterialRequest::TextureRefs::*member;
  };
  constexpr Spec kSpecs[] = {
      {"Base Color", &MaterialRequest::TextureRefs::baseColor},
      {"Metallic Roughness", &MaterialRequest::TextureRefs::metallicRoughness},
      {"Normal", &MaterialRequest::TextureRefs::normal},
      {"Emissive", &MaterialRequest::TextureRefs::emissive},
      {"Occlusion", &MaterialRequest::TextureRefs::occlusion},
  };

  std::vector<MaterialTextureEntry> entries;
  for (const Spec &spec : kSpecs) {
    const TextureRef ref = record.textureRefs.*(spec.member);
    if (!isValid(ref)) {
      continue;
    }
    entries.push_back(MaterialTextureEntry{.label = spec.label, .ref = ref});
  }
  return entries;
}

constexpr std::array<int, 5> kAtlasResolutionSteps = {1024, 2048, 3072, 4096,
                                                      8192};

constexpr std::array<const char *, 5> kAtlasResolutionStepLabels = {
    "1K (1024)", "2K (2048)", "3K (3072)", "4K (4096)", "8K (8192)"};

void setPathText(std::array<char, 512> &buffer, std::string_view value) {
  buffer.fill('\0');
  const size_t copyCount = std::min(buffer.size() - 1u, value.size());
  if (copyCount > 0) {
    std::memcpy(buffer.data(), value.data(), copyCount);
  }
  buffer[copyCount] = '\0';
}

void syncOutputPathFromSource(FontCompilerUiState &state) {
  const std::filesystem::path sourcePath{std::string(state.sourcePath.data())};
  const std::string stem = sourcePath.stem().string();
  if (stem.empty()) {
    return;
  }

  if (state.outputDirectory.empty()) {
    state.outputDirectory = std::filesystem::path("assets") / "fonts";
  }
  const std::filesystem::path resolved =
      (state.outputDirectory / (stem + ".nfont")).lexically_normal();
  setPathText(state.outputPath, resolved.generic_string());
}

void refreshNfontAssetList(FontCompilerUiState &state) {
  state.availableNfonts.clear();
  std::error_code ec;
  if (!std::filesystem::exists(state.outputDirectory, ec) ||
      !std::filesystem::is_directory(state.outputDirectory, ec)) {
    state.selectedNfontIndex = -1;
    return;
  }

  for (const auto &entry :
       std::filesystem::directory_iterator(state.outputDirectory, ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_regular_file(ec)) {
      continue;
    }
    const std::filesystem::path path = entry.path();
    if (path.extension() == ".nfont") {
      state.availableNfonts.push_back(path.lexically_normal());
    }
  }
  std::sort(state.availableNfonts.begin(), state.availableNfonts.end());

  const std::filesystem::path currentPath{
      std::string(state.selectedNfontPath.data())};
  state.selectedNfontIndex = -1;
  for (size_t i = 0; i < state.availableNfonts.size(); ++i) {
    if (state.availableNfonts[i] == currentPath) {
      state.selectedNfontIndex = static_cast<int>(i);
      break;
    }
  }
  if (state.selectedNfontIndex < 0 && !state.availableNfonts.empty()) {
    state.selectedNfontIndex = 0;
    setPathText(state.selectedNfontPath,
                state.availableNfonts.front().generic_string());
  }
}

[[nodiscard]] int closestAtlasStepIndex(int value) {
  int bestIndex = 0;
  int bestDistance = std::abs(kAtlasResolutionSteps[0] - value);
  for (size_t i = 1; i < kAtlasResolutionSteps.size(); ++i) {
    const int distance = std::abs(kAtlasResolutionSteps[i] - value);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = static_cast<int>(i);
    }
  }
  return bestIndex;
}

struct LogModel {
  std::deque<LogLine> lines;
  std::uint64_t lastSequence = 0;
  bool seededFromFile = false;

  void clear() {
    lines.clear();
    lastSequence = 0;
    seededFromFile = false;
  }

  void trimLinesToCapacity() {
    while (lines.size() > kMaxLogLines) {
      lines.pop_front();
    }
  }

  static std::filesystem::path findLatestLogFile() {
    std::error_code ec;
    const std::filesystem::path logDir("logs");
    if (!std::filesystem::exists(logDir, ec)) {
      return {};
    }

    std::filesystem::path latest;
    std::filesystem::file_time_type latestTime{};
    for (const auto &entry : std::filesystem::directory_iterator(logDir, ec)) {
      if (ec) {
        break;
      }
      if (!entry.is_regular_file(ec)) {
        continue;
      }
      const auto path = entry.path();
      if (path.extension() != ".log") {
        continue;
      }
      const auto writeTime = entry.last_write_time(ec);
      if (ec) {
        continue;
      }
      if (latest.empty() || writeTime > latestTime) {
        latest = path;
        latestTime = writeTime;
      }
    }
    return latest;
  }

  static std::pair<LogLevel, std::string> parseLevelTag(std::string_view line) {
    for (const auto &meta : kLogLevels) {
      if (line.size() >= meta.tag.size() &&
          line.substr(0, meta.tag.size()) == meta.tag) {
        std::string msg(line.substr(meta.tag.size()));
        if (!msg.empty() && msg.front() == ' ') {
          msg.erase(0, 1);
        }
        return {meta.level, std::move(msg)};
      }
    }
    return {LogLevel::Info, std::string(line)};
  }

  void seedFromFileIfNeeded(LogFilterState &filterState) {
    if (seededFromFile) {
      return;
    }
    seededFromFile = true;

    const std::filesystem::path logPath = findLatestLogFile();
    if (logPath.empty()) {
      return;
    }

    std::ifstream file(logPath);
    if (!file.is_open()) {
      return;
    }

    std::string line;
    while (std::getline(file, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      auto [level, message] = parseLevelTag(line);
      LogLine entry{};
      entry.level = level;
      entry.message = std::move(message);
      lines.push_back(std::move(entry));
    }

    trimLinesToCapacity();
    if (!lines.empty()) {
      filterState.requestScroll = true;
    }
  }

  void appendEntries(const std::vector<LogEntry> &entries) {
    for (const auto &entry : entries) {
      LogLine line{};
      line.level = entry.level;
      line.message = entry.message;
      lines.push_back(std::move(line));
    }
    trimLinesToCapacity();
  }

  void update(LogFilterState &filterState) {
    seedFromFileIfNeeded(filterState);

    std::vector<LogEntry> entries;
    const LogReadResult result = readLogEntriesSince(lastSequence, entries);
    if (result.truncated) {
      lines.clear();
      lastSequence = result.lastSequence;
    }
    if (!entries.empty()) {
      lastSequence = result.lastSequence;
      appendEntries(entries);
      filterState.requestScroll = true;
    }
  }
};

std::string_view logTagFor(LogLevel level) {
  for (const auto &meta : kLogLevels) {
    if (meta.level == level) {
      return meta.tag;
    }
  }
  return "[Info]";
}

void drawInlineCheckbox(const char *label, bool &value) {
  ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
  ImGui::Checkbox(label, &value);
}

void drawLogToolbar(LogModel &model, LogFilterState &filterState) {
  if (ImGui::Button("Clear")) {
    model.clear();
    filterState.requestScroll = true;
  }

  drawInlineCheckbox("Auto-scroll", filterState.autoScroll);

  ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
  filterState.textFilter.Draw("Filter", kLogFilterWidth);

  struct Toggle {
    const char *label;
    bool *enabled;
  };
  Toggle toggles[] = {
      {"Trace", &filterState.showTrace}, {"Debug", &filterState.showDebug},
      {"Info", &filterState.showInfo},   {"Warn", &filterState.showWarning},
      {"Fatal", &filterState.showFatal},
  };
  for (const Toggle &toggle : toggles) {
    drawInlineCheckbox(toggle.label, *toggle.enabled);
  }
}

void drawLogMessages(const LogModel &model, LogFilterState &filterState,
                     std::pmr::memory_resource *scratchResource) {
  std::pmr::vector<size_t> visibleIndices(
      scratchResource ? scratchResource : std::pmr::get_default_resource());
  visibleIndices.reserve(model.lines.size());
  for (size_t lineIndex = 0; lineIndex < model.lines.size(); ++lineIndex) {
    const auto &line = model.lines[lineIndex];
    if (!filterState.levelEnabled(line.level)) {
      continue;
    }
    if (!filterState.textFilter.PassFilter(line.message.c_str())) {
      continue;
    }
    visibleIndices.push_back(lineIndex);
  }

  ImGui::BeginChild("LogScroll", ImVec2(0.0f, 0.0f), false,
                    ImGuiWindowFlags_HorizontalScrollbar);

  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(visibleIndices.size()));
  while (clipper.Step()) {
    for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
      const size_t visibleLineIndex = visibleIndices[static_cast<size_t>(i)];
      const LogLine &line = model.lines[visibleLineIndex];
      const std::string_view tag = logTagFor(line.level);
      ImGui::TextUnformatted(tag.data(), tag.data() + tag.size());
      ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
      ImGui::TextUnformatted(line.message.c_str());
    }
  }

  if (filterState.autoScroll && filterState.requestScroll) {
    ImGui::SetScrollHereY(1.0f);
    filterState.requestScroll = false;
  }
  ImGui::EndChild();
}

void drawInspectorHeader(std::string_view label) {
  ImGui::TextUnformatted(label.data(), label.data() + label.size());
  ImGui::Separator();
}

bool passKindUsesFeatureToggle(PassInspectorKind kind) {
  switch (kind) {
  case PassInspectorKind::Skybox:
  case PassInspectorKind::Opaque:
  case PassInspectorKind::Transmission:
  case PassInspectorKind::Transparent:
  case PassInspectorKind::Debug:
    return true;
  case PassInspectorKind::Composite:
  case PassInspectorKind::Generic:
    return false;
  }
  return false;
}

bool *renderSettingToggleForPassKind(RenderSettings &renderSettings,
                                     PassInspectorKind kind) {
  switch (kind) {
  case PassInspectorKind::Skybox:
    return &renderSettings.skybox.enabled;
  case PassInspectorKind::Opaque:
    return &renderSettings.opaque.enabled;
  case PassInspectorKind::Transmission:
    return &renderSettings.transmission.enabled;
  case PassInspectorKind::Transparent:
    return &renderSettings.transparent.enabled;
  case PassInspectorKind::Debug:
    return &renderSettings.debug.enabled;
  case PassInspectorKind::Composite:
  case PassInspectorKind::Generic:
    return nullptr;
  }
  return nullptr;
}

bool isPassInFamily(const RenderPipelinePassInfo &candidate,
                    const RenderPipelinePassInfo &selected,
                    PassInspectorKind selectedKind) {
  if (!selected.featureName.empty() &&
      candidate.featureName == selected.featureName) {
    return true;
  }
  return classifyPassInspector(candidate.featureName, candidate.passName) ==
         selectedKind;
}

bool isPassFamilyEnabled(RenderPipeline *renderPipeline,
                         const RenderPipelinePassInfo &selected,
                         PassInspectorKind kind) {
  if (renderPipeline == nullptr) {
    return false;
  }
  bool sawFamilyPass = false;
  for (size_t passIndex = 0; passIndex < renderPipeline->passCount();
       ++passIndex) {
    const auto candidate = renderPipeline->passInfo(passIndex);
    if (!candidate.has_value() || !isPassInFamily(*candidate, selected, kind)) {
      continue;
    }
    sawFamilyPass = true;
    if (!candidate->enabled) {
      return false;
    }
  }
  return sawFamilyPass;
}

void setPassFamilyEnabled(RenderPipeline *renderPipeline,
                          const RenderPipelinePassInfo &selected,
                          PassInspectorKind kind, bool enabled) {
  if (renderPipeline == nullptr) {
    return;
  }
  for (size_t passIndex = 0; passIndex < renderPipeline->passCount();
       ++passIndex) {
    const auto candidate = renderPipeline->passInfo(passIndex);
    if (!candidate.has_value() || !isPassInFamily(*candidate, selected, kind)) {
      continue;
    }
    renderPipeline->setPassEnabled(candidate->index, enabled);
  }
}

void syncFeatureToggleToRenderSettings(RenderSettings &renderSettings,
                                       PassInspectorKind kind, bool enabled) {
  if (bool *const toggle = renderSettingToggleForPassKind(renderSettings, kind);
      toggle != nullptr) {
    *toggle = enabled;
  }
}

void drawSkyboxSettings(RenderSettings::SkyboxSettings &skybox) {
  ImGui::Text("Skybox background: %s", skybox.enabled ? "enabled" : "disabled");
}

void drawOpaqueSettings(RenderSettings::OpaqueSettings &opaque) {
  constexpr const char *kDebugModes[] = {
      "None",
      "Wire Overlay",
      "Wireframe Only",
      "Tess Patch (Edges + Heatmap)",
  };
  int debugMode = static_cast<int>(opaque.debugVisualization);
  debugMode =
      std::clamp(debugMode, 0, static_cast<int>(IM_ARRAYSIZE(kDebugModes)) - 1);
  if (ImGui::Combo("Debug Visualization##OpaquePass", &debugMode, kDebugModes,
                   IM_ARRAYSIZE(kDebugModes))) {
    opaque.debugVisualization =
        static_cast<OpaqueDebugVisualization>(debugMode);
  }
  if (opaque.debugVisualization ==
      OpaqueDebugVisualization::TessPatchEdgesHeatmap) {
    ImGui::TextUnformatted(
        "Patch mode auto-enables tessellation for visualization.");
  }

  ImGui::Separator();
  ImGui::TextUnformatted("Depth");
  ImGui::Checkbox("Enable Depth Pre-pass##OpaquePass",
                  &opaque.enableDepthPrepass);
  ImGui::Checkbox("Enable Depth Pyramid##OpaquePass",
                  &opaque.enableDepthPyramid);

  ImGui::Separator();
  ImGui::TextUnformatted("Mesh LOD");
  ImGui::Checkbox("Enable Indirect Draws##OpaquePass",
                  &opaque.enableIndirectDraw);
  ImGui::Checkbox("Enable Instanced Draws##OpaquePass",
                  &opaque.enableInstancedDraw);
  ImGui::Checkbox("Enable Mesh LOD##OpaquePass", &opaque.enableMeshLod);
  ImGui::SliderInt("Forced LOD##OpaquePass", &opaque.forcedMeshLod, -1, 3);

  float lodThresholds[3] = {
      opaque.meshLodDistanceThresholds.x,
      opaque.meshLodDistanceThresholds.y,
      opaque.meshLodDistanceThresholds.z,
  };
  if (ImGui::SliderFloat3("LOD Distance##OpaquePass", lodThresholds, 0.5f,
                          128.0f, "%.1f")) {
    std::sort(std::begin(lodThresholds), std::end(lodThresholds));
    opaque.meshLodDistanceThresholds =
        glm::vec3(lodThresholds[0], lodThresholds[1], lodThresholds[2]);
  }

  ImGui::Separator();
  ImGui::TextUnformatted("Tessellation");
  ImGui::Checkbox("Enable Tessellation##OpaquePass",
                  &opaque.enableTessellation);
  ImGui::SliderFloat("Tess Near##OpaquePass", &opaque.tessNearDistance, 0.0f,
                     256.0f, "%.2f");
  ImGui::SliderFloat("Tess Far##OpaquePass", &opaque.tessFarDistance, 0.0f,
                     512.0f, "%.2f");
  ImGui::SliderFloat("Tess Min##OpaquePass", &opaque.tessMinFactor, 1.0f, 64.0f,
                     "%.2f");
  ImGui::SliderFloat("Tess Max##OpaquePass", &opaque.tessMaxFactor, 1.0f, 64.0f,
                     "%.2f");
  int tessMaxInstances = static_cast<int>(
      std::min<uint32_t>(opaque.tessMaxInstances, kUiMaxTessInstances));
  if (ImGui::SliderInt("Tess Max Inst##OpaquePass", &tessMaxInstances, 0,
                       4096)) {
    opaque.tessMaxInstances =
        static_cast<uint32_t>(std::max(tessMaxInstances, 0));
  }
  opaque.tessNearDistance = std::max(0.0f, opaque.tessNearDistance);
  opaque.tessFarDistance =
      std::max(opaque.tessFarDistance, opaque.tessNearDistance + 1.0e-3f);
  opaque.tessMinFactor = std::clamp(opaque.tessMinFactor, 1.0f, 64.0f);
  opaque.tessMaxFactor =
      std::clamp(opaque.tessMaxFactor, opaque.tessMinFactor, 64.0f);
  opaque.tessMaxInstances =
      std::min<uint32_t>(opaque.tessMaxInstances, kUiMaxTessInstances);
}

void drawDebugSettings(RenderSettings::DebugSettings &debug) {
  ImGui::Checkbox("Model Bounds##DebugPasses", &debug.modelBounds);
  ImGui::Checkbox("Grid##DebugPasses", &debug.grid);
  ImGui::Checkbox("Light Icons##DebugPasses", &debug.lightIcons);
}

void drawTransparentSettings(RenderSettings::TransparentSettings &transparent) {
  ImGui::Text("Transparent blending: %s",
              transparent.enabled ? "enabled" : "disabled");
}

void drawTransmissionSettings(
    RenderSettings::TransmissionSettings &transmission) {
  ImGui::Text("Transmission shading: %s",
              transmission.enabled ? "enabled" : "disabled");
}

void drawCompositeSettings() {
  ImGui::TextUnformatted("Composite runs only when an offscreen frame color");
  ImGui::TextUnformatted("target is produced and needs presentation.");
}

void drawTextureFilteringWindow(bool &open, RenderSettings &renderSettings,
                                const GPUDevice &gpu) {
  if (!ImGui::Begin(kTextureFilteringWindowName, &open)) {
    ImGui::End();
    return;
  }

  auto &settings = renderSettings.textureFiltering;
  sanitizeTextureFilteringSettings(settings);

  int modeIndex = static_cast<int>(settings.mode);
  modeIndex = std::clamp(modeIndex, 0,
                         static_cast<int>(kTextureFilterModeLabels.size()) - 1);
  if (ImGui::Combo("Mode", &modeIndex, kTextureFilterModeLabels.data(),
                   static_cast<int>(kTextureFilterModeLabels.size()))) {
    settings.mode = static_cast<TextureFilterMode>(modeIndex);
  }

  int anisotropyIndex = 2;
  for (int i = 0; i < static_cast<int>(kTextureFilterAnisotropyLevels.size());
       ++i) {
    if (settings.anisotropy ==
        kTextureFilterAnisotropyLevels[static_cast<size_t>(i)]) {
      anisotropyIndex = i;
      break;
    }
  }
  if (ImGui::Combo("Anisotropy", &anisotropyIndex,
                   kTextureFilterAnisotropyLabels.data(),
                   static_cast<int>(kTextureFilterAnisotropyLabels.size()))) {
    settings.anisotropy =
        kTextureFilterAnisotropyLevels[static_cast<size_t>(std::clamp(
            anisotropyIndex, 0,
            static_cast<int>(kTextureFilterAnisotropyLevels.size()) - 1))];
  }

  ImGui::Separator();
  ImGui::TextUnformatted("Bilinear disables mip blending.");
  ImGui::TextUnformatted("Trilinear blends between mip levels.");
  ImGui::TextUnformatted(
      "Anisotropic improves oblique-angle texture sampling.");
  ImGui::Separator();

  const uint8_t maxAnisotropy = gpu.getMaxSamplerAnisotropy();
  const TextureFilterMode effectiveMode =
      effectiveTextureFilterMode(settings, maxAnisotropy);
  ImGui::Text("Requested: %s", textureFilterModeDisplayName(settings.mode));
  ImGui::Text("Effective: %s", textureFilterModeDisplayName(effectiveMode));
  ImGui::Text("Requested Anisotropy: %ux", settings.anisotropy);
  ImGui::Text("Max Backend Anisotropy: %ux", maxAnisotropy);
  if (settings.mode == TextureFilterMode::Anisotropic && maxAnisotropy <= 1u) {
    ImGui::TextUnformatted(
        "Backend fallback: anisotropy unavailable, using trilinear.");
  }

  ImGui::End();
}

std::string makePassListLabel(const RenderPipelinePassInfo &passInfo) {
  std::string label;
  label.reserve(passInfo.passName.size() + passInfo.featureName.size() + 4u);
  label.append(passInfo.passName.begin(), passInfo.passName.end());
  if (!passInfo.featureName.empty()) {
    label.append("##");
    label.append(passInfo.featureName.begin(), passInfo.featureName.end());
    label.push_back('_');
    label.append(std::to_string(passInfo.index));
  }
  return label;
}

void drawPassList(RenderSettings &renderSettings,
                  RenderPipeline *renderPipeline, size_t &selectedPassIndex) {
  ImGui::TextUnformatted("Passes");
  ImGui::Separator();
  if (renderPipeline == nullptr || renderPipeline->passCount() == 0u) {
    ImGui::TextDisabled("No pipeline passes registered.");
    return;
  }

  if (selectedPassIndex >= renderPipeline->passCount()) {
    selectedPassIndex = 0u;
  }

  for (size_t passIndex = 0; passIndex < renderPipeline->passCount();
       ++passIndex) {
    const std::optional<RenderPipelinePassInfo> passInfo =
        renderPipeline->passInfo(passIndex);
    if (!passInfo.has_value()) {
      continue;
    }
    const PassInspectorKind kind =
        classifyPassInspector(passInfo->featureName, passInfo->passName);
    bool enabled = passKindUsesFeatureToggle(kind)
                       ? isPassFamilyEnabled(renderPipeline, *passInfo, kind)
                       : passInfo->enabled;
    if (ImGui::Checkbox(
            ("##PassEnabled" + std::to_string(passInfo->index)).c_str(),
            &enabled)) {
      if (passKindUsesFeatureToggle(kind)) {
        setPassFamilyEnabled(renderPipeline, *passInfo, kind, enabled);
        syncFeatureToggleToRenderSettings(renderSettings, kind, enabled);
      } else {
        renderPipeline->setPassEnabled(passInfo->index, enabled);
      }
    }
    ImGui::SameLine();
    const std::string label = makePassListLabel(*passInfo);
    const bool isSelected = selectedPassIndex == passInfo->index;
    if (ImGui::Selectable(label.c_str(), isSelected)) {
      selectedPassIndex = passInfo->index;
    }
  }
}

void drawPassInspector(RenderSettings &renderSettings,
                       RenderPipeline *renderPipeline,
                       size_t &selectedPassIndex) {
  ImGui::BeginChild("PassPanel", ImVec2(0.0f, 0.0f), false,
                    ImGuiWindowFlags_NoScrollbar);

  if (ImGui::BeginTable("PassInspectorTable", 2,
                        ImGuiTableFlags_BordersInnerV |
                            ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableSetupColumn("PassList", ImGuiTableColumnFlags_WidthFixed,
                            kPassListWidth);
    ImGui::TableSetupColumn("PassSettings", ImGuiTableColumnFlags_WidthStretch,
                            0.0f);

    ImGui::TableNextColumn();
    drawPassList(renderSettings, renderPipeline, selectedPassIndex);

    ImGui::TableNextColumn();
    if (renderPipeline == nullptr || renderPipeline->passCount() == 0u) {
      drawInspectorHeader("No Pass Selected");
      ImGui::TextUnformatted("RenderPipeline is unavailable.");
    } else {
      selectedPassIndex =
          std::min(selectedPassIndex, renderPipeline->passCount() - 1u);
      const std::optional<RenderPipelinePassInfo> passInfo =
          renderPipeline->passInfo(selectedPassIndex);
      if (!passInfo.has_value()) {
        drawInspectorHeader("No Pass Selected");
        ImGui::TextUnformatted("Selected pass entry is unavailable.");
      } else {
        std::string title(passInfo->passName);
        if (!passInfo->featureName.empty()) {
          title.append(" (");
          title.append(passInfo->featureName);
          title.push_back(')');
        }
        drawInspectorHeader(title);
        const PassInspectorKind kind =
            classifyPassInspector(passInfo->featureName, passInfo->passName);
        bool enabled =
            passKindUsesFeatureToggle(kind)
                ? isPassFamilyEnabled(renderPipeline, *passInfo, kind)
                : passInfo->enabled;
        if (ImGui::Checkbox("Enabled", &enabled)) {
          if (passKindUsesFeatureToggle(kind)) {
            setPassFamilyEnabled(renderPipeline, *passInfo, kind, enabled);
            syncFeatureToggleToRenderSettings(renderSettings, kind, enabled);
          } else {
            renderPipeline->setPassEnabled(passInfo->index, enabled);
          }
        }
        ImGui::Separator();

        switch (kind) {
        case PassInspectorKind::Skybox:
          drawSkyboxSettings(renderSettings.skybox);
          break;
        case PassInspectorKind::Opaque:
          drawOpaqueSettings(renderSettings.opaque);
          break;
        case PassInspectorKind::Transmission:
          drawTransmissionSettings(renderSettings.transmission);
          break;
        case PassInspectorKind::Transparent:
          drawTransparentSettings(renderSettings.transparent);
          break;
        case PassInspectorKind::Composite:
          drawCompositeSettings();
          break;
        case PassInspectorKind::Debug:
          drawDebugSettings(renderSettings.debug);
          break;
        case PassInspectorKind::Generic:
          ImGui::TextUnformatted("This pass has no dedicated inspector yet.");
          break;
        }
      }
    }

    ImGui::EndTable();
  }

  ImGui::EndChild();
}

void drawLogWindow(LogModel &model, LogFilterState &filterState,
                   std::pmr::memory_resource *scratchResource) {
  drawLogToolbar(model, filterState);
  ImGui::Separator();
  drawLogMessages(model, filterState, scratchResource);
}

void drawRenderPassesWindow(bool &open, RenderSettings &renderSettings,
                            RenderPipeline *renderPipeline,
                            size_t &selectedPassIndex) {
  if (!ImGui::Begin(kRenderPassesWindowName, &open)) {
    ImGui::End();
    return;
  }
  drawPassInspector(renderSettings, renderPipeline, selectedPassIndex);
  ImGui::End();
}

void drawFontCompilerWindow(bool &open, FontCompilerUiState &state,
                            TextSystem *textSystem, void *ownerWindowHandle) {
  if (!state.nfontListInitialized) {
    refreshNfontAssetList(state);
    state.nfontListInitialized = true;
  }

  if (state.compileInFlight && state.compileFuture.valid()) {
    const auto waitResult =
        state.compileFuture.wait_for(std::chrono::seconds(0));
    if (waitResult == std::future_status::ready) {
      auto compileResult = state.compileFuture.get();
      state.compileInFlight = false;
      if (compileResult.hasError()) {
        state.error = compileResult.error();
      } else {
        state.lastReport = compileResult.value();
        std::ostringstream status;
        status << "Generated " << state.lastReport.outputPath.string()
               << " | glyphs=" << state.lastReport.glyphCount
               << " atlas=" << state.lastReport.atlasWidth << "x"
               << state.lastReport.atlasHeight
               << " bytes=" << state.lastReport.bytesWritten;
        state.status = status.str();
        setPathText(
            state.selectedNfontPath,
            state.lastReport.outputPath.lexically_normal().generic_string());
        refreshNfontAssetList(state);
      }
    }
  }

  if (!ImGui::Begin(kFontCompilerWindowName, &open)) {
    ImGui::End();
    return;
  }

  bool sourceEdited = ImGui::InputText(
      "Source TTF/OTF", state.sourcePath.data(), state.sourcePath.size());
  ImGui::SameLine();
  if (ImGui::Button("Browse...##FontSource")) {
    if (const auto selectedPath =
            state.fileDialog.openFontFile(ownerWindowHandle)) {
      setPathText(state.sourcePath, selectedPath->generic_string());
      sourceEdited = true;
    }
  }

  if (sourceEdited && state.autoOutputName) {
    syncOutputPathFromSource(state);
  }

  ImGui::InputText("Output .nfont", state.outputPath.data(),
                   state.outputPath.size());
  ImGui::SameLine();
  if (ImGui::Checkbox("Auto name", &state.autoOutputName) &&
      state.autoOutputName) {
    syncOutputPathFromSource(state);
  }

  constexpr const char *kCharsetOptions[] = {"ASCII", "Latin-1"};
  ImGui::Combo("Charset", &state.charsetPreset, kCharsetOptions,
               IM_ARRAYSIZE(kCharsetOptions));

  ImGui::SliderFloat("Minimum EM Size", &state.minimumEmSize, 8.0f, 128.0f,
                     "%.1f");
  ImGui::SliderFloat("PX Range", &state.pxRange, 1.0f, 16.0f, "%.1f");
  ImGui::SliderFloat("Outer PX Padding", &state.outerPixelPadding, 0.0f, 16.0f,
                     "%.1f");
  ImGui::SliderInt("Atlas Spacing", &state.atlasSpacing, 0, 16);
  ImGui::Checkbox("RGBA16F Atlas", &state.useRgba16fAtlas);
  if (ImGui::SliderInt("Atlas Width Step", &state.atlasWidthPreset, 0,
                       static_cast<int>(kAtlasResolutionSteps.size()) - 1)) {
    state.atlasWidthPreset =
        std::clamp(state.atlasWidthPreset, 0,
                   static_cast<int>(kAtlasResolutionSteps.size()) - 1);
    state.maxAtlasWidth =
        kAtlasResolutionSteps[static_cast<size_t>(state.atlasWidthPreset)];
  }
  ImGui::SameLine();
  ImGui::TextUnformatted(kAtlasResolutionStepLabels[static_cast<size_t>(
      std::clamp(state.atlasWidthPreset, 0,
                 static_cast<int>(kAtlasResolutionStepLabels.size()) - 1))]);
  if (ImGui::InputInt("Max Atlas Width", &state.maxAtlasWidth)) {
    state.maxAtlasWidth = std::clamp(state.maxAtlasWidth, 1, 8192);
    state.atlasWidthPreset = closestAtlasStepIndex(state.maxAtlasWidth);
  }

  if (ImGui::SliderInt("Atlas Height Step", &state.atlasHeightPreset, 0,
                       static_cast<int>(kAtlasResolutionSteps.size()) - 1)) {
    state.atlasHeightPreset =
        std::clamp(state.atlasHeightPreset, 0,
                   static_cast<int>(kAtlasResolutionSteps.size()) - 1);
    state.maxAtlasHeight =
        kAtlasResolutionSteps[static_cast<size_t>(state.atlasHeightPreset)];
  }
  ImGui::SameLine();
  ImGui::TextUnformatted(kAtlasResolutionStepLabels[static_cast<size_t>(
      std::clamp(state.atlasHeightPreset, 0,
                 static_cast<int>(kAtlasResolutionStepLabels.size()) - 1))]);
  if (ImGui::InputInt("Max Atlas Height", &state.maxAtlasHeight)) {
    state.maxAtlasHeight = std::clamp(state.maxAtlasHeight, 1, 8192);
    state.atlasHeightPreset = closestAtlasStepIndex(state.maxAtlasHeight);
  }

  ImGui::SliderInt("Threads", &state.threadCount, 0, 32);

  const bool wasCompileInFlight = state.compileInFlight;
  if (wasCompileInFlight) {
    ImGui::BeginDisabled();
  }
  if (ImGui::Button("Generate .nfont")) {
    state.status.clear();
    state.error.clear();

    NFontCompileConfig config{};
    config.sourceFontPath = std::string(state.sourcePath.data());
    config.outputFontPath = std::string(state.outputPath.data());
    config.charset = state.charsetPreset == 0 ? NFontCharsetPreset::Ascii
                                              : NFontCharsetPreset::Latin1;
    config.minimumEmSize = state.minimumEmSize;
    config.pxRange = state.pxRange;
    config.outerPixelPadding = state.outerPixelPadding;
    config.atlasSpacing =
        state.atlasSpacing > 0 ? static_cast<uint32_t>(state.atlasSpacing) : 0u;
    config.useRgba16fAtlas = state.useRgba16fAtlas;
    config.maxAtlasWidth = state.maxAtlasWidth > 0
                               ? static_cast<uint32_t>(state.maxAtlasWidth)
                               : 0u;
    config.maxAtlasHeight = state.maxAtlasHeight > 0
                                ? static_cast<uint32_t>(state.maxAtlasHeight)
                                : 0u;
    config.threadCount =
        state.threadCount > 0 ? static_cast<uint32_t>(state.threadCount) : 0u;

    state.compileFuture = std::async(std::launch::async, [config]() {
                            return compileNFontFromFontFile(config);
                          }).share();
    state.compileInFlight = true;
  }
  if (wasCompileInFlight) {
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextUnformatted("Compiling...");
  }

  if (!state.status.empty()) {
    ImGui::Spacing();
    ImGui::TextUnformatted(state.status.c_str());
  }
  if (!state.error.empty()) {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s",
                       state.error.c_str());
  }

  ImGui::Separator();
  ImGui::TextUnformatted("Global Text Font");

  if (textSystem == nullptr) {
    ImGui::TextUnformatted("TextSystem is not available.");
    ImGui::End();
    return;
  }

  if (ImGui::Button("Refresh .nfont List")) {
    refreshNfontAssetList(state);
  }
  ImGui::SameLine();
  ImGui::Text("Dir: %s", state.outputDirectory.generic_string().c_str());

  const std::string previewLabel =
      state.selectedNfontIndex >= 0 &&
              static_cast<size_t>(state.selectedNfontIndex) <
                  state.availableNfonts.size()
          ? state.availableNfonts[static_cast<size_t>(state.selectedNfontIndex)]
                .filename()
                .string()
          : std::string("<none>");
  if (ImGui::BeginCombo("Available .nfont", previewLabel.c_str())) {
    for (size_t i = 0; i < state.availableNfonts.size(); ++i) {
      const bool selected = state.selectedNfontIndex == static_cast<int>(i);
      const std::string label = state.availableNfonts[i].filename().string();
      if (ImGui::Selectable(label.c_str(), selected)) {
        state.selectedNfontIndex = static_cast<int>(i);
        setPathText(state.selectedNfontPath,
                    state.availableNfonts[i].generic_string());
      }
      if (selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

  if (ImGui::InputText("Selected .nfont", state.selectedNfontPath.data(),
                       state.selectedNfontPath.size())) {
    const std::filesystem::path selectedPath(
        std::string(state.selectedNfontPath.data()));
    state.selectedNfontIndex = -1;
    for (size_t i = 0; i < state.availableNfonts.size(); ++i) {
      if (state.availableNfonts[i] == selectedPath) {
        state.selectedNfontIndex = static_cast<int>(i);
        break;
      }
    }
  }

  state.globalFontSizePx = std::clamp(state.globalFontSizePx, 8.0f, 256.0f);
  ImGui::SliderFloat("Global Font Size (px)", &state.globalFontSizePx, 8.0f,
                     256.0f, "%.1f");
  if (ImGui::Button("Apply Global Font")) {
    state.globalStatus.clear();
    state.globalError.clear();
    textSystem->setDefaultFontSizePx(state.globalFontSizePx);

    const std::filesystem::path selectedPath(
        std::string(state.selectedNfontPath.data()));
    if (selectedPath.empty()) {
      state.globalError = "No .nfont selected";
    } else {
      auto loadResult = textSystem->loadAndSetDefaultFont(
          selectedPath.generic_string(), selectedPath.stem().string());
      if (loadResult.hasError()) {
        state.globalError = loadResult.error();
      } else {
        std::ostringstream oss;
        oss << "Applied " << selectedPath.filename().string() << " at "
            << state.globalFontSizePx << "px";
        state.globalStatus = oss.str();
      }
    }
  }

  if (!state.globalStatus.empty()) {
    ImGui::Spacing();
    ImGui::TextUnformatted(state.globalStatus.c_str());
  }
  if (!state.globalError.empty()) {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s",
                       state.globalError.c_str());
  }

  ImGui::End();
}

void drawBakeryWindow(bool &open, BakeryUiState &state,
                      bakery::BakerySystem *bakery,
                      std::pmr::memory_resource *scratchResource,
                      void *ownerWindowHandle) {
  if (!ImGui::Begin(kBakeryWindowName, &open)) {
    ImGui::End();
    return;
  }

  if (bakery == nullptr) {
    ImGui::TextUnformatted("Bakery system is not available.");
    ImGui::End();
    return;
  }

  ImGui::Checkbox("Force Rebuild", &state.forceRebuild);

  if (ImGui::Button("Queue BRDF LUT")) {
    state.status.clear();
    state.error.clear();
    auto enqueueResult =
        bakery->enqueue(bakery::BakeRequest{bakery::BrdfLutBakeRequest{
            .forceRebuild = state.forceRebuild,
        }});
    if (enqueueResult.hasError()) {
      state.error = enqueueResult.error();
    } else {
      std::ostringstream oss;
      oss << "Queued BRDF LUT job #" << enqueueResult.value().value;
      state.status = oss.str();
    }
  }

  ImGui::InputText("Env HDR Path", state.envHdrPath.data(),
                   state.envHdrPath.size());
  ImGui::SameLine();
  if (ImGui::Button("Browse...##EnvHdr")) {
    static constexpr std::array<FileDialogFilter, 4> kHdrFilters = {
        FileDialogFilter{"HDR Files (*.hdr;*.exr)", "*.hdr;*.exr"},
        FileDialogFilter{"Radiance HDR (*.hdr)", "*.hdr"},
        FileDialogFilter{"OpenEXR (*.exr)", "*.exr"},
        FileDialogFilter{"All Files (*.*)", "*.*"},
    };
    OpenFileRequest request{};
    request.title = "Select Environment HDR";
    request.filters = kHdrFilters;
    request.defaultExtension = "hdr";
    request.ownerWindowHandle = ownerWindowHandle;
    if (const auto selectedPath = state.fileDialog.openFile(request)) {
      setPathText(state.envHdrPath, selectedPath->generic_string());
    }
  }
  if (ImGui::Button("Queue Env Prefilter")) {
    state.status.clear();
    state.error.clear();
    auto enqueueResult =
        bakery->enqueue(bakery::BakeRequest{bakery::EnvmapPrefilterBakeRequest{
            .environmentHdrPath =
                std::filesystem::path(std::string(state.envHdrPath.data())),
            .forceRebuild = state.forceRebuild,
        }});
    if (enqueueResult.hasError()) {
      state.error = enqueueResult.error();
    } else {
      std::ostringstream oss;
      oss << "Queued Env Prefilter job #" << enqueueResult.value().value;
      state.status = oss.str();
    }
  }

  ImGui::Separator();
  ImGui::InputText("Scene Path", state.scenePath.data(),
                   state.scenePath.size());
  ImGui::SameLine();
  if (ImGui::Button("Browse...##SceneBake")) {
    static constexpr std::array<FileDialogFilter, 3> kSceneFilters = {
        FileDialogFilter{"Scene Files (*.gltf;*.glb)", "*.gltf;*.glb"},
        FileDialogFilter{"glTF (*.gltf)", "*.gltf"},
        FileDialogFilter{"GLB (*.glb)", "*.glb"},
    };
    OpenFileRequest request{};
    request.title = "Select Scene for Portable Asset Bake";
    request.filters = kSceneFilters;
    request.defaultExtension = "gltf";
    request.ownerWindowHandle = ownerWindowHandle;
    if (const auto selectedPath = state.fileDialog.openFile(request)) {
      setPathText(state.scenePath, selectedPath->generic_string());
    }
  }

  ImGui::Checkbox("Prebuild BC7", &state.prebuildBc7);
  ImGui::SameLine();
  ImGui::Checkbox("Prebuild ETC2", &state.prebuildEtc2);
  ImGui::SameLine();
  ImGui::Checkbox("Prebuild RGBA8", &state.prebuildRgba8);
  ImGui::TextUnformatted(
      "Scene Portable Assets writes mipmapped KTX2 for PNG/KTX sources.");
  ImGui::TextUnformatted(
      "External DDS textures stay authored and are not rebaked here.");

  if (ImGui::Button("Queue Scene Portable Assets")) {
    state.status.clear();
    state.error.clear();

    std::vector<bakery::ScenePortableTextureTarget> prebuildTargets{};
    if (state.prebuildBc7) {
      prebuildTargets.push_back(bakery::ScenePortableTextureTarget::BC7);
    }
    if (state.prebuildEtc2) {
      prebuildTargets.push_back(bakery::ScenePortableTextureTarget::ETC2);
    }
    if (state.prebuildRgba8) {
      prebuildTargets.push_back(bakery::ScenePortableTextureTarget::RGBA8);
    }

    auto enqueueResult = bakery->enqueue(
        bakery::BakeRequest{bakery::ScenePortableAssetsBakeRequest{
            .scenePath =
                std::filesystem::path(std::string(state.scenePath.data())),
            .prebuildNativeTargets = std::move(prebuildTargets),
            .forceRebuild = state.forceRebuild,
        }});
    if (enqueueResult.hasError()) {
      state.error = enqueueResult.error();
    } else {
      std::ostringstream oss;
      oss << "Queued Scene Portable Assets job #"
          << enqueueResult.value().value;
      state.status = oss.str();
    }
  }

  if (!state.status.empty()) {
    ImGui::Spacing();
    ImGui::TextUnformatted(state.status.c_str());
  }
  if (!state.error.empty()) {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s",
                       state.error.c_str());
  }

  std::pmr::vector<bakery::BakeJobSnapshot> jobs =
      bakery->snapshotJobs(scratchResource);
  ImGui::Separator();
  ImGui::Text("Jobs: %d", static_cast<int>(jobs.size()));
  if (ImGui::BeginTable("BakeryJobs", 6,
                        ImGuiTableFlags_BordersInnerV |
                            ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_ScrollY,
                        ImVec2(0.0f, 240.0f))) {
    ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 52.0f);
    ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed, 120.0f);
    ImGui::TableSetupColumn("State", ImGuiTableColumnFlags_WidthFixed, 110.0f);
    ImGui::TableSetupColumn("Progress", ImGuiTableColumnFlags_WidthFixed,
                            90.0f);
    ImGui::TableSetupColumn("Summary", ImGuiTableColumnFlags_WidthStretch,
                            0.0f);
    ImGui::TableSetupColumn("Error", ImGuiTableColumnFlags_WidthStretch, 0.0f);
    ImGui::TableHeadersRow();

    for (const bakery::BakeJobSnapshot &job : jobs) {
      ImGui::TableNextRow();

      ImGui::TableNextColumn();
      ImGui::Text("%llu", static_cast<unsigned long long>(job.id.value));

      ImGui::TableNextColumn();
      ImGui::TextUnformatted(bakeJobKindName(job.kind));

      ImGui::TableNextColumn();
      ImGui::TextUnformatted(bakeJobStateName(job.state));

      ImGui::TableNextColumn();
      ImGui::Text("%.0f%%", std::clamp(job.progress01, 0.0f, 1.0f) * 100.0f);

      ImGui::TableNextColumn();
      ImGui::TextUnformatted(job.summary.c_str());

      ImGui::TableNextColumn();
      if (job.error.empty()) {
        ImGui::TextUnformatted("-");
      } else {
        ImGui::TextWrapped("%s", job.error.c_str());
      }
    }
    ImGui::EndTable();
  }

  ImGui::End();
}

[[nodiscard]] std::string_view
resolveTelemetryPassName(const RenderGraphTelemetrySnapshot &snapshot,
                         uint32_t passIndex) {
  if (passIndex >= snapshot.passNames.size()) {
    return "unnamed_pass";
  }
  const std::pmr::string &name = snapshot.passNames[passIndex];
  return name.empty() ? std::string_view("unnamed_pass")
                      : std::string_view(name.data(), name.size());
}

const char *drawBufferBindingTargetName(
    RenderGraphCompileResult::DrawBufferBindingTarget target) {
  switch (target) {
  case RenderGraphCompileResult::DrawBufferBindingTarget::Vertex:
    return "vertex";
  case RenderGraphCompileResult::DrawBufferBindingTarget::Index:
    return "index";
  case RenderGraphCompileResult::DrawBufferBindingTarget::Indirect:
    return "indirect";
  case RenderGraphCompileResult::DrawBufferBindingTarget::IndirectCount:
    return "indirect_count";
  }
  return "unknown";
}

const char *passTextureBindingTargetName(
    RenderGraphCompileResult::PassTextureBindingTarget target) {
  switch (target) {
  case RenderGraphCompileResult::PassTextureBindingTarget::Color:
    return "color";
  case RenderGraphCompileResult::PassTextureBindingTarget::Depth:
    return "depth";
  }
  return "unknown";
}

void syncTelemetryDumpPath(RenderGraphTelemetryUiState &state,
                           const RenderGraphTelemetryService *telemetry) {
  if (telemetry == nullptr) {
    return;
  }
  const std::string currentPath = state.outputPath.data();
  const std::string suggestedPath =
      telemetry->suggestDumpPath().generic_string();
  if (state.initializedOutputPath && currentPath != state.lastSuggestedPath) {
    return;
  }
  setPathText(state.outputPath, suggestedPath);
  state.lastSuggestedPath = suggestedPath;
  state.initializedOutputPath = true;
}

void drawTextView(std::string_view text) {
  ImGui::TextUnformatted(text.data(), text.data() + text.size());
}

void drawTelemetrySummary(
    const RenderGraphTelemetrySnapshot::Summary &summary) {
  if (!ImGui::BeginTable("RenderGraphTelemetrySummary", 2,
                         ImGuiTableFlags_BordersInnerV |
                             ImGuiTableFlags_SizingStretchProp)) {
    return;
  }

  const auto drawRow = [](const char *label, auto value) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(label);
    ImGui::TableNextColumn();
    ImGui::Text("%llu", static_cast<unsigned long long>(value));
  };

  drawRow("Frame Index", summary.frameIndex);
  drawRow("Declared Passes", summary.declaredPassCount);
  drawRow("Culled Passes", summary.culledPassCount);
  drawRow("Root Passes", summary.rootPassCount);
  drawRow("Pass Count", summary.passCount);
  drawRow("Edge Count", summary.edgeCount);
  drawRow("Imported Textures", summary.importedTextures);
  drawRow("Transient Textures", summary.transientTextures);
  drawRow("Imported Buffers", summary.importedBuffers);
  drawRow("Transient Buffers", summary.transientBuffers);
  drawRow("Texture Lifetimes", summary.transientTextureLifetimeCount);
  drawRow("Buffer Lifetimes", summary.transientBufferLifetimeCount);
  drawRow("Texture Physicals", summary.transientTexturePhysicalCount);
  drawRow("Buffer Physicals", summary.transientBufferPhysicalCount);
  drawRow("Resolved Dependency Slots",
          summary.resolvedDependencyBufferSlotCount);
  drawRow("Resolved Pre-Dispatch Slots",
          summary.resolvedPreDispatchDependencyBufferSlotCount);
  drawRow("Unresolved Texture Bindings", summary.unresolvedTextureBindingCount);
  drawRow("Unresolved Dependency Bindings",
          summary.unresolvedDependencyBufferBindingCount);
  drawRow("Unresolved Pre-Dispatch Bindings",
          summary.unresolvedPreDispatchDependencyBufferBindingCount);
  drawRow("Unresolved Draw Bindings", summary.unresolvedDrawBufferBindingCount);

  ImGui::EndTable();
}

template <typename Fn>
void drawTelemetryTableSection(const char *header, const char *tableId,
                               int columns, bool hasRows, float height,
                               Fn &&drawRows) {
  if (!ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) {
    return;
  }
  if (!hasRows) {
    ImGui::TextUnformatted("<none>");
    return;
  }

  if (ImGui::BeginTable(tableId, columns,
                        ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY,
                        ImVec2(0.0f, height))) {
    drawRows();
    ImGui::EndTable();
  }
}

void drawRenderGraphTelemetryWindow(RenderGraphTelemetryUiState &state,
                                    RenderGraphTelemetryService *telemetry,
                                    void *ownerWindowHandle) {
  syncTelemetryDumpPath(state, telemetry);

  ImGui::InputText("Output .txt", state.outputPath.data(),
                   state.outputPath.size());
  ImGui::SameLine();
  if (ImGui::Button("Browse...##RenderGraphTelemetry")) {
    static constexpr std::array<FileDialogFilter, 2> kTelemetryFilters = {
        FileDialogFilter{"Text Files (*.txt)", "*.txt"},
        FileDialogFilter{"All Files (*.*)", "*.*"},
    };
    SaveFileRequest request{};
    request.title = "Save Render Graph Telemetry";
    request.filters = kTelemetryFilters;
    request.defaultExtension = "txt";
    request.initialPath = state.outputPath.data();
    request.ownerWindowHandle = ownerWindowHandle;
    if (const auto selectedPath = state.fileDialog.saveFile(request)) {
      setPathText(state.outputPath, selectedPath->generic_string());
      state.initializedOutputPath = true;
    }
  }

  const bool canDump = telemetry != nullptr && telemetry->hasSnapshot();
  if (!canDump) {
    ImGui::BeginDisabled();
  }
  if (ImGui::Button("Dump Current Telemetry")) {
    state.status.clear();
    state.error.clear();
    auto dumpResult = telemetry->writeLatestTextDump(state.outputPath.data());
    if (dumpResult.hasError()) {
      state.error = dumpResult.error();
    } else {
      state.status =
          std::string("Wrote telemetry to ") + state.outputPath.data();
    }
  }
  if (!canDump) {
    ImGui::EndDisabled();
  }

  if (!state.status.empty()) {
    ImGui::Spacing();
    ImGui::TextUnformatted(state.status.c_str());
  }
  if (!state.error.empty()) {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s",
                       state.error.c_str());
  }

  const RenderGraphTelemetrySnapshot *snapshot =
      telemetry != nullptr ? telemetry->latestSnapshot() : nullptr;
  if (snapshot == nullptr) {
    ImGui::Spacing();
    ImGui::TextUnformatted("No render-graph telemetry has been captured yet.");
    return;
  }

  ImGui::Separator();
  drawTelemetrySummary(snapshot->summary);

  drawTelemetryTableSection(
      "Passes", "RenderGraphTelemetryPasses", 2, !snapshot->passNames.empty(),
      160.0f, [&]() {
        ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthFixed,
                                64.0f);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (uint32_t passIndex = 0; passIndex < snapshot->passNames.size();
             ++passIndex) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::Text("%u", passIndex);
          ImGui::TableNextColumn();
          drawTextView(resolveTelemetryPassName(*snapshot, passIndex));
        }
      });

  drawTelemetryTableSection(
      "Edges", "RenderGraphTelemetryEdges", 4, !snapshot->edges.empty(), 140.0f,
      [&]() {
        ImGui::TableSetupColumn("Before", ImGuiTableColumnFlags_WidthFixed,
                                60.0f);
        ImGui::TableSetupColumn("Before Name",
                                ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("After", ImGuiTableColumnFlags_WidthFixed,
                                60.0f);
        ImGui::TableSetupColumn("After Name",
                                ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (const auto &edge : snapshot->edges) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::Text("%u", edge.before);
          ImGui::TableNextColumn();
          drawTextView(resolveTelemetryPassName(*snapshot, edge.before));
          ImGui::TableNextColumn();
          ImGui::Text("%u", edge.after);
          ImGui::TableNextColumn();
          drawTextView(resolveTelemetryPassName(*snapshot, edge.after));
        }
      });

  drawTelemetryTableSection(
      "Execution Order", "RenderGraphTelemetryExecution", 3,
      !snapshot->orderedPassIndices.empty(), 140.0f, [&]() {
        ImGui::TableSetupColumn("Rank", ImGuiTableColumnFlags_WidthFixed,
                                60.0f);
        ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthFixed,
                                60.0f);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (uint32_t rank = 0; rank < snapshot->orderedPassIndices.size();
             ++rank) {
          const uint32_t passIndex = snapshot->orderedPassIndices[rank];
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::Text("%u", rank);
          ImGui::TableNextColumn();
          ImGui::Text("%u", passIndex);
          ImGui::TableNextColumn();
          drawTextView(resolveTelemetryPassName(*snapshot, passIndex));
        }
      });

  drawTelemetryTableSection(
      "Transient Lifetimes", "RenderGraphTelemetryTextureLifetimes", 4,
      !snapshot->transientTextureLifetimes.empty() ||
          !snapshot->transientBufferLifetimes.empty(),
      180.0f, [&]() {
        ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Resource", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("First", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Last", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableHeadersRow();
        for (const auto &lifetime : snapshot->transientTextureLifetimes) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("tex");
          ImGui::TableNextColumn();
          ImGui::Text("%u", lifetime.resourceIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", lifetime.firstExecutionIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", lifetime.lastExecutionIndex);
        }
        for (const auto &lifetime : snapshot->transientBufferLifetimes) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("buf");
          ImGui::TableNextColumn();
          ImGui::Text("%u", lifetime.resourceIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", lifetime.firstExecutionIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", lifetime.lastExecutionIndex);
        }
      });

  drawTelemetryTableSection(
      "Allocations", "RenderGraphTelemetryAllocations", 4,
      !snapshot->transientTextureAllocations.empty() ||
          !snapshot->transientBufferAllocations.empty(),
      180.0f, [&]() {
        ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Resource", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Physical", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Map", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (const auto &allocation : snapshot->transientTextureAllocations) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("tex");
          ImGui::TableNextColumn();
          ImGui::Text("%u", allocation.resourceIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", allocation.allocationIndex);
          ImGui::TableNextColumn();
          ImGui::Text("tex[%u] -> phys[%u]", allocation.resourceIndex,
                      allocation.allocationIndex);
        }
        for (const auto &allocation : snapshot->transientBufferAllocations) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("buf");
          ImGui::TableNextColumn();
          ImGui::Text("%u", allocation.resourceIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", allocation.allocationIndex);
          ImGui::TableNextColumn();
          ImGui::Text("buf[%u] -> phys[%u]", allocation.resourceIndex,
                      allocation.allocationIndex);
        }
      });

  drawTelemetryTableSection(
      "Allocation Maps", "RenderGraphTelemetryAllocationMaps", 3,
      !snapshot->transientTextureAllocationByResource.empty() ||
          !snapshot->transientBufferAllocationByResource.empty(),
      180.0f, [&]() {
        ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Resource", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Physical", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableHeadersRow();
        for (uint32_t i = 0;
             i < snapshot->transientTextureAllocationByResource.size(); ++i) {
          const uint32_t physical =
              snapshot->transientTextureAllocationByResource[i];
          if (physical == UINT32_MAX) {
            continue;
          }
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("tex");
          ImGui::TableNextColumn();
          ImGui::Text("%u", i);
          ImGui::TableNextColumn();
          ImGui::Text("%u", physical);
        }
        for (uint32_t i = 0;
             i < snapshot->transientBufferAllocationByResource.size(); ++i) {
          const uint32_t physical =
              snapshot->transientBufferAllocationByResource[i];
          if (physical == UINT32_MAX) {
            continue;
          }
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("buf");
          ImGui::TableNextColumn();
          ImGui::Text("%u", i);
          ImGui::TableNextColumn();
          ImGui::Text("%u", physical);
        }
      });

  drawTelemetryTableSection(
      "Physical Allocations", "RenderGraphTelemetryPhysicalAllocations", 5,
      !snapshot->transientTexturePhysicalAllocations.empty() ||
          !snapshot->transientBufferPhysicalAllocations.empty(),
      200.0f, [&]() {
        ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Physical", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Representative",
                                ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableSetupColumn("Format/Usage",
                                ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableSetupColumn("Details", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (const auto &physical :
             snapshot->transientTexturePhysicalAllocations) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("tex");
          ImGui::TableNextColumn();
          ImGui::Text("%u", physical.allocationIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", physical.representativeResourceIndex);
          ImGui::TableNextColumn();
          ImGui::TextUnformatted(formatDisplayName(physical.desc.format));
          ImGui::TableNextColumn();
          ImGui::Text("%ux%ux%u layers=%u samples=%u mips=%u",
                      physical.desc.dimensions.width,
                      physical.desc.dimensions.height,
                      physical.desc.dimensions.depth, physical.desc.numLayers,
                      physical.desc.numSamples, physical.desc.numMipLevels);
        }
        for (const auto &physical :
             snapshot->transientBufferPhysicalAllocations) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("buf");
          ImGui::TableNextColumn();
          ImGui::Text("%u", physical.allocationIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", physical.representativeResourceIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", static_cast<uint32_t>(physical.desc.usage));
          ImGui::TableNextColumn();
          ImGui::Text("storage=%u size=%zu",
                      static_cast<uint32_t>(physical.desc.storage),
                      physical.desc.size);
        }
      });

  drawTelemetryTableSection(
      "Bindings", "RenderGraphTelemetryBindings", 5,
      !snapshot->unresolvedTextureBindings.empty() ||
          !snapshot->resolvedDependencyBuffers.empty() ||
          !snapshot->unresolvedDependencyBufferBindings.empty() ||
          !snapshot->unresolvedPreDispatchDependencyBufferBindings.empty() ||
          !snapshot->unresolvedDrawBufferBindings.empty(),
      220.0f, [&]() {
        ImGui::TableSetupColumn("Section", ImGuiTableColumnFlags_WidthFixed,
                                110.0f);
        ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthFixed,
                                60.0f);
        ImGui::TableSetupColumn("Item", ImGuiTableColumnFlags_WidthFixed,
                                80.0f);
        ImGui::TableSetupColumn("Target", ImGuiTableColumnFlags_WidthFixed,
                                110.0f);
        ImGui::TableSetupColumn("Resource", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (const auto &binding : snapshot->unresolvedTextureBindings) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("pass_tex");
          ImGui::TableNextColumn();
          ImGui::Text("%u", binding.orderedPassIndex);
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("-");
          ImGui::TableNextColumn();
          ImGui::TextUnformatted(passTextureBindingTargetName(binding.target));
          ImGui::TableNextColumn();
          ImGui::Text("tex[%u]", binding.textureResourceIndex);
        }
        for (uint32_t slot = 0;
             slot < snapshot->resolvedDependencyBuffers.size(); ++slot) {
          const BufferHandle handle = snapshot->resolvedDependencyBuffers[slot];
          if (!nuri::isValid(handle)) {
            continue;
          }
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("dep_slot");
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("-");
          ImGui::TableNextColumn();
          ImGui::Text("%u", slot);
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("resolved");
          ImGui::TableNextColumn();
          ImGui::Text("handle=(%u,%u)", handle.index, handle.generation);
        }
        for (const auto &binding :
             snapshot->unresolvedDependencyBufferBindings) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("dep_buf");
          ImGui::TableNextColumn();
          ImGui::Text("%u", binding.orderedPassIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", binding.dependencyBufferIndex);
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("dependency");
          ImGui::TableNextColumn();
          ImGui::Text("buf[%u]", binding.bufferResourceIndex);
        }
        for (const auto &binding :
             snapshot->unresolvedPreDispatchDependencyBufferBindings) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("pre_dep");
          ImGui::TableNextColumn();
          ImGui::Text("%u", binding.orderedPassIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u/%u", binding.preDispatchIndex,
                      binding.dependencyBufferIndex);
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("pre_dispatch");
          ImGui::TableNextColumn();
          ImGui::Text("buf[%u]", binding.bufferResourceIndex);
        }
        for (const auto &binding : snapshot->unresolvedDrawBufferBindings) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("draw_buf");
          ImGui::TableNextColumn();
          ImGui::Text("%u", binding.orderedPassIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", binding.drawIndex);
          ImGui::TableNextColumn();
          ImGui::TextUnformatted(drawBufferBindingTargetName(binding.target));
          ImGui::TableNextColumn();
          ImGui::Text("buf[%u]", binding.bufferResourceIndex);
        }
      });

  drawTelemetryTableSection(
      "Ranges", "RenderGraphTelemetryRanges", 4,
      !snapshot->dependencyBufferRangesByPass.empty() ||
          !snapshot->preDispatchRangesByPass.empty() ||
          !snapshot->preDispatchDependencyRanges.empty() ||
          !snapshot->drawRangesByPass.empty(),
      220.0f, [&]() {
        ImGui::TableSetupColumn("Section", ImGuiTableColumnFlags_WidthFixed,
                                110.0f);
        ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Offset", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableHeadersRow();
        for (uint32_t i = 0; i < snapshot->dependencyBufferRangesByPass.size();
             ++i) {
          const auto &range = snapshot->dependencyBufferRangesByPass[i];
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("dep_pass");
          ImGui::TableNextColumn();
          ImGui::Text("%u", i);
          ImGui::TableNextColumn();
          ImGui::Text("%u", range.offset);
          ImGui::TableNextColumn();
          ImGui::Text("%u", range.count);
        }
        for (uint32_t i = 0; i < snapshot->preDispatchRangesByPass.size();
             ++i) {
          const auto &range = snapshot->preDispatchRangesByPass[i];
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("pre_pass");
          ImGui::TableNextColumn();
          ImGui::Text("%u", i);
          ImGui::TableNextColumn();
          ImGui::Text("%u", range.offset);
          ImGui::TableNextColumn();
          ImGui::Text("%u", range.count);
        }
        for (uint32_t i = 0; i < snapshot->preDispatchDependencyRanges.size();
             ++i) {
          const auto &range = snapshot->preDispatchDependencyRanges[i];
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("pre_dep");
          ImGui::TableNextColumn();
          ImGui::Text("%u", i);
          ImGui::TableNextColumn();
          ImGui::Text("%u", range.offset);
          ImGui::TableNextColumn();
          ImGui::Text("%u", range.count);
        }
        for (uint32_t i = 0; i < snapshot->drawRangesByPass.size(); ++i) {
          const auto &range = snapshot->drawRangesByPass[i];
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("draw_pass");
          ImGui::TableNextColumn();
          ImGui::Text("%u", i);
          ImGui::TableNextColumn();
          ImGui::Text("%u", range.offset);
          ImGui::TableNextColumn();
          ImGui::Text("%u", range.count);
        }
      });
}

void setDockspaceWindowPlacement(const ImGuiViewport *viewport) {
  if (!viewport) {
    return;
  }
  ImGui::SetNextWindowPos(viewport->WorkPos);
  ImGui::SetNextWindowSize(viewport->WorkSize);
  ImGui::SetNextWindowViewport(viewport->ID);
}

void setLogWindowPlacementWithoutDock(const ImGuiViewport *viewport) {
  if (!viewport) {
    return;
  }
  const float height = std::max(180.0f, viewport->WorkSize.y * 0.25f);
  const ImVec2 position(viewport->WorkPos.x,
                        viewport->WorkPos.y + viewport->WorkSize.y - height);
  const ImVec2 size(viewport->WorkSize.x, height);
  ImGui::SetNextWindowPos(position, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(size, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowViewport(viewport->ID);
}

void drawFpsOverlay(const FPSCounter &fpsCounter, LinearGraph &fpsGraph,
                    LinearGraph &frametimeGraph,
                    const RenderFrameMetrics &frameMetrics,
                    const TelemetryOverlayUiState &telemetryState,
                    float overlayRightBoundaryX) {
  if (!telemetryState.overlayEnabled) {
    return;
  }
  if (const ImGuiViewport *viewport = ImGui::GetMainViewport()) {
    const float viewportRight = viewport->WorkPos.x + viewport->WorkSize.x;
    const float rightBoundary =
        overlayRightBoundaryX > 0.0f
            ? std::min(overlayRightBoundaryX, viewportRight) - 15.0f
            : viewportRight - 15.0f;
    ImGui::SetNextWindowPos({rightBoundary, viewport->WorkPos.y + 15.0f},
                            ImGuiCond_Always, {1.0f, 0.0f});
  }
  ImGui::SetNextWindowBgAlpha(0.30f);
  const float overlayHeight =
      telemetryState.showGraphs ? kMetricGraphWindowHeight : 140.0f;
  ImGui::SetNextWindowSize(ImVec2(kMetricGraphWindowWidth, overlayHeight),
                           ImGuiCond_Always);
  if (ImGui::Begin("##FPS", nullptr,
                   ImGuiWindowFlags_NoDecoration |
                       ImGuiWindowFlags_NoSavedSettings |
                       ImGuiWindowFlags_NoFocusOnAppearing |
                       ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove)) {
    const float fps = fpsCounter.getFPS();
    const float milliseconds = fps > 0.0f ? 1000.0f / fps : 0.0f;
    bool drewStats = false;
    if (telemetryState.showFpsMs) {
      ImGui::Text("FPS : %i", static_cast<int>(fps));
      ImGui::Text("Ms  : %.1f", milliseconds);
      drewStats = true;
    }
    if (telemetryState.showInstanceStats) {
      ImGui::Text("Inst: %u / %u", frameMetrics.opaque.visibleInstances,
                  frameMetrics.opaque.totalInstances);
      drewStats = true;
    }
    if (telemetryState.showDrawTessStats) {
      ImGui::Text("Draw: %u (Tess: %u)  Tess Inst: %u",
                  frameMetrics.opaque.instancedDraws,
                  frameMetrics.opaque.tessellatedDraws,
                  frameMetrics.opaque.tessellatedInstances);
      drewStats = true;
    }
    if (telemetryState.showIndirectStats) {
      ImGui::Text("Indirect: %u calls / %u cmds",
                  frameMetrics.opaque.indirectDrawCalls,
                  frameMetrics.opaque.indirectCommands);
      drewStats = true;
    }
    if (telemetryState.showDebugDrawStats) {
      ImGui::Text("Debug Draws: %u (Fallback: %u)",
                  frameMetrics.opaque.debugOverlayDraws,
                  frameMetrics.opaque.debugOverlayFallbackDraws);
      drewStats = true;
    }
    if (telemetryState.showPatchHeatmap) {
      ImGui::Text("Patch Heatmap: %u",
                  frameMetrics.opaque.debugPatchHeatmapDraws);
      drewStats = true;
    }
    if (telemetryState.showDispatchStats) {
      ImGui::Text("Dispatch: %u x%u", frameMetrics.opaque.computeDispatches,
                  frameMetrics.opaque.computeDispatchX);
      drewStats = true;
    }

    if (telemetryState.showGraphs) {
      if (drewStats) {
        ImGui::Separator();
      }
      const float availableGraphHeight = ImGui::GetContentRegionAvail().y;
      const float perGraphHeight = std::max(availableGraphHeight * 0.5f, 1.0f);
      const ImVec2 itemSpacing = ImGui::GetStyle().ItemSpacing;
      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,
                          ImVec2(itemSpacing.x, 0.0f));

      LinearGraphStyle graphStyle{
          .heightPixels = perGraphHeight,
          .lineColorRgba = IM_COL32(64, 224, 128, 255),
          .fillUnderLine = true,
      };
      fpsGraph.draw("FPS Graph##Metrics", "FPS", graphStyle);

      graphStyle.lineColorRgba = IM_COL32(255, 160, 64, 255);
      frametimeGraph.draw("Frametime Graph##Metrics", "Frametime (ms)",
                          graphStyle);
      ImGui::PopStyleVar();
    }
  }
  ImGui::End();
}

ImGuiWindowFlags dockspaceWindowFlags() {
  return ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
         ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
         ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground |
         ImGuiWindowFlags_NoSavedSettings;
}

#ifdef IMGUI_HAS_DOCK
struct DockLayoutState {
  ImGuiID logDockId = 0;
  ImGuiID hierarchyDockId = 0;
  ImGuiID inspectorDockId = 0;
  bool built = false;

  void ensureLayout(ImGuiID dockspaceId, const ImGuiViewport *viewport) {
    if (built || dockspaceId == 0 || !viewport) {
      return;
    }

    const auto dockNodeFlags = static_cast<ImGuiDockNodeFlags>(
        static_cast<int>(ImGuiDockNodeFlags_DockSpace) |
        static_cast<int>(ImGuiDockNodeFlags_PassthruCentralNode));

    ImGui::DockBuilderRemoveNode(dockspaceId);
    ImGui::DockBuilderAddNode(dockspaceId, dockNodeFlags);
    ImGui::DockBuilderSetNodeSize(dockspaceId, viewport->WorkSize);

    ImGuiID dockMain = dockspaceId;
    ImGuiID dockBottom = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Down,
                                                     0.25f, nullptr, &dockMain);
    ImGuiID dockLeft = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Left,
                                                   0.22f, nullptr, &dockMain);
    ImGuiID dockRight = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Right,
                                                    0.28f, nullptr, &dockMain);

    logDockId = dockBottom;
    hierarchyDockId = dockLeft;
    inspectorDockId = dockRight;
    ImGui::DockBuilderDockWindow(kLogWindowName, logDockId);
    ImGui::DockBuilderDockWindow(kHierarchyWindowName, hierarchyDockId);
    ImGui::DockBuilderDockWindow(kInspectorWindowName, inspectorDockId);
    ImGui::DockBuilderFinish(dockspaceId);
    built = true;
  }
};
#endif

#ifdef IMGUI_HAS_DOCK
using MaybeDockLayoutState = DockLayoutState;
#else
struct MaybeDockLayoutState {};
#endif

} // namespace

struct ImGuiEditor::Impl {
  Impl(Window &windowIn, GPUDevice &gpuIn, const EditorServices &services)
      : window(windowIn), gpu(gpuIn), scene(services.scene),
        resources(services.resources), renderPipeline(services.renderPipeline),
        selectionState(services.selectionState != nullptr
                           ? services.selectionState
                           : &localSelectionState),
        textSystem(services.textSystem), cameraSystem(services.cameraSystem),
        bakery(services.bakery),
        renderGraphTelemetry(services.renderGraphTelemetry) {}

  ~Impl() { resetSceneUiState(); }

  RenderableInspectorState *findRenderableInspectorState(RenderableId id) {
    for (RenderableInspectorState &state : renderableInspectorStates) {
      if (state.renderableId == id) {
        return &state;
      }
    }
    return nullptr;
  }

  RenderableInspectorState &ensureRenderableInspectorState(RenderableId id) {
    if (RenderableInspectorState *state = findRenderableInspectorState(id);
        state != nullptr) {
      return *state;
    }
    renderableInspectorStates.push_back(RenderableInspectorState{});
    RenderableInspectorState &state = renderableInspectorStates.back();
    state.renderableId = id;
    state.selectedTextureIndex = 0;
    return state;
  }

  void releaseOwnedOverride(RenderableInspectorState &state) {
    if (resources != nullptr && isValid(state.ownedOverride)) {
      resources->release(state.ownedOverride);
    }
    state.ownedOverride = kInvalidMaterialRef;
  }

  void clearRenderableOverride(RenderableInspectorState &state) {
    if (scene != nullptr) {
      (void)scene->graph().clearRenderableMaterialOverride(state.renderableId);
    }
    releaseOwnedOverride(state);
  }

  void resetSceneUiState() {
    for (RenderableInspectorState &state : renderableInspectorStates) {
      releaseOwnedOverride(state);
    }
    renderableInspectorStates.clear();
    invalidateLightEditorDraft(lightEditorDraft);
    lastObservedSelectionNode = kInvalidNodeId;
    pendingRevealSelection = false;
    suppressRevealForNextSelectionChange = false;
    hierarchyTopologyCacheValid = false;
    hierarchyStatsCacheValid = false;
    cachedSelectedPathLeaf = kInvalidNodeId;
    cachedLightCount = 0u;
    cachedRenderableCount = 0u;
    hierarchyNodeTopology.clear();
    hierarchyNodeStats.clear();
    selectedPathNodeFlags.clear();
    selectedPathChildIndices.clear();
    hierarchyVisibleRows.clear();
    hierarchyNodeOpenFlags.clear();
    hierarchyOpenBatchKeys.clear();
    hierarchySelectedRowIndex = -1;
    hierarchySceneRootOpen = true;
    if (selectionState != nullptr) {
      selectionState->clear();
    }
  }

  void validateSelectionState() {
    if (scene == nullptr || selectionState == nullptr) {
      return;
    }
    if (!isValid(selectionState->node)) {
      selectionState->clear();
      return;
    }
    if (!selectionNodeStillValid(scene->graph(), *selectionState)) {
      selectionState->clear();
      return;
    }
    if (selectionState->kind == SceneSelectionKind::NodeRenderable) {
      const Renderable *renderable =
          scene->renderable(selectionState->renderableIndex);
      if (renderable == nullptr ||
          renderable->id != selectionState->renderableId ||
          renderable->node != selectionState->node) {
        selectionState->kind = SceneSelectionKind::Node;
        selectionState->renderableId = kInvalidRenderableId;
        selectionState->renderableIndex = 0u;
      }
    } else if (selectionState->kind == SceneSelectionKind::Light) {
      LightDesc light{};
      NodeId lightNode = kInvalidNodeId;
      if (!scene->graph().getLightDesc(selectionState->lightId, light) ||
          !scene->graph().getLightNode(selectionState->lightId, lightNode) ||
          lightNode != selectionState->node) {
        selectionState->kind = SceneSelectionKind::Node;
        selectionState->lightId = kInvalidLightId;
      }
    }
  }

  void syncSelectionWindows() {
    if (selectionState == nullptr) {
      lastObservedSelectionNode = kInvalidNodeId;
      pendingRevealSelection = false;
      suppressRevealForNextSelectionChange = false;
      return;
    }

    if (selectionState->node != lastObservedSelectionNode) {
      if (isValid(selectionState->node)) {
        showHierarchyWindow = true;
        showInspectorWindow = true;
        pendingRevealSelection = !suppressRevealForNextSelectionChange;
      }
      suppressRevealForNextSelectionChange = false;
      lastObservedSelectionNode = selectionState->node;
    }
  }

  void rebuildHierarchyFrameCache() {
    if (scene == nullptr) {
      hierarchyTopologyCacheValid = false;
      hierarchyStatsCacheValid = false;
      cachedSelectedPathLeaf = kInvalidNodeId;
      hierarchyNodeTopology.clear();
      hierarchyNodeStats.clear();
      selectedPathNodeFlags.clear();
      selectedPathChildIndices.clear();
      hierarchyVisibleRows.clear();
      hierarchyNodeOpenFlags.clear();
      hierarchyOpenBatchKeys.clear();
      hierarchySelectedRowIndex = -1;
      return;
    }

    uint32_t currentLightCount = 0u;
    uint32_t maxLightNodeValue = 0u;
    scene->graph().forEachLightId([&](LightId lightId) {
      NodeId node = kInvalidNodeId;
      if (scene->graph().getLightNode(lightId, node) && isValid(node)) {
        ++currentLightCount;
        maxLightNodeValue =
            std::max(maxLightNodeValue, static_cast<uint32_t>(indexOf(node)));
      }
    });

    const uint32_t currentRenderableCount =
        static_cast<uint32_t>(scene->renderables().size());
    if (hierarchyStatsCacheValid &&
        (currentRenderableCount != cachedRenderableCount ||
         currentLightCount != cachedLightCount)) {
      hierarchyTopologyCacheValid = false;
    }
    if (!hierarchyTopologyCacheValid) {
      NURI_PROFILER_ZONE("ImGuiEditor::HierarchyWindow::BuildTopologyCache",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      hierarchyNodeTopology.clear();
      std::vector<NodeId> stack;
      stack.push_back(scene->graph().rootNode());
      while (!stack.empty()) {
        const NodeId node = stack.back();
        stack.pop_back();
        if (!isValid(node)) {
          continue;
        }

        const size_t nodeSlot = hierarchyNodeSlot(node);
        if (nodeSlot >= hierarchyNodeTopology.size()) {
          hierarchyNodeTopology.resize(nodeSlot + 1u);
        }
        HierarchyNodeTopology &entry = hierarchyNodeTopology[nodeSlot];
        entry.labelName = nodeDisplayName(scene->graph(), node);
        entry.children = collectChildNodes(scene->graph(), node);
        for (auto it = entry.children.rbegin(); it != entry.children.rend();
             ++it) {
          stack.push_back(*it);
        }
      }
      hierarchyTopologyCacheValid = true;
      if (hierarchyNodeOpenFlags.size() < hierarchyNodeTopology.size()) {
        hierarchyNodeOpenFlags.resize(hierarchyNodeTopology.size(), 0u);
      }
      NURI_PROFILER_ZONE_END();
    }
    if (!hierarchyStatsCacheValid ||
        currentRenderableCount != cachedRenderableCount ||
        currentLightCount != cachedLightCount) {
      NURI_PROFILER_ZONE("ImGuiEditor::HierarchyWindow::BuildNodeStats",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      uint32_t maxNodeIndex = maxLightNodeValue;
      for (const Renderable &renderable : scene->renderables()) {
        if (!isValid(renderable.node)) {
          continue;
        }
        maxNodeIndex = std::max(
            maxNodeIndex, static_cast<uint32_t>(indexOf(renderable.node)));
      }

      hierarchyNodeStats.assign(static_cast<size_t>(maxNodeIndex) + 1u,
                                HierarchyNodeStats{});
      for (const Renderable &renderable : scene->renderables()) {
        if (!isValid(renderable.node)) {
          continue;
        }
        ++hierarchyNodeStats[hierarchyNodeSlot(renderable.node)]
              .renderableCount;
      }
      scene->graph().forEachLightId([&](LightId lightId) {
        NodeId node = kInvalidNodeId;
        if (scene->graph().getLightNode(lightId, node) && isValid(node)) {
          ++hierarchyNodeStats[hierarchyNodeSlot(node)].lightCount;
        }
      });

      cachedRenderableCount = currentRenderableCount;
      cachedLightCount = currentLightCount;
      hierarchyStatsCacheValid = true;
      NURI_PROFILER_ZONE_END();
    }

    const NodeId selectedLeaf =
        selectionState != nullptr ? selectionState->node : kInvalidNodeId;
    if (selectedLeaf != cachedSelectedPathLeaf) {
      NURI_PROFILER_ZONE("ImGuiEditor::HierarchyWindow::BuildSelectedPath",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      selectedPathNodeFlags.assign(hierarchyNodeStats.size(), 0u);
      selectedPathChildIndices.assign(hierarchyNodeStats.size(), -1);
      if (isValid(selectedLeaf)) {
        NodeId current = selectedLeaf;
        while (isValid(current)) {
          const size_t currentSlot = hierarchyNodeSlot(current);
          if (currentSlot >= selectedPathNodeFlags.size()) {
            selectedPathNodeFlags.resize(currentSlot + 1u, 0u);
            selectedPathChildIndices.resize(currentSlot + 1u, -1);
          }
          selectedPathNodeFlags[currentSlot] = 1u;
          NodeId parent = kInvalidNodeId;
          if (!scene->graph().getNodeParent(current, parent)) {
            break;
          }
          if (isValid(parent)) {
            const size_t parentSlot = hierarchyNodeSlot(parent);
            if (parentSlot >= selectedPathChildIndices.size()) {
              selectedPathChildIndices.resize(parentSlot + 1u, -1);
            }
            const std::vector<NodeId> &siblings =
                hierarchyTopology(parent).children;
            const auto it =
                std::find(siblings.begin(), siblings.end(), current);
            if (it != siblings.end()) {
              selectedPathChildIndices[parentSlot] =
                  static_cast<int32_t>(std::distance(siblings.begin(), it));
            }
          }
          current = parent;
        }
      }
      cachedSelectedPathLeaf = selectedLeaf;
      NURI_PROFILER_ZONE_END();
    }
  }

  [[nodiscard]] const HierarchyNodeStats &hierarchyStats(NodeId node) const {
    static const HierarchyNodeStats kEmptyStats{};
    const size_t nodeSlot = hierarchyNodeSlot(node);
    return isValid(node) && nodeSlot < hierarchyNodeStats.size()
               ? hierarchyNodeStats[nodeSlot]
               : kEmptyStats;
  }

  [[nodiscard]] const HierarchyNodeTopology &
  hierarchyTopology(NodeId node) const {
    static const HierarchyNodeTopology kEmptyTopology{};
    const size_t nodeSlot = hierarchyNodeSlot(node);
    return isValid(node) && nodeSlot < hierarchyNodeTopology.size()
               ? hierarchyNodeTopology[nodeSlot]
               : kEmptyTopology;
  }

  [[nodiscard]] int32_t selectedChildIndexForParent(NodeId node) const {
    const size_t nodeSlot = hierarchyNodeSlot(node);
    return isValid(node) && nodeSlot < selectedPathChildIndices.size()
               ? selectedPathChildIndices[nodeSlot]
               : -1;
  }

  [[nodiscard]] bool nodeIsOnSelectedPath(NodeId node) const {
    const size_t nodeSlot = hierarchyNodeSlot(node);
    return isValid(node) && nodeSlot < selectedPathNodeFlags.size() &&
           selectedPathNodeFlags[nodeSlot] != 0u;
  }

  [[nodiscard]] bool childRangeContainsSelectedChild(NodeId parentNode,
                                                     size_t beginIndex,
                                                     size_t endIndex) const {
    const int32_t selectedChildIndex = selectedChildIndexForParent(parentNode);
    return selectedChildIndex >= 0 &&
           beginIndex <= static_cast<size_t>(selectedChildIndex) &&
           static_cast<size_t>(selectedChildIndex) < endIndex;
  }

  [[nodiscard]] static size_t hierarchyBatchSpan(size_t count) {
    size_t span = kHierarchyBatchSize;
    while (((count + span - 1u) / span) > kHierarchyBatchSize) {
      span *= kHierarchyBatchSize;
    }
    return span;
  }

  [[nodiscard]] static uint64_t
  hierarchyBatchKey(NodeId parentNode, size_t beginIndex, size_t endIndex) {
    uint64_t key = static_cast<uint64_t>(parentNode.value);
    key ^= 0x9e3779b97f4a7c15ull + (key << 6u) + (key >> 2u) +
           static_cast<uint64_t>(beginIndex);
    key ^= 0x9e3779b97f4a7c15ull + (key << 6u) + (key >> 2u) +
           static_cast<uint64_t>(endIndex);
    return key;
  }

  void drawNodeTransformEditor(NodeId node) {
    if (scene == nullptr || !isValid(node)) {
      ImGui::TextUnformatted("No node selected.");
      return;
    }

    SceneGraph &graph = scene->graph();
    (void)graph.syncWorldTransforms();
    glm::mat4 localMatrix(1.0f);
    glm::mat4 worldMatrix(1.0f);
    if (!graph.getNodeLocalTransform(node, localMatrix)) {
      ImGui::TextUnformatted("Selected node is no longer valid.");
      return;
    }
    (void)graph.getCachedNodeWorldTransform(node, worldMatrix);

    float translation[3]{};
    float rotation[3]{};
    float scale[3]{};
    ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(localMatrix),
                                          translation, rotation, scale);
    bool changed = false;
    changed |= ImGui::InputFloat3("Translation", translation, "%.3f");
    changed |= ImGui::InputFloat3("Rotation", rotation, "%.3f");
    changed |= ImGui::InputFloat3("Scale", scale, "%.3f");
    if (changed) {
      glm::mat4 recomposed(1.0f);
      ImGuizmo::RecomposeMatrixFromComponents(translation, rotation, scale,
                                              glm::value_ptr(recomposed));
      (void)graph.setNodeLocalTransform(node, recomposed);
      localMatrix = recomposed;
      (void)graph.syncWorldTransforms();
      (void)graph.getCachedNodeWorldTransform(node, worldMatrix);
    }

    const glm::vec3 worldPosition = glm::vec3(worldMatrix[3]);
    ImGui::SeparatorText("World");
    ImGui::Text("Position: %.3f %.3f %.3f", worldPosition.x, worldPosition.y,
                worldPosition.z);
    ImGui::Text("Node Handle: %u", indexOf(node));
  }

  bool applyMaterialOverride(RenderableInspectorState &state,
                             const MaterialRecord &sourceRecord,
                             const MaterialDesc &desc,
                             const MaterialRequest::TextureRefs &textureRefs) {
    if (resources == nullptr || scene == nullptr) {
      return false;
    }

    auto acquireResult = resources->acquireMaterial(MaterialRequest{
        .desc = desc,
        .textureRefs = textureRefs,
        .debugName = !sourceRecord.debugName.empty()
                         ? std::string(sourceRecord.debugName) + " (Override)"
                         : std::string("Editor Override"),
        .sourceIdentity = "editor_override/renderable_" +
                          std::to_string(state.renderableId.value),
    });
    if (acquireResult.hasError()) {
      NURI_LOG_WARNING("ImGuiEditor: failed to acquire override material: %s",
                       acquireResult.error().c_str());
      return false;
    }

    const MaterialRef newRef = acquireResult.value();
    if (!scene->graph().setRenderableMaterialOverride(state.renderableId,
                                                      newRef)) {
      resources->release(newRef);
      NURI_LOG_WARNING("ImGuiEditor: failed to attach override material to "
                       "renderable %u",
                       state.renderableId.value);
      return false;
    }

    releaseOwnedOverride(state);
    state.ownedOverride = newRef;
    return true;
  }

  void drawMaterialViewer(const MaterialRecord &record,
                          RenderableInspectorState &state, bool editable) {
    MaterialDesc editedDesc = record.desc;
    const MaterialRequest::TextureRefs textureRefs = record.textureRefs;

    if (!editable) {
      ImGui::BeginDisabled();
    }

    bool changed = false;
    changed |= ImGui::ColorEdit4("Base Color",
                                 glm::value_ptr(editedDesc.baseColorFactor));
    changed |= ImGui::ColorEdit3("Emissive",
                                 glm::value_ptr(editedDesc.emissiveFactor));
    changed |= ImGui::SliderFloat(
        "Emissive Strength", &editedDesc.emissiveStrength, 0.0f, 32.0f, "%.3f");
    changed |= ImGui::SliderFloat("Metallic", &editedDesc.metallicFactor, 0.0f,
                                  1.0f, "%.3f");
    changed |= ImGui::SliderFloat("Roughness", &editedDesc.roughnessFactor,
                                  0.0f, 1.0f, "%.3f");
    constexpr const char *kAlphaModes[] = {"Opaque", "Mask", "Blend"};
    int alphaMode = static_cast<int>(editedDesc.alphaMode);
    if (ImGui::Combo("Alpha Mode", &alphaMode, kAlphaModes,
                     IM_ARRAYSIZE(kAlphaModes))) {
      editedDesc.alphaMode = static_cast<MaterialAlphaMode>(std::clamp(
          alphaMode, 0, static_cast<int>(IM_ARRAYSIZE(kAlphaModes)) - 1));
      changed = true;
    }
    changed |= ImGui::SliderFloat("Alpha Cutoff", &editedDesc.alphaCutoff, 0.0f,
                                  1.0f, "%.3f");
    changed |= ImGui::Checkbox("Double Sided", &editedDesc.doubleSided);

    if (!editable) {
      ImGui::EndDisabled();
    }

    if (editable && changed) {
      applyMaterialOverride(state, record, editedDesc, textureRefs);
    }

    ImGui::SeparatorText("Textures");
    const std::vector<MaterialTextureEntry> textures =
        buildMaterialTextureEntries(record);
    if (textures.empty()) {
      ImGui::TextUnformatted("No previewable textures.");
      return;
    }
    state.selectedTextureIndex = std::clamp(
        state.selectedTextureIndex, 0, static_cast<int>(textures.size()) - 1);
    if (ImGui::BeginListBox("Slots")) {
      for (int index = 0; index < static_cast<int>(textures.size()); ++index) {
        const bool selected = state.selectedTextureIndex == index;
        if (ImGui::Selectable(textures[index].label, selected)) {
          state.selectedTextureIndex = index;
        }
        if (selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndListBox();
    }

    const MaterialTextureEntry &textureEntry =
        textures[static_cast<size_t>(state.selectedTextureIndex)];
    const TextureRecord *textureRecord =
        resources != nullptr ? resources->tryGet(textureEntry.ref) : nullptr;
    if (textureRecord == nullptr) {
      ImGui::TextUnformatted("Selected texture is unavailable.");
      return;
    }
    ImGui::Text("Preview: %s", textureEntry.label);
    if (textureRecord->bindlessIndex != kInvalidTextureBindlessIndex) {
      ImGui::Image(toImTextureId(textureRecord->bindlessIndex),
                   ImVec2(192.0f, 192.0f));
    }
    if (!textureRecord->debugName.empty()) {
      ImGui::Text("Debug: %s", textureRecord->debugName.c_str());
    }
    if (!textureRecord->canonicalPath.empty()) {
      ImGui::TextWrapped("Path: %s", textureRecord->canonicalPath.c_str());
    }
    ImGui::Text("Size: %ux%u", textureRecord->dimensions.width,
                textureRecord->dimensions.height);
    ImGui::Text("Format: %s", formatDisplayName(textureRecord->format));
    ImGui::Text("Mip Levels: %u", textureRecord->numMipLevels);
  }

  void drawRenderableInspector(const Renderable &renderable) {
    if (resources == nullptr) {
      ImGui::TextUnformatted("Resource manager unavailable.");
      return;
    }
    RenderableInspectorState &state =
        ensureRenderableInspectorState(renderable.id);
    std::vector<MaterialSourceEntry> sourceEntries =
        buildMaterialSourceEntries(renderable, resources);
    state.baselineSlotIndex = std::clamp(
        state.baselineSlotIndex, 0, static_cast<int>(sourceEntries.size()) - 1);

    const bool hasOverride = isValid(renderable.materialOverride);
    const MaterialRef displayedMaterial =
        hasOverride
            ? renderable.materialOverride
            : sourceEntries[static_cast<size_t>(state.baselineSlotIndex)].ref;
    const MaterialRecord *displayedRecord =
        resources->tryGet(displayedMaterial);

    ImGui::SeparatorText("Renderable");
    ImGui::Text("Renderable Index: %u", selectionState->renderableIndex);
    ImGui::Text("Renderable Handle: %u", renderable.id.value);

    if (const ModelRecord *modelRecord = resources->tryGet(renderable.model);
        modelRecord != nullptr) {
      if (!modelRecord->canonicalPath.empty()) {
        ImGui::TextWrapped("Model: %s", modelRecord->canonicalPath.c_str());
      }
    }

    ImGui::SeparatorText("Material");
    if (!hasOverride && sourceEntries.size() > 1u) {
      std::vector<const char *> labels;
      labels.reserve(sourceEntries.size());
      for (const MaterialSourceEntry &entry : sourceEntries) {
        labels.push_back(entry.label.c_str());
      }
      ImGui::Combo("Baseline Slot", &state.baselineSlotIndex, labels.data(),
                   static_cast<int>(labels.size()));
      for (const MaterialSourceEntry &entry : sourceEntries) {
        ImGui::BulletText("%s", entry.label.c_str());
      }
      ImGui::TextUnformatted(
          "Uniform override will replace all source-material slots.");
    }

    bool overrideEnabled = hasOverride;
    if (ImGui::Checkbox("Uniform Material Override", &overrideEnabled)) {
      if (overrideEnabled) {
        const MaterialRecord *sourceRecord = displayedRecord;
        if (sourceRecord != nullptr) {
          (void)applyMaterialOverride(state, *sourceRecord, sourceRecord->desc,
                                      sourceRecord->textureRefs);
        }
      } else {
        clearRenderableOverride(state);
      }
    }

    if (displayedRecord == nullptr) {
      ImGui::TextUnformatted("Material record unavailable.");
      return;
    }
    if (!displayedRecord->debugName.empty()) {
      ImGui::Text("Debug Name: %s", displayedRecord->debugName.c_str());
    }
    drawMaterialViewer(*displayedRecord, state, overrideEnabled);
  }

  void drawLightInspector(LightId lightId) {
    if (scene == nullptr) {
      return;
    }
    LightDesc light{};
    if (!scene->graph().getLightDesc(lightId, light)) {
      ImGui::TextUnformatted("Selected light is no longer valid.");
      return;
    }
    ImGui::SeparatorText("Light");
    ImGui::Text("Type: %s", lightTypeName(light.type));
    ImGui::Text("Light Slot: %u", indexOf(lightId));
    drawLightEditor(scene->graph(), lightId, light, lightEditorDraft);
  }

  void drawInspectorWindow() {
    if (!ImGui::Begin(kInspectorWindowName, &showInspectorWindow)) {
      inspectorWindowVisible = true;
      inspectorWindowMinX = ImGui::GetWindowPos().x;
      ImGui::End();
      return;
    }
    inspectorWindowVisible = true;
    inspectorWindowMinX = ImGui::GetWindowPos().x;
    if (scene == nullptr || selectionState == nullptr ||
        !isValid(selectionState->node)) {
      ImGui::TextUnformatted("No scene selection.");
      ImGui::End();
      return;
    }

    ImGui::TextUnformatted(
        nodeDisplayName(scene->graph(), selectionState->node).c_str());
    ImGui::Separator();
    drawNodeTransformEditor(selectionState->node);

    if (selectionState->kind == SceneSelectionKind::NodeRenderable) {
      if (const Renderable *renderable =
              scene->renderable(selectionState->renderableIndex);
          renderable != nullptr &&
          renderable->id == selectionState->renderableId) {
        drawRenderableInspector(*renderable);
      }
    } else if (selectionState->kind == SceneSelectionKind::Light) {
      drawLightInspector(selectionState->lightId);
    } else {
      uint32_t renderableCount = 0u;
      scene->graph().forEachRenderableOnNode(
          selectionState->node, [&](RenderableId) { ++renderableCount; });
      uint32_t lightCount = 0u;
      scene->graph().forEachLightOnNode(selectionState->node,
                                        [&](LightId) { ++lightCount; });
      ImGui::SeparatorText("Components");
      ImGui::Text("Renderables: %u", renderableCount);
      ImGui::Text("Lights: %u", lightCount);
    }

    ImGui::End();
  }

  void drawScenesSection() {
    ImGui::SeparatorText("Scenes");
    if (sceneSelectionState.nameViews.empty()) {
      ImGui::TextUnformatted("No scenes available.");
      return;
    }

    int selectedIndex = sceneSelectionState.selectedIndex;
    if (ImGui::Combo("Scene", &selectedIndex,
                     sceneSelectionState.nameViews.data(),
                     static_cast<int>(sceneSelectionState.nameViews.size())) &&
        selectedIndex != sceneSelectionState.selectedIndex) {
      sceneSelectionState.selectedIndex = selectedIndex;
      sceneSelectionState.pendingSelectionRequest =
          sceneSelectionState.ids[static_cast<size_t>(selectedIndex)];
    }
    if (!sceneSelectionState.hotkeyHint.empty()) {
      ImGui::TextUnformatted(sceneSelectionState.hotkeyHint.c_str());
    }
  }

  [[nodiscard]] bool isHierarchyNodeOpen(NodeId node) const {
    const size_t nodeSlot = hierarchyNodeSlot(node);
    return isValid(node) && nodeSlot < hierarchyNodeOpenFlags.size() &&
           hierarchyNodeOpenFlags[nodeSlot] != 0u;
  }

  void setHierarchyNodeOpen(NodeId node, bool open) {
    if (!isValid(node)) {
      return;
    }
    const size_t nodeSlot = hierarchyNodeSlot(node);
    if (nodeSlot >= hierarchyNodeOpenFlags.size()) {
      hierarchyNodeOpenFlags.resize(nodeSlot + 1u, 0u);
    }
    hierarchyNodeOpenFlags[nodeSlot] = open ? 1u : 0u;
  }

  [[nodiscard]] bool isHierarchyBatchOpen(NodeId parentNode, size_t beginIndex,
                                          size_t endIndex) const {
    return hierarchyOpenBatchKeys.contains(
        hierarchyBatchKey(parentNode, beginIndex, endIndex));
  }

  void setHierarchyBatchOpen(NodeId parentNode, size_t beginIndex,
                             size_t endIndex, bool open) {
    const uint64_t key = hierarchyBatchKey(parentNode, beginIndex, endIndex);
    if (open) {
      hierarchyOpenBatchKeys.insert(key);
    } else {
      hierarchyOpenBatchKeys.erase(key);
    }
  }

  void revealHierarchyBatchPath(NodeId parentNode, size_t childIndex) {
    const std::vector<NodeId> &children =
        hierarchyTopology(parentNode).children;
    if (childIndex >= children.size()) {
      return;
    }

    size_t beginIndex = 0u;
    size_t endIndex = children.size();
    while (endIndex - beginIndex > kHierarchyBatchSize) {
      const size_t childSpan = hierarchyBatchSpan(endIndex - beginIndex);
      const size_t relativeIndex = childIndex - beginIndex;
      const size_t subBegin =
          beginIndex + (relativeIndex / childSpan) * childSpan;
      const size_t subEnd = std::min(subBegin + childSpan, endIndex);
      setHierarchyBatchOpen(parentNode, subBegin, subEnd, true);
      beginIndex = subBegin;
      endIndex = subEnd;
    }
  }

  void applyPendingHierarchyReveal() {
    if (!pendingRevealSelection || selectionState == nullptr ||
        !isValid(selectionState->node)) {
      return;
    }

    hierarchySceneRootOpen = true;
    NodeId current = selectionState->node;
    while (isValid(current)) {
      if (!hierarchyTopology(current).children.empty()) {
        setHierarchyNodeOpen(current, true);
      }
      NodeId parent = kInvalidNodeId;
      if (!scene->graph().getNodeParent(current, parent)) {
        break;
      }
      if (isValid(parent)) {
        const int32_t childIndex = selectedChildIndexForParent(parent);
        if (childIndex >= 0) {
          revealHierarchyBatchPath(parent, static_cast<size_t>(childIndex));
          setHierarchyNodeOpen(parent, true);
        }
      }
      current = parent;
    }
  }

  void appendHierarchyVisibleRowsForChildren(
      NodeId parentNode, const std::vector<NodeId> &children, int depth,
      size_t beginIndex, size_t endIndex) {
    const size_t clampedEnd = std::min(endIndex, children.size());
    if (beginIndex >= clampedEnd) {
      return;
    }

    const size_t rangeCount = clampedEnd - beginIndex;
    if (rangeCount <= kHierarchyBatchSize) {
      for (size_t index = beginIndex; index < clampedEnd; ++index) {
        const NodeId child = children[index];
        hierarchyVisibleRows.push_back(HierarchyVisibleRow{
            .kind = HierarchyRowKind::Node,
            .depth = depth,
            .node = child,
        });
        if (selectionState != nullptr && selectionState->node == child) {
          hierarchySelectedRowIndex =
              static_cast<int>(hierarchyVisibleRows.size()) - 1;
        }
        if (isHierarchyNodeOpen(child)) {
          const HierarchyNodeTopology &childTopology = hierarchyTopology(child);
          appendHierarchyVisibleRowsForChildren(child, childTopology.children,
                                                depth + 1u, 0u,
                                                childTopology.children.size());
        }
      }
      return;
    }

    const size_t childSpan = hierarchyBatchSpan(rangeCount);
    for (size_t index = beginIndex; index < clampedEnd; index += childSpan) {
      const size_t subEnd = std::min(index + childSpan, clampedEnd);
      hierarchyVisibleRows.push_back(HierarchyVisibleRow{
          .kind = HierarchyRowKind::Batch,
          .depth = depth,
          .node = parentNode,
          .beginIndex = index,
          .endIndex = subEnd,
      });
      if (isHierarchyBatchOpen(parentNode, index, subEnd)) {
        appendHierarchyVisibleRowsForChildren(parentNode, children, depth + 1u,
                                              index, subEnd);
      }
    }
  }

  void rebuildHierarchyVisibleRows() {
    hierarchyVisibleRows.clear();
    hierarchySelectedRowIndex = -1;
    if (scene == nullptr) {
      return;
    }

    hierarchyVisibleRows.push_back(
        HierarchyVisibleRow{.kind = HierarchyRowKind::SceneRoot, .depth = 0});
    if (!hierarchySceneRootOpen) {
      return;
    }

    const std::vector<NodeId> &children =
        hierarchyTopology(scene->graph().rootNode()).children;
    appendHierarchyVisibleRowsForChildren(scene->graph().rootNode(), children,
                                          1, 0u, children.size());
  }

  void drawHierarchyRow(const HierarchyVisibleRow &row,
                        std::string_view sceneLabel) {
    if (scene == nullptr || selectionState == nullptr) {
      return;
    }

    const float indentWidth =
        static_cast<float>(row.depth) * ImGui::GetTreeNodeToLabelSpacing();
    if (indentWidth > 0.0f) {
      ImGui::Indent(indentWidth);
    }

    if (row.kind == HierarchyRowKind::SceneRoot) {
      ImGui::SetNextItemOpen(hierarchySceneRootOpen, ImGuiCond_Always);
      const bool open =
          ImGui::TreeNodeEx("active_scene_root",
                            ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                ImGuiTreeNodeFlags_SpanAvailWidth |
                                ImGuiTreeNodeFlags_OpenOnArrow |
                                ImGuiTreeNodeFlags_OpenOnDoubleClick,
                            "%s", std::string(sceneLabel).c_str());
      hierarchySceneRootOpen = open;
    } else if (row.kind == HierarchyRowKind::Batch) {
      const bool isOpen =
          isHierarchyBatchOpen(row.node, row.beginIndex, row.endIndex);
      const bool containsSelected = childRangeContainsSelectedChild(
          row.node, row.beginIndex, row.endIndex);
      ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                 ImGuiTreeNodeFlags_SpanAvailWidth |
                                 ImGuiTreeNodeFlags_OpenOnArrow |
                                 ImGuiTreeNodeFlags_OpenOnDoubleClick;
      if (containsSelected) {
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
      }
      const std::string label = "More " + std::to_string(row.beginIndex + 1u) +
                                "-" + std::to_string(row.endIndex) + " (" +
                                std::to_string(row.endIndex - row.beginIndex) +
                                ")";
      ImGui::PushID(static_cast<int>(row.node.value));
      ImGui::PushID(static_cast<int>(row.beginIndex));
      ImGui::PushID(static_cast<int>(row.endIndex));
      ImGui::SetNextItemOpen(isOpen, ImGuiCond_Always);
      const bool open = ImGui::TreeNodeEx("batch", flags, "%s", label.c_str());
      if (open != isOpen) {
        setHierarchyBatchOpen(row.node, row.beginIndex, row.endIndex, open);
      }
      ImGui::PopID();
      ImGui::PopID();
      ImGui::PopID();
    } else {
      const HierarchyNodeTopology &topology = hierarchyTopology(row.node);
      const HierarchyNodeStats &stats = hierarchyStats(row.node);
      const bool hasChildren = !topology.children.empty();
      const bool isOpen = hasChildren && isHierarchyNodeOpen(row.node);
      ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                 ImGuiTreeNodeFlags_SpanAvailWidth |
                                 ImGuiTreeNodeFlags_OpenOnArrow |
                                 ImGuiTreeNodeFlags_OpenOnDoubleClick;
      if (!hasChildren) {
        flags |= ImGuiTreeNodeFlags_Leaf;
      }
      if (selectionState->node == row.node) {
        flags |= ImGuiTreeNodeFlags_Selected;
      }
      std::string label = topology.labelName +
                          "  R:" + std::to_string(stats.renderableCount) +
                          "  L:" + std::to_string(stats.lightCount);
      ImGui::SetNextItemOpen(isOpen, ImGuiCond_Always);
      ImGui::PushID(static_cast<int>(row.node.value));
      const bool open = ImGui::TreeNodeEx("node", flags, "%s", label.c_str());
      if (hasChildren && open != isOpen) {
        setHierarchyNodeOpen(row.node, open);
      }
      if (ImGui::IsItemClicked()) {
        suppressRevealForNextSelectionChange = true;
        applyNodeSelection(*scene, row.node, *selectionState);
      }
      ImGui::PopID();
    }

    if (indentWidth > 0.0f) {
      ImGui::Unindent(indentWidth);
    }
  }

  void drawHierarchyWindow() {
    if (!ImGui::Begin(kHierarchyWindowName, &showHierarchyWindow)) {
      ImGui::End();
      return;
    }
    if (scene == nullptr) {
      ImGui::TextUnformatted("No scene available.");
      ImGui::End();
      return;
    }

    NURI_PROFILER_ZONE("ImGuiEditor::HierarchyWindow::ScenesSection",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    drawScenesSection();
    ImGui::SeparatorText("Hierarchy");
    NURI_PROFILER_ZONE_END();

    rebuildHierarchyFrameCache();
    applyPendingHierarchyReveal();
    rebuildHierarchyVisibleRows();

    NURI_PROFILER_ZONE("ImGuiEditor::HierarchyWindow::DrawTree",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    if (hierarchyVisibleRows.size() <= 1u) {
      ImGui::TextUnformatted("Scene graph is empty.");
    } else {
      const bool hasSceneName =
          !sceneSelectionState.names.empty() &&
          sceneSelectionState.selectedIndex >= 0 &&
          sceneSelectionState.selectedIndex <
              static_cast<int>(sceneSelectionState.names.size());
      const std::string sceneLabel =
          hasSceneName
              ? sceneSelectionState.names[sceneSelectionState.selectedIndex]
              : std::string("Active Scene");
      if (pendingRevealSelection && hierarchySelectedRowIndex >= 0) {
        const float lineHeight = ImGui::GetTextLineHeightWithSpacing();
        const float visibleHeight =
            std::max(ImGui::GetContentRegionAvail().y, lineHeight * 3.0f);
        const float targetY = std::max(
            0.0f, static_cast<float>(hierarchySelectedRowIndex) * lineHeight -
                      visibleHeight * 0.35f);
        ImGui::SetScrollY(targetY);
        pendingRevealSelection = false;
      }
      ImGuiListClipper clipper;
      clipper.Begin(static_cast<int>(hierarchyVisibleRows.size()),
                    ImGui::GetTextLineHeightWithSpacing());
      while (clipper.Step()) {
        for (int rowIndex = clipper.DisplayStart; rowIndex < clipper.DisplayEnd;
             ++rowIndex) {
          drawHierarchyRow(hierarchyVisibleRows[static_cast<size_t>(rowIndex)],
                           sceneLabel);
        }
      }
    }
    NURI_PROFILER_ZONE_END();
    ImGui::End();
  }

  void drawMainMenuBar() {
    if (!ImGui::BeginMainMenuBar()) {
      return;
    }

    if (ImGui::BeginMenu("Menu")) {
      ImGui::MenuItem("Bakery", nullptr, &showBakeryWindow);
      ImGui::MenuItem("Font Compiler", nullptr, &showFontCompilerWindow);
      ImGui::MenuItem("Hierarchy", nullptr, &showHierarchyWindow);
      ImGui::MenuItem("Inspector", nullptr, &showInspectorWindow);
      ImGui::MenuItem("Render Passes", nullptr, &showRenderPassesWindow);
      ImGui::MenuItem("Lights", nullptr, &showLightsWindow);
      if (ImGui::BeginMenu("Texture Filtering")) {
        auto &settings = renderSettings.textureFiltering;
        sanitizeTextureFilteringSettings(settings);
        const bool bilinear = settings.mode == TextureFilterMode::Bilinear;
        const bool trilinear = settings.mode == TextureFilterMode::Trilinear;
        const bool anisotropic =
            settings.mode == TextureFilterMode::Anisotropic;
        if (ImGui::MenuItem("Bilinear", nullptr, bilinear)) {
          settings.mode = TextureFilterMode::Bilinear;
        }
        if (ImGui::MenuItem("Trilinear", nullptr, trilinear)) {
          settings.mode = TextureFilterMode::Trilinear;
        }
        if (ImGui::MenuItem("Anisotropic", nullptr, anisotropic)) {
          settings.mode = TextureFilterMode::Anisotropic;
        }
        ImGui::Separator();
        ImGui::MenuItem("Settings Window", nullptr,
                        &showTextureFilteringWindow);
        ImGui::EndMenu();
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Debug")) {
      ImGui::MenuItem("Render Graph Telemetry", nullptr,
                      &showRenderGraphTelemetryWindow);
      ImGui::MenuItem("Gizmo Controls", nullptr, &showGizmoControlsWindow);
      ImGui::MenuItem("Telemetry", nullptr, &showTelemetrySettingsWindow);
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Camera")) {
      ImGui::MenuItem("Controller", nullptr, &showCameraControllerWindow);
      ImGui::MenuItem("Help", nullptr, &showCameraHelpWindow);
      ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();
  }

  void updateMetricGraphs(double deltaSeconds) {
    graphSampleAccumulatorSeconds += deltaSeconds;
    if (graphSampleAccumulatorSeconds < kMetricGraphUpdateIntervalSeconds) {
      return;
    }

    const float frametimeMs =
        sanitizeSample(static_cast<float>(deltaSeconds * 1000.0));
    const float instantFps =
        sanitizeSample(deltaSeconds > kMetricSampleMinDeltaSeconds
                           ? static_cast<float>(1.0 / deltaSeconds)
                           : 0.0f);
    fpsGraph->pushSample(instantFps);
    frametimeGraph->pushSample(frametimeMs);
    graphSampleAccumulatorSeconds = std::fmod(
        graphSampleAccumulatorSeconds, kMetricGraphUpdateIntervalSeconds);
  }

  void beginFrame() {
    NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
    platform->newFrame();
    ImGui::NewFrame();
    inspectorWindowVisible = false;
    inspectorWindowMinX = 0.0f;
    drawMainMenuBar();
    drawDockspaceRoot();
  }

  Result<RenderGraphGraphicsPassDesc, std::string> endFrame() {
    NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);

    if (telemetryOverlayState.showImGuiMetricsWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::ShowMetricsWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      ImGui::ShowMetricsWindow(&telemetryOverlayState.showImGuiMetricsWindow);
      NURI_PROFILER_ZONE_END();
    }

    NURI_PROFILER_ZONE("ImGuiEditor::UpdateMetricsAndLogs",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    fpsCounter.tick(frameDeltaSeconds, true);
    updateMetricGraphs(std::max(frameDeltaSeconds, 0.0));

    logUpdateAccumulatorSeconds += std::max(frameDeltaSeconds, 0.0);
    if (logUpdateAccumulatorSeconds >= kLogUpdateIntervalSeconds) {
      logModel.update(logFilterState);
      logUpdateAccumulatorSeconds =
          std::fmod(logUpdateAccumulatorSeconds, kLogUpdateIntervalSeconds);
    }
    validateSelectionState();
    syncSelectionWindows();
    NURI_PROFILER_ZONE_END();

#ifdef IMGUI_HAS_DOCK
    if (dockLayoutState.logDockId != 0) {
      ImGui::SetNextWindowDockID(dockLayoutState.logDockId, ImGuiCond_Once);
    }
#endif

    const ImGuiViewport *viewport = ImGui::GetMainViewport();
#ifndef IMGUI_HAS_DOCK
    setLogWindowPlacementWithoutDock(viewport);
#endif

    ScopedScratch scopedScratch(scratchArena);
    if (showHierarchyWindow) {
#ifdef IMGUI_HAS_DOCK
      if (dockLayoutState.hierarchyDockId != 0) {
        ImGui::SetNextWindowDockID(dockLayoutState.hierarchyDockId,
                                   ImGuiCond_Once);
      }
#endif
      NURI_PROFILER_ZONE("ImGuiEditor::DrawHierarchyWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawHierarchyWindow();
      NURI_PROFILER_ZONE_END();
    }
    if (showInspectorWindow) {
#ifdef IMGUI_HAS_DOCK
      if (dockLayoutState.inspectorDockId != 0) {
        ImGui::SetNextWindowDockID(dockLayoutState.inspectorDockId,
                                   ImGuiCond_Once);
      }
#endif
      NURI_PROFILER_ZONE("ImGuiEditor::DrawInspectorWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawInspectorWindow();
      NURI_PROFILER_ZONE_END();
    }

    NURI_PROFILER_ZONE("ImGuiEditor::DrawLogWindow",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    ImGui::Begin(kLogWindowName);
    drawLogWindow(logModel, logFilterState, scopedScratch.resource());
    ImGui::End();

    if (showRenderGraphTelemetryWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawRenderGraphTelemetryWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      if (ImGui::Begin(kRenderGraphTelemetryWindowName,
                       &showRenderGraphTelemetryWindow)) {
        drawRenderGraphTelemetryWindow(telemetryState, renderGraphTelemetry,
                                       window.nativeHandle());
      }
      ImGui::End();
      NURI_PROFILER_ZONE_END();
    }
    if (showFontCompilerWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawFontCompilerWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawFontCompilerWindow(showFontCompilerWindow, fontCompilerState,
                             textSystem, window.nativeHandle());
      NURI_PROFILER_ZONE_END();
    }
    if (showBakeryWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawBakeryWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawBakeryWindow(showBakeryWindow, bakeryState, bakery,
                       scopedScratch.resource(), window.nativeHandle());
      NURI_PROFILER_ZONE_END();
    }
    if (showRenderPassesWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawRenderPassesWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawRenderPassesWindow(showRenderPassesWindow, renderSettings,
                             renderPipeline, selectedPassIndex);
      NURI_PROFILER_ZONE_END();
    }
    if (showTextureFilteringWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawTextureFilteringWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawTextureFilteringWindow(showTextureFilteringWindow, renderSettings,
                                 gpu);
      NURI_PROFILER_ZONE_END();
    }
    if (showCameraControllerWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawCameraControllerWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      if (cameraSystem != nullptr &&
          ImGui::Begin(kCameraControllerWindowName,
                       &showCameraControllerWindow)) {
        drawCameraControllerContents(*cameraSystem, cameraControllerState);
      }
      if (cameraSystem == nullptr) {
        showCameraControllerWindow = false;
      } else {
        ImGui::End();
      }
      NURI_PROFILER_ZONE_END();
    }
    if (showCameraHelpWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawCameraHelpWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      if (ImGui::Begin(kCameraHelpWindowName, &showCameraHelpWindow)) {
        drawCameraHelpContents();
      }
      ImGui::End();
      NURI_PROFILER_ZONE_END();
    }
    if (showTelemetrySettingsWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawTelemetrySettingsWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      if (ImGui::Begin(kTelemetryWindowName, &showTelemetrySettingsWindow)) {
        ImGui::Checkbox("Show Overlay", &telemetryOverlayState.overlayEnabled);
        ImGui::Separator();
        ImGui::Checkbox("FPS + ms", &telemetryOverlayState.showFpsMs);
        ImGui::Checkbox("Instance counts",
                        &telemetryOverlayState.showInstanceStats);
        ImGui::Checkbox("Draw / tess stats",
                        &telemetryOverlayState.showDrawTessStats);
        ImGui::Checkbox("Indirect stats",
                        &telemetryOverlayState.showIndirectStats);
        ImGui::Checkbox("Debug draw stats",
                        &telemetryOverlayState.showDebugDrawStats);
        ImGui::Checkbox("Patch heatmap",
                        &telemetryOverlayState.showPatchHeatmap);
        ImGui::Checkbox("Dispatch stats",
                        &telemetryOverlayState.showDispatchStats);
        ImGui::Checkbox("Graphs", &telemetryOverlayState.showGraphs);
        ImGui::Separator();
        ImGui::Checkbox("ImGui Metrics Window",
                        &telemetryOverlayState.showImGuiMetricsWindow);
      }
      ImGui::End();
      NURI_PROFILER_ZONE_END();
    }
    NURI_PROFILER_ZONE_END();

    NURI_PROFILER_ZONE("ImGuiEditor::DrawFpsOverlay",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    float overlayRightBoundaryX = 0.0f;
    if (inspectorWindowVisible) {
      if (const ImGuiViewport *viewport = ImGui::GetMainViewport();
          viewport != nullptr &&
          inspectorWindowMinX >
              viewport->WorkPos.x + viewport->WorkSize.x * 0.5f) {
        overlayRightBoundaryX = inspectorWindowMinX;
      }
    }
    drawFpsOverlay(fpsCounter, *fpsGraph, *frametimeGraph, frameMetrics,
                   telemetryOverlayState, overlayRightBoundaryX);
    NURI_PROFILER_ZONE_END();

    NURI_PROFILER_ZONE("ImGuiEditor::FinalizeImGuiFrame",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    ImGui::EndFrame();
    ImGui::Render();
    NURI_PROFILER_ZONE_END();

    const auto passResult = [&]() {
      Result<RenderGraphGraphicsPassDesc, std::string> result =
          Result<RenderGraphGraphicsPassDesc, std::string>::makeError(
              std::string{});
      NURI_PROFILER_ZONE("ImGuiEditor::BuildGraphicsPassDesc",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      result =
          renderer->buildGraphicsPassDesc(gpu.getSwapchainFormat(), frameIndex);
      NURI_PROFILER_ZONE_END();
      return result;
    }();
    return passResult;
  }

  void drawDockspaceRoot() {
#ifdef IMGUI_HAS_DOCK
    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    setDockspaceWindowPlacement(viewport);

    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

    ImGui::Begin(kDockspaceWindowName, nullptr, dockspaceWindowFlags());
    ImGui::PopStyleVar(3);

    const ImGuiDockNodeFlags dockFlags = ImGuiDockNodeFlags_PassthruCentralNode;
    const ImGuiID dockspaceId = ImGui::GetID(kDockspaceRootId);
    ImGui::DockSpace(dockspaceId, ImVec2(0.0f, 0.0f), dockFlags);
    dockLayoutState.ensureLayout(dockspaceId, viewport);

    ImGui::End();
#endif
  }

  Window &window;
  GPUDevice &gpu;
  std::unique_ptr<ImGuiGlfwPlatform> platform;
  std::unique_ptr<ImGuiGpuRenderer> renderer;
  FPSCounter fpsCounter{0.5f};
  std::unique_ptr<LinearGraph> fpsGraph =
      createImPlotLinearGraph(kMetricGraphSampleCount);
  std::unique_ptr<LinearGraph> frametimeGraph =
      createImPlotLinearGraph(kMetricGraphSampleCount);
  double graphSampleAccumulatorSeconds = kMetricGraphUpdateIntervalSeconds;
  double logUpdateAccumulatorSeconds = kLogUpdateIntervalSeconds;
  bool showBakeryWindow = false;
  bool showFontCompilerWindow = false;
  bool showHierarchyWindow = true;
  bool showInspectorWindow = true;
  bool showRenderPassesWindow = false;
  bool showLightsWindow = false;
  bool showTextureFilteringWindow = false;
  bool showRenderGraphTelemetryWindow = false;
  bool showGizmoControlsWindow = false;
  bool showTelemetrySettingsWindow = false;
  bool showCameraControllerWindow = false;
  bool showCameraHelpWindow = false;
  bool inspectorWindowVisible = false;
  RenderSettings renderSettings{};
  RenderFrameMetrics frameMetrics{};
  size_t selectedPassIndex = 0u;
  double frameDeltaSeconds = 0.0;
  uint64_t frameIndex = 0;
  float inspectorWindowMinX = 0.0f;
  LogModel logModel;
  LogFilterState logFilterState;
  RenderGraphTelemetryUiState telemetryState;
  TelemetryOverlayUiState telemetryOverlayState;
  FontCompilerUiState fontCompilerState;
  BakeryUiState bakeryState;
  SceneSelectionUiState sceneSelectionState;
  CameraControllerWidgetState cameraControllerState{};
  MaybeDockLayoutState dockLayoutState;
  ScratchArena scratchArena;
  std::vector<RenderableInspectorState> renderableInspectorStates{};
  LightEditorDraft lightEditorDraft{};
  SceneEditorSelectionState localSelectionState{};
  NodeId lastObservedSelectionNode = kInvalidNodeId;
  bool pendingRevealSelection = false;
  bool suppressRevealForNextSelectionChange = false;
  bool hierarchyStatsCacheValid = false;
  bool hierarchyTopologyCacheValid = false;
  uint32_t cachedRenderableCount = 0u;
  uint32_t cachedLightCount = 0u;
  NodeId cachedSelectedPathLeaf = kInvalidNodeId;
  std::vector<HierarchyNodeTopology> hierarchyNodeTopology{};
  std::vector<HierarchyNodeStats> hierarchyNodeStats{};
  std::vector<uint8_t> selectedPathNodeFlags{};
  std::vector<int32_t> selectedPathChildIndices{};
  std::vector<HierarchyVisibleRow> hierarchyVisibleRows{};
  std::vector<uint8_t> hierarchyNodeOpenFlags{};
  std::unordered_set<uint64_t> hierarchyOpenBatchKeys{};
  int hierarchySelectedRowIndex = -1;
  bool hierarchySceneRootOpen = true;
  RenderScene *scene = nullptr;
  ResourceManager *resources = nullptr;
  RenderPipeline *renderPipeline = nullptr;
  SceneEditorSelectionState *selectionState = nullptr;
  TextSystem *textSystem = nullptr;
  CameraSystem *cameraSystem = nullptr;
  bakery::BakerySystem *bakery = nullptr;
  RenderGraphTelemetryService *renderGraphTelemetry = nullptr;
};

std::unique_ptr<ImGuiEditor>
ImGuiEditor::create(Window &window, GPUDevice &gpu,
                    const EditorServices &services) {
  return std::unique_ptr<ImGuiEditor>(new ImGuiEditor(window, gpu, services));
}

ImGuiEditor::ImGuiEditor(Window &window, GPUDevice &gpu,
                         const EditorServices &services)
    : impl_(std::make_unique<Impl>(window, gpu, services)) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();

  ImGuiIO &io = ImGui::GetIO();
  // Ensure the default font exists before the first NewFrame().
  // (Some builds/configs can end up with an empty font atlas otherwise.)
  if (io.Fonts && io.Fonts->Fonts.empty()) {
    io.Fonts->AddFontDefault();
  }
  io.FontDefault =
      io.Fonts && !io.Fonts->Fonts.empty() ? io.Fonts->Fonts[0] : nullptr;
  if (io.Fonts) {
    io.Fonts->Build();
  }
#if defined(ImGuiConfigFlags_DockingEnable) || defined(IMGUI_HAS_DOCK)
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  // Make docking work without needing to hold Shift.
  io.ConfigDockingWithShift = false;
#endif
  io.IniFilename = nullptr;

  impl_->platform = ImGuiGlfwPlatform::create(impl_->window);
  impl_->renderer = ImGuiGpuRenderer::create(impl_->gpu);
}

ImGuiEditor::~ImGuiEditor() {
  impl_->resetSceneUiState();
  impl_->renderer.reset();
  impl_->platform.reset();
  if (ImPlot::GetCurrentContext() != nullptr) {
    ImPlot::DestroyContext();
  }
  ImGui::DestroyContext();
}

void ImGuiEditor::setFrameDeltaSeconds(double deltaTime) {
  if (!impl_) {
    return;
  }
  if (!std::isfinite(deltaTime) || deltaTime < 0.0) {
    impl_->frameDeltaSeconds = 0.0;
    return;
  }
  impl_->frameDeltaSeconds = deltaTime;
}

void ImGuiEditor::setFrameIndex(uint64_t frameIndex) {
  if (!impl_) {
    return;
  }
  impl_->frameIndex = frameIndex;
}

void ImGuiEditor::setFrameMetrics(const RenderFrameMetrics &metrics) {
  if (!impl_) {
    return;
  }
  impl_->frameMetrics = metrics;
}

void ImGuiEditor::setRenderSettings(const RenderSettings &settings) {
  if (!impl_) {
    return;
  }
  impl_->renderSettings = settings;
  sanitizeTextureFilteringSettings(impl_->renderSettings.textureFiltering);
}

void ImGuiEditor::syncCameraControllerWidgetStateFromCamera(
    const Camera &camera) {
  if (!impl_) {
    return;
  }
  nuri::syncCameraControllerWidgetStateFromCamera(camera,
                                                  impl_->cameraControllerState);
}

void ImGuiEditor::setSceneSelectionUi(
    std::span<const EditorSceneSelectionOption> scenes,
    std::string_view selectedSceneId, uint64_t version,
    std::string_view hotkeyHint) {
  if (!impl_) {
    return;
  }
  impl_->sceneSelectionState.set(scenes, selectedSceneId, version, hotkeyHint);
}

void ImGuiEditor::resetSceneUiState() {
  if (!impl_) {
    return;
  }
  impl_->resetSceneUiState();
}

std::optional<std::string> ImGuiEditor::takeSceneSelectionRequest() {
  return impl_
             ? std::exchange(impl_->sceneSelectionState.pendingSelectionRequest,
                             std::nullopt)
             : std::nullopt;
}

bool *ImGuiEditor::gizmoControlsWindowOpenState() {
  return impl_ ? &impl_->showGizmoControlsWindow : nullptr;
}

bool *ImGuiEditor::lightsWindowOpenState() {
  return impl_ ? &impl_->showLightsWindow : nullptr;
}

bool ImGuiEditor::isGizmoControlsWindowOpen() const {
  return impl_ != nullptr && impl_->showGizmoControlsWindow;
}

bool ImGuiEditor::isLightsWindowOpen() const {
  return impl_ != nullptr && impl_->showLightsWindow;
}

RenderSettings ImGuiEditor::renderSettings() const {
  if (!impl_) {
    return RenderSettings{};
  }
  RenderSettings settings = impl_->renderSettings;
  sanitizeTextureFilteringSettings(settings.textureFiltering);
  return settings;
}

bool ImGuiEditor::wantsCaptureKeyboard() const {
  if (!ImGui::GetCurrentContext()) {
    return false;
  }
  const ImGuiIO &io = ImGui::GetIO();
  return io.WantCaptureKeyboard;
}

bool ImGuiEditor::wantsCaptureMouse() const {
  if (!ImGui::GetCurrentContext()) {
    return false;
  }
  const ImGuiIO &io = ImGui::GetIO();
  return io.WantCaptureMouse;
}

void ImGuiEditor::beginFrame() { impl_->beginFrame(); }

Result<RenderGraphGraphicsPassDesc, std::string> ImGuiEditor::endFrame() {
  return impl_->endFrame();
}

} // namespace nuri
