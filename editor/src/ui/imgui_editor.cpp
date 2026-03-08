#include "nuri/editor_pch.h"

#include "nuri/ui/imgui_editor.h"

#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/core/runtime_config.h"
#include "nuri/bakery/bakery_system.h"
#include "nuri/core/window.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/imgui_gpu_renderer.h"
#include "nuri/gfx/render_graph/render_graph_telemetry.h"
#include "nuri/platform/imgui_glfw_platform.h"
#include "nuri/resources/storage/font/nfont_compiler.h"
#include "nuri/text/text_system.h"
#include "nuri/ui/file_dialog_widget.h"
#include "nuri/ui/linear_graph.h"
#include "nuri/utils/fsp_counter.h"

namespace nuri {

namespace {

constexpr size_t kMaxLogLines = 2000;
constexpr float kLogFilterWidth = 200.0f;
constexpr float kLayerPanelWidth = 360.0f;
constexpr float kLayerListWidth = 140.0f;
constexpr double kMetricGraphUpdateIntervalSeconds = 0.04;
constexpr double kLogUpdateIntervalSeconds = 0.10;
constexpr float kMetricGraphWindowWidth = 300.0f;
constexpr float kMetricGraphWindowHeight = 280.0f;
constexpr double kMetricSampleMinDeltaSeconds = 1.0e-6;
constexpr std::size_t kMetricGraphSampleCount = 240;
constexpr uint32_t kUiMaxTessInstances = 65536u;
constexpr const char *kDockspaceWindowName = "NuriDockspace";
constexpr const char *kDockspaceRootId = "NuriDockspace##Root";
constexpr const char *kLogWindowName = "Log";
constexpr const char *kRenderGraphTelemetryWindowName =
    "Render Graph Telemetry";
constexpr const char *kFontCompilerWindowName = "Font Compiler";
constexpr const char *kBakeryWindowName = "Bakery";
constexpr const char *kCameraControllerWindowName = "Camera Controller";
constexpr const char *kScenePresetWindowName = "Scene Preset";
constexpr const char *kSelectionWindowName = "Selection";

enum class LayerSelection : uint8_t {
  Skybox,
  Opaque,
  Debug,
};

const std::array<LayerSelection, 3> kRenderLayers = {
    LayerSelection::Skybox,
    LayerSelection::Opaque,
    LayerSelection::Debug,
};

const char *layerDisplayName(LayerSelection layer) {
  switch (layer) {
  case LayerSelection::Skybox:
    return "Skybox";
  case LayerSelection::Opaque:
    return "Opaque";
  case LayerSelection::Debug:
    return "Debug";
  }
  return "Unknown";
}

const char *bakeJobKindName(bakery::BakeJobKind kind) {
  switch (kind) {
  case bakery::BakeJobKind::BrdfLut:
    return "BRDF LUT";
  case bakery::BakeJobKind::EnvmapPrefilter:
    return "Envmap Prefilter";
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
    {LogLevel::Fatal, "[Fatal]"},
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
  bool forceRebuild = false;
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

void drawInspectorHeader(LayerSelection layer) {
  ImGui::TextUnformatted(layerDisplayName(layer));
  ImGui::Separator();
}

void drawSkyboxSettings(RenderSettings::SkyboxSettings &skybox) {
  ImGui::Checkbox("Enabled##SkyboxLayer", &skybox.enabled);
}

void drawOpaqueSettings(RenderSettings::OpaqueSettings &opaque) {
  ImGui::Checkbox("Enabled##OpaqueLayer", &opaque.enabled);
  constexpr const char *kDebugModes[] = {
      "None",
      "Wire Overlay",
      "Wireframe Only",
      "Tess Patch (Edges + Heatmap)",
  };
  int debugMode = static_cast<int>(opaque.debugVisualization);
  debugMode =
      std::clamp(debugMode, 0, static_cast<int>(IM_ARRAYSIZE(kDebugModes)) - 1);
  if (ImGui::Combo("Debug Visualization##OpaqueLayer", &debugMode, kDebugModes,
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
  ImGui::TextUnformatted("Mesh LOD");
  ImGui::Checkbox("Enable Indirect Draws##OpaqueLayer",
                  &opaque.enableIndirectDraw);
  ImGui::Checkbox("Enable Instanced Draws##OpaqueLayer",
                  &opaque.enableInstancedDraw);
  ImGui::Checkbox("Enable Mesh LOD##OpaqueLayer", &opaque.enableMeshLod);
  ImGui::SliderInt("Forced LOD##OpaqueLayer", &opaque.forcedMeshLod, -1, 3);

  float lodThresholds[3] = {
      opaque.meshLodDistanceThresholds.x,
      opaque.meshLodDistanceThresholds.y,
      opaque.meshLodDistanceThresholds.z,
  };
  if (ImGui::SliderFloat3("LOD Distance##OpaqueLayer", lodThresholds, 0.5f,
                          128.0f, "%.1f")) {
    std::sort(std::begin(lodThresholds), std::end(lodThresholds));
    opaque.meshLodDistanceThresholds =
        glm::vec3(lodThresholds[0], lodThresholds[1], lodThresholds[2]);
  }

  ImGui::Separator();
  ImGui::TextUnformatted("Tessellation");
  ImGui::Checkbox("Enable Tessellation##OpaqueLayer",
                  &opaque.enableTessellation);
  ImGui::SliderFloat("Tess Near##OpaqueLayer", &opaque.tessNearDistance, 0.0f,
                     256.0f, "%.2f");
  ImGui::SliderFloat("Tess Far##OpaqueLayer", &opaque.tessFarDistance, 0.0f,
                     512.0f, "%.2f");
  ImGui::SliderFloat("Tess Min##OpaqueLayer", &opaque.tessMinFactor, 1.0f,
                     64.0f, "%.2f");
  ImGui::SliderFloat("Tess Max##OpaqueLayer", &opaque.tessMaxFactor, 1.0f,
                     64.0f, "%.2f");
  int tessMaxInstances = static_cast<int>(
      std::min<uint32_t>(opaque.tessMaxInstances, kUiMaxTessInstances));
  if (ImGui::SliderInt("Tess Max Inst##OpaqueLayer", &tessMaxInstances, 0,
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
  ImGui::Checkbox("Enabled##DebugLayer", &debug.enabled);
  ImGui::Checkbox("Model Bounds##DebugLayer", &debug.modelBounds);
  ImGui::Checkbox("Grid##DebugLayer", &debug.grid);
}

void drawLayerList(LayerSelection &selectedLayer) {
  ImGui::TextUnformatted("Layers");
  ImGui::Separator();
  for (const LayerSelection layer : kRenderLayers) {
    const bool isSelected = selectedLayer == layer;
    if (ImGui::Selectable(layerDisplayName(layer), isSelected)) {
      selectedLayer = layer;
    }
  }
}

void drawLayerInspector(RenderSettings &renderSettings,
                        LayerSelection &selectedLayer) {
  ImGui::BeginChild("LayerPanel", ImVec2(0.0f, 0.0f), false,
                    ImGuiWindowFlags_NoScrollbar);

  if (ImGui::BeginTable("LayerInspectorTable", 2,
                        ImGuiTableFlags_BordersInnerV |
                            ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableSetupColumn("LayerList", ImGuiTableColumnFlags_WidthFixed,
                            kLayerListWidth);
    ImGui::TableSetupColumn("LayerSettings", ImGuiTableColumnFlags_WidthStretch,
                            0.0f);

    ImGui::TableNextColumn();
    drawLayerList(selectedLayer);

    ImGui::TableNextColumn();
    drawInspectorHeader(selectedLayer);
    switch (selectedLayer) {
    case LayerSelection::Skybox:
      drawSkyboxSettings(renderSettings.skybox);
      break;
    case LayerSelection::Opaque:
      drawOpaqueSettings(renderSettings.opaque);
      break;
    case LayerSelection::Debug:
      drawDebugSettings(renderSettings.debug);
      break;
    }

    ImGui::EndTable();
  }

  ImGui::EndChild();
}

void drawLogWindow(LogModel &model, LogFilterState &filterState,
                   RenderSettings &renderSettings,
                   LayerSelection &selectedLayer,
                   std::pmr::memory_resource *scratchResource) {
  drawLogToolbar(model, filterState);
  ImGui::Separator();

  const ImGuiTableFlags tableFlags =
      ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable;
  if (!ImGui::BeginTable("LogAndLayerTable", 2, tableFlags)) {
    drawLogMessages(model, filterState, scratchResource);
    return;
  }

  ImGui::TableSetupColumn("Logs", ImGuiTableColumnFlags_WidthStretch, 0.0f);
  ImGui::TableSetupColumn("Layer Inspector", ImGuiTableColumnFlags_WidthFixed,
                          kLayerPanelWidth);

  ImGui::TableNextColumn();
  drawLogMessages(model, filterState, scratchResource);

  ImGui::TableNextColumn();
  drawLayerInspector(renderSettings, selectedLayer);

  ImGui::EndTable();
}

void drawFontCompilerWindow(FontCompilerUiState &state, TextSystem *textSystem,
                            void *ownerWindowHandle) {
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

  if (!ImGui::Begin(kFontCompilerWindowName)) {
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

void drawBakeryWindow(BakeryUiState &state, bakery::BakerySystem *bakery,
                      std::pmr::memory_resource *scratchResource,
                      void *ownerWindowHandle) {
  if (!ImGui::Begin(kBakeryWindowName)) {
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
    auto enqueueResult = bakery->enqueue(bakery::BakeRequest{bakery::BrdfLutBakeRequest{
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

  ImGui::InputText("Env HDR Path", state.envHdrPath.data(), state.envHdrPath.size());
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
    auto enqueueResult = bakery->enqueue(
        bakery::BakeRequest{bakery::EnvmapPrefilterBakeRequest{
        .environmentHdrPath = std::filesystem::path(std::string(state.envHdrPath.data())),
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
  if (ImGui::BeginTable(
          "BakeryJobs", 6,
          ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable |
              ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
          ImVec2(0.0f, 240.0f))) {
    ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 52.0f);
    ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed, 120.0f);
    ImGui::TableSetupColumn("State", ImGuiTableColumnFlags_WidthFixed, 110.0f);
    ImGui::TableSetupColumn("Progress", ImGuiTableColumnFlags_WidthFixed, 90.0f);
    ImGui::TableSetupColumn("Summary", ImGuiTableColumnFlags_WidthStretch, 0.0f);
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
  const std::string suggestedPath = telemetry->suggestDumpPath().generic_string();
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

void drawTelemetrySummary(const RenderGraphTelemetrySnapshot::Summary &summary) {
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
  drawRow("Resolved Dependency Slots", summary.resolvedDependencyBufferSlotCount);
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
                        ImGuiTableFlags_BordersInnerV |
                            ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_Resizable |
                            ImGuiTableFlags_ScrollY,
                        ImVec2(0.0f, height))) {
    drawRows();
    ImGui::EndTable();
  }
}

void drawRenderGraphTelemetryWindow(RenderGraphTelemetryUiState &state,
                                    RenderGraphTelemetryService *telemetry,
                                    void *ownerWindowHandle) {
  syncTelemetryDumpPath(state, telemetry);

  ImGui::InputText("Output .txt", state.outputPath.data(), state.outputPath.size());
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
      state.status = std::string("Wrote telemetry to ") + state.outputPath.data();
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
        ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthFixed, 64.0f);
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
        ImGui::TableSetupColumn("Before", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableSetupColumn("Before Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("After", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableSetupColumn("After Name", ImGuiTableColumnFlags_WidthStretch);
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
        ImGui::TableSetupColumn("Rank", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthFixed, 60.0f);
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
        ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Resource", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("First", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Last", ImGuiTableColumnFlags_WidthFixed, 70.0f);
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
        ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Resource", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Physical", ImGuiTableColumnFlags_WidthFixed, 70.0f);
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
        ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Resource", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Physical", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableHeadersRow();
        for (uint32_t i = 0; i < snapshot->transientTextureAllocationByResource.size();
             ++i) {
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
        for (uint32_t i = 0; i < snapshot->transientBufferAllocationByResource.size();
             ++i) {
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
        ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Physical", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Representative", ImGuiTableColumnFlags_WidthFixed,
                                110.0f);
        ImGui::TableSetupColumn("Format/Usage", ImGuiTableColumnFlags_WidthFixed,
                                110.0f);
        ImGui::TableSetupColumn("Details", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (const auto &physical : snapshot->transientTexturePhysicalAllocations) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("tex");
          ImGui::TableNextColumn();
          ImGui::Text("%u", physical.allocationIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", physical.representativeResourceIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", static_cast<uint32_t>(physical.desc.format));
          ImGui::TableNextColumn();
          ImGui::Text("%ux%ux%u layers=%u samples=%u mips=%u",
                      physical.desc.dimensions.width,
                      physical.desc.dimensions.height,
                      physical.desc.dimensions.depth, physical.desc.numLayers,
                      physical.desc.numSamples, physical.desc.numMipLevels);
        }
        for (const auto &physical : snapshot->transientBufferPhysicalAllocations) {
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
        ImGui::TableSetupColumn("Section", ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableSetupColumn("Item", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Target", ImGuiTableColumnFlags_WidthFixed, 110.0f);
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
        for (uint32_t slot = 0; slot < snapshot->resolvedDependencyBuffers.size();
             ++slot) {
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
        for (const auto &binding : snapshot->unresolvedDependencyBufferBindings) {
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
        ImGui::TableSetupColumn("Section", ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Offset", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableHeadersRow();
        for (uint32_t i = 0; i < snapshot->dependencyBufferRangesByPass.size(); ++i) {
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
        for (uint32_t i = 0; i < snapshot->preDispatchRangesByPass.size(); ++i) {
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
        for (uint32_t i = 0; i < snapshot->preDispatchDependencyRanges.size(); ++i) {
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
  ImGui::SetNextWindowPos(viewport->Pos);
  ImGui::SetNextWindowSize(viewport->Size);
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
                    const RenderFrameMetrics &frameMetrics) {
  if (const ImGuiViewport *viewport = ImGui::GetMainViewport()) {
    ImGui::SetNextWindowPos({viewport->WorkPos.x + viewport->WorkSize.x - 15.0f,
                             viewport->WorkPos.y + 15.0f},
                            ImGuiCond_Always, {1.0f, 0.0f});
  }
  ImGui::SetNextWindowBgAlpha(0.30f);
  ImGui::SetNextWindowSize(
      ImVec2(kMetricGraphWindowWidth, kMetricGraphWindowHeight),
      ImGuiCond_Always);
  if (ImGui::Begin("##FPS", nullptr,
                   ImGuiWindowFlags_NoDecoration |
                       ImGuiWindowFlags_NoSavedSettings |
                       ImGuiWindowFlags_NoFocusOnAppearing |
                       ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove)) {
    const float fps = fpsCounter.getFPS();
    const float milliseconds = fps > 0.0f ? 1000.0f / fps : 0.0f;
    ImGui::Text("FPS : %i", static_cast<int>(fps));
    ImGui::Text("Ms  : %.1f", milliseconds);
    ImGui::Text("Inst: %u / %u", frameMetrics.opaque.visibleInstances,
                frameMetrics.opaque.totalInstances);
    ImGui::Text("Draw: %u (Tess: %u)  Tess Inst: %u",
                frameMetrics.opaque.instancedDraws,
                frameMetrics.opaque.tessellatedDraws,
                frameMetrics.opaque.tessellatedInstances);
    ImGui::Text("Indirect: %u calls / %u cmds",
                frameMetrics.opaque.indirectDrawCalls,
                frameMetrics.opaque.indirectCommands);
    ImGui::Text("Debug Draws: %u (Fallback: %u)",
                frameMetrics.opaque.debugOverlayDraws,
                frameMetrics.opaque.debugOverlayFallbackDraws);
    ImGui::Text("Patch Heatmap: %u",
                frameMetrics.opaque.debugPatchHeatmapDraws);
    ImGui::Text("Dispatch: %u x%u", frameMetrics.opaque.computeDispatches,
                frameMetrics.opaque.computeDispatchX);
    ImGui::Separator();

    const float availableGraphHeight = ImGui::GetContentRegionAvail().y;
    const float perGraphHeight = std::max(availableGraphHeight * 0.5f, 1.0f);
    const ImVec2 itemSpacing = ImGui::GetStyle().ItemSpacing;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(itemSpacing.x, 0.0f));

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
    ImGui::DockBuilderSetNodeSize(dockspaceId, viewport->Size);

    ImGuiID dockMain = dockspaceId;
    ImGuiID dockBottom = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Down,
                                                     0.25f, nullptr, &dockMain);
    ImGuiID dockBottomLeft = ImGui::DockBuilderSplitNode(
        dockBottom, ImGuiDir_Left, 0.28f, nullptr, &dockBottom);

    logDockId = dockBottom;
    ImGui::DockBuilderDockWindow(kCameraControllerWindowName, dockBottomLeft);
    ImGui::DockBuilderDockWindow(kScenePresetWindowName, dockBottomLeft);
    ImGui::DockBuilderDockWindow(kSelectionWindowName, dockBottomLeft);
    ImGui::DockBuilderDockWindow(kLogWindowName, logDockId);
    ImGui::DockBuilderDockWindow(kRenderGraphTelemetryWindowName, logDockId);
    ImGui::DockBuilderDockWindow(kFontCompilerWindowName, logDockId);
    ImGui::DockBuilderDockWindow(kBakeryWindowName, logDockId);
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
      : window(windowIn), gpu(gpuIn), textSystem(services.textSystem),
        bakery(services.bakery),
        renderGraphTelemetry(services.renderGraphTelemetry) {}

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
    drawDockspaceRoot();
  }

  Result<RenderGraphGraphicsPassDesc, std::string> endFrame() {
    NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);

    if (showMetricsWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::ShowMetricsWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      ImGui::ShowMetricsWindow(&showMetricsWindow);
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

    NURI_PROFILER_ZONE("ImGuiEditor::DrawLogWindow",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    ScopedScratch scopedScratch(scratchArena);
    ImGui::Begin(kLogWindowName);
    drawLogWindow(logModel, logFilterState, renderSettings, selectedLayer,
                  scopedScratch.resource());
    ImGui::End();
#ifdef IMGUI_HAS_DOCK
    if (dockLayoutState.logDockId != 0) {
      ImGui::SetNextWindowDockID(dockLayoutState.logDockId, ImGuiCond_Once);
    }
#endif
    if (ImGui::Begin(kRenderGraphTelemetryWindowName)) {
      drawRenderGraphTelemetryWindow(telemetryState, renderGraphTelemetry,
                                     window.nativeHandle());
    }
    ImGui::End();
    drawFontCompilerWindow(fontCompilerState, textSystem,
                           window.nativeHandle());
    drawBakeryWindow(bakeryState, bakery, scopedScratch.resource(),
                     window.nativeHandle());
    NURI_PROFILER_ZONE_END();

    NURI_PROFILER_ZONE("ImGuiEditor::DrawFpsOverlay",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    drawFpsOverlay(fpsCounter, *fpsGraph, *frametimeGraph, frameMetrics);
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
  bool showMetricsWindow = false;
  RenderSettings renderSettings{};
  RenderFrameMetrics frameMetrics{};
  LayerSelection selectedLayer = LayerSelection::Opaque;
  double frameDeltaSeconds = 0.0;
  uint64_t frameIndex = 0;
  LogModel logModel;
  LogFilterState logFilterState;
  RenderGraphTelemetryUiState telemetryState;
  FontCompilerUiState fontCompilerState;
  BakeryUiState bakeryState;
  MaybeDockLayoutState dockLayoutState;
  ScratchArena scratchArena;
  TextSystem *textSystem = nullptr;
  bakery::BakerySystem *bakery = nullptr;
  RenderGraphTelemetryService *renderGraphTelemetry = nullptr;
};

std::unique_ptr<ImGuiEditor>
ImGuiEditor::create(Window &window, GPUDevice &gpu, EventManager &events,
                    const EditorServices &services) {
  return std::unique_ptr<ImGuiEditor>(
      new ImGuiEditor(window, gpu, events, services));
}

ImGuiEditor::ImGuiEditor(Window &window, GPUDevice &gpu, EventManager &events,
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

  impl_->platform = ImGuiGlfwPlatform::create(impl_->window, events);
  impl_->renderer = ImGuiGpuRenderer::create(impl_->gpu);
}

ImGuiEditor::~ImGuiEditor() {
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
}

RenderSettings ImGuiEditor::renderSettings() const {
  return impl_ ? impl_->renderSettings : RenderSettings{};
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
