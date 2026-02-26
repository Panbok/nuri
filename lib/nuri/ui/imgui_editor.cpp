#include "nuri/pch.h"

#include "nuri/ui/imgui_editor.h"

#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/core/runtime_config.h"
#include "nuri/core/window.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/imgui_gpu_renderer.h"
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
constexpr const char *kFontCompilerWindowName = "Font Compiler";
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

void drawFontCompilerWindow(FontCompilerUiState &state,
                            TextSystem *textSystem) {
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
    if (const auto selectedPath = state.fileDialog.openFontFile()) {
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

  if (state.compileInFlight) {
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
  if (state.compileInFlight) {
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
    ImGui::DockBuilderDockWindow(kFontCompilerWindowName, logDockId);
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
      : window(windowIn), gpu(gpuIn), textSystem(services.textSystem) {}

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

  Result<RenderPass, std::string> endFrame() {
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
    drawFontCompilerWindow(fontCompilerState, textSystem);
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
      Result<RenderPass, std::string> result =
          Result<RenderPass, std::string>::makeError(std::string{});
      NURI_PROFILER_ZONE("ImGuiEditor::BuildRenderPass",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      result = renderer->buildRenderPass(gpu.getSwapchainFormat(), frameIndex);
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
  FontCompilerUiState fontCompilerState;
  MaybeDockLayoutState dockLayoutState;
  ScratchArena scratchArena;
  TextSystem *textSystem = nullptr;
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

Result<RenderPass, std::string> ImGuiEditor::endFrame() {
  return impl_->endFrame();
}

} // namespace nuri
