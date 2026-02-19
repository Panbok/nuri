#include "nuri/ui/imgui_editor.h"

#include "nuri/core/profiling.h"
#include "nuri/core/window.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/imgui_gpu_renderer.h"
#include "nuri/platform/imgui_glfw_platform.h"
#include "nuri/ui/linear_graph.h"
#include "nuri/utils/fsp_counter.h"

#include <imgui.h>
#include <imgui_internal.h>
#if __has_include(<implot.h>)
#include <implot.h>
#else
#include <implot/implot.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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
constexpr const char *kCameraControllerWindowName = "Camera Controller";

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

void drawLogMessages(const LogModel &model, LogFilterState &filterState) {
  std::vector<size_t> visibleIndices;
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
  ImGui::Checkbox("Wireframe##OpaqueLayer", &opaque.wireframe);

  ImGui::Separator();
  ImGui::TextUnformatted("Mesh LOD");
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
  int tessMaxInstances =
      static_cast<int>(std::min<uint32_t>(opaque.tessMaxInstances,
                                          kUiMaxTessInstances));
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
                   LayerSelection &selectedLayer) {
  drawLogToolbar(model, filterState);
  ImGui::Separator();

  const ImGuiTableFlags tableFlags =
      ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable;
  if (!ImGui::BeginTable("LogAndLayerTable", 2, tableFlags)) {
    drawLogMessages(model, filterState);
    return;
  }

  ImGui::TableSetupColumn("Logs", ImGuiTableColumnFlags_WidthStretch, 0.0f);
  ImGui::TableSetupColumn("Layer Inspector", ImGuiTableColumnFlags_WidthFixed,
                          kLayerPanelWidth);

  ImGui::TableNextColumn();
  drawLogMessages(model, filterState);

  ImGui::TableNextColumn();
  drawLayerInspector(renderSettings, selectedLayer);

  ImGui::EndTable();
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
    ImGui::Text("Inst: %u / %u",
                frameMetrics.opaque.visibleInstances,
                frameMetrics.opaque.totalInstances);
    ImGui::Text("Draw: %u (Tess: %u)  Tess Inst: %u",
                frameMetrics.opaque.instancedDraws,
                frameMetrics.opaque.tessellatedDraws,
                frameMetrics.opaque.tessellatedInstances);
    ImGui::Text("Dispatch: %u x%u",
                frameMetrics.opaque.computeDispatches,
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
    ImGui::DockBuilderDockWindow(kLogWindowName, logDockId);
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
  Impl(Window &windowIn, GPUDevice &gpuIn) : window(windowIn), gpu(gpuIn) {}

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
      ImGui::ShowMetricsWindow(&showMetricsWindow);
    }

    fpsCounter.tick(frameDeltaSeconds, true);
    updateMetricGraphs(std::max(frameDeltaSeconds, 0.0));

    logUpdateAccumulatorSeconds += std::max(frameDeltaSeconds, 0.0);
    if (logUpdateAccumulatorSeconds >= kLogUpdateIntervalSeconds) {
      logModel.update(logFilterState);
      logUpdateAccumulatorSeconds = std::fmod(
          logUpdateAccumulatorSeconds, kLogUpdateIntervalSeconds);
    }

#ifdef IMGUI_HAS_DOCK
    if (dockLayoutState.logDockId != 0) {
      ImGui::SetNextWindowDockID(dockLayoutState.logDockId, ImGuiCond_Once);
    }
#endif

    const ImGuiViewport *viewport = ImGui::GetMainViewport();
#ifndef IMGUI_HAS_DOCK
    setLogWindowPlacementWithoutDock(viewport);
#endif

    ImGui::Begin(kLogWindowName);
    drawLogWindow(logModel, logFilterState, renderSettings, selectedLayer);
    ImGui::End();

    drawFpsOverlay(fpsCounter, *fpsGraph, *frametimeGraph, frameMetrics);

    ImGui::EndFrame();
    ImGui::Render();

    return renderer->buildRenderPass(gpu.getSwapchainFormat(), frameIndex);
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
  MaybeDockLayoutState dockLayoutState;
};

std::unique_ptr<ImGuiEditor> ImGuiEditor::create(Window &window, GPUDevice &gpu,
                                                 EventManager &events) {
  return std::unique_ptr<ImGuiEditor>(new ImGuiEditor(window, gpu, events));
}

ImGuiEditor::ImGuiEditor(Window &window, GPUDevice &gpu, EventManager &events)
    : impl_(std::make_unique<Impl>(window, gpu)) {
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
