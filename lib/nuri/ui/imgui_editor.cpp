#include "nuri/ui/imgui_editor.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/core/window.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/imgui_gpu_renderer.h"
#include "nuri/platform/imgui_glfw_platform.h"
#include "nuri/utils/FPSCounter.h"

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace nuri {

namespace {

constexpr size_t kMaxLogLines = 2000;
constexpr float kLogFilterWidth = 200.0f;
constexpr float kTagHorizontalPadding = 6.0f;
constexpr float kTagCornerRounding = 3.0f;
constexpr const char *kDockspaceWindowName = "NuriDockspace";
constexpr const char *kDockspaceRootId = "NuriDockspace##Root";
constexpr const char *kLogWindowName = "Log";

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
    constexpr std::string_view tags[] = {"[Trace]", "[Debug]", "[Info]",
                                         "[Warn]", "[Fatal]"};
    constexpr LogLevel levels[] = {LogLevel::Trace, LogLevel::Debug,
                                   LogLevel::Info, LogLevel::Warning,
                                   LogLevel::Fatal};

    for (size_t i = 0; i < std::size(tags); ++i) {
      if (line.size() >= tags[i].size() &&
          line.substr(0, tags[i].size()) == tags[i]) {
        std::string msg(line.substr(tags[i].size()));
        if (!msg.empty() && msg.front() == ' ') {
          msg.erase(0, 1);
        }
        return {levels[i], std::move(msg)};
      }
    }
    return {LogLevel::Info, std::string(line)};
  }

  void seedFromFileIfNeeded(LogFilterState &filterState) {
    if (seededFromFile || !lines.empty() || lastSequence != 0) {
      return;
    }

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
    seededFromFile = true;
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
    std::vector<LogEntry> entries;
    const LogReadResult result = readLogEntriesSince(lastSequence, entries);
    if (result.truncated) {
      lines.clear();
      lastSequence = result.lastSequence;
    }
    if (entries.empty()) {
      seedFromFileIfNeeded(filterState);
    } else {
      lastSequence = result.lastSequence;
      appendEntries(entries);
      filterState.requestScroll = true;
    }
  }
};

struct LogStyle {
  const char *tag;
  ImVec4 color;
};

struct LogStyleCatalog {
  static LogStyle styleFor(LogLevel level) {
    switch (level) {
    case LogLevel::Trace:
      return {"[Trace]", ImVec4(0.7f, 0.7f, 0.7f, 1.0f)};
    case LogLevel::Debug:
      return {"[Debug]", ImVec4(0.4f, 0.8f, 0.9f, 1.0f)};
    case LogLevel::Info:
      return {"[Info]", ImVec4(0.9f, 0.9f, 0.9f, 1.0f)};
    case LogLevel::Warning:
      return {"[Warn]", ImVec4(1.0f, 0.8f, 0.2f, 1.0f)};
    case LogLevel::Fatal:
      return {"[Fatal]", ImVec4(1.0f, 0.3f, 0.3f, 1.0f)};
    }
    return {"[Info]", ImVec4(0.9f, 0.9f, 0.9f, 1.0f)};
  }

  static ImU32 textColorFor(const ImVec4 &background) {
    const float luma =
        0.299f * background.x + 0.587f * background.y + 0.114f * background.z;
    const ImVec4 foreground = luma > 0.6f ? ImVec4(0.0f, 0.0f, 0.0f, 1.0f)
                                          : ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    return ImGui::ColorConvertFloat4ToU32(foreground);
  }

  static void drawTagPill(LogLevel level) {
    const LogStyle style = styleFor(level);
    const ImU32 backgroundColor = ImGui::ColorConvertFloat4ToU32(style.color);
    const ImU32 foregroundColor = textColorFor(style.color);

    const ImVec2 textSize = ImGui::CalcTextSize(style.tag);
    const ImGuiStyle &imguiStyle = ImGui::GetStyle();
    const ImVec2 padding(kTagHorizontalPadding,
                         std::max(2.0f, imguiStyle.FramePadding.y * 0.5f));
    const ImVec2 size(textSize.x + padding.x * 2.0f,
                      textSize.y + padding.y * 2.0f);

    const ImVec2 position = ImGui::GetCursorScreenPos();
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(position,
                            ImVec2(position.x + size.x, position.y + size.y),
                            backgroundColor, kTagCornerRounding);
    drawList->AddText(ImVec2(position.x + padding.x, position.y + padding.y),
                      foregroundColor, style.tag);
    ImGui::Dummy(size);
  }
};

void drawInlineItemGap() {
  ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
}

void drawInlineLevelToggle(const char *label, bool &value) {
  drawInlineItemGap();
  ImGui::Checkbox(label, &value);
}

struct LogWindowWidget {
  static void draw(LogModel &model, LogFilterState &filterState) {
    drawToolbar(model, filterState);
    ImGui::Separator();
    drawMessages(model, filterState);
  }

private:
  static void drawToolbar(LogModel &model, LogFilterState &filterState) {
    if (ImGui::Button("Clear")) {
      model.clear();
      filterState.requestScroll = true;
    }

    drawInlineItemGap();
    ImGui::Checkbox("Auto-scroll", &filterState.autoScroll);

    drawInlineItemGap();
    filterState.textFilter.Draw("Filter", kLogFilterWidth);

    drawInlineLevelToggle("Trace", filterState.showTrace);
    drawInlineLevelToggle("Debug", filterState.showDebug);
    drawInlineLevelToggle("Info", filterState.showInfo);
    drawInlineLevelToggle("Warn", filterState.showWarning);
    drawInlineLevelToggle("Fatal", filterState.showFatal);
  }

  static void drawMessages(const LogModel &model, LogFilterState &filterState) {
    std::vector<size_t> visibleIndices;
    visibleIndices.reserve(model.lines.size());
    for (size_t i = 0; i < model.lines.size(); ++i) {
      const auto &line = model.lines[i];
      if (!filterState.levelEnabled(line.level)) {
        continue;
      }
      if (!filterState.textFilter.PassFilter(line.message.c_str())) {
        continue;
      }
      visibleIndices.push_back(i);
    }

    ImGui::BeginChild("LogScroll", ImVec2(0.0f, 0.0f), false,
                      ImGuiWindowFlags_HorizontalScrollbar);

    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(visibleIndices.size()));
    while (clipper.Step()) {
      for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
        const size_t lineIdx = visibleIndices[static_cast<size_t>(i)];
        const auto &line = model.lines[lineIdx];
        LogStyleCatalog::drawTagPill(line.level);
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
};

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

void drawFpsOverlay(const FPSCounter &fpsCounter) {
  if (const ImGuiViewport *viewport = ImGui::GetMainViewport()) {
    ImGui::SetNextWindowPos({viewport->WorkPos.x + viewport->WorkSize.x - 15.0f,
                             viewport->WorkPos.y + 15.0f},
                            ImGuiCond_Always, {1.0f, 0.0f});
  }
  ImGui::SetNextWindowBgAlpha(0.30f);
  ImGui::SetNextWindowSize(ImVec2(ImGui::CalcTextSize("FPS : _______").x, 0));
  if (ImGui::Begin("##FPS", nullptr,
                   ImGuiWindowFlags_NoDecoration |
                       ImGuiWindowFlags_AlwaysAutoResize |
                       ImGuiWindowFlags_NoSavedSettings |
                       ImGuiWindowFlags_NoFocusOnAppearing |
                       ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove)) {
    const float fps = fpsCounter.getFPS();
    const float milliseconds = fps > 0.0f ? 1000.0f / fps : 0.0f;
    ImGui::Text("FPS : %i", static_cast<int>(fps));
    ImGui::Text("Ms  : %.1f", milliseconds);
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

    logDockId = dockBottom;
    ImGui::DockBuilderDockWindow(kLogWindowName, dockBottom);
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
    logModel.update(logFilterState);

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
    LogWindowWidget::draw(logModel, logFilterState);
    ImGui::End();

    drawFpsOverlay(fpsCounter);

    ImGui::EndFrame();
    ImGui::Render();

    return renderer->buildRenderPass(gpu.getSwapchainFormat());
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
  bool showMetricsWindow = false;
  double frameDeltaSeconds = 0.0;
  LogModel logModel;
  LogFilterState logFilterState;
  MaybeDockLayoutState dockLayoutState;
};

std::unique_ptr<ImGuiEditor> ImGuiEditor::create(Window &window,
                                                 GPUDevice &gpu) {
  return std::unique_ptr<ImGuiEditor>(new ImGuiEditor(window, gpu));
}

ImGuiEditor::ImGuiEditor(Window &window, GPUDevice &gpu)
    : impl_(std::make_unique<Impl>(window, gpu)) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();

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
  impl_->renderer.reset();
  impl_->platform.reset();
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

void ImGuiEditor::beginFrame() { impl_->beginFrame(); }

Result<RenderPass, std::string> ImGuiEditor::endFrame() {
  return impl_->endFrame();
}

} // namespace nuri
