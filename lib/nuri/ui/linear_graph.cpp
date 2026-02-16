#include "nuri/pch.h"

#include "nuri/ui/linear_graph.h"

namespace nuri {

namespace {

constexpr float kDefaultPlotWidthPixels = 220.0f;

const char *cacheText(std::pmr::string &buffer, std::string_view text) {
  buffer.assign(text.data(), text.size());
  return buffer.c_str();
}

class ImPlotLinearGraph final : public LinearGraph {
public:
  explicit ImPlotLinearGraph(std::size_t capacity)
      : capacity_(std::clamp<std::size_t>(capacity, 1,
                                          static_cast<std::size_t>(INT_MAX))),
        samples_(&memoryResource_), contiguousSamples_(&memoryResource_),
        plotIdBuffer_(&memoryResource_), seriesLabelBuffer_(&memoryResource_),
        shadedLabelBuffer_(&memoryResource_) {
    samples_.resize(capacity_, 0.0f);
    contiguousSamples_.resize(capacity_, 0.0f);
  }

  void pushSample(float value) override {
    if (!std::isfinite(value)) {
      value = 0.0f;
    }

    samples_[writeIndex_] = value;
    writeIndex_ = (writeIndex_ + 1) % capacity_;
    if (count_ < capacity_) {
      ++count_;
    }
  }

  void clear() override {
    writeIndex_ = 0;
    count_ = 0;
  }

  void draw(std::string_view plotId, std::string_view seriesLabel,
            const LinearGraphStyle &style) override {
    const char *plotIdCStr = cacheText(plotIdBuffer_, plotId);
    const char *seriesLabelCStr = cacheText(seriesLabelBuffer_, seriesLabel);
    const float plotHeight = std::max(style.heightPixels, 1.0f);
    float plotWidth = ImGui::GetContentRegionAvail().x;
    if (plotWidth < 1.0f) {
      plotWidth = kDefaultPlotWidthPixels;
    }

    constexpr ImPlotFlags kPlotFlags = ImPlotFlags_CanvasOnly;
    constexpr ImPlotAxisFlags kAxisFlags =
        ImPlotAxisFlags_NoDecorations | ImPlotAxisFlags_AutoFit;

    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0.0f, 0.0f));
    if (!ImPlot::BeginPlot(plotIdCStr, ImVec2(plotWidth, plotHeight),
                           kPlotFlags)) {
      ImPlot::PopStyleVar();
      return;
    }

    ImPlot::SetupAxes(nullptr, nullptr, kAxisFlags, kAxisFlags);
    if (count_ > 0) {
      updateContiguousSamples();

      ImPlot::PushStyleColor(ImPlotCol_Line, style.lineColorRgba);
      ImPlot::PlotLine(seriesLabelCStr, contiguousSamples_.data(),
                       static_cast<int>(count_));
      ImPlot::PopStyleColor();

      if (style.fillUnderLine) {
        shadedLabelBuffer_.assign(seriesLabel.data(), seriesLabel.size());
        shadedLabelBuffer_.append("##fill");
        ImVec4 fillColor = ImGui::ColorConvertU32ToFloat4(style.lineColorRgba);
        fillColor.w *= 0.25f;
        ImPlot::PushStyleColor(ImPlotCol_Fill, fillColor);
        ImPlot::PlotShaded(shadedLabelBuffer_.c_str(),
                           contiguousSamples_.data(), static_cast<int>(count_),
                           0.0);
        ImPlot::PopStyleColor();
      }
    }

    ImPlot::EndPlot();
    ImPlot::PopStyleVar();
  }

private:
  void updateContiguousSamples() {
    const std::size_t startIndex = (count_ == capacity_) ? writeIndex_ : 0;
    const std::size_t firstChunk = std::min(count_, capacity_ - startIndex);
    std::copy_n(samples_.data() + startIndex, firstChunk,
                contiguousSamples_.data());

    const std::size_t secondChunk = count_ - firstChunk;
    if (secondChunk > 0) {
      std::copy_n(samples_.data(), secondChunk,
                  contiguousSamples_.data() + firstChunk);
    }
  }

  std::pmr::unsynchronized_pool_resource memoryResource_{};
  std::size_t capacity_ = 0;
  std::size_t writeIndex_ = 0;
  std::size_t count_ = 0;
  std::pmr::vector<float> samples_;
  std::pmr::vector<float> contiguousSamples_;
  std::pmr::string plotIdBuffer_;
  std::pmr::string seriesLabelBuffer_;
  std::pmr::string shadedLabelBuffer_;
};

} // namespace

std::unique_ptr<LinearGraph> createImPlotLinearGraph(std::size_t capacity) {
  return std::make_unique<ImPlotLinearGraph>(capacity);
}

} // namespace nuri
