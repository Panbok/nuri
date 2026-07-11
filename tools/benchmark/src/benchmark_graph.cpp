#include "nuri/tools/benchmark/benchmark_graph.h"

#include "nuri/tools/core/html_report.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string_view>

namespace nuri::tools::benchmark {
namespace {

[[nodiscard]] std::string escapeHtml(std::string_view text) {
  return nuri::tools::core::htmlEscape(text);
}

[[nodiscard]] double statisticValue(const MetricStats &stats,
                                    std::string_view statistic) {
  if (statistic == "min") {
    return stats.min;
  }
  if (statistic == "median") {
    return stats.median;
  }
  if (statistic == "p90") {
    return stats.p90;
  }
  if (statistic == "p95") {
    return stats.p95;
  }
  if (statistic == "p99") {
    return stats.p99;
  }
  if (statistic == "max") {
    return stats.max;
  }
  if (statistic == "mean") {
    return stats.mean;
  }
  if (statistic == "stddev") {
    return stats.stddev;
  }
  if (statistic == "mad") {
    return stats.mad;
  }
  if (statistic == "iqr") {
    return stats.iqr;
  }
  if (statistic == "cv" || statistic == "coefficientOfVariation") {
    return stats.coefficientOfVariation;
  }
  return std::numeric_limits<double>::quiet_NaN();
}

[[nodiscard]] bool validStatistic(std::string_view statistic) {
  static constexpr std::array values{
      std::string_view("min"),  std::string_view("median"),
      std::string_view("p90"),  std::string_view("p95"),
      std::string_view("p99"),  std::string_view("max"),
      std::string_view("mean"), std::string_view("stddev"),
      std::string_view("mad"),  std::string_view("iqr"),
      std::string_view("cv"),   std::string_view("coefficientOfVariation"),
  };
  return std::find(values.begin(), values.end(), statistic) != values.end();
}

[[nodiscard]] bool metricLooksIntegral(std::string_view metric) {
  return metric.ends_with("_count") || metric.ends_with("_counts") ||
         metric.ends_with("_draws") || metric.ends_with("_passes") ||
         metric.ends_with("_instances") || metric.ends_with("_commands") ||
         metric.ends_with("_dispatches") || metric.ends_with("_cascades") ||
         metric.ends_with("cascades") || metric.ends_with("_textures") ||
         metric.ends_with("_allocations") ||
         metric.ends_with("_reallocations") || metric.ends_with("_entries") ||
         metric.ends_with("_budget") || metric.ends_with("_levels") ||
         metric.ends_with("_calls") || metric.ends_with("_size") ||
         metric.ends_with("_frames");
}

[[nodiscard]] std::string metricUnit(std::string_view metric) {
  if (metric.ends_with("_ms")) {
    return " ms";
  }
  if (metric.ends_with("_mb")) {
    return " MiB";
  }
  if (metric.ends_with("_percent")) {
    return "%";
  }
  if (metric.ends_with("_ratio")) {
    return "";
  }
  return "";
}

[[nodiscard]] std::string formatMetricValue(std::string_view metric,
                                            double value) {
  if (!std::isfinite(value)) {
    return "missing";
  }
  std::ostringstream out;
  if (metricLooksIntegral(metric)) {
    out << std::fixed << std::setprecision(0) << value;
  } else if (metric.ends_with("_mb") && value >= 100.0) {
    out << std::fixed << std::setprecision(1) << value;
  } else {
    out << std::fixed << std::setprecision(3) << value;
  }
  out << metricUnit(metric);
  return out.str();
}

[[nodiscard]] std::string metricCategory(std::string_view metric) {
  if (metric.starts_with("cpu.")) {
    return "CPU timings";
  }
  if (metric == "gpu.scopes_sum_ms" || metric.starts_with("gpu.scopes.")) {
    return "GPU pass timings";
  }
  if (metric.starts_with("rendergraph.pass.")) {
    return "Render graph pass timings";
  }
  if (metric.starts_with("rendergraph.summary.")) {
    return "Render graph structure";
  }
  if (metric.starts_with("memory.process.")) {
    return "Process memory";
  }
  if (metric.starts_with("memory.pmr.")) {
    return "Benchmark PMR pools";
  }
  if (metric.starts_with("gpu.memory.")) {
    return "GPU frame memory estimates";
  }
  if (metric.starts_with("renderer.")) {
    return "Renderer work metrics";
  }
  return "Other metrics";
}

[[nodiscard]] std::string metricCategoryClass(std::string_view category) {
  if (category == "CPU timings") {
    return "cat-cpu";
  }
  if (category == "GPU pass timings") {
    return "cat-gpu";
  }
  if (category == "Render graph pass timings") {
    return "cat-rg-pass";
  }
  if (category == "Render graph structure") {
    return "cat-rg";
  }
  if (category == "Process memory") {
    return "cat-memory";
  }
  if (category == "Benchmark PMR pools") {
    return "cat-pmr";
  }
  if (category == "GPU frame memory estimates") {
    return "cat-gpu-memory";
  }
  if (category == "Renderer work metrics") {
    return "cat-renderer";
  }
  return "cat-other";
}

[[nodiscard]] std::string metricCategoryDescription(std::string_view category) {
  if (category == "CPU timings") {
    return "Wall-clock CPU intervals around benchmark frame work.";
  }
  if (category == "GPU pass timings") {
    return "Resolved GPU timer scopes for renderer passes.";
  }
  if (category == "Render graph pass timings") {
    return "Per render-graph pass CPU recording and GPU execution timing.";
  }
  if (category == "Render graph structure") {
    return "Render graph pass, barrier, submission, and transient resource "
           "counts.";
  }
  if (category == "Process memory") {
    return "OS process memory sampled after measured frames.";
  }
  if (category == "Benchmark PMR pools") {
    return "Upstream bytes reserved by renderer, pipeline, and scene PMR "
           "pools.";
  }
  if (category == "GPU frame memory estimates") {
    return "Renderer-reported frame texture memory estimates.";
  }
  if (category == "Renderer work metrics") {
    return "Draw, pass, dispatch, allocation, and resource counters.";
  }
  return "Additional numeric measurements from the selected reports.";
}

[[nodiscard]] int metricCategoryRank(std::string_view metric) {
  const std::string category = metricCategory(metric);
  if (category == "CPU timings") {
    return 0;
  }
  if (category == "GPU pass timings") {
    return 1;
  }
  if (category == "Render graph pass timings") {
    return 2;
  }
  if (category == "Render graph structure") {
    return 3;
  }
  if (category == "Process memory") {
    return 4;
  }
  if (category == "Benchmark PMR pools") {
    return 5;
  }
  if (category == "GPU frame memory estimates") {
    return 6;
  }
  if (category == "Renderer work metrics") {
    return 7;
  }
  return 8;
}

[[nodiscard]] bool metricLess(const std::string &lhs, const std::string &rhs) {
  const int lhsRank = metricCategoryRank(lhs);
  const int rhsRank = metricCategoryRank(rhs);
  if (lhsRank != rhsRank) {
    return lhsRank < rhsRank;
  }
  return lhs < rhs;
}

[[nodiscard]] std::string reportLabel(const BenchmarkReport &report,
                                      size_t index) {
  if (!report.benchmarkCase.id.empty()) {
    return report.benchmarkCase.id;
  }
  return "report " + std::to_string(index + 1u);
}

[[nodiscard]] std::vector<std::string>
makeReportLabels(std::span<const BenchmarkReport> reports) {
  std::map<std::string, uint32_t> counts;
  for (size_t i = 0; i < reports.size(); ++i) {
    ++counts[reportLabel(reports[i], i)];
  }

  std::map<std::string, uint32_t> seen;
  std::vector<std::string> labels;
  labels.reserve(reports.size());
  for (size_t i = 0; i < reports.size(); ++i) {
    std::string label = reportLabel(reports[i], i);
    if (counts[label] > 1u) {
      const uint32_t ordinal = ++seen[label];
      label += " #" + std::to_string(ordinal);
    }
    labels.push_back(std::move(label));
  }
  return labels;
}

[[nodiscard]] std::vector<std::string>
selectMetrics(std::span<const BenchmarkReport> reports,
              const BenchmarkGraphOptions &options) {
  if (!options.metrics.empty()) {
    return options.metrics;
  }

  std::set<std::string> available;
  for (const BenchmarkReport &report : reports) {
    for (const auto &[metricId, stats] : report.stats) {
      (void)stats;
      if (metricId.starts_with("rendergraph.pass.")) {
        continue;
      }
      available.insert(metricId);
    }
  }

  std::vector<std::string> selected(available.begin(), available.end());
  std::sort(selected.begin(), selected.end(), metricLess);
  return selected;
}

[[nodiscard]] std::vector<std::string>
selectStatistics(const BenchmarkGraphOptions &options) {
  if (options.statistics.empty()) {
    return {"median", "p95", "p99"};
  }
  return options.statistics;
}

struct PassTimingMetricKey {
  uint32_t orderedPassIndex = UINT32_MAX;
  std::string slug{};
};

struct PassTimingRow {
  std::string reportLabel{};
  uint32_t orderedPassIndex = UINT32_MAX;
  std::string passName{};
  double cpuMedian = std::numeric_limits<double>::quiet_NaN();
  double cpuP95 = std::numeric_limits<double>::quiet_NaN();
  double gpuMedian = std::numeric_limits<double>::quiet_NaN();
  double gpuP95 = std::numeric_limits<double>::quiet_NaN();
};

struct TracyZoneRow {
  std::string reportLabel{};
  BenchmarkTracyZoneStats zone{};
};

[[nodiscard]] bool parsePassTimingMetric(std::string_view metric,
                                         PassTimingMetricKey &key, bool &cpu) {
  constexpr std::string_view kPrefix = "rendergraph.pass.";
  if (!metric.starts_with(kPrefix)) {
    return false;
  }
  std::string_view rest = metric.substr(kPrefix.size());
  const size_t firstDot = rest.find('.');
  const size_t lastDot = rest.rfind('.');
  if (firstDot == std::string_view::npos || lastDot == std::string_view::npos ||
      firstDot == lastDot) {
    return false;
  }
  const std::string_view indexText = rest.substr(0u, firstDot);
  const std::string_view suffix = rest.substr(lastDot + 1u);
  if (suffix == "cpu_ms") {
    cpu = true;
  } else if (suffix == "gpu_ms") {
    cpu = false;
  } else {
    return false;
  }
  uint32_t index = 0u;
  for (const char c : indexText) {
    if (c < '0' || c > '9') {
      return false;
    }
    index = index * 10u + static_cast<uint32_t>(c - '0');
  }
  key.orderedPassIndex = index;
  key.slug.assign(rest.substr(firstDot + 1u, lastDot - firstDot - 1u));
  return true;
}

[[nodiscard]] std::string prettyPassName(std::string_view slug) {
  std::string out(slug);
  for (char &c : out) {
    if (c == '_') {
      c = ' ';
    }
  }
  return out.empty() ? std::string("unnamed pass") : out;
}

[[nodiscard]] double finiteMax(std::initializer_list<double> values) {
  double out = 0.0;
  for (const double value : values) {
    if (std::isfinite(value)) {
      out = std::max(out, value);
    }
  }
  return out;
}

[[nodiscard]] std::vector<PassTimingRow>
collectPassTimingRows(std::span<const BenchmarkReport> reports,
                      std::span<const std::string> labels) {
  std::vector<PassTimingRow> rows;
  for (size_t reportIndex = 0u; reportIndex < reports.size(); ++reportIndex) {
    std::map<std::pair<uint32_t, std::string>, PassTimingRow> reportRows;
    for (const auto &[metricId, stats] : reports[reportIndex].stats) {
      PassTimingMetricKey key{};
      bool cpu = false;
      if (!parsePassTimingMetric(metricId, key, cpu)) {
        continue;
      }
      PassTimingRow &row =
          reportRows[std::make_pair(key.orderedPassIndex, key.slug)];
      row.reportLabel = labels[reportIndex];
      row.orderedPassIndex = key.orderedPassIndex;
      row.passName = prettyPassName(key.slug);
      if (cpu) {
        row.cpuMedian = stats.median;
        row.cpuP95 = stats.p95;
      } else {
        row.gpuMedian = stats.median;
        row.gpuP95 = stats.p95;
      }
    }
    for (auto &[key, row] : reportRows) {
      (void)key;
      rows.push_back(std::move(row));
    }
  }
  std::sort(rows.begin(), rows.end(),
            [](const PassTimingRow &lhs, const PassTimingRow &rhs) {
              const double lhsMax = finiteMax(
                  {lhs.cpuP95, lhs.gpuP95, lhs.cpuMedian, lhs.gpuMedian});
              const double rhsMax = finiteMax(
                  {rhs.cpuP95, rhs.gpuP95, rhs.cpuMedian, rhs.gpuMedian});
              if (lhsMax != rhsMax) {
                return lhsMax > rhsMax;
              }
              if (lhs.reportLabel != rhs.reportLabel) {
                return lhs.reportLabel < rhs.reportLabel;
              }
              return lhs.orderedPassIndex < rhs.orderedPassIndex;
            });
  return rows;
}

[[nodiscard]] std::string formatNsAsMs(double ns) {
  return formatMetricValue("tracy.zone_ms", ns / 1'000'000.0);
}

[[nodiscard]] std::string formatPercent(double value) {
  if (!std::isfinite(value)) {
    return "missing";
  }
  std::ostringstream out;
  out << std::fixed << std::setprecision(2) << value << "%";
  return out.str();
}

[[nodiscard]] std::string formatCount(uint64_t value) {
  std::ostringstream out;
  out << value;
  return out.str();
}

[[nodiscard]] std::string fileHref(const std::filesystem::path &path) {
  const std::string absolute = std::filesystem::absolute(path).generic_string();
#if defined(_WIN32)
  return "file:///" + absolute;
#else
  return "file://" + absolute;
#endif
}

[[nodiscard]] std::string artifactLink(const std::filesystem::path &path,
                                       std::string_view shortLabel = {}) {
  if (path.empty()) {
    return "missing";
  }
  const std::string label =
      shortLabel.empty()
          ? (path.filename().empty() ? path.generic_string()
                                     : path.filename().generic_string())
          : std::string(shortLabel);
  return "<a href=\"" + escapeHtml(fileHref(path)) + "\">" + escapeHtml(label) +
         "</a>";
}

[[nodiscard]] std::string sourceLocation(const BenchmarkTracyZoneStats &zone) {
  if (zone.sourceFile.empty()) {
    return "";
  }
  std::string text = zone.sourceFile.filename().generic_string();
  if (zone.sourceLine > 0u) {
    text += ":" + std::to_string(zone.sourceLine);
  }
  return text;
}

[[nodiscard]] std::string
flameSourceLocation(const BenchmarkTracyFlameNode &node) {
  if (node.sourceFile.empty()) {
    return "";
  }
  std::string text = node.sourceFile.filename().generic_string();
  if (node.sourceLine > 0u) {
    text += ":" + std::to_string(node.sourceLine);
  }
  return text;
}

[[nodiscard]] uint32_t stableHash(std::string_view text) {
  uint32_t hash = 2166136261u;
  for (const char c : text) {
    hash ^= static_cast<unsigned char>(c);
    hash *= 16777619u;
  }
  return hash;
}

[[nodiscard]] std::string flameColor(std::string_view name) {
  const uint32_t hue = stableHash(name) % 360u;
  return "hsl(" + std::to_string(hue) + ",68%,62%)";
}

[[nodiscard]] std::string flameNodeTitle(const BenchmarkTracyFlameNode &node) {
  std::ostringstream title;
  title << node.name;
  if (!node.thread.empty()) {
    title << "\nThread: " << node.thread;
  }
  const std::string source = flameSourceLocation(node);
  if (!source.empty()) {
    title << "\nSource: " << source;
  }
  title << "\nTotal: " << formatNsAsMs(static_cast<double>(node.totalNs))
        << "\nSelf: " << formatNsAsMs(static_cast<double>(node.selfNs))
        << "\nCount: " << node.count;
  return title.str();
}

void appendFlameNodeSvg(std::ostringstream &out,
                        const BenchmarkTracyFlameNode &node, uint32_t depth,
                        double x, double width, uint32_t maxDepth,
                        size_t svgIndex, uint64_t &nodeIndex) {
  if (width <= 0.05 || node.totalNs == 0u) {
    return;
  }

  constexpr double kRowHeight = 23.0;
  constexpr double kRectInset = 1.0;
  const double y =
      static_cast<double>(maxDepth - std::min(depth, maxDepth)) * kRowHeight;
  const double rectHeight = kRowHeight - kRectInset;
  const uint64_t currentNode = nodeIndex++;
  const std::string clipId =
      "fg_clip_" + std::to_string(svgIndex) + "_" + std::to_string(currentNode);

  const std::string nodeTitle = flameNodeTitle(node);
  out << "<g class=\"flame-node\" tabindex=\""
      << (currentNode == 0u ? "0" : "-1")
      << "\" role=\"button\" aria-describedby=\"tracy-flame-help-" << svgIndex
      << "\" aria-label=\"" << escapeHtml(nodeTitle) << "\" data-name=\""
      << escapeHtml(node.name) << "\" data-thread=\"" << escapeHtml(node.thread)
      << "\" data-source=\"" << escapeHtml(flameSourceLocation(node))
      << "\" data-total=\""
      << escapeHtml(formatNsAsMs(static_cast<double>(node.totalNs)))
      << "\" data-self=\""
      << escapeHtml(formatNsAsMs(static_cast<double>(node.selfNs)))
      << "\" data-count=\"" << node.count << "\" data-x=\"" << std::fixed
      << std::setprecision(2) << x << "\" data-width=\"" << width
      << "\" data-depth=\"" << depth << "\"><title>" << escapeHtml(nodeTitle)
      << "</title><clipPath id=\"" << clipId << "\"><rect x=\"" << std::fixed
      << std::setprecision(2) << x << "\" y=\"" << y << "\" width=\"" << width
      << "\" height=\"" << rectHeight
      << "\"/></clipPath><rect class=\"flame-rect\" x=\"" << x << "\" y=\"" << y
      << "\" width=\"" << width << "\" height=\"" << rectHeight << "\" fill=\""
      << flameColor(node.name) << "\"/>";
  if (width >= 38.0) {
    out << "<text class=\"flame-label\" x=\"" << x + 4.0 << "\" y=\""
        << y + 15.0 << "\" clip-path=\"url(#" << clipId << ")\">"
        << escapeHtml(node.name) << "</text>";
  }
  out << "</g>";

  double childX = x;
  for (const BenchmarkTracyFlameNode &child : node.children) {
    const double childWidth = width * (static_cast<double>(child.totalNs) /
                                       static_cast<double>(node.totalNs));
    appendFlameNodeSvg(out, child, depth + 1u, childX, childWidth, maxDepth,
                       svgIndex, nodeIndex);
    childX += childWidth;
  }
}

void appendTracyFlameGraphs(std::ostringstream &out,
                            std::span<const BenchmarkReport> reports,
                            std::span<const std::string> labels) {
  bool hasFlameGraph = false;
  for (const BenchmarkReport &report : reports) {
    hasFlameGraph = hasFlameGraph || report.tracy.flameGraph.root.totalNs > 0u;
  }
  if (!hasFlameGraph) {
    out << "<h3>Tracy Flame Graph</h3><p class=\"empty\">No Tracy flame graph "
           "events were exported.</p>";
    return;
  }

  out << "<h3>Tracy Flame Graph</h3><p>Width represents inclusive time; "
         "vertical depth represents the call stack. Select a zone to inspect "
         "it and zoom horizontally.</p>";
  for (size_t i = 0u; i < reports.size(); ++i) {
    const BenchmarkTracyFlameGraph &flameGraph = reports[i].tracy.flameGraph;
    if (flameGraph.root.totalNs == 0u) {
      continue;
    }

    const uint32_t maxDepth = std::max<uint32_t>(flameGraph.maxDepth, 1u);
    const double height = (static_cast<double>(maxDepth) + 1.0) * 23.0;
    out << "<article class=\"flame-report\"><div "
           "class=\"flame-report-head\"><b>"
        << escapeHtml(labels[i]) << "</b><span>"
        << (flameGraph.frameScoped ? "BenchmarkFrame scoped" : "full capture")
        << " | " << flameGraph.retainedNodeCount << " nodes | "
        << artifactLink(flameGraph.eventsCsvPath)
        << "</span></div><div class=\"flame-toolbar\" role=\"search\" "
           "aria-label=\"Filter Tracy zones for "
        << escapeHtml(labels[i])
        << "\"><div class=\"control-group\"><label "
           "for=\"tracy-flame-search-"
        << i << "\">Find zone</label><input id=\"tracy-flame-search-" << i
        << "\" type=\"search\" placeholder=\"e.g. RenderGraph, shadow, "
           "submit\"></div><button type=\"button\" "
           "data-action=\"flame-reset\">Reset zoom</button><p "
           "class=\"results-count\" id=\"tracy-flame-results-"
        << i
        << "\" aria-live=\"polite\"></p></div><p class=\"flame-help\" "
           "id=\"tracy-flame-help-"
        << i
        << "\">Use arrow keys to move between zones, Enter or Space to zoom, "
           "and Escape to reset.</p><div class=\"flame-canvas\">"
        << "<svg class=\"tracy-flame-svg\" viewBox=\"0 0 1200 " << std::fixed
        << std::setprecision(2) << height << "\" data-base-viewbox=\"0 0 1200 "
        << height
        << "\" role=\"group\" aria-label=\"Interactive Tracy flame graph for "
        << escapeHtml(labels[i]) << "\">";
    uint64_t nodeIndex = 0u;
    appendFlameNodeSvg(out, flameGraph.root, 0u, 0.0, 1200.0, maxDepth, i,
                       nodeIndex);
    out << "</svg></div><aside class=\"flame-detail\" "
           "id=\"tracy-flame-detail-"
        << i
        << "\" aria-live=\"polite\"><strong>Zone details</strong><p>Focus or "
           "select a flame block to inspect its inclusive time, self time, "
           "count, thread, and source.</p></aside></article>";
  }
}

[[nodiscard]] std::vector<TracyZoneRow>
collectTracyZoneRows(std::span<const BenchmarkReport> reports,
                     std::span<const std::string> labels, bool selfTime) {
  std::vector<TracyZoneRow> rows;
  for (size_t reportIndex = 0u; reportIndex < reports.size(); ++reportIndex) {
    const std::vector<BenchmarkTracyZoneStats> &zones =
        selfTime ? reports[reportIndex].tracy.selfZones
                 : reports[reportIndex].tracy.zones;
    for (const BenchmarkTracyZoneStats &zone : zones) {
      rows.push_back(
          TracyZoneRow{.reportLabel = labels[reportIndex], .zone = zone});
    }
  }
  std::sort(rows.begin(), rows.end(),
            [](const TracyZoneRow &lhs, const TracyZoneRow &rhs) {
              if (lhs.zone.totalNs != rhs.zone.totalNs) {
                return lhs.zone.totalNs > rhs.zone.totalNs;
              }
              if (lhs.reportLabel != rhs.reportLabel) {
                return lhs.reportLabel < rhs.reportLabel;
              }
              return lhs.zone.name < rhs.zone.name;
            });
  constexpr size_t kMaxTracyRows = 80u;
  if (rows.size() > kMaxTracyRows) {
    rows.resize(kMaxTracyRows);
  }
  return rows;
}

void appendTracyZoneTable(std::ostringstream &out,
                          std::span<const BenchmarkReport> reports,
                          std::span<const std::string> labels, bool selfTime) {
  const std::vector<TracyZoneRow> rows =
      collectTracyZoneRows(reports, labels, selfTime);
  out << "<h3>"
      << (selfTime ? "Top Tracy Self-Time Zones" : "Top Tracy Inclusive Zones")
      << "</h3>";
  if (rows.empty()) {
    out << "<p class=\"empty\">No Tracy zone rows were exported.</p>";
    return;
  }

  out << "<div class=\"table-wrap tracy-table-wrap\"><table><thead><tr>"
         "<th scope=\"col\">Report</th><th scope=\"col\">Zone</th>"
         "<th scope=\"col\">Source</th><th scope=\"col\">Count</th>"
         "<th scope=\"col\">Total</th><th scope=\"col\">Mean</th>"
         "<th scope=\"col\">Max</th><th scope=\"col\">Share</th>"
         "</tr></thead><tbody>";
  for (const TracyZoneRow &row : rows) {
    out << "<tr><td>" << escapeHtml(row.reportLabel) << "</td><td>"
        << escapeHtml(row.zone.name) << "</td><td>"
        << escapeHtml(sourceLocation(row.zone)) << "</td><td>"
        << formatCount(row.zone.count) << "</td><td>"
        << formatNsAsMs(static_cast<double>(row.zone.totalNs)) << "</td><td>"
        << formatNsAsMs(row.zone.meanNs) << "</td><td>"
        << formatNsAsMs(static_cast<double>(row.zone.maxNs)) << "</td><td>"
        << formatPercent(row.zone.totalPercent) << "</td></tr>";
  }
  out << "</tbody></table></div>";
}

void appendReportTable(std::ostringstream &out,
                       std::span<const BenchmarkReport> reports,
                       std::span<const std::string> labels) {
  out << "<section class=\"panel\" id=\"reports\"><div "
         "class=\"section-heading\">"
         "<div><h2>Input reports</h2><p>Environment and collection validity "
         "for "
         "each plotted run.</p></div><span class=\"section-count\">"
      << reports.size()
      << " reports</span></div><div class=\"table-wrap report-table-wrap\">"
         "<table><caption>Benchmark inputs and comparison readiness</caption>"
         "<thead><tr><th scope=\"col\">Report</th><th scope=\"col\">Case</th>"
         "<th scope=\"col\">Resolution</th><th scope=\"col\">Backend</th>"
         "<th scope=\"col\">Build</th><th scope=\"col\">Frames</th>"
         "<th scope=\"col\">Tracy</th><th scope=\"col\">GPU drain</th>"
         "<th scope=\"col\">Comparison</th></tr></thead><tbody>";
  for (size_t i = 0; i < reports.size(); ++i) {
    const BenchmarkReport &report = reports[i];
    out << "<tr><td>" << escapeHtml(labels[i]) << "</td><td>"
        << escapeHtml(report.benchmarkCase.id) << "</td><td>"
        << report.benchmarkCase.resolution[0] << "x"
        << report.benchmarkCase.resolution[1] << "</td><td>"
        << escapeHtml(report.environment.gpuBackend) << "</td><td>"
        << escapeHtml(report.environment.buildType) << "</td><td>"
        << report.run.warmupFrames << " / " << report.run.measurementFrames
        << " / " << report.run.cooldownFrames << "</td><td>"
        << (report.environment.tracyEnabled ? "cpu" : "off")
        << (report.environment.tracyGpuEnabled ? "+gpu" : "") << "</td><td>"
        << (report.timingDrain.drainComplete ? "complete" : "timeout")
        << ", missing " << report.timingDrain.missingGpuTimingFrames
        << ", dropped " << report.timingDrain.droppedGpuTimingReports
        << "</td><td><span class=\"status-badge status-";
    if (report.run.validForComparison) {
      out << "pass\">ready";
    } else if (report.environment.tracyDiagnostic) {
      out << "warn\">diagnostic-only";
    } else {
      out << "fail\">not ready";
    }
    out << "</span></td></tr>";
  }
  out << "</tbody></table></div></section>";
}

void appendTracySection(std::ostringstream &out,
                        std::span<const BenchmarkReport> reports,
                        std::span<const std::string> labels) {
  bool hasTracy = false;
  for (const BenchmarkReport &report : reports) {
    hasTracy = hasTracy || report.tracy.available ||
               !report.artifacts.tracyArtifacts.empty() ||
               report.environment.tracyDiagnostic;
  }
  if (!hasTracy) {
    out << "<section class=\"panel tracy-panel\" id=\"tracy\"><div "
           "class=\"section-head\"><div><h2>Tracy diagnostics</h2><p>CPU/GPU "
           "call-stack evidence for targeted performance investigation.</p>"
           "</div><span class=\"status-badge status-unavailable\">not "
           "collected</span></div><div class=\"diagnostics tone-warn\" "
           "role=\"status\"><h3>Tracy diagnostics were not collected</h3>"
           "<p>Every input report was recorded with Tracy disabled. Metric "
           "charts remain valid, but there is no trace or flame graph to "
           "inspect.</p><p><strong>Collect it:</strong> build with the "
           "<code>cpu-gpu</code> profiling mode and run the case with "
           "<code>--tracy-diagnostic</code>.</p></div></section>";
    return;
  }

  out << "<section class=\"panel tracy-panel\" id=\"tracy\"><div "
         "class=\"section-head\">"
         "<div><h2>Tracy Capture Stats</h2><p>Diagnostic capture summary and "
         "top zones exported from tracy-csvexport. Inclusive rows include "
         "child-zone time; self-time rows remove child-zone time.</p></div>"
         "<span>diagnostic</span></div><div class=\"table-wrap\"><table "
         "class=\"tracy-summary-table\">"
         "<caption>Tracy diagnostic capture availability and exported "
         "artifacts</caption>"
         "<thead><tr><th scope=\"col\">Report</th><th scope=\"col\">Status</th>"
         "<th scope=\"col\">Frames</th><th scope=\"col\">Span</th>"
         "<th scope=\"col\">Zone events</th><th scope=\"col\">Inclusive "
         "rows</th>"
         "<th scope=\"col\">Self rows</th><th scope=\"col\">Artifacts</th>"
         "</tr></thead><tbody>";
  for (size_t i = 0u; i < reports.size(); ++i) {
    const BenchmarkReport &report = reports[i];
    out << "<tr><td>" << escapeHtml(labels[i])
        << "</td><td><span class=\"status-badge status-"
        << (report.tracy.available ? "pass\">captured"
                                   : "unavailable\">not available")
        << "</span>"
        << "</td><td>" << formatCount(report.tracy.captureFrameCount)
        << "</td><td>" << std::fixed << std::setprecision(2)
        << report.tracy.captureTimeSpanSeconds << " s</td><td>"
        << formatCount(report.tracy.captureZoneEventCount) << "</td><td>"
        << report.tracy.zones.size() << "</td><td>"
        << report.tracy.selfZones.size()
        << "</td><td><nav class=\"artifact-links\" aria-label=\"Tracy "
           "artifacts for "
        << escapeHtml(labels[i]) << "\">";
    bool hasArtifact = false;
    const auto appendArtifact = [&](const std::filesystem::path &path,
                                    std::string_view label) {
      if (path.empty()) {
        return;
      }
      hasArtifact = true;
      out << artifactLink(path, label);
    };
    appendArtifact(report.tracy.tracePath, "trace");
    appendArtifact(report.tracy.zonesCsvPath, "inclusive CSV");
    appendArtifact(report.tracy.selfZonesCsvPath, "self-time CSV");
    appendArtifact(report.tracy.flameGraph.eventsCsvPath, "flame events");
    if (!hasArtifact) {
      out << "<span class=\"muted\">not produced</span>";
    }
    out << "</nav></td></tr>";
  }
  out << "</tbody></table></div>";
  appendTracyFlameGraphs(out, reports, labels);
  appendTracyZoneTable(out, reports, labels, false);
  appendTracyZoneTable(out, reports, labels, true);
  out << "</section>";
}

void appendPassTimingSection(std::ostringstream &out,
                             std::span<const BenchmarkReport> reports,
                             std::span<const std::string> labels) {
  const std::vector<PassTimingRow> rows =
      collectPassTimingRows(reports, labels);
  if (rows.empty()) {
    return;
  }
  out << "<section class=\"panel pass-panel\" id=\"passes\"><div "
         "class=\"section-head\"><div>"
         "<h2>Render Graph Pass Timings</h2><p>Slowest measured pass rows "
         "across "
         "reports. CPU is command recording time; GPU is backend timer-query "
         "execution time when available.</p></div><span>"
      << rows.size()
      << " rows</span></div><div class=\"table-wrap pass-table-wrap\"><table>"
         "<caption>Slowest render-graph pass timings across all input "
         "reports</caption>"
         "<thead><tr><th scope=\"col\">Report</th><th scope=\"col\">Pass</th>"
         "<th scope=\"col\">Name</th><th scope=\"col\">CPU median</th>"
         "<th scope=\"col\">CPU p95</th><th scope=\"col\">GPU median</th>"
         "<th scope=\"col\">GPU p95</th></tr></thead><tbody>";
  for (const PassTimingRow &row : rows) {
    out << "<tr><td>" << escapeHtml(row.reportLabel) << "</td><td>"
        << row.orderedPassIndex << "</td><td>" << escapeHtml(row.passName)
        << "</td><td>"
        << formatMetricValue("rendergraph.pass.cpu_ms", row.cpuMedian)
        << "</td><td>"
        << formatMetricValue("rendergraph.pass.cpu_ms", row.cpuP95)
        << "</td><td>"
        << formatMetricValue("rendergraph.pass.gpu_ms", row.gpuMedian)
        << "</td><td>"
        << formatMetricValue("rendergraph.pass.gpu_ms", row.gpuP95)
        << "</td></tr>";
  }
  out << "</tbody></table></div></section>";
}

void appendMetricCard(std::ostringstream &out,
                      std::span<const BenchmarkReport> reports,
                      std::span<const std::string> labels,
                      std::string_view metric, std::string_view statistic) {
  std::vector<double> values;
  values.reserve(reports.size());
  double maxValue = 0.0;
  for (const BenchmarkReport &report : reports) {
    const auto it = report.stats.find(std::string(metric));
    const double value = it == report.stats.end()
                             ? std::numeric_limits<double>::quiet_NaN()
                             : statisticValue(it->second, statistic);
    values.push_back(value);
    if (std::isfinite(value)) {
      maxValue = std::max(maxValue, value);
    }
  }

  out << "<section class=\"metric-card\" data-metric=\"" << escapeHtml(metric)
      << "\" data-statistic=\"" << escapeHtml(statistic)
      << "\"><div class=\"metric-head\"><h3>" << escapeHtml(metric)
      << "</h3><span>" << escapeHtml(statistic) << "</span></div>";
  if (maxValue <= 0.0) {
    out << "<p class=\"empty\">No values available for this metric.</p>";
  } else {
    out << "<div class=\"bars\">";
    for (size_t i = 0; i < reports.size(); ++i) {
      const double value = values[i];
      const double percent =
          std::isfinite(value)
              ? std::clamp((value / maxValue) * 100.0, 0.0, 100.0)
              : 0.0;
      out << "<div class=\"bar-row\"><div class=\"bar-label\">"
          << escapeHtml(labels[i])
          << "</div><div class=\"bar-track\" role=\"img\" aria-label=\""
          << escapeHtml(labels[i]) << " " << escapeHtml(metric) << " "
          << escapeHtml(statistic) << ": "
          << escapeHtml(formatMetricValue(metric, value))
          << "\"><div class=\"bar-fill\" "
             "style=\"width:"
          << std::fixed << std::setprecision(2) << percent
          << "%\"></div></div><div class=\"bar-value\">"
          << formatMetricValue(metric, value) << "</div></div>";
    }
    out << "</div>";
  }
  out << "</section>";
}

void appendMetricSection(std::ostringstream &out,
                         std::span<const BenchmarkReport> reports,
                         std::span<const std::string> labels,
                         std::span<const std::string> metrics,
                         std::span<const std::string> statistics) {
  std::string currentCategory;
  bool open = false;
  for (const std::string &metric : metrics) {
    const std::string category = metricCategory(metric);
    if (category != currentCategory) {
      if (open) {
        out << "</div></section>";
      }
      currentCategory = category;
      open = true;
      out << "<section class=\"metric-section "
          << metricCategoryClass(currentCategory) << "\" data-category=\""
          << escapeHtml(currentCategory)
          << "\"><div class=\"section-head\"><div><h2>"
          << escapeHtml(currentCategory) << "</h2><p>"
          << escapeHtml(metricCategoryDescription(currentCategory))
          << "</p></div><span>" << escapeHtml(std::to_string(statistics.size()))
          << " stats</span></div><div class=\"metric-grid\">";
    }
    for (const std::string &statistic : statistics) {
      appendMetricCard(out, reports, labels, metric, statistic);
    }
  }
  if (open) {
    out << "</div></section>";
  }
}

} // namespace

Result<std::string, std::string>
writeBenchmarkGraphHtml(std::span<const BenchmarkReport> reports,
                        const BenchmarkGraphOptions &options) {
  if (reports.empty()) {
    return Result<std::string, std::string>::makeError(
        "writeBenchmarkGraphHtml: no reports provided");
  }

  const std::vector<std::string> metrics = selectMetrics(reports, options);
  const std::vector<std::string> statistics = selectStatistics(options);
  for (const std::string &statistic : statistics) {
    if (!validStatistic(statistic)) {
      return Result<std::string, std::string>::makeError(
          "writeBenchmarkGraphHtml: unsupported statistic '" + statistic + "'");
    }
  }

  const std::vector<std::string> labels = makeReportLabels(reports);
  size_t validReports = 0u;
  size_t tracyReports = 0u;
  size_t warningCount = 0u;
  for (const BenchmarkReport &report : reports) {
    validReports += report.run.validForComparison ? 1u : 0u;
    tracyReports += report.tracy.available ? 1u : 0u;
    warningCount += report.warnings.size();
  }
  std::ostringstream out;
  nuri::tools::core::beginHtmlReport(out, options.title, R"CSS(
.section-head{display:flex;align-items:flex-end;justify-content:space-between;gap:1rem;margin-bottom:.8rem}.section-head p{margin:.3rem 0 0;color:var(--muted)}.section-head>span{flex:0 0 auto;color:var(--muted);font-size:.82rem}
.metric-section{min-width:0;margin:1.4rem 0}.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,27rem),1fr));gap:.8rem}
.metric-card{padding:1rem}.metric-head{display:flex;align-items:flex-start;justify-content:space-between;gap:.8rem;margin-bottom:.8rem}.metric-head span{flex:0 0 auto;padding:.22rem .55rem;border-radius:999px;background:var(--accent);color:var(--accent-ink);font-size:.75rem;font-weight:850}
.bars{display:grid;gap:.65rem}.bar-row{display:grid;grid-template-columns:minmax(8rem,1.3fr) minmax(8rem,3fr) 7rem;gap:.65rem;align-items:center}.bar-label{overflow:hidden;color:var(--text);font-size:.8rem;text-overflow:ellipsis;white-space:nowrap}.bar-track{height:1.35rem;overflow:hidden;border:1px solid var(--line);border-radius:6px;background:var(--surface-raised)}.bar-fill{height:100%;border-radius:5px;background:linear-gradient(90deg,var(--accent),var(--link))}.bar-value{text-align:right;color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
.cat-cpu .bar-fill{background:linear-gradient(90deg,#f2a65a,var(--warn))}.cat-gpu .bar-fill{background:linear-gradient(90deg,var(--accent),#67e8f9)}.cat-rg-pass .bar-fill{background:linear-gradient(90deg,#f97316,var(--warn))}.cat-rg .bar-fill{background:linear-gradient(90deg,#38bdf8,var(--accent))}.cat-memory .bar-fill{background:linear-gradient(90deg,#ff91a4,#f2a65a)}.cat-pmr .bar-fill{background:linear-gradient(90deg,#c89bff,var(--link))}.cat-gpu-memory .bar-fill{background:linear-gradient(90deg,#67e8f9,var(--link))}.cat-renderer .bar-fill{background:linear-gradient(90deg,var(--link),#c89bff)}
.report-table-wrap table{min-width:86rem}.report-table-wrap th,.report-table-wrap td{white-space:nowrap}.report-table-wrap th:nth-child(-n+2),.report-table-wrap td:nth-child(-n+2){min-width:18rem;white-space:normal}.pass-table-wrap{max-height:40rem}.pass-table-wrap table{min-width:58rem}.pass-table-wrap td:nth-child(n+4){text-align:right;font-variant-numeric:tabular-nums}.tracy-panel h3{margin:1.2rem 0 .4rem}.tracy-table-wrap{max-height:40rem}.tracy-table-wrap table{min-width:64rem}.tracy-summary-table th,.tracy-summary-table td{white-space:nowrap}.tracy-summary-table td:first-child{min-width:16rem;white-space:normal}.artifact-links{display:flex;min-width:18rem;flex-wrap:wrap;gap:.4rem}.artifact-links a{padding:.2rem .45rem;border:1px solid var(--line);border-radius:999px;white-space:nowrap;text-decoration:none}.flame-report{margin-top:.7rem;overflow:hidden;border:1px solid var(--line);border-radius:var(--radius-small);background:#0d131a}.flame-report-head{display:flex;flex-wrap:wrap;justify-content:space-between;gap:.5rem 1rem;padding:.7rem .8rem;border-bottom:1px solid var(--line)}.flame-report-head span{color:var(--muted)}.flame-toolbar{display:flex;flex-wrap:wrap;align-items:end;gap:.65rem;padding:.75rem .8rem;border-bottom:1px solid var(--line);background:#111a24}.flame-toolbar .control-group{flex:1 1 18rem}.flame-toolbar .results-count{margin-left:auto}.flame-help{margin:0;padding:.55rem .8rem;color:var(--muted);font-size:.8rem}.flame-canvas{overflow:auto;border-block:1px solid var(--line);background:#0d131a}.tracy-flame-svg{display:block;min-width:64rem;width:100%;height:auto;background:#0d131a}.flame-node{cursor:pointer;transition:opacity .12s ease}.flame-rect{stroke:#0d131a;stroke-width:.8}.flame-label{fill:#081018;font-size:11px;font-weight:700;pointer-events:none}.flame-node:focus{outline:none}.flame-node:focus .flame-rect,.flame-node.is-selected .flame-rect{stroke:#fff;stroke-width:3}.flame-node.is-match .flame-rect{stroke:#f8d66d;stroke-width:3}.flame-node.is-muted{opacity:.16}.flame-detail{min-height:4.8rem;padding:.75rem .8rem;background:#111a24}.flame-detail p{margin:.35rem 0 0;color:var(--muted)}.flame-detail dl{display:flex;flex-wrap:wrap;gap:.35rem 1rem;margin:.45rem 0 0}.flame-detail dt{color:var(--muted);font-size:.74rem;font-weight:800;text-transform:uppercase}.flame-detail dd{margin:0;font-family:ui-monospace,SFMono-Regular,Consolas,monospace}.empty{color:var(--muted)}
@media(max-width:720px){.bar-row{grid-template-columns:1fr}.bar-value{text-align:left}.section-head{align-items:flex-start;flex-direction:column}}
)CSS");
  out << "<header class=\"report-header\"><p class=\"report-kicker\">Renderer "
         "performance report</p><div class=\"title-row\"><h1>"
      << escapeHtml(options.title) << "</h1><span class=\"status-badge status-"
      << (validReports == reports.size() ? "pass\">ready" : "warn\">review")
      << "</span></div><p class=\"lede\">Cross-run benchmark metrics, "
         "render-graph "
         "pass timings, and optional Tracy diagnostics. Timing values use "
         "milliseconds; memory values use MiB.</p><nav class=\"report-nav\" "
         "aria-label=\"Report sections\"><a href=\"#overview\">Overview</a>"
         "<a href=\"#reports\">Inputs</a><a href=\"#passes\">Pass timings</a>"
         "<a href=\"#tracy\">Tracy</a><a href=\"#metrics\">Metrics</a>"
         "<button type=\"button\" "
         "data-action=\"theme\" aria-pressed=\"false\">Light theme</button>"
         "</nav></header><main id=\"main-content\" tabindex=\"-1\">"
         "<section id=\"overview\" aria-labelledby=\"overview-title\"><div "
         "class=\"section-heading\"><div><h2 id=\"overview-title\">Run overview"
         "</h2><p>Collection readiness and report coverage.</p></div></div>"
         "<div class=\"summary-grid\"><div class=\"summary-card\"><strong>"
      << reports.size()
      << "</strong><span>input reports</span></div><div class=\"summary-card\">"
         "<strong class=\"tone-pass\">"
      << validReports
      << "</strong><span>comparison-ready</span></div><div "
         "class=\"summary-card\">"
         "<strong>"
      << metrics.size()
      << "</strong><span>metrics</span></div><div class=\"summary-card\">"
         "<strong>"
      << statistics.size()
      << "</strong><span>statistics per metric</span></div><div "
         "class=\"summary-card\"><strong>"
      << tracyReports
      << "</strong><span>Tracy captures</span></div><div "
         "class=\"summary-card\">"
         "<strong class=\"tone-warn\">"
      << warningCount
      << "</strong><span>collection warnings</span></div></div></section>";

  appendReportTable(out, reports, labels);
  appendPassTimingSection(out, reports, labels);
  appendTracySection(out, reports, labels);
  out << "<section id=\"metrics\" aria-labelledby=\"metrics-title\"><div "
         "class=\"section-heading\"><div><h2 id=\"metrics-title\">Metric "
         "comparison charts</h2><p>Each chart is normalized to its largest "
         "displayed value; exact values remain visible.</p></div><span "
         "class=\"section-count\">"
      << metrics.size() * statistics.size() << " charts</span></div>";
  if (metrics.empty()) {
    out << "<section class=\"panel\"><h2>No metrics</h2><p "
           "class=\"empty-state\">"
           "No matching metrics were found in the selected "
           "reports.</p></section>";
  } else {
    out << "<div class=\"toolbar\" role=\"search\" aria-label=\"Filter metric "
           "charts\"><div class=\"control-group\"><label for=\"metric-search\">"
           "Search metrics or categories</label><input id=\"metric-search\" "
           "type=\"search\" placeholder=\"e.g. gpu, memory, render_submit\">"
           "</div><div class=\"control-group\"><label for=\"metric-statistic\">"
           "Statistic</label><select id=\"metric-statistic\"><option "
           "value=\"\">"
           "All statistics</option>";
    for (const std::string &statistic : statistics) {
      out << "<option>" << escapeHtml(statistic) << "</option>";
    }
    out << "</select></div><p id=\"metric-results\" class=\"results-count\" "
           "aria-live=\"polite\"></p></div>";
    appendMetricSection(out, reports, labels, metrics, statistics);
  }
  out << "</section></main>";
  nuri::tools::core::endHtmlReport(out, R"JS(
(() => {
  const search = document.querySelector('#metric-search');
  const statistic = document.querySelector('#metric-statistic');
  const cards = [...document.querySelectorAll('.metric-card')];
  const sections = [...document.querySelectorAll('.metric-section')];
  const results = document.querySelector('#metric-results');
  const filter = () => {
    const query = (search?.value || '').trim().toLowerCase();
    const selected = statistic?.value || '';
    let visible = 0;
    cards.forEach((card) => {
      const category = card.closest('.metric-section')?.dataset.category || '';
      const show = (!query || `${card.dataset.metric} ${category}`.toLowerCase().includes(query)) &&
        (!selected || card.dataset.statistic === selected);
      card.hidden = !show;
      if (show) visible += 1;
    });
    sections.forEach((section) => {
      section.hidden = !section.querySelector('.metric-card:not([hidden])');
    });
    if (results) results.textContent = `${visible} of ${cards.length} charts shown`;
  };
  search?.addEventListener('input', filter);
  statistic?.addEventListener('change', filter);
  filter();

  document.querySelectorAll('.flame-report').forEach((report) => {
    const svg = report.querySelector('.tracy-flame-svg');
    const nodes = [...report.querySelectorAll('.flame-node')];
    const zoneSearch = report.querySelector('input[type="search"]');
    const zoneResults = report.querySelector('.flame-toolbar .results-count');
    const reset = report.querySelector('[data-action="flame-reset"]');
    const detail = report.querySelector('.flame-detail');
    if (!svg || !nodes.length || !detail) return;
    const baseViewBox = svg.dataset.baseViewbox;
    const baseHeight = Number(baseViewBox?.split(/\s+/)[3] || 1);
    const addDetail = (list, label, value) => {
      if (!value) return;
      const term = document.createElement('dt');
      term.textContent = label;
      const description = document.createElement('dd');
      description.textContent = value;
      list.append(term, description);
    };
    const showDetail = (node) => {
      detail.replaceChildren();
      const title = document.createElement('strong');
      title.textContent = node.dataset.name || 'Zone details';
      const list = document.createElement('dl');
      addDetail(list, 'Inclusive', node.dataset.total);
      addDetail(list, 'Self', node.dataset.self);
      addDetail(list, 'Calls', node.dataset.count);
      addDetail(list, 'Thread', node.dataset.thread);
      addDetail(list, 'Source', node.dataset.source);
      detail.append(title, list);
    };
    const setRovingFocus = (node) => {
      nodes.forEach((candidate) => candidate.setAttribute('tabindex', candidate === node ? '0' : '-1'));
    };
    const selectNode = (node, moveFocus) => {
      nodes.forEach((candidate) => candidate.classList.toggle('is-selected', candidate === node));
      setRovingFocus(node);
      showDetail(node);
      const x = Number(node.dataset.x || 0);
      const width = Math.max(Number(node.dataset.width || 1200), 1);
      svg.setAttribute('viewBox', `${x} 0 ${width} ${baseHeight}`);
      if (moveFocus) node.focus();
    };
    const resetView = (moveFocus = false) => {
      svg.setAttribute('viewBox', baseViewBox);
      nodes.forEach((node) => node.classList.remove('is-selected'));
      setRovingFocus(nodes[0]);
      showDetail(nodes[0]);
      if (moveFocus) nodes[0].focus();
    };
    nodes.forEach((node, index) => {
      node.addEventListener('focus', () => showDetail(node));
      node.addEventListener('click', () => selectNode(node, false));
      node.addEventListener('keydown', (event) => {
        let next = index;
        if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = Math.min(index + 1, nodes.length - 1);
        else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = Math.max(index - 1, 0);
        else if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = nodes.length - 1;
        else if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          selectNode(node, false);
          return;
        } else if (event.key === 'Escape') {
          event.preventDefault();
          resetView(true);
          return;
        } else return;
        event.preventDefault();
        setRovingFocus(nodes[next]);
        nodes[next].focus();
      });
    });
    zoneSearch?.addEventListener('input', () => {
      const query = zoneSearch.value.trim().toLowerCase();
      let matches = 0;
      nodes.forEach((node) => {
        const match = !query || (node.dataset.name || '').toLowerCase().includes(query);
        node.classList.toggle('is-match', Boolean(query) && match);
        node.classList.toggle('is-muted', Boolean(query) && !match);
        if (match) matches += 1;
      });
      if (zoneResults) zoneResults.textContent = query ? `${matches} of ${nodes.length} zones match` : `${nodes.length} zones`;
    });
    reset?.addEventListener('click', () => resetView(true));
    if (zoneResults) zoneResults.textContent = `${nodes.length} zones`;
    showDetail(nodes[0]);
  });
})();
)JS");
  return Result<std::string, std::string>::makeResult(out.str());
}

Result<bool, std::string>
writeBenchmarkGraphHtmlFile(std::span<const BenchmarkReport> reports,
                            const BenchmarkGraphOptions &options,
                            const std::filesystem::path &path) {
  auto html = writeBenchmarkGraphHtml(reports, options);
  if (html.hasError()) {
    return Result<bool, std::string>::makeError(html.error());
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return Result<bool, std::string>::makeError(
        "writeBenchmarkGraphHtmlFile: failed to open " + path.string());
  }
  file << html.value();
  return Result<bool, std::string>::makeResult(true);
}

namespace {

[[nodiscard]] std::string statusClass(std::string_view status) {
  if (status == "fail") {
    return "status-fail";
  }
  if (status == "warn") {
    return "status-warn";
  }
  return "status-pass";
}

[[nodiscard]] std::string formatDeltaValue(std::string_view metric,
                                           double value) {
  std::string formatted = formatMetricValue(metric, value);
  if (std::isfinite(value) && value > 0.0) {
    formatted.insert(formatted.begin(), '+');
  }
  return formatted;
}

[[nodiscard]] std::string formatDeltaPercent(double value) {
  if (!std::isfinite(value)) {
    return "missing";
  }
  std::ostringstream out;
  if (value > 0.0) {
    out << '+';
  }
  out << std::fixed << std::setprecision(2) << value << '%';
  return out.str();
}

[[nodiscard]] bool
comparisonMetricSelected(const BenchmarkMetricComparison &metric,
                         const BenchmarkGraphOptions &options) {
  if (!options.metrics.empty() &&
      std::find(options.metrics.begin(), options.metrics.end(),
                metric.metricId) == options.metrics.end()) {
    return false;
  }
  if (!options.statistics.empty() &&
      std::find(options.statistics.begin(), options.statistics.end(),
                metric.statistic) == options.statistics.end()) {
    return false;
  }
  return true;
}

[[nodiscard]] std::vector<const BenchmarkMetricComparison *>
selectComparisonMetrics(const BenchmarkComparisonReport &comparison,
                        const BenchmarkGraphOptions &options) {
  std::vector<const BenchmarkMetricComparison *> selected;
  for (const BenchmarkMetricComparison &metric : comparison.metrics) {
    if (comparisonMetricSelected(metric, options)) {
      selected.push_back(&metric);
    }
  }
  std::sort(selected.begin(), selected.end(),
            [](const BenchmarkMetricComparison *lhs,
               const BenchmarkMetricComparison *rhs) {
              if (lhs->status != rhs->status) {
                const auto rank = [](std::string_view status) {
                  if (status == "fail") {
                    return 0;
                  }
                  if (status == "warn") {
                    return 1;
                  }
                  return 2;
                };
                return rank(lhs->status) < rank(rhs->status);
              }
              if (lhs->required != rhs->required) {
                return lhs->required && !rhs->required;
              }
              if (metricLess(lhs->metricId, rhs->metricId)) {
                return true;
              }
              if (metricLess(rhs->metricId, lhs->metricId)) {
                return false;
              }
              if (lhs->statistic != rhs->statistic) {
                return lhs->statistic < rhs->statistic;
              }
              return false;
            });
  return selected;
}

void appendMessageList(std::ostringstream &out, std::string_view title,
                       const std::vector<std::string> &messages,
                       std::string_view cssClass) {
  if (messages.empty()) {
    return;
  }
  out << "<section class=\"panel " << escapeHtml(cssClass) << "\"><h2>"
      << escapeHtml(title) << "</h2><ul class=\"messages\">";
  for (const std::string &message : messages) {
    out << "<li>" << escapeHtml(message) << "</li>";
  }
  out << "</ul></section>";
}

void appendComparisonMetricRows(
    std::ostringstream &out,
    const std::vector<const BenchmarkMetricComparison *> &metrics) {
  std::string currentCategory;
  bool open = false;
  for (const BenchmarkMetricComparison *metric : metrics) {
    const std::string category = metricCategory(metric->metricId);
    if (category != currentCategory) {
      if (open) {
        out << "</tbody></table></div></details>";
      }
      currentCategory = category;
      open = true;
      out << "<details class=\"compare-group\" open><summary><span>"
          << escapeHtml(category) << "</span><small>"
          << escapeHtml(metricCategoryDescription(category))
          << "</small></summary><div class=\"table-wrap\"><table>"
             "<caption>Baseline-to-current metric comparison</caption>"
             "<thead><tr><th scope=\"col\">Metric</th><th "
             "scope=\"col\">Stat</th>"
             "<th scope=\"col\">Required</th><th scope=\"col\">Baseline</th>"
             "<th scope=\"col\">Current</th><th scope=\"col\">Delta</th>"
             "<th scope=\"col\">Delta %</th><th scope=\"col\">Robust "
             "effect</th>"
             "<th scope=\"col\">Delta interval</th><th scope=\"col\">Noise</th>"
             "<th scope=\"col\">Repeat confidence</th><th "
             "scope=\"col\">Status</th>"
             "</tr></thead><tbody>";
    }
    out << "<tr class=\"comparison-row " << statusClass(metric->status)
        << "\" data-status=\"" << escapeHtml(metric->status)
        << "\" data-metric=\"" << escapeHtml(metric->metricId) << "\"><td>"
        << escapeHtml(metric->metricId) << "</td><td>"
        << escapeHtml(metric->statistic) << "</td><td>"
        << (metric->required ? "yes" : "no") << "</td><td>"
        << formatMetricValue(metric->metricId, metric->baseline) << "</td><td>"
        << formatMetricValue(metric->metricId, metric->current) << "</td><td>"
        << formatDeltaValue(metric->metricId, metric->delta) << "</td><td>"
        << (metric->deltaPercentAvailable
                ? formatDeltaPercent(metric->deltaPercent)
                : std::string("n/a"))
        << "</td><td>";
    if (metric->repeatComparison.has_value()) {
      const RepeatComparisonStats &repeat = *metric->repeatComparison;
      out << std::fixed << std::setprecision(2) << repeat.robustEffect
          << "</td><td>["
          << formatDeltaValue(metric->metricId, repeat.confidenceLow) << ", "
          << formatDeltaValue(metric->metricId, repeat.confidenceHigh)
          << "]</td><td>" << std::fixed << std::setprecision(2)
          << repeat.noiseScore << "</td><td>"
          << (metric->repeatObservationsIndependent
                  ? (repeat.lowConfidence ? "low" : "estimated")
                  : "investigative")
          << " (" << escapeHtml(metric->repeatObservationUnit) << ")";
    } else {
      out << "n/a</td><td>n/a</td><td>n/a</td><td>not collected";
    }
    out << "</td><td><span class=\"status-badge status-"
        << escapeHtml(metric->status) << "\">" << escapeHtml(metric->status)
        << "</span></td></tr>";
  }
  if (open) {
    out << "</tbody></table></div></details>";
  }
}

} // namespace

Result<std::string, std::string>
writeBenchmarkComparisonHtml(const BenchmarkReport &baseline,
                             const BenchmarkReport &current,
                             const BenchmarkComparisonReport &comparison,
                             const BenchmarkGraphOptions &options) {
  const std::vector<const BenchmarkMetricComparison *> metrics =
      selectComparisonMetrics(comparison, options);
  uint32_t passCount = 0u;
  uint32_t warnCount = 0u;
  uint32_t failCount = 0u;
  uint32_t requiredCount = 0u;
  for (const BenchmarkMetricComparison *metric : metrics) {
    if (metric->required) {
      ++requiredCount;
    }
    if (metric->status == "fail") {
      ++failCount;
    } else if (metric->status == "warn") {
      ++warnCount;
    } else {
      ++passCount;
    }
  }

  const std::string title =
      options.title.empty() ? "Nuri Benchmark Comparison" : options.title;
  const std::string_view overallStatus =
      !comparison.valid                          ? std::string_view{"invalid"}
      : comparison.regression || failCount != 0u ? std::string_view{"fail"}
      : warnCount != 0u                          ? std::string_view{"warn"}
                                                 : std::string_view{"pass"};
  std::ostringstream out;
  nuri::tools::core::beginHtmlReport(out, title, R"CSS(
.compare-group{margin:.8rem 0;overflow:hidden}.compare-group summary{display:grid;grid-template-columns:minmax(11rem,1fr) 2fr;gap:.8rem;align-items:center;padding:.85rem 1rem;border-bottom:1px solid var(--line)}.compare-group summary span{font-weight:850}.compare-group summary small{color:var(--muted)}.compare-group table{min-width:76rem}.comparison-row td:nth-child(4),.comparison-row td:nth-child(5),.comparison-row td:nth-child(6),.comparison-row td:nth-child(7){text-align:right;font-variant-numeric:tabular-nums}.status-warn td:nth-child(6),.status-warn td:nth-child(7){color:var(--warn)}.status-fail td:nth-child(6),.status-fail td:nth-child(7){color:var(--fail)}
.error-panel{color:var(--fail);border-color:var(--fail)}.warning-panel{color:var(--warn);border-color:var(--warn)}
@media(max-width:760px){.compare-group summary{grid-template-columns:1fr}}
)CSS");
  out << "<header class=\"report-header\"><p class=\"report-kicker\">"
         "Benchmark regression analysis</p><div class=\"title-row\"><h1>"
      << escapeHtml(title) << "</h1><span class=\"status-badge status-"
      << overallStatus << "\">" << overallStatus
      << "</span></div><p class=\"lede\">Baseline-to-current deltas, required "
         "metric gates, repeat confidence, and comparison authority.</p><nav "
         "class=\"report-nav\" aria-label=\"Report sections\"><a "
         "href=\"#overview\">Overview</a><a href=\"#identity\">Run identity</a>"
         "<a href=\"#metric-comparisons\">Metrics</a><button type=\"button\" "
         "data-action=\"theme\" aria-pressed=\"false\">Light theme</button>"
         "</nav></header><main id=\"main-content\" tabindex=\"-1\"><section "
         "id=\"overview\" aria-labelledby=\"overview-title\"><div "
         "class=\"section-heading\"><div><h2 id=\"overview-title\">Comparison "
         "overview</h2><p>Gate outcome and selected metric coverage.</p></div>"
         "</div><div class=\"summary-grid\"><div class=\"summary-card\">"
         "<strong class=\"tone-"
      << (comparison.valid ? "pass\">valid" : "fail\">invalid")
      << "</strong><span>comparison contract</span></div><div "
         "class=\"summary-card\">"
         "<strong class=\"tone-"
      << (comparison.authoritative ? "pass\">yes" : "warn\">no")
      << "</strong><span>authoritative</span></div><div class=\"summary-card\">"
         "<strong class=\"tone-"
      << (comparison.regression ? "fail\">yes" : "pass\">no")
      << "</strong><span>regression</span></div><div class=\"summary-card\">"
         "<strong class=\"tone-fail\">"
      << failCount
      << "</strong><span>failed metrics</span></div><div "
         "class=\"summary-card\">"
         "<strong class=\"tone-warn\">"
      << warnCount
      << "</strong><span>warned metrics</span></div><div "
         "class=\"summary-card\">"
         "<strong class=\"tone-pass\">"
      << passCount
      << "</strong><span>passed metrics</span></div><div "
         "class=\"summary-card\">"
         "<strong>"
      << requiredCount
      << "</strong><span>required metrics</span></div></div>"
         "</section>";

  appendMessageList(out, "Errors", comparison.errors, "error-panel");
  appendMessageList(out, "Warnings", comparison.warnings, "warning-panel");
  out << "<section class=\"panel\" id=\"identity\"><div "
         "class=\"section-heading\"><div><h2>Run identity</h2><p>Verify "
         "hardware, build, and commit alignment before interpreting deltas."
         "</p></div></div><div class=\"identity-grid\"><article "
         "class=\"identity-card\"><p class=\"report-kicker\">Baseline</p><h3>"
      << escapeHtml(baseline.benchmarkCase.id)
      << "</h3><dl><dt>Backend</dt><dd>"
      << escapeHtml(baseline.environment.gpuBackend) << "</dd><dt>GPU</dt><dd>"
      << escapeHtml(baseline.environment.gpuDeviceName)
      << "</dd><dt>Build</dt><dd>" << escapeHtml(baseline.environment.buildType)
      << "</dd><dt>Commit</dt><dd><code>"
      << escapeHtml(baseline.environment.commitHash)
      << "</code></dd></dl></article><article class=\"identity-card\">"
         "<p class=\"report-kicker\">Current</p><h3>"
      << escapeHtml(current.benchmarkCase.id) << "</h3><dl><dt>Backend</dt><dd>"
      << escapeHtml(current.environment.gpuBackend) << "</dd><dt>GPU</dt><dd>"
      << escapeHtml(current.environment.gpuDeviceName)
      << "</dd><dt>Build</dt><dd>" << escapeHtml(current.environment.buildType)
      << "</dd><dt>Commit</dt><dd><code>"
      << escapeHtml(current.environment.commitHash)
      << "</code></dd></dl></article></div></section><section "
         "id=\"metric-comparisons\" "
         "aria-labelledby=\"comparison-metrics-title\">"
         "<div class=\"section-heading\"><div><h2 "
         "id=\"comparison-metrics-title\">Metric comparisons</h2><p>Failures "
         "sort first; filters never change the computed gate outcome.</p></div>"
         "<span class=\"section-count\">"
      << metrics.size() << " rows</span></div>";
  if (metrics.empty()) {
    out << "<p class=\"empty-state\">No comparison metric rows matched the "
           "selected filters.</p>";
  } else {
    out << "<div class=\"toolbar\" role=\"search\" aria-label=\"Filter metric "
           "comparisons\"><div class=\"control-group\"><label "
           "for=\"comparison-search\">Search metric</label><input "
           "id=\"comparison-search\" type=\"search\" placeholder=\"Metric id\">"
           "</div><div class=\"control-group\"><label "
           "for=\"comparison-status\">Status</label><select "
           "id=\"comparison-status\"><option value=\"\">All statuses</option>"
           "<option>fail</option><option>warn</option><option>pass</option>"
           "</select></div><p id=\"comparison-results\" "
           "class=\"results-count\" aria-live=\"polite\"></p></div>";
    appendComparisonMetricRows(out, metrics);
  }
  out << "</section></main>";
  nuri::tools::core::endHtmlReport(out, R"JS(
(() => {
  const search = document.querySelector('#comparison-search');
  const status = document.querySelector('#comparison-status');
  const rows = [...document.querySelectorAll('.comparison-row')];
  const groups = [...document.querySelectorAll('.compare-group')];
  const results = document.querySelector('#comparison-results');
  const filter = () => {
    const query = (search?.value || '').trim().toLowerCase();
    const selected = status?.value || '';
    let visible = 0;
    rows.forEach((row) => {
      const show = (!query || row.dataset.metric.toLowerCase().includes(query)) &&
        (!selected || row.dataset.status === selected);
      row.hidden = !show;
      if (show) visible += 1;
    });
    groups.forEach((group) => { group.hidden = !group.querySelector('.comparison-row:not([hidden])'); });
    if (results) results.textContent = `${visible} of ${rows.length} metric rows shown`;
  };
  search?.addEventListener('input', filter);
  status?.addEventListener('change', filter);
  filter();
})();
)JS");
  return Result<std::string, std::string>::makeResult(out.str());
}

Result<bool, std::string> writeBenchmarkComparisonHtmlFile(
    const BenchmarkReport &baseline, const BenchmarkReport &current,
    const BenchmarkComparisonReport &comparison,
    const BenchmarkGraphOptions &options, const std::filesystem::path &path) {
  auto html =
      writeBenchmarkComparisonHtml(baseline, current, comparison, options);
  if (html.hasError()) {
    return Result<bool, std::string>::makeError(html.error());
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return Result<bool, std::string>::makeError(
        "writeBenchmarkComparisonHtmlFile: failed to open " + path.string());
  }
  file << html.value();
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri::tools::benchmark
