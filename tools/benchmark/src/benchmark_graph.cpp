#include "nuri/tools/benchmark/benchmark_graph.h"

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
  std::string out;
  out.reserve(text.size());
  for (const char c : text) {
    switch (c) {
    case '&':
      out += "&amp;";
      break;
    case '<':
      out += "&lt;";
      break;
    case '>':
      out += "&gt;";
      break;
    case '"':
      out += "&quot;";
      break;
    case '\'':
      out += "&#39;";
      break;
    default:
      out += c;
      break;
    }
  }
  return out;
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
      std::string_view("min"),    std::string_view("median"),
      std::string_view("p90"),    std::string_view("p95"),
      std::string_view("max"),    std::string_view("mean"),
      std::string_view("stddev"), std::string_view("mad"),
      std::string_view("iqr"),    std::string_view("cv"),
      std::string_view("coefficientOfVariation"),
  };
  return std::find(values.begin(), values.end(), statistic) != values.end();
}

[[nodiscard]] bool metricLooksIntegral(std::string_view metric) {
  return metric.ends_with("_count") || metric.ends_with("_counts") ||
         metric.ends_with("_draws") || metric.ends_with("_passes") ||
         metric.ends_with("_instances") || metric.ends_with("_commands") ||
         metric.ends_with("_dispatches") || metric.ends_with("_cascades") ||
         metric.ends_with("cascades") ||
         metric.ends_with("_textures") || metric.ends_with("_allocations") ||
         metric.ends_with("_reallocations") || metric.ends_with("_entries") ||
         metric.ends_with("_budget") || metric.ends_with("_levels") ||
         metric.ends_with("_calls");
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
  if (category == "Process memory") {
    return "OS process memory sampled after measured frames.";
  }
  if (category == "Benchmark PMR pools") {
    return "Upstream bytes reserved by renderer, pipeline, and scene PMR pools.";
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
  if (category == "Process memory") {
    return 2;
  }
  if (category == "Benchmark PMR pools") {
    return 3;
  }
  if (category == "GPU frame memory estimates") {
    return 4;
  }
  if (category == "Renderer work metrics") {
    return 5;
  }
  return 6;
}

[[nodiscard]] bool metricLess(const std::string &lhs,
                              const std::string &rhs) {
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
    return {"median", "p95"};
  }
  return options.statistics;
}

void appendReportTable(std::ostringstream &out,
                       std::span<const BenchmarkReport> reports,
                       std::span<const std::string> labels) {
  out << "<section class=\"panel\"><h2>Reports</h2><div class=\"table-wrap\">"
         "<table><thead><tr><th>Report</th><th>Case</th><th>Resolution</th>"
         "<th>Backend</th><th>Build</th><th>Valid</th></tr></thead><tbody>";
  for (size_t i = 0; i < reports.size(); ++i) {
    const BenchmarkReport &report = reports[i];
    out << "<tr><td>" << escapeHtml(labels[i]) << "</td><td>"
        << escapeHtml(report.benchmarkCase.id) << "</td><td>"
        << report.benchmarkCase.resolution[0] << "x"
        << report.benchmarkCase.resolution[1] << "</td><td>"
        << escapeHtml(report.environment.gpuBackend) << "</td><td>"
        << escapeHtml(report.environment.buildType) << "</td><td>"
        << (report.run.validForComparison ? "yes" : "no") << "</td></tr>";
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

  out << "<section class=\"metric-card\"><div class=\"metric-head\"><h2>"
      << escapeHtml(metric) << "</h2><span>" << escapeHtml(statistic)
      << "</span></div>";
  if (maxValue <= 0.0) {
    out << "<p class=\"empty\">No values available for this metric.</p>";
  } else {
    out << "<div class=\"bars\">";
    for (size_t i = 0; i < reports.size(); ++i) {
      const double value = values[i];
      const double percent =
          std::isfinite(value) ? std::clamp((value / maxValue) * 100.0, 0.0,
                                            100.0)
                               : 0.0;
      out << "<div class=\"bar-row\"><div class=\"bar-label\">"
          << escapeHtml(labels[i])
          << "</div><div class=\"bar-track\"><div class=\"bar-fill\" style=\"width:"
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
          << metricCategoryClass(currentCategory)
          << "\"><div class=\"section-head\"><div><h2>"
          << escapeHtml(currentCategory) << "</h2><p>"
          << escapeHtml(metricCategoryDescription(currentCategory))
          << "</p></div><span>"
          << escapeHtml(std::to_string(statistics.size()))
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
  std::ostringstream out;
  out << "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
         "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
         "<title>"
      << escapeHtml(options.title)
      << "</title><style>"
         ":root{color-scheme:dark;--bg:#101318;--panel:#181d25;--panel2:#202734;"
         "--text:#eef3f8;--muted:#aab4c0;--grid:#2d3542;--accent:#52d1a8;"
         "--accent2:#6aa7ff;--warn:#ffd166;--bad:#ff6b6b;--violet:#c084fc;"
         "--amber:#f6c177;--rose:#ff7a90;--cyan:#67e8f9}"
         "*{box-sizing:border-box}body{margin:0;background:linear-gradient(180deg,#101318,#151a22);"
         "color:var(--text);font:14px/1.45 Inter,Segoe UI,system-ui,sans-serif}"
         "main{max-width:1360px;margin:0 auto;padding:32px 22px 48px}"
         "header{margin-bottom:22px}h1{font-size:32px;line-height:1.1;margin:0 0 8px}"
         "h2{font-size:16px;margin:0}.subtle{color:var(--muted);max-width:760px}"
         ".summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}"
         ".stat{background:var(--panel);border:1px solid var(--grid);border-radius:8px;padding:14px}"
         ".stat b{display:block;font-size:24px}.panel,.metric-card{background:rgba(24,29,37,.94);"
         "border:1px solid var(--grid);border-radius:8px;padding:18px;"
         "box-shadow:0 12px 36px rgba(0,0,0,.22)}.table-wrap{overflow:auto;margin-top:12px}"
         ".panel{margin:16px 0}.metric-section{margin:26px 0 0}.section-head{display:flex;"
         "align-items:flex-end;justify-content:space-between;gap:18px;margin:0 0 12px}"
         ".section-head h2{font-size:20px}.section-head p{margin:5px 0 0;color:var(--muted)}"
         ".section-head span{color:var(--muted);border:1px solid var(--grid);border-radius:999px;padding:4px 10px}"
         ".metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}"
         "table{width:100%;border-collapse:collapse;min-width:680px}th,td{text-align:left;"
         "border-bottom:1px solid var(--grid);padding:9px 10px}th{color:var(--muted);font-weight:600}"
         ".metric-head{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:14px}"
         ".metric-head span{color:#0b1117;background:var(--accent);border-radius:999px;padding:3px 10px;font-weight:700}"
         ".bars{display:grid;gap:10px}.bar-row{display:grid;grid-template-columns:minmax(190px,1.4fr) minmax(160px,4fr) 112px;"
         "gap:12px;align-items:center}.bar-label{color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}"
         ".bar-track{height:22px;border-radius:6px;background:var(--panel2);border:1px solid var(--grid);overflow:hidden}"
         ".bar-fill{height:100%;border-radius:5px;background:linear-gradient(90deg,var(--accent),var(--accent2));"
         "box-shadow:0 0 18px rgba(82,209,168,.35)}.bar-value{font-variant-numeric:tabular-nums;color:var(--muted);text-align:right}"
         ".cat-cpu .bar-fill{background:linear-gradient(90deg,var(--amber),var(--warn))}"
         ".cat-gpu .bar-fill{background:linear-gradient(90deg,var(--accent),var(--cyan))}"
         ".cat-memory .bar-fill{background:linear-gradient(90deg,var(--rose),var(--amber))}"
         ".cat-pmr .bar-fill{background:linear-gradient(90deg,var(--violet),var(--accent2))}"
         ".cat-gpu-memory .bar-fill{background:linear-gradient(90deg,var(--cyan),var(--accent2))}"
         ".cat-renderer .bar-fill{background:linear-gradient(90deg,var(--accent2),var(--violet))}"
         ".empty{color:var(--muted);margin:12px 0 0}@media(max-width:720px){.bar-row,.section-head{grid-template-columns:1fr;"
         "display:grid;gap:6px}.metric-grid{grid-template-columns:1fr}.bar-value{text-align:left}h1{font-size:26px}}"
         "</style></head><body><main><header><h1>"
      << escapeHtml(options.title)
      << "</h1><p class=\"subtle\">Self-contained benchmark graph generated "
         "from nuri JSON reports. Units come from each metric suffix: timings "
         "use ms, memory uses MiB, and renderer counters are unitless.</p></header>"
         "<section class=\"summary\"><div class=\"stat\"><b>"
      << reports.size()
      << "</b><span>reports</span></div><div class=\"stat\"><b>"
      << metrics.size()
      << "</b><span>metrics</span></div><div class=\"stat\"><b>"
      << statistics.size() << "</b><span>statistics</span></div></section>";

  appendReportTable(out, reports, labels);
  if (metrics.empty()) {
    out << "<section class=\"panel\"><h2>No Metrics</h2><p class=\"empty\">"
           "No matching metrics were found in the selected reports.</p></section>";
  } else {
    appendMetricSection(out, reports, labels, metrics, statistics);
  }
  out << "</main></body></html>\n";
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

} // namespace nuri::tools::benchmark
