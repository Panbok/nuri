#include "nuri/tools/snapshot/snapshot_html_report.h"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace nuri::tools::snapshot {
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
    default:
      out.push_back(c);
      break;
    }
  }
  return out;
}

[[nodiscard]] std::string relPath(const std::filesystem::path &from,
                                  const std::filesystem::path &to) {
  if (to.empty()) {
    return {};
  }
  std::error_code ec;
  std::filesystem::path relative =
      std::filesystem::relative(to, from.parent_path(), ec);
  return (ec ? to : relative).generic_string();
}

void writeCaptureCard(std::ostringstream &out, const SnapshotReport &report,
                      const SnapshotCaptureReport &capture,
                      const std::filesystem::path &htmlPath) {
  out << "<section class=\"capture status-" << escapeHtml(capture.status)
      << "\">\n"
      << "<h2>" << escapeHtml(capture.target) << "</h2>\n"
      << "<p><strong>" << escapeHtml(capture.status) << "</strong> "
      << escapeHtml(capture.statusReason) << "</p>\n"
      << "<dl>"
      << "<dt>Profile</dt><dd>" << escapeHtml(capture.profile) << "</dd>"
      << "<dt>Format</dt><dd>" << escapeHtml(capture.format) << "</dd>"
      << "<dt>Size</dt><dd>" << capture.width << "x" << capture.height
      << "</dd>"
      << "<dt>Producer</dt><dd>" << escapeHtml(capture.producerPassLabel)
      << "</dd>"
      << "</dl>\n";
  if (!capture.preview.empty()) {
    out << "<img alt=\"" << escapeHtml(capture.target) << " preview\" src=\""
        << escapeHtml(
               relPath(htmlPath, report.artifacts.caseDir / capture.preview))
        << "\">\n";
  }
  if (!capture.diff.empty()) {
    out << "<img alt=\"" << escapeHtml(capture.target) << " diff\" src=\""
        << escapeHtml(
               relPath(htmlPath, report.artifacts.caseDir / capture.diff))
        << "\">\n";
  }
  out << "<p class=\"paths\">";
  if (!capture.actual.empty()) {
    out << "actual: " << escapeHtml(capture.actual.generic_string()) << " ";
  }
  if (!capture.expected.empty()) {
    out << "expected: " << escapeHtml(capture.expected.generic_string()) << " ";
  }
  out << "</p>\n</section>\n";
}

} // namespace

Result<std::string, std::string>
writeSnapshotHtmlReport(const SnapshotReport &report) {
  std::ostringstream out;
  out << "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n"
      << "<title>Nuri Snapshot " << escapeHtml(report.snapshotCase.id)
      << "</title>\n"
      << "<style>"
      << "body{font-family:system-ui,sans-serif;margin:24px;background:#f7f7f5;"
         "color:#191919}"
      << "header{margin-bottom:24px}"
      << ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax("
         "320px,1fr));gap:16px}"
      << ".capture{background:white;border:1px solid "
         "#d8d8d2;border-radius:8px;padding:16px}"
      << ".capture img{display:block;max-width:100%;height:auto;border:1px "
         "solid #ddd;margin:8px 0}"
      << ".status-fail{border-color:#b42318}.status-pass{border-color:#1b7f3a}"
      << ".status-missing_baseline,.status-missing_capture_point{border-color:#"
         "a15c00}"
      << "dt{font-weight:600;float:left;clear:left;width:90px}dd{margin-left:"
         "100px}"
      << ".paths{font-family:ui-monospace,monospace;font-size:12px;color:#555}"
      << "</style></head><body>\n";
  out << "<header><h1>" << escapeHtml(report.snapshotCase.id) << "</h1>\n"
      << "<p>" << escapeHtml(report.generatedAtUtc) << " "
      << escapeHtml(report.environment.gpuBackend) << " "
      << escapeHtml(report.environment.resolvedWindowMode) << "</p>\n"
      << "</header>\n<main class=\"grid\">\n";
  for (const SnapshotCaptureReport &capture : report.captures) {
    writeCaptureCard(out, report, capture, report.artifacts.caseHtml);
  }
  out << "</main></body></html>\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

Result<bool, std::string>
writeSnapshotHtmlReportFile(const SnapshotReport &report,
                            const std::filesystem::path &path) {
  SnapshotReport htmlReport = report;
  htmlReport.artifacts.caseHtml = path;
  auto html = writeSnapshotHtmlReport(htmlReport);
  if (html.hasError()) {
    return Result<bool, std::string>::makeError(html.error());
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return Result<bool, std::string>::makeError(
        "writeSnapshotHtmlReportFile: failed to open " + path.string());
  }
  file << html.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<std::string, std::string>
writeSnapshotSuiteHtml(std::span<const SnapshotReport> reports,
                       std::string_view suite) {
  std::ostringstream out;
  out << "<!doctype html><html><head><meta charset=\"utf-8\"><title>Nuri "
      << escapeHtml(suite)
      << " Snapshots</"
         "title><style>body{font-family:system-ui,sans-serif;margin:24px}"
      << "li{margin:8px 0}.fail{color:#b42318}.pass{color:#1b7f3a}</style>"
      << "</head><body><h1>" << escapeHtml(suite) << "</h1><ul>\n";
  for (const SnapshotReport &report : reports) {
    bool failed = false;
    for (const SnapshotCaptureReport &capture : report.captures) {
      failed = failed || capture.status == "fail";
    }
    out << "<li class=\"" << (failed ? "fail" : "pass") << "\"><a href=\""
        << escapeHtml((std::filesystem::path("cases") / report.snapshotCase.id /
                       "report.html")
                          .generic_string())
        << "\">" << escapeHtml(report.snapshotCase.id) << "</a></li>\n";
  }
  out << "</ul></body></html>\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

Result<bool, std::string>
writeSnapshotSuiteHtmlFile(std::span<const SnapshotReport> reports,
                           std::string_view suite,
                           const std::filesystem::path &path) {
  auto html = writeSnapshotSuiteHtml(reports, suite);
  if (html.hasError()) {
    return Result<bool, std::string>::makeError(html.error());
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return Result<bool, std::string>::makeError(
        "writeSnapshotSuiteHtmlFile: failed to open " + path.string());
  }
  file << html.value();
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri::tools::snapshot
