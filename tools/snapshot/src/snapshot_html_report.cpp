#include "nuri/tools/snapshot/snapshot_html_report.h"

#include "nuri/tools/core/atomic_file.h"
#include "nuri/tools/core/html_report.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <sstream>
#include <string_view>
#include <vector>

namespace nuri::tools::snapshot {
namespace {

[[nodiscard]] std::string escapeHtml(std::string_view text) {
  return nuri::tools::core::htmlEscape(text);
}

[[nodiscard]] std::string readableStatus(std::string_view status) {
  std::string label(status);
  std::replace(label.begin(), label.end(), '_', ' ');
  return label;
}

[[nodiscard]] std::string pathToUtf8(const std::filesystem::path &path) {
  const std::u8string encoded = path.generic_u8string();
  return {reinterpret_cast<const char *>(encoded.data()), encoded.size()};
}

[[nodiscard]] std::string relPath(const std::filesystem::path &from,
                                  const std::filesystem::path &to) {
  if (to.empty()) {
    return {};
  }
  std::error_code error;
  const std::filesystem::path relative =
      std::filesystem::relative(to, from.parent_path(), error);
  return pathToUtf8(error ? to : relative);
}

[[nodiscard]] std::filesystem::path
artifactPath(const SnapshotReport &report, const std::filesystem::path &path) {
  return path.is_absolute() ? path : report.artifacts.caseDir / path;
}

[[nodiscard]] std::string artifactHref(const SnapshotReport &report,
                                       const std::filesystem::path &htmlPath,
                                       const std::filesystem::path &path) {
  return path.empty() ? std::string{}
                      : relPath(htmlPath, artifactPath(report, path));
}

void writeImagePanel(std::ostringstream &out, std::string_view label,
                     std::string_view href, std::string_view alt,
                     std::string_view cssClass = {}) {
  out << "<figure class=\"image-panel\"><figcaption>" << escapeHtml(label)
      << "</figcaption>";
  if (href.empty()) {
    out << "<div class=\"image-missing\" role=\"img\" aria-label=\""
        << escapeHtml(std::string(label) + " image not produced")
        << "\"><span aria-hidden=\"true\">—</span><small>Not produced</small>"
           "</div>";
  } else {
    out << "<a class=\"image-link\" href=\"" << escapeHtml(href)
        << "\" aria-label=\"Open " << escapeHtml(label)
        << " image at full resolution"
        << "\"><img decoding=\"async\" class=\"" << escapeHtml(cssClass)
        << "\" alt=\"" << escapeHtml(alt) << "\" src=\"" << escapeHtml(href)
        << "\"></a>";
  }
  out << "</figure>";
}

void writeMetricTable(std::ostringstream &out,
                      const SnapshotCaptureReport &capture) {
  const SnapshotCompareMetrics &metrics = capture.metrics;
  const SnapshotSemanticMetrics &semantic = capture.semanticMetrics;
  out << "<div class=\"table-wrap metrics-wrap\"><table class=\"metrics\">"
         "<caption>Image comparison and semantic error metrics</caption>"
         "<thead><tr><th scope=\"col\">Metric</th><th scope=\"col\">Value</th>"
         "</tr></thead><tbody>"
      << "<tr><td>mean absolute error</td><td class=\"numeric\">"
      << metrics.meanAbsError
      << "</td></tr><tr><td>RMSE</td><td class=\"numeric\">" << metrics.rmse
      << "</td></tr><tr><td>p99 absolute error</td><td class=\"numeric\">"
      << metrics.p99AbsError
      << "</td></tr><tr><td>max absolute error</td><td class=\"numeric\">"
      << metrics.maxAbsError
      << "</td></tr><tr><td>failing / compared values"
         "</td><td class=\"numeric\">"
      << metrics.failingValues << " / " << metrics.comparedValues
      << "</td></tr><tr><td>semantic mean / max (RMSE / p99)</td>"
         "<td class=\"numeric\">"
      << semantic.meanError << " / " << semantic.maxError << " ("
      << semantic.rmse << " / " << semantic.p99Error << ") "
      << escapeHtml(semantic.unit)
      << "</td></tr><tr><td>valid / ignored / changed pixels</td>"
         "<td class=\"numeric\">"
      << semantic.validPixels << " / " << semantic.ignoredPixels << " / "
      << semantic.changedPixels << " (" << semantic.failingPixels
      << " over threshold)</td></tr>";
  if (!semantic.secondaryUnit.empty()) {
    out << "<tr><td>secondary mean / RMSE / p99 / max</td>"
           "<td class=\"numeric\">"
        << semantic.meanSecondaryError << " / " << semantic.secondaryRmse
        << " / " << semantic.p99SecondaryError << " / "
        << semantic.maxSecondaryError << " "
        << escapeHtml(semantic.secondaryUnit) << "</td></tr>";
  }
  if (capture.profile == "mask") {
    out << "<tr><td>mask TP / TN / FP / FN</td><td class=\"numeric\">"
        << semantic.truePositivePixels << " / " << semantic.trueNegativePixels
        << " / " << semantic.falsePositivePixels << " / "
        << semantic.falseNegativePixels << "</td></tr>"
        << "<tr><td>mask intersection over union</td><td class=\"numeric\">"
        << semantic.intersectionOverUnion << "</td></tr>";
  }
  if (semantic.changedBoundsValid) {
    out << "<tr><td>changed bounds</td><td class=\"numeric\">("
        << semantic.minChangedX << ", " << semantic.minChangedY << ")–("
        << semantic.maxChangedX << ", " << semantic.maxChangedY
        << ")</td></tr>";
  }
  out << "<tr><td>max error coordinate</td><td class=\"numeric\">("
      << semantic.maxErrorX << ", " << semantic.maxErrorY
      << ")</td></tr></tbody></table></div>";
  if (!capture.failedThresholds.empty()) {
    out << "<div class=\"failed-thresholds diagnostics tone-fail\" "
           "role=\"alert\"><strong>Failed thresholds</strong><ul>";
    for (size_t index = 0u; index < capture.failedThresholds.size(); ++index) {
      out << "<li><code>" << escapeHtml(capture.failedThresholds[index])
          << "</code></li>";
    }
    out << "</ul></div>";
  }
}

void writeCaptureCard(std::ostringstream &out, const SnapshotReport &report,
                      const SnapshotCaptureReport &capture,
                      const std::filesystem::path &htmlPath, size_t index) {
  const std::string actualHref =
      artifactHref(report, htmlPath,
                   capture.preview.empty() ? capture.actual : capture.preview);
  const std::string expectedHref =
      artifactHref(report, htmlPath, capture.expected);
  const std::string diffHref = artifactHref(report, htmlPath, capture.diff);
  const bool comparisonExpected = capture.status != "captured" ||
                                  !expectedHref.empty() || !diffHref.empty();
  const std::string viewerId = "viewer-" + std::to_string(index);
  const std::string captureId = "capture-" + std::to_string(index);
  const std::string statusLabel = readableStatus(capture.status);

  out << "<article class=\"capture status-" << escapeHtml(capture.status)
      << " status-border-" << escapeHtml(capture.status) << "\" data-status=\""
      << escapeHtml(capture.status) << "\" data-target=\""
      << escapeHtml(capture.target) << "\" id=\"" << captureId
      << "\">\n<div class=\"card-heading\"><div><p class=\"report-kicker\">"
         "Capture "
      << index + 1u << "</p><h2>" << escapeHtml(capture.target)
      << "</h2></div><span class=\"status-badge status-"
      << escapeHtml(capture.status) << "\">" << escapeHtml(statusLabel)
      << "</span></div><p class=\"capture-reason\"><strong>Result:</strong> "
      << escapeHtml(readableStatus(capture.statusReason)) << "</p>\n"
      << "<div class=\"capture-meta\"><dl><dt>Profile</dt><dd>"
      << escapeHtml(capture.profile) << "</dd><dt>Semantic kind</dt><dd>"
      << escapeHtml(capture.kind) << "</dd><dt>Format</dt><dd>"
      << escapeHtml(capture.format) << "</dd><dt>Color space</dt><dd>"
      << escapeHtml(capture.colorSpace) << "</dd><dt>Size</dt><dd>"
      << capture.width << "×" << capture.height << " mip " << capture.mip
      << " layer " << capture.layer << "</dd><dt>Producer</dt><dd>"
      << escapeHtml(capture.producerPassLabel) << "</dd><dt>Required</dt><dd>"
      << (capture.required ? "yes" : "no") << "</dd></dl></div>\n";

  out << "<div class=\"viewer\" id=\"" << viewerId
      << "\" aria-label=\"Image evidence for " << escapeHtml(capture.target)
      << "\"><div class=\"triptych\">";
  writeImagePanel(out, "Actual", actualHref, capture.target + " actual",
                  "actual-image");
  if (comparisonExpected) {
    writeImagePanel(out, "Expected", expectedHref, capture.target + " expected",
                    "expected-image");
    writeImagePanel(out, "Semantic diff", diffHref, capture.target + " diff",
                    "diff-image");
  }
  out << "</div>";
  if (!comparisonExpected) {
    out << "<p class=\"empty-state\">Comparison was not requested; this "
           "report contains capture evidence only.</p>";
  }
  if (!actualHref.empty() && !expectedHref.empty()) {
    out << "<details class=\"overlay-tool\"><summary>Interactive overlay "
           "inspection</summary><div class=\"overlay-stage\" tabindex=\"0\" "
           "aria-label=\"Scrollable expected and actual image overlay\">"
           "<img class=\"overlay-expected\" src=\""
        << escapeHtml(expectedHref)
        << "\" alt=\"Expected image overlay\"><img class=\"overlay-actual\" "
           "src=\""
        << escapeHtml(actualHref)
        << "\" alt=\"Actual image overlay\"></div><div "
           "class=\"viewer-controls\">"
           "<button type=\"button\" data-action=\"blink\" "
           "aria-pressed=\"false\">"
           "Blink actual / expected</button>"
           "<label>Actual opacity <span><input data-action=\"opacity\" "
           "type=\"range\" min=\"0\" max=\"1\" step=\"0.01\" "
           "value=\"0.5\"><output>50%</output></span></label>"
           "<label>Zoom <span><input data-action=\"zoom\" type=\"range\" "
           "min=\"1\" "
           "max=\"8\" step=\"0.25\" "
           "value=\"1\"><output>1×</output></span></label>";
    if (!diffHref.empty()) {
      out << "<label>Diff amplification <span><input data-action=\"amplify\" "
             "type=\"range\" min=\"1\" max=\"8\" step=\"0.25\" "
             "value=\"1\"><output>1×</output></span></label>";
    }
    out << "</div></details>";
  }
  out << "</div>";
  if (comparisonExpected) {
    out << "<details class=\"metric-details\" open><summary>Metrics and "
           "thresholds</summary>";
    writeMetricTable(out, capture);
    out << "</details>";
  }

  out << "<nav class=\"artifact-links paths\" aria-label=\"Capture "
         "artifacts\">";
  const std::array links{
      std::pair{std::string_view("raw actual"), capture.actual},
      std::pair{std::string_view("descriptor"), capture.actualMetadata},
      std::pair{std::string_view("actual preview"), capture.preview},
      std::pair{std::string_view("expected preview"), capture.expected},
      std::pair{std::string_view("diff"), capture.diff}};
  bool first = true;
  for (const auto &[label, path] : links) {
    const std::string href = artifactHref(report, htmlPath, path);
    if (href.empty()) {
      continue;
    }
    if (!first) {
      out << "<span aria-hidden=\"true\">·</span>";
    }
    first = false;
    out << "<a href=\"" << escapeHtml(href) << "\">" << escapeHtml(label)
        << "</a>";
  }
  if (first) {
    out << "<span class=\"muted\">No artifact files were published.</span>";
  }
  out << "</nav>\n</article>\n";
}

[[nodiscard]] std::string_view reportStatus(const SnapshotReport &report) {
  if (!report.errors.empty()) {
    return "error";
  }
  std::string_view status = "pass";
  for (const SnapshotCaptureReport &capture : report.captures) {
    if (capture.status == "runtime_error" || capture.status == "invalid") {
      return "error";
    }
    if (capture.status == "fail") {
      status = "fail";
    } else if ((capture.status == "missing_baseline" ||
                capture.status == "environment_unavailable" ||
                capture.status == "missing_capture_point") &&
               status != "fail") {
      status = "unavailable";
    } else if (capture.status == "investigative" && status == "pass") {
      status = "investigative";
    }
  }
  return status;
}

[[nodiscard]] int statusRank(std::string_view status) {
  if (status == "error") {
    return 0;
  }
  if (status == "fail") {
    return 1;
  }
  if (status == "unavailable") {
    return 2;
  }
  if (status == "investigative") {
    return 3;
  }
  return 4;
}

} // namespace

Result<std::string, std::string>
writeSnapshotHtmlReport(const SnapshotReport &report) {
  const std::string_view overallStatus = reportStatus(report);
  size_t failedCaptures = 0u;
  size_t unavailableCaptures = 0u;
  size_t investigativeCaptures = 0u;
  size_t passedCaptures = 0u;
  size_t failedThresholds = 0u;
  for (const SnapshotCaptureReport &capture : report.captures) {
    failedThresholds += capture.failedThresholds.size();
    if (capture.status == "fail" || capture.status == "runtime_error" ||
        capture.status == "invalid") {
      ++failedCaptures;
    } else if (capture.status == "investigative") {
      ++investigativeCaptures;
    } else if (capture.status == "missing_baseline" ||
               capture.status == "environment_unavailable" ||
               capture.status == "missing_capture_point") {
      ++unavailableCaptures;
    } else {
      ++passedCaptures;
    }
  }

  std::ostringstream out;
  nuri::tools::core::beginHtmlReport(
      out, "Nuri Snapshot " + report.snapshotCase.id, R"CSS(
.capture.status-border-fail,.capture.status-border-runtime_error,.capture.status-border-invalid{border-left:5px solid var(--fail)}
.capture.status-border-pass,.capture.status-border-captured{border-left:5px solid var(--pass)}
.capture.status-border-missing_baseline,.capture.status-border-missing_capture_point,.capture.status-border-environment_unavailable,.capture.status-border-investigative{border-left:5px solid var(--warn)}
.capture-reason{margin:.55rem 0;color:var(--muted)}
.capture-meta{margin:.85rem 0;padding:.75rem;border:1px solid var(--line);border-radius:var(--radius-small);background:var(--surface-soft)}
.triptych{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,18rem),1fr));gap:.8rem}
.image-panel{min-width:0;margin:0}.image-panel figcaption{margin-bottom:.4rem;font-weight:800}
.image-link{display:block;border-radius:var(--radius-small);overflow:hidden;background:#05080a}
.image-panel img,.overlay-stage img{display:block;width:100%;height:auto;border:1px solid var(--line);background:#05080a}
.image-missing{display:grid;min-height:10rem;place-items:center;align-content:center;gap:.25rem;border:1px dashed var(--line-strong);border-radius:var(--radius-small);background:var(--surface-soft);color:var(--muted)}
.image-missing span{font-size:2rem}.overlay-tool,.metric-details{margin-top:1rem;border-top:1px solid var(--line)}
.overlay-tool>summary,.metric-details>summary{padding:.8rem 0;font-weight:800}
.overlay-stage{display:grid;max-width:100%;max-height:70vh;overflow:auto;border:1px solid var(--line);border-radius:var(--radius-small);background:#05080a}
.overlay-stage img{grid-area:1/1;max-width:none;transform-origin:top left}.overlay-actual{opacity:.5}
.viewer-controls{display:flex;flex-wrap:wrap;gap:.75rem;align-items:end;padding:.75rem 0}
.viewer-controls label{display:flex;flex:1 1 13rem;flex-direction:column;gap:.3rem;color:var(--muted);font-size:.8rem;font-weight:750}
.viewer-controls label span{display:flex;align-items:center;gap:.55rem}.viewer-controls output{min-width:3rem;color:var(--text);font-variant-numeric:tabular-nums}
.metrics{min-width:36rem}.metrics td:last-child{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
.failed-thresholds{margin-top:.75rem}.artifact-links{display:flex;flex-wrap:wrap;gap:.45rem;margin-top:1rem;font-size:.8rem}.artifact-links a{padding:.25rem .45rem;border:1px solid var(--line);border-radius:999px;text-decoration:none}
.capture-list{min-width:0}
@media(max-width:900px){.triptych{grid-template-columns:1fr}.metrics{min-width:30rem}}
)CSS");

  out << "<header class=\"report-header\"><p class=\"report-kicker\">Visual "
         "snapshot report</p><div class=\"title-row\"><h1>"
      << escapeHtml(report.snapshotCase.id)
      << "</h1><span class=\"status-badge status-" << escapeHtml(overallStatus)
      << "\">" << escapeHtml(readableStatus(overallStatus))
      << "</span></div><p class=\"lede\">Pixel and semantic comparison "
         "evidence, "
         "ordered for fast failure triage.</p><ul class=\"meta-list\">"
         "<li><strong>Generated:</strong> "
      << escapeHtml(report.generatedAtUtc)
      << "</li><li><strong>Backend:</strong> "
      << escapeHtml(report.environment.gpuBackend)
      << "</li><li><strong>GPU:</strong> "
      << escapeHtml(report.environment.gpuDeviceName)
      << "</li><li><strong>Window:</strong> "
      << escapeHtml(report.environment.resolvedWindowMode)
      << "</li><li><strong>Commit:</strong> <code>"
      << escapeHtml(report.environment.commitHash)
      << "</code>"
         "</li></ul><nav class=\"report-nav\" aria-label=\"Report sections\">"
         "<a href=\"#overview\">Overview</a><a href=\"#captures\">Captures</a>"
         "<a href=\"#environment\">Environment</a><button type=\"button\" "
         "data-action=\"theme\" aria-pressed=\"false\">Light theme</button>"
         "</nav>";
  if (!report.reproduceCommand.empty()) {
    out << "<div class=\"command-box\"><code id=\"reproduce-command\">"
        << escapeHtml(report.reproduceCommand)
        << "</code><button type=\"button\" "
           "data-copy-target=\"#reproduce-command\">"
           "Copy command</button></div>";
  }
  out << "</header><main id=\"main-content\" tabindex=\"-1\">"
         "<section id=\"overview\" aria-labelledby=\"overview-title\">"
         "<div class=\"section-heading\"><div><h2 id=\"overview-title\">Run "
         "overview</h2><p>The outcome and evidence volume at a "
         "glance.</p></div>"
         "</div><div class=\"summary-grid\">"
         "<div class=\"summary-card\"><strong class=\"tone-fail\">"
      << failedCaptures
      << "</strong><span>failed captures</span></div>"
         "<div class=\"summary-card\"><strong class=\"tone-warn\">"
      << unavailableCaptures
      << "</strong><span>unavailable captures</span></div>"
         "<div class=\"summary-card\"><strong class=\"tone-warn\">"
      << investigativeCaptures
      << "</strong><span>investigative captures</span></div>"
         "<div class=\"summary-card\"><strong class=\"tone-pass\">"
      << passedCaptures
      << "</strong><span>passed / captured</span></div>"
         "<div class=\"summary-card\"><strong>"
      << failedThresholds
      << "</strong><span>failed thresholds</span></div>"
         "<div class=\"summary-card\"><strong>"
      << report.warnings.size()
      << "</strong><span>warnings</span></div>"
         "<div class=\"summary-card\"><strong>"
      << report.errors.size()
      << "</strong><span>errors</span></div></div>"
         "</section>";

  if (!report.warnings.empty()) {
    out << "<section class=\"diagnostics tone-warn\" role=\"status\">"
           "<h2>Warnings</h2><ul>";
    for (const std::string &warning : report.warnings) {
      out << "<li>" << escapeHtml(warning) << "</li>";
    }
    out << "</ul></section>";
  }
  if (!report.errors.empty()) {
    out << "<section class=\"diagnostics tone-fail\" role=\"alert\">"
           "<h2>Errors</h2><ul>";
    for (const std::string &error : report.errors) {
      out << "<li>" << escapeHtml(error) << "</li>";
    }
    out << "</ul></section>";
  }

  out << "<details class=\"panel\" id=\"environment\"><summary><strong>Run "
         "environment and baseline authority</strong></summary><div "
         "class=\"detail-grid\">"
         "<dl><dt>Baseline profile</dt><dd>"
      << escapeHtml(report.baselineProfile)
      << "</dd><dt>Profile compatible</dt><dd>"
      << (report.baselineProfileCompatible ? "yes" : "no")
      << "</dd><dt>Build</dt><dd>" << escapeHtml(report.environment.buildType)
      << "</dd><dt>Driver</dt><dd>"
      << escapeHtml(report.environment.gpuDriverVersion)
      << "</dd></dl><dl><dt>Present mode</dt><dd>"
      << escapeHtml(report.environment.resolvedPresentMode)
      << "</dd><dt>Capture sync</dt><dd>"
      << escapeHtml(report.captureSynchronization)
      << "</dd><dt>Working tree</dt><dd>"
      << (report.environment.dirty ? "dirty" : "clean")
      << "</dd><dt>Render workers</dt><dd>"
      << report.environment.renderGraphWorkerCount << "</dd></dl></div>";
  if (!report.baselineProfileIncompatibilityReasons.empty()) {
    out << "<div class=\"diagnostics tone-warn\"><h3>Profile incompatibility "
           "reasons</h3><ul>";
    for (const std::string &reason :
         report.baselineProfileIncompatibilityReasons) {
      out << "<li>" << escapeHtml(reason) << "</li>";
    }
    out << "</ul></div>";
  }
  out << "</details><section id=\"captures\" "
         "aria-labelledby=\"captures-title\">"
         "<div class=\"section-heading\"><div><h2 "
         "id=\"captures-title\">Capture "
         "evidence</h2><p>Failures and unavailable evidence remain visible "
         "when "
         "filtering.</p></div><span class=\"section-count\">"
      << report.captures.size() << " captures</span></div>";
  if (!report.captures.empty()) {
    out << "<div class=\"toolbar\" role=\"search\" aria-label=\"Filter "
           "captures\">"
           "<div class=\"control-group\"><label for=\"capture-search\">Search "
           "target</label><input id=\"capture-search\" type=\"search\" "
           "placeholder=\"e.g. final_color\"></div><div "
           "class=\"control-group\">"
           "<label for=\"capture-status\">Status</label><select "
           "id=\"capture-status\">"
           "<option value=\"\">All statuses</option><option "
           "value=\"fail\">Fail</option>"
           "<option value=\"pass\">Pass</option><option "
           "value=\"captured\">Captured</option>"
           "<option value=\"investigative\">Investigative</option>"
           "<option value=\"missing_baseline\">Missing baseline</option>"
           "<option value=\"missing_capture_point\">Missing capture "
           "point</option>"
           "<option value=\"environment_unavailable\">Environment "
           "unavailable</option>"
           "<option value=\"runtime_error\">Runtime "
           "error</option></select></div>"
           "<p id=\"capture-results\" class=\"results-count\" "
           "aria-live=\"polite\"></p>"
           "</div>";
  }
  out << "<div class=\"capture-list\">";
  for (size_t index = 0u; index < report.captures.size(); ++index) {
    writeCaptureCard(out, report, report.captures[index],
                     report.artifacts.caseHtml, index);
  }
  if (report.captures.empty()) {
    out << "<p class=\"empty-state\">No capture evidence was produced for this "
           "case.</p>";
  }
  out << "</div></section></main>";
  nuri::tools::core::endHtmlReport(out, R"JS(
(() => {
  const search = document.querySelector('#capture-search');
  const status = document.querySelector('#capture-status');
  const cards = [...document.querySelectorAll('.capture')];
  const results = document.querySelector('#capture-results');
  const applyFilter = () => {
    const query = (search?.value || '').trim().toLowerCase();
    const selectedStatus = status?.value || '';
    let visible = 0;
    cards.forEach((card) => {
      const show = (!query || card.dataset.target.toLowerCase().includes(query)) &&
        (!selectedStatus || card.dataset.status === selectedStatus);
      card.hidden = !show;
      if (show) visible += 1;
    });
    if (results) results.textContent = `${visible} of ${cards.length} captures shown`;
  };
  search?.addEventListener('input', applyFilter);
  status?.addEventListener('change', applyFilter);
  applyFilter();

  document.querySelectorAll('.viewer').forEach((viewer) => {
    const actual = viewer.querySelector('.overlay-actual');
    const diff = viewer.querySelector('.diff-image');
    let timer = 0;
    const opacity = viewer.querySelector('[data-action="opacity"]');
    opacity?.addEventListener('input', () => {
      if (actual) actual.style.opacity = opacity.value;
      const output = opacity.parentElement?.querySelector('output');
      if (output) output.textContent = `${Math.round(Number(opacity.value) * 100)}%`;
    });
    const zoom = viewer.querySelector('[data-action="zoom"]');
    zoom?.addEventListener('input', () => {
      viewer.querySelectorAll('.overlay-stage img').forEach((image) => {
        image.style.transform = `scale(${zoom.value})`;
      });
      const output = zoom.parentElement?.querySelector('output');
      if (output) output.textContent = `${zoom.value}×`;
    });
    const amplify = viewer.querySelector('[data-action="amplify"]');
    amplify?.addEventListener('input', () => {
      if (diff) diff.style.filter = `contrast(${amplify.value})`;
      const output = amplify.parentElement?.querySelector('output');
      if (output) output.textContent = `${amplify.value}×`;
    });
    const blink = viewer.querySelector('[data-action="blink"]');
    blink?.addEventListener('click', () => {
      if (!actual) return;
      if (timer) {
        clearInterval(timer);
        timer = 0;
        actual.style.visibility = 'visible';
        blink.setAttribute('aria-pressed', 'false');
      } else {
        timer = window.setInterval(() => {
          actual.style.visibility = actual.style.visibility === 'hidden' ? 'visible' : 'hidden';
        }, 450);
        blink.setAttribute('aria-pressed', 'true');
      }
    });
  });
})();
)JS");
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
  auto written = nuri::tools::core::atomicWriteTextFile(path, html.value());
  if (written.hasError()) {
    return Result<bool, std::string>::makeError(
        "writeSnapshotHtmlReportFile: " + written.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<std::string, std::string>
writeSnapshotSuiteHtml(std::span<const SnapshotReport> reports,
                       std::string_view suite,
                       const std::filesystem::path &htmlPath) {
  std::vector<const SnapshotReport *> ordered;
  ordered.reserve(reports.size());
  std::array<size_t, 5u> counts{};
  for (const SnapshotReport &report : reports) {
    ordered.push_back(&report);
    ++counts[static_cast<size_t>(statusRank(reportStatus(report)))];
  }
  std::stable_sort(ordered.begin(), ordered.end(),
                   [](const auto *left, const auto *right) {
                     return statusRank(reportStatus(*left)) <
                            statusRank(reportStatus(*right));
                   });

  std::string_view suiteStatus = "pass";
  if (counts[0] != 0u) {
    suiteStatus = "error";
  } else if (counts[1] != 0u) {
    suiteStatus = "fail";
  } else if (counts[2] != 0u) {
    suiteStatus = "unavailable";
  } else if (counts[3] != 0u) {
    suiteStatus = "investigative";
  }

  std::ostringstream out;
  nuri::tools::core::beginHtmlReport(
      out, "Nuri " + std::string(suite) + " snapshot suite",
      R"CSS(
.case-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,22rem),1fr));gap:.8rem;margin:0;padding:0;list-style:none}
.case-card{min-width:0;padding:1rem;border:1px solid var(--line);border-left-width:5px;border-radius:var(--radius);background:var(--surface)}
.case-card.status-error,.case-card.status-fail{border-left-color:var(--fail)}
.case-card.status-unavailable,.case-card.status-investigative{border-left-color:var(--warn)}
.case-card.status-pass{border-left-color:var(--pass)}
.case-link{display:block;margin:.55rem 0 .3rem;color:var(--text);font-size:1rem;font-weight:800;overflow-wrap:anywhere}
.case-meta{display:flex;flex-wrap:wrap;gap:.4rem .75rem;margin:0;color:var(--muted);font-size:.8rem}
)CSS");
  out << "<header class=\"report-header\"><p class=\"report-kicker\">Visual "
         "snapshot suite</p><div class=\"title-row\"><h1>"
      << escapeHtml(suite) << "</h1><span class=\"status-badge status-"
      << suiteStatus << "\">" << escapeHtml(readableStatus(suiteStatus))
      << "</span></div><p class=\"lede\">All cases ranked by severity, with "
         "search and status filtering for quick triage.</p><nav "
         "class=\"report-nav\" "
         "aria-label=\"Report actions\"><a href=\"#cases\">Cases</a>"
         "<button type=\"button\" data-action=\"theme\" aria-pressed=\"false\">"
         "Light theme</button></nav></header><main id=\"main-content\" "
         "tabindex=\"-1\"><section aria-labelledby=\"suite-overview\">"
         "<div class=\"section-heading\"><div><h2 id=\"suite-overview\">Suite "
         "overview</h2><p>Case outcomes from the selected snapshot "
         "run.</p></div>"
         "</div><div class=\"summary-grid\"><div class=\"summary-card\">"
         "<strong class=\"tone-fail\">"
      << counts[0]
      << "</strong><span>errors</span></div><div class=\"summary-card\">"
         "<strong class=\"tone-fail\">"
      << counts[1]
      << "</strong><span>failures</span></div><div class=\"summary-card\">"
         "<strong class=\"tone-warn\">"
      << counts[2]
      << "</strong><span>unavailable</span></div><div class=\"summary-card\">"
         "<strong class=\"tone-warn\">"
      << counts[3]
      << "</strong><span>investigative</span></div><div class=\"summary-card\">"
         "<strong class=\"tone-pass\">"
      << counts[4]
      << "</strong><span>passed</span></div></div></section>"
         "<section id=\"cases\" aria-labelledby=\"cases-title\"><div "
         "class=\"section-heading\"><div><h2 id=\"cases-title\">Cases</h2>"
         "<p>Open a case for images, metrics, thresholds, and reproduction "
         "details."
         "</p></div><span class=\"section-count\">"
      << ordered.size()
      << " total</span></div><div class=\"toolbar\" role=\"search\" "
         "aria-label=\"Filter snapshot cases\"><div class=\"control-group\">"
         "<label for=\"search\">Search cases</label><input id=\"search\" "
         "type=\"search\" placeholder=\"Case id\"></div><div "
         "class=\"control-group\">"
         "<label for=\"status\">Status</label><select id=\"status\">"
         "<option value=\"\">All statuses</option><option>error</option>"
         "<option>fail</option><option>unavailable</option>"
         "<option>investigative</option><option>pass</option></select></div>"
         "<p id=\"case-results\" class=\"results-count\" aria-live=\"polite\">"
         "</p></div><ol class=\"case-grid\">";
  for (const SnapshotReport *report : ordered) {
    const std::string_view status = reportStatus(*report);
    const std::string reportLink =
        htmlPath.empty() ? pathToUtf8(std::filesystem::path("cases") /
                                      report->snapshotCase.id / "report.html")
                         : relPath(htmlPath, report->artifacts.caseHtml);
    out << "<li class=\"case-card status-" << status << "\" data-status=\""
        << status << "\" data-case=\"" << escapeHtml(report->snapshotCase.id)
        << "\"><span class=\"status-badge status-" << status << "\">"
        << escapeHtml(readableStatus(status))
        << "</span><a class=\"case-link\" href=\"" << escapeHtml(reportLink)
        << "\">" << escapeHtml(report->snapshotCase.id)
        << "</a><p class=\"case-meta\"><span>" << report->captures.size()
        << " captures</span><span>" << report->warnings.size()
        << " warnings</span><span>" << report->errors.size()
        << " errors</span></p></li>";
  }
  if (ordered.empty()) {
    out << "<li class=\"empty-state\">No snapshot cases were selected.</li>";
  }
  out << "</ol></section></main>";
  nuri::tools::core::endHtmlReport(out, R"JS(
(() => {
  const search = document.querySelector('#search');
  const status = document.querySelector('#status');
  const cases = [...document.querySelectorAll('.case-card')];
  const results = document.querySelector('#case-results');
  const filter = () => {
    const query = (search?.value || '').trim().toLowerCase();
    const selected = status?.value || '';
    let visible = 0;
    cases.forEach((item) => {
      const show = (!query || item.dataset.case.toLowerCase().includes(query)) &&
        (!selected || item.dataset.status === selected);
      item.hidden = !show;
      if (show) visible += 1;
    });
    if (results) results.textContent = `${visible} of ${cases.length} cases shown`;
  };
  search?.addEventListener('input', filter);
  status?.addEventListener('change', filter);
  filter();
})();
)JS");
  return Result<std::string, std::string>::makeResult(out.str());
}

Result<bool, std::string>
writeSnapshotSuiteHtmlFile(std::span<const SnapshotReport> reports,
                           std::string_view suite,
                           const std::filesystem::path &path) {
  auto html = writeSnapshotSuiteHtml(reports, suite, path);
  if (html.hasError()) {
    return Result<bool, std::string>::makeError(html.error());
  }
  auto written = nuri::tools::core::atomicWriteTextFile(path, html.value());
  if (written.hasError()) {
    return Result<bool, std::string>::makeError("writeSnapshotSuiteHtmlFile: " +
                                                written.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri::tools::snapshot
