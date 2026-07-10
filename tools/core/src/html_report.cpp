#include "nuri/tools/core/html_report.h"

#include <ostream>

namespace nuri::tools::core {
namespace {

constexpr std::string_view kReportCss = R"CSS(
:root {
  color-scheme: dark;
  --page: #0a1017;
  --page-accent: #101b28;
  --surface: #111a24;
  --surface-raised: #172332;
  --surface-soft: #0e1721;
  --text: #f4f7fa;
  --muted: #afbdca;
  --line: #34475a;
  --line-strong: #52677c;
  --accent: #72d6c0;
  --accent-ink: #06251f;
  --link: #8ecaff;
  --pass: #75dda2;
  --warn: #ffd36a;
  --fail: #ff9292;
  --info: #8ecaff;
  --shadow: 0 18px 52px rgb(0 0 0 / 0.28);
  --radius: 14px;
  --radius-small: 9px;
  --focus: #ffe08a;
}
:root[data-theme="light"] {
  color-scheme: light;
  --page: #f3f6f9;
  --page-accent: #e7eef5;
  --surface: #ffffff;
  --surface-raised: #f8fafc;
  --surface-soft: #edf3f7;
  --text: #14202b;
  --muted: #526474;
  --line: #c4d0da;
  --line-strong: #8799a8;
  --accent: #087c68;
  --accent-ink: #ffffff;
  --link: #005fa9;
  --pass: #08783d;
  --warn: #8b5a00;
  --fail: #b4232a;
  --info: #005fa9;
  --shadow: 0 16px 42px rgb(25 48 68 / 0.14);
  --focus: #6d4d00;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  min-width: 0;
  overflow-x: hidden;
  background:
    radial-gradient(circle at 85% -10%, color-mix(in srgb, var(--accent) 12%, transparent), transparent 34rem),
    linear-gradient(180deg, var(--page-accent), var(--page) 22rem);
  color: var(--text);
  font: 15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  text-rendering: optimizeLegibility;
}
a { color: var(--link); text-underline-offset: 0.18em; }
a:hover { text-decoration-thickness: 2px; }
button, input, select { font: inherit; }
button, select, input[type="search"] {
  min-height: 44px;
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-small);
  background: var(--surface-raised);
  color: var(--text);
}
button { cursor: pointer; padding: 0.55rem 0.8rem; font-weight: 700; }
button:hover { border-color: var(--accent); background: var(--surface-soft); }
input[type="search"], select { padding: 0.55rem 0.7rem; }
input[type="range"] { min-width: 8rem; accent-color: var(--accent); }
:focus-visible { outline: 3px solid var(--focus); outline-offset: 3px; }
::selection { background: var(--accent); color: var(--accent-ink); }
.skip-link {
  position: fixed;
  z-index: 100;
  inset: 0.7rem auto auto 0.7rem;
  transform: translateY(-160%);
  padding: 0.65rem 0.9rem;
  border-radius: var(--radius-small);
  background: var(--accent);
  color: var(--accent-ink);
  font-weight: 800;
}
.skip-link:focus { transform: translateY(0); }
.sr-only {
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  padding: 0 !important;
  margin: -1px !important;
  overflow: hidden !important;
  clip: rect(0, 0, 0, 0) !important;
  white-space: nowrap !important;
  border: 0 !important;
}
.report-shell {
  width: min(calc(100% - 2rem), 1500px);
  min-width: 0;
  margin: 0 auto;
  padding: 1.35rem 0 3rem;
}
.report-header {
  position: relative;
  overflow: hidden;
  padding: clamp(1.25rem, 3vw, 2.2rem);
  border: 1px solid var(--line);
  border-radius: calc(var(--radius) + 4px);
  background: color-mix(in srgb, var(--surface) 94%, transparent);
  box-shadow: var(--shadow);
}
.report-header::after {
  content: "";
  position: absolute;
  width: 16rem;
  height: 16rem;
  right: -8rem;
  top: -9rem;
  border-radius: 50%;
  background: color-mix(in srgb, var(--accent) 18%, transparent);
  pointer-events: none;
}
.report-kicker {
  margin: 0 0 0.4rem;
  color: var(--accent);
  font-size: 0.76rem;
  font-weight: 850;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.title-row, .section-heading, .card-heading, .toolbar, .header-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  min-width: 0;
}
.title-row { align-items: flex-start; }
h1, h2, h3 { line-height: 1.2; overflow-wrap: anywhere; }
h1 { margin: 0; font-size: clamp(1.8rem, 4vw, 3rem); letter-spacing: -0.035em; }
h2 { margin: 0; font-size: clamp(1.15rem, 2vw, 1.45rem); }
h3 { margin: 0; font-size: 1rem; }
p { max-width: 78ch; }
.lede { margin: 0.65rem 0 0; color: var(--muted); font-size: 1rem; }
.muted, .subtle, .environment { color: var(--muted); }
.meta-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem 1rem;
  margin: 1rem 0 0;
  padding: 0;
  color: var(--muted);
  list-style: none;
  font-size: 0.88rem;
}
.meta-list li { overflow-wrap: anywhere; }
.report-nav {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  margin: 1rem 0 0;
}
.report-nav a, .report-nav button {
  min-height: 40px;
  display: inline-flex;
  align-items: center;
  padding: 0.45rem 0.7rem;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: var(--surface-soft);
  color: var(--text);
  font-size: 0.86rem;
  font-weight: 720;
  text-decoration: none;
}
.report-nav a:hover { border-color: var(--accent); }
.status-badge {
  display: inline-flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 0.42rem;
  min-height: 2rem;
  padding: 0.28rem 0.65rem;
  border: 1px solid currentColor;
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 850;
  letter-spacing: 0.045em;
  line-height: 1;
  text-transform: uppercase;
}
.status-badge::before { content: ""; width: 0.5rem; height: 0.5rem; border-radius: 50%; background: currentColor; }
.status-pass, .tone-pass { color: var(--pass); }
.status-warn, .status-investigative, .status-missing_baseline,
.status-missing_capture_point, .status-environment_unavailable,
.status-unavailable, .tone-warn { color: var(--warn); }
.status-fail, .status-error, .status-invalid, .status-runtime_error, .tone-fail { color: var(--fail); }
.status-captured, .status-pending, .tone-info { color: var(--info); }
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 10rem), 1fr));
  gap: 0.75rem;
  margin: 1rem 0;
}
.summary-card {
  min-width: 0;
  padding: 1rem;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--surface);
}
.summary-card strong { display: block; font-size: clamp(1.35rem, 3vw, 2rem); line-height: 1.1; font-variant-numeric: tabular-nums; }
.summary-card span { display: block; margin-top: 0.35rem; color: var(--muted); font-size: 0.82rem; }
.panel, .checkpoint, .capture, .metric-card, .compare-group {
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: color-mix(in srgb, var(--surface) 96%, transparent);
  box-shadow: 0 8px 28px rgb(0 0 0 / 0.12);
}
.panel, .checkpoint, .capture { margin: 1rem 0; padding: clamp(1rem, 2vw, 1.35rem); }
.section-heading { align-items: flex-end; margin-bottom: 0.85rem; }
.section-heading p { margin: 0.35rem 0 0; color: var(--muted); }
.section-count { flex: 0 0 auto; color: var(--muted); font-variant-numeric: tabular-nums; }
.toolbar {
  align-items: end;
  flex-wrap: wrap;
  margin: 1rem 0;
  padding: 0.8rem;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--surface);
}
.control-group { display: flex; flex: 1 1 14rem; flex-direction: column; gap: 0.28rem; min-width: 0; }
.control-group label { color: var(--muted); font-size: 0.78rem; font-weight: 750; }
.control-group input, .control-group select { width: 100%; }
.results-count { margin: 0; color: var(--muted); font-size: 0.88rem; }
.table-wrap {
  width: 100%;
  min-width: 0;
  overflow-x: auto;
  margin-top: 0.75rem;
  border: 1px solid var(--line);
  border-radius: var(--radius-small);
  background: var(--surface-soft);
  overscroll-behavior-inline: contain;
  scrollbar-color: var(--line-strong) transparent;
}
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
caption { padding: 0.7rem 0.8rem; color: var(--muted); text-align: left; }
th, td { padding: 0.68rem 0.75rem; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; overflow-wrap: anywhere; }
th { position: sticky; top: 0; z-index: 1; background: var(--surface-raised); color: var(--muted); font-size: 0.75rem; letter-spacing: 0.035em; text-transform: uppercase; }
tbody tr:last-child td { border-bottom: 0; }
tbody tr:hover { background: color-mix(in srgb, var(--accent) 7%, transparent); }
.numeric { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.mono, code, .paths { font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; }
code { overflow-wrap: anywhere; }
.command-box {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 0.6rem;
  align-items: center;
  margin-top: 1rem;
  padding: 0.75rem;
  border: 1px solid var(--line);
  border-radius: var(--radius-small);
  background: var(--surface-soft);
}
.command-box code { min-width: 0; }
.diagnostics {
  margin: 0.9rem 0 0;
  padding: 0.85rem 1rem;
  border: 1px solid currentColor;
  border-radius: var(--radius-small);
  background: var(--surface-soft);
}
.diagnostics h2, .diagnostics h3 { color: currentColor; }
.diagnostics ul, .messages { margin: 0.55rem 0 0; padding-left: 1.25rem; }
.diagnostics li, .messages li { margin: 0.25rem 0; overflow-wrap: anywhere; }
details { min-width: 0; }
summary { cursor: pointer; }
summary::marker { color: var(--accent); }
.empty-state { padding: 1.2rem; border: 1px dashed var(--line-strong); border-radius: var(--radius-small); color: var(--muted); text-align: center; }
.identity-grid, .detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 15rem), 1fr)); gap: 0.75rem; }
.identity-card { min-width: 0; padding: 0.85rem; border: 1px solid var(--line); border-radius: var(--radius-small); background: var(--surface-soft); }
.identity-card p { margin: 0.35rem 0 0; overflow-wrap: anywhere; }
dl { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 0.35rem 0.85rem; }
dt { color: var(--muted); font-size: 0.8rem; font-weight: 750; }
dd { min-width: 0; margin: 0; overflow-wrap: anywhere; }
.report-footer { margin-top: 2rem; padding: 1rem 0; border-top: 1px solid var(--line); color: var(--muted); font-size: 0.8rem; }
[hidden] { display: none !important; }
@media (max-width: 760px) {
  .report-shell { width: min(calc(100% - 1rem), 1500px); padding-top: 0.5rem; }
  .report-header, .panel, .checkpoint, .capture { border-radius: 11px; }
  .title-row, .section-heading, .card-heading { align-items: flex-start; flex-direction: column; }
  .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .command-box { grid-template-columns: 1fr; }
  .command-box button { justify-self: start; }
  dl { grid-template-columns: 1fr; gap: 0.05rem; }
  dd { margin-bottom: 0.55rem; }
  .report-nav { gap: 0.35rem; }
}
@media (max-width: 420px) {
  .summary-grid { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; animation-iteration-count: 1 !important; }
}
@media print {
  :root { color-scheme: light; --page: #fff; --page-accent: #fff; --surface: #fff; --surface-raised: #fff; --surface-soft: #fff; --text: #000; --muted: #333; --line: #aaa; --line-strong: #666; --link: #000; --shadow: none; }
  body { background: #fff; }
  .report-shell { width: 100%; padding: 0; }
  .report-nav, .toolbar, button { display: none !important; }
  .report-header, .panel, .checkpoint, .capture, .metric-card, .compare-group { break-inside: avoid; box-shadow: none; }
  .table-wrap { overflow: visible; }
}
)CSS";

constexpr std::string_view kReportScript = R"JS(
(() => {
  const root = document.documentElement;
  const themeButton = document.querySelector('[data-action="theme"]');
  const announce = (message) => {
    const target = document.querySelector('#report-announcer');
    if (target) target.textContent = message;
  };
  const setThemeButton = () => {
    if (!themeButton) return;
    const light = root.dataset.theme === 'light';
    themeButton.setAttribute('aria-pressed', String(light));
    themeButton.textContent = light ? 'Dark theme' : 'Light theme';
  };
  themeButton?.addEventListener('click', () => {
    const next = root.dataset.theme === 'light' ? 'dark' : 'light';
    root.dataset.theme = next;
    try { localStorage.setItem('nuri-report-theme', next); } catch (_) {}
    setThemeButton();
    announce(`${next} theme active`);
  });
  setThemeButton();
  document.querySelectorAll('[data-copy-target]').forEach((button) => {
    button.addEventListener('click', async () => {
      const target = document.querySelector(button.dataset.copyTarget);
      if (!target) return;
      try {
        await navigator.clipboard.writeText(target.textContent || '');
        announce('Command copied to clipboard');
        const old = button.textContent;
        button.textContent = 'Copied';
        window.setTimeout(() => { button.textContent = old; }, 1400);
      } catch (_) {
        announce('Clipboard unavailable; select and copy the command manually');
      }
    });
  });
})();
)JS";

} // namespace

std::string htmlEscape(std::string_view text) {
  std::string escaped;
  escaped.reserve(text.size());
  for (const char c : text) {
    switch (c) {
    case '&':
      escaped += "&amp;";
      break;
    case '<':
      escaped += "&lt;";
      break;
    case '>':
      escaped += "&gt;";
      break;
    case '"':
      escaped += "&quot;";
      break;
    case '\'':
      escaped += "&#39;";
      break;
    default:
      escaped.push_back(c);
      break;
    }
  }
  return escaped;
}

void beginHtmlReport(std::ostream &out, std::string_view title,
                     std::string_view additionalCss) {
  out << "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
         "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
         "<meta name=\"color-scheme\" content=\"dark light\"><title>"
      << htmlEscape(title)
      << "</title><script>try{const t=localStorage.getItem('nuri-report-theme');"
         "if(t)document.documentElement.dataset.theme=t}catch(_){}</script><style>"
      << kReportCss << additionalCss
      << "</style></head><body><a class=\"skip-link\" href=\"#main-content\">"
         "Skip to report content</a><div id=\"report-announcer\" class=\"sr-only\" "
         "aria-live=\"polite\" aria-atomic=\"true\"></div>"
         "<div class=\"report-shell\">";
}

void endHtmlReport(std::ostream &out, std::string_view additionalScript) {
  out << "<footer class=\"report-footer\">Generated by Nuri renderer tooling. "
         "This report is self-contained and works offline.</footer></div><script>"
      << kReportScript << additionalScript << "</script></body></html>\n";
}

} // namespace nuri::tools::core
