#pragma once

#include <iosfwd>
#include <string>
#include <string_view>

namespace nuri::tools::core {

[[nodiscard]] std::string htmlEscape(std::string_view text);

// Starts a self-contained, accessible HTML report. The caller writes report
// content into the open .report-shell element and must finish with
// endHtmlReport().
void beginHtmlReport(std::ostream &out, std::string_view title,
                     std::string_view additionalCss = {});
void endHtmlReport(std::ostream &out, std::string_view additionalScript = {});

} // namespace nuri::tools::core
