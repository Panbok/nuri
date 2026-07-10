#include "nuri/tools/benchmark/benchmark_measurements.h"

#include <algorithm>

namespace nuri::tools::benchmark {

std::string_view BenchmarkMeasurement::id() const noexcept {
  if (!registered()) {
    return ownedId;
  }
  const BenchmarkMetricDescriptor *descriptor =
      benchmarkMetricDescriptor(registeredIndex);
  return descriptor != nullptr ? descriptor->idOrRule : std::string_view{};
}

BenchmarkFrameMeasurements::BenchmarkFrameMeasurements(
    std::initializer_list<std::pair<std::string_view, double>> values) {
  values_.reserve(values.size());
  for (const auto &[id, value] : values) {
    (*this)[id] = value;
  }
}

void BenchmarkFrameMeasurements::appendRegistered(BenchmarkMetricIndex index,
                                                  double value) {
  values_.push_back(BenchmarkMeasurement{.registeredIndex = index,
                                         .second = value});
}

void BenchmarkFrameMeasurements::appendOwned(std::string id, double value) {
  values_.push_back(BenchmarkMeasurement{.ownedId = std::move(id),
                                         .second = value});
}

std::pair<BenchmarkFrameMeasurements::iterator, bool>
BenchmarkFrameMeasurements::emplace(std::string id, double value) {
  const auto existing = find(id);
  if (existing != end()) {
    return {existing, false};
  }
  if (const auto index = findExactBenchmarkMetricIndex(id); index.has_value()) {
    appendRegistered(*index, value);
  } else {
    appendOwned(std::move(id), value);
  }
  return {std::prev(end()), true};
}

double &BenchmarkFrameMeasurements::operator[](std::string_view id) {
  const auto existing = find(id);
  if (existing != end()) {
    return existing->second;
  }
  if (const auto index = findExactBenchmarkMetricIndex(id); index.has_value()) {
    appendRegistered(*index, 0.0);
  } else {
    appendOwned(std::string(id), 0.0);
  }
  return values_.back().second;
}

BenchmarkFrameMeasurements::iterator
BenchmarkFrameMeasurements::find(std::string_view id) noexcept {
  return std::find_if(values_.begin(), values_.end(),
                      [id](const BenchmarkMeasurement &value) {
                        return value.id() == id;
                      });
}

BenchmarkFrameMeasurements::const_iterator
BenchmarkFrameMeasurements::find(std::string_view id) const noexcept {
  return std::find_if(values_.begin(), values_.end(),
                      [id](const BenchmarkMeasurement &value) {
                        return value.id() == id;
                      });
}

size_t BenchmarkFrameMeasurements::count(std::string_view id) const noexcept {
  return find(id) == end() ? 0u : 1u;
}

size_t BenchmarkFrameMeasurements::registeredCount() const noexcept {
  return static_cast<size_t>(
      std::count_if(values_.begin(), values_.end(),
                    [](const BenchmarkMeasurement &value) {
                      return value.registered();
                    }));
}

size_t BenchmarkFrameMeasurements::ownedMetricIdCount() const noexcept {
  return values_.size() - registeredCount();
}

} // namespace nuri::tools::benchmark
