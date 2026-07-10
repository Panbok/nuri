#pragma once

#include "nuri/tools/benchmark/benchmark_metric_registry.h"

#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nuri::tools::benchmark {

struct BenchmarkMeasurement {
  BenchmarkMetricIndex registeredIndex{};
  std::string ownedId{};
  double second = 0.0;

  [[nodiscard]] bool registered() const noexcept {
    return registeredIndex.valid();
  }
  [[nodiscard]] std::string_view id() const noexcept;
};

// Per-frame metric storage is contiguous. Published exact metrics retain only
// their registry index; only dynamic pass timings and unknown metrics read from
// legacy reports own an ID string.
class BenchmarkFrameMeasurements final {
public:
  using value_type = BenchmarkMeasurement;
  using container_type = std::vector<value_type>;
  using iterator = container_type::iterator;
  using const_iterator = container_type::const_iterator;

  BenchmarkFrameMeasurements() = default;
  BenchmarkFrameMeasurements(
      std::initializer_list<std::pair<std::string_view, double>> values);

  void reserve(size_t count) { values_.reserve(count); }
  [[nodiscard]] size_t capacity() const noexcept { return values_.capacity(); }
  [[nodiscard]] size_t size() const noexcept { return values_.size(); }
  [[nodiscard]] bool empty() const noexcept { return values_.empty(); }

  void appendRegistered(BenchmarkMetricIndex index, double value);
  void appendOwned(std::string id, double value);
  std::pair<iterator, bool> emplace(std::string id, double value);
  double &operator[](std::string_view id);

  [[nodiscard]] iterator find(std::string_view id) noexcept;
  [[nodiscard]] const_iterator find(std::string_view id) const noexcept;
  [[nodiscard]] size_t count(std::string_view id) const noexcept;
  [[nodiscard]] size_t registeredCount() const noexcept;
  [[nodiscard]] size_t ownedMetricIdCount() const noexcept;

  [[nodiscard]] iterator begin() noexcept { return values_.begin(); }
  [[nodiscard]] const_iterator begin() const noexcept { return values_.begin(); }
  [[nodiscard]] iterator end() noexcept { return values_.end(); }
  [[nodiscard]] const_iterator end() const noexcept { return values_.end(); }

private:
  container_type values_{};
};

} // namespace nuri::tools::benchmark
