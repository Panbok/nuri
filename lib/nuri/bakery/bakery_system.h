#pragma once

#include "nuri/bakery/bakery_types.h"
#include "nuri/core/result.h"
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"

#include <memory>
#include <memory_resource>

namespace nuri {
class GPUDevice;
}

namespace nuri::bakery {

class NURI_API BakerySystem final {
public:
  struct CreateDesc {
    nuri::GPUDevice &gpu;
    RuntimeConfig config;
    BakeryExecutionProfile profile = BakeryExecutionProfile::Interactive;
  };

  static Result<std::unique_ptr<BakerySystem>, std::string>
  create(CreateDesc desc);

  ~BakerySystem();

  BakerySystem(const BakerySystem &) = delete;
  BakerySystem &operator=(const BakerySystem &) = delete;
  BakerySystem(BakerySystem &&) = delete;
  BakerySystem &operator=(BakerySystem &&) = delete;

  Result<BakeJobId, std::string> enqueue(BakeRequest request);
  void tick();
  [[nodiscard]] std::pmr::vector<BakeJobSnapshot>
  snapshotJobs(std::pmr::memory_resource *mem) const;
  [[nodiscard]] bool hasActiveJobs() const noexcept;
  void requestCancel(BakeJobId id);
  void setExecutionProfile(BakeryExecutionProfile profile);
  [[nodiscard]] BakeryExecutionProfile executionProfile() const noexcept;

private:
  explicit BakerySystem(CreateDesc desc);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace nuri::bakery
