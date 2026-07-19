#pragma once
#include <cstddef>
#include <memory_resource>
#include <utility>
#include <vector>
namespace nuri {

template <typename ResourceRecord, typename CompletionToken>
class GpuRetirementQueue {
public:
  explicit GpuRetirementQueue(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : pending_(memory) {}
  void reserve(size_t capacity) { pending_.reserve(capacity); }
  void retire(ResourceRecord resource, CompletionToken lastUse) {
    pending_.push_back(Pending{
        .resource = std::move(resource),
        .lastUse = std::move(lastUse),
    });
  }
  template <typename IsComplete> size_t collectIf(IsComplete &&isComplete) {
    size_t writeIndex = 0u;
    size_t collected = 0u;
    for (size_t readIndex = 0u; readIndex < pending_.size(); ++readIndex) {
      Pending &entry = pending_[readIndex];
      if (isComplete(entry.lastUse)) {
        ++collected;
        continue;
      }
      if (writeIndex != readIndex) {
        pending_[writeIndex] = std::move(entry);
      }
      ++writeIndex;
    }
    pending_.resize(writeIndex);
    return collected;
  }
  size_t collectThrough(const CompletionToken &completed) {
    return collectIf([&completed](const CompletionToken &lastUse) {
      return lastUse <= completed;
    });
  }
  void releaseAllAfterIdle() { pending_.clear(); }
  [[nodiscard]] size_t pendingCount() const noexcept { return pending_.size(); }
  [[nodiscard]] size_t capacity() const noexcept { return pending_.capacity(); }
  [[nodiscard]] bool empty() const noexcept { return pending_.empty(); }

private:
  struct Pending {
    ResourceRecord resource;
    CompletionToken lastUse;
  };
  std::pmr::vector<Pending> pending_;
};

} // namespace nuri
