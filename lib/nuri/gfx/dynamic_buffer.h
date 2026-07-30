#pragma once
#include "nuri/core/result.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/resources/gpu/buffer.h"
#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <memory_resource>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>
namespace nuri {

[[nodiscard]] constexpr size_t
nextDynamicBufferCapacity(size_t currentCapacity, size_t requiredCapacity) {
  constexpr size_t kMinimumCapacity = 256u;
  constexpr size_t kCapacityAlignment = 256u;
  if (currentCapacity >= requiredCapacity) {
    return currentCapacity;
  }
  const size_t growthBase = std::max(currentCapacity, kMinimumCapacity);
  const size_t geometricCapacity = growthBase + growthBase / 2u;
  const size_t targetCapacity = std::max(requiredCapacity, geometricCapacity);
  return ((targetCapacity + kCapacityAlignment - 1u) / kCapacityAlignment) *
         kCapacityAlignment;
}

struct DynamicBufferAcquisition {
  BufferHandle buffer{};
  size_t capacityBytes = 0u;
  size_t lane = 0u;
  bool replaced = false;
};

class DynamicBufferRing {
  struct Lane {
    std::unique_ptr<Buffer> buffer;
    size_t capacityBytes = 0u;
    SubmissionHandle submission{};
  };

public:
  DynamicBufferRing(
      GPUDevice &gpu, BufferDesc policy, std::string_view debugName,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : gpu_(gpu), policy_(policy), debugName_(debugName), lanes_(memory) {
    policy_.size = 0u;
  }

  [[nodiscard]] Result<DynamicBufferAcquisition, std::string>
  acquire(uint64_t frameIndex, size_t requiredBytes, size_t minimumLaneCount) {
    if (pendingLane_) {
      return Result<DynamicBufferAcquisition, std::string>::makeError(
          "dynamic buffer ring already has a prepared lane");
    }
    minimumLaneCount = std::max(minimumLaneCount, size_t{1u});
    if (lanes_.size() < minimumLaneCount) {
      lanes_.resize(minimumLaneCount);
    }
    const size_t preferredLane = frameIndex % minimumLaneCount;
    size_t laneIndex = preferredLane;
    if (!makeReusable(lanes_[laneIndex])) {
      const auto available = std::ranges::find_if(
          lanes_, [this](Lane &lane) { return makeReusable(lane); });
      if (available == lanes_.end()) {
        laneIndex = lanes_.size();
        lanes_.emplace_back();
      } else {
        laneIndex = static_cast<size_t>(available - lanes_.begin());
      }
    }

    Lane &lane = lanes_[laneIndex];
    bool replaced = false;
    if (!lane.buffer || !lane.buffer->valid() ||
        lane.capacityBytes < requiredBytes) {
      BufferDesc desc = policy_;
      desc.size = nextDynamicBufferCapacity(lane.capacityBytes, requiredBytes);
      auto replacement = Buffer::create(gpu_, desc, debugName_);
      if (replacement.hasError()) {
        return Result<DynamicBufferAcquisition, std::string>::makeError(
            replacement.error());
      }
      lane.buffer = std::move(replacement.value());
      lane.capacityBytes = desc.size;
      replaced = true;
    }
    pendingLane_ = laneIndex;
    return Result<DynamicBufferAcquisition, std::string>::makeResult(
        DynamicBufferAcquisition{.buffer = lane.buffer->handle(),
                                 .capacityBytes = lane.capacityBytes,
                                 .lane = laneIndex,
                                 .replaced = replaced});
  }

  [[nodiscard]] Result<DynamicBufferAcquisition, std::string>
  upload(uint64_t frameIndex, size_t minimumLaneCount,
         std::span<const std::byte> bytes) {
    auto acquisition = acquire(frameIndex, bytes.size(), minimumLaneCount);
    if (acquisition.hasError()) {
      return acquisition;
    }
    auto update = gpu_.updateBuffer(acquisition.value().buffer, bytes, 0u);
    if (update.hasError()) {
      pendingLane_.reset();
      return Result<DynamicBufferAcquisition, std::string>::makeError(
          update.error());
    }
    return acquisition;
  }

  void submitPrepared(SubmissionHandle submission) noexcept {
    if (!pendingLane_) {
      return;
    }
    lanes_[*pendingLane_].submission = submission;
    pendingLane_.reset();
  }

  void abandonPrepared() noexcept { pendingLane_.reset(); }

  void reset() noexcept {
    pendingLane_.reset();
    lanes_.clear();
  }

  [[nodiscard]] size_t laneCount() const noexcept { return lanes_.size(); }

private:
  [[nodiscard]] bool makeReusable(Lane &lane) const {
    if (!nuri::isValid(lane.submission)) {
      return true;
    }
    if (!gpu_.isSubmissionComplete(lane.submission)) {
      return false;
    }
    lane.submission = {};
    return true;
  }

  GPUDevice &gpu_;
  BufferDesc policy_{};
  std::string debugName_;
  std::pmr::vector<Lane> lanes_;
  std::optional<size_t> pendingLane_{};
};

struct DynamicBufferLaneAcquisition {
  size_t lane = 0u;
  bool grew = false;
};

struct DynamicBufferRoleDesc {
  BufferUsage usage = BufferUsage::Storage;
  Storage storage = Storage::Device;
  size_t minimumBytes = 1u;
  size_t alignmentBytes = 1u;
  std::string_view debugName;
};

class DynamicBufferRoleRing {
  struct LaneBuffer {
    std::unique_ptr<Buffer> buffer;
    size_t capacityBytes = 0u;
  };

  struct Role {
    explicit Role(std::pmr::memory_resource *memory) : lanes(memory) {}
    std::pmr::vector<LaneBuffer> lanes;
    BufferDesc desc{};
    std::string debugName;
    size_t minimumBytes = 1u;
    size_t alignmentBytes = 1u;
    bool configured = false;
  };

public:
  DynamicBufferRoleRing(
      GPUDevice &gpu, std::initializer_list<DynamicBufferRoleDesc> roleDescs,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : gpu_(gpu), roles_(memory), submissions_(memory) {
    roles_.reserve(roleDescs.size());
    for (const DynamicBufferRoleDesc &desc : roleDescs) {
      roles_.emplace_back(memory);
      Role &role = roles_.back();
      role.desc.usage = desc.usage;
      role.desc.storage = desc.storage;
      role.minimumBytes = std::max(desc.minimumBytes, size_t{1u});
      role.alignmentBytes = std::max(desc.alignmentBytes, size_t{1u});
      role.debugName.assign(desc.debugName);
    }
  }

  [[nodiscard]] Result<bool, std::string>
  ensureLaneCount(size_t minimumLaneCount) {
    minimumLaneCount = std::max(minimumLaneCount, size_t{1u});
    if (submissions_.size() >= minimumLaneCount) {
      return Result<bool, std::string>::makeResult(false);
    }
    const size_t previousCount = submissions_.size();
    submissions_.resize(minimumLaneCount);
    for (Role &role : roles_) {
      role.lanes.resize(minimumLaneCount);
      if (!role.configured) {
        continue;
      }
      for (size_t lane = previousCount; lane < minimumLaneCount; ++lane) {
        auto result = ensureLaneBuffer(role, lane);
        if (result.hasError()) {
          return Result<bool, std::string>::makeError(result.error());
        }
      }
    }
    return Result<bool, std::string>::makeResult(true);
  }

  [[nodiscard]] Result<DynamicBufferLaneAcquisition, std::string>
  acquire(uint64_t frameIndex, size_t minimumLaneCount) {
    if (pendingLane_) {
      return Result<DynamicBufferLaneAcquisition, std::string>::makeError(
          "dynamic buffer role ring already has a prepared lane");
    }
    auto countResult = ensureLaneCount(minimumLaneCount);
    if (countResult.hasError()) {
      return Result<DynamicBufferLaneAcquisition, std::string>::makeError(
          countResult.error());
    }
    minimumLaneCount = std::max(minimumLaneCount, size_t{1u});
    size_t lane = frameIndex % minimumLaneCount;
    bool grew = countResult.value();
    if (!makeReusable(lane)) {
      size_t available = submissions_.size();
      for (size_t candidate = 0u; candidate < submissions_.size();
           ++candidate) {
        if (makeReusable(candidate)) {
          available = candidate;
          break;
        }
      }
      if (available == submissions_.size()) {
        lane = submissions_.size();
        auto growResult = ensureLaneCount(lane + 1u);
        if (growResult.hasError()) {
          return Result<DynamicBufferLaneAcquisition, std::string>::makeError(
              growResult.error());
        }
        grew = true;
      } else {
        lane = available;
      }
    }
    pendingLane_ = lane;
    return Result<DynamicBufferLaneAcquisition, std::string>::makeResult(
        {.lane = lane, .grew = grew});
  }

  template <typename OnReplacement>
  [[nodiscard]] Result<bool, std::string>
  ensureRole(size_t roleIndex, size_t requiredBytes,
             OnReplacement &&onReplacement) {
    if (roleIndex >= roles_.size()) {
      return Result<bool, std::string>::makeError(
          "dynamic buffer role index is out of range");
    }
    Role &role = roles_[roleIndex];
    const size_t requested = std::max(requiredBytes, role.minimumBytes);
    role.desc.size =
        ((requested + role.alignmentBytes - 1u) / role.alignmentBytes) *
        role.alignmentBytes;
    role.configured = true;
    bool replaced = false;
    for (size_t lane = 0u; lane < submissions_.size(); ++lane) {
      auto result = ensureLaneBuffer(role, lane);
      if (result.hasError()) {
        return Result<bool, std::string>::makeError(result.error());
      }
      if (result.value()) {
        replaced = true;
        onReplacement(lane);
      }
    }
    return Result<bool, std::string>::makeResult(replaced);
  }

  [[nodiscard]] Result<bool, std::string> ensureRole(size_t role,
                                                     size_t requiredBytes) {
    return ensureRole(role, requiredBytes, [](size_t) {});
  }

  [[nodiscard]] BufferHandle handle(size_t role, size_t lane) const noexcept {
    if (role >= roles_.size() || lane >= roles_[role].lanes.size()) {
      return {};
    }
    const auto &buffer = roles_[role].lanes[lane].buffer;
    return buffer ? buffer->handle() : BufferHandle{};
  }

  [[nodiscard]] size_t capacity(size_t role, size_t lane) const noexcept {
    return role < roles_.size() && lane < roles_[role].lanes.size()
               ? roles_[role].lanes[lane].capacityBytes
               : 0u;
  }

  [[nodiscard]] bool valid(size_t role, size_t lane) const noexcept {
    return nuri::isValid(handle(role, lane));
  }

  [[nodiscard]] size_t laneCount() const noexcept {
    return submissions_.size();
  }

  [[nodiscard]] std::optional<size_t> preparedLane() const noexcept {
    return pendingLane_;
  }

  [[nodiscard]] bool completed(size_t lane) {
    return lane < submissions_.size() && makeReusable(lane);
  }

  void submitPrepared(SubmissionHandle submission) noexcept {
    if (!pendingLane_) {
      return;
    }
    submissions_[*pendingLane_] = submission;
    pendingLane_.reset();
  }

  void abandonPrepared() noexcept { pendingLane_.reset(); }

  void reset() noexcept {
    pendingLane_.reset();
    submissions_.clear();
    for (Role &role : roles_) {
      role.lanes.clear();
      role.configured = false;
    }
  }

private:
  [[nodiscard]] Result<bool, std::string> ensureLaneBuffer(Role &role,
                                                           size_t lane) {
    LaneBuffer &slot = role.lanes[lane];
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= role.desc.size) {
      return Result<bool, std::string>::makeResult(false);
    }
    BufferDesc desc = role.desc;
    desc.size = nextDynamicBufferCapacity(slot.capacityBytes, desc.size);
    auto replacement = Buffer::create(gpu_, desc, role.debugName);
    if (replacement.hasError()) {
      return Result<bool, std::string>::makeError(replacement.error());
    }
    slot.buffer = std::move(replacement.value());
    slot.capacityBytes = desc.size;
    return Result<bool, std::string>::makeResult(true);
  }

  [[nodiscard]] bool makeReusable(size_t lane) {
    SubmissionHandle &submission = submissions_[lane];
    if (!nuri::isValid(submission)) {
      return true;
    }
    if (!gpu_.isSubmissionComplete(submission)) {
      return false;
    }
    submission = {};
    return true;
  }

  GPUDevice &gpu_;
  std::pmr::vector<Role> roles_;
  std::pmr::vector<SubmissionHandle> submissions_;
  std::optional<size_t> pendingLane_{};
};

[[nodiscard]] inline Result<bool, std::string>
ensureDynamicBufferCapacity(GPUDevice &gpu, std::unique_ptr<Buffer> &buffer,
                            size_t &capacityBytes, BufferDesc desc,
                            std::string_view debugName) {
  if (buffer && buffer->valid() && capacityBytes >= desc.size) {
    return Result<bool, std::string>::makeResult(false);
  }
  desc.size = nextDynamicBufferCapacity(capacityBytes, desc.size);
  auto replacementResult = Buffer::create(gpu, desc, debugName);
  if (replacementResult.hasError()) {
    return Result<bool, std::string>::makeError(replacementResult.error());
  }
  std::unique_ptr<Buffer> previous = std::move(buffer);
  buffer = std::move(replacementResult.value());
  capacityBytes = desc.size;
  previous.reset();
  return Result<bool, std::string>::makeResult(true);
}

inline void retireDynamicBuffer(std::unique_ptr<Buffer> &buffer,
                                size_t &capacityBytes) {
  buffer.reset();
  capacityBytes = 0u;
}

} // namespace nuri
