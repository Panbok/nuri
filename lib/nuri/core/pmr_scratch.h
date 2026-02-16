#pragma once

#include <exception>
#include <memory_resource>

namespace nuri {

class ScratchArena final {
public:
  explicit ScratchArena(
      std::pmr::memory_resource *upstream = std::pmr::get_default_resource())
      : pool_(upstream ? upstream : std::pmr::get_default_resource()),
        arena_(&pool_) {}

  ScratchArena(const ScratchArena &) = delete;
  ScratchArena &operator=(const ScratchArena &) = delete;
  ScratchArena(ScratchArena &&) = delete;
  ScratchArena &operator=(ScratchArena &&) = delete;

  [[nodiscard]] std::pmr::memory_resource *resource() noexcept {
    return &arena_;
  }

  void reset() noexcept { arena_.release(); }

private:
  friend class ScopedScratch;
  std::pmr::unsynchronized_pool_resource pool_;
  std::pmr::monotonic_buffer_resource arena_;
  bool scopeActive_ = false;
};

class ScopedScratch final {
public:
  explicit ScopedScratch(ScratchArena &arena) noexcept : arena_(arena) {
    if (arena_.scopeActive_) {
      std::terminate();
    }
    arena_.scopeActive_ = true;
  }
  ~ScopedScratch() noexcept {
    arena_.scopeActive_ = false;
    arena_.reset();
  }

  ScopedScratch(const ScopedScratch &) = delete;
  ScopedScratch &operator=(const ScopedScratch &) = delete;
  ScopedScratch(ScopedScratch &&) = delete;
  ScopedScratch &operator=(ScopedScratch &&) = delete;

  // Scratch-backed allocations must not outlive this guard's scope.
  // Do not nest ScopedScratch guards over the same ScratchArena.
  [[nodiscard]] std::pmr::memory_resource *resource() noexcept {
    return arena_.resource();
  }

private:
  ScratchArena &arena_;
};

} // namespace nuri
