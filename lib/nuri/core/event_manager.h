#pragma once

#include "nuri/core/log.h"
#include "nuri/defines.h"

#include <algorithm>
#include <atomic>

namespace nuri {

enum class EventChannel : uint8_t {
  Generic = 0,
  RawInput = 1,
  Input = 2,
};

struct SubscriptionToken {
  EventChannel channel = EventChannel::Generic;
  uint32_t typeId = UINT32_MAX;
  uint32_t handlerId = 0;
};

class NURI_API EventManager {
public:
  explicit EventManager(std::pmr::memory_resource &upstream);
  ~EventManager();

  EventManager(const EventManager &) = delete;
  EventManager &operator=(const EventManager &) = delete;
  EventManager(EventManager &&) = delete;
  EventManager &operator=(EventManager &&) = delete;

  template <typename T> using Handler = bool (*)(const T &event, void *user);

  template <typename T>
  SubscriptionToken subscribe(EventChannel channel, Handler<T> handler,
                              void *user, int32_t priority = 0);
  bool unsubscribe(const SubscriptionToken &token);
  template <typename T>
  void emit(const T &event, EventChannel channel = EventChannel::Generic);

  void dispatch(EventChannel channel);
  void clear(EventChannel channel);
  void clear();

private:
  static constexpr size_t kChannelCount =
      static_cast<size_t>(EventChannel::Input) + 1;

  struct QueuedEvent {
    uint32_t typeId = 0;
    const void *data = nullptr;
  };

  struct HandlerListBase {
    virtual ~HandlerListBase() = default;
    virtual bool dispatch(const void *event, bool stopOnConsume) = 0;
    virtual bool unsubscribe(uint32_t handlerId) = 0;
    virtual void compact() = 0;
  };

  template <typename T> struct HandlerList final : HandlerListBase {
    struct Entry {
      uint32_t id = 0;
      int32_t priority = 0;
      Handler<T> fn = nullptr;
      void *user = nullptr;
      bool active = true;
    };

    explicit HandlerList(std::pmr::memory_resource &mr) : entries(&mr) {}

    bool dispatch(const void *event, bool stopOnConsume) override {
      const T &typed = *static_cast<const T *>(event);
      inDispatch = true;

      bool consumed = false;
      for (Entry &entry : entries) {
        if (!entry.active || entry.fn == nullptr) {
          continue;
        }
        const bool handled = entry.fn(typed, entry.user);
        if (stopOnConsume && handled) {
          consumed = true;
          break;
        }
      }

      inDispatch = false;
      if (needsCompact) {
        compact();
      }
      return consumed;
    }

    void add(uint32_t id, int32_t priority, Handler<T> fn, void *user) {
      entries.push_back({
          .id = id,
          .priority = priority,
          .fn = fn,
          .user = user,
          .active = true,
      });

      std::stable_sort(entries.begin(), entries.end(),
                       [](const Entry &a, const Entry &b) {
                         if (a.priority != b.priority) {
                           return a.priority > b.priority;
                         }
                         return a.id < b.id;
                       });
    }

    bool unsubscribe(uint32_t handlerId) override {
      for (Entry &entry : entries) {
        if (entry.id != handlerId || !entry.active) {
          continue;
        }

        if (inDispatch) {
          entry.active = false;
          needsCompact = true;
          return true;
        }

        entry.active = false;
        compact();
        return true;
      }
      return false;
    }

    void compact() override {
      entries.erase(
          std::remove_if(entries.begin(), entries.end(),
                         [](const Entry &entry) { return !entry.active; }),
          entries.end());
      needsCompact = false;
    }

    std::pmr::vector<Entry> entries;
    bool inDispatch = false;
    bool needsCompact = false;
  };

  struct HandlerListSlot {
    HandlerListBase *list = nullptr;
    void (*destroy)(HandlerListBase *, std::pmr::memory_resource &) = nullptr;
  };

  struct ChannelState {
    explicit ChannelState(std::pmr::memory_resource &mr)
        : queue(&mr), handlerLists(&mr) {}

    std::pmr::vector<QueuedEvent> queue;
    std::pmr::vector<HandlerListSlot> handlerLists;
  };

  template <typename T>
  static void destroyList(HandlerListBase *list,
                          std::pmr::memory_resource &mr) {
    auto *typed = static_cast<HandlerList<T> *>(list);
    std::destroy_at(typed);
    std::pmr::polymorphic_allocator<HandlerList<T>> alloc(&mr);
    alloc.deallocate(typed, 1);
  }

  static constexpr size_t channelIndex(EventChannel channel) {
    return static_cast<size_t>(channel);
  }

  ChannelState &stateFor(EventChannel channel) {
    const size_t idx = channelIndex(channel);
    NURI_ASSERT(idx < kChannelCount, "Invalid event channel index");
    return channels_[idx];
  }

  template <typename T> HandlerList<T> &getOrCreateList(EventChannel channel) {
    const uint32_t id = typeId<T>();
    ChannelState &state = stateFor(channel);
    std::pmr::vector<HandlerListSlot> &handlerSlots = state.handlerLists;
    if (handlerSlots.size() <= id) {
      handlerSlots.resize(id + 1);
    }

    HandlerListSlot &slot = handlerSlots[id];
    if (!slot.list) {
      std::pmr::polymorphic_allocator<HandlerList<T>> alloc(&upstream_);
      HandlerList<T> *list = alloc.allocate(1);
      std::construct_at(list, upstream_);
      slot.list = list;
      slot.destroy = &destroyList<T>;
    }

    return *static_cast<HandlerList<T> *>(slot.list);
  }

  static uint32_t acquireTypeId();

  template <typename T> static uint32_t typeId() {
    static const uint32_t id = acquireTypeId();
    return id;
  }

  bool allQueuesEmpty() const {
    for (const ChannelState &state : channels_) {
      if (!state.queue.empty()) {
        return false;
      }
    }
    return true;
  }

  std::pmr::memory_resource &upstream_;
  std::pmr::monotonic_buffer_resource arena_;
  std::array<ChannelState, kChannelCount> channels_;
  std::atomic<uint32_t> nextHandlerId_{1};
};

template <typename T>
SubscriptionToken EventManager::subscribe(EventChannel channel,
                                          Handler<T> handler, void *user,
                                          int32_t priority) {
  NURI_ASSERT(handler != nullptr, "EventManager handler must not be null");
  const uint32_t handlerId =
      nextHandlerId_.fetch_add(1, std::memory_order_relaxed);
  getOrCreateList<T>(channel).add(handlerId, priority, handler, user);
  NURI_LOG_DEBUG(
      "EventManager::subscribe: Subscribed to type ID %u on channel %u",
      typeId<T>(), static_cast<uint32_t>(channel));
  return SubscriptionToken{
      .channel = channel,
      .typeId = typeId<T>(),
      .handlerId = handlerId,
  };
}

template <typename T>
void EventManager::emit(const T &event, EventChannel channel) {
  static_assert(std::is_trivially_copyable_v<T>,
                "EventManager events must be trivially copyable");
  static_assert(std::is_trivially_destructible_v<T>,
                "EventManager events must be trivially destructible");

  void *storage = arena_.allocate(sizeof(T), alignof(T));
  std::memcpy(storage, &event, sizeof(T));
  stateFor(channel).queue.push_back({typeId<T>(), storage});
}

} // namespace nuri
