#include "event_manager.h"

namespace nuri {
namespace {
std::atomic<uint32_t> g_nextTypeId{0};
} // namespace

EventManager::EventManager(std::pmr::memory_resource &upstream)
    : upstream_(upstream), arena_(&upstream_),
      channels_(makeChannels(upstream_, std::make_index_sequence<kChannelCount>{})) {
  NURI_LOG_DEBUG("EventManager::EventManager: Event manager created");
}

EventManager::~EventManager() {
  for (ChannelState &state : channels_) {
    for (HandlerListSlot &slot : state.handlerLists) {
      if (!slot.list) {
        continue;
      }
      slot.destroy(slot.list, upstream_);
      slot.list = nullptr;
      slot.destroy = nullptr;
    }
    state.queue.clear();
  }
  arena_.release();
  NURI_LOG_DEBUG("EventManager::~EventManager: Event manager destroyed");
}

bool EventManager::unsubscribe(const SubscriptionToken &token) {
  if (token.typeId == UINT32_MAX || token.handlerId == 0) {
    return false;
  }

  std::pmr::vector<HandlerListSlot> &slots =
      stateFor(token.channel).handlerLists;
  if (token.typeId >= slots.size()) {
    return false;
  }

  HandlerListSlot &slot = slots[token.typeId];
  if (!slot.list) {
    return false;
  }

  NURI_LOG_DEBUG("EventManager::unsubscribe: Unsubscribing from type ID %u",
                 token.typeId);

  return slot.list->unsubscribe(token.handlerId);
}

void EventManager::dispatch(EventChannel channel) {
  ChannelState &state = stateFor(channel);
  std::pmr::vector<QueuedEvent> &queue = state.queue;
  std::pmr::vector<HandlerListSlot> &handlerLists = state.handlerLists;
  const bool stopOnConsume = (channel == EventChannel::Input);

  for (const QueuedEvent &event : queue) {
    if (event.typeId >= handlerLists.size()) {
      continue;
    }

    HandlerListSlot &slot = handlerLists[event.typeId];
    if (!slot.list) {
      continue;
    }

    (void)slot.list->dispatch(event.data, stopOnConsume);
  }

  clear(channel);
}

void EventManager::clear(EventChannel channel) {
  std::pmr::vector<QueuedEvent> &queue = stateFor(channel).queue;
  queue.clear();
  if (allQueuesEmpty()) {
    arena_.release();
  }
}

void EventManager::clear() {
  for (ChannelState &state : channels_) {
    state.queue.clear();
  }
  arena_.release();
}

uint32_t EventManager::acquireTypeId() {
  return g_nextTypeId.fetch_add(1, std::memory_order_relaxed);
}

} // namespace nuri
