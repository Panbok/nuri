#pragma once

#include <ankerl/unordered_dense.h>

#include <memory_resource>
#include <unordered_map>
#include <utility>

namespace nuri {

template <typename Key, typename Value, typename Hash = std::hash<Key>,
          typename Eq = std::equal_to<Key>>
using HashMap = ankerl::unordered_dense::map<Key, Value, Hash, Eq>;

template <typename Key, typename Value, typename Hash = std::hash<Key>,
          typename Eq = std::equal_to<Key>>
using PmrHashMap = std::pmr::unordered_map<Key, Value, Hash, Eq>;

} // namespace nuri
