#pragma once

#include <ankerl/unordered_dense.h>

#include <functional>
#include <memory_resource>

namespace nuri {

template <typename Key, typename Value, typename Hash = std::hash<Key>,
          typename Eq = std::equal_to<Key>>
using HashMap = ankerl::unordered_dense::map<Key, Value, Hash, Eq>;

template <typename Key, typename Value, typename Hash = std::hash<Key>,
          typename Eq = std::equal_to<Key>>
using PmrHashMap = ankerl::unordered_dense::pmr::map<Key, Value, Hash, Eq>;

} // namespace nuri
