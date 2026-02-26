#pragma once

#include <ankerl/unordered_dense.h>

#include <functional>

namespace nuri {

template <typename Key, typename Hash = std::hash<Key>,
          typename Eq = std::equal_to<Key>>
using HashSet = ankerl::unordered_dense::set<Key, Hash, Eq>;

} // namespace nuri
