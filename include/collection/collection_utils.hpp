/**
 * @file collection_utils.hpp
 * @brief Utility functions for collection operations
 * 
 * This header provides a set of template utilities for common collection operations
 * that are not readily available in the standard library. These utilities are designed
 * to work with any STL-compatible container types.
    *
 */

#pragma once

#include <optional>

namespace wheel {

    /**
     * @brief Finds the first element in a collection that matches a predicate
     * 
     * Iterates through a collection and returns the first element that satisfies
     * the given predicate function. If no matching element is found, returns std::nullopt.
     * 
     * @tparam T The type of elements stored in the collection
     * @tparam COLLECTION The container type (deduced from parameter)
     * @tparam PREDICATE The predicate function type (deduced from parameter)
     * 
     * @param collection The collection to search through
     * @param predicate A callable that takes an element of type T and returns a boolean
     * 
     * @return std::optional<T> The first matching element if found, std::nullopt otherwise
     * 
     * @example
     *   std::vector<int> numbers = {1, 2, 3, 4, 5};
     *   auto even = wheel::find_any_match<int>(numbers, [](int n) { return n % 2 == 0; });
     *   // even contains 2
     */
    template <typename T, typename COLLECTION, typename PREDICATE>
    std::optional<T> find_any_match(const COLLECTION &collection, PREDICATE predicate) {
        for (const auto &item : collection) {
            if (predicate(item)) {
                return item;
            }
        }
        return std::nullopt;
    }

} // namespace wheel
