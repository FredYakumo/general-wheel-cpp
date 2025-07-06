/**
 * @file concurrent_hashset.hpp
 * @brief A thread-safe unordered set
 */

#pragma once

#include "../shared_guarded_ref.hpp"
#include <functional>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_set>

namespace wheel {

    /**
     * @brief Thread-safe unordered set container with concurrent access support
     *
     * @tparam Value The value type of the set
     * @tparam Hash The hash function object type (default: std::hash<Value>)
     * @tparam KeyEqual The value equality comparison function object type (default: std::equal_to<Value>)
     * @tparam Allocator The allocator type (default: std::allocator<Value>)
     *
     * This implementation provides basic thread safety guarantees for all
     * operations using reader-writer locks (shared_mutex). Read operations can
     * execute concurrently, while write operations require exclusive access.
     */
    template <typename Value, typename Hash = std::hash<Value>, 
              typename Equal = std::equal_to<Value>,
              typename Allocator = std::allocator<Value>>
    class concurrent_hashset {
      private:
        template <class Lock, class Set> class range_proxy;

      public:
        using value_type = Value;
        using size_type = typename std::unordered_set<Value, Hash, Equal, Allocator>::size_type;
        using hasher = Hash;
        using key_equal = Equal;
        using allocator_type = Allocator;

        /**
         * @brief Constructs an empty container
         */
        concurrent_hashset() = default;

        /**
         * @brief Constructs an empty container with specified allocator
         */
        explicit concurrent_hashset(const Allocator& alloc) : m_set(alloc) {}

        /**
         * @brief Copy constructor
         */
        concurrent_hashset(const concurrent_hashset& other) {
            std::unique_lock lock(m_mutex, std::defer_lock);
            std::shared_lock other_lock(other.m_mutex, std::defer_lock);
            std::lock(lock, other_lock);
            m_set = other.m_set;
        }

        /**
         * @brief Move constructor
         */
        concurrent_hashset(concurrent_hashset&& other) noexcept {
            std::unique_lock lock(m_mutex, std::defer_lock);
            std::unique_lock other_lock(other.m_mutex, std::defer_lock);
            std::lock(lock, other_lock);
            m_set = std::move(other.m_set);
        }

        /**
         * @brief Copy assignment operator
         */
        concurrent_hashset& operator=(const concurrent_hashset& other) {
            if (this != &other) {
                std::unique_lock lock(m_mutex, std::defer_lock);
                std::shared_lock other_lock(other.m_mutex, std::defer_lock);
                std::lock(lock, other_lock);
                m_set = other.m_set;
            }
            return *this;
        }

        /**
         * @brief Move assignment operator
         */
        concurrent_hashset& operator=(concurrent_hashset&& other) noexcept {
            if (this != &other) {
                std::unique_lock lock(m_mutex, std::defer_lock);
                std::unique_lock other_lock(other.m_mutex, std::defer_lock);
                std::lock(lock, other_lock);
                m_set = std::move(other.m_set);
            }
            return *this;
        }

        /**
         * @brief Attempts to insert an element
         * @return true if insertion succeeded, false if value already exists
         */
        bool insert(const Value& value) {
            std::unique_lock lock(m_mutex);
            return m_set.insert(value).second;
        }

        /**
         * @brief Attempts to insert an element using move semantics
         * @return true if insertion succeeded, false if value already exists
         */
        bool insert(Value&& value) {
            std::unique_lock lock(m_mutex);
            return m_set.insert(std::move(value)).second;
        }

        /**
         * @brief Checks if container contains the specified value
         */
        bool contains(const Value& value) const {
            std::shared_lock lock(m_mutex);
            return m_set.contains(value);
        }

        /**
         * @brief Removes the specified value
         * @return true if value was removed, false if it didn't exist
         */
        bool erase(const Value& value) {
            std::unique_lock lock(m_mutex);
            return m_set.erase(value) > 0;
        }

        /**
         * @brief Returns the number of elements
         */
        size_type size() const {
            std::shared_lock lock(m_mutex);
            return m_set.size();
        }

        /**
         * @brief Checks if the container is empty
         */
        bool empty() const {
            std::shared_lock lock(m_mutex);
            return m_set.empty();
        }

        /**
         * @brief Erases all elements
         */
        void clear() {
            std::unique_lock lock(m_mutex);
            m_set.clear();
        }

        /**
         * @brief Applies given function to each element (read-only)
         */
        template <typename Func>
        void for_each(Func func) const {
            std::shared_lock lock(m_mutex);
            for (const auto& value : m_set) {
                func(value);
            }
        }

        /**
         * @brief Returns a range-like object for read-only iteration
         */
        auto iter() const {
            return range_proxy<std::shared_lock<std::shared_mutex>,
                             const std::unordered_set<Value, Hash, Equal, Allocator>>{m_set, m_mutex};
        }

        /**
         * @brief Returns a range-like object for mutable iteration
         */
        auto iter() {
            return range_proxy<std::unique_lock<std::shared_mutex>,
                             std::unordered_set<Value, Hash, Equal, Allocator>>{m_set, m_mutex};
        }

        /**
         * @brief Returns the allocator associated with the container
         */
        allocator_type get_allocator() const noexcept {
            return m_set.get_allocator();
        }

      private:
        template <class Lock, class Set>
        class range_proxy {
          public:
            range_proxy(Set& set, typename Lock::mutex_type& mutex)
                : set_(set), lock_(mutex) {}

            auto begin() { return set_.begin(); }
            auto end() { return set_.end(); }

          private:
            Set& set_;
            Lock lock_;
        };

        std::unordered_set<Value, Hash, Equal, Allocator> m_set;
        mutable std::shared_mutex m_mutex;
    };

} // namespace wheel