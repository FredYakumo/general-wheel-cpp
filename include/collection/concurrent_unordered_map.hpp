/**
 * @file concurrent_unordered_map.hpp
 * @brief A thread-safe unordered map
 */

#pragma once

#include "../shared_guarded_ref.hpp"
#include <functional>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <utility>

namespace wheel {

    /**
     * @brief Thread-safe unordered map container with concurrent access support
     *
     * @tparam Key The key type of the map
     * @tparam Value The mapped type of the map
     * @tparam Hash The hash function object type (default: std::hash<Key>)
     * @tparam KeyEqual The key equality comparison function object type (default:
     * std::equal_to<Key>)
     * @tparam Allocator The allocator type (default: std::allocator<std::pair<const
     * Key, Value>>)
     *
     * This implementation provides basic thread safety guarantees for all
     * operations using reader-writer locks (shared_mutex). Read operations can
     * execute concurrently, while write operations require exclusive access.
     */
    template <typename Key, typename Value, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>,
              typename Allocator = std::allocator<std::pair<const Key, Value>>>
    class concurrent_unordered_map {
      private:
        template <class Lock, class Map> class range_proxy;

      public:
        using key_type = Key;
        using mapped_type = Value;
        using value_type = std::pair<const Key, Value>;
        using size_type = typename std::unordered_map<Key, Value, Hash, KeyEqual, Allocator>::size_type;
        using hasher = Hash;
        using key_equal = KeyEqual;
        using allocator_type = Allocator;

        /**
         * @brief Constructs an empty container with default-constructed allocator
         */
        concurrent_unordered_map() = default;

        /**
         * @brief Constructs an empty container with specified allocator
         * @param alloc The allocator to use for all memory allocations
         */
        explicit concurrent_unordered_map(const Allocator &alloc) : m_map(alloc) {}

        /**
         * @brief Copy constructor.
         */
        concurrent_unordered_map(const concurrent_unordered_map &other) {
            std::unique_lock lock(m_mutex, std::defer_lock);
            std::shared_lock other_lock(other.m_mutex, std::defer_lock);

            // Lock both mutexes without deadlock
            std::lock(lock, other_lock);
            m_map = other.m_map;
        }

        /**
         * @brief Move constructor.
         */
        concurrent_unordered_map(concurrent_unordered_map &&other) noexcept {
            std::unique_lock lock(m_mutex, std::defer_lock);
            std::shared_lock other_lock(other.m_mutex, std::defer_lock);

            // Lock both mutexes without deadlock
            std::lock(lock, other_lock);
            m_map = std::move(other.m_map);
        }

        /**
         * @brief Copy assignment operator.
         */
        concurrent_unordered_map &operator=(const concurrent_unordered_map &other) {
            if (this != &other) {
                std::unique_lock lock(m_mutex, std::defer_lock);
                std::shared_lock other_lock(other.m_mutex, std::defer_lock);

                // Lock both mutexes without deadlock
                std::lock(lock, other_lock);

                m_map = other.m_map;
            }
            return *this;
        }

        /**
         * @brief Move assignment operator.
         */
        concurrent_unordered_map &operator=(concurrent_unordered_map &&other) noexcept {
            if (this != &other) {
                std::unique_lock lock(m_mutex, std::defer_lock);
                std::shared_lock other_lock(other.m_mutex, std::defer_lock);

                // Lock both mutexes without deadlock
                std::lock(lock, other_lock);
                m_map = std::move(other.m_map);
            }
            return *this;
        }

        /**
         * @brief Move constructor that takes an rvalue reference to an std::unordered_map.
         * @param map The unordered_map to be moved into the concurrent_unordered_map.
         */
        explicit concurrent_unordered_map(std::unordered_map<Key, Value, Hash, KeyEqual, Allocator> &&map)
            : m_map(std::move(map)) {}

        /**
         * @brief Move assignment operator from std::unordered_map.
         *
         * Atomically replaces the contents of the concurrent map with those of another unordered_map.
         * The operation is thread-safe and performed under a lock.
         *
         * @param other The unordered_map to move from.
         * @return Reference to this map after the move operation.
         */
        concurrent_unordered_map &operator=(std::unordered_map<Key, Value, Hash, KeyEqual, Allocator> &&other) {
            std::unique_lock lock(m_mutex);
            m_map = std::move(other);
            return *this;
        }

        /**
         * @brief Inserts or updates an element in the container
         * @param key The key of the element to insert/update
         * @param value The value to associate with the key
         */
        void insert_or_assign(const Key &key, const Value &value) {
            std::unique_lock unique_lock(m_mutex);
            m_map.insert_or_assign(key, value);
        }

        /**
         * @brief Attempts to insert an element if the key doesn't exist
         * @return true if insertion succeeded, false if key already exists
         */
        bool try_insert(const Key &key, const Value &value) {
            std::unique_lock lock(m_mutex);
            return m_map.try_emplace(key, value).second;
        }

        /**
         * @brief Finds an element with given key
         * @return A locked reference to the value if found, empty otherwise
         */
        std::optional<shared_guarded_ref<const Value, std::shared_mutex>> find(const Key &key) const {
            std::shared_lock lock(m_mutex);
            auto it = m_map.find(key);
            if (it != std::cend(m_map)) {
                return shared_guarded_ref<const Value, std::shared_mutex>(it->second, std::move(lock));
            }
            return std::nullopt;
        }

        /**
         * @brief Finds an element with given key
         * @return A locked reference to the value if found, empty otherwise
         */
        std::optional<shared_guarded_ref<Value, std::shared_mutex>> find(const Key &key) {
            std::shared_lock lock(m_mutex);
            auto it = m_map.find(key);
            if (it != std::cend(m_map)) {
                return shared_guarded_ref<Value, std::shared_mutex>(it->second, std::move(lock));
            }
            return std::nullopt;
        }

        /**
         * @brief Removes the element with given key
         * @return true if element was removed, false if key didn't exist
         */
        bool erase(const Key &key) {
            std::unique_lock lock(m_mutex);
            return m_map.erase(key) > 0;
        }

        /**
         * @brief Checks if container contains element with specific key
         */
        bool contains(const Key &key) const {
            std::shared_lock lock(m_mutex);
            return m_map.contains(key);
        }

        /**
         * @brief Returns the number of elements in the container
         */
        size_type size() const {
            std::shared_lock lock(m_mutex);
            return m_map.size();
        }

        /**
         * @brief Checks if the container is empty
         */
        bool empty() const {
            std::shared_lock lock(m_mutex);
            return m_map.empty();
        }

        /**
         * @brief Erases all elements from the container
         */
        void clear() {
            std::unique_lock lock(m_mutex);
            m_map.clear();
        }

        /**
         * @brief Applies given function to each element (read-only)
         * @tparam Func Callable type accepting (key, value) parameters
         * @param func The function to apply
         */
        template <typename Func> void for_each(Func func) const {
            std::shared_lock lock(m_mutex);
            for (const auto &[key, value] : m_map) {
                func(key, value);
            }
        }

        /**
         * @brief Applies given function to each element (allows modification)
         * @tparam Func Callable type accepting (key, value) parameters
         * @param func The function to apply
         */
        template <typename Func> void modify(Func func) {
            std::unique_lock lock(m_mutex);
            for (auto &[key, value] : m_map) {
                func(key, value);
            }
        }

        /**
         * @brief Returns a range-like object for read-only iteration.
         * The returned proxy object holds a shared lock for the duration of the iteration.
         * @return A proxy object with begin() and end() methods.
         */
        auto iter() const {
            return range_proxy<std::shared_lock<std::shared_mutex>,
                               const std::unordered_map<Key, Value, Hash, KeyEqual, Allocator>>{m_map, m_mutex};
        }

        /**
         * @brief Returns a range-like object for mutable iteration.
         * The returned proxy object holds a unique lock for the duration of the iteration.
         * @return A proxy object with begin() and end() methods.
         */
        auto iter() {
            return range_proxy<std::unique_lock<std::shared_mutex>,
                               std::unordered_map<Key, Value, Hash, KeyEqual, Allocator>>{m_map, m_mutex};
        }

        /**
         * @brief Provides thread-safe access to the underlying container (read-only)
         * @tparam Func Callable type accepting const reference to the underlying map
         * @return Result of the function invocation
         */
        template <typename Func> auto access(Func func) const {
            std::shared_lock lock(m_mutex);
            return func(m_map);
        }

        /**
         * @brief Provides thread-safe access to the underlying container (read-write)
         * @tparam Func Callable type accepting reference to the underlying map
         * @return Result of the function invocation
         */
        template <typename Func> auto modify_map(Func func) {
            std::unique_lock lock(m_mutex);
            return func(m_map);
        }

        /**
         * @brief Returns the allocator associated with the container
         */
        allocator_type get_allocator() const noexcept { return m_map.get_allocator(); }

        /**
         * @brief Gets or creates a value for the given key
         * @param key The key to look up
         * @param value_factory A factory function that creates the default value if key doesn't exist
         * @return A locked reference to the value (existing or newly created)
         */
        template <typename ValueFactory>
        shared_guarded_ref<Value, std::shared_mutex> get_or_create_value(const Key &key, ValueFactory &&value_factory) {
            // Try with shared lock
            std::shared_lock shared_lock(m_mutex);
            auto it = m_map.find(key);
            if (it != std::cend(m_map)) {
                return shared_guarded_ref<Value, std::shared_mutex>(it->second, std::move(shared_lock));
            }

            // Key not found, upgrade to unique lock
            shared_lock.unlock();
            std::unique_lock unique_lock(m_mutex);

            // Double-check pattern in case another thread inserted while upgrading
            it = m_map.find(key);
            if (it != std::cend(m_map)) {
                unique_lock.unlock();
                return shared_guarded_ref<Value, std::shared_mutex>(it->second, std::shared_lock(m_mutex));
            }

            // insert new value
            auto [new_it, inserted] = m_map.try_emplace(key, std::forward<ValueFactory>(value_factory)());

            unique_lock.unlock();
            return shared_guarded_ref<Value, std::shared_mutex>(new_it->second, std::shared_lock(m_mutex));
        }

        /**
         * @brief Gets or emplace a value for the given key
         * @param key The key to look up
         * @param Args Arguments that use to construct value directly.
         * @return A locked reference to the value (existing or newly created)
         */
        template <typename... Args>
        shared_guarded_ref<Value, std::shared_mutex> get_or_emplace_value(const Key &key, Args &&...args) {
            // Try with shared lock
            std::shared_lock shared_lock(m_mutex);
            auto it = m_map.find(key);
            if (it != std::cend(m_map)) {
                return shared_guarded_ref<Value, std::shared_mutex>(it->second, std::move(shared_lock));
            }

            // Key not found, upgrade to unique lock
            shared_lock.unlock();
            std::unique_lock unique_lock(m_mutex);

            // Double-check pattern in case another thread inserted while upgrading
            it = m_map.find(key);
            if (it != std::cend(m_map)) {
                unique_lock.unlock();
                return shared_guarded_ref<Value, std::shared_mutex>(it->second, std::shared_lock(m_mutex));
            }

            // insert new value
            auto [new_it, inserted] = m_map.try_emplace(key, std::forward<Args>(args)...);

            unique_lock.unlock();
            return shared_guarded_ref<Value, std::shared_mutex>(new_it->second, std::shared_lock(m_mutex));
        }

        std::optional<Value> pop(const Key &key) {
            // Try with shared lock
            std::shared_lock shared_lock(m_mutex);
            auto it = m_map.find(key);
            if (it == std::cend(m_map)) {
                return std::nullopt;
            }

            // Key found, upgrade to unique lock
            shared_lock.unlock();
            std::unique_lock unique_lock(m_mutex);

            // Double-check pattern in case another thread removed while upgrading
            it = m_map.find(key);
            if (it == std::cend(m_map)) {
                return std::nullopt;
            }

            // Get the value and remove the entry
            Value value = std::move(it->second);
            m_map.erase(it);

            return value;
        }

      private:
        template <class Lock, class Map> class range_proxy {
          public:
            range_proxy(Map &map, typename Lock::mutex_type &mutex) : map_(map), lock_(mutex) {}

            auto begin() { return map_.begin(); }
            auto end() { return map_.end(); }

          private:
            Map &map_;
            Lock lock_;
        };

        std::unordered_map<Key, Value, Hash, KeyEqual, Allocator> m_map;
        mutable std::shared_mutex m_mutex;
    };

} // namespace wheel