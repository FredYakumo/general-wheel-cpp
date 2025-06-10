/**
 * @file concurrent_vector.hpp
 * @brief A thread-safe vector implementation with concurrent access support
 */

#pragma once

#include <algorithm>
#include <functional>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <utility>
#include <vector>
#include "../shared_guarded_ref.hpp"

namespace wheel {
    /**
     * @brief Thread-safe vector container with concurrent access support
     *
     * @tparam T The element type stored in the vector
     * @tparam Allocator The allocator type (default: std::allocator<T>)
     *
     * This implementation provides thread safety guarantees for all operations
     * using a combination of reader-writer locks (shared_mutex) for read operations
     * and exclusive locks (unique_lock) for write operations. The design prioritizes
     * thread safety while maintaining reasonable performance characteristics.
     */
    template <typename T, typename Allocator = std::allocator<T>> class concurrent_vector {
      public:
        using value_type = T;
        using allocator_type = Allocator;
        using size_type = typename std::vector<T, Allocator>::size_type;
        using reference = T &;
        using const_reference = const T &;

        /**
         * @brief Constructs an empty container with default-constructed allocator
         */
        concurrent_vector() = default;

        /**
         * @brief Constructs an empty container with specified allocator
         * @param alloc The allocator to use for all memory allocations
         */
        explicit concurrent_vector(const Allocator &alloc) : m_vec(alloc) {}

        /**
         * @brief Appends the given element to the end of the container
         * @param value The value of the element to append
         */
        void push_back(const T &value) {
            std::unique_lock lock(m_mutex);
            m_vec.push_back(value);
        }

        /**
         * @brief Appends the given element to the end of the container (move version)
         * @param value The value of the element to append
         */
        void push_back(T &&value) {
            std::unique_lock lock(m_mutex);
            m_vec.push_back(std::move(value));
        }

        /**
         * @brief Accesses specified element with bounds checking
         * @param pos Position of the element to return
         * @return std::optional containing the guarded element if pos is valid
         */
        std::optional<shared_guarded_ref<const T, std::shared_mutex>> at(size_type pos) const {
            std::shared_lock lock(m_mutex);
            if (pos >= m_vec.size()) {
                return std::nullopt;
            }
            return shared_guarded_ref<const T, std::shared_mutex>(m_vec[pos], std::move(lock));
        }

        /**
         * @brief Accesses specified element with bounds checking
         * @param pos Position of the element to return
         * @return std::optional containing the guarded element if pos is valid
         */
        std::optional<shared_guarded_ref<T, std::shared_mutex>> at(size_type pos) {
            std::shared_lock lock(m_mutex);
            if (pos >= m_vec.size()) {
                return std::nullopt;
            }
            return shared_guarded_ref<T, std::shared_mutex>(m_vec[pos], std::move(lock));
        }

        /**
         * @brief Accesses specified element without bounds checking
         * @param pos Position of the element to return
         * @return std::optional containing the guarded element if pos is valid
         */
        std::optional<shared_guarded_ref<const T, std::shared_mutex>> operator[](size_type pos) const {
            std::shared_lock lock(m_mutex);
            if (pos >= m_vec.size()) {
                return std::nullopt;
            }
            return shared_guarded_ref<const T, std::shared_mutex>(m_vec[pos], std::move(lock));
        }

        /**
         * @brief Accesses specified element without bounds checking
         * @param pos Position of the element to return
         * @return std::optional containing the guarded element if pos is valid
         */
        std::optional<shared_guarded_ref<T, std::shared_mutex>> operator[](size_type pos) {
            std::shared_lock lock(m_mutex);
            if (pos >= m_vec.size()) {
                return std::nullopt;
            }
            return shared_guarded_ref<T, std::shared_mutex>(m_vec[pos], std::move(lock));
        }

        /**
         * @brief Returns the number of elements in the container
         */
        size_type size() const {
            std::shared_lock lock(m_mutex);
            return m_vec.size();
        }

        /**
         * @brief Checks if the container is empty
         */
        bool empty() const {
            std::shared_lock lock(m_mutex);
            return m_vec.empty();
        }

        /**
         * @brief Erases all elements from the container
         */
        void clear() {
            std::unique_lock lock(m_mutex);
            m_vec.clear();
        }

        /**
         * @brief Reserves storage for at least the specified number of elements
         * @param new_cap New capacity of the vector
         */
        void reserve(size_type new_cap) {
            std::unique_lock lock(m_mutex);
            m_vec.reserve(new_cap);
        }

        /**
         * @brief Returns the number of elements that can be held in allocated storage
         */
        size_type capacity() const {
            std::shared_lock lock(m_mutex);
            return m_vec.capacity();
        }

        /**
         * @brief Reduces memory usage by freeing unused memory
         */
        void shrink_to_fit() {
            std::unique_lock lock(m_mutex);
            m_vec.shrink_to_fit();
        }

        /**
         * @brief Applies given function to each element (read-only)
         * @tparam Func Callable type accepting (const_reference) parameter
         * @param func The function to apply
         */
        template <typename Func> void for_each(Func func) const {
            std::shared_lock lock(m_mutex);
            for (const auto &element : m_vec) {
                func(element);
            }
        }

        /**
         * @brief Applies given function to each element (allows modification)
         * @tparam Func Callable type accepting (reference) parameter
         * @param func The function to apply
         */
        template <typename Func> void modify(Func func) {
            std::unique_lock lock(m_mutex);
            for (auto &element : m_vec) {
                func(element);
            }
        }

        /**
         * @brief Provides thread-safe access to the underlying container (read-only)
         * @tparam Func Callable type accepting const reference to the underlying vector
         * @return Result of the function invocation
         */
        template <typename Func> auto access(Func func) const {
            std::shared_lock lock(m_mutex);
            return func(m_vec);
        }

        /**
         * @brief Provides thread-safe access to the underlying container (read-write)
         * @tparam Func Callable type accepting reference to the underlying vector
         * @return Result of the function invocation
         */
        template <typename Func> auto modify_vector(Func func) {
            std::unique_lock lock(m_mutex);
            return func(m_vec);
        }

        /**
         * @brief Returns the allocator associated with the container
         */
        allocator_type get_allocator() const noexcept { return m_vec.get_allocator(); }

        /**
         * @brief Finds the first element satisfying specific criteria
         * @tparam Predicate Type of the predicate function
         * @param pred Predicate function which returns true for the required element
         * @return std::optional containing the const guarded reference to the first matching element if found
         */
        template <typename Predicate>
        std::optional<shared_guarded_ref<const T, std::shared_mutex>> find_if(Predicate pred) const {
            std::shared_lock lock(m_mutex);
            auto it = std::find_if(std::cbegin(m_vec), std::cend(m_vec), pred);
            if (it != std::cend(m_vec)) {
                return shared_guarded_ref<const T, std::shared_mutex>(*it, std::move(lock));
            }
            return std::nullopt;
        }

        /**
         * @brief Finds the first element satisfying specific criteria
         * @tparam Predicate Type of the predicate function
         * @param pred Predicate function which returns true for the required element
         * @return std::optional containing the guarded reference to the first matching element if found
         */
        template <typename Predicate>
        std::optional<shared_guarded_ref<T, std::shared_mutex>> find_if(Predicate pred) {
            std::shared_lock lock(m_mutex);
            auto it = std::find_if(std::cbegin(m_vec), std::cend(m_vec), pred);
            if (it != std::cend(m_vec)) {
                return shared_guarded_ref<T, std::shared_mutex>(*it, std::move(lock));
            }
            return std::nullopt;
        }

        /**
         * @brief Thread-safe iterator wrapper for concurrent_vector
         */
        class const_iterator {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = T;
            using difference_type = std::ptrdiff_t;
            using pointer = const T*;
            using reference = const T&;

            const_iterator(const const_iterator&) = delete;
            const_iterator& operator=(const const_iterator&) = delete;

            const_iterator(const_iterator&& other) noexcept
                : m_vec(other.m_vec)
                , m_lock(std::move(other.m_lock))
                , m_it(other.m_it) {}

            reference operator*() const { return *m_it; }
            pointer operator->() const { return &(*m_it); }

            const_iterator& operator++() {
                ++m_it;
                return *this;
            }

            const_iterator operator++(int) {
                const_iterator tmp = std::move(*this);
                ++m_it;
                return tmp;
            }

            bool operator==(const const_iterator& other) const {
                return m_it == other.m_it;
            }

            bool operator!=(const const_iterator& other) const {
                return !(*this == other);
            }

        private:
            friend class concurrent_vector;
            const concurrent_vector* m_vec;
            std::shared_lock<std::shared_mutex> m_lock;
            typename std::vector<T, Allocator>::const_iterator m_it;

            const_iterator(const concurrent_vector* vec, std::shared_lock<std::shared_mutex> lock,
                          typename std::vector<T, Allocator>::const_iterator it)
                : m_vec(vec)
                , m_lock(std::move(lock))
                , m_it(it) {}
        };

        /**
         * @brief Thread-safe iterator wrapper for concurrent_vector that allows modification
         */
        class iterator {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = T;
            using difference_type = std::ptrdiff_t;
            using pointer = T*;
            using reference = T&;

            iterator(const iterator&) = delete;
            iterator& operator=(const iterator&) = delete;

            iterator(iterator&& other) noexcept
                : m_vec(other.m_vec)
                , m_lock(std::move(other.m_lock))
                , m_it(other.m_it) {}

            reference operator*() { return *m_it; }
            pointer operator->() { return &(*m_it); }

            iterator& operator++() {
                ++m_it;
                return *this;
            }

            iterator operator++(int) {
                iterator tmp = std::move(*this);
                ++m_it;
                return tmp;
            }

            bool operator==(const iterator& other) const {
                return m_it == other.m_it;
            }

            bool operator!=(const iterator& other) const {
                return !(*this == other);
            }

            // Allow conversion to const_iterator
            operator const_iterator() const {
                return const_iterator(m_vec, std::shared_lock<std::shared_mutex>(m_vec->m_mutex), m_it);
            }

        private:
            friend class concurrent_vector;
            concurrent_vector* m_vec;
            std::shared_lock<std::shared_mutex> m_lock;
            typename std::vector<T, Allocator>::iterator m_it;

            iterator(concurrent_vector* vec, std::shared_lock<std::shared_mutex> lock,
                    typename std::vector<T, Allocator>::iterator it)
                : m_vec(vec)
                , m_lock(std::move(lock))
                , m_it(it) {}
        };

        /**
         * @brief Returns a thread-safe iterator to the beginning
         */
        const_iterator cbegin() const {
            std::shared_lock lock(m_mutex);
            return const_iterator(this, std::move(lock), m_vec.begin());
        }

        /**
         * @brief Returns a thread-safe iterator to the end
         */
        const_iterator cend() const {
            std::shared_lock lock(m_mutex);
            return const_iterator(this, std::move(lock), m_vec.end());
        }

        /**
         * @brief Returns a thread-safe iterator to the beginning that allows modification
         */
        iterator begin() {
            std::shared_lock lock(m_mutex);
            return iterator(this, std::move(lock), m_vec.begin());
        }

        /**
         * @brief Returns a thread-safe iterator to the end that allows modification
         */
        iterator end() {
            std::shared_lock lock(m_mutex);
            return iterator(this, std::move(lock), m_vec.end());
        }

      private:
        std::vector<T, Allocator> m_vec;
        mutable std::shared_mutex m_mutex;
    };
} // namespace wheel