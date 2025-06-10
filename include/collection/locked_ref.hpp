#pragma once
#include <mutex>

namespace wheel {
    /**
     * @brief A wrapper class that holds both a reference and an associated lock
     *
     * @tparam T The type of the wrapped reference
     * @tparam Mutex The type of the mutex
     */
    template <typename T, typename Mutex> class locked_reference {
      public:
        locked_reference(T &ref, std::unique_lock<Mutex> lock) : m_ref(ref), m_lock(std::move(lock)) {}

        // Provide access to the underlying reference
        T &get() noexcept { return m_ref; }
        const T &get() const noexcept { return m_ref; }

        // Allow implicit conversion to reference
        operator T &() noexcept { return m_ref; }
        operator const T &() const noexcept { return m_ref; }

        // Prevent copying
        locked_reference(const locked_reference &) = delete;
        locked_reference &operator=(const locked_reference &) = delete;

        // Allow moving
        locked_reference(locked_reference &&) = default;
        locked_reference &operator=(locked_reference &&) = default;

      private:
        T &m_ref;
        std::unique_lock<Mutex> m_lock;
    };
} // namespace wheel