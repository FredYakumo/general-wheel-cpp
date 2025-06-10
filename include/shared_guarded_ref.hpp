#pragma once
#include <mutex>

namespace wheel {
    /**
     * @brief A wrapper class that holds both a reference and an associated lock
     *
     * @tparam T The type of the wrapped reference
     * @tparam Mutex The type of the mutex
     */
    template <typename T, typename Mutex> class shared_guarded_ref {
      public:
        shared_guarded_ref(T &ref, std::unique_lock<Mutex> lock) : m_ref(ref), m_lock(std::move(lock)) {}

        // Provide access to the underlying reference
        T &get() noexcept { return m_ref; }
        const T &get() const noexcept { return m_ref; }

        // Allow implicit conversion to reference
        operator T &() noexcept { return m_ref; }
        operator const T &() const noexcept { return m_ref; }

        // Add implicit conversion to T
        operator T() const { return m_ref; }

        // Prevent copying
        shared_guarded_ref(const shared_guarded_ref &) = delete;
        shared_guarded_ref &operator=(const shared_guarded_ref &) = delete;

        // Allow moving
        shared_guarded_ref(shared_guarded_ref &&) = default;
        shared_guarded_ref &operator=(shared_guarded_ref &&) = default;

      private:
        T &m_ref;
        std::unique_lock<Mutex> m_lock;
    };
} // namespace wheel