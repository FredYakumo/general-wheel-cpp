#pragma once

#include "vec_ops.hpp"
#include <stdexcept>

namespace wheel::linalg_boost {

    /**
     * @brief Public API for calculating dot product between two vectors
     *
     * @param a Pointer to the first vector
     * @param b Pointer to the second vector
     * @param size Size of the vectors
     * @return Dot product result
     * @throws std::invalid_argument If size is zero
     * @note Automatically selects the best implementation based on available hardware
     */
    inline float dot_product(const float *a, const float *b, size_t size) {
        if (size == 0)
            throw std::invalid_argument("dot_product: size must > 0");

#if defined(__aarch64__)
#if defined(LINALG_USE_ASM)
        return detail::dot_product_asm_aarch64(a, b, size);
#else
        return detail::dot_product_neon(a, b, size);
#endif
#else
        return detail::dot_product_scalar(a, b, size);
#endif
    }

    /**
     * @brief Public API for calculating cosine similarity between two vectors
     *
     * @param a Pointer to the first vector
     * @param b Pointer to the second vector
     * @param size Size of the vectors
     * @return Cosine similarity value in range [-1, 1]
     * @throws std::invalid_argument If size is zero
     * @note Returns 0 when either vector is a zero vector
     * @note Automatically selects the best implementation based on available hardware
     */
    inline float cosine_similarity(const float *a, const float *b, size_t size) {
        if (size == 0)
            throw std::invalid_argument("cosine_similarity: size must > 0");
#if defined(__aarch64__)
        return detail::cosine_similarity_neon(a, b, size);
#else
        return detail::cosine_similarity_scalar(a, b, size);
#endif
    }
} // namespace wheel::linalg_boost