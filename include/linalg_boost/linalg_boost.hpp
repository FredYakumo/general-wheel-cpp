#pragma once

#include "vec_ops.hpp"
#include <stdexcept>

namespace wheel::linalg_boost {

    /**
     * @brief Calculate dot product between two vectors
     *
     * @param a Pointer to the first vector
     * @param b Pointer to the second vector
     * @param size Size of the vectors
     * @return Dot product result
     * @throws std::invalid_argument If size is zero
     * @note Performance optimized using hardware instruction sets
     */
    inline float dot_product(const float *a, const float *b, size_t size) {
        if (size == 0)
            throw std::invalid_argument("dot_product: size must > 0");

#ifdef __aarch64__
#ifdef LINALG_USE_ASM
        return detail::dot_product_asm_aarch64(a, b, size);
#else
        return detail::dot_product_neon(a, b, size);
#endif
#else
        return detail::dot_product_scalar(a, b, size);
#endif
    }

    /**
     * @brief Calculate cosine similarity between two vectors
     *
     * @param a Pointer to the first vector
     * @param b Pointer to the second vector
     * @param size Size of the vectors
     * @return Cosine similarity value in range [-1, 1]
     * @throws std::invalid_argument If size is zero
     * @note Returns 0 when either vector is a zero vector
     * @note Performance optimized using hardware instruction sets
     */
    inline float cosine_similarity(const float *a, const float *b, size_t size) {
        if (size == 0)
            throw std::invalid_argument("cosine_similarity: size must > 0");
#ifdef __aarch64__
#ifdef LINALG_USE_ASM
        return detail::cosine_similarity_asm_arm64(a, b, size);
#else
        return detail::cosine_similarity_neon(a, b, size);
#endif
#else
        return detail::cosine_similarity_scalar(a, b, size);
#endif
    }
    
        /**
     * @brief Calculate cosine similarity for multiple vectors against a reference vector
     * 
     * Efficiently computes similarity values for multiple input vectors against a single reference vector.
     * The implementation uses batch processing to maximize performance.
     * 
     * @param a Array of vector pointers
     * @param b Pointer to the reference vector
     * @param n Size of each vector
     * @param batch_size Number of vectors in a
     * @param results Output array to store similarity results
     * @note Performance optimized using hardware instruction sets
     */
    inline void batch_cosine_similarity(const float **a, const float *b, size_t n, size_t batch_size, float *results) {
#ifdef LINALG_USE_ASM
#ifdef __aarch64__
        detail::batch_cosine_similarity_asm_arm64(a, b, n, batch_size, results);
        return;
#endif // __aarch64__
#endif // LINALG_USE_ASM

#ifdef __ARM_NEON
        detail::batch_cosine_similarity_neon(a, b, n, batch_size, results);
        return;
#endif // __ARM_NEON

        detail::batch_cosine_similarity_scalar(a, b, n, batch_size, results);
    }
} // namespace wheel::linalg_boost