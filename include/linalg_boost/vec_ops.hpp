#pragma once

#include <cmath>
#include <cstddef>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace wheel::linalg_boost::detail {
#ifdef __aarch64__
    /**
     * @brief NEON-optimized implementation of dot product
     *
     * @param a Pointer to the first vector
     * @param b Pointer to the second vector
     * @param n Size of the vectors
     * @return Dot product result
     */
    inline float dot_product_neon(const float *a, const float *b, size_t n) {
        size_t i = 0;

        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);

        // Process 8 elements per iteration (8-way unroll)
        for (; i + 8 <= n; i += 8) {
            float32x4_t a0 = vld1q_f32(a + i);
            float32x4_t b0 = vld1q_f32(b + i);
            float32x4_t a1 = vld1q_f32(a + i + 4);
            float32x4_t b1 = vld1q_f32(b + i + 4);

            acc0 = vmlaq_f32(acc0, a0, b0);
            acc1 = vmlaq_f32(acc1, a1, b1);
        }

        // Process 4 elements if possible (4-way)
        if (i + 4 <= n) {
            float32x4_t a0 = vld1q_f32(a + i);
            float32x4_t b0 = vld1q_f32(b + i);
            acc0 = vmlaq_f32(acc0, a0, b0);
            i += 4;
        }

        // Perform horizontal sum of accumulator vectors
        float32x4_t acc = vaddq_f32(acc0, acc1);
        float sum = vaddvq_f32(acc); // Reduce 4 lanes to a single value

        // Process remaining elements
        for (; i < n; ++i)
            sum += a[i] * b[i];
        return sum;
    }

    /**
     * @brief NEON-optimized implementation of cosine similarity
     *
     * @param a Pointer to the first vector
     * @param b Pointer to the second vector
     * @param n Size of the vectors
     * @return Cosine similarity value in range [-1, 1]
     * @note Returns 0 when either vector is a zero vector
     */
    inline float cosine_similarity_neon(const float *a, const float *b, size_t n) {
        size_t i = 0;

        float32x4_t dot0 = vdupq_n_f32(0.0f), dot1 = vdupq_n_f32(0.0f);
        float32x4_t a0s = vdupq_n_f32(0.0f), a1s = vdupq_n_f32(0.0f);
        float32x4_t b0s = vdupq_n_f32(0.0f), b1s = vdupq_n_f32(0.0f);

        for (; i + 8 <= n; i += 8) {
            float32x4_t a0v = vld1q_f32(a + i);
            float32x4_t b0v = vld1q_f32(b + i);
            float32x4_t a1v = vld1q_f32(a + i + 4);
            float32x4_t b1v = vld1q_f32(b + i + 4);

            dot0 = vmlaq_f32(dot0, a0v, b0v);
            dot1 = vmlaq_f32(dot1, a1v, b1v);

            a0s = vmlaq_f32(a0s, a0v, a0v);
            a1s = vmlaq_f32(a1s, a1v, a1v);

            b0s = vmlaq_f32(b0s, b0v, b0v);
            b1s = vmlaq_f32(b1s, b1v, b1v);
        }

        if (i + 4 <= n) {
            float32x4_t av = vld1q_f32(a + i);
            float32x4_t bv = vld1q_f32(b + i);

            dot0 = vmlaq_f32(dot0, av, bv);
            a0s = vmlaq_f32(a0s, av, av);
            b0s = vmlaq_f32(b0s, bv, bv);

            i += 4;
        }

        float32x4_t dotv = vaddq_f32(dot0, dot1);
        float32x4_t asv = vaddq_f32(a0s, a1s);
        float32x4_t bsv = vaddq_f32(b0s, b1s);

        float dot = vaddvq_f32(dotv);
        float aa = vaddvq_f32(asv);
        float bb = vaddvq_f32(bsv);

        for (; i < n; ++i) {
            float ai = a[i], bi = b[i];
            dot += ai * bi;
            aa += ai * ai;
            bb += bi * bi;
        }

        float denom = std::sqrt(aa) * std::sqrt(bb);
        if (denom == 0.0f)
            return 0.0f;
        return dot / denom;
    }

    /**
     * @brief Optional AArch64 inline-assembly implementation of dot product
     *
     * @note Processes elements in 4-float chunks, with tail handling in C++
     * @note Enable via CMake option LINALG_USE_ASM (ON by default on AArch64)
     */
#ifdef LINALG_USE_ASM
    /**
     * @brief Assembly-optimized implementation of dot product for AArch64
     *
     * @param a Pointer to the first vector
     * @param b Pointer to the second vector
     * @param n Size of the vectors
     * @return Dot product result
     */
    inline float dot_product_asm_aarch64(const float *a, const float *b, size_t n) {
        size_t blocks = n / 4; // 4 floats per iteration
        float sum = 0.0f;

        if (blocks) {
            const float *pa = a;
            const float *pb = b;
            asm volatile("eor v0.16b, v0.16b, v0.16b           \n" // acc = 0
                         "1:                                     \n"
                         "ld1 {v1.4s}, [%[pa]], #16             \n"
                         "ld1 {v2.4s}, [%[pb]], #16             \n"
                         "fmla v0.4s, v1.4s, v2.4s              \n"
                         "subs %[blocks], %[blocks], #1         \n"
                         "b.ne 1b                                \n"
                         // horizontal reduce v0
                         "faddp v0.4s, v0.4s, v0.4s             \n" // [a0+a1, a2+a3, a0+a1, a2+a3]
                         "faddp v0.2s, v0.2s, v0.2s             \n" // [sum, sum, ..., ...]
                         "fmov %w[sum], s0                       \n"
                         : [sum] "=&r"(sum), [pa] "+r"(pa), [pb] "+r"(pb), [blocks] "+r"(blocks)
                         :
                         : "v0", "v1", "v2", "cc", "memory");
        }

        // Process remaining elements
        for (size_t i = (n & ~size_t(3)); i < n; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /**
     * @brief Assembly-optimized implementation of cosine similarity for AArch64
     *
     * @param a Pointer to the first vector
     * @param b Pointer to the second vector
     * @param n Size of the vectors
     * @return Cosine similarity value in range [-1, 1]
     * @note Returns 0 when either vector is a zero vector
     */
    inline float cosine_similarity_asm_aarch64(const float *a, const float *b, size_t n) {
        size_t blocks = n / 4; // 4 floats per iteration
        float dot = 0.0f;
        float aa = 0.0f; // Sum of squares for vector a
        float bb = 0.0f; // Sum of squares for vector b

        if (blocks) {
            const float *pa = a;
            const float *pb = b;
            asm volatile(
                "eor v0.16b, v0.16b, v0.16b           \n" // dot_product = 0
                "eor v3.16b, v3.16b, v3.16b           \n" // aa = 0
                "eor v4.16b, v4.16b, v4.16b           \n" // bb = 0
                "1:                                     \n"
                "ld1 {v1.4s}, [%[pa]], #16             \n" // Load 4 floats from a
                "ld1 {v2.4s}, [%[pb]], #16             \n" // Load 4 floats from b
                "fmla v0.4s, v1.4s, v2.4s              \n" // dot += a * b
                "fmla v3.4s, v1.4s, v1.4s              \n" // aa += a * a
                "fmla v4.4s, v2.4s, v2.4s              \n" // bb += b * b
                "subs %[blocks], %[blocks], #1         \n"
                "b.ne 1b                                \n"
                // horizontal reduce v0, v3, v4
                "faddp v0.4s, v0.4s, v0.4s             \n" // Reduce dot product
                "faddp v0.2s, v0.2s, v0.2s             \n"
                "fmov %w[dot], s0                       \n"

                "faddp v3.4s, v3.4s, v3.4s             \n" // Reduce aa
                "faddp v3.2s, v3.2s, v3.2s             \n"
                "fmov %w[aa], s3                        \n"

                "faddp v4.4s, v4.4s, v4.4s             \n" // Reduce bb
                "faddp v4.2s, v4.2s, v4.2s             \n"
                "fmov %w[bb], s4                        \n"
                : [dot] "=&r"(dot), [aa] "=&r"(aa), [bb] "=&r"(bb), [pa] "+r"(pa), [pb] "+r"(pb), [blocks] "+r"(blocks)
                :
                : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
        }

        // Process remaining elements
        for (size_t i = (n & ~size_t(3)); i < n; ++i) {
            const float ai = a[i], bi = b[i];
            dot += ai * bi;
            aa += ai * ai;
            bb += bi * bi;
        }

        const float denom = std::sqrt(aa) * std::sqrt(bb);
        if (denom == 0.0f)
            return 0.0f; // Return 0 when either vector is a zero vector
        return dot / denom;
    }
#endif // LINALG_USE_ASM
#endif // __aarch64__

    /**
     * @brief Portable scalar implementation of dot product
     *
     * @param a Pointer to the first vector
     * @param b Pointer to the second vector
     * @param n Size of the vectors
     * @return Dot product result
     */
    inline float dot_product_scalar(const float *a, const float *b, size_t n) {
        float acc = 0.0f;
        for (size_t i = 0; i < n; ++i)
            acc += a[i] * b[i];
        return acc;
    }

    /**
     * @brief Portable scalar implementation of cosine similarity
     *
     * @param a Pointer to the first vector
     * @param b Pointer to the second vector
     * @param n Size of the vectors
     * @return Cosine similarity value in range [-1, 1]
     * @note Returns 0 when either vector is a zero vector
     */
    inline float cosine_similarity_scalar(const float *a, const float *b, size_t n) {
        float dot = 0.0f, aa = 0.0f, bb = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            const float ai = a[i], bi = b[i];
            dot += ai * bi;
            aa += ai * ai;
            bb += bi * bi;
        }
        const float denom = std::sqrt(aa) * std::sqrt(bb);
        if (denom == 0.0f)
            return 0.0f; // Return 0 when either vector is a zero vector
        return dot / denom;
    }

} // namespace wheel::linalg_boost::detail
