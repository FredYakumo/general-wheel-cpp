#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>

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
#ifdef LINALG_USE_ASM
        // Use the assembly implementation when LINALG_USE_ASM is defined
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
#else
        // Use the NEON intrinsics implementation
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
#endif // __LINALG_USE_ASM__
    }

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

#ifdef __aarch64__
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
#ifdef LINALG_USE_ASM
        // Use the assembly implementation when LINALG_USE_ASM is defined
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
#else
        // Use the NEON intrinsics implementation
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
#endif // __LINALG_USE_ASM__
    }

#endif // __aarch64__

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

#ifdef __aarch64__
    /**
     * @brief Optimized NEON implementation of batch cosine similarity
     *
     * @param a Array of vector pointers
     * @param b Pointer to the reference vector
     * @param n Size of each vector
     * @param batch_size Number of vectors in a
     * @param results Output array to store similarity results
     * @note Returns 0 when either vector is a zero vector
     */
    inline void batch_cosine_similarity_neon(const float **a, const float *b, size_t n, size_t batch_size,
                                             float *results) {
#ifdef LINALG_USE_ASM
        // Use the assembly implementation when LINALG_USE_ASM is defined
        // Precompute b's squared sum
        size_t blocks = n / 4; // 4 floats per iteration
        float bb = 0.0f;       // Sum of squares for vector b

        if (blocks) {
            const float *pb = b;
            size_t blocks_copy = blocks;

            asm volatile("eor v4.16b, v4.16b, v4.16b           \n" // bb = 0

                         "1:                                     \n"
                         "ld1 {v2.4s}, [%[pb]], #16             \n" // Load 4 floats from b
                         "fmla v4.4s, v2.4s, v2.4s              \n" // bb += b * b
                         "subs %[blocks], %[blocks], #1         \n"
                         "b.ne 1b                                \n"

                         // horizontal reduce v4
                         "faddp v4.4s, v4.4s, v4.4s             \n" // Reduce bb
                         "faddp v4.2s, v4.2s, v4.2s             \n"
                         "fmov %w[bb], s4                        \n"
                         : [bb] "=&r"(bb), [pb] "+r"(pb), [blocks] "+r"(blocks_copy)
                         :
                         : "v2", "v4", "cc", "memory");
        }

        // Process remaining elements for b
        for (size_t i = (n & ~size_t(3)); i < n; ++i) {
            const float bi = b[i];
            bb += bi * bi;
        }

        if (bb == 0.0f) {
            // b = zero vector, all similarities are 0
            for (size_t k = 0; k < batch_size; ++k) {
                results[k] = 0.0f;
            }
            return;
        }

        const float sqrt_bb = std::sqrt(bb);

        // Process vectors in pairs if possible
        const size_t batch_step = 2;
        const size_t aligned_batch_size = (batch_size / batch_step) * batch_step;

        // Process pairs of vectors simultaneously using dual registers
        for (size_t k = 0; k < aligned_batch_size; k += batch_step) {
            const float *avec0 = a[k];
            const float *avec1 = a[k + 1];
            float dot0 = 0.0f, dot1 = 0.0f;
            float aa0 = 0.0f, aa1 = 0.0f;

            if (blocks) {
                const float *pa0 = avec0;
                const float *pa1 = avec1;
                const float *pb = b;
                size_t blocks_copy = blocks;

                asm volatile(
                    // Initialize accumulators
                    "eor v0.16b, v0.16b, v0.16b           \n" // dot0 = 0
                    "eor v1.16b, v1.16b, v1.16b           \n" // dot1 = 0

                    "eor v3.16b, v3.16b, v3.16b           \n" // aa0 = 0
                    "eor v5.16b, v5.16b, v5.16b           \n" // aa1 = 0

                    "1:                                     \n"
                    // Load vector elements
                    "ld1 {v6.4s}, [%[pa0]], #16            \n" // Load 4 floats from a0
                    "ld1 {v7.4s}, [%[pa1]], #16            \n" // Load 4 floats from a1
                    "ld1 {v2.4s}, [%[pb]], #16             \n" // Load 4 floats from b

                    // Calculate
                    "fmla v0.4s, v6.4s, v2.4s              \n" // dot0 += a0 * b
                    "fmla v1.4s, v7.4s, v2.4s              \n" // dot1 += a1 * b
                    "fmla v3.4s, v6.4s, v6.4s              \n" // aa0 += a0 * a0
                    "fmla v5.4s, v7.4s, v7.4s              \n" // aa1 += a1 * a1
                    "subs %[blocks], %[blocks], #1         \n"
                    "b.ne 1b                                \n"

                    // Horizontal reduction
                    "faddp v0.4s, v0.4s, v0.4s             \n" // Reduce dot0
                    "faddp v0.2s, v0.2s, v0.2s             \n"
                    "fmov %w[dot0], s0                      \n"
                    "faddp v1.4s, v1.4s, v1.4s             \n" // Reduce dot1
                    "faddp v1.2s, v1.2s, v1.2s             \n"
                    "fmov %w[dot1], s1                      \n"
                    "faddp v3.4s, v3.4s, v3.4s             \n" // Reduce aa0
                    "faddp v3.2s, v3.2s, v3.2s             \n"
                    "fmov %w[aa0], s3                      \n"
                    "faddp v5.4s, v5.4s, v5.4s             \n" // Reduce aa1
                    "faddp v5.2s, v5.2s, v5.2s             \n"
                    "fmov %w[aa1], s5                      \n"

                    : [dot0] "=&r"(dot0), [dot1] "=&r"(dot1), [aa0] "=&r"(aa0), [aa1] "=&r"(aa1), [pa0] "+r"(pa0),
                      [pa1] "+r"(pa1), [pb] "+r"(pb), [blocks] "+r"(blocks_copy)
                    :
                    : "v0", "v1", "v2", "v3", "v5", "v6", "v7", "cc", "memory");
            }

            // Process remaining elements
            for (size_t i = (n & ~size_t(3)); i < n; ++i) {
                const float bi = b[i];

                const float a0i = avec0[i];
                const float a1i = avec1[i];

                dot0 += a0i * bi;
                dot1 += a1i * bi;

                aa0 += a0i * a0i;
                aa1 += a1i * a1i;
            }

            // Calculate final results
            if (aa0 == 0.0f) {
                results[k] = 0.0f; // a0 is a zero vector
            } else {
                results[k] = dot0 / (std::sqrt(aa0) * sqrt_bb);
            }

            if (aa1 == 0.0f) {
                results[k + 1] = 0.0f; // a1 is a zero vector
            } else {
                results[k + 1] = dot1 / (std::sqrt(aa1) * sqrt_bb);
            }
        }

        // Process remaining vectors individually
        for (size_t k = aligned_batch_size; k < batch_size; ++k) {
            const float *avec = a[k];
            float dot = 0.0f;
            float aa = 0.0f;

            if (blocks) {
                const float *pa = avec;
                const float *pb = b;
                size_t blocks_copy = blocks;

                asm volatile(
                    "eor v0.16b, v0.16b, v0.16b           \n" // dot_product = 0
                    "eor v3.16b, v3.16b, v3.16b           \n" // aa = 0

                    "1:                                     \n"
                    "ld1 {v1.4s}, [%[pa]], #16             \n" // Load 4 floats from a
                    "ld1 {v2.4s}, [%[pb]], #16             \n" // Load 4 floats from b
                    "fmla v0.4s, v1.4s, v2.4s              \n" // dot += a * b
                    "fmla v3.4s, v1.4s, v1.4s              \n" // aa += a * a
                    "subs %[blocks], %[blocks], #1         \n"
                    "b.ne 1b                                \n"

                    // horizontal reduce v0, v3
                    "faddp v0.4s, v0.4s, v0.4s             \n" // Reduce dot product
                    "faddp v0.2s, v0.2s, v0.2s             \n"
                    "fmov %w[dot], s0                       \n"
                    "faddp v3.4s, v3.4s, v3.4s             \n" // Reduce aa
                    "faddp v3.2s, v3.2s, v3.2s             \n"
                    "fmov %w[aa], s3                        \n"
                    : [dot] "=&r"(dot), [aa] "=&r"(aa), [pa] "+r"(pa), [pb] "+r"(pb), [blocks] "+r"(blocks_copy)
                    :
                    : "v0", "v1", "v2", "v3", "cc", "memory");
            }

            // Process remaining elements
            for (size_t i = (n & ~size_t(3)); i < n; ++i) {
                const float ai = avec[i], bi = b[i];
                dot += ai * bi;
                aa += ai * ai;
            }

            if (aa == 0.0f) {
                results[k] = 0.0f; // a is a zero vector
            } else {
                results[k] = dot / (std::sqrt(aa) * sqrt_bb);
            }
        }
#else
        // Use the NEON intrinsics implementation
        // Precompute b's squared sum using NEON
        float32x4_t b0s = vdupq_n_f32(0.0f), b1s = vdupq_n_f32(0.0f);
        float bb = 0.0f;

        // Load and cache the reference vector elements in advance
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            float32x4_t b0v = vld1q_f32(b + i);
            float32x4_t b1v = vld1q_f32(b + i + 4);

            b0s = vmlaq_f32(b0s, b0v, b0v);
            b1s = vmlaq_f32(b1s, b1v, b1v);
        }

        if (i + 4 <= n) {
            float32x4_t bv = vld1q_f32(b + i);
            b0s = vmlaq_f32(b0s, bv, bv);
            i += 4;
        }

        float32x4_t bsv = vaddq_f32(b0s, b1s);
        bb = vaddvq_f32(bsv);

        for (; i < n; ++i) {
            float bi = b[i];
            bb += bi * bi;
        }

        if (bb == 0.0f) {
            // b = zero vector, all similarities are 0
            for (size_t k = 0; k < batch_size; ++k) {
                results[k] = 0.0f;
            }
            return;
        }

        const float sqrt_bb = std::sqrt(bb);

        // Process vectors in batches of 4 if possible
        const size_t batch_step = 4;
        const size_t aligned_batch_size = (batch_size / batch_step) * batch_step;

        // Process main batch in groups of 4
        for (size_t k = 0; k < aligned_batch_size; k += batch_step) {
            const float *avec0 = a[k];
            const float *avec1 = a[k + 1];
            const float *avec2 = a[k + 2];
            const float *avec3 = a[k + 3];

            float dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float aa[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            // Process vector elements in chunks
            size_t j = 0;
            for (; j + 4 <= n; j += 4) {
                // Load reference vector once
                float32x4_t bv = vld1q_f32(b + j);

                // Process 4 input vectors in parallel
                float32x4_t av0 = vld1q_f32(avec0 + j);
                float32x4_t av1 = vld1q_f32(avec1 + j);
                float32x4_t av2 = vld1q_f32(avec2 + j);
                float32x4_t av3 = vld1q_f32(avec3 + j);

                // Calculate dot products
                float32x4_t dot0 = vmulq_f32(av0, bv);
                float32x4_t dot1 = vmulq_f32(av1, bv);
                float32x4_t dot2 = vmulq_f32(av2, bv);
                float32x4_t dot3 = vmulq_f32(av3, bv);

                // Calculate squared sums
                float32x4_t as0 = vmulq_f32(av0, av0);
                float32x4_t as1 = vmulq_f32(av1, av1);
                float32x4_t as2 = vmulq_f32(av2, av2);
                float32x4_t as3 = vmulq_f32(av3, av3);

                // Horizontal sum accumulation
                dot[0] += vaddvq_f32(dot0);
                dot[1] += vaddvq_f32(dot1);
                dot[2] += vaddvq_f32(dot2);
                dot[3] += vaddvq_f32(dot3);

                aa[0] += vaddvq_f32(as0);
                aa[1] += vaddvq_f32(as1);
                aa[2] += vaddvq_f32(as2);
                aa[3] += vaddvq_f32(as3);
            }

            // Process remaining elements
            for (; j < n; ++j) {
                const float bj = b[j];

                const float a0j = avec0[j];
                const float a1j = avec1[j];
                const float a2j = avec2[j];
                const float a3j = avec3[j];

                dot[0] += a0j * bj;
                dot[1] += a1j * bj;
                dot[2] += a2j * bj;
                dot[3] += a3j * bj;

                aa[0] += a0j * a0j;
                aa[1] += a1j * a1j;
                aa[2] += a2j * a2j;
                aa[3] += a3j * a3j;
            }

            // Calculate final results
            for (size_t idx = 0; idx < batch_step; ++idx) {
                if (aa[idx] == 0.0f) {
                    results[k + idx] = 0.0f; // a is a zero vector
                } else {
                    results[k + idx] = dot[idx] / (std::sqrt(aa[idx]) * sqrt_bb);
                }
            }
        }

        // Process remaining vectors individually
        for (size_t k = aligned_batch_size; k < batch_size; ++k) {
            const float *avec = a[k];
            float32x4_t dot0 = vdupq_n_f32(0.0f), dot1 = vdupq_n_f32(0.0f);
            float32x4_t a0s = vdupq_n_f32(0.0f), a1s = vdupq_n_f32(0.0f);

            i = 0;
            for (; i + 8 <= n; i += 8) {
                float32x4_t a0v = vld1q_f32(avec + i);
                float32x4_t b0v = vld1q_f32(b + i);
                float32x4_t a1v = vld1q_f32(avec + i + 4);
                float32x4_t b1v = vld1q_f32(b + i + 4);

                dot0 = vmlaq_f32(dot0, a0v, b0v);
                dot1 = vmlaq_f32(dot1, a1v, b1v);

                a0s = vmlaq_f32(a0s, a0v, a0v);
                a1s = vmlaq_f32(a1s, a1v, a1v);
            }

            if (i + 4 <= n) {
                float32x4_t av = vld1q_f32(avec + i);
                float32x4_t bv = vld1q_f32(b + i);

                dot0 = vmlaq_f32(dot0, av, bv);
                a0s = vmlaq_f32(a0s, av, av);

                i += 4;
            }

            float32x4_t dotv = vaddq_f32(dot0, dot1);
            float32x4_t asv = vaddq_f32(a0s, a1s);

            float dot = vaddvq_f32(dotv);
            float aa = vaddvq_f32(asv);

            for (; i < n; ++i) {
                float ai = avec[i], bi = b[i];
                dot += ai * bi;
                aa += ai * ai;
            }

            if (aa == 0.0f) {
                results[k] = 0.0f; // a is a zero vector
            } else {
                results[k] = dot / (std::sqrt(aa) * sqrt_bb);
            }
        }
#endif // __LINALG_USE_ASM__
    }

#endif // __aarch64__

    /**
     * @brief Optimized portable scalar implementation of batch cosine similarity
     *
     * @param a Array of vector pointers
     * @param b Pointer to the reference vector
     * @param n Size of each vector
     * @param batch_size Number of vectors in a
     * @param results Output array to store similarity results
     * @note Returns 0 when either vector is a zero vector
     */
    inline void batch_cosine_similarity_scalar(const float **a, const float *b, size_t n, size_t batch_size,
                                               float *results) {
        // Precompute b's squared sum once for all comparisons
        float bb = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            const float bi = b[i];
            bb += bi * bi;
        }

        if (bb == 0.0f) {
            // b = zero vector, all similarities are 0
            for (size_t k = 0; k < batch_size; ++k) {
                results[k] = 0.0f;
            }
            return;
        }

        const float sqrt_bb = std::sqrt(bb);

        // Calculate similarity for each vector in the batch
        for (size_t k = 0; k < batch_size; ++k) {
            const float *avec = a[k];
            float dot = 0.0f, aa = 0.0f;

            for (size_t i = 0; i < n; ++i) {
                const float ai = avec[i], bi = b[i];
                dot += ai * bi;
                aa += ai * ai;
            }

            if (aa == 0.0f) {
                results[k] = 0.0f; // a is a zero vector
            } else {
                results[k] = dot / (std::sqrt(aa) * sqrt_bb);
            }
        }
    }

#ifdef __aarch64__
    /**
     * @brief Optimized NEON implementation of mean pooling
     *
     * @param vectors Array of vector pointers
     * @param n Size of each vector
     * @param num_vectors Number of vectors to average
     * @param result Output vector to store the result
     */
    inline void mean_pooling_neon(const float **vectors, size_t n, size_t num_vectors, float *result) {
#ifdef LINALG_USE_ASM
        const size_t blocks = n / 4;
        const float scale = 1.0f / static_cast<float>(num_vectors);

        for (size_t i = 0; i < blocks; ++i) {
            float *res_ptr = result + (i * 4);
            asm volatile("eor v0.16b, v0.16b, v0.16b           \n" // sum = 0
                         "st1 {v0.4s}, [%[res_ptr]]            \n" // Store zeros
                         :
                         : [res_ptr] "r"(res_ptr)
                         : "v0", "memory");
        }

        // Zero remaining
        for (size_t i = blocks * 4; i < n; ++i) {
            result[i] = 0.0f;
        }

        // Sum all vectors
        for (size_t k = 0; k < num_vectors; ++k) {
            const float *vec_ptr = vectors[k];

            // Process 4-element blocks
            for (size_t i = 0; i < blocks; ++i) {
                size_t offset = i * 4;
                float *res_ptr = result + offset;

                asm volatile("ld1 {v0.4s}, [%[res_ptr]]            \n" // Load current sum
                             "ld1 {v1.4s}, [%[vec_ptr]]            \n" // Load vector chunk
                             "fadd v0.4s, v0.4s, v1.4s             \n" // sum += vec
                             "st1 {v0.4s}, [%[res_ptr]]            \n" // Store updated sum
                             :
                             : [res_ptr] "r"(res_ptr), [vec_ptr] "r"(vec_ptr + offset)
                             : "v0", "v1", "memory");
            }

            // Process remaining
            for (size_t i = blocks * 4; i < n; ++i) {
                result[i] += vec_ptr[i];
            }
        }

        // multiply by scale factor
        for (size_t i = 0; i < blocks; ++i) {
            float *res_ptr = result + (i * 4);

            asm volatile("fmov s2, %w[scale]                   \n" // Load scale factor
                         "dup v2.4s, v2.s[0]                   \n" // Duplicate to all lanes
                         "ld1 {v0.4s}, [%[res_ptr]]            \n" // Load sum
                         "fmul v0.4s, v0.4s, v2.4s             \n" // result = sum * scale
                         "st1 {v0.4s}, [%[res_ptr]]            \n" // Store result
                         :
                         : [scale] "r"(scale), [res_ptr] "r"(res_ptr)
                         : "v0", "v2", "memory");
        }

        // Scale remaining elements
        for (size_t i = blocks * 4; i < n; ++i) {
            result[i] *= scale;
        }
#else
        // Use the NEON intrinsics implementation
        for (size_t i = 0; i + 4 <= n; i += 4) {
            vst1q_f32(result + i, vdupq_n_f32(0.0f));
        }
        for (size_t i = (n & ~size_t(3)); i < n; ++i) {
            result[i] = 0.0f;
        }

        // Sum all vectors
        for (size_t k = 0; k < num_vectors; ++k) {
            const float *vec = vectors[k];
            for (size_t i = 0; i + 4 <= n; i += 4) {
                float32x4_t current_sum = vld1q_f32(result + i);
                float32x4_t vec_chunk = vld1q_f32(vec + i);
                vst1q_f32(result + i, vaddq_f32(current_sum, vec_chunk));
            }

            // remaining
            for (size_t i = (n & ~size_t(3)); i < n; ++i) {
                result[i] += vec[i];
            }
        }

        // scale by 1/num_vectors
        float32x4_t scale_vec = vdupq_n_f32(1.0f / static_cast<float>(num_vectors));
        for (size_t i = 0; i + 4 <= n; i += 4) {
            float32x4_t sum_vec = vld1q_f32(result + i);
            float32x4_t result_vec = vmulq_f32(sum_vec, scale_vec);
            vst1q_f32(result + i, result_vec);
        }

        // scale remaining
        for (size_t i = (n & ~size_t(3)); i < n; ++i) {
            result[i] /= static_cast<float>(num_vectors);
        }
#endif // __LINALG_USE_ASM__
    }
#endif // __aarch64__

    /**
     * @brief Portable scalar implementation of mean pooling
     *
     * @param vectors Array of vector pointers
     * @param n Size of each vector
     * @param num_vectors Number of vectors to average
     * @param result Output vector to store the result
     */
    inline void mean_pooling_scalar(const float **vectors, size_t n, size_t num_vectors, float *result) {
        std::fill(result, result + n, 0.0f);

        // Sum all vectors
        for (size_t k = 0; k < num_vectors; ++k) {
            const float *vec = vectors[k];
            for (size_t i = 0; i < n; ++i) {
                result[i] += vec[i];
            }
        }

        // Divide by number of vectors
        const float scale = 1.0f / static_cast<float>(num_vectors);
        for (size_t i = 0; i < n; ++i) {
            result[i] *= scale;
        }
    }

#ifdef __aarch64__
    /**
     * @brief NEON-optimized implementation of batch mean pooling
     *
     * @param matrix Array of batch pointers, where each batch contains channel_dim vectors of feature_dim elements
     * @param feature_dim Size of each feature vector (inner dimension)
     * @param channel_dim Number of vectors to average per batch item (middle dimension)
     * @param batch_size Number of batches to process (outer dimension)
     * @param results Output array to store batch results
     */
    inline void batch_channel_mean_pooling_neon(const float ***matrix, size_t feature_dim, size_t channel_dim,
                                        size_t batch_size, float **results) {
#ifdef LINALG_USE_ASM
        // Use the assembly implementation when LINALG_USE_ASM is defined
        const size_t blocks = feature_dim / 4;
        const float scale = 1.0f / static_cast<float>(channel_dim);

        // Process each batch
        for (size_t b = 0; b < batch_size; ++b) {
            float *result = results[b];
            const float **batch_vectors = matrix[b];

            // Initialize
            for (size_t i = 0; i < blocks; ++i) {
                float *res_ptr = result + (i * 4);
                asm volatile("eor v0.16b, v0.16b, v0.16b           \n" // sum = 0
                             "st1 {v0.4s}, [%[res_ptr]]            \n" // Store zeros
                             :
                             : [res_ptr] "r"(res_ptr)
                             : "v0", "memory");
            }

            // remaining
            for (size_t i = blocks * 4; i < feature_dim; ++i) {
                result[i] = 0.0f;
            }

            // Sum all
            for (size_t c = 0; c < channel_dim; ++c) {
                const float *vec_ptr = batch_vectors[c];

                // Process in blocks of 4 floats
                for (size_t i = 0; i < blocks; ++i) {
                    size_t offset = i * 4;
                    float *res_ptr = result + offset;

                    asm volatile("ld1 {v0.4s}, [%[res_ptr]]            \n" // Load current sum
                                 "ld1 {v1.4s}, [%[vec_ptr]]            \n" // Load vector chunk
                                 "fadd v0.4s, v0.4s, v1.4s             \n" // sum += vec
                                 "st1 {v0.4s}, [%[res_ptr]]            \n" // Store updated sum
                                 :
                                 : [res_ptr] "r"(res_ptr), [vec_ptr] "r"(vec_ptr + offset)
                                 : "v0", "v1", "memory");
                }

                // remaining
                for (size_t i = blocks * 4; i < feature_dim; ++i) {
                    result[i] += vec_ptr[i];
                }
            }

            // scale the results
            for (size_t i = 0; i < blocks; ++i) {
                float *res_ptr = result + (i * 4);

                asm volatile("fmov s2, %w[scale]                   \n" // Load scale factor
                             "dup v2.4s, v2.s[0]                   \n" // Duplicate to all lanes
                             "ld1 {v0.4s}, [%[res_ptr]]            \n" // Load sum
                             "fmul v0.4s, v0.4s, v2.4s             \n" // result = sum * scale
                             "st1 {v0.4s}, [%[res_ptr]]            \n" // Store result
                             :
                             : [scale] "r"(scale), [res_ptr] "r"(res_ptr)
                             : "v0", "v2", "memory");
            }

            // Scale remaining
            for (size_t i = blocks * 4; i < feature_dim; ++i) {
                result[i] *= scale;
            }
        }
#else
        // Use the NEON intrinsics implementation
        const float scale = 1.0f / static_cast<float>(channel_dim);
        float32x4_t scale_vec = vdupq_n_f32(scale);

        // Process each
        for (size_t b = 0; b < batch_size; ++b) {
            float *result = results[b];
            const float **batch_vectors = matrix[b];

            for (size_t i = 0; i + 4 <= feature_dim; i += 4) {
                vst1q_f32(result + i, vdupq_n_f32(0.0f));
            }
            for (size_t i = (feature_dim & ~size_t(3)); i < feature_dim; ++i) {
                result[i] = 0.0f;
            }

            // sum all vectors
            for (size_t c = 0; c < channel_dim; ++c) {
                const float *vec = batch_vectors[c];

                // Process in chunks of 4 floats
                for (size_t i = 0; i + 4 <= feature_dim; i += 4) {
                    float32x4_t current_sum = vld1q_f32(result + i);
                    float32x4_t vec_chunk = vld1q_f32(vec + i);
                    vst1q_f32(result + i, vaddq_f32(current_sum, vec_chunk));
                }

                // Process remaining elements
                for (size_t i = (feature_dim & ~size_t(3)); i < feature_dim; ++i) {
                    result[i] += vec[i];
                }
            }

            // scale the results
            for (size_t i = 0; i + 4 <= feature_dim; i += 4) {
                float32x4_t sum_vec = vld1q_f32(result + i);
                float32x4_t result_vec = vmulq_f32(sum_vec, scale_vec);
                vst1q_f32(result + i, result_vec);
            }

            // scale remaining
            for (size_t i = (feature_dim & ~size_t(3)); i < feature_dim; ++i) {
                result[i] *= scale;
            }
        }
#endif // __LINALG_USE_ASM__
    }
#endif // __aarch64__

    /**
     * @brief Portable scalar implementation of batch mean pooling
     *
     * @param matrix Array of batch pointers, where each batch contains channel_dim vectors of feature_dim elements
     * @param feature_dim Size of each feature vector (inner dimension)
     * @param channel_dim Number of vectors to average per batch item (middle dimension)
     * @param batch_size Number of batches to process (outer dimension)
     * @param results Output array to store batch results
     */
    inline void batch_channel_mean_pooling_scalar(const float ***matrix, size_t feature_dim, size_t channel_dim,
                                          size_t batch_size, float **results) {
        const float scale = 1.0f / static_cast<float>(channel_dim);

        // Process each batch independently
        for (size_t b = 0; b < batch_size; ++b) {
            // Zero initialize result for this batch
            std::fill(results[b], results[b] + feature_dim, 0.0f);

            // Sum all vectors in this batch
            for (size_t c = 0; c < channel_dim; ++c) {
                const float *vec = matrix[b][c];
                for (size_t i = 0; i < feature_dim; ++i) {
                    results[b][i] += vec[i];
                }
            }

            // Scale the results
            for (size_t i = 0; i < feature_dim; ++i) {
                results[b][i] *= scale;
            }
        }
    }
    
#ifdef __aarch64__
    /**
     * @brief NEON-optimized implementation of batch feature mean pooling
     *
     * @param matrix Array of batch pointers, where each batch contains feature_dim elements
     * @param feature_dim Size of each feature vector
     * @param batch_size Number of batches to process
     * @param results Output array to store batch results (must be pre-allocated with batch_size elements)
     */
    inline void batch_feature_mean_pooling_neon(const float **matrix, size_t feature_dim, size_t batch_size,
                                             float *results) {
#ifdef LINALG_USE_ASM
        const float inv_feature_dim = 1.0f / static_cast<float>(feature_dim);
        const size_t blocks = feature_dim / 4;
        
        // Process batches in pairs if possible
        const size_t batch_step = 2;
        const size_t aligned_batch_size = (batch_size / batch_step) * batch_step;
        
        // Process pairs of batches simultaneously
        for (size_t b = 0; b < aligned_batch_size; b += batch_step) {
            const float *vec_ptr0 = matrix[b];
            const float *vec_ptr1 = matrix[b + 1];
            float sum0 = 0.0f, sum1 = 0.0f;
            
            if (blocks) {
                size_t blocks_copy = blocks;
                asm volatile(
                    "eor v0.16b, v0.16b, v0.16b           \n" // sum0 = 0
                    "eor v2.16b, v2.16b, v2.16b           \n" // sum1 = 0

                    "1:                                     \n"
                    "ld1 {v1.4s}, [%[vec_ptr0]], #16       \n" // Load 4 floats from vector 0
                    "ld1 {v3.4s}, [%[vec_ptr1]], #16       \n" // Load 4 floats from vector 1
                    "fadd v0.4s, v0.4s, v1.4s              \n" // sum0 += vector0 elements
                    "fadd v2.4s, v2.4s, v3.4s              \n" // sum1 += vector1 elements
                    "subs %[blocks], %[blocks], #1         \n"
                    "b.ne 1b                                \n"

                    // Horizontal reduction
                    "faddp v0.4s, v0.4s, v0.4s             \n" // Reduce sum0
                    "faddp v0.2s, v0.2s, v0.2s             \n"
                    "fmov %w[sum0], s0                      \n"
                    
                    "faddp v2.4s, v2.4s, v2.4s             \n" // Reduce sum1
                    "faddp v2.2s, v2.2s, v2.2s             \n"
                    "fmov %w[sum1], s2                      \n"
                    : [sum0] "=&r"(sum0), [sum1] "=&r"(sum1), 
                      [vec_ptr0] "+r"(vec_ptr0), [vec_ptr1] "+r"(vec_ptr1), 
                      [blocks] "+r"(blocks_copy)
                    :
                    : "v0", "v1", "v2", "v3", "cc", "memory");
            }
            
            // Process remaining elements
            const size_t remaining_start = (feature_dim & ~size_t(3));
            for (size_t i = remaining_start; i < feature_dim; ++i) {
                sum0 += matrix[b][i];
                sum1 += matrix[b + 1][i];
            }
            
            // Scale and store results
            results[b] = sum0 * inv_feature_dim;
            results[b + 1] = sum1 * inv_feature_dim;
        }
        
        // Process any remaining batches
        for (size_t b = aligned_batch_size; b < batch_size; ++b) {
            const float *vec_ptr = matrix[b];
            float sum = 0.0f;
            
            if (blocks) {
                size_t blocks_copy = blocks;
                asm volatile(
                    "eor v0.16b, v0.16b, v0.16b           \n" // sum = 0

                    "1:                                     \n"
                    "ld1 {v1.4s}, [%[vec_ptr]], #16        \n" // Load 4 floats from vector
                    "fadd v0.4s, v0.4s, v1.4s              \n" // sum += vector elements
                    "subs %[blocks], %[blocks], #1         \n"
                    "b.ne 1b                                \n"

                    // Horizontal reduction
                    "faddp v0.4s, v0.4s, v0.4s             \n" // Reduce sum
                    "faddp v0.2s, v0.2s, v0.2s             \n"
                    "fmov %w[sum], s0                       \n"
                    : [sum] "=&r"(sum), [vec_ptr] "+r"(vec_ptr), [blocks] "+r"(blocks_copy)
                    :
                    : "v0", "v1", "cc", "memory");
            }
            
            // Process remaining elements
            const size_t remaining_start = (feature_dim & ~size_t(3));
            for (size_t i = remaining_start; i < feature_dim; ++i) {
                sum += matrix[b][i];
            }
            
            results[b] = sum * inv_feature_dim;
        }
#else
        // Use the NEON intrinsics implementation
        const float inv_feature_dim = 1.0f / static_cast<float>(feature_dim);
        const float32x4_t inv_dim_vec = vdupq_n_f32(inv_feature_dim);
        
        // Process batches in groups of 4 if possible
        const size_t batch_step = 4;
        const size_t aligned_batch_size = (batch_size / batch_step) * batch_step;
        
        // Process 4 batches at once for better throughput
        for (size_t b = 0; b < aligned_batch_size; b += batch_step) {
            float32x4_t sums[4] = {
                vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f)
            };
            
            // Sum across feature dimension in chunks of 4
            size_t i = 0;
            for (; i + 4 <= feature_dim; i += 4) {
                // Load 4 elements from each of the 4 batches
                float32x4_t v0 = vld1q_f32(matrix[b] + i);
                float32x4_t v1 = vld1q_f32(matrix[b+1] + i);
                float32x4_t v2 = vld1q_f32(matrix[b+2] + i);
                float32x4_t v3 = vld1q_f32(matrix[b+3] + i);
                
                sums[0] = vaddq_f32(sums[0], v0);
                sums[1] = vaddq_f32(sums[1], v1);
                sums[2] = vaddq_f32(sums[2], v2);
                sums[3] = vaddq_f32(sums[3], v3);
            }
            
            // Horizontal sum for each batch
            float batch_sums[4] = {
                vaddvq_f32(sums[0]),
                vaddvq_f32(sums[1]),
                vaddvq_f32(sums[2]),
                vaddvq_f32(sums[3])
            };
            
            // Add remaining elements for each batch
            for (; i < feature_dim; ++i) {
                batch_sums[0] += matrix[b][i];
                batch_sums[1] += matrix[b+1][i];
                batch_sums[2] += matrix[b+2][i];
                batch_sums[3] += matrix[b+3][i];
            }
            
            // Calculate means
            results[b] = batch_sums[0] * inv_feature_dim;
            results[b+1] = batch_sums[1] * inv_feature_dim;
            results[b+2] = batch_sums[2] * inv_feature_dim;
            results[b+3] = batch_sums[3] * inv_feature_dim;
        }
        
        // Process remaining batches individually
        for (size_t b = aligned_batch_size; b < batch_size; ++b) {
            const float *vec = matrix[b];
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            
            // Process in chunks of 4 elements
            size_t i = 0;
            for (; i + 4 <= feature_dim; i += 4) {
                float32x4_t vec_chunk = vld1q_f32(vec + i);
                sum_vec = vaddq_f32(sum_vec, vec_chunk);
            }
            
            // Horizontal sum
            float sum = vaddvq_f32(sum_vec);
            
            // Add remaining elements
            for (; i < feature_dim; ++i) {
                sum += vec[i];
            }
            
            // Calculate mean
            results[b] = sum * inv_feature_dim;
        }
#endif // __LINALG_USE_ASM__
    }
#endif // __aarch64__

    /**
     * @brief Portable scalar implementation of batch feature mean pooling
     *
     * @param matrix Array of batch pointers, where each batch contains feature_dim elements
     * @param feature_dim Size of each feature vector
     * @param batch_size Number of batches to process
     * @param results Output array to store batch results (must be pre-allocated with batch_size elements)
     */
    inline void batch_feature_mean_pooling_scalar(const float **matrix, size_t feature_dim, size_t batch_size,
                                              float *results) {
        const float inv_feature_dim = 1.0f / static_cast<float>(feature_dim);
        
        // Use standard algorithms for each batch
        for (size_t b = 0; b < batch_size; ++b) {
            // Use std::accumulate to sum all elements in the feature vector
            results[b] = std::accumulate(matrix[b], matrix[b] + feature_dim, 0.0f) * inv_feature_dim;
        }
    }

} // namespace wheel::linalg_boost::detail
