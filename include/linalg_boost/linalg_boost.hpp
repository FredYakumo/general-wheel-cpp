#pragma once

#include "vec_ops.hpp"
#include <stdexcept>
#include <vector>
#include <algorithm>

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
     * @brief Calculate dot product for multiple vectors against a reference vector
     *
     * @param a Array of vector pointers
     * @param b Pointer to the reference vector
     * @param n Size of each vector
     * @param batch_size Number of vectors in a
     * @param results Output array to store dot product results (must be pre-allocated with batch_size elements)
     * @throws std::invalid_argument If n is zero or results is null
     * @note Performance optimized using hardware instruction sets
     */
    inline void batch_dot_product(const float **a, const float *b, size_t n, size_t batch_size, float *results) {
        if (n == 0)
            throw std::invalid_argument("batch_dot_product: vector size must > 0");
        if (results == nullptr)
            throw std::invalid_argument("batch_dot_product: results pointer cannot be null");

        for (size_t k = 0; k < batch_size; ++k) {
#ifdef __aarch64__
#ifdef LINALG_USE_ASM
            results[k] = detail::dot_product_asm_aarch64(a[k], b, n);
#else
            results[k] = detail::dot_product_neon(a[k], b, n);
#endif
#else
            results[k] = detail::dot_product_scalar(a[k], b, n);
#endif
        }
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
     * @param results Output array to store similarity results (must be pre-allocated with batch_size elements)
     * @throws std::invalid_argument If n is zero or results is null
     * @note Performance optimized using hardware instruction sets
     * @note Automatically selects the best implementation based on available hardware
     */
    inline void batch_cosine_similarity(const float **a, const float *b, size_t n, size_t batch_size, float *results) {
        if (n == 0)
            throw std::invalid_argument("batch_cosine_similarity: vector size must > 0");
        if (results == nullptr)
            throw std::invalid_argument("batch_cosine_similarity: results pointer cannot be null");
            
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
    
    /**
     * @brief Batch calculate top-k similar vectors from a collection
     * 
     * Finds the k most similar vectors to the reference vector and returns their indices
     * and similarity scores in descending order of similarity.
     * 
     * @param vectors Array of vector pointers in the collection
     * @param ref_vector Reference vector to compare against
     * @param vec_size Size of each vector
     * @param collection_size Number of vectors in the collection
     * @param k Number of top results to return
     * @param indices Output array to store indices of top-k similar vectors (must be pre-allocated with k elements)
     * @param scores Output array to store similarity scores of top-k vectors (must be pre-allocated with k elements)
     * @throws std::invalid_argument If vec_size is zero, k is zero, or k > collection_size
     */
    inline void top_k_similar(const float **vectors, const float *ref_vector, size_t vec_size, 
                              size_t collection_size, size_t k, size_t *indices, float *scores) {
        if (vec_size == 0)
            throw std::invalid_argument("top_k_similar: vector size must > 0");
        if (k == 0)
            throw std::invalid_argument("top_k_similar: k must > 0");
        if (k > collection_size)
            throw std::invalid_argument("top_k_similar: k cannot be greater than collection_size");
            
        // Calculate all similarities
        std::vector<float> all_similarities(collection_size);
        batch_cosine_similarity(vectors, ref_vector, vec_size, collection_size, all_similarities.data());
        
        // Create index vector
        std::vector<size_t> all_indices(collection_size);
        for (size_t i = 0; i < collection_size; ++i) {
            all_indices[i] = i;
        }
        
        // Partial sort to find top-k
        std::partial_sort(all_indices.begin(), all_indices.begin() + k, all_indices.end(),
            [&all_similarities](size_t a, size_t b) {
                return all_similarities[a] > all_similarities[b];
            });
            
        // Copy results
        for (size_t i = 0; i < k; ++i) {
            indices[i] = all_indices[i];
            scores[i] = all_similarities[all_indices[i]];
        }
    }
} // namespace wheel::linalg_boost