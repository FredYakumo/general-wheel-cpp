#pragma once

#include "vec_ops.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>

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
     * @brief Calculate dot product between two std::vector<float>
     *
     * @param a First vector
     * @param b Second vector
     * @return Dot product result
     * @throws std::invalid_argument If vectors are empty or have different sizes
     * @note Wraps the pointer-based implementation
     */
    inline float dot_product(const std::vector<float> &a, const std::vector<float> &b) {
        if (a.empty())
            throw std::invalid_argument("dot_product: vectors must not be empty");
        if (a.size() != b.size())
            throw std::invalid_argument("dot_product: vectors must have the same size");

        return dot_product(a.data(), b.data(), a.size());
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
     * @brief Calculate dot product for multiple vectors against a reference vector
     *
     * @param a Vector of input vectors
     * @param b Reference vector
     * @return Vector containing dot product results
     * @throws std::invalid_argument If vectors are empty or have inconsistent sizes
     * @note Wraps the pointer-based implementation
     */
    inline std::vector<float> batch_dot_product(const std::vector<std::vector<float>> &a, const std::vector<float> &b) {
        if (a.empty())
            throw std::invalid_argument("batch_dot_product: input vector must not be empty");
        if (b.empty())
            throw std::invalid_argument("batch_dot_product: reference vector must not be empty");

        // Verify all vectors have the same size as the reference vector
        for (const auto &vec : a) {
            if (vec.size() != b.size())
                throw std::invalid_argument(
                    "batch_dot_product: all vectors must have the same size as the reference vector");
        }

        // Prepare array of pointers
        std::vector<const float *> vec_ptrs(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            vec_ptrs[i] = a[i].data();
        }

        // Prepare results vector
        std::vector<float> results(a.size());

        batch_dot_product(vec_ptrs.data(), b.data(), b.size(), a.size(), results.data());

        return results;
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
     * @brief Calculate cosine similarity between two std::vector<float>
     *
     * @param a First vector
     * @param b Second vector
     * @return Cosine similarity value in range [-1, 1]
     * @throws std::invalid_argument If vectors are empty or have different sizes
     * @note Returns 0 when either vector is a zero vector
     * @note Wraps the pointer-based implementation
     */
    inline float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b) {
        if (a.empty())
            throw std::invalid_argument("cosine_similarity: vectors must not be empty");
        if (a.size() != b.size())
            throw std::invalid_argument("cosine_similarity: vectors must have the same size");

        return cosine_similarity(a.data(), b.data(), a.size());
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
     * @brief Calculate cosine similarity for multiple vectors against a reference vector
     *
     * Efficiently computes similarity values for multiple input vectors against a single reference vector.
     *
     * @param a Vector of input vectors
     * @param b Reference vector
     * @return Vector containing similarity results
     * @throws std::invalid_argument If vectors are empty or have inconsistent sizes
     * @note Wraps the pointer-based implementation
     * @note Automatically selects the best implementation based on available hardware
     */
    inline std::vector<float> batch_cosine_similarity(const std::vector<std::vector<float>> &a,
                                                      const std::vector<float> &b) {
        if (a.empty())
            throw std::invalid_argument("batch_cosine_similarity: input vector must not be empty");
        if (b.empty())
            throw std::invalid_argument("batch_cosine_similarity: reference vector must not be empty");

        // Verify all vectors have the same size as the reference vector
        for (const auto &vec : a) {
            if (vec.size() != b.size())
                throw std::invalid_argument(
                    "batch_cosine_similarity: all vectors must have the same size as the reference vector");
        }

        // Prepare array of pointers
        std::vector<const float *> vec_ptrs(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            vec_ptrs[i] = a[i].data();
        }

        // Prepare results vector
        std::vector<float> results(a.size());

        batch_cosine_similarity(vec_ptrs.data(), b.data(), b.size(), a.size(), results.data());

        return results;
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
    inline void top_k_similar(const float **vectors, const float *ref_vector, size_t vec_size, size_t collection_size,
                              size_t k, size_t *indices, float *scores) {
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
        std::partial_sort(
            all_indices.begin(), all_indices.begin() + k, all_indices.end(),
            [&all_similarities](size_t a, size_t b) { return all_similarities[a] > all_similarities[b]; });

        // Copy results
        for (size_t i = 0; i < k; ++i) {
            indices[i] = all_indices[i];
            scores[i] = all_similarities[all_indices[i]];
        }
    }

    /**
     * @brief Batch calculate top-k similar vectors from a collection
     *
     * Finds the k most similar vectors to the reference vector and returns their indices
     * and similarity scores in descending order of similarity.
     *
     * @param vectors Collection of vectors to compare against the reference vector
     * @param ref_vector Reference vector to compare against
     * @param k Number of top results to return
     * @return A pair of vectors: first contains indices, second contains corresponding similarity scores
     * @throws std::invalid_argument If vectors are empty, k is zero, or k > collection size
     * @note Wraps the pointer-based implementation
     */
    inline std::pair<std::vector<size_t>, std::vector<float>>
    top_k_similar(const std::vector<std::vector<float>> &vectors, const std::vector<float> &ref_vector, size_t k) {
        if (vectors.empty())
            throw std::invalid_argument("top_k_similar: vector collection must not be empty");
        if (ref_vector.empty())
            throw std::invalid_argument("top_k_similar: reference vector must not be empty");
        if (k == 0)
            throw std::invalid_argument("top_k_similar: k must > 0");
        if (k > vectors.size())
            throw std::invalid_argument("top_k_similar: k cannot be greater than collection_size");

        // Verify all vectors have the same size as the reference vector
        for (const auto &vec : vectors) {
            if (vec.size() != ref_vector.size())
                throw std::invalid_argument(
                    "top_k_similar: all vectors must have the same size as the reference vector");
        }

        // Prepare array of pointers
        std::vector<const float *> vec_ptrs(vectors.size());
        for (size_t i = 0; i < vectors.size(); ++i) {
            vec_ptrs[i] = vectors[i].data();
        }

        // Prepare result vectors
        std::vector<size_t> indices(k);
        std::vector<float> scores(k);

        top_k_similar(vec_ptrs.data(), ref_vector.data(), ref_vector.size(), vectors.size(), k, indices.data(),
                      scores.data());

        return {indices, scores};
    }
} // namespace wheel::linalg_boost