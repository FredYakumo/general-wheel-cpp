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
        return detail::dot_product_neon(a, b, size);
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
            results[k] = detail::dot_product_neon(a[k], b, n);
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
        return detail::cosine_similarity_neon(a, b, size);
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

#ifdef __aarch64__
        detail::batch_cosine_similarity_neon(a, b, n, batch_size, results);
        return;
#else
        detail::batch_cosine_similarity_scalar(a, b, n, batch_size, results);
#endif // __aarch64__
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

    /**
     * @brief Calculate mean pooling of multiple vectors
     *
     * Computes the element-wise average of multiple input vectors.
     *
     * @param vectors Array of vector pointers
     * @param n Size of each vector
     * @param num_vectors Number of vectors to average
     * @param result Output vector to store the result (must be pre-allocated with n elements)
     * @throws std::invalid_argument If n is zero, num_vectors is zero, or result is null
     * @note Performance optimized using hardware instruction sets when available
     */
    inline void mean_pooling(const float **vectors, size_t n, size_t num_vectors, float *result) {
        if (n == 0)
            throw std::invalid_argument("mean_pooling: vector size must > 0");
        if (num_vectors == 0)
            throw std::invalid_argument("mean_pooling: number of vectors must > 0");
        if (result == nullptr)
            throw std::invalid_argument("mean_pooling: result pointer cannot be null");

#ifdef __aarch64__
        detail::mean_pooling_neon(vectors, n, num_vectors, result);
#else
        detail::mean_pooling_scalar(vectors, n, num_vectors, result);
#endif
    }

    /**
     * @brief Calculate mean pooling of multiple vectors
     *
     * Computes the element-wise average of multiple input vectors.
     *
     * @param vectors Vector of input vectors
     * @return Vector containing the mean-pooled result
     * @throws std::invalid_argument If vectors are empty or have inconsistent sizes
     * @note Wraps the pointer-based implementation
     */
    inline std::vector<float> mean_pooling(const std::vector<std::vector<float>> &vectors) {
        if (vectors.empty())
            throw std::invalid_argument("mean_pooling: input vector must not be empty");

        const size_t vec_size = vectors[0].size();
        if (vec_size == 0)
            throw std::invalid_argument("mean_pooling: vectors must not be empty");

        // Verify all vectors have the same size
        for (const auto &vec : vectors) {
            if (vec.size() != vec_size)
                throw std::invalid_argument("mean_pooling: all vectors must have the same size");
        }

        // Prepare array of pointers
        std::vector<const float *> vec_ptrs(vectors.size());
        for (size_t i = 0; i < vectors.size(); ++i) {
            vec_ptrs[i] = vectors[i].data();
        }

        // Prepare result vector
        std::vector<float> result(vec_size, 0.0f);

        mean_pooling(vec_ptrs.data(), vec_size, vectors.size(), result.data());

        return result;
    }

    /**
     * @brief Batch mean pooling for a matrix (batch of vectors)
     *
     * Computes the mean pooling vector for each vector in the batch matrix.
     * For a matrix of size [batch_size x channel_dim x feature_dim], returns a vector of size [batch_size x
     * feature_dim].
     *
     * @param matrix Array of batch pointers, where each batch contains channel_dim vectors of feature_dim elements
     * @param feature_dim Size of each feature vector (inner dimension)
     * @param channel_dim Number of vectors to average per batch item (middle dimension)
     * @param batch_size Number of batches to process (outer dimension)
     * @param results Output array to store batch results (must be pre-allocated with batch_size * feature_dim elements)
     * @throws std::invalid_argument If feature_dim is zero, channel_dim is zero, or results is null
     * @note Performance optimized using hardware instruction sets when available
     */
    inline void batch_mean_pooling(const float ***matrix, size_t feature_dim, size_t channel_dim, size_t batch_size,
                                   float **results) {
        if (feature_dim == 0)
            throw std::invalid_argument("batch_mean_pooling: feature dimension must > 0");
        if (channel_dim == 0)
            throw std::invalid_argument("batch_mean_pooling: channel dimension must > 0");
        if (results == nullptr)
            throw std::invalid_argument("batch_mean_pooling: results pointer cannot be null");

#ifdef __aarch64__
        detail::batch_mean_pooling_neon(matrix, feature_dim, channel_dim, batch_size, results);
#else
        detail::batch_mean_pooling_scalar(matrix, feature_dim, channel_dim, batch_size, results);
#endif
    }

    /**
     * @brief Batch mean pooling for a matrix (batch of vectors)
     *
     * Computes the mean pooling vector for each vector in the batch matrix.
     * For a 3D vector of size [batch_size][channel_dim][feature_dim], returns a 2D vector of size
     * [batch_size][feature_dim].
     *
     * @param matrix 3D vector representing batches of channels of feature vectors
     * @return 2D vector containing the mean-pooled results for each batch
     * @throws std::invalid_argument If matrix is empty or has inconsistent dimensions
     * @note Wraps the pointer-based implementation with hardware-specific optimizations
     */
    inline std::vector<std::vector<float>>
    batch_mean_pooling(const std::vector<std::vector<std::vector<float>>> &matrix) {
        if (matrix.empty())
            throw std::invalid_argument("batch_mean_pooling: input matrix must not be empty");

        const size_t batch_size = matrix.size();
        const size_t channel_dim = matrix[0].size();

        if (channel_dim == 0)
            throw std::invalid_argument("batch_mean_pooling: channel dimension must > 0");

        const size_t feature_dim = matrix[0][0].size();

        if (feature_dim == 0)
            throw std::invalid_argument("batch_mean_pooling: feature dimension must > 0");

        // Verify all dimensions are consistent
        for (const auto &batch : matrix) {
            if (batch.size() != channel_dim)
                throw std::invalid_argument("batch_mean_pooling: all batches must have the same channel dimension");

            for (const auto &channel : batch) {
                if (channel.size() != feature_dim)
                    throw std::invalid_argument("batch_mean_pooling: all vectors must have the same feature dimension");
            }
        }

        // Prepare array of batch pointers
        std::vector<const float **> batch_ptrs(batch_size);
        std::vector<std::vector<const float *>> channel_ptrs(batch_size);

        for (size_t b = 0; b < batch_size; ++b) {
            channel_ptrs[b].resize(channel_dim);
            for (size_t c = 0; c < channel_dim; ++c) {
                channel_ptrs[b][c] = matrix[b][c].data();
            }
            batch_ptrs[b] = channel_ptrs[b].data();
        }

        // Prepare result vectors
        std::vector<std::vector<float>> results(batch_size, std::vector<float>(feature_dim, 0.0f));
        std::vector<float *> result_ptrs(batch_size);

        for (size_t b = 0; b < batch_size; ++b) {
            result_ptrs[b] = results[b].data();
        }

        // Call the pointer-based implementation with optimizations
        batch_mean_pooling(batch_ptrs.data(), feature_dim, channel_dim, batch_size, result_ptrs.data());

        return results;
    }
} // namespace wheel::linalg_boost