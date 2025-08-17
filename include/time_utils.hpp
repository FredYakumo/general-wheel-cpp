/**
 * @file time_utils.hpp
 * @brief Utility helpers for measuring execution time of arbitrary callables.
 */

#pragma once

#include <chrono>
#include <type_traits>
#include <utility>
#include <functional>

namespace wheel {

/**
 * @brief Measure the execution duration of a callable (function, lambda, functor).
 *
 * Executes the provided callable with the forwarded arguments and returns the
 * elapsed time as a std::chrono::nanoseconds duration. The return value of the
 * callable (if any) is discarded. Use @ref measure_duration_with_result if you
 * also need the result value.
 *
 * Thread-safety: This function itself is thread-safe; it does not share state.
 *
 * @tparam F Callable type.
 * @tparam Args Argument types.
 * @param f The callable to execute.
 * @param args Arguments to forward to the callable.
 * @return std::chrono::nanoseconds Execution time duration.
 */
template <typename F, typename... Args>
inline std::chrono::nanoseconds measure_duration(F&& f, Args&&... args) {
	using clock = std::chrono::high_resolution_clock;
	const auto start = clock::now();
	std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
	const auto end = clock::now();
	return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
}

/**
 * @brief Measure execution duration and return both the result and duration.
 *
 * Executes the callable and returns a std::pair containing the callable's
 * result and the measured duration in nanoseconds. Overload is enabled only
 * for non-void returning callables.
 *
 * Thread-safety: This function itself is thread-safe; it does not share state.
 *
 * @tparam F Callable type.
 * @tparam Args Argument types.
 * @param f The callable to execute.
 * @param args Arguments to forward to the callable.
 * @return std::pair<R, std::chrono::nanoseconds> where R is the callable's return type.
 */
template <typename F, typename... Args,
		  typename R = std::invoke_result_t<F, Args...>,
		  typename = std::enable_if_t<!std::is_void_v<R>>>
inline std::pair<R, std::chrono::nanoseconds> measure_duration_with_result(F&& f, Args&&... args) {
	using clock = std::chrono::high_resolution_clock;
	const auto start = clock::now();
	R result = std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
	const auto end = clock::now();
	return { std::move(result), std::chrono::duration_cast<std::chrono::nanoseconds>(end - start) };
}

/**
 * @brief Void-return specialization returning only the duration.
 *
 * This overload is provided for symmetry; for void-returning callables it is
 * equivalent to calling @ref measure_duration.
 *
 * @tparam F Callable type.
 * @tparam Args Argument types.
 * @return std::chrono::nanoseconds Execution time duration.
 */
template <typename F, typename... Args,
		  typename R = std::invoke_result_t<F, Args...>,
		  typename = std::enable_if_t<std::is_void_v<R>>, typename = void>
inline std::chrono::nanoseconds measure_duration_with_result(F&& f, Args&&... args) {
	return measure_duration(std::forward<F>(f), std::forward<Args>(args)...);
}

} // namespace wheel
