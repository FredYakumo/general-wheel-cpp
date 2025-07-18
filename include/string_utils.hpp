#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <tuple>

namespace wheel {

    inline std::string_view ltrim(const std::string_view str) {
        const size_t start = str.find_first_not_of(" \n\r");
        return (start == std::string_view::npos) ? std::string_view() : str.substr(start);
    }

    inline std::string_view rtrim(const std::string_view str) {
        const size_t end = str.find_last_not_of(" \n\r");
        return (end == std::string_view::npos) ? std::string_view() : str.substr(0, end + 1);
    }

/**
 * Checks if a byte is the leading byte of a UTF-8 character.
 *
 * @param c The byte to check.
 * @return True if the byte is a UTF-8 leading byte, false otherwise.
 */
#include <cstddef>
    inline bool is_utf8_leader_byte(unsigned char c) {
        return (c & 0xC0) != 0x80; // Leading bytes do not start with 0b10xxxxxx.
    }

    /**
     * Calculates the length of a UTF-8 character based on its leading byte.
     *
     * @param c The leading byte of the UTF-8 character.
     * @return The length of the UTF-8 character in bytes (1-4). If the byte is invalid, returns 1.
     */
    inline size_t utf8_char_length(unsigned char c) {
        if ((c & 0x80) == 0x00) {
            return 1; // 0xxxxxxx: 1-byte character (ASCII).
        } else if ((c & 0xE0) == 0xC0) {
            return 2; // 110xxxxx: 2-byte character.
        } else if ((c & 0xF0) == 0xE0) {
            return 3; // 1110xxxx: 3-byte character.
        } else if ((c & 0xF8) == 0xF0) {
            return 4; // 11110xxx: 4-byte character.
        }
        return 1; // Invalid UTF-8 byte, treat as a single byte.
    }

    /**
     * Replaces all occurrences of a pattern in a string with a replacement string, ensuring UTF-8 compatibility.
     *
     * @param str The input string to process.
     * @param pattern The pattern to search for and replace.
     * @param replace The string to replace the pattern with.
     * @return A new string with all occurrences of the pattern replaced.
     */
    inline std::string replace_str(std::string_view str, std::string_view pattern, std::string_view replace) {
        if (pattern.empty()) {
            return std::string(str); // If the pattern is empty, return the original string.
        }
        std::string result;
        size_t start = 0;
        while (start <= str.size()) {
            // Find the next occurrence of the pattern.
            const size_t pos = str.find(pattern, start);
            if (pos == std::string_view::npos) {
                break; // No more matches fsnd.
            }
            // Ensure the match starts at a valid UTF-8 leader byte.
            if (!is_utf8_leader_byte(str[pos])) {
                start = pos + 1; // Skip invalid UTF-8 sequences.
                continue;
            }
            // Verify the match is complete and valid.
            if (pos + pattern.size() > str.size() || str.substr(pos, pattern.size()) != pattern) {
                const size_t step = utf8_char_length(str[pos]); // Handle multi-byte UTF-8 characters.
                start = pos + (step > 0 ? step : 1);            // Move to the next character.
                continue;
            }
            // Append the part of the string before the match.
            result.append(str.data() + start, pos - start);
            // Append the replacement string.
            result.append(replace.data(), replace.size());
            // Move the start position past the matched pattern.
            start = pos + pattern.size();
        }
        // Append the remaining part of the string after the last match.
        result.append(str.data() + start, str.size() - start);
        return result;
    }

    template <typename STR_ITER>
    inline std::string join_str(STR_ITER cbegin, STR_ITER cend, const std::string_view delimiter = ",") {
        std::string result;
        for (auto it = cbegin; it != cend; ++it) {
            if (it->empty()) {
                continue;
            }
            if (!result.empty()) {
                result += delimiter;
            }
            result += *it;
        }
        return result;
    }

    template <typename STR_ITER>
    inline std::string
    join_str(STR_ITER cbegin, STR_ITER cend, const std::string_view delimiter,
             std::function<std::string(const typename std::iterator_traits<STR_ITER>::value_type &)> transform) {
        std::string result;
        for (auto it = cbegin; it != cend; ++it) {
            auto mapped = transform(*it);
            if (mapped.empty()) {
                continue;
            }
            if (!result.empty()) {
                result += delimiter;
            }
            result += mapped;
        }
        return result;
    }

    inline void remove_text_between_markers(std::string &str, const std::string &start_marker,
                                            const std::string &end_marker) {
        size_t start_pos = str.find(start_marker);
        size_t end_pos = str.find(end_marker);

        if (start_pos != std::string::npos && end_pos != std::string::npos && end_pos > start_pos) {
            str.erase(start_pos, end_pos - start_pos + end_marker.length());
        }
    }

#include <cassert>
#include <string_view>

    class Utf8Splitter {
      public:
        Utf8Splitter(std::string_view str, size_t max_chars) : m_str(str), m_max_chars(max_chars) {
            assert(max_chars > 0 && "Max characters must be greater than 0");
        }

        class Iterator {
          public:
            Iterator(std::string_view str, size_t max_chars, size_t start_pos)
                : m_str(str), m_max_chars(max_chars), m_current_pos(start_pos) {
                if (start_pos != std::string_view::npos) {
                    find_next_boundary();
                }
            }

            std::string_view operator*() const {
                if (m_current_pos == std::string_view::npos)
                    return {};
                return m_str.substr(m_current_pos, m_next_pos - m_current_pos);
            }

            Iterator &operator++() {
                m_current_pos = m_next_pos;
                if (m_current_pos != std::string_view::npos) {
                    find_next_boundary();
                }
                return *this;
            }

            bool operator!=(const Iterator &other) const { return m_current_pos != other.m_current_pos; }

            Iterator operator++(int) {
                auto tmp = *this;
                ++*this;
                return tmp;
            }

          private:
            void find_next_boundary() {
                m_next_pos = m_current_pos;
                size_t chars_count = 0;

                while (chars_count < m_max_chars && m_next_pos < m_str.size()) {
                    const auto code_point_len = utf8_code_point_length(m_str[m_next_pos]);

                    // 检查是否有足够的字节组成完整字符
                    if (m_next_pos + code_point_len > m_str.size())
                        break;

                    m_next_pos += code_point_len;
                    ++chars_count;
                }

                if (chars_count == 0) {
                    m_current_pos = m_next_pos = std::string_view::npos;
                }
            }

            static size_t utf8_code_point_length(char first_byte) noexcept {
                const auto uc = static_cast<unsigned char>(first_byte);
                if (uc < 0x80)
                    return 1; // ASCII
                if ((uc & 0xE0) == 0xC0)
                    return 2; // 2-byte
                if ((uc & 0xF0) == 0xE0)
                    return 3; // 3-byte
                if ((uc & 0xF8) == 0xF0)
                    return 4; // 4-byte
                return 1;     // 无效序列按单字节处理
            }

            std::string_view m_str;
            size_t m_max_chars;
            size_t m_current_pos;
            size_t m_next_pos = std::string_view::npos;
        };

        Iterator begin() const { return {m_str, m_max_chars, 0}; }
        Iterator end() const { return {m_str, m_max_chars, std::string_view::npos}; }

      private:
        std::string_view m_str;
        size_t m_max_chars;
    };

    class SplitString {
      public:
        SplitString(const std::string_view str, const char delimiter)
            : m_str(str), m_delimiter(delimiter), m_start(0) {}

        /**
         * @brief Gets the I-th component of the split string.
         * @tparam I The index of the component to retrieve.
         * @return A string_view of the I-th component.
         * @note This enables structured bindings, e.g., auto [key, value] = SplitString("key=value", '=');
         */
        template <size_t I>
        std::string_view get() const {
            static_assert(I < 2, "Index out of bounds for SplitString structured binding");
            const size_t pos = m_str.find(m_delimiter);

            if (pos == std::string_view::npos) {
                if constexpr (I == 0) {
                    return m_str;
                } else {
                    throw std::out_of_range("SplitString has no second component");
                }
            }

            if constexpr (I == 0) {
                return m_str.substr(0, pos);
            } else { // I == 1
                return m_str.substr(pos + 1);
            }
        }

        class Iterator {
          public:
            Iterator(const std::string_view str, const char delimiter, const size_t start)
                : m_str(str), m_delimiter(delimiter), m_start(start) {
                if (str.empty()) {
                    m_start = m_end = std::string_view::npos;
                } else {
                    find_next();
                }
            }

            std::string_view operator*() const {
                return m_str.substr(m_start,
                                    (m_end == std::string_view::npos) ? std::string_view::npos : m_end - m_start);
            }

            Iterator &operator++() {
                m_start = m_end;
                if (m_start != std::string_view::npos) {
                    ++m_start;
                    find_next();
                }
                return *this;
            }

            bool operator!=(const Iterator &other) const { return m_start != other.m_start; }
            Iterator operator++(int) {
                const Iterator tmp = *this;
                ++*this;
                return tmp;
            }

          private:
            void find_next() { m_end = m_str.find(m_delimiter, m_start); }

            std::string_view m_str;
            char m_delimiter;
            size_t m_start;
            size_t m_end{};
        };

        [[nodiscard]] Iterator begin() const { return {m_str, m_delimiter, m_start}; }
        [[nodiscard]] Iterator end() const { return {m_str, m_delimiter, std::string_view::npos}; }

      private:
        std::string_view m_str;
        char m_delimiter;
        size_t m_start;
    };

    inline std::string to_lower_str(std::string_view str) {
        std::string result;
        result.reserve(str.size());
        for (const auto &c : str) {
            result += static_cast<char>(std::tolower(c));
        }
        return result;
    }

    inline std::string to_upper_str(std::string_view str) {
        std::string result;
        result.reserve(str.size());
        for (const auto &c : str) {
            result += static_cast<char>(std::toupper(c));
        }
        return result;
    }

    inline std::string url_encode(std::string_view str) {
        std::string result;
        result.reserve(str.size() * 3); // Reserve space for potential encoding

        for (unsigned char c : str) {
            if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
                result += c;
            } else {
                result += '%';
                result += "0123456789ABCDEF"[c >> 4];
                result += "0123456789ABCDEF"[c & 15];
            }
        }
        return result;
    }

} // namespace wheel

namespace std {
template <>
struct tuple_size<wheel::SplitString> : std::integral_constant<size_t, 2> {};

template <size_t I>
struct tuple_element<I, wheel::SplitString> {
    static_assert(I < 2, "Index out of bounds for SplitString structured binding");
    using type = std::string_view;
};
} // namespace std