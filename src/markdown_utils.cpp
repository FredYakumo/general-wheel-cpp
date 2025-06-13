#include "markdown_utils.h"
#include <regex>

std::regex table_pattern(R"(\|.*\|\s*\n\|[-:|]+\|\s*\n(\|.*\|\s*\n)+)");

std::vector<std::regex> rich_text_pattern = {
    std::regex(R"(\*\*.+\*\*)"),     // 加粗文本
    std::regex(R"(\*.+\*)"),         // 斜体文本
    std::regex(R"(\#+.+)"),          // 标题
    std::regex(R"($$.+$$$.+$)"),     // 链接
    std::regex(R"(!$$.+$$$.+$)"),    // 图片
    std::regex(R"(\d+\.\s+.+)"),     // 有序列表
    std::regex(R"([-*+]\s+.+)"),     // 无序列表
    std::regex(R"(`{1,3}.+`{1,3})"), // 代码块
    std::regex(R"(~~.+~~)"),         // 删除线
    std::regex(R"(\|.*\|)")          // 表格行（单独检测）
};
namespace wheel {
    bool contains_markdown_table(const std::string &text) { return std::regex_search(text, table_pattern); }

    bool contains_rich_text_features(const std::string &text) {
        for (const auto &pattern : rich_text_pattern) {
            if (std::regex_search(text, pattern)) {
                return true;
            }
        }

        return false;
    }
} // namespace wheel