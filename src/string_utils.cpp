#include "string_utils.hpp"
#include <regex>

std::regex table_pattern(R"(\|.*\|\s*\n\|[-:|]+\|\s*\n(\|.*\|\s*\n)+)");

bool contains_markdown_table(const std::string &text) { return std::regex_search(text, table_pattern); }

std::vector<std::regex> patterns = {
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

bool contains_rich_text_features(const std::string &text) {
    for (const auto &pattern : patterns) {
        if (std::regex_search(text, pattern)) {
            return true;
        }
    }

    return false;
}