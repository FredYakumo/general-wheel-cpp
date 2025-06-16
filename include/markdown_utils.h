#pragma once

#include "string_utils.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace wheel {
    bool contains_markdown_table(const std::string &text);
    
    bool contains_rich_text_features(const std::string &text);
    
    struct MarkdownNode {
        std::string text;   // 原始块
        std::optional<std::string> table_text;  // 如果存在markdown表格
        std::optional<std::string> rich_text;   // 如果存在富文本
        std::optional<std::string> code_text;   // 如果存在代码块
        std::optional<std::string> latex_text; // 如果存在latex块
        std::optional<std::string> render_html_text; // 转换为html后的内容
    };

    std::vector<MarkdownNode> parse_markdown(const std::string& markdown);
    
    std::string markdown_table_to_html(const std::string &markdown, size_t border_width_px = 1);
    
    std::string markdown_rich_text_to_html(const std::string &text);

    std::string code_block_text_to_html(const std::string &code_text);
} // namespace wheel