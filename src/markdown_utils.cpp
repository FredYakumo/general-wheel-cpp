#include "markdown_utils.h"
#include "code_default_highlight.hpp"
#include "latex_display_script.hpp"
#include "string_utils.hpp"
#include <cctype>
#include <regex>
#include <sstream>

// std::regex table_pattern(R"(\|.*\|\s*\n\|[-:|]+\|\s*\n(\|.*\|\s*\n)+)");
std::regex table_pattern(R"(\|.*\|)");

std::regex single_line_latex(R"(\[.+\]\(.+\)|(?:\$[^$\n]+\$)|(?:\\[\(\[].*?\\[\)\]])|(?:\\begin\{.*?\}.*?\\end\{.*?\}))"); // 单行LaTex

std::vector<std::regex> rich_text_pattern = {std::regex(R"(\*\*.+\*\*)"),     // 加粗文本
                                             std::regex(R"(\*.+\*)"),         // 斜体文本
                                             std::regex(R"(\#+.+)"),          // 标题
                                             std::regex(R"($$.+$$$.+$)"),     // 链接
                                             std::regex(R"(!$$.+$$$.+$)"),    // 图片
                                             std::regex(R"(\d+\.\s+.+)"),     // 有序列表
                                             std::regex(R"([-*+]\s+.+)"),     // 无序列表
                                             std::regex(R"(`{1,3}.+`{1,3})"), // 代码块
                                             std::regex(R"(~~.+~~)"),         // 删除线
                                             std::regex(R"(\|.*\|)"),         // 表格行（单独检测）
                                            //  latex_link_pattern
                                             };
namespace wheel {
    bool contains_markdown_table(const std::string &text) { return std::regex_search(text, table_pattern); }

    bool contains_rich_text_features(const std::string &text) {
        // if (text.find('[') != std::string::npos && text.find(']') != std::string::npos) {
        //     return true;
        // }
        for (const auto &pattern : rich_text_pattern) {
            if (std::regex_search(text, pattern)) {
                return true;
            }
        }

        return false;
    }

    std::string markdown_table_to_html(const std::string &markdown, size_t border_width_px) {
        std::vector<std::string> lines;

        std::vector<std::vector<std::string>> table_data;
        bool in_table = false;

        for (const auto &line : SplitString(markdown, '\n')) {
            if (line.empty())
                continue;

            if (line[0] == '|') {
                in_table = true;
                auto cells = SplitString(line.substr(1, line.size() - 2), '|');
                std::vector<std::string> row;
                for (const auto &cell : cells) {
                    row.push_back(std::string(ltrim(rtrim(cell))));
                }
                table_data.push_back(std::move(row));
            } else if (in_table) {
                break;
            }
        }

        if (table_data.size() < 2) {
            return "";
        }

        std::string html = "<table style=\"border: 1px solid black; border-collapse: collapse;\">\n";

        // Header
        html += "  <thead>\n    <tr>\n";
        for (const auto &cell : table_data[0]) {
            html += "      <th style=\"border: ";
            html += std::to_string(border_width_px) + "px solid black; padding: 8px;\">" + cell + "</th>\n";
        }
        html += "    </tr>\n  </thead>\n";

        // Body
        html += "  <tbody>\n";
        for (size_t i = 2; i < table_data.size(); i++) {
            html += "    <tr>\n";
            for (const auto &cell : table_data[i]) {
                html += "      <td style=\"border: ";
                html += std::to_string(border_width_px) + "px solid black; padding: 8px;\">" + cell + "</td>\n";
            }
            html += "    </tr>\n";
        }
        html += "  </tbody>\n";

        html += "</table>";
        return html;
    }

    std::regex bold_regex(R"(\*\*(.+?)\*\*)");
    std::regex italic_regex(R"(\*(.+?)\*)");
    std::regex strikethrough_regex(R"(~~(.+?)~~)");
    std::regex header_regex(R"(^(#{1,6})\s(.+)$)");
    std::regex background_text_regex(R"(`([^`]+)`)");

    std::string markdown_rich_text_to_html(const std::string &text) {
        std::string result = text;

        // Handle escaped square brackets
        result = replace_str(result, "\\[", "&#91;"); // Replace \[ with &#91;
        result = replace_str(result, "\\]", "&#93;"); // Replace \] with &#93;

        // Handle normal square brackets (if they should be converted to specific HTML)
        // For example, if they represent links:
        std::regex link_pattern(R"(\[(.*?)\]\((.*?)\))");
        result = std::regex_replace(result, link_pattern, "<a href=\"$2\">$1</a>");

        std::istringstream iss(result);
        std::ostringstream oss;
        std::string line;

        while (std::getline(iss, line)) {
            std::smatch header_match;
            if (std::regex_match(line, header_match, header_regex)) {
                int level = header_match[1].length(); // # 的数量 (1-6)
                std::string header_text = header_match[2];
                auto level_str = std::to_string(level);
                line = "<h" + level_str + ">" + header_text + "</h" + level_str + ">";
            }
            oss << line << "\n";
        }

        result = oss.str();
        // 处理其他格式
        result = std::regex_replace(result, bold_regex, "<strong>$1</strong>");
        result = std::regex_replace(result, italic_regex, "<em>$1</em>");
        result = std::regex_replace(result, strikethrough_regex, "<del>$1</del>");
        result = std::regex_replace(
            result, background_text_regex,
            "<code style='background-color: #a0a0a0; padding: 2px 4px; border-radius: 3px;'>$1</code>");
        // result = replace_str(result, "\n", "<br/>");

        // Add MathJax
        std::string max_jax_script = "<script id='MathJax-script'>" + LATEX_DISPLAY_SCRIPT + "</script>";

        return max_jax_script + result;
    }

    inline bool is_code_block_line(std::string_view line) { return ltrim(line).find("```") == 0; }

    inline bool is_latex_block_begin_line(std::string_view line) {
        auto lt = to_lower_str(ltrim(line));
        return lt.find("$$") == 0 || lt.find("\\[") == 0 || lt.find("\\begin{equation}") == 0;
    }

    inline bool is_latex_block_end_line(std::string_view line) {
        auto lt = to_lower_str(ltrim(line));
        return lt.find("$$") == 0 || lt.find("\\]") == 0 || lt.find("\\end{equation}") == 0;
    }

    std::string code_block_text_to_html(const std::string &code_text) {
        std::string code;
        // Replace < and > with their HTML entities
        code = replace_str(code, "<", "&lt;");
        code = replace_str(code, ">", "&gt;");
        std::string language;
        for (const auto &line : SplitString(code, '\n')) {
            if (is_code_block_line(line)) {
                if (line.length() > 3) {
                    language = line.substr(3); // Extract language after ```
                }
                continue; // 跳过代码块的开始和结束标记
            }
            code += std::string(line) + "\n"; // 保留代码行
        }

        std::string html = R"(<!-- 引入 highlight.js -->
        <style>)" + CODE_HIGHLIGHT_CSS +
                           R"(</style>
        <script>)" + CODE_HIGHLIGHT_JS +
                           R"(</script>
        
        <!-- 使用代码块 -->
        <pre><code class="language-)" +
                           (language.empty() ? "plaintext" : language) + R"(">)";

        // Process indent
        // code = replace_str(code, "\t", "    ");
        html += code;
        html += R"(</code></pre>
        <!-- 初始化 highlight.js -->
        <script>hljs.highlightAll();</script>
        )";
        return html;
    }

    std::string latex_block_text_to_html(const std::string &latex_text) {
        // Replace < and > with their HTML entities
        std::string latex = replace_str(latex_text, "<", "&lt;");
        latex = replace_str(latex, ">", "&gt;");

        // Add MathJax script
        std::string math_jax_script = "<script id='MathJax-script'>" + LATEX_DISPLAY_SCRIPT + "</script>";

        // Wrap in a div with class for MathJax processing
        return math_jax_script + "<div class='mathjax'>" + latex + "</div>";
    }

    std::vector<MarkdownNode> parse_markdown(const std::string &markdown) {
        std::vector<MarkdownNode> nodes;
        auto lines = SplitString(markdown, '\n');

        MarkdownNode current_node;
        bool in_code_block = false;
        bool in_table = false;
        bool in_latex_block = false;
        std::string code_block_content;
        std::string table_content;
        std::string latex_block_content;

        bool on_skip_code_markdown = false;

        for (const auto &l : lines) {
            if (l.empty())
                continue;
            std::string line{l};

            // Process code block
            if (is_code_block_line(line)) {
                // Skip ```markdown
                auto lt = to_lower_str(ltrim(line));
                if (lt.find("```markdown") == 0 || lt.find("```md") == 0 || lt.find("```text") == 0) {
                    on_skip_code_markdown = true;
                    continue;
                } else if (lt.find("```") == 0 && on_skip_code_markdown) {
                    on_skip_code_markdown = false;
                }

                if (!in_code_block) {
                    // Start code block
                    if (!current_node.text.empty() && !current_node.table_text && !current_node.code_text) {
                        nodes.push_back(current_node);
                        current_node = MarkdownNode();
                    }
                    in_code_block = true;
                    code_block_content = line + "\n";
                } else {
                    // End code
                    code_block_content += line;
                    current_node.code_text = code_block_content;
                    current_node.text = code_block_content;
                    nodes.push_back(current_node);
                    current_node = MarkdownNode();
                    in_code_block = false;
                }
                continue;
            }

            if (in_code_block) {
                code_block_content += line + "\n";
                continue;
            }

            // Check if line contains LaTeX link
            if (std::regex_search(line, single_line_latex)) {
                if (!current_node.text.empty() && !current_node.latex_text && !current_node.table_text && !current_node.code_text) {
                    nodes.push_back(current_node);
                    current_node = MarkdownNode();
                }
                current_node.latex_text = line;
                current_node.text = line;
                nodes.push_back(current_node);
                current_node = MarkdownNode();
                continue;
            }

            if (is_latex_block_begin_line(line)) {
                if (!in_latex_block) {
                    // Start LaTeX block
                    if (!current_node.text.empty() && !current_node.table_text && !current_node.code_text) {
                        nodes.push_back(current_node);
                        current_node = MarkdownNode();
                    }
                    in_latex_block = true;
                    latex_block_content = line + "\n";
                }
                continue;
            } else if (is_latex_block_end_line(line)) {
                if (in_latex_block) {
                    // End LaTeX block
                    latex_block_content += line + "\n";
                    current_node.latex_text = latex_block_content;
                    current_node.text = latex_block_content;
                    nodes.push_back(current_node);
                    current_node = MarkdownNode();
                    in_latex_block = false;
                }
                continue;
            }

            if (in_latex_block) {
                latex_block_content += line + "\n";
                continue;
            }

            // Start table
            if (contains_markdown_table(line)) {
                if (!in_table) {
                    // start table
                    if (!current_node.text.empty() && !current_node.table_text && !current_node.code_text) {
                        nodes.push_back(current_node);
                        current_node = MarkdownNode();
                    }
                    in_table = true;
                    table_content = line + "\n";
                } else {
                    // Start table
                    table_content += line + "\n";
                }
                continue;
            }

            if (in_table) {
                // 表格结束
                current_node.table_text = table_content;
                current_node.text = table_content;
                nodes.push_back(current_node);
                current_node = MarkdownNode();
                in_table = false;
            }

            // normal text or rich text
            if (contains_rich_text_features(line)) {
                if (!current_node.rich_text) {
                    current_node.rich_text = line;
                } else {
                    *current_node.rich_text += "\n" + line;
                }
            }

            if (!current_node.text.empty()) {
                current_node.text += "\n";
            }
            current_node.text += line;
        }

        // Add last node
        if (!current_node.text.empty() || current_node.table_text || current_node.code_text || current_node.rich_text) {
            nodes.push_back(current_node);
        }

        for (auto &node : nodes) {
            if (node.table_text.has_value()) {
                node.render_html_text = markdown_table_to_html(*node.table_text);
            } else if (node.code_text.has_value()) {
                node.render_html_text = code_block_text_to_html(*node.code_text);
            } else if (node.rich_text.has_value()) {
                node.render_html_text = markdown_rich_text_to_html(*node.rich_text);
            } else if (node.latex_text.has_value()) {
                node.render_html_text = latex_block_text_to_html(*node.latex_text);
            }
        }

        return nodes;
    }
} // namespace wheel