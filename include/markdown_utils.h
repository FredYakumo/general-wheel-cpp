#pragma once

#include "string_utils.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace wheel {
    bool contains_markdown_table(const std::string &text);
    
    bool contains_rich_text_features(const std::string &text);
    
    inline std::string markdown_table_to_html(const std::string &markdown, size_t border_width_px = 1) {
        std::vector<std::string> lines;
        size_t start = 0;
        size_t end = markdown.find('\n');

        while (end != std::string::npos) {
            lines.push_back(markdown.substr(start, end - start));
            start = end + 1;
            end = markdown.find('\n', start);
        }
        lines.push_back(markdown.substr(start));

        std::vector<std::vector<std::string>> table_data;
        bool in_table = false;

        for (const auto &line : lines) {
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
} // namespace wheel