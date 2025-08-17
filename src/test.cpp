#include "linalg_boost/linalg_boost.hpp"
#include "markdown_utils.h"
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <random>
#include <string>
#include <vector>

// Helper function to check if two float values are approximately equal
bool almost_equal(float a, float b, float epsilon = 1e-6f) { return std::abs(a - b) < epsilon; }

// Test function for dot product
void test_dot_product() {
    std::cout << "Testing dot product function...\n";
    bool all_passed = true;

    // Test case 1: Simple vectors
    {
        std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};
        float result = wheel::linalg_boost::dot_product(a.data(), b.data(), a.size());
        float expected = 1.0f * 5.0f + 2.0f * 6.0f + 3.0f * 7.0f + 4.0f * 8.0f; // 70.0f
        std::cout << "  Case 1: result = " << result << ", expected = " << expected << "\n";
        if (!almost_equal(result, expected)) {
            std::cout << "  FAILED: Test case 1\n";
            all_passed = false;
        }
    }

    // Test case 2: Orthogonal vectors
    {
        std::vector<float> a = {1.0f, 0.0f, 0.0f};
        std::vector<float> b = {0.0f, 1.0f, 0.0f};
        float result = wheel::linalg_boost::dot_product(a.data(), b.data(), a.size());
        float expected = 0.0f;
        std::cout << "  Case 2: result = " << result << ", expected = " << expected << "\n";
        if (!almost_equal(result, expected)) {
            std::cout << "  FAILED: Test case 2\n";
            all_passed = false;
        }
    }

    // Test case 3: Longer vectors to test SIMD optimization
    {
        std::vector<float> a(16, 1.0f);
        std::vector<float> b(16, 2.0f);
        float result = wheel::linalg_boost::dot_product(a.data(), b.data(), a.size());
        float expected = 32.0f; // 16 elements * 1.0 * 2.0
        std::cout << "  Case 3: result = " << result << ", expected = " << expected << "\n";
        if (!almost_equal(result, expected)) {
            std::cout << "  FAILED: Test case 3\n";
            all_passed = false;
        }
    }

    if (all_passed) {
        std::cout << "Dot product tests passed!\n\n";
    } else {
        std::cout << "Some dot product tests failed!\n\n";
    }
}

// Test function for cosine similarity
void test_cosine_similarity() {
    std::cout << "Testing cosine similarity function...\n";
    bool all_passed = true;

    // Test case 1: Identical vectors
    {
        std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};
        float result = wheel::linalg_boost::cosine_similarity(a.data(), b.data(), a.size());
        float expected = 1.0f;
        std::cout << "  Case 1: result = " << result << ", expected = " << expected << "\n";
        if (!almost_equal(result, expected)) {
            std::cout << "  FAILED: Test case 1\n";
            all_passed = false;
        }
    }

    // Test case 2: Orthogonal vectors
    {
        std::vector<float> a = {1.0f, 0.0f, 0.0f};
        std::vector<float> b = {0.0f, 1.0f, 0.0f};
        float result = wheel::linalg_boost::cosine_similarity(a.data(), b.data(), a.size());
        float expected = 0.0f;
        std::cout << "  Case 2: result = " << result << ", expected = " << expected << "\n";
        if (!almost_equal(result, expected)) {
            std::cout << "  FAILED: Test case 2\n";
            all_passed = false;
        }
    }

    // Test case 3: Opposite direction vectors
    {
        std::vector<float> a = {1.0f, 2.0f, 3.0f};
        std::vector<float> b = {-1.0f, -2.0f, -3.0f};
        float result = wheel::linalg_boost::cosine_similarity(a.data(), b.data(), a.size());
        float expected = -1.0f;
        std::cout << "  Case 3: result = " << result << ", expected = " << expected << "\n";
        if (!almost_equal(result, expected)) {
            std::cout << "  FAILED: Test case 3\n";
            all_passed = false;
        }
    }

    // Test case 4: Longer vectors to test SIMD optimization
    {
        std::vector<float> a(16, 1.0f);
        std::vector<float> b(16, 2.0f);
        float result = wheel::linalg_boost::cosine_similarity(a.data(), b.data(), a.size());
        float expected = 1.0f; // Vectors in same direction
        std::cout << "  Case 4: result = " << result << ", expected = " << expected << "\n";
        if (!almost_equal(result, expected)) {
            std::cout << "  FAILED: Test case 4\n";
            all_passed = false;
        }
    }

    if (all_passed) {
        std::cout << "Cosine similarity tests passed!\n\n";
    } else {
        std::cout << "Some cosine similarity tests failed!\n\n";
    }
}

// Test function for markdown parsing and rendering
void test_markdown_parsing() {
    std::cout << "Testing markdown parsing and rendering...\n";
    
    // Test markdown content
    std::string markdown_text = R"(这个问题需要从**弹道学基础原理**和**创伤弹道学**两个维度来分析。让我一步步拆解：

### 1. **停止力（Stopping Power）的本质**
**停止力**指子弹使目标立即丧失行动能力的效力，核心取决于：
- **能量传递效率**：子弹在目标体内释放动能的百分比
- **创伤空腔体积**：子弹在组织中造成的永久性损伤区域大小
- **神经/骨骼系统破坏**：对关键生理结构的直接损伤

### 2. **.45 ACP手枪弹的特性**（以典型230格令弹头为例）
```math
动能公式：E_k = \frac{1}{2}mv^2
```
- **低速重弹**：初速约260m/s，质量约15g
- **大直径**：11.43mm弹径（步枪弹通常5.56-7.62mm）
- **能量传递机制**：
  - 击中目标后迅速减速，**动能释放率>85%**
  - 形成直径达8-10cm的**永久空腔**
  - 在肌肉组织产生剧烈液压冲击波

### 3. **高穿透性步枪弹的特性**（如5.56×45mm M855）
- **高速轻弹**：初速94

0m/s，质量仅4g
- **小直径**：5.7mm弹径
- **穿透机制**：
  - 弹道稳定，**动能释放率<30%**
  - 主要依靠临时空腔（组织弹性会部分恢复）
  - 设计目标：穿透防弹衣/掩体

### 4. **关键对比实验数据**
| 参数             | .45 ACP         | 5.56mm步枪弹     |
|------------------|-----------------|-----------------|
| 组织停留时间     | 450-600μs       | 80-150μs        |
| 永久空腔直径     | 8-12cm          | 3-5cm           |
| FBI弹道凝胶穿透深度 | 35-45cm         | 55-70cm         |
| 能量释放率       | >85%            | 20-40%          |

### 5. **医学机制差异**
- **.45 ACP**：大质量低速弹头造成**挤压性创伤**，直接破坏神

经丛和骨骼结构，触发痛觉休克反应
- **高穿透步枪弹**：形成狭窄通道，依赖弹头翻滚/破碎增强杀伤，但**过度穿透**导致能量浪费

### 结论
**.45手枪弹在无防护人体目标上具有更优停止力的根本原因**：
1. **质量优势**（动量p=mv更大）带来更强的机械冲击效应
2. **低穿透深度**确保能量在目标体内充分释放
3. **大截面创伤**直接破坏更多生理功能单元

> 注：此结论仅适用于**近距离无防护人体目标**。在穿甲需求场景下，高穿透步枪弹仍是更优选择。
以下是代码例子
```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. 准备数据（带噪声的线性数据）
torch.manual_seed(42)  # 确保可重复性
X = torch.linspace(0, 10, 100).view(-1, 1)  # 100个0-10之间的点
true_slope = 2.5
true_intercept = 1.0
noise = torch.randn(X.size()) * 1.5  # 添加高斯噪声
y = true_slope * X + true_intercept + noise

# 2. 定义神经网络模型（最简单的线性层）
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()


        self.linear = nn.Linear(1, 1)  # 单输入单输出的线性层
        
    def forward(self, x):
        return self.linear(x)

# 3. 初始化模型、损失函数和优化器
model = LinearRegression()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降

# 4. 训练循环
epochs = 1000
loss_history = []

for epoch in range(epochs):
    # 前向传播
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # 反向传播和优化
    optimizer.zero_grad()  # 清除历史梯度
    loss.backward()        # 计算梯度
   

 optimizer.step()       # 更新权重
    
    # 记录损失
    loss_history.append(loss.item())
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. 结果可视化
# 绘制原始数据和预测线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(X.numpy(), y.numpy(), label='Original data', alpha=0.6)
plt.plot(X.numpy(), model(X).detach().numpy(), 'r-', lw=3, label='Fitted line')
plt.title('Linear Regression Result')
plt.legend()

# 绘制损失曲线
plt.subplot(1, 2, 2)
plt.plot

(loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

plt.tight_layout()
plt.show()

# 6. 打印训练后的参数
trained_slope = model.linear.weight.item()
trained_intercept = model.linear.bias.item()
print(f"\nTrue parameters: slope={true_slope}, intercept={true_intercept}")
print(f"Trained parameters: slope={trained_slope:.4f}, intercept={trained_intercept:.4f}")
```)";

    // Parse and display markdown elements
    auto parse = wheel::parse_markdown(markdown_text);
    for (const auto &node : parse) {
        if (node.table_text) {
            std::cout << "Table: " << *node.table_text << "\n\n";
            continue;
        }
        if (node.rich_text) {
            std::cout << "Rich Text: " << *node.rich_text << "\n\n";
            continue;
        }
        if (node.code_text) {
            std::cout << "Code: " << *node.code_text << "\n\n";
            continue;
        }

        std::cout << "Text: " << node.text << "\n\n";
    }

    // Write rendered HTML to file
    std::ofstream out_fs("rendered_markdown.html");
    if (!out_fs.is_open()) {
        std::cerr << "Failed to open file for writing.\n";
        return;
    }
    for (const auto &node : parse) {
        if (node.render_html_text) {
            out_fs << *node.render_html_text << "\n";
        } else {
            out_fs << node.text << "\n";
        }
    }

    std::cout << "Rendered HTML Text has been write to " << "rendered_markdown.html" << "\n";
}

// Generate random vector data for performance testing
std::vector<float> generate_random_vector(size_t size, float min_val = -10.0f, float max_val = 10.0f) {
    std::vector<float> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    for (size_t i = 0; i < size; ++i) {
        vec[i] = dist(gen);
    }
    
    return vec;
}

// Performance test for dot product function
void test_dot_product_performance() {
    std::cout << "\n---------- Dot Product Performance Test ----------\n";
    
    // Test vector sizes
    const std::vector<size_t> sizes = {1000, 10000, 100000, 1000000};
    const int num_iterations = 100; // Number of iterations for each test
    const int num_epochs = 5;       // Number of epochs for averaging
    
    for (auto size : sizes) {
        // Generate random vectors
        auto vec_a = generate_random_vector(size);
        auto vec_b = generate_random_vector(size);
        
        // Variables to store results to prevent compiler optimization
        volatile float result_optimized = 0.0f;
        volatile float result_scalar = 0.0f;
        
        double total_duration_optimized = 0.0;
        double total_duration_scalar = 0.0;
        double min_duration_optimized = std::numeric_limits<double>::max();
        double min_duration_scalar = std::numeric_limits<double>::max();
        double max_duration_optimized = 0.0;
        double max_duration_scalar = 0.0;
        
        // Run multiple epochs for more accurate measurements
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            // Measure optimized implementation
            auto start_optimized = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_iterations; ++i) {
                result_optimized = wheel::linalg_boost::dot_product(vec_a.data(), vec_b.data(), size);
            }
            auto end_optimized = std::chrono::high_resolution_clock::now();
            auto duration_optimized = std::chrono::duration_cast<std::chrono::microseconds>(
                end_optimized - start_optimized).count() / static_cast<double>(num_iterations);
            
            total_duration_optimized += duration_optimized;
            min_duration_optimized = std::min(min_duration_optimized, duration_optimized);
            max_duration_optimized = std::max(max_duration_optimized, duration_optimized);
            
            // Measure scalar implementation
            auto start_scalar = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_iterations; ++i) {
                result_scalar = wheel::linalg_boost::detail::dot_product_scalar(vec_a.data(), vec_b.data(), size);
            }
            auto end_scalar = std::chrono::high_resolution_clock::now();
            auto duration_scalar = std::chrono::duration_cast<std::chrono::microseconds>(
                end_scalar - start_scalar).count() / static_cast<double>(num_iterations);
            
            total_duration_scalar += duration_scalar;
            min_duration_scalar = std::min(min_duration_scalar, duration_scalar);
            max_duration_scalar = std::max(max_duration_scalar, duration_scalar);
            
            // Print progress
            std::cout << "  Epoch " << (epoch + 1) << "/" << num_epochs << " completed\r" << std::flush;
        }
        
        // Calculate average durations
        double avg_duration_optimized = total_duration_optimized / num_epochs;
        double avg_duration_scalar = total_duration_scalar / num_epochs;
        
        // Calculate speedup
        double speedup = avg_duration_scalar / avg_duration_optimized;
        
        // Print results
        std::cout << "\nVector size: " << size << "\n";
        std::cout << "  Optimized implementation: " << std::fixed << std::setprecision(2) 
                  << "min = " << min_duration_optimized << " µs, "
                  << "max = " << max_duration_optimized << " µs, "
                  << "avg = " << avg_duration_optimized << " µs\n";
        std::cout << "  Scalar implementation: " << std::fixed << std::setprecision(2) 
                  << "min = " << min_duration_scalar << " µs, "
                  << "max = " << max_duration_scalar << " µs, "
                  << "avg = " << avg_duration_scalar << " µs\n";
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n\n";
    }
}

// Performance test for cosine similarity function
void test_cosine_similarity_performance() {
    std::cout << "\n---------- Cosine Similarity Performance Test ----------\n";
    
    // Test vector sizes
    const std::vector<size_t> sizes = {1000, 10000, 100000, 1000000};
    const int num_iterations = 100; // Number of iterations for each test
    const int num_epochs = 5;       // Number of epochs for averaging
    
    for (auto size : sizes) {
        // Generate random vectors
        auto vec_a = generate_random_vector(size);
        auto vec_b = generate_random_vector(size);
        
        // Variables to store results to prevent compiler optimization
        volatile float result_optimized = 0.0f;
        volatile float result_scalar = 0.0f;
        
        double total_duration_optimized = 0.0;
        double total_duration_scalar = 0.0;
        double min_duration_optimized = std::numeric_limits<double>::max();
        double min_duration_scalar = std::numeric_limits<double>::max();
        double max_duration_optimized = 0.0;
        double max_duration_scalar = 0.0;
        
        // Run multiple epochs for more accurate measurements
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            // Measure optimized implementation
            auto start_optimized = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_iterations; ++i) {
                result_optimized = wheel::linalg_boost::cosine_similarity(vec_a.data(), vec_b.data(), size);
            }
            auto end_optimized = std::chrono::high_resolution_clock::now();
            auto duration_optimized = std::chrono::duration_cast<std::chrono::microseconds>(
                end_optimized - start_optimized).count() / static_cast<double>(num_iterations);
            
            total_duration_optimized += duration_optimized;
            min_duration_optimized = std::min(min_duration_optimized, duration_optimized);
            max_duration_optimized = std::max(max_duration_optimized, duration_optimized);
            
            // Measure scalar implementation
            auto start_scalar = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_iterations; ++i) {
                result_scalar = wheel::linalg_boost::detail::cosine_similarity_scalar(vec_a.data(), vec_b.data(), size);
            }
            auto end_scalar = std::chrono::high_resolution_clock::now();
            auto duration_scalar = std::chrono::duration_cast<std::chrono::microseconds>(
                end_scalar - start_scalar).count() / static_cast<double>(num_iterations);
            
            total_duration_scalar += duration_scalar;
            min_duration_scalar = std::min(min_duration_scalar, duration_scalar);
            max_duration_scalar = std::max(max_duration_scalar, duration_scalar);
            
            // Print progress
            std::cout << "  Epoch " << (epoch + 1) << "/" << num_epochs << " completed\r" << std::flush;
        }
        
        // Calculate average durations
        double avg_duration_optimized = total_duration_optimized / num_epochs;
        double avg_duration_scalar = total_duration_scalar / num_epochs;
        
        // Calculate speedup
        double speedup = avg_duration_scalar / avg_duration_optimized;
        
        // Print results
        std::cout << "\nVector size: " << size << "\n";
        std::cout << "  Optimized implementation: " << std::fixed << std::setprecision(2) 
                  << "min = " << min_duration_optimized << " µs, "
                  << "max = " << max_duration_optimized << " µs, "
                  << "avg = " << avg_duration_optimized << " µs\n";
        std::cout << "  Scalar implementation: " << std::fixed << std::setprecision(2) 
                  << "min = " << min_duration_scalar << " µs, "
                  << "max = " << max_duration_scalar << " µs, "
                  << "avg = " << avg_duration_scalar << " µs\n";
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n\n";
    }
}

// Test function for batch cosine similarity
void test_batch_cosine_similarity() {
    std::cout << "Testing batch cosine similarity function...\n";
    bool all_passed = true;

    // Test case 1: Batch of identical vectors
    {
        const size_t vector_size = 4;
        const size_t batch_size = 3;
        
        // Create reference vector
        std::vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};
        
        // Create batch of vectors
        std::vector<std::vector<float>> batch_vectors = {
            {1.0f, 2.0f, 3.0f, 4.0f},  // Identical to reference (cosine = 1.0)
            {2.0f, 4.0f, 6.0f, 8.0f},  // Scaled version of reference (cosine = 1.0)
            {4.0f, 3.0f, 2.0f, 1.0f}   // Different vector
        };
        
        // Create array of pointers for batch API
        std::vector<const float*> a_ptrs(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            a_ptrs[i] = batch_vectors[i].data();
        }
        
        // Results array
        std::vector<float> results(batch_size, 0.0f);
        
        // Call batch function
        wheel::linalg_boost::batch_cosine_similarity(a_ptrs.data(), b.data(), vector_size, batch_size, results.data());
        
        // Expected results
        std::vector<float> expected = {
            1.0f,  // Identical vector
            1.0f,  // Scaled vector (same direction)
            0.6667f  // Approximately 0.6667 for the given vectors
        };
        
        // Check results
        for (size_t i = 0; i < batch_size; ++i) {
            std::cout << "  Case 1, vector " << i << ": result = " << results[i] 
                      << ", expected " << (i < 2 ? "exactly " : "approximately ") << expected[i] << "\n";
            
            if (i < 2 && !almost_equal(results[i], expected[i])) {
                std::cout << "  FAILED: Test case 1, vector " << i << "\n";
                all_passed = false;
            } else if (i == 2 && std::abs(results[i] - expected[i]) > 0.01f) {
                std::cout << "  FAILED: Test case 1, vector " << i << "\n";
                all_passed = false;
            }
        }
    }
    
    // Test case 2: Zero vector handling
    {
        const size_t vector_size = 3;
        const size_t batch_size = 2;
        
        // Reference vector is zero
        std::vector<float> zero_ref = {0.0f, 0.0f, 0.0f};
        
        // Batch vectors
        std::vector<std::vector<float>> batch_vectors = {
            {1.0f, 2.0f, 3.0f},  // Normal vector
            {0.0f, 0.0f, 0.0f}   // Zero vector
        };
        
        // Create array of pointers
        std::vector<const float*> a_ptrs(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            a_ptrs[i] = batch_vectors[i].data();
        }
        
        // Results array
        std::vector<float> results(batch_size, -99.0f);  // Initialize with invalid value
        
        // Call batch function with zero reference vector
        wheel::linalg_boost::batch_cosine_similarity(a_ptrs.data(), zero_ref.data(), vector_size, batch_size, results.data());
        
        // All results should be 0.0 when reference vector is zero
        for (size_t i = 0; i < batch_size; ++i) {
            std::cout << "  Case 2a, vector " << i << ": result = " << results[i] << ", expected = 0.0\n";
            if (!almost_equal(results[i], 0.0f)) {
                std::cout << "  FAILED: Test case 2a, vector " << i << "\n";
                all_passed = false;
            }
        }
        
        // Now test with normal reference vector but zero in batch
        std::vector<float> normal_ref = {1.0f, 2.0f, 3.0f};
        
        // Reset results
        std::fill(results.begin(), results.end(), -99.0f);
        
        // Call batch function
        wheel::linalg_boost::batch_cosine_similarity(a_ptrs.data(), normal_ref.data(), vector_size, batch_size, results.data());
        
        // Expected: first should be 1.0 (same vector), second should be 0.0 (zero vector)
        std::vector<float> expected = {1.0f, 0.0f};
        
        for (size_t i = 0; i < batch_size; ++i) {
            std::cout << "  Case 2b, vector " << i << ": result = " << results[i] << ", expected = " << expected[i] << "\n";
            if (!almost_equal(results[i], expected[i])) {
                std::cout << "  FAILED: Test case 2b, vector " << i << "\n";
                all_passed = false;
            }
        }
    }
    
    if (all_passed) {
        std::cout << "Batch cosine similarity tests passed!\n\n";
    } else {
        std::cout << "Some batch cosine similarity tests failed!\n\n";
    }
}

// Performance test for batch cosine similarity function
void test_batch_cosine_similarity_performance() {
    std::cout << "\n---------- Batch Cosine Similarity Performance Test ----------\n";
    
    // Test vector sizes
    const std::vector<size_t> sizes = {1000, 10000, 100000};
    const std::vector<size_t> batch_sizes = {10, 50, 100};
    const int num_iterations = 20; // Number of iterations for each test
    const int num_epochs = 5;      // Number of epochs for averaging
    
    for (auto size : sizes) {
        for (auto batch_size : batch_sizes) {
            std::cout << "\nVector size: " << size << ", Batch size: " << batch_size << "\n";
            
            // Generate reference vector
            auto ref_vec = generate_random_vector(size);
            
            // Generate batch of random vectors
            std::vector<std::vector<float>> batch_vectors;
            std::vector<const float*> batch_ptrs(batch_size);
            
            for (size_t i = 0; i < batch_size; ++i) {
                batch_vectors.push_back(generate_random_vector(size));
                batch_ptrs[i] = batch_vectors[i].data();
            }
            
            std::vector<float> batch_results(batch_size);
            std::vector<float> single_results(batch_size);
            
            // Variables for timing
            double total_duration_batch = 0.0;
            double total_duration_single = 0.0;
            double min_duration_batch = std::numeric_limits<double>::max();
            double min_duration_single = std::numeric_limits<double>::max();
            double max_duration_batch = 0.0;
            double max_duration_single = 0.0;
            
            // Run multiple epochs for more accurate measurements
            for (int epoch = 0; epoch < num_epochs; ++epoch) {
                // Measure batch implementation
                auto start_batch = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < num_iterations; ++i) {
                    wheel::linalg_boost::batch_cosine_similarity(
                        batch_ptrs.data(), ref_vec.data(), size, batch_size, batch_results.data());
                }
                auto end_batch = std::chrono::high_resolution_clock::now();
                auto duration_batch = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_batch - start_batch).count() / static_cast<double>(num_iterations);
                
                total_duration_batch += duration_batch;
                min_duration_batch = std::min(min_duration_batch, duration_batch);
                max_duration_batch = std::max(max_duration_batch, duration_batch);
                
                // Measure multiple single calls implementation
                auto start_single = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < num_iterations; ++i) {
                    for (size_t j = 0; j < batch_size; ++j) {
                        single_results[j] = wheel::linalg_boost::cosine_similarity(
                            batch_vectors[j].data(), ref_vec.data(), size);
                    }
                }
                auto end_single = std::chrono::high_resolution_clock::now();
                auto duration_single = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_single - start_single).count() / static_cast<double>(num_iterations);
                
                total_duration_single += duration_single;
                min_duration_single = std::min(min_duration_single, duration_single);
                max_duration_single = std::max(max_duration_single, duration_single);
                
                // Print progress
                std::cout << "  Epoch " << (epoch + 1) << "/" << num_epochs << " completed\r" << std::flush;
            }
            
            // Calculate average durations
            double avg_duration_batch = total_duration_batch / num_epochs;
            double avg_duration_single = total_duration_single / num_epochs;
            
            // Calculate speedup
            double speedup = avg_duration_single / avg_duration_batch;
            
            // Print results
            std::cout << "\n  Batch implementation: " << std::fixed << std::setprecision(2) 
                      << "min = " << min_duration_batch << " µs, "
                      << "max = " << max_duration_batch << " µs, "
                      << "avg = " << avg_duration_batch << " µs\n";
            std::cout << "  Multiple single calls: " << std::fixed << std::setprecision(2) 
                      << "min = " << min_duration_single << " µs, "
                      << "max = " << max_duration_single << " µs, "
                      << "avg = " << avg_duration_single << " µs\n";
            std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
            
            // Verify results match between batch and single calls
            bool results_match = true;
            for (size_t i = 0; i < batch_size; ++i) {
                if (!almost_equal(batch_results[i], single_results[i])) {
                    results_match = false;
                    break;
                }
            }
            std::cout << "  Results match: " << (results_match ? "Yes" : "No") << "\n";
        }
    }
}

// Test function for mean pooling
void test_mean_pooling() {
    std::cout << "Testing mean pooling function...\n";
    bool all_passed = true;

    // Test case 1: Simple average of two vectors
    {
        std::vector<std::vector<float>> vectors = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {5.0f, 6.0f, 7.0f, 8.0f}
        };
        
        std::vector<float> result = wheel::linalg_boost::mean_pooling(vectors);
        
        std::vector<float> expected = {3.0f, 4.0f, 5.0f, 6.0f}; // Average of the two vectors
        
        std::cout << "  Case 1: ";
        for (size_t i = 0; i < result.size(); ++i) {
            std::cout << result[i] << " ";
            if (!almost_equal(result[i], expected[i])) {
                std::cout << "\n  FAILED: Test case 1 at index " << i 
                          << ", got " << result[i] << ", expected " << expected[i] << "\n";
                all_passed = false;
            }
        }
        std::cout << "\n";
    }

    // Test case 2: Average of multiple vectors
    {
        std::vector<std::vector<float>> vectors = {
            {1.0f, 1.0f, 1.0f},
            {2.0f, 2.0f, 2.0f},
            {3.0f, 3.0f, 3.0f},
            {4.0f, 4.0f, 4.0f}
        };
        
        std::vector<float> result = wheel::linalg_boost::mean_pooling(vectors);
        
        std::vector<float> expected = {2.5f, 2.5f, 2.5f}; // (1+2+3+4)/4 = 2.5
        
        std::cout << "  Case 2: ";
        for (size_t i = 0; i < result.size(); ++i) {
            std::cout << result[i] << " ";
            if (!almost_equal(result[i], expected[i])) {
                std::cout << "\n  FAILED: Test case 2 at index " << i 
                          << ", got " << result[i] << ", expected " << expected[i] << "\n";
                all_passed = false;
            }
        }
        std::cout << "\n";
    }

    // Test case 3: Handling longer vectors (to test SIMD optimization)
    {
        const size_t vector_size = 16;
        std::vector<std::vector<float>> vectors;
        
        // Create 8 vectors with increasing values
        for (size_t k = 0; k < 8; ++k) {
            std::vector<float> vec(vector_size, static_cast<float>(k + 1));
            vectors.push_back(vec);
        }
        
        std::vector<float> result = wheel::linalg_boost::mean_pooling(vectors);
        
        // Expected: average is (1+2+3+4+5+6+7+8)/8 = 4.5
        std::vector<float> expected(vector_size, 4.5f);
        
        std::cout << "  Case 3: ";
        bool case_passed = true;
        for (size_t i = 0; i < result.size(); ++i) {
            if (!almost_equal(result[i], expected[i])) {
                std::cout << "\n  FAILED: Test case 3 at index " << i 
                          << ", got " << result[i] << ", expected " << expected[i] << "\n";
                all_passed = false;
                case_passed = false;
                break;
            }
        }
        std::cout << (case_passed ? "All elements match expected value 4.5\n" : "\n");
    }

    // Test case 4: Different values at different positions
    {
        std::vector<std::vector<float>> vectors = {
            {1.0f, 10.0f, 100.0f},
            {3.0f, 30.0f, 300.0f},
            {5.0f, 50.0f, 500.0f}
        };
        
        std::vector<float> result = wheel::linalg_boost::mean_pooling(vectors);
        
        std::vector<float> expected = {3.0f, 30.0f, 300.0f}; // Position-wise average
        
        std::cout << "  Case 4: ";
        for (size_t i = 0; i < result.size(); ++i) {
            std::cout << result[i] << " ";
            if (!almost_equal(result[i], expected[i])) {
                std::cout << "\n  FAILED: Test case 4 at index " << i 
                          << ", got " << result[i] << ", expected " << expected[i] << "\n";
                all_passed = false;
            }
        }
        std::cout << "\n";
    }

    if (all_passed) {
        std::cout << "Mean pooling tests passed!\n\n";
    } else {
        std::cout << "Some mean pooling tests failed!\n\n";
    }
}

// Performance test for mean pooling function
void test_mean_pooling_performance() {
    std::cout << "\n---------- Mean Pooling Performance Test ----------\n";
    
    // Test different vector sizes
    const std::vector<size_t> sizes = {1000, 10000, 100000};
    const std::vector<size_t> num_vectors_list = {2, 10, 50};
    const int num_iterations = 50; // Number of iterations for each test
    const int num_epochs = 5;      // Number of epochs for averaging
    
    for (auto size : sizes) {
        for (auto num_vectors : num_vectors_list) {
            std::cout << "\nVector size: " << size << ", Number of vectors: " << num_vectors << "\n";
            
            // Generate random vectors
            std::vector<std::vector<float>> vectors;
            std::vector<const float*> vec_ptrs(num_vectors);
            
            for (size_t i = 0; i < num_vectors; ++i) {
                vectors.push_back(generate_random_vector(size));
                vec_ptrs[i] = vectors[i].data();
            }
            
            // Prepare result vector for optimized implementation
            std::vector<float> result_optimized(size);
            
            // Prepare result vector for scalar implementation
            std::vector<float> result_scalar(size);
            
            // Variables for timing
            double total_duration_optimized = 0.0;
            double total_duration_scalar = 0.0;
            double min_duration_optimized = std::numeric_limits<double>::max();
            double min_duration_scalar = std::numeric_limits<double>::max();
            double max_duration_optimized = 0.0;
            double max_duration_scalar = 0.0;
            
            // Run multiple epochs for more accurate measurements
            for (int epoch = 0; epoch < num_epochs; ++epoch) {
                // Measure optimized implementation
                auto start_optimized = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < num_iterations; ++i) {
                    wheel::linalg_boost::mean_pooling(vec_ptrs.data(), size, num_vectors, result_optimized.data());
                }
                auto end_optimized = std::chrono::high_resolution_clock::now();
                auto duration_optimized = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_optimized - start_optimized).count() / static_cast<double>(num_iterations);
                
                total_duration_optimized += duration_optimized;
                min_duration_optimized = std::min(min_duration_optimized, duration_optimized);
                max_duration_optimized = std::max(max_duration_optimized, duration_optimized);
                
                // Measure scalar implementation
                auto start_scalar = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < num_iterations; ++i) {
                    wheel::linalg_boost::detail::mean_pooling_scalar(vec_ptrs.data(), size, num_vectors, result_scalar.data());
                }
                auto end_scalar = std::chrono::high_resolution_clock::now();
                auto duration_scalar = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_scalar - start_scalar).count() / static_cast<double>(num_iterations);
                
                total_duration_scalar += duration_scalar;
                min_duration_scalar = std::min(min_duration_scalar, duration_scalar);
                max_duration_scalar = std::max(max_duration_scalar, duration_scalar);
                
                // Print progress
                std::cout << "  Epoch " << (epoch + 1) << "/" << num_epochs << " completed\r" << std::flush;
            }
            
            // Calculate average durations
            double avg_duration_optimized = total_duration_optimized / num_epochs;
            double avg_duration_scalar = total_duration_scalar / num_epochs;
            
            // Calculate speedup
            double speedup = avg_duration_scalar / avg_duration_optimized;
            
            // Print results
            std::cout << "\n  Optimized implementation: " << std::fixed << std::setprecision(2) 
                      << "min = " << min_duration_optimized << " µs, "
                      << "max = " << max_duration_optimized << " µs, "
                      << "avg = " << avg_duration_optimized << " µs\n";
            std::cout << "  Scalar implementation: " << std::fixed << std::setprecision(2) 
                      << "min = " << min_duration_scalar << " µs, "
                      << "max = " << max_duration_scalar << " µs, "
                      << "avg = " << avg_duration_scalar << " µs\n";
            std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
            
            // Verify results match between optimized and scalar implementations
            bool results_match = true;
            for (size_t i = 0; i < size; ++i) {
                if (!almost_equal(result_optimized[i], result_scalar[i])) {
                    results_match = false;
                    break;
                }
            }
            std::cout << "  Results match: " << (results_match ? "Yes" : "No") << "\n";
        }
    }
}

int main() {
    // Run all tests
    test_dot_product();
    test_cosine_similarity();
    test_batch_cosine_similarity();
    test_mean_pooling();
    test_markdown_parsing();
    
    // Run performance tests
    test_dot_product_performance();
    test_cosine_similarity_performance();
    test_batch_cosine_similarity_performance();
    test_mean_pooling_performance();
    
    return 0;
}