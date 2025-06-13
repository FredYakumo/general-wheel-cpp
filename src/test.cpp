#include "markdown_utils.h"
#include <string>
#include <iostream>
#include <fstream>
int main() {
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

    auto parse = wheel::parse_markdown(markdown_text);
    for (const auto &node : parse) {

        // if (node.render_html_text) {
        //     std::cout << "Render HTML: " << *node.render_html_text << "\n\n";

        // }


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

    std::ofstream out_fs("rendered_markdown.html");
    if (!out_fs.is_open()) {
        std::cerr << "Failed to open file for writing.\n";
        return 1;
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