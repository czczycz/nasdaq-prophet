import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 超参数设置
input_size = 2  # 每天有2个特征：成交量和收盘指数
hidden_size = 64  # LSTM 隐藏层神经元数
num_layers = 2  # LSTM 层数
output_size = 2  # 输出同样是成交量和收盘指数
seq_length = 30  # 输入序列长度
num_epochs = 100  # 训练轮数
learning_rate = 0.001


# 定义基于 LSTM 的预测模型
class NasdaqPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(NasdaqPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch_size, seq_length, input_size]
        out, (hn, cn) = self.lstm(x)
        # 取序列最后一个时刻的输出作为特征表示
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# 这里构造一个模拟数据集，实际应用中请替换为真实数据
def generate_dummy_data(num_samples=1000):
    """
    构造模拟数据：
    - 成交量采用随机波动模拟
    - 收盘指数采用随机游走模拟
    """
    data = []
    for i in range(num_samples + seq_length):
        if i == 0:
            volume = np.random.rand() * 1e6  # 初始成交量
            close = np.random.rand() * 1e4  # 初始收盘价
        else:
            # 加入一定随机噪声
            volume = data[-1][0] + np.random.randn() * 1e4
            close = data[-1][1] + np.random.randn() * 10
        data.append([volume, close])
    data = np.array(data)

    X, y = [], []
    for i in range(num_samples):
        X.append(data[i: i + seq_length])  # 过去 30 天数据
        y.append(data[i + seq_length])  # 第 31 天数据
    return np.array(X), np.array(y)


# 准备数据
X, y = generate_dummy_data(num_samples=1000)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 初始化模型、损失函数和优化器
model = NasdaqPredictor(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练过程
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X)  # 前向传播
    loss = criterion(outputs, y)

    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 模型预测：给定最近30天数据，预测下一天的成交量和收盘指数
model.eval()
with torch.no_grad():
    # 示例中直接使用数据集中的最后一段作为预测输入
    new_data = X[-1].unsqueeze(0)  # shape: [1, seq_length, input_size]
    prediction = model(new_data)
    print("预测的成交量和收盘指数:", prediction.numpy())
