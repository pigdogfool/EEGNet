# 创建虚拟输入数据和目标数据
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset

def generate_realistic_sEEG(num_samples=1628, num_channels=256, time_steps=1024, num_classes=21, sr=1000):
    """
    生成模拟真实句子阅读任务的sEEG数据
    参数:
        num_samples: 总样本数 (4*407)
        num_channels: 电极通道数 (256)
        time_steps: 时间步长 (1024)
        num_classes: 音节类别数 (10)
        sr: 采样率 (Hz)
    返回:
        x: sEEG数据 [num_samples, num_channels, time_steps]
        y: 标签 [num_samples]
    """
    # 基础参数
    duration = time_steps / sr  # 信号时长(秒)
    t = torch.linspace(0, duration, time_steps)

    # 为每个类别定义不同的神经活动模式
    class_patterns = []
    for c in range(num_classes):
        # 1. 生成类别特定的时频模式
        freq = 5 + c * 2  # 基础频率随类别变化(5-25Hz)
        erp = 0.5 * torch.sin(2 * np.pi * freq * t)  # ERP成分
        erp *= torch.exp(-(t - 0.3)**2 / 0.05)      # 在300ms附近增强

        # 2. 添加高频Gamma活动（语言处理相关）
        gamma = 0.3 * torch.randn(time_steps)
        gamma *= (t > 0.2) & (t < 0.8)  # 在200-800ms出现

        # 3. 组合模式
        class_patterns.append(erp + gamma)

    # 生成每个样本
    x = torch.zeros(num_samples, num_channels, time_steps)
    y = torch.randint(0, num_classes, (num_samples,))

    for i in range(num_samples):
        c = y[i]
        base_pattern = class_patterns[c]

        # 4. 空间传播模拟（左侧语言区更强）
        for ch in range(num_channels):
            # 通道权重：假设前50个通道在左侧语言区
            if ch < 50:
                weight = 1.0 + 0.5 * torch.rand(1)  # 左侧增强
                delay = torch.rand(1) * 0.02         # 0-20ms传导延迟
            else:
                weight = 0.2 + 0.3 * torch.rand(1)   # 其他区域弱激活
                delay = torch.rand(1) * 0.05

            # 应用通道特异性调制
            delay_samples = int(delay * sr)
            shifted_pattern = torch.roll(base_pattern, delay_samples)
            x[i, ch] = weight * shifted_pattern + 0.1 * torch.randn(time_steps)  # 添加噪声

    # 全局标准化
    x = (x - x.mean()) / x.std()
    return x, y

# 生成数据
def generate_attributes(x):
    """
    x: 输入sEEG信号 [batch_size, 256, 1024]
    返回: 
    - y_poa: [batch_size] (0-5)
    - y_moa: [batch_size] (0-4)
    - y_asp: [batch_size] (0-1)
    - y_dev: [batch_size] (0-1)
    """
    if x.dim() == 2:  # 如果是单个样本
        x = x.unsqueeze(0)  # 增加一个批量维度，形状变为 [1, 256, 1024]
    # 1. POA由前10个通道的均值决定
    poa_feat = x[:, :10, :].mean(dim=(1, 2))  # [batch_size]
    y_poa = (poa_feat * 6).long() % 6  # 映射到0-5

    # 2. MOA由中间50-60通道的方差决定
    moa_feat = x[:, 50:60, :].var(dim=2).mean(dim=1)  # [batch_size]
    y_moa = (moa_feat * 5).long() % 5  # 映射到0-4

    # 3. 送气由高频能量（100-150通道）决定
    hf_energy = x[:, 100:150, :].abs().mean(dim=(1, 2))
    y_asp = (hf_energy > hf_energy.median()).long()  # 二分类

    # 4. 清浊由低频能量（0-50通道）决定
    lf_energy = x[:, :50, :].abs().mean(dim=(1, 2))
    y_dev = (lf_energy > lf_energy.median()).long()  # 二分类
    attr = [y_poa, y_moa, y_asp, y_dev]
    return attr

x, y = generate_realistic_sEEG()
x_test, y_test = generate_realistic_sEEG(num_samples=100)

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], generate_attributes(self.x[idx])
# 实例化数据集
dataset = MyDataset(x, y)
test_dataset = MyDataset(x_test, y_test)
# 实例化 DataLoader
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=0)
# 定义卷积层
class CNNBackBone(nn.Module):

    def __init__(self, kernal_size, stride):
        super(CNNBackBone, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(256,
                      64,
                      kernel_size=kernal_size,
                      stride=stride,
                      padding=kernal_size // 2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            self._make_block(64,
                             128,
                             kernel_size=7,
                             stride=1,
                             pool_size=3,
                             pool_stride=2),  # 块1
            self._make_block(128,
                             128,
                             kernel_size=5,
                             stride=1,
                             pool_size=3,
                             pool_stride=2),  # 块2
            self._make_block(128,
                             256,
                             kernel_size=3,
                             stride=1,
                             pool_size=3,
                             pool_stride=2),  # 块3
            self._make_block(256,
                             256,
                             kernel_size=3,
                             stride=1,
                             pool_size=3,
                             pool_stride=2)  # 块4
        )


    def _make_block(self, in_channels, out_channels, kernel_size, stride,
                    pool_size, pool_stride):
        return nn.Sequential(
            nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=kernel_size // 2), nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size,
                         stride=pool_stride,
                         padding=pool_size // 2))

    def forward(self, x):
        # x shape: [batch_size, 256, 1024]
        x = self.stem(x)
        x = self.blocks(x)
        return x

class AAIModule(nn.Module):
    def __init__(self, c_f=256, d_poa=6, d_moa=5, d_asp=2, d_dev=2):
        """
        Args:
            c_f: 输入特征的隐藏维度 (来自CNN backbone的z的维度)
            d_poa: POA属性的类别数 (如6类: 双唇、唇齿、齿龈等)
            d_moa: MOA属性的类别数 (如5类: 爆破音、塞擦音等)
            d_asp: 送气属性的类别数 (如2类: 送气/不送气)
            d_dev: 清浊属性的类别数 (如2类: 清音/浊音)
        """
        super().__init__()
        self.d_poa, self.d_moa, self.d_asp, self.d_dev = d_poa, d_moa, d_asp, d_dev
        self.d_total = d_poa + d_moa + d_asp + d_dev

        # 初始化可学习的原型矩阵
        self.W_poa = nn.Parameter(torch.randn(c_f, d_poa))
        self.W_moa = nn.Parameter(torch.randn(c_f, d_moa))
        self.W_asp = nn.Parameter(torch.randn(c_f, d_asp))
        self.W_dev = nn.Parameter(torch.randn(c_f, d_dev))

        # 初始化邻接矩阵A (根据语音学规则)
        self.A = self._init_adjacency_matrix()

        # 预测头
        self.init_mlp = nn.Sequential(
            nn.Linear(c_f, 256),
            nn.ReLU(),
            nn.Linear(256, 21)  # 21个声母类别
        )

    def _init_adjacency_matrix(self):
        """初始化邻接矩阵A，定义属性间的关系"""
        A = torch.zeros(self.d_total, self.d_total)

        # 假设某些属性共存（需根据实际语音学规则填充）
        # 例如：双唇(b) + 爆破音(p) + 不送气 + 清音 可能共存
        A[:self.d_poa, self.d_poa:self.d_poa+self.d_moa] = 1  # POA与MOA关联
        A[self.d_poa:self.d_poa+self.d_moa, -self.d_dev:] = 1  # MOA与清浊关联
        A = A + A.T  # 对称化
        A = A + torch.eye(self.d_total)  # 添加自环
        return nn.Parameter(A, requires_grad=True)  # 允许训练中更新

    def forward(self, z):
        """
        Args:
            z: CNN提取的特征 [batch_size, c_f, T] → 展平后 [batch_size, c_f]
        Returns:
            attr_logits: 属性分类logits [batch_size, d_total]
            init_logits: 声母分类logits [batch_size, 21]
        """

        # 计算属性空间表示
        W = torch.cat([self.W_poa, self.W_moa, self.W_asp, self.W_dev], dim=1)  # [c_f, d_total]
        W_hat = F.relu(W @ self.A)
        # 属性分类
        attr_logits = z @ W_hat

        # 声母分类
        init_logits = self.init_mlp(z)

        return attr_logits, init_logits

class SeegModel(nn.Module):
    """整合CNN和AAI的端到端模型"""

    def __init__(self):
        super().__init__()
        self.cnn = CNNBackBone(kernal_size=7, stride=2)  # 卷积神经网络

    def forward(self, x):
        features = self.cnn(x)  # CNN提取特征
        features = features.view(features.size(0), -1)
        mlp = nn.Sequential(
            nn.Linear(features.shape[1], 256),
            nn.Linear(256, 256)
        )
        features = mlp(features)  # MLP处理特征
        self.aai = AAIModule()  # 输入维度与CNN最后一层匹配
        attr, initial = self.aai(features)  # AAI预测
        return attr, initial
# 定义均方误差损失函数和随机梯度下降优化器
model = SeegModel()  # 实例化模型
if __name__ == "__main__":
    # model = CNNBackBone(kernal_size=7, stride=2)  # 实例化模型
    criterion = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # 学习率为0.01
    attr_index = [6,5,2,2]
    # 执行梯度下降算法进行模型训练
    for epoch in range(9):  # 迭代9次
        # 前向传播，计算预测值
        total_loss = 0
        for batch_idx, (data, initial, attr) in enumerate(dataloader):
            attr_pre, initial_pre = model(data)  # 计算损失
            loss_initial = criterion(initial_pre, initial)
            loss_attr = 0
            for i in range(4):
                loss_attr += criterion(attr_pre[:, :attr_index[i]], attr[i].squeeze(-1))
                attr_pre = attr_pre[:, attr_index[i]:]
            loss = 0.5 * loss_initial + loss_attr
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/50], Loss: {total_loss / len(dataloader):.4f}")
    # for i in range(13):
    #     for batch_idx, (data, initial, attr) in enumerate(dataloader):
    #         y = model(data)
    #         loss = criterion(y, initial)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     print(f"Epoch [{i+1}/50], Loss: {loss.item():.4f}")
    model.eval()  # 设置为评估模式
    initial_correct = 0
    attr_correct = 0

    with torch.no_grad():  # 评估时不需要计算梯度
        for batch_idx, (data, initial, attr) in enumerate(test_loader):
            attr_pre, initial_pre = model(data)  # 计算损失
            initial_correct += (torch.argmax(initial_pre, dim=1) == initial).sum().item()
            for i in range(4):
                attr_correct+=(torch.argmax(attr_pre[:, :attr_index[i]], dim=1) == attr[i].squeeze(-1)).sum().item()
                attr_pre = attr_pre[:, attr_index[i]:]
    # with torch.no_grad():
    #     for batch_idx, (data, initial, attr) in enumerate(test_loader):
    #         initial_pre = model(data)
    #         initial_correct += (torch.argmax(initial_pre, dim=1) == initial).sum().item()
            
    # initial_accuracy = 100 * initial_correct / len(test_loader.dataset)
    # # attr_accuracy = attr_correct / (400 * len(test_loader.dataset))
    # print(f"Initial Test Accuracy: {initial_accuracy:.2f}%")
    # # print(f"Attr Test Accuracy: {attr_accuracy:.2f}%")
