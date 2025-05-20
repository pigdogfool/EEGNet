import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import gamma
import librosa
import h5py
import soundfile as sf
from tqdm import tqdm

# 参数设置
num_samples = 1024          # 音频数量
duration = 2.0             # 信号时长(秒)
fs_audio = 16000           # 音频采样率
fs_sEEG = 1000             # sEEG采样率
n_sEEG_channels = 256      # sEEG通道数
tone_labels = [0, 1, 2, 3]  # 汉语四声

def generate_audio(tone_type, duration=duration, fs=fs_audio):
    """生成带音调特征的合成元音音频"""
    t = np.linspace(0, duration, int(fs * duration))
    
    # 基础频率曲线（根据音调变化）
    if tone_type == 0:   # 高平调
        f0 = 220 * np.ones_like(t)  
    elif tone_type == 1: # 升调
        f0 = 120 + 100 * t / duration
    elif tone_type == 2: # 降升调
        f0 = 180 - 80 * np.sin(2 * np.pi * 0.5 * t / duration)
    else:                      # 降调
        f0 = 220 - 100 * t / duration
    
    # 生成元音（/a/的共振峰结构）
    formants = {
        'F1': 700 + 50*np.random.randn(), 
        'F2': 1200 + 100*np.random.randn(),
        'F3': 2500 + 150*np.random.randn()
    }
    
    # 源-滤波器模型合成音频
    harmonic = 0.5 * np.sin(2 * np.pi * np.cumsum(f0) / fs)
    bandwidth = 100  # 带宽
    for f in formants.values():
        harmonic = librosa.effects.preemphasis(harmonic, coef=0.98)
        low = (f - bandwidth) / (fs / 2)
        high = (f + bandwidth) / (fs / 2)
        b, a = signal.butter(4, [low, high], 'bandpass')
        harmonic = signal.lfilter(b, a, harmonic)
    
    # 添加噪声和包络
    audio = harmonic * np.exp(-0.5 * t) + 0.01 * np.random.randn(len(t))
    return audio / np.max(np.abs(audio)), formants

# 生成所有数据
audio_signals = []
audio_features = []
tone_labels = []  # 修改点2：直接存储数值标签

for i in range(num_samples):
    tone = i % 4  # 修改点3：直接循环0-3
    audio, formants = generate_audio(tone)
    audio_signals.append(audio)
    audio_features.append([formants['F1'], formants['F2'], formants['F3']])
    tone_labels.append(tone)  # 直接添加数值

audio_signals = np.array(audio_signals)
audio_features = np.array(audio_features)
tone_labels = np.array(tone_labels)  # 修改点4：直接转为numpy数组


def generate_sEEG(formants, tone_type, fs=fs_sEEG):
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    sEEG = np.zeros((n_sEEG_channels, n_samples))
    
    # 根据数值标签选择调制模式
    if tone_type == 0:   # tone1
        mod = 40 * np.ones_like(t)
    elif tone_type == 1: # tone2
        mod = 30 + 20 * t
    elif tone_type == 2: # tone3
        mod = 50 * np.sin(2*np.pi*2*t) + 40
    else:                # tone4
        mod = 60 - 20 * t
    
    for ch in range(n_sEEG_channels):
        gamma = 0.2 * np.sin(2 * np.pi * mod * t)
        pac = 0.5 * np.sin(2*np.pi*formants[0]/100 * t) * gamma
        sEEG[ch] = pac + 0.1 * np.random.randn(n_samples)
    
    return sEEG

# 生成sEEG
sEEG_signals = np.stack([generate_sEEG(audio_features[i], tone_labels[i]) 
                      for i in range(num_samples)])

# 保存数据（不再需要编码步骤）
with h5py.File('tone_sEEG_dataset.h5', 'w') as f:
    f.create_dataset('audio', data=audio_signals)
    f.create_dataset('sEEG', data=sEEG_signals)
    f.create_dataset('audio_features', data=audio_features)
    f.create_dataset('tone_labels', data=tone_labels)  # 直接保存数值
    f.attrs['tone_mapping'] = '0:tone1, 1:tone2, 2:tone3, 3:tone4'  # 

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# 1. 定义自定义Dataset类
class AudioEEGDataset(Dataset):
    def __init__(self, h5_path, mode='train', transform=None):
        """
        参数:
            h5_path: HDF5文件路径
            mode: train/val/test模式
            transform: 数据增强变换
        """
        self.h5_path = h5_path
        self.mode = mode
        self.transform = transform
        
        # 读取HDF5元数据
        with h5py.File(h5_path, 'r') as f:
            self.total_samples = len(f['sEEG'])
            self.sEEG_shape = f['sEEG'].shape[1:]
            self.audio_shape = f['audio'].shape[1:]
        
        # 划分训练/验证/测试集 (8:1:1)
        indices = np.arange(self.total_samples)
        train_idx = int(0.8 * self.total_samples)
        val_idx = int(0.9 * self.total_samples)
        
        if mode == 'train':
            self.indices = indices[:train_idx]
        elif mode == 'val':
            self.indices = indices[train_idx:val_idx]
        else:  # test
            self.indices = indices[val_idx:]
        
        # 初始化标准化器
        self.sEEG_scaler = StandardScaler()
        self.audio_scaler = StandardScaler()
        
        if mode == 'train':
            self._init_scalers()
        else:
            # 加载训练集的标准化参数
            self.sEEG_scaler.mean_ = np.load('sEEG_scaler_mean.npy')
            self.sEEG_scaler.scale_ = np.load('sEEG_scaler_scale.npy')
            self.audio_scaler.mean_ = np.load('audio_scaler_mean.npy')
            self.audio_scaler.scale_ = np.load('audio_scaler_scale.npy')

    def _init_scalers(self):
        """用训练集数据初始化标准化参数"""
        if self.mode != 'train':
            return
            
        with h5py.File(self.h5_path, 'r') as f:
            sEEG_data = f['sEEG'][self.indices]
            audio_data = f['audio'][self.indices]
        
        # 计算sEEG各通道的均值和方差
        self.sEEG_scaler.fit(sEEG_data.reshape(-1, self.sEEG_shape[0]))
        self.audio_scaler.fit(audio_data.reshape(-1, self.audio_shape[0]))
        
        # 保存标准化参数
        np.save('sEEG_scaler_mean.npy', self.sEEG_scaler.mean_)
        np.save('sEEG_scaler_scale.npy', self.sEEG_scaler.scale_)
        np.save('audio_scaler_mean.npy', self.audio_scaler.mean_)
        np.save('audio_scaler_scale.npy', self.audio_scaler.scale_)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            real_idx = self.indices[idx]
            sEEG = f['sEEG'][real_idx]  # (channels, time)
            features = f['audio_features'][real_idx]  # (3,)
            label = f['tone_labels'][real_idx]  # scalar
            # 标准化处理
        sEEG = self.sEEG_scaler.transform(sEEG.T).T  # 各通道单独标准化
        
        # 转换为torch张量
        sEEG = torch.FloatTensor(sEEG)  # (channels, time)
        features = torch.FloatTensor(features)  # (3,)
        label = torch.LongTensor([label])  # (1,)
        
        # 数据增强
        if self.transform:
            sEEG = self.transform(sEEG)
            
        return {
            'sEEG': sEEG,
            'features': features,
            'label': label.squeeze()
        }

# 2. 加载数据集
dataset = AudioEEGDataset('tone_sEEG_dataset.h5')

# 3. 创建数据加载器
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)    

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

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(256, 21))

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
        # x = self.head(x)
        return x
        
class Tone(nn.Module):
    def __init__(self):
        super(Tone, self).__init__()
        self.v = 1 #自由度
        self.cnn = CNNBackBone(kernal_size=7, stride=2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(256, 4))
    
    def _pairwise_euclidean(self, x):
        """
        计算成对欧式距离
        """
        if x.dim() == 3:
            batch_size, channels, time_steps = x.size()
            # 展平为 [batch_size, channels * time_steps]
            x_flat = x.view(batch_size, -1)
        elif x.dim() == 2:
            x_flat = x  # 如果已经是二维的，直接使用
        else:
            raise ValueError(f"Unexpected input dimensions: {x.dim()}")

        # 计算平方和
        x_norm = torch.sum(x_flat ** 2, dim=1, keepdim=True)  # [batch_size, 1]

        # 计算成对距离的平方
        dist = x_norm + x_norm.t() - 2 * torch.mm(x_flat, x_flat.t())  # [batch_size, batch_size]

        # 取平方根并避免数值问题
        dist = torch.sqrt(torch.clamp(dist, min=1e-6))  # [batch_size, batch_size]

        return dist
    
    def _student_t_similarity(self, D):
        """学生t分布相似性转换"""
        nu = self.v
        coeff = gamma((nu + 1) / 2) / (np.sqrt(nu * np.pi) * gamma(nu / 2))
        S = coeff * (1 + D ** 2 / nu) ** (-(nu + 1) / 2)
        return S
    
    def forward(self,x ,f):
        CNNFeature = self.cnn(x)
        D_n = self._pairwise_euclidean(CNNFeature)
        D_f = self._pairwise_euclidean(f)
        S_n = self._student_t_similarity(D_n)  # 神经相似性
        S_f = self._student_t_similarity(D_f)  # 声学相似性
        
        # 3. 计算KL散度损失
        loss_nar = -torch.mean(
            S_f * torch.log(S_n + 1e-6) + (1 - S_f) * torch.log(1 - S_n + 1e-6)
        )
        return loss_nar, self.classifier(CNNFeature)
model = Tone()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for epoch in range(10):
    for batch in dataloader:
        sEEG = batch['sEEG']
        features = batch['features']
        label = batch['label']
        loss, pred = model(sEEG, features)
        loss = 0.2*loss + criterion(pred, label)
    print(f"Epoch {epoch+1}/{10}, Loss: {loss.item():.4f}")

dataloader = DataLoader(AudioEEGDataset('tone_sEEG_dataset.h5', mode='val'), batch_size=batch_size, shuffle=False, num_workers=0)
with torch.no_grad():
    for batch in dataloader:
        sEEG = batch['sEEG']
        features = batch['features']
        label = batch['label']
        loss, pred = model(sEEG, features)
        pred_label = torch.argmax(pred, dim=1)
        acc = (pred_label == label).float().mean()
    print(f"Test Accuracy: {acc.item():.4f}")