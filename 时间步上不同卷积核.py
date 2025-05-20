import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BartForConditionalGeneration
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# ----------
templates = [
    "我在{地点}看到了{数量}{物体}",
    "今天{动作}了{数量}{物体}感觉{情绪}",
    "{人名}在{地点}{动作}了{时间}",
    "因为{原因}所以我很{情绪}"
]

lexicon = {
    "地点": ["北海公园", "学校", "小超市", "大客厅", "办公室"],
    "物体": ["苹果", "书", "电脑", "杯子", "衣服"],
    "动作": ["吃", "买", "读", "喝", "洗"],
    "情绪": ["高兴", "难过", "惊讶", "无聊", "兴奋"],
    "数量": ["一个", "两个", "三个", "几个", "很多"],
    "人名": ["小明", "老师", "同事", "朋友", "家人"],
    "时间": ["十分钟", "一小时", "半天", "一整天"],
    "原因": ["天气太好", "太累", "考试通过", "工作完成"]
}

def generate_meaningful_sentence():
    template = np.random.choice(templates)
    while True:
        sentence = template.format(**{k: np.random.choice(v) for k, v in lexicon.items()})
        if len(sentence) == 12:  # 确保正好12字
            return sentence

# ------------
def generate_related_eeg(sentence):
    # 语义特征编码（简单示例：按关键词类型分配不同脑区）
    features = {
        "物体": (0, 50),    # 枕叶（视觉处理）
        "动作": (50, 100),  # 运动皮层
        "情绪": (100, 150), # 前额叶
        "地点": (150, 200)  # 海马（空间记忆）
    }
    
    # 初始化256通道EEG（1秒采样率1024Hz）
    eeg = np.random.normal(0, 0.5, (256, 1024))
    
    # 根据句子成分增强特定频段（模拟语义激活）
    for word_type, (ch_start, ch_end) in features.items():
        if any(word in sentence for word in lexicon[word_type]):
            # 在相关通道添加事件相关电位（ERP）
            eeg[ch_start:ch_end, 300:700] += np.random.normal(1.0, 0.2, (ch_end-ch_start, 400))
    
    return eeg

# ------------
train_dataset = []
val_dataset = []
print("Generating training dataset...")
for _ in tqdm(range(56), desc="Training Data Progress"):
    sentence = generate_meaningful_sentence()
    eeg = generate_related_eeg(sentence)
    train_dataset.append({"sentence": sentence, "eeg": eeg})
print("Training dataset generation complete!")

for _ in range(12):
    sentence_val = generate_meaningful_sentence()
    eeg_val = generate_related_eeg(sentence_val)
    val_dataset.append({"sentence": sentence_val, "eeg": eeg_val})
with open("sentences_eeg.pkl", "wb") as f:
    pickle.dump(train_dataset, f)
with open("sentences_eeg1.pkl", "wb") as f:
    pickle.dump(val_dataset, f)

class EEGDataset(Dataset):
    def __init__(self, pkl_file, transform=None):
        """
        EEG 数据集加载器
        参数:
            pkl_file: 包含句子和脑电信号的 .pkl 文件路径
            npz_file: 包含脑电信号的 .npz 文件路径（可选）
            transform: 数据预处理或增强方法（可选）
        """
        self.transform = transform
        self.data = pd.read_pickle(pkl_file)
        self.eeg_data = [sample["eeg"] for sample in self.data]
        self.sentences = [sample["sentence"] for sample in self.data]

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.eeg_data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据
        参数:
            idx: 数据索引
        返回:
            eeg: 脑电信号张量，形状为 [256, 1024]
            sentence: 对应的句子
        """
        eeg = self.eeg_data[idx]
        sentence = self.sentences[idx]

        # 如果有预处理方法，应用到 EEG 数据
        if self.transform:
            eeg = self.transform(eeg)

        # 转换为 PyTorch 张量
        eeg = torch.tensor(eeg, dtype=torch.float32)

        return eeg, sentence

class TimeConv(nn.Module):
    def __init__(self, in_channels, kernel_size_list = [3, 5, 7]):
        """
        不同长度卷积核在时间步上进行卷积，通道数不变，最后拼接起来输入到 transformer 中
        参数:
            in_channels: 输入通道数（例如 EEG 通道数）
            kernel_size_list: 卷积核大小的列表
        """
        super(TimeConv, self).__init__()
        # 创建多个卷积层，每个卷积核只在时间步上卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,  # 输入通道数
                      out_channels=in_channels,  # 输出通道数保持不变
                      kernel_size=kernel_size,  # 卷积核大小
                      stride=1,  # 时间步上逐步滑动
                      padding=kernel_size // 2,
                      groups = in_channels) # 保证输出时间步大小不变
            for kernel_size in kernel_size_list
        ])

    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入张量，形状为 [batch_size, in_channels, time_steps]
        返回:
            融合后的张量，形状为 [batch_size, embedding_dim, time_steps]
        """
        # 对每个卷积层进行卷积操作
        conv_outputs = [conv(x) for conv in self.convs]
        for i in range(len(conv_outputs)):
            a = nn.Conv1d(256, 256, kernel_size=1)
            conv_outputs[i] = a(conv_outputs[i])
        # 将所有卷积层的输出在通道维度上拼接
        fused = torch.cat(conv_outputs, dim=1)  # [batch_size, embedding_dim(768), time_steps]
        return fused

class EEGToText(nn.Module):
    def __init__(self, in_channels, kernel_size_list, embedding_dim, bart_model_name):
        super(EEGToText, self).__init__()
        self.time_conv = TimeConv(in_channels, kernel_size_list)
        self.embedding_dim = embedding_dim
        self.projection = nn.Conv1d(in_channels * len(kernel_size_list), embedding_dim, kernel_size=1)  # 投影到 BART 的嵌入维度
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)  # 加载预训练的 BART 模型

    def forward(self, eeg, labels, attention_mask=None):
        # 1. EEG 特征提取
        eeg_features = self.time_conv(eeg)  
        eeg_embeddings = self.projection(eeg_features)  
        eeg_embeddings = eeg_embeddings.permute(0, 2, 1)  # 转换为 [batch_size, time_steps, embedding_dim]
        # 2. BART 编码-解码
        outputs = self.bart(inputs_embeds=eeg_embeddings, labels=labels, attention_mask=attention_mask)
        return outputs.loss  # [batch_size, seq_len, vocab_size]


# 测试代码
# 输入参数
batch_size = 8
in_channels = 256  # EEG 通道数
embedding_dim = 768  # BERT 嵌入维度
bart_model_name = 'fnlp/bart-base-chinese'  # BART 模型名称
kernel_size_list = [3, 5, 7]  # 不同的卷积核大小
# 数据文件路径
pkl_file = "sentences_eeg.pkl"
pkl_file1 = "sentences_eeg1.pkl"
# 创建数据集
dataset = EEGDataset(pkl_file)
val_dataset = EEGDataset(pkl_file1)
# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
dataloader1 = DataLoader(val_dataset, batch_size=8, shuffle=True)
model = EEGToText(in_channels, kernel_size_list, embedding_dim, bart_model_name)
torch.save(model, 'full_model.pth')
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(bart_model_name)
# 测试 DataLoader
for epoch in range(1):
    for batch_idx, (eeg_batch, sentence_batch) in enumerate(dataloader):
        decoder_input_ids = tokenizer(sentence_batch, return_tensors="pt", padding=True, truncation=False).input_ids
        loss = model(eeg_batch, decoder_input_ids)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss.backward()
        optimizer.step()
        print(f"Batch {batch_idx + 1}, Loss: {loss.item()}")
with torch.no_grad():
    for batch_idx, (egg_batch, sentence_batch) in enumerate(dataloader1):
        eeg_embeddings = model.time_conv(egg_batch)
        eeg_embeddings = model.projection(eeg_embeddings)
        eeg_embeddings = eeg_embeddings.permute(0, 2, 1)
        encoder_outputs = model.bart.model.encoder(inputs_embeds=eeg_embeddings)
        generated_ids = model.bart.generate(
            encoder_outputs=encoder_outputs,
            max_length=14,  # 设置生成的最大长度
            num_beams=5,  # 使用 Beam Search
            early_stopping=True
        )
        for i in range(len(sentence_batch)):
            predicted_sentences = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predicted_sentences[i] = predicted_sentences[i].replace(" ", "")
            print(f"Predicted sentences:\n{predicted_sentences[i]}")
            print(f"True sentences:\n{sentence_batch[i]}")