import os
import glob
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler        # 新版导入，解决弃用警告
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# ==================== 配置 ====================
data_root = r"D:\xinxiduikangkechengsheji\matlab"
batch_size = 128
epochs = 30
lr = 0.001
num_classes = 8
iq_len = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True

# 需要训练的 SNR 列表：-10 ~ 10 dB，步长2
TARGET_SNRS = list(range(-10, 12, 2))

# ==================== 数据集类 ====================
class SignalDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        mod_folders = [f for f in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(f)]
        label_map = {'BPSK':0, 'QPSK':1, '8PSK':2, '16QAM':3, '64QAM':4,
                     '4FSK':5, '16APSK':6, '32APSK':7}
        for mod_path in mod_folders:
            mod_name = os.path.basename(mod_path)
            if mod_name not in label_map:
                print(f"Warning: unknown folder {mod_name}, skip")
                continue
            label = label_map[mod_name]
            mat_files = glob.glob(os.path.join(mod_path, "*.mat"))
            for file in mat_files:
                mat = sio.loadmat(file)
                data_batch = mat['data_batch']   # shape: (N, 2*iq_len)
                # 读取信噪比信息
                if 'snr_labels' in mat:
                    snr_arr = mat['snr_labels'].flatten()
                elif 'snr' in mat:
                    snr_arr = mat['snr'].flatten()
                else:
                    snr_arr = np.zeros(data_batch.shape[0])
                for i in range(data_batch.shape[0]):
                    iq_row = data_batch[i, :]
                    iq = iq_row.reshape(2, iq_len).astype(np.float32)
                    iq_tensor = torch.from_numpy(iq)
                    snr_val = snr_arr[i] if i < len(snr_arr) else snr_arr[0]
                    self.samples.append((iq_tensor, label, snr_val))
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx][0], self.samples[idx][1], self.samples[idx][2]

# ==================== 轻量化多尺度卷积+LSTM网络 ====================
class LightMultiScaleLSTM(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(2, 32, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(2, 32, kernel_size=15, padding=7)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.down_conv = nn.Conv1d(2, 16, kernel_size=4, stride=4)
        self.lstm = nn.LSTM(input_size=16, hidden_size=64, num_layers=1,
                            batch_first=True, dropout=0, bidirectional=False)

        concat_dim = 32*3 + 64
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 多尺度特征
        out1 = self.global_pool(self.relu(self.bn(self.conv1(x)))).flatten(1)
        out2 = self.global_pool(self.relu(self.bn(self.conv2(x)))).flatten(1)
        out3 = self.global_pool(self.relu(self.bn(self.conv3(x)))).flatten(1)
        multi_scale = torch.cat([out1, out2, out3], dim=1)

        # 时序特征
        x_d = self.down_conv(x)
        x_lstm = x_d.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_lstm)
        lstm_feat = lstm_out[:, -1, :]

        combined = torch.cat([multi_scale, lstm_feat], dim=1)
        out = self.classifier(combined)
        return out

# ==================== 加载全部数据并按 SNR 分组 ====================
full_dataset = SignalDataset(data_root)

# 提取所有样本的 SNR（四舍五入到整数，方便匹配）
all_snrs = []
for i in range(len(full_dataset)):
    _, _, s = full_dataset[i]
    all_snrs.append(int(round(s)))   # 转为整数 dB
all_snrs = np.array(all_snrs)

# 存储每个 SNR 的最终测试准确率
snr_accuracies = {}

# ==================== 对每个 SNR 单独训练 ====================
for target_snr in TARGET_SNRS:
    # 筛选当前 SNR 的样本索引
    indices = np.where(all_snrs == target_snr)[0]
    if len(indices) == 0:
        print(f"SNR {target_snr} dB: 无数据，跳过")
        continue
    print(f"\n{'='*50}\nTraining for SNR = {target_snr} dB, samples = {len(indices)}\n{'='*50}")

    # 创建子集并划分训练/测试集
    snr_subset = torch.utils.data.Subset(full_dataset, indices)
    train_len = int(0.7 * len(snr_subset))
    test_len = len(snr_subset) - train_len
    train_ds, test_ds = random_split(snr_subset, [train_len, test_len],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # 初始化模型、损失函数、优化器
    model = LightMultiScaleLSTM(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    best_acc = 0.0
    best_model_state = None

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"SNR {target_snr} Epoch {epoch+1}/{epochs}", leave=False)
        for iq, labels, _ in loop:
            iq, labels = iq.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):   # 新版写法
                outputs = model(iq)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # 每个 epoch 后测试
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for iq, labels, _ in test_loader:
                iq, labels = iq.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast(device_type='cuda'):
                    outputs = model(iq)
                    _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        if acc > best_acc:
            best_acc = acc
            # 深拷贝模型参数，避免后续更新影响
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

    # 保存当前 SNR 的最佳模型
    torch.save(best_model_state, f"best_model_snr{target_snr}.pth")
    snr_accuracies[target_snr] = best_acc
    print(f"SNR {target_snr} dB 训练完成，最佳测试准确率 = {best_acc:.4f}")

    # 额外输出分类报告
    model.load_state_dict(torch.load(f"best_model_snr{target_snr}.pth", map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for iq, labels, _ in test_loader:
            iq = iq.to(device)
            with autocast(device_type='cuda'):
                outputs = model(iq)
                _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    class_names = ['BPSK','QPSK','8PSK','16QAM','64QAM','4FSK','16APSK','32APSK']
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))

# ==================== 绘制准确率随 SNR 变化曲线 ====================
snr_list = sorted(snr_accuracies.keys())
acc_list = [snr_accuracies[s] for s in snr_list]
plt.figure()
plt.plot(snr_list, acc_list, marker='o')
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy')
plt.title('Modulation Recognition Accuracy vs SNR (Separate Training)')
plt.grid(True)
plt.savefig('acc_vs_snr_separate.png')
plt.show()

print("\n所有 SNR 训练完毕！准确率曲线已保存为 acc_vs_snr_separate.png")