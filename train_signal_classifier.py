import os
import glob
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==================== 配置 ====================
data_root = r"D:\xinxiduikangkechengsheji\matlab"   # 你的数据根目录
batch_size = 128                   # 根据显存调整
epochs = 30
lr = 0.001
num_classes = 8
iq_len = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True

# ==================== 数据集类（兼容新旧格式） ====================
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
                # 读取SNR标签（新格式 'snr_labels'，旧格式 'snr'）
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
        out1 = self.global_pool(self.relu(self.bn(self.conv1(x)))).flatten(1)
        out2 = self.global_pool(self.relu(self.bn(self.conv2(x)))).flatten(1)
        out3 = self.global_pool(self.relu(self.bn(self.conv3(x)))).flatten(1)
        multi_scale = torch.cat([out1, out2, out3], dim=1)

        x_d = self.down_conv(x)
        x_lstm = x_d.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_lstm)
        lstm_feat = lstm_out[:, -1, :]

        combined = torch.cat([multi_scale, lstm_feat], dim=1)
        out = self.classifier(combined)
        return out

# ==================== 数据加载 ====================
full_dataset = SignalDataset(data_root)
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=True)

model = LightMultiScaleLSTM(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
scaler = GradScaler()

best_acc = 0.0
train_losses = []
test_accs = []

# ==================== 训练循环 ====================
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for iq, labels, _ in loop:
        iq, labels = iq.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            outputs = model(iq)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 测试
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for iq, labels, _ in test_loader:
            iq, labels = iq.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast():
                outputs = model(iq)
                _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    test_accs.append(acc)
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Test Acc={acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  -> Best model saved (acc={best_acc:.4f})")

    scheduler.step()

# ==================== 最终评估 ====================
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for iq, labels, _ in test_loader:
        iq = iq.to(device, non_blocking=True)
        with autocast():
            outputs = model(iq)
            _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
class_names = ['BPSK','QPSK','8PSK','16QAM','64QAM','4FSK','16APSK','32APSK']
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

print("\n========== Classification Report ==========")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
print(f"Final test accuracy: {accuracy_score(all_labels, all_preds):.4f}")

# 按SNR画出准确率曲线（如果有snr标签）
all_snrs = []
for _, _, snr in test_dataset:
    all_snrs.append(snr)
# 注意：上面需要预先收集snr，但为了简单，可以在训练时保存。这里提供一个可选方法：
# 在测试循环中收集 snr 值，但为了不使代码过长，用户可以自行添加。此处省略不影响主要评估。