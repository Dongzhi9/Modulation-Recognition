import os
import glob
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# ==================== 配置 ====================
data_root = r"D:\xinxiduikangkechengsheji\matlab"
batch_size = 128
epochs = 60              # 增加训练轮数
lr = 0.001
num_classes = 8
iq_len = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True

TARGET_SNRS = list(range(-10, 12, 2))
class_names = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '4FSK', '16APSK', '32APSK']

# ==================== 数据集类 ====================
class SignalDataset(Dataset):
    def __init__(self, root_dir, normalize=True):
        self.samples = []
        self.normalize = normalize
        mod_folders = [f for f in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(f)]
        label_map = {'BPSK': 0, 'QPSK': 1, '8PSK': 2, '16QAM': 3, '64QAM': 4,
                     '4FSK': 5, '16APSK': 6, '32APSK': 7}
        for mod_path in mod_folders:
            mod_name = os.path.basename(mod_path)
            if mod_name not in label_map:
                print(f"Warning: unknown folder {mod_name}, skip")
                continue
            label = label_map[mod_name]
            mat_files = glob.glob(os.path.join(mod_path, "*.mat"))
            for file in mat_files:
                mat = sio.loadmat(file)
                data_batch = mat['data_batch']  # shape: (N, 2*iq_len)
                if 'snr_labels' in mat:
                    snr_arr = mat['snr_labels'].flatten()
                elif 'snr' in mat:
                    snr_arr = mat['snr'].flatten()
                else:
                    snr_arr = np.zeros(data_batch.shape[0])
                for i in range(data_batch.shape[0]):
                    iq_row = data_batch[i, :]
                    iq = iq_row.reshape(2, iq_len).astype(np.float32)
                    snr_val = snr_arr[i] if i < len(snr_arr) else snr_arr[0]
                    self.samples.append((iq, label, snr_val))
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        iq, label, snr = self.samples[idx]
        # 数据标准化：每个样本除以其幅度的RMS，使功率归一化到1
        if self.normalize:
            iq_tensor = torch.from_numpy(iq)
            rms = torch.sqrt((iq_tensor ** 2).mean())
            if rms > 1e-8:
                iq_tensor = iq_tensor / rms
        else:
            iq_tensor = torch.from_numpy(iq)
        return iq_tensor, label, snr


# ==================== 改进的多尺度深度学习网络 ====================
class ImprovedMultiScaleLSTM(nn.Module):
    """
    改进点：
    1. 每个卷积分支使用2层卷积 → 构建层次化特征，池化前有更多抽象
    2. 每个卷积独立BatchNorm
    3. 双向LSTM，使用所有时间步的均值而非仅最后一步
    4. 输入层用InstanceNorm做样本级归一化
    """
    def __init__(self, num_classes=8):
        super().__init__()

        # 输入归一化（样本级，不依赖batch统计量）
        self.input_norm = nn.InstanceNorm1d(2, affine=False)

        # ----- 卷积分支1: 细尺度 (kernel=3) -----
        self.conv1a = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm1d(32)
        self.conv1b = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm1d(64)

        # ----- 卷积分支2: 中尺度 (kernel=7) -----
        self.conv2a = nn.Conv1d(2, 32, kernel_size=7, padding=3)
        self.bn2a = nn.BatchNorm1d(32)
        self.conv2b = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2b = nn.BatchNorm1d(64)

        # ----- 卷积分支3: 粗尺度 (kernel=15) -----
        self.conv3a = nn.Conv1d(2, 32, kernel_size=15, padding=7)
        self.bn3a = nn.BatchNorm1d(32)
        self.conv3b = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn3b = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ----- LSTM路径（增强）-----
        self.down_conv = nn.Conv1d(2, 32, kernel_size=4, stride=4)
        self.bn_lstm = nn.BatchNorm1d(32)
        # 双向LSTM，2层，输出维度 = hidden_size * 2
        self.lstm = nn.LSTM(
            input_size=32, hidden_size=64, num_layers=2,
            batch_first=True, dropout=0.2, bidirectional=True
        )

        # ----- 分类器 -----
        # 3个卷积分支各64维 + LSTM 128维(64*2) = 320维
        classifier_dim = 64 * 3 + 128
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 输入归一化
        x = self.input_norm(x)

        # 卷积分支1: 细尺度
        out1 = self.relu(self.bn1a(self.conv1a(x)))
        out1 = self.global_pool(self.relu(self.bn1b(self.conv1b(out1)))).flatten(1)

        # 卷积分支2: 中尺度
        out2 = self.relu(self.bn2a(self.conv2a(x)))
        out2 = self.global_pool(self.relu(self.bn2b(self.conv2b(out2)))).flatten(1)

        # 卷积分支3: 粗尺度
        out3 = self.relu(self.bn3a(self.conv3a(x)))
        out3 = self.global_pool(self.relu(self.bn3b(self.conv3b(out3)))).flatten(1)

        # LSTM路径：下采样 → 双向LSTM → 时间步平均
        x_d = self.relu(self.bn_lstm(self.down_conv(x)))     # (B, 32, 256)
        x_lstm = x_d.permute(0, 2, 1)                        # (B, 256, 32)
        lstm_out, _ = self.lstm(x_lstm)                      # (B, 256, 128)
        lstm_feat = lstm_out.mean(dim=1)                     # (B, 128) 所有时间步平均

        # 拼接所有特征
        combined = torch.cat([out1, out2, out3, lstm_feat], dim=1)
        return self.classifier(combined)


# ==================== 辅助函数 ====================
def get_stratified_indices(labels, train_ratio=0.7, random_state=42):
    """按类别比例分层采样，保证训练/测试集各类别比例一致"""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=random_state)
    train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
    return train_idx, test_idx


def evaluate_model(model, loader, device):
    """评估模型，返回准确率、预测值和真实标签"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for iq, labels, _ in loader:
            iq, labels = iq.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast(device_type='cuda'):
                outputs = model(iq)
                _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc, np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(labels, preds, class_names, snr, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix (SNR={snr} dB)')
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    # 标注数字
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


# ==================== 主程序 ====================
def main():
    # 加载全部数据
    print("Loading dataset...")
    full_dataset = SignalDataset(data_root, normalize=True)

    # 提取所有样本的 SNR 和 label
    all_snrs = []
    all_labels = []
    for i in range(len(full_dataset)):
        _, label, s = full_dataset[i]
        all_snrs.append(int(round(s)))
        all_labels.append(label)
    all_snrs = np.array(all_snrs)
    all_labels = np.array(all_labels)

    # 存储结果
    snr_accuracies = {}
    snr_reports = {}
    snr_f1_macro = {}
    snr_f1_weighted = {}

    # ==================== 对每个 SNR 单独训练 ====================
    for target_snr in TARGET_SNRS:
        indices = np.where(all_snrs == target_snr)[0]
        if len(indices) == 0:
            print(f"SNR {target_snr} dB: 无数据，跳过")
            continue

        labels_at_snr = all_labels[indices]
        print(f"\n{'='*60}")
        print(f"Training for SNR = {target_snr} dB, samples = {len(indices)}")
        # 统计各类别样本数
        unique, counts = np.unique(labels_at_snr, return_counts=True)
        class_counts = dict(zip(unique, counts))
        print(f"Class distribution: { {class_names[int(k)]: int(v) for k, v in class_counts.items()} }")

        # 分层采样（保证每类在训练/测试集中比例一致）
        train_idx_local, test_idx_local = get_stratified_indices(
            labels_at_snr, train_ratio=0.7, random_state=42
        )
        train_indices = indices[train_idx_local]
        test_indices = indices[test_idx_local]

        train_ds = Subset(full_dataset, train_indices)
        test_ds = Subset(full_dataset, test_indices)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )

        # 初始化模型
        model = ImprovedMultiScaleLSTM(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        # 余弦退火 + 线性预热
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = GradScaler()

        best_acc = 0.0
        best_model_state = None
        no_improve_epochs = 0
        early_stop_patience = 15

        # 训练循环
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            loop = tqdm(
                train_loader,
                desc=f"SNR {target_snr:+3d}dB Epoch {epoch+1}/{epochs}",
                leave=False
            )
            for iq, labels, _ in loop:
                iq, labels = iq.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    outputs = model(iq)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                # 梯度裁剪，防止梯度爆炸
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            # 每个 epoch 后测试
            acc, _, _ = evaluate_model(model, test_loader, device)
            if acc > best_acc:
                best_acc = acc
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, "
                      f"test_acc={acc:.4f}, best={best_acc:.4f}")

            # 早停
            if no_improve_epochs >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # 保存最佳模型
        model_path = f"improved_model_snr{target_snr}.pth"
        torch.save(best_model_state, model_path)
        snr_accuracies[target_snr] = best_acc
        print(f"SNR {target_snr:3d} dB 完成 ✓ 最佳准确率 = {best_acc:.4f}  模型已保存")

        # 评估最佳模型，输出详细报告
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        acc, all_preds, all_labels_eval = evaluate_model(model, test_loader, device)

        report = classification_report(
            all_labels_eval, all_preds,
            target_names=class_names, digits=4, zero_division=0
        )
        snr_reports[target_snr] = (report, all_labels_eval, all_preds)
        print(report)

        # 绘制混淆矩阵
        cm_path = f"cm_snr{target_snr}.png"
        plot_confusion_matrix(all_labels_eval, all_preds, class_names, target_snr, cm_path)
        print(f"  混淆矩阵已保存: {cm_path}")

        # 计算 F1 分数
        f1_macro = f1_score(all_labels_eval, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels_eval, all_preds, average='weighted', zero_division=0)
        snr_f1_macro[target_snr] = f1_macro
        snr_f1_weighted[target_snr] = f1_weighted
        print(f"  F1 (macro) = {f1_macro:.4f}, F1 (weighted) = {f1_weighted:.4f}")

    # ==================== 绘制结果曲线 ====================
    snr_list = sorted(snr_accuracies.keys())
    acc_list = [snr_accuracies[s] for s in snr_list]
    plt.figure(figsize=(10, 6))
    plt.plot(snr_list, acc_list, marker='o', linewidth=2, markersize=8)
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Improved Model: Modulation Recognition Accuracy vs SNR', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    for s, a in zip(snr_list, acc_list):
        plt.text(s, a + 0.02, f'{a:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('improved_acc_vs_snr.png', dpi=150)
    plt.show()

    # ==================== 绘制 F1-score 曲线 ====================
    f1_macro_list = [snr_f1_macro[s] for s in snr_list]
    f1_weighted_list = [snr_f1_weighted[s] for s in snr_list]

    plt.figure(figsize=(12, 6))
    plt.plot(snr_list, f1_macro_list, marker='s', linewidth=2, markersize=8,
             label='Macro Avg F1', color='green')
    plt.plot(snr_list, f1_weighted_list, marker='^', linewidth=2, markersize=8,
             label='Weighted Avg F1', color='orange')
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.title('Improved Model: Modulation Recognition F1 Score vs SNR', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)
    for s, (m, w) in zip(snr_list, zip(f1_macro_list, f1_weighted_list)):
        plt.text(s, m + 0.02, f'{m:.3f}', ha='center', va='bottom', fontsize=8, color='green')
        plt.text(s, w - 0.03, f'{w:.3f}', ha='center', va='top', fontsize=8, color='orange')
    plt.tight_layout()
    plt.savefig('improved_f1_vs_snr.png', dpi=150)
    plt.show()

    # ==================== 汇总报告 ====================
    print("\n\n" + "="*60)
    print("                   FINAL SUMMARY")
    print("="*60)
    for snr in sorted(snr_accuracies.keys()):
        acc = snr_accuracies[snr]
        report, _, _ = snr_reports[snr]
        # 从报告中提取每个类别的准确率
        lines = report.strip().split('\n')
        per_class = {}
        for line in lines[2:-3]:  # 跳过表头和汇总行
            parts = line.split()
            if len(parts) >= 5:
                cls = parts[0]
                try:
                    prec = float(parts[1])
                    rec = float(parts[2])
                    f1 = float(parts[3])
                    per_class[cls] = (prec, rec, f1)
                except ValueError:
                    pass

        print(f"\nSNR = {snr:3d} dB | Overall Acc = {acc:.4f} | F1(macro) = {snr_f1_macro[snr]:.4f} | F1(weighted) = {snr_f1_weighted[snr]:.4f}")
        for cls, (p, r, f1) in per_class.items():
            marker = " ⚠️" if r < 0.1 else " ✅" if r > 0.8 else ""
            print(f"  {cls:8s}: Prec={p:.3f} Rec={r:.3f} F1={f1:.3f}{marker}")
        print(f"  Best model saved as: improved_model_snr{snr}.pth")

    print("\n✅ 所有 SNR 训练完毕！")


if __name__ == '__main__':
    main()
