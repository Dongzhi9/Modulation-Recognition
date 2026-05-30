import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from collections import Counter

data_root = r"D:\xinxiduikangkechengsheji\matlab"

# 类别名称及顺序（必须与训练时完全一致）
class_names = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '4FSK', '16APSK', '32APSK']
label_map = {name: idx for idx, name in enumerate(class_names)}

# 统计每个类别的样本总数
sample_counts = Counter()
mod_folders = [f for f in glob.glob(os.path.join(data_root, "*")) if os.path.isdir(f)]

for mod_path in mod_folders:
    mod_name = os.path.basename(mod_path)
    if mod_name not in label_map:
        print(f"⚠️ 忽略未知文件夹: {mod_name}")
        continue
    mat_files = glob.glob(os.path.join(mod_path, "*.mat"))
    total_samples = 0
    for mat_file in mat_files:
        mat = sio.loadmat(mat_file)
        data_batch = mat['data_batch']
        total_samples += data_batch.shape[0]
    sample_counts[mod_name] = total_samples

print("\n========== 各类别样本数量 ==========")
for name in class_names:
    cnt = sample_counts.get(name, 0)
    print(f"{name:10s} : {cnt:6d} 样本")

# 检查是否均衡
print("\n========== 可视化星座图 ==========")
for mod_name in class_names:
    mod_path = os.path.join(data_root, mod_name)
    if not os.path.isdir(mod_path):
        print(f"❌ 文件夹不存在: {mod_name}")
        continue
    # 选取一个 SNR 较高的文件（例如最后几个 SNR 之一）
    mat_files = sorted(glob.glob(os.path.join(mod_path, "*.mat")))
    if not mat_files:
        print(f"❌ {mod_name} 下没有 .mat 文件")
        continue
    # 选择 SNR 最大的文件（文件名含 dB，如 '8PSK_10dB.mat'）
    # 简单起见，选最后一个（按字典序 SNR 高的通常在后面）
    test_file = mat_files[-1]
    mat = sio.loadmat(test_file)
    data_batch = mat['data_batch']
    # 取第一个样本
    iq_row = data_batch[0, :]          # shape (2048,)
    iq = iq_row.reshape(2, 1024)       # (2, 1024)
    I, Q = iq[0], iq[1]
    
    # 画星座图（每10个点采样一次，避免过密）
    plt.figure(figsize=(5,5))
    plt.scatter(I[::10], Q[::10], s=2, alpha=0.7)
    plt.title(f"{mod_name} (SNR={os.path.basename(test_file)[:-4].split('_')[-1]}dB)")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{mod_name}_constellation.png", dpi=150)
    plt.show()
    print(f"✅ {mod_name} 星座图已保存为 {mod_name}_constellation.png")