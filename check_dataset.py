import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

mods = ['8PSK', '16QAM', '64QAM']
snr = 10

for mod in mods:
    f = rf"D:\xinxiduikangkechengsheji\matlab\{mod}\{mod}_{snr}dB.mat"
    mat = sio.loadmat(f)
    data = mat['data_batch']
    iq = data[0].reshape(2, 1024)
    I, Q = iq[0], iq[1]
    plt.figure(figsize=(5,5))
    plt.scatter(I[::10], Q[::10], s=2, alpha=0.7)
    plt.title(f"{mod} @ SNR={snr}dB")
    plt.axis('equal')
    plt.grid()
    plt.show()