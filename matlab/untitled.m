% ==================== 生成调制识别数据集（随机SNR，宽范围）====================
clear; clc;

% 参数设置
mod_types = {'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '4FSK', '16APSK', '32APSK'};
mod_idx = 0:7;
num_samples_per_snr = 1000;   % 每种调制总样本数（不再分SNR文件）
sps = 8;
symbols_per_frame = 128;
iq_len = symbols_per_frame * sps;   % 1024
rolloff = 0.35;
filter_span = 6;
tx_filter = rcosdesign(rolloff, filter_span, sps);

% SNR 随机范围（dB）
snr_min = -6;
snr_max = 20;

rng(2026);   % 固定随机种子，保证可重复
output_root = 'D:\xinxiduikangkechengsheji\matlab';
if ~exist(output_root, 'dir')
    mkdir(output_root);
end

% ==================== 主循环 ====================
for m = 1:length(mod_types)
    mod_type = mod_types{m};
    label = mod_idx(m);
    mod_dir = fullfile(output_root, mod_type);
    if ~exist(mod_dir, 'dir')
        mkdir(mod_dir);
    end
    
    fprintf('Generating %s ...\n', mod_type);
    % 预分配
    data_batch = zeros(num_samples_per_snr, 2*iq_len);
    labels = label * ones(num_samples_per_snr, 1);
    snr_labels = zeros(num_samples_per_snr, 1);
    
    for n = 1:num_samples_per_snr
        % 随机信噪比
        snr = snr_min + (snr_max - snr_min) * rand();
        snr_labels(n) = snr;
        
        % 生成随机比特
        bits_per_sym = get_bits_per_symbol(mod_type);
        total_bits = symbols_per_frame * bits_per_sym;
        bits = randi([0 1], 1, total_bits);
        
        % 调制映射
        sym = modulate_symbols(bits, mod_type);
        sym = sym(:)';
        
        % 上采样+成型滤波
        upsampled = upsample(sym, sps);
        iq_signal = filter(tx_filter, 1, upsampled);
        iq_signal = iq_signal(:)';
        
        % 调整长度
        if length(iq_signal) < iq_len
            iq_signal = [iq_signal, zeros(1, iq_len - length(iq_signal))];
        else
            iq_signal = iq_signal(1:iq_len);
        end
        
        % 添加 AWGN
        signal_power = mean(abs(iq_signal).^2);
        noise_power = signal_power / (10^(snr/10));
        noise = sqrt(noise_power/2) * (randn(1, iq_len) + 1j*randn(1, iq_len));
        rx_signal = iq_signal + noise;
        rx_signal = rx_signal(:)';
        
        % 存储（实部 + 虚部拼接）
        data_batch(n, 1:iq_len) = real(rx_signal);
        data_batch(n, iq_len+1:end) = imag(rx_signal);
    end
    
    % 保存为一个 .mat 文件（每种调制所有样本放在一个文件中）
    filename = sprintf('%s_all.mat', mod_type);
    save(fullfile(mod_dir, filename), 'data_batch', 'labels', 'snr_labels', 'iq_len', 'mod_type', 'snr_min', 'snr_max');
end

fprintf('数据集生成完成！保存在 %s\n', output_root);

% ==================== 所有函数定义（必须放在文件末尾）====================
function bits_per_sym = get_bits_per_symbol(mod_type)
    switch mod_type
        case 'BPSK',      bits_per_sym = 1;
        case 'QPSK',      bits_per_sym = 2;
        case '8PSK',      bits_per_sym = 3;
        case '16QAM',     bits_per_sym = 4;
        case '64QAM',     bits_per_sym = 6;
        case '4FSK',      bits_per_sym = 2;
        case '16APSK',    bits_per_sym = 4;
        case '32APSK',    bits_per_sym = 5;
        otherwise,        bits_per_sym = 2;
    end
end

function dec = bin2dec_array(bits_row)
    N = length(bits_row);
    dec = bits_row * (2.^(N-1:-1:0))';
end

function sym = modulate_symbols(bits, mod_type)
    bits_per_sym = get_bits_per_symbol(mod_type);
    num_syms = length(bits) / bits_per_sym;
    bits_mat = reshape(bits, bits_per_sym, num_syms)';
    idx = zeros(num_syms, 1);
    for i = 1:num_syms
        idx(i) = bin2dec_array(bits_mat(i, :));
    end
    
    switch mod_type
        case 'BPSK'
            sym = 2*idx - 1;
        case 'QPSK'
            I = 2*rem(idx,2) - 1;
            Q = 2*rem(floor(idx/2),2) - 1;
            sym = (I + 1j*Q) / sqrt(2);
        case '8PSK'
            % 标准 8PSK，相位偏移 π/8
            angles = (0:7)*2*pi/8 + pi/8;
            sym = exp(1j * angles(idx+1));
        case '16QAM'
            const = [-3-3j, -3-1j, -3+3j, -3+1j, -1-3j, -1-1j, -1+3j, -1+1j, ...
                      3-3j,  3-1j,  3+3j,  3+1j,  1-3j,  1-1j,  1+3j,  1+1j] / sqrt(10);
            sym = const(idx+1);
        case '64QAM'
            I_vals = [-7, -5, -3, -1, 1, 3, 5, 7] / sqrt(42);
            Q_vals = I_vals;
            const = reshape(I_vals' + 1j*Q_vals, 1, 64);
            sym = const(idx+1);
        case '4FSK'
            freq_dev = (idx - 1.5) / 2;
            sym = exp(1j * 2*pi * freq_dev);
        case '16APSK'
            % 16APSK: 内环4点，外环12点
            r1 = 1; r2 = 2.2;
            angles1 = (0:3)*2*pi/4 + pi/4;
            angles2 = (0:11)*2*pi/12;
            points = [r1*exp(1j*angles1), r2*exp(1j*angles2)];
            sym = points(idx+1);
        case '32APSK'
            % 32APSK: 内环4点，中环12点，外环16点
            r1 = 1; r2 = 2; r3 = 3;
            angles1 = (0:3)*2*pi/4 + pi/4;
            angles2 = (0:11)*2*pi/12;
            angles3 = (0:15)*2*pi/16;
            points = [r1*exp(1j*angles1), r2*exp(1j*angles2), r3*exp(1j*angles3)];
            sym = points(idx+1);
        otherwise
            sym = zeros(1, num_syms);
    end
    sym = sym(:)';
end