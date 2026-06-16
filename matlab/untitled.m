% ==================== 严格按任务书：-10:2:10 dB，每类每SNR 1000个信号 ====================
clear; clc;

% 参数设置
mod_types = {'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '4FSK', '16APSK', '32APSK'};
mod_idx = 0:7;
snr_dB = -10:2:10;             
num_samples_per_snr = 1000;    
sps = 8;                       % 每个符号采样点数
symbols_per_frame = 128;       % 每帧符号数
iq_len = symbols_per_frame * sps;   % 1024
rolloff = 0.35;
filter_span = 6;
tx_filter = rcosdesign(rolloff, filter_span, sps);
filter_delay = filter_span * sps / 2;   % RRC 群延迟

rng(2026);
output_root = 'D:\xinxiduikangkechengsheji\matlab';
if ~exist(output_root, 'dir')
    mkdir(output_root);
end

% 主循环
for m = 1:length(mod_types)
    mod_type = mod_types{m};
    label = mod_idx(m);
    mod_dir = fullfile(output_root, mod_type);
    if ~exist(mod_dir, 'dir')
        mkdir(mod_dir);
    end
    
    for snr = snr_dB
        fprintf('Generating %s, SNR=%d dB ...\n', mod_type, snr);
        data_batch = zeros(num_samples_per_snr, 2*iq_len);
        labels = label * ones(num_samples_per_snr, 1);
        snr_labels = snr * ones(num_samples_per_snr, 1);
        
        for n = 1:num_samples_per_snr
            bits_per_sym = get_bits_per_symbol(mod_type);
            total_bits = symbols_per_frame * bits_per_sym;
            bits = randi([0 1], 1, total_bits);
            
            % 根据是否为 FSK 分别处理
            if ~contains(mod_type, 'FSK')
                % 线性调制：生成复符号 -> 上采样 -> 脉冲成形
                sym = modulate_symbols(bits, mod_type);
                upsampled = upsample(sym, sps);
                iq_signal = filter(tx_filter, 1, upsampled);
                iq_signal = iq_signal(filter_delay+1 : end);  % 去掉滤波器延迟
                if length(iq_signal) < iq_len
                    iq_signal = [iq_signal, zeros(1, iq_len - length(iq_signal))];
                else
                    iq_signal = iq_signal(1:iq_len);
                end
            else
                % FSK：生成频率偏移，逐符号构造复正弦波
                freq_dev = modulate_fsk_freq(bits, mod_type);
                iq_signal = [];
                t_symbol = (0:sps-1) / sps;
                for k = 1:length(freq_dev)
                    tone = exp(1j * 2*pi * freq_dev(k) * t_symbol);
                    iq_signal = [iq_signal, tone];
                end
                if length(iq_signal) < iq_len
                    iq_signal = [iq_signal, zeros(1, iq_len - length(iq_signal))];
                else
                    iq_signal = iq_signal(1:iq_len);
                end
            end
            
            % 添加 AWGN
            signal_power = mean(abs(iq_signal).^2);
            noise_power = signal_power / (10^(snr/10));
            noise = sqrt(noise_power/2) * (randn(1, iq_len) + 1j*randn(1, iq_len));
            rx_signal = iq_signal + noise;
            
            % 存储实部 + 虚部分开交错
            data_batch(n, 1:iq_len) = real(rx_signal);
            data_batch(n, iq_len+1:end) = imag(rx_signal);
        end
        
        % 保存文件
        filename = sprintf('%s_%ddB.mat', mod_type, snr);
        save(fullfile(mod_dir, filename), 'data_batch', 'labels', 'snr_labels', 'iq_len', 'mod_type', 'snr');
    end
end

fprintf('数据集生成完成！保存在 %s\n', output_root);

% ==================== 辅助函数 ====================
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
            sym = 2*idx - 1;                     % ±1，功率1
        case 'QPSK'
            I = 2*mod(idx,2) - 1;
            Q = 2*floor(idx/2) - 1;
            sym = (I + 1j*Q) / sqrt(2);          % 功率1
        case '8PSK'
            angles = idx * 2*pi/8 + pi/8;
            sym = exp(1j * angles);              % 功率1
        case '16QAM'
            sym = qammod(idx, 16, 'gray') / sqrt(10);   % 格雷码，归一化功率1
        case '64QAM'
            sym = qammod(idx, 64, 'gray') / sqrt(42);   % 格雷码，归一化功率1
        case '16APSK'
            r1 = 1; r2 = 2.2;
            angles1 = (0:3)*2*pi/4 + pi/4;
            angles2 = (0:11)*2*pi/12;
            points = [r1*exp(1j*angles1), r2*exp(1j*angles2)];
            sym = points(idx+1);
            avg_pwr = mean(abs(points).^2);
            sym = sym / sqrt(avg_pwr);            % 归一化功率1
        case '32APSK'
            r1 = 1; r2 = 2; r3 = 3;
            angles1 = (0:3)*2*pi/4 + pi/4;
            angles2 = (0:11)*2*pi/12;
            angles3 = (0:15)*2*pi/16;
            points = [r1*exp(1j*angles1), r2*exp(1j*angles2), r3*exp(1j*angles3)];
            sym = points(idx+1);
            avg_pwr = mean(abs(points).^2);
            sym = sym / sqrt(avg_pwr);            % 归一化功率1
        otherwise
            sym = zeros(1, num_syms);
    end
    sym = sym(:)';
end

function freq_dev = modulate_fsk_freq(bits, mod_type)
    bits_per_sym = get_bits_per_symbol(mod_type);
    num_syms = length(bits) / bits_per_sym;
    bits_mat = reshape(bits, bits_per_sym, num_syms)';
    idx = zeros(num_syms, 1);
    for i = 1:num_syms
        idx(i) = bin2dec_array(bits_mat(i, :));
    end
    freq_dev = (idx - 1.5) / 2;   % -0.75, -0.25, 0.25, 0.75
end