import os
import glob
import random
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# =================配置参数=================
SAMPLING_RATE = 1000  # Hz
NOTCH_FREQ = 50  # 工频干扰 Hz
NOTCH_QUALITY = 30  # 陷波器质量因子
DATE_LABEL = "20251217"
OUTPUT_ROOT_NAME = f"{DATE_LABEL}_LFP Analysis"

# 时频分析参数
CWT_FREQ_MIN = 1  # 保持 1Hz，避免 0Hz 除零错误
CWT_FREQ_MAX = 70  # 【已修改】上限调整为 70Hz
CWT_TIME_WINDOW_SEC = 300  # 5分钟

# === 绘图标尺设置 (固定标尺以支持组间对比) ===
# 【调整方案：上限设为 50】
# 0.01 - 50: 进一步扩大动态范围，区分强信号层次
PLOT_VMIN = 0.01
PLOT_VMAX = 50

# 绘图风格设置
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# =================功能函数=================

def apply_notch_filter(data, fs=SAMPLING_RATE, freq=NOTCH_FREQ, q=NOTCH_QUALITY):
    """50Hz 陷波滤波"""
    b, a = signal.iirnotch(freq, q, fs)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def complex_morlet_cwt(data, fs, freqs):
    """执行小波变换 (手动计算复Morlet)"""
    n_cycles = np.linspace(7, 3, len(freqs))
    scales = (n_cycles * fs) / (2 * np.pi * freqs)

    cwt_matrix = np.zeros((len(freqs), len(data)), dtype=complex)

    for i, (f, s, c) in enumerate(zip(freqs, scales, n_cycles)):
        M = int(s * 10)
        x = np.arange(0, M) - (M - 1.0) / 2
        x = x / s
        wavelet = np.pi ** (-0.25) * np.exp(1j * c * x) * np.exp(-0.5 * x ** 2)
        cwt_row = signal.convolve(data, wavelet, mode='same')
        cwt_matrix[i, :] = cwt_row

    return cwt_matrix


def select_files_gui():
    """GUI选择输入文件夹"""
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="选择原始LFP数据文件夹")
    if not folder_path: return None, None
    files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    if not files: return None, None
    files.sort()
    return files, folder_path


def create_output_dir_gui():
    """GUI选择输出路径"""
    export_path = filedialog.askdirectory(title="选择结果导出路径")
    if not export_path: return None
    target_dir = os.path.join(export_path, OUTPUT_ROOT_NAME)
    if not os.path.exists(target_dir): os.makedirs(target_dir)
    return target_dir


def get_group_label(filename):
    """从文件名判断组别"""
    if "Ctrl" in filename:
        return "Ctrl"
    elif "CUMS" in filename:
        return "CUMS"
    else:
        return "Unknown"


# =================主程序逻辑=================

def main():
    files, src_dir = select_files_gui()
    if not files: return
    output_dir = create_output_dir_gui()
    if not output_dir: return

    print("\n" + "=" * 50 + "\n开始处理数据...\n" + "=" * 50)

    file_pbar = tqdm(files, desc="处理文件", unit="file")

    for file_path in file_pbar:
        file_name = os.path.basename(file_path)
        file_pbar.set_postfix(current_file=file_name)

        group_label = get_group_label(file_name)

        try:
            df = pd.read_excel(file_path, header=0)
        except Exception as e:
            print(f"\n[Error] 无法读取文件 {file_name}: {e}")
            continue

        num_cols = df.shape[1]
        num_animals = num_cols // 2

        animal_pbar = tqdm(range(num_animals), desc=f"解析 {file_name} 动物", leave=False)

        for i in animal_pbar:
            animal_idx = i + 1
            col_idx_val = i * 2 + 1
            if col_idx_val >= num_cols: break
            raw_series = df.iloc[:, col_idx_val].dropna()
            if len(raw_series) == 0: continue

            data_raw = raw_series.values
            data_filtered = apply_notch_filter(data_raw)
            total_points = len(data_filtered)

            # ------------------------------------------------
            # 任务 2.3: 描绘 LFP 原始波形
            # ------------------------------------------------
            n_points_1s = int(1 * SAMPLING_RATE)

            if total_points > n_points_1s:
                for k in range(10):
                    start_idx = random.randint(0, total_points - n_points_1s)
                    end_idx = start_idx + n_points_1s

                    segment_data = data_filtered[start_idx:end_idx]
                    segment_time = np.linspace(0, 1, n_points_1s)

                    # 1. 导出原始波形数据
                    wave_df = pd.DataFrame({
                        "Time (s)": segment_time,
                        "Voltage (uV)": segment_data
                    })
                    wave_filename = f"{DATE_LABEL}_LFP_raw wave data_{group_label}_animal {animal_idx}_time {k + 1}.xlsx"
                    wave_save_path = os.path.join(output_dir, wave_filename)
                    wave_df.to_excel(wave_save_path, index=False)

                    # 2. 绘制波形图
                    plt.figure(figsize=(10, 4))
                    plt.plot(segment_time, segment_data, color='black', linewidth=0.8)
                    plt.title(f"{group_label} - Animal {animal_idx} - Raw LFP Wave (Sample {k + 1})")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Voltage (uV)")
                    plt.tight_layout()

                    img_filename = f"{DATE_LABEL}_LFP_raw wave_{group_label}_animal {animal_idx}_time {k + 1}.png"
                    plt.savefig(os.path.join(output_dir, img_filename), dpi=300)
                    plt.close()

            # ------------------------------------------------
            # 任务 2.4: 时频图绘制
            # ------------------------------------------------
            n_points_5min = int(CWT_TIME_WINDOW_SEC * SAMPLING_RATE)

            if total_points < n_points_5min:
                tqdm.write(f"[Warning] Animal {animal_idx} ({file_name}): 数据不足5分钟，跳过时频图。")
            else:
                start_idx_cwt = random.randint(0, total_points - n_points_5min)
                end_idx_cwt = start_idx_cwt + n_points_5min
                segment_cwt = data_filtered[start_idx_cwt:end_idx_cwt]
                freqs = np.arange(CWT_FREQ_MIN, CWT_FREQ_MAX + 1)

                # 计算 CWT 和 能量
                cwt_complex = complex_morlet_cwt(segment_cwt, SAMPLING_RATE, freqs)
                power = np.abs(cwt_complex) ** 2

                # 绘制时频图
                plt.figure(figsize=(12, 7))
                times_plot = np.linspace(0, CWT_TIME_WINDOW_SEC, len(segment_cwt))

                # 使用调整后的固定标尺 (0.01 - 50)
                plt.pcolormesh(times_plot, freqs, power, shading='gouraud', cmap='jet',
                               norm=LogNorm(vmin=PLOT_VMIN, vmax=PLOT_VMAX))

                plt.title(f"{group_label} - Animal {animal_idx} - LFP Spectrogram (CWT)", pad=20)
                plt.xlabel("Time (s)", labelpad=15, fontsize=12)
                plt.ylabel("Frequency (Hz)", labelpad=15, fontsize=12)
                plt.tick_params(axis='both', which='major', pad=10, labelsize=10)

                # 【已修改】纵坐标范围设置为 0-70，与分析频率一致
                plt.ylim(0, 70)
                plt.xlim(0, 300)

                cbar = plt.colorbar(pad=0.05)
                cbar.set_label(f'Power (uV^2) - Log Scale [Fixed: {PLOT_VMIN}-{PLOT_VMAX}]', labelpad=15)

                plt.tight_layout(pad=3.0)
                spec_img_name = f"{DATE_LABEL}_LFP_spectrogram_{group_label}_animal {animal_idx}.png"
                plt.savefig(os.path.join(output_dir, spec_img_name), dpi=300, bbox_inches='tight')
                plt.close()

    print("\n" + "=" * 50 + f"\n所有分析已完成！结果已保存在: {output_dir}\n" + "=" * 50)


if __name__ == "__main__":
    main()