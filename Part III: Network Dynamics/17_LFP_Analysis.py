import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, iirnotch, welch, filtfilt
from tkinter import Tk, filedialog, simpledialog
from tqdm import tqdm
from datetime import datetime
import warnings

# 忽略运行时警告
warnings.filterwarnings("ignore")


# ===========================
# 1. 信号处理函数定义
# ===========================

def apply_notch_filter(data, fs=1000.0, freq=50.0, Q=30.0):
    """
    应用陷波滤波器去除工频干扰 (50Hz)
    """
    b, a = iirnotch(w0=freq, Q=Q, fs=fs)
    y = filtfilt(b, a, data)
    return y


def apply_bandpass_filter_sos(data, lowcut, highcut, fs=1000.0, order=4):
    """
    应用 SOS (Second-Order Sections) 带通滤波器
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    y = sosfiltfilt(sos, data)
    return y


def calculate_psd(data, fs=1000.0):
    """
    计算 PSD (Welch 方法)
    """
    f, Pxx = welch(data, fs=fs, nperseg=2000)
    return f, Pxx


# ===========================
# 2. 主程序逻辑
# ===========================

def main():
    root = Tk()
    root.withdraw()

    # 获取当前系统日期
    current_date = datetime.now().strftime("%Y%m%d")

    print(f"=== LFP 数据分析程序 (v4.0 统计分析优化版) ===")
    print(f"检测到今天日期: {current_date}\n")

    # --- 2.1 读取原始数据文件 ---
    print("[步骤 1/5] 请选择包含原始数据文件的文件夹...")
    input_folder = filedialog.askdirectory(title="选择原始数据文件夹")
    if not input_folder:
        return

    all_files = glob.glob(os.path.join(input_folder, "*.xlsx"))
    if not all_files:
        print("未找到 .xlsx 文件！")
        return

    # 排序逻辑
    file_names = [os.path.basename(f) for f in all_files]
    sort_choice = simpledialog.askstring("文件排序",
                                         f"检测到 {len(all_files)} 个文件。\n请输入排序方式 (1: 升序, 2: 降序, 3: 默认):",
                                         initialvalue="1")

    if sort_choice == '1':
        all_files.sort()
    elif sort_choice == '2':
        all_files.sort(reverse=True)

    # --- 2.2 创建输出路径 ---
    print("\n[步骤 2/5] 请选择数据导出的保存路径...")
    output_base_path = filedialog.askdirectory(title="选择输出保存路径")
    if not output_base_path:
        return

    output_folder_name = f"{current_date}_LFP Analysis"
    output_folder = os.path.join(output_base_path, output_folder_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 定义频段
    bands = {
        'delta': (0.5, 3.5),
        'theta': (4, 7),
        'alpha': (8, 13),
        'beta': (15, 28),
        'gamma': (30, 70)
    }

    # 用于汇总数据的列表
    summary_data_storage = {
        'Ctrl': [],
        'CUMS': []
    }

    # --- 开始处理文件循环 ---
    print("\n[步骤 3/5] 开始批量处理文件...")

    for file_path in tqdm(all_files, desc="总进度", unit="file"):
        file_name = os.path.basename(file_path)

        # 智能识别分组
        if "Ctrl" in file_name:
            group_name = "Ctrl"
        elif "CUMS" in file_name:
            group_name = "CUMS"
        else:
            group_name = "Unknown"

        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f"读取失败: {file_name}")
            continue

        num_animals = df.shape[1] // 2

        for i in tqdm(range(num_animals), desc=f"处理 {group_name} 组动物", leave=False):
            raw_animal_idx = i + 1
            animal_id_str = f"{group_name}_{raw_animal_idx}"

            time_col = df.iloc[:, 2 * i]
            val_col = df.iloc[:, 2 * i + 1]
            # 简单校验
            if time_col.isna().all() or val_col.isna().all():
                continue

            valid_mask = ~np.isnan(val_col)
            t_data = time_col[valid_mask].values
            v_data = val_col[valid_mask].values

            if len(v_data) < 1000:
                continue

            # --- 预处理 ---
            v_clean = apply_notch_filter(v_data)

            # --- 随机截取 1s ---
            max_start_idx = len(v_clean) - 1000
            start_idx = random.randint(0, max_start_idx)
            end_idx = start_idx + 1000

            t_1s = t_data[start_idx:end_idx]
            v_clean_1s = v_clean[start_idx:end_idx]

            # --- 2.3 导出原始波形 ---
            raw_wave_filename = f"{current_date}_LFP_raw wave data_animal {animal_id_str}.xlsx"
            pd.DataFrame({'Timestamp': t_1s, 'Value': v_clean_1s}).to_excel(
                os.path.join(output_folder, raw_wave_filename), index=False
            )

            plt.figure(figsize=(10, 4))
            plt.plot(t_1s, v_clean_1s, color='black', linewidth=0.8)
            plt.title(f"Animal {animal_id_str} Raw LFP (1s) - {current_date}")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage")
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"{current_date}_LFP_raw wave_animal {animal_id_str}.png"), dpi=150)
            plt.close()

            # --- 2.4 导出频段波形 ---
            for band_name, (low, high) in bands.items():
                v_band = apply_bandpass_filter_sos(v_clean, low, high)
                v_band_1s = v_band[start_idx:end_idx]

                # 导出 Excel
                band_xlsx_name = f"{current_date}_LFP_{band_name}_raw wave data_animal {animal_id_str}.xlsx"
                pd.DataFrame({'Timestamp': t_1s, 'Value': v_band_1s}).to_excel(
                    os.path.join(output_folder, band_xlsx_name), index=False
                )

                # 绘图
                plt.figure(figsize=(10, 4))
                plt.plot(t_1s, v_band_1s, color='blue', linewidth=0.8)
                plt.title(f"Animal {animal_id_str} {band_name.capitalize()} Waveform")
                plt.xlabel("Time (s)")
                plt.ylabel("Voltage")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_folder, f"{current_date}_LFP_{band_name}_raw wave_animal {animal_id_str}.png"),
                    dpi=150)
                plt.close()

            # --- 2.5 计算 PSD 并汇总 ---
            f, Pxx = calculate_psd(v_clean)
            freq_mask = (f >= 0.5) & (f <= 100)
            f_sel = f[freq_mask]
            Pxx_sel = Pxx[freq_mask]
            Pxx_log = np.log10(Pxx_sel)

            # 导出 Un-transformed 和 Log-transformed 详情表
            pd.DataFrame({'Frequency (Hz)': f_sel, 'PSD (V^2/Hz)': Pxx_sel}).to_excel(
                os.path.join(output_folder, f"{current_date}_LFP_PSD_untransformed data_animal {animal_id_str}.xlsx"),
                index=False
            )
            pd.DataFrame({'Frequency (Hz)': f_sel, 'PSD (Log(V^2/Hz))': Pxx_log}).to_excel(
                os.path.join(output_folder, f"{current_date}_LFP_PSD_log transformation_animal {animal_id_str}.xlsx"),
                index=False
            )

            # 汇总数据 (内存中暂存长格式)
            for band_name, (low, high) in bands.items():
                idx_band = (f_sel >= low) & (f_sel <= high)
                if np.sum(idx_band) > 0:
                    mean_log_psd = np.mean(Pxx_log[idx_band])
                else:
                    mean_log_psd = np.nan

                if group_name in summary_data_storage:
                    summary_data_storage[group_name].append({
                        'Animal ID': animal_id_str,
                        'Group': group_name,
                        'Band': band_name,
                        'Mean Log PSD': mean_log_psd
                    })

    # --- 最后步骤：生成汇总 Excel 文件 (转换为宽格式) ---
    print("\n[步骤 4/5] 正在转换格式并生成汇总表格...")

    # 定义期望的列顺序 (生理学顺序)
    desired_band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    # 处理并导出 Ctrl 组
    if summary_data_storage['Ctrl']:
        df_long = pd.DataFrame(summary_data_storage['Ctrl'])

        # 【关键步骤】 Pivot: 长格式转宽格式
        # Index 是动物ID，Column 是频段名，Value 是PSD值
        df_wide = df_long.pivot(index=['Animal ID', 'Group'], columns='Band', values='Mean Log PSD').reset_index()

        # 重新排列列的顺序 (保证是 delta -> theta -> alpha...)
        cols = ['Animal ID', 'Group'] + [b for b in desired_band_order if b in df_wide.columns]
        df_wide = df_wide[cols]

        save_path = os.path.join(output_folder, f"{current_date}_LFP_PSD_log transformation_5 bands_Ctrl.xlsx")
        df_wide.to_excel(save_path, index=False)
        print(f" - Ctrl 组 (5 bands) 汇总表已生成")

    # 处理并导出 CUMS 组
    if summary_data_storage['CUMS']:
        df_long = pd.DataFrame(summary_data_storage['CUMS'])

        # Pivot
        df_wide = df_long.pivot(index=['Animal ID', 'Group'], columns='Band', values='Mean Log PSD').reset_index()

        # Reorder columns
        cols = ['Animal ID', 'Group'] + [b for b in desired_band_order if b in df_wide.columns]
        df_wide = df_wide[cols]

        save_path = os.path.join(output_folder, f"{current_date}_LFP_PSD_log transformation_5 bands_CUMS.xlsx")
        df_wide.to_excel(save_path, index=False)
        print(f" - CUMS 组 (5 bands) 汇总表已生成")

    print("\n" + "=" * 40)
    print("所有分析任务已完成！")
    print(f"结果文件夹: {output_folder}")
    print("=" * 40)


if __name__ == "__main__":
    main()