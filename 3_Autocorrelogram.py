import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox


# --- 计算函数 ---

def calculate_autocorrelogram(spike_times_sec, max_lag_ms, bin_size_ms):
    """
    【已修改】计算单个神经元的双侧对称自相关图数据。

    参数:
    spike_times_sec (array): 神经元放电时刻 (单位: 秒)。
    max_lag_ms (float): 最大时间差 (单位: 毫秒)。
    bin_size_ms (float): 时间窗格大小 (单位: 毫秒)。

    返回:
    autocorr_counts (array): 每个时间窗格中的原始脉冲计数值。
    bins (array): 时间窗格的边界。
    """
    spike_times_ms = spike_times_sec.dropna().to_numpy() * 1000

    # 定义双侧对称的时间边界
    bins = np.arange(-max_lag_ms, max_lag_ms + bin_size_ms, bin_size_ms)

    if len(spike_times_ms) < 2:
        return np.zeros(len(bins) - 1), bins

    # 计算所有正负时间差
    diffs = []
    for i in range(len(spike_times_ms)):
        for j in range(i + 1, len(spike_times_ms)):
            diff = spike_times_ms[j] - spike_times_ms[i]
            if diff <= max_lag_ms:
                diffs.extend([diff, -diff])  # 同时记录正负时间差
            else:
                break

    diffs = np.array(diffs)

    autocorr_counts, _ = np.histogram(diffs, bins=bins)

    # 将t=0的中心仓置零 (因为它代表脉冲自身的相关性，无意义)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    zero_bin_index = np.argmin(np.abs(bin_centers))
    if np.abs(bin_centers[zero_bin_index]) < bin_size_ms / 2.0:
        autocorr_counts[zero_bin_index] = 0

    return autocorr_counts, bins


def normalize_autocorrelogram(autocorr_counts, num_spikes, bin_size_ms):
    """
    将自相关图的原始计数值归一化为放电率 (Hz)。
    """
    if num_spikes == 0:
        return autocorr_counts.astype(float)

    bin_size_s = bin_size_ms / 1000.0
    autocorr_hz = autocorr_counts / (num_spikes * bin_size_s)
    return autocorr_hz


def calculate_metrics(autocorr_counts, bins, bin_size_ms):
    """
    【已修改】根据双侧自相关图的右半部分(t>0)计算各项指标。
    """
    peak_time_ms = np.nan
    refractory_period_ms = np.nan
    burst_index = np.nan
    fwhm_ms = np.nan

    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # --- 只在正时间差(右半部分)进行计算 ---
    positive_mask = bin_centers > 0
    autocorr_positive = autocorr_counts[positive_mask]
    bin_centers_positive = bin_centers[positive_mask]

    if np.sum(autocorr_positive) > 0:
        # 1. 不应期
        first_spike_idx = np.where(autocorr_positive > 0)[0]
        if len(first_spike_idx) > 0:
            refractory_period_ms = bin_centers_positive[first_spike_idx[0]]

        # 2. 峰值时间 (从3ms后开始搜索)
        search_start_time = 3  # ms
        peak_search_mask = bin_centers_positive >= search_start_time
        if np.any(peak_search_mask) and np.sum(autocorr_positive[peak_search_mask]) > 0:
            peak_idx_relative = np.argmax(autocorr_positive[peak_search_mask])
            peak_idx_absolute = np.where(peak_search_mask)[0][peak_idx_relative]

            peak_time_ms = bin_centers_positive[peak_idx_absolute]
            peak_value = autocorr_positive[peak_idx_absolute]

            # 4. 半峰全宽 (FWHM)
            if peak_value > 0:
                half_max = peak_value / 2.0
                left_idx = np.where(autocorr_positive[:peak_idx_absolute + 1] < half_max)[0]
                right_idx = np.where(autocorr_positive[peak_idx_absolute:] < half_max)[0]

                if len(left_idx) > 0 and len(right_idx) > 0:
                    left_boundary_idx = left_idx[-1]
                    right_boundary_idx = right_idx[0] + peak_idx_absolute
                    fwhm_ms = (bin_centers_positive[right_boundary_idx] - bin_centers_positive[left_boundary_idx])

        # 3. 爆发指数
        early_window_mask = (bin_centers_positive >= 3) & (bin_centers_positive <= 15)
        late_window_mask = (bin_centers_positive >= 50) & (bin_centers_positive <= 100)

        if np.any(early_window_mask) and np.any(late_window_mask):
            mean_early = np.mean(autocorr_positive[early_window_mask])
            mean_late = np.mean(autocorr_positive[late_window_mask])

            if mean_late > 0:
                burst_index = mean_early / mean_late
            elif mean_early > 0 and mean_late == 0:
                burst_index = np.inf

    return {
        "Autocorrelogram_Peak_Time": peak_time_ms,
        "Refractory_Period": refractory_period_ms,
        "Burst_Index": burst_index,
        "FWHM": fwhm_ms
    }


# --- GUI 和主流程函数 ---

def main():
    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo("提示", "请选择包含原始数据 (.xlsx) 文件的文件夹。")
    input_dir = filedialog.askdirectory(title="请选择原始数据文件夹")
    if not input_dir: return

    files = glob.glob(os.path.join(input_dir, '*.xlsx'))
    if not files:
        messagebox.showerror("错误", "所选文件夹中未找到任何 .xlsx 文件。");
        return

    order_choice = simpledialog.askstring("文件顺序", "请选择文件分析顺序:\n1: 升序\n2: 降序\n3: 默认顺序", parent=root)
    if order_choice == '1':
        files.sort()
    elif order_choice == '2':
        files.sort(reverse=True)

    file_list_str = "\n".join([os.path.basename(f) for f in files])
    print(f"程序将按以下顺序处理文件:\n{file_list_str}")
    messagebox.showinfo("文件处理顺序", f"程序将按以下顺序处理文件:\n\n{file_list_str}")

    output_dir_parent = filedialog.askdirectory(title="请选择结果保存路径")
    if not output_dir_parent: return

    date_str = datetime.now().strftime('%Y%m%d')
    output_main_dir = os.path.join(output_dir_parent, f"{date_str}_C.4.e_Autocorrelogram_raw")
    output_png_dir = os.path.join(output_main_dir, f"{date_str}_All Neurons Autocorrelograms_png")
    os.makedirs(output_png_dir, exist_ok=True)

    numeric_filepath = os.path.join(output_main_dir, f"{date_str}_All Neurons Autocorrelogram_numeric.xlsx")
    summary_filepath = os.path.join(output_main_dir, f"{date_str}_All Neurons Autocorrelogram_summary.xlsx")

    pd.DataFrame(columns=["Number", "Autocorrelogram_Peak_Time", "Refractory_Period", "Burst_Index", "FWHM"]).to_excel(
        summary_filepath, index=False)

    params_str = simpledialog.askstring("参数设置", "请输入参数: 最大时间差(ms),时间窗格(ms)\n例如: 100,1", parent=root)
    try:
        max_lag_ms, bin_size_ms = map(float, params_str.split(','))
    except (TypeError, ValueError):
        messagebox.showerror("错误", "参数格式不正确，程序将使用默认值 (100, 1)。")
        max_lag_ms, bin_size_ms = 100.0, 1.0

    # 预先计算一次横坐标，用于numeric文件的表头
    _, bins_template = calculate_autocorrelogram(pd.Series([]), max_lag_ms, bin_size_ms)
    bin_centers_template = (bins_template[:-1] + bins_template[1:]) / 2.0
    numeric_data = {'Time_ms': bin_centers_template}

    all_summary_data = []

    try:
        with tqdm(total=len(files), desc="文件处理进度") as file_pbar:
            for file_path in files:
                filename = os.path.basename(file_path)
                file_pbar.set_description(f"正在处理: {filename}")

                df = pd.read_excel(file_path, header=0)
                neuron_numbers = df.columns.tolist()

                with tqdm(total=len(neuron_numbers), desc=f"神经元进度 ({filename})", leave=False) as neuron_pbar:
                    for neuron_num in neuron_numbers:
                        neuron_pbar.set_description(f"处理中: 神经元 {neuron_num}")

                        spike_times = df[neuron_num].iloc[1:]
                        valid_spike_times = spike_times.dropna()
                        num_spikes = len(valid_spike_times)

                        autocorr_counts, bins_calc = calculate_autocorrelogram(valid_spike_times, max_lag_ms,
                                                                               bin_size_ms)
                        autocorr_hz = normalize_autocorrelogram(autocorr_counts, num_spikes, bin_size_ms)

                        numeric_data[neuron_num] = autocorr_hz

                        # 【更新】绘制折线图
                        bin_centers = (bins_calc[:-1] + bins_calc[1:]) / 2.0
                        plt.figure(figsize=(8, 6))
                        plt.plot(bin_centers, autocorr_hz)  # 改为折线图
                        plt.axvline(0, color='gray', linestyle='--', linewidth=1)  # 添加中心线
                        plt.title(f'Autocorrelogram for Neuron {neuron_num}')
                        plt.xlabel('Time Lag (ms)')
                        plt.ylabel('Firing Rate (Hz)')
                        plt.xlim(-max_lag_ms, max_lag_ms)  # 更新X轴范围

                        plt.show(block=False)
                        plt.pause(1)

                        png_filename = f"{date_str}_{neuron_num}_Autocorrelogram.png"
                        plt.savefig(os.path.join(output_png_dir, png_filename), dpi=600)
                        plt.close()

                        metrics = calculate_metrics(autocorr_counts, bins_calc, bin_size_ms)

                        summary_row = {"Number": neuron_num, **metrics}
                        all_summary_data.append(summary_row)

                        neuron_pbar.update(1)
                file_pbar.update(1)

        print("正在保存数值结果...")
        pd.DataFrame(numeric_data).to_excel(numeric_filepath, index=False)

        print("正在保存统计摘要...")
        summary_final_df = pd.DataFrame(all_summary_data)
        with pd.ExcelWriter(summary_filepath, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            summary_final_df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)

        messagebox.showinfo("完成", "所有分析已完成并成功保存！")
        print("\n--- 程序运行结束 ---")

    except Exception as e:
        messagebox.showerror("发生错误", f"程序运行中出现错误: \n{e}")
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()