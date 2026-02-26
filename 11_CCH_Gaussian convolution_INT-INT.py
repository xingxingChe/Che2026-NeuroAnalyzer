# -*- coding: utf-8 -*-
"""
INT-INT 抑制性突触连接分析脚本 (V_Gaussian_Fixed_Export)
--------------------------------------------------
核心算法: Sauer & Bartos, 2022 (eLife) - Gaussian Convolution
功能更新:
1. 修复了未导出 CCH 原始数据和基线数据的遗漏。
2. 现在会为每一对神经元生成包含 [Time_Lag, Raw_CCH, Baseline] 的 Excel 文件。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.stats import poisson
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, Listbox, END, Toplevel
from datetime import datetime


# ---------------------------------------------------
# 1. 核心分析函数
# ---------------------------------------------------

def calculate_cch(spike_train1, spike_train2, bin_size=0.0004, lag=0.05):
    """计算互相关图 (CCH)。"""
    num_bins = int(2 * lag / bin_size)
    if num_bins % 2 == 0:
        num_bins += 1
    bins = np.linspace(-lag, lag, num_bins + 1)

    cch = np.zeros(num_bins)
    for spike_time in spike_train1:
        # 优化速度：仅切片相关范围
        time_diffs = spike_train2 - spike_time
        # 过滤出 lag 范围内的差值
        valid_diffs = time_diffs[(time_diffs >= -lag) & (time_diffs <= lag)]
        if len(valid_diffs) > 0:
            hist, _ = np.histogram(valid_diffs, bins=bins)
            cch += hist

    bin_centers = (bins[:-1] + bins[1:]) / 2
    return cch, bin_centers


def create_hollow_gaussian(sigma_ms, hollow_fraction, bin_size_ms):
    """创建部分空心高斯核 (保持不变)。"""
    sigma_bins = sigma_ms / bin_size_ms
    size = int(sigma_bins * 8)
    x = np.arange(-size, size + 1)
    gaussian = np.exp(-(x ** 2) / (2 * sigma_bins ** 2))
    hollow_gaussian = np.exp(-(x ** 2) / (2 * (sigma_bins * (1 - hollow_fraction)) ** 2))
    kernel = gaussian - hollow_gaussian
    return kernel / np.sum(kernel)


def calculate_baseline(cch, sigma=10, hollow_fraction=0.6, bin_size=0.4):
    """计算基线 (保持不变)。"""
    kernel = create_hollow_gaussian(sigma, hollow_fraction, bin_size)
    baseline = convolve(cch, kernel, mode='same')
    return baseline


def perform_inhibition_test(cch, baseline, bins, p_thresh=0.001):
    """
    [关键修改] 对 CCH 进行抑制性连接的显著性检验。
    寻找在 0.8ms - 3.0ms 范围内显著低于基线的'谷'。
    """
    mono_window = (bins >= 0.0008) & (bins <= 0.0030)

    n_obs = np.sum(cch[mono_window])  # 观察到的脉冲数
    n_exp = np.sum(baseline[mono_window])  # 期望的基线脉冲数

    # 如果观察值比期望值还高，肯定不是抑制
    if n_obs >= n_exp:
        return False, 1.0, f"Observed({n_obs}) >= Expected({n_exp:.2f})"

    # 泊松累积分布函数 (CDF) 左尾检验
    p_val = poisson.cdf(n_obs, n_exp)

    is_inhibited = p_val < p_thresh

    summary = (
        f"Inhibition Test Results (INT -> INT):\n"
        f"-----------------------------------\n"
        f"Window Analysis (0.8ms to 3.0ms):\n"
        f"   - Observed Spikes (n_obs): {n_obs}\n"
        f"   - Expected Baseline (n_exp): {n_exp:.4f}\n"
        f"   - Probability (P_val): {p_val:.8f}\n"
        f"   - Criterion: P_val < {p_thresh}\n\n"
        f"Conclusion:\n"
        f"-----------------------------------\n"
        f"A significant INHIBITORY connection was {'DETECTED' if is_inhibited else 'NOT DETECTED'}.\n"
    )
    return is_inhibited, p_val, summary


def calculate_inhibition_strength(cch, baseline, bins, pre_spike_count):
    """计算抑制强度 (Spike Suppression Probability)。"""
    mono_window = (bins >= 0.0008) & (bins <= 0.0030)
    deficit_spikes = np.sum(baseline[mono_window]) - np.sum(cch[mono_window])

    if pre_spike_count == 0 or deficit_spikes < 0:
        return 0.0
    return deficit_spikes / pre_spike_count


# ---------------------------------------------------
# 3. 绘图与文件导出函数
# ---------------------------------------------------

def plot_and_save_cch_inhibition(cch, baseline, bins, pair_id, animal_id, is_connected, output_folder):
    """绘制 CCH 图。"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = np.mean(np.diff(bins))

    # 绘制直方图
    ax.bar(bins, cch, width=bar_width, color='gray', alpha=0.6, label='Raw CCH', edgecolor='none')
    # 绘制基线
    ax.plot(bins, baseline, color='black', linewidth=1.5, linestyle='--', label='Baseline')
    # 标记分析窗口
    ax.axvspan(0.0008, 0.0030, color='blue', alpha=0.1, label='Analysis Window')

    if is_connected:
        mono_window_indices = (bins >= 0.0008) & (bins <= 0.0030)
        ax.fill_between(bins[mono_window_indices],
                        cch[mono_window_indices],
                        baseline[mono_window_indices],
                        where=(baseline[mono_window_indices] > cch[mono_window_indices]),
                        color='blue', alpha=0.5, label='Inhibitory Deficit')

    ax.set_title(f"INT-INT Inhibition Analysis: {animal_id} - {pair_id}", fontsize=16)
    ax.set_xlabel("Time Lag (s)", fontsize=12)
    ax.set_ylabel("Spike Count", fontsize=12)
    ax.legend(loc='upper right')
    ax.set_xlim(-0.03, 0.03)

    status_text = "Inhibition Detected" if is_connected else "No Inhibition"
    status_color = 'blue' if is_connected else 'gray'

    ax.text(0.05, 0.95, status_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', fc=status_color, alpha=0.2, ec=status_color))

    suffix = "Inhibition" if is_connected else "No_Inhibition"
    date_str = datetime.now().strftime('%Y%m%d')
    filename = f"{date_str}_CCH_INT-INT_{animal_id}_{pair_id}_{suffix}.png"
    output_path = os.path.join(output_folder, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_numeric_data(data, filename, output_folder):
    """保存数值数据到 Excel (新增功能点)。"""
    df = pd.DataFrame(data)
    output_path = os.path.join(output_folder, filename)
    df.to_excel(output_path, index=False)


def save_text_data(text, filename, output_folder):
    output_path = os.path.join(output_folder, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)


# ---------------------------------------------------
# 4. GUI 部分
# ---------------------------------------------------
class CustomSortDialog(Toplevel):
    def __init__(self, parent, file_list):
        super().__init__(parent)
        self.title("自定义文件分析顺序")
        self.geometry("400x500")
        self.file_list = list(file_list)
        self.result = self.file_list
        tk.Label(self, text="请拖动或使用按钮调整文件顺序:").pack(pady=10)
        self.listbox = Listbox(self, selectmode=tk.SINGLE)
        for f in self.file_list:
            self.listbox.insert(END, f)
        self.listbox.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="上移", command=self.move_up).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="下移", command=self.move_down).pack(side=tk.LEFT, padx=5)
        tk.Button(self, text="确定", command=self.on_ok).pack(pady=10)
        self.transient(parent)
        self.grab_set()
        self.wait_window()

    def move_up(self):
        try:
            idx = self.listbox.curselection()[0]
            if idx > 0:
                item = self.listbox.get(idx)
                self.listbox.delete(idx)
                self.listbox.insert(idx - 1, item)
                self.listbox.selection_set(idx - 1)
        except IndexError:
            pass

    def move_down(self):
        try:
            idx = self.listbox.curselection()[0]
            if idx < self.listbox.size() - 1:
                item = self.listbox.get(idx)
                self.listbox.delete(idx)
                self.listbox.insert(idx + 1, item)
                self.listbox.selection_set(idx + 1)
        except IndexError:
            pass

    def on_ok(self):
        self.result = list(self.listbox.get(0, END))
        self.destroy()


def get_user_inputs():
    root = tk.Tk()
    root.withdraw()
    source_dir = filedialog.askdirectory(title="请选择包含原始数据 (.xlsx) 的文件夹")
    if not source_dir: return None, None, None
    try:
        files = sorted([f for f in os.listdir(source_dir) if f.endswith('.xlsx')])
        if not files:
            messagebox.showerror("错误", "未找到 .xlsx 文件。")
            return None, None, None
    except Exception:
        return None, None, None

    sort_choice = simpledialog.askstring("文件顺序", "请选择顺序:\n1: 升序\n2: 降序\n3: 自定义", parent=root)
    if sort_choice == '2':
        files.sort(reverse=True)
    elif sort_choice == '3':
        dialog = CustomSortDialog(root, files)
        files = dialog.result
    else:
        files.sort()

    output_dir = filedialog.askdirectory(title="请选择输出文件夹")
    if not output_dir: return None, None, None
    return source_dir, output_dir, files


# ---------------------------------------------------
# 5. 主程序逻辑
# ---------------------------------------------------

def main():
    P_THRESH = 0.001

    source_folder, output_folder, files_to_process = get_user_inputs()
    if not all([source_folder, output_folder, files_to_process]): return

    date_str = datetime.now().strftime('%Y%m%d')
    main_output_folder_name = f"{date_str}_INT_INT_Inhibition_Gaussian_Analysis"
    main_output_path = os.path.join(output_folder, main_output_folder_name)
    os.makedirs(main_output_path, exist_ok=True)

    summary_file_name = f"{date_str}_INT_INT_Summary.xlsx"
    summary_file_path = os.path.join(main_output_path, summary_file_name)
    summary_data = []

    print(f"Results will be saved to: {main_output_path}")

    try:
        with tqdm(total=len(files_to_process), desc="Total Progress") as pbar_files:
            for filename in files_to_process:
                animal_id = os.path.splitext(filename)[0]
                pbar_files.set_description(f"Processing: {animal_id}")

                numeric_path = os.path.join(main_output_path, f"{animal_id}_Details")
                os.makedirs(numeric_path, exist_ok=True)

                file_path = os.path.join(source_folder, filename)
                df = pd.read_excel(file_path, header=None)

                neuron_ids = df.iloc[0, :].astype(str)
                neuron_types = df.iloc[1, :].astype(int)

                int_neurons = {}
                for i, n_type in enumerate(neuron_types):
                    if n_type == 0:  # INT
                        spike_times = df.iloc[2:, i].dropna().to_numpy()
                        if len(spike_times) >= 500:
                            int_neurons[neuron_ids[i]] = spike_times

                int_ids = list(int_neurons.keys())
                total_pairs = len(int_ids) * (len(int_ids) - 1)

                if total_pairs == 0:
                    pbar_files.update(1)
                    continue

                with tqdm(total=total_pairs, desc=f"{animal_id} Pairs", leave=False) as pbar_pairs:
                    for pre_id in int_ids:
                        for post_id in int_ids:
                            if pre_id == post_id:
                                continue

                            pair_id = f"Pre_{pre_id}-Post_{post_id}"
                            pre_spikes = int_neurons[pre_id]
                            post_spikes = int_neurons[post_id]

                            # 1. 计算
                            cch, bins = calculate_cch(pre_spikes, post_spikes)
                            baseline = calculate_baseline(cch)

                            # 2. 检验
                            is_inhibited, p_val, sig_summary = perform_inhibition_test(
                                cch, baseline, bins, P_THRESH
                            )

                            # 3. 绘图
                            plot_and_save_cch_inhibition(cch, baseline, bins, pair_id, animal_id, is_inhibited,
                                                         main_output_path)

                            suffix = "Inhibition" if is_inhibited else "No_Inhibition"

                            # 4. 保存统计文本
                            save_text_data(sig_summary, f"{pair_id}_{suffix}_stats.txt", numeric_path)

                            # 5. [新增] 保存具体的 CCH 和基线数据到 Excel
                            raw_data_dict = {
                                "Time_Lag_s": bins,
                                "Raw_CCH_Counts": cch,
                                "Gaussian_Baseline": baseline
                            }
                            save_numeric_data(raw_data_dict, f"{pair_id}_{suffix}_CCH_Data.xlsx", numeric_path)

                            # 6. 汇总
                            if is_inhibited:
                                strength = calculate_inhibition_strength(cch, baseline, bins, len(pre_spikes))
                                summary_data.append({
                                    "Animal_ID": animal_id,
                                    "Pre_INT_ID": pre_id,
                                    "Post_INT_ID": post_id,
                                    "Spike_Suppression_Probability": strength,
                                    "P_value": p_val
                                })

                            pbar_pairs.update(1)
                pbar_files.update(1)

        # 保存总结文件
        if summary_data:
            pd.DataFrame(summary_data).to_excel(summary_file_path, index=False)
            print(f"Summary saved: {summary_file_path}")
        else:
            print("No inhibitory connections detected.")
            pd.DataFrame(columns=["Animal_ID", "Pre_INT_ID", "Post_INT_ID", "Spike_Suppression_Probability",
                                  "P_value"]).to_excel(
                summary_file_path, index=False)

        messagebox.showinfo("Done", "INT-INT Inhibition Analysis Complete!")

    except Exception as e:
        import traceback
        traceback.print_exc()
        messagebox.showerror("Error", f"Runtime Error: {e}")


if __name__ == "__main__":
    main()