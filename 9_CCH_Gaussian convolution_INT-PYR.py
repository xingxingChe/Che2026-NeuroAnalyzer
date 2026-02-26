# -*- coding: utf-8 -*-
"""
INT-PYR 抑制性突触连接分析脚本 (V_Gaussian_Fixed)
--------------------------------------------------
修复说明: 修正了 bin_edges 与 cch 长度不匹配导致的 IndexError。
核心算法: Sauer & Bartos, 2022 (eLife) - Gaussian Convolution
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


# =========================================================================
# 1. 核心分析函数 (Gaussian Convolution - Inhibition)
# =========================================================================

def calculate_cch(spike_train1, spike_train2, bin_size=0.0004, lag=0.05):
    """计算互相关图"""
    num_bins = int(2 * lag / bin_size)
    if num_bins % 2 == 0:
        num_bins += 1
    bins = np.linspace(-lag, lag, num_bins + 1)

    cch = np.zeros(num_bins)
    for t1 in spike_train1:
        # 优化速度：只切片相关范围
        relevant_t2 = spike_train2[(spike_train2 >= t1 - lag) & (spike_train2 <= t1 + lag)]
        if len(relevant_t2) > 0:
            hist, _ = np.histogram(relevant_t2 - t1, bins=bins)
            cch += hist

    bin_centers = (bins[:-1] + bins[1:]) / 2
    return cch, bin_centers, bins


def create_hollow_gaussian(sigma_ms, hollow_fraction, bin_size_ms):
    """创建部分空心高斯核 (Sauer & Bartos, 2022)"""
    sigma_bins = sigma_ms / bin_size_ms
    size = int(sigma_bins * 8)
    x = np.arange(-size, size + 1)
    gaussian = np.exp(-(x ** 2) / (2 * sigma_bins ** 2))
    hollow_gaussian = np.exp(-(x ** 2) / (2 * (sigma_bins * (1 - hollow_fraction)) ** 2))
    kernel = gaussian - hollow_gaussian
    return kernel / np.sum(kernel)


def calculate_baseline(cch, sigma=10, hollow_fraction=0.6, bin_size=0.4):
    """卷积计算基线"""
    kernel = create_hollow_gaussian(sigma, hollow_fraction, bin_size)
    baseline = convolve(cch, kernel, mode='same')
    return baseline


def perform_inhibition_test_gaussian(cch, baseline, bin_centers, p_thresh=0.001):
    """
    [核心逻辑] 高斯法检测抑制性连接
    注意：这里的 bins 必须传入 bin_centers (长度与 cch 一致)
    """
    # 使用 bin_centers 进行掩码生成，确保长度为 N
    mono_window = (bin_centers >= 0.0008) & (bin_centers <= 0.0030)

    n_obs = np.sum(cch[mono_window])  # 观察值
    n_exp = np.sum(baseline[mono_window])  # 期望值 (基线)

    # 如果观察值 >= 期望值，显然没有抑制
    if n_obs >= n_exp:
        return False, 1.0, 0.0

    # 泊松左尾检验 (计算观察到 <= n_obs 的概率)
    p_val = poisson.cdf(n_obs, n_exp)

    is_inhibited = p_val < p_thresh

    # 计算缺失的脉冲数 (Deficit)
    deficit = n_exp - n_obs

    return is_inhibited, p_val, deficit


# =========================================================================
# 2. 绘图与导出
# =========================================================================

def plot_gaussian_inhibition(cch, baseline, bin_centers, pair_id, animal_id, is_inhibited, ssp, output_folder):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # 计算柱宽
    if len(bin_centers) > 1:
        bar_width = bin_centers[1] - bin_centers[0]
    else:
        bar_width = 0.0004

    # 绘制 CCH 和 基线 (使用 bin_centers 作为 x 轴)
    ax.bar(bin_centers, cch, width=bar_width, color='gray', alpha=0.5, label='Raw CCH', edgecolor='none')
    ax.plot(bin_centers, baseline, color='black', linewidth=1.5, linestyle='--', label='Gaussian Baseline')

    # 分析窗口
    ax.axvspan(0.0008, 0.0030, color='blue', alpha=0.1, label='Window (0.8-3ms)')

    # 高亮抑制区域 (Deficit)
    if is_inhibited:
        window_mask = (bin_centers >= 0.0008) & (bin_centers <= 0.0030)
        ax.fill_between(bin_centers[window_mask],
                        cch[window_mask],
                        baseline[window_mask],
                        where=(baseline[window_mask] > cch[window_mask]),
                        color='blue', alpha=0.6, label='Suppressed Spikes')

    ax.set_title(f"INT->PYR Inhibition (Gaussian): {animal_id} - {pair_id}", fontsize=14)
    ax.set_xlabel("Time Lag (s)", fontsize=12)
    ax.set_ylabel("Spike Count", fontsize=12)
    ax.set_xlim(-0.03, 0.03)
    ax.legend(loc='upper right')

    status_text = f"Suppression Detected\nSSP: {ssp:.4f}" if is_inhibited else "No Suppression"
    status_color = 'blue' if is_inhibited else 'gray'
    ax.text(0.05, 0.95, status_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', fc=status_color, alpha=0.2))

    suffix = "Inhibition" if is_inhibited else "No_Inhibition"
    date_str = datetime.now().strftime('%Y%m%d')
    filename = f"{date_str}_Gaussian_INT-PYR_{animal_id}_{pair_id}_{suffix}.png"
    output_path = os.path.join(output_folder, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_numeric_data(data, filename, output_folder):
    df = pd.DataFrame(data)
    path = os.path.join(output_folder, filename)
    df.to_excel(path, index=False)


# =========================================================================
# 3. GUI 交互
# =========================================================================

class CustomSortDialog(Toplevel):
    def __init__(self, parent, file_list):
        super().__init__(parent)
        self.title("自定义文件分析顺序")
        self.geometry("400x500")
        self.file_list = list(file_list)
        self.result = self.file_list
        tk.Label(self, text="请拖动或使用按钮调整文件顺序:").pack(pady=10)
        self.listbox = Listbox(self, selectmode=tk.SINGLE)
        for f in self.file_list: self.listbox.insert(END, f)
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
        if not files: return None, None, None
    except:
        return None, None, None
    sort_choice = simpledialog.askstring("文件顺序", "请选择顺序:\n1: 升序\n2: 降序\n3: 自定义", parent=root)
    if sort_choice == '2':
        files.sort(reverse=True)
    elif sort_choice == '3':
        d = CustomSortDialog(root, files)
        files = d.result
    else:
        files.sort()
    output_dir = filedialog.askdirectory(title="请选择输出文件夹")
    return source_dir, output_dir, files


# =========================================================================
# 4. 主程序
# =========================================================================

def main():
    # 参数配置
    P_THRESH = 0.001  # Sauer & Bartos 标准
    MIN_SPIKES = 500

    source_folder, output_folder, files_to_process = get_user_inputs()
    if not all([source_folder, output_folder, files_to_process]): return

    date_str = datetime.now().strftime('%Y%m%d')
    # 标注 Gaussian 方法
    main_output_name = f"{date_str}_INT_PYR_Inhibition_Gaussian_Analysis"
    main_output_path = os.path.join(output_folder, main_output_name)
    os.makedirs(main_output_path, exist_ok=True)

    summary_file_path = os.path.join(main_output_path, f"{date_str}_INT_PYR_Gaussian_Summary_SSP.xlsx")
    summary_data = []

    print(f"Results will be saved to: {main_output_path}")
    print(f"Analyzing: INT (Pre) -> PYR (Post)")
    print(f"Algorithm: Gaussian Convolution (Sauer & Bartos, 2022)")

    try:
        with tqdm(total=len(files_to_process), desc="Processing Files") as pbar_files:
            for filename in files_to_process:
                animal_id = os.path.splitext(filename)[0]
                pbar_files.set_description(f"File: {animal_id}")

                detail_folder = os.path.join(main_output_path, f"{animal_id}_Details")
                os.makedirs(detail_folder, exist_ok=True)

                file_path = os.path.join(source_folder, filename)
                df = pd.read_excel(file_path, header=None)
                neuron_ids = df.iloc[0, :].astype(str)
                neuron_types = df.iloc[1, :].astype(int)

                # 提取神经元
                int_neurons = {}  # Pre (Type 0)
                pyr_neurons = {}  # Post (Type 1)

                for i, n_type in enumerate(neuron_types):
                    spikes = df.iloc[2:, i].dropna().to_numpy()
                    if len(spikes) >= MIN_SPIKES:
                        if n_type == 0:
                            int_neurons[neuron_ids[i]] = spikes
                        elif n_type == 1:
                            pyr_neurons[neuron_ids[i]] = spikes

                int_ids = list(int_neurons.keys())
                pyr_ids = list(pyr_neurons.keys())
                pairs_count = len(int_ids) * len(pyr_ids)

                if pairs_count == 0:
                    pbar_files.update(1)
                    continue

                # 配对分析 INT -> PYR
                with tqdm(total=pairs_count, desc=f"  Analysing INT->PYR ({animal_id})", leave=False) as pbar_pairs:
                    for pre_id in int_ids:
                        for post_id in pyr_ids:
                            if pre_id == post_id: continue

                            pair_id = f"PreINT_{pre_id}-PostPYR_{post_id}"
                            pre_spikes = int_neurons[pre_id]
                            post_spikes = pyr_neurons[post_id]

                            # 1. 计算 CCH (注意：这里我们明确解包出三个变量)
                            # cch: counts (len=N)
                            # bin_centers: 中心点 (len=N)
                            # bin_edges: 边缘点 (len=N+1)
                            cch, bin_centers, bin_edges = calculate_cch(pre_spikes, post_spikes)

                            # 2. 计算基线 (Gaussian)
                            baseline = calculate_baseline(cch)

                            # 3. 统计检验 (Inhibition)
                            # 关键修正：这里传入 bin_centers (长度与 cch 一致)，而不是 bin_edges
                            is_inhibited, p_val, deficit = perform_inhibition_test_gaussian(
                                cch, baseline, bin_centers, P_THRESH
                            )

                            # 4. 计算 SSP
                            ssp = deficit / len(pre_spikes) if len(pre_spikes) > 0 else 0.0

                            # 5. 绘图
                            # 关键修正：这里传入 bin_centers
                            plot_gaussian_inhibition(cch, baseline, bin_centers, pair_id, animal_id, is_inhibited, ssp,
                                                     main_output_path)

                            # 6. 保存数据
                            # 关键修正：保存 bin_centers
                            suffix = "Inhibition" if is_inhibited else "No_Inhibition"
                            save_numeric_data({
                                'time_lag_s': bin_centers,
                                'raw_cch': cch,
                                'baseline': baseline
                            }, f"{pair_id}_{suffix}_data.xlsx", detail_folder)

                            # 7. 汇总
                            if is_inhibited:
                                summary_data.append({
                                    "Animal_ID": animal_id,
                                    "Pre_INT_ID": pre_id,
                                    "Post_PYR_ID": post_id,
                                    "Spike_Suppression_Probability": ssp,
                                    "P_value": p_val,
                                    "Method": "Gaussian_Convolution"
                                })

                            pbar_pairs.update(1)
                pbar_files.update(1)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(summary_file_path, index=False)
            print(f"\nSummary saved: {summary_file_path}")
        else:
            print("\nNo INT->PYR inhibitory connections detected.")
            pd.DataFrame(columns=["Animal_ID", "Pre_INT_ID", "Post_PYR_ID", "Spike_Suppression_Probability"]).to_excel(
                summary_file_path, index=False)

        messagebox.showinfo("Complete", "INT-PYR Gaussian Analysis Finished!")

    except Exception as e:
        import traceback
        traceback.print_exc()
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


if __name__ == "__main__":
    main()