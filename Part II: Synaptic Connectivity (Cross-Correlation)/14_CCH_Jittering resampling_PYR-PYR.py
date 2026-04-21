# -*- coding: utf-8 -*-
"""
PYR-PYR 兴奋性突触连接分析脚本 (V_Jittering)
--------------------------------------------------
分析方向: Pyramidal (Pre) -> Pyramidal (Post)
核心算法: Fujisawa et al., 2008 (Nature Neuroscience) - Jittering Resampling
量化指标: Spike Transmission Probability (STP)

注意:
根据 Sauer & Bartos (2022) 和 Fujisawa (2008)，PYR-PYR 连接在体内记录中极难检测。
检出率低属于正常现象 (通常 < 0.5%)。

功能:
1. 识别 Type 1 (PYR) 神经元。
2. 对每一对 PYR-PYR 进行 [-5ms, +5ms] 随机抖动 (1000次)。
3. 构建 99.5% 显著性上界。
4. 导出 Excel 和图表 (红色高亮)。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, Listbox, END, Toplevel
from datetime import datetime


# =========================================================================
# 1. 核心分析函数 (Jittering - Excitation)
# =========================================================================

def calculate_cch(spike_train1, spike_train2, bin_size=0.0004, lag=0.05):
    """计算 CCH"""
    num_bins = int(2 * lag / bin_size)
    if num_bins % 2 == 0:
        num_bins += 1
    bins = np.linspace(-lag, lag, num_bins + 1)

    all_diffs = []
    for t1 in spike_train1:
        relevant_t2 = spike_train2[(spike_train2 >= t1 - lag) & (spike_train2 <= t1 + lag)]
        all_diffs.append(relevant_t2 - t1)

    if all_diffs:
        all_diffs = np.concatenate(all_diffs)
        cch, _ = np.histogram(all_diffs, bins=bins)
    else:
        cch = np.zeros(num_bins)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    return cch, bin_centers, bins


def fast_cch_for_jitter(pre_spikes, post_spikes, bins, lag):
    """快速 CCH 用于循环"""
    all_diffs = []
    for t1 in pre_spikes:
        relevant_t2 = post_spikes[(post_spikes >= t1 - lag) & (post_spikes <= t1 + lag)]
        if len(relevant_t2) > 0:
            all_diffs.append(relevant_t2 - t1)

    if len(all_diffs) > 0:
        all_diffs = np.concatenate(all_diffs)
        counts, _ = np.histogram(all_diffs, bins=bins)
        return counts
    else:
        return np.zeros(len(bins) - 1)


def run_jittering_test_excitation(pre_spikes, post_spikes, n_iter=1000, jitter_window=0.005):
    """
    [核心算法] Fujisawa Jittering Method (Excitation)
    """
    LAG = 0.05
    BIN_SIZE = 0.0004
    # PYR-PYR 的潜伏期通常也很快，维持 1-4ms 窗口
    ANALYSIS_WINDOW_MIN = 0.0010
    ANALYSIS_WINDOW_MAX = 0.0040

    raw_cch, bin_centers, bins_edges = calculate_cch(pre_spikes, post_spikes, BIN_SIZE, LAG)

    # Jittering Loop
    surrogate_cchs = np.zeros((n_iter, len(raw_cch)))
    noise = np.random.uniform(-jitter_window, jitter_window, (n_iter, len(pre_spikes)))

    for i in range(n_iter):
        jittered_pre = pre_spikes + noise[i, :]
        jittered_pre.sort()
        surrogate_cchs[i, :] = fast_cch_for_jitter(jittered_pre, post_spikes, bins_edges, LAG)

    baseline_mean = np.mean(surrogate_cchs, axis=0)
    # 99.5% 上界判定
    upper_bound = np.percentile(surrogate_cchs, 99.5, axis=0)

    window_mask = (bin_centers >= ANALYSIS_WINDOW_MIN) & (bin_centers <= ANALYSIS_WINDOW_MAX)
    is_connected = np.any(raw_cch[window_mask] > upper_bound[window_mask])

    # 计算 STP
    excess_spikes = np.sum(raw_cch[window_mask]) - np.sum(baseline_mean[window_mask])
    if excess_spikes < 0:
        excess_spikes = 0.0

    stp = excess_spikes / len(pre_spikes) if len(pre_spikes) > 0 else 0.0

    return {
        "is_connected": is_connected,
        "stp": stp,
        "raw_cch": raw_cch,
        "baseline_mean": baseline_mean,
        "upper_bound": upper_bound,
        "bin_centers": bin_centers,
        "bins_edges": bins_edges
    }


# =========================================================================
# 2. 绘图与导出
# =========================================================================

def plot_jitter_result_excitation(res, pair_id, animal_id, output_folder):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = res['bin_centers']
    bar_width = np.mean(np.diff(bins))

    ax.bar(bins, res['raw_cch'], width=bar_width, color='gray', alpha=0.5, label='Raw CCH', edgecolor='none')
    ax.plot(bins, res['baseline_mean'], color='black', linewidth=1.5, label='Expected Baseline')
    ax.plot(bins, res['upper_bound'], color='red', linestyle='--', linewidth=1.2, label='99.5% Bound')

    ax.axvspan(0.001, 0.004, color='orange', alpha=0.1, label='Window (1-4ms)')

    if res['is_connected']:
        window_mask = (bins >= 0.001) & (bins <= 0.004)
        ax.fill_between(bins[window_mask],
                        res['raw_cch'][window_mask],
                        res['baseline_mean'][window_mask],
                        where=(res['raw_cch'][window_mask] > res['baseline_mean'][window_mask]),
                        color='red', alpha=0.6, label='Excess Spikes')

    ax.set_title(f"PYR->PYR Excitation: {animal_id} - {pair_id}", fontsize=14)
    ax.set_xlabel("Time Lag (s)", fontsize=12)
    ax.set_ylabel("Spike Count", fontsize=12)
    ax.set_xlim(-0.03, 0.03)
    ax.legend(loc='upper right', frameon=True)

    status_text = f"Connection Detected\nSTP: {res['stp']:.4f}" if res['is_connected'] else "No Connection"
    status_color = 'red' if res['is_connected'] else 'gray'
    ax.text(0.05, 0.95, status_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', fc=status_color, alpha=0.2))

    suffix = "Synapse" if res['is_connected'] else "No_Synapse"
    date_str = datetime.now().strftime('%Y%m%d')
    filename = f"{date_str}_Jitter_PYR-PYR_{animal_id}_{pair_id}_{suffix}.png"
    output_path = os.path.join(output_folder, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_details_to_excel(res, filename, output_folder):
    df = pd.DataFrame({
        'time_lag_s': res['bin_centers'],
        'raw_cch': res['raw_cch'],
        'baseline_mean': res['baseline_mean'],
        'upper_bound_99.5pct': res['upper_bound']
    })
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
    JITTER_ITER = 1000
    MIN_SPIKES = 500

    source_folder, output_folder, files_to_process = get_user_inputs()
    if not all([source_folder, output_folder, files_to_process]): return

    date_str = datetime.now().strftime('%Y%m%d')
    # 标注 PYR_PYR
    main_output_name = f"{date_str}_PYR_PYR_Excitatory_Jittering_Analysis"
    main_output_path = os.path.join(output_folder, main_output_name)
    os.makedirs(main_output_path, exist_ok=True)

    summary_file_path = os.path.join(main_output_path, f"{date_str}_PYR_PYR_Summary_STP.xlsx")
    summary_data = []

    print(f"Results will be saved to: {main_output_path}")
    print(f"Analyzing: PYR (Pre) -> PYR (Post)")
    print(f"Algorithm: Jittering Resampling ({JITTER_ITER} iterations)")
    print("Warning: PYR-PYR connections are typically rare in vivo.")

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

                # 提取 PYR
                pyr_neurons = {}
                for i, n_type in enumerate(neuron_types):
                    if n_type == 1:  # PYR
                        spikes = df.iloc[2:, i].dropna().to_numpy()
                        if len(spikes) >= MIN_SPIKES:
                            pyr_neurons[neuron_ids[i]] = spikes

                pyr_ids = list(pyr_neurons.keys())
                # PYR -> PYR 配对
                pairs_count = len(pyr_ids) * (len(pyr_ids) - 1)

                if pairs_count == 0:
                    pbar_files.update(1)
                    continue

                with tqdm(total=pairs_count, desc=f"  Analysing PYR->PYR ({animal_id})", leave=False) as pbar_pairs:
                    for pre_id in pyr_ids:
                        for post_id in pyr_ids:
                            # 必须跳过自连接
                            if pre_id == post_id: continue

                            pair_id = f"PrePYR_{pre_id}-PostPYR_{post_id}"
                            pre_spikes = pyr_neurons[pre_id]
                            post_spikes = pyr_neurons[post_id]

                            # --- 运行 Jitter 分析 (兴奋性) ---
                            result = run_jittering_test_excitation(
                                pre_spikes,
                                post_spikes,
                                n_iter=JITTER_ITER
                            )

                            # --- 绘图 ---
                            plot_jitter_result_excitation(result, pair_id, animal_id, main_output_path)

                            # --- 保存原始数据 ---
                            suffix = "Synapse" if result['is_connected'] else "No_Synapse"
                            save_details_to_excel(result, f"{pair_id}_{suffix}_data.xlsx", detail_folder)

                            # --- 记录汇总 ---
                            if result['is_connected']:
                                summary_data.append({
                                    "Animal_ID": animal_id,
                                    "Pre_PYR_ID": pre_id,
                                    "Post_PYR_ID": post_id,
                                    "Spike_Transmission_Probability": result['stp'],
                                    "Pre_Spike_Count": len(pre_spikes),
                                    "Method": "Jittering_1000x"
                                })

                            pbar_pairs.update(1)

                pbar_files.update(1)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(summary_file_path, index=False)
            print(f"\nSummary saved successfully: {summary_file_path}")
        else:
            print("\nNo PYR->PYR excitatory connections detected.")
            pd.DataFrame(columns=["Animal_ID", "Pre_PYR_ID", "Post_PYR_ID", "Spike_Transmission_Probability"]).to_excel(
                summary_file_path, index=False)

        messagebox.showinfo("Complete", "PYR-PYR Analysis Finished!")

    except Exception as e:
        import traceback
        traceback.print_exc()
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


if __name__ == "__main__":
    main()