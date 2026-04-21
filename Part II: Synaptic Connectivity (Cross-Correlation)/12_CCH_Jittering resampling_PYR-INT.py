# -*- coding: utf-8 -*-
"""
PYR-INT 兴奋性突触连接分析脚本 (V_Jittering)
--------------------------------------------------
分析方向: Pyramidal Neuron (Pre) -> Interneuron (Post)
核心算法: Fujisawa et al., 2008 (Nature Neuroscience) - Jittering Resampling
量化指标: Spike Transmission Probability (STP)

功能:
1. 识别 Type 1 (PYR) 和 Type 0 (INT) 神经元。
2. 通过 [-5ms, +5ms] 随机抖动生成 1000 组替代数据。
3. 构建 99.5% 显著性上界 (Upper Bound)。
4. 计算 Spike Transmission Probability (STP)。
5. 导出 Excel 和图表 (红色高亮兴奋峰)。
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
# 1. 核心分析函数 (Jittering Algorithm - Excitation)
# =========================================================================

def calculate_cch(spike_train1, spike_train2, bin_size=0.0004, lag=0.05):
    """
    计算互相关图 (CCH)。
    spike_train1: 突触前 (PYR)
    spike_train2: 突触后 (INT)
    """
    num_bins = int(2 * lag / bin_size)
    if num_bins % 2 == 0:
        num_bins += 1
    bins = np.linspace(-lag, lag, num_bins + 1)

    all_diffs = []
    for t1 in spike_train1:
        # 只选取 lag 范围内的 spikes
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
    """Jitter 循环专用快速 CCH"""
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
    [核心算法] Fujisawa Jittering Method for Excitation
    """
    # 参数设置
    LAG = 0.05
    BIN_SIZE = 0.0004
    # 分析窗口：典型的单突触兴奋延迟 1-4ms
    ANALYSIS_WINDOW_MIN = 0.0010
    ANALYSIS_WINDOW_MAX = 0.0040

    # 原始 CCH
    raw_cch, bin_centers, bins_edges = calculate_cch(pre_spikes, post_spikes, BIN_SIZE, LAG)

    # Jittering 循环
    surrogate_cchs = np.zeros((n_iter, len(raw_cch)))
    # 预生成随机数加速
    noise = np.random.uniform(-jitter_window, jitter_window, (n_iter, len(pre_spikes)))

    for i in range(n_iter):
        jittered_pre = pre_spikes + noise[i, :]
        jittered_pre.sort()
        surrogate_cchs[i, :] = fast_cch_for_jitter(jittered_pre, post_spikes, bins_edges, LAG)

    # 统计分析
    baseline_mean = np.mean(surrogate_cchs, axis=0)

    # -------------------------------------------------------------
    # 关键修改点：计算上界 (Upper Bound)
    # Fujisawa (2008) 使用 99% global band，这里使用 99.5% pointwise 作为严格判定
    # -------------------------------------------------------------
    upper_bound = np.percentile(surrogate_cchs, 99.5, axis=0)

    # 判定：是否显著兴奋
    # 逻辑：在分析窗口内，原始 CCH 是否突破了上界
    window_mask = (bin_centers >= ANALYSIS_WINDOW_MIN) & (bin_centers <= ANALYSIS_WINDOW_MAX)
    is_connected = np.any(raw_cch[window_mask] > upper_bound[window_mask])

    # 计算 Spike Transmission Probability (STP)
    # 公式：(Observed - Expected_Baseline) / N_pre
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
# 2. 绘图与导出函数
# =========================================================================

def plot_jitter_result_excitation(res, pair_id, animal_id, output_folder):
    """绘制兴奋性连接结果图 (红色系)"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = res['bin_centers']
    bar_width = np.mean(np.diff(bins))

    # CCH
    ax.bar(bins, res['raw_cch'], width=bar_width, color='gray', alpha=0.5, label='Raw CCH', edgecolor='none')
    # 基线
    ax.plot(bins, res['baseline_mean'], color='black', linewidth=1.5, label='Expected Baseline (Jitter Mean)')
    # 上界 (红色虚线)
    ax.plot(bins, res['upper_bound'], color='red', linestyle='--', linewidth=1.2, label='99.5% Significance Bound')

    # 分析窗口
    ax.axvspan(0.001, 0.004, color='orange', alpha=0.1, label='Analysis Window (1-4ms)')

    # 高亮显著区域 (Excess)
    if res['is_connected']:
        window_mask = (bins >= 0.001) & (bins <= 0.004)
        ax.fill_between(bins[window_mask],
                        res['raw_cch'][window_mask],
                        res['baseline_mean'][window_mask],
                        where=(res['raw_cch'][window_mask] > res['baseline_mean'][window_mask]),
                        color='red', alpha=0.6, label='Excess Spikes (Transmission)')

    ax.set_title(f"PYR->INT Excitation: {animal_id} - {pair_id}", fontsize=14)
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
    filename = f"{date_str}_Jitter_PYR-INT_{animal_id}_{pair_id}_{suffix}.png"
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
# 3. GUI 交互 (保持一致)
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
    JITTER_ITER = 1000
    MIN_SPIKES = 500

    source_folder, output_folder, files_to_process = get_user_inputs()
    if not all([source_folder, output_folder, files_to_process]): return

    date_str = datetime.now().strftime('%Y%m%d')
    # 明确标注这是 PYR_INT 的 Jittering 分析
    main_output_name = f"{date_str}_PYR_INT_Excitatory_Jittering_Analysis"
    main_output_path = os.path.join(output_folder, main_output_name)
    os.makedirs(main_output_path, exist_ok=True)

    summary_file_path = os.path.join(main_output_path, f"{date_str}_PYR_INT_Summary_STP.xlsx")
    summary_data = []

    print(f"Results will be saved to: {main_output_path}")
    print(f"Analyzing: PYR (Pre) -> INT (Post)")
    print(f"Algorithm: Jittering Resampling ({JITTER_ITER} iterations) - Excitation Detection")

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

                # 分类提取神经元
                pyr_neurons = {}  # Pre (Type 1)
                int_neurons = {}  # Post (Type 0)

                for i, n_type in enumerate(neuron_types):
                    spikes = df.iloc[2:, i].dropna().to_numpy()
                    if len(spikes) >= MIN_SPIKES:
                        if n_type == 1:  # PYR
                            pyr_neurons[neuron_ids[i]] = spikes
                        elif n_type == 0:  # INT
                            int_neurons[neuron_ids[i]] = spikes

                pyr_ids = list(pyr_neurons.keys())
                int_ids = list(int_neurons.keys())

                pairs_count = len(pyr_ids) * len(int_ids)

                if pairs_count == 0:
                    pbar_files.update(1)
                    continue

                # 配对循环: PYR (Pre) -> INT (Post)
                with tqdm(total=pairs_count, desc=f"  Analysing PYR->INT ({animal_id})", leave=False) as pbar_pairs:
                    for pre_id in pyr_ids:
                        for post_id in int_ids:
                            # 即使ID相同也不用跳过 (不同类型ID通常不同)
                            if pre_id == post_id: continue

                            pair_id = f"PrePYR_{pre_id}-PostINT_{post_id}"
                            pre_spikes = pyr_neurons[pre_id]
                            post_spikes = int_neurons[post_id]

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
                                    "Post_INT_ID": post_id,
                                    "Spike_Transmission_Probability": result['stp'],
                                    "Pre_Spike_Count": len(pre_spikes),
                                    "Method": "Jittering_1000x"
                                })

                            pbar_pairs.update(1)

                pbar_files.update(1)

        # 保存总表
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(summary_file_path, index=False)
            print(f"\nSummary saved successfully: {summary_file_path}")
        else:
            print("\nNo PYR->INT excitatory connections detected.")
            pd.DataFrame(columns=["Animal_ID", "Pre_PYR_ID", "Post_INT_ID", "Spike_Transmission_Probability"]).to_excel(
                summary_file_path, index=False)

        messagebox.showinfo("Complete", "PYR-INT Excitation Analysis Finished!")

    except Exception as e:
        import traceback
        traceback.print_exc()
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


if __name__ == "__main__":
    main()