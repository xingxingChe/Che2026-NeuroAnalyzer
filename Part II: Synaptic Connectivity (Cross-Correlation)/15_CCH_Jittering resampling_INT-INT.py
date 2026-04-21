# -*- coding: utf-8 -*-
"""
INT-INT 抑制性突触连接分析脚本 (V_Jittering)
--------------------------------------------------
算法依据: Fujisawa et al., 2008 (Nature Neuroscience)
核心方法: Jittering Resampling (抖动重采样)
量化指标: Spike Suppression Probability (SSP)

功能:
1. 通过 [-5ms, +5ms] 的随机抖动生成 1000 组替代数据。
2. 计算抖动数据的均值作为“期望基线” (Expected Baseline)。
3. 计算抖动数据的 0.5% 分位数作为“显著性下界” (Lower Significance Band)。
4. 如果原始 CCH 在 1-4ms 窗口内显著低于下界，则判定为抑制性连接。
5. 导出 Excel 和图表。
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
# 1. 核心分析函数 (Jittering Algorithm)
# =========================================================================

def calculate_cch(spike_train1, spike_train2, bin_size=0.0004, lag=0.05):
    """
    计算互相关图 (CCH)。
    spike_train1: 突触前 (Pre)
    spike_train2: 突触后 (Post)
    """
    num_bins = int(2 * lag / bin_size)
    if num_bins % 2 == 0:
        num_bins += 1
    bins = np.linspace(-lag, lag, num_bins + 1)

    # 计算时间差
    # 优化：为了加速，这里不使用循环，而是利用广播或搜索
    # 但由于脉冲数不一致，最稳妥且易读的仍是双循环或 Searchsorted
    # 鉴于 jitter 循环在外部，这里保持简单逻辑，或者使用 numpy 直方图加速

    all_diffs = []
    # 简单的遍历逻辑 (对于 Jittering 内部循环，我们需要更快的速度，见 run_jitter_test)
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
    """
    为 Jitter 循环优化的快速 CCH 计算函数。
    只计算 counts，不返回 bin_centers。
    """
    all_diffs = []
    # 预筛选以减少计算量
    # 这是一个简单的优化，针对典型的稀疏脉冲数据
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


def run_jittering_test(pre_spikes, post_spikes, n_iter=1000, jitter_window=0.005):
    """
    [核心算法] Fujisawa Jittering Method
    1. 计算原始 CCH
    2. 循环 1000 次：
       - 将 pre_spikes 在 [-5ms, +5ms] 内随机抖动
       - 重新计算 CCH
    3. 统计 1000 个替代 CCH 的分布
    """
    # 1. 基础参数
    LAG = 0.05
    BIN_SIZE = 0.0004
    # 定义分析窗口：Fujisawa 2008 使用 1-4ms
    ANALYSIS_WINDOW_MIN = 0.0010
    ANALYSIS_WINDOW_MAX = 0.0040

    # 2. 计算原始 CCH
    raw_cch, bin_centers, bins_edges = calculate_cch(pre_spikes, post_spikes, BIN_SIZE, LAG)

    # 3. Jittering 循环 (生成替代数据)
    surrogate_cchs = np.zeros((n_iter, len(raw_cch)))

    # 预先生成随机偏移量矩阵以加速 (n_iter x n_spikes)
    # 这样避免在循环内反复调用 random
    noise = np.random.uniform(-jitter_window, jitter_window, (n_iter, len(pre_spikes)))

    # 注意：为了进度条不刷屏，这里不加 tqdm，或者只在外部加
    for i in range(n_iter):
        jittered_pre = pre_spikes + noise[i, :]
        # 必须重新排序，尽管histogram通常不强制要求排序，但为了逻辑严谨
        jittered_pre.sort()
        surrogate_cchs[i, :] = fast_cch_for_jitter(jittered_pre, post_spikes, bins_edges, LAG)

    # 4. 统计分析
    # 基线 (Expected Baseline): 替代数据的均值
    baseline_mean = np.mean(surrogate_cchs, axis=0)

    # 显著性边界 (Global Bands / Pointwise Bands)
    # Fujisawa 使用 global bands，这里为了简化计算且保持严谨，
    # 我们使用逐点的 0.5% 分位数 (Pointwise 0.5 percentile) 作为下界。
    # 意味着：只有 0.5% 的随机情况会比这个还低。如果原始数据比这还低，那肯定显著。
    lower_bound = np.percentile(surrogate_cchs, 0.5, axis=0)

    # 5. 判定抑制性连接
    # 逻辑：在分析窗口 (1-4ms) 内，原始 CCH 是否显著低于下界 (lower_bound)
    # 或者是 原始 CCH 是否显著低于基线 (mean) 且 P 值极低

    window_mask = (bin_centers >= ANALYSIS_WINDOW_MIN) & (bin_centers <= ANALYSIS_WINDOW_MAX)

    # 检查窗口内是否有 bin 突破了下界 (Raw < Lower Bound)
    # 并且该 bin 的计数要显著少于基线 (这在 definition 上是必然的)
    is_inhibited = np.any(raw_cch[window_mask] < lower_bound[window_mask])

    # 6. 计算 Spike Suppression Probability (SSP)
    # 公式: (Expected_Baseline - Observed) / N_pre
    # 只计算分析窗口内的亏损
    deficit_spikes = np.sum(baseline_mean[window_mask]) - np.sum(raw_cch[window_mask])

    # 如果算出来是负数（没有抑制），则归零
    if deficit_spikes < 0:
        deficit_spikes = 0.0

    ssp = deficit_spikes / len(pre_spikes) if len(pre_spikes) > 0 else 0.0

    return {
        "is_inhibited": is_inhibited,
        "ssp": ssp,
        "raw_cch": raw_cch,
        "baseline_mean": baseline_mean,
        "lower_bound": lower_bound,
        "bin_centers": bin_centers,
        "bins_edges": bins_edges
    }


# =========================================================================
# 2. 绘图与导出函数
# =========================================================================

def plot_jitter_result(res, pair_id, animal_id, output_folder):
    """绘制包含显著性边界的 Jitter 分析图"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = res['bin_centers']
    bar_width = np.mean(np.diff(bins))

    # 1. 绘制原始 CCH (灰色柱状)
    ax.bar(bins, res['raw_cch'], width=bar_width, color='gray', alpha=0.5, label='Raw CCH', edgecolor='none')

    # 2. 绘制 Jitter 基线 (黑色实线)
    ax.plot(bins, res['baseline_mean'], color='black', linewidth=1.5, label='Expected Baseline (Jitter Mean)')

    # 3. 绘制显著性下界 (红色虚线)
    ax.plot(bins, res['lower_bound'], color='red', linestyle='--', linewidth=1.2, label='0.5% Significance Bound')

    # 4. 标记分析窗口 (1-4ms)
    ax.axvspan(0.001, 0.004, color='blue', alpha=0.1, label='Analysis Window (1-4ms)')

    # 5. 高亮显示抑制区域 (Deficit)
    if res['is_inhibited']:
        # 找到窗口内低于基线的部分
        window_mask = (bins >= 0.001) & (bins <= 0.004)
        ax.fill_between(bins[window_mask],
                        res['raw_cch'][window_mask],
                        res['baseline_mean'][window_mask],
                        where=(res['baseline_mean'][window_mask] > res['raw_cch'][window_mask]),
                        color='blue', alpha=0.6, label='Suppressed Spikes')

    ax.set_title(f"Jitter Inhibition Analysis: {animal_id} - {pair_id}", fontsize=14)
    ax.set_xlabel("Time Lag (s)", fontsize=12)
    ax.set_ylabel("Spike Count", fontsize=12)
    ax.set_xlim(-0.03, 0.03)  # 聚焦中心
    ax.legend(loc='upper right', frameon=True)

    # 状态标签
    status_text = f"Suppression Detected\nSSP: {res['ssp']:.4f}" if res['is_inhibited'] else "No Suppression"
    status_color = 'blue' if res['is_inhibited'] else 'gray'
    ax.text(0.05, 0.95, status_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', fc=status_color, alpha=0.2))

    # 保存
    suffix = "Inhibition" if res['is_inhibited'] else "No_Inhibition"
    date_str = datetime.now().strftime('%Y%m%d')
    filename = f"{date_str}_Jitter_INT-INT_{animal_id}_{pair_id}_{suffix}.png"
    output_path = os.path.join(output_folder, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_details_to_excel(res, filename, output_folder):
    """保存详细的 CCH 数据以便后续重画"""
    df = pd.DataFrame({
        'time_lag_s': res['bin_centers'],
        'raw_cch': res['raw_cch'],
        'baseline_mean': res['baseline_mean'],
        'lower_bound_0.5pct': res['lower_bound']
    })
    path = os.path.join(output_folder, filename)
    df.to_excel(path, index=False)


# =========================================================================
# 3. GUI 交互 (保持原样)
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
    # 配置参数
    JITTER_ITER = 1000  # 抖动次数，Fujisawa 推荐 1000
    MIN_SPIKES = 500  # 最小脉冲数要求

    source_folder, output_folder, files_to_process = get_user_inputs()
    if not all([source_folder, output_folder, files_to_process]): return

    date_str = datetime.now().strftime('%Y%m%d')
    main_output_name = f"{date_str}_INT_INT_Jittering_Analysis"
    main_output_path = os.path.join(output_folder, main_output_name)
    os.makedirs(main_output_path, exist_ok=True)

    summary_file_path = os.path.join(main_output_path, f"{date_str}_Summary_SSP.xlsx")
    summary_data = []

    print(f"Results will be saved to: {main_output_path}")
    print(f"Algorithm: Jittering Resampling ({JITTER_ITER} iterations)")
    print(f"Metric: Spike Suppression Probability (SSP)")

    try:
        # 总体进度条
        with tqdm(total=len(files_to_process), desc="Processing Files") as pbar_files:
            for filename in files_to_process:
                animal_id = os.path.splitext(filename)[0]
                pbar_files.set_description(f"File: {animal_id}")

                # 创建详情文件夹
                detail_folder = os.path.join(main_output_path, f"{animal_id}_Details")
                os.makedirs(detail_folder, exist_ok=True)

                # 读取数据
                file_path = os.path.join(source_folder, filename)
                df = pd.read_excel(file_path, header=None)
                neuron_ids = df.iloc[0, :].astype(str)
                neuron_types = df.iloc[1, :].astype(int)

                # 提取 INT 数据
                int_neurons = {}
                for i, n_type in enumerate(neuron_types):
                    if n_type == 0:  # INT
                        spikes = df.iloc[2:, i].dropna().to_numpy()
                        if len(spikes) >= MIN_SPIKES:
                            int_neurons[neuron_ids[i]] = spikes

                int_ids = list(int_neurons.keys())
                pairs_count = len(int_ids) * (len(int_ids) - 1)

                if pairs_count == 0:
                    pbar_files.update(1)
                    continue

                # 配对循环
                with tqdm(total=pairs_count, desc=f"  Analysing Pairs ({animal_id})", leave=False) as pbar_pairs:
                    for pre_id in int_ids:
                        for post_id in int_ids:
                            if pre_id == post_id: continue

                            pair_id = f"Pre_{pre_id}-Post_{post_id}"
                            pre_spikes = int_neurons[pre_id]
                            post_spikes = int_neurons[post_id]

                            # --- 运行 Jitter 分析 ---
                            result = run_jittering_test(
                                pre_spikes,
                                post_spikes,
                                n_iter=JITTER_ITER
                            )

                            # --- 绘图 ---
                            plot_jitter_result(result, pair_id, animal_id, main_output_path)

                            # --- 保存原始数据 ---
                            suffix = "Inhibition" if result['is_inhibited'] else "No_Inhibition"
                            save_details_to_excel(result, f"{pair_id}_{suffix}_data.xlsx", detail_folder)

                            # --- 记录汇总 ---
                            if result['is_inhibited']:
                                summary_data.append({
                                    "Animal_ID": animal_id,
                                    "Pre_INT_ID": pre_id,
                                    "Post_INT_ID": post_id,
                                    "Spike_Suppression_Probability": result['ssp'],
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
            print("\nNo inhibitory connections detected across all files.")
            pd.DataFrame(columns=["Animal_ID", "Pre_INT_ID", "Post_INT_ID", "Spike_Suppression_Probability"]).to_excel(
                summary_file_path, index=False)

        messagebox.showinfo("Complete", "Analysis Finished!\nCheck output folder for results.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


if __name__ == "__main__":
    main()