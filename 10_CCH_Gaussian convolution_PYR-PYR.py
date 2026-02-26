# -*- coding: utf-8 -*-
"""
神经元放电互相关分析脚本 (V5 - PYR-PYR 专用版)
修改说明:
1. 基于 V4 (严格检验版) 修改。
2. 逻辑变更为检测 锥体细胞 (Source) -> 锥体细胞 (Target) 的突触连接。
3. 自动跳过自身对自身的比较 (Autocorrelation)。
4. 保持 Sauer & Bartos (2022) 的严格统计阈值。
"""

# ---------------------------------------------------
# 1. 导入所需库
# ---------------------------------------------------
import os
import sys
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
# 2. 核心分析函数 (保持不变)
# ---------------------------------------------------

def calculate_cch(spike_train1, spike_train2, bin_size=0.0004, lag=0.05):
    """
    计算两个神经元放电序列的互相关图 (CCH)。
    spike_train1: 参考神经元 (Presynaptic/Source)
    spike_train2: 目标神经元 (Postsynaptic/Target)
    """
    num_bins = int(2 * lag / bin_size)
    if num_bins % 2 == 0:
        num_bins += 1
    bins = np.linspace(-lag, lag, num_bins + 1)
    cch = np.zeros(num_bins)
    for spike_time in spike_train1:
        time_diffs = spike_train2 - spike_time
        hist, _ = np.histogram(time_diffs, bins=bins)
        cch += hist
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return cch, bin_centers


def create_hollow_gaussian(sigma_ms, hollow_fraction, bin_size_ms):
    """创建一个部分空心的高斯核用于基线计算。"""
    sigma_bins = sigma_ms / bin_size_ms
    size = int(sigma_bins * 8)
    x = np.arange(-size, size + 1)
    gaussian = np.exp(-(x ** 2) / (2 * sigma_bins ** 2))
    hollow_gaussian = np.exp(-(x ** 2) / (2 * (sigma_bins * (1 - hollow_fraction)) ** 2))
    kernel = gaussian - hollow_gaussian
    return kernel / np.sum(kernel)


def calculate_baseline(cch, sigma=10, hollow_fraction=0.6, bin_size=0.4):
    """使用部分空心的高斯核卷积来计算 CCH 的基线。"""
    kernel = create_hollow_gaussian(sigma, hollow_fraction, bin_size)
    baseline = convolve(cch, kernel, mode='same')
    return baseline


def poisson_prob_continuity_corrected(n, mu):
    """根据论文中的公式计算带有连续性校正的泊松概率。"""
    if mu <= 0:
        return 1.0 if n <= 0 else 0.0
    prob = 1 - poisson.cdf(n - 1, mu)
    correction = 0.5 * poisson.pmf(n, mu)
    return prob - correction


def perform_significance_test(cch, baseline, bins, p_syn_thresh, p_causal_thresh):
    """
    对 CCH 进行显著性检验以判断是否存在突触连接。
    """
    # 窗口定义保持不变：0.8ms - 2.8ms 为典型的单突触延迟窗口
    mono_window = (bins >= 0.0008) & (bins <= 0.0028)
    anti_causal_window = (bins >= -0.002) & (bins <= 0.0)

    n_syn = np.sum(cch[mono_window])
    b_syn = np.sum(baseline[mono_window])
    n_anti = np.sum(cch[anti_causal_window])

    p_syn = poisson_prob_continuity_corrected(n_syn, b_syn)
    p_causal = poisson_prob_continuity_corrected(n_syn, n_anti)

    is_connected = (p_syn < p_syn_thresh) and (p_causal < p_causal_thresh)

    summary = (
        f"Significance Test Results (PYR->PYR):\n"
        f"-----------------------------------\n"
        f"1. Monosynaptic Window Analysis (0.8ms to 2.8ms):\n"
        f"   - Observed Spikes (n_syn): {n_syn}\n"
        f"   - Expected Baseline (b_syn): {b_syn:.4f}\n"
        f"   - Significance Probability (P_syn): {p_syn:.6f}\n"
        f"   - Criterion: P_syn < {p_syn_thresh}\n\n"
        f"2. Causality Analysis (vs -2ms to 0ms):\n"
        f"   - Anti-causal Spikes (n_anti): {n_anti}\n"
        f"   - Significance Probability (P_causal): {p_causal:.6f}\n"
        f"   - Criterion: P_causal < {p_causal_thresh}\n\n"
        f"Conclusion:\n"
        f"-----------------------------------\n"
        f"A significant synaptic connection was {'DETECTED' if is_connected else 'NOT DETECTED'}.\n"
    )
    return is_connected, p_syn, p_causal, summary


def calculate_spike_transmission_prob(cch, baseline, bins, source_spike_count):
    """计算尖峰传递概率。"""
    mono_window = (bins >= 0.0008) & (bins <= 0.0028)
    excess_spikes = np.sum(cch[mono_window]) - np.sum(baseline[mono_window])
    if source_spike_count == 0 or excess_spikes < 0:
        return 0.0
    return excess_spikes / source_spike_count


# ---------------------------------------------------
# 3. 绘图与文件导出函数
# ---------------------------------------------------

def plot_and_save_cch(cch, baseline, bins, pair_id, animal_id, is_connected, output_folder):
    """绘制 CCH 图并保存。"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = np.mean(np.diff(bins))

    # 绘图颜色微调：PYR-PYR 常用深蓝色或黑色表示
    ax.bar(bins, cch, width=bar_width, color='#2c3e50', alpha=0.8, label='Raw CCH')
    ax.plot(bins, baseline, color='#e74c3c', linewidth=2, linestyle='--', label='Baseline')
    ax.axvspan(0.0008, 0.0028, color='blue', alpha=0.1, label='Monosynaptic Window')

    ax.set_title(f"PYR-PYR CCH: {animal_id} - {pair_id}", fontsize=16)
    ax.set_xlabel("Time Lag (s) [Source -> Target]", fontsize=12)
    ax.set_ylabel("Spike Count", fontsize=12)
    ax.legend()
    ax.set_xlim(-0.05, 0.05)

    status_text = "Synapse Detected" if is_connected else "No Synapse Detected"
    status_color = 'green' if is_connected else 'darkred'
    ax.text(0.95, 0.95, status_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', fc=status_color, alpha=0.3))

    suffix = "Synapse" if is_connected else "No_Synapse"
    date_str = datetime.now().strftime('%Y%m%d')
    # 文件名增加 PYR-PYR 标识
    filename = f"{date_str}_PYR-PYR_CCH_{animal_id}_{pair_id}_{suffix}.png"
    output_path = os.path.join(output_folder, filename)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


def save_numeric_data(data, filename, output_folder):
    """将数值数据保存为 .xlsx 文件。"""
    df = pd.DataFrame(data)
    output_path = os.path.join(output_folder, filename)
    df.to_excel(output_path, index=False)


def save_text_data(text, filename, output_folder):
    """将文本数据保存为 .txt 文件。"""
    output_path = os.path.join(output_folder, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)


# ---------------------------------------------------
# 4. 用户界面 (GUI) 函数 (保持不变)
# ---------------------------------------------------

class CustomSortDialog(Toplevel):
    """一个让用户自定义文件顺序的对话框。"""

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
    """通过GUI获取所有用户输入。"""
    root = tk.Tk()
    root.withdraw()
    source_dir = filedialog.askdirectory(title="请选择包含原始数据 (.xlsx) 的文件夹")
    if not source_dir:
        messagebox.showerror("错误", "未选择文件夹，程序将退出。")
        return None, None, None
    try:
        files = sorted([f for f in os.listdir(source_dir) if f.endswith('.xlsx')])
        if not files:
            messagebox.showerror("错误", f"文件夹 '{source_dir}' 中未找到 .xlsx 文件。")
            return None, None, None
    except Exception as e:
        messagebox.showerror("错误", f"读取文件列表失败: {e}")
        return None, None, None
    sort_choice = simpledialog.askstring("文件顺序", "请选择文件分析顺序:\n1: 升序 (默认)\n2: 降序\n3: 自定义",
                                         parent=root)
    if sort_choice == '2':
        files.sort(reverse=True)
    elif sort_choice == '3':
        dialog = CustomSortDialog(root, files)
        files = dialog.result
    else:
        files.sort()
    order_message = "代码将按以下顺序处理文件:\n\n" + "\n".join(files)
    messagebox.showinfo("文件处理顺序", order_message)
    output_dir = filedialog.askdirectory(title="请选择一个文件夹用于存放分析结果")
    if not output_dir:
        messagebox.showerror("错误", "未选择输出文件夹，程序将退出。")
        return None, None, None
    return source_dir, output_dir, files


# ---------------------------------------------------
# 5. 主程序逻辑 (核心修改区域)
# ---------------------------------------------------

def main():
    """主执行函数"""

    # =========================================================================
    # *** 检验水平参数 ***
    # 保持 Sauer & Bartos (2022) 的严格检验标准
    P_SYN_THRESHOLD = 0.001
    P_CAUSAL_THRESHOLD = 0.0026
    # =========================================================================

    source_folder, output_folder, files_to_process = get_user_inputs()
    if not all([source_folder, output_folder, files_to_process]):
        print("用户取消了操作或输入不完整，程序退出。")
        return

    date_str = datetime.now().strftime('%Y%m%d')
    # 修改输出文件夹名称，明确标记为 PYR-PYR 分析
    main_output_folder_name = f"{date_str}_PYR-PYR_Analysis_Results"
    main_output_path = os.path.join(output_folder, main_output_folder_name)
    os.makedirs(main_output_path, exist_ok=True)

    summary_file_name = f"{date_str}_PYR-PYR_Connection_Summary.xlsx"
    summary_file_path = os.path.join(main_output_path, summary_file_name)

    # 用于在内存中存储结果的列表
    summary_data = []

    print(f"Results will be saved to: {main_output_path}")
    print(f"Using STRICT significance thresholds: P_syn < {P_SYN_THRESHOLD}, P_causal < {P_CAUSAL_THRESHOLD}")
    print("Mode: Pyramidal (Source) -> Pyramidal (Target)")

    try:
        with tqdm(total=len(files_to_process), desc="Overall Progress") as pbar_files:
            for filename in files_to_process:
                animal_id = os.path.splitext(filename)[0]
                pbar_files.set_description(f"Processing: {animal_id}")

                numeric_results_folder_name = f"{date_str}_PYR-PYR_Numeric_{animal_id}"
                numeric_results_path = os.path.join(main_output_path, numeric_results_folder_name)
                os.makedirs(numeric_results_path, exist_ok=True)

                file_path = os.path.join(source_folder, filename)
                df = pd.read_excel(file_path, header=None)

                neuron_ids = df.iloc[0, :].astype(str)
                neuron_types = df.iloc[1, :].astype(int)

                pyr_neurons = {}
                # 只读取 PYR (Type 1)，忽略 INT (Type 0)
                for i, n_type in enumerate(neuron_types):
                    neuron_id = neuron_ids[i]
                    spike_times = df.iloc[2:, i].dropna().to_numpy()
                    # 保持最小 spike 数量限制，保证统计效力
                    if len(spike_times) >= 500:
                        if n_type == 1:
                            pyr_neurons[neuron_id] = spike_times

                # PYR 到 PYR 的循环逻辑
                # 需要计算排列数 P(n, 2) = n * (n-1)
                pyr_ids = list(pyr_neurons.keys())
                num_pyr = len(pyr_ids)
                total_pairs = num_pyr * (num_pyr - 1) if num_pyr > 1 else 0

                if total_pairs == 0:
                    pbar_files.update(1)
                    continue

                with tqdm(total=total_pairs, desc=f"{animal_id} PYR Pairs", leave=False) as pbar_pairs:
                    # 外层循环：Source (Pre-synaptic)
                    for source_id in pyr_ids:
                        source_spikes = pyr_neurons[source_id]

                        # 内层循环：Target (Post-synaptic)
                        for target_id in pyr_ids:
                            # *** 关键修改：跳过自己对自己 ***
                            if source_id == target_id:
                                continue

                            target_spikes = pyr_neurons[target_id]

                            pair_id = f"Source_{source_id}-Target_{target_id}"
                            pbar_pairs.set_description(f"Analyzing: {pair_id}")

                            # 计算 CCH (Source -> Target)
                            cch, bins = calculate_cch(source_spikes, target_spikes)
                            baseline = calculate_baseline(cch)

                            is_connected, _, _, sig_summary = perform_significance_test(
                                cch, baseline, bins, P_SYN_THRESHOLD, P_CAUSAL_THRESHOLD
                            )

                            plot_and_save_cch(cch, baseline, bins, pair_id, animal_id, is_connected, main_output_path)

                            suffix = "Synapse" if is_connected else "No_Synapse"
                            save_numeric_data(
                                {'time_lag_s': bins, 'spike_count': cch},
                                f"{date_str}_CCH_{pair_id}_{suffix}_raw_data.xlsx", numeric_results_path
                            )
                            # 仅在检测到连接或为了调试时保存Baseline数据，避免文件过多(此处保持全部保存)
                            save_numeric_data(
                                {'time_lag_s': bins, 'baseline': baseline},
                                f"{date_str}_Baseline_{pair_id}_{suffix}_raw_data.xlsx", numeric_results_path
                            )
                            save_text_data(
                                sig_summary,
                                f"{date_str}_Significance_Test_{pair_id}_{suffix}.txt", numeric_results_path
                            )

                            if is_connected:
                                transmission_prob = calculate_spike_transmission_prob(
                                    cch, baseline, bins, len(source_spikes)
                                )
                                # 将结果追加到内存列表
                                summary_data.append({
                                    "Animal_ID": animal_id,
                                    "Source_Neuron": source_id,
                                    "Target_Neuron": target_id,
                                    "Spike_Transmission_Probability": transmission_prob
                                })

                            pbar_pairs.update(1)

                pbar_files.update(1)

        # --- 所有文件处理完毕后，一次性写入总结文件 ---
        if summary_data:
            final_summary_df = pd.DataFrame(summary_data)
            # 调整列顺序以符合阅读习惯
            final_summary_df = final_summary_df[
                ["Animal_ID", "Source_Neuron", "Target_Neuron", "Spike_Transmission_Probability"]]
            final_summary_df.to_excel(summary_file_path, index=False)
            print(f"\nSummary file created successfully at: {summary_file_path}")
        else:
            print("\nNo PYR-PYR synaptic connections were detected. An empty summary file was created.")
            pd.DataFrame(
                columns=["Animal_ID", "Source_Neuron", "Target_Neuron", "Spike_Transmission_Probability"]).to_excel(
                summary_file_path, index=False)

        messagebox.showinfo("Complete", "PYR-PYR Analysis Completed Successfully!")

    except Exception as e:
        error_message = f"An error occurred during execution: {e}\n\nPlease check your data file format and content."
        print(error_message)
        import traceback
        traceback.print_exc()
        messagebox.showerror("Runtime Error", error_message)


if __name__ == "__main__":
    main()