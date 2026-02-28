import os
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from itertools import combinations
import tkinter as tk
from tkinter import filedialog

# =============================================================================
# 全局参数设置 (Global Parameters)
# =============================================================================
# 降采样目标放电率 (Hz) - 基于 CUMS 组 PYR 的较低平均放电率设定
TARGET_PYR_RATE = 6.765 


# =============================================================================
# Helper Functions for Analysis
# =============================================================================

def downsample_spikes(spike_train, target_rate, total_recording_time):
    """
    通过无放回的随机抽样，将单个神经元的 Spike train 抽稀至 target_rate，并返回重新排序后的时间戳数组。
    """
    current_count = len(spike_train)
    if current_count == 0 or total_recording_time <= 0:
        return spike_train

    current_rate = current_count / total_recording_time
    
    # 如果原始放电率已经低于或等于目标放电率，则不需要降采样
    if current_rate <= target_rate:
        return spike_train

    target_count = int(target_rate * total_recording_time)
    if target_count <= 0:
        return np.array([])

    # 无放回的随机抽样
    sampled_spikes = np.random.choice(spike_train, size=target_count, replace=False)
    # 必须对抽样后的时间戳重新排序，以保证时间序列的正确性
    return np.sort(sampled_spikes)


def detect_assemblies(spike_count_matrix, num_to_extract=2):
    """
    Detects cell assemblies using ICA, with a fixed number of components to extract.
    """
    num_neurons, _ = spike_count_matrix.shape

    if num_neurons < 2:
        return np.array([])

    scaler = StandardScaler()
    z_scored_matrix = scaler.fit_transform(spike_count_matrix.T).T

    n_assemblies = num_to_extract

    if num_neurons < n_assemblies:
        print(f"      [警告] ⚠️ 神经元数量({num_neurons})少于要提取的集群数({n_assemblies})。跳过此文件的集群提取。")
        return np.array([])

    ica = FastICA(n_components=n_assemblies, whiten='unit-variance', max_iter=1000, tol=0.001)
    try:
        ica.fit(z_scored_matrix.T)
    except Exception as e:
        print(f"      [错误] ❌ ICA拟合失败: {e}。跳过此文件的集群分析。")
        return np.array([])

    weight_vectors = ica.components_

    for i in range(n_assemblies):
        vec = weight_vectors[i, :]
        if np.abs(np.min(vec)) > np.max(vec):
            weight_vectors[i, :] = -vec
        norm = np.linalg.norm(vec)
        if norm > 0:
            weight_vectors[i, :] /= np.linalg.norm(weight_vectors[i, :])

    return weight_vectors


def reconstruct_and_quantify_activations(spike_count_matrix, weight_vectors, T_total):
    """
    Reconstructs assembly activations and quantifies their strength and frequency.
    """
    if weight_vectors.shape[0] == 0:
        return np.nan, np.nan

    smoothed_counts = gaussian_filter1d(spike_count_matrix, sigma=1, axis=1)
    scaler = StandardScaler()
    z_scored_smoothed = scaler.fit_transform(smoothed_counts.T).T

    all_strengths, all_frequencies = [], []

    for i in range(weight_vectors.shape[0]):
        w = weight_vectors[i, :]
        P = np.outer(w, w)
        activation_strength_t = np.sum((z_scored_smoothed.T @ P) * z_scored_smoothed.T, axis=1)
        activations = activation_strength_t[activation_strength_t > 5]

        if len(activations) > 0:
            avg_strength = np.mean(activations)
            activation_events = np.sum((activation_strength_t[:-1] <= 5) & (activation_strength_t[1:] > 5))
            frequency = activation_events / T_total
        else:
            avg_strength = np.nan
            frequency = 0.0

        all_strengths.append(avg_strength)
        all_frequencies.append(frequency)

    return np.nanmean(all_strengths), np.nanmean(all_frequencies)


def calculate_pairwise_sync(pyr_spike_times, T_total):
    """
    Calculates normalized pairwise synchrony for all PYR pairs.
    """
    if len(pyr_spike_times) < 2: return []
    neuron_ids = list(pyr_spike_times.keys())
    sync_values = []
    # *** 参数调整：同步窗口大小调整回25ms ***
    window_size = 0.025
    frequencies = {nid: len(spikes) / T_total for nid, spikes in pyr_spike_times.items() if T_total > 0}

    for id1, id2 in combinations(neuron_ids, 2):
        spikes1, spikes2 = pyr_spike_times[id1], pyr_spike_times[id2]
        if len(spikes1) == 0 or len(spikes2) == 0: continue

        coincident_spikes = 0
        # A more efficient way to count coincident spikes
        for t1 in spikes1:
            # Find insertion point to quickly check the window
            idx = np.searchsorted(spikes2, t1 - window_size / 2, side='left')
            while idx < len(spikes2) and spikes2[idx] <= t1 + window_size / 2:
                coincident_spikes += 1
                idx += 1

        expected_chance = 2 * frequencies.get(id2, 0) * window_size * len(spikes1)
        if expected_chance > 0:
            sync_values.append(coincident_spikes / expected_chance)
    return sync_values


def calculate_cofiring_validation(spike_count_matrix, weight_vectors):
    """
    Calculates co-firing coefficients (Pearson's r) for high- vs. low-weight neurons.
    """
    if weight_vectors.shape[0] == 0: return []
    validation_data = []
    for asm_idx, w_vec in enumerate(weight_vectors):
        threshold = 2 * np.std(w_vec)
        high_indices = np.where(w_vec > threshold)[0]
        low_indices = np.where(w_vec <= threshold)[0]
        if len(high_indices) >= 2:
            for i, j in combinations(high_indices, 2):
                with np.errstate(divide='ignore', invalid='ignore'):  # Handle constant spike trains
                    corr = np.corrcoef(spike_count_matrix[i, :], spike_count_matrix[j, :])[0, 1]
                if not np.isnan(corr):
                    validation_data.append(
                        {"Assembly_Index": asm_idx, "Weight_Category": "High", "Co-firing_Coefficient_r": corr})
        if len(low_indices) >= 2:
            num_to_sample = len(high_indices) * (len(high_indices) - 1) // 2
            num_to_sample = min(num_to_sample, 100)  # Sample a reasonable number of pairs
            if num_to_sample > 0:
                low_pairs = list(combinations(low_indices, 2))
                sample_indices = np.random.choice(len(low_pairs), min(len(low_pairs), num_to_sample), replace=False)
                for idx in sample_indices:
                    i, j = low_pairs[idx]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        corr = np.corrcoef(spike_count_matrix[i, :], spike_count_matrix[j, :])[0, 1]
                    if not np.isnan(corr):
                        validation_data.append(
                            {"Assembly_Index": asm_idx, "Weight_Category": "Low", "Co-firing_Coefficient_r": corr})
    return validation_data


def process_single_file(file_path, save_fig5b_data=False, output_dir="."):
    """
    Main processing pipeline for a single mouse data file.
    """
    mouse_id = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n==================================================")
    print(f"🔄 开始处理小鼠数据: {mouse_id}")
    print(f"==================================================")

    print(f"  -> [1/7] 📂 正在读取 Excel 数据...")
    df = pd.read_excel(file_path)
    neuron_types = df.iloc[0]
    pyr_columns = neuron_types[neuron_types == 1].index

    if len(pyr_columns) < 2:
        print(f"  -> [警告] ⚠️ PYR数量({len(pyr_columns)})少于2，无法进行网络分析，跳过此文件。")
        return None, None, None

    print(f"  -> [2/7] 🧩 成功识别 {len(pyr_columns)} 个锥体细胞(PYR)，正在提取 Spike Times...")
    pyr_spike_times = {col: df[col][1:].dropna().to_numpy() for col in pyr_columns}

    T_total = 0
    for spikes in pyr_spike_times.values():
        if len(spikes) > 0 and spikes.max() > T_total:
            T_total = spikes.max()

    # =========================================================================
    # 群体降采样控制 (Population Down-sampling Control)
    # =========================================================================
    is_ctrl = 'ctrl' in mouse_id.lower()
    print(f"  -> [3/7] ⚖️ 组别判定与群体降采样...")

    if is_ctrl:
        print(f"      ✅ 检测到 Ctrl 组数据，正在执行群体降采样 (目标放电率: {TARGET_PYR_RATE} Hz)...")
        downsample_count = 0
        for col in pyr_columns:
            original_spikes = pyr_spike_times[col]
            original_rate = len(original_spikes) / T_total if T_total > 0 else 0
            if original_rate > TARGET_PYR_RATE:
                downsample_count += 1
            downsampled_spikes = downsample_spikes(original_spikes, TARGET_PYR_RATE, T_total)
            pyr_spike_times[col] = downsampled_spikes
        print(f"      ✅ 降采样完成！共对 {downsample_count} 个高放电率神经元进行了抽稀处理。")
    else:
        print(f"      ✅ 检测到非 Ctrl 组数据 (如 CUMS)，跳过降采样，保持原始放电率。")
    # =========================================================================

    print(f"  -> [4/7] 📊 正在划分时间窗 (Bin size = 25ms) 并构建 Spike Count Matrix...")
    bin_size = 0.025
    bins = np.arange(0, T_total + bin_size, bin_size)
    spike_count_matrix = np.zeros((len(pyr_columns), len(bins) - 1))
    
    for i, neuron_id in enumerate(pyr_columns):
        spikes = pyr_spike_times[neuron_id]
        if len(spikes) > 0:
            spike_count_matrix[i, :], _ = np.histogram(spikes, bins=bins)

    scaler = StandardScaler()
    z_scored_matrix = scaler.fit_transform(spike_count_matrix.T).T
    covariance_matrix = np.cov(z_scored_matrix)

    print(f"  -> [5/7] 🧠 正在使用 FastICA 算法提取神经元同步集群 (Cell Assemblies)...")
    weight_vectors = detect_assemblies(spike_count_matrix, num_to_extract=2)
    print(f"      ✅ 尝试提取 2 个集群，成功找到 {weight_vectors.shape[0]} 个。")

    if save_fig5b_data and weight_vectors.shape[0] > 0:
        cov_path = os.path.join(output_dir, f"{mouse_id}_covariance_matrix.csv")
        weights_path = os.path.join(output_dir, f"{mouse_id}_weight_vectors.csv")
        np.savetxt(cov_path, covariance_matrix, delimiter=",")
        np.savetxt(weights_path, weight_vectors, delimiter=",")
        print(f"      💾 已将协方差矩阵和权重向量保存用于 Fig.5b 作图。")

    print(f"  -> [6/7] 📈 正在重建集群激活轨迹，并计算激活强度与频率...")
    avg_strength, avg_frequency = reconstruct_and_quantify_activations(spike_count_matrix, weight_vectors, T_total)
    
    print(f"  -> [7/7] 🔗 正在计算成对神经元同步性 (Pairwise Sync) 以及验证共发放系数...")
    sync_values = calculate_pairwise_sync(pyr_spike_times, T_total)
    validation_data = calculate_cofiring_validation(spike_count_matrix, weight_vectors)

    summary = {"Mouse_ID": mouse_id, "Avg_Assembly_Strength": avg_strength, "Avg_Assembly_Frequency_Hz": avg_frequency}
    sync_df = pd.DataFrame({"Mouse_ID": mouse_id, "Normalized_Sync_Value": sync_values})
    validation_df = pd.DataFrame(validation_data)
    if not validation_df.empty:
        validation_df["Mouse_ID"] = mouse_id

    print(f"✅ 文件 {mouse_id} 分析流程全部完成！")
    return summary, sync_df, validation_df


# =============================================================================
# Main Execution Block
# =============================================================================

def main():
    root = tk.Tk()
    root.withdraw()

    print("🌟 欢迎使用 mPFC 神经元集群分析工具！")
    
    input_dir = filedialog.askdirectory(title="请选择包含原始数据 (.xlsx) 的文件夹")
    if not input_dir:
        print("❌ 未选择输入文件夹，程序退出。")
        return

    output_dir = filedialog.askdirectory(title="请选择保存结果的文件夹")
    if not output_dir:
        print("❌ 未选择输出文件夹，程序退出。")
        return

    all_files = [f for f in os.listdir(input_dir) if f.endswith(".xlsx") and not f.startswith("~")]
    if not all_files:
        print("❌ 在所选文件夹中未找到任何有效的 .xlsx 文件。")
        return

    print(f"\n📁 成功找到以下 {len(all_files)} 个文件准备分析:")
    for i, fname in enumerate(all_files):
        print(f"  [{i + 1}] {fname}")

    processing_order = []
    while not processing_order:
        choice = input("\n⌨️ 请选择文件分析顺序 (输入数字并按回车):\n"
                       "  1: 升序排列 (默认)\n"
                       "  2: 降序排列\n"
                       "  3: 自定义顺序\n"
                       "👉 你的选择: ")
        if choice == '1' or choice == '':
            processing_order = sorted(all_files)
        elif choice == '2':
            processing_order = sorted(all_files, reverse=True)
        elif choice == '3':
            custom_order_str = input(f"请输入文件编号 (1-{len(all_files)})，以英文逗号分隔 (例如: 3,1,2): ")
            try:
                order_indices = [int(x.strip()) - 1 for x in custom_order_str.split(',')]
                # More robust check for custom order input
                if len(order_indices) == len(all_files) and len(set(order_indices)) == len(all_files) and all(
                        0 <= i < len(all_files) for i in order_indices):
                    processing_order = [all_files[i] for i in order_indices]
                else:
                    print("⚠️ 错误: 输入的编号有重复、超出范围或数量不正确，请重试。")
            except (ValueError, IndexError):
                print("⚠️ 错误: 请确保输入的是以逗号分隔的有效数字，请重试。")
        else:
            print("⚠️ 无效输入，请输入1, 2, 或 3。")

    print("\n🚀 即将按以下顺序开启分析狂飙模式:")
    for fname in processing_order:
        print(f"  🔜 {fname}")
    
    all_summary_data, all_sync_dfs, all_validation_dfs = [], [], []

    for filename in processing_order:
        file_path = os.path.join(input_dir, filename)
        summary, sync_df, validation_df = process_single_file(file_path, save_fig5b_data=True, output_dir=output_dir)

        if summary:
            all_summary_data.append(summary)
            all_sync_dfs.append(sync_df)
            if validation_df is not None and not validation_df.empty:
                all_validation_dfs.append(validation_df)

    if not all_summary_data:
        print("\n 分析结束，但未能生成任何有效的汇总数据。请检查原始文件。")
        return

    print("\n==================================================")
    print("💾 正在整合所有结果并写入最终的 Excel 文件...")
    final_summary_df = pd.DataFrame(all_summary_data)
    final_sync_df = pd.concat(all_sync_dfs, ignore_index=True) if all_sync_dfs else pd.DataFrame()
    final_validation_df = pd.concat(all_validation_dfs, ignore_index=True) if all_validation_dfs else pd.DataFrame()

    output_path = os.path.join(output_dir, "analysis_results_final.xlsx")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        final_summary_df.to_excel(writer, sheet_name="Assembly_Summary", index=False)
        final_sync_df.to_excel(writer, sheet_name="Pairwise_Sync_Data", index=False)
        if not final_validation_df.empty:
            final_validation_df.to_excel(writer, sheet_name="Cofiring_Validation_Data", index=False)

    print(f"🎉 恭喜维卡！所有分析均已顺利完成！")
    print(f"📄 最终结果总表已保存至: {output_path}")
    print("==================================================\n")


if __name__ == "__main__":
    main()