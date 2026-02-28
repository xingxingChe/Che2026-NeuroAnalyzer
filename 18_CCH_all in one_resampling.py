# -*- coding: utf-8 -*-
"""
CCH Master Pipeline (V_Console_Interactive_Fixed)
功能：
1. 集成 Gaussian convolution 与 Jittering resampling。
2. 支持全连接类型。
3. 严格针对 CUMS / Control 组的突触前单向降采样预处理。
4. 修复了 Gaussian Excitation 里的 IndexError。
5. 增加了对 "ctrl" 等文件名的兼容识别。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.stats import poisson
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
from datetime import datetime

# =========================================================================
# 1. 降采样预处理引擎
# =========================================================================

TARGET_PYR_RATE = 6.765
TARGET_INT_RATE = 11.50

def downsample_spikes(spike_train, target_rate, total_recording_time):
    current_rate = len(spike_train) / total_recording_time
    if current_rate <= target_rate:
        return spike_train
    
    target_count = int(target_rate * total_recording_time)
    if target_count <= 0:
        return np.array([])
        
    downsampled = np.random.choice(spike_train, size=target_count, replace=False)
    downsampled.sort()
    return downsampled

# =========================================================================
# 2. 通用数学与核心 CCH 函数
# =========================================================================

def calculate_cch(spike_train1, spike_train2, bin_size=0.0004, lag=0.05):
    num_bins = int(2 * lag / bin_size)
    if num_bins % 2 == 0: num_bins += 1
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
    all_diffs = []
    for t1 in pre_spikes:
        relevant_t2 = post_spikes[(post_spikes >= t1 - lag) & (post_spikes <= t1 + lag)]
        if len(relevant_t2) > 0: all_diffs.append(relevant_t2 - t1)
    if all_diffs:
        all_diffs = np.concatenate(all_diffs)
        counts, _ = np.histogram(all_diffs, bins=bins)
        return counts
    return np.zeros(len(bins) - 1)

def create_hollow_gaussian(sigma_ms, hollow_fraction, bin_size_ms):
    sigma_bins = sigma_ms / bin_size_ms
    size = int(sigma_bins * 8)
    x = np.arange(-size, size + 1)
    gaussian = np.exp(-(x ** 2) / (2 * sigma_bins ** 2))
    hollow_gaussian = np.exp(-(x ** 2) / (2 * (sigma_bins * (1 - hollow_fraction)) ** 2))
    kernel = gaussian - hollow_gaussian
    return kernel / np.sum(kernel)

def calculate_baseline(cch, sigma=10, hollow_fraction=0.6, bin_size=0.4):
    kernel = create_hollow_gaussian(sigma, hollow_fraction, bin_size)
    return convolve(cch, kernel, mode='same')

def poisson_prob_continuity_corrected(n, mu):
    if mu <= 0: return 1.0 if n <= 0 else 0.0
    prob = 1 - poisson.cdf(n - 1, mu)
    return prob - 0.5 * poisson.pmf(n, mu)

# =========================================================================
# 3. 统计学检验模块
# =========================================================================

def analyze_gaussian_excitation(cch, baseline, bins, pre_spike_count, p_syn_thresh=0.001, p_causal_thresh=0.0026):
    mono_window = (bins >= 0.0008) & (bins <= 0.0028)
    anti_causal_window = (bins >= -0.002) & (bins <= 0.0)
    n_syn, b_syn = np.sum(cch[mono_window]), np.sum(baseline[mono_window])
    n_anti = np.sum(cch[anti_causal_window])
    
    p_syn = poisson_prob_continuity_corrected(n_syn, b_syn)
    p_causal = poisson_prob_continuity_corrected(n_syn, n_anti)
    is_connected = (p_syn < p_syn_thresh) and (p_causal < p_causal_thresh)
    
    excess_spikes = n_syn - b_syn
    stp = (excess_spikes / pre_spike_count) if (pre_spike_count > 0 and excess_spikes > 0) else 0.0
    return is_connected, stp, p_syn, p_causal

def analyze_jitter_excitation(pre_spikes, post_spikes, n_iter=1000, jitter_window=0.005):
    raw_cch, bin_centers, bins_edges = calculate_cch(pre_spikes, post_spikes)
    surrogate_cchs = np.zeros((n_iter, len(raw_cch)))
    noise = np.random.uniform(-jitter_window, jitter_window, (n_iter, len(pre_spikes)))
    for i in range(n_iter):
        jittered_pre = np.sort(pre_spikes + noise[i, :])
        surrogate_cchs[i, :] = fast_cch_for_jitter(jittered_pre, post_spikes, bins_edges, 0.05)
        
    baseline_mean = np.mean(surrogate_cchs, axis=0)
    upper_bound = np.percentile(surrogate_cchs, 99.5, axis=0)
    window_mask = (bin_centers >= 0.0010) & (bin_centers <= 0.0040)
    is_connected = np.any(raw_cch[window_mask] > upper_bound[window_mask])
    
    excess_spikes = np.sum(raw_cch[window_mask]) - np.sum(baseline_mean[window_mask])
    stp = excess_spikes / len(pre_spikes) if (len(pre_spikes) > 0 and excess_spikes > 0) else 0.0
    return {"is_connected": is_connected, "metric": stp, "raw_cch": raw_cch, "baseline": baseline_mean, "bound": upper_bound, "bins": bin_centers}

def analyze_gaussian_inhibition(cch, baseline, bin_centers, pre_spike_count, p_thresh=0.001):
    mono_window = (bin_centers >= 0.0008) & (bin_centers <= 0.0030)
    n_obs, n_exp = np.sum(cch[mono_window]), np.sum(baseline[mono_window])
    
    if n_obs >= n_exp: return False, 0.0, 1.0
    p_val = poisson.cdf(n_obs, n_exp)
    is_inhibited = p_val < p_thresh
    
    deficit = n_exp - n_obs
    ssp = deficit / pre_spike_count if pre_spike_count > 0 else 0.0
    return is_inhibited, ssp, p_val

def analyze_jitter_inhibition(pre_spikes, post_spikes, n_iter=1000, jitter_window=0.005):
    raw_cch, bin_centers, bins_edges = calculate_cch(pre_spikes, post_spikes)
    surrogate_cchs = np.zeros((n_iter, len(raw_cch)))
    noise = np.random.uniform(-jitter_window, jitter_window, (n_iter, len(pre_spikes)))
    for i in range(n_iter):
        jittered_pre = np.sort(pre_spikes + noise[i, :])
        surrogate_cchs[i, :] = fast_cch_for_jitter(jittered_pre, post_spikes, bins_edges, 0.05)
        
    baseline_mean = np.mean(surrogate_cchs, axis=0)
    lower_bound = np.percentile(surrogate_cchs, 0.5, axis=0)
    window_mask = (bin_centers >= 0.0010) & (bin_centers <= 0.0040)
    is_inhibited = np.any(raw_cch[window_mask] < lower_bound[window_mask])
    
    deficit_spikes = np.sum(baseline_mean[window_mask]) - np.sum(raw_cch[window_mask])
    ssp = deficit_spikes / len(pre_spikes) if (len(pre_spikes) > 0 and deficit_spikes > 0) else 0.0
    return {"is_connected": is_inhibited, "metric": ssp, "raw_cch": raw_cch, "baseline": baseline_mean, "bound": lower_bound, "bins": bin_centers}

# =========================================================================
# 4. 统一绘图与导出模块
# =========================================================================

def plot_and_save(bins, cch, baseline, bound, is_connected, metric, conn_type, algo, pair_id, animal_id, output_folder):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = np.mean(np.diff(bins)) if len(bins)>1 else 0.0004
    
    is_exc = conn_type in ['PYR->INT', 'PYR->PYR']
    base_color = '#e74c3c' if is_exc else 'blue'
    fill_color = 'red' if is_exc else 'blue'
    window_color = 'orange' if is_exc else 'blue'
    win_min, win_max = (0.0008, 0.0028) if (algo == 'Gaussian' and is_exc) else (0.001, 0.004) if algo == 'Jittering' else (0.0008, 0.0030)
    
    ax.bar(bins, cch, width=bar_width, color='gray' if algo=='Jittering' else ('#2c3e50' if conn_type=='PYR->PYR' else 'black'), alpha=0.5 if algo=='Jittering' else 0.8, label='Raw CCH', edgecolor='none')
    ax.plot(bins, baseline, color='black' if algo=='Jittering' else base_color, linewidth=1.5 if algo=='Jittering' else 2, linestyle='--', label='Baseline')
    if bound is not None:
        ax.plot(bins, bound, color='red', linestyle='--', linewidth=1.2, label='Significance Bound')
        
    ax.axvspan(win_min, win_max, color=window_color, alpha=0.1, label=f'Analysis Window')
    
    if is_connected:
        window_mask = (bins >= win_min) & (bins <= win_max)
        if is_exc:
            ax.fill_between(bins[window_mask], cch[window_mask], baseline[window_mask], where=(cch[window_mask] > baseline[window_mask]), color=fill_color, alpha=0.6, label='Excess Spikes')
        else:
            ax.fill_between(bins[window_mask], cch[window_mask], baseline[window_mask], where=(baseline[window_mask] > cch[window_mask]), color=fill_color, alpha=0.6, label='Suppressed Spikes')

    ax.set_title(f"{conn_type} ({algo}): {animal_id} - {pair_id}", fontsize=14)
    ax.set_xlim(-0.05 if algo=='Gaussian' else -0.03, 0.05 if algo=='Gaussian' else 0.03)
    ax.legend(loc='upper right')
    
    status_str = f"{'Synapse' if is_exc else 'Suppression'} Detected\nMetric: {metric:.4f}" if is_connected else "No Connection"
    ax.text(0.95 if algo=='Gaussian' else 0.05, 0.95, status_str, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right' if algo=='Gaussian' else 'left', bbox=dict(boxstyle='round', fc=fill_color if is_connected else 'gray', alpha=0.2))

    suffix = "Detected" if is_connected else "None"
    filename = f"{datetime.now().strftime('%Y%m%d')}_{algo}_{conn_type.replace('->','_')}_{animal_id}_{pair_id}_{suffix}.png"
    plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_numeric(data_dict, filename, output_folder):
    pd.DataFrame(data_dict).to_excel(os.path.join(output_folder, filename), index=False)


# =========================================================================
# 5. 主程序与控制台工作流
# =========================================================================

def main():
    root = tk.Tk()
    root.withdraw() 
    
    print("\n=====================================================")
    print("🟢 CCH 全能分析工作站 (纯净交互版) 已启动！")
    print("👉 第 1 步：请在弹出的窗口中选择【原始数据文件夹】📂")
    print("=====================================================")
    
    source_folder = filedialog.askdirectory(title="选择包含原始数据 (.xlsx) 的文件夹")
    if not source_folder: 
        print("❌ 你取消了选择，程序退出。")
        return
    
    files_to_process = sorted([f for f in os.listdir(source_folder) if f.endswith('.xlsx')])
    if not files_to_process: 
        print("❌ 所选文件夹内没有找到 .xlsx 文件，程序退出。")
        return
        
    print(f"✅ 成功锁定包含 {len(files_to_process)} 个 Excel 文件的目标文件夹！\n")
    print("👉 第 2 步：请选择【结果保存文件夹】📂")
    output_folder = filedialog.askdirectory(title="选择输出结果保存文件夹")
    if not output_folder: 
        print("❌ 你取消了选择，程序退出。")
        return
    
    print("✅ 输出路径设置成功！\n")
    
    print("👉 第 3 步：请在下方控制台直接输入参数！(直接按回车代表使用默认开启/默认值)")
    print("-----------------------------------------------------")
    
    while True:
        rec_time_str = input("▶️ 请输入总录音时长 (秒) [默认 600]: ").strip()
        if not rec_time_str:
            rec_time = 600.0
            break
        try:
            rec_time = float(rec_time_str)
            break
        except ValueError:
            print("❌ 输入无效，请输入纯数字 (例如 600)！")

    print("\n▶️ 请选择分析算法 (输入 y 开启，n 关闭，直接回车默认开启):")
    run_gauss = input("   - 运行 Gaussian Convolution? (y/n) [默认 y]: ").strip().lower() != 'n'
    run_jitter = input("   - 运行 Jittering Resampling? (y/n) [默认 y]: ").strip().lower() != 'n'

    print("\n▶️ 请选择要分析的连接类型 (输入 y 开启，n 关闭，直接回车默认开启):")
    c_pi = input("   - PYR (Pre) -> INT (Post) [Excitation] (y/n) [默认 y]: ").strip().lower() != 'n'
    c_ip = input("   - INT (Pre) -> PYR (Post) [Inhibition] (y/n) [默认 y]: ").strip().lower() != 'n'
    c_pp = input("   - PYR (Pre) -> PYR (Post) [Excitation] (y/n) [默认 y]: ").strip().lower() != 'n'
    c_ii = input("   - INT (Pre) -> INT (Post) [Inhibition] (y/n) [默认 y]: ").strip().lower() != 'n'

    conns = []
    if c_pi: conns.append(('PYR', 'INT'))
    if c_ip: conns.append(('INT', 'PYR'))
    if c_pp: conns.append(('PYR', 'PYR'))
    if c_ii: conns.append(('INT', 'INT'))
    
    if not conns or not (run_gauss or run_jitter): 
        print("❌ 未选择有效分析配置，程序退出。")
        return

    print("\n✅ 参数配置完毕！即将进入高强度运算阶段...")
    print("=====================================================\n")
    
    date_str = datetime.now().strftime('%Y%m%d')
    main_out_path = os.path.join(output_folder, f"{date_str}_Master_CCH_Analysis")
    os.makedirs(main_out_path, exist_ok=True)
    
    summary_data = []

    with tqdm(total=len(files_to_process), desc="Total Progress") as pbar_files:
        for filename in files_to_process:
            animal_id = os.path.splitext(filename)[0]
            animal_id_lower = animal_id.lower()
            
            # --- 核心逻辑 1：识别组别 (加入对 ctrl 的识别) ---
            if 'control' in animal_id_lower or 'con' in animal_id_lower or 'ctrl' in animal_id_lower: 
                group = 'Control'
            elif 'cums' in animal_id_lower: 
                group = 'CUMS'
            else: 
                group = 'Unknown'
                
            detail_folder = os.path.join(main_out_path, f"{animal_id}_Details")
            os.makedirs(detail_folder, exist_ok=True)
            
            tqdm.write(f"\n⏳ [读取中] 正在死磕大文件: {filename} (组别被识别为: {group}) ...")
            df = pd.read_excel(os.path.join(source_folder, filename), header=None)
            tqdm.write(f"✅ [读取完成] 成功提取！开始筛选有效神经元并计算 CCH...")
            
            neuron_ids, neuron_types = df.iloc[0, :].astype(str), df.iloc[1, :].astype(int)
            
            spikes_dict = {'PYR': {}, 'INT': {}}
            for i, n_type in enumerate(neuron_types):
                spk = df.iloc[2:, i].dropna().to_numpy()
                if len(spk) >= 500:
                    if n_type == 1: spikes_dict['PYR'][neuron_ids[i]] = spk
                    elif n_type == 0: spikes_dict['INT'][neuron_ids[i]] = spk

            for (pre_type, post_type) in conns:
                pre_ids = list(spikes_dict[pre_type].keys())
                post_ids = list(spikes_dict[post_type].keys())
                if not pre_ids or not post_ids: continue
                
                for pre_id in pre_ids:
                    for post_id in post_ids:
                        if pre_id == post_id: continue # 防止自相关
                        
                        raw_pre_spikes = spikes_dict[pre_type][pre_id]
                        post_spikes = spikes_dict[post_type][post_id]
                        conn_name = f"{pre_type}->{post_type}"
                        pair_id = f"Pre{pre_type}_{pre_id}-Post{post_type}_{post_id}"
                        
                        # --- 核心逻辑 2：极致严谨的突触前单向降采样 ---
                        pre_spikes_analysis = raw_pre_spikes.copy()
                        if pre_type == 'PYR' and group == 'Control':
                            pre_spikes_analysis = downsample_spikes(raw_pre_spikes, TARGET_PYR_RATE, rec_time)
                        elif pre_type == 'INT' and group == 'CUMS':
                            pre_spikes_analysis = downsample_spikes(raw_pre_spikes, TARGET_INT_RATE, rec_time)

                        # 若降采样后脉冲太少，跳过
                        if len(pre_spikes_analysis) < 100: 
                            continue 

                        # --- Pipeline A: Gaussian ---
                        if run_gauss:
                            cch, bins_centers, bins_edges = calculate_cch(pre_spikes_analysis, post_spikes)
                            baseline = calculate_baseline(cch)
                            
                            if pre_type == 'PYR': # Excitation (修复了此处传参为 bins_centers)
                                is_conn, metric, p1, p2 = analyze_gaussian_excitation(cch, baseline, bins_centers, len(pre_spikes_analysis))
                            else: # Inhibition
                                is_conn, metric, p_val = analyze_gaussian_inhibition(cch, baseline, bins_centers, len(pre_spikes_analysis))
                                
                            plot_and_save(bins_centers, cch, baseline, None, is_conn, metric, conn_name, 'Gaussian', pair_id, animal_id, detail_folder)
                            if is_conn:
                                summary_data.append({"Animal_ID": animal_id, "Group": group, "Connection": conn_name, "Pair": pair_id, "Method": "Gaussian", "Metric_STP_SSP": metric})
                                save_numeric({'Time': bins_centers, 'CCH': cch, 'Base': baseline}, f"{pair_id}_Gaussian_data.xlsx", detail_folder)

                        # --- Pipeline B: Jittering ---
                        if run_jitter:
                            if pre_type == 'PYR':
                                res = analyze_jitter_excitation(pre_spikes_analysis, post_spikes)
                            else:
                                res = analyze_jitter_inhibition(pre_spikes_analysis, post_spikes)
                                
                            plot_and_save(res['bins'], res['raw_cch'], res['baseline'], res['bound'], res['is_connected'], res['metric'], conn_name, 'Jittering', pair_id, animal_id, detail_folder)
                            if res['is_connected']:
                                summary_data.append({"Animal_ID": animal_id, "Group": group, "Connection": conn_name, "Pair": pair_id, "Method": "Jittering", "Metric_STP_SSP": res['metric']})
                                save_numeric({'Time': res['bins'], 'CCH': res['raw_cch'], 'Base': res['baseline'], 'Bound': res['bound']}, f"{pair_id}_Jitter_data.xlsx", detail_folder)
                                
            pbar_files.update(1)

    print("\n=====================================================")
    if summary_data:
        pd.DataFrame(summary_data).to_excel(os.path.join(main_out_path, f"{date_str}_Master_Summary.xlsx"), index=False)
        print("🎉 恭喜！全系列分析完美收官！")
        print("📂 已生成总表，请前往你选择的输出文件夹查看。")
    else:
        print("🛑 分析结束，当前数据集未检测到任何显著的突触连接。")
    print("=====================================================")

if __name__ == "__main__":
    main()