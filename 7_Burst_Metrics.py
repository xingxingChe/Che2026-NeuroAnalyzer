import os
import tkinter as tk
from tkinter import filedialog, simpledialog
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm


# --- 计算核心函数 ---

def calculate_burst_metrics(spike_times_s, recording_duration_s=180):
    """
    计算关于爆发(Burst)的各项指标。
    一个脉冲被认为是爆发的一部分，如果其前向或后向ISI <= 10ms。
    """
    spike_times_ms = np.array(spike_times_s) * 1000
    n_spikes = len(spike_times_ms)

    # 处理脉冲数过少的情况
    if n_spikes < 2:
        return 0, 0, 0, 0, 0

    isis = np.diff(spike_times_ms)

    # 标记每个脉冲是否属于爆发
    is_in_burst = np.zeros(n_spikes, dtype=bool)
    # 检查后向ISI (从第二个脉冲开始)
    is_in_burst[1:] = isis <= 10
    # 检查前向ISI (直到倒数第二个脉冲)
    is_in_burst[:-1] = is_in_burst[:-1] | (isis <= 10)

    spikes_in_bursts_count = np.sum(is_in_burst)

    if spikes_in_bursts_count == 0:
        return 0, 0, 0, 0, 0

    percentage_spikes_in_bursts = (spikes_in_bursts_count / n_spikes) * 100 if n_spikes > 0 else 0

    # 计算爆发事件的总数 (一个或多个连续的爆发脉冲算作一个事件)
    burst_switches = np.diff(is_in_burst.astype(int))
    # 每个从非爆发到爆发的转换 (+1) 就是一个新爆发的开始
    total_bursts = np.sum(burst_switches == 1)
    # 如果第一个脉冲就在爆发中，也算一个爆发事件
    if is_in_burst[0]:
        total_bursts += 1

    burst_rate = total_bursts / recording_duration_s

    # 爆发内的ISIs
    # 一个ISI在爆发内，当且仅当它连接的两个脉冲都在爆发中
    intra_burst_mask = is_in_burst[:-1] & is_in_burst[1:]
    intra_burst_isis = isis[intra_burst_mask]

    # 爆发内平均放电率 (Hz)
    if len(intra_burst_isis) > 0:
        # 瞬时放电率 = 1000 / isi (ms)
        mean_intra_burst_rate = np.mean(1000 / intra_burst_isis)
    else:
        mean_intra_burst_rate = 0

    mean_spikes_per_burst = spikes_in_bursts_count / total_bursts if total_bursts > 0 else 0

    return total_bursts, burst_rate, mean_intra_burst_rate, mean_spikes_per_burst, percentage_spikes_in_bursts


def calculate_cv(isis):
    """计算变异系数 (CV)"""
    if len(isis) < 2:
        return np.nan
    mean_isi = np.mean(isis)
    std_isi = np.std(isis, ddof=1)  # ddof=1 for sample standard deviation
    return std_isi / mean_isi if mean_isi > 0 else np.nan


def calculate_cv2(isis):
    """计算局部变异系数 (CV2)"""
    if len(isis) < 2:
        return np.nan
    # 计算相邻ISI对的差异
    cv2_values = 2 * np.abs(isis[1:] - isis[:-1]) / (isis[1:] + isis[:-1])
    return np.mean(cv2_values)


def calculate_fano_factor(spike_times_s, duration_s=180, bin_width_s=1.0):
    """计算法诺因子 (Fano Factor)"""
    if len(spike_times_s) == 0:
        return np.nan

    # 创建时间窗口
    bins = np.arange(0, duration_s + bin_width_s, bin_width_s)
    # 计算每个窗口内的脉冲数
    counts, _ = np.histogram(spike_times_s, bins=bins)

    mean_count = np.mean(counts)
    var_count = np.var(counts, ddof=1)

    return var_count / mean_count if mean_count > 0 else np.nan


# --- 主程序 ---

def main():
    # 创建一个Tkinter根窗口并隐藏它
    root = tk.Tk()
    root.withdraw()

    print("程序启动...")

    # 1. 让用户选择原始数据文件所在文件夹
    input_dir = filedialog.askdirectory(title="请选择包含原始数据文件(.xlsx)的文件夹")
    if not input_dir:
        print("未选择文件夹，程序退出。")
        return

    # 2. 读取文件夹中所有xlsx文件
    try:
        all_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx') and not f.startswith('~')]
        if not all_files:
            print(f"错误：在文件夹 '{input_dir}' 中未找到任何 .xlsx 文件。")
            return
    except FileNotFoundError:
        print(f"错误：找不到文件夹 '{input_dir}'。")
        return

    # 3. 让用户选择文件分析顺序
    # 使用simpledialog来创建一个简单的选择窗口
    root.deiconify()  # 临时显示一个窗口来承载对话框
    sort_choice = simpledialog.askstring("文件排序", "请选择文件分析顺序:\n 1: 文件名升序 (默认)\n 2: 文件名降序",
                                         parent=root)
    root.withdraw()  # 再次隐藏

    if sort_choice == '2':
        all_files.sort(reverse=True)
        sort_order_msg = "降序"
    else:
        all_files.sort()
        sort_order_msg = "升序"

    # 4. 告知用户代码将以什么顺序处理文件
    print("-" * 50)
    print(f"文件将以 [{sort_order_msg}] 顺序处理:")
    for f in all_files:
        print(f"  - {f}")
    print("-" * 50)

    # 5. 提示用户选择创建输出文件夹的路径
    output_parent_dir = filedialog.askdirectory(title="请选择保存结果的文件夹路径")
    if not output_parent_dir:
        print("未选择输出路径，程序退出。")
        return

    # 6. 创建输出文件夹和Excel文件
    today_str = datetime.now().strftime("%Y%m%d")
    output_folder_name = f"{today_str}_D.2.3_Burst指标"
    output_excel_name = f"{today_str}_Burst指标.xlsx"

    output_folder_path = os.path.join(output_parent_dir, output_folder_name)
    output_excel_path = os.path.join(output_folder_path, output_excel_name)

    os.makedirs(output_folder_path, exist_ok=True)
    print(f"结果将保存在: {output_excel_path}")

    # 7. 准备存储所有结果的列表
    all_results = []
    headers = [
        "神经元编号", "神经元种类", "爆发总数", "爆发率",
        "爆发内平均放电率", "每次爆发的平均脉冲数", "爆发中脉冲占比",
        "变异系数", "局部变异系数", "法诺因子"
    ]

    # 8. 开始处理文件 (外层循环，带文件进度条)
    for filename in tqdm(all_files, desc="文件处理进度", unit="个文件"):
        file_path = os.path.join(input_dir, filename)

        try:
            # 读取Excel，指定第一行为表头，第二行数据从实际第三行开始
            # 我们先读取所有数据，再手动处理头两行
            df = pd.read_excel(file_path, header=None)
            neuron_ids = df.iloc[0, :]
            neuron_types = df.iloc[1, :]
            spike_data = df.iloc[2:, :]
            spike_data.columns = neuron_ids  # 将神经元编号设置为列名
        except Exception as e:
            print(f"\n读取文件 '{filename}' 时出错: {e}，已跳过此文件。")
            continue

        # 内层循环，处理文件中的每个神经元 (带神经元进度条)
        for i in tqdm(range(len(neuron_ids)), desc=f"处理神经元 @ {filename}", leave=False, unit="个神经元"):
            neuron_id = neuron_ids[i]
            neuron_type = neuron_types[i]

            # 提取该神经元所有放电时刻，并移除空值
            spike_times = spike_data[neuron_id].dropna().to_numpy()

            # --- 开始计算 ---
            if len(spike_times) < 2:
                # 如果脉冲数不足，无法计算ISI，所有指标记为0或NaN
                total_bursts, burst_rate, mean_intra_burst_rate, \
                    mean_spikes_per_burst, percentage_spikes_in_bursts = (0, 0, 0, 0, 0)
                cv, cv2, ff = (np.nan, np.nan, np.nan)
            else:
                isis_ms = np.diff(spike_times) * 1000

                # 计算爆发指标
                total_bursts, burst_rate, mean_intra_burst_rate, \
                    mean_spikes_per_burst, percentage_spikes_in_bursts = \
                    calculate_burst_metrics(spike_times, recording_duration_s=180)

                # 计算变异性指标
                cv = calculate_cv(isis_ms)
                cv2 = calculate_cv2(isis_ms)
                ff = calculate_fano_factor(spike_times, duration_s=180)

            # 将结果存入字典
            result_row = {
                "神经元编号": neuron_id,
                "神经元种类": 'PYR' if neuron_type == 1 else 'INT',
                "爆发总数": total_bursts,
                "爆发率": burst_rate,
                "爆发内平均放电率": mean_intra_burst_rate,
                "每次爆发的平均脉冲数": mean_spikes_per_burst,
                "爆发中脉冲占比": percentage_spikes_in_bursts,
                "变异系数": cv,
                "局部变异系数": cv2,
                "法诺因子": ff
            }
            all_results.append(result_row)

    # 9. 所有文件处理完毕后，一次性将结果写入Excel
    if all_results:
        results_df = pd.DataFrame(all_results, columns=headers)
        results_df.to_excel(output_excel_path, index=False, engine='openpyxl')
        print("\n" + "=" * 50)
        print("处理完成！所有结果已成功保存。")
        print("=" * 50)
    else:
        print("\n警告：没有处理任何数据，未生成结果文件。")


if __name__ == "__main__":
    main()