# -*- coding: utf-8 -*-
"""
此脚本用于计算神经元放电的ISI（Inter-Spike Interval），
并生成频数分布统计和直方图。
"""

import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime


def select_folder(title="请选择一个文件夹"):
    """
    弹出一个窗口让用户选择文件夹。
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    folder_path = filedialog.askdirectory(title=title)
    return folder_path


def get_file_order(file_list):
    """
    让用户选择文件处理顺序。
    """
    while True:
        print("\n请选择文件处理顺序:")
        print("  1: 文件名升序 (e.g., A, B, C)")
        print("  2: 文件名降序 (e.g., C, B, A)")
        print("  3: 系统默认顺序")
        choice = input("请输入选项 (1/2/3): ")

        if choice == '1':
            return sorted(file_list)
        elif choice == '2':
            return sorted(file_list, reverse=True)
        elif choice == '3':
            return file_list
        else:
            print("无效输入，请输入 1, 2, 或 3。")


def main():
    """
    主函数，执行整个ISI分析流程。
    """
    # 1. 让用户选择原始数据文件夹
    input_dir = select_folder("请选择原始数据文件所在文件夹")
    if not input_dir:
        print("未选择文件夹，程序退出。")
        return

    print(f"已选择数据文件夹: {input_dir}")

    # 2. 读取文件夹中所有.xlsx文件
    try:
        all_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx') and not f.startswith('~')]
        if not all_files:
            print("错误：在所选文件夹中未找到任何 .xlsx 文件。")
            return
    except Exception as e:
        print(f"读取文件列表时出错: {e}")
        return

    # 3. 让用户选择文件分析顺序
    sorted_files = get_file_order(all_files)

    # 4. 告知用户处理顺序
    print("\n代码将按照以下顺序处理文件:")
    for i, filename in enumerate(sorted_files):
        print(f"  {i + 1}. {filename}")
    print("-" * 30)

    # 5. 让用户选择结果保存路径
    output_dir = select_folder("请选择一个路径用于保存分析结果")
    if not output_dir:
        print("未选择保存路径，程序退出。")
        return

    # 6. 创建输出文件夹和文件结构
    date_str = datetime.now().strftime("%Y%m%d")
    main_output_folder_name = f"{date_str}_C.4.c.d_ISI频数分布"
    main_output_path = os.path.join(output_dir, main_output_folder_name)

    histograms_folder_name = f"{date_str}_All Neurons ISI histograms_png"
    histograms_path = os.path.join(main_output_path, histograms_folder_name)

    isi_excel_filename = f"{date_str}_All Neurons ISI.xlsx"
    isi_excel_path = os.path.join(main_output_path, isi_excel_filename)

    try:
        os.makedirs(main_output_path, exist_ok=True)
        os.makedirs(histograms_path, exist_ok=True)
        print(f"已创建主输出文件夹: {main_output_path}")
        print(f"将在此处保存所有直方图: {histograms_path}")
        print(f"将在此处保存ISI数据: {isi_excel_path}")
    except Exception as e:
        print(f"创建输出目录时出错: {e}")
        return

    # 准备一个字典来收集所有神经元的ISI数据，以便最后一次性写入Excel
    all_neurons_isi_data = {}

    # 7. 开始处理文件，并添加外层进度条
    for filename in tqdm(sorted_files, desc="处理文件进度"):
        file_path = os.path.join(input_dir, filename)

        try:
            df_input = pd.read_excel(file_path)
        except Exception as e:
            print(f"\n警告：读取文件 {filename}失败，已跳过。错误: {e}")
            continue

        # 内层循环处理文件中的每个神经元（每列），并添加内层进度条
        for neuron_id_str in tqdm(df_input.columns, desc=f"处理 {filename} 中的神经元", leave=False):
            neuron_col = df_input[neuron_id_str]

            # 提取信息
            neuron_id = str(neuron_id_str)  # 确保编号是字符串
            neuron_type_code = neuron_col.iloc[0]
            neuron_type_str = "PYR" if neuron_type_code == 1 else "INT"

            # 提取放电时刻数据，并移除空值
            spike_times_s = neuron_col.iloc[1:].dropna().to_numpy(dtype=float)

            # 确保至少有两次放电才能计算ISI
            if len(spike_times_s) < 2:
                continue

            # a. 计算ISI并转换为毫秒
            isi_s = np.diff(spike_times_s)
            isi_ms = isi_s * 1000

            # b. 将结果存入字典
            # 列头是神经元编号，第一行是种类，后面是ISI数据
            all_neurons_isi_data[neuron_id] = [neuron_type_str] + list(isi_ms)

            # c. 定义频数分布的bins
            # 1-10ms (20个), 10-100ms (20个), 100-1000ms (20个)
            # 使用对数空间来创建在对数尺度上均匀的bins
            bins1 = np.logspace(np.log10(1), np.log10(10), 21)
            bins2 = np.logspace(np.log10(10), np.log10(100), 21)
            bins3 = np.logspace(np.log10(100), np.log10(1000), 21)
            # 合并bins，并去除重复的边界（10和100）
            bins = np.concatenate((bins1[:-1], bins2[:-1], bins3))

            # d. 计算频数分布
            counts, bin_edges = np.histogram(isi_ms, bins=bins)

            # e. 绘制直方图
            plt.figure(figsize=(12, 7))
            # 使用对数坐标轴能更好地展示这种分布
            plt.xscale('log')
            # 使用bar图绘制，宽度与bin的宽度相对应
            plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge', edgecolor='black')

            plt.title(f"ISI Frequency Distribution for Neuron {neuron_id} ({neuron_type_str})", fontsize=16)
            plt.xlabel("Inter-Spike Interval (ms) [Log Scale]", fontsize=12)
            plt.ylabel("Frequency (Count)", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            # f. 保存直方图图片
            png_filename = f"{date_str}_{neuron_id}_{neuron_type_str}_ISI histogram.png"
            png_path = os.path.join(histograms_path, png_filename)
            plt.savefig(png_path, dpi=600)
            plt.close()  # 关闭图像，释放内存

            # g. 保存直方图的原始数据到Excel
            bin_labels = [f"{bin_edges[i]:.3f} - {bin_edges[i + 1]:.3f}" for i in range(len(counts))]
            hist_data_df = pd.DataFrame({
                "Bin_Range (ms)": bin_labels,
                "Count": counts
            })

            excel_hist_filename = f"{date_str}_{neuron_id}_{neuron_type_str}_ISI histogram.xlsx"
            excel_hist_path = os.path.join(histograms_path, excel_hist_filename)
            hist_data_df.to_excel(excel_hist_path, index=False, engine='openpyxl')

    # 8. 所有循环结束后，将收集到的所有ISI数据写入一个Excel文件
    print("\n正在将所有神经元的ISI数据写入Excel文件，请稍候...")
    try:
        # 将字典转换为DataFrame，处理不同长度的ISI序列
        # 注意：pandas的版本可能会影响此操作，dict comprehension是比较稳妥的方式
        final_isi_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_neurons_isi_data.items()]))

        # 将第一行（神经元种类）作为单独的行插入
        # 由于我们把种类和数据放在了一个列表里，现在需要调整
        final_df_transposed = final_isi_df.transpose()
        final_df_transposed.columns = ['Type'] + [f'ISI_{i + 1}' for i in range(final_df_transposed.shape[1] - 1)]

        # 重新构建最终的DataFrame，格式为：
        # Row 0: Neuron_ID (作为列名)
        # Row 1: Type
        # Row 2...: ISI data
        output_df = pd.DataFrame()
        for col_name in final_isi_df.columns:
            # 将列表的第一项作为种类，其余作为数据
            neuron_type_val = final_isi_df[col_name].iloc[0]
            isi_values = final_isi_df[col_name].iloc[1:].dropna()

            # 创建一个临时的Series，包含种类和数据
            temp_series = pd.Series([neuron_type_val] + list(isi_values), name=col_name)

            # 合并到输出DataFrame
            output_df = pd.concat([output_df, temp_series], axis=1)

        # 写入文件
        # 注意：不写入索引和默认的header，因为我们的数据结构已经包含了header信息
        output_df.to_excel(isi_excel_path, index=False, header=True, engine='openpyxl')

        print(f"成功将所有ISI数据写入: {isi_excel_path}")

    except Exception as e:
        print(f"\n写入主ISI Excel文件时出错: {e}")

    print("\n--- 所有任务已完成 ---")


if __name__ == "__main__":
    main()