import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import re
from datetime import date
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog


# --- 实用工具函数 ---
def select_folder():
    """
    弹出一个窗口让用户选择要处理的文件夹
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    folder_path = filedialog.askdirectory(title="请选择包含CSV波形文件的文件夹")
    return folder_path


def sanitize_filename(name):
    """
    清理字符串，使其可以安全地用作文件名。
    """
    return re.sub(r'[\\/*?:"<>|]', '_', name)


# --- 新增函数：用于文件排序 ---
def get_sorted_file_list(files, folder_path):
    """
    向用户展示排序选项并返回排序后的文件列表。
    """
    while True:
        print("\n" + "=" * 50)
        print("请选择文件处理顺序:")
        print("  1. 按文件名升序 (例如: A.csv, B.csv, ...)")
        print("  2. 按文件名降序 (例如: Z.csv, Y.csv, ...)")
        print("  3. 按修改时间升序 (从最旧的文件到最新)")
        print("  4. 按修改时间降序 (从最新的文件到最旧)")
        print("  5. 手动指定顺序")
        print("=" * 50)

        choice = input("请输入选项编号 (1-5): ")

        if choice == '1':
            return sorted(files)
        elif choice == '2':
            return sorted(files, reverse=True)
        elif choice == '3':
            # 根据文件的修改时间进行升序排序
            return sorted(files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
        elif choice == '4':
            # 根据文件的修改时间进行降序排序
            return sorted(files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
        elif choice == '5':
            print("\n当前文件列表:")
            for i, f in enumerate(files):
                print(f"  {i + 1}: {f}")

            while True:
                order_str = input("\n请输入新的文件顺序，用逗号分隔 (例如: 3,1,2): ")
                try:
                    order_indices = [int(i.strip()) - 1 for i in order_str.split(',')]
                    # 检查输入的索引是否有效
                    if all(0 <= i < len(files) for i in order_indices) and len(order_indices) == len(files):
                        return [files[i] for i in order_indices]
                    else:
                        print("错误：输入的编号无效或数量不匹配，请重新输入。")
                except ValueError:
                    print("错误：输入格式不正确，请输入数字并用逗号分隔。")
        else:
            print("无效的选项，请输入1到5之间的数字。")


# --- 核心分析与绘图函数 (保持不变) ---
def read_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"\n读取文件 {os.path.basename(file_path)} 出错: {e}")
        return None


def plot_waveform(time, voltage, std_dev, neuron_name, save_path):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(time, voltage, 'b-', label='Waveform')
    plt.fill_between(time, voltage - std_dev, voltage + std_dev,
                     color='gray', alpha=0.2, label='Standard Deviation')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage')
    clean_title_name = neuron_name.replace('_wf vs. AllFile', '')
    plt.title(f'Waveform of {clean_title_name}')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(save_path, dpi=600, format='png', bbox_inches='tight')
    except Exception as e:
        print(f"\n保存图片出错: {e}")
    plt.close(fig)


def calculate_peak_fwhm(time, voltage):
    try:
        t_dense = np.linspace(time.min(), time.max(), len(time) * 10)
        f = interp1d(time, voltage, kind='cubic')
        v_dense = f(t_dense)
        positive_mask = v_dense > 0
        if not np.any(positive_mask): return np.nan
        t_positive = t_dense[positive_mask]
        v_positive = v_dense[positive_mask]
        peak_value = np.max(v_positive)
        half_max = peak_value / 2
        indices = np.where(v_positive >= half_max)[0]
        if len(indices) < 2: return np.nan
        left_idx, right_idx = indices[0], indices[-1]
        if left_idx > 0:
            x1, x2 = t_positive[left_idx - 1], t_positive[left_idx]
            y1, y2 = v_positive[left_idx - 1] - half_max, v_positive[left_idx] - half_max
            t_left = x1 + (x2 - x1) * (-y1) / (y2 - y1)
        else:
            t_left = t_positive[left_idx]
        if right_idx < len(v_positive) - 1:
            x1, x2 = t_positive[right_idx], t_positive[right_idx + 1]
            y1, y2 = v_positive[right_idx] - half_max, v_positive[right_idx + 1] - half_max
            t_right = x1 + (x2 - x1) * (-y1) / (y2 - y1)
        else:
            t_right = t_positive[right_idx]
        fwhm = abs(t_right - t_left)
        if 0 < fwhm < (time.max() - time.min()): return fwhm
    except Exception:
        pass
    return np.nan


def calculate_valley_fwhm(time, voltage):
    try:
        t_dense = np.linspace(time.min(), time.max(), len(time) * 10)
        f = interp1d(time, voltage, kind='cubic')
        v_dense = f(t_dense)
        negative_mask = v_dense < 0
        if not np.any(negative_mask): return np.nan
        valley_value = np.min(v_dense[negative_mask])
        half_max = valley_value / 2
        crossings = np.where(np.diff(v_dense <= half_max))[0]
        if len(crossings) >= 2:
            left_idx, right_idx = crossings[0], crossings[-1]
            x1, x2 = t_dense[left_idx], t_dense[left_idx + 1]
            y1, y2 = v_dense[left_idx] - half_max, v_dense[left_idx + 1] - half_max
            t_left = x1 + (x2 - x1) * (-y1) / (y2 - y1)
            x1, x2 = t_dense[right_idx], t_dense[right_idx + 1]
            y1, y2 = v_dense[right_idx] - half_max, v_dense[right_idx + 1] - half_max
            t_right = x1 + (x2 - x1) * (-y1) / (y2 - y1)
            fwhm = abs(t_right - t_left)
            if 0 < fwhm < (time.max() - time.min()): return fwhm
    except Exception:
        pass
    return np.nan


def calculate_characteristics(time, voltage):
    peak_value = np.max(voltage)
    valley_value = np.min(voltage)
    peak_fwhm = calculate_peak_fwhm(time, voltage)
    valley_fwhm = calculate_valley_fwhm(time, voltage)
    valley_index = np.argmin(voltage)
    peak_index = np.argmax(voltage)
    valley_to_peak = abs(time[peak_index] - time[valley_index])
    peak_valley_ratio = abs(peak_value / valley_value) if valley_value != 0 else np.nan
    valley_peak_ratio = abs(valley_value / peak_value) if peak_value != 0 else np.nan
    return {
        'Peak FWHM': round(peak_fwhm, 6) if not np.isnan(peak_fwhm) else np.nan,
        'Valley FWHM': round(valley_fwhm, 6) if not np.isnan(valley_fwhm) else np.nan,
        'Valley-to-Peak Time': round(valley_to_peak, 6),
        'Peak Value': round(peak_value, 6), 'Valley Value': round(valley_value, 6),
        'Peak/Valley': round(peak_valley_ratio, 6) if not np.isnan(peak_valley_ratio) else np.nan,
        'Valley/Peak': round(valley_peak_ratio, 6) if not np.isnan(valley_peak_ratio) else np.nan
    }


def write_results(results_file, filename, neuron_name, characteristics):
    row = [filename, neuron_name] + list(characteristics.values())
    header = ['Filename', 'Neuron'] + list(characteristics.keys())
    file_exists = os.path.exists(results_file)
    with open(results_file, 'a', newline='') as f:
        if not file_exists:
            f.write(','.join(header) + '\n')
        f.write(','.join(map(str, row)) + '\n')


def identify_neurons(data):
    neuron_columns = []
    for col in data.columns:
        if '_wf vs. AllFile' in col:
            match = re.search(r'CH\d+([a-zA-Z])_', col)
            if match:
                neuron_columns.append(col)
    return neuron_columns


# --- 修改后的 main 函数 ---
def main():
    """
    主程序 - 自动化处理流程
    """
    input_folder = select_folder()
    if not input_folder:
        print("未选择文件夹，程序退出。")
        return

    print(f"已选择文件夹: {input_folder}")

    today_str = date.today().strftime("%Y%m%d")
    base_path = os.path.dirname(input_folder)
    png_folder = os.path.join(base_path, f"{today_str}_waveform_png")
    csv_folder = os.path.join(base_path, f"{today_str}_waveformCharacteristics_csv")
    os.makedirs(png_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)
    output_csv_file = os.path.join(csv_folder, f"characteristics_results_{today_str}.csv")

    print(f"图片将保存至: {png_folder}")
    print(f"表格将保存至: {csv_folder}")

    csv_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]
    if not csv_files:
        print("在所选文件夹中未找到任何CSV文件。")
        return

    # --- 新增：调用排序函数 ---
    sorted_csv_files = get_sorted_file_list(csv_files, input_folder)

    print("\n即将开始处理文件，最终顺序为:")
    for f in sorted_csv_files:
        print(f"  -> {f}")

    # 使用排序后的文件列表进行处理
    for filename in tqdm(sorted_csv_files, desc="文件处理进度"):
        file_path = os.path.join(input_folder, filename)

        data = read_data(file_path)
        if data is None: continue

        if 'Time (ms)' not in data.columns:
            print(f"\n文件 {filename} 中缺少 'Time (ms)' 列，已跳过。")
            continue
        time = data['Time (ms)'].values

        neuron_columns = identify_neurons(data)
        if not neuron_columns: continue

        for neuron in tqdm(neuron_columns, desc=f"处理 {filename[:15]}...", leave=False):
            voltage = data[neuron].values
            std_col = neuron.replace('_wf vs. AllFile', '_wf St.Dev. vs. AllFile')
            std_dev = data[std_col].values if std_col in data.columns else np.zeros_like(voltage)

            characteristics = calculate_characteristics(time, voltage)
            write_results(output_csv_file, filename, neuron, characteristics)

            base_filename = os.path.splitext(filename)[0]
            clean_neuron_name = sanitize_filename(neuron.replace('_wf vs. AllFile', ''))
            plot_filename = f"{today_str}_波形图_{base_filename}_{clean_neuron_name}.png"
            plot_save_path = os.path.join(png_folder, plot_filename)
            plot_waveform(time, voltage, std_dev, neuron, plot_save_path)

    print("\n\n所有文件处理完成！")


if __name__ == "__main__":
    main()