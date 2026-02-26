import pandas as pd
import os
import glob
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox


def analyze_neuron_data():
    """
    主函数，用于执行神经元放电数据分析的完整流程。
    """
    # --- 步骤 1: 弹出窗口让用户选择文件夹 ---
    try:
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        folder_path = filedialog.askdirectory(title='请选择包含原始数据文件的文件夹')

        if not folder_path:
            messagebox.showinfo("提示", "您没有选择任何文件夹，程序已退出。")
            print("没有选择文件夹，程序退出。")
            return
    except Exception as e:
        print(f"创建图形界面失败，可能是环境不支持。错误: {e}")
        print("请尝试在支持图形界面的环境中运行此脚本。")
        return

    print(f"已选择文件夹: {folder_path}")

    # --- 步骤 2: 读取文件夹中所有 .xlsx 文件 ---
    search_pattern = os.path.join(folder_path, '*.xlsx')
    file_list = glob.glob(search_pattern)

    if not file_list:
        messagebox.showerror("错误", f"在文件夹 '{folder_path}' 中没有找到任何 .xlsx 文件。")
        print("未找到 .xlsx 文件，程序退出。")
        return

    # --- 步骤 3: 让用户选择文件分析顺序 ---
    sort_choice_window = tk.Tk()
    sort_choice_window.withdraw()
    sort_choice = simpledialog.askstring(
        "排序方式",
        "请选择文件处理顺序:\n 1: 文件名升序\n 2: 文件名降序\n (输入其他任意值将按默认顺序处理)",
        parent=sort_choice_window
    )
    sort_choice_window.destroy()

    if sort_choice == '1':
        file_list.sort()
        print("文件将按升序处理。")
    elif sort_choice == '2':
        file_list.sort(reverse=True)
        print("文件将按降序处理。")
    else:
        print("文件将按默认顺序处理。")

    # --- 步骤 4: 创建输出文件并准备写入 ---
    output_filename = '20250915_AllNeuron_SpikeNums_and_Frequency.xlsx'
    output_path = os.path.join(folder_path, output_filename)
    results_data = []
    headers = ['原始数据文件名', '神经元编号', '放电次数', '放电频率 (Hz)']

    RECORDING_TIME_SECONDS = 180.0

    # --- 步骤 5: 对所有文件进行循环操作并添加进度条 ---
    print("\n开始处理文件...")
    try:
        with tqdm(total=len(file_list), desc="文件处理进度") as pbar:
            for file_path in file_list:
                try:
                    original_filename = os.path.basename(file_path)

                    df = pd.read_excel(file_path)

                    for neuron_id in df.columns:
                        spike_count = df[neuron_id].count()
                        frequency = spike_count / RECORDING_TIME_SECONDS
                        results_data.append([
                            original_filename,
                            str(neuron_id),  # 确保神经元编号是字符串
                            spike_count,
                            frequency
                        ])
                except Exception as e:
                    print(f"\n处理文件 {original_filename} 时发生错误: {e}")

                pbar.update(1)

        # --- 步骤 6: 将所有结果写入新的Excel文件 ---
        if results_data:
            print("\n正在将结果写入Excel文件...")
            results_df = pd.DataFrame(results_data, columns=headers)
            results_df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"处理完成！结果已保存至: {output_path}")
            messagebox.showinfo("完成", f"所有文件处理完成！\n结果已保存至:\n{output_path}")
        else:
            print("没有处理任何数据，未生成结果文件。")
            messagebox.showwarning("警告", "没有成功处理任何数据，请检查您的数据文件。")

    except Exception as e:
        print(f"\n程序运行期间发生严重错误: {e}")
        messagebox.showerror("严重错误", f"程序运行期间发生严重错误: {e}")


if __name__ == '__main__':
    analyze_neuron_data()