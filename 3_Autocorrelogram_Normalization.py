import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, messagebox


def main():
    """
    主函数，执行标准化和重新绘图的流程。
    """
    # 隐藏主Tkinter窗口
    root = tk.Tk()
    root.withdraw()

    # 1. 弹窗让用户选择 numeric.xlsx 文件
    messagebox.showinfo("选择文件", "请选择您想要进行标准化处理的 '..._numeric.xlsx' 文件。")
    input_filepath = filedialog.askopenfilename(
        title="请选择 numeric.xlsx 文件",
        filetypes=[("Excel files", "*.xlsx")]
    )

    if not input_filepath:
        print("未选择文件，程序退出。")
        return

    print(f"正在读取文件: {input_filepath}")

    try:
        # 读取数据
        df = pd.read_excel(input_filepath)

        # 复制DataFrame用于标准化
        df_normalized = df.copy()

        # 2. 对每个神经元列进行 Min-Max 标准化 (0-1)
        # 第一列是 'Time_ms'，我们从第二列开始处理
        neuron_columns = df.columns[1:]

        print("正在对每个神经元的数据进行标准化 (0-1)...")
        for col_name in tqdm(neuron_columns, desc="标准化进度"):
            column_data = df[col_name]
            min_val = column_data.min()
            max_val = column_data.max()

            # 处理分母为0的特殊情况 (例如，如果一列数据完全相同)
            if (max_val - min_val) == 0:
                df_normalized[col_name] = 0
            else:
                df_normalized[col_name] = (column_data - min_val) / (max_val - min_val)

        # 3. 构造输出路径并保存新的Excel文件
        input_dir = os.path.dirname(input_filepath)
        input_filename_base = os.path.basename(input_filepath).replace('_numeric.xlsx', '')

        output_excel_path = os.path.join(input_dir, f"{input_filename_base}_numeric_normalized.xlsx")

        df_normalized.to_excel(output_excel_path, index=False)
        print(f"标准化数据已成功保存到: {output_excel_path}")

        # 4. 创建新的PNG文件夹
        output_png_dir = os.path.join(input_dir, f"{input_filename_base}_Autocorrelograms_normalized_png")
        os.makedirs(output_png_dir, exist_ok=True)
        print(f"新的PNG图将保存到: {output_png_dir}")

        # 5. 使用新数据重新绘制和导出所有图像
        time_data = df_normalized['Time_ms']

        # 从文件名中提取日期部分，以便命名图片
        date_str = input_filename_base.split('_')[0]

        for neuron_id in tqdm(neuron_columns, desc="生成新图像"):
            normalized_acorr_data = df_normalized[neuron_id]

            plt.figure(figsize=(8, 6))
            plt.plot(time_data, normalized_acorr_data)

            # 样式设置
            plt.axvline(0, color='gray', linestyle='--', linewidth=1)
            plt.title(f'Normalized Autocorrelogram for Neuron {neuron_id}')
            plt.xlabel('Time Lag (ms)')
            plt.ylabel('Normalized Firing Rate')  # Y轴标签更新
            plt.ylim(0, 1.05)  # 设置Y轴范围，顶部留一点空间
            plt.xlim(time_data.min(), time_data.max())

            # 构造新的文件名并保存
            png_filename = f"{date_str}_{neuron_id}_Autocorrelogram_normalized.png"
            plt.savefig(os.path.join(output_png_dir, png_filename), dpi=600)
            plt.close()  # 关闭图像以释放内存

        messagebox.showinfo("完成", "所有数据已成功标准化，新图表也已全部生成！")
        print("\n--- 程序运行结束 ---")

    except Exception as e:
        messagebox.showerror("发生错误", f"程序运行中出现错误: \n{e}")
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()