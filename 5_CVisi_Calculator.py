import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
from tqdm import tqdm


def calculate_isi_stats():
    """
    该函数通过图形化界面让用户选择一个包含神经元ISI数据的Excel文件，
    然后计算每个神经元ISI的均值、标准差和变异系数（CV），
    最后将结果保存到一个新的Excel文件中。
    """
    # 1. 弹出窗口让用户选择原始数据文件
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    print("正在打开文件选择窗口，请选择您的原始数据文件...")
    file_path = filedialog.askopenfilename(
        title="请选择原始ISI数据文件",
        filetypes=[("Excel files", "*.xlsx")]
    )

    if not file_path:
        print("您没有选择任何文件，程序已退出。")
        return

    print(f"文件已选择: {file_path}")

    try:
        # 2. 读取Excel文件
        # header=None 表示第一行不是表头
        df = pd.read_excel(file_path, header=None)

        # 3. 在原始文件相同路径下建立输出文件
        output_dir = os.path.dirname(file_path)
        output_filename = "20250916_CVisi.xlsx"
        output_path = os.path.join(output_dir, output_filename)

        # 准备一个列表来存储所有神经元的计算结果
        results_list = []

        # 4. 遍历每一列（每个神经元）并进行计算
        print("开始计算，请稍候...")
        # 使用tqdm创建进度条
        for col in tqdm(df.columns, desc="处理进度"):
            # 提取信息
            neuron_id = df.loc[0, col]
            neuron_type = df.loc[1, col]

            # 提取ISI数据（从第三行开始），并转换为数值类型，忽略无法转换的错误
            isi_data = pd.to_numeric(df.loc[2:, col], errors='coerce').dropna()

            if isi_data.empty:
                # 如果该列没有有效的ISI数据，则跳过
                continue

            # 计算均值
            isi_mean = isi_data.mean()

            # 计算标准差
            isi_std = isi_data.std()

            # 计算变异系数 CVisi
            if isi_mean != 0:
                cv_isi = isi_std / isi_mean
            else:
                cv_isi = np.nan  # 如果均值为0，则CV无意义，记为NaN

            # 将结果添加到列表中
            results_list.append({
                "神经元编号": neuron_id,
                "神经元种类": neuron_type,
                "ISI的均值": isi_mean,
                "ISI的标准差": isi_std,
                "CVisi": cv_isi
            })

        # 5. 将结果列表转换为DataFrame并写入新的Excel文件
        if results_list:
            results_df = pd.DataFrame(results_list)

            # 将DataFrame写入Excel文件，不包含索引列
            results_df.to_excel(output_path, index=False)

            print("\n处理完成！")
            print(f"结果已成功保存至: {output_path}")
        else:
            print("未能从文件中处理任何数据。")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")


# --- 程序主入口 ---
if __name__ == "__main__":
    calculate_isi_stats()