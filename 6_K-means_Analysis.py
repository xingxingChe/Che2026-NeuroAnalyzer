import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from tkinter import Tk, filedialog

# --- 新功能：弹出文件选择窗口 ---
# 隐藏Tkinter的根窗口
root = Tk()
root.withdraw()

# 打开文件选择对话框，让用户选择CSV文件
# filetypes 参数限制了用户只能选择.csv文件
input_filepath = filedialog.askopenfilename(
    title="请选择原始数据文件",
    filetypes=[("CSV files", "*.csv")]
)

# 如果用户没有选择文件（点了取消），则退出程序
if not input_filepath:
    print("没有选择文件，程序已退出。")
    exit()

# --- 新功能：自动设置输出路径 ---
# 从用户选择的文件路径中获取所在目录
base_directory = os.path.dirname(input_filepath)
# 将结果文件路径设置在与原始文件相同的目录下
output_filepath = os.path.join(base_directory, 'k_means_results.csv')

try:
    # 读取CSV文件
    data = pd.read_csv(input_filepath)

    # 检查所需列是否存在
    required_columns = ['Peak FWHM', 'Valley FWHM']
    if not all(col in data.columns for col in required_columns):
        print(f"错误：文件 {input_filepath} 中缺少必需的列。")
        print(f"请确保文件包含以下列：{', '.join(required_columns)}")
        exit()

    # 提取需要的数据
    X = data[['Peak FWHM', 'Valley FWHM']].values

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
    plt.xlabel('Peak Full Width at Half Maximum (FWHM)')
    plt.ylabel('Valley Full Width at Half Maximum (FWHM)')
    plt.title('Neural Spike Waveform Characteristics')
    plt.grid(True)

    # 显示图形并等待用户关闭
    print("正在显示数据散点图，请关闭图形窗口以继续...")
    plt.show(block=True)

    # 执行k-means聚类
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # 创建结果DataFrame
    results = pd.DataFrame({
        'Neuron_ID': data.iloc[:, 0],  # 假设第一列是神经元编号
        'Cluster': cluster_labels
    })

    # 保存结果到新文件
    results.to_csv(output_filepath, index=False)

    print("\n聚类完成！")
    print(f"结果已保存到：{output_filepath}\n")

    # --- 新功能：计算并显示每个集群的比例 ---
    print("集群分析结果：")
    # 计算每个集群标签的数量
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    total_count = len(cluster_labels)

    for cluster_id, count in cluster_counts.items():
        percentage = (count / total_count) * 100
        print(f"  - 集群 {cluster_id}: {count} 个数据点, 占比: {percentage:.2f}%")

except FileNotFoundError:
    print(f"错误：找不到文件 {input_filepath}")
except Exception as e:
    print(f"发生错误：{str(e)}")