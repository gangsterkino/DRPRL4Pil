# 示例数据，实际数据应根据您的实验结果填充
import matplotlib.pyplot as plt
import numpy as np

# 假设有四个实验的数据，每个实验六个尺度上的指标数值
experiments = ['Baseline', 'Without Saliency', 'Without Multiscale', 'Without Both']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'top-k Consistency', "Cohen's kappa"]
scales = np.arange(6)  # 尺度0到5

# 示例数据，实际数据应根据您的实验结果填充
# 每个实验每个指标在每个尺度上的数值
# 格式：experiments_metrics[scale][experiment_index]
experiments_metrics = {
    0: [[0.92, 0.89, 0.91, 0.90, 0.88, 0.90], [0.89, 0.87, 0.88, 0.87, 0.85, 0.87], [0.90, 0.88, 0.89, 0.88, 0.86, 0.88], [0.85, 0.82, 0.84, 0.83, 0.81, 0.83]],
    1: [[0.89, 0.86, 0.88, 0.87, 0.85, 0.87], [0.86, 0.83, 0.85, 0.84, 0.82, 0.84], [0.87, 0.85, 0.86, 0.85, 0.83, 0.85], [0.82, 0.80, 0.81, 0.80, 0.79, 0.81]],
    2: [[0.90, 0.87, 0.89, 0.88, 0.86, 0.88], [0.88, 0.85, 0.87, 0.86, 0.84, 0.86], [0.89, 0.87, 0.88, 0.87, 0.86, 0.87], [0.84, 0.82, 0.83, 0.82, 0.81, 0.82]],
    3: [[0.85, 0.82, 0.84, 0.83, 0.81, 0.83], [0.83, 0.80, 0.82, 0.81, 0.79, 0.81], [0.84, 0.82, 0.83, 0.82, 0.81, 0.82], [0.81, 0.78, 0.79, 0.78, 0.77, 0.78]]
}

# 绘制每个指标的图表
for metric in metrics:
    plt.figure(figsize=(10, 6))
    for i, experiment in enumerate(experiments):
        plt.plot(scales, [experiments_metrics[scale][i][metrics.index(metric)] for scale in scales], marker='o', label=experiment)

    plt.title(f'{metric} Comparison Across Scales')
    plt.xlabel('Scale')
    plt.ylabel(metric)
    plt.xticks(scales)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
