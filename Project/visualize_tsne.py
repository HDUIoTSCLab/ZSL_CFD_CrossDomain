import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def plot_tsne(real_embed_list, real_label_list, synth_embed_list, synth_label_list, title="t-SNE Visualization"):
    vectors_tsne = []
    labels_tsne = []

    if real_embed_list:
        vectors_tsne.extend(real_embed_list)
        labels_tsne.extend(real_label_list)

    if synth_embed_list:
        vectors_tsne.extend(synth_embed_list)
        labels_tsne.extend(synth_label_list)

    # ✅ 强制统一维度
    vectors_tsne_fixed = []
    for v in vectors_tsne:
        v = np.asarray(v).squeeze()
        assert v.ndim == 1, f"向量维度错误: {v.shape}"
        vectors_tsne_fixed.append(v)
    vectors_tsne = np.array(vectors_tsne_fixed)

    # ✅ t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=min(30, len(vectors_tsne) // 3), random_state=42)
    vectors_2d = tsne.fit_transform(vectors_tsne)

    # ✅ 可视化
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(set(labels_tsne))
    palette = sns.color_palette("hls", len(unique_labels))
    label2color = {label: palette[i] for i, label in enumerate(unique_labels)}
    plotted_labels = set()

    for i, (x, y) in enumerate(vectors_2d):
        label = labels_tsne[i]
        marker = 'o' if 'Real' in label else 'x'
        color = label2color[label]
        if label not in plotted_labels:
            plt.scatter(x, y, c=[color], marker=marker, s=100, alpha=0.8, label=label)
            plotted_labels.add(label)
        else:
            plt.scatter(x, y, c=[color], marker=marker, s=100, alpha=0.8)

    plt.legend(loc='best', fontsize=9)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
