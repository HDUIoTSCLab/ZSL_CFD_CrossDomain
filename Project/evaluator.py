import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from config import SAMPLES_PER_SIMPLE_CLASS, COMPOSITE_DEF, TEST_COMPOSITE, TEST_DIRS, TRAIN_TYPES,SAMPLE_LENGTH,MAX_SAMPLES_PER_CLASS
from data_loader import load_data_from_dir, load_data_per_fault_class
from models.embedding_net import EmbeddingNet
from models.transformer_synth import TransformerSynthesizer
from utils import get_class_centers
from visualize_tsne import plot_tsne
import torch.nn.functional as F
from sklearn.metrics import classification_report
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_on_all_test_dirs(model_embed, synthesizer, label_centers, test_dirs):
    acc_list = []
    for test_dir in test_dirs:
        samples_per_class_dict = {label: COMPOSITE_DEF[label]['num'] for label in COMPOSITE_DEF}
        X_test, Y_test = load_data_from_dir(test_dir, TEST_COMPOSITE, samples_per_class_dict)
        if len(X_test) == 0:
            acc_list.append(0.0)
            continue
        with torch.no_grad():
            test_embed = model_embed(torch.tensor(X_test, dtype=torch.float32).to(next(model_embed.parameters()).device)).cpu().numpy()
        test_labels = [label for label in Y_test if label in TEST_COMPOSITE]
        test_embed_tensor = torch.tensor(test_embed[[i for i, l in enumerate(Y_test) if l in TEST_COMPOSITE]], dtype=torch.float32).to(next(model_embed.parameters()).device)
        test_targets = torch.tensor([TEST_COMPOSITE.index(l) for l in test_labels], dtype=torch.long).to(test_embed_tensor.device)
        with torch.no_grad():
            synth_prototypes = []
            for label in TEST_COMPOSITE:
                parts = COMPOSITE_DEF[label]['parts']
                part_vecs = [label_centers[p].unsqueeze(0) for p in parts]
                syn = synthesizer(part_vecs)
                synth_prototypes.append(syn)
            synth_prototypes_tensor = torch.cat(synth_prototypes, dim=0)
            sims = torch.matmul(test_embed_tensor, synth_prototypes_tensor.transpose(0, 1))
            preds = torch.argmax(sims, dim=1)
            correct = (preds == test_targets).sum().item()
            acc = correct / len(test_targets) if len(test_targets) > 0 else 0.0
            acc_list.append(acc)
    return acc_list

def evaluate_best_model(model_path="best_model.pt", runs=10):
    print(f"\n🧪 开始使用最佳模型 {model_path} 进行测试工况评估，重复运行 {runs} 次...")

    all_run_accs = []
    all_dir_accs = {dir.split('/')[-1]: [] for dir in TEST_DIRS}
    all_fault_precisions = {label: [] for label in TEST_COMPOSITE}

    for run_idx in range(runs):
        print(f"\n🚀 第 {run_idx + 1}/{runs} 次评估开始...")
        trial_start = time.time()

        X_train, Y_train = load_data_per_fault_class(SAMPLES_PER_SIMPLE_CLASS)
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)

        model_embed = EmbeddingNet(SAMPLE_LENGTH // 2, 128).to(device)
        synthesizer = TransformerSynthesizer(128).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        model_embed.load_state_dict(checkpoint['embed'])
        synthesizer.load_state_dict(checkpoint['synth'])

        model_embed.eval()
        synthesizer.eval()

        with torch.no_grad():
            embed_train = model_embed(X_tensor).cpu().numpy()
        centers = get_class_centers(embed_train, Y_train, TRAIN_TYPES)
        label_centers = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in centers.items()}

        overall_preds = []
        overall_trues = []

        for test_dir in TEST_DIRS:
            dir_name = test_dir.split('/')[-1]
            samples_per_class_dict = {label: COMPOSITE_DEF[label]['num'] for label in COMPOSITE_DEF}
            X_test, Y_test = load_data_from_dir(test_dir, TEST_COMPOSITE, samples_per_class_dict)
            if len(X_test) == 0:
                print(f"⚠️ 工况 {dir_name} 无有效样本")
                continue

            with torch.no_grad():
                test_embed_tensor = model_embed(torch.tensor(X_test, dtype=torch.float32).to(device))
                synth_prototypes = []
                for label in TEST_COMPOSITE:
                    parts = COMPOSITE_DEF[label]['parts']
                    part_vecs = [label_centers[p].unsqueeze(0) for p in parts]
                    syn = synthesizer(part_vecs)
                    synth_prototypes.append(syn)
                synth_prototypes_tensor = torch.cat(synth_prototypes, dim=0)
                sims = torch.matmul(test_embed_tensor, synth_prototypes_tensor.T)
                preds = torch.argmax(sims, dim=1).cpu().numpy()

            test_targets = [TEST_COMPOSITE.index(label) for label in Y_test]
            overall_preds.extend(preds)
            overall_trues.extend(test_targets)

            acc = np.mean(preds == np.array(test_targets))
            all_dir_accs[dir_name].append(acc)

        # 计算总准确率与分类报告
        total_acc = accuracy_score(overall_trues, overall_preds)
        all_run_accs.append(total_acc)
        print(f"🎯 第 {run_idx + 1} 次评估总准确率: {total_acc:.4f}")
        print(f"🕓 用时：{time.time() - trial_start:.2f} 秒")

        # 收集每类的 precision
        report = classification_report(overall_trues, overall_preds, target_names=TEST_COMPOSITE, digits=4, output_dict=True)
        for label in TEST_COMPOSITE:
            precision = report[label]['precision']
            all_fault_precisions[label].append(precision)

    # ================== 结果统计 ==================
    print("\n=================== 📊 评估统计结果 ===================")
    print(f"🎯 所有工况平均准确率: {np.mean(all_run_accs):.4f} ± {np.std(all_run_accs):.4f}")

    print("\n🗂 各测试工况平均准确率：")
    for dir_name, acc_list in all_dir_accs.items():
        acc_mean = np.mean(acc_list)
        acc_std = np.std(acc_list)
        print(f" - 工况 {dir_name:<8} | 平均准确率: {acc_mean:.4f} ± {acc_std:.4f}")

    print("\n📌 每类复合故障平均准确率（Precision）：")
    for label in TEST_COMPOSITE:
        mean_p = np.mean(all_fault_precisions[label])
        std_p = np.std(all_fault_precisions[label])
        print(f" - 故障 {label:<6} | 平均 Precision: {mean_p:.4f} ± {std_p:.4f}")


def compare_tsne_real_vs_synth(
    model_path="best_model.pt",
    tsne_test_dir="./Data/F800",
    composite_classes=TEST_COMPOSITE,
    sample_num=MAX_SAMPLES_PER_CLASS
):

    print(f"\n📌 开始可视化对比：{tsne_test_dir} 中的复合故障 {composite_classes}...\n")

    # 加载模型
    model_embed = EmbeddingNet(SAMPLE_LENGTH // 2, 128).to(device)
    synthesizer = TransformerSynthesizer(128).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model_embed.load_state_dict(checkpoint['embed'])
    synthesizer.load_state_dict(checkpoint['synth'])
    model_embed.eval()
    synthesizer.eval()

    # 获取训练集单一故障嵌入中心
    X_train, Y_train = load_data_per_fault_class(SAMPLES_PER_SIMPLE_CLASS)
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    with torch.no_grad():
        embed_train = model_embed(X_tensor).cpu().numpy()
    centers = get_class_centers(embed_train, Y_train, TRAIN_TYPES)
    label_centers = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in centers.items()}

    real_embed_list = []
    real_label_list = []
    synth_embed_list = []
    synth_label_list = []

    for label in composite_classes:
        # 真实样本嵌入（每条样本都加入）
        X_val, Y_val = load_data_from_dir(
            data_dir=tsne_test_dir,
            fault_types=[label],
            samples_per_class_dict={label: sample_num}
        )
        if len(X_val) > 0:
            val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            with torch.no_grad():
                embed = model_embed(val_tensor).cpu().numpy()
            for e in embed:
                real_embed_list.append(e)
                real_label_list.append(f"{label}_Real")

        # 合成语义中心
        parts = COMPOSITE_DEF[label]["parts"]
        part_vecs = [label_centers[p].unsqueeze(0) for p in parts]
        with torch.no_grad():
            syn = synthesizer(part_vecs).cpu().numpy()
        synth_embed_list.append(syn.squeeze(0))
        synth_label_list.append(f"{label}_Synth")

    # 可视化对比
    plot_tsne(real_embed_list, real_label_list, synth_embed_list, synth_label_list,
              title=f"{tsne_test_dir.split('/')[-1]} 工况下：真实语义 vs 合成语义对比")

def compare_real_vs_generated_on_target(
    model_path="best_model.pt",
    real_dir="./Data/S800",
    synth_path="./synthetic_s800_composite.npy",
    composite_classes=TEST_COMPOSITE,
    sample_num=MAX_SAMPLES_PER_CLASS
):
    """
    在目标工况下对比真实复合故障样本与合成样本的语义分布（t-SNE 可视化）
    """
    print(f"\n📌 正在对比目标工况 {real_dir} 下的真实与合成复合故障语义分布...\n")
    from visualize_tsne import plot_tsne

    model_embed = EmbeddingNet(SAMPLE_LENGTH // 2, 128).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model_embed.load_state_dict(checkpoint['embed'])
    model_embed.eval()

    real_embed_list = []
    real_label_list = []
    synth_embed_list = []
    synth_label_list = []

    # 加载合成数据
    synth_dict = np.load(synth_path, allow_pickle=True).item()

    for label in composite_classes:
        # 真实复合故障数据
        X_real, Y_real = load_data_from_dir(
            data_dir=real_dir,
            fault_types=[label],
            samples_per_class_dict={label: sample_num}
        )
        if len(X_real) > 0:
            with torch.no_grad():
                emb = model_embed(torch.tensor(X_real, dtype=torch.float32).to(device)).cpu().numpy()
            for e in emb:
                real_embed_list.append(e)
                real_label_list.append(f"{label}_Real")

        # 合成数据
        X_synth = synth_dict.get(label, None)
        if X_synth is not None and len(X_synth) > 0:
            X_synth = X_synth[:sample_num]
            with torch.no_grad():
                emb_s = model_embed(torch.tensor(X_synth, dtype=torch.float32).to(device)).cpu().numpy()
            for e in emb_s:
                synth_embed_list.append(e)
                synth_label_list.append(f"{label}_Synth")

    plot_tsne(real_embed_list, real_label_list, synth_embed_list, synth_label_list,
              title=f"{real_dir.split('/')[-1]} 工况下：真实 vs 合成复合故障语义对比")


def compare_real_and_synth_semantics_between_domains(
        model_path="best_model.pt",
        source_dir="./Data/F0",
        target_dir="./Data/S800",
        synth_path="./synthetic_s800_composite.npy",
        composite_classes=None,
        sample_num=100
):
    """
    对比源工况、目标工况真实复合故障样本 与 迁移生成复合故障样本的语义嵌入表示（t-SNE 可视化）

    - 源工况真实样本：圆形 'o'
    - 目标工况真实样本：三角形 '^'
    - 迁移生成样本：方形 's'
    """

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    print(f"\n📌 正在对比源工况 {source_dir}、目标工况 {target_dir} 与迁移样本的复合故障语义嵌入分布...\n")

    if composite_classes is None:
        composite_classes = ["IOF", "IBF", "ICF", "OBF", "IOBF"]

    # 加载嵌入网络
    model_embed = EmbeddingNet(SAMPLE_LENGTH // 2, 128).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model_embed.load_state_dict(checkpoint['embed'])
    model_embed.eval()

    # 加载迁移生成的数据
    synth_dict = np.load(synth_path, allow_pickle=True).item()

    embed_list = []
    label_list = []
    marker_list = []

    for label in composite_classes:
        # 源工况真实复合故障
        X_source, _ = load_data_from_dir(
            data_dir=source_dir,
            fault_types=[label],
            samples_per_class_dict={label: sample_num}
        )
        if len(X_source) > 0:
            with torch.no_grad():
                emb_s = model_embed(torch.tensor(X_source, dtype=torch.float32).to(device)).cpu().numpy()
            embed_list.extend(emb_s)
            label_list.extend([f"{label}_Source"] * len(emb_s))
            marker_list.extend(['o'] * len(emb_s))  # 圆形

        # 目标工况真实复合故障
        X_target, _ = load_data_from_dir(
            data_dir=target_dir,
            fault_types=[label],
            samples_per_class_dict={label: sample_num}
        )
        if len(X_target) > 0:
            with torch.no_grad():
                emb_t = model_embed(torch.tensor(X_target, dtype=torch.float32).to(device)).cpu().numpy()
            embed_list.extend(emb_t)
            label_list.extend([f"{label}_Target"] * len(emb_t))
            marker_list.extend(['^'] * len(emb_t))  # 三角形

        # 迁移生成复合故障
        X_synth = synth_dict.get(label, None)
        if X_synth is not None and len(X_synth) > 0:
            X_synth = X_synth[:sample_num]
            with torch.no_grad():
                emb_gen = model_embed(torch.tensor(X_synth, dtype=torch.float32).to(device)).cpu().numpy()
            embed_list.extend(emb_gen)
            label_list.extend([f"{label}_Synth"] * len(emb_gen))
            marker_list.extend(['s'] * len(emb_gen))  # 方形

    # t-SNE 降维
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    tsne_result = tsne.fit_transform(np.array(embed_list))

    # 可视化
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(list(set(label_list)))
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    for i, label in enumerate(unique_labels):
        indices = [j for j, l in enumerate(label_list) if l == label]
        marker = marker_list[indices[0]]
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1],
                    label=label, marker=marker, s=120)#调节各类故障的图标大小120 or 40

    plt.title(f"复合故障语义分布对比：{source_dir.split('/')[-1]} → {target_dir.split('/')[-1]}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

