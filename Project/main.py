import torch
import numpy as np
from config import SAMPLES_PER_SIMPLE_CLASS, COMPOSITE_DEF, TEST_COMPOSITE, TEST_DIRS, TRAIN_TYPES, GENERATE_COMPOUND_FAULT_NUM,SOURCE_DIR,TARGET_DIR,SAMPLE_LENGTH
from data_loader import load_data_per_fault_class, load_few_shot_composite_samples_by_type
from models.embedding_net import EmbeddingNet
from models.transformer_synth import TransformerSynthesizer
from trainer.train_embed import generate_semihard_triplets, run_manual_training
from trainer.train_synth import train_synthesizer
from evaluator import evaluate_on_all_test_dirs, evaluate_best_model,compare_tsne_real_vs_synth, compare_real_vs_generated_on_target,compare_real_and_synth_semantics_between_domains

from utils import get_class_centers
import time
from synthetic_fault_generator import generate_target_compound_fault

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_pipeline(embed_epochs=10, synth_epochs=200, trials=10):
    best_score = -1
    best_state = None

    for trial in range(trials):
        print(f"\n🚀 [Trial {trial + 1}/{trials}] 开始训练...")
        trial_start = time.time()

        # 🧠 加载训练数据
        X_train, Y_train = load_data_per_fault_class(SAMPLES_PER_SIMPLE_CLASS)
        model_embed = EmbeddingNet(SAMPLE_LENGTH // 2, 128).to(device)
        synthesizer = TransformerSynthesizer(128).to(device)
        optimizer = torch.optim.Adam(model_embed.parameters(), lr=1e-3)

        # 🔁 生成 Triplets
        print("🔧 生成 Triplet 训练对...")
        triplets = generate_semihard_triplets(X_train, Y_train, margin=10.0, max_triplets=3000)
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)

        # 🧠 阶段一：训练嵌入网络
        print(f"🧠 开始 EmbeddingNet 训练（{embed_epochs} epochs）...")
        for epoch in range(embed_epochs):
            run_manual_training(X_tensor, triplets, model_embed, optimizer)
        print(f"✅ EmbeddingNet 训练完成，共执行 {embed_epochs} 轮训练。")

        # ✅ 计算类中心
        with torch.no_grad():
            embed_train = model_embed(X_tensor).cpu().numpy()
        centers = get_class_centers(embed_train, Y_train, TRAIN_TYPES)
        label_centers = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in centers.items()}

        # 🧠 阶段二：训练合成器
        print(f"✨ 开始 Synthesizer 训练（{synth_epochs} epochs）...")

        #合成复合故障数据,使用迁移的复合故障
        train_synthesizer(
            model_embed, synthesizer, label_centers, COMPOSITE_DEF,
            "./synthetic_composite.npy",sample_num=GENERATE_COMPOUND_FAULT_NUM, epochs=synth_epochs,
            print_every=50  # ✅ 每 50 个 epoch 打印一次损失
        )
        '''
        # 合成复合故障数据,不使用迁移的复合故障
        train_synthesizer(model_embed, synthesizer, label_centers, COMPOSITE_DEF, data_source=None,
                          epochs=synth_epochs,print_every=50  # ✅ 每 50 个 epoch 打印一次损失
        )
        '''

        # 📊 模型评估
        print(f"📊 开始评估第 {trial + 1} 次模型...")
        accs = evaluate_on_all_test_dirs(model_embed, synthesizer, label_centers, TEST_DIRS)
        score = np.mean(accs)
        print(f"🎯 当前轮性能：Score = {score:.4f}")

        # ✅ 记录最佳模型
        if score > best_score:
            best_score = score
            best_state = {'embed': model_embed.state_dict(), 'synth': synthesizer.state_dict()}
            print("🔥 更新最佳模型参数")

        print(f"🕓 第 {trial + 1} 次训练总耗时：{time.time() - trial_start:.2f} 秒")

    torch.save(best_state, 'best_model.pt')
    print(f"\n🏆 训练完成，最佳模型已保存为 'best_model.pt'，最佳评分: {best_score:.4f}")



if __name__ == "__main__":

    #模型训练
    # 生成复合故障
    generate_target_compound_fault(source_dir=SOURCE_DIR,target_dir=TARGET_DIR,sample_per_class=GENERATE_COMPOUND_FAULT_NUM, epochs=2000)
    run_pipeline()
    #模型测试集
    evaluate_best_model("best_model.pt")





