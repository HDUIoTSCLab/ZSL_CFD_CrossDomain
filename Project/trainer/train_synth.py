import torch
import torch.nn as nn
import numpy as np
from config import SAMPLE_LENGTH, TEST_COMPOSITE
from data_loader import load_data_from_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_synthesizer(model_embed, synthesizer, label_centers, COMPOSITE_DEF,
                      data_source=None, epochs=1000, sample_num=100, print_every=50):
    composite_labels = list(COMPOSITE_DEF.keys())
    label2idx = {label: idx for idx, label in enumerate(composite_labels)}

    optimizer = torch.optim.Adam(synthesizer.parameters(), lr=1e-3)
    triplet_criterion = nn.TripletMarginLoss(margin=1.0)
    ce_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        synthesizer.train()
        synth_prototypes = []
        label_list = []

        # ✅ 合成当前所有复合标签的语义表示
        for label in composite_labels:
            parts = COMPOSITE_DEF[label]["parts"]
            part_vecs = [label_centers[p].unsqueeze(0) for p in parts]
            syn = synthesizer(part_vecs)  # shape: [1, dim]
            synth_prototypes.append(syn)
            label_list.append(label2idx[label])

        synth_tensor = torch.cat(synth_prototypes, dim=0)  # [num_classes, dim]

        # === 分支1：使用真实复合样本 ===
        if callable(data_source) or (isinstance(data_source, str) and data_source.endswith(".npy")):
            if isinstance(data_source, str):
                synth_dict = np.load(data_source, allow_pickle=True).item()
                X_val, Y_val = [], []
                for label in composite_labels:
                    data = synth_dict.get(label, None)
                    if data is not None:
                        data = data[:sample_num]
                        X_val.append(data)
                        Y_val += [label] * len(data)
                X_val = np.concatenate(X_val, axis=0)
            else:
                X_val, Y_val = data_source()

            val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            with torch.no_grad():
                val_embed = model_embed(val_tensor)
            val_targets = torch.tensor([label2idx[l] for l in Y_val], dtype=torch.long).to(device)

        # === 分支2：使用合成语义自身作为训练样本 ===
        else:
            val_embed = synth_tensor.detach()
            val_targets = torch.tensor(label_list, dtype=torch.long).to(device)

        # === Loss 计算 ===
        sims = torch.matmul(val_embed, synth_tensor.T)  # [N, num_classes]
        ce_loss = ce_loss_fn(sims, val_targets)

        # Triplet Loss
        triplet_loss_total = 0.0
        count = 0
        for i, label_idx in enumerate(val_targets):
            anchor = synth_tensor[label_idx].unsqueeze(0)
            positive = val_embed[i].unsqueeze(0)
            neg_indices = [j for j in range(len(val_targets)) if j != label_idx]
            if not neg_indices:
                continue
            j = np.random.choice(neg_indices)
            negative = val_embed[j].unsqueeze(0)
            triplet_loss_total += triplet_criterion(anchor, positive, negative)
            count += 1
        triplet_loss_avg = triplet_loss_total / max(count, 1)

        # === 总损失 ===
        loss = ce_loss + triplet_loss_avg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"[Epoch {epoch+1:03d}] Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}, Triplet: {triplet_loss_avg.item():.4f})")
