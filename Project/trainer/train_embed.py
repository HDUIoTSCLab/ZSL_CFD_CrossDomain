import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import euclidean_distances

def generate_semihard_triplets(X, Y, margin=1.0, max_triplets=3000):
    triplets = []
    distances = euclidean_distances(X, X)
    label_to_indices = {label: np.where(Y == label)[0] for label in np.unique(Y)}
    for anchor_idx in range(len(X)):
        anchor_label = Y[anchor_idx]
        positive_indices = label_to_indices[anchor_label]
        negative_indices = np.concatenate([label_to_indices[l] for l in label_to_indices if l != anchor_label])
        positive_indices = positive_indices[positive_indices != anchor_idx]
        if len(positive_indices) == 0:
            continue
        pos_dists = distances[anchor_idx][positive_indices]
        sorted_pos = positive_indices[np.argsort(pos_dists)]
        semi_pos = sorted_pos[min(3, len(sorted_pos) - 1)]
        neg_dists = distances[anchor_idx][negative_indices]
        hard_negatives = negative_indices[(neg_dists > distances[anchor_idx][semi_pos]) & (neg_dists < distances[anchor_idx][semi_pos] + margin)]
        if len(hard_negatives) == 0:
            continue
        semi_neg = np.random.choice(hard_negatives)
        triplets.append((anchor_idx, semi_pos, semi_neg))
        if len(triplets) >= max_triplets:
            break
    return triplets

def run_manual_training(X_tensor, triplets, model_embed, optimizer, batch_size=1024):
    model_embed.train()
    np.random.shuffle(triplets)
    for i in range(0, len(triplets), batch_size):
        batch = triplets[i:i + batch_size]
        if len(batch) == 0:
            continue
        a, p, n = zip(*batch)
        anchor = X_tensor[list(a)]
        positive = X_tensor[list(p)]
        negative = X_tensor[list(n)]
        anchor_embed = model_embed(anchor)
        positive_embed = model_embed(positive)
        negative_embed = model_embed(negative)
        loss = nn.TripletMarginLoss(margin=1)(anchor_embed, positive_embed, negative_embed)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()