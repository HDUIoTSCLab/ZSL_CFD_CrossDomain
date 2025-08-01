import os, re
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
from config import SAMPLE_LENGTH, fft_dim,COMPOSITE_DEF
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim=fft_dim, latent_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = self.fc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, output_dim=fft_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, z):
        return self.fc(z)

def compute_mmd(x1, x2, bandwidth=1.0):
    def rbf_kernel(a, b, gamma):
        aa, bb = a.unsqueeze(1), b.unsqueeze(0)
        return torch.exp(-gamma * ((aa - bb) ** 2).sum(2))

    gamma = 1.0 / (2 * bandwidth ** 2)
    kxx = rbf_kernel(x1, x1, gamma).mean()
    kyy = rbf_kernel(x2, x2, gamma).mean()
    kxy = rbf_kernel(x1, x2, gamma).mean()
    return kxx + kyy - 2 * kxy

def load_fft_from_dir(data_dir, fault_type, max_num=100):
    X, count = [], 0
    for fname in os.listdir(data_dir):
        if not fname.endswith('.mat'):
            continue
        match = re.search(r'\d+\s+([A-Z]+)_', fname)
        if not match or match.group(1) != fault_type:
            continue
        data = sio.loadmat(os.path.join(data_dir, fname)).get('Data', None)
        if data is None:
            continue
        data = data[:, 2]
        for _ in range(max_num):
            if len(data) < SAMPLE_LENGTH:
                break
            start = np.random.randint(0, len(data) - SAMPLE_LENGTH)
            segment = data[start:start + SAMPLE_LENGTH]
            fft = np.abs(np.fft.fft(segment))[:fft_dim]
            fft = (fft - np.mean(fft)) / (np.std(fft) + 1e-8)
            X.append(fft)
            count += 1
            if count >= max_num:
                break
    return np.array(X)

def generate_target_compound_fault(
    source_dir="./Data/F0", target_dir="./Data/S800",
    save_path="./synthetic_composite.npy", sample_per_class=100, epochs=100
):
    encoder, decoder = Encoder().to(device), Decoder().to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    MSE = nn.MSELoss()

    norm_src = load_fft_from_dir(source_dir, "N", max_num=sample_per_class)
    norm_tgt = load_fft_from_dir(target_dir, "N", max_num=sample_per_class)
    real_data = {label: load_fft_from_dir(source_dir, label, max_num=sample_per_class) for label in COMPOSITE_DEF}

    norm_src_tensor = torch.tensor(norm_src, dtype=torch.float32).to(device)
    norm_tgt_tensor = torch.tensor(norm_tgt, dtype=torch.float32).to(device)

    trial_start = time.time()

    for epoch in range(epochs):
        total_loss = 0
        for label, data in real_data.items():
            x = torch.tensor(data, dtype=torch.float32).to(device)
            z, mu, logvar = encoder(x)
            recon = decoder(z)
            recon_loss = MSE(recon, x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            z_src, _, _ = encoder(norm_src_tensor)
            z_tgt, _, _ = encoder(norm_tgt_tensor)
            mmd_loss = compute_mmd(z_src, z_tgt, bandwidth=1.0)
            loss = recon_loss + kl_loss + 0.1 * mmd_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 100 == 0:
            print(f"[Epoch {epoch+1}] Total Loss: {total_loss:.4f}")

    encoder.eval(); decoder.eval()
    all_synth = {}
    with torch.no_grad():
        for label, data in real_data.items():
            x = torch.tensor(data, dtype=torch.float32).to(device)
            z, _, _ = encoder(x)
            synth = decoder(z).cpu().numpy()
            all_synth[label] = synth

    print(f"复合故障样本生成总耗时：{time.time() - trial_start:.2f} 秒")

    np.save(save_path, all_synth)
    print(f"📦 合成数据已保存至: {save_path}")

