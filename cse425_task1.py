
#easy task
# 1) Setup
!pip -q install torch torchaudio librosa numpy pandas scikit-learn umap-learn matplotlib seaborn

import os, pandas as pd, numpy as np, torch, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import librosa

# Paths
CSV_PATH = "/content/data/lyrics/tracks.csv"
RESULTS_DIR = "/content/results"
os.makedirs(f"{RESULTS_DIR}/figures", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/metrics", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/latent", exist_ok=True)


# 2) Audio features (MFCC + deltas, aggregated stats)
def compute_mfcc_features(path, sr=22050, n_mfcc=20, hop_length=512):
    y, sr = librosa.load(path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    def agg(X):
        return np.concatenate([X.mean(axis=1), X.std(axis=1)])
    feat = np.concatenate([agg(mfcc), agg(delta), agg(delta2)]).astype(np.float32)
    return feat

def load_tracks(csv_path):
    df = pd.read_csv(csv_path)
    return df

def build_feature_matrix(df):
    feats, meta = [], []
    for _, row in df.iterrows():
        f = compute_mfcc_features(row['audio_path'])
        feats.append(f)
        meta.append({'track_id': row['track_id'], 'language': row.get('language', None)})
    X = np.vstack(feats)
    return X, meta

df = load_tracks(CSV_PATH)
X, meta = build_feature_matrix(df)

# 3) Basic VAE (MLP)
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld, recon_loss.detach(), kld.detach()

# 4) Train VAE
from torch.utils.data import TensorDataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Xnorm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
tensor = torch.tensor(Xnorm, dtype=torch.float32)
ds = TensorDataset(tensor)
dl = DataLoader(ds, batch_size=64, shuffle=True)

model = VAE(input_dim=X.shape[1], latent_dim=16).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for ep in range(50):
    model.train()
    losses = []
    for (batch,) in dl:
        batch = batch.to(device)
        opt.zero_grad()
        recon, mu, logvar = model(batch)
        loss, rl, kld = vae_loss(recon, batch, mu, logvar)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    if (ep+1) % 5 == 0:
        print(f"Epoch {ep+1}/50 | loss={np.mean(losses):.4f}")

# 5) Latent extraction
with torch.no_grad():
    mu, logvar = model.encode(tensor.to(device))
    Z = mu.cpu().numpy()
np.save(f"{RESULTS_DIR}/latent/Z_easy.npy", Z)

# 6) Clustering: K-Means on latent + PCA baseline
k = 2
km_latent = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(Z)
labels_latent = km_latent.labels_

pca = PCA(n_components=16, random_state=42)
Xp = pca.fit_transform(Xnorm)
km_pca = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(Xp)
labels_pca = km_pca.labels_

# 7) Visualization t-SNE & UMAP
T = TSNE(n_components=2, random_state=42).fit_transform(Z)
plt.figure(figsize=(6,5)); plt.scatter(T[:,0], T[:,1], c=labels_latent, cmap='tab10', s=12)
plt.title("t-SNE (VAE latent)"); plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/figures/tsne_latent_easy.png"); plt.close()

U = umap.UMAP(n_components=2, random_state=42).fit_transform(Z)
plt.figure(figsize=(6,5)); plt.scatter(U[:,0], U[:,1], c=labels_latent, cmap='tab10', s=12)
plt.title("UMAP (VAE latent)"); plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/figures/umap_latent_easy.png"); plt.close()

# 8) Metrics: Silhouette, CH, DB; ARI & NMI
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def unsup_scores(X2D, labels):
    if len(set(labels)) <= 1:
        return {}
    return {
        'silhouette': float(silhouette_score(X2D, labels)),
        'calinski_harabasz': float(calinski_harabasz_score(X2D, labels)),
        'davies_bouldin': float(davies_bouldin_score(X2D, labels))
    }

scores_latent = unsup_scores(Z, labels_latent)
scores_pca = unsup_scores(Xp, labels_pca)

lang_labels = [m['language'] for m in meta]
if all(isinstance(x, str) for x in lang_labels):
    uniq = {v:i for i,v in enumerate(sorted(set(lang_labels)))}
    y_true = np.array([uniq[v] for v in lang_labels])
    sup_latent = {
        'ARI': float(adjusted_rand_score(y_true, labels_latent)),
        'NMI': float(normalized_mutual_info_score(y_true, labels_latent))
    }
    sup_pca = {
        'ARI': float(adjusted_rand_score(y_true, labels_pca)),
        'NMI': float(normalized_mutual_info_score(y_true, labels_pca))
    }
else:
    sup_latent, sup_pca = {}, {}

pd.DataFrame([scores_latent, scores_pca], index=['VAE+KMeans','PCA+KMeans']).to_csv(f"{RESULTS_DIR}/metrics/unsupervised_easy.csv")
pd.DataFrame([sup_latent, sup_pca], index=['VAE+KMeans','PCA+KMeans']).to_csv(f"{RESULTS_DIR}/metrics/with_labels_easy.csv")

print("Easy task complete. Results saved to:", RESULTS_DIR)