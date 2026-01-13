
#hard task

# 1) Setup
!pip -q install torch torchaudio librosa numpy pandas scikit-learn umap-learn matplotlib seaborn sentence-transformers

import os, numpy as np, pandas as pd, torch, torch.nn as nn, matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

CSV_PATH = "/content/data/lyrics/tracks.csv"
RESULTS_DIR = "/content/results"
os.makedirs(f"{RESULTS_DIR}/figures", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/metrics", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/latent", exist_ok=True)

# 2) Audio summary stats (reusing MFCC aggregation)
import librosa
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
    return pd.read_csv(csv_path)

df = load_tracks(CSV_PATH)
Xa = np.vstack([compute_mfcc_features(p) for p in df['audio_path']])

# 3) Lyrics embeddings
embedder = SentenceTransformer("sentence-transformers/LaBSE")
texts = df['lyrics_text'].fillna("").tolist()
Zl = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)

# 4) Optional genre one-hot
genres = df.get('genre', pd.Series(['unknown']*len(df))).fillna('unknown').tolist()
g_map = {g:i for i,g in enumerate(sorted(set(genres)))}
G = np.eye(len(g_map))[ [ g_map[g] for g in genres ] ].astype(np.float32)

# 5) Multi-modal feature vector
Xmm = np.concatenate([Xa, Zl, G], axis=1).astype(np.float32)
Xmm_norm = (Xmm - Xmm.mean(axis=0)) / (Xmm.std(axis=0) + 1e-8)
Xten = torch.tensor(Xmm_norm, dtype=torch.float32)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Xten = Xten.to(device)

# 6) Beta-VAE (disentanglement)
class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden_dim=256, beta=4.0):
        super().__init__()
        self.beta = beta
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        recon = self.dec(z)
        recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kld
        return loss, recon_loss, kld, mu

beta_model = BetaVAE(input_dim=Xmm.shape[1], latent_dim=16, beta=4.0).to(device)
opt = torch.optim.Adam(beta_model.parameters(), lr=1e-3)

for ep in range(50):
    opt.zero_grad()
    loss, rl, kld, mu = beta_model(Xten)
    loss.backward()
    opt.step()
    if (ep+1) % 5 == 0:
        print(f"[BetaVAE] Epoch {ep+1}/50 loss={loss.item():.4f} RL={rl.item():.4f} KLD={kld.item():.4f}")

with torch.no_grad():
    _, _, _, mu = beta_model(Xten)
    Z_beta = mu.cpu().numpy()
np.save(f"{RESULTS_DIR}/latent/Z_beta_hard.npy", Z_beta)

# 7) Conditional VAE (condition on language)
langs = df['language'].fillna('en').tolist()
l_map = {v:i for i,v in enumerate(sorted(set(langs)))}  # e.g., {'bn':0,'en':1}
C_lang = np.eye(len(l_map))[ [ l_map[l] for l in langs ] ].astype(np.float32)
Cten = torch.tensor(C_lang, dtype=torch.float32).to(device)

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim=16, hidden_dim=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, c):
        h = self.enc(torch.cat([x, c], dim=1))
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        recon = self.dec(torch.cat([z, c], dim=1))
        recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld
        return loss, recon_loss, kld, mu

cvae = ConditionalVAE(input_dim=Xmm.shape[1], cond_dim=C_lang.shape[1], latent_dim=16).to(device)
opt2 = torch.optim.Adam(cvae.parameters(), lr=1e-3)

for ep in range(50):
    opt2.zero_grad()
    loss, rl, kld, mu_c = cvae(Xten, Cten)
    loss.backward()
    opt2.step()
    if (ep+1) % 5 == 0:
        print(f"[CVAE] Epoch {ep+1}/50 loss={loss.item():.4f} RL={rl.item():.4f} KLD={kld.item():.4f}")

with torch.no_grad():
    _, _, _, mu_c = cvae(Xten, Cten)
    Z_c = mu_c.cpu().numpy()
np.save(f"{RESULTS_DIR}/latent/Z_cvae_hard.npy", Z_c)

# 8) Clustering + visualization
k = len(l_map)  # typically 2 for bn/en
labels_beta = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(Z_beta)
labels_cvae = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(Z_c)

# t-SNE and UMAP
T_beta = TSNE(n_components=2, random_state=42).fit_transform(Z_beta)
plt.figure(figsize=(6,5)); plt.scatter(T_beta[:,0], T_beta[:,1], c=labels_beta, cmap='tab10', s=12)
plt.title("t-SNE (Beta-VAE latent)"); plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/figures/tsne_beta_hard.png"); plt.close()

U_c = umap.UMAP(n_components=2, random_state=42).fit_transform(Z_c)
plt.figure(figsize=(6,5)); plt.scatter(U_c[:,0], U_c[:,1], c=labels_cvae, cmap='tab10', s=12)
plt.title("UMAP (CVAE latent)"); plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/figures/umap_cvae_hard.png"); plt.close()

# 9) Metrics: unsupervised and with labels
def unsup_scores(X2D, labels):
    if len(set(labels)) <= 1:
        return {}
    return {
        'silhouette': float(silhouette_score(X2D, labels)),
        'calinski_harabasz': float(calinski_harabasz_score(X2D, labels)),
        'davies_bouldin': float(davies_bouldin_score(X2D, labels))
    }

unsup_beta = unsup_scores(Z_beta, labels_beta)
unsup_cvae = unsup_scores(Z_c, labels_cvae)

# Supervised (ARI/NMI) with language labels
uniq = {v:i for i,v in enumerate(sorted(set(langs)))}
y_true = np.array([uniq[v] for v in langs])
sup_beta = {'ARI': float(adjusted_rand_score(y_true, labels_beta)),
            'NMI': float(normalized_mutual_info_score(y_true, labels_beta))}
sup_cvae = {'ARI': float(adjusted_rand_score(y_true, labels_cvae)),
            'NMI': float(normalized_mutual_info_score(y_true, labels_cvae))}

pd.DataFrame([unsup_beta, unsup_cvae], index=['BetaVAE','CVAE']).to_csv(f"{RESULTS_DIR}/metrics/unsupervised_hard.csv")
pd.DataFrame([sup_beta, sup_cvae], index=['BetaVAE','CVAE']).to_csv(f"{RESULTS_DIR}/metrics/with_labels_hard.csv")

print("Hard task complete. Results saved to:", RESULTS_DIR)