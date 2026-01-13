
#medium task
# 1) Setup
!pip -q install torch torchaudio librosa numpy pandas scikit-learn umap-learn matplotlib seaborn sentence-transformers

import os, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
import librosa, torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

CSV_PATH = "/content/data/lyrics/tracks.csv"
RESULTS_DIR = "/content/results"
os.makedirs(f"{RESULTS_DIR}/figures", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/metrics", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/latent", exist_ok=True)

# 2) Log-mel spectrograms
def compute_logmel(path, sr=22050, n_mels=128, n_fft=1024, hop_length=256, max_frames=256):
    y, sr = librosa.load(path, sr=sr, mono=True)
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    LM = librosa.power_to_db(M, ref=np.max)
    if LM.shape[1] < max_frames:
        pad = max_frames - LM.shape[1]
        LM = np.pad(LM, ((0,0),(0,pad)), mode='constant', constant_values=LM.min())
    else:
        LM = LM[:, :max_frames]
    LM = (LM - LM.mean()) / (LM.std() + 1e-8)
    return LM.astype(np.float32)  # [128,256]

def load_tracks(csv_path):
    return pd.read_csv(csv_path)

df = load_tracks(CSV_PATH)
Xmel = np.stack([compute_logmel(p) for p in df['audio_path']])  # [N,128,256]
Xmel_t = torch.tensor(Xmel).unsqueeze(1)  # [N,1,128,256]

# 3) Conv-VAE
class ConvVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU()
        )
        self.flatten = nn.Flatten()
        # Input 128x256 -> downsample /8 -> 16x32 feature map
        self.fc_mu = nn.Linear(128 * 16 * 32, latent_dim)
        self.fc_logvar = nn.Linear(128 * 16 * 32, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 128 * 16 * 32)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def encode(self, x):
        h = self.enc(x)
        h = self.flatten(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 128, 16, 32)
        xhat = self.dec(h)
        return xhat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ConvVAE(in_channels=1, latent_dim=32).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4) Train Conv-VAE (full-batch for simplicity; adjust for memory)
Xmel_t = Xmel_t.to(device)
for ep in range(30):
    model.train()
    opt.zero_grad()
    recon, mu, logvar = model(Xmel_t)
    recon_loss = nn.functional.mse_loss(recon, Xmel_t)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kld
    loss.backward()
    opt.step()
    if (ep+1) % 5 == 0:
        print(f"Epoch {ep+1}/30 | loss={loss.item():.4f} | RL={recon_loss.item():.4f} | KLD={kld.item():.4f}")

with torch.no_grad():
    mu, _ = model.encode(Xmel_t)
    Z_audio = mu.cpu().numpy()
np.save(f"{RESULTS_DIR}/latent/Z_audio_medium.npy", Z_audio)

# 5) Lyrics embeddings (multilingual LaBSE)
embedder = SentenceTransformer("sentence-transformers/LaBSE")
texts = df['lyrics_text'].fillna("").tolist()
Z_lyrics = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)
np.save(f"{RESULTS_DIR}/latent/Z_lyrics_medium.npy", Z_lyrics)

# 6) Fusion and clustering variants
Z_fused = np.concatenate([Z_audio, Z_lyrics], axis=1)
np.save(f"{RESULTS_DIR}/latent/Z_fused_medium.npy", Z_fused)

k = 2
labels_km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(Z_fused)
labels_ag = AgglomerativeClustering(n_clusters=k).fit_predict(Z_fused)
labels_db = DBSCAN(eps=0.5, min_samples=5).fit_predict(Z_fused)

# 7) Visualization
T = TSNE(n_components=2, random_state=42).fit_transform(Z_fused)
plt.figure(figsize=(6,5)); plt.scatter(T[:,0], T[:,1], c=labels_km, cmap='tab10', s=12)
plt.title("t-SNE (audio+lyrics fused)"); plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/figures/tsne_fused_medium.png"); plt.close()

U = umap.UMAP(n_components=2, random_state=42).fit_transform(Z_fused)
plt.figure(figsize=(6,5)); plt.scatter(U[:,0], U[:,1], c=labels_km, cmap='tab10', s=12)
plt.title("UMAP (audio+lyrics fused)"); plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/figures/umap_fused_medium.png"); plt.close()

# 8) Metrics
def unsup_scores(X2D, labels):
    if len(set(labels)) <= 1:
        return {}
    return {
        'silhouette': float(silhouette_score(X2D, labels)),
        'calinski_harabasz': float(calinski_harabasz_score(X2D, labels)),
        'davies_bouldin': float(davies_bouldin_score(X2D, labels))
    }

unsup_km = unsup_scores(Z_fused, labels_km)
unsup_ag = unsup_scores(Z_fused, labels_ag)
unsup_db = unsup_scores(Z_fused, labels_db)

lang_labels = df['language'].tolist()
uniq = {v:i for i,v in enumerate(sorted(set(lang_labels)))}
y_true = np.array([uniq[v] for v in lang_labels])
sup_km = {
    'ARI': float(adjusted_rand_score(y_true, labels_km)),
    'NMI': float(normalized_mutual_info_score(y_true, labels_km))
}

pd.DataFrame([unsup_km, unsup_ag, unsup_db], index=['KMeans','Agglomerative','DBSCAN']).to_csv(f"{RESULTS_DIR}/metrics/unsupervised_medium.csv")
pd.DataFrame([sup_km], index=['KMeans']).to_csv(f"{RESULTS_DIR}/metrics/with_labels_medium.csv")

# Optional: reconstruction comparison plot
with torch.no_grad():
    recon, _, _ = model(Xmel_t[:4])
R = recon.cpu().numpy()[0,0]
I = Xmel[0]
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(I, aspect='auto', origin='lower'); plt.title("Input log-mel")
plt.subplot(1,2,2); plt.imshow(R, aspect='auto', origin='lower'); plt.title("Reconstruction")
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/figures/reconstruction_medium.png"); plt.close()

print("Medium task complete. Results saved to:", RESULTS_DIR)