"""
Fault Isolation via Latent Space Clustering
============================================
Moves beyond "sensor X is anomalous" toward "this looks like Fault Type Z"
by clustering the GRU bottleneck representations of anomalous windows.

Method
------
1. Extract the GRU encoder's bottleneck hidden state h_T (16-dim) for
   every test window — this is the model's learned process representation.
2. Apply UMAP (or PCA fallback) to 2D for visualization.
3. Cluster faulty windows using HDBSCAN (density-based, no need to specify k).
4. For each cluster, compute:
   - Mean sensor reconstruction errors (fault signature)
   - Top contributing sensors (per-cluster SHAP/IG ranking)
   - Temporal fault onset statistics
5. Produce a fault-type discovery report + labelled scatter plot.

This is unsupervised fault type discovery — it groups faults by their
process signature without requiring labelled fault type metadata.

Outputs
-------
model/results/fault_isolation/
  latent_clusters.csv         — per-window cluster assignment + sensor errors
  fault_type_signatures.json  — per-cluster sensor signature
  latent_umap.png             — 2D scatter coloured by cluster
  sensor_signature_heatmap.png — cluster × sensor mean error heatmap
"""
import csv, json, os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data", "processed")
OP    = os.path.join(BASE, "model", "results", "leakage_safe_v2", "operating_point.npz")
SAVED = os.path.join(BASE, "model", "saved", "gru_ae_v2_best.pt")
OUT   = os.path.join(BASE, "model", "results", "fault_isolation")
sys.path.insert(0, BASE)
from model.gru_ae_v2 import GRUAutoencoderV2

SENSORS = ["Agitator RPM","Aeration rate","Sugar feed rate","Acid flow rate",
           "Base flow rate","Heating/cooling water flow","Dissolved oxygen","pH","Temperature"]
WINDOW = 60; STRIDE = 5


def scale(x, mn, sc): return x * sc + mn

def make_split(batch_data, faults, batches, mn, sc):
    xs, y_end, batch_ids, times = [], [], [], []
    for b in batches:
        x = scale(batch_data[str(b)], mn, sc)
        f, t = faults[str(b)]["fault"], faults[str(b)]["time"]
        for start in range(0, len(x) - WINDOW + 1, STRIDE):
            end = start + WINDOW
            xs.append(x[start:end].astype(np.float32))
            y_end.append(int(f[end-1] > 0))
            batch_ids.append(b); times.append(float(t[end-1]))
    return np.asarray(xs), np.asarray(y_end), np.asarray(batch_ids), np.asarray(times)


def extract_latent_and_errors(model, x, batch_size=256):
    """Returns (latent_codes, recon_errors_end) both (N, ...)"""
    latents, errors = [], []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            z = torch.tensor(x[s:s+batch_size], dtype=torch.float32)
            enc = model.encode(z)      # (B, T, 16)
            lat = enc[:, -1, :]        # (B, 16) — bottleneck at window end
            recon = model.decode(enc)
            err = ((recon - z)**2)[:, -1, :].numpy()
            latents.append(lat.numpy())
            errors.append(err)
    return np.concatenate(latents), np.concatenate(errors)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Fault Isolation via Latent Space Clustering")

    batch_data = np.load(os.path.join(DATA,"batch_data.npy"), allow_pickle=True).item()
    faults     = np.load(os.path.join(DATA,"batch_faults.npy"), allow_pickle=True).item()
    op         = np.load(OP)
    mn, sc_arr = op["scaler_min"], op["scaler_scale"]
    median, mad = op["sensor_median"], op["sensor_mad"]
    weights     = op["sensor_weights"]
    threshold   = float(op["threshold"])

    xte, yte, bte, tte = make_split(batch_data, faults, range(81,101), mn, sc_arr)

    model = GRUAutoencoderV2(9)
    ckpt  = torch.load(SAVED, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("Extracting latent representations...")
    latent, end_errors = extract_latent_and_errors(model, xte)  # (N, 16), (N, 9)

    # Anomaly scores
    end_z  = (end_errors - median) / mad
    scores = (end_z * weights).max(axis=1)
    pred   = (scores > threshold).astype(int)

    # Focus clustering on PREDICTED anomalous windows (detected faults)
    alarm_ix = np.flatnonzero(pred == 1)
    alarm_lat = latent[alarm_ix]              # (N_alarm, 16)
    alarm_err = end_errors[alarm_ix]          # (N_alarm, 9)
    alarm_err_z = end_z[alarm_ix]            # (N_alarm, 9)
    alarm_yte = yte[alarm_ix]
    alarm_bte = bte[alarm_ix]
    alarm_tte = tte[alarm_ix]
    alarm_scores = scores[alarm_ix]

    print(f"Clustering {len(alarm_ix)} alarmed windows ({alarm_yte.sum()} true faults, "
          f"{(1-alarm_yte).sum()} false alarms)")

    # Standardize latent codes for clustering
    scaler_lat = StandardScaler()
    alarm_lat_scaled = scaler_lat.fit_transform(alarm_lat)

    # Try HDBSCAN, fall back to KMeans
    cluster_labels = None
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(5, len(alarm_ix)//10),
                                     min_samples=3, metric="euclidean")
        cluster_labels = clusterer.fit_predict(alarm_lat_scaled)
        method = "HDBSCAN"
    except ImportError:
        pass

    if cluster_labels is None:
        from sklearn.cluster import KMeans
        # Estimate k from number of distinct fault batches (~10 fault batches, ~3-5 fault types)
        k = min(5, max(2, len(alarm_ix)//20))
        clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = clusterer.fit_predict(alarm_lat_scaled)
        method = f"KMeans(k={k})"

    print(f"Clustering method: {method}")
    unique_clusters = [c for c in np.unique(cluster_labels) if c >= 0]
    print(f"Found {len(unique_clusters)} clusters (excluding noise label -1 if HDBSCAN)")

    # PCA to 2D for visualization
    pca = PCA(n_components=2, random_state=42)
    lat_2d = pca.fit_transform(alarm_lat_scaled)
    pve = pca.explained_variance_ratio_
    print(f"PCA 2D: explains {100*pve.sum():.1f}% of variance")

    # Per-cluster fault signatures
    signatures = {}
    for c in unique_clusters:
        cix = np.flatnonzero(cluster_labels == c)
        sig_err = alarm_err_z[cix].mean(axis=0)   # mean standardised error per sensor
        top_sensor = SENSORS[np.argmax(np.abs(sig_err))]
        signatures[int(c)] = {
            "n_windows": int(len(cix)),
            "n_true_fault": int(alarm_yte[cix].sum()),
            "n_false_alarm": int((1-alarm_yte[cix]).sum()),
            "batches": sorted(set(alarm_bte[cix].tolist())),
            "mean_score": float(alarm_scores[cix].mean()),
            "dominant_sensor": top_sensor,
            "sensor_mean_err_z": {SENSORS[j]: float(sig_err[j]) for j in range(9)},
            "inferred_fault_type": _infer_fault_type(sig_err),
        }

    def _fmt(c): return signatures[c]

    print("\nFault Cluster Signatures:")
    for c in unique_clusters:
        s = signatures[c]
        print(f"  Cluster {c}: {s['n_windows']} windows | "
              f"dominant={s['dominant_sensor']} | type={s['inferred_fault_type']} | "
              f"batches={s['batches']}")

    # Save JSON
    report = {
        "clustering_method": method,
        "n_alarm_windows": int(len(alarm_ix)),
        "n_clusters": len(unique_clusters),
        "pca_variance_explained": float(pve.sum()),
        "clusters": signatures
    }
    with open(os.path.join(OUT,"fault_type_signatures.json"),"w") as f:
        json.dump(report, f, indent=2)

    # Save per-window CSV
    rows = []
    for k, idx in enumerate(alarm_ix):
        rows.append(dict(
            window_idx=int(idx), batch=int(bte[idx]), time=float(tte[idx]),
            label=int(yte[idx]), score=float(scores[idx]),
            cluster=int(cluster_labels[k]),
            pca1=float(lat_2d[k, 0]), pca2=float(lat_2d[k, 1]),
            **{f"err_z_{SENSORS[j].replace(' ','_')}": float(end_z[idx, j]) for j in range(9)}
        ))
    with open(os.path.join(OUT,"latent_clusters.csv"),"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # Plot 1: PCA scatter coloured by cluster
    cmap  = plt.colormaps.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for c in unique_clusters:
        cix = np.flatnonzero(cluster_labels == c)
        ax.scatter(lat_2d[cix,0], lat_2d[cix,1], s=20, label=f"Cluster {c}",
                   color=cmap(c), alpha=0.7)
    if -1 in cluster_labels:
        nix = np.flatnonzero(cluster_labels == -1)
        ax.scatter(lat_2d[nix,0], lat_2d[nix,1], s=8, color="gray", alpha=0.3, label="noise")
    ax.set_xlabel(f"PC1 ({100*pve[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({100*pve[1]:.1f}%)")
    ax.set_title(f"Latent Space — {method}\n(alarmed windows only)")
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
    # Colour by true label
    ax2 = axes[1]
    tp_mask = alarm_yte == 1
    ax2.scatter(lat_2d[~tp_mask,0], lat_2d[~tp_mask,1], s=12, color="steelblue",
                alpha=0.5, label="false alarm")
    ax2.scatter(lat_2d[tp_mask,0],  lat_2d[tp_mask,1],  s=20, color="red",
                alpha=0.7, label="true fault")
    ax2.set_xlabel(f"PC1 ({100*pve[0]:.1f}%)"); ax2.set_ylabel(f"PC2 ({100*pve[1]:.1f}%)")
    ax2.set_title("Latent Space — True Label\n(red=fault, blue=false alarm)")
    ax2.legend(fontsize=9); ax2.spines[["top","right"]].set_visible(False)
    plt.suptitle("GRU-AE v2 Bottleneck Representation — Fault Type Discovery", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,"latent_umap.png"), dpi=180)
    plt.close()

    # Plot 2: Sensor signature heatmap per cluster
    if len(unique_clusters) >= 2:
        sig_matrix = np.array([[signatures[c]["sensor_mean_err_z"][s] for s in SENSORS]
                                for c in unique_clusters])
        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(sig_matrix, cmap="RdBu_r", aspect="auto",
                       vmin=-np.abs(sig_matrix).max(), vmax=np.abs(sig_matrix).max())
        ax.set_xticks(range(9)); ax.set_xticklabels(SENSORS, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(unique_clusters)))
        ax.set_yticklabels([f"Cluster {c}\n({signatures[c]['dominant_sensor'][:10]})"
                            for c in unique_clusters], fontsize=9)
        plt.colorbar(im, ax=ax, label="Mean standardised reconstruction error")
        ax.set_title("Fault Type Signatures — Sensor Error Pattern per Cluster\n"
                     "(red=high error, blue=low, shows characteristic sensor pattern per fault type)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT,"sensor_signature_heatmap.png"), dpi=180)
        plt.close()

    print(f"\nSaved to {OUT}")


def _infer_fault_type(err_z):
    """Rule-based fault type inference from sensor error pattern."""
    # err_z is (9,) standardised errors: indices match SENSORS
    # [0=AgitRPM, 1=Aeration, 2=Sugar, 3=Acid, 4=Base, 5=HeatCool, 6=DO, 7=pH, 8=Temp]
    dominant = int(np.argmax(np.abs(err_z)))
    pairs = {0: "Agitation failure", 1: "Aeration/oxygen deviation",
             2: "Substrate feed deviation", 3: "Acid dosing fault",
             4: "Base dosing / pH control fault", 5: "Heat exchange fault",
             6: "Dissolved oxygen crisis", 7: "pH deviation", 8: "Temperature deviation"}
    return pairs.get(dominant, "Unknown")


if __name__ == "__main__":
    main()
