import torch, pandas as pd, numpy as np, matplotlib.pyplot as plt, joblib
from src.models import Generator
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ───────── CONFIG ────────────────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 10_000
RAW         = Path("Dataset.csv")
PREPROC     = Path("preprocess.joblib")
SYNTH       = Path("synthetic_traffic.csv")    # root-level output
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Helpers ─────────────────────────────────────────────────────────────────
def add_octets(df: pd.DataFrame) -> pd.DataFrame:
    """Split id.orig_h / id.resp_h into four octet columns each (as ints)."""
    for side in ["orig", "resp"]:
        parts = df[f"id.{side}_h"].str.split('.', expand=True).astype(int)
        parts.columns = [f"{side}_ip_{i+1}" for i in range(4)]
        df = pd.concat([df, parts], axis=1)
    return df.drop(columns=["id.orig_h", "id.resp_h"])

def invert_transform(X_scaled: np.ndarray, pipe):
    """Reverse StandardScaler + OneHotEncoder → human-readable DataFrame."""
    ct = pipe.named_steps["ct"]                         # ColumnTransformer
    ohe: OneHotEncoder = ct.named_transformers_["cat"]
    num_scaler        = ct.named_transformers_["num"]

    cat_len = len(ohe.get_feature_names_out())
    X_cat, X_num = X_scaled[:, :cat_len], X_scaled[:, cat_len:]

    df_cat = pd.DataFrame(ohe.inverse_transform(X_cat),
                          columns=ohe.feature_names_in_)
    df_num = pd.DataFrame(num_scaler.inverse_transform(X_num),
                          columns=num_scaler.feature_names_in_)
    return pd.concat([df_cat, df_num], axis=1)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ╭──────────────────────── READ + PREPROCESS ONE ROW (for dim) ──────────╮
    df_sample = add_octets(pd.read_csv(RAW).iloc[:1].copy())
    pipe = joblib.load(PREPROC)
    feature_dim = pipe.named_steps["ct"].transform(df_sample).shape[1]
    # ╰────────────────────────────────────────────────────────────────────────╯

    # ─── Load Generator weights safely (weights_only=True) ───────────────────
    G = Generator(128, feature_dim)
    G.load_state_dict(torch.load("G.pt", map_location="cpu", weights_only=True))
    G.eval()

    # ─── Generate synthetic samples ──────────────────────────────────────────
    with torch.no_grad():
        noise = torch.randn(N_SAMPLES, 128)
        fake  = G(noise).numpy()

    df_fake = invert_transform(fake, pipe)
    df_fake.to_csv(SYNTH, index=False)
    print(f"✅ Synthetic samples saved → {SYNTH}")

    # ─── Prepare *real* data in the exact octet format ───────────────────────
    df_real = add_octets(pd.read_csv(RAW).copy())

    # ─── Quick distribution visual (first 4 IP-octet cols) ───────────────────
    numeric_cols = ["orig_ip_1", "orig_ip_2", "resp_ip_1", "resp_ip_2"]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, col in enumerate(numeric_cols):
        ax = axs[i // 2, i % 2]
        ax.hist(df_real[col], bins=30, alpha=0.6, label="Real", color="blue")
        ax.hist(df_fake[col], bins=30, alpha=0.6, label="Fake", color="orange")
        ax.set_title(col); ax.legend()
    plt.tight_layout(); plt.suptitle("IP-Octet Distributions", y=1.02)
    plt.show()

    # ─── GAN-detection sanity test (logistic reg) ────────────────────────────
    X_real = pipe.transform(df_real)
    X_fake = pipe.transform(df_fake)

    # Convert sparse → dense once for stacking
    X      = np.vstack([X_real.toarray(), X_fake.toarray()])
    y      = np.concatenate([np.ones(X_real.shape[0]),
                             np.zeros(X_fake.shape[0])])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=SEED)
    clf = LogisticRegression(max_iter=1000); clf.fit(X_tr, y_tr)
    y_hat = clf.predict(X_te)

    print("\n── GAN Detection Report ──")
    print(classification_report(y_te, y_hat, target_names=["Fake", "Real"]))
    print(f"ROC-AUC: {roc_auc_score(y_te, clf.predict_proba(X_te)[:,1]):.4f}")

if __name__ == "__main__":
    main()
