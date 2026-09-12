# Running the TabPFN-3 arm yourself

Everything is built and tested. You do two things I can't: accept the licence,
and run the model. Then I analyse the output.

## Why this split

TabPFN-3's weights are publicly hosted (the HF API reports `gated=False`), but the
TABPFN-3 Non-Commercial License v1.0 says agreement is formed **by the act of
downloading**, and `tabpfn/model_loading.py` calls `ensure_license_accepted()`
before any download. Your academic use qualifies as non-commercial. Accepting is a
legal act, so it is yours to perform, not mine.

## Step 1 — build the venv (~2 GB, one command)

```bash
cd C:/Users/UseR/Downloads/ICLR-Nafis
C:/Users/UseR/AppData/Local/Programs/Python/Python312/python.exe -m venv .venv-tabpfn3
.venv-tabpfn3/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.7.1+cpu"
.venv-tabpfn3/Scripts/python.exe -m pip install "tabpfn==8.5.0" scikit-learn pandas pyarrow openml joblib pyyaml psutil
```

## Step 2 — accept the licence

Either visit <https://huggingface.co/Prior-Labs/tabpfn_3>, read the LICENSE, accept,
then `.venv-tabpfn3/Scripts/python.exe -m pip install huggingface_hub` and
`huggingface-cli login` — or just run step 3 and complete the interactive prompt
the package shows the first time.

Sanity check (downloads the checkpoint, ~1 min):

```bash
.venv-tabpfn3/Scripts/python.exe -c "import os; os.environ['CUDA_VISIBLE_DEVICES']=''
from tabpfn import TabPFNClassifier
from sklearn.datasets import make_classification
X,y=make_classification(n_samples=300,n_features=8,n_informative=5,n_classes=3,n_clusters_per_class=1,random_state=0)
c=TabPFNClassifier(device='cpu',random_state=0); c.fit(X[:200],y[:200])
print('TabPFN-3 OK', c.predict_proba(X[200:]).shape)"
```

## Step 3 — run the probe (~1–2 h on CPU, resumable)

```bash
.venv/Scripts/python.exe src/experiments/m15_v3_cross_generation.py
```

It is the **same paired rotation probe** as the v2 and v1 arms — same datasets,
same split seed, same rotation matrices, limits matched to the M9 v2 run. Results
are cached by config hash, so you can stop and restart it freely.

## Step 4 — tell me it's done

I run `src/experiments/m14_analyze.py` (extended to pick up `m15v3`) and report
v1 vs v2 vs v3 together.

## What we already know without it

| generation | relative degradation under rotation | datasets worse | p |
|---|---|---|---|
| v1 (ICLR 2023) | +4.9% | 21/30 | 0.027 |
| v2 (Nature 2025) | +11.3% | 29/30 | 6.5e-09 |

v2 is significantly more rotation-sensitive than v1 (paired, relative, p=1.0e-05).
The v3 arm tests whether that trend continued, reversed, or plateaued — any of
which is publishable. The cross-generation objection is **already answered** by
v1-vs-v2; v3 would strengthen it against the current production model.
