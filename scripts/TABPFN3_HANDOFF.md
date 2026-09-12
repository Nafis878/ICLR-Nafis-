# Running the TabPFN-3 arm

Everything is built and tested. One step is yours and cannot be delegated:
accepting the licence. Then the run is one command.

## The gate, as actually verified

`tabpfn==8.5.0` calls `ensure_license_accepted()` in `model_loading._download_model`
before any weights are loaded. We ran it and captured the exact failure:

```
tabpfn.errors.TabPFNLicenseError: TabPFN requires a one-time license acceptance to
download model weights for local inference, but no interactive terminal is available.
```

Reading `tabpfn/browser_auth.py` confirms what the gate checks. It is **not**
HuggingFace authentication, and it is not satisfied by having the checkpoint on
disk (we do: the 203 MB `tabpfn-v3-classifier-v3_default.ckpt` is in the HF cache,
and `Prior-Labs/tabpfn_3` reports `gated=False`). The gate requires:

1. a **Prior Labs account** at <https://ux.priorlabs.ai>,
2. the **TABPFN-3 Non-Commercial License** accepted on that account's Licenses tab,
3. a **vendor-issued API key** presented as `TABPFN_TOKEN`, which the package
   verifies against a Prior Labs **server** on every cold start
   (`verify_token` -> `check_license_accepted`).

So auditing post-v2 TabPFN requires a credential the vendor issues and can revoke,
and a live callback to vendor infrastructure at model-load time. That is a
stronger barrier than a click-through licence, and it is reported as a finding in
its own right -- see the "Honest limits" section of the README.

Academic use qualifies as non-commercial. Accepting is a legal act by an account
holder, so it is the researcher's to perform.

## Step 1 -- build the venv (~2 GB, one command) [DONE]

```bash
C:/Users/UseR/AppData/Local/Programs/Python/Python312/python.exe -m venv .venv-tabpfn3
.venv-tabpfn3/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.7.1+cpu"
.venv-tabpfn3/Scripts/python.exe -m pip install "tabpfn==8.5.0" scikit-learn pandas pyarrow openml joblib pyyaml psutil
```

## Step 2 -- accept the licence and get a token  <-- THE ONLY OUTSTANDING STEP

1. Open <https://ux.priorlabs.ai>, log in or register.
2. Accept the TABPFN-3 Non-Commercial License on the **Licenses** tab.
3. Copy the API key from <https://ux.priorlabs.ai/account>.
4. Export it, then run step 3 in the same shell:

```bash
export TABPFN_TOKEN="<your-api-key>"        # PowerShell: $env:TABPFN_TOKEN="<key>"
```

Verify in ~30 s (this is what the run does on its first cell):

```bash
.venv-tabpfn3/Scripts/python.exe -c "import os; os.environ['CUDA_VISIBLE_DEVICES']=''
from tabpfn import TabPFNClassifier
from sklearn.datasets import make_classification
X,y=make_classification(n_samples=300,n_features=8,n_informative=5,n_classes=3,n_clusters_per_class=1,random_state=0)
c=TabPFNClassifier(device='cpu',random_state=0); c.fit(X[:200],y[:200])
print('TabPFN-3 OK', c.predict_proba(X[200:]).shape)"
```

Do **not** paste the key into any file in this repo; it is a secret and the repo
is public. The environment variable is read directly by the worker subprocess.

## Step 3 -- run the probe (~1-2 h on CPU, resumable)

```bash
.venv/Scripts/python.exe src/experiments/m15_v3_cross_generation.py
```

`m15_v3_cross_generation.py` preflights the token and fails in seconds with these
instructions if it is missing, rather than after loading datasets.

It is the **same paired rotation probe** as the v2 and v1 arms -- same datasets,
same split seed, same rotation matrices, limits matched to the M9 v2 run. Results
are cached by config hash, so it can be stopped and restarted freely.

## Step 4 -- analyse

```bash
.venv/Scripts/python.exe src/experiments/m14_analyze.py     # picks up m15v3 automatically
```

## What is already established without it

| generation | relative degradation under rotation | datasets worse | p |
|---|---|---|---|
| v1 (ICLR 2023) | +4.9% | 21/30 | 0.040 |
| v2 (Nature 2025) | +11.3% | 86/95 | 7.7e-16 |

On the 30 datasets measured in **both** generations the paired degradation is
+0.0037 (v1) vs +0.0199 (v2), Wilcoxon p = 1.0e-05: v2 is significantly more
rotation-sensitive than v1.

The cross-generation objection is therefore **already answered** by v1-vs-v2. The
v3 arm would turn a two-point trend into three and test it against the current
production model -- strengthening the result, not carrying it.
