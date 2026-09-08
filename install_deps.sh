#!/bin/bash
set -x
VPY=".venv/Scripts/python.exe"
"$VPY" -m pip install --upgrade pip setuptools wheel
"$VPY" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.7.1+cpu"
"$VPY" -m pip install "scikit-learn==1.6.1" "tabpfn==2.2.1"
"$VPY" -m pip install lightgbm xgboost catboost openml joblib pyyaml pyarrow matplotlib pandas scipy psutil pytest
echo "INSTALL_DONE"
