# ----------------------------
# Data Preparation for Tourism
# ----------------------------
# - Loads dataset from Hugging Face dataset repo
# - Splits into X/y and train/test (20% test, stratified)
# - Saves Xtrain/Xtest/ytrain/ytest locally as CSV
# - Uploads these CSVs back to the HF dataset repo
#
# Requirements:
#   pip install huggingface_hub pandas scikit-learn fsspec
#
# Notes:
#   - Expects HF_TOKEN in env if your repo is private (public works without).
#   - Dataset path: hf://datasets/Ansh91/tourism/tourism.csv
# ----------------------------

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# ----------------------------
# Config
# ----------------------------
HF_REPO_ID = "Ansh91/tourism"      # dataset repository (you already created it)
DATASET_PATH = f"hf://datasets/{HF_REPO_ID}/tourism.csv"
TARGET_COL = "ProdTaken"
ID_COLS = ["CustomerID"]           # dropped if present
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ----------------------------
# Load dataset from HF
# ----------------------------
print(f"Loading dataset from: {DATASET_PATH}")
try:
    df = pd.read_csv(DATASET_PATH)
except Exception as e:
    # Fallback: download file first if hf:// path is not available in this environment
    print(f"Direct hf:// read failed: {e}\nUsing hf_hub_download fallback...")
    from huggingface_hub import hf_hub_download
    local_csv = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename="tourism.csv",
        token=os.getenv("HF_TOKEN", None)
    )
    df = pd.read_csv(local_csv)

print("Dataset loaded successfully.")
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# ----------------------------
# Basic checks & light cleanup
# ----------------------------
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found. Found: {list(df.columns)}")

# Drop ID columns if present
for col in ID_COLS:
    if col in df.columns:
        df = df.drop(columns=[col])

# Remove rows with missing target
df = df[~df[TARGET_COL].isna()].copy()

# Ensure binary int target
df[TARGET_COL] = df[TARGET_COL].astype(int)

# ----------------------------
# Build X, y
# ----------------------------
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print("X shape:", X.shape, " y shape:", y.shape)
print("Target positive rate:", float(y.mean()))

# ----------------------------
# Train / Test split (20%, stratified)
# ----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
print("Xtrain:", Xtrain.shape, "Xtest:", Xtest.shape)
print("ytrain:", ytrain.shape, "ytest:", ytest.shape)

# ----------------------------
# Save locally
# ----------------------------
Xtrain_path = "Xtrain.csv"
Xtest_path  = "Xtest.csv"
ytrain_path = "ytrain.csv"
ytest_path  = "ytest.csv"

Xtrain.to_csv(Xtrain_path, index=False)
Xtest.to_csv(Xtest_path, index=False)
ytrain.to_csv(ytrain_path, index=False)
ytest.to_csv(ytest_path, index=False)

print(f"Saved: {Xtrain_path}, {Xtest_path}, {ytrain_path}, {ytest_path}")

# ----------------------------
# Upload to HF dataset repo
# ----------------------------
api = HfApi(token=os.getenv("HF_TOKEN", None))

files_to_upload = [Xtrain_path, Xtest_path, ytrain_path, ytest_path]
for fp in files_to_upload:
    print(f"Uploading {fp} to {HF_REPO_ID} (dataset repo)...")
    api.upload_file(
        path_or_fileobj=fp,
        path_in_repo=os.path.basename(fp),  # store at repo root
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )
print("All prepared files uploaded successfully.")
