# for data manipulation
import pandas as pd
# preprocessing / pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# model, tuning, metrics
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
# serialization & os
import joblib, os
# HF Hub
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
# experiment tracking
import mlflow

# ---------------------------
# MLflow setup (use env if set, else fallback)
# ---------------------------
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "tourism-wellness-package"))

HF_DATASET_REPO = "Ansh91/tourism"                       # dataset repo with Xtrain/Xtest/ytrain/ytest
HF_MODEL_REPO   = os.getenv("HF_MODEL_REPO", "Ansh91/tourism")  # model repo to upload artifact
HF_TOKEN        = os.getenv("HF_TOKEN", None)

# ✅ workspace-relative artifacts dir (works in Actions & Colab)
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", os.path.join(os.getcwd(), "artifacts"))
os.makedirs(ARTIFACT_DIR, exist_ok=True)
MODEL_BASENAME = "best_tourism_model_v1.joblib"
MODEL_PATH = os.path.join(ARTIFACT_DIR, MODEL_BASENAME)

def read_csv_from_hf(repo_id: str, filename: str, repo_type: str = "dataset") -> pd.DataFrame:
    """Try hf:// first; if it fails (private/no auth), fall back to hf_hub_download."""
    path = f"hf://{'datasets' if repo_type=='dataset' else repo_type}/{repo_id}/{filename}"
    try:
        return pd.read_csv(path)
    except Exception:
        local = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, token=HF_TOKEN)
        return pd.read_csv(local)

# ---------------------------
# Load prepared splits from HF dataset repo
# ---------------------------
Xtrain = read_csv_from_hf(HF_DATASET_REPO, "Xtrain.csv")
Xtest  = read_csv_from_hf(HF_DATASET_REPO, "Xtest.csv")
ytrain = read_csv_from_hf(HF_DATASET_REPO, "ytrain.csv").squeeze()
ytest  = read_csv_from_hf(HF_DATASET_REPO, "ytest.csv").squeeze()

print("Prepared splits loaded successfully.")
print("Xtrain:", Xtrain.shape, "Xtest:", Xtest.shape, "ytrain:", ytrain.shape, "ytest:", ytest.shape)

# ---------------------------
# Features per tourism schema (keep only those present)
# ---------------------------
numeric_features = [
    "Age","CityTier","NumberOfPersonVisiting","PreferredPropertyStar","NumberOfTrips",
    "Passport","OwnCar","NumberOfChildrenVisiting","MonthlyIncome",
    "PitchSatisfactionScore","NumberOfFollowups","DurationOfPitch"
]
categorical_features = [
    "TypeofContact","Occupation","Gender","MaritalStatus","Designation","ProductPitched"
]
numeric_features     = [c for c in numeric_features if c in Xtrain.columns]
categorical_features = [c for c in categorical_features if c in Xtrain.columns]

# ---------------------------
# Class imbalance (scale_pos_weight = neg/pos)
# ---------------------------
neg = int((ytrain == 0).sum())
pos = int((ytrain == 1).sum())
scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0

# ---------------------------
# Preprocessor & model
# ---------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
    remainder="drop"
)
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    eval_metric="logloss",
)
param_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__colsample_bytree": [0.4, 0.5, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.5, 0.6],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__reg_lambda": [0.4, 0.5, 0.6],
}
model_pipeline = make_pipeline(preprocessor, xgb_model)

# ---------------------------
# Train + track
# ---------------------------
with mlflow.start_run():
    search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_grid,
        n_iter=12, cv=2, n_jobs=-1, random_state=42
    )
    search.fit(Xtrain, ytrain)

    # nested runs logging (optional; keep like your sample)
    results = search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", float(results["mean_test_score"][i]))
            mlflow.log_metric("std_test_score",  float(results["std_test_score"][i]))

    mlflow.log_params(search.best_params_)

    # evaluate at fixed threshold
    best_model = search.best_estimator_
    thr = 0.45
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= thr).astype(int)
    y_pred_test  = (best_model.predict_proba(Xtest)[:, 1]  >= thr).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True, zero_division=0)
    test_report  = classification_report(ytest,  y_pred_test,  output_dict=True, zero_division=0)

    mlflow.log_metrics({
        "train_accuracy": float(train_report['accuracy']),
        "train_precision": float(train_report['1']['precision']),
        "train_recall":    float(train_report['1']['recall']),
        "train_f1-score":  float(train_report['1']['f1-score']),
        "test_accuracy":   float(test_report['accuracy']),
        "test_precision":  float(test_report['1']['precision']),
        "test_recall":     float(test_report['1']['recall']),
        "test_f1-score":   float(test_report['1']['f1-score'])
    })

    # ---------------------------
    # Save & upload model  (NO /content paths)
    # ---------------------------
    joblib.dump(best_model, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH, artifact_path="model")
    print(f"Model saved locally at: {MODEL_PATH}")

    api = HfApi(token=HF_TOKEN)
    repo_id = HF_MODEL_REPO
    repo_type = "model"
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repo '{repo_id}' exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{repo_id}' not found. Creating...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=HF_TOKEN)
        print(f"Model repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=MODEL_PATH,
        path_in_repo=os.path.basename(MODEL_PATH),
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Uploaded model to HF: {repo_id}/{os.path.basename(MODEL_PATH)}")
