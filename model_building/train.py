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
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
# experiment tracking
import mlflow

# ---------------------------
# MLflow setup (use env if set, else fallback)
# ---------------------------
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "tourism-wellness-package"))

# ---------------------------
# Load prepared splits from HF dataset repo
# ---------------------------
Xtrain_path = "hf://datasets/Ansh91/tourism/Xtrain.csv"
Xtest_path  = "hf://datasets/Ansh91/tourism/Xtest.csv"
ytrain_path = "hf://datasets/Ansh91/tourism/ytrain.csv"
ytest_path  = "hf://datasets/Ansh91/tourism/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()  # to 1D
ytest  = pd.read_csv(ytest_path).squeeze()

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
numeric_features      = [c for c in numeric_features if c in Xtrain.columns]
categorical_features  = [c for c in categorical_features if c in Xtrain.columns]

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

# Hyperparameter grid (sample-style)
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

    # log each tried set as nested run (like sample)
    results = search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", float(results["mean_test_score"][i]))
            mlflow.log_metric("std_test_score",  float(results["std_test_score"][i]))

    # log best params on main run
    mlflow.log_params(search.best_params_)

    # evaluate with sample’s fixed threshold = 0.45
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
    # Save & upload model
    # ---------------------------
    os.makedirs("/content/tourism/artifacts", exist_ok=True)
    model_path = "/content/tourism/artifacts/best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved locally at: {model_path}")

    # Upload to Hugging Face (model repo)
    hf_token = os.getenv("HF_TOKEN", None)
    api = HfApi(token=hf_token)
    repo_id = os.getenv("HF_MODEL_REPO", "Ansh91/tourism")  # override via env if you prefer
    repo_type = "model"

    # create model repo if missing
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repo '{repo_id}' exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{repo_id}' not found. Creating...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=hf_token)
        print(f"Model repo '{repo_id}' created.")

    # upload artifact
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=os.path.basename(model_path),
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Uploaded model to HF: {repo_id}/{os.path.basename(model_path)}")
