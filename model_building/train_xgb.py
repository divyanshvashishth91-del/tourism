import os
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import mlflow

# -------------------------
# Config / paths
# -------------------------
CSV_PATH = "/content/tourism/tourism.csv"
TARGET_COL = "ProdTaken"
ARTIFACT_DIR = "/content/tourism/artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Respect env if notebook set them already
if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
if os.getenv("MLFLOW_EXPERIMENT_NAME"):
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(CSV_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier if present
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not in CSV.")

# X / y and split
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature lists per spec (only keep ones actually present)
numeric_features = [
    "Age","CityTier","NumberOfPersonVisiting","PreferredPropertyStar","NumberOfTrips",
    "Passport","OwnCar","NumberOfChildrenVisiting","MonthlyIncome",
    "PitchSatisfactionScore","NumberOfFollowups","DurationOfPitch"
]
categorical_features = [
    "TypeofContact","Occupation","Gender","MaritalStatus","Designation","ProductPitched"
]
numeric_features = [c for c in numeric_features if c in Xtrain.columns]
categorical_features = [c for c in categorical_features if c in Xtrain.columns]

# Class imbalance for XGBoost
neg, pos = (ytrain == 0).sum(), (ytrain == 1).sum()
scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0

# Preprocessor + model (sample style)
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
    remainder="drop",
)
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1,
    tree_method="hist", eval_metric="logloss"
)
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Hyperparameter grid (sample style)
param_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__colsample_bytree": [0.4, 0.5, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.5, 0.6],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__reg_lambda": [0.4, 0.5, 0.6],
}

# -------------------------
# Train + log (prints MLflow links like your sample)
# -------------------------
with mlflow.start_run():
    parent_run = mlflow.active_run()
    exp_id = parent_run.info.experiment_id
    base_url = os.getenv("MLFLOW_TRACKING_URI", "")
    print()  # spacing

    grid_search = RandomizedSearchCV(
        model_pipeline, param_distributions=param_grid,
        n_iter=12, cv=2, n_jobs=-1, random_state=42
    )
    grid_search.fit(Xtrain, ytrain)

    # Log each tried set as nested run and PRINT THE LINK
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            child_run = mlflow.active_run()
            run_id = child_run.info.run_id
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", float(results["mean_test_score"][i]))
            mlflow.log_metric("std_test_score", float(results["std_test_score"][i]))
            # Print run link (same format as your screenshot)
            if base_url.startswith("http"):
                print(f"🔗 View run at: {base_url}/#/experiments/{exp_id}/runs/{run_id}")
    # Also print the experiment page link once
    if base_url.startswith("http"):
        print(f"🔗 View experiment at: {base_url}/#/experiments/{exp_id}")

    # Log best params in the parent run
    mlflow.log_params(grid_search.best_params_)

    # Evaluate with sample’s threshold
    best_model = grid_search.best_estimator_
    thr = 0.45

    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= thr).astype(int)
    y_pred_test  = (best_model.predict_proba(Xtest)[:, 1]  >= thr).astype(int)

    train_rep = classification_report(ytrain, y_pred_train, output_dict=True, zero_division=0)
    test_rep  = classification_report(ytest,  y_pred_test,  output_dict=True, zero_division=0)

    mlflow.log_metrics({
        "train_accuracy": float(train_rep["accuracy"]),
        "train_precision": float(train_rep["1"]["precision"]),
        "train_recall": float(train_rep["1"]["recall"]),
        "train_f1-score": float(train_rep["1"]["f1-score"]),
        "test_accuracy": float(test_rep["accuracy"]),
        "test_precision": float(test_rep["1"]["precision"]),
        "test_recall": float(test_rep["1"]["recall"]),
        "test_f1-score": float(test_rep["1"]["f1-score"]),
    })

    # Save model artifact
    model_path = os.path.join(ARTIFACT_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)

print("✅ Training complete. Best model saved at /content/tourism/artifacts/best_model.pkl")
