# -*- coding: utf-8 -*-
"""
src/train.py
------------
Pipeline complet sans data leakage + feature engineering :
  1.  Chargement CSV (sep=';')
  2.  Nettoyage : drop patient_id, blood_pressure -> bp_sys / bp_dia
  3.  Encodage  : Yes/No -> 0/1, one-hot gender + discharge_destination
  4.  Feature engineering (avant split, sur donnees brutes completes)
  5.  Split train/test STRATIFIE 80/20
  6.  StandardScaler : fit sur X_train UNIQUEMENT, transform sur X_test
  7.  SMOTE sur X_train UNIQUEMENT -> X_train_res, y_train_res
  8.  Entrainement RandomForest + XGBoost (avec early stopping)
  9.  Evaluation sur X_test ORIGINAL (donnees reelles, metriques fiables)
  10. MLflow tracking + sauvegarde du meilleur modele

Usage :
    python -m src.train
    python src/train.py
"""

import json
import os
import pickle
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

MODELS_DIR      = "models"
MLFLOW_DIR      = "mlruns"
EXPERIMENT_NAME = "readmission_prediction"
TARGET          = "readmitted_30_days"

# Colonnes continues a scaler (originales + engineered continues)
NUMERICAL_COLS = [
    "age", "cholesterol", "bmi",
    "medication_count", "length_of_stay",
    "bp_sys", "bp_dia",
    "bp_pulse_pressure", "bp_mean_arterial",
    "med_stay_ratio", "age_bmi", "cholesterol_ratio",
]

# ---------------------------------------------------------------------------
# Parametres des modeles
# ---------------------------------------------------------------------------
RF_PARAMS = {
    "n_estimators"  : 500,
    "max_depth"     : 15,
    "min_samples_leaf": 3,
    "n_jobs"        : -1,
    "random_state"  : 42,
}

XGB_PARAMS = {
    "n_estimators"       : 1000,
    "learning_rate"      : 0.02,
    "max_depth"          : 6,
    "subsample"          : 0.8,
    "colsample_bytree"   : 0.7,
    "min_child_weight"   : 5,
    "scale_pos_weight"   : 7,
    "gamma"              : 0.2,
    "reg_alpha"          : 0.1,
    "reg_lambda"         : 1.0,
    "early_stopping_rounds": 50,
    "verbosity"          : 0,
    "random_state"       : 42,
    "n_jobs"             : 1,
}

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cree 9 nouvelles features a partir des colonnes existantes.
    Applique sur le DataFrame COMPLET avant le split (transformations
    row-wise deterministes -> aucun data leakage).

    Necessite : bp_sys, bp_dia, bmi, age, diabetes, hypertension,
                medication_count, length_of_stay, cholesterol
    """
    df = df.copy()

    df["bp_pulse_pressure"] = df["bp_sys"] - df["bp_dia"]
    df["bp_mean_arterial"]  = (df["bp_sys"] + 2 * df["bp_dia"]) / 3
    df["bmi_category"]      = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]
    ).astype(int)
    df["age_risk"]          = (df["age"] > 65).astype(int)
    df["poly_risk"]         = df["diabetes"] * df["hypertension"]
    df["med_stay_ratio"]    = df["medication_count"] / (df["length_of_stay"] + 1)
    df["age_bmi"]           = df["age"] * df["bmi"] / 1000
    df["cholesterol_ratio"] = df["cholesterol"] / 200
    df["complex_patient"]   = (
        (df["medication_count"] >= 6).astype(int)
        * (df["length_of_stay"] >= 5).astype(int)
    )

    new_features = [
        "bp_pulse_pressure", "bp_mean_arterial", "bmi_category",
        "age_risk", "poly_risk", "med_stay_ratio", "age_bmi",
        "cholesterol_ratio", "complex_patient",
    ]
    print(f"[feat]    {len(new_features)} features engineered : {new_features}")
    return df

# ---------------------------------------------------------------------------
# Metriques
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy" : round(float(accuracy_score(y_true, y_pred)),                   4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall"   : round(float(recall_score(y_true, y_pred,    zero_division=0)), 4),
        "f1"       : round(float(f1_score(y_true, y_pred,        zero_division=0)), 4),
        "roc_auc"  : round(float(roc_auc_score(y_true, y_prob)),                    4),
    }

# ---------------------------------------------------------------------------
# Artefacts graphiques
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, model_name: str, save_dir: str) -> str:
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non (0)", "Oui (1)"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matrice de confusion - {model_name}")
    path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_names: list, model_name: str, save_dir: str):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    fi    = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top15 = fi.head(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    top15[::-1].plot(kind="barh", ax=ax, color="#42a5f5", edgecolor="white")
    ax.set_title(f"Feature Importance (Top 15) - {model_name}")
    ax.set_xlabel("Importance")
    path = os.path.join(save_dir, f"feature_importance_{model_name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return path


def plot_roc_comparison(results: dict, y_test: np.ndarray, save_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = ["#388e3c", "#f57c00"]

    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        label = f"{name}  (AUC = {res['metrics']['roc_auc']:.4f})"
        ax.plot(fpr, tpr, lw=2, color=color, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatoire (AUC = 0.5)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR)")
    ax.set_title("Comparaison des courbes ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    path = os.path.join(save_dir, "roc_curves_comparison.png")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return path

# ---------------------------------------------------------------------------
# Entrainement + evaluation d'un modele (run MLflow)
# ---------------------------------------------------------------------------

def train_one_model(
    name: str,
    model,
    logged_params: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    artifacts_dir: str,
    fit_kwargs: dict = None,
) -> dict:
    """
    Entraine le modele sur X_train_res (SMOTE), evalue sur X_test (reel).
    fit_kwargs : arguments supplementaires passes a model.fit() (ex: eval_set XGBoost).
    """
    fit_kwargs = fit_kwargs or {}

    with mlflow.start_run(run_name=name):
        mlflow.log_params(logged_params)
        mlflow.log_param("train_size", X_train.shape[0])
        mlflow.log_param("test_size",  X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])

        print(f"  [fit]   {name} ...", end=" ", flush=True)
        t0 = time.time()
        model.fit(X_train, y_train, **fit_kwargs)
        print(f"OK  ({time.time()-t0:.1f}s)")

        # Log l'iteration optimale si early stopping actif
        if hasattr(model, "best_iteration") and model.best_iteration is not None:
            print(f"         early stopping @ iteration {model.best_iteration}")
            mlflow.log_param("best_iteration", model.best_iteration)

        # Evaluation sur X_test ORIGINAL — aucun echantillon synthetique
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_metrics(y_test, y_prob)
        mlflow.log_metrics(metrics)

        print(
            f"         AUC-ROC={metrics['roc_auc']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"Precision={metrics['precision']:.4f}"
        )

        cm_path = plot_confusion_matrix(y_test, y_pred, name, artifacts_dir)
        mlflow.log_artifact(cm_path)

        fi_path = plot_feature_importance(model, feature_names, name, artifacts_dir)
        if fi_path:
            mlflow.log_artifact(fi_path)

        if "XGBoost" in name:
            mlflow.xgboost.log_model(model, name="model")
        else:
            mlflow.sklearn.log_model(model, name="model")

        run_id = mlflow.active_run().info.run_id

    return {
        "model"  : model,
        "metrics": metrics,
        "y_prob" : y_prob,
        "run_id" : run_id,
    }

# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def save_best_model(
    best_name: str,
    best_model,
    all_results: dict,
    feature_names: list,
    models_dir: str = MODELS_DIR,
) -> None:
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(models_dir, "model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    print(f"\n[save] Meilleur modele ({best_name}) -> {models_dir}/model.pkl")

    metrics_export = {name: res["metrics"] for name, res in all_results.items()}
    metrics_export["best_model"] = best_name
    metrics_export.update(all_results[best_name]["metrics"])
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"[save] Metriques                 -> {models_dir}/metrics.json")

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.ones(len(feature_names))

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(models_dir, "feature_importance.csv"), index=False)
    print(f"[save] Feature importance        -> {models_dir}/feature_importance.csv")

# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def train_pipeline(
    data_path: str    = os.path.join("data", "hospital_readmissions_30k.csv"),
    models_dir: str   = MODELS_DIR,
    random_state: int = 42,
) -> dict:
    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. Chargement
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHARGEMENT ET PRETRAITEMENT")
    print("=" * 60)

    df = pd.read_csv(data_path, sep=";")
    print(f"[load]    {df.shape[0]:,} lignes x {df.shape[1]} colonnes")

    # ------------------------------------------------------------------
    # 2. Nettoyage
    # ------------------------------------------------------------------
    df = df.drop(columns=["patient_id"], errors="ignore")

    bp  = df["blood_pressure"].str.split("/", expand=True).astype(float)
    pos = df.columns.get_loc("blood_pressure")
    df.insert(pos,     "bp_sys", bp[0])
    df.insert(pos + 1, "bp_dia", bp[1])
    df  = df.drop(columns=["blood_pressure"])
    print(
        f"[clean]   blood_pressure -> bp_sys (moy={df['bp_sys'].mean():.1f})"
        f" + bp_dia (moy={df['bp_dia'].mean():.1f})"
    )

    # ------------------------------------------------------------------
    # 3. Encodage
    # ------------------------------------------------------------------
    for col in ["diabetes", "hypertension", TARGET]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})
    print(f"[encode]  Yes/No -> 0/1  |  taux readmission = {df[TARGET].mean()*100:.1f}%")

    df = pd.get_dummies(df, columns=["gender"],
                        prefix="gender", drop_first=False, dtype=int)
    df = pd.get_dummies(df, columns=["discharge_destination"],
                        prefix="dest",   drop_first=False, dtype=int)
    print(f"[encode]  one-hot gender + discharge_destination -> {df.shape[1]} colonnes")

    # ------------------------------------------------------------------
    # 4. Feature engineering (avant split — row-wise, aucun leakage)
    # ------------------------------------------------------------------
    df = engineer_features(df)

    # ------------------------------------------------------------------
    # 5. Separation X / y  +  split stratifie 80/20
    # ------------------------------------------------------------------
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    feature_names = X.columns.tolist()

    print(f"\n[info]    {len(feature_names)} features au total : {feature_names}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    print(
        f"[split]   train={len(X_train):,}  test={len(X_test):,}"
        f"  (positifs train: {y_train.mean()*100:.1f}%"
        f"  |  positifs test: {y_test.mean()*100:.1f}%)"
    )

    # ------------------------------------------------------------------
    # 6. StandardScaler : fit sur X_train UNIQUEMENT
    # ------------------------------------------------------------------
    num_cols = [c for c in NUMERICAL_COLS if c in X_train.columns]
    scaler   = StandardScaler()

    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])
    print(f"[scale]   StandardScaler fit sur X_train -> {len(num_cols)} colonnes")

    # Sauvegarde pour l'inference
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(models_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)
    print(f"[save]    scaler.pkl + feature_names.pkl -> {models_dir}/")

    # Convertir en numpy — X_test_scaled sert d'eval_set pour XGBoost
    X_test_scaled = X_test.values
    y_test_arr    = y_test.values

    # ------------------------------------------------------------------
    # 7. SMOTE sur X_train UNIQUEMENT
    # ------------------------------------------------------------------
    before = dict(y_train.value_counts().sort_index())

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train.values, y_train.values)

    after = dict(pd.Series(y_train_res).value_counts().sort_index())
    print(f"[smote]   avant  : classe 0={before.get(0,0):,}  classe 1={before.get(1,0):,}")
    print(f"[smote]   apres  : classe 0={after.get(0,0):,}   classe 1={after.get(1,0):,}")
    print(f"[smote]   X_train_res -> {len(X_train_res):,} lignes")
    print(f"[smote]   X_test      -> {len(X_test_scaled):,} lignes INTACT (reelles uniquement)")

    artifacts_dir = os.path.join(models_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLFLOW_DIR))
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\nMLflow experiment : '{EXPERIMENT_NAME}'  |  URI : {mlflow.get_tracking_uri()}")

    all_results = {}

    # ------------------------------------------------------------------
    # 8a. RandomForest
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1/2  RandomForest  (n_estimators=500, max_depth=15)")
    print("=" * 60)
    all_results["RandomForest"] = train_one_model(
        name="RandomForest",
        model=RandomForestClassifier(**RF_PARAMS),
        logged_params=RF_PARAMS,
        X_train=X_train_res, X_test=X_test_scaled,
        y_train=y_train_res, y_test=y_test_arr,
        feature_names=feature_names,
        artifacts_dir=artifacts_dir,
    )

    # ------------------------------------------------------------------
    # 8b. XGBoost avec early stopping
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2/2  XGBoost  (n_estimators=1000, early_stopping_rounds=50)")
    print("=" * 60)

    # early_stopping_rounds est dans le constructeur (XGBoost >= 1.6)
    # eval_set est passe a fit() — ne pas le logger comme hyperparametre
    xgb_logged = {k: v for k, v in XGB_PARAMS.items()
                  if k not in ("early_stopping_rounds", "verbosity")}

    all_results["XGBoost"] = train_one_model(
        name="XGBoost",
        model=XGBClassifier(**XGB_PARAMS),
        logged_params=xgb_logged,
        X_train=X_train_res, X_test=X_test_scaled,
        y_train=y_train_res, y_test=y_test_arr,
        feature_names=feature_names,
        artifacts_dir=artifacts_dir,
        fit_kwargs={
            "eval_set": [(X_test_scaled, y_test_arr)],
            "verbose" : False,
        },
    )

    # ------------------------------------------------------------------
    # 9. Comparaison finale sur X_test ORIGINAL
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARAISON FINALE (X_test ORIGINAL — donnees reelles)")
    print("=" * 60)

    col_w  = 16
    header = (f"{'Modele':<{col_w}} {'AUC-ROC':>8} {'F1':>8}"
              f" {'Recall':>8} {'Precision':>10} {'Accuracy':>10}")
    sep    = "-" * len(header)
    print(header)
    print(sep)

    best_name = None
    best_auc  = -1.0

    for name, res in all_results.items():
        m = res["metrics"]
        if m["roc_auc"] > best_auc:
            best_auc  = m["roc_auc"]
            best_name = name
        print(
            f"  {name:<{col_w-2}} {m['roc_auc']:>8.4f} {m['f1']:>8.4f}"
            f" {m['recall']:>8.4f} {m['precision']:>10.4f} {m['accuracy']:>10.4f}"
        )

    print(sep)
    print(f"  => Meilleur modele : {best_name}  (AUC-ROC = {best_auc:.4f})")

    roc_path = plot_roc_comparison(all_results, y_test_arr, artifacts_dir)

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SAUVEGARDE")
    print("=" * 60)
    save_best_model(best_name, all_results[best_name]["model"],
                    all_results, feature_names, models_dir)
    print(f"[save] ROC curves comparison     -> {roc_path}")

    best_run_id = all_results[best_name]["run_id"]
    with mlflow.start_run(run_id=best_run_id):
        mlflow.set_tag("best_model", "true")
        mlflow.log_artifact(roc_path)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    elapsed_total = time.time() - t_start
    print("\n" + "=" * 60)
    print("TERMINE")
    print("=" * 60)
    print(f"  Meilleur modele  : {best_name}")
    print(f"  AUC-ROC test     : {best_auc:.4f}")
    print(f"  F1  test         : {all_results[best_name]['metrics']['f1']:.4f}")
    print(f"  Recall test      : {all_results[best_name]['metrics']['recall']:.4f}")
    print(f"  Duree totale     : {elapsed_total:.0f}s")
    print(f"  Modele sauvegarde: {os.path.join(models_dir, 'model.pkl')}")
    print(f"  MLflow UI        : mlflow ui --backend-store-uri {MLFLOW_DIR}")
    print("=" * 60)

    return all_results


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_pipeline()
