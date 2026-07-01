# -*- coding: utf-8 -*-
"""
src/preprocess.py
-----------------
Pipeline de pretraitement pour la prediction de readmission hospitaliere.

Etapes :
  1. Chargement  - read_csv(sep=';')
  2. Nettoyage   - suppression de patient_id, decomposition blood_pressure
  3. Encodage    - Yes/No -> 0/1, one-hot sur gender et discharge_destination
  4. Split       - train/test 80/20 stratifie sur la cible
  5. Scaling     - StandardScaler fitte sur train uniquement
  6. SMOTE       - reequilibrage sur train uniquement
  7. Sauvegarde  - scaler.pkl + feature_names.pkl dans models/
"""

import os
import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Chemins par defaut
# ---------------------------------------------------------------------------
DATA_PATH  = os.path.join("data", "hospital_readmissions_30k.csv")
MODELS_DIR = "models"
TARGET     = "readmitted_30_days"

# Colonnes continues a scaler (apres decomposition blood_pressure)
NUMERICAL_COLS = [
    "age", "cholesterol", "bmi",
    "medication_count", "length_of_stay",
    "bp_sys", "bp_dia",
]

# ---------------------------------------------------------------------------
# 1. Chargement
# ---------------------------------------------------------------------------

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Charge le CSV avec separateur ';'."""
    df = pd.read_csv(path, sep=";")
    print(f"[load]    {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
    return df


# ---------------------------------------------------------------------------
# 2. Nettoyage
# ---------------------------------------------------------------------------

def drop_id(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime patient_id (identifiant non predictif)."""
    df = df.drop(columns=["patient_id"], errors="ignore")
    print(f"[clean]   patient_id supprime -> {df.shape[1]} colonnes restantes")
    return df


def split_blood_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose blood_pressure (format '130/72') en deux colonnes numeriques :
      - bp_sys : pression systolique
      - bp_dia : pression diastolique
    Supprime la colonne originale blood_pressure.
    """
    bp = df["blood_pressure"].str.split("/", expand=True).astype(float)
    insert_pos = df.columns.get_loc("blood_pressure")
    df.insert(insert_pos,     "bp_sys", bp[0])
    df.insert(insert_pos + 1, "bp_dia", bp[1])
    df = df.drop(columns=["blood_pressure"])
    print(
        f"[clean]   blood_pressure -> bp_sys (moy={df['bp_sys'].mean():.1f})"
        f" + bp_dia (moy={df['bp_dia'].mean():.1f})"
    )
    return df


# ---------------------------------------------------------------------------
# 3. Encodage
# ---------------------------------------------------------------------------

def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode Yes/No -> 1/0 sur les colonnes binaires :
      diabetes, hypertension, readmitted_30_days.
    """
    binary_cols = ["diabetes", "hypertension", TARGET]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})
    readmit_rate = df[TARGET].mean() * 100
    print(
        f"[encode]  binaire Yes/No -> 0/1 sur {binary_cols}"
        f"  |  taux readmission = {readmit_rate:.1f}%"
    )
    return df


def encode_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode gender et discharge_destination.
    drop_first=False pour conserver toutes les modalites (interpretabilite).
    Prefixes : gender_  et  dest_
    """
    df = pd.get_dummies(
        df, columns=["gender"],
        prefix="gender", drop_first=False, dtype=int
    )
    df = pd.get_dummies(
        df, columns=["discharge_destination"],
        prefix="dest", drop_first=False, dtype=int
    )
    gender_cols = [c for c in df.columns if c.startswith("gender_")]
    dest_cols   = [c for c in df.columns if c.startswith("dest_")]
    print(f"[encode]  one-hot gender    -> {gender_cols}")
    print(f"[encode]  one-hot discharge -> {dest_cols}")
    return df


# ---------------------------------------------------------------------------
# 4. Split train / test
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Separe features et cible, puis effectue un split 80/20 stratifie
    pour conserver le ratio de classes dans chaque ensemble.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(
        f"[split]   train={len(X_train):,}  test={len(X_test):,}"
        f"  (stratifie | positifs train: {y_train.mean()*100:.1f}%"
        f" | positifs test: {y_test.mean()*100:.1f}%)"
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 5. Scaling
# ---------------------------------------------------------------------------

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: list = None,
) -> tuple:
    """
    Applique StandardScaler uniquement sur les colonnes continues.
    Fitte sur X_train uniquement pour eviter le data leakage.
    Les colonnes binaires / one-hot ne sont pas scalees.
    Retourne les DataFrames avec les memes noms de colonnes.
    """
    if num_cols is None:
        num_cols = [c for c in NUMERICAL_COLS if c in X_train.columns]

    scaler  = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    print(f"[scale]   StandardScaler fitte sur train -> {num_cols}")
    return X_train, X_test, scaler


# ---------------------------------------------------------------------------
# 6. SMOTE
# ---------------------------------------------------------------------------

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple:
    """
    Applique SMOTE sur le train set uniquement pour reequilibrer les classes.
    Le test set reste intact (distribution reelle).
    """
    before = dict(y_train.value_counts().sort_index())
    smote  = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    after  = dict(pd.Series(y_res).value_counts().sort_index())

    print(f"[smote]   avant  : classe 0={before.get(0,0):,}  classe 1={before.get(1,0):,}")
    print(f"[smote]   apres  : classe 0={after.get(0,0):,}   classe 1={after.get(1,0):,}")
    print(f"[smote]   train reequilibre -> {len(X_res):,} lignes")
    return X_res, y_res


# ---------------------------------------------------------------------------
# 7. Sauvegarde des artefacts
# ---------------------------------------------------------------------------

def save_artifacts(
    scaler: StandardScaler,
    feature_names: list,
    models_dir: str = MODELS_DIR,
) -> None:
    """Sauvegarde scaler.pkl et feature_names.pkl dans models/."""
    os.makedirs(models_dir, exist_ok=True)

    scaler_path   = os.path.join(models_dir, "scaler.pkl")
    features_path = os.path.join(models_dir, "feature_names.pkl")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(features_path, "wb") as f:
        pickle.dump(feature_names, f)

    print(f"[save]    scaler        -> {scaler_path}")
    print(f"[save]    feature_names -> {features_path}  ({len(feature_names)} features)")


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def preprocess_pipeline(
    path: str         = DATA_PATH,
    test_size: float  = 0.2,
    random_state: int = 42,
    use_smote: bool   = True,
    save: bool        = True,
    models_dir: str   = MODELS_DIR,
) -> tuple:
    """
    Execute le pipeline complet de bout en bout.

    Retourne
    --------
    X_train      : np.ndarray  (reequilibre si use_smote=True)
    X_test       : np.ndarray
    y_train      : np.ndarray
    y_test       : np.ndarray
    scaler       : StandardScaler fitte
    feature_names: list[str]  (ordre des colonnes apres encodage)
    """
    print("=" * 55)
    print("PIPELINE DE PRETRAITEMENT")
    print("=" * 55)

    df = load_data(path)
    df = drop_id(df)
    df = split_blood_pressure(df)
    df = encode_binary(df)
    df = encode_onehot(df)

    all_cols = df.drop(columns=[TARGET]).columns.tolist()
    print(f"\n[info]    {len(all_cols)} features apres encodage:")
    print(f"          {all_cols}\n")

    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)

    feature_names = X_train.columns.tolist()

    X_train, X_test, scaler = scale_features(X_train, X_test)

    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state)

    if save:
        save_artifacts(scaler, feature_names, models_dir)

    # Conversion finale en numpy
    X_train = X_train if isinstance(X_train, np.ndarray) else np.array(X_train)
    X_test  = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_train = y_train if isinstance(y_train, np.ndarray) else np.array(y_train)
    y_test  = y_test.values if isinstance(y_test, pd.Series) else y_test

    print("\n" + "=" * 55)
    print("RESUME FINAL")
    print("=" * 55)
    print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}    y_test  : {y_test.shape}")
    print(f"  Features ({len(feature_names)}) : {feature_names}")
    print("=" * 55)

    return X_train, X_test, y_train, y_test, scaler, feature_names


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_pipeline()
