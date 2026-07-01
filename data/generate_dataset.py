# -*- coding: utf-8 -*-
"""
data/generate_dataset.py
------------------------
Genere hospital_readmissions_30k.csv avec de vraies correlations medicales.

8 regles de risque :
  1. Age > 70            -> risque augmente
  2. Sejour > 7 jours    -> risque fortement augmente
  3. Diabete + HTA       -> risque tres eleve (synergique)
  4. BMI > 35            -> risque augmente
  5. Medicaments > 7     -> patient complexe, risque augmente
  6. Nursing_Facility    -> risque plus eleve que Home
  7. Cholesterol > 280   -> risque augmente
  8. bp_sys > 160        -> hypertension severe, risque augmente

Taux de readmission cible : 20-25 % (calibration automatique via brentq).

Usage :
    python data/generate_dataset.py
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

SEED        = 42
N           = 30_000
TARGET_RATE = 0.22          # taux de readmission vise (22 %)
OUTPUT_PATH = os.path.join("data", "hospital_readmissions_30k.csv")

np.random.seed(SEED)
rng = np.random.default_rng(SEED)

print("=" * 60)
print("GENERATION DU DATASET — CORRELATIONS MEDICALES REELLES")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Features de base
# ---------------------------------------------------------------------------
print("\n[1/6] Generation des features de base...")

patient_id = np.arange(1, N + 1)

# Distribution d'age realiste (adultes hospitalises)
age = np.clip(
    np.round(np.random.normal(57, 20, N)).astype(int), 18, 90
)

gender = np.random.choice(
    ["Male", "Female", "Other"], N, p=[1 / 3, 1 / 3, 1 / 3]
)

# Comorbidites independantes ~ 50/50
diabetes_bin     = rng.binomial(1, 0.50, N)
hypertension_bin = rng.binomial(1, 0.50, N)

# ---------------------------------------------------------------------------
# 2. Features correlees medicalement
# ---------------------------------------------------------------------------
print("[2/6] Generation des features correlees...")

# --- Pression arterielle : plus elevee si hypertension ---
bp_sys_base = np.where(hypertension_bin == 1, 152, 124)
bp_sys = np.clip(
    np.round(bp_sys_base + rng.normal(0, 14, N)).astype(int), 105, 185
)
# P(bp_sys > 160 | HTA)   ~  25 %
# P(bp_sys > 160 | non-HTA) ~  1 %

bp_dia_base = np.where(hypertension_bin == 1, 94, 79)
bp_dia = np.clip(
    np.round(bp_dia_base + rng.normal(0, 8, N)).astype(int), 60, 110
)

# --- Cholesterol : legerement plus eleve chez diabetiques ---
chol_base = np.where(diabetes_bin == 1, 238, 215)
cholesterol = np.clip(
    np.round(chol_base + rng.normal(0, 38, N)).astype(int), 150, 300
)

# --- BMI : plus eleve chez diabetiques ---
bmi_base = 29.0 + 2.0 * diabetes_bin
bmi = np.clip(
    np.round(bmi_base + rng.normal(0, 5.8, N), 1), 18.0, 40.0
)

# --- Nombre de medicaments : croissant avec l'age et les comorbidites ---
med_base = (
    2.0
    + np.clip((age - 40) * 0.05, 0.0, 3.5)
    + 1.8 * diabetes_bin
    + 1.2 * hypertension_bin
)
medication_count = np.clip(
    np.round(med_base + rng.normal(0, 1.8, N)).astype(int), 0, 10
)

# --- Duree de sejour : plus longue pour les personnes agees / polymedicamentees ---
stay_base = (
    4.0
    + 0.8  * (age > 70).astype(float)
    + 0.9  * (medication_count > 7).astype(float)
    + 0.6  * (diabetes_bin & hypertension_bin).astype(float)
)
length_of_stay = np.clip(
    np.round(stay_base + rng.normal(0, 2.4, N)).astype(int), 1, 10
)

# --- Destination de sortie : depend de l'age et des comorbidites ---
nursing_prob = np.clip(
    0.05
    + 0.005 * np.maximum(0, age - 60)
    + 0.07  * (diabetes_bin & hypertension_bin),
    0.02, 0.40,
)
rehab_prob = np.clip(0.14 + 0.002 * np.maximum(0, age - 50), 0.10, 0.30)
rehab_prob = np.minimum(rehab_prob, 1.0 - nursing_prob - 0.10)
home_prob  = np.maximum(1.0 - nursing_prob - rehab_prob, 0.10)

# Renormalisation pour garantir sum = 1
total = home_prob + rehab_prob + nursing_prob
home_prob    /= total
rehab_prob   /= total
nursing_prob /= total

discharge_destination = np.array([
    rng.choice(["Home", "Rehab", "Nursing_Facility"], p=[h, r, n])
    for h, r, n in zip(home_prob, rehab_prob, nursing_prob)
])

# ---------------------------------------------------------------------------
# 3. Probabilite de readmission — 8 regles medicales (modele logistique)
# ---------------------------------------------------------------------------
print("[3/6] Calcul des probabilites de readmission (8 regles)...")

# Logit sans intercept (regles + bruit individuel)
logit_rules = (
    # Regle 1 : Age eleve
    0.60 * (age > 70).astype(float)
    + 0.30 * (age > 80).astype(float)

    # Regle 2 : Sejour long (fort impact)
    + 0.95 * (length_of_stay > 7).astype(float)
    + 0.35 * (length_of_stay > 9).astype(float)

    # Regle 3 : Diabete + HTA ensemble (synergie)
    + 1.05 * (diabetes_bin & hypertension_bin).astype(float)
    + 0.18 * diabetes_bin.astype(float)
    + 0.15 * hypertension_bin.astype(float)

    # Regle 4 : Obesite severe BMI > 35
    + 0.55 * (bmi > 35).astype(float)

    # Regle 5 : Polymedicamentation (patient complexe)
    + 0.75 * (medication_count > 7).astype(float)

    # Regle 6 : Destination de sortie
    + 0.75 * (discharge_destination == "Nursing_Facility").astype(float)
    + 0.22 * (discharge_destination == "Rehab").astype(float)

    # Regle 7 : Cholesterol eleve > 280
    + 0.42 * (cholesterol > 280).astype(float)

    # Regle 8 : Hypertension severe bp_sys > 160
    + 0.65 * (bp_sys > 160).astype(float)
    + 0.30 * (bp_sys > 170).astype(float)

    # Bruit gaussien individuel
    + rng.normal(0, 0.55, N)
)

# --- Calibration automatique de l'intercept via brentq ---
def expected_rate(intercept: float) -> float:
    prob = 1.0 / (1.0 + np.exp(-(logit_rules + intercept)))
    return prob.mean() - TARGET_RATE

intercept_opt = brentq(expected_rate, -15.0, 10.0, xtol=1e-8)
print(f"   Intercept optimal : {intercept_opt:.4f}  (cible = {TARGET_RATE*100:.0f}%)")

prob = 1.0 / (1.0 + np.exp(-(logit_rules + intercept_opt)))

# Generation des outcomes binaires
readmitted_bin  = rng.binomial(1, prob)
readmitted      = np.where(readmitted_bin == 1, "Yes", "No")
actual_rate     = readmitted_bin.mean() * 100
print(f"   Taux effectif     : {actual_rate:.1f}%")

# ---------------------------------------------------------------------------
# 4. Construction du DataFrame final
# ---------------------------------------------------------------------------
print("[4/6] Construction du DataFrame...")

blood_pressure = np.array([f"{s}/{d}" for s, d in zip(bp_sys, bp_dia)])

df = pd.DataFrame({
    "patient_id"           : patient_id,
    "age"                  : age,
    "gender"               : gender,
    "blood_pressure"       : blood_pressure,
    "cholesterol"          : cholesterol,
    "bmi"                  : bmi,
    "diabetes"             : np.where(diabetes_bin == 1, "Yes", "No"),
    "hypertension"         : np.where(hypertension_bin == 1, "Yes", "No"),
    "medication_count"     : medication_count,
    "length_of_stay"       : length_of_stay,
    "discharge_destination": discharge_destination,
    "readmitted_30_days"   : readmitted,
})

# ---------------------------------------------------------------------------
# 5. Validation et statistiques
# ---------------------------------------------------------------------------
print("[5/6] Validation...")

assert df.shape == (N, 12), f"Shape inattendue : {df.shape}"
assert df.isnull().sum().sum() == 0, "Valeurs nulles detectees !"
assert df.columns.tolist() == [
    "patient_id", "age", "gender", "blood_pressure", "cholesterol",
    "bmi", "diabetes", "hypertension", "medication_count",
    "length_of_stay", "discharge_destination", "readmitted_30_days",
], "Colonnes incorrectes !"

print("\n" + "=" * 60)
print("STATISTIQUES DU DATASET GENERE")
print("=" * 60)

print(f"\n  Lignes x Colonnes  : {df.shape[0]:,} x {df.shape[1]}")
print(f"  Valeurs nulles     : {df.isnull().sum().sum()}")
print(f"  Taux readmission   : {actual_rate:.1f}%  "
      f"({readmitted_bin.sum():,} positifs / {N:,})")

print("\n  --- Variables numeriques ---")
bp_expanded = df["blood_pressure"].str.split("/", expand=True).astype(float)
df_stats    = df[["age", "cholesterol", "bmi",
                  "medication_count", "length_of_stay"]].copy()
df_stats["bp_sys"] = bp_expanded[0]
df_stats["bp_dia"] = bp_expanded[1]
print(df_stats.describe().round(2).to_string())

print("\n  --- Variables categorielles ---")
for col in ["gender", "diabetes", "hypertension",
            "discharge_destination", "readmitted_30_days"]:
    counts = df[col].value_counts()
    parts  = [f"{v}: {counts[v]:,} ({counts[v]/N*100:.1f}%)" for v in counts.index]
    print(f"  {col:<26} {' | '.join(parts)}")

print("\n  --- Verification des regles medicales ---")
readm = df["readmitted_30_days"] == "Yes"

rules = [
    ("Age > 70",              df["age"] > 70),
    ("Sejour > 7j",           df["length_of_stay"] > 7),
    ("Diabete + HTA",         (df["diabetes"] == "Yes") & (df["hypertension"] == "Yes")),
    ("BMI > 35",              df["bmi"] > 35),
    ("Medicaments > 7",       df["medication_count"] > 7),
    ("Nursing_Facility",      df["discharge_destination"] == "Nursing_Facility"),
    ("Cholesterol > 280",     df["cholesterol"] > 280),
    ("bp_sys > 160",          df_stats["bp_sys"] > 160),
]

print(f"  {'Regle':<25} {'Prevalence':>10}  {'Taux readm.':>12}  {'Ratio vs base':>13}")
print("  " + "-" * 65)
baseline = readm.mean() * 100
for label, mask in rules:
    prev  = mask.mean() * 100
    rate  = readm[mask].mean() * 100 if mask.sum() > 0 else 0.0
    ratio = rate / baseline if baseline > 0 else 0.0
    print(f"  {label:<25} {prev:>9.1f}%  {rate:>11.1f}%  {ratio:>12.1f}x")
print(f"  {'(baseline global)':<25} {'100.0':>9}%  {baseline:>11.1f}%")

# ---------------------------------------------------------------------------
# 6. Sauvegarde
# ---------------------------------------------------------------------------
print(f"\n[6/6] Sauvegarde -> {OUTPUT_PATH}")
os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
df.to_csv(OUTPUT_PATH, sep=";", index=False)
print(f"   {N:,} patients  x  {df.shape[1]} colonnes  |  sep=';'")

print("\n" + "=" * 60)
print("DATASET GENERE AVEC SUCCES")
print("=" * 60)

# ---------------------------------------------------------------------------
# Lancement automatique de src/train.py
# ---------------------------------------------------------------------------
train_script = os.path.join(PROJECT_ROOT, "src", "train.py")
print(f"\nLancement de {train_script}...\n")

result = subprocess.run(
    [sys.executable, train_script],
    cwd=PROJECT_ROOT,
)
sys.exit(result.returncode)
