# -*- coding: utf-8 -*-
"""
data/prepare_theme_dataset.py
------------------------------
Convertit le nouveau dataset "DONNEES_NETTOYEES_THEME (1).xlsx" (fourni par
l'utilisateur) vers le format attendu par le pipeline du projet
(data/hospital_readmissions_30k.csv, separateur ';').

Transformations :
  - Ajout de patient_id (1..N)
  - Fusion bp_systolic / bp_diastolic -> blood_pressure ("sys/dia")
  - Encodage 0/1 -> No/Yes pour diabetes, hypertension, readmitted_30_days
  - Reordonnancement des colonnes pour matcher le schema historique

Usage :
    python data/prepare_theme_dataset.py
"""

import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_XLSX  = os.path.join(PROJECT_ROOT, "DONNEES_NETTOYEES_THEME (1).xlsx")
OUTPUT_CSV   = os.path.join(PROJECT_ROOT, "data", "hospital_readmissions_30k.csv")

COLUMN_ORDER = [
    "patient_id", "nom", "prenom", "age", "gender", "blood_pressure",
    "cholesterol", "bmi", "diabetes", "hypertension", "medication_count",
    "length_of_stay", "discharge_destination", "readmitted_30_days",
]


def main():
    df = pd.read_excel(SOURCE_XLSX)
    print(f"[load]    {df.shape[0]:,} lignes x {df.shape[1]} colonnes depuis {SOURCE_XLSX}")

    df.insert(0, "patient_id", range(1, len(df) + 1))

    df["blood_pressure"] = (
        df["bp_systolic"].astype(int).astype(str)
        + "/"
        + df["bp_diastolic"].astype(int).astype(str)
    )
    df = df.drop(columns=["bp_systolic", "bp_diastolic"])

    for col in ["diabetes", "hypertension", "readmitted_30_days"]:
        df[col] = df[col].map({1: "Yes", 0: "No"})

    df = df[COLUMN_ORDER]

    df.to_csv(OUTPUT_CSV, sep=";", index=False, encoding="utf-8")
    print(f"[save]    {df.shape[0]:,} lignes x {df.shape[1]} colonnes -> {OUTPUT_CSV}")
    print(f"          colonnes : {df.columns.tolist()}")
    print(f"          taux readmission : {(df['readmitted_30_days']=='Yes').mean()*100:.2f}%")


if __name__ == "__main__":
    main()
