"""
data/add_names.py
Ajoute les colonnes 'nom' et 'prenom' au dataset avec des noms marocains.
Assignation aléatoire fixe (random_state=42) selon le genre du patient.
"""

import numpy as np
import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).parent / "hospital_readmissions_30k.csv"

PRENOMS_M = [
    "Mohamed", "Ahmed", "Youssef", "Karim", "Omar", "Hassan", "Amine",
    "Mehdi", "Zakaria", "Hamza", "Bilal", "Soufiane", "Rachid", "Khalid",
    "Nabil", "Tarik", "Saad", "Reda", "Hicham", "Othmane",
    "Ayoub", "Adil", "Iliass", "Badr", "Mouad",
]

PRENOMS_F = [
    "Fatima", "Salma", "Zineb", "Nadia", "Sara", "Imane", "Khadija",
    "Meriem", "Houda", "Laila", "Samira", "Widad", "Hafsa", "Ghita",
    "Loubna", "Chaimae", "Hajar", "Malak", "Rim", "Ikram",
    "Soukaina", "Yasmine", "Sanaa", "Btissam", "Rajae",
]

PRENOMS_OTHER = [
    "Rayan", "Nour", "Sami", "Hana", "Wael",
    "Dina", "Fares", "Lina", "Anis", "Jana",
]

NOMS_FAMILLE = [
    "Alami", "Benali", "Moussaoui", "El Idrissi", "Laik", "Lamris",
    "Bennani", "Tazi", "Fassi", "Chraibi", "Berrada", "Chah",
    "Kettani", "Filali", "Sebti", "Benhaddou", "El Ouazzani",
    "Tahiri", "Skali", "Bensouda", "Cherkaoui", "El Mansouri",
    "Regragui", "Bouhout", "Hajji",
]


def assign_names(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n   = len(df)

    prenoms = []
    for genre in df["gender"]:
        if genre == "Male":
            pool = PRENOMS_M
        elif genre == "Female":
            pool = PRENOMS_F
        else:
            pool = PRENOMS_OTHER
        prenoms.append(pool[rng.integers(len(pool))])

    noms = [NOMS_FAMILLE[i] for i in rng.integers(len(NOMS_FAMILLE), size=n)]

    df = df.copy()
    # Supprimer les colonnes existantes si présentes
    for col in ["nom", "prenom"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df.insert(1, "nom",    noms)
    df.insert(2, "prenom", prenoms)
    return df


def main():
    print(f"Chargement : {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, sep=";")
    print(f"  {len(df):,} patients, {df.shape[1]} colonnes")

    df = assign_names(df)

    print(f"  Colonnes après ajout : {list(df.columns)}")
    df.to_csv(CSV_PATH, sep=";", index=False, encoding="utf-8")
    print(f"  Sauvegardé : {CSV_PATH}\n")
    print("=== 5 premières lignes ===")
    print(df[["patient_id", "nom", "prenom", "gender"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
