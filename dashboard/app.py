# dashboard/app.py
import json
import os
import pickle
import warnings
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Système de Prédiction de Réadmission",
    page_icon="🏥",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
API_URL       = os.getenv("API_URL", "http://localhost:8000")
DATA_PATH     = "data/hospital_readmissions_30k.csv"
PATIENTS_PATH = "data/patients.json"
MODEL_PATH    = "models/model.pkl"
SCALER_PATH   = "models/scaler.pkl"
FEATURES_PATH = "models/feature_names.pkl"
METRICS_PATH  = "models/metrics.json"
FI_PATH       = "models/feature_importance.csv"

NUMERICAL_COLS = [
    "age", "cholesterol", "bmi", "medication_count", "length_of_stay",
    "bp_sys", "bp_dia", "bp_pulse_pressure", "bp_mean_arterial",
    "med_stay_ratio", "age_bmi", "cholesterol_ratio",
]

RISK_COLORS = {"Low": "#28a745", "Medium": "#fd7e14", "High": "#dc3545"}
RISK_LABELS = {"Low": "Faible",  "Medium": "Moyen",   "High": "Élevé"}

USERS = {
    "medecin": {
        "password": "medecin123",
        "role":     "medecin",
        "nom":      "Dr. Emily Richardson",
        "specialite": "Cardiologue — Médecine interne",
        "email":    "dr.richardson@chu-hospital.com",
    },
    "admin": {
        "password": "admin123",
        "role":     "admin",
        "nom":      "Administrateur",
        "specialite": "Gestion du système",
        "email":    "admin@hopital.fr",
    },
}

DEST_LABELS = {
    "Home":             "Domicile",
    "Nursing_Facility": "EHPAD / Soins infirmiers",
    "Rehab":            "Rééducation",
}

# ── Listes de noms pour génération automatique ───────────────────────────────
_PRENOMS_H = [
    "Mohamed", "Ahmed", "Youssef", "Hassan", "Ibrahim", "Omar", "Ali",
    "Rachid", "Karim", "Mehdi", "Amine", "Soufiane", "Hamza", "Othmane",
    "Saad", "Khalid", "Nabil", "Tarik", "Badr", "Hicham",
    "Pierre", "Jean", "Marc", "Luc", "Thomas", "Nicolas", "Antoine",
    "François", "Julien", "Maxime",
]
_PRENOMS_F = [
    "Fatima", "Salma", "Nadia", "Laila", "Zineb", "Khadija", "Amina",
    "Samira", "Houda", "Meryem", "Hafsa", "Sara", "Imane", "Yasmine",
    "Nour", "Siham", "Oumaima", "Rim", "Hajar", "Widad",
    "Marie", "Sophie", "Isabelle", "Camille", "Julie", "Lucie",
    "Emma", "Céline", "Claire", "Léa",
]
_NOMS_FAMILLE = [
    "Alami", "Benali", "Chah", "El Idrissi", "Fassi", "Ghali", "Hassani",
    "Idrissi", "Jebari", "Karimi", "Lahlou", "Mansouri", "Naji", "Ouali",
    "Rahimi", "Saidi", "Tazi", "Wahbi", "Ziani", "Berrada",
    "Belhaj", "Benkirane", "Chraibi", "Tahiri", "Filali",
    "Kabbaj", "Lyoussi", "Moussaoui", "Raissouni", "Sqalli",
    "Dupont", "Martin", "Bernard", "Lefebvre", "Moreau",
    "Simon", "Laurent", "Leroy", "Roux", "Girard",
]

# ─────────────────────────────────────────────────────────────────────────────
# CSS global
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── layout ── */
[data-testid="stSidebar"] section { padding-top: 10px; }
.block-container { padding-top: 1.5rem; }

/* ── login (styles post-connexion uniquement) ── */

/* ── cards ── */
.stat-card {
    background: white; border-radius: 14px;
    padding: 22px 24px; box-shadow: 0 2px 12px rgba(0,0,0,.06);
    border-left: 5px solid #1976d2; margin-bottom: 4px;
}
.profile-card {
    background: white; border-radius: 16px;
    padding: 36px 40px; max-width: 700px;
    box-shadow: 0 2px 18px rgba(0,0,0,.07);
    border-top: 5px solid #1976d2;
}
.risk-card {
    text-align: center; padding: 28px 20px;
    border-radius: 16px; margin-bottom: 12px;
}

/* ── table header ── */
.table-header {
    background: #f5f7fa; padding: 10px 6px;
    border-radius: 8px; margin-bottom: 6px;
    font-weight: 700; color: #555; font-size: 13px;
}

/* ── sidebar nav ── */
.nav-label {
    font-size: 11px; color: #9e9e9e; letter-spacing: 1.2px;
    text-transform: uppercase; margin-bottom: 4px; padding-left: 4px;
}

/* ── metrics ── */
div[data-testid="stMetricValue"] { font-size: 1.65rem !important; }
div[data-testid="stMetricLabel"] { font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_names():
    with open(FEATURES_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_PATH, sep=";")

@st.cache_data
def load_metrics():
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)

def load_patients() -> list:
    if not os.path.exists(PATIENTS_PATH):
        return []
    with open(PATIENTS_PATH, encoding="utf-8") as f:
        return json.load(f)

def save_patients(patients: list) -> None:
    os.makedirs(os.path.dirname(PATIENTS_PATH), exist_ok=True)
    with open(PATIENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(patients, f, ensure_ascii=False, indent=2)


@st.cache_data
def load_patients_csv() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, sep=";")


def _show_patient_card_csv(p, key_suffix: str = ""):
    """Affiche la fiche détaillée d'un patient CSV + bouton Prédire."""
    genre_fr = {"Male": "Homme", "Female": "Femme", "Other": "Autre"}
    pid      = int(p["patient_id"])
    nom      = str(p.get("nom", "—"))
    prenom   = str(p.get("prenom", "—"))
    age      = int(p.get("age", 0))
    genre    = genre_fr.get(str(p.get("gender", "")), "—")
    bp_sys   = p.get("bp_systolic", "—")
    bp_dia   = p.get("bp_diastolic", "—")
    bp       = f"{int(bp_sys)}/{int(bp_dia)}" if bp_sys != "—" else "—"
    chol     = str(p.get("cholesterol", "—"))
    bmi_val  = str(p.get("bmi", "—"))
    diab_fr  = "Oui" if int(p.get("diabetes", 0)) == 1 else "Non"
    hype_fr  = "Oui" if int(p.get("hypertension", 0)) == 1 else "Non"
    nmed     = int(p.get("medication_count", 0))
    j        = int(p.get("length_of_stay", 0))
    dest     = DEST_LABELS.get(str(p.get("discharge_destination", "")), "—")

    # ── En-tête ──────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:#f8faff; border:1.5px solid #c5d8f8; border-radius:14px;"
        f" padding:26px 32px; margin-top:14px;'>"
        f"<div style='font-weight:800; color:#1565c0; font-size:18px; margin-bottom:16px;'>"
        f"📋 Fiche patient — {prenom} {nom}"
        f"<span style='font-size:12px; color:#9e9e9e; font-weight:400; margin-left:14px;'>"
        f"ID #{pid}</span></div>",
        unsafe_allow_html=True,
    )

    # ── Bannière risque (si prédiction déjà disponible) ──────────────────────
    if "probabilite" in p and "risque" in p:
        prob  = float(p["probabilite"])
        risk  = str(p["risque"])
        color = RISK_COLORS.get(risk, "#888")
        label = RISK_LABELS.get(risk, risk)
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:14px; margin-bottom:20px;"
            f" background:{color}18; border:1.5px solid {color}55;"
            f" border-radius:10px; padding:12px 18px;'>"
            f"<span style='font-size:22px; font-weight:900; color:{color};'>{prob*100:.1f}%</span>"
            f"<span style='background:{color}33; color:{color}; font-weight:800;"
            f" padding:4px 14px; border-radius:20px; font-size:13px;'>{label}</span>"
            f"<span style='color:#9e9e9e; font-size:12px;'>probabilité de réadmission à 30 jours</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Grille 3 colonnes ────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            "<div style='font-size:11px; color:#9e9e9e; text-transform:uppercase;"
            " letter-spacing:.8px; margin-bottom:8px;"
            " border-bottom:1px solid #dde8f8; padding-bottom:4px;'>Identité</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Nom :** {nom}")
        st.markdown(f"**Prénom :** {prenom}")
        st.markdown(f"**Âge :** {age} ans")
        st.markdown(f"**Genre :** {genre}")

    with c2:
        st.markdown(
            "<div style='font-size:11px; color:#9e9e9e; text-transform:uppercase;"
            " letter-spacing:.8px; margin-bottom:8px;"
            " border-bottom:1px solid #dde8f8; padding-bottom:4px;'>Paramètres vitaux</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Pression artérielle :** {bp} mmHg")
        st.markdown(f"**Cholestérol :** {chol} mg/dL")
        st.markdown(f"**IMC (BMI) :** {bmi_val} kg/m²")

    with c3:
        st.markdown(
            "<div style='font-size:11px; color:#9e9e9e; text-transform:uppercase;"
            " letter-spacing:.8px; margin-bottom:8px;"
            " border-bottom:1px solid #dde8f8; padding-bottom:4px;'>Informations cliniques</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Diabète :** {diab_fr}")
        st.markdown(f"**Hypertension :** {hype_fr}")
        st.markdown(f"**Médicaments :** {nmed}")
        st.markdown(f"**Durée de séjour :** {j} jour{'s' if j > 1 else ''}")
        st.markdown(f"**Destination sortie :** {dest}")

    # ── Fermeture div principale + bouton ────────────────────────────────────
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button(
        "🔬 Prédire le risque de réadmission",
        type="primary",
        use_container_width=True,
        key=f"pred_card_{pid}_{key_suffix}",
    ):
        st.session_state.pred_patient_data = {k: (v.item() if hasattr(v, "item") else v)
                                               for k, v in p.items()}
        st.session_state.pred_prob         = None
        st.session_state.pred_risk         = None
        st.session_state.medecin_page      = "Prédiction"
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — identique à src/train.py
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_input(
    age, bmi, cholesterol, medication_count, length_of_stay,
    bp_sys, bp_dia, diabetes_bin, hypertension_bin,
    gender, discharge_destination,
    scaler, feature_names,
) -> pd.DataFrame:
    age_f  = float(age)
    bmi_f  = float(bmi)
    chol_f = float(cholesterol)
    meds_i = int(medication_count)
    stay_i = int(length_of_stay)
    sys_f  = float(bp_sys)
    dia_f  = float(bp_dia)
    db     = int(diabetes_bin)
    ht     = int(hypertension_bin)

    bmi_cat = pd.cut([bmi_f], bins=[0, 18.5, 25, 30, 100],
                     labels=[0, 1, 2, 3], include_lowest=True)
    bmi_category = int(bmi_cat[0]) if pd.notna(bmi_cat[0]) else 1

    row_d = {
        "age":               age_f,
        "bp_sys":            sys_f,
        "bp_dia":            dia_f,
        "cholesterol":       chol_f,
        "bmi":               bmi_f,
        "diabetes":          db,
        "hypertension":      ht,
        "medication_count":  float(meds_i),
        "length_of_stay":    float(stay_i),
        "gender_Female":     int(gender == "Female"),
        "gender_Male":       int(gender == "Male"),
        "gender_Other":      int(gender == "Other"),
        "dest_Home":         int(discharge_destination == "Home"),
        "dest_Nursing_Facility": int(discharge_destination == "Nursing_Facility"),
        "dest_Rehab":        int(discharge_destination == "Rehab"),
        "bp_pulse_pressure": sys_f - dia_f,
        "bp_mean_arterial":  (sys_f + 2 * dia_f) / 3,
        "bmi_category":      bmi_category,
        "age_risk":          int(age_f > 65),
        "poly_risk":         db * ht,
        "med_stay_ratio":    meds_i / (stay_i + 1),
        "age_bmi":           age_f * bmi_f / 1000,
        "cholesterol_ratio": chol_f / 200,
        "complex_patient":   int(meds_i >= 6 and stay_i >= 5),
    }
    df = pd.DataFrame([row_d]).reindex(columns=feature_names, fill_value=0)
    df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])
    return df


def preprocess_batch(df_raw: pd.DataFrame, scaler, feature_names) -> pd.DataFrame:
    """Prétraitement vectorisé sur l'ensemble du dataset (identique à src/train.py)."""
    bp_sys = df_raw["bp_systolic"].astype(float).fillna(120.0)
    bp_dia = df_raw["bp_diastolic"].astype(float).fillna(80.0)

    age   = df_raw["age"].astype(float)
    bmi   = df_raw["bmi"].astype(float)
    chol  = df_raw["cholesterol"].astype(float)
    meds  = df_raw["medication_count"].astype(float)
    stay  = df_raw["length_of_stay"].astype(float)
    db    = df_raw["diabetes"].astype(int)
    ht    = df_raw["hypertension"].astype(int)

    out = pd.DataFrame(index=df_raw.index)
    out["age"]             = age
    out["bp_sys"]          = bp_sys
    out["bp_dia"]          = bp_dia
    out["cholesterol"]     = chol
    out["bmi"]             = bmi
    out["diabetes"]        = db
    out["hypertension"]    = ht
    out["medication_count"] = meds
    out["length_of_stay"]  = stay

    out["gender_Female"] = (df_raw["gender"] == "Female").astype(int)
    out["gender_Male"]   = (df_raw["gender"] == "Male").astype(int)
    out["gender_Other"]  = (df_raw["gender"] == "Other").astype(int)

    out["dest_Home"]             = (df_raw["discharge_destination"] == "Home").astype(int)
    out["dest_Nursing_Facility"] = (df_raw["discharge_destination"] == "Nursing_Facility").astype(int)
    out["dest_Rehab"]            = (df_raw["discharge_destination"] == "Rehab").astype(int)

    out["bp_pulse_pressure"] = bp_sys - bp_dia
    out["bp_mean_arterial"]  = (bp_sys + 2 * bp_dia) / 3
    out["bmi_category"]      = pd.cut(
        bmi, bins=[0, 18.5, 25, 30, 100],
        labels=[0, 1, 2, 3], include_lowest=True,
    ).astype(float).fillna(1).astype(int)
    out["age_risk"]          = (age > 65).astype(int)
    out["poly_risk"]         = db * ht
    out["med_stay_ratio"]    = meds / (stay + 1)
    out["age_bmi"]           = age * bmi / 1000
    out["cholesterol_ratio"] = chol / 200
    out["complex_patient"]   = ((meds >= 6) & (stay >= 5)).astype(int)

    out = out.reindex(columns=feature_names, fill_value=0)
    out[NUMERICAL_COLS] = scaler.transform(out[NUMERICAL_COLS])
    return out


@st.cache_data(show_spinner=False)
def predict_all_patients() -> pd.DataFrame:
    """Prédiction XGBoost en batch sur les 30 000 patients (mise en cache)."""
    df_raw     = load_patients_csv()
    model_obj  = load_model()
    scaler_obj = load_scaler()
    feat_names = load_feature_names()
    X          = preprocess_batch(df_raw, scaler_obj, feat_names)
    probs      = model_obj.predict_proba(X.values)[:, 1]
    result     = df_raw.copy()
    result["probabilite"] = probs
    result["risque"]      = np.where(probs >= 0.60, "High",
                            np.where(probs >= 0.30, "Medium", "Low"))
    return result


def get_risk_level(prob: float) -> str:
    if prob < 0.30: return "Low"
    if prob < 0.60: return "Medium"
    return "High"


def run_prediction(patient: dict):
    bp_sys = float(patient.get("bp_systolic", 120))
    bp_dia = float(patient.get("bp_diastolic", 80))
    db_bin = int(patient.get("diabetes", 0))
    ht_bin = int(patient.get("hypertension", 0))

    prob = None
    payload = {
        "age": int(patient["age"]), "bmi": float(patient["bmi"]),
        "cholesterol": float(patient["cholesterol"]),
        "medication_count": int(patient["medication_count"]),
        "length_of_stay": int(patient["length_of_stay"]),
        "bp_sys": bp_sys, "bp_dia": bp_dia,
        "diabetes": db_bin, "hypertension": ht_bin,
        "gender": patient.get("gender", "Other"),
        "discharge_destination": patient.get("discharge_destination", "Home"),
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=3)
        if resp.status_code == 200:
            d    = resp.json()
            prob = d.get("probability", d.get("readmitted_probability"))
    except Exception:
        pass

    if prob is None:
        model_obj  = load_model()
        scaler_obj = load_scaler()
        feat_names = load_feature_names()
        X_df = preprocess_input(
            int(patient["age"]), float(patient["bmi"]), float(patient["cholesterol"]),
            int(patient["medication_count"]), int(patient["length_of_stay"]),
            bp_sys, bp_dia, db_bin, ht_bin,
            patient.get("gender", "Other"),
            patient.get("discharge_destination", "Home"),
            scaler_obj, feat_names,
        )
        prob = float(model_obj.predict_proba(X_df.values)[0, 1])

    return prob, get_risk_level(prob)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "logged_in":         False,
    "user_role":         None,
    "username":          None,
    "pred_prob":         None,
    "pred_risk":         None,
    "pred_pid":          None,
    "pred_patient_data": None,
    "selected_csv_pid":  None,
    "medecin_page":      "Gestion des patients",
    "admin_page":        "Tableau de bord",
}
for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# PAGE : LOGIN
# ─────────────────────────────────────────────────────────────────────────────
def show_login():
    st.markdown("""
    <style>
    /* ── Masquer sidebar ── */
    [data-testid="stSidebar"]        { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }

    /* ── Fond dégradé bleu médical ── */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #0f2340 50%, #1a3a5c 100%) !important;
        min-height: 100vh;
    }
    section[data-testid="stAppViewContainer"] { background: transparent !important; }
    [data-testid="stHeader"] {
        background: transparent !important;
        box-shadow: none !important;
    }
    .block-container { background: transparent !important; padding-top: 3rem !important; }

    /* ── Carte login (le container st.form) ── */
    [data-testid="stForm"] {
        background: #ffffff !important;
        border-radius: 24px !important;
        padding: 52px 52px 40px !important;
        box-shadow:
            0 32px 90px rgba(0, 0, 0, .55),
            0 0 0 1px rgba(255, 255, 255, .06) !important;
        border: none !important;
        overflow: hidden !important;
    }

    /* ── Champs de saisie ── */
    [data-testid="stForm"] [data-testid="stTextInput"] input {
        border: 1.5px solid #dce6f4 !important;
        border-radius: 10px !important;
        background: #f7faff !important;
        color: #1a2a4a !important;
        font-size: 15px !important;
        padding: 12px 16px !important;
        transition: border-color .2s, box-shadow .2s !important;
    }
    [data-testid="stForm"] [data-testid="stTextInput"] input:focus {
        border-color: #1565c0 !important;
        box-shadow: 0 0 0 3px rgba(21, 101, 192, .13) !important;
        background: #ffffff !important;
        outline: none !important;
    }
    [data-testid="stForm"] [data-testid="stTextInput"] label { display: none !important; }

    /* ── Bouton connexion ── */
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #1248a0 0%, #1565c0 50%, #1e88e5 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
        padding: 14px !important;
        box-shadow: 0 6px 22px rgba(21, 101, 192, .45) !important;
        transition: all .25s ease !important;
        width: 100% !important;
        margin-top: 8px !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        box-shadow: 0 10px 30px rgba(21, 101, 192, .65) !important;
        transform: translateY(-2px) !important;
    }
    [data-testid="stFormSubmitButton"] > button:active {
        transform: translateY(0) !important;
    }

    /* ── Alerte erreur ── */
    [data-testid="stAlert"] { border-radius: 10px !important; margin-top: 10px !important; }
    </style>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 1.6, 1])
    with center:
        with st.form("login_form"):

            # ── Logo + Titre ──────────────────────────────────────────────
            st.markdown("""
            <div style="text-align:center; margin-bottom:34px;">
                <div style="width:90px; height:90px;
                            background:linear-gradient(135deg,#1248a0 0%,#1e88e5 100%);
                            border-radius:50%; display:inline-flex; align-items:center;
                            justify-content:center; font-size:42px;
                            box-shadow:0 10px 32px rgba(21,101,192,.45);
                            margin-bottom:22px;">
                    🏥
                </div>
                <div style="font-size:22px; font-weight:800; color:#0d2952;
                            line-height:1.35; letter-spacing:-.3px; margin-bottom:10px;">
                    Système de Prédiction<br>de Réadmission Hospitalière
                </div>
                <div style="font-size:11px; font-weight:700; color:#1565c0;
                            letter-spacing:2.5px; text-transform:uppercase; margin-bottom:18px;">
                    CHU — Centre Hospitalier Universitaire
                </div>
                <div style="height:1px;
                            background:linear-gradient(90deg,transparent,#dce6f4,transparent);
                            margin-bottom:18px;"></div>
                <div style="font-size:12px; color:#8fa5bc; font-style:italic;">
                    Accès réservé au personnel médical autorisé
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Champ utilisateur ─────────────────────────────────────────
            st.markdown(
                "<div style='font-size:12.5px; font-weight:600; color:#4a5f78;"
                " margin-bottom:5px;'>👤 &nbsp; Nom d'utilisateur</div>",
                unsafe_allow_html=True,
            )
            username = st.text_input(
                "u", placeholder="Entrez votre identifiant",
                label_visibility="collapsed",
            )

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            # ── Champ mot de passe ────────────────────────────────────────
            st.markdown(
                "<div style='font-size:12.5px; font-weight:600; color:#4a5f78;"
                " margin-bottom:5px;'>🔒 &nbsp; Mot de passe</div>",
                unsafe_allow_html=True,
            )
            password = st.text_input(
                "p", type="password", placeholder="••••••••",
                label_visibility="collapsed",
            )

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            submitted = st.form_submit_button(
                "Se connecter", use_container_width=True, type="primary",
            )

            # ── Pied de carte ─────────────────────────────────────────────
            st.markdown(
                "<div style='margin-top:26px; padding-top:16px;"
                " border-top:1px solid #eef2f8; text-align:center;'>"
                "<div style='color:#b8cad8; font-size:10.5px; letter-spacing:.3px;'>"
                "medecin / medecin123 &nbsp;·&nbsp; admin / admin123"
                "</div>"
                "<div style='color:#8fa5bc; font-size:10.5px; margin-top:8px;"
                " letter-spacing:.3px;'>"
                "© 2025 CHU — Système IA de prédiction clinique"
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        if submitted:
            user = USERS.get(username.strip())
            if user and user["password"] == password:
                st.session_state.logged_in = True
                st.session_state.user_role = user["role"]
                st.session_state.username  = username.strip()
                st.rerun()
            else:
                st.error("Identifiants incorrects. Vérifiez votre nom d'utilisateur et mot de passe.")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def show_sidebar():
    role      = st.session_state.user_role
    user_info = USERS[st.session_state.username]
    avatar    = "👨‍⚕️" if role == "medecin" else "⚙️"

    badge_bg    = "#e3f2fd" if role == "medecin" else "#fff3e0"
    badge_color = "#1565c0" if role == "medecin" else "#e65100"
    badge_label = "MÉDECIN" if role == "medecin" else "ADMINISTRATEUR"

    if role == "medecin":
        pages     = [("👥", "Gestion des patients"), ("🔬", "Prédiction"), ("👤", "Mon profil")]
        state_key = "medecin_page"
    else:
        pages     = [("📊", "Tableau de bord"), ("📈", "Performance des modèles"), ("🔍", "Exploration des données")]
        state_key = "admin_page"

    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:20px 0 12px;">
            <div style="font-size:52px;">{avatar}</div>
            <div style="font-weight:800; color:#1565c0; font-size:17px; margin-top:6px;">
                {user_info['nom']}
            </div>
            <div style="color:#9e9e9e; font-size:12px; margin-top:3px;">
                {user_info['specialite']}
            </div>
            <span style="display:inline-block; background:{badge_bg}; color:{badge_color};
                         padding:4px 16px; border-radius:20px; font-size:11px;
                         font-weight:800; margin-top:10px; letter-spacing:.6px;">
                {badge_label}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("<div class='nav-label'>Navigation</div>", unsafe_allow_html=True)

        for icon, label in pages:
            is_active = st.session_state[state_key] == label
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{label}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state[state_key] = label
                st.rerun()

        st.divider()
        if st.button("🚪  Déconnexion", use_container_width=True):
            for k in list(DEFAULTS.keys()):
                st.session_state[k] = DEFAULTS[k]
            st.rerun()

        st.markdown(
            "<div style='text-align:center; color:#ccc; font-size:11px; margin-top:20px;'>"
            "© 2025 Hôpital — v2.0</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Composant : résultat de prédiction + SHAP
# ─────────────────────────────────────────────────────────────────────────────
def show_prediction_results(prob: float, risk: str, patient: dict):
    color   = RISK_COLORS[risk]
    risk_fr = RISK_LABELS[risk]

    st.divider()
    st.subheader("Résultat de la prédiction")

    col_risk, col_gauge = st.columns([1, 2])

    with col_risk:
        st.markdown(f"""
        <div class="risk-card" style="background:{color}14; border:2px solid {color};">
            <div style="font-size:11px; color:#999; letter-spacing:1.5px;
                        text-transform:uppercase; margin-bottom:10px;">Niveau de risque</div>
            <div style="font-size:44px; font-weight:900; color:{color}; letter-spacing:2px;">
                {risk_fr}
            </div>
            <div style="font-size:36px; font-weight:700; color:#1a1a2e; margin-top:10px;">
                {prob * 100:.1f}&nbsp;%
            </div>
            <div style="font-size:12px; color:#bbb; margin-top:6px;">
                probabilité de réadmission à 30 jours
            </div>
        </div>
        <div style="font-size:11px; color:#888; padding:10px 8px; background:#f8f9fa;
                    border-radius:8px; text-align:center; line-height:2.2;">
            &lt; 30&thinsp;% &rarr; <span style="color:#28a745; font-weight:700;">Faible</span><br>
            30–60&thinsp;% &rarr; <span style="color:#fd7e14; font-weight:700;">Moyen</span><br>
            &ge; 60&thinsp;% &rarr; <span style="color:#dc3545; font-weight:700;">Élevé</span>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 42, "color": color}},
            title={"text": "Probabilité de réadmission à 30 jours", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickfont": {"size": 12}},
                "bar":  {"color": color, "thickness": 0.28},
                "bgcolor": "white",
                "steps": [
                    {"range": [0,  30],  "color": "#d4edda"},
                    {"range": [30, 60],  "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "#333", "width": 3},
                    "thickness": 0.85,
                    "value": prob * 100,
                },
            },
        ))
        gauge.update_layout(height=300, margin={"t": 60, "b": 10, "l": 20, "r": 20})
        st.plotly_chart(gauge, use_container_width=True)

    st.subheader("Explication SHAP — contribution de chaque variable")
    _show_shap(patient)


def _show_shap(patient: dict):
    try:
        import shap

        bp_sys = float(patient.get("bp_systolic", 120))
        bp_dia = float(patient.get("bp_diastolic", 80))
        db_bin = int(patient.get("diabetes", 0))
        ht_bin = int(patient.get("hypertension", 0))

        model_obj  = load_model()
        scaler_obj = load_scaler()
        feat_names = load_feature_names()
        X_df = preprocess_input(
            int(patient["age"]), float(patient["bmi"]), float(patient["cholesterol"]),
            int(patient["medication_count"]), int(patient["length_of_stay"]),
            bp_sys, bp_dia, db_bin, ht_bin,
            patient.get("gender", "Other"),
            patient.get("discharge_destination", "Home"),
            scaler_obj, feat_names,
        )
        explainer   = shap.TreeExplainer(model_obj)
        shap_values = explainer(X_df)
        shap.plots.waterfall(shap_values[0], show=False)
        fig_shap = plt.gcf()
        fig_shap.set_size_inches(10, 6)
        st.pyplot(fig_shap)
        plt.close("all")

    except ImportError:
        st.info("Installez SHAP pour ce graphe : `pip install shap`")
        _show_feature_importance_fallback()
    except Exception as exc:
        st.warning(f"SHAP indisponible : {exc}")
        _show_feature_importance_fallback()


def _show_feature_importance_fallback():
    try:
        fi = pd.read_csv(FI_PATH).sort_values("importance", ascending=False).head(10)
        fig_fi = px.bar(
            fi, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Blues",
            title="Top 10 variables les plus importantes (importance globale)",
            labels={"importance": "Importance", "feature": "Variable"},
        )
        fig_fi.update_layout(yaxis={"autorange": "reversed"}, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)
    except FileNotFoundError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# MÉDECIN — Page 1 : Gestion des patients
# ─────────────────────────────────────────────────────────────────────────────
def _table_row(row, W: list, key_prefix: str):
    """Affiche une ligne du tableau + fiche inline si sélectionnée."""
    pid   = int(row["patient_id"])
    prob  = float(row["probabilite"])
    risk  = str(row["risque"])
    color = RISK_COLORS.get(risk, "#888")

    cols = st.columns(W)
    cols[0].write(row["nom"])
    cols[1].write(row["prenom"])
    cols[2].write(f"{int(row['age'])} ans")
    cols[3].markdown(
        f"<span style='color:{color}; font-weight:800; font-size:15px;'>"
        f"{prob*100:.1f}%</span>",
        unsafe_allow_html=True,
    )
    cols[4].markdown(
        f"<span style='background:{color}22; color:{color}; font-weight:700;"
        f" padding:3px 10px; border-radius:12px; font-size:12px;'>"
        f"{RISK_LABELS.get(risk, risk)}</span>",
        unsafe_allow_html=True,
    )
    b_fiche, b_pred = cols[5].columns(2)
    if b_fiche.button("👁 Fiche", key=f"{key_prefix}_f_{pid}"):
        st.session_state.selected_csv_pid = (
            None if st.session_state.selected_csv_pid == pid else pid
        )
        st.rerun()
    if b_pred.button("🔬 Prédire", key=f"{key_prefix}_p_{pid}", type="primary"):
        st.session_state.pred_patient_data = {
            k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()
        }
        st.session_state.pred_prob         = None
        st.session_state.pred_risk         = None
        st.session_state.selected_csv_pid  = None
        st.session_state.medecin_page      = "Prédiction"
        st.rerun()
    if st.session_state.selected_csv_pid == pid:
        _show_patient_card_csv(row, key_suffix=key_prefix)


def _table_header(W: list):
    hdr = st.columns(W)
    for c, lbl in zip(hdr, [
        "**Nom**", "**Prénom**", "**Âge**", "**Probabilité**", "**Risque**", "**Actions**",
    ]):
        c.markdown(lbl)
    st.markdown("<hr style='margin:4px 0 6px; border-color:#e8e8e8;'>", unsafe_allow_html=True)


def page_gestion_patients():
    st.title("👥 Gestion des patients")

    with st.spinner("Analyse de la base de données en cours…"):
        try:
            df = predict_all_patients()
        except FileNotFoundError:
            st.error(f"Dataset ou modèle introuvable. Vérifiez {DATA_PATH} et {MODEL_PATH}.")
            return
        except Exception as exc:
            st.error(f"Erreur lors du chargement : {exc}")
            return

    nb_total = len(df)
    nb_high  = int((df["risque"] == "High").sum())
    nb_med   = int((df["risque"] == "Medium").sum())
    nb_low   = nb_total - nb_high - nb_med

    # ── KPIs ────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total patients",   f"{nb_total:,}")
    k2.metric("🔴 Risque élevé",  f"{nb_high:,}",
              delta=f"{nb_high/nb_total*100:.1f}% du total", delta_color="inverse")
    k3.metric("🟠 Risque moyen",  f"{nb_med:,}",
              delta=f"{nb_med/nb_total*100:.1f}% du total",  delta_color="off")
    k4.metric("🟢 Risque faible", f"{nb_low:,}")

    # ── Section 1 : Patients à risque élevé ─────────────────────────────────
    st.divider()
    st.markdown("### 🔴 Patients à risque élevé de réadmission")

    high_risk = df[df["risque"] == "High"].sort_values("probabilite", ascending=False)

    if nb_high == 0:
        st.success("Aucun patient à risque élevé détecté dans la base.")
    else:
        top25 = high_risk.head(25)
        extra = nb_high - 25
        note  = (f" · {extra:,} autres accessibles via la recherche ci-dessous"
                 if extra > 0 else "")
        st.caption(
            f"**{nb_high:,}** patients avec probabilité > 60 % — "
            f"affichage des **{min(nb_high, 25)} plus critiques**{note}"
        )
        W = [1.0, 1.0, 0.5, 1.0, 0.9, 1.4]
        _table_header(W)
        for _, row in top25.iterrows():
            _table_row(row, W, key_prefix="h")

    # ── Section 2 : Recherche ────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔍 Rechercher un patient")

    query = st.text_input(
        "",
        placeholder="Rechercher par nom ou prénom…",
        label_visibility="collapsed",
    )

    if not query.strip():
        st.markdown(
            "<div style='color:#9e9e9e; font-size:13px; text-align:center; padding:24px 0;'>"
            "Tapez un nom ou un prénom pour rechercher dans la base de données."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    q       = query.strip().lower()
    mask    = (df["nom"].str.lower().str.contains(q, na=False) |
               df["prenom"].str.lower().str.contains(q, na=False))
    results = df[mask]
    nb      = len(results)

    if nb == 0:
        st.warning(f"Aucun patient trouvé pour **« {query} »**.")
        return

    top = results.head(20)
    st.caption(f"**{nb}** résultat(s) — {min(nb, 20)} affiché(s), triés par risque décroissant")
    top = top.sort_values("probabilite", ascending=False)

    W = [1.0, 1.0, 0.5, 1.0, 0.9, 1.4]
    _table_header(W)
    for _, row in top.iterrows():
        _table_row(row, W, key_prefix="s")


# ─────────────────────────────────────────────────────────────────────────────
# MÉDECIN — Page 2 : Prédiction
# ─────────────────────────────────────────────────────────────────────────────
def page_prediction_medecin():
    st.title("🔬 Prédiction du risque de réadmission")

    # ── Patient sélectionné depuis la Gestion des patients (CSV) ───────────
    if st.session_state.pred_patient_data is not None:
        patient  = st.session_state.pred_patient_data
        genre_fr = {"Male": "Homme", "Female": "Femme", "Other": "Autre"}

        if st.button("← Retour à la liste des patients"):
            st.session_state.pred_patient_data = None
            st.session_state.pred_prob         = None
            st.session_state.pred_risk         = None
            st.session_state.medecin_page      = "Gestion des patients"
            st.rerun()

        st.divider()
        st.subheader(f"Fiche patient — {patient.get('prenom','—')} {patient.get('nom','—')}")

        diab_fr = "Oui" if int(patient.get("diabetes", 0)) == 1 else "Non"
        hype_fr = "Oui" if int(patient.get("hypertension", 0)) == 1 else "Non"
        j       = int(patient.get("length_of_stay", 0))
        nmed    = int(patient.get("medication_count", 0))
        bp_val  = f"{int(patient.get('bp_systolic', '—'))}/{int(patient.get('bp_diastolic', '—'))} mmHg"

        left, right = st.columns(2)
        with left:
            st.markdown(
                "<h5 style='color:#1565c0; border-bottom:2px solid #1565c0;"
                " padding-bottom:6px; margin-bottom:14px;'>Informations démographiques</h5>",
                unsafe_allow_html=True,
            )
            st.table(pd.DataFrame({
                "Champ": ["Nom", "Prénom", "Âge", "Genre", "IMC (BMI)", "Pression artérielle"],
                "Valeur": [
                    patient.get("nom", "—"),
                    patient.get("prenom", "—"),
                    f"{patient['age']} ans",
                    genre_fr.get(str(patient.get("gender", "")), "—"),
                    f"{patient.get('bmi', '—')} kg/m²",
                    bp_val,
                ],
            }).set_index("Champ"))

        with right:
            st.markdown(
                "<h5 style='color:#28a745; border-bottom:2px solid #28a745;"
                " padding-bottom:6px; margin-bottom:14px;'>Informations médicales</h5>",
                unsafe_allow_html=True,
            )
            st.table(pd.DataFrame({
                "Champ": [
                    "Diabète", "Hypertension", "Cholestérol",
                    "Médicaments", "Durée du séjour", "Destination à la sortie",
                ],
                "Valeur": [
                    diab_fr, hype_fr,
                    f"{patient.get('cholesterol', '—')} mg/dL",
                    f"{nmed} médicament{'s' if nmed > 1 else ''}",
                    f"{j} jour{'s' if j > 1 else ''}",
                    DEST_LABELS.get(str(patient.get("discharge_destination", "")), "—"),
                ],
            }).set_index("Champ"))

        st.divider()
        if st.button("Prédire le risque de réadmission", type="primary", use_container_width=True,
                     key="pred_csv_btn"):
            with st.spinner("Calcul en cours…"):
                try:
                    prob, risk = run_prediction(patient)
                    st.session_state.pred_prob = prob
                    st.session_state.pred_risk = risk
                except FileNotFoundError:
                    st.error("Modèle introuvable — entraînez le modèle : `python -m src.train`")
                    st.stop()
                except Exception as exc:
                    st.error(f"Erreur de prédiction : {exc}")
                    st.stop()

        if st.session_state.pred_prob is not None:
            show_prediction_results(st.session_state.pred_prob, st.session_state.pred_risk, patient)
        return

    # ── Aucun patient sélectionné ───────────────────────────────────────────
    st.info(
        "Aucun patient sélectionné. "
        "Recherchez un patient dans la page **Gestion des patients** et cliquez sur **Prédire**."
    )
    if st.button("Aller à la Gestion des patients", type="primary"):
        st.session_state.medecin_page = "Gestion des patients"
        st.rerun()

    # ── Compatibilité : patients JSON ───────────────────────────────────────
    patients = load_patients()
    if not patients:
        return

    st.markdown("### Sélection du patient")
    options_labels = [
        f"#{p['id']} — {p['prenom']} {p['nom']}  ({p['age']} ans)"
        for p in patients
    ]

    default_idx = 0
    if st.session_state.pred_pid is not None:
        ids = [p["id"] for p in patients]
        if st.session_state.pred_pid in ids:
            default_idx = ids.index(st.session_state.pred_pid)

    selected_label = st.selectbox("Choisir un patient", options_labels, index=default_idx)
    selected_id    = int(selected_label.split(" —")[0].replace("#", "").strip())
    patient        = next(p for p in patients if p["id"] == selected_id)

    if st.session_state.pred_pid != selected_id:
        st.session_state.pred_pid  = selected_id
        st.session_state.pred_prob = None
        st.session_state.pred_risk = None

    # ── Fiche patient ───────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"Fiche patient — {patient['prenom']} {patient['nom']}")

    genre_fr = {"Male": "Homme", "Female": "Femme", "Other": "Autre"}
    left, right = st.columns(2)

    with left:
        st.markdown(
            "<h5 style='color:#1565c0; border-bottom:2px solid #1565c0;"
            " padding-bottom:6px; margin-bottom:14px;'>Informations démographiques</h5>",
            unsafe_allow_html=True,
        )
        st.table(pd.DataFrame({
            "Champ": ["Nom", "Prénom", "Âge", "Genre", "IMC (BMI)", "Pression artérielle"],
            "Valeur": [
                patient["nom"],
                patient["prenom"],
                f"{patient['age']} ans",
                genre_fr.get(patient.get("gender", ""), patient.get("gender", "—")),
                f"{patient['bmi']} kg/m²",
                f"{patient.get('bp_systolic','—')}/{patient.get('bp_diastolic','—')} mmHg",
            ],
        }).set_index("Champ"))

    with right:
        st.markdown(
            "<h5 style='color:#28a745; border-bottom:2px solid #28a745;"
            " padding-bottom:6px; margin-bottom:14px;'>Informations médicales</h5>",
            unsafe_allow_html=True,
        )
        j    = patient["length_of_stay"]
        nmed = patient["medication_count"]
        st.table(pd.DataFrame({
            "Champ": [
                "Diabète", "Hypertension", "Cholestérol",
                "Médicaments", "Durée du séjour", "Destination à la sortie",
            ],
            "Valeur": [
                patient["diabetes"],
                patient["hypertension"],
                f"{patient['cholesterol']} mg/dL",
                f"{nmed} médicament{'s' if nmed > 1 else ''}",
                f"{j} jour{'s' if j > 1 else ''}",
                DEST_LABELS.get(patient.get("discharge_destination", "Home"), "—"),
            ],
        }).set_index("Champ"))

    # ── Bouton prédiction ───────────────────────────────────────────────────
    st.divider()
    if st.button(
        "Prédire le risque de réadmission",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Calcul en cours…"):
            try:
                prob, risk = run_prediction(patient)
                st.session_state.pred_prob = prob
                st.session_state.pred_risk = risk
            except FileNotFoundError:
                st.error("Modèle introuvable — entraînez le modèle : `python -m src.train`")
                st.stop()
            except Exception as exc:
                st.error(f"Erreur de prédiction : {exc}")
                st.stop()

    if st.session_state.pred_prob is not None and st.session_state.pred_pid == selected_id:
        show_prediction_results(
            st.session_state.pred_prob,
            st.session_state.pred_risk,
            patient,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MÉDECIN — Page 3 : Mon profil
# ─────────────────────────────────────────────────────────────────────────────
def page_profil_medecin():
    st.title("👤 Mon profil")

    user_info = USERS[st.session_state.username]

    # Charger les prédictions pour les statistiques (cache = instantané)
    try:
        df_pred = predict_all_patients()
        nb_total     = len(df_pred)
        nb_high      = int((df_pred["risque"] == "High").sum())
        nb_med       = int((df_pred["risque"] == "Medium").sum())
        taux_readmit = f"{nb_high / nb_total * 100:.1f} %"
    except Exception:
        nb_total, nb_high, nb_med = 30_000, 0, 0
        taux_readmit = "—"

    # ── Carte profil ─────────────────────────────────────────────────────────
    st.markdown(
        f"<div class='profile-card'>"
        f"<div style='display:flex; align-items:center; gap:28px;'>"
        f"<div style='font-size:80px; line-height:1;'>👩‍⚕️</div>"
        f"<div>"
        f"<div style='font-size:26px; font-weight:800; color:#1565c0;'>{user_info['nom']}</div>"
        f"<div style='color:#9e9e9e; font-size:14px; margin-top:4px;'>{user_info['specialite']}</div>"
        f"<div style='color:#9e9e9e; font-size:13px; margin-top:3px;'>{user_info['email']}</div>"
        f"<span style='display:inline-block; background:#e3f2fd; color:#1565c0;"
        f" padding:5px 18px; border-radius:20px; font-size:12px;"
        f" font-weight:800; margin-top:12px; letter-spacing:.6px;'>MÉDECIN</span>"
        f"</div></div>"
        f"<hr style='margin:28px 0 20px; border-color:#f0f0f0;'>"
        f"<div style='display:grid; grid-template-columns:1fr 1fr 1fr 1fr 1fr; gap:18px;'>"

        # Stat 1 — Identifiant
        f"<div>"
        f"<div style='color:#bbb; font-size:11px; text-transform:uppercase; letter-spacing:1px;'>"
        f"Identifiant</div>"
        f"<div style='font-weight:700; color:#333; margin-top:5px; font-size:15px;'>"
        f"{st.session_state.username}</div>"
        f"</div>"

        # Stat 2 — Patients suivis (high risk)
        f"<div>"
        f"<div style='color:#bbb; font-size:11px; text-transform:uppercase; letter-spacing:1px;'>"
        f"Patients suivis</div>"
        f"<div style='font-weight:800; color:#dc3545; margin-top:5px; font-size:26px;'>"
        f"{nb_high:,}</div>"
        f"<div style='font-size:10px; color:#dc3545; margin-top:2px;'>risque élevé</div>"
        f"</div>"

        # Stat 3 — Total système
        f"<div>"
        f"<div style='color:#bbb; font-size:11px; text-transform:uppercase; letter-spacing:1px;'>"
        f"Total système</div>"
        f"<div style='font-weight:800; color:#555; margin-top:5px; font-size:26px;'>"
        f"{nb_total:,}</div>"
        f"<div style='font-size:10px; color:#9e9e9e; margin-top:2px;'>patients</div>"
        f"</div>"

        # Stat 4 — Risque élevé %
        f"<div>"
        f"<div style='color:#bbb; font-size:11px; text-transform:uppercase; letter-spacing:1px;'>"
        f"Risque élevé</div>"
        f"<div style='font-weight:800; color:#fd7e14; margin-top:5px; font-size:26px;'>"
        f"{taux_readmit}</div>"
        f"<div style='font-size:10px; color:#fd7e14; margin-top:2px;'>du total</div>"
        f"</div>"

        # Stat 5 — Statut
        f"<div>"
        f"<div style='color:#bbb; font-size:11px; text-transform:uppercase; letter-spacing:1px;'>"
        f"Statut</div>"
        f"<div style='font-weight:700; color:#28a745; margin-top:5px; font-size:15px;'>"
        f"● Connecté</div>"
        f"</div>"

        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ── Patients les plus critiques ──────────────────────────────────────────
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.subheader("Patients les plus critiques à surveiller")

    if nb_high == 0:
        st.info("Aucun patient à risque élevé détecté dans la base.")
        return

    top_critical = (
        df_pred[df_pred["risque"] == "High"]
        .sort_values("probabilite", ascending=False)
        .head(10)
    )

    st.caption(
        f"Top 10 patients sur **{nb_high:,}** à risque élevé "
        f"(probabilité > 60 %) — triés par criticité décroissante"
    )

    genre_fr = {"Male": "Homme", "Female": "Femme", "Other": "Autre"}
    color_h  = RISK_COLORS["High"]

    W = [0.6, 0.9, 0.9, 0.4, 0.7, 1.0, 0.9]
    hdr = st.columns(W)
    for c, lbl in zip(hdr, [
        "**ID**", "**Nom**", "**Prénom**", "**Âge**",
        "**Genre**", "**Probabilité**", "**Risque**",
    ]):
        c.markdown(lbl)
    st.markdown("<hr style='margin:4px 0 6px; border-color:#ffd5d5;'>",
                unsafe_allow_html=True)

    for _, row in top_critical.iterrows():
        prob  = float(row["probabilite"])
        cols  = st.columns(W)
        cols[0].write(f"#{int(row['patient_id'])}")
        cols[1].write(str(row["nom"]))
        cols[2].write(str(row["prenom"]))
        cols[3].write(f"{int(row['age'])} ans")
        cols[4].write(genre_fr.get(str(row["gender"]), "—"))
        cols[5].markdown(
            f"<span style='color:{color_h}; font-weight:800; font-size:15px;'>"
            f"{prob*100:.1f}%</span>",
            unsafe_allow_html=True,
        )
        cols[6].markdown(
            f"<span style='background:{color_h}22; color:{color_h}; font-weight:700;"
            f" padding:3px 10px; border-radius:12px; font-size:12px;'>Élevé</span>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN — Page 1 : Tableau de bord
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard_admin():
    st.title("📊 Tableau de bord — Vue globale")

    try:
        df = load_raw_data()
    except FileNotFoundError:
        st.error("Dataset introuvable : data/hospital_readmissions_30k.csv")
        return

    n_patients    = len(df)
    readmit_rate  = round(df["readmitted_30_days"].mean() * 100, 1)
    avg_age       = df["age"].mean()
    avg_stay      = df["length_of_stay"].mean()
    diabetes_rate = df["diabetes"].mean() * 100

    try:
        df_pred = predict_all_patients()
        nb_high = int((df_pred["risque"] == "High").sum())
        nb_med  = int((df_pred["risque"] == "Medium").sum())
    except Exception:
        nb_high = nb_med = 0

    # ── KPIs — ligne 1 ──────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total patients",     f"{n_patients:,}")
    k2.metric("Taux de réadmission", f"{readmit_rate} %")
    k3.metric("Âge moyen",          f"{avg_age:.0f} ans")
    k4.metric("Séjour moyen",       f"{avg_stay:.1f} j")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("🔴 Patients risque élevé",  f"{nb_high:,}")
    k6.metric("🟠 Patients risque moyen",  f"{nb_med:,}")
    k7.metric("Taux diabète",              f"{diabetes_rate:.1f} %")
    k8.metric("Meilleur modèle (AUC-ROC)", "XGBoost — 0.740")

    # ── Graphes ligne 1 : réadmissions + genre ───────────────────────────────
    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        n_readmis     = int(df["readmitted_30_days"].sum())
        n_non_readmis = n_patients - n_readmis
        fig_pie = px.pie(
            values=[n_readmis, n_non_readmis],
            names=["Réadmis", "Non réadmis"],
            color=["Réadmis", "Non réadmis"],
            color_discrete_map={"Réadmis": "#dc3545", "Non réadmis": "#28a745"},
            title="Répartition des réadmissions à 30 jours",
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=14)
        fig_pie.update_layout(legend_title="Statut")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        genre_series = df["gender"].map({"Male": "Homme", "Female": "Femme", "Other": "Homme"}).value_counts()
        fig_g = px.pie(
            values=genre_series.values, names=genre_series.index,
            color=genre_series.index,
            color_discrete_map={"Homme": "#1565c0", "Femme": "#e91e8c"},
            title="Répartition par genre",
        )
        fig_g.update_traces(textinfo="percent+label", textfont_size=14)
        st.plotly_chart(fig_g, use_container_width=True)

    # ── Graphes ligne 2 : distributions âge + séjour ────────────────────────
    st.divider()
    col_c, col_d = st.columns(2)

    with col_c:
        fig_age = px.histogram(
            df, x="age", nbins=30, color_discrete_sequence=["#1565c0"],
            title="Distribution des âges des patients",
        )
        fig_age.update_layout(
            bargap=0.06, xaxis_title="Âge (années)",
            yaxis_title="Nombre de patients",
            plot_bgcolor="#fafbff",
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col_d:
        fig_sej = px.histogram(
            df, x="length_of_stay", nbins=20, color_discrete_sequence=["#42a5f5"],
            title="Distribution de la durée de séjour",
        )
        fig_sej.update_layout(
            bargap=0.06, xaxis_title="Durée de séjour (jours)",
            yaxis_title="Nombre de patients",
            plot_bgcolor="#fafbff",
        )
        st.plotly_chart(fig_sej, use_container_width=True)

    # ── Graphes ligne 3 : taux par tranche d'âge + destinations ─────────────
    st.divider()
    col_e, col_f = st.columns(2)

    with col_e:
        df_tr = df.copy()
        df_tr["Tranche d'âge"] = pd.cut(
            df_tr["age"], bins=[0, 40, 60, 75, 110],
            labels=["0–40 ans", "40–60 ans", "60–75 ans", "75+ ans"],
        )
        rate_age = (
            df_tr.groupby("Tranche d'âge", observed=True)["readmitted_30_days"]
            .apply(lambda x: round(x.mean() * 100, 1))
            .reset_index(name="Taux (%)")
        )
        fig_rate = px.bar(
            rate_age, x="Tranche d'âge", y="Taux (%)",
            color="Taux (%)", color_continuous_scale="Reds",
            text="Taux (%)",
            title="Taux de réadmission par tranche d'âge",
            labels={"Taux (%)": "Taux (%)"},
        )
        fig_rate.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_rate.update_layout(coloraxis_showscale=False, plot_bgcolor="#fafbff",
                               yaxis_title="Taux de réadmission (%)")
        st.plotly_chart(fig_rate, use_container_width=True)

    with col_f:
        df_dest = df.groupby(["discharge_destination", "readmitted_30_days"]).size().reset_index(name="count")
        df_dest["Statut"]      = df_dest["readmitted_30_days"].map({1: "Réadmis", 0: "Non réadmis"})
        df_dest["Destination"] = df_dest["discharge_destination"].map(DEST_LABELS)
        fig_dest = px.bar(
            df_dest, x="Destination", y="count", color="Statut", barmode="group",
            color_discrete_map={"Réadmis": "#dc3545", "Non réadmis": "#28a745"},
            title="Réadmissions par destination de sortie",
        )
        fig_dest.update_layout(plot_bgcolor="#fafbff")
        st.plotly_chart(fig_dest, use_container_width=True)

    # ── Graphe ligne 4 : boîte à moustaches durée/âge ───────────────────────
    st.divider()
    df_box = df.copy()
    df_box["Tranche d'âge"] = pd.cut(
        df_box["age"], bins=[0, 40, 60, 75, 110],
        labels=["< 40 ans", "40–60 ans", "60–75 ans", "> 75 ans"],
    )
    df_box["Réadmission"] = df_box["readmitted_30_days"].map({1: "Oui", 0: "Non"})
    fig_box2 = px.box(
        df_box, x="Tranche d'âge", y="length_of_stay", color="Réadmission",
        color_discrete_map={"Oui": "#dc3545", "Non": "#28a745"},
        labels={"length_of_stay": "Durée de séjour (jours)"},
        title="Durée de séjour selon la tranche d'âge et le statut de réadmission",
    )
    fig_box2.update_layout(plot_bgcolor="#fafbff")
    st.plotly_chart(fig_box2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN — Page 2 : Performance des modèles
# ─────────────────────────────────────────────────────────────────────────────
_PERF = {
    "Régression Logistique": {"AUC-ROC": 0.56, "F1": 0.17, "Rappel": 0.22, "Précision": 0.15},
    "Random Forest":         {"AUC-ROC": 0.55, "F1": 0.18, "Rappel": 0.25, "Précision": 0.14},
    "XGBoost":               {"AUC-ROC": 0.74, "F1": 0.46, "Rappel": 0.79, "Précision": 0.32},
}
_BEST = "XGBoost"

def page_performance_modeles():
    st.title("📈 Performance des modèles MLflow")

    # ── Meilleur modèle — KPIs ───────────────────────────────────────────────
    st.markdown(
        "<div style='background:#d4edda; border-left:5px solid #28a745; border-radius:10px;"
        " padding:14px 20px; margin-bottom:20px;'>"
        "<b style='color:#155724; font-size:15px;'>✅ Modèle sélectionné : XGBoost</b>"
        "<span style='color:#155724; font-size:13px; margin-left:12px;'>"
        "Meilleure combinaison AUC-ROC + Rappel pour la détection des réadmissions à risque élevé"
        "</span></div>",
        unsafe_allow_html=True,
    )
    b = _PERF[_BEST]
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("AUC-ROC",   f"{b['AUC-ROC']:.2f}")
    b2.metric("F1-Score",  f"{b['F1']:.2f}")
    b3.metric("Rappel",    f"{b['Rappel']:.2f}")
    b4.metric("Précision", f"{b['Précision']:.2f}")

    # ── Tableau comparatif ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Comparaison des 3 modèles")

    df_met = pd.DataFrame(_PERF).T

    def style_table(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for col in df.columns:
            max_val = df[col].max()
            for idx in df.index:
                if df.loc[idx, col] == max_val:
                    styles.loc[idx, col] = "background-color:#d4edda; font-weight:bold; color:#155724"
        styles.loc[_BEST] = styles.loc[_BEST].apply(
            lambda x: x + "; border-left:3px solid #28a745" if x else "border-left:3px solid #28a745"
        )
        return styles

    st.dataframe(
        df_met.style.apply(style_table, axis=None).format("{:.2f}"),
        use_container_width=True,
    )
    st.caption("Cellule verte = meilleure valeur pour la métrique · Ligne XGBoost bordée en vert")

    # ── Graphe comparatif ────────────────────────────────────────────────────
    st.divider()
    df_melt = df_met.reset_index().rename(columns={"index": "Modèle"}).melt(
        id_vars="Modèle", var_name="Métrique", value_name="Score"
    )
    color_map = {"Régression Logistique": "#90caf9", "Random Forest": "#42a5f5", "XGBoost": "#1565c0"}
    fig_bar = px.bar(
        df_melt, x="Métrique", y="Score", color="Modèle", barmode="group",
        color_discrete_map=color_map,
        title="Comparaison des performances par modèle",
        text="Score",
    )
    fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_bar.update_layout(yaxis_range=[0, 1.05], plot_bgcolor="#fafbff",
                          legend_title="Modèle")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Explication XGBoost ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Pourquoi XGBoost a été sélectionné ?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Critères de sélection clinique :**
- **AUC-ROC = 0.74** : le plus élevé des 3 modèles (+34% vs Random Forest)
- **Rappel = 0.79** : détecte 79% des vrais cas de réadmission — crucial pour ne pas rater un patient à risque
- **F1 = 0.46** : meilleur équilibre précision/rappel
- Gradient boosting robuste aux données déséquilibrées (classe "réadmis" minoritaire)
        """)
    with col2:
        st.markdown("""
**Limites des autres modèles :**
- *Régression Logistique* : AUC-ROC = 0.56, Rappel = 0.22 → rate 78% des réadmissions réelles
- *Random Forest* : AUC-ROC = 0.55, légèrement meilleur en Rappel (0.25) mais insuffisant
- Les deux modèles linéaires ne capturent pas les interactions non-linéaires entre variables cliniques

**Paramètres XGBoost optimisés :**
`scale_pos_weight=7` pour compenser le déséquilibre de classes,
`early_stopping_rounds=50`, `n_estimators=1000`
        """)

    # ── Importance des variables ─────────────────────────────────────────────
    st.divider()
    st.subheader("Importance des variables — XGBoost")
    try:
        fi = pd.read_csv(FI_PATH).sort_values("importance", ascending=False)
        fi_fr = {
            "poly_risk": "Risque combiné (diabète × hypertension)",
            "diabetes": "Diabète", "hypertension": "Hypertension",
            "age_risk": "Âge > 65 ans", "dest_Home": "Sortie à domicile",
            "length_of_stay": "Durée de séjour", "gender_Female": "Genre féminin",
            "medication_count": "Nombre de médicaments", "bmi_category": "Catégorie IMC",
            "dest_Nursing_Facility": "Sortie en EHPAD", "age": "Âge",
            "cholesterol_ratio": "Rapport cholestérol", "age_bmi": "Âge × IMC",
            "complex_patient": "Patient complexe", "bp_pulse_pressure": "Pression différentielle",
        }
        fi["feature_fr"] = fi["feature"].map(fi_fr).fillna(fi["feature"])
        fig_fi = px.bar(
            fi.head(15), x="importance", y="feature_fr", orientation="h",
            color="importance", color_continuous_scale="Blues",
            title="Top 15 variables les plus influentes sur la prédiction",
            labels={"importance": "Importance", "feature_fr": "Variable"},
        )
        fig_fi.update_layout(yaxis={"autorange": "reversed"}, coloraxis_showscale=False,
                             height=500, plot_bgcolor="#fafbff")
        st.plotly_chart(fig_fi, use_container_width=True)
    except FileNotFoundError:
        st.warning("models/feature_importance.csv introuvable.")


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN — Page 3 : Exploration des données
# ─────────────────────────────────────────────────────────────────────────────
def page_exploration_donnees():
    st.title("🔍 Exploration des données")

    try:
        df = load_raw_data()
    except FileNotFoundError:
        st.error("Dataset introuvable : data/hospital_readmissions_30k.csv")
        return

    df_display = df.drop(columns=["patient_id"], errors="ignore")
    st.caption(f"Dataset : **{len(df):,}** patients · **{df_display.shape[1]}** variables")

    with st.expander("Aperçu du dataset (10 premières lignes)"):
        st.dataframe(df_display.head(10), use_container_width=True)

    # Colonne réadmission en français (réutilisée)
    df_base = df.copy()
    df_base["Réadmission"] = df_base["readmitted_30_days"].map({1: "Oui", 0: "Non"})

    # ── Boxplots IMC + Âge selon réadmission ────────────────────────────────
    st.divider()
    st.subheader("Variables cliniques selon le statut de réadmission")
    col_a, col_b = st.columns(2)

    with col_a:
        fig_bmi = px.box(
            df_base, x="Réadmission", y="bmi", color="Réadmission",
            color_discrete_map={"Oui": "#dc3545", "Non": "#28a745"},
            points="outliers", labels={"bmi": "IMC (BMI)"},
            title="Distribution de l'IMC selon la réadmission",
        )
        fig_bmi.update_layout(plot_bgcolor="#fafbff", showlegend=False)
        st.plotly_chart(fig_bmi, use_container_width=True)

    with col_b:
        fig_age_box = px.box(
            df_base, x="Réadmission", y="age", color="Réadmission",
            color_discrete_map={"Oui": "#dc3545", "Non": "#28a745"},
            points="outliers", labels={"age": "Âge (années)"},
            title="Distribution de l'âge selon la réadmission",
        )
        fig_age_box.update_layout(plot_bgcolor="#fafbff", showlegend=False)
        st.plotly_chart(fig_age_box, use_container_width=True)

    # ── Distribution cholestérol + taux par destination ─────────────────────
    st.divider()
    col_c, col_d = st.columns(2)

    with col_c:
        fig_chol = px.histogram(
            df_base, x="cholesterol", nbins=30, color="Réadmission",
            color_discrete_map={"Oui": "#dc3545", "Non": "#28a745"},
            barmode="overlay", opacity=0.7,
            title="Distribution du cholestérol",
            labels={"cholesterol": "Cholestérol (mg/dL)", "count": "Nombre"},
        )
        fig_chol.update_layout(bargap=0.04, plot_bgcolor="#fafbff")
        st.plotly_chart(fig_chol, use_container_width=True)

    with col_d:
        df_dest_rate = (
            df.groupby("discharge_destination")["readmitted_30_days"]
            .apply(lambda x: round(x.mean() * 100, 1))
            .reset_index(name="Taux de réadmission (%)")
        )
        df_dest_rate["Destination"] = df_dest_rate["discharge_destination"].map(DEST_LABELS)
        fig_dest_r = px.bar(
            df_dest_rate, x="Destination", y="Taux de réadmission (%)",
            color="Taux de réadmission (%)", color_continuous_scale="Reds",
            text="Taux de réadmission (%)",
            title="Taux de réadmission selon la destination de sortie",
        )
        fig_dest_r.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_dest_r.update_layout(coloraxis_showscale=False, plot_bgcolor="#fafbff")
        st.plotly_chart(fig_dest_r, use_container_width=True)

    # ── Diabète + Hypertension ───────────────────────────────────────────────
    st.divider()
    col_e, col_f = st.columns(2)

    with col_e:
        df_diab = df.groupby(["diabetes", "readmitted_30_days"]).size().reset_index(name="count")
        df_diab["Statut"]  = df_diab["readmitted_30_days"].map({1: "Réadmis", 0: "Non réadmis"})
        df_diab["Diabète"] = df_diab["diabetes"].map({1: "Oui", 0: "Non"})
        fig_diab = px.bar(
            df_diab, x="Diabète", y="count", color="Statut", barmode="group",
            color_discrete_map={"Réadmis": "#dc3545", "Non réadmis": "#28a745"},
            title="Réadmissions selon le diabète",
        )
        fig_diab.update_layout(plot_bgcolor="#fafbff")
        st.plotly_chart(fig_diab, use_container_width=True)

    with col_f:
        df_ht = df.groupby(["hypertension", "readmitted_30_days"]).size().reset_index(name="count")
        df_ht["Statut"]       = df_ht["readmitted_30_days"].map({1: "Réadmis", 0: "Non réadmis"})
        df_ht["Hypertension"] = df_ht["hypertension"].map({1: "Oui", 0: "Non"})
        fig_ht = px.bar(
            df_ht, x="Hypertension", y="count", color="Statut", barmode="group",
            color_discrete_map={"Réadmis": "#dc3545", "Non réadmis": "#28a745"},
            title="Réadmissions selon l'hypertension",
        )
        fig_ht.update_layout(plot_bgcolor="#fafbff")
        st.plotly_chart(fig_ht, use_container_width=True)

    # ── Heatmap de corrélation ───────────────────────────────────────────────
    st.divider()
    st.subheader("Heatmap de corrélation des variables numériques")
    df_corr = df.copy()
    df_corr["Réadmission"]   = df_corr["readmitted_30_days"]
    df_corr["Diabète"]       = df_corr["diabetes"]
    df_corr["Hypertension"]  = df_corr["hypertension"]
    df_corr["Pression sys."] = df_corr["bp_systolic"]
    df_corr["Pression dia."] = df_corr["bp_diastolic"]
    df_corr = df_corr.rename(columns={
        "age": "Âge", "cholesterol": "Cholestérol", "bmi": "IMC",
        "medication_count": "Médicaments", "length_of_stay": "Durée séjour",
    })
    num_cols_corr = ["Âge", "Cholestérol", "IMC", "Médicaments", "Durée séjour",
                     "Pression sys.", "Pression dia.", "Diabète", "Hypertension", "Réadmission"]
    corr = df_corr[num_cols_corr].corr()
    fig_corr = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        text_auto=".2f", title="Matrice de corrélation — variables cliniques", aspect="auto",
    )
    fig_corr.update_layout(height=580)
    st.plotly_chart(fig_corr, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTAGE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    show_login()
else:
    show_sidebar()
    role = st.session_state.user_role

    if role == "medecin":
        page = st.session_state.medecin_page
        if page == "Gestion des patients":
            page_gestion_patients()
        elif page == "Prédiction":
            page_prediction_medecin()
        elif page == "Mon profil":
            page_profil_medecin()

    elif role == "admin":
        page = st.session_state.admin_page
        if page == "Tableau de bord":
            page_dashboard_admin()
        elif page == "Performance des modèles":
            page_performance_modeles()
        elif page == "Exploration des données":
            page_exploration_donnees()
