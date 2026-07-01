# -*- coding: utf-8 -*-
"""
Diagramme de cas d'utilisation -- Acteur : Administrateur
(version mise a jour : authentification, tableau de bord, performance des
modeles ML, exploration des donnees)

Genere : uml/use_case_admin.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from uml_common import (
    CLR_BG, CLR_UC_GRN_F, CLR_UC_GRN_E, CLR_UC_GRN_TXT, CLR_ARROW_GRN,
    CLR_ACTOR, CLR_TITLE, CLR_SUBTITLE, CLR_INCLUDE, CLR_EXTEND,
    UC_W, UC_H, FONT_TITLE,
    stick_figure, draw_uc, assoc, relation_path, system_boundary, footer,
)

FIG_W, FIG_H = 21, 14

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
ax.set_facecolor(CLR_BG)
fig.patch.set_facecolor(CLR_BG)

# ── Titre general ───────────────────────────────────────────────────────────
ax.text(FIG_W / 2, FIG_H - 0.45,
        "Diagramme de cas d'utilisation -- Acteur : Administrateur",
        ha="center", va="center",
        fontsize=FONT_TITLE + 3, fontweight="bold", color=CLR_TITLE)
ax.text(FIG_W / 2, FIG_H - 0.95,
        "Plateforme de prediction du risque de readmission hospitaliere a 30 jours",
        ha="center", va="center",
        fontsize=12, color=CLR_SUBTITLE, style="italic")

# ── Frontiere du systeme ────────────────────────────────────────────────────
system_boundary(ax, 2.0, 0.55, 18.4, 11.7,
                 "Espace Administrateur (acces authentifie)")

# ── Acteur ───────────────────────────────────────────────────────────────────
ACTOR_X, ACTOR_Y = 1.05, 6.4
stick_figure(ax, ACTOR_X, ACTOR_Y, "Administrateur")

# ── Colonnes de cas d'utilisation ───────────────────────────────────────────
COL1_X = 7.6
COL2_X = 15.6
ROWS_Y = [10.3, 7.7, 5.1, 2.5]

uc_kwargs = dict(face=CLR_UC_GRN_F, edge=CLR_UC_GRN_E, txt_color=CLR_UC_GRN_TXT)

# Colonne 1 -- acces, navigation et donnees
draw_uc(ax, COL1_X, ROWS_Y[0], "S'authentifier",
        "Connexion securisee par identifiant\net mot de passe (role Administrateur)", **uc_kwargs)
draw_uc(ax, COL1_X, ROWS_Y[1], "Consulter le tableau de bord",
        "Vue globale : indicateurs cles,\nrepartition des risques sur 30 000 patients", **uc_kwargs)
draw_uc(ax, COL1_X, ROWS_Y[2], "Explorer les donnees du dataset",
        "Distributions, correlations et\nstatistiques descriptives", **uc_kwargs)
draw_uc(ax, COL1_X, ROWS_Y[3], "Se deconnecter",
        "Cloture securisee de la session\nen cours", **uc_kwargs)

# Colonne 2 -- performance des modeles
draw_uc(ax, COL2_X, ROWS_Y[0], "Comparer les performances\ndes modeles ML",
        "Regression Logistique, Random Forest\net XGBoost", **uc_kwargs)
draw_uc(ax, COL2_X, ROWS_Y[1], "Consulter les metriques\nde suivi (MLflow)",
        "AUC-ROC, F1-score, Precision,\nRappel pour chaque modele", **uc_kwargs)
draw_uc(ax, COL2_X, ROWS_Y[2], "Voir l'importance\ndes variables",
        "Variables les plus determinantes\ndu modele XGBoost retenu", **uc_kwargs)

# ── Associations acteur -> cas d'utilisation ────────────────────────────────
assoc(ax, ACTOR_X, ACTOR_Y, COL1_X, ROWS_Y[0], CLR_ARROW_GRN)   # S'authentifier
assoc(ax, ACTOR_X, ACTOR_Y, COL1_X, ROWS_Y[1], CLR_ARROW_GRN)   # Tableau de bord
assoc(ax, ACTOR_X, ACTOR_Y, COL1_X, ROWS_Y[2], CLR_ARROW_GRN)   # Explorer les donnees
assoc(ax, ACTOR_X, ACTOR_Y, COL1_X, ROWS_Y[3], CLR_ARROW_GRN)   # Se deconnecter
assoc(ax, ACTOR_X, ACTOR_Y, COL2_X, ROWS_Y[0], CLR_ARROW_GRN)   # Comparer les performances

# ── Relations <<include>> / <<extend>> ──────────────────────────────────────
edge_c2_right = COL2_X + UC_W / 2

# "Comparer les performances des modeles ML" <<include>> "Consulter les metriques (MLflow)"
relation_path(ax,
               x_from=edge_c2_right, y_from=ROWS_Y[0],
               x_to=edge_c2_right,   y_to=ROWS_Y[1],
               via_x=18.6, label="<<include>>", color=CLR_INCLUDE, label_dx=0.12)

# "Voir l'importance des variables" <<extend>> "Comparer les performances des modeles ML"
relation_path(ax,
               x_from=edge_c2_right, y_from=ROWS_Y[2],
               x_to=edge_c2_right,   y_to=ROWS_Y[0] + 0.15,
               via_x=19.35, label="<<extend>>", color=CLR_EXTEND, label_dx=0.12)

# ── Legende ──────────────────────────────────────────────────────────────────
legend_elems = [
    Line2D([0], [0], color=CLR_ARROW_GRN, lw=2, marker=">", markersize=6,
           label="Association (acteur -> cas d'utilisation)"),
    Line2D([0], [0], color=CLR_INCLUDE, lw=1.8, linestyle="--",
           label="<<include>> -- toujours execute"),
    Line2D([0], [0], color=CLR_EXTEND, lw=1.8, linestyle="--",
           label="<<extend>> -- execute sous condition"),
    mpatches.Patch(facecolor=CLR_UC_GRN_F, edgecolor=CLR_UC_GRN_E,
                   label="Cas d'utilisation (Administrateur)"),
]
ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.012),
          ncol=4, fontsize=10.5, framealpha=0.93, edgecolor="#BBBBBB",
          handlelength=2.4)

plt.tight_layout()
plt.savefig("use_case_admin.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print("OK -> use_case_admin.png")
