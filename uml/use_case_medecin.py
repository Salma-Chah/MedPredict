# -*- coding: utf-8 -*-
"""
Diagramme de cas d'utilisation -- Acteur : Medecin
(version mise a jour : authentification, recherche de patient par nom/prenom,
fiche patient, prediction de readmission, profil medecin)

Genere : uml/use_case_medecin.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from uml_common import (
    CLR_BG, CLR_UC_BLUE_F, CLR_UC_BLUE_E, CLR_UC_BLUE_TXT, CLR_ARROW_BLUE,
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
        "Diagramme de cas d'utilisation -- Acteur : Medecin",
        ha="center", va="center",
        fontsize=FONT_TITLE + 3, fontweight="bold", color=CLR_TITLE)
ax.text(FIG_W / 2, FIG_H - 0.95,
        "Plateforme de prediction du risque de readmission hospitaliere a 30 jours",
        ha="center", va="center",
        fontsize=12, color=CLR_SUBTITLE, style="italic")

# ── Frontiere du systeme ────────────────────────────────────────────────────
system_boundary(ax, 2.0, 0.55, 18.4, 11.7,
                 "Espace Medecin (acces authentifie)")

# ── Acteur ───────────────────────────────────────────────────────────────────
ACTOR_X, ACTOR_Y = 1.05, 6.4
stick_figure(ax, ACTOR_X, ACTOR_Y, "Medecin")

# ── Colonnes de cas d'utilisation ───────────────────────────────────────────
COL1_X = 7.6
COL2_X = 15.6
ROWS_Y = [10.3, 7.7, 5.1, 2.5]

uc_kwargs = dict(face=CLR_UC_BLUE_F, edge=CLR_UC_BLUE_E, txt_color=CLR_UC_BLUE_TXT)

# Colonne 1 -- acces et navigation
draw_uc(ax, COL1_X, ROWS_Y[0], "S'authentifier",
        "Connexion securisee par identifiant\net mot de passe (role Medecin)", **uc_kwargs)
draw_uc(ax, COL1_X, ROWS_Y[1], "Rechercher un patient",
        "Recherche instantanee dans la base\npar nom et/ou prenom", **uc_kwargs)
draw_uc(ax, COL1_X, ROWS_Y[2], "Consulter la fiche d'un patient",
        "Donnees demographiques, vitales\net cliniques completes", **uc_kwargs)
draw_uc(ax, COL1_X, ROWS_Y[3], "Consulter mon profil",
        "Identite, specialite et indicateurs\nde suivi personnel", **uc_kwargs)

# Colonne 2 -- prediction et session
draw_uc(ax, COL2_X, ROWS_Y[0], "Se deconnecter",
        "Cloture securisee de la session\nen cours", **uc_kwargs)
draw_uc(ax, COL2_X, ROWS_Y[1], "Obtenir une prediction\nde readmission",
        "Calcul de la probabilite de\nreadmission a 30 jours (modele ML)", **uc_kwargs)
draw_uc(ax, COL2_X, ROWS_Y[2], "Voir le niveau de risque",
        "Classification automatique\nFaible / Moyen / Eleve", **uc_kwargs)
draw_uc(ax, COL2_X, ROWS_Y[3], "Voir l'explication SHAP",
        "Variables ayant le plus\ninfluence la prediction", **uc_kwargs)

# ── Associations acteur -> cas d'utilisation ────────────────────────────────
assoc(ax, ACTOR_X, ACTOR_Y, COL1_X, ROWS_Y[0], CLR_ARROW_BLUE)   # S'authentifier
assoc(ax, ACTOR_X, ACTOR_Y, COL1_X, ROWS_Y[1], CLR_ARROW_BLUE)   # Rechercher un patient
assoc(ax, ACTOR_X, ACTOR_Y, COL1_X, ROWS_Y[2], CLR_ARROW_BLUE)   # Consulter la fiche
assoc(ax, ACTOR_X, ACTOR_Y, COL1_X, ROWS_Y[3], CLR_ARROW_BLUE)   # Consulter mon profil
assoc(ax, ACTOR_X, ACTOR_Y, COL2_X, ROWS_Y[0], CLR_ARROW_BLUE)   # Se deconnecter
assoc(ax, ACTOR_X, ACTOR_Y, COL2_X, ROWS_Y[1], CLR_ARROW_BLUE)   # Obtenir une prediction

# ── Relations <<include>> / <<extend>> ──────────────────────────────────────
edge_c1_right = COL1_X + UC_W / 2
edge_c2_left  = COL2_X - UC_W / 2
edge_c2_right = COL2_X + UC_W / 2

# "Obtenir une prediction" <<include>> "Consulter la fiche d'un patient"
relation_path(ax,
               x_from=edge_c2_left,  y_from=ROWS_Y[1],
               x_to=edge_c1_right,   y_to=ROWS_Y[2],
               via_x=11.6, label="<<include>>", color=CLR_INCLUDE, label_dx=0.15)

# "Voir le niveau de risque" <<extend>> "Obtenir une prediction"
relation_path(ax,
               x_from=edge_c2_right, y_from=ROWS_Y[2],
               x_to=edge_c2_right,   y_to=ROWS_Y[1] - 0.15,
               via_x=18.6, label="<<extend>>", color=CLR_EXTEND, label_dx=0.12)

# "Voir l'explication SHAP" <<include>> "Obtenir une prediction"
relation_path(ax,
               x_from=edge_c2_right, y_from=ROWS_Y[3],
               x_to=edge_c2_right,   y_to=ROWS_Y[1] + 0.15,
               via_x=19.35, label="<<include>>", color=CLR_INCLUDE, label_dx=0.12)

# ── Legende ──────────────────────────────────────────────────────────────────
legend_elems = [
    Line2D([0], [0], color=CLR_ARROW_BLUE, lw=2, marker=">", markersize=6,
           label="Association (acteur -> cas d'utilisation)"),
    Line2D([0], [0], color=CLR_INCLUDE, lw=1.8, linestyle="--",
           label="<<include>> -- toujours execute"),
    Line2D([0], [0], color=CLR_EXTEND, lw=1.8, linestyle="--",
           label="<<extend>> -- execute sous condition"),
    mpatches.Patch(facecolor=CLR_UC_BLUE_F, edgecolor=CLR_UC_BLUE_E,
                   label="Cas d'utilisation (Medecin)"),
]
ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.012),
          ncol=4, fontsize=10.5, framealpha=0.93, edgecolor="#BBBBBB",
          handlelength=2.4)

plt.tight_layout()
plt.savefig("use_case_medecin.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print("OK -> use_case_medecin.png")
