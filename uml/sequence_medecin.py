# -*- coding: utf-8 -*-
"""
Diagramme de sequence -- Parcours Medecin (version mise a jour)
Connexion -> recherche d'un patient par nom/prenom -> consultation de la
fiche patient (pre-remplie) -> prediction de readmission + explication SHAP.

Genere : uml/sequence_medecin.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Participants ─────────────────────────────────────────────────────────────
participants = [
    "Medecin",
    "Dashboard\nStreamlit",
    "Base de\npatients (CSV)",
    "Modele\nXGBoost",
    "SHAP\nExplainer",
]
n = len(participants)

fig_w, fig_h = 20, 20
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")
fig.patch.set_facecolor("#fdfefe")

xs = np.linspace(1.8, fig_w - 1.8, n)

# ── Header boxes ────────────────────────────────────────────────────────────
box_h = 0.80
header_y = fig_h - 1.2
colors = ["#d5e8d4", "#dae8fc", "#fff2cc", "#f8cecc", "#e1d5e7"]
for label, x, col in zip(participants, xs, colors):
    bw = 2.4
    rect = FancyBboxPatch((x - bw / 2, header_y - box_h / 2), bw, box_h,
                          boxstyle="round,pad=0.12",
                          facecolor=col, edgecolor="#555", linewidth=1.6,
                          zorder=4)
    ax.add_patch(rect)
    ax.text(x, header_y, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#333", zorder=5)

# ── Lifelines ───────────────────────────────────────────────────────────────
lifeline_top = header_y - box_h / 2
lifeline_bot = 0.9
for x in xs:
    ax.plot([x, x], [lifeline_top, lifeline_bot],
            color="#aaa", lw=1.2, linestyle="--", zorder=1)

# ── Activation boxes ─────────────────────────────────────────────────────────
act_w = 0.18


def activation(ax_, x_, y_top, y_bot, col="#bbbbbb"):
    rect = FancyBboxPatch((x_ - act_w / 2, y_bot), act_w, y_top - y_bot,
                          boxstyle="square,pad=0",
                          facecolor=col, edgecolor="#666", linewidth=1,
                          zorder=3)
    ax_.add_patch(rect)


# ── Messages ────────────────────────────────────────────────────────────────
# Format : (from_idx, to_idx, label, is_return)
msg_y_start = header_y - box_h / 2 - 0.55
step = 1.10

messages = [
    (0, 1, "Saisir identifiant\net mot de passe",                       False),
    (1, 1, "Verifier les identifiants\n(dictionnaire USERS)",           False),  # self-msg
    (1, 0, "Acces autorise :\nafficher Espace Medecin",                 True),
    (0, 1, "Rechercher un patient\n(nom / prenom)",                     False),
    (1, 2, "Filtrer les patients\npar nom / prenom",                    False),
    (2, 1, "Liste des patients\ncorrespondants",                        True),
    (1, 0, "Afficher resultats +\nfiche patient pre-remplie",           True),
    (0, 1, "Cliquer \"Predire le risque\nde readmission\"",             False),
    (1, 1, "Pretraitement &\nencodage des features",                    False),  # self-msg
    (1, 3, "predict_proba(X_scaled)",                                   False),
    (3, 1, "probabilite brute (0..1)",                                  True),
    (1, 4, "shap_values(X)",                                            False),
    (4, 1, "valeurs SHAP +\nfeatures importantes",                      True),
    (1, 1, "Calcul du niveau de risque\n(Faible / Moyen / Eleve)",      False),  # self-msg
    (1, 0, "Afficher jauge + niveau\nde risque + explication SHAP",    True),
]

ys = [msg_y_start - k * step for k in range(len(messages))]

# Activation spans per participant index
act_spans = {i: [] for i in range(n)}
act_spans[0] = [(ys[0], ys[-1])]            # Medecin : actif tout le long
act_spans[1] = [(ys[0], ys[-1])]            # Dashboard : orchestrateur central
act_spans[2] = [(ys[4], ys[5])]             # Base de patients : recherche
act_spans[3] = [(ys[9], ys[10])]            # Modele XGBoost : prediction
act_spans[4] = [(ys[11], ys[12])]           # SHAP Explainer : explication

for idx, spans in act_spans.items():
    for (y_top, y_bot) in spans:
        activation(ax, xs[idx], y_top, y_bot,
                   col=colors[idx] if colors[idx] != "#fdfefe" else "#ddd")

# ── Dessin des messages ───────────────────────────────────────────────────────
arrow_props_fwd = dict(arrowstyle="-|>", color="#1a5276",
                       lw=1.5, mutation_scale=14)
arrow_props_ret = dict(arrowstyle="-|>", color="#7f8c8d",
                       lw=1.3, linestyle="dashed", mutation_scale=12)

for k, (fi, ti, label, is_ret) in enumerate(messages):
    y = ys[k]
    props = arrow_props_ret if is_ret else arrow_props_fwd
    color = "#7f8c8d" if is_ret else "#1a5276"
    x_from = xs[fi]
    x_to = xs[ti]

    if fi == ti:
        # Message reflexif : petite boucle a droite de la ligne de vie
        offset = 0.55
        ax.annotate("", xy=(x_from, y - 0.22), xytext=(x_from, y),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=1.3, mutation_scale=12,
                                    connectionstyle="arc,angleA=-30,angleB=30,"
                                                    "armA=50,armB=50,rad=0"))
        ax.plot([x_from, x_from + offset, x_from + offset, x_from],
                [y, y, y - 0.22, y - 0.22],
                color=color, lw=1.3, linestyle="--" if is_ret else "-")
        ax.annotate("", xy=(x_from, y - 0.22), xytext=(x_from + offset, y - 0.22),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=1.3, mutation_scale=11))
        lines = label.split("\n")
        ax.text(x_from + offset + 0.12, y - 0.11, "\n".join(lines),
                ha="left", va="center", fontsize=8.2, color=color,
                style="italic" if is_ret else "normal")
    else:
        ax.annotate("", xy=(x_to, y), xytext=(x_from, y),
                    arrowprops=props, zorder=5)
        mid_x = (x_from + x_to) / 2
        lines = label.split("\n")
        style = "italic" if is_ret else "normal"
        ax.text(mid_x, y + 0.08, "\n".join(lines),
                ha="center", va="bottom", fontsize=8.8,
                color=color, style=style,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.75))

# ── Numerotation des etapes ────────────────────────────────────────────────
for k, y in enumerate(ys):
    ax.text(0.4, y, f"{k + 1}.", ha="center", va="center",
            fontsize=9, color="#555", fontweight="bold")

# ── Titre ────────────────────────────────────────────────────────────────────
ax.text(fig_w / 2, fig_h - 0.38,
        "Diagramme de sequence -- Parcours Medecin "
        "(connexion, recherche patient, prediction et explication SHAP)",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#1a5276")

# ── Legende ──────────────────────────────────────────────────────────────────
fwd_line = mpatches.Patch(facecolor="#1a5276", label="Message (appel synchrone)")
ret_line = mpatches.Patch(facecolor="#7f8c8d", label="Message (retour / reponse)")
ax.legend(handles=[fwd_line, ret_line], loc="lower right",
          fontsize=9, framealpha=0.9, edgecolor="#aaa")

fig.savefig("sequence_medecin.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print("OK -> sequence_medecin.png")
