# -*- coding: utf-8 -*-
"""
Diagramme de classes -- Plateforme de prediction du risque de readmission
hospitaliere a 30 jours (version mise a jour : authentification, profils
Medecin / Administrateur, recherche patient, prediction + explication SHAP).

Disposition centree en "diamant" autour de l'entite Prediction, avec une
note explicative resumant l'organisation generale.

Genere : uml/class_diagram.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D

from uml_common import CLR_BG, CLR_TITLE, CLR_SUBTITLE, FONT_TITLE

FIG_W, FIG_H = 21, 14.6

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
ax.set_facecolor(CLR_BG)
fig.patch.set_facecolor(CLR_BG)

LINE_H   = 0.32
HEADER_H = 0.62


def draw_class(ax, x, y, w, name, attrs, methods, header_color,
               txt_color="white", stereotype=None):
    """Dessine une classe UML (compartiments nom / attributs / methodes)."""
    attrs_h   = max(len(attrs), 1) * LINE_H + 0.16
    methods_h = max(len(methods), 1) * LINE_H + 0.16
    h = HEADER_H + attrs_h + methods_h

    ax.add_patch(Rectangle((x, y), w, h, facecolor="white",
                            edgecolor="#333333", lw=1.4, zorder=3))
    ax.add_patch(Rectangle((x, y + attrs_h + methods_h), w, HEADER_H,
                            facecolor=header_color, edgecolor="#333333",
                            lw=1.4, zorder=4))

    title_y = y + attrs_h + methods_h + HEADER_H / 2
    if stereotype:
        ax.text(x + w / 2, title_y + 0.14, stereotype, ha="center", va="center",
                fontsize=9, style="italic", color=txt_color, zorder=5)
        ax.text(x + w / 2, title_y - 0.13, name, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=txt_color, zorder=5)
    else:
        ax.text(x + w / 2, title_y, name, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=txt_color, zorder=5)

    ax.plot([x, x + w], [y + methods_h + attrs_h, y + methods_h + attrs_h],
            color="#333333", lw=1.0, zorder=4)
    ax.plot([x, x + w], [y + methods_h, y + methods_h],
            color="#333333", lw=1.0, zorder=4)

    for i, a in enumerate(attrs):
        ay = y + methods_h + attrs_h - 0.21 - i * LINE_H
        ax.text(x + 0.18, ay, a, ha="left", va="center",
                fontsize=9.2, family="monospace", color="#1A1A1A", zorder=5)
    for i, m in enumerate(methods):
        my = y + methods_h - 0.21 - i * LINE_H
        ax.text(x + 0.18, my, m, ha="left", va="center",
                fontsize=9.2, family="monospace", color="#1A1A1A", zorder=5)

    return (x, y, w, h)


def anchor(box, where):
    x, y, w, h = box
    return {
        "top":         (x + w / 2, y + h),
        "bottom":      (x + w / 2, y),
        "left":        (x,         y + h / 2),
        "right":       (x + w,     y + h / 2),
        "topleft":     (x,         y + h),
        "topright":    (x + w,     y + h),
        "bottomleft":  (x,         y),
        "bottomright": (x + w,     y),
    }[where]


def inheritance(ax, p_from, p_to, color="#333333"):
    """Fleche de generalisation (triangle creux) de la sous-classe vers la
    super-classe."""
    ax.annotate("", xy=p_to, xytext=p_from,
                 arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                                 mutation_scale=24, fc="white"),
                 zorder=2)


def association(ax, p1, p2, label, m1="", m2="", color="#555555"):
    """Association avec multiplicites aux deux extremites et nom au centre."""
    x1, y1 = p1
    x2, y2 = p2
    ax.plot([x1, x2], [y1, y2], color=color, lw=1.4, zorder=2)

    dx, dy = x2 - x1, y2 - y1
    dist = (dx ** 2 + dy ** 2) ** 0.5
    ux, uy = dx / dist, dy / dist
    off = 0.45

    if m1:
        ax.text(x1 + ux * off, y1 + uy * off, m1, ha="center", va="center",
                fontsize=9.5, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none",
                          alpha=0.85), zorder=6)
    if m2:
        ax.text(x2 - ux * off, y2 - uy * off, m2, ha="center", va="center",
                fontsize=9.5, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none",
                          alpha=0.85), zorder=6)

    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mx, my, label, ha="center", va="center",
            fontsize=10, fontweight="bold", style="italic", color=CLR_TITLE,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#BBBBBB",
                      alpha=0.93, lw=1.0), zorder=7)


def draw_note(ax, x, y, w, h, text, fold=0.5):
    """Note UML (rectangle a coin replie) avec texte explicatif."""
    pts = [(x, y), (x + w, y), (x + w, y + h - fold),
           (x + w - fold, y + h), (x, y + h)]
    ax.add_patch(Polygon(pts, closed=True, facecolor="#FCF3CF",
                          edgecolor="#B7950B", lw=1.3, zorder=3))
    ax.plot([x + w - fold, x + w - fold], [y + h - fold, y + h],
            color="#B7950B", lw=1.0, zorder=4)
    ax.plot([x + w - fold, x + w], [y + h - fold, y + h - fold],
            color="#B7950B", lw=1.0, zorder=4)
    ax.text(x + w / 2, y + h / 2 - fold / 4, text,
            ha="center", va="center", fontsize=9.3, color="#7D6608",
            style="italic", multialignment="left", zorder=5)


# ── Titre general ────────────────────────────────────────────────────────────
ax.text(FIG_W / 2, FIG_H - 0.40,
        "Diagramme de classes -- Plateforme de prediction de readmission hospitaliere",
        ha="center", va="center",
        fontsize=FONT_TITLE + 3, fontweight="bold", color=CLR_TITLE)
ax.text(FIG_W / 2, FIG_H - 0.85,
        "Modele conceptuel centre sur l'entite Prediction"
        " (utilisateurs, patient, modele ML, explication SHAP)",
        ha="center", va="center", fontsize=12, color=CLR_SUBTITLE, style="italic")

# ── Classes (disposition en "diamant" : Utilisateur en tete, Prediction au
#    centre, Patient / ModeleML de part et d'autre, ExplicationSHAP dessous) ──
utilisateur = draw_class(
    ax, 7.3, 9.5, 5.4,
    "Utilisateur",
    ["- idUtilisateur : str", "- nom : str", "- motDePasse : str",
     "- role : str", "- email : str"],
    ["+ seConnecter(login, motDePasse) : bool",
     "+ seDeconnecter() : void"],
    header_color="#34495E", stereotype="<<abstract>>",
)

medecin = draw_class(
    ax, 1.0, 6.5, 6.0,
    "Medecin",
    ["- specialite : str"],
    ["+ rechercherPatient(nom, prenom) : List<Patient>",
     "+ consulterFiche(patientId) : Patient",
     "+ demanderPrediction(patient) : Prediction",
     "+ consulterProfil() : void"],
    header_color="#1A5276",
)

administrateur = draw_class(
    ax, 14.2, 6.7, 6.0,
    "Administrateur",
    [],
    ["+ consulterTableauDeBord() : void",
     "+ comparerModeles() : List<ModeleML>",
     "+ explorerDonnees() : void"],
    header_color="#1E8449",
)

prediction = draw_class(
    ax, 7.3, 3.5, 5.4,
    "Prediction",
    ["- probabilite : float", "- niveauRisque : str", "- dateCalcul : datetime"],
    ["+ calculerNiveauRisque(probabilite) : str"],
    header_color="#6C3483",
)

patient = draw_class(
    ax, 1.0, 1.04, 6.0,
    "Patient",
    ["- patientId : int", "- nom : str", "- prenom : str", "- age : int",
     "- genre : str", "- bmi : float", "- cholesterol : float",
     "- tensionArterielle : str", "- diabete : bool", "- hypertension : bool"],
    ["+ getDonneesCliniques() : dict"],
    header_color="#B9770E",
)

modele_ml = draw_class(
    ax, 14.2, 2.2, 6.0,
    "ModeleML",
    ["- nom : str", "- version : str", "- aucRoc : float", "- f1Score : float",
     "- precision : float", "- rappel : float"],
    ["+ predict(donnees) : float", "+ entrainer(X, y) : void"],
    header_color="#117A65",
)

explication_shap = draw_class(
    ax, 7.3, 0.6, 5.4,
    "ExplicationSHAP",
    ["- valeursShap : list", "- featuresImportantes : list"],
    ["+ genererWaterfallPlot() : Figure"],
    header_color="#922B21",
)

# ── Generalisations (heritage) -- Medecin et Administrateur heritent
#    des proprietes communes de Utilisateur ────────────────────────────────
inheritance(ax, anchor(medecin, "top"), (8.8, 9.5))
inheritance(ax, anchor(administrateur, "top"), (11.2, 9.5))

# ── Associations -- toutes centrees autour de Prediction ──────────────────
association(ax, anchor(medecin, "bottom"), anchor(prediction, "topleft"),
             "effectue", m1="1", m2="0..*")

association(ax, anchor(patient, "topright"), anchor(prediction, "left"),
             "concerne", m1="1", m2="0..*")

association(ax, anchor(prediction, "right"), anchor(modele_ml, "left"),
             "utilise", m1="0..*", m2="1")

association(ax, anchor(prediction, "bottom"), anchor(explication_shap, "top"),
             "genere", m1="1", m2="0..1")

association(ax, anchor(administrateur, "bottom"), anchor(modele_ml, "top"),
             "evalue", m1="1", m2="*")

# ── Note explicative ─────────────────────────────────────────────────────────
note_box = (14.2, 9.7, 6.0, 2.6)
draw_note(ax, *note_box,
          "Utilisateur (abstrait) regroupe\n"
          "l'authentification commune a\n"
          "Medecin et Administrateur.\n"
          "Prediction est l'entite centrale :\n"
          "elle relie Patient, ModeleML\n"
          "et ExplicationSHAP.")
ax.plot([note_box[0], anchor(utilisateur, "right")[0]],
        [note_box[1] + note_box[3] / 2, anchor(utilisateur, "right")[1]],
        color="#B7950B", lw=1.0, linestyle="--", zorder=2)

# ── Legende ──────────────────────────────────────────────────────────────────
legend_elems = [
    Line2D([0], [0], color="#333333", lw=1.6, marker=">", markersize=8,
           markerfacecolor="white", label="Generalisation (heritage)"),
    Line2D([0], [0], color="#555555", lw=1.4,
           label="Association (avec multiplicites et role)"),
    mpatches.Patch(facecolor="white", edgecolor="#333333",
                   label="Classe : nom / attributs / methodes"),
    mpatches.Patch(facecolor="#FCF3CF", edgecolor="#B7950B",
                   label="Note explicative (UML)"),
]
ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.012),
          ncol=4, fontsize=10.5, framealpha=0.93, edgecolor="#BBBBBB",
          handlelength=2.6)

plt.tight_layout()
plt.savefig("class_diagram.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print("OK -> class_diagram.png")
