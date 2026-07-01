"""
Génère deux diagrammes UML haute résolution pour le projet
de prédiction de réadmission hospitalière :
  - uml/usecase.png  : Diagramme de cas d'utilisation
  - uml/sequence.png : Diagramme de séquence
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def draw_actor(ax, x, y, label, fontsize=10):
    """Stick-figure actor."""
    head_r = 0.18
    # head
    head = plt.Circle((x, y + 0.55 + head_r), head_r,
                       color="#2c3e50", zorder=4)
    ax.add_patch(head)
    # body
    ax.plot([x, x], [y + 0.55, y + 0.15], color="#2c3e50", lw=2, zorder=4)
    # arms
    ax.plot([x - 0.28, x + 0.28], [y + 0.42, y + 0.42],
            color="#2c3e50", lw=2, zorder=4)
    # legs
    ax.plot([x, x - 0.22], [y + 0.15, y - 0.15], color="#2c3e50", lw=2, zorder=4)
    ax.plot([x, x + 0.22], [y + 0.15, y - 0.15], color="#2c3e50", lw=2, zorder=4)
    # label
    ax.text(x, y - 0.30, label, ha="center", va="top",
            fontsize=fontsize, fontweight="bold", color="#2c3e50")


def draw_usecase(ax, cx, cy, w, h, text, fontsize=9, color="#d6eaf8"):
    """Ellipse use-case."""
    ell = Ellipse((cx, cy), w, h, facecolor=color,
                  edgecolor="#1a5276", linewidth=1.8, zorder=3)
    ax.add_patch(ell)
    # word-wrap: split on spaces if text is long
    words = text.split()
    lines, line = [], []
    for w_ in words:
        line.append(w_)
        if len(" ".join(line)) > 20:
            lines.append(" ".join(line[:-1]))
            line = [w_]
    lines.append(" ".join(line))
    ax.text(cx, cy, "\n".join(lines), ha="center", va="center",
            fontsize=fontsize, color="#1a5276",
            multialignment="center", zorder=4)


def draw_assoc(ax, x1, y1, x2, y2):
    """Simple association line."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-", color="#555555",
                                lw=1.4, connectionstyle="arc3,rad=0.0"),
                zorder=2)


# ─────────────────────────────────────────────────────────────
# 1. Diagramme de cas d'utilisation
# ─────────────────────────────────────────────────────────────

def draw_usecase_diagram():
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#fdfefe")

    # ── System boundary ──────────────────────────────────────
    sys_rect = FancyBboxPatch((2.6, 0.8), 12.8, 10.4,
                              boxstyle="round,pad=0.15",
                              facecolor="#eaf4fb", edgecolor="#1a5276",
                              linewidth=2.5, zorder=1)
    ax.add_patch(sys_rect)
    ax.text(9.0, 11.45, "Système de Prédiction de Réadmission Hospitalière",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="#1a5276")

    # ── Actors ───────────────────────────────────────────────
    draw_actor(ax, 1.25, 5.5, "Médecin", fontsize=11)
    draw_actor(ax, 16.75, 5.5, "Admin", fontsize=11)

    # ── Use-cases – Médecin (left column) ────────────────────
    medecin_ucs = [
        (6.0, 9.8,  "Saisir données\npatient"),
        (6.0, 7.8,  "Obtenir\nprédiction"),
        (6.0, 5.8,  "Voir niveau risque\nLow / Medium / High"),
        (6.0, 3.8,  "Voir explication\nSHAP"),
    ]
    for cx, cy, txt in medecin_ucs:
        draw_usecase(ax, cx, cy, 3.8, 1.2, txt, fontsize=9.5)
        draw_assoc(ax, 1.75, 6.28, cx - 1.9, cy)

    # ── Use-cases – Admin (right column) ─────────────────────
    admin_ucs = [
        (12.0, 9.8, "Consulter statistiques\ndataset"),
        (12.0, 7.4, "Comparer les 3\nmodèles ML"),
        (12.0, 5.0, "Voir métriques\nMLflow"),
    ]
    for cx, cy, txt in admin_ucs:
        draw_usecase(ax, cx, cy, 3.8, 1.2, txt, fontsize=9.5)
        draw_assoc(ax, 16.25, 6.28, cx + 1.9, cy)

    # ── <<include>> between Obtenir prédiction & sub-cases ───
    # Obtenir prédiction → Voir niveau risque
    ax.annotate("", xy=(6.0, 4.42), xytext=(6.0, 7.2),
                arrowprops=dict(arrowstyle="->", color="#7f8c8d",
                                lw=1.2, linestyle="dashed"))
    ax.text(6.35, 6.1, "«include»", fontsize=7.5, color="#7f8c8d",
            style="italic")

    # Obtenir prédiction → Voir explication SHAP
    ax.annotate("", xy=(6.0, 4.42), xytext=(6.0, 7.2),
                arrowprops=dict(arrowstyle="->", color="#7f8c8d",
                                lw=1.2, linestyle="dashed"))

    # draw explicit dashed lines
    ax.plot([6.0, 6.0], [7.2, 6.42], color="#7f8c8d",
            lw=1.2, linestyle="--", zorder=2)
    ax.plot([6.0, 6.0], [5.38, 4.42], color="#7f8c8d",
            lw=1.2, linestyle="--", zorder=2)
    ax.annotate("", xy=(6.0, 4.42), xytext=(6.0, 4.7),
                arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.2))
    ax.text(6.1, 4.9, "«include»", fontsize=7.5, color="#7f8c8d", style="italic")

    # ── Title ────────────────────────────────────────────────
    ax.text(9.0, 0.35, "Diagramme de Cas d'Utilisation — Prédiction de Réadmission",
            ha="center", va="center", fontsize=11, style="italic", color="#555555")

    # ── Legend ───────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor="#d6eaf8", edgecolor="#1a5276",
                       label="Cas d'utilisation"),
        mpatches.Patch(facecolor="#2c3e50", label="Acteur"),
        mpatches.Patch(facecolor="none", edgecolor="#7f8c8d",
                       linestyle="--", label="Relation «include»"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              fontsize=9, framealpha=0.9, edgecolor="#aaa")

    out = os.path.join(OUTPUT_DIR, "usecase.png")
    fig.savefig(out, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] usecase.png  ->  {out}")


# ─────────────────────────────────────────────────────────────
# 2. Diagramme de séquence
# ─────────────────────────────────────────────────────────────

def draw_sequence_diagram():
    # Participants
    participants = [
        "Médecin",
        "Dashboard\nStreamlit",
        "API\nFastAPI",
        "Modèle\nXGBoost",
        "SHAP\nExplainer",
    ]
    n = len(participants)

    fig_w, fig_h = 20, 14
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("#fdfefe")

    # X positions of lifelines (evenly spaced)
    xs = np.linspace(1.8, fig_w - 1.8, n)

    # ── Header boxes ─────────────────────────────────────────
    box_h = 0.80
    header_y = fig_h - 1.2
    colors = ["#d5e8d4", "#dae8fc", "#fff2cc", "#f8cecc", "#e1d5e7"]
    for i, (label, x, col) in enumerate(zip(participants, xs, colors)):
        bw = 2.2
        rect = FancyBboxPatch((x - bw / 2, header_y - box_h / 2), bw, box_h,
                              boxstyle="round,pad=0.12",
                              facecolor=col, edgecolor="#555", linewidth=1.6,
                              zorder=4)
        ax.add_patch(rect)
        ax.text(x, header_y, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#333", zorder=5)

    # ── Lifelines ─────────────────────────────────────────────
    lifeline_top = header_y - box_h / 2
    lifeline_bot = 0.9
    for x in xs:
        ax.plot([x, x], [lifeline_top, lifeline_bot],
                color="#aaa", lw=1.2, linestyle="--", zorder=1)

    # ── Activation boxes ─────────────────────────────────────
    # We'll draw these once we know the message positions
    act_w = 0.18

    def activation(ax_, x_, y_top, y_bot, col="#bbbbbb"):
        rect = FancyBboxPatch((x_ - act_w / 2, y_bot), act_w, y_top - y_bot,
                              boxstyle="square,pad=0",
                              facecolor=col, edgecolor="#666", linewidth=1,
                              zorder=3)
        ax_.add_patch(rect)

    # ── Messages ─────────────────────────────────────────────
    # Format: (from_idx, to_idx, label, is_return, y)
    msg_y_start = header_y - box_h / 2 - 0.55
    step = 1.35

    messages = [
        # (from, to, label,              is_return)
        (0, 1, "Saisir données patient\n(formulaire)",             False),
        (1, 2, "POST /predict\n{données patient JSON}",            False),
        (2, 2, "Prétraitement &\nencoding features",               False),   # self-msg
        (2, 3, "predict_proba(X_scaled)",                          False),
        (3, 2, "probabilité brute (0..1)",                         True),
        (2, 4, "shap_values(X)",                                   False),
        (4, 2, "valeurs SHAP +\nfeatures importantes",             True),
        (2, 2, "Calcul niveau risque\n(Low / Medium / High)",      False),   # self-msg
        (2, 1, "probabilité + niveau risque\n+ explication SHAP",  True),
        (1, 0, "Afficher résultat\n(jauge + niveau + SHAP)",       True),
    ]

    # Activation spans per participant index
    act_spans = {i: [] for i in range(n)}

    ys = []
    for k, (fi, ti, label, is_ret) in enumerate(messages):
        y = msg_y_start - k * step
        ys.append(y)

    # Record activation windows
    # Médecin: active throughout
    act_spans[0] = [(ys[0], ys[-1])]
    # Dashboard: from msg0 receive to last msg
    act_spans[1] = [(ys[1], ys[-1])]
    # FastAPI: from msg1 receive to msg8
    act_spans[2] = [(ys[2], ys[8])]
    # XGBoost: msg3 → msg4
    act_spans[3] = [(ys[3], ys[4])]
    # SHAP: msg5 → msg6
    act_spans[4] = [(ys[5], ys[6])]

    for idx, spans in act_spans.items():
        for (y_top, y_bot) in spans:
            activation(ax, xs[idx], y_top, y_bot,
                       col=colors[idx] if colors[idx] != "#fdfefe" else "#ddd")

    # Draw messages
    arrow_props_fwd = dict(arrowstyle="-|>", color="#1a5276",
                           lw=1.5, mutation_scale=14)
    arrow_props_ret = dict(arrowstyle="-|>", color="#7f8c8d",
                           lw=1.3, linestyle="dashed", mutation_scale=12)

    for k, (fi, ti, label, is_ret) in enumerate(messages):
        y = ys[k]
        props = arrow_props_ret if is_ret else arrow_props_fwd
        color = "#7f8c8d" if is_ret else "#1a5276"
        x_from = xs[fi]
        x_to   = xs[ti]

        if fi == ti:
            # Self-message: small loop on the right
            offset = 0.55
            ax.annotate("", xy=(x_from, y - 0.22),
                        xytext=(x_from, y),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.3, mutation_scale=12,
                                        connectionstyle=f"arc,angleA=-30,angleB=30,"
                                                        f"armA=50,armB=50,rad=0"))
            ax.plot([x_from, x_from + offset, x_from + offset, x_from],
                    [y, y, y - 0.22, y - 0.22],
                    color=color, lw=1.3, linestyle="--" if is_ret else "-")
            ax.annotate("", xy=(x_from, y - 0.22),
                        xytext=(x_from + offset, y - 0.22),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.3, mutation_scale=11))
            lines = label.split("\n")
            ax.text(x_from + offset + 0.12, y - 0.11, "\n".join(lines),
                    ha="left", va="center", fontsize=8.2, color=color,
                    style="italic" if is_ret else "normal")
        else:
            ax.annotate("", xy=(x_to, y), xytext=(x_from, y),
                        arrowprops=props, zorder=5)
            # label above arrow
            mid_x = (x_from + x_to) / 2
            lines = label.split("\n")
            va = "bottom"
            offset_y = 0.08
            if is_ret:
                style = "italic"
            else:
                style = "normal"
            ax.text(mid_x, y + offset_y, "\n".join(lines),
                    ha="center", va="bottom", fontsize=8.8,
                    color=color, style=style,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.75))

    # ── Step numbers ────────────────────────────────────────
    for k, y in enumerate(ys):
        ax.text(0.4, y, f"{k+1}.", ha="center", va="center",
                fontsize=9, color="#555", fontweight="bold")

    # ── Title ───────────────────────────────────────────────
    ax.text(fig_w / 2, fig_h - 0.38,
            "Diagramme de Séquence — Prédiction de Réadmission Hospitalière",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="#1a5276")

    # ── Legend ──────────────────────────────────────────────
    fwd_line = mpatches.Patch(facecolor="#1a5276",
                              label="Message (appel synchrone)")
    ret_line = mpatches.Patch(facecolor="#7f8c8d",
                              label="Message (retour / réponse)")
    ax.legend(handles=[fwd_line, ret_line], loc="lower right",
              fontsize=9, framealpha=0.9, edgecolor="#aaa")

    out = os.path.join(OUTPUT_DIR, "sequence.png")
    fig.savefig(out, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] sequence.png ->  {out}")


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    draw_usecase_diagram()
    draw_sequence_diagram()
    print("\nDiagrammes générés avec succès dans le dossier uml/")
