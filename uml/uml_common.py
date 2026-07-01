# -*- coding: utf-8 -*-
"""
Primitives de dessin partagees pour les diagrammes UML du projet
(palette de couleurs + helpers matplotlib reutilises par les scripts
use_case_medecin.py, use_case_admin.py, class_diagram.py, sequence_medecin.py).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
from matplotlib.lines import Line2D

# ── Palette ──────────────────────────────────────────────────────────────────
CLR_BG          = "#FAFBFC"
CLR_SYS_FACE    = "#EEF4FB"
CLR_SYS_EDGE    = "#2E4057"

CLR_UC_BLUE_F   = "#D6EAF8"
CLR_UC_BLUE_E   = "#1A5276"
CLR_UC_BLUE_TXT = "#154360"
CLR_ARROW_BLUE  = "#1A5276"

CLR_UC_GRN_F    = "#D5F5E3"
CLR_UC_GRN_E    = "#1E8449"
CLR_UC_GRN_TXT  = "#145A32"
CLR_ARROW_GRN   = "#1E8449"

CLR_UC_GOLD_F   = "#FDEBD0"
CLR_UC_GOLD_E   = "#B9770E"
CLR_UC_GOLD_TXT = "#7E5109"

CLR_ACTOR       = "#2C3E50"
CLR_TITLE       = "#2E4057"
CLR_SUBTITLE    = "#666666"
CLR_NOTE        = "#666666"
CLR_INCLUDE     = "#6C3483"   # violet — <<include>>
CLR_EXTEND      = "#B7770D"   # ambre  — <<extend>>

# ── Dimensions par defaut ──────────────────────────────────────────────────────
UC_W       = 4.60
UC_H       = 0.85
FONT_TITLE = 16
FONT_ACTOR = 14
FONT_UC    = 13
FONT_NOTE  = 11


# ── Primitives ────────────────────────────────────────────────────────────────

def stick_figure(ax, cx, cy, label, fontsize=FONT_ACTOR):
    """Acteur UML (bonhomme baton) avec etiquette en dessous."""
    r        = 0.26
    body_top = cy + 0.62
    body_bot = cy + 0.12
    arm_y    = cy + 0.44
    leg_bot  = cy - 0.30

    kw = dict(color=CLR_ACTOR, lw=2.5, solid_capstyle="round", zorder=5)
    ax.add_patch(plt.Circle((cx, body_top + r), r,
                             color=CLR_ACTOR, fill=False, lw=2.5, zorder=5))
    ax.plot([cx, cx],               [body_top, body_bot], **kw)
    ax.plot([cx - 0.38, cx + 0.38], [arm_y, arm_y],       **kw)
    ax.plot([cx, cx - 0.30],        [body_bot, leg_bot],  **kw)
    ax.plot([cx, cx + 0.30],        [body_bot, leg_bot],  **kw)
    ax.text(cx, cy - 0.44, label,
            ha="center", va="top",
            fontsize=fontsize, fontweight="bold", color=CLR_ACTOR, zorder=5)


def draw_uc(ax, cx, cy, title, note, face, edge, txt_color, w=UC_W, h=UC_H):
    """Ellipse de cas d'utilisation : titre en gras + note en italique dessous."""
    ax.add_patch(Ellipse((cx, cy), w, h,
                          facecolor=face, edgecolor=edge, linewidth=2.2, zorder=3))
    ax.text(cx, cy, title,
            ha="center", va="center",
            fontsize=FONT_UC, fontweight="bold", color=txt_color,
            multialignment="center", zorder=4)
    if note:
        ax.text(cx, cy - h / 2 - 0.26, note,
                ha="center", va="top",
                fontsize=FONT_NOTE, color=CLR_NOTE, style="italic",
                multialignment="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.14", fc="#F4F6F7",
                          ec="#D5D8DC", alpha=0.80, lw=0.9))


def assoc(ax, x_actor, y_actor, cx_uc, cy_uc, color, w=UC_W):
    """Fleche d'association (trait plein) de l'acteur vers le bord de l'ellipse."""
    is_left   = x_actor < cx_uc
    x_uc_edge = cx_uc - w / 2 if is_left else cx_uc + w / 2
    ax.annotate("",
                 xy=(x_uc_edge, cy_uc), xytext=(x_actor, y_actor),
                 arrowprops=dict(arrowstyle="-|>", color=color,
                                 lw=1.8, mutation_scale=16,
                                 connectionstyle="arc3,rad=0.0"),
                 zorder=2)


def relation_path(ax, x_from, y_from, x_to, y_to, via_x, label, color, label_dx=0.12):
    """
    Relation <<include>>/<<extend>> en pointilles : trajet en equerre
    (horizontal -> vertical -> horizontal) passant par `via_x`,
    de (x_from, y_from) vers (x_to, y_to). La fleche pointe vers le point cible.
    """
    ax.plot([x_from, via_x], [y_from, y_from],
            color=color, lw=1.6, linestyle="--", zorder=2)
    ax.plot([via_x, via_x], [y_from, y_to],
            color=color, lw=1.6, linestyle="--", zorder=2)
    ax.plot([via_x, x_to], [y_to, y_to],
            color=color, lw=1.6, linestyle="--", zorder=2)
    ax.annotate("",
                 xy=(x_to, y_to), xytext=(via_x, y_to),
                 arrowprops=dict(arrowstyle="-|>", color=color,
                                 lw=1.6, mutation_scale=14),
                 zorder=3)
    mid_y = (y_from + y_to) / 2
    ha = "left" if label_dx > 0 else "right"
    ax.text(via_x + label_dx, mid_y, label,
            ha=ha, va="center",
            fontsize=10, color=color, style="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.14", fc="white",
                      ec=color, alpha=0.90, lw=1.0),
            zorder=6)


def build_legend(ax, elems, ncol=4, loc="lower center", bbox=(0.5, 0.002)):
    ax.legend(handles=elems, loc=loc, bbox_to_anchor=bbox,
              ncol=ncol, fontsize=10.5,
              framealpha=0.93, edgecolor="#BBBBBB", handlelength=2.2)


def footer(ax, fig_w, text):
    ax.text(fig_w / 2, 0.22, text,
            ha="center", va="center",
            fontsize=10, style="italic", color=CLR_SUBTITLE)


def system_boundary(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.20",
        facecolor=CLR_SYS_FACE, edgecolor=CLR_SYS_EDGE,
        linewidth=3, zorder=1))
    title_y = y + h - 0.56
    ax.text(x + w / 2, title_y, title,
            ha="center", va="center",
            fontsize=FONT_TITLE, fontweight="bold", color=CLR_TITLE, zorder=4)
    sep_y = y + h - 1.12
    ax.plot([x + 0.5, x + w - 0.5], [sep_y, sep_y],
            color=CLR_SYS_EDGE, lw=1.2, linestyle="--", alpha=0.40, zorder=2)
    return sep_y
