# -*- coding: utf-8 -*-
"""
Génère rapport_technique.pdf depuis rapport_technique.md
Dépendance : fpdf2  (pip install fpdf2)
"""

import re
import os
from fpdf import FPDF

MD_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rapport_technique.md")
PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rapport_technique.pdf")

# ── Palette couleurs ──────────────────────────────────────────────────────────
C_BLUE_DARK  = (30,  64,  87)   # titres principaux
C_BLUE_MID   = (26,  82, 118)   # titres secondaires
C_BLUE_LIGHT = (214,234,248)    # fond en-têtes tableaux
C_GREEN_DARK = (30, 132, 73)    # titre H3
C_GREEN_LIGHT= (213,245,227)    # fond lignes paires tableau
C_CODE_BG    = (245,245,245)    # fond bloc code
C_CODE_BORDER= (180,180,180)
C_GRAY_TEXT  = (90,  90,  90)   # texte secondaire
C_BLACK      = (30,  30,  30)   # corps du texte
C_WHITE      = (255,255,255)
C_SEP        = (180,200,220)    # filet de séparation


# Caractères Unicode non supportés par les polices de base (latin-1)
UNICODE_REPLACEMENTS = {
    "–": "-",    # – (en dash)
    "—": "-",    # — (em dash)
    "→": "->",   # → (flèche)
    "∈": " dans ",  # ∈
    "─": "-",    # ─ (box drawing horizontal)
    "│": "|",    # │ (box drawing vertical)
    "└": "`",    # └ (box drawing up-right)
    "├": "|",    # ├ (box drawing vertical-right)
}


def sanitize_unicode(text: str) -> str:
    """Remplace les caractères Unicode non supportés par des équivalents ASCII."""
    for src, dst in UNICODE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text


def hex_strip(text: str) -> str:
    """Retire les balises markdown inline (* ** ` ~~)."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'`(.+?)`',       r'\1', text)
    return sanitize_unicode(text)


class RapportPDF(FPDF):

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(20, 20, 20)

    # ── En-tête de page ───────────────────────────────────────────────────────
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", style="I", size=8)
        self.set_text_color(*C_GRAY_TEXT)
        self.cell(0, 6, sanitize_unicode("Rapport Technique — Prédiction de Réadmission Hospitalière"),
                  align="L")
        self.set_font("Helvetica", style="I", size=8)
        self.cell(0, 6, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*C_SEP)
        self.set_line_width(0.3)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(3)

    # ── Pied de page ──────────────────────────────────────────────────────────
    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-14)
        self.set_draw_color(*C_SEP)
        self.line(20, self.get_y(), 190, self.get_y())
        self.set_font("Helvetica", style="I", size=7.5)
        self.set_text_color(*C_GRAY_TEXT)
        self.cell(0, 8, "PFA 2025-2026", align="C")

    # ── Page de garde ─────────────────────────────────────────────────────────
    def page_de_garde(self):
        self.add_page()

        # Bande supérieure
        self.set_fill_color(*C_BLUE_DARK)
        self.rect(0, 0, 210, 55, style="F")

        # Sous-bande colorée
        self.set_fill_color(*C_BLUE_MID)
        self.rect(0, 55, 210, 6, style="F")

        # Titre principal
        self.set_y(18)
        self.set_font("Helvetica", style="B", size=22)
        self.set_text_color(*C_WHITE)
        self.multi_cell(0, 10, "Rapport Technique\nProjet de Fin d'Année", align="C")

        # Sous-titre
        self.ln(2)
        self.set_font("Helvetica", style="I", size=13)
        self.set_text_color(190, 220, 240)
        self.cell(0, 8,
                  "Système de Prédiction de Réadmission Hospitalière (30 jours)",
                  align="C", new_x="LMARGIN", new_y="NEXT")

        # Bloc descriptif centré
        self.set_y(80)
        self.set_font("Helvetica", size=11)
        self.set_text_color(*C_BLACK)
        self.set_fill_color(*C_BLUE_LIGHT)
        self.multi_cell(
            0, 8,
            "Ce document présente l'ensemble du travail réalisé dans le cadre\n"
            "du projet de fin d'année : nettoyage des données, entraînement\n"
            "des modèles de Machine Learning, déploiement via FastAPI et\n"
            "Streamlit, ainsi que la conception UML du système.",
            align="C", fill=True, border=1
        )

        # Infos projet
        self.ln(12)
        infos = [
            ("Dataset",     "hospital_readmissions_30k.csv — 30 000 patients"),
            ("Modèles",     "Logistic Regression · Random Forest · XGBoost"),
            ("Outils",      "Python · scikit-learn · MLflow · FastAPI · Streamlit · SHAP"),
            ("Auteur",      "Salma Chah"),
            ("Date",        "Avril 2026"),
        ]
        self.set_font("Helvetica", size=10)
        for label, value in infos:
            self.set_font("Helvetica", style="B", size=10)
            self.set_text_color(*C_BLUE_DARK)
            self.cell(38, 8, label + " :", align="R")
            self.set_font("Helvetica", size=10)
            self.set_text_color(*C_BLACK)
            self.cell(0, 8, sanitize_unicode(value), new_x="LMARGIN", new_y="NEXT")

        # Bande inférieure décorative
        self.set_fill_color(*C_BLUE_DARK)
        self.rect(0, 275, 210, 22, style="F")
        self.set_y(280)
        self.set_font("Helvetica", style="I", size=8)
        self.set_text_color(*C_WHITE)
        self.cell(0, 6, "PFA 2025-2026  |  Classification binaire supervisée", align="C")

    # ── Helpers de rendu ──────────────────────────────────────────────────────

    def h1(self, text):
        """Titre de section principal."""
        self.ln(5)
        # Bande colorée pleine largeur
        self.set_fill_color(*C_BLUE_DARK)
        self.set_text_color(*C_WHITE)
        self.set_font("Helvetica", style="B", size=14)
        self.cell(0, 9, "  " + hex_strip(text), fill=True,
                  new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_text_color(*C_BLACK)

    def h2(self, text):
        """Titre de sous-section."""
        self.ln(4)
        self.set_font("Helvetica", style="B", size=12)
        self.set_text_color(*C_BLUE_MID)
        # Filet gauche coloré
        x, y = self.get_x(), self.get_y()
        self.set_fill_color(*C_BLUE_MID)
        self.rect(20, y, 1.5, 7, style="F")
        self.set_x(23)
        self.cell(0, 7, hex_strip(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_text_color(*C_BLACK)

    def h3(self, text):
        """Titre de sous-sous-section."""
        self.ln(3)
        self.set_font("Helvetica", style="B", size=11)
        self.set_text_color(*C_GREEN_DARK)
        self.cell(0, 6, hex_strip(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_text_color(*C_BLACK)

    def body(self, text):
        """Paragraphe normal avec support du gras inline."""
        self.set_font("Helvetica", size=10)
        self.set_text_color(*C_BLACK)
        # Gestion du gras inline **...**
        parts = re.split(r'(\*\*[^*]+\*\*)', text)
        line_parts = []
        for p in parts:
            m = re.match(r'\*\*([^*]+)\*\*', p)
            if m:
                line_parts.append(("B", m.group(1)))
            else:
                clean = re.sub(r'`([^`]+)`', r'\1', p)
                if clean:
                    line_parts.append(("", clean))
        if not line_parts:
            return

        # Écriture multi-style sur une seule ligne (multi_cell pour wrap)
        full_text = "".join(v for _, v in line_parts)
        full_text = hex_strip(full_text)
        self.set_font("Helvetica", size=10)
        self.multi_cell(0, 5.5, full_text, new_x="LMARGIN", new_y="NEXT")
        self.ln(0.5)

    def bullet(self, text, level=0):
        """Point de liste."""
        indent = 6 + level * 5
        self.set_x(20 + indent)
        self.set_font("Helvetica", size=10)
        self.set_text_color(*C_BLACK)
        symbol = chr(149) if level == 0 else "-"
        clean  = hex_strip(text)
        self.cell(5, 5.5, symbol)
        available = 170 - indent - 5
        self.multi_cell(available, 5.5, clean, new_x="LMARGIN", new_y="NEXT")

    def code_block(self, lines):
        """Bloc de code avec fond gris."""
        self.ln(2)
        self.set_fill_color(*C_CODE_BG)
        self.set_draw_color(*C_CODE_BORDER)
        self.set_font("Courier", size=8.5)
        self.set_text_color(50, 50, 50)
        content = "\n".join(sanitize_unicode(l) for l in lines)
        self.multi_cell(0, 5, content, fill=True, border=1,
                        new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_font("Helvetica", size=10)
        self.set_text_color(*C_BLACK)

    def hr(self):
        """Ligne de séparation horizontale."""
        self.ln(3)
        self.set_draw_color(*C_SEP)
        self.set_line_width(0.4)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(4)

    def table(self, rows):
        """
        Tableau à partir d'une liste de lignes markdown.
        rows[0] = en-tête, rows[1] = séparateur ---, rows[2:] = données
        """
        if len(rows) < 2:
            return

        # Parse colonnes
        def parse_row(line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            return cells

        header_cells = parse_row(rows[0])
        data_rows    = [parse_row(r) for r in rows[2:] if r.strip()]
        n_cols       = len(header_cells)
        if n_cols == 0:
            return

        page_w  = 170   # largeur utile (210 - 2×20)
        col_w   = page_w / n_cols

        self.ln(3)

        # En-tête
        self.set_fill_color(*C_BLUE_DARK)
        self.set_text_color(*C_WHITE)
        self.set_font("Helvetica", style="B", size=9)
        for cell in header_cells:
            self.cell(col_w, 7, hex_strip(cell), border=1, fill=True, align="C")
        self.ln()

        # Lignes de données
        self.set_font("Helvetica", size=8.8)
        for i, row in enumerate(data_rows):
            if i % 2 == 0:
                self.set_fill_color(*C_WHITE)
            else:
                self.set_fill_color(*C_BLUE_LIGHT)
            self.set_text_color(*C_BLACK)

            # Hauteur dynamique selon contenu le plus long
            row_padded = (row + [""] * n_cols)[:n_cols]
            max_lines  = max(
                len(self.multi_cell(col_w, 5, hex_strip(c),
                                    dry_run=True, output="LINES"))
                for c in row_padded
            )
            row_h = max(6, max_lines * 5)

            for j, cell in enumerate(row_padded):
                x, y = self.get_x(), self.get_y()
                self.set_fill_color(
                    *(C_WHITE if i % 2 == 0 else C_BLUE_LIGHT)
                )
                self.multi_cell(col_w, row_h / max(max_lines, 1),
                                hex_strip(cell),
                                border=1, fill=True, align="L",
                                max_line_height=5)
                self.set_xy(x + col_w, y)
            self.ln(row_h)

        self.ln(3)
        self.set_text_color(*C_BLACK)


# ── Parseur Markdown → PDF ────────────────────────────────────────────────────

def render(pdf: RapportPDF, md_text: str):
    lines      = md_text.splitlines()
    i          = 0
    table_buf  = []
    code_buf   = []
    in_code    = False

    while i < len(lines):
        raw  = lines[i]
        line = raw.strip()

        # ── Blocs de code ````
        if line.startswith("```"):
            if not in_code:
                in_code   = True
                code_buf  = []
            else:
                in_code = False
                pdf.code_block(code_buf)
                code_buf = []
            i += 1
            continue

        if in_code:
            code_buf.append(raw)
            i += 1
            continue

        # ── Tableaux
        if line.startswith("|"):
            table_buf.append(line)
            i += 1
            continue
        elif table_buf:
            pdf.table(table_buf)
            table_buf = []

        # ── Titres
        if line.startswith("#### "):
            pdf.h3(line[5:])
        elif line.startswith("### "):
            pdf.h3(line[4:])
        elif line.startswith("## "):
            pdf.h2(line[3:])
        elif re.match(r'^# ', line):
            # H1 du document = titre de page
            text = line[2:]
            if "Rapport Technique" in text or "Système" in text:
                pass   # déjà sur la page de garde
            else:
                pdf.h1(text)

        # ── Séparateurs ---
        elif re.match(r'^-{3,}$', line):
            pdf.hr()

        # ── Listes à puces
        elif line.startswith("- ") or line.startswith("* "):
            pdf.bullet(line[2:])

        # ── Lignes de code inline seules (commençant par ``)
        elif line.startswith("`") and line.endswith("`"):
            pdf.code_block([line.strip("`")])

        # ── Lignes vides
        elif line == "":
            pdf.ln(2)

        # ── Paragraphes normaux
        else:
            pdf.body(line)

        i += 1

    # Tableau ou code non fermés
    if table_buf:
        pdf.table(table_buf)
    if code_buf:
        pdf.code_block(code_buf)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(MD_PATH, encoding="utf-8") as f:
        md_text = f.read()

    pdf = RapportPDF()

    # Page de garde
    pdf.page_de_garde()

    # Contenu : on commence après le titre H1 et H2 de la première ligne
    pdf.add_page()
    # Retire les deux premières lignes (# Titre et ## sous-titre) déjà sur la garde
    lines = md_text.splitlines()
    start = 0
    for idx, l in enumerate(lines):
        if l.startswith("---") and idx > 2:
            start = idx + 1
            break
    body_text = "\n".join(lines[start:])

    render(pdf, body_text)

    pdf.output(PDF_PATH)
    print(f"[OK] PDF genere -> {PDF_PATH}")


if __name__ == "__main__":
    main()
