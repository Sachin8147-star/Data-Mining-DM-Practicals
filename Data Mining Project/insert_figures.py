"""
insert_figures.py
-----------------
Fills the grey "Insert Figure N here" boxes in the Word report with the
graphs created by breast_cancer_data_mining.py.

HOW TO USE
  1. Run  breast_cancer_data_mining.py   ->  creates the 'figures' folder
  2. (Optional) take screenshots of your console output and save them as
        screenshots/output_1.png, output_2.png, output_3.png
     (png / jpg / jpeg all work)
  3. Keep these in ONE folder:
        Breast_Cancer_Data_Mining_Report.docx
        insert_figures.py
        figures/
        screenshots/   (optional)
  4. Run:  python insert_figures.py
  5. Open  Breast_Cancer_Data_Mining_Report_FINAL.docx

Install once:  pip install python-docx
"""

import glob
import os
import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches

REPORT = "Breast_Cancer_Data_Mining_Report.docx"
OUTPUT = "Breast_Cancer_Data_Mining_Report_FINAL.docx"
FIG_DIR = "figures"
SHOT_DIR = "screenshots"

# figure number in report -> (file in figures folder, width in inches)
FIGURES = {
    1: ("01_class_distribution.png", 2.9),
    2: ("02_feature_histograms.png", 3.0),
    3: ("03_correlation_heatmap.png", 4.4),
    4: ("04_feature_importance.png", 3.0),
    5: ("07_roc_curves.png", 2.7),
    6: ("05_pca.png", 6.0),
    7: ("08_confusion_matrix.png", 2.9),
    8: ("10_kmeans_clusters.png", 2.9),
}
SHOT_WIDTH = 6.2  # inches


def all_paragraphs(parent):
    """Every paragraph in the document, including those inside tables."""
    for p in parent.paragraphs:
        yield p
    for t in getattr(parent, "tables", []):
        for row in t.rows:
            for cell in row.cells:
                yield from all_paragraphs(cell)


def find_screenshot(n):
    for ext in ("png", "jpg", "jpeg"):
        hits = glob.glob(os.path.join(SHOT_DIR, f"output_{n}.{ext}"))
        if hits:
            return hits[0]
    return None


def clear_box_style(paragraph):
    """Remove the dashed border and grey fill of the placeholder cell."""
    tc = paragraph._p.getparent()
    if tc.tag != qn("w:tc") or tc.tcPr is None:
        return
    for tag in ("w:tcBorders", "w:shd"):
        el = tc.tcPr.find(qn(tag))
        if el is not None:
            tc.tcPr.remove(el)


def replace_with_picture(paragraph, path, width_in):
    for run in list(paragraph.runs):
        run._element.getparent().remove(run._element)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = paragraph.paragraph_format
    pf.space_before = Inches(0.05)
    pf.space_after = Inches(0.03)
    pf.keep_with_next = True
    paragraph.add_run().add_picture(path, width=Inches(width_in))
    clear_box_style(paragraph)


def main():
    if not os.path.exists(REPORT):
        raise SystemExit(f"Cannot find {REPORT} in this folder.")

    doc = Document(REPORT)
    done, missing = [], []

    for p in list(all_paragraphs(doc)):
        text = p.text

        m = re.search(r"Insert Figure (\d+)", text)
        if m:
            n = int(m.group(1))
            fname, width = FIGURES.get(n, (None, None))
            path = os.path.join(FIG_DIR, fname) if fname else None
            if path and os.path.exists(path):
                replace_with_picture(p, path, width)
                done.append(f"Figure {n}")
            else:
                missing.append(f"Figure {n} ({path})")
            continue

        m = re.search(r"Insert Output Screenshot (\d+)", text)
        if m:
            n = int(m.group(1))
            path = find_screenshot(n)
            if path:
                replace_with_picture(p, path, SHOT_WIDTH)
                done.append(f"Output screenshot {n}")
            else:
                missing.append(f"Output screenshot {n} (screenshots/output_{n}.png)")

    doc.save(OUTPUT)
    print("Inserted:", ", ".join(done) if done else "nothing")
    if missing:
        print("Not found (box left in place):")
        for x in missing:
            print("   -", x)
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
