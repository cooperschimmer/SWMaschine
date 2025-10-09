
"""
Hart-kodiertes Skript:
- CSV einlesen (Pfad unten in CSV_PATH fest eintragen)
- Spalten X, Y, MAT erkennen 
- Zeilen mit höchstem und niedrigstem MAT finden
- Diese zwei Zeilen ausgeben
- Von XY0 aus jeweils 5 mm radial nach außen verschieben
- Neue Koordinaten anzeigen und als CSV speichern 
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ======= HIER EINSTELLEN =======
CSV_PATHS = [  # <--- Pfade zu deinen CSVs eintragen
    ("AD.csv", "result_AD.csv"),
    ("ID.csv", "result_ID.csv"),
]
XY0 = (0.0, 0.0)                       # <--- Mittelpunkt des Kreises (x0, y0) in mm
RADIAL_OFFSET_MM = 5.0                 # Abstand nach außen in mm
# =================================


def find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise KeyError(f"Spalte nicht gefunden. Gesucht: {candidates}. Vorhanden: {list(df.columns)}")


def load_csv_hardcoded(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV-Datei nicht gefunden: {path}")
    # Gängige Separatoren probieren
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep)
            # Plausibilitätscheck: mindestens 2 numerische Spalten
            if df.select_dtypes(include=[np.number]).shape[1] >= 2:
                return df
        except Exception:
            continue
    # Fallback
    return pd.read_csv(path, engine="python")


def compute_outward_point(x, y, x0, y0, offset_mm):
    vec = np.array([x - x0, y - y0], dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        direction = np.array([1.0, 0.0])  # Sonderfall: Punkt = Mittelpunkt -> nehme +X
    else:
        direction = vec / norm
    out_xy = np.array([x, y], dtype=float) + direction * float(offset_mm)
    return float(out_xy[0]), float(out_xy[1])


ORIGINAL_X_POSITIONS_PER_FILE = {}
NEW_X_POSITIONS_PER_FILE = {}
NEW_XY_POSITIONS_PER_FILE = {}
LAST_RUN_SUMMARY = []


def process_file(csv_path: str, out_csv: str):
    df = load_csv_hardcoded(csv_path)

    # Spalten auflösen
    mat_col = find_col(df, ["mat", "materialaufmass", "materialaufmaß", "aufmass", "aufmaß","mATERIALCONDITION mat"])
    x_col = find_col(df, ["x", "x_mm", "x [mm]", "pos_x", "xpos", "xcoord", "X","Actual X"])
    y_col = find_col(df, ["y", "y_mm", "y [mm]", "pos_y", "ypos", "ycoord", "Y","Actual Y"])

    # In Zahlen konvertieren
    for c in (x_col, y_col, mat_col):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    dfn = df.dropna(subset=[x_col, y_col, mat_col]).copy()
    if dfn.empty:
        print("Fehler: Keine gültigen numerischen Werte in X/Y/MAT gefunden.", file=sys.stderr)
        sys.exit(2)

    # Max/Min MAT (Vorzeichen-unabhängig)
    abs_mat = dfn[mat_col].abs()
    idx_max = abs_mat.idxmax()
    idx_min = abs_mat.idxmin()
    row_max = dfn.loc[idx_max].copy()
    row_min = dfn.loc[idx_min].copy()

    # Ergebnisse (Originalzeilen) ausgeben
    print(f"=== Originalzeilen ({csv_path}) ===")
    print("-- MAT_MAX --")
    print(row_max.to_string())
    print("\n-- MAT_MIN --")
    print(row_min.to_string())

    # Outward-5mm berechnen
    x0, y0 = float(XY0[0]), float(XY0[1])

    x_max, y_max = float(row_max[x_col]), float(row_max[y_col])
    x_min, y_min = float(row_min[x_col]), float(row_min[y_col])

    x_max_out, y_max_out = compute_outward_point(x_max, y_max, x0, y0, RADIAL_OFFSET_MM)
    x_min_out, y_min_out = compute_outward_point(x_min, y_min, x0, y0, RADIAL_OFFSET_MM)

    # Zusammenfassungstabelle
    results_rows = []
    for label, x, y, mat, xo, yo in [
        ("MAT_MAX_ABS", x_max, y_max, float(row_max[mat_col]), x_max_out, y_max_out),
        ("MAT_MIN_ABS", x_min, y_min, float(row_min[mat_col]), x_min_out, y_min_out),
    ]:
        results_rows.append({
            "Fall": label,
            "X": x, "Y": y, "MAT": mat,
            "X0": x0, "Y0": y0,
            "X_out_5mm": xo, "Y_out_5mm": yo,
            "Hinweis": "Outward = weg von (X0,Y0) entlang der Linie durch (X,Y)"
        })
    results = pd.DataFrame(results_rows)

    # Speichern
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    print("\n=== Neue XY (5mm outward) ===")
    print(f"  MAX -> ({x_max_out:.6f}, {y_max_out:.6f})")
    print(f"  MIN -> ({x_min_out:.6f}, {y_min_out:.6f})")
    print(f"\nErgebnis gespeichert: {out_path}")
    print("\n" + "-" * 60 + "\n")

    ORIGINAL_X_POSITIONS_PER_FILE[csv_path] = {
        "MAT_MAX_ABS": x_max,
        "MAT_MIN_ABS": x_min,
    }
    NEW_X_POSITIONS_PER_FILE[csv_path] = {
        "MAT_MAX_ABS": x_max_out,
        "MAT_MIN_ABS": x_min_out,
    }
    NEW_XY_POSITIONS_PER_FILE[csv_path] = {
        "MAT_MAX_ABS": (x_max_out, y_max_out),
        "MAT_MIN_ABS": (x_min_out, y_min_out),
    }

    return {
        "csv_path": csv_path,
        "output_csv": str(out_path),
        "original_x": ORIGINAL_X_POSITIONS_PER_FILE[csv_path],
        "new_x": NEW_X_POSITIONS_PER_FILE[csv_path],
        "new_xy": NEW_XY_POSITIONS_PER_FILE[csv_path],
    }


def main():
    if not CSV_PATHS:
        print("Keine CSVs konfiguriert. Bitte CSV_PATHS anpassen.", file=sys.stderr)
        sys.exit(1)

    summary = []
    for csv_path, out_csv in CSV_PATHS:
        try:
            result = process_file(csv_path, out_csv)
            summary.append(result)
        except Exception as exc:
            print(f"Fehler bei Verarbeitung von {csv_path}: {exc}", file=sys.stderr)
    return summary


if __name__ == "__main__":
    LAST_RUN_SUMMARY = main()
