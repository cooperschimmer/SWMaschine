#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_scan_vectors.py

Liest zwei CSV-Dateien mit synchronen Scan-Bahnen (links/rechts), führt vektorisierte Berechnungen (Kreuzprodukt, Skalarprodukt etc.) durch und leitet generische ABC-Orientierungen ab. Ergebnisse werden als CSV ausgegeben.

Erwartete Spalten pro CSV:
  - Positionen aktuell:   actual_x, actual_y, actual_z
  - Positionen nominal:   nominal_x, nominal_y, nominal_z
  - Richtungsvektoren aktuell: actual_i, actual_j, actual_k
  - Richtungsvektoren nominal: nominal_i, nominal_j, nominal_k

Hinweis ABC:
Die Funktion vector_to_abc() bildet einen Richtungsvektor (angenommene Werkzeug-Z-Achse im Werkstück-KS) auf Euler-Winkel in Z-Y-X-Konvention (Rz * Ry * Rx) ab. Zur Eindeutigkeit wird eine Referenz-X-Achse gewählt, um eine vollständige Rotationsmatrix zu konstruieren, aus der ZYX-Winkel extrahiert werden. Die Zuordnung A=Rx, B=Ry, C=Rz ist generisch und maschinen-/kinematikabhängig ggf. anzupassen.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


# --------------------------- Konfiguration ---------------------------

# Standard-Pfade/-Einstellungen (können bei Bedarf zentral angepasst werden)
BASE_DIR = Path(__file__).resolve().parent

LEFT_CSV_PATH = "Output_CSV_Li.csv"
RIGHT_CSV_PATH = "Output_CSV_Re.csv"
RESULT_CSV_PATH = "results.csv"
RESULT_SIMPLE_CSV_PATH = "results_simple.csv"
USE_MODE = "both"  # "actual", "nominal" oder "both"

REQUIRED_COLS = [
    "actual_x", "actual_y", "actual_z",
    "nominal_x", "nominal_y", "nominal_z",
    "actual_i", "actual_j", "actual_k",
    "nominal_i", "nominal_j", "nominal_k",
]


def resolve_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p


def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s | %(message)s",
    )


def read_csv_checked(path: str) -> List[Dict[str, float]]:
    """Liest eine CSV-Datei ein und bringt die Spaltennamen auf das erwartete Schema."""

    rename_map: Dict[str, str] = {
        "NOM_X": "nominal_x",
        "NOM_Y": "nominal_y",
        "NOM_Z": "nominal_z",
        "ACT_X": "actual_x",
        "ACT_Y": "actual_y",
        "ACT_Z": "actual_z",
        "NOM_I": "nominal_i",
        "NOM_J": "nominal_j",
        "NOM_K": "nominal_k",
        "ACT_I": "actual_i",
        "ACT_J": "actual_j",
        "ACT_K": "actual_k",
    }

    file_path = resolve_path(path)

    try:
        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            rows_raw = list(reader)
    except Exception as e:
        logging.error(f"Fehler beim Einlesen der CSV '{path}': {e}")
        raise

    if not rows_raw:
        return []

    rows: List[Dict[str, float]] = []
    for row in rows_raw:
        normalized: Dict[str, float] = {}
        for key, value in row.items():
            if key is None:
                continue
            key_clean = key.strip()
            mapped = rename_map.get(key_clean, key_clean)
            try:
                normalized[mapped] = float(value.strip()) if value is not None else math.nan
            except (AttributeError, ValueError):
                normalized[mapped] = math.nan
        rows.append(normalized)

    missing = [c for c in REQUIRED_COLS if c not in rows[0]]
    if missing:
        available = sorted(rows[0].keys())
        logging.error(
            f"In '{path}' fehlen erforderliche Spalten: {missing}\nVorhandene Spalten: {available}"
        )
        raise ValueError(f"Fehlende Spalten in {path}: {missing}")

    return rows


def add_suffix(rows: List[Dict[str, float]], suffix: str) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for row in rows:
        out.append({f"{key}_{suffix}": value for key, value in row.items()})
    return out


# --------------------------- Math Helpers ---------------------------

def vector_sub(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vector_cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vector_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vector_norm(v: Sequence[float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def normalize_vector(v: Sequence[float], eps: float = 1e-12) -> Tuple[Tuple[float, float, float], float]:
    n = vector_norm(v)
    if not math.isfinite(n) or n < eps:
        return (math.nan, math.nan, math.nan), n
    return (v[0] / n, v[1] / n, v[2] / n), n


def zyx_euler_from_R(R: Sequence[Sequence[float]], eps: float = 1e-12) -> Tuple[float, float, float]:
    """
    Extrahiert Z-Y-X Euler-Winkel (yaw=z, pitch=y, roll=x) aus einer 3x3 Rotationsmatrix.
    Rückgabe in Radiant.
    """
    # sy = sqrt(R[0,0]^2 + R[1,0]^2)
    sy = math.sqrt(R[0][0] ** 2 + R[1][0] ** 2)
    if sy > eps:
        yaw = math.atan2(R[1][0], R[0][0])      # z
        pitch = math.atan2(-R[2][0], sy)        # y
        roll = math.atan2(R[2][1], R[2][2])     # x
    else:
        # Gimbal-Lock: R[0,0] ~ R[1,0] ~ 0
        yaw = math.atan2(-R[0][1], R[1][1])     # z
        pitch = math.atan2(-R[2][0], sy)        # y (sy≈0)
        roll = 0.0                              # x
    return yaw, pitch, roll


def vector_to_abc(v: Sequence[float], eps: float = 1e-12) -> Tuple[float, float, float]:
    """
    Erzeugt eine plausible Orientierung (A,B,C) aus einem Richtungsvektor v.
    Annahme: v ist die gewünschte Werkzeug-Z-Achse im Werkstückkoordinatensystem.
    Vorgehen:
      1) Normiere v -> z.
      2) Wähle Referenz x_ref = [1,0,0] (falls parallel zu z, nutze [0,1,0]).
      3) y = norm(cross(z, x_ref)); x = cross(y, z).
      4) R = [x y z] (Spalten als Achsen).
      5) Extrahiere ZYX-Euler (yaw=z, pitch=y, roll=x). Mappe auf A=Rx, B=Ry, C=Rz.
    Rückgabe: Winkel in Grad als (A, B, C).
    Hinweis: Maschinenabhängig! Ggf. an Kinematik (Kopf-/Tischachsen, Vorzeichen, Reihenfolge) anpassen.
    """
    normed, n = normalize_vector(v, eps=eps)
    if not math.isfinite(n) or n < eps:
        return math.nan, math.nan, math.nan

    z = normed
    x_ref = (1.0, 0.0, 0.0)
    cross_z_xref = vector_cross(z, x_ref)
    if vector_norm(cross_z_xref) < 1e-6:
        x_ref = (0.0, 1.0, 0.0)
        cross_z_xref = vector_cross(z, x_ref)

    y_vec = cross_z_xref
    y_norm = vector_norm(y_vec)
    if y_norm < eps:
        return math.nan, math.nan, math.nan
    y = (y_vec[0] / y_norm, y_vec[1] / y_norm, y_vec[2] / y_norm)
    x_vec = vector_cross(y, z)
    x = x_vec

    R = (
        (x[0], y[0], z[0]),
        (x[1], y[1], z[1]),
        (x[2], y[2], z[2]),
    )

    yaw, pitch, roll = zyx_euler_from_R(R)
    # Map: A=roll (Rx), B=pitch (Ry), C=yaw (Rz)
    A = math.degrees(roll)
    B = math.degrees(pitch)
    C = math.degrees(yaw)
    return A, B, C


# --------------------------- Core Computation ---------------------------

def compute_block(
    rows_L: List[Dict[str, float]],
    rows_R: List[Dict[str, float]],
    kind: str,
    prefix: str,
) -> Tuple[List[Dict[str, float]], List[str]]:
    """
    Berechnet alle Werte für 'actual' oder 'nominal'.
    kind   ∈ {'actual','nominal'}
    prefix ∈ {'', 'act_', 'nom_'}
    """
    assert kind in ("actual", "nominal")

    columns_order = [
        f"{prefix}{kind}_x_L",
        f"{prefix}{kind}_y_L",
        f"{prefix}{kind}_z_L",
        f"{prefix}{kind}_x_R",
        f"{prefix}{kind}_y_R",
        f"{prefix}{kind}_z_R",
        f"{prefix}{kind}_i_L",
        f"{prefix}{kind}_j_L",
        f"{prefix}{kind}_k_L",
        f"{prefix}{kind}_i_R",
        f"{prefix}{kind}_j_R",
        f"{prefix}{kind}_k_R",
        f"{prefix}d_x",
        f"{prefix}d_y",
        f"{prefix}d_z",
        f"{prefix}d_norm",
        f"{prefix}c_x",
        f"{prefix}c_y",
        f"{prefix}c_z",
        f"{prefix}c_norm",
        f"{prefix}dot_c_d",
        f"{prefix}dot_n",
        f"{prefix}angle_n_deg",
        f"{prefix}A_L",
        f"{prefix}B_L",
        f"{prefix}C_L",
        f"{prefix}A_R",
        f"{prefix}B_R",
        f"{prefix}C_R",
        f"{prefix}A_c",
        f"{prefix}B_c",
        f"{prefix}C_c",
    ]

    result_rows: List[Dict[str, float]] = []
    warn_nL = False
    warn_nR = False

    for row_L, row_R in zip(rows_L, rows_R):
        pL = (
            row_L[f"{kind}_x_L"],
            row_L[f"{kind}_y_L"],
            row_L[f"{kind}_z_L"],
        )
        pR = (
            row_R[f"{kind}_x_R"],
            row_R[f"{kind}_y_R"],
            row_R[f"{kind}_z_R"],
        )
        d_vec = vector_sub(pR, pL)
        d_norm = vector_norm(d_vec)

        nL_raw = (
            row_L[f"{kind}_i_L"],
            row_L[f"{kind}_j_L"],
            row_L[f"{kind}_k_L"],
        )
        nR_raw = (
            row_R[f"{kind}_i_R"],
            row_R[f"{kind}_j_R"],
            row_R[f"{kind}_k_R"],
        )

        nL_normed, nL_norm = normalize_vector(nL_raw)
        nR_normed, nR_norm = normalize_vector(nR_raw)

        if not math.isfinite(nL_norm) or nL_norm < 1e-12:
            warn_nL = True
        if not math.isfinite(nR_norm) or nR_norm < 1e-12:
            warn_nR = True

        if any(math.isnan(component) for component in nL_normed) or any(
            math.isnan(component) for component in nR_normed
        ):
            c_vec = (math.nan, math.nan, math.nan)
            c_norm = math.nan
            dot_c_d = math.nan
            dot_n = math.nan
        else:
            c_vec = vector_cross(nR_normed, nL_normed)
            c_norm = vector_norm(c_vec)
            dot_c_d = vector_dot(c_vec, d_vec)
            dot_n = max(-1.0, min(1.0, vector_dot(nR_normed, nL_normed)))

        angle_n_deg = math.degrees(math.acos(dot_n)) if math.isfinite(dot_n) else math.nan

        A_L, B_L, C_L = vector_to_abc(nL_normed) if math.isfinite(nL_norm) and nL_norm >= 1e-12 else (math.nan, math.nan, math.nan)
        A_R, B_R, C_R = vector_to_abc(nR_normed) if math.isfinite(nR_norm) and nR_norm >= 1e-12 else (math.nan, math.nan, math.nan)

        if math.isfinite(c_norm) and c_norm > 1e-12 and all(math.isfinite(v) for v in c_vec):
            c_unit = (c_vec[0] / c_norm, c_vec[1] / c_norm, c_vec[2] / c_norm)
            A_c, B_c, C_c = vector_to_abc(c_unit)
        else:
            A_c, B_c, C_c = (math.nan, math.nan, math.nan)

        result_rows.append(
            {
                f"{prefix}{kind}_x_L": pL[0],
                f"{prefix}{kind}_y_L": pL[1],
                f"{prefix}{kind}_z_L": pL[2],
                f"{prefix}{kind}_x_R": pR[0],
                f"{prefix}{kind}_y_R": pR[1],
                f"{prefix}{kind}_z_R": pR[2],
                f"{prefix}{kind}_i_L": nL_raw[0],
                f"{prefix}{kind}_j_L": nL_raw[1],
                f"{prefix}{kind}_k_L": nL_raw[2],
                f"{prefix}{kind}_i_R": nR_raw[0],
                f"{prefix}{kind}_j_R": nR_raw[1],
                f"{prefix}{kind}_k_R": nR_raw[2],
                f"{prefix}d_x": d_vec[0],
                f"{prefix}d_y": d_vec[1],
                f"{prefix}d_z": d_vec[2],
                f"{prefix}d_norm": d_norm,
                f"{prefix}c_x": c_vec[0],
                f"{prefix}c_y": c_vec[1],
                f"{prefix}c_z": c_vec[2],
                f"{prefix}c_norm": c_norm,
                f"{prefix}dot_c_d": dot_c_d,
                f"{prefix}dot_n": dot_n,
                f"{prefix}angle_n_deg": angle_n_deg,
                f"{prefix}A_L": A_L,
                f"{prefix}B_L": B_L,
                f"{prefix}C_L": C_L,
                f"{prefix}A_R": A_R,
                f"{prefix}B_R": B_R,
                f"{prefix}C_R": C_R,
                f"{prefix}A_c": A_c,
                f"{prefix}B_c": B_c,
                f"{prefix}C_c": C_c,
            }
        )

    if warn_nL:
        logging.warning(f"[{kind}] Einige n_L haben sehr kleine Normen – Werte auf NaN gesetzt.")
    if warn_nR:
        logging.warning(f"[{kind}] Einige n_R haben sehr kleine Normen – Werte auf NaN gesetzt.")

    return result_rows, columns_order


def main() -> None:
    setup_logging("INFO")

    logging.info("Einlesen der CSV-Dateien ...")
    rows_L_raw = read_csv_checked(LEFT_CSV_PATH)
    rows_R_raw = read_csv_checked(RIGHT_CSV_PATH)

    if len(rows_L_raw) != len(rows_R_raw):
        raise ValueError(f"Unterschiedliche Zeilenzahl: left={len(rows_L_raw)} vs right={len(rows_R_raw)}")

    # Suffixe anwenden
    rows_L = add_suffix(rows_L_raw, "L")
    rows_R = add_suffix(rows_R_raw, "R")

    if USE_MODE not in {"actual", "nominal", "both"}:
        raise ValueError("USE_MODE muss 'actual', 'nominal' oder 'both' sein.")

    blocks: List[Tuple[str, str, List[Dict[str, float]], List[str]]] = []

    if USE_MODE in ("actual", "both"):
        logging.info("Berechne Block: ACTUAL")
        prefix = "act_" if USE_MODE == "both" else ""
        block_rows, cols = compute_block(rows_L, rows_R, kind="actual", prefix=prefix)
        blocks.append(("actual", prefix, block_rows, cols))

    if USE_MODE in ("nominal", "both"):
        logging.info("Berechne Block: NOMINAL")
        prefix = "nom_" if USE_MODE == "both" else ""
        block_rows, cols = compute_block(rows_L, rows_R, kind="nominal", prefix=prefix)
        blocks.append(("nominal", prefix, block_rows, cols))

    fieldnames = ["index"]
    for _, _, _, cols in blocks:
        fieldnames.extend(cols)

    delta_columns: List[str] = []
    if USE_MODE == "both":
        delta_columns = [
            "delta_A_L",
            "delta_B_L",
            "delta_C_L",
            "delta_A_R",
            "delta_B_R",
            "delta_C_R",
            "delta_A_c",
            "delta_B_c",
            "delta_C_c",
        ]
        fieldnames.extend(delta_columns)

    result_rows: List[Dict[str, float]] = []
    for idx in range(len(rows_L)):
        row_out: Dict[str, float] = {"index": idx}
        for _, _, block_rows, _ in blocks:
            row_out.update(block_rows[idx])

        if USE_MODE == "both":
            for delta_key in delta_columns:
                base_key = delta_key.replace("delta_", "")
                act_key = f"act_{base_key}"
                nom_key = f"nom_{base_key}"
                act_val = row_out.get(act_key)
                nom_val = row_out.get(nom_key)
                if isinstance(act_val, float) and isinstance(nom_val, float) and math.isfinite(act_val) and math.isfinite(nom_val):
                    row_out[delta_key] = act_val - nom_val
                else:
                    row_out[delta_key] = math.nan

        result_rows.append(row_out)

    result_path = resolve_path(RESULT_CSV_PATH)

    logging.info(f"Schreibe Ergebnis nach '{result_path}' ...")
    try:
        with result_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in result_rows:
                formatted = {}
                for key in fieldnames:
                    value = row.get(key, "")
                    if isinstance(value, float) and math.isnan(value):
                        formatted[key] = ""
                    else:
                        formatted[key] = value
                writer.writerow(formatted)
    except Exception as e:
        logging.error(f"Fehler beim Schreiben der Ergebnis-CSV: {e}")
        raise

    # Vereinfachte Ergebnis-Datei mit Kerninformationen
    simple_path = resolve_path(RESULT_SIMPLE_CSV_PATH)
    simple_fieldnames = ["index"]

    def collect_simple_keys(kind: str, prefix: str) -> List[str]:
        keys = [
            f"{prefix}d_x",
            f"{prefix}d_y",
            f"{prefix}d_z",
            f"{prefix}c_x",
            f"{prefix}c_y",
            f"{prefix}c_z",
            f"{prefix}dot_c_d",
            f"{prefix}dot_n",
            f"{prefix}A_L",
            f"{prefix}B_L",
            f"{prefix}C_L",
            f"{prefix}A_R",
            f"{prefix}B_R",
            f"{prefix}C_R",
            f"{prefix}A_c",
            f"{prefix}B_c",
            f"{prefix}C_c",
        ]
        vec_key = "actual" if kind == "actual" else "nominal"
        for side in ("L", "R"):
            keys.extend(
                [
                    f"{prefix}{vec_key}_i_{side}",
                    f"{prefix}{vec_key}_j_{side}",
                    f"{prefix}{vec_key}_k_{side}",
                ]
            )
        return keys

    for kind, prefix, _, _ in blocks:
        simple_keys = collect_simple_keys(kind, prefix)
        simple_fieldnames.extend(simple_keys)

    logging.info(f"Schreibe vereinfachtes Ergebnis nach '{simple_path}' ...")

    try:
        with simple_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=simple_fieldnames)
            writer.writeheader()
            for row in result_rows:
                formatted = {}
                for key in simple_fieldnames:
                    value = row.get(key, "")
                    if isinstance(value, float) and math.isnan(value):
                        formatted[key] = ""
                    else:
                        formatted[key] = value
                writer.writerow(formatted)
    except Exception as e:
        logging.error(f"Fehler beim Schreiben der vereinfachten Ergebnis-CSV: {e}")
        raise

    logging.info("Fertig.")


if __name__ == "__main__":
    main()
