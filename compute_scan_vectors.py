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

import logging
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd


# --------------------------- Konfiguration ---------------------------

# Standard-Pfade/-Einstellungen (können bei Bedarf zentral angepasst werden)
LEFT_CSV_PATH = "Output_CSV_Li.csv"
RIGHT_CSV_PATH = "Output_CSV_Re.csv"
RESULT_CSV_PATH = "results.csv"
USE_MODE = "both"  # "actual", "nominal" oder "both"

REQUIRED_COLS = [
    "actual_x", "actual_y", "actual_z",
    "nominal_x", "nominal_y", "nominal_z",
    "actual_i", "actual_j", "actual_k",
    "nominal_i", "nominal_j", "nominal_k",
]


def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s | %(message)s",
    )


def read_csv_checked(path: str) -> pd.DataFrame:
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

    try:
        df = pd.read_csv(path, sep=",", decimal=".", encoding="utf-8", skipinitialspace=True)
    except Exception as e:
        logging.error(f"Fehler beim Einlesen der CSV '{path}': {e}")
        raise

    # Whitespace entfernen und gewünschte Namen zuordnen
    df.columns = df.columns.str.strip()
    df = df.rename(columns=rename_map)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        logging.error(
            f"In '{path}' fehlen erforderliche Spalten: {missing}\nVorhandene Spalten: {list(df.columns)}"
        )
        raise ValueError(f"Fehlende Spalten in {path}: {missing}")

    return df


def add_suffix(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{c}_{suffix}" for c in df.columns]
    return df


# --------------------------- Math Helpers ---------------------------

def normalize_rows(arr: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalisiert zeilenweise 3D-Vektoren.
    Rückgabe: (normierte Vektoren, Normen). Bei Norm < eps wird die Zeile mit NaN gefüllt.
    """
    norms = np.linalg.norm(arr, axis=1)
    out = np.empty_like(arr, dtype=float)
    mask_ok = norms > eps
    out[mask_ok] = arr[mask_ok] / norms[mask_ok, None]
    out[~mask_ok] = np.nan
    return out, norms


def zyx_euler_from_R(R: np.ndarray, eps: float = 1e-12) -> Tuple[float, float, float]:
    """
    Extrahiert Z-Y-X Euler-Winkel (yaw=z, pitch=y, roll=x) aus einer 3x3 Rotationsmatrix.
    Rückgabe in Radiant.
    """
    # sy = sqrt(R[0,0]^2 + R[1,0]^2)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > eps:
        yaw = np.arctan2(R[1, 0], R[0, 0])      # z
        pitch = np.arctan2(-R[2, 0], sy)        # y
        roll = np.arctan2(R[2, 1], R[2, 2])     # x
    else:
        # Gimbal-Lock: R[0,0] ~ R[1,0] ~ 0
        yaw = np.arctan2(-R[0, 1], R[1, 1])     # z
        pitch = np.arctan2(-R[2, 0], sy)        # y (sy≈0)
        roll = 0.0                              # x
    return yaw, pitch, roll


def vector_to_abc(v: np.ndarray, eps: float = 1e-12) -> Tuple[float, float, float]:
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
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < eps:
        return np.nan, np.nan, np.nan

    z = v / n
    x_ref = np.array([1.0, 0.0, 0.0])
    if np.linalg.norm(np.cross(z, x_ref)) < 1e-6:
        x_ref = np.array([0.0, 1.0, 0.0])

    y = np.cross(z, x_ref)
    y_n = np.linalg.norm(y)
    if y_n < eps:
        return np.nan, np.nan, np.nan
    y = y / y_n
    x = np.cross(y, z)

    R = np.stack([x, y, z], axis=1)  # columns are axes

    yaw, pitch, roll = zyx_euler_from_R(R)
    # Map: A=roll (Rx), B=pitch (Ry), C=yaw (Rz)
    A = np.degrees(roll)
    B = np.degrees(pitch)
    C = np.degrees(yaw)
    return A, B, C


# --------------------------- Core Computation ---------------------------

def compute_block(dfL: pd.DataFrame, dfR: pd.DataFrame, kind: str, prefix: str) -> pd.DataFrame:
    """
    Berechnet alle Werte für 'actual' oder 'nominal'.
    kind   ∈ {'actual','nominal'}
    prefix ∈ {'', 'act_', 'nom_'}
    """
    assert kind in ("actual", "nominal")

    # Positionsarrays
    pL = dfL[[f"{kind}_x_L", f"{kind}_y_L", f"{kind}_z_L"]].to_numpy(dtype=float)
    pR = dfR[[f"{kind}_x_R", f"{kind}_y_R", f"{kind}_z_R"]].to_numpy(dtype=float)
    d = pR - pL
    d_x, d_y, d_z = d[:, 0], d[:, 1], d[:, 2]
    d_norm = np.linalg.norm(d, axis=1)

    # Normal-/Richtungsvektoren
    nL_raw = dfL[[f"{kind}_i_L", f"{kind}_j_L", f"{kind}_k_L"]].to_numpy(dtype=float)
    nR_raw = dfR[[f"{kind}_i_R", f"{kind}_j_R", f"{kind}_k_R"]].to_numpy(dtype=float)

    nL, nL_norms = normalize_rows(nL_raw)
    nR, nR_norms = normalize_rows(nR_raw)

    if np.any(~np.isfinite(nL).all(axis=1)):
        logging.warning(f"[{kind}] Einige n_L haben sehr kleine Normen – Werte auf NaN gesetzt.")
    if np.any(~np.isfinite(nR).all(axis=1)):
        logging.warning(f"[{kind}] Einige n_R haben sehr kleine Normen – Werte auf NaN gesetzt.")

    # Kreuzprodukt und Skalarprodukt
    c = np.cross(nR, nL)  # Reihenfolge R × L
    c_x, c_y, c_z = c[:, 0], c[:, 1], c[:, 2]
    c_norm = np.linalg.norm(c, axis=1)

    dot_c_d = np.einsum("ij,ij->i", c, d)
    dot_n = np.einsum("ij,ij->i", nR, nL)
    # Clip für numerische Stabilität
    dot_n_clipped = np.clip(dot_n, -1.0, 1.0)
    angle_n_deg = np.degrees(np.arccos(dot_n_clipped))

    # ABC aus nL, nR, c
    A_L = np.empty(len(dfL)); B_L = np.empty(len(dfL)); C_L = np.empty(len(dfL))
    A_R = np.empty(len(dfL)); B_R = np.empty(len(dfL)); C_R = np.empty(len(dfL))
    A_c = np.empty(len(dfL)); B_c = np.empty(len(dfL)); C_c = np.empty(len(dfL))

    for idx in range(len(dfL)):
        a, b, c_ang = vector_to_abc(nL[idx]) if np.all(np.isfinite(nL[idx])) else (np.nan, np.nan, np.nan)
        A_L[idx], B_L[idx], C_L[idx] = a, b, c_ang
        a, b, c_ang = vector_to_abc(nR[idx]) if np.all(np.isfinite(nR[idx])) else (np.nan, np.nan, np.nan)
        A_R[idx], B_R[idx], C_R[idx] = a, b, c_ang

        if np.isfinite(c_norm[idx]) and c_norm[idx] > 1e-12 and np.all(np.isfinite(c[idx])):
            a, b, c_ang = vector_to_abc(c[idx] / c_norm[idx])
        else:
            a, b, c_ang = (np.nan, np.nan, np.nan)
        A_c[idx], B_c[idx], C_c[idx] = a, b, c_ang

    # Ergebnis-DataFrame bauen (mit Präfix)
    out = pd.DataFrame({
        f"{prefix}d_x": d_x,
        f"{prefix}d_y": d_y,
        f"{prefix}d_z": d_z,
        f"{prefix}d_norm": d_norm,
        f"{prefix}c_x": c_x,
        f"{prefix}c_y": c_y,
        f"{prefix}c_z": c_z,
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
    })

    # Zusätzlich Positions- & Richtungs-Komponenten (links/rechts) mit aufnehmen
    pos_cols = pd.DataFrame({
        f"{prefix}{kind}_x_L": pL[:, 0],
        f"{prefix}{kind}_y_L": pL[:, 1],
        f"{prefix}{kind}_z_L": pL[:, 2],
        f"{prefix}{kind}_x_R": pR[:, 0],
        f"{prefix}{kind}_y_R": pR[:, 1],
        f"{prefix}{kind}_z_R": pR[:, 2],
        f"{prefix}{kind}_i_L": nL_raw[:, 0],
        f"{prefix}{kind}_j_L": nL_raw[:, 1],
        f"{prefix}{kind}_k_L": nL_raw[:, 2],
        f"{prefix}{kind}_i_R": nR_raw[:, 0],
        f"{prefix}{kind}_j_R": nR_raw[:, 1],
        f"{prefix}{kind}_k_R": nR_raw[:, 2],
    })

    # Reihenfolge gemäß Spezifikation: erst pos/ijk, dann d, c, dot, Winkel, ABC
    ordered = pd.concat([pos_cols, out], axis=1)
    return ordered


def main() -> None:
    setup_logging("INFO")

    logging.info("Einlesen der CSV-Dateien ...")
    dfL_raw = read_csv_checked(LEFT_CSV_PATH)
    dfR_raw = read_csv_checked(RIGHT_CSV_PATH)

    if len(dfL_raw) != len(dfR_raw):
        raise ValueError(f"Unterschiedliche Zeilenzahl: left={len(dfL_raw)} vs right={len(dfR_raw)}")

    # Suffixe anwenden
    dfL = add_suffix(dfL_raw, "L")
    dfR = add_suffix(dfR_raw, "R")

    # Index-Spalte (0-basiert)
    index_series = pd.Series(range(len(dfL)), name="index")

    result_blocks: List[pd.DataFrame] = []

    if USE_MODE in ("actual", "both"):
        logging.info("Berechne Block: ACTUAL")
        block_act = compute_block(dfL, dfR, kind="actual", prefix="act_" if USE_MODE == "both" else "")
        result_blocks.append(block_act)

    if USE_MODE in ("nominal", "both"):
        logging.info("Berechne Block: NOMINAL")
        block_nom = compute_block(dfL, dfR, kind="nominal", prefix="nom_" if USE_MODE == "both" else "")
        result_blocks.append(block_nom)

    if not result_blocks:
        raise ValueError("USE_MODE muss 'actual', 'nominal' oder 'both' sein.")

    # Zusammenführen
    result_df = pd.concat([index_series] + result_blocks, axis=1)

    # Spaltenreihenfolge ggf. minimal sortieren: index zuerst, Rest wie gebaut
    cols = list(result_df.columns)
    if cols[0] != "index":
        cols = ["index"] + [c for c in cols if c != "index"]
        result_df = result_df[cols]

    # Schreiben
    logging.info(f"Schreibe Ergebnis nach '{RESULT_CSV_PATH}' ...")
    try:
        result_df.to_csv(RESULT_CSV_PATH, sep=",", encoding="utf-8", index=False)
    except Exception as e:
        logging.error(f"Fehler beim Schreiben der Ergebnis-CSV: {e}")
        raise

    logging.info("Fertig.")


if __name__ == "__main__":
    main()
