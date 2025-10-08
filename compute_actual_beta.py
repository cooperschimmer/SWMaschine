#!/usr/bin/env python3
"""Berechnet B-Achsen-Winkel basierend auf ACT-Punktdaten aus zwei CSV-Dateien.

Dieses Skript liest zwei CSV-Dateien ein, die synchronisierte Punkte der linken
und rechten Bahnseite enthalten. Die Dateien müssen die Spalten ``ACT_X``,
``ACT_Y`` und ``ACT_Z`` enthalten. Für jedes Punktpaar wird der
Verbindungsvektor bestimmt und daraus der Winkel berechnet, den eine B-Achse
(Rotation um die Y-Achse) einnehmen muss, damit die Werkzeug-Z-Achse auf diesen
Vektor zeigt.

Die Ergebnisse werden in eine CSV-Datei geschrieben, die alle verwendeten
Punktkoordinaten, den Verbindungsvektor sowie zwei Winkel enthält:

``winkel_beta1_ausrichtung``
    Gibt an, wie groß der Winkel zwischen der Werkzeug-Z-Achse und dem
    Verbindungsvektor ist, wenn die B-Achse positiv (um Y) gedreht wird. Ein
    Wert von ``0`` bedeutet, dass der Vektor in +Z-Richtung zeigt; ``90``
    bedeutet Ausrichtung entlang +X.

``winkel_beta2_maschine``
    Ergänzungswinkel zu ``90°``. Dieser Wert ist hilfreich, wenn ein
    Steuerungssystem den Winkel relativ zur +X-Ausrichtung benötigt.

``Mathematisches Vorgehen´´
    Vektor der Punkte v-> = Pb (X Y Z)-Pa (X Y Z) = ( I J K)
    Werkzeugachsen Vektor zur normalisierung z-> = ( 0 0 1)
    Kreuzprodukt = n-> = v-> * z->
    Skalarprodukt = (v-> * z->) / 
"""

#from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

DEFAULT_LEFT = "Output_CSV_Li.csv"
DEFAULT_RIGHT = "Output_CSV_Re.csv"
DEFAULT_RESULT = "results_beta.csv"

ACT_KEYS = ("ACT_X", "ACT_Y", "ACT_Z")


class CsvFormatError(RuntimeError):
    """Wird ausgegeben, wenn Pflichtspalten in einer Eingabedatei fehlen."""


def _resolve(path: Union[str, Path]) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p


def _read_points(path: Union[str, Path]) -> List[Dict[str, float]]:
    file_path = _resolve(path)
    try:
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            rows = list(reader)
    except Exception as exc:  # pragma: no cover - reine Fehlermeldung
        raise RuntimeError(f"Fehler beim Lesen der CSV '{path}': {exc}") from exc

    if not rows:
        return []

    missing = [key for key in ACT_KEYS if key not in rows[0]]
    if missing:
        raise CsvFormatError(
            f"In der Datei '{path}' fehlen folgende Spalten: {', '.join(missing)}"
        )

    parsed: List[Dict[str, float]] = []
    for row in rows:
        parsed_row: Dict[str, float] = {}
        for key in ACT_KEYS:
            value = row.get(key)
            try:
                parsed_row[key] = float(value) if value is not None else math.nan
            except ValueError:
                parsed_row[key] = math.nan
        parsed.append(parsed_row)
    return parsed


def _pairwise(left: Iterable[Dict[str, float]], right: Iterable[Dict[str, float]]) -> Iterable[Tuple[Dict[str, float], Dict[str, float]]]:
    for idx, (row_left, row_right) in enumerate(zip(left, right)):
        yield idx, row_left, row_right


def _compute_beta_angles(dx: float, dy: float, dz: float) -> Tuple[float, float, float]:
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length == 0.0 or not math.isfinite(length):
        return math.nan, math.nan, math.nan

    beta_align = math.degrees(math.atan2(dx, dz))
    beta_machine = 90.0 - beta_align
    return length, beta_align, beta_machine


def process(left: List[Dict[str, float]], right: List[Dict[str, float]]) -> List[Dict[str, float]]:
    if len(left) != len(right):
        raise RuntimeError(
            f"Unterschiedliche Zeilenzahl: links={len(left)} vs rechts={len(right)}"
        )

    results: List[Dict[str, float]] = []
    for idx, row_left, row_right in _pairwise(left, right):
        dx = row_right["ACT_X"] - row_left["ACT_X"]
        dy = row_right["ACT_Y"] - row_left["ACT_Y"]
        dz = row_right["ACT_Z"] - row_left["ACT_Z"]
        length, beta_align, beta_machine = _compute_beta_angles(dx, dy, dz)

        results.append(
            {
                "index": idx,
                "act_x_L": row_left["ACT_X"],
                "act_y_L": row_left["ACT_Y"],
                "act_z_L": row_left["ACT_Z"],
                "act_x_R": row_right["ACT_X"],
                "act_y_R": row_right["ACT_Y"],
                "act_z_R": row_right["ACT_Z"],
                "delta_x": dx,
                "delta_y": dy,
                "delta_z": dz,
                "abstand": length,
                "winkel_beta1_ausrichtung": beta_align,
                "winkel_beta2_maschine": beta_machine,
            }
        )
    return results


def _write_results(result_path: Union[str, Path], rows: List[Dict[str, float]]) -> None:
    path = _resolve(result_path)
    fieldnames = [
        "index",
        "act_x_L",
        "act_y_L",
        "act_z_L",
        "act_x_R",
        "act_y_R",
        "act_z_R",
        "delta_x",
        "delta_y",
        "delta_z",
        "abstand",
        "winkel_beta1_ausrichtung",
        "winkel_beta2_maschine",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Berechnet B-Achsen-Winkel aus ACT-Punktdaten zweier CSV-Dateien."
    )
    parser.add_argument("--left", default=DEFAULT_LEFT, help="Pfad zur linken CSV-Datei")
    parser.add_argument(
        "--right", default=DEFAULT_RIGHT, help="Pfad zur rechten CSV-Datei"
    )
    parser.add_argument(
        "--output", default=DEFAULT_RESULT, help="Pfad für die Ergebnis-CSV"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_rows = _read_points(args.left)
    right_rows = _read_points(args.right)
    results = process(left_rows, right_rows)
    _write_results(args.output, results)


if __name__ == "__main__":
    main()
