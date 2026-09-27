import csv
from pathlib import Path
from datetime import date


def read_manifest(path):
    p = Path(path).resolve()
    with p.open(newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    required = {"path", "region", "date", "split", "cloudy", "clear", "mask"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must have columns: {', '.join(sorted(required))}")
    for row in rows:
        row["date_obj"] = date.fromisoformat(row["date"])
        if row["split"] not in ("train", "validation", "test", "reference"):
            raise ValueError(f"Invalid split {row['split']!r}")
        row["path"] = (p.parent / row["path"]).resolve()
        if not row["path"].is_file(): raise FileNotFoundError(row["path"])
        # cloudy/clear/mask identify arrays in the NPZ, not filesystem paths.
        for key in ("cloudy", "clear", "mask"):
            row[key] = row[key] or key
    return rows


def assert_region_disjoint(rows):
    """Fail closed on spatial leakage; use date-disjoint mode explicitly if necessary."""
    groups = {}
    for r in rows: 
        if r["split"] != "reference": groups.setdefault(r["region"], set()).add(r["split"])
    leaked = {k: v for k, v in groups.items() if len(v) > 1}
    if leaked:
        raise ValueError(f"Regions occur in multiple splits (leakage): {leaked}")
    return True
