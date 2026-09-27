from datetime import date
import numpy as np


def select_references(target, candidates, max_refs=5, season_days=45):
    """Rank same-region, earlier clear acquisitions. Never select future/test labels."""
    tdate = target["date_obj"] if hasattr(target["date_obj"], "toordinal") else date.fromisoformat(str(target["date"]))
    ranked = []
    for c in candidates:
        cdate = c["date_obj"] if hasattr(c["date_obj"], "toordinal") else date.fromisoformat(str(c["date"]))
        if c["region"] != target["region"] or cdate >= tdate or c.get("split") not in (target.get("split"), "reference"):
            continue
        if not c.get("clear") or not c["path"].is_file(): continue
        gap = (tdate - cdate).days
        season = abs((tdate.timetuple().tm_yday - cdate.timetuple().tm_yday + 183) % 366 - 183)
        # Prefer closer seasonal analogues, then nearby acquisition; transparent ranking.
        score = gap / 365.0 + 2.0 * season / max(season_days, 1)
        # Optional manifest quality fields: cloud_percent [0,100], quality_score [0,1],
        # spectral_distance (nonnegative, computed over clear overlap).
        try: score += max(0.0, float(c.get("cloud_percent", 0.0))) / 100.0
        except (TypeError, ValueError): pass
        try: score += 1.0 - min(1.0, max(0.0, float(c.get("quality_score", 1.0))))
        except (TypeError, ValueError): pass
        try: score += max(0.0, float(c.get("spectral_distance", 0.0)))
        except (TypeError, ValueError): pass
        ranked.append((score, c))
    ranked.sort(key=lambda z: z[0])
    return [r for _, r in ranked[:max_refs]]


def spectral_similarity(a, b, valid=None):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if a.shape != b.shape: raise ValueError("Spectral similarity requires aligned matching arrays")
    m = np.ones(a.shape[-2:], bool) if valid is None else np.asarray(valid, bool)
    return float(np.mean(np.abs(a[:, m] - b[:, m]))) if m.any() else float("inf")
