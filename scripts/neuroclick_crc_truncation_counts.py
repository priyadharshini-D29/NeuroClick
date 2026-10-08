"""Truncated-visit counts quoted in Sect. 3.2 of the paper, from the frozen audit manifest.

The audit (01_audit_buy_nobuy_events.py) detects visits only from gaze samples strictly before
min(page-interval end, cutoff) and records last_gaze_time = end of the last pre-cutoff visit
(last in-window sample + one eye-tracker sample). A visit was therefore still in progress at the
cutoff when last_gaze_time lies within one eye-tracker sample (1/120 s) of the cutoff.
The first visit is the truncated one only when it is the sole visit.
Run from the repository root: python scripts/neuroclick_crc_truncation_counts.py"""
import pandas as pd

a = pd.read_csv("results/audit_full_42_v2/subject_product_audit.csv")
c = a[a.considered == 1]
b = c[c.label_buy == 1]
tol = 1.0 / 120 * 1.0001
clipped = b.last_gaze_time >= b.predecision_cutoff_time - tol
single = b.n_gaze_visits == 1
print(f"pairs {len(a)} | excluded (no >=100 ms visit before cutoff) {(a.considered == 0).sum()} "
      f"(with a click: {((a.considered == 0) & (a.label_buy == 1)).sum()}) | retained {len(c)} "
      f"(Buy {len(b)}, NoBuy {len(c) - len(b)})")
print(f"Buy sequences whose last pre-cutoff visit was in progress at the cutoff: {clipped.sum()} / {len(b)}")
print(f"single-visit Buy sequences: {single.sum()}; of which the (first and only) visit was truncated: "
      f"{(clipped & single).sum()} = {(clipped & single).sum() / len(b):.1%} of all Buy first visits")
