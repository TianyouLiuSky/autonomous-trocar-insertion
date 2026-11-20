#!/usr/bin/env python3
"""Offline analysis of correspondence_log.csv (written by the '`' key in
home_to_rcm_insert_v2_precision.py). This is the "evidence" half of the
ISMR headline claim -- "the microscope-measured RCM point and the
depth-camera(D405)-measured RCM point are the same point" -- turned into
publication-grade statistics instead of a handful of printed numbers.

No ROS / robot / camera dependency -- pure CSV analysis, safe to run
anywhere (including off-robot) once correspondence_log.csv has been copied
over from the lab machine.

What this computes, and why each piece is here (ISMR/method-comparison
framing, not just "mean and std"):

  1. Campaign-integrity check: for a "10 trials from 10 DIFFERENT home
     positions" claim to be defensible to a reviewer, this script does NOT
     just trust the trial_index column -- it independently re-derives
     "how many distinct homes were actually used" from the logged
     home_x/y/z_mm columns themselves (clustering by a minimum-separation
     threshold), and prints a clear PASS/FAIL against the campaign's own
     stated n>=10-distinct-homes design. If the raw numbers don't support
     the claim, this script says so instead of silently assuming they do.

  2. Headline discrepancy statistics (raw AND offset-corrected), with a
     95% CI via Student's t (not a normal approximation -- n is small,
     typically ~10, so the t-distribution's fatter tails are the honest
     choice here).

  3. Bland-Altman-style agreement analysis -- THE standard statistical
     tool reviewers in a biomedical/surgical-robotics venue expect for
     exactly this "do two measurement methods agree" question. Classic
     Bland-Altman is defined for a single scalar measured two ways; here
     each trial yields a 3D point from each sensor, so this script reports
     Bland-Altman per-axis (mic_x-d405_x, mic_y-d405_y, mic_z-d405_z: bias
     + 95% limits of agreement = bias +/- 1.96*SD) AND treats the scalar
     3D discrepancy magnitude the same way as a secondary, more intuitive
     "how far apart are they" summary. Both views are saved as a single
     multi-panel figure.

  4. Generalization sanity check: Pearson correlation between
     (distance of this trial's home from the first trial's home) and
     (this trial's discrepancy), explicitly to pre-empt the obvious
     reviewer question "did it only work well from one favorable starting
     configuration?" -- reported with an explicit small-n caveat (do not
     overclaim a null correlation as proof of generalization with n~10).

Usage:
    python3 correspondence_report.py [path/to/correspondence_log.csv] [--no-plot]
"""

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from runtime_session import ROOT, data_path
import csv
import os
import sys

import numpy as np

try:
    from scipy import stats as _scipy_stats
except ImportError:
    _scipy_stats = None

HOME_DIVERSITY_MIN_MM = float(os.environ.get("HOME_DIVERSITY_MIN_MM", "5.0"))


def _f(row, key):
    v = row.get(key)
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_rows(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit("%s is empty" % path)
    return rows


def count_distinct_homes(rows, min_sep_mm=HOME_DIVERSITY_MIN_MM):
    """Re-derive the number of distinct home positions actually used,
    directly from logged home_x/y/z_mm -- a simple greedy clustering
    (any point within min_sep_mm of an already-accepted cluster center
    joins that cluster) rather than trusting trial_index at face value.
    Returns (n_clusters, cluster_assignment_per_row, centers)."""
    pts, idxs = [], []
    for i, r in enumerate(rows):
        x, y, z = _f(r, "home_x_mm"), _f(r, "home_y_mm"), _f(r, "home_z_mm")
        if x is None or y is None or z is None:
            continue
        pts.append((x, y, z))
        idxs.append(i)
    centers = []
    assign = {}
    for i, p in zip(idxs, pts):
        p = np.array(p)
        best_c, best_d = None, None
        for ci, c in enumerate(centers):
            d = float(np.linalg.norm(p - c))
            if best_d is None or d < best_d:
                best_c, best_d = ci, d
        if best_c is not None and best_d < min_sep_mm:
            assign[i] = best_c
        else:
            centers.append(p)
            assign[i] = len(centers) - 1
    return len(centers), assign, centers


def ci95_t(arr):
    """95% CI half-width via Student's t (honest for small n; falls back
    to a normal-approximation 1.96*SEM if scipy is unavailable)."""
    arr = np.asarray(arr, float)
    n = len(arr)
    if n < 2:
        return float("nan")
    sem = arr.std(ddof=1) / np.sqrt(n)
    if _scipy_stats is not None:
        tcrit = _scipy_stats.t.ppf(0.975, df=n - 1)
    else:
        tcrit = 1.96
    return float(tcrit * sem)


def bland_altman(a, b):
    """Classic Bland-Altman pair -> (mean, diff, bias, loa_lo, loa_hi)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mean = (a + b) / 2.0
    diff = a - b
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    return mean, diff, bias, bias - 1.96 * sd, bias + 1.96 * sd


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    no_plot = "--no-plot" in sys.argv
    path = args[0] if args else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "correspondence_log.csv")
    if not os.path.isfile(path):
        raise SystemExit("no such file: %s (press '`' at least once in "
                          "home_to_rcm_insert_v2_precision.py first)" % path)

    rows = load_rows(path)
    print("=== correspondence_log.csv summary (%s) ===" % path)
    print("total rows: %d\n" % len(rows))

    # ---- 1. campaign-integrity check ------------------------------------
    n_homes, assign, centers = count_distinct_homes(rows)
    n_with_home = sum(1 for r in rows if _f(r, "home_x_mm") is not None)
    print("--- campaign integrity (10+ DIFFERENT home positions requirement) ---")
    if n_with_home == 0:
        print("  !! no home_x/y/z_mm logged in any row (older script version?) -- "
              "cannot independently verify the 'different home per trial' claim.")
    else:
        print("  rows with a logged home position: %d/%d" % (n_with_home, len(rows)))
        print("  distinct home-position clusters found (>=%.1fmm apart): %d"
              % (HOME_DIVERSITY_MIN_MM, n_homes))
        verdict = "PASS" if n_homes >= 10 else ("MARGINAL" if n_homes >= 2 else "FAIL")
        print("  verdict: %s (campaign design target: >=10 distinct homes)" % verdict)
        if verdict != "PASS":
            print("  !! this campaign's rows do NOT yet independently demonstrate "
                  ">=10 distinct starting configurations -- either more trials from "
                  "new homes are needed, or HOME_DIVERSITY_MIN_MM (currently %.1fmm) "
                  "needs reviewing against the robot's actual home-to-home travel."
                  % HOME_DIVERSITY_MIN_MM)
    print()

    # ---- 2. headline discrepancy stats -----------------------------------
    raw = np.array([v for v in (_f(r, "discrepancy_raw_mm") for r in rows) if v is not None])
    corr = np.array([v for v in (_f(r, "discrepancy_corrected_mm") for r in rows) if v is not None])
    print("--- discrepancy (||mic_point - d405_point||) ---")
    for name, arr in (("raw (no ring->entry offset applied)", raw),
                       ("corrected (offset-compensated)", corr)):
        if len(arr) == 0:
            print("  %s: no data" % name)
            continue
        ci = ci95_t(arr)
        print("  %s: n=%d  mean=%.4fmm  std=%.4fmm  median=%.4fmm  "
              "min=%.4fmm  max=%.4fmm  95%%CI=[%.4f, %.4f]mm" %
              (name, len(arr), arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0.0,
               np.median(arr), arr.min(), arr.max(),
               arr.mean() - ci, arr.mean() + ci))
    print()

    # ---- 3. Bland-Altman per-axis + magnitude -----------------------------
    axes = ["x", "y", "z"]
    ba_results = {}
    have_xyz = all(_f(rows[0], "mic_%s_mm" % ax) is not None or True for ax in axes)
    print("--- Bland-Altman agreement (mic vs D405-corrected, per axis) ---")
    mic_xyz = {ax: np.array([v for v in (_f(r, "mic_%s_mm" % ax) for r in rows) if v is not None]) for ax in axes}
    d405_xyz = {ax: np.array([v for v in (_f(r, "d405_corrected_%s_mm" % ax) for r in rows) if v is not None]) for ax in axes}
    for ax in axes:
        a, b = mic_xyz[ax], d405_xyz[ax]
        n = min(len(a), len(b))
        if n < 2:
            print("  %s-axis: insufficient paired data (n=%d)" % (ax, n))
            continue
        a, b = a[:n], b[:n]
        mean, diff, bias, lo, hi = bland_altman(a, b)
        ba_results[ax] = (mean, diff, bias, lo, hi)
        print("  %s-axis: n=%d  bias=%+.4fmm  95%% limits of agreement=[%+.4f, %+.4f]mm"
              % (ax, n, bias, lo, hi))
    if corr is not None and len(corr) >= 2:
        ba_mag_mean, ba_mag_diff, ba_mag_bias, ba_mag_lo, ba_mag_hi = \
            bland_altman(corr, np.zeros_like(corr))
        print("  magnitude (discrepancy vs. zero -- 'how far apart, period'): "
              "n=%d  mean=%.4fmm  95%% upper limit=%.4fmm"
              % (len(corr), corr.mean(), corr.mean() + 1.96 * (corr.std(ddof=1) if len(corr) > 1 else 0.0)))
    print()

    # ---- 4. generalization sanity check (home spread vs discrepancy) -----
    print("--- generalization check: does discrepancy depend on starting home? ---")
    home_dist_from_first, disc_for_corr = [], []
    first_home = None
    for r in rows:
        x, y, z = _f(r, "home_x_mm"), _f(r, "home_y_mm"), _f(r, "home_z_mm")
        if x is None:
            continue
        if first_home is None:
            first_home = np.array([x, y, z])
            continue
        d_corr = _f(r, "discrepancy_corrected_mm")
        if d_corr is None:
            continue
        home_dist_from_first.append(float(np.linalg.norm(np.array([x, y, z]) - first_home)))
        disc_for_corr.append(d_corr)
    if len(home_dist_from_first) >= 3:
        home_dist_from_first = np.array(home_dist_from_first)
        disc_for_corr = np.array(disc_for_corr)
        if _scipy_stats is not None:
            r_val, p_val = _scipy_stats.pearsonr(home_dist_from_first, disc_for_corr)
            print("  Pearson r(home-distance-from-trial-1, discrepancy) = %.3f (p=%.3f), n=%d"
                  % (r_val, p_val, len(home_dist_from_first)))
        else:
            r_val = float(np.corrcoef(home_dist_from_first, disc_for_corr)[0, 1])
            print("  Pearson r(home-distance-from-trial-1, discrepancy) = %.3f, n=%d (scipy "
                  "unavailable, no p-value)" % (r_val, len(home_dist_from_first)))
        print("  !! CAVEAT: n is small (~10) -- a non-significant/near-zero r here is weak "
              "evidence FOR generalization, not proof of it. Report as 'no detected "
              "dependence on starting configuration in this campaign', not 'proven "
              "pose-invariant'.")
    else:
        print("  not enough paired (home, discrepancy) rows to compute (need >=3, have %d)"
              % len(home_dist_from_first))
    print()

    print("--- honest limitations (for the writeup) ---")
    print("  - single-session and/or single-mount unless this CSV spans multiple physical")
    print("    microscope re-mounts between trials -- check home-position diversity above")
    print("    is not a substitute for an actual re-mount/re-calibration diversity claim.")
    print("  - n~10 trials gives wide 95% CIs on the discrepancy mean; do not round down")
    print("    reported CIs when writing the paper's numbers.")
    print("  - Bland-Altman limits of agreement assume differences are roughly normally")
    print("    distributed and variance-homogeneous across the measurement range; with")
    print("    n~10 this assumption itself is only weakly checkable -- say so explicitly")
    print("    rather than silently relying on it.")

    if not no_plot:
        try:
            _make_plots(rows, raw, corr, ba_results, path)
        except Exception as e:
            print("\n  !! plot generation failed (non-fatal, stats above are unaffected): %s" % e)


def _make_plots(rows, raw, corr, ba_results, csv_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = os.path.splitext(csv_path)[0] + "_report.png"
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))

    # panel 1: per-trial raw vs corrected discrepancy
    ax0 = axs[0, 0]
    trial_idx = list(range(1, len(rows) + 1))
    raw_full = [(_f(r, "discrepancy_raw_mm")) for r in rows]
    corr_full = [(_f(r, "discrepancy_corrected_mm")) for r in rows]
    ax0.plot(trial_idx, [v if v is not None else np.nan for v in raw_full],
             "o-", color="tab:orange", label="raw", alpha=0.8)
    ax0.plot(trial_idx, [v if v is not None else np.nan for v in corr_full],
             "s-", color="tab:green", label="corrected", alpha=0.8)
    ax0.axhline(0.5, color="gray", ls="--", lw=1, label="0.5mm reference")
    ax0.set_xlabel("trial (row order)")
    ax0.set_ylabel("||mic - D405|| discrepancy (mm)")
    ax0.set_title("Per-trial discrepancy")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)

    # panel 2: histogram of corrected discrepancy
    ax1 = axs[0, 1]
    if len(corr) > 0:
        ax1.hist(corr, bins=min(10, max(3, len(corr) // 2)), color="tab:green", alpha=0.7)
        ax1.axvline(float(np.mean(corr)), color="k", ls="-", lw=1.5, label="mean")
    ax1.set_xlabel("corrected discrepancy (mm)")
    ax1.set_ylabel("count")
    ax1.set_title("Distribution of corrected discrepancy")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # panel 3: Bland-Altman (magnitude vs zero, i.e. simple discrepancy-vs-mean)
    ax2 = axs[1, 0]
    if len(corr) > 0:
        mean_idx = np.arange(len(corr))
        ax2.scatter(mean_idx, corr, color="tab:blue")
        bias = float(corr.mean())
        sd = float(corr.std(ddof=1)) if len(corr) > 1 else 0.0
        ax2.axhline(bias, color="k", lw=1.5, label="mean=%.3fmm" % bias)
        ax2.axhline(bias + 1.96 * sd, color="r", ls="--", lw=1, label="+1.96SD")
        ax2.axhline(max(bias - 1.96 * sd, 0), color="r", ls="--", lw=1, label="-1.96SD")
    ax2.set_xlabel("trial")
    ax2.set_ylabel("corrected discrepancy (mm)")
    ax2.set_title("Bland-Altman-style limits (magnitude)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # panel 4: per-axis Bland-Altman bias + LoA
    ax3 = axs[1, 1]
    axes_names = list(ba_results.keys())
    if axes_names:
        biases = [ba_results[a][2] for a in axes_names]
        los = [ba_results[a][3] for a in axes_names]
        his = [ba_results[a][4] for a in axes_names]
        xs = np.arange(len(axes_names))
        ax3.errorbar(xs, biases, yerr=[np.array(biases) - np.array(los), np.array(his) - np.array(biases)],
                     fmt="o", capsize=5, color="tab:purple")
        ax3.axhline(0, color="gray", ls="--", lw=1)
        ax3.set_xticks(xs)
        ax3.set_xticklabels(["%s-axis" % a for a in axes_names])
    ax3.set_ylabel("mic - D405 bias (mm), with 95% LoA")
    ax3.set_title("Bland-Altman per-axis bias")
    ax3.grid(alpha=0.3)

    fig.suptitle("Microscope <-> D405 RCM-point correspondence campaign", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    print("\n  figure saved: %s" % out_png)


if __name__ == "__main__":
    main()
