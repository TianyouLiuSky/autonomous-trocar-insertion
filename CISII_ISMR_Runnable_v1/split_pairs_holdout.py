#!/usr/bin/env python3
"""Split handeye_data/pairs.csv into a training set + a genuinely held-out
validation set, for use with handeye_reproj_refine.py's HANDEYE_VAL_CSV.

Why a random split isn't quite enough on its own: if your 100 poses were
collected as clusters (e.g. several orientations captured back-to-back at
each of ~20 positions), a purely random row-wise split will put highly
correlated near-duplicate poses on BOTH sides, making the validation set
look artificially easy (it's not truly "new" information). This script
optionally clusters by position first (--cluster-mm) and holds out whole
clusters, which is a fairer test of generalization to genuinely new poses.

Usage:
  python3 split_pairs_holdout.py [pairs.csv] [--frac 0.2] [--cluster-mm 5] [--seed 0]
Outputs (next to the input file):
  pairs_train.csv   (use as the 1st CLI arg to handeye_reproj_refine.py)
  pairs_val.csv      (point HANDEYE_VAL_CSV at this)
"""
from __future__ import print_function
import argparse
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_csv", nargs="?", default="handeye_data/pairs.csv")
    ap.add_argument("--frac", type=float, default=0.2, help="fraction of poses to hold out")
    ap.add_argument("--cluster-mm", type=float, default=5.0,
                     help="group poses within this FrameEE-position distance before splitting "
                          "(0 disables clustering, i.e. plain random row split)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.pairs_csv) as f:
        header = f.readline()
        rows = [line.strip() for line in f if line.strip()]
    data = np.array([[float(v) for v in r.split(",")] for r in rows])
    n = len(data)
    Pbe = data[:, 3:6]

    if args.cluster_mm > 0:
        cluster_id = -np.ones(n, dtype=int)
        next_id = 0
        for i in range(n):
            if cluster_id[i] >= 0:
                continue
            cluster_id[i] = next_id
            for j in range(i + 1, n):
                if cluster_id[j] < 0 and np.linalg.norm(Pbe[i] - Pbe[j]) < args.cluster_mm:
                    cluster_id[j] = next_id
            next_id += 1
        n_clusters = next_id
        rng = np.random.RandomState(args.seed)
        cluster_order = rng.permutation(n_clusters)
        n_val_clusters = max(1, int(round(n_clusters * args.frac)))
        val_clusters = set(cluster_order[:n_val_clusters].tolist())
        val_mask = np.array([cluster_id[i] in val_clusters for i in range(n)])
        print("  %d poses grouped into %d position-clusters (<%.1fmm apart); "
              "holding out %d whole clusters" % (n, n_clusters, args.cluster_mm, n_val_clusters))
    else:
        rng = np.random.RandomState(args.seed)
        n_val = max(1, int(round(n * args.frac)))
        val_idx = rng.choice(n, n_val, replace=False)
        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_idx] = True

    train_rows = [rows[i] for i in range(n) if not val_mask[i]]
    val_rows = [rows[i] for i in range(n) if val_mask[i]]

    outdir = os.path.dirname(os.path.abspath(args.pairs_csv))
    train_path = os.path.join(outdir, "pairs_train.csv")
    val_path = os.path.join(outdir, "pairs_val.csv")
    with open(train_path, "w") as f:
        f.write(header)
        f.write("\n".join(train_rows) + "\n")
    with open(val_path, "w") as f:
        f.write(header)
        f.write("\n".join(val_rows) + "\n")
    print("  train: %d poses -> %s" % (len(train_rows), train_path))
    print("  val:   %d poses -> %s" % (len(val_rows), val_path))
    print("\n  next: python3 handeye_reproj_refine.py %s handeye_T.npz handeye_T_reproj.npz" % train_path)
    print("        (with HANDEYE_VAL_CSV=%s set)" % val_path)


if __name__ == "__main__":
    main()
