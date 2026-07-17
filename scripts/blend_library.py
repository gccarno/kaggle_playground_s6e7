#!/usr/bin/env python3
"""Blend our per-run probability artifacts with the public OOF library and/or
each other, choosing weights that generalize (fixed few-DOF blends + a bounded
simplex search validated on a held-out fold), and write a submission.

Why this is leakage-safe to blend: every source here is out-of-fold on the SAME
frozen split -- our stack uses StratifiedKFold(5, shuffle=True, random_state=42),
and szymonkapiski/s6e7-student-health-full-oof-library was built on the identical
split, so OOF row i is a held-out prediction for train row i in every source.
(Verified once: each library member's argmax OOF balanced accuracy reproduces its
manifest value exactly.)

Sources
  - our runs:   experiments/preds/<run_id>/{oof,test}_proba[_<learner>].csv
                (columns: id[,y_true], at-risk, fit, unhealthy)
  - library:    experiments/bank/oof_lib/{oof,test}_<name>.npy  (shape (N,3),
                columns [at-risk, fit, unhealthy], row order == train/test.csv)

Balanced accuracy is the metric (mean per-class recall). The decision rule is
plain argmax on the blended probabilities -- every source is already trained
class-balanced, so argmax is the right rule (matches how the LB-confirmed public
blends were scored). Fixed weights are preferred: OOF gains from many-DOF search
have historically NOT transferred to the LB here (see experiments/runs.csv notes),
whereas a fixed few-model blend (e.g. the public 0.6/0.4) does.

Usage:
  python scripts/blend_library.py \
      --sources v11:experiments/preds/447f3b7c/oof_proba_ftplr.csv \
                pub:oof_lib:pub_nawfeel_realmlp \
                fs:oof_lib:realmlp_fs_clogit \
      --weights v11=1,pub=2,fs=2 \
      --out experiments/blends
A source spec is NAME:PATH (our csv) or NAME:oof_lib:MEMBER (library .npy).
Omit --weights to only print an OOF report of singles + a small search.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CLS = ["at-risk", "fit", "unhealthy"]
C2I = {c: i for i, c in enumerate(CLS)}
LIB = REPO / "experiments" / "bank" / "oof_lib"
N_TRAIN, N_TEST = 690088, 295753


def _load_source(spec):
    """spec = 'NAME:PATH' (our oof csv; its test_ sibling is derived) or
    'NAME:oof_lib:MEMBER'. Returns (name, oof (N_TRAIN,3), test (N_TEST,3))."""
    parts = spec.split(":")
    name = parts[0]
    if parts[1] == "oof_lib":
        member = parts[2]
        oof = np.load(LIB / f"oof_{member}.npy").astype(np.float64)
        test = np.load(LIB / f"test_{member}.npy").astype(np.float64)
    else:
        oof_path = Path(parts[1])
        if not oof_path.is_absolute():
            oof_path = REPO / oof_path
        test_path = Path(str(oof_path).replace("oof_proba", "test_proba"))
        odf = pd.read_csv(oof_path).sort_values("id").reset_index(drop=True)
        tdf = pd.read_csv(test_path).sort_values("id").reset_index(drop=True)
        oof = odf[CLS].values.astype(np.float64)
        test = tdf[CLS].values.astype(np.float64)
    assert oof.shape == (N_TRAIN, 3) and test.shape == (N_TEST, 3), (name, oof.shape, test.shape)
    return name, oof, test


def _y_true():
    """Recover integer y_true from any of our archived oof_proba*.csv (has y_true),
    in train.csv id order."""
    for p in sorted((REPO / "experiments" / "preds").rglob("oof_proba*.csv")):
        df = pd.read_csv(p)
        if "y_true" in df.columns:
            df = df.sort_values("id").reset_index(drop=True)
            return df["y_true"].map(C2I).values.astype(np.int64)
    raise SystemExit("no archived oof_proba*.csv with a y_true column found")


def make_ba(y):
    masks = [y == k for k in range(3)]
    def ba(pred):
        return float(np.mean([(pred[m] == k).mean() for k, m in enumerate(masks)]))
    return ba


def blend(mats, w):
    w = np.asarray(w, float)
    return sum(wi * m for wi, m in zip(w, mats)) / w.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", required=True, help="NAME:PATH or NAME:oof_lib:MEMBER specs")
    ap.add_argument("--weights", default="", help="comma list NAME=w; omit to only report")
    ap.add_argument("--out", default="experiments/blends", help="dir to write the chosen blend under")
    ap.add_argument("--search", action="store_true", help="run a bounded held-out-validated simplex search")
    args = ap.parse_args()

    y = _y_true()
    ba = make_ba(y)
    names, oofs, tests = [], [], []
    for spec in args.sources:
        n, o, t = _load_source(spec)
        names.append(n); oofs.append(o); tests.append(t)
        print(f"  {n:24s} single OOF BA = {ba(o.argmax(1)):.5f}")

    if args.search:
        from sklearn.model_selection import StratifiedKFold
        tr, va = next(iter(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)))
        ba_tr, ba_va = make_ba(y[tr]), make_ba(y[va])
        rng = np.random.default_rng(0)
        best_w, best = np.ones(len(names)) / len(names), -1.0
        best_va = ba_va(blend([o[va] for o in oofs], best_w).argmax(1))
        for _ in range(4000):
            w = rng.dirichlet(np.full(len(names), 0.5))
            s = ba_tr(blend([o[tr] for o in oofs], w).argmax(1))   # fit on the 4-fold complement
            if s > best:
                va_s = ba_va(blend([o[va] for o in oofs], w).argmax(1))  # validate on held-out fold
                if va_s >= best_va:
                    best, best_w, best_va = s, w, va_s
        print(f"  search weights {dict(zip(names, best_w.round(3)))}  full-OOF BA "
              f"{ba(blend(oofs, best_w).argmax(1)):.5f}  (held-out {best_va:.5f})")

    if not args.weights:
        return
    wmap = dict(kv.split("=") for kv in args.weights.split(","))
    w = np.array([float(wmap[n]) for n in names])
    oof_ba = ba(blend(oofs, w).argmax(1))
    pred = blend(tests, w).argmax(1)
    print(f"\nchosen weights {dict(zip(names, w))}  ->  OOF BA {oof_ba:.5f}")

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = REPO / args.out / stamp
    out.mkdir(parents=True, exist_ok=True)
    tids = np.arange(N_TRAIN, N_TRAIN + N_TEST)
    sub = pd.DataFrame({"id": tids, "health_condition": [CLS[i] for i in pred]})
    sub.to_csv(out / "submission.csv", index=False)
    json.dump({"sources": args.sources, "weights": dict(zip(names, w.tolist())),
               "oof_bal_acc": round(oof_ba, 5)}, open(out / "manifest.json", "w"), indent=2)
    print(f"wrote {out/'submission.csv'}  ({sub['health_condition'].value_counts(normalize=True).round(4).to_dict()})")


if __name__ == "__main__":
    main()
