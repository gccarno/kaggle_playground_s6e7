#!/usr/bin/env python
"""Build a single-model submission from one run's per-learner probability artifacts.

One multi-learner kernel run archives {oof,test}_proba_<learner>.csv per base learner
(experiments/preds/<run_id>/), so each learner's LB score costs zero extra GPU: this
script turns one learner's fold-averaged test probabilities into a submission.csv,
after running the same per-class decision-weight search the main notebook uses on
that learner's own OOF (adopted only if it beats raw argmax by WEIGHT_MARGIN).

Usage:
  python scripts/make_single_sub.py --run 447f3b7c --learner ftplr
  python scripts/make_single_sub.py --run 447f3b7c --learner xgb --submit

Writes experiments/singles/<run_id>_<learner>/submission.csv (+ manifest) and appends
a row to experiments/runs.csv. With --submit it also submits the file directly
(kaggle competitions submit -f ...); fill public_lb_score into runs.csv afterwards.
"""
import argparse, csv, json, subprocess, sys, uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_CSV = REPO_ROOT / "experiments" / "runs.csv"
WEIGHT_MARGIN = 5e-4  # same adopt-only-if-clearly-better guard as the notebooks


def optimize_class_weights(y_true, proba, n_rounds=3, grid=np.linspace(0.75, 1.35, 25)):
    """Coordinate-ascent per-class decision weights (copied from ply-s6e7-ft.ipynb)."""
    w = np.ones(proba.shape[1])
    best_score = balanced_accuracy_score(y_true, proba.argmax(1))
    for _ in range(n_rounds):
        improved = False
        for k in range(proba.shape[1]):
            best_wk, best_local = w[k], best_score
            for cand in grid:
                w_try = w.copy(); w_try[k] = cand
                s = balanced_accuracy_score(y_true, (proba * w_try).argmax(1))
                if s > best_local:
                    best_local, best_wk = s, cand
            if best_local > best_score:
                w[k] = best_wk; best_score = best_local; improved = True
        if not improved:
            break
    return w, best_score


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run_id under experiments/preds/ (or a preds dir path)")
    ap.add_argument("--learner", required=True,
                    help="base-learner suffix of the proba files (xgb, mlp, ftplr, node, ...); "
                         "'blend' uses the run's overall test_proba.csv/oof_proba.csv")
    ap.add_argument("--competition", default="playground-series-s6e7")
    ap.add_argument("--submit", action="store_true", help="also submit the file to the leaderboard")
    args = ap.parse_args()

    preds_dir = Path(args.run)
    if not preds_dir.is_dir():
        preds_dir = REPO_ROOT / "experiments" / "preds" / args.run
    if not preds_dir.is_dir():
        sys.exit(f"preds dir not found: {preds_dir}")
    run_id = preds_dir.name

    suffix = "" if args.learner == "blend" else f"_{args.learner}"
    oof_path, test_path = preds_dir / f"oof_proba{suffix}.csv", preds_dir / f"test_proba{suffix}.csv"
    for p in (oof_path, test_path):
        if not p.is_file():
            have = sorted(q.name for q in preds_dir.glob("*proba*.csv"))
            sys.exit(f"missing {p.name} in {preds_dir}\navailable: {have}")

    oof, test = pd.read_csv(oof_path), pd.read_csv(test_path)
    classes = [c for c in oof.columns if c not in ("id", "y_true")]
    y = oof["y_true"].map({c: i for i, c in enumerate(classes)}).values
    P_oof, P_test = oof[classes].values, test[classes].values

    ba_raw = balanced_accuracy_score(y, P_oof.argmax(1))
    w, ba_w = optimize_class_weights(y, P_oof)
    if ba_w >= ba_raw + WEIGHT_MARGIN:
        weights, method, ba_final = w, "search", ba_w
    else:
        weights, method, ba_final = np.ones(len(classes)), "none", ba_raw
    print(f"{run_id}/{args.learner}: OOF bal_acc raw={ba_raw:.5f}  "
          f"weighted={ba_w:.5f} (w={np.round(w, 3).tolist()})  adopted={method}")

    out_dir = REPO_ROOT / "experiments" / "singles" / f"{run_id}_{args.learner}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = (P_test * weights).argmax(1)
    sub = pd.DataFrame({"id": test["id"], "health_condition": [classes[i] for i in pred]})
    sub_path = out_dir / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"wrote {sub_path}")
    print(sub["health_condition"].value_counts(normalize=True).round(4).to_string())

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_run": run_id, "learner": args.learner,
        "oof_bal_acc_raw": round(float(ba_raw), 5),
        "class_weight_method": method,
        "class_weights": np.round(weights, 4).tolist(),
        "final_oof_bal_acc": round(float(ba_final), 5),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with RUNS_CSV.open(newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames
    row = {k: "" for k in fieldnames}
    row.update({
        "run_id": uuid.uuid4().hex[:8],
        "timestamp_utc": manifest["created_utc"],
        "description": f"single[{args.learner}] of {run_id}",
        "class_weight_method": method,
        "final_oof_bal_acc": manifest["final_oof_bal_acc"],
        "notes": "single-model file submission; fill public_lb_score after submitting",
        "preds_dir": out_dir.relative_to(REPO_ROOT).as_posix(),
    })
    for i, k in enumerate(("class_weight_w0", "class_weight_w1", "class_weight_w2")):
        if k in row and i < len(weights):
            row[k] = round(float(weights[i]), 4)
    with RUNS_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
    print(f"appended runs.csv row {row['run_id']}")

    if args.submit:
        msg = f"single {args.learner} from run {run_id} (OOF {ba_final:.5f})"
        cmd = ["kaggle", "competitions", "submit", args.competition, "-f", str(sub_path), "-m", msg]
        print("submitting:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print(f"\nto submit:\n  kaggle competitions submit {args.competition} "
              f"-f {sub_path} -m \"single {args.learner} from {run_id}\"")


if __name__ == "__main__":
    main()
