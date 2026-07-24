#!/usr/bin/env python3
"""Push ply-s6e7-ft.ipynb to Kaggle, poll until it finishes, and append the
run's metrics (parsed from the notebook's RUN_METRICS_JSON print) to
experiments/runs.csv.

Usage:
    python scripts/collect_run.py --description "seed bagging + lightgbm + calibration"
    python scripts/collect_run.py --description "..." --no-push   # kernel already running
    python scripts/collect_run.py --description "..." --submit    # also submit to the leaderboard

Designed to be run via the harness's background Bash execution -- a full
push -> queue -> GPU run -> complete cycle can take a long time.
"""
import argparse, csv, json, re, shutil, subprocess, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_METADATA = REPO_ROOT / "kernel-metadata.json"
RUNS_CSV = REPO_ROOT / "experiments" / "runs.csv"

RUN_COLUMNS = [
    "run_id", "timestamp_utc", "git_commit", "git_dirty", "kernel_ref", "kernel_version",
    "description", "n_seeds", "hpo_sample_frac", "notebook_runtime_sec",
    "xgb_oof", "cat_oof", "lr_oof", "ft_oof", "lgb_oof", "mlp_oof",
    "meta_oof_raw", "meta_oof_calibrated", "calibration_used",
    "class_weight_w0", "class_weight_w1", "class_weight_w2", "class_weight_method",
    "final_oof_bal_acc", "public_lb_score", "notes", "preds_dir",
]

# Prediction artifacts the notebook writes alongside submission.csv; archived
# per run so ply-s6e7-blend.ipynb (and the probe engine) can ensemble across
# submissions later. Includes the per-base-learner probability files the stack
# emits (test_proba_<learner>.csv / oof_proba_<learner>.csv) so each learner is
# usable as an independent blend source.
PRED_FILES = ["submission.csv", "test_proba.csv", "oof_proba.csv"]
for _lk in ("xgb", "mlp", "ftplr", "catnat", "minority", "node", "tabm", "grande",
            "lgbte", "catte", "hgbte", "et", "ftplr2", "cnn1d", "nca"):
    PRED_FILES += [f"test_proba_{_lk}.csv", f"oof_proba_{_lk}.csv"]


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)


def git_commit():
    out = run(["git", "rev-parse", "HEAD"])
    return out.stdout.strip() if out.returncode == 0 else ""


def git_dirty():
    out = run(["git", "status", "--porcelain"])
    return bool(out.stdout.strip())


def kernel_ref():
    meta = json.loads(KERNEL_METADATA.read_text(encoding="utf-8"))
    return meta["id"]


def push_kernel():
    out = run(["kaggle", "kernels", "push", "-p", str(REPO_ROOT)])
    print(out.stdout); print(out.stderr, file=sys.stderr)
    if out.returncode != 0:
        raise RuntimeError(f"kaggle kernels push failed: {out.stderr}")
    # Best-effort: kaggle-cli's exact push-success wording isn't guaranteed across
    # versions, so this may return None -- that's fine, only --submit needs a
    # version and falls back to submitting the kernel's latest version instead.
    m = re.search(r"[Vv]ersion\s+(\d+)", out.stdout + out.stderr)
    return m.group(1) if m else None


def poll_status(ref, poll_interval, timeout_min):
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        out = run(["kaggle", "kernels", "status", ref])
        text = (out.stdout + out.stderr).strip()
        m = re.search(r"KernelWorkerStatus\.(\w+)", text)
        status = m.group(1) if m else text
        print(f"  status: {status}")
        if status in ("COMPLETE", "ERROR", "CANCELLED"):
            return status
        time.sleep(poll_interval)
    raise TimeoutError(f"Kernel did not finish within {timeout_min} minutes")


def _extract_stdout_text(raw_bytes):
    """Kaggle's kernel .log file is a JSON array of {stream_name, time, data}
    entries, not plain text -- reassemble the stdout stream into one string.
    Falls back to the raw decoded text if the file isn't in that format."""
    text = raw_bytes.decode("utf-8", errors="replace")
    try:
        entries = json.loads(text)
    except json.JSONDecodeError:
        return text
    return "".join(e.get("data", "") for e in entries if e.get("stream_name") in ("stdout", None))


def pull_run_metrics(ref, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    out = run(["kaggle", "kernels", "output", ref, "-p", str(workdir), "--force"])
    print(out.stdout); print(out.stderr, file=sys.stderr)
    if out.returncode != 0:
        raise RuntimeError(f"kaggle kernels output failed: {out.stderr}")
    pattern = re.compile(r"RUN_METRICS_JSON:(\{.*\})")
    for p in workdir.rglob("*"):
        if not p.is_file():
            continue
        try:
            raw = p.read_bytes()
        except Exception:
            continue
        text = _extract_stdout_text(raw)
        m = pattern.search(text)
        if m:
            return json.loads(m.group(1))
    raise RuntimeError(f"RUN_METRICS_JSON not found in any file under {workdir}")


def submit_and_get_score(competition, ref, version, message, poll_interval, timeout_min,
                         file_fallback=None):
    # Prefer submitting the pulled submission.csv directly: kernel-based
    # submission has failed for this competition both without a version
    # ("version required") and with one (400 on CreateCodeSubmission), while
    # file submission works -- playground comps accept plain files.
    if file_fallback is not None and Path(file_fallback).is_file():
        print(f"  Submitting file directly: {file_fallback}")
        cmd = ["kaggle", "competitions", "submit", competition, "-f", str(file_fallback), "-m", message]
    elif version:
        cmd = ["kaggle", "competitions", "submit", competition, "-k", ref, "-v", version, "-m", message]
    else:
        cmd = ["kaggle", "competitions", "submit", competition, "-k", ref, "-m", message]
    out = run(cmd)
    print(out.stdout); print(out.stderr, file=sys.stderr)
    if out.returncode != 0:
        raise RuntimeError(f"kaggle competitions submit failed: {out.stderr}")

    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        out = run(["kaggle", "competitions", "submissions", competition, "--csv"])
        rows = list(csv.DictReader(out.stdout.splitlines()))
        if rows:
            latest = rows[0]  # assumed most-recent-first, matches the web UI ordering
            score = latest.get("publicScore") or latest.get("public_score")
            if score not in (None, "", "None", "pending"):
                return score
        time.sleep(poll_interval)
    print("  WARNING: timed out waiting for a public score; leaving public_lb_score blank")
    return ""


def archive_preds(workdir, run_id):
    """Copy the run's prediction artifacts into experiments/preds/<run_id>/ so
    they survive later kernel versions (Kaggle only serves the latest output).
    Returns the repo-relative dir, or '' if nothing was found to archive."""
    dest = REPO_ROOT / "experiments" / "preds" / run_id
    copied = []
    for name in PRED_FILES:
        matches = sorted(p for p in workdir.rglob(name) if p.is_file())
        if matches:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(matches[0], dest / name)
            copied.append(name)
    if not copied:
        print(f"  WARNING: no prediction artifacts ({', '.join(PRED_FILES)}) found under {workdir}")
        return ""
    print(f"  Archived {', '.join(copied)} -> {dest}")
    return dest.relative_to(REPO_ROOT).as_posix()


def append_run_row(row):
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not RUNS_CSV.exists()
    with RUNS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--description", required=True, help="Short note on what changed this run")
    ap.add_argument("--no-push", action="store_true", help="Skip pushing; poll an already-running kernel")
    ap.add_argument("--submit", action="store_true", help="Also submit submission.csv to the leaderboard")
    ap.add_argument("--competition", default="playground-series-s6e7")
    ap.add_argument("--poll-interval", type=int, default=60, help="Seconds between status checks")
    ap.add_argument("--timeout-min", type=int, default=180, help="Max minutes to wait for completion")
    ap.add_argument("--notes", default="", help="Freeform notes column")
    args = ap.parse_args()

    ref = kernel_ref()
    commit = git_commit()
    dirty = git_dirty()
    if dirty:
        print("WARNING: working tree has uncommitted changes; the pushed kernel may not match HEAD.")

    version = None
    if not args.no_push:
        version = push_kernel()
        print(f"Pushed. Detected version: {version!r} (best-effort parse; may be None)")

    print(f"Polling {ref} for completion...")
    status = poll_status(ref, args.poll_interval, args.timeout_min)
    if status != "COMPLETE":
        raise RuntimeError(f"Kernel finished with status {status}, not COMPLETE -- not logging a run row.")

    workdir = REPO_ROOT / ".kaggle_output" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics = pull_run_metrics(ref, workdir)
    print("Parsed RUN_METRICS_JSON:", json.dumps(metrics, indent=2))

    run_id = uuid.uuid4().hex[:8]
    preds_dir = archive_preds(workdir, run_id)

    score = ""
    if args.submit:
        sub_file = next(iter(sorted(p for p in workdir.rglob("submission.csv") if p.is_file())), None)
        score = submit_and_get_score(args.competition, ref, version, args.description,
                                      args.poll_interval, args.timeout_min, file_fallback=sub_file)
        print(f"Public LB score: {score}")

    base = metrics.get("base_oof_bal_acc", {})
    cw = metrics.get("class_weights", [None, None, None])
    row = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "kernel_ref": ref,
        "kernel_version": version or "",
        "description": args.description,
        "n_seeds": metrics.get("n_seeds", ""),
        "hpo_sample_frac": metrics.get("hpo_sample_frac", ""),
        "notebook_runtime_sec": metrics.get("notebook_runtime_sec", ""),
        "xgb_oof": base.get("xgb", ""),
        "cat_oof": base.get("cat", ""),
        "lr_oof": base.get("lr", ""),
        "ft_oof": base.get("ftplr", ""),   # FT-PLR (v11+) reuses the historical ft_oof column
        "lgb_oof": base.get("lgb", ""),
        "mlp_oof": base.get("mlp", ""),
        "meta_oof_raw": metrics.get("meta_oof_bal_acc_raw", ""),
        "meta_oof_calibrated": metrics.get("meta_oof_bal_acc_calibrated", ""),
        "calibration_used": metrics.get("calibration_used", ""),
        "class_weight_w0": cw[0] if len(cw) > 0 else "",
        "class_weight_w1": cw[1] if len(cw) > 1 else "",
        "class_weight_w2": cw[2] if len(cw) > 2 else "",
        "class_weight_method": metrics.get("class_weight_method", ""),
        "final_oof_bal_acc": metrics.get("final_oof_bal_acc", ""),
        "public_lb_score": score,
        "notes": args.notes,
        "preds_dir": preds_dir,
    }
    append_run_row(row)
    print(f"\nAppended run {row['run_id']} to {RUNS_CSV}")


if __name__ == "__main__":
    main()
