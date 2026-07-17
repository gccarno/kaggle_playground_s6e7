#!/usr/bin/env python3
"""Anchor+probe submission (E5, "signal-engine style"): take the best available
public submission as an immutable anchor and flip only a small set of test rows
where OUR models are highly reliable AND the public submission bank agrees the
anchor is the outlier. Hard fallback: the exact anchor when nothing clears the gates.

Evidence combined per candidate row (anchor_label != our_blend_argmax):
  - our blend confidence  : max prob of the 0.95085 blend (pub+fs+ftplr, 2:2:1)
  - our blend reliability : OOF precision (Wilson lower bound) of that blend for
                            (predicted class, confidence bin) -- how often we're
                            right when this confident, measured leak-free on OOF
  - bank consensus        : LB-score-weighted vote over the ~20 decoded public
                            submissions for our class vs the anchor's class
A row is flippable only if reliability >= REL_MIN, blend conf >= CONF_MIN, and the
bank net-supports our class over the anchor's by >= BANK_MARGIN. The top-N by a
combined score are flipped (N small; the anchor is already ~0.9511).

Nothing here is fit on test labels. Prints a report; writes a submission only with --apply.
"""
import argparse, glob, json, math, time
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
CLS = ["at-risk", "fit", "unhealthy"]; C2I = {c: i for i, c in enumerate(CLS)}
BANK = REPO / "experiments" / "bank"; LIB = BANK / "oof_lib"
V11 = REPO / "experiments" / "preds" / "447f3b7c"
N_TRAIN, N_TEST = 690088, 295753
ANCHOR = "0.95112.csv"
CONF_MIN, REL_MIN, BANK_MARGIN, N_MAX = 0.92, 0.97, 0.15, 7
CONF_BINS = np.array([0.0, 0.5, 0.7, 0.85, 0.92, 0.96, 1.0001])


def wilson_lower(succ, tot, z=1.645):
    if tot <= 0: return 0.0
    p = succ / tot; den = 1 + z*z/tot
    c = p + z*z/(2*tot); r = z*math.sqrt((p*(1-p) + z*z/(4*tot))/tot)
    return max(0.0, (c - r)/den)


def blend_probs(which):  # which in {'oof','test'}
    ft = pd.read_csv(V11 / f"{'oof' if which=='oof' else 'test'}_proba_ftplr.csv").sort_values("id")
    ftp = ft[CLS].values.astype(np.float64)
    suf = "oof" if which == "oof" else "test"
    pub = np.load(LIB / f"{suf}_pub_nawfeel_realmlp.npy").astype(np.float64)
    fs  = np.load(LIB / f"{suf}_realmlp_fs_clogit.npy").astype(np.float64)
    return (2*pub + 2*fs + 1*ftp) / 5.0, ft["id"].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write + (optionally) prepare submission")
    ap.add_argument("--n", type=int, default=N_MAX, help="max rows to flip")
    args = ap.parse_args()

    # our blend OOF (for reliability) + y_true
    oof, _ = blend_probs("oof")
    y = pd.read_csv(V11 / "oof_proba.csv").sort_values("id")["y_true"].map(C2I).values
    oof_pred = oof.argmax(1); oof_conf = oof.max(1)
    oof_bin = np.clip(np.digitize(oof_conf, CONF_BINS) - 1, 0, len(CONF_BINS) - 2)
    rel = {}  # (class, bin) -> Wilson lower bound of precision
    for c in range(3):
        for b in range(len(CONF_BINS) - 1):
            m = (oof_pred == c) & (oof_bin == b)
            rel[(c, b)] = wilson_lower(int(((y == c) & m).sum()), int(m.sum()))

    # our blend TEST proba
    test, tids = blend_probs("test")
    t_pred = test.argmax(1); t_conf = test.max(1)
    t_bin = np.clip(np.digitize(t_conf, CONF_BINS) - 1, 0, len(CONF_BINS) - 2)

    # anchor labels aligned to tids
    anc = pd.read_csv(BANK / ANCHOR).set_index("id").reindex(tids)["health_condition"].map(C2I).values

    # public bank consensus (LB-score-weighted vote over the decoded vectors)
    man = pd.read_csv(BANK / "manifest.csv")
    man = man[man["source"] == "embedded_she"]
    vecs, scores = [], []
    for _, r in man.iterrows():
        v = pd.read_csv(BANK / r["name"]).set_index("id").reindex(tids)["health_condition"].map(C2I).values
        vecs.append(v); scores.append(float(r["lb_score"]))
    V = np.stack(vecs); scores = np.array(scores)
    w = np.exp((scores - scores.max()) / 0.0007); w /= w.sum()
    vote = np.zeros((N_TEST, 3))
    for c in range(3):
        vote[:, c] = ((V == c) * w[:, None]).sum(0)

    # candidates: our confident pick disagrees with the anchor
    cand = np.where(t_pred != anc)[0]
    rows = []
    for i in cand:
        c_ours, c_anc = t_pred[i], anc[i]
        r = rel[(c_ours, t_bin[i])]
        bank_net = vote[i, c_ours] - vote[i, c_anc]
        ok = (t_conf[i] >= CONF_MIN) and (r >= REL_MIN) and (bank_net >= BANK_MARGIN)
        rows.append((i, c_ours, c_anc, t_conf[i], r, bank_net, ok))
    df = pd.DataFrame(rows, columns=["idx", "ours", "anchor", "conf", "rel", "bank_net", "ok"])
    passed = df[df["ok"]].copy()
    passed["score"] = passed["rel"] * passed["conf"] * (0.5 + passed["bank_net"])
    passed = passed.sort_values("score", ascending=False)

    print(f"anchor={ANCHOR}  candidates(disagree)={len(df)}  clearing gates={len(passed)}  "
          f"(CONF>={CONF_MIN}, REL>={REL_MIN}, BANK_MARGIN>={BANK_MARGIN})")
    print(f"reliability at (class,top-bin): " +
          ", ".join(f"{CLS[c]}={rel[(c,len(CONF_BINS)-2)]:.3f}" for c in range(3)))
    show = passed.head(args.n).copy()
    for _, r in show.iterrows():
        print(f"  id={tids[int(r['idx'])]:>7}  {CLS[r['anchor']]:>9} -> {CLS[r['ours']]:<9}  "
              f"conf={r['conf']:.3f} rel={r['rel']:.3f} bank_net={r['bank_net']:+.3f}")

    if not args.apply:
        return
    flips = passed.head(args.n)["idx"].astype(int).to_numpy()
    out_labels = anc.copy()
    out_labels[flips] = t_pred[flips]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = REPO / "experiments" / "probes" / stamp; out.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id": tids, "health_condition": [CLS[i] for i in out_labels]})
    sub.to_csv(out / "submission.csv", index=False)
    (out / "anchor.csv").write_bytes((BANK / ANCHOR).read_bytes())
    json.dump({"anchor": ANCHOR, "n_flipped": int(len(flips)),
               "flipped_ids": [int(tids[i]) for i in flips],
               "gates": {"conf": CONF_MIN, "rel": REL_MIN, "bank_margin": BANK_MARGIN}},
              open(out / "manifest.json", "w"), indent=2)
    print(f"\nwrote {out/'submission.csv'}  (flipped {len(flips)} of {N_TEST} rows vs anchor)")
    print(f"STAMP={stamp}")


if __name__ == "__main__":
    main()
