"""Turn sorter JSONL logs into an accuracy table and per-stage latency (median / p95).

    python evaluate.py logs/offline_20261113_101500.jsonl
    python evaluate.py logs/*.jsonl --group device        # compare runs (CPU vs GPU vs NPU ...)
    python evaluate.py LOG --suggest-threshold            # best THRESHOLD for this model+device
    python evaluate.py LOG --markdown > results.md        # paste into the design write-up

Accuracy needs ground truth: offline runs log it from the folder names. For live
runs, pass --labels with one label per item in order (e.g. g,g,d,g) or skip it
and get latency only.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tabulate import tabulate

import settings

STAGES = ["pick", "capture", "detect", "infer_per_view", "decide", "recheck", "place", "total"]


def load(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["_log"] = Path(p).name
                rows.append(r)
    return rows


def defect_type(r):
    if r.get("source"):
        return Path(r["source"]).parent.name
    return r.get("label") or "?"


def accuracy_tables(rows, fmt):
    labelled = [r for r in rows if r.get("label") in ("good", "defect")]
    if not labelled:
        return "no ground-truth labels in these logs (latency only)\n"
    good_a, bad_a = settings.ACTION_FOR["good"], settings.ACTION_FOR["defect"]
    cm = Counter((r["label"], r["action"]) for r in labelled)
    correct = sum(1 for r in labelled if r["action"] == settings.ACTION_FOR[r["label"]])
    tp, fn = cm[("defect", bad_a)], sum(v for (lab, a), v in cm.items() if lab == "defect" and a != bad_a)
    fp, tn = sum(v for (lab, a), v in cm.items() if lab == "good" and a != good_a), cm[("good", good_a)]
    out = []
    out.append(tabulate(
        [["good", tn, fp], ["defect", fn, tp]],
        headers=["truth \\ action", good_a, bad_a], tablefmt=fmt))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    out.append(tabulate([[len(labelled), f"{correct / len(labelled):.1%}", f"{prec:.1%}", f"{rec:.1%}",
                          f"{tn / (tn + fp):.1%}" if tn + fp else "nan"]],
                        headers=["items", "accuracy", "defect precision", "defect recall", "good recall"],
                        tablefmt=fmt))
    per = defaultdict(lambda: [0, 0])
    for r in labelled:
        k = defect_type(r)
        per[k][0] += 1
        per[k][1] += r["action"] == settings.ACTION_FOR[r["label"]]
    out.append(tabulate([[k, n, f"{c / n:.1%}", f"{np.median([r['score'] for r in labelled if defect_type(r) == k]):.3f}"]
                         for k, (n, c) in sorted(per.items())],
                        headers=["type", "n", "correct", "median score"], tablefmt=fmt))
    wrong = [r for r in labelled if r["action"] != settings.ACTION_FOR[r["label"]]]
    if wrong:
        out.append("mistakes:\n" + "\n".join(
            f"  item {r['item']:4} {defect_type(r):15} score={r['score']:.3f} -> {r['action']}  {r.get('source') or ''}"
            for r in wrong[:15]))
    return "\n\n".join(out) + "\n"


def latency_table(rows, fmt, group=None):
    groups = defaultdict(list)
    for r in rows:
        groups[str(r.get(group)) if group else "all"].append(r)
    table = []
    for g, rs in sorted(groups.items()):
        for s in STAGES:
            vals = []
            for r in rs:
                v = r.get("ms", {}).get(s)
                if v is None:
                    continue
                vals.extend(v if isinstance(v, list) else [v])
            if vals:
                table.append([g, s, len(vals), f"{np.median(vals):.1f}", f"{np.percentile(vals, 95):.1f}",
                              f"{max(vals):.1f}"])
        tot = [r["ms"]["total"] for r in rs if "total" in r.get("ms", {})]
        if tot:
            table.append([g, "items/min", len(tot), f"{60000 / np.median(tot):.1f}", "", ""])
    return tabulate(table, headers=[group or "", "stage (ms)", "n", "median", "p95", "max"], tablefmt=fmt) + "\n"


def suggest_threshold(rows):
    labelled = [r for r in rows if r.get("label") in ("good", "defect")]
    if not labelled:
        return "need labelled rows to suggest a threshold\n"
    y = np.array([r["label"] == "defect" for r in labelled])
    s = np.array([r["score"] for r in labelled])
    if y.all() or not y.any():
        return "need both good and defect items to suggest a threshold\n"
    from sklearn.metrics import roc_auc_score

    u = np.unique(s)
    cands = (u[:-1] + u[1:]) / 2 if len(u) > 1 else u  # midpoints between neighbouring scores
    accs = np.array([((s >= t) == y).mean() for t in cands])
    best = accs.max()
    ok = cands[accs == best]
    mid_good, mid_bad = s[~y].max(), s[y].min()
    lines = [f"image AUROC {roc_auc_score(y, s):.4f}",
             f"current THRESHOLD {settings.THRESHOLD}: accuracy {((s >= settings.THRESHOLD) == y).mean():.1%}",
             f"best accuracy {best:.1%} for thresholds in [{ok.min():.3f}, {ok.max():.3f}]",
             f"highest good score {mid_good:.3f}, lowest defect score {mid_bad:.3f}"]
    if mid_bad > mid_good:
        lines.append(f"perfectly separable: suggest THRESHOLD = {(mid_good + mid_bad) / 2:.3f} (midpoint)")
    else:
        lines.append(f"suggest THRESHOLD = {float(np.median(ok)):.3f}; overlap -> more views, crop, or better data")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--group", default=None, help="group latency by a log field: device, model, _log, views")
    ap.add_argument("--labels", default=None, help="ground truth for live runs, comma separated g/d per item")
    ap.add_argument("--suggest-threshold", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    fmt = "github" if args.markdown else "simple"

    rows = load(args.logs)
    if args.labels:
        labels = [{"g": "good", "d": "defect"}.get(x.strip()[0].lower()) for x in args.labels.split(",")]
        for r, lab in zip(rows, labels):
            r["label"] = lab
    models = Counter((r.get("model"), r.get("device")) for r in rows)
    print(f"{len(rows)} items from {len(args.logs)} log(s); model/device: {dict(models)}\n")
    print("## Accuracy\n" + accuracy_tables(rows, fmt))
    print("## Latency\n" + latency_table(rows, fmt, args.group))
    if args.suggest_threshold:
        print("## Threshold\n" + suggest_threshold(rows))


if __name__ == "__main__":
    main()
