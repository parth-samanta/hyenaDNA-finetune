import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score


def scores(labels, probabilities):
    y, prob = np.asarray(labels), np.asarray(probabilities)
    pred = (prob > .5).astype(int)
    both = len(set(y)) == 2
    return dict(n=len(y), positives=int(y.sum()),
        f1_macro=float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        f1_binary=float(f1_score(y, pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y, pred)) if len(set(y) | set(pred)) > 1 else 0., accuracy=float(accuracy_score(y, pred)),
        auroc=float(roc_auc_score(y, prob)) if both else None,
        auprc=float(average_precision_score(y, prob)) if both else None)


def by_task(rows, probabilities):
    result = {}
    for task in sorted({r["task"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["task"] == task]
        result[task] = scores([rows[i]["label"] for i in idx], [probabilities[i] for i in idx])
    return result


def slices(rows, probabilities):
    result = {}
    for field in ("subtype", "species", "chrom", "length_bin", "gc_bin"):
        buckets = {}
        for i, r in enumerate(rows):
            if field == "length_bin":
                value = str((len(r["sequence"]) // 100) * 100)
            elif field == "gc_bin":
                seq = r["sequence"]
                known = len(seq) - seq.count("N")
                gc = (seq.count("G") + seq.count("C")) / max(1, known)
                value = str(min(9, int(gc * 10)) / 10)
            else:
                value = r.get(field, "")
            if value != "":
                buckets.setdefault((r["task"], value), []).append(i)
        for (task, value), idx in buckets.items():
            result["/".join([task, field, value])] = scores(
                [rows[i]["label"] for i in idx], [probabilities[i] for i in idx])
    return result

