import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

TASKS = {"promoter_all": 300, "promoter_non_tata": 300, "promoter_tata": 300,
         "splice_sites_acceptor": 600, "splice_sites_donor": 600}
VOCAB = dict(zip("ACGTN", range(7, 12)))
FIELDS = ["id", "sequence", "label", "task", "split", "group", "site", "subtype", "species", "chrom"]


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty data file: " + str(path))
    for r in rows:
        for key in ("id", "sequence", "label", "task", "split"):
            if not r.get(key):
                raise ValueError("Missing column/value: " + key)
        r["sequence"] = r["sequence"].upper()
        if set(r["sequence"]) - set(VOCAB):
            raise ValueError("Non-ACGTN bases: " + r["id"])
        r["label"] = int(r["label"])
        if r["label"] not in (0, 1):
            raise ValueError("This project expects binary labels")
        if r["split"] not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        r["site"] = int(r["site"]) if r.get("site", "") != "" else -1
        if r["site"] != -1 and not 0 <= r["site"] < len(r["sequence"]):
            raise ValueError("site is a zero-based nucleotide index: " + r["id"])
        for key in FIELDS:
            r.setdefault(key, "")
    if len({(r["task"], r["id"]) for r in rows}) != len(rows):
        raise ValueError("IDs must be unique within each task")
    return rows


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            r = dict(row)
            if r.get("site") == -1:
                r["site"] = ""
            w.writerow(r)


def groups_for(rows):
    # Connect exact duplicates and supplied groups, including across tasks.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, r in enumerate(rows):
        keys = [("seq", r["sequence"])]
        if r.get("group"):
            keys.append(("group", r["group"]))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    return [find(i) for i in range(len(rows))]


def prepare(rows, output, val_fraction=.1, seed=2026):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0,1)")
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError("Prepared data already exists: " + str(output))
    # Reject contradictory duplicates within a task; cross-task labels may differ.
    labels = {}
    for r in rows:
        key = (r["task"], r["sequence"])
        if key in labels and labels[key] != r["label"]:
            raise ValueError("Conflicting labels for an identical sequence in " + r["task"])
        labels[key] = r["label"]
    groups = groups_for(rows)
    occupied = defaultdict(set)
    for r, g in zip(rows, groups):
        occupied[g].add(r["split"])
    if any(len(s) > 1 for s in occupied.values()):
        raise ValueError("A sequence or group crosses supplied splits. Resolve overlap before preparing data.")
    if not any(r["split"] == "val" for r in rows):
        train_indices = [i for i, r in enumerate(rows) if r["split"] == "train"]
        strata = [rows[i]["task"] + ":" + str(rows[i]["label"]) for i in train_indices]
        counts = Counter(strata)
        best = None
        splitter = GroupShuffleSplit(n_splits=128, test_size=val_fraction, random_state=seed)
        for _, va in splitter.split(train_indices, groups=[groups[i] for i in train_indices]):
            vc = Counter(strata[j] for j in va)
            if any(vc[k] == 0 or vc[k] == n for k, n in counts.items()):
                continue
            score = sum(abs(vc[k] / n - val_fraction) for k, n in counts.items())
            if best is None or score < best[0]:
                best = score, va
        if best is None:
            raise ValueError("Cannot create a grouped validation split with both classes per task")
        for j in best[1]:
            rows[train_indices[j]]["split"] = "val"
    for task in sorted({r["task"] for r in rows}):
        for split in ("train", "val", "test"):
            subset = [r for r in rows if r["task"] == task and r["split"] == split]
            if {r["label"] for r in subset} != {0, 1}:
                raise ValueError("Need both classes in %s/%s" % (task, split))
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ("train", "val", "test"):
        write_rows(output / (split + ".csv"), [r for r in rows if r["split"] == split])
        files[split] = sha(output / (split + ".csv"))
    counts = Counter((r["task"], r["split"], r["label"]) for r in rows)
    write_json(output / "manifest.json", dict(files=files, split_seed=seed,
        counts={"/".join(map(str, k)): v for k, v in sorted(counts.items())},
        grouping="exact sequence plus supplied group; not automatic homology clustering"))


def load_split(root, split, tasks):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / (split + ".csv")
    if sha(path) != manifest["files"][split]:
        raise ValueError("Prepared data changed: " + str(path))
    rows = [r for r in read_rows(path) if r["task"] in tasks]
    if {r["task"] for r in rows} != set(tasks):
        raise ValueError("Missing requested tasks in " + split)
    return rows


def subsample(rows, fraction, seed):
    if fraction == 1:
        return rows
    grouped = defaultdict(list)
    for i, r in enumerate(rows):
        grouped[(r["task"], r["label"])].append(i)
    keep = []
    for key, indices in sorted(grouped.items()):
        indices.sort(key=lambda i: hashlib.sha256((str(seed) + rows[i]["id"] + rows[i]["task"]).encode()).hexdigest())
        keep.extend(indices[:max(1, round(len(indices) * fraction))])
    return [rows[i] for i in sorted(keep)]


class SequenceDataset(Dataset):
    def __init__(self, rows, tasks, length, rc=False):
        self.rows, self.tasks, self.length, self.rc = rows, tasks, length, rc
        if any(len(r["sequence"]) > length for r in rows):
            raise ValueError("Sequence exceeds max_length. Crop in data preparation, not in the loader.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        seq, site = r["sequence"], r["site"]
        if self.rc and torch.rand(()).item() < .5:
            seq = seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
            site = len(seq) - 1 - site if site >= 0 else site
        ids = torch.full((self.length,), 4, dtype=torch.long)
        ids[:len(seq)] = torch.tensor([VOCAB[c] for c in seq])
        return dict(ids=ids, mask=ids.ne(4), label=torch.tensor(r["label"]),
                    task=torch.tensor(self.tasks.index(r["task"])), site=torch.tensor(site), index=torch.tensor(i))


def reverse_complement(ids, mask):
    mapping = torch.tensor([0, 1, 2, 3, 4, 5, 6, 10, 9, 8, 7, 11], device=ids.device)
    lengths = mask.sum(1)
    positions = lengths[:, None] - 1 - torch.arange(ids.shape[1], device=ids.device)
    rc = mapping[ids.gather(1, positions.clamp_min(0))]
    return rc.masked_fill(~mask, 4)

