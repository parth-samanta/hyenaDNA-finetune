"""Generate runs, execute them, and select configs using validation scores."""
import argparse
import csv
import copy
import json
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from hyenadna.config import load_config
from hyenadna.data import TASKS, write_json

HEADS = ["mean", "max", "last", "mean_max_last", "attention", "multi_attention", "regional", "cnn",
         "regional_attention", "residual_regional_attention"]


def generate(base, output, tasks, heads, seeds, methods=("frozen",), trials=1):
    cfg = load_config(base)
    out = Path(output).resolve()
    if (out / "manifest.json").exists():
        raise FileExistsError("Choose a new experiment folder")
    if trials < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Need positive trials and unique seeds")
    if not tasks or not heads or not methods or set(methods) - {"frozen", "partial", "lora"}:
        raise ValueError("Choose tasks, heads and frozen/partial/lora methods")
    if any(len(set(xs)) != len(xs) for xs in (tasks, heads, methods)):
        raise ValueError("Remove duplicate tasks, heads or methods")
    if cfg["bidirectional"] or cfg["rc_augmentation"] or cfg["tapt_epochs"] or cfg["model"] != "hyena":
        raise ValueError("Keep the aggregation study on single-view HyenaDNA without TAPT or augmentation")
    for key in ("data", "checkpoint", "model_config"):
        if cfg[key]:
            cfg[key] = str(Path(cfg[key]).resolve())
    out.mkdir(parents=True, exist_ok=True)
    # Every head sees the same learning-rate candidates and seeds.
    rates = [cfg["head_lr"]] + random.Random(42).choices([2e-4, 5e-4, 1e-3, 2e-3], k=trials - 1)
    manifest = []
    for task in tasks:
        for method in methods:
            for head in heads:
                for trial, lr in enumerate(rates):
                    family = "/".join([task, method, head])
                    group = family + "/trial_%02d" % trial
                    for seed in seeds:
                        run = out / "runs" / group / ("seed_%d" % seed)
                        item = copy.deepcopy(cfg)
                        item.update(name=group, output=str(run), tasks=[task], method=method, head=head,
                                    seed=seed, head_lr=lr, max_length=max(cfg["max_length"], TASKS.get(task, cfg["max_length"])),
                                    cache_features=(method == "frozen" and head in {"mean", "max", "last", "mean_max_last", "regional", "local_global", "site_regions"}))
                        item = load_config(item)
                        config_path = out / "configs" / group / ("seed_%d.json" % seed)
                        config_path.parent.mkdir(parents=True, exist_ok=True)
                        write_json(config_path, item)
                        manifest.append(dict(family=family, group=group, seed=seed,
                                             config=str(config_path), output=str(run)))
    write_json(out / "manifest.json", manifest)
    print("Generated", len(manifest), "runs; none started.")
    return out / "manifest.json"

def archive_interrupted(out, cfg, study):
    """Preserve a recognized interrupted run before restarting from epoch one."""
    out, study = Path(out).resolve(), Path(study).resolve()
    runs = study / "runs"
    if not out.is_relative_to(runs) or out == runs or Path(cfg["output"]).resolve() != out:
        raise ValueError("Interrupted output must be a run directory inside this study: " + str(out))
    if (out / "complete.json").exists():
        raise ValueError("Cannot archive a completed run")
    recorded = out / "config.json"
    if not recorded.exists() or load_config(recorded) != cfg:
        raise FileExistsError("Existing output is not an interrupted run with the same config: " + str(out))
    archives = (study / "interrupted").resolve()
    if not archives.is_relative_to(study) or archives == study:
        raise ValueError("Interrupted archives must stay inside this study")
    archives.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = archives / ("%s_%s_seed%d_%s_%s" %
        (cfg["method"], cfg["head"], cfg["seed"], stamp, uuid4().hex[:8]))
    out.rename(archive)
    print("Preserved interrupted run:", archive, flush=True)
    print("Restarting this run from epoch 1; optimizer state was not saved.", flush=True)
    return archive


def run_manifest(path):
    from hyenadna.engine import fit
    from filelock import FileLock, Timeout
    path = Path(path).resolve()
    rows = json.loads(path.read_text())
    lock = FileLock(str(path.parent / ".sweep.lock"), timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise RuntimeError("Another process is running this study. Stop it before restarting the sweep.") from exc
    try:
        for i, row in enumerate(rows):
            out = Path(row["output"])
            if (out / "complete.json").exists():
                print("Already complete:", out)
                continue
            if out.exists() and any(out.iterdir()):
                archive_interrupted(out, load_config(row["config"]), path.parent)
            print("Run %d/%d: %s" % (i + 1, len(rows), row["group"]), flush=True)
            fit(row["config"])
    finally:
        lock.release()


def report(path):
    path = Path(path)
    rows = json.loads(path.read_text())
    if not rows:
        raise ValueError("Empty manifest")
    groups, families = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    validation = {}
    for group, runs in groups.items():
        if not all((Path(r["output"]) / "complete.json").exists() for r in runs):
            raise ValueError("Incomplete group: " + group)
        vals = [json.loads((Path(r["output"]) / "validation.json").read_text())["selection_score"] for r in runs]
        validation[group] = dict(mean=statistics.mean(vals), std=statistics.stdev(vals) if len(vals) > 1 else None,
                                 seeds=[r["seed"] for r in runs])
        families[runs[0]["family"]].append(group)
    winners = {f: sorted(gs, key=lambda g: (-validation[g]["mean"], g))[0] for f, gs in families.items()}
    selected = [r for r in rows if r["group"] in winners.values()]
    results = {}
    for family, winner in winners.items():
        runs = groups[winner]
        if not all((Path(r["output"]) / "test.json").exists() for r in runs):
            continue
        values = defaultdict(list)
        for r in runs:
            metrics = json.loads((Path(r["output"]) / "test.json").read_text())["metrics"]
            for task, scores in metrics.items():
                for key, value in scores.items():
                    if key not in {"n", "positives"}:
                        values[task + "/" + key].append(value)
        results[family] = {k: (None if any(x is None for x in vs) else
            dict(mean=statistics.mean(vs), std=statistics.stdev(vs) if len(vs) > 1 else None, n=len(vs)))
            for k, vs in values.items()}
    selection_path = path.parent / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selected:
        raise ValueError("Selection is already saved. Use a new experiment directory for changed selection.")
    write_json(selection_path, selected)
    write_json(path.parent / "summary.json", dict(validation=validation, winners=winners, test=results))
    comparison = []
    for family, group in winners.items():
        run_rows = groups[group]
        config = json.loads(Path(run_rows[0]["config"]).read_text())
        cost = [json.loads((Path(r["output"]) / "complete.json").read_text()) for r in run_rows]
        row = dict(task=config["tasks"][0], method=config["method"], aggregation=config["head"],
            seeds=len(run_rows), selected_head_lr=config["head_lr"],
            validation_f1_mean=validation[group]["mean"], validation_f1_std=validation[group]["std"],
            trainable_parameters=cost[0]["trainable_parameters"],
            aggregation_parameters=cost[0]["aggregation_parameters"],
            projection_parameters=cost[0]["projection_parameters"],
            seconds_mean=statistics.mean(c["elapsed_seconds"] for c in cost))
        for metric in ("f1_macro", "f1_binary", "mcc", "auroc", "auprc", "accuracy"):
            result = results.get(family, {}).get(row["task"] + "/" + metric)
            row["test_" + metric + "_mean"] = result["mean"] if result else None
            row["test_" + metric + "_std"] = result["std"] if result else None
        comparison.append(row)
    with open(path.parent / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    print("Saved", path.parent / "summary.json")


def test_selected(path):
    from hyenadna.engine import test
    selection = Path(path).parent / "selection.json"
    if not selection.exists():
        raise FileNotFoundError("Run report first to freeze validation-based selection")
    rows = json.loads(selection.read_text())
    for row in rows:
        if not (Path(row["output"]) / "test.json").exists():
            test(row["output"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["generate", "run", "report", "test-selected"])
    p.add_argument("--manifest", default="experiments/aggregation/manifest.json")
    p.add_argument("--base", default="configs/frozen_mean.json")
    p.add_argument("--tasks", nargs="+", default=["promoter_all", "splice_sites_acceptor"])
    p.add_argument("--heads", nargs="+", default=HEADS)
    p.add_argument("--methods", nargs="+", choices=["frozen", "partial", "lora"], default=["frozen"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--trials", type=int, default=1)
    a = p.parse_args()
    if a.action == "generate":
        generate(a.base, Path(a.manifest).parent, a.tasks, a.heads, a.seeds, a.methods, a.trials)
    else:
        {"run": run_manifest, "report": report, "test-selected": test_selected}[a.action](a.manifest)


