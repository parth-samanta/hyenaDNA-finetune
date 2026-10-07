"""One entry point: prepare, train, select, test, and export a study."""
import argparse
import copy
import csv
import json
import os
import random
import re
import shutil
from pathlib import Path

from filelock import FileLock, Timeout

from .config import load_config
from .data import TASKS
from .experiments import generate, report, run_manifest, test_selected


def find_project(project=None):
    if project is not None:
        root = Path(project).expanduser().resolve()
        if not (root / "pyproject.toml").is_file():
            raise FileNotFoundError("Project needs pyproject.toml: " + str(root))
        return root
    for start in (Path.cwd(), Path(__file__).resolve().parents[2]):
        for root in (start, *start.parents):
            if (root / "pyproject.toml").is_file() and (root / "configs/study.json").is_file():
                return root.resolve()
    raise FileNotFoundError("Run from the project folder or pass --project PATH")


def load_study(root, path="configs/study.json"):
    path = Path(path)
    path = (path if path.is_absolute() else root / path).resolve()
    supplied = json.loads(path.read_text(encoding="utf-8"))
    fields = {"name", "base", "tasks", "heads", "seeds", "methods", "lr_trials", "data_revision"}
    if set(supplied) != fields:
        raise ValueError("Study settings require exactly: " + ", ".join(sorted(fields)))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", supplied["name"]):
        raise ValueError("Study name must be a simple folder name")
    for key in ("tasks", "heads", "seeds", "methods"):
        values = supplied[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError("Study needs nonempty, unique " + key)
    for key in ("tasks", "heads", "methods"):
        if any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value)
               for value in supplied[key]):
            raise ValueError(key + " must contain simple names")
    if any(type(seed) is not int or seed < 0 for seed in supplied["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    if type(supplied["lr_trials"]) is not int or supplied["lr_trials"] < 1:
        raise ValueError("lr_trials must be a positive integer")
    if set(supplied["methods"]) - {"frozen", "partial", "lora"}:
        raise ValueError("Study supports frozen, partial or lora methods")
    if not isinstance(supplied["data_revision"], str) or not supplied["data_revision"]:
        raise ValueError("data_revision must be a nonempty revision")
    base_path = Path(supplied["base"])
    cfg = load_config(base_path if base_path.is_absolute() else root / base_path)
    from .paper_data import PAPER_REVISION
    if supplied["tasks"] != ["splice_sites_acceptor"] or supplied["data_revision"] != PAPER_REVISION:
        raise ValueError("Study requires the original acceptor dataset and pinned revision")
    data_name = "paper_splice_sites_acceptor"
    cfg["data"] = str(root / "data/prepared" / data_name)
    for key in ("checkpoint", "model_config"):
        if cfg[key]:
            value = Path(cfg[key])
            cfg[key] = str((value if value.is_absolute() else root / value).resolve())
    # Validate each requested head/method using the package's config contract.
    for method in supplied["methods"]:
        for head in supplied["heads"]:
            load_config(dict(cfg, method=method, head=head))
    if cfg["bidirectional"] or cfg["rc_augmentation"] or cfg["tapt_epochs"] or cfg["model"] != "hyena":
        raise ValueError("This comparison requires single-view HyenaDNA without TAPT/augmentation")
    return supplied, cfg


def validate_manifest(manifest, study, base):
    """Reject changed controls instead of mixing old results with new settings."""
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    rates = [base["head_lr"]] + random.Random(42).choices([2e-4, 5e-4, 1e-3, 2e-3], k=study["lr_trials"] - 1)
    expected = {}
    for task in study["tasks"]:
        for method in study["methods"]:
            for head in study["heads"]:
                for trial, lr in enumerate(rates):
                    family = "/".join([task, method, head])
                    group = family + "/trial_%02d" % trial
                    for seed in study["seeds"]:
                        cfg = copy.deepcopy(base)
                        output = manifest.parent / "runs" / group / ("seed_%d" % seed)
                        config = manifest.parent / "configs" / group / ("seed_%d.json" % seed)
                        cfg.update(name=group, output=str(output), tasks=[task], method=method,
                                   head=head, seed=seed, head_lr=lr,
                                   max_length=max(base["max_length"], TASKS.get(task, base["max_length"])),
                                   cache_features=(method == "frozen" and head in {
                                       "mean", "max", "last", "mean_max_last", "regional", "local_global", "site_regions"}))
                        expected[(group, seed)] = (load_config(cfg), str(config), str(output), family)
    seen = set()
    for row in rows:
        key = (row["group"], row["seed"])
        if key in seen or key not in expected:
            raise ValueError("Manifest does not match study settings; use a new study name")
        cfg, config, output, family = expected[key]
        if (Path(row["config"]).resolve() != Path(config) or
                Path(row["output"]).resolve() != Path(output) or row["family"] != family):
            raise ValueError("Manifest paths/families do not match this project: " + str(key))
        recorded = load_config(config)
        changed = sorted(k for k in cfg if cfg[k] != recorded[k])
        if changed:
            raise ValueError("Saved study controls changed (" + ", ".join(changed) + "); use a new study name")
        completed_config = Path(output) / "config.json"
        if (Path(output) / "complete.json").exists() and load_config(completed_config) != recorded:
            raise ValueError("Completed run config differs from manifest config: " + output)
        seen.add(key)
    if seen != set(expected):
        raise ValueError("Manifest is missing requested runs; use a new study name")
    return rows


def export_results(root, manifest, study_path="configs/study.json"):
    destination = root / "reports" / manifest.parent.name
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("comparison.csv", "summary.json"):
        shutil.copyfile(manifest.parent / name, destination / name)
    with (destination / "comparison.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    lines = ["# " + manifest.parent.name, "",
             "Validation selects checkpoints and LR trials; saved test results are reused.", "",
             "Mean +/- sample SD across seeds; all scores use the same prepared split.", "",
             "| Aggregation | Seeds | Validation macro F1 | Test macro F1 | Test AUROC | Trainable parameters |",
             "| --- | --- | --- | --- | --- | --- |"]
    print("\nFinal comparison (mean +/- sample SD across seeds):")
    for row in rows:
        validation = "%.4f +/- %s" % (float(row["validation_f1_mean"]),
                       "%.4f" % float(row["validation_f1_std"]) if row["validation_f1_std"] else "N/A (one seed)")
        test = ("%.4f +/- %s" % (float(row["test_f1_macro_mean"]),
                "%.4f" % float(row["test_f1_macro_std"]) if row["test_f1_macro_std"] else "N/A (one seed)")
                if row["test_f1_macro_mean"] else "pending")
        auroc = "%.4f" % float(row["test_auroc_mean"]) if row["test_auroc_mean"] else "pending"
        lines.append("| %s | %s | %s | %s | %s | %s |" %
                     (row["aggregation"], row["seeds"], validation, test, auroc, row["trainable_parameters"]))
        print("  %-29s test macro F1: %s" % (row["aggregation"], test))
    lines.extend(["", "The CSV's auprc columns contain sklearn average precision (AP), not trapezoidal PR area.",
                  "", "Full run configs, histories, predictions and model checkpoints remain in results/" + manifest.parent.name + "/.",
                  "Training settings: configs/base.json. Study definition: " + str(study_path) + ".", ""])
    manifest_rows = json.loads(manifest.read_text())
    run_config = json.loads(Path(manifest_rows[0]["config"]).read_text())
    source_path = Path(run_config["data"]) / "source.json"
    if source_path.is_file():
        source = json.loads(source_path.read_text())
        lines.extend(["Dataset source: " + source["repository"] + ".",
                      "Dataset revision: `" + source["revision"] + "`.", ""])
        if source["repository"] == "hyenadna/hyena-dna-nt6":
            lines.extend(["Uses the original paper's acceptor FASTA files with partial fine-tuning.",
                          "All original training examples are used. Validation and test both use the original paper holdout, matching its loader.",
                          "The reported test score is on the same holdout used for checkpoint selection, not an independent test set.",
                          "This is an aggregation comparison on the paper dataset, not an exact reproduction of the paper's full fine-tuning protocol.", ""])
    (destination / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nReadable results:", destination / "README.md")
    print("CSV:", destination / "comparison.csv")
    return destination


def run_study(project=None, study_path="configs/study.json"):
    root = find_project(project)
    # Existing data utilities resolve cache paths from cwd. Spyder and CLI share
    # the same root so opening the launcher from another folder remains reliable.
    os.chdir(root)
    os.environ["HF_HOME"] = str(root / "data/cache/huggingface")
    os.environ["HF_HUB_CACHE"] = str(root / "data/cache/huggingface/hub")
    study, cfg = load_study(root, study_path)
    folder = root / "results" / study["name"]
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "manifest.json"
    try:
        with FileLock(str(folder / ".workflow.lock"), timeout=0):
            if not manifest.exists():
                if cfg["pretrained"] and (not Path(cfg["checkpoint"]).is_file() or not Path(cfg["model_config"]).is_file()):
                    if Path(cfg["checkpoint"]).name != "weights.ckpt" or Path(cfg["model_config"]) != Path(cfg["checkpoint"]).with_name("config.json"):
                        raise FileNotFoundError("Provide the configured checkpoint/config files before running")
                    from .download_checkpoint import download
                    print("Preparing pretrained checkpoint...", flush=True)
                    download(output=str(Path(cfg["checkpoint"]).parent))
                from .paper_data import prepare_paper
                print("Preparing the original paper dataset...", flush=True)
                prepare_paper(study["tasks"], cfg["data"], revision=study["data_revision"])
                generate(cfg, folder, study["tasks"], study["heads"], study["seeds"], study["methods"], study["lr_trials"])
            rows = validate_manifest(manifest, study, cfg)
            source_path = Path(cfg["data"]) / "source.json"
            if source_path.is_file() and study["data_revision"] != "main":
                source = json.loads(source_path.read_text())
                if source["revision"] != study["data_revision"]:
                    raise ValueError("Prepared dataset revision differs from study settings; use a new study name/data folder")
                from .paper_data import PAPER_SPLIT_PROTOCOL
                if source.get("split_protocol") != PAPER_SPLIT_PROTOCOL:
                    raise ValueError("Prepared original data must match the paper's val-to-test split protocol")
            complete = sum((Path(row["output"]) / "complete.json").is_file() for row in rows)
            print("Study: %s | %d/%d runs complete" % (study["name"], complete, len(rows)), flush=True)
            if complete < len(rows):
                print("Training pending runs; completed runs are kept...", flush=True)
                run_manifest(manifest)
            else:
                print("All training runs already complete; reusing them.", flush=True)
            print("Selecting checkpoints and trials using validation...", flush=True)
            report(manifest)
            selected = json.loads((folder / "selection.json").read_text())
            pending = sum(not (Path(row["output"]) / "test.json").is_file() for row in selected)
            if pending:
                print("Evaluating %d selected checkpoints on test..." % pending, flush=True)
                test_selected(manifest)
                report(manifest)
            else:
                print("All selected test results already saved; reusing them.", flush=True)
            return export_results(root, manifest, study_path)
    except Timeout as exc:
        raise RuntimeError("Another workflow is running this study. Stop it before starting again.") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run/resume the complete HyenaDNA pooling study automatically.")
    parser.add_argument("--project", help="Project folder (default: detected from cwd or source checkout)")
    parser.add_argument("--study", default="configs/study.json", help="Study definition; no action names required")
    args = parser.parse_args(argv)
    try:
        run_study(args.project, args.study)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")
