"""
Every trained adapter, by version.

Each adapter directory's litigpt_manifest.json is the record of that version:
what it was trained on, with which config and code, and how training went.
This module reads those records and keeps no index of its own that could fall
out of step with them.

    python -m litigpt.registry list
    python -m litigpt.registry import /path/to/litigpt_<version>.tar.gz
    python -m litigpt.registry stamp models/litigpt_top30_lora --notes "..."

`import` is how an adapter comes home from a pod. It lands in
models/<version>/ and refuses to overwrite: two runs of the same config share
an output_dir on the pod, and copying the second over the first locally is
how a version is lost.
"""

import argparse
import json
import logging
import shutil
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

from litigpt.manifest import MANIFEST_FILENAME, read_manifest, stamp_manifest

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = "models"
# Every blind-evaluation log, local and pulled from pods (blind_eval.pod.jsonl).
DEFAULT_EVAL_LOGS = "data/eval/*.jsonl"


def find_adapters(models_dir: str | Path) -> list[tuple[Path, dict | None]]:
    """Adapter directories one level under models_dir, with their manifests."""
    found = []
    for path in sorted(Path(models_dir).iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        manifest = read_manifest(path)
        has_weights = (path / "adapter_model.safetensors").exists() or any(path.glob("checkpoint-*"))
        if manifest is not None or has_weights:
            found.append((path, manifest))
    return found


def blind_eval_scores(eval_logs: str) -> dict[str, dict]:
    """
    Model-condition rounds per adapter, over every log matching the glob:
    count, correct, and mean chance.

    Keyed by the manifest version each round recorded, or, for rounds logged
    before rounds recorded one, by the last component of the adapter path.
    Oracle rounds are a ceiling for the cohort, not a score for any version,
    and are left out.
    """
    scores: dict[str, dict] = defaultdict(lambda: {"rounds": 0, "correct": 0, "chance": 0.0})
    pattern = Path(eval_logs)
    for path in sorted(pattern.parent.glob(pattern.name)):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("source", "model") != "model":
                    continue
                key = record.get("model_version") or Path(str(record.get("model_path", ""))).name
                entry = scores[key]
                entry["rounds"] += 1
                entry["correct"] += bool(record.get("correct"))
                entry["chance"] += float(record.get("chance") or 0)
    return dict(scores)


def _filters(manifest: dict) -> str:
    data = (manifest.get("config") or {}).get("data") or {}
    if "min_comment_words" in data:
        floor = f">={data['min_comment_words']}w"
    else:
        floor = f">={data.get('min_comment_length', '?')}c"
    score = data.get("min_score")
    return f"{floor} <={data.get('max_comment_length', '?')}c score>={score}"


def describe(path: Path, manifest: dict | None, scores: dict[str, dict]) -> dict:
    """One row of the version table."""
    if manifest is None:
        return {"version": "(no manifest)", "directory": path.name}
    metrics = manifest.get("metrics") or {}
    blind = scores.get(manifest.get("version") or "") or scores.get(path.name)
    blind_text = ""
    if blind and blind["rounds"]:
        blind_text = (f"{blind['correct']}/{blind['rounds']} "
                      f"({100 * blind['correct'] / blind['rounds']:.0f}%, chance "
                      f"{100 * blind['chance'] / blind['rounds']:.0f}%)")
    commit = ((manifest.get("git") or {}).get("commit") or "")[:7]
    return {
        "version": manifest.get("version") or "(unversioned)",
        "directory": path.name,
        "base": (manifest.get("base_model") or "").rsplit("/", maxsplit=1)[-1],
        "users": len(manifest.get("users") or []),
        "examples": (manifest.get("dataset") or {}).get("train", {}).get("lines"),
        "filters": _filters(manifest),
        "best_eval": metrics.get("best_eval_loss"),
        "steps": metrics.get("global_step"),
        "status": manifest.get("status"),
        "commit": commit,
        "blind_eval": blind_text,
        "notes": manifest.get("notes") or "",
    }


def list_versions(models_dir: str | Path = DEFAULT_MODELS_DIR,
                  eval_logs: str = DEFAULT_EVAL_LOGS) -> list[dict]:
    """Every adapter under models_dir, oldest version first."""
    scores = blind_eval_scores(eval_logs)
    rows = [describe(path, manifest, scores) for path, manifest in find_adapters(models_dir)]
    return sorted(rows, key=lambda r: (r["version"].startswith("("), r["version"]))


def import_adapter(source: str | Path, models_dir: str | Path = DEFAULT_MODELS_DIR) -> Path:
    """
    Put an adapter -- a watchdog archive or a directory -- at models/<version>/.

    Refuses when the adapter has no version (stamp it first) or when that
    version is already present.
    """
    source = Path(source)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=models_dir, prefix=".importing-") as tmp:
        staged = Path(tmp) / "adapter"
        if source.is_dir():
            shutil.copytree(source, staged)
        elif tarfile.is_tarfile(source):
            with tarfile.open(source) as archive:
                # "data" refuses absolute paths, links out of the tree and
                # device files: the archive comes off a rented machine.
                archive.extractall(staged, filter="data")
        else:
            raise ValueError(f"{source} is neither a directory nor a tar archive")

        manifest = read_manifest(staged)
        if manifest is None:
            raise ValueError(f"{source} has no {MANIFEST_FILENAME}; nothing says what it is")
        version = manifest.get("version")
        if not version:
            raise ValueError(
                f"{source} has a manifest without a version. Stamp it first: "
                "python -m litigpt.registry stamp <dir>"
            )
        target = models_dir / version
        if target.exists():
            raise FileExistsError(f"{target} already exists; version {version} is imported")
        staged.rename(target)

    logger.info("Imported %s as %s", source, target)
    return target


def _print_table(rows: list[dict]) -> None:
    columns = ["version", "directory", "base", "users", "examples", "filters",
               "best_eval", "steps", "commit", "blind_eval", "notes"]
    cells = [[("" if r.get(c) is None else str(r.get(c))) for c in columns] for r in rows]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(columns)]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths, strict=True)))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths, strict=True)))


def main():
    parser = argparse.ArgumentParser(description="Trained adapters, by version")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="Every adapter, with how it was trained and scored")
    p_list.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    p_list.add_argument("--eval-logs", default=DEFAULT_EVAL_LOGS,
                        help="Glob of blind-evaluation logs to score versions from")
    p_list.add_argument("--json", action="store_true", help="Rows as JSON")

    p_import = sub.add_parser("import", help="Archive or directory -> models/<version>/")
    p_import.add_argument("source")
    p_import.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)

    p_stamp = sub.add_parser("stamp", help="Give an older adapter a version and metrics")
    p_stamp.add_argument("adapter_dir")
    p_stamp.add_argument("--notes", help="Why this version exists")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if args.command == "list":
        rows = list_versions(args.models_dir, args.eval_logs)
        if args.json:
            print(json.dumps(rows, indent=2, ensure_ascii=False))
        else:
            _print_table(rows)
    elif args.command == "import":
        print(import_adapter(args.source, args.models_dir))
    elif args.command == "stamp":
        manifest = stamp_manifest(args.adapter_dir, notes=args.notes)
        logger.info("%s is version %s (%s)", args.adapter_dir, manifest["version"],
                    manifest.get("version_source", "already versioned"))


if __name__ == "__main__":
    main()
