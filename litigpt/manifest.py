"""
The training manifest: what an adapter was trained on, stored beside it.

An adapter directory used to hold weights, tokenizer and PEFT's config, and
nothing about the data. The cohort of real users it impersonates lived only
in data/processed/users_metadata.json on whichever machine ran extraction,
and every interface read the persona list from there. On a machine where that
file came from a different extraction -- or did not exist -- the interfaces
offered a list that did not match the weights, and nothing said so.

litigpt_manifest.json is written into model.output_dir before training starts,
copied into every checkpoint, and completed after the final save. Because it
sits inside the adapter directory, it travels with the adapter: the RunPod
watchdog tars that directory whole, so the manifest comes home in the archive
with no extra step.

The cohort is read from train.jsonl rather than from config or extraction
metadata. With top_n selection data.target_usernames is empty, and the
extraction metadata describes what was extracted, not what was trained on.
The training file is what the adapter actually saw.
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "litigpt_manifest.json"
SCHEMA_VERSION = 1

_TRACKED_LIBRARIES = (
    "torch", "transformers", "trl", "peft", "datasets",
    "accelerate", "bitsandbytes", "mlflow",
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def cohort_from_training_data(train_path: str | Path) -> dict[str, int]:
    """
    Count examples per username in a training jsonl.

    Ordered by count, largest first, then by name, so the persona an interface
    picks by default is the best-represented one and the order is stable.
    """
    counts: Counter = Counter()
    with open(train_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                counts[json.loads(line).get("username")] += 1

    if None in counts:
        raise ValueError(
            f"{counts[None]} examples in {train_path} have no 'username' field, "
            "so the manifest cannot record who the adapter impersonates. "
            "Re-run the preprocess step."
        )
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def file_fingerprint(path: str | Path) -> dict:
    """sha256, size and line count, enough to tell whether two copies match."""
    digest = hashlib.sha256()
    lines = 0
    size = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
            lines += chunk.count(b"\n")
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size, "lines": lines}


def git_state(repo_dir: str | Path | None = None) -> dict:
    """
    The commit the code was at, or nulls.

    A pod set up by copying the tree rather than cloning has no .git, and that
    must not stop a training run -- the config and library versions carry most
    of what reproducing it needs.
    """
    cwd = Path(repo_dir) if repo_dir else Path(__file__).resolve().parent.parent
    try:
        # git is resolved from PATH on purpose: its location differs per machine.
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd,  # noqa: S607
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd,  # noqa: S607
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError) as exc:
        logger.info("No git commit recorded in the manifest: %s", exc)
        return {"commit": None, "dirty": None}


def library_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for name in _TRACKED_LIBRARIES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def adapted_modules(adapter_dir: str | Path) -> list[str] | None:
    """
    The module names the adapter's weights actually contain.

    Read from the safetensors, not adapter_config.json: PEFT writes the
    requested target list into the config whether or not each name matched,
    which is how a run once claimed seven modules and adapted two.
    """
    path = Path(adapter_dir) / "adapter_model.safetensors"
    if not path.exists():
        return None
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as fh:
        # base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.weight
        return sorted({key.split(".")[-3] for key in fh.keys()})  # noqa: SIM118 -- safe_open is not a dict


def build_manifest(config, config_path: str | None, *, provenance: str) -> dict:
    """Assemble a manifest for a run about to start (or, for backfill, past)."""
    training_dir = Path(config.data.training_dir)
    train_path = training_dir / "train.jsonl"
    val_path = training_dir / "val.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found. Run the preprocess step before training."
        )

    counts = cohort_from_training_data(train_path)
    dataset = {"train": file_fingerprint(train_path)}
    if val_path.exists():
        dataset["val"] = file_fingerprint(val_path)

    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance,
        "status": "started",
        "started_at": _now(),
        "finished_at": None,
        "base_model": config.model.base_model,
        "users": list(counts),
        "examples_per_user": counts,
        "dataset": dataset,
        "config_file": str(config_path) if config_path else None,
        "config": config.model_dump(),
        "git": git_state(),
        "libraries": library_versions(),
        "adapted_modules": None,
    }


def write_manifest(manifest: dict, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_FILENAME
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def read_manifest(directory: str | Path) -> dict | None:
    """The manifest in an adapter directory, or None if it has none."""
    path = Path(directory) / MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unreadable manifest %s: %s", path, exc)
        return None


def finalize_manifest(directory: str | Path) -> dict | None:
    """Mark the run complete and record what the saved weights contain."""
    manifest = read_manifest(directory)
    if manifest is None:
        logger.warning("No manifest in %s to finalize", directory)
        return None
    manifest["status"] = "completed"
    manifest["finished_at"] = _now()
    manifest["adapted_modules"] = adapted_modules(directory)
    write_manifest(manifest, directory)
    return manifest


def load_adapter_users(adapter_dir: str | Path | None) -> list[str]:
    """The cohort an adapter was trained on, or [] if it carries no manifest."""
    if not adapter_dir:
        return []
    manifest = read_manifest(adapter_dir)
    return list(manifest.get("users", [])) if manifest else []


# ----------------------------------------------------------------------
# Backfill: adapters trained before the manifest existed
# ----------------------------------------------------------------------


def _mlflow_user_counts(db_path: str | Path) -> dict[str, int]:
    """Per-user example counts the training run logged, from its MLflow DB."""
    import sqlite3

    with sqlite3.connect(str(db_path)) as conn:
        run = conn.execute(
            "select run_uuid from runs where name like 'train%' "
            "order by start_time desc limit 1"
        ).fetchone()
        if run is None:
            raise ValueError(f"No training run found in {db_path}")
        rows = conn.execute(
            "select key, value from metrics where run_uuid = ? "
            "and key like 'user\\_%\\_samples' escape '\\'",
            (run[0],),
        ).fetchall()
    return {key[len("user_"):-len("_samples")]: int(value) for key, value in rows}


def backfill(adapter_dir: str, config_path: str, mlflow_db: str | None) -> dict:
    """
    Write a manifest for an adapter trained without one.

    The local training data is only trustworthy as a description of the run
    if it is the same data the run used. With the run's MLflow DB given, the
    per-user counts it logged are compared against the local file and any
    difference refuses the backfill. Without it, the manifest says it is
    unverified.

    Git commit and library versions are left null: the current checkout and
    environment are not the ones that trained the adapter.
    """
    from litigpt.config import Config

    if not (Path(adapter_dir) / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_dir}")

    config = Config.from_yaml(config_path)

    if mlflow_db:
        logged = _mlflow_user_counts(mlflow_db)
        local = cohort_from_training_data(Path(config.data.training_dir) / "train.jsonl")
        if logged != local:
            differing = sorted(
                u for u in set(logged) | set(local) if logged.get(u) != local.get(u)
            )
            raise ValueError(
                f"Local training data does not match the run's MLflow record for "
                f"{len(differing)} users ({', '.join(differing[:5])}...). "
                "Refusing to describe the adapter with data it was not trained on."
            )
        check = f"per-user counts verified against {Path(mlflow_db).name}"
    else:
        check = "UNVERIFIED: local data not checked against the run's record"

    manifest = build_manifest(
        config, config_path,
        provenance=f"backfilled {_now()} from local data; {check}",
    )
    manifest["started_at"] = None
    manifest["git"] = {"commit": None, "dirty": None}
    manifest["libraries"] = None
    manifest["status"] = "completed"
    manifest["adapted_modules"] = adapted_modules(adapter_dir)
    write_manifest(manifest, adapter_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Write a manifest for an adapter trained before manifests existed"
    )
    parser.add_argument("--model", required=True, help="Adapter directory")
    parser.add_argument("--config", required=True,
                        help="Config the adapter was trained with")
    parser.add_argument("--mlflow-db",
                        help="The run's mlflow.db, to verify the local data matches")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing manifest")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if read_manifest(args.model) is not None and not args.force:
        raise SystemExit(f"{args.model} already has a manifest; --force to overwrite")

    manifest = backfill(args.model, args.config, args.mlflow_db)
    logger.info("Wrote %s: %d users, %s",
                Path(args.model) / MANIFEST_FILENAME,
                len(manifest["users"]), manifest["provenance"])


if __name__ == "__main__":
    main()
