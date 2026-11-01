"""
Versions: telling one trained adapter from the next, and not losing any.

Adapter directories were named by hand in model.output_dir, and a second run
of the same config wrote to the same path -- on the pod, and again when the
result was copied home over the first. Nothing recorded which run a
directory held, or how its training went. Every failure here is silent: an
adapter is replaced, or a blind-eval score is credited to the wrong one.
"""

import json
import sqlite3
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from litigpt.config import Config
from litigpt.manifest import (
    build_manifest,
    finalize_manifest,
    make_version,
    read_manifest,
    refuse_to_overwrite,
    stamp_manifest,
    write_manifest,
)
from litigpt.registry import blind_eval_scores, import_adapter, list_versions

ROOT = Path(__file__).resolve().parent.parent

LOG_HISTORY = [
    {"step": 10, "epoch": 0.5, "loss": 3.1},
    {"step": 20, "epoch": 1.0, "eval_loss": 2.95},
    {"step": 30, "epoch": 1.5, "loss": 2.7},
    {"step": 40, "epoch": 2.0, "eval_loss": 2.90},
    {"step": 40, "epoch": 2.0, "train_runtime": 123.4, "train_loss": 2.8},
]


def _config(tmp_path):
    config = Config()
    config.data.training_dir = str(tmp_path / "training")
    config.model.output_dir = str(tmp_path / "adapter")
    config.model.base_model = "Qwen/Qwen2.5-7B-Instruct"
    config.model.notes = "five-word floor, posts as context"
    (tmp_path / "training").mkdir()
    row = {"messages": [{"role": "assistant", "content": "x"}], "username": "bob"}
    (tmp_path / "training" / "train.jsonl").write_text(json.dumps(row) + "\n")
    return config


def _adapter(directory, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    save_file({"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(1)},
              str(directory / "adapter_model.safetensors"))
    if manifest is not None:
        write_manifest(manifest, directory)
    return directory


def _checkpoint(directory, step, log_history):
    ckpt = directory / f"checkpoint-{step}"
    ckpt.mkdir(parents=True)
    (ckpt / "trainer_state.json").write_text(json.dumps(
        {"global_step": step, "epoch": 2.0, "log_history": log_history}
    ))


class TestVersion:
    def test_format_sorts_by_time_and_names_the_base(self):
        v = make_version("Qwen/Qwen2.5-7B-Instruct", datetime(2026, 11, 2, 14, 30, tzinfo=UTC))
        assert v == "20261102-1430-qwen2.5-7b-instruct"
        earlier = make_version("microsoft/phi-3-mini-4k-instruct",
                               datetime(2026, 9, 21, 18, 42, tzinfo=UTC))
        assert earlier < v

    def test_manifest_carries_version_and_notes_from_the_start(self, tmp_path):
        manifest = build_manifest(_config(tmp_path), None, provenance="test")
        assert manifest["version"].endswith("-qwen2.5-7b-instruct")
        assert manifest["notes"] == "five-word floor, posts as context"
        assert manifest["metrics"] is None


class TestMetrics:
    def test_finalize_records_how_training_went(self, tmp_path):
        config = _config(tmp_path)
        out = _adapter(Path(config.model.output_dir),
                       build_manifest(config, None, provenance="test"))
        metrics = finalize_manifest(out, log_history=LOG_HISTORY)["metrics"]
        assert metrics["eval_loss"] == [[20, 2.95], [40, 2.9]]
        assert metrics["best_eval_loss"] == 2.9
        assert metrics["final_train_loss"] == 2.7
        assert metrics["train_runtime_s"] == 123.4

    def test_without_history_it_reads_the_latest_checkpoint(self, tmp_path):
        config = _config(tmp_path)
        out = _adapter(Path(config.model.output_dir),
                       build_manifest(config, None, provenance="test"))
        _checkpoint(out, 9, [{"step": 9, "eval_loss": 9.0}])
        # Logged every 10 steps, saved at 43: the step count is the state's.
        _checkpoint(out, 43, LOG_HISTORY[:4])
        metrics = finalize_manifest(out)["metrics"]
        assert metrics["best_eval_loss"] == 2.9
        assert metrics["global_step"] == 43


class TestNoOverwrite:
    def test_a_completed_adapter_is_not_trained_over(self, tmp_path):
        _adapter(tmp_path, {"status": "completed", "version": "v1"})
        with pytest.raises(FileExistsError, match="v1"):
            refuse_to_overwrite(tmp_path)

    def test_a_crashed_run_or_an_empty_directory_is_fine(self, tmp_path):
        refuse_to_overwrite(tmp_path / "new")
        _adapter(tmp_path / "crashed", {"status": "started"})
        refuse_to_overwrite(tmp_path / "crashed")

    def test_the_pipeline_checks_before_training(self):
        source = (ROOT / "litigpt" / "pipeline.py").read_text(encoding="utf-8")
        assert "refuse_to_overwrite(model_cfg.output_dir)" in source
        assert "log_history=hf_trainer.state.log_history" in source


class TestStamp:
    def test_older_adapter_gets_its_run_date_not_today(self, tmp_path):
        out = _adapter(tmp_path / "old", {
            "schema_version": 1, "status": "completed", "started_at": None,
            "base_model": "microsoft/phi-3-mini-4k-instruct", "users": ["bob"],
        })
        (out / "run_artifacts").mkdir()
        with sqlite3.connect(out / "run_artifacts" / "mlflow.db") as conn:
            conn.execute("create table runs (name text, start_time integer)")
            started = datetime(2026, 9, 21, 18, 42, tzinfo=UTC)
            conn.execute("insert into runs values ('train_30users_phi', ?)",
                         (int(started.timestamp() * 1000),))
        _checkpoint(out, 3352, LOG_HISTORY)

        manifest = stamp_manifest(out, notes="v2")
        assert manifest["version"] == "20260921-1842-phi-3-mini-4k-instruct"
        assert manifest["version_source"] == "run_artifacts/mlflow.db"
        assert manifest["metrics"]["global_step"] == 3352
        assert manifest["schema_version"] == 2

        again = stamp_manifest(out)
        assert again["version"] == manifest["version"]
        assert again["notes"] == "v2"


class TestImport:
    def _versioned(self, tmp_path, version="20261102-1430-qwen2.5-7b-instruct"):
        return _adapter(tmp_path / "pulled", {"version": version, "status": "completed"})

    def test_a_directory_lands_under_its_version(self, tmp_path):
        target = import_adapter(self._versioned(tmp_path), tmp_path / "models")
        assert target == tmp_path / "models" / "20261102-1430-qwen2.5-7b-instruct"
        assert (target / "adapter_model.safetensors").exists()

    def test_a_watchdog_archive_lands_under_its_version(self, tmp_path):
        src = self._versioned(tmp_path)
        (tmp_path / "mlflow.db").write_bytes(b"db")
        archive = tmp_path / "litigpt_v.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(src, arcname=".")
            tar.add(tmp_path / "mlflow.db", arcname="run_artifacts/mlflow.db")
        target = import_adapter(archive, tmp_path / "models")
        assert (target / "run_artifacts" / "mlflow.db").exists()
        assert read_manifest(target)["version"] == "20261102-1430-qwen2.5-7b-instruct"

    def test_an_imported_version_is_never_overwritten(self, tmp_path):
        src = self._versioned(tmp_path)
        import_adapter(src, tmp_path / "models")
        with pytest.raises(FileExistsError):
            import_adapter(src, tmp_path / "models")
        # Nothing half-imported is left behind.
        assert [p.name for p in (tmp_path / "models").iterdir()] == [
            "20261102-1430-qwen2.5-7b-instruct"
        ]

    def test_an_unversioned_adapter_is_refused(self, tmp_path):
        src = _adapter(tmp_path / "pulled", {"status": "completed"})
        with pytest.raises(ValueError, match="stamp"):
            import_adapter(src, tmp_path / "models")


class TestScores:
    def test_rounds_are_credited_to_the_version_they_judged(self, tmp_path):
        log = tmp_path / "eval" / "blind_eval.jsonl"
        log.parent.mkdir()
        rounds = [
            {"source": "model", "model_version": "v2", "model_path": "/pod/models/x",
             "correct": True, "chance": 0.25},
            {"source": "model", "model_version": "v2", "model_path": "models/x",
             "correct": False, "chance": 0.25},
            # Before rounds recorded a version: matched on the directory name.
            {"source": "model", "model_path": "/workspace/models/old_dir",
             "correct": True, "chance": 0.5},
            {"source": "oracle", "model_path": "(oracle)", "correct": True, "chance": 0.25},
        ]
        log.write_text("\n".join(json.dumps(r) for r in rounds))
        scores = blind_eval_scores(str(tmp_path / "eval" / "*.jsonl"))
        assert scores["v2"] == {"rounds": 2, "correct": 1, "chance": 0.5}
        assert scores["old_dir"]["rounds"] == 1
        assert "(oracle)" not in scores

    def test_list_shows_every_adapter_including_unrecorded_ones(self, tmp_path):
        models = tmp_path / "models"
        _adapter(models / "b", {"version": "20261102-1430-q", "status": "completed",
                                "users": ["a"], "metrics": {"best_eval_loss": 2.8}})
        _adapter(models / "a", {"version": "20260921-1842-p", "status": "completed",
                                "users": ["a"]})
        _adapter(models / "legacy", None)
        rows = list_versions(models, str(tmp_path / "none" / "*.jsonl"))
        assert [r["version"] for r in rows] == [
            "20260921-1842-p", "20261102-1430-q", "(no manifest)",
        ]
        assert rows[1]["best_eval"] == 2.8
