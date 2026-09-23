"""
The training manifest, which ties an adapter to the users it impersonates.

Before it existed, the adapter directory recorded nothing about its data and
every interface took its persona list from data/processed/users_metadata.json
-- the last extraction on whatever machine launched it. A different
extraction, or none, gave a list that did not match the weights, and nothing
raised. The MLflow run of the top-30 adapter was tagged users='' for the same
reason: the tag read data.target_usernames, which top_n selection leaves empty.
"""

import json
import logging
import sqlite3
import types
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from safetensors.torch import save_file

from litigpt.config import Config
from litigpt.manifest import (
    MANIFEST_FILENAME,
    backfill,
    build_manifest,
    cohort_from_training_data,
    finalize_manifest,
    read_manifest,
    write_manifest,
)
from litigpt.model_utils import resolve_available_users
from litigpt.training.trainer import ManifestToCheckpoints

ROOT = Path(__file__).resolve().parent.parent


def _write_jsonl(path, usernames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps({
            "messages": [
                {"role": "system", "content": f"Sei {u}"},
                {"role": "user", "content": "ciao"},
                {"role": "assistant", "content": "ciao"},
            ],
            "username": u,
        }) + "\n" for u in usernames)


def _config(tmp_path, users=("bob", "alice", "bob", "carol", "bob", "alice")):
    config = Config()
    config.data.training_dir = str(tmp_path / "training")
    config.data.processed_dir = str(tmp_path / "processed")
    config.model.output_dir = str(tmp_path / "adapter")
    _write_jsonl(tmp_path / "training" / "train.jsonl", users)
    _write_jsonl(tmp_path / "training" / "val.jsonl", users[:2])
    return config


def _fake_adapter(directory, modules=("qkv_proj", "o_proj")):
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        f"base_model.model.model.layers.0.self_attn.{m}.lora_{ab}.weight": torch.zeros(2, 2)
        for m in modules for ab in "AB"
    }
    save_file(tensors, str(directory / "adapter_model.safetensors"))


def _processed(tmp_path, users):
    d = tmp_path / "processed"
    d.mkdir(parents=True, exist_ok=True)
    (d / "users_metadata.json").write_text(json.dumps({"users": users}))


class TestCohort:
    def test_counts_and_orders_by_representation(self, tmp_path):
        config = _config(tmp_path)
        counts = cohort_from_training_data(Path(config.data.training_dir) / "train.jsonl")
        assert counts == {"bob": 3, "alice": 2, "carol": 1}
        assert list(counts) == ["bob", "alice", "carol"]

    def test_refuses_examples_without_a_username(self, tmp_path):
        path = tmp_path / "train.jsonl"
        path.write_text(json.dumps({"messages": []}) + "\n")
        with pytest.raises(ValueError, match="no 'username'"):
            cohort_from_training_data(path)


class TestManifest:
    def test_records_cohort_from_data_when_config_names_nobody(self, tmp_path):
        # The top_n case: target_usernames empty, cohort only in the data.
        config = _config(tmp_path)
        assert config.data.target_usernames == []
        manifest = build_manifest(config, None, provenance="test")
        assert manifest["users"] == ["bob", "alice", "carol"]
        assert manifest["dataset"]["train"]["lines"] == 6
        assert manifest["config"]["model"]["base_model"] == config.model.base_model

    def test_finalize_reads_modules_from_the_weights(self, tmp_path):
        config = _config(tmp_path)
        out = Path(config.model.output_dir)
        write_manifest(build_manifest(config, None, provenance="test"), out)
        assert read_manifest(out)["status"] == "started"

        _fake_adapter(out, modules=("qkv_proj", "o_proj"))
        finalize_manifest(out)

        manifest = read_manifest(out)
        assert manifest["status"] == "completed"
        assert manifest["adapted_modules"] == ["o_proj", "qkv_proj"]

    def test_missing_training_data_fails_before_any_gpu_work(self, tmp_path):
        config = Config()
        config.data.training_dir = str(tmp_path / "nothing")
        with pytest.raises(FileNotFoundError, match="preprocess"):
            build_manifest(config, None, provenance="test")


class TestCheckpointCopy:
    def test_each_checkpoint_carries_the_manifest(self, tmp_path):
        # A crashed run is archived from its newest checkpoint, not output_dir.
        (tmp_path / MANIFEST_FILENAME).write_text('{"users": ["bob"]}')
        (tmp_path / "checkpoint-100").mkdir()
        args = types.SimpleNamespace(output_dir=str(tmp_path))
        state = types.SimpleNamespace(global_step=100)

        ManifestToCheckpoints().on_save(args, state, control=None)

        assert read_manifest(tmp_path / "checkpoint-100") == {"users": ["bob"]}


class TestResolveUsers:
    def test_adapter_manifest_wins_over_local_extraction(self, tmp_path, caplog):
        adapter = tmp_path / "adapter"
        write_manifest({"users": ["bob", "alice"]}, adapter)
        _processed(tmp_path, ["bob", "alice", "zed"])

        with caplog.at_level(logging.WARNING):
            users = resolve_available_users(str(adapter), str(tmp_path / "processed"))

        assert users == ["bob", "alice"]
        assert "zed" in caplog.text

    def test_adapter_without_manifest_falls_back_and_says_so(self, tmp_path, caplog):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        _processed(tmp_path, ["bob"])

        with caplog.at_level(logging.WARNING):
            users = resolve_available_users(str(adapter), str(tmp_path / "processed"))

        assert users == ["bob"]
        assert MANIFEST_FILENAME in caplog.text

    def test_works_with_no_local_extraction_at_all(self, tmp_path):
        # A fresh clone with only the adapter pulled from the pod.
        adapter = tmp_path / "adapter"
        write_manifest({"users": ["bob"]}, adapter)
        assert resolve_available_users(str(adapter), str(tmp_path / "absent")) == ["bob"]


class TestBackfill:
    def _mlflow_db(self, path, counts):
        conn = sqlite3.connect(str(path))
        conn.execute("create table runs (run_uuid, name, start_time)")
        conn.execute("create table metrics (run_uuid, key, value)")
        conn.execute("insert into runs values ('r1', 'train__phi', 1)")
        for user, n in counts.items():
            conn.execute("insert into metrics values ('r1', ?, ?)", (f"user_{user}_samples", n))
        conn.commit()
        conn.close()

    def _config_file(self, tmp_path, config):
        import yaml

        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config.model_dump()))
        return str(path)

    def test_verified_backfill(self, tmp_path):
        config = _config(tmp_path)
        _fake_adapter(Path(config.model.output_dir))
        db = tmp_path / "mlflow.db"
        self._mlflow_db(db, {"bob": 3, "alice": 2, "carol": 1})

        manifest = backfill(config.model.output_dir, self._config_file(tmp_path, config), str(db))

        assert manifest["users"] == ["bob", "alice", "carol"]
        assert "verified" in manifest["provenance"]
        assert manifest["git"]["commit"] is None  # this checkout did not train it

    def test_refuses_when_local_data_is_not_what_was_trained(self, tmp_path):
        config = _config(tmp_path)
        _fake_adapter(Path(config.model.output_dir))
        db = tmp_path / "mlflow.db"
        self._mlflow_db(db, {"bob": 3, "alice": 2, "dave": 1})

        with pytest.raises(ValueError, match="does not match"):
            backfill(config.model.output_dir, self._config_file(tmp_path, config), str(db))
        assert read_manifest(config.model.output_dir) is None


class TestInterfacesUseTheAdapterList:
    """Source guard, like TestNoSecondImplementation: the interfaces need a
    loaded model to exercise, so check they do not read the extraction list
    directly."""

    MODULES: ClassVar[list[str]] = [
        "litigpt/interface/gradio_app.py",
        "litigpt/interface/ollama.py",
        "litigpt/interface/blind_eval.py",
    ]

    @pytest.mark.parametrize("module", MODULES)
    def test_no_direct_read_of_users_metadata(self, module):
        source = (ROOT / module).read_text(encoding="utf-8")
        assert "load_user_metadata" not in source
        assert "resolve_available_users" in source
