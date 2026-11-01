"""
Every key in a tracked config file must exist in the schema.

Config.from_yaml drops keys the Pydantic models do not declare, without a
word. A renamed setting left behind in a YAML file -- min_comment_length after
it became min_comment_words -- would keep looking authoritative while the run
used the default instead.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from litigpt.config import Config

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = sorted(p.name for p in ROOT.glob("config.*.yaml"))


def _unknown_keys(raw: dict, model: type[BaseModel], prefix: str = "") -> list[str]:
    unknown = []
    for key, value in raw.items():
        field = model.model_fields.get(key)
        if field is None:
            unknown.append(prefix + key)
            continue
        annotation = field.annotation
        if isinstance(value, dict) and isinstance(annotation, type) and issubclass(annotation, BaseModel):
            unknown += _unknown_keys(value, annotation, f"{prefix}{key}.")
    return unknown


@pytest.mark.parametrize("name", CONFIGS)
def test_every_key_is_in_the_schema(name):
    raw = yaml.safe_load((ROOT / name).read_text(encoding="utf-8")) or {}
    assert _unknown_keys(raw, Config) == [], f"{name} has keys config.py ignores"


@pytest.mark.parametrize("name", CONFIGS)
def test_loads(name):
    Config.from_yaml(str(ROOT / name))


def test_there_are_configs_to_check():
    assert "config.default.yaml" in CONFIGS
