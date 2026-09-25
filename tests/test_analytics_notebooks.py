"""
The analytics notebooks are committed without outputs.

They are the only tracked notebooks (.gitignore excludes *.ipynb elsewhere),
and an executed one carries tables and plots of real usernames and comment
text. Nothing else would stop that reaching git: `git add` does not look
inside the file. Strip them with `uv run nbstripout *.ipynb` in analytics/.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((ROOT / "analytics").glob("*.ipynb"))


def test_notebooks_are_found():
    # Guards the glob: an empty list would make the test below pass vacuously.
    assert NOTEBOOKS


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_outputs(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert not cell.get("outputs"), f"{path.name}: cell {i} has outputs"
        assert cell.get("execution_count") is None, f"{path.name}: cell {i} was run"
