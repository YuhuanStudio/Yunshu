"""Feature x modality audit: every modality column decides every feature, fail closed.

path_matrix.json ``modality_columns`` has one column per modality (image, video, audio_in,
omni_speech, ocr). Each column needs a cell for exactly the features in ``features``.
``engaged`` cites code (file + a grep string that must occur in that file) and carries a
``real_check`` id; ``missing`` / ``partial`` need an owner, an impact and a reason;
``na`` needs a reason. Static inspection only: no MLX import, no GPU.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATRIX = json.loads((ROOT / "python/yunshu_engine/path_matrix.json").read_text())
COLUMNS = MATRIX["modality_columns"]
STATUSES = {"engaged", "na", "missing", "partial"}
IMPACTS = {"high", "medium", "low"}
MODALITIES = ("image", "video", "audio_in", "omni_speech", "ocr")


def test_modality_columns_are_the_five_declared_modalities():
    assert set(COLUMNS) == {"_doc", *MODALITIES}
    assert COLUMNS["_doc"]


def test_every_column_covers_exactly_the_features():
    features = set(MATRIX["features"])
    for name in MODALITIES:
        column = COLUMNS[name]
        missing = features - set(column)
        extra = set(column) - features
        assert not missing, f"{name} lacks cells for {sorted(missing)}"
        assert not extra, f"{name} has cells for unknown features {sorted(extra)}"


def test_cells_are_decided_with_their_required_fields():
    for name in MODALITIES:
        for feature, cell in COLUMNS[name].items():
            where = f"{name}/{feature}"
            assert cell.get("s") in STATUSES, where
            if cell["s"] == "engaged":
                assert cell.get("ev"), f"{where}: engaged needs ev"
                assert cell.get("real_check"), f"{where}: engaged needs real_check"
            if cell["s"] in ("missing", "partial"):
                assert cell.get("owner"), f"{where}: {cell['s']} needs owner"
                assert cell.get("impact") in IMPACTS, (
                    f"{where}: {cell['s']} needs impact"
                )
                assert cell.get("why"), f"{where}: {cell['s']} needs why"
            if cell["s"] == "na":
                assert cell.get("why"), f"{where}: na needs a physical reason"


def test_engaged_evidence_strings_occur_in_their_files():
    for name in MODALITIES:
        for feature, cell in COLUMNS[name].items():
            for rel, needle in cell.get("ev", []):
                path = ROOT / rel
                assert path.is_file(), f"{name}/{feature}: {rel} does not exist"
                assert needle in path.read_text(), (
                    f"{name}/{feature}: '{needle}' not in {rel}"
                )
