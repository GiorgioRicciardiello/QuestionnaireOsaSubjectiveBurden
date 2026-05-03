"""Load human-readable feature names from ``dataset/encoding.json``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_feature_display_names(encoding_path: Path) -> Dict[str, str]:
    """Build the mapping ``raw_column_name → publication display name``.

    Parameters
    ----------
    encoding_path
        Absolute path to the ``encoding.json`` file shipped with the
        cleaned database.

    Returns
    -------
    dict
        ``{column_name: definition}`` pairs.  Missing or malformed
        entries are silently skipped — the returned mapping is therefore
        always safe to feed to ``DataFrame.rename``.
    """
    with encoding_path.open("r", encoding="utf-8") as fh:
        encoding = json.load(fh)

    return {
        feat: data["definition"]
        for feat, data in encoding.items()
        if isinstance(data, dict) and "definition" in data
    }
