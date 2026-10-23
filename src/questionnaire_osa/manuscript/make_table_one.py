"""Entry point: regenerate the cohort descriptive table (Supplementary Table S3).

Run with::

    python -m questionnaire_osa.manuscript.make_table_one

The cohort is the same adult slice (``dem_age >= SETTINGS["min_age"]``)
used to train every model, so the N reported here matches the models.
Output goes to :data:`config.PATHS["tbl_table_one"]`.
"""

from __future__ import annotations

import json
import logging

from config import PATHS, SETTINGS
from config.config import ensure_output_dirs

from ..data import load_sleep_data
from ..reporting import build_table_one, write_table_one

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ensure_output_dirs()

    df = load_sleep_data(PATHS["database"], min_age=SETTINGS["min_age"])
    logger.info("Loaded %s adult subjects (dem_age >= %d)", f"{len(df):,}", SETTINGS["min_age"])

    with PATHS["encoding_json"].open("r", encoding="utf-8") as fh:
        encoding = json.load(fh)

    table = build_table_one(df, encoding)
    write_table_one(table, PATHS["tbl_table_one"])
    logger.info("Wrote %s (%d rows)", PATHS["tbl_table_one"], len(table))


if __name__ == "__main__":
    main()
