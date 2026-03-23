#!/usr/bin/env python3
"""Seed Content rows from data/resources.csv.

Usage:
  ./.venv/bin/python scripts/seed_content_from_resources.py
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Any

import pandas as pd

# Ensure repo root imports work even if run from another cwd.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import helper_functions as helper
from app.flask_db import create_app
from app.db_models import Content, db


def clean_value(value) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def resolve_resource_type(row: pd.Series, map_types: bool) -> str | None:
    medium = clean_value(row.get("medium"))
    if medium is None:
        return None

    if not map_types:
        return str(medium)

    return helper.map_resource_medium_to_type(str(medium), default=str(medium))


def seed_content(map_types: bool = False) -> None:
    cfg, root = helper.load_config()

    data_dir = pathlib.Path(cfg["data"]["path"])
    resources_csv = root / data_dir / "resources.csv"

    if not resources_csv.exists():
        raise FileNotFoundError(f"Could not find resources file: {resources_csv}")

    resources_df = pd.read_csv(resources_csv)

    app = create_app()
    created = 0
    skipped = 0

    with app.app_context():
        existing_content_ids = {
            row[0]
            for row in db.session.query(Content.content_id).all()
        }

        for _, row in resources_df.iterrows():
            print(row)
            csv_id = clean_value(row.get("id"))
            if csv_id is None:
                skipped += 1
                continue

            content_id = str(csv_id)
            if content_id in existing_content_ids:
                skipped += 1
                continue

            content_metadata = {
                "url": clean_value(row.get("url")),
                "topic": clean_value(row.get("topic")),
                "date": clean_value(row.get("date")),
                "author": clean_value(row.get("author")),
                "medium": clean_value(row.get("medium")),
                "respath": clean_value(row.get("respath")),
                "txtpath": clean_value(row.get("txtpath")),
                "venue": clean_value(row.get("venue")),
                "status": clean_value(row.get("status")),
            }

            db.session.add(
                Content(
                    content_id=content_id,
                    title=clean_value(row.get("title")),
                    resource_type=resolve_resource_type(row, map_types=map_types),
                    content_metadata=content_metadata,
                )
            )
            existing_content_ids.add(content_id)
            created += 1

        db.session.commit()

    print(f"Seed complete. Created: {created}, Skipped: {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed content rows from the resources CSV.")
    parser.add_argument(
        "--map-resource-types",
        action="store_true",
        help="Map medium values to coarse resource types using the README mapping.",
    )
    args = parser.parse_args()
    seed_content(map_types=args.map_resource_types)
