import pathlib
import sys
import unittest

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import helper_functions as helper
from scripts.populate_content_database import resolve_resource_type


class TestResourceTypeMapping(unittest.TestCase):
    def test_readme_resource_mappings_are_normalized(self):
        self.assertEqual(helper.map_resource_medium_to_type("survey"), "resource")
        self.assertEqual(helper.map_resource_medium_to_type("Library"), "resource")
        self.assertEqual(helper.map_resource_medium_to_type("Resources"), "resource")
        self.assertEqual(helper.map_resource_medium_to_type("NACLO Problems"), "assignment")

    def test_unknown_medium_returns_default(self):
        self.assertEqual(
            helper.map_resource_medium_to_type("worksheet", default="worksheet"),
            "worksheet",
        )
        self.assertIsNone(helper.map_resource_medium_to_type("worksheet"))

    def test_seed_resolution_keeps_raw_medium_without_flag(self):
        row = pd.Series({"medium": "paper"})
        self.assertEqual(resolve_resource_type(row, map_types=False), "paper")

    def test_seed_resolution_maps_medium_with_flag(self):
        row = pd.Series({"medium": "paper"})
        self.assertEqual(resolve_resource_type(row, map_types=True), "resource")


if __name__ == "__main__":
    unittest.main()