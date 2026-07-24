from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from Dataset_Manifest import classify_instance_name
from Generate_Precedence_Stress import validate_dataset


class PrecedenceStressDatasetTests(unittest.TestCase):
    def test_generated_dataset_passes_all_structural_checks(self) -> None:
        summary = validate_dataset(
            PROJECT_ROOT / "data_precedence_stress",
            PROJECT_ROOT / "data_table03_origin",
        )
        self.assertEqual(
            summary,
            {
                "source_instances": 20,
                "gamma_levels": 3,
                "generated_instances": 60,
            },
        )

    def test_generated_density_suffixes_are_precedence_variants(self) -> None:
        for gamma in (30, 35, 40):
            self.assertEqual(
                classify_instance_name(f"tic-12.prec{gamma}.dzn"),
                ("precedence", f"prec{gamma}"),
            )


if __name__ == "__main__":
    unittest.main()
