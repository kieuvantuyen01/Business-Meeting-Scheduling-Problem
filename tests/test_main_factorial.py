from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from Main import (
    VARIANT_CODES,
    VARIANT_FACTOR_NAMES,
    parse_args,
    precedence_configurations,
    write_aggregate_csv,
)


class MainPrecedenceFactorialTests(unittest.TestCase):
    def test_imp12_plus_machine_code_and_display_name_are_explicit(self) -> None:
        self.assertEqual(VARIANT_CODES["imp12+"], "IC12P")
        self.assertEqual(VARIANT_FACTOR_NAMES["imp12+"], "IC12+")

    def test_default_and_explicit_cli_generate_the_expected_cells(self) -> None:
        default_args = parse_args([])
        self.assertEqual(default_args.maxsat_backend, "uwrmaxsat")
        self.assertEqual(default_args.sat_backend, "cadical")
        self.assertEqual(
            precedence_configurations(default_args),
            [
                ("pairwise", "direct"),
                ("pairwise", "distance_closure"),
                ("sparse_suffix", "direct"),
                ("sparse_suffix", "distance_closure"),
            ],
        )

        legacy_args = parse_args(["--precedence-mode", "both"])
        self.assertEqual(
            precedence_configurations(legacy_args),
            [
                ("pairwise", "direct"),
                ("sparse_suffix", "distance_closure"),
            ],
        )

        one_row = parse_args(
            [
                "--precedence-encoding",
                "sparse_suffix",
                "--precedence-graph",
                "direct",
            ]
        )
        self.assertEqual(
            precedence_configurations(one_row),
            [("sparse_suffix", "direct")],
        )

    def test_legacy_and_independent_flags_cannot_be_mixed(self) -> None:
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--precedence-mode",
                        "traditional",
                        "--precedence-graph",
                        "direct",
                    ]
                )

    def test_aggregate_csv_keeps_all_four_factorial_rows_separate(self) -> None:
        results = []
        for precedence_encoding in ("pairwise", "sparse_suffix"):
            for precedence_graph in ("direct", "distance_closure"):
                results.append(
                    {
                        "instance": "micro",
                        "precedence_encoding": precedence_encoding,
                        "precedence_graph": precedence_graph,
                        "precedence_configuration": (
                            f"{precedence_encoding}+{precedence_graph}"
                        ),
                        "solver": "maxsat",
                        "domain_mode": "reduced",
                        "encoding_variant": "imp12+",
                        "sat_result": "SAT",
                        "runtime_seconds": 0.1,
                        "idle_range_pstar": 0,
                    }
                )

        with tempfile.TemporaryDirectory(prefix="b2b_factorial_test_") as temp:
            output = Path(temp) / "aggregate.csv"
            write_aggregate_csv(output, results)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {
                (row["precedence_encoding"], row["precedence_graph"])
                for row in rows
            },
            {
                ("pairwise", "direct"),
                ("pairwise", "distance_closure"),
                ("sparse_suffix", "direct"),
                ("sparse_suffix", "distance_closure"),
            },
        )
        self.assertNotIn("staircase", rows[0])


if __name__ == "__main__":
    unittest.main()
