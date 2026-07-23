from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    output_name: str
    family: str
    path_count: int
    unique_content_count: int
    precedence_factorial: bool

    @property
    def configurations_per_instance(self) -> int:
        return 24 if self.precedence_factorial else 6


DATASET_SPECS = (
    DatasetSpec("data_table03_origin", "original", 20, 20, False),
    DatasetSpec("data_table06_forb", "forbidden", 40, 26, False),
    DatasetSpec("data_table07_fixed", "fixed", 40, 40, False),
    DatasetSpec("data_table08_prec", "precedence", 40, 40, True),
)

ORG_CONFIGURATION_LABEL = "ORG-F-PW-DE-PSC-IRP-UW-OBIC12P"
ORG_ENCODING_VARIANT = "org_old_best_ic12plus"
ORG_IMPLIED_PACKAGE_CODE = "OBIC12P"
ORG_IMPLIED_PACKAGE_NAME = "OldBestIC12+"
ORG_CONFIGURATION_ID = (
    "baseline1__model-org_old_best_maxsat__m-full__p-pairwise__"
    "g-direct__b-per_slot_cardinality__o-idle_range_pstar__"
    "s-uwrmaxsat__i-old_best_ic12plus__fairness-none"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def expected_main_cells(spec: DatasetSpec) -> set[tuple[str, str, str, str]]:
    precedence_encodings = (
        ("pairwise", "sparse_suffix")
        if spec.precedence_factorial
        else ("pairwise",)
    )
    precedence_graphs = (
        ("direct", "distance_closure")
        if spec.precedence_factorial
        else ("direct",)
    )
    return set(
        itertools.product(
            ("full", "reduced"),
            ("IncrementalSAT", "NonIncrementalSAT", "UWrMaxSAT"),
            precedence_encodings,
            precedence_graphs,
        )
    )


def _main_cell(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("domain_mode", ""),
        row.get("optimization_engine", ""),
        row.get("precedence_encoding", ""),
        row.get("precedence_graph", ""),
    )


def official_forbidden_instance_names(noves_dir: Path) -> set[str]:
    if not noves_dir.is_dir():
        raise FileNotFoundError(noves_dir)
    return {
        path.stem
        for path in noves_dir.glob("*.dzn")
        if ".forb0003.dzn" in path.name or ".forb0007.dzn" in path.name
    }


def source_alias_names(rows: list[dict[str, str]]) -> set[str]:
    names: set[str] = set()
    for row in rows:
        aliases = row.get("source_alias_paths", "")
        if aliases:
            names.update(
                Path(alias.strip()).stem
                for alias in aliases.split(" | ")
                if alias.strip()
            )
        elif row.get("instance"):
            names.add(row["instance"])
    return names


def validate_output(
    output: Path,
    noves_dir: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    all_main_instances: set[str] = set()
    all_org_instances: set[str] = set()
    all_main_aliases: set[str] = set()
    all_org_aliases: set[str] = set()
    all_main_content_ids: set[str] = set()
    all_org_content_ids: set[str] = set()
    main_total = 0
    org_total = 0

    for spec in DATASET_SPECS:
        main_path = output / "main" / f"{spec.output_name}_detailed.csv"
        org_path = output / "org" / f"{spec.output_name}_org_new.csv"
        try:
            main_rows = read_csv(main_path)
        except FileNotFoundError:
            errors.append(f"missing main CSV: {main_path}")
            continue
        try:
            org_rows = read_csv(org_path)
        except FileNotFoundError:
            errors.append(f"missing ORG CSV: {org_path}")
            continue

        main_total += len(main_rows)
        org_total += len(org_rows)
        main_by_instance: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in main_rows:
            main_by_instance[row.get("instance", "")].append(row)
        org_by_instance: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in org_rows:
            org_by_instance[row.get("instance", "")].append(row)

        if len(main_by_instance) != spec.unique_content_count:
            errors.append(
                f"{spec.output_name}: main has {len(main_by_instance)} content "
                f"representatives, expected {spec.unique_content_count}"
            )
        if len(org_by_instance) != spec.unique_content_count:
            errors.append(
                f"{spec.output_name}: ORG has {len(org_by_instance)} content "
                f"representatives, expected {spec.unique_content_count}"
            )
        if set(main_by_instance) != set(org_by_instance):
            errors.append(
                f"{spec.output_name}: main/ORG representative sets differ"
            )
        main_aliases = source_alias_names(main_rows)
        org_aliases = source_alias_names(org_rows)
        if len(main_aliases) != spec.path_count:
            errors.append(
                f"{spec.output_name}: main covers {len(main_aliases)} source "
                f"paths, expected {spec.path_count}"
            )
        if len(org_aliases) != spec.path_count:
            errors.append(
                f"{spec.output_name}: ORG covers {len(org_aliases)} source "
                f"paths, expected {spec.path_count}"
            )
        if main_aliases != org_aliases:
            errors.append(
                f"{spec.output_name}: main/ORG source-alias sets differ"
            )
        if spec.family == "forbidden" and noves_dir is not None:
            try:
                official_names = official_forbidden_instance_names(noves_dir)
            except FileNotFoundError:
                errors.append(f"missing noves directory: {noves_dir}")
            else:
                if main_aliases != official_names:
                    errors.append(
                        f"{spec.output_name}: source aliases do not match the "
                        "40 official Forbidden paths in noves"
                    )
        main_content_ids = {
            row.get("instance_content_id", "") for row in main_rows
        }
        org_content_ids = {
            row.get("instance_content_id", "") for row in org_rows
        }
        if len(main_content_ids) != spec.unique_content_count:
            errors.append(
                f"{spec.output_name}: main has {len(main_content_ids)} unique "
                f"contents, expected {spec.unique_content_count}"
            )
        if len(org_content_ids) != spec.unique_content_count:
            errors.append(
                f"{spec.output_name}: ORG has {len(org_content_ids)} unique "
                f"contents, expected {spec.unique_content_count}"
            )
        if main_content_ids != org_content_ids:
            errors.append(
                f"{spec.output_name}: main/ORG content-ID sets differ"
            )

        expected_cells = expected_main_cells(spec)
        for instance_name, rows in main_by_instance.items():
            cells = [_main_cell(row) for row in rows]
            if len(rows) != spec.configurations_per_instance:
                errors.append(
                    f"{spec.output_name}/{instance_name}: {len(rows)} main rows, "
                    f"expected {spec.configurations_per_instance}"
                )
            if set(cells) != expected_cells:
                missing = sorted(expected_cells - set(cells))
                extra = sorted(set(cells) - expected_cells)
                errors.append(
                    f"{spec.output_name}/{instance_name}: bad main matrix; "
                    f"missing={missing}, extra={extra}"
                )
            if len(cells) != len(set(cells)):
                errors.append(
                    f"{spec.output_name}/{instance_name}: duplicate main cells"
                )
            for row in rows:
                if row.get("status") not in {"OPTIMAL", "UNSAT"}:
                    errors.append(
                        f"{spec.output_name}/{instance_name}: incomplete Main "
                        f"status {row.get('status')!r}"
                    )
                if row.get("runtime_censored", "").lower() == "true":
                    errors.append(
                        f"{spec.output_name}/{instance_name}: censored Main row"
                    )
                if row.get("error_type"):
                    errors.append(
                        f"{spec.output_name}/{instance_name}: Main error "
                        f"{row.get('error_type')!r}"
                    )
                if row.get("instance_family") != spec.family:
                    errors.append(
                        f"{spec.output_name}/{instance_name}: family is "
                        f"{row.get('instance_family')!r}, expected {spec.family!r}"
                    )
                if row.get("idle_encoding") != "span_threshold":
                    errors.append(
                        f"{spec.output_name}/{instance_name}: unexpected B factor"
                    )
                if row.get("objective_code") != "IRP":
                    errors.append(
                        f"{spec.output_name}/{instance_name}: unexpected O factor"
                    )
                if row.get("encoding_variant") != "imp12+":
                    errors.append(
                        f"{spec.output_name}/{instance_name}: unexpected I variant"
                    )
                if row.get("factor_i") != "IC12+":
                    errors.append(
                        f"{spec.output_name}/{instance_name}: unexpected factor_i"
                    )
                if row.get("implied_constraints_code") != "IC12P":
                    errors.append(
                        f"{spec.output_name}/{instance_name}: unexpected I code"
                    )

        for instance_name, rows in org_by_instance.items():
            if len(rows) != 1:
                errors.append(
                    f"{spec.output_name}/{instance_name}: {len(rows)} ORG rows, "
                    "expected 1"
                )
            for row in rows:
                if row.get("status") not in {"OPTIMAL", "UNSAT"}:
                    errors.append(
                        f"{spec.output_name}/{instance_name}: incomplete ORG "
                        f"status {row.get('status')!r}"
                    )
                if row.get("runtime_censored", "").lower() == "true":
                    errors.append(
                        f"{spec.output_name}/{instance_name}: censored ORG row"
                    )
                if row.get("error_type"):
                    errors.append(
                        f"{spec.output_name}/{instance_name}: ORG error "
                        f"{row.get('error_type')!r}"
                    )
                if row.get("instance_family") != spec.family:
                    errors.append(
                        f"{spec.output_name}/{instance_name}: ORG family is "
                        f"{row.get('instance_family')!r}, expected {spec.family!r}"
                    )
                if row.get("configuration_label") != ORG_CONFIGURATION_LABEL:
                    errors.append(
                        f"{spec.output_name}/{instance_name}: unexpected ORG label"
                    )
                if row.get("configuration_id") != ORG_CONFIGURATION_ID:
                    errors.append(
                        f"{spec.output_name}/{instance_name}: unexpected ORG ID"
                    )
                if row.get("encoding_variant") != ORG_ENCODING_VARIANT:
                    errors.append(
                        f"{spec.output_name}/{instance_name}: unexpected ORG variant"
                    )
                if row.get("factor_i") != ORG_IMPLIED_PACKAGE_NAME:
                    errors.append(
                        f"{spec.output_name}/{instance_name}: unexpected ORG factor_i"
                    )
                if (
                    row.get("implied_constraints_code")
                    != ORG_IMPLIED_PACKAGE_CODE
                ):
                    errors.append(
                        f"{spec.output_name}/{instance_name}: unexpected ORG I code"
                    )
                if row.get("precedence_mode") != "traditional":
                    errors.append(
                        f"{spec.output_name}/{instance_name}: missing ORG "
                        "precedence_mode"
                    )
                if (
                    row.get("precedence_configuration")
                    != "pairwise+direct"
                ):
                    errors.append(
                        f"{spec.output_name}/{instance_name}: missing ORG "
                        "precedence_configuration"
                    )
                objective = row.get("objective_value", "")
                best = row.get("best_value", "")
                if row.get("status") == "OPTIMAL" and not objective:
                    errors.append(
                        f"{spec.output_name}/{instance_name}: OPTIMAL ORG row has "
                        "no objective_value"
                    )
                if best and objective != best:
                    errors.append(
                        f"{spec.output_name}/{instance_name}: ORG objective_value "
                        "does not equal best_value"
                    )

        duplicate_main = all_main_instances.intersection(main_by_instance)
        if duplicate_main:
            errors.append(
                f"{spec.output_name}: main paths repeated across families: "
                f"{sorted(duplicate_main)}"
            )
        duplicate_org = all_org_instances.intersection(org_by_instance)
        if duplicate_org:
            errors.append(
                f"{spec.output_name}: ORG paths repeated across families: "
                f"{sorted(duplicate_org)}"
            )
        duplicate_main_content = all_main_content_ids.intersection(
            main_content_ids
        )
        if duplicate_main_content:
            errors.append(
                f"{spec.output_name}: main contents repeated across families: "
                f"{sorted(duplicate_main_content)}"
            )
        duplicate_org_content = all_org_content_ids.intersection(org_content_ids)
        if duplicate_org_content:
            errors.append(
                f"{spec.output_name}: ORG contents repeated across families: "
                f"{sorted(duplicate_org_content)}"
            )
        all_main_instances.update(main_by_instance)
        all_org_instances.update(org_by_instance)
        all_main_aliases.update(main_aliases)
        all_org_aliases.update(org_aliases)
        all_main_content_ids.update(main_content_ids)
        all_org_content_ids.update(org_content_ids)

    if main_total != 1476:
        errors.append(f"main total is {main_total}, expected 1476")
    if org_total != 126:
        errors.append(f"ORG total is {org_total}, expected 126")
    if len(all_main_instances) != 126:
        errors.append(
            "main content-representative total is "
            f"{len(all_main_instances)}, expected 126"
        )
    if len(all_org_instances) != 126:
        errors.append(
            "ORG content-representative total is "
            f"{len(all_org_instances)}, expected 126"
        )
    if len(all_main_aliases) != 140:
        errors.append(
            f"main source-path coverage is {len(all_main_aliases)}, expected 140"
        )
    if len(all_org_aliases) != 140:
        errors.append(
            f"ORG source-path coverage is {len(all_org_aliases)}, expected 140"
        )
    if len(all_main_content_ids) != 126:
        errors.append(
            f"main unique-content total is {len(all_main_content_ids)}, expected 126"
        )
    if len(all_org_content_ids) != 126:
        errors.append(
            f"ORG unique-content total is {len(all_org_content_ids)}, expected 126"
        )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the canonical official IC12+ benchmark matrix."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="run output directory containing main/ and org/",
    )
    parser.add_argument(
        "--noves-dir",
        type=Path,
        help="official extracted noves directory used for Forbidden paths",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_output(args.output, args.noves_dir)
    if errors:
        print("OFFICIAL RUN VALIDATION: FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "OFFICIAL RUN VALIDATION: PASSED "
        "(140 official paths / 126 unique contents, "
        "1476 computed main rows / 1560 logical path cells, "
        "126 computed ORG rows / 140 logical paths)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
