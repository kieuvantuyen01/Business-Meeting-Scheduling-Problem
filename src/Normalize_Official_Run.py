from __future__ import annotations

import argparse
import csv
from pathlib import Path

from Validate_Official_Run import (
    DATASET_SPECS,
    ORG_CONFIGURATION_ID,
    ORG_CONFIGURATION_LABEL,
    ORG_ENCODING_VARIANT,
    ORG_IMPLIED_PACKAGE_CODE,
    ORG_IMPLIED_PACKAGE_NAME,
    source_alias_names,
    validate_output,
)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_org_metadata(row: dict[str, str]) -> None:
    row["configuration_label"] = ORG_CONFIGURATION_LABEL
    row["configuration_id"] = ORG_CONFIGURATION_ID
    row["configuration_key"] = ORG_CONFIGURATION_ID
    row["factor_i"] = ORG_IMPLIED_PACKAGE_NAME
    row["encoding_variant"] = ORG_ENCODING_VARIANT
    row["implied_constraints_code"] = ORG_IMPLIED_PACKAGE_CODE
    row["precedence_mode"] = "traditional"
    row["precedence_configuration"] = "pairwise+direct"
    row["objective_value"] = row.get("solver_cost") or row.get("best_value", "")


def normalize_run(source: Path, destination: Path) -> list[str]:
    report: list[str] = []
    for spec in DATASET_SPECS:
        detailed_name = f"{spec.output_name}_detailed.csv"
        detailed_source = source / "main" / detailed_name
        main_fields, all_main_rows = read_rows(detailed_source)
        main_rows = [
            row
            for row in all_main_rows
            if row.get("instance_family") == spec.family
        ]
        write_rows(destination / "main" / detailed_name, main_fields, main_rows)

        aggregate_name = f"{spec.output_name}_aggregate.csv"
        aggregate_source = source / "main" / aggregate_name
        aggregate_fields, all_aggregate_rows = read_rows(aggregate_source)
        kept_instances = {row["instance"] for row in main_rows}
        aggregate_rows = [
            row
            for row in all_aggregate_rows
            if row.get("instance") in kept_instances
        ]
        write_rows(
            destination / "main" / aggregate_name,
            aggregate_fields,
            aggregate_rows,
        )

        org_name = f"{spec.output_name}_org_new.csv"
        org_source = source / "org" / org_name
        org_fields, all_org_rows = read_rows(org_source)
        org_rows = [
            row
            for row in all_org_rows
            if row.get("instance_family") == spec.family
        ]
        for row in org_rows:
            normalize_org_metadata(row)
        write_rows(destination / "org" / org_name, org_fields, org_rows)

        report.append(
            f"{spec.family}: Main {len(main_rows)}/{len(all_main_rows)} rows; "
            f"ORG {len(org_rows)}/{len(all_org_rows)} rows; "
            f"source paths {len(source_alias_names(main_rows))}"
        )

    errors = validate_output(destination)
    if errors:
        raise RuntimeError(
            "normalized output failed validation:\n- " + "\n- ".join(errors)
        )
    report.append(
        "validated: 1476 computed Main rows, 126 computed ORG rows, "
        "covering 140 official paths"
    )
    report_path = destination / "normalization_report.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter cross-family rows from an existing official run and "
            "normalize metadata without rerunning solvers."
        )
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        print(f"ERROR: output directory is not empty: {args.output}")
        return 2
    try:
        report = normalize_run(args.source, args.output)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    for line in report:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
