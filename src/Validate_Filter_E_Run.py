from __future__ import annotations

import argparse
import csv
import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from Dataset_Manifest import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TERMINAL_STATUSES = {"OPTIMAL", "UNSAT", "TIMEOUT"}
EXPECTED_ENGINES = ("IncrementalSAT", "NonIncrementalSAT", "UWrMaxSAT")
EXPECTED_ENCODINGS = ("pairwise", "sparse_suffix")
EXPECTED_GRAPHS = ("direct", "distance_closure")
ENGINE_METADATA = {
    "IncrementalSAT": ("incremental", "CaDiCaL", "IS-CD", "cadical"),
    "NonIncrementalSAT": ("multiple", "CaDiCaL", "NIS-CD", "cadical"),
    "UWrMaxSAT": ("maxsat", "UWrMaxSAT", "UW", "uwrmaxsat"),
}
ENCODING_METADATA = {
    "pairwise": ("Pairwise", "PW"),
    "sparse_suffix": ("SparseSuffix", "SS"),
}
GRAPH_METADATA = {
    "direct": ("Direct-E", "DE"),
    "distance_closure": ("DistanceClosure-E*", "DC"),
}
EXPECTED_CELLS = set(
    itertools.product(
        EXPECTED_ENGINES,
        EXPECTED_ENCODINGS,
        EXPECTED_GRAPHS,
    )
)


@dataclass(frozen=True)
class DatasetSpec:
    output_name: str
    manifest: Path
    variants: tuple[str, ...]

    @property
    def instance_count(self) -> int:
        return 20 * len(self.variants)

    @property
    def configurations_per_instance(self) -> int:
        return len(EXPECTED_CELLS)

    @property
    def run_count(self) -> int:
        return self.instance_count * self.configurations_per_instance


DATASET_SPECS = (
    DatasetSpec(
        "official",
        PROJECT_ROOT / "instances_manifest.csv",
        ("prec15", "prec25"),
    ),
    DatasetSpec(
        "stress",
        PROJECT_ROOT / "data_precedence_stress" / "instances_manifest.csv",
        ("prec30", "prec35", "prec40"),
    ),
    DatasetSpec(
        "stress-high",
        PROJECT_ROOT
        / "data_precedence_stress_high"
        / "instances_manifest.csv",
        ("prec50", "prec60"),
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def precedence_manifest_rows(spec: DatasetSpec) -> list[dict[str, str]]:
    rows = [
        row
        for row in read_csv(spec.manifest)
        if row.get("family") == "precedence"
    ]
    return rows


def validate_manifests() -> list[str]:
    errors: list[str] = []
    all_content_ids: set[str] = set()
    all_instance_names: set[str] = set()

    for spec in DATASET_SPECS:
        try:
            rows = precedence_manifest_rows(spec)
        except FileNotFoundError:
            errors.append(f"missing manifest: {spec.manifest}")
            continue

        if len(rows) != spec.instance_count:
            errors.append(
                f"{spec.output_name}: manifest has {len(rows)} precedence "
                f"instances, expected {spec.instance_count}"
            )
        variant_counts = Counter(row.get("variant", "") for row in rows)
        expected_variant_counts = Counter({variant: 20 for variant in spec.variants})
        if variant_counts != expected_variant_counts:
            errors.append(
                f"{spec.output_name}: variant counts are "
                f"{dict(sorted(variant_counts.items()))}, expected "
                f"{dict(sorted(expected_variant_counts.items()))}"
            )

        content_ids = {row.get("content_id", "") for row in rows}
        instance_names = {row.get("canonical_instance", "") for row in rows}
        if "" in content_ids or len(content_ids) != len(rows):
            errors.append(
                f"{spec.output_name}: content IDs are missing or duplicated"
            )
        if "" in instance_names or len(instance_names) != len(rows):
            errors.append(
                f"{spec.output_name}: canonical instance names are missing "
                "or duplicated"
            )
        duplicate_content_ids = all_content_ids.intersection(content_ids)
        if duplicate_content_ids:
            errors.append(
                f"{spec.output_name}: content IDs overlap an earlier dataset: "
                f"{sorted(duplicate_content_ids)}"
            )
        duplicate_instance_names = all_instance_names.intersection(instance_names)
        if duplicate_instance_names:
            errors.append(
                f"{spec.output_name}: instance names overlap an earlier dataset: "
                f"{sorted(duplicate_instance_names)}"
            )

        for row in rows:
            try:
                edge_count = int(row.get("n_direct_precedence_edges", ""))
            except ValueError:
                edge_count = 0
            if edge_count <= 0:
                errors.append(
                    f"{spec.output_name}/{row.get('canonical_instance', '')}: "
                    "manifest does not record a positive direct-precedence count"
                )
            run_path = Path(row.get("canonical_run_path", ""))
            if not run_path.is_absolute():
                run_path = spec.manifest.parent / run_path
            if not run_path.is_file():
                errors.append(
                    f"{spec.output_name}/{row.get('canonical_instance', '')}: "
                    f"missing run path {run_path}"
                )
            elif file_sha256(run_path) != row.get("sha256"):
                errors.append(
                    f"{spec.output_name}/{row.get('canonical_instance', '')}: "
                    "run-path SHA-256 differs from manifest"
                )

        all_content_ids.update(content_ids)
        all_instance_names.update(instance_names)

    expected_instances = sum(spec.instance_count for spec in DATASET_SPECS)
    if len(all_content_ids) != expected_instances:
        errors.append(
            f"all datasets contain {len(all_content_ids)} unique content IDs, "
            f"expected {expected_instances}"
        )
    if len(all_instance_names) != expected_instances:
        errors.append(
            f"all datasets contain {len(all_instance_names)} unique instance "
            f"names, expected {expected_instances}"
        )
    return errors


def _cell(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("optimization_engine", ""),
        row.get("precedence_encoding", ""),
        row.get("precedence_graph", ""),
    )


def _is_true(value: str) -> bool:
    return value.strip().lower() == "true"


def _integer(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _expected_identity(
    engine: str,
    precedence_encoding: str,
    precedence_graph: str,
) -> tuple[str, str]:
    _, _, engine_code, backend_code = ENGINE_METADATA[engine]
    _, encoding_code = ENCODING_METADATA[precedence_encoding]
    _, graph_code = GRAPH_METADATA[precedence_graph]
    label = (
        f"R-FE-{encoding_code}-{graph_code}-ST-IRP-{engine_code}-IC12P"
    )
    identifier = "__".join(
        (
            "cfg4",
            "m-reduced",
            "f-direct",
            f"p-{precedence_encoding}",
            f"g-{precedence_graph}",
            "b-span_threshold",
            "o-idle_range_pstar",
            f"s-{engine.lower()}",
            "i-imp12plus",
            f"backend-{backend_code}",
        )
    )
    return label, identifier


def validate_output(output: Path) -> list[str]:
    errors = validate_manifests()
    total_rows = 0
    all_result_content_ids: set[str] = set()

    for spec in DATASET_SPECS:
        detailed_path = output / "main" / f"{spec.output_name}_detailed.csv"
        try:
            rows = read_csv(detailed_path)
            manifest_rows = precedence_manifest_rows(spec)
        except FileNotFoundError as exc:
            errors.append(f"missing CSV: {exc}")
            continue

        total_rows += len(rows)
        expected_by_content = {
            row["content_id"]: row for row in manifest_rows
        }
        rows_by_content: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            rows_by_content[row.get("instance_content_id", "")].append(row)

        if set(rows_by_content) != set(expected_by_content):
            missing = sorted(set(expected_by_content) - set(rows_by_content))
            extra = sorted(set(rows_by_content) - set(expected_by_content))
            errors.append(
                f"{spec.output_name}: result content IDs differ from manifest; "
                f"missing={missing}, extra={extra}"
            )

        for content_id, instance_rows in rows_by_content.items():
            manifest_row = expected_by_content.get(content_id)
            instance_name = (
                manifest_row["canonical_instance"]
                if manifest_row is not None
                else content_id
            )
            cells = [_cell(row) for row in instance_rows]
            if len(instance_rows) != spec.configurations_per_instance:
                errors.append(
                    f"{spec.output_name}/{instance_name}: "
                    f"{len(instance_rows)} rows, expected "
                    f"{spec.configurations_per_instance}"
                )
            if set(cells) != EXPECTED_CELLS:
                errors.append(
                    f"{spec.output_name}/{instance_name}: bad Filter-E matrix; "
                    f"missing={sorted(EXPECTED_CELLS - set(cells))}, "
                    f"extra={sorted(set(cells) - EXPECTED_CELLS)}"
                )
            if len(cells) != len(set(cells)):
                errors.append(
                    f"{spec.output_name}/{instance_name}: duplicate Filter-E cells"
                )

            reduced_candidate_counts: set[int] = set()
            optimum_values: set[int] = set()
            statuses: set[str] = set()
            for row in instance_rows:
                engine, precedence_encoding, precedence_graph = _cell(row)
                expected_label = ""
                expected_identifier = ""
                if (engine, precedence_encoding, precedence_graph) in EXPECTED_CELLS:
                    expected_label, expected_identifier = _expected_identity(
                        engine,
                        precedence_encoding,
                        precedence_graph,
                    )
                if manifest_row is not None:
                    if row.get("instance") != manifest_row["canonical_instance"]:
                        errors.append(
                            f"{spec.output_name}/{instance_name}: result instance "
                            "name differs from manifest"
                        )
                    if row.get("instance_variant") != manifest_row["variant"]:
                        errors.append(
                            f"{spec.output_name}/{instance_name}: result variant "
                            "differs from manifest"
                        )
                    if row.get("instance_sha256") != manifest_row["sha256"]:
                        errors.append(
                            f"{spec.output_name}/{instance_name}: result SHA-256 "
                            "differs from manifest"
                        )
                    for key in (
                        "source_alias_count",
                        "source_alias_paths",
                        "repository_alias_count",
                        "repository_alias_paths",
                        "dataset_source_page",
                        "dataset_archive_url",
                        "dataset_archive_sha256",
                    ):
                        if row.get(key, "") != manifest_row.get(key, ""):
                            errors.append(
                                f"{spec.output_name}/{instance_name}: {key} "
                                "differs from manifest"
                            )
                if row.get("instance_family") != "precedence":
                    errors.append(
                        f"{spec.output_name}/{instance_name}: family is not "
                        "precedence"
                    )
                if row.get("domain_mode") != "reduced":
                    errors.append(
                        f"{spec.output_name}/{instance_name}: Filter-E row is not "
                        "Reduced"
                    )
                if row.get("domain_filter_graph") != "direct":
                    errors.append(
                        f"{spec.output_name}/{instance_name}: "
                        "domain_filter_graph is not direct"
                    )
                if row.get("factor_f") != "Filter-E":
                    errors.append(
                        f"{spec.output_name}/{instance_name}: factor_f is not "
                        "Filter-E"
                    )
                if row.get("factor_m") != "Reduced":
                    errors.append(
                        f"{spec.output_name}/{instance_name}: factor_m is not "
                        "Reduced"
                    )
                if row.get("factor_b") != "SpanThreshold":
                    errors.append(
                        f"{spec.output_name}/{instance_name}: unexpected factor_b"
                    )
                if row.get("factor_o") != "IdleRangePstar":
                    errors.append(
                        f"{spec.output_name}/{instance_name}: unexpected factor_o"
                    )
                configuration_id = row.get("configuration_id", "")
                if row.get("configuration_label") != expected_label:
                    errors.append(
                        f"{spec.output_name}/{instance_name}: unexpected "
                        "configuration label"
                    )
                if configuration_id != expected_identifier:
                    errors.append(
                        f"{spec.output_name}/{instance_name}: unexpected "
                        "configuration ID"
                    )
                if row.get("configuration_key") != configuration_id:
                    errors.append(
                        f"{spec.output_name}/{instance_name}: configuration key "
                        "differs from ID"
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
                if row.get("idle_encoding") != "span_threshold":
                    errors.append(
                        f"{spec.output_name}/{instance_name}: unexpected B encoding"
                    )
                if row.get("objective_code") != "IRP":
                    errors.append(
                        f"{spec.output_name}/{instance_name}: unexpected O code"
                    )
                if precedence_encoding in ENCODING_METADATA:
                    if (
                        row.get("factor_p")
                        != ENCODING_METADATA[precedence_encoding][0]
                    ):
                        errors.append(
                            f"{spec.output_name}/{instance_name}: factor_p does "
                            "not match precedence_encoding"
                        )
                if precedence_graph in GRAPH_METADATA:
                    if row.get("factor_g") != GRAPH_METADATA[precedence_graph][0]:
                        errors.append(
                            f"{spec.output_name}/{instance_name}: factor_g does "
                            "not match precedence_graph"
                        )
                if engine in ENGINE_METADATA:
                    solver_name, solver_backend, _, _ = ENGINE_METADATA[engine]
                    if row.get("solver") != solver_name:
                        errors.append(
                            f"{spec.output_name}/{instance_name}: solver does not "
                            "match optimization engine"
                        )
                    if row.get("solver_backend") != solver_backend:
                        errors.append(
                            f"{spec.output_name}/{instance_name}: backend does not "
                            "match optimization engine"
                        )
                    if row.get("factor_s") != engine:
                        errors.append(
                            f"{spec.output_name}/{instance_name}: factor_s does "
                            "not match optimization engine"
                        )
                    if engine == "UWrMaxSAT":
                        if not row.get("solver_binary"):
                            errors.append(
                                f"{spec.output_name}/{instance_name}: missing "
                                "UWrMaxSAT binary"
                            )
                        if len(row.get("solver_binary_sha256", "")) != 64:
                            errors.append(
                                f"{spec.output_name}/{instance_name}: missing "
                                "UWrMaxSAT binary SHA-256"
                            )
                        if (
                            row.get("status") != "TIMEOUT"
                            and not row.get("solver_command")
                        ):
                            errors.append(
                                f"{spec.output_name}/{instance_name}: missing "
                                "UWrMaxSAT command"
                            )
                    elif row.get("solver_version") != "1.5.3":
                        errors.append(
                            f"{spec.output_name}/{instance_name}: SAT backend is "
                            "not CaDiCaL 1.5.3"
                        )

                status = row.get("status", "")
                statuses.add(status)
                if status not in TERMINAL_STATUSES:
                    errors.append(
                        f"{spec.output_name}/{instance_name}: non-terminal status "
                        f"{status!r}"
                    )
                if status == "TIMEOUT" and not _is_true(
                    row.get("runtime_censored", "")
                ):
                    errors.append(
                        f"{spec.output_name}/{instance_name}: TIMEOUT is not "
                        "marked runtime_censored"
                    )
                if status != "TIMEOUT" and _is_true(
                    row.get("runtime_censored", "")
                ):
                    errors.append(
                        f"{spec.output_name}/{instance_name}: completed row is "
                        "marked runtime_censored"
                    )
                for key in ("validation_errors", "error_type", "error_message"):
                    if row.get(key):
                        errors.append(
                            f"{spec.output_name}/{instance_name}: non-empty {key}"
                        )

                reduced_count = _integer(
                    row.get("reduced_schedule_candidates", "")
                )
                active_count = _integer(
                    row.get("active_schedule_candidates", "")
                )
                if reduced_count is None or active_count != reduced_count:
                    errors.append(
                        f"{spec.output_name}/{instance_name}: inconsistent "
                        "Reduced candidate counts"
                    )
                else:
                    reduced_candidate_counts.add(reduced_count)

                direct_edges = _integer(row.get("precedence_direct_edges", ""))
                closure_edges = _integer(row.get("precedence_closure_edges", ""))
                relation_edges = _integer(
                    row.get("precedence_relation_edges", "")
                )
                manifest_direct_edges = (
                    _integer(manifest_row["n_direct_precedence_edges"])
                    if manifest_row is not None
                    else None
                )
                if direct_edges != manifest_direct_edges:
                    errors.append(
                        f"{spec.output_name}/{instance_name}: direct-edge count "
                        "differs from manifest"
                    )
                expected_relations = (
                    direct_edges
                    if precedence_graph == "direct"
                    else closure_edges
                )
                if relation_edges != expected_relations:
                    errors.append(
                        f"{spec.output_name}/{instance_name}: relation-edge count "
                        "does not match G"
                    )

                if status == "OPTIMAL":
                    objective_values = {
                        _integer(row.get(key, ""))
                        for key in (
                            "best_value",
                            "proven_optimum",
                            "objective_value",
                            "idle_range_pstar",
                        )
                    }
                    if None in objective_values or len(objective_values) != 1:
                        errors.append(
                            f"{spec.output_name}/{instance_name}: inconsistent "
                            "optimal objective fields"
                        )
                    else:
                        optimum_values.update(objective_values)

                for key in (
                    "run_started_utc",
                    "timeout_seconds",
                    "git_commit",
                    "git_dirty",
                    "runner_command",
                    "runtime_scope",
                ):
                    if row.get(key, "") == "":
                        errors.append(
                            f"{spec.output_name}/{instance_name}: missing "
                            f"provenance field {key}"
                        )
                runner_command = row.get("runner_command", "")
                for required_arg in (
                    "--domain-mode reduced",
                    "--domain-filter-graph direct",
                    "--precedence-encoding both",
                    "--precedence-graph both",
                ):
                    if required_arg not in runner_command:
                        errors.append(
                            f"{spec.output_name}/{instance_name}: runner command "
                            f"does not contain {required_arg!r}"
                        )

            if len(reduced_candidate_counts) != 1:
                errors.append(
                    f"{spec.output_name}/{instance_name}: Reduced candidate "
                    "count differs across Filter-E cells"
                )
            if len(optimum_values) > 1:
                errors.append(
                    f"{spec.output_name}/{instance_name}: proven optima differ "
                    "across Filter-E cells"
                )
            if "UNSAT" in statuses and "OPTIMAL" in statuses:
                errors.append(
                    f"{spec.output_name}/{instance_name}: SAT/UNSAT disagreement"
                )

        duplicate_content_ids = all_result_content_ids.intersection(rows_by_content)
        if duplicate_content_ids:
            errors.append(
                f"{spec.output_name}: result content IDs overlap an earlier "
                f"dataset: {sorted(duplicate_content_ids)}"
            )
        all_result_content_ids.update(rows_by_content)

        if len(rows) != spec.run_count:
            errors.append(
                f"{spec.output_name}: {len(rows)} rows, expected {spec.run_count}"
            )

    expected_total = sum(spec.run_count for spec in DATASET_SPECS)
    if total_rows != expected_total:
        errors.append(f"Filter-E total is {total_rows}, expected {expected_total}")
    return errors


def print_plan() -> None:
    print("Filter-E precedence matrix")
    for spec in DATASET_SPECS:
        variants = "/".join(spec.variants)
        print(
            f"- {spec.output_name}: {spec.instance_count} instances "
            f"({variants}) x {spec.configurations_per_instance} = "
            f"{spec.run_count} runs"
        )
    print(
        "- total: "
        f"{sum(spec.instance_count for spec in DATASET_SPECS)} instances x "
        f"{len(EXPECTED_CELLS)} = "
        f"{sum(spec.run_count for spec in DATASET_SPECS)} runs"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate all Reduced Filter-E cells across official precedence, "
            "stress, and stress-high datasets."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="run output directory containing main/*_detailed.csv",
    )
    parser.add_argument(
        "--check-manifests-only",
        action="store_true",
        help="validate dataset coverage and print the run plan without CSVs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.check_manifests_only and args.output is None:
        raise SystemExit("--output is required unless --check-manifests-only is used")

    errors = (
        validate_manifests()
        if args.check_manifests_only
        else validate_output(args.output)
    )
    print_plan()
    if errors:
        print("FILTER-E RUN VALIDATION: FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("FILTER-E RUN VALIDATION: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
