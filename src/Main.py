from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import queue as queue_module
import time
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # Memory remains optional for benchmark portability.
    psutil = None

from IncrementalSAT_Solver import B2BIncrementalSATSolver
from MaxSAT_Solver import (
    B2BMaxSATSolver,
    UWRMAXSAT_NOT_FOUND_MESSAGE,
    executable_sha256,
    resolve_uwrmaxsat_binary,
)
from Multiple_SAT import B2BMultipleSATSolver
from SAT_Backend import require_sat_backend


VARIANTS = ["basic", "imp1", "imp2", "imp12", "imp12+"]
SOLVERS = ["incremental", "multiple", "maxsat"]
PRECEDENCE_MODES = ["traditional", "staircase"]
PRECEDENCE_ENCODINGS = ["pairwise", "sparse_suffix"]
PRECEDENCE_GRAPHS = ["direct", "distance_closure"]
DOMAIN_MODES = ["full", "reduced"]
MAXSAT_BACKENDS = ["uwrmaxsat", "rc2"]
SAT_BACKENDS = ["cadical", "glucose"]
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.05
QUEUE_GRACE_SECONDS = 1.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the conference model: minimize IdleRange(P*) over "
            "Full and/or Reduced meeting-slot domains."
        )
    )
    parser.add_argument("--instance", help="single .dzn instance")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="directory containing .dzn instances when --instance is omitted",
    )
    parser.add_argument(
        "--solver",
        choices=[*SOLVERS, "all"],
        default="all",
    )
    parser.add_argument(
        "--maxsat-backend",
        choices=MAXSAT_BACKENDS,
        default="uwrmaxsat",
        help="UWrMaxSAT is required by default; RC2 is development-only",
    )
    parser.add_argument(
        "--uwrmaxsat-bin",
        help="pinned UWrMaxSAT executable; otherwise use UWRMAXSAT_BIN/local/PATH",
    )
    parser.add_argument(
        "--uwrmaxsat-sha256",
        help="optional expected SHA-256 of the pinned UWrMaxSAT executable",
    )
    parser.add_argument(
        "--sat-backend",
        choices=SAT_BACKENDS,
        default="cadical",
        help="CaDiCaL 1.5.3 is required by default; Glucose is development-only",
    )
    parser.add_argument(
        "--precedence-mode",
        choices=[*PRECEDENCE_MODES, "both"],
        help="deprecated composite alias; use the independent P/G flags",
    )
    parser.add_argument(
        "--precedence-encoding",
        choices=[*PRECEDENCE_ENCODINGS, "both"],
        help="P factor; defaults to both",
    )
    parser.add_argument(
        "--precedence-graph",
        choices=[*PRECEDENCE_GRAPHS, "both"],
        help="G factor; defaults to both",
    )
    parser.add_argument(
        "--encoding-variant",
        choices=[*VARIANTS, "all"],
        default="all",
    )
    parser.add_argument(
        "--domain-mode",
        choices=[*DOMAIN_MODES, "both"],
        default="both",
        help="Full=MxT variables; Reduced=exact preprocessing fixpoint",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=7200.0,
        help="wall-clock timeout per run in seconds",
    )
    parser.add_argument(
        "--csv",
        default="table3_results.csv",
        help="aggregated Table-3 CSV path",
    )
    parser.add_argument(
        "--long-csv",
        help="detailed CSV path; defaults to <csv-stem>_detailed.csv",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.precedence_mode is not None and (
        args.precedence_encoding is not None or args.precedence_graph is not None
    ):
        parser.error(
            "--precedence-mode cannot be combined with independent P/G flags"
        )
    return args


def selected(choice: str, values: list[str], all_name: str = "all") -> list[str]:
    return values if choice == all_name else [choice]


def precedence_configurations(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Return controlled P×G cells or the requested deprecated composites."""

    if args.precedence_mode is not None:
        legacy = {
            "traditional": ("pairwise", "direct"),
            "staircase": ("sparse_suffix", "distance_closure"),
        }
        modes = selected(args.precedence_mode, PRECEDENCE_MODES, "both")
        return [legacy[mode] for mode in modes]

    encodings = selected(
        args.precedence_encoding or "both",
        PRECEDENCE_ENCODINGS,
        "both",
    )
    graphs = selected(
        args.precedence_graph or "both",
        PRECEDENCE_GRAPHS,
        "both",
    )
    return [
        (precedence_encoding, precedence_graph)
        for precedence_encoding in encodings
        for precedence_graph in graphs
    ]


def legacy_precedence_mode(
    precedence_encoding: str,
    precedence_graph: str,
) -> str:
    if (precedence_encoding, precedence_graph) == ("pairwise", "direct"):
        return "traditional"
    if (precedence_encoding, precedence_graph) == (
        "sparse_suffix",
        "distance_closure",
    ):
        return "staircase"
    return "factorial"


def collect_instances(instance: str | None, data_dir: str) -> list[Path]:
    if instance:
        path = Path(instance)
        if not path.is_file():
            raise FileNotFoundError(path)
        return [path]

    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    instances = sorted(directory.glob("*.dzn"))
    if not instances:
        raise FileNotFoundError(f"No .dzn files found in {directory}")
    return instances


def require_solver_environment(
    solvers: list[str],
    *,
    maxsat_backend: str,
    uwrmaxsat_bin: str | None,
    uwrmaxsat_sha256: str | None,
    sat_backend: str,
) -> None:
    """Validate all requested production backends before the first run."""

    if "maxsat" in solvers and maxsat_backend == "uwrmaxsat":
        binary = resolve_uwrmaxsat_binary(uwrmaxsat_bin)
        if binary is None:
            raise FileNotFoundError(UWRMAXSAT_NOT_FOUND_MESSAGE)
        actual_sha256 = executable_sha256(binary)
        expected_sha256 = (uwrmaxsat_sha256 or "").strip().lower()
        if expected_sha256 and (
            len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)
        ):
            raise ValueError(
                "--uwrmaxsat-sha256 must be a 64-character hex digest"
            )
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise RuntimeError(
                "UWrMaxSAT executable SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

    if any(solver in {"incremental", "multiple"} for solver in solvers):
        require_sat_backend(sat_backend)


def status_to_sat_result(status: str | None) -> str:
    normalized = (status or "ERROR").upper()
    if normalized in {"OPTIMAL", "SAT", "SATISFIABLE", "OPTIMUM FOUND"}:
        return "SAT"
    if normalized in {"UNSAT", "UNSATISFIABLE"}:
        return "UNSAT"
    if normalized == "TIMEOUT":
        return "TIMEOUT"
    return "ERROR"


def serialize_list(values: list[int] | tuple[int, ...] | None) -> str:
    return "" if values is None else ",".join(str(value) for value in values)


def serialize_assignment(values: list[int] | None) -> str:
    if values is None:
        return ""
    return ",".join(str(value + 1) if value >= 0 else "-" for value in values)


def serialize_schedule(meetings_per_slot: list[list[int]] | None) -> str:
    if meetings_per_slot is None:
        return ""
    return " | ".join(
        f"{slot}:" + " ".join(f"M{meeting + 1}" for meeting in meetings)
        for slot, meetings in enumerate(meetings_per_slot, start=1)
    )


def _solver_class(solver_name: str):
    if solver_name == "incremental":
        return B2BIncrementalSATSolver
    if solver_name == "multiple":
        return B2BMultipleSATSolver
    if solver_name == "maxsat":
        return B2BMaxSATSolver
    raise ValueError(f"Unknown solver: {solver_name}")


def _formula_metadata(solver_name: str, solver_object: Any) -> dict[str, Any]:
    artifacts = solver_object.artifacts
    n_soft = len(artifacts.objective_lits) if solver_name == "maxsat" else 0
    return {
        "message_type": "metadata",
        "objective": artifacts.objective_name,
        "objective_participant_count": len(artifacts.objective_participants),
        "objective_participants": serialize_list(
            tuple(participant + 1 for participant in artifacts.objective_participants)
        ),
        "precedence_mode": artifacts.precedence_mode,
        "precedence_encoding": artifacts.precedence_encoding,
        "precedence_graph": artifacts.precedence_graph,
        "precedence_configuration": artifacts.precedence_configuration,
        "domain_mode": artifacts.domain_mode,
        "full_schedule_candidates": artifacts.full_schedule_candidates,
        "unary_eligible_schedule_candidates": (
            artifacts.unary_eligible_schedule_candidates
        ),
        "initial_schedule_candidates": artifacts.initial_schedule_candidates,
        "reduced_schedule_candidates": artifacts.reduced_schedule_candidates,
        "active_schedule_candidates": artifacts.active_schedule_candidates,
        "unary_removed_schedule_candidates": (
            artifacts.unary_removed_schedule_candidates
        ),
        "preprocessing_removed_schedule_candidates": (
            artifacts.preprocessing_removed_schedule_candidates
        ),
        # Backward-compatible alias: removals made by exact preprocessing.
        "removed_schedule_candidates": artifacts.removed_schedule_candidates,
        "n_vars": artifacts.n_vars,
        "n_hard_clauses": artifacts.n_clauses,
        "n_soft_clauses": n_soft,
        "n_total_clauses": artifacts.n_clauses + n_soft,
        "n_objective_lits": len(artifacts.objective_lits),
        "precedence_direct_edges": artifacts.precedence_direct_edges,
        "precedence_closure_edges": artifacts.precedence_transitive_edges,
        "precedence_max_distance": artifacts.precedence_max_distance,
        "precedence_relation_edges": artifacts.precedence_relation_edges,
        "precedence_pairwise_clauses": artifacts.precedence_pairwise_clauses,
        "precedence_sparse_link_clauses": (
            artifacts.precedence_sparse_link_clauses
        ),
        "precedence_unique_suffix_cuts": (
            artifacts.precedence_unique_suffix_cuts
        ),
        "enabled_constraints": " | ".join(artifacts.enabled_constraints),
        "solver_backend": getattr(solver_object, "solver_backend", ""),
        "solver_version": getattr(solver_object, "solver_version", ""),
        "solver_binary": getattr(solver_object, "solver_binary", ""),
        "solver_command": getattr(solver_object, "solver_command", ""),
        "maxsat_backend_preference": getattr(solver_object, "backend", ""),
        "sat_backend_preference": getattr(solver_object, "solver_name", ""),
        "resolved_uwrmaxsat_bin": str(
            getattr(solver_object, "resolved_uwrmaxsat_bin", "") or ""
        ),
        "solver_binary_sha256": getattr(
            solver_object, "uwrmaxsat_binary_sha256", ""
        ),
    }


def _result_payload(
    result: dict[str, Any],
    *,
    solver_name: str,
    precedence_encoding: str,
    precedence_graph: str,
    encoding_variant: str,
    domain_mode: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    stats = result.get("stats")
    payload = {
        "message_type": "result",
        "status": result.get("status", "ERROR"),
        "solver": solver_name,
        "solver_backend": result.get("solver_backend", result.get("solver", "")),
        "solver_binary": result.get("solver_binary", ""),
        "solver_binary_sha256": result.get("solver_binary_sha256", ""),
        "solver_version": result.get("solver_version", ""),
        "solver_command": result.get("solver_command", ""),
        "solver_message": result.get("solver_message", ""),
        "maxsat_backend_preference": result.get("maxsat_backend_preference", ""),
        "sat_backend_preference": result.get("sat_backend_preference", ""),
        "precedence_mode": result.get(
            "precedence_mode",
            legacy_precedence_mode(precedence_encoding, precedence_graph),
        ),
        "precedence_encoding": precedence_encoding,
        "precedence_graph": precedence_graph,
        "precedence_configuration": (
            f"{precedence_encoding}+{precedence_graph}"
        ),
        "encoding_variant": encoding_variant,
        "domain_mode": domain_mode,
        "runtime_seconds": round(runtime_seconds, 6),
        "objective": result.get("objective", "internal_idle_slot_range_pstar"),
        "objective_value": result.get("objective_value"),
        "proven_optimum": result.get("proven_optimum"),
        "objective_participant_count": result.get(
            "objective_participant_count"
        ),
        "objective_participants": serialize_list(
            result.get("objective_participants")
        ),
        "idle_range_pstar": None if stats is None else stats.idle_range,
        "all_participant_idle_range": (
            None if stats is None else stats.all_participant_idle_range
        ),
        "total_internal_idle_slots": (
            None if stats is None else stats.total_internal_idle_slots
        ),
        "participant_internal_idle_slots": (
            ""
            if stats is None
            else serialize_list(stats.participant_internal_idle_slots)
        ),
        "busy_participants_per_slot": (
            "" if stats is None else serialize_list(stats.busy_participants_per_slot)
        ),
        "assignment": serialize_assignment(result.get("assignment")),
        "schedule_by_slot": (
            "" if stats is None else serialize_schedule(stats.meetings_per_slot)
        ),
        "validation_errors": "; ".join(result.get("validation_errors", [])),
        "error_type": "",
        "error_message": "",
    }
    payload["sat_result"] = status_to_sat_result(payload["status"])
    return payload


def _worker(
    solver_name: str,
    instance_path: str,
    precedence_encoding: str,
    precedence_graph: str,
    encoding_variant: str,
    domain_mode: str,
    maxsat_backend: str,
    uwrmaxsat_bin: str | None,
    uwrmaxsat_sha256: str | None,
    sat_backend: str,
    solver_timeout: float,
    verbose: bool,
    output: mp.Queue[Any],
) -> None:
    started = time.perf_counter()
    try:
        solver_kwargs: dict[str, Any] = {
            "instance_or_path": instance_path,
            "precedence_encoding": precedence_encoding,
            "precedence_graph": precedence_graph,
            "encoding_variant": encoding_variant,
            "domain_mode": domain_mode,
        }
        if solver_name == "maxsat":
            solver_kwargs.update(
                backend=maxsat_backend,
                uwrmaxsat_bin=uwrmaxsat_bin,
                uwrmaxsat_sha256=uwrmaxsat_sha256,
                uwrmaxsat_timeout=solver_timeout,
            )
        else:
            solver_kwargs["solver_name"] = sat_backend
        solver_object = _solver_class(solver_name)(
            **solver_kwargs
        )
        output.put(_formula_metadata(solver_name, solver_object))
        result = solver_object.solve(verbose=verbose)
        output.put(
            _result_payload(
                result,
                solver_name=solver_name,
                precedence_encoding=precedence_encoding,
                precedence_graph=precedence_graph,
                encoding_variant=encoding_variant,
                domain_mode=domain_mode,
                runtime_seconds=time.perf_counter() - started,
            )
        )
    except BaseException as exc:
        output.put(
            {
                "message_type": "result",
                "status": "ERROR",
                "sat_result": "ERROR",
                "solver": solver_name,
                "precedence_mode": legacy_precedence_mode(
                    precedence_encoding, precedence_graph
                ),
                "precedence_encoding": precedence_encoding,
                "precedence_graph": precedence_graph,
                "precedence_configuration": (
                    f"{precedence_encoding}+{precedence_graph}"
                ),
                "encoding_variant": encoding_variant,
                "domain_mode": domain_mode,
                "runtime_seconds": round(time.perf_counter() - started, 6),
                "objective": "internal_idle_slot_range_pstar",
                "maxsat_backend_preference": (
                    maxsat_backend if solver_name == "maxsat" else ""
                ),
                "sat_backend_preference": (
                    sat_backend if solver_name != "maxsat" else ""
                ),
                "validation_errors": "",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )


def _process_tree_rss_bytes(pid: int) -> int | None:
    if psutil is None:
        return None
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.Error, OSError):
        return None
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except (psutil.Error, OSError):
            continue
    return total


def _terminate_process_tree(process: mp.Process, grace_seconds: float = 5.0) -> None:
    """Terminate the worker and any external solver children it launched."""

    if process.pid is None:
        return

    if psutil is None:
        process.terminate()
        process.join(grace_seconds)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(grace_seconds)
        return

    try:
        root = psutil.Process(process.pid)
        children = root.children(recursive=True)
    except (psutil.Error, OSError):
        children = []

    for child in children:
        try:
            child.terminate()
        except (psutil.Error, OSError):
            continue

    process.terminate()
    process.join(grace_seconds)

    _, alive_children = psutil.wait_procs(children, timeout=grace_seconds)
    for child in alive_children:
        try:
            child.kill()
        except (psutil.Error, OSError):
            continue

    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(grace_seconds)


def _drain_queue(
    output: mp.Queue[Any],
    metadata: dict[str, Any],
    result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    while True:
        try:
            message = output.get_nowait()
        except queue_module.Empty:
            return result
        if message.get("message_type") == "metadata":
            metadata.update(
                {key: value for key, value in message.items() if key != "message_type"}
            )
        elif message.get("message_type") == "result":
            result = message


def _terminal_payload(
    status: str,
    *,
    solver_name: str,
    precedence_encoding: str,
    precedence_graph: str,
    encoding_variant: str,
    domain_mode: str,
    maxsat_backend: str,
    sat_backend: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "status": status,
        "sat_result": status_to_sat_result(status),
        "solver": solver_name,
        "precedence_mode": legacy_precedence_mode(
            precedence_encoding, precedence_graph
        ),
        "precedence_encoding": precedence_encoding,
        "precedence_graph": precedence_graph,
        "precedence_configuration": (
            f"{precedence_encoding}+{precedence_graph}"
        ),
        "encoding_variant": encoding_variant,
        "domain_mode": domain_mode,
        "runtime_seconds": round(runtime_seconds, 6),
        "objective": "internal_idle_slot_range_pstar",
        "maxsat_backend_preference": (
            maxsat_backend if solver_name == "maxsat" else ""
        ),
        "sat_backend_preference": sat_backend if solver_name != "maxsat" else "",
        "validation_errors": "",
        "error_type": "",
        "error_message": "",
    }


def run_with_timeout(
    solver_name: str,
    instance_path: Path,
    precedence_encoding: str,
    precedence_graph: str,
    encoding_variant: str,
    domain_mode: str,
    maxsat_backend: str,
    uwrmaxsat_bin: str | None,
    uwrmaxsat_sha256: str | None,
    sat_backend: str,
    timeout_seconds: float,
    verbose: bool,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    output: mp.Queue[Any] = context.Queue()
    process = context.Process(
        target=_worker,
        args=(
            solver_name,
            str(instance_path),
            precedence_encoding,
            precedence_graph,
            encoding_variant,
            domain_mode,
            maxsat_backend,
            uwrmaxsat_bin,
            uwrmaxsat_sha256,
            sat_backend,
            timeout_seconds,
            verbose,
            output,
        ),
    )
    started = time.perf_counter()
    deadline = started + timeout_seconds
    metadata: dict[str, Any] = {}
    result: dict[str, Any] | None = None
    peak_rss_bytes: int | None = None
    process.start()

    while process.is_alive() and time.perf_counter() < deadline:
        current_rss = _process_tree_rss_bytes(process.pid)
        if current_rss is not None:
            peak_rss_bytes = max(peak_rss_bytes or 0, current_rss)
        result = _drain_queue(output, metadata, result)
        process.join(
            min(
                MEMORY_SAMPLE_INTERVAL_SECONDS,
                max(0.0, deadline - time.perf_counter()),
            )
        )

    if process.is_alive():
        _terminate_process_tree(process)
        result = _terminal_payload(
            "TIMEOUT",
            solver_name=solver_name,
            precedence_encoding=precedence_encoding,
            precedence_graph=precedence_graph,
            encoding_variant=encoding_variant,
            domain_mode=domain_mode,
            maxsat_backend=maxsat_backend,
            sat_backend=sat_backend,
            runtime_seconds=time.perf_counter() - started,
        )
    else:
        process.join()
        grace_deadline = time.perf_counter() + QUEUE_GRACE_SECONDS
        while result is None and time.perf_counter() < grace_deadline:
            result = _drain_queue(output, metadata, result)
            if result is None:
                time.sleep(0.01)
        if result is None:
            result = _terminal_payload(
                "ERROR",
                solver_name=solver_name,
                precedence_encoding=precedence_encoding,
                precedence_graph=precedence_graph,
                encoding_variant=encoding_variant,
                domain_mode=domain_mode,
                maxsat_backend=maxsat_backend,
                sat_backend=sat_backend,
                runtime_seconds=time.perf_counter() - started,
            )
            result["error_type"] = "NoWorkerPayload"
            result["error_message"] = "Worker returned no result"

    for key, value in metadata.items():
        result.setdefault(key, value)
    result["peak_memory_mb"] = (
        None
        if peak_rss_bytes is None
        else round(peak_rss_bytes / (1024 * 1024), 3)
    )
    result["memory_metric"] = "peak_process_tree_rss_mb"
    result["sat_result"] = status_to_sat_result(result.get("status"))
    output.close()
    output.join_thread()
    return result


def _detailed_csv_path(csv_path: str) -> Path:
    path = Path(csv_path)
    suffix = path.suffix or ".csv"
    return path.with_name(f"{path.stem}_detailed{suffix}")


def write_detailed_csv(path: Path, results: list[dict[str, Any]]) -> None:
    preferred_fields = [
        "instance",
        "sat_result",
        "status",
        "runtime_seconds",
        "peak_memory_mb",
        "memory_metric",
        "solver",
        "solver_backend",
        "solver_binary",
        "solver_binary_sha256",
        "solver_version",
        "solver_command",
        "solver_message",
        "maxsat_backend_preference",
        "sat_backend_preference",
        "resolved_uwrmaxsat_bin",
        "precedence_mode",
        "precedence_encoding",
        "precedence_graph",
        "precedence_configuration",
        "encoding_variant",
        "domain_mode",
        "objective",
        "objective_value",
        "proven_optimum",
        "objective_participant_count",
        "objective_participants",
        "idle_range_pstar",
        "all_participant_idle_range",
        "total_internal_idle_slots",
        "initial_schedule_candidates",
        "full_schedule_candidates",
        "unary_eligible_schedule_candidates",
        "reduced_schedule_candidates",
        "active_schedule_candidates",
        "unary_removed_schedule_candidates",
        "preprocessing_removed_schedule_candidates",
        "removed_schedule_candidates",
        "n_vars",
        "n_hard_clauses",
        "n_soft_clauses",
        "n_total_clauses",
        "n_objective_lits",
        "precedence_direct_edges",
        "precedence_closure_edges",
        "precedence_max_distance",
        "precedence_relation_edges",
        "precedence_pairwise_clauses",
        "precedence_sparse_link_clauses",
        "precedence_unique_suffix_cuts",
        "participant_internal_idle_slots",
        "busy_participants_per_slot",
        "assignment",
        "schedule_by_slot",
        "validation_errors",
        "enabled_constraints",
        "error_type",
        "error_message",
    ]
    extra_fields = sorted(
        {
            key
            for result in results
            for key in result
            if key not in preferred_fields
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[*preferred_fields, *extra_fields],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)


def format_table_cell(result: dict[str, Any]) -> str:
    status = result.get("sat_result")
    if status == "TIMEOUT":
        return "TO"
    if status == "UNSAT":
        return "UNSAT"
    if status == "ERROR":
        return "ERR"
    runtime = result.get("runtime_seconds")
    objective = result.get("idle_range_pstar")
    return f"{float(runtime):.1f} {objective}"


def write_aggregate_csv(path: Path, results: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for result in results:
        key = (
            result["instance"],
            result["precedence_encoding"],
            result["precedence_graph"],
            result["solver"],
            result["domain_mode"],
        )
        row = grouped.setdefault(
            key,
            {
                "instance": result["instance"],
                "precedence_encoding": result["precedence_encoding"],
                "precedence_graph": result["precedence_graph"],
                "precedence_configuration": result["precedence_configuration"],
                "solver": result["solver"],
                "objective": "IdleRange(P*)",
                "domain_mode": result["domain_mode"],
            },
        )
        row[result["encoding_variant"]] = format_table_cell(result)

    rows = list(grouped.values())
    for row in rows:
        for variant in VARIANTS:
            row.setdefault(variant, "-")
    rows.sort(
        key=lambda row: (
            row["instance"],
            row["domain_mode"],
            row["precedence_encoding"],
            row["precedence_graph"],
            row["solver"],
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "instance",
        "precedence_encoding",
        "precedence_graph",
        "precedence_configuration",
        "solver",
        "objective",
        "domain_mode",
        *VARIANTS,
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        instances = collect_instances(args.instance, args.data_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    solvers = selected(args.solver, SOLVERS)
    try:
        require_solver_environment(
            solvers,
            maxsat_backend=args.maxsat_backend,
            uwrmaxsat_bin=args.uwrmaxsat_bin,
            uwrmaxsat_sha256=args.uwrmaxsat_sha256,
            sat_backend=args.sat_backend,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    precedence_cells = precedence_configurations(args)
    variants = selected(args.encoding_variant, VARIANTS)
    domain_modes = selected(args.domain_mode, DOMAIN_MODES, "both")
    total_runs = (
        len(instances)
        * len(solvers)
        * len(precedence_cells)
        * len(variants)
        * len(domain_modes)
    )
    results: list[dict[str, Any]] = []
    current_run = 0

    print(f"B2B conference benchmark: {total_runs} run(s), objective=IdleRange(P*)")
    print(f"Domain mode(s): {', '.join(domain_modes)}")
    if "maxsat" in solvers:
        print(f"MaxSAT backend: {args.maxsat_backend}")
    if any(solver in {"incremental", "multiple"} for solver in solvers):
        print(f"SAT backend: {args.sat_backend}")
    print(
        "Precedence P×G cell(s): "
        + ", ".join(
            f"{precedence_encoding}+{precedence_graph}"
            for precedence_encoding, precedence_graph in precedence_cells
        )
    )
    for instance_path in instances:
        for domain_mode in domain_modes:
            for precedence_encoding, precedence_graph in precedence_cells:
                for solver_name in solvers:
                    for variant in variants:
                        current_run += 1
                        print(
                            f"[{current_run}/{total_runs}] {instance_path.stem} | "
                            f"{domain_mode} | {solver_name} | "
                            f"P={precedence_encoding} | G={precedence_graph} | "
                            f"{variant}",
                            flush=True,
                        )
                        result = run_with_timeout(
                            solver_name,
                            instance_path,
                            precedence_encoding,
                            precedence_graph,
                            variant,
                            domain_mode,
                            args.maxsat_backend,
                            args.uwrmaxsat_bin,
                            args.uwrmaxsat_sha256,
                            args.sat_backend,
                            args.timeout,
                            args.verbose,
                        )
                        result = {"instance": instance_path.stem, **result}
                        results.append(result)
                        print(
                            f"    {result['sat_result']} | "
                            f"IdleRange(P*)={result.get('idle_range_pstar')} | "
                            f"time={result.get('runtime_seconds')}s",
                            flush=True,
                        )

    detailed_path = Path(args.long_csv) if args.long_csv else _detailed_csv_path(args.csv)
    aggregate_path = Path(args.csv)
    write_detailed_csv(detailed_path, results)
    write_aggregate_csv(aggregate_path, results)
    print(f"Detailed CSV: {detailed_path}")
    print(f"Aggregate CSV: {aggregate_path}")

    errors = sum(result["sat_result"] == "ERROR" for result in results)
    timeouts = sum(result["sat_result"] == "TIMEOUT" for result in results)
    print(f"Completed with errors={errors}, timeouts={timeouts}.")
    return 2 if errors else 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
