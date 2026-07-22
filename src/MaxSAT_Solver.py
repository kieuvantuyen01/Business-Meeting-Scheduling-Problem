from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

from B2B_Instance import B2BInstance, B2BSATModel, B2BSolutionStats, read_instance


MaxSATBackend = Literal["auto", "uwrmaxsat", "rc2"]
VALID_MAXSAT_BACKENDS = {"auto", "uwrmaxsat", "rc2"}
DEFAULT_UWRMAXSAT_TIMEOUT_SECONDS = 3600.0


def _ensure_instance(instance_or_path: B2BInstance | str | Path) -> B2BInstance:
    return (
        instance_or_path
        if isinstance(instance_or_path, B2BInstance)
        else read_instance(instance_or_path)
    )


def _positive_float_from_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number; got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}")
    return value


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and (os.name == "nt" or os.access(path, os.X_OK))


def _resolve_uwrmaxsat_binary(explicit_path: str | Path | None = None) -> Path | None:
    """Find UWrMaxSAT using the same local layout as ORG_new, then PATH."""

    script_dir = Path(__file__).resolve().parent
    configured = explicit_path or os.environ.get("UWRMAXSAT_BIN")
    candidates: list[Path] = []

    if configured:
        configured_text = os.path.expandvars(os.path.expanduser(str(configured)))
        configured_path = Path(configured_text)
        candidates.append(configured_path)
        if configured_path.parent == Path("."):
            located = shutil.which(configured_text)
            if located:
                candidates.append(Path(located))

    executable_name = "uwrmaxsat.exe" if os.name == "nt" else "uwrmaxsat"
    candidates.extend(
        [
            script_dir / "uwrmaxsat" / "build" / "release" / "bin" / executable_name,
            script_dir / "UWrMaxSat" / "build" / "release" / "bin" / executable_name,
            script_dir / "UWrMaxSAT" / "build" / "release" / "bin" / executable_name,
        ]
    )

    located = shutil.which("uwrmaxsat")
    if located:
        candidates.append(Path(located))

    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.resolve(strict=False))
        if normalized in seen:
            continue
        seen.add(normalized)
        if _is_executable_file(candidate):
            return candidate.resolve()
    return None


def _parse_uwrmaxsat_output(output: str) -> tuple[str | None, int | None, list[int]]:
    """Parse the standard ``s``, ``o`` and ``v`` lines emitted by UWrMaxSAT."""

    status: str | None = None
    cost: int | None = None
    model: list[int] = []

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("s "):
            status = line[2:].strip()
        elif line.startswith("o "):
            try:
                cost = int(line[2:].strip())
            except ValueError:
                continue
        elif line.startswith("v "):
            for token in line[2:].split():
                try:
                    literal = int(token)
                except ValueError:
                    continue
                if literal != 0:
                    model.append(literal)

    return status, cost, model


class B2BMaxSATSolver:
    """MaxSAT optimization of the internal-idle-slot range over P*.

    The true count of the shared model's objective literals is exactly
    ``max_{p in P*} B(p) - min_{p in P*} B(p)``, where
    ``P* = {p : |M_p| >= 2}``. By default the solver first tries UWrMaxSAT and
    falls back to PySAT RC2 only when no executable is available. No hard
    objective cap or secondary lexicographic objective is included.
    """

    def __init__(
        self,
        instance_or_path: B2BInstance | str | Path,
        precedence_mode: str | None = None,
        encoding_variant: str = "imp12+",
        domain_mode: str = "reduced",
        *,
        precedence_encoding: str | None = None,
        precedence_graph: str | None = None,
        backend: MaxSATBackend | str | None = None,
        uwrmaxsat_bin: str | Path | None = None,
        uwrmaxsat_timeout: float | None = None,
    ) -> None:
        selected_backend = (
            backend or os.environ.get("B2B_MAXSAT_BACKEND", "auto")
        ).lower()
        if selected_backend not in VALID_MAXSAT_BACKENDS:
            raise ValueError(
                f"Unknown MaxSAT backend={selected_backend!r}; expected one of "
                f"{sorted(VALID_MAXSAT_BACKENDS)}"
            )

        timeout = (
            _positive_float_from_env(
                "UWRMAXSAT_TIMEOUT",
                DEFAULT_UWRMAXSAT_TIMEOUT_SECONDS,
            )
            if uwrmaxsat_timeout is None
            else float(uwrmaxsat_timeout)
        )
        if timeout <= 0:
            raise ValueError("uwrmaxsat_timeout must be positive")

        if (
            precedence_mode is None
            and precedence_encoding is None
            and precedence_graph is None
        ):
            precedence_mode = "traditional"

        self.inst = _ensure_instance(instance_or_path)
        self.model = B2BSATModel(
            inst=self.inst,
            precedence_mode=precedence_mode,
            precedence_encoding=precedence_encoding,
            precedence_graph=precedence_graph,
            encoding_variant=encoding_variant,
            domain_mode=domain_mode,
        )
        self.artifacts = self.model.build_base_cnf()
        self.backend: MaxSATBackend = selected_backend  # type: ignore[assignment]
        self.uwrmaxsat_bin = uwrmaxsat_bin
        self.uwrmaxsat_timeout = timeout
        self.resolved_uwrmaxsat_bin = _resolve_uwrmaxsat_binary(uwrmaxsat_bin)

    def _build_wcnf(self) -> WCNF:
        return self.model.build_wcnf()

    def _pack_result(
        self,
        status: str,
        assignment: list[int] | None,
        stats: B2BSolutionStats | None,
        checks: list[str] | None = None,
        *,
        solver_cost: int | None = None,
        solver_backend: str,
        solver_message: str = "",
    ) -> dict[str, Any]:
        return {
            "status": status,
            "solver": "MaxSAT",
            "solver_backend": solver_backend,
            "solver_binary": (
                str(self.resolved_uwrmaxsat_bin)
                if solver_backend == "UWrMaxSAT"
                and self.resolved_uwrmaxsat_bin is not None
                else ""
            ),
            "solver_message": solver_message,
            "maxsat_backend_preference": self.backend,
            "precedence_mode": self.artifacts.precedence_mode,
            "precedence_encoding": self.artifacts.precedence_encoding,
            "precedence_graph": self.artifacts.precedence_graph,
            "precedence_configuration": (
                self.artifacts.precedence_configuration
            ),
            "encoding_variant": self.artifacts.encoding_variant,
            "domain_mode": self.artifacts.domain_mode,
            "objective": self.artifacts.objective_name,
            "objective_participant_count": len(
                self.artifacts.objective_participants
            ),
            "objective_participants": tuple(
                participant + 1
                for participant in self.artifacts.objective_participants
            ),
            "objective_value": (
                stats.objective_gap if stats is not None else solver_cost
            ),
            "proven_optimum": solver_cost,
            "solver_cost": solver_cost,
            "assignment": assignment,
            "stats": stats,
            "validation_errors": checks or [],
            "n_vars": self.artifacts.n_vars,
            "n_clauses": self.artifacts.n_clauses,
            "n_hard_clauses": self.artifacts.n_clauses,
            "n_soft": len(self.artifacts.objective_lits),
            "n_soft_clauses": len(self.artifacts.objective_lits),
            "n_objective_lits": len(self.artifacts.objective_lits),
            "full_schedule_candidates": (
                self.artifacts.full_schedule_candidates
            ),
            "unary_eligible_schedule_candidates": (
                self.artifacts.unary_eligible_schedule_candidates
            ),
            "initial_schedule_candidates": (
                self.artifacts.initial_schedule_candidates
            ),
            "reduced_schedule_candidates": (
                self.artifacts.reduced_schedule_candidates
            ),
            "active_schedule_candidates": (
                self.artifacts.active_schedule_candidates
            ),
            "unary_removed_schedule_candidates": (
                self.artifacts.unary_removed_schedule_candidates
            ),
            "preprocessing_removed_schedule_candidates": (
                self.artifacts.preprocessing_removed_schedule_candidates
            ),
            "removed_schedule_candidates": (
                self.artifacts.removed_schedule_candidates
            ),
            "precedence_direct_edges": self.artifacts.precedence_direct_edges,
            "precedence_closure_edges": (
                self.artifacts.precedence_transitive_edges
            ),
            "precedence_max_distance": self.artifacts.precedence_max_distance,
            "precedence_relation_edges": (
                self.artifacts.precedence_relation_edges
            ),
            "precedence_pairwise_clauses": (
                self.artifacts.precedence_pairwise_clauses
            ),
            "precedence_sparse_link_clauses": (
                self.artifacts.precedence_sparse_link_clauses
            ),
            "precedence_unique_suffix_cuts": (
                self.artifacts.precedence_unique_suffix_cuts
            ),
            "enabled_constraints": self.artifacts.enabled_constraints,
        }

    def _validate_model(
        self,
        sat_model: list[int],
        solver_cost: int,
    ) -> tuple[list[int], B2BSolutionStats, list[str]]:
        assignment = self.model.decode_assignment(sat_model)
        stats = self.model.compute_stats(assignment)
        checks = self.model.validate_assignment(assignment)
        checks.extend(
            self.model.objective_consistency_errors(
                sat_model,
                stats,
                solver_cost=solver_cost,
            )
        )
        return assignment, stats, checks

    def _solve_with_rc2(
        self,
        verbose: bool,
        *,
        fallback_reason: str = "",
    ) -> dict[str, Any]:
        with RC2(self._build_wcnf()) as solver:
            sat_model = solver.compute()
            if sat_model is None:
                return self._pack_result(
                    "UNSAT",
                    None,
                    None,
                    solver_backend="RC2",
                    solver_message=fallback_reason,
                )
            solver_cost = int(solver.cost)

        assignment, stats, checks = self._validate_model(sat_model, solver_cost)
        if verbose:
            prefix = f"{fallback_reason}; " if fallback_reason else ""
            print(
                f"[MaxSAT/RC2] {prefix}optimum IdleRange(P*)="
                f"{stats.objective_gap} (cost={solver_cost})"
            )
        return self._pack_result(
            "OPTIMAL" if not checks else "ERROR",
            assignment,
            stats,
            checks,
            solver_cost=solver_cost,
            solver_backend="RC2",
            solver_message=fallback_reason,
        )

    def _solve_with_uwrmaxsat(
        self,
        binary: Path,
        verbose: bool,
    ) -> dict[str, Any]:
        wcnf = self._build_wcnf()
        safe_stem = Path(self.inst.instance_name).stem or "instance"

        with tempfile.TemporaryDirectory(prefix="b2b_uwrmaxsat_") as temp_dir:
            wcnf_path = Path(temp_dir) / f"{safe_stem}.wcnf"
            wcnf.to_file(str(wcnf_path))
            command = [str(binary), "-m", str(wcnf_path)]

            if verbose:
                print(
                    "[MaxSAT/UWrMaxSAT] running "
                    f"{binary} with timeout={self.uwrmaxsat_timeout:g}s"
                )

            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.uwrmaxsat_timeout,
                    check=False,
                    start_new_session=(os.name != "nt"),
                )
            except subprocess.TimeoutExpired:
                message = (
                    "UWrMaxSAT timed out after "
                    f"{self.uwrmaxsat_timeout:g} seconds"
                )
                if verbose:
                    print(f"[MaxSAT/UWrMaxSAT] {message}")
                return self._pack_result(
                    "TIMEOUT",
                    None,
                    None,
                    solver_backend="UWrMaxSAT",
                    solver_message=message,
                )
            except OSError as exc:
                message = f"cannot execute UWrMaxSAT: {exc}"
                return self._pack_result(
                    "ERROR",
                    None,
                    None,
                    solver_backend="UWrMaxSAT",
                    solver_message=message,
                )

        combined_output = "\n".join(
            part for part in (completed.stdout, completed.stderr) if part
        )
        raw_status, parsed_cost, sat_model = _parse_uwrmaxsat_output(combined_output)
        normalized_status = (raw_status or "").strip().upper()

        if normalized_status in {"UNSAT", "UNSATISFIABLE"}:
            return self._pack_result(
                "UNSAT",
                None,
                None,
                solver_backend="UWrMaxSAT",
                solver_message=(
                    f"process exit code={completed.returncode}"
                    if completed.returncode != 0
                    else ""
                ),
            )

        optimum_statuses = {"OPTIMUM FOUND", "OPTIMAL", "OPTIMUM"}
        if normalized_status not in optimum_statuses:
            details = raw_status or "missing status line"
            message = (
                f"UWrMaxSAT did not report an optimum ({details}); "
                f"exit code={completed.returncode}"
            )
            stderr_tail = completed.stderr.strip().splitlines()
            if stderr_tail:
                message += f"; stderr: {stderr_tail[-1]}"
            return self._pack_result(
                "ERROR",
                None,
                None,
                solver_backend="UWrMaxSAT",
                solver_message=message,
            )

        if not sat_model and self.artifacts.n_vars > 0:
            return self._pack_result(
                "ERROR",
                None,
                None,
                solver_backend="UWrMaxSAT",
                solver_message="UWrMaxSAT reported an optimum but returned no model",
            )

        solver_cost = (
            parsed_cost
            if parsed_cost is not None
            else self.model.encoded_objective_value(sat_model)
        )
        assignment, stats, validation_errors = self._validate_model(
            sat_model,
            solver_cost,
        )

        solver_messages: list[str] = [
            f"process exit code={completed.returncode}"
        ]

        # Exit code chỉ là cảnh báo quy trình, không phải lỗi của nghiệm.
        if completed.returncode not in {0, 30}:
            solver_messages.append(
                "UWrMaxSAT reported OPTIMUM FOUND with an unexpected process "
                f"exit code {completed.returncode}"
            )

        if verbose:
            print(
                "[MaxSAT/UWrMaxSAT] optimum IdleRange(P*)="
                f"{stats.objective_gap} (cost={solver_cost})"
            )

            print(
                "[MaxSAT/UWrMaxSAT] "
                f"process exit code={completed.returncode}"
            )

            if validation_errors:
                print("[MaxSAT/UWrMaxSAT] validation errors:")
                for error in validation_errors:
                    print(f"  - {error}")

        return self._pack_result(
            "OPTIMAL" if not validation_errors else "ERROR",
            assignment,
            stats,
            validation_errors,
            solver_cost=solver_cost,
            solver_backend="UWrMaxSAT",
            solver_message="; ".join(solver_messages),
        )

    def solve(self, verbose: bool = False) -> dict[str, Any]:
        if self.backend == "rc2":
            return self._solve_with_rc2(verbose)

        binary = self.resolved_uwrmaxsat_bin
        if binary is not None:
            return self._solve_with_uwrmaxsat(binary, verbose)

        message = (
            "UWrMaxSAT executable not found; set UWRMAXSAT_BIN or place it at "
            "uwrmaxsat/build/release/bin/uwrmaxsat"
        )
        if self.backend == "uwrmaxsat":
            if verbose:
                print(f"[MaxSAT/UWrMaxSAT] {message}")
            return self._pack_result(
                "ERROR",
                None,
                None,
                solver_backend="UWrMaxSAT",
                solver_message=message,
            )

        if verbose:
            print(f"[MaxSAT] {message}; falling back to RC2")
        return self._solve_with_rc2(verbose, fallback_reason=message)


def solve_b2b(
    instance_or_path: B2BInstance | str | Path,
    precedence_mode: str | None = None,
    encoding_variant: str = "imp12+",
    verbose: bool = False,
    domain_mode: str = "reduced",
    *,
    precedence_encoding: str | None = None,
    precedence_graph: str | None = None,
    backend: MaxSATBackend | str | None = None,
    uwrmaxsat_bin: str | Path | None = None,
    uwrmaxsat_timeout: float | None = None,
) -> dict[str, Any]:
    return B2BMaxSATSolver(
        instance_or_path=instance_or_path,
        precedence_mode=precedence_mode,
        precedence_encoding=precedence_encoding,
        precedence_graph=precedence_graph,
        encoding_variant=encoding_variant,
        domain_mode=domain_mode,
        backend=backend,
        uwrmaxsat_bin=uwrmaxsat_bin,
        uwrmaxsat_timeout=uwrmaxsat_timeout,
    ).solve(verbose=verbose)


def solve_b2b_traditional(
    instance_or_path: B2BInstance | str | Path,
    encoding_variant: str = "imp12+",
    verbose: bool = False,
    domain_mode: str = "reduced",
    **kwargs: Any,
) -> dict[str, Any]:
    return solve_b2b(
        instance_or_path=instance_or_path,
        precedence_mode="traditional",
        encoding_variant=encoding_variant,
        verbose=verbose,
        domain_mode=domain_mode,
        **kwargs,
    )


def solve_b2b_staircase(
    instance_or_path: B2BInstance | str | Path,
    encoding_variant: str = "imp12+",
    verbose: bool = False,
    domain_mode: str = "reduced",
    **kwargs: Any,
) -> dict[str, Any]:
    return solve_b2b(
        instance_or_path=instance_or_path,
        precedence_mode="staircase",
        encoding_variant=encoding_variant,
        verbose=verbose,
        domain_mode=domain_mode,
        **kwargs,
    )
