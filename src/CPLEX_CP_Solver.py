from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from B2B_Instance import B2BInstance
from Exact_Model_Common import (
    ExactModelArtifacts,
    ExactModelContext,
    exact_result,
    load_exact_context,
    make_exact_artifacts,
    with_model_counts,
)


@dataclass(frozen=True)
class CPMeetingTimeSpec:
    time_domains: tuple[tuple[int, ...], ...]
    all_different_groups: tuple[tuple[int, ...], ...]
    capacity_values: tuple[int, ...]
    precedence_relations: tuple[tuple[int, int, int], ...]
    objective_participants: tuple[int, ...]

    @property
    def n_global_constraints(self) -> int:
        return len(self.all_different_groups) + len(self.capacity_values)

    @property
    def n_linear_constraints(self) -> int:
        return len(self.precedence_relations) + sum(
            not domain for domain in self.time_domains
        )


CP_ENABLED_CONSTRAINTS = (
    "one reduced-domain integer time variable per meeting",
    "global all_diff per participant",
    "global count per slot for table capacity",
    "distance-labelled transitive precedence closure",
    "native min/max internal-idle expressions",
    "native max/min range objective over P*",
)


def build_cp_meeting_time_spec(
    context: ExactModelContext,
) -> tuple[CPMeetingTimeSpec, ExactModelArtifacts]:
    groups = tuple(
        tuple(meetings)
        for meetings in context.inst.meetings_by_business
        if len(meetings) >= 2
    )
    relations = tuple(
        (pred, post, distance)
        for post, distances in enumerate(context.graph.longest_distance)
        for pred, distance in sorted(distances.items())
    )
    spec = CPMeetingTimeSpec(
        time_domains=context.domains,
        all_different_groups=groups,
        capacity_values=tuple(range(context.inst.n_total_slots)),
        precedence_relations=relations,
        objective_participants=context.objective_participants,
    )
    artifacts = make_exact_artifacts(
        context,
        formalism="CP",
        model_family="CP-MeetingTime-SpanRange",
        formulation_name="CP-MeetingTime-SpanRange-Reduced-DistanceClosure",
        precedence_encoding="native_cp",
        objective_encoding="native_min_max_span_range",
        enabled_constraints=CP_ENABLED_CONSTRAINTS,
    )
    artifacts = with_model_counts(
        artifacts,
        n_vars=context.inst.n_meetings,
        n_primary_variables=context.inst.n_meetings,
        n_integer_variables=context.inst.n_meetings,
        n_linear_constraints=spec.n_linear_constraints,
        n_global_constraints=spec.n_global_constraints,
    )
    return spec, artifacts


class B2BCPLEXCPSolver:
    """CP Optimizer meeting-time model with native global constraints."""

    solver_backend = "CPLEX-CP"
    solver_binary = ""

    def __init__(
        self,
        instance_or_path: B2BInstance | str | Path,
        *,
        domain_mode: str = "reduced",
        solver_timeout: float | None = None,
        threads: int = 1,
        random_seed: int = 0,
    ) -> None:
        self.context = load_exact_context(
            instance_or_path,
            domain_mode=domain_mode,
        )
        self.cp_spec, self.artifacts = build_cp_meeting_time_spec(self.context)
        self.solver_timeout = solver_timeout
        self.threads = threads
        self.random_seed = random_seed
        self.solver_version = ""
        self.solver_command = (
            f"docplex.cp/CP Optimizer; Workers={threads}; "
            f"RandomSeed={random_seed}"
        )

    def solve(self, *, verbose: bool = False) -> dict[str, Any]:
        backend_build_started = time.perf_counter()
        try:
            import docplex
            from docplex.cp.expression import integer_var
            from docplex.cp.model import CpoModel
            from docplex.cp.modeler import (
                all_diff,
                count,
                max as cp_max,
                min as cp_min,
                minimize,
            )
        except ImportError as exc:
            raise RuntimeError(
                "CPLEX CP requires the optional 'docplex' package and a "
                "working CP Optimizer executable/license; no fallback is used"
            ) from exc

        model = CpoModel(name="B2B_CP_MeetingTime_SpanRange")
        horizon = self.context.inst.n_total_slots
        time_variables: list[Any] = []
        for meeting, domain in enumerate(self.cp_spec.time_domains):
            effective_domain = domain or tuple(range(horizon))
            variable = integer_var(
                domain=effective_domain,
                name=f"time_{meeting}",
            )
            time_variables.append(variable)
            if not domain:
                model.add(variable != variable)

        for group in self.cp_spec.all_different_groups:
            model.add(all_diff([time_variables[meeting] for meeting in group]))

        # `count()` is CP Optimizer's global occurrence-count expression.
        for slot in self.cp_spec.capacity_values:
            model.add(
                count(time_variables, slot) <= self.context.inst.n_tables
            )

        for pred, post, distance in self.cp_spec.precedence_relations:
            model.add(
                time_variables[pred] + distance <= time_variables[post]
            )

        idle_expressions: list[Any] = []
        for participant in self.context.objective_participants:
            participant_times = [
                time_variables[meeting]
                for meeting in self.context.inst.meetings_by_business[
                    participant
                ]
            ]
            idle_expressions.append(
                cp_max(participant_times)
                - cp_min(participant_times)
                + 1
                - len(participant_times)
            )
        delta = (
            cp_max(idle_expressions) - cp_min(idle_expressions)
            if len(idle_expressions) >= 2
            else 0
        )
        model.add(minimize(delta))

        solve_kwargs: dict[str, Any] = {
            "Workers": self.threads,
            "RandomSeed": self.random_seed,
            "LogVerbosity": "Normal" if verbose else "Quiet",
        }
        backend_model_construction_seconds = (
            time.perf_counter() - backend_build_started
        )
        if self.solver_timeout is not None:
            solve_kwargs["TimeLimit"] = max(
                0.001,
                self.solver_timeout
                - backend_model_construction_seconds,
            )
        result = model.solve(**solve_kwargs)

        solve_status = str(result.get_solve_status() or "")
        search_status = str(result.get_search_status() or "")
        stop_cause = str(result.get_stop_cause() or "")
        fail_status = (
            ""
            if stop_cause
            else str(result.get_fail_status() or "")
        )
        normalized = " ".join(
            (solve_status, search_status, stop_cause, fail_status)
        ).lower()
        solution = result.get_solution()
        has_solution = solution is not None
        if has_solution and "optimal" in normalized:
            status = "OPTIMAL"
        elif "infeasible" in normalized:
            status = "UNSAT"
        elif "limit" in f"{stop_cause} {fail_status}".lower():
            status = "TIMEOUT"
        elif has_solution:
            status = "FEASIBLE"
        else:
            status = "ERROR"

        assignment = (
            [
                int(round(result.get_value(variable)))
                for variable in time_variables
            ]
            if has_solution
            else None
        )
        objective_value = (
            result.get_objective_value() if has_solution else None
        )
        infos = result.get_solver_infos()
        solver_version = ""
        cp_branches = None
        cp_fails = None
        if infos is not None:
            getter = getattr(infos, "get", lambda _key, default=None: default)
            solver_version = str(getter("SolverVersion", "") or "")
            cp_branches = getter("NumberOfBranches", None)
            cp_fails = getter("NumberOfFails", None)
        self.solver_version = (
            solver_version
            or f"DOcplex {getattr(docplex, '__version__', '')}".strip()
        )
        return exact_result(
            self.context,
            self.artifacts,
            status=status,
            solver_backend=self.solver_backend,
            solver_version=self.solver_version,
            assignment=assignment,
            objective_value=objective_value,
            best_bound=result.get_objective_bound(),
            optimality_gap=(
                result.get_objective_gap() if has_solution else None
            ),
            proven_optimum=status == "OPTIMAL",
            solver_message=" | ".join(
                value
                for value in (
                    solve_status,
                    search_status,
                    stop_cause,
                    fail_status,
                )
                if value
            ),
            metrics={
                "backend_model_construction_seconds": round(
                    backend_model_construction_seconds,
                    6,
                ),
                "cp_branches": cp_branches,
                "cp_fails": cp_fails,
                "n_optimizer_calls": 1,
            },
        )
