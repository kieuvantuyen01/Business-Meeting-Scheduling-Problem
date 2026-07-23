from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from B2B_Instance import (
    B2BInstance,
    B2BSolutionStats,
    PrecedenceGraphInfo,
    build_precedence_graph,
    compute_solution_stats,
    objective_participants,
    original_eligible_slots,
    read_instance,
    reduce_domains_with_precedence,
    validate_schedule_assignment,
)


OBJECTIVE_NAME = "internal_idle_slot_range_pstar"
EXACT_DOMAIN_MODE = "reduced"
EXACT_PRECEDENCE_GRAPH = "distance_closure"


@dataclass(frozen=True)
class ExactModelContext:
    """Solver-independent instance data for the exact MIP/CP baselines."""

    inst: B2BInstance
    graph: PrecedenceGraphInfo
    domains: tuple[tuple[int, ...], ...]
    objective_participants: tuple[int, ...]
    full_schedule_candidates: int
    unary_eligible_schedule_candidates: int
    initial_schedule_candidates: int
    reduced_schedule_candidates: int

    @property
    def active_schedule_candidates(self) -> int:
        return self.reduced_schedule_candidates

    @property
    def unary_removed_schedule_candidates(self) -> int:
        return (
            self.full_schedule_candidates
            - self.unary_eligible_schedule_candidates
        )

    @property
    def preprocessing_removed_schedule_candidates(self) -> int:
        return (
            self.initial_schedule_candidates
            - self.reduced_schedule_candidates
        )


@dataclass(frozen=True)
class ExactModelArtifacts:
    """Common reporting schema for a MIP or CP Optimizer formulation."""

    formalism: str
    model_family: str
    formulation_name: str
    objective_name: str
    objective_participants: tuple[int, ...]
    objective_encoding: str
    domain_mode: str
    precedence_mode: str
    precedence_encoding: str
    precedence_graph: str
    precedence_configuration: str
    enabled_constraints: tuple[str, ...]
    full_schedule_candidates: int
    unary_eligible_schedule_candidates: int
    initial_schedule_candidates: int
    reduced_schedule_candidates: int
    active_schedule_candidates: int
    unary_removed_schedule_candidates: int
    preprocessing_removed_schedule_candidates: int
    removed_schedule_candidates: int
    precedence_direct_edges: int
    precedence_transitive_edges: int
    precedence_cycle_nodes: tuple[int, ...]
    precedence_max_distance: int
    precedence_relation_edges: int
    n_vars: int = 0
    n_primary_variables: int = 0
    n_auxiliary_variables: int = 0
    n_binary_variables: int = 0
    n_integer_variables: int = 0
    n_continuous_variables: int = 0
    n_linear_constraints: int = 0
    n_global_constraints: int = 0
    n_nonzeros: int = 0


def load_exact_context(
    instance_or_path: B2BInstance | str | Path,
    *,
    domain_mode: str = EXACT_DOMAIN_MODE,
) -> ExactModelContext:
    """Build the canonical exact-baseline context.

    Exact baselines intentionally use one fixed configuration: zero-based
    reduced domains and the distance-labelled transitive precedence closure.
    This prevents SAT-specific P/G/I factors from being multiplied into the
    Gurobi and CPLEX comparison cells.
    """

    if domain_mode != EXACT_DOMAIN_MODE:
        raise ValueError(
            "Exact MIP/CP baselines require domain_mode='reduced'; "
            "SAT/MaxSAT-only domain ablations must not be applied to them"
        )
    inst = (
        instance_or_path
        if isinstance(instance_or_path, B2BInstance)
        else read_instance(instance_or_path)
    )
    graph = build_precedence_graph(inst.precedences)
    unary_count = sum(
        len(original_eligible_slots(inst, meeting))
        for meeting in range(inst.n_meetings)
    )
    domains, initial_count, reduced_count = reduce_domains_with_precedence(
        inst,
        graph,
    )
    return ExactModelContext(
        inst=inst,
        graph=graph,
        domains=tuple(tuple(domain) for domain in domains),
        objective_participants=objective_participants(inst),
        full_schedule_candidates=inst.n_meetings * inst.n_total_slots,
        unary_eligible_schedule_candidates=unary_count,
        initial_schedule_candidates=initial_count,
        reduced_schedule_candidates=reduced_count,
    )


def make_exact_artifacts(
    context: ExactModelContext,
    *,
    formalism: str,
    model_family: str,
    formulation_name: str,
    precedence_encoding: str,
    objective_encoding: str,
    enabled_constraints: tuple[str, ...],
) -> ExactModelArtifacts:
    graph = context.graph
    return ExactModelArtifacts(
        formalism=formalism,
        model_family=model_family,
        formulation_name=formulation_name,
        objective_name=OBJECTIVE_NAME,
        objective_participants=context.objective_participants,
        objective_encoding=objective_encoding,
        domain_mode=EXACT_DOMAIN_MODE,
        precedence_mode="exact_native",
        precedence_encoding=precedence_encoding,
        precedence_graph=EXACT_PRECEDENCE_GRAPH,
        precedence_configuration=(
            f"{precedence_encoding}+{EXACT_PRECEDENCE_GRAPH}"
        ),
        enabled_constraints=enabled_constraints,
        full_schedule_candidates=context.full_schedule_candidates,
        unary_eligible_schedule_candidates=(
            context.unary_eligible_schedule_candidates
        ),
        initial_schedule_candidates=context.initial_schedule_candidates,
        reduced_schedule_candidates=context.reduced_schedule_candidates,
        active_schedule_candidates=context.active_schedule_candidates,
        unary_removed_schedule_candidates=(
            context.unary_removed_schedule_candidates
        ),
        preprocessing_removed_schedule_candidates=(
            context.preprocessing_removed_schedule_candidates
        ),
        removed_schedule_candidates=(
            context.preprocessing_removed_schedule_candidates
        ),
        precedence_direct_edges=graph.direct_edge_count,
        precedence_transitive_edges=graph.transitive_edge_count,
        precedence_cycle_nodes=graph.cycle_nodes,
        precedence_max_distance=graph.max_chain_distance,
        precedence_relation_edges=graph.transitive_edge_count,
    )


def with_model_counts(
    artifacts: ExactModelArtifacts,
    *,
    n_vars: int,
    n_primary_variables: int,
    n_binary_variables: int = 0,
    n_integer_variables: int = 0,
    n_continuous_variables: int = 0,
    n_linear_constraints: int = 0,
    n_global_constraints: int = 0,
    n_nonzeros: int = 0,
) -> ExactModelArtifacts:
    return replace(
        artifacts,
        n_vars=n_vars,
        n_primary_variables=n_primary_variables,
        n_auxiliary_variables=max(0, n_vars - n_primary_variables),
        n_binary_variables=n_binary_variables,
        n_integer_variables=n_integer_variables,
        n_continuous_variables=n_continuous_variables,
        n_linear_constraints=n_linear_constraints,
        n_global_constraints=n_global_constraints,
        n_nonzeros=n_nonzeros,
    )


def normalize_integer(value: float | int | None) -> int | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return int(round(numeric))


def normalize_number(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    rounded = round(numeric)
    return int(rounded) if abs(numeric - rounded) <= 1e-9 else numeric


def exact_result(
    context: ExactModelContext,
    artifacts: ExactModelArtifacts,
    *,
    status: str,
    solver_backend: str,
    solver_version: str,
    assignment: list[int] | None = None,
    objective_value: float | int | None = None,
    best_bound: float | int | None = None,
    optimality_gap: float | None = None,
    proven_optimum: bool = False,
    solver_message: str = "",
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stats: B2BSolutionStats | None = None
    validation_errors: list[str] = []
    normalized_objective = normalize_integer(objective_value)
    if assignment is not None:
        validation_errors = validate_schedule_assignment(
            context.inst,
            assignment,
            graph=context.graph,
        )
        stats = compute_solution_stats(
            context.inst,
            assignment,
            participants=context.objective_participants,
        )
        if (
            normalized_objective is not None
            and normalized_objective != stats.objective_gap
        ):
            validation_errors.append(
                "objective mismatch: "
                f"solver={normalized_objective}, schedule={stats.objective_gap}"
            )
        if normalized_objective is None:
            normalized_objective = stats.objective_gap

    if validation_errors:
        status = "ERROR"
        proven_optimum = False

    result: dict[str, Any] = {
        "status": status,
        "solver": solver_backend,
        "solver_backend": solver_backend,
        "solver_version": solver_version,
        "solver_message": solver_message,
        "objective": OBJECTIVE_NAME,
        "objective_value": normalized_objective,
        "best_bound": (
            normalize_number(best_bound)
            if best_bound is not None
            else None
        ),
        "optimality_gap": optimality_gap,
        "proven_optimum": (
            normalized_objective if proven_optimum else None
        ),
        "objective_participant_count": len(
            context.objective_participants
        ),
        "objective_participants": tuple(
            participant + 1
            for participant in context.objective_participants
        ),
        "assignment": assignment,
        "stats": stats,
        "validation_errors": validation_errors,
        "formalism": artifacts.formalism,
        "model_family": artifacts.model_family,
        "formulation_name": artifacts.formulation_name,
    }
    if metrics:
        result.update(metrics)
    return result
