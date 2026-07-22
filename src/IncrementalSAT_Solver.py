from __future__ import annotations

from pathlib import Path
from typing import Any

from pysat.card import ITotalizer

from B2B_Instance import B2BInstance, B2BSATModel, B2BSolutionStats, read_instance
from SAT_Backend import (
    create_sat_solver,
    normalize_sat_backend,
    sat_backend_label,
    sat_backend_version,
)


def _ensure_instance(instance_or_path: B2BInstance | str | Path) -> B2BInstance:
    return (
        instance_or_path
        if isinstance(instance_or_path, B2BInstance)
        else read_instance(instance_or_path)
    )


def _new_solver(clauses: list[list[int]], preferred: str = "cadical"):
    """Create exactly the requested SAT backend without fallback."""

    return create_sat_solver(clauses, preferred)


class B2BIncrementalSATSolver:
    """Incremental SAT optimization of the idle-slot range over P*.

    One SAT solver is retained while an incremental totalizer imposes upper
    bounds on ``max_{p in P*} B(p) - min_{p in P*} B(p)`` through assumptions.
    No hard objective cap or secondary Lexicographic objective is generated.
    """

    def __init__(
        self,
        instance_or_path: B2BInstance | str | Path,
        precedence_mode: str | None = None,
        encoding_variant: str = "imp12+",
        solver_name: str = "cadical",
        domain_mode: str = "reduced",
        *,
        precedence_encoding: str | None = None,
        precedence_graph: str | None = None,
    ) -> None:
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
        self.solver_name = normalize_sat_backend(solver_name)
        self.solver_backend = sat_backend_label(self.solver_name)
        self.solver_version = sat_backend_version(self.solver_name)

    def _pack_result(
        self,
        status: str,
        assignment: list[int] | None,
        stats: B2BSolutionStats | None,
        checks: list[str] | None = None,
        *,
        proven_optimum: int | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "solver": "IncrementalSAT",
            "solver_backend": self.solver_backend,
            "solver_version": self.solver_version,
            "sat_backend_preference": self.solver_name,
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
                stats.objective_gap if stats is not None else proven_optimum
            ),
            "proven_optimum": proven_optimum,
            "assignment": assignment,
            "stats": stats,
            "validation_errors": checks or [],
            "n_vars": self.artifacts.n_vars,
            "n_clauses": self.artifacts.n_clauses,
            "n_hard_clauses": self.artifacts.n_clauses,
            "n_soft": 0,
            "n_soft_clauses": 0,
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

    def _evaluate_sat_model(
        self,
        sat_model: list[int],
        *,
        imposed_bound: int | None = None,
    ) -> tuple[list[int], B2BSolutionStats, list[str]]:
        assignment = self.model.decode_assignment(sat_model)
        stats = self.model.compute_stats(assignment)
        checks = self.model.validate_assignment(assignment)
        checks.extend(
            self.model.objective_consistency_errors(
                sat_model,
                stats,
                imposed_bound=imposed_bound,
            )
        )
        return assignment, stats, checks

    def solve(self, verbose: bool = False) -> dict[str, Any]:
        with _new_solver(self.artifacts.cnf.clauses, self.solver_name) as solver:
            if not solver.solve():
                return self._pack_result("UNSAT", None, None)

            initial_model = solver.get_model()
            best_assignment, best_stats, checks = self._evaluate_sat_model(
                initial_model
            )
            if checks:
                return self._pack_result("ERROR", best_assignment, best_stats, checks)

            best_objective = best_stats.objective_gap
            if verbose:
                print(f"[IncrementalSAT] initial IdleRange(P*)={best_objective}")
            if best_objective == 0:
                return self._pack_result(
                    "OPTIMAL",
                    best_assignment,
                    best_stats,
                    proven_optimum=0,
                )

            objective_lits = self.artifacts.objective_lits
            if not objective_lits:
                return self._pack_result(
                    "ERROR",
                    best_assignment,
                    best_stats,
                    ["nonzero schedule gap but no objective literals were generated"],
                )

            with ITotalizer(
                lits=objective_lits,
                ubound=best_objective,
                top_id=self.artifacts.n_vars,
            ) as totalizer:
                solver.append_formula(totalizer.cnf.clauses)
                low, high = 0, best_objective - 1

                while low <= high:
                    bound = (low + high) // 2
                    # rhs[bound] means at least bound+1 objective literals are true.
                    satisfiable = solver.solve(assumptions=[-totalizer.rhs[bound]])
                    if verbose:
                        print(
                            f"[IncrementalSAT] IdleRange(P*) <= {bound}: "
                            f"{'SAT' if satisfiable else 'UNSAT'}"
                        )
                    if satisfiable:
                        candidate_model = solver.get_model()
                        (
                            candidate_assignment,
                            candidate_stats,
                            candidate_checks,
                        ) = self._evaluate_sat_model(
                            candidate_model,
                            imposed_bound=bound,
                        )
                        if candidate_checks:
                            return self._pack_result(
                                "ERROR",
                                candidate_assignment,
                                candidate_stats,
                                candidate_checks,
                            )
                        best_assignment = candidate_assignment
                        best_stats = candidate_stats
                        high = bound - 1
                    else:
                        low = bound + 1

            final_checks = self.model.validate_assignment(best_assignment)
            if best_stats.objective_gap != low:
                final_checks.append(
                    "optimization mismatch: "
                    f"proven optimum={low}, schedule gap={best_stats.objective_gap}"
                )
            return self._pack_result(
                "OPTIMAL" if not final_checks else "ERROR",
                best_assignment,
                best_stats,
                final_checks,
                proven_optimum=low,
            )


def solve_b2b(
    instance_or_path: B2BInstance | str | Path,
    precedence_mode: str | None = None,
    encoding_variant: str = "imp12+",
    verbose: bool = False,
    solver_name: str = "cadical",
    domain_mode: str = "reduced",
    *,
    precedence_encoding: str | None = None,
    precedence_graph: str | None = None,
) -> dict[str, Any]:
    return B2BIncrementalSATSolver(
        instance_or_path=instance_or_path,
        precedence_mode=precedence_mode,
        precedence_encoding=precedence_encoding,
        precedence_graph=precedence_graph,
        encoding_variant=encoding_variant,
        solver_name=solver_name,
        domain_mode=domain_mode,
    ).solve(verbose=verbose)


def solve_b2b_traditional(
    instance_or_path: B2BInstance | str | Path,
    encoding_variant: str = "imp12+",
    verbose: bool = False,
    domain_mode: str = "reduced",
) -> dict[str, Any]:
    return solve_b2b(
        instance_or_path=instance_or_path,
        precedence_mode="traditional",
        encoding_variant=encoding_variant,
        verbose=verbose,
        domain_mode=domain_mode,
    )


def solve_b2b_staircase(
    instance_or_path: B2BInstance | str | Path,
    encoding_variant: str = "imp12+",
    verbose: bool = False,
    domain_mode: str = "reduced",
) -> dict[str, Any]:
    return solve_b2b(
        instance_or_path=instance_or_path,
        precedence_mode="staircase",
        encoding_variant=encoding_variant,
        verbose=verbose,
        domain_mode=domain_mode,
    )
