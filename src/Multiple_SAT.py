from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pysat.card import CardEnc, EncType

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


class B2BMultipleSATSolver:
    """Repeated-SAT optimization of the internal-idle-slot range over P*.

    Every candidate bound is solved in a fresh SAT solver. The objective is
    only ``max_{p in P*} B(p) - min_{p in P*} B(p)``; no hard objective cap or
    secondary Lexicographic objective is generated.
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
        self.n_optimizer_calls = 0
        self.n_bound_encodings = 0
        self.optimizer_added_variables_peak = 0
        self.optimizer_added_clauses_peak = 0
        self.optimizer_added_literals_peak = 0
        self.optimizer_added_clauses_cumulative = 0

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
            "solver": "MultipleSAT",
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
            "n_optimizer_calls": self.n_optimizer_calls,
            "n_bound_encodings": self.n_bound_encodings,
            "optimizer_added_variables_peak": (
                self.optimizer_added_variables_peak
            ),
            "optimizer_added_clauses_peak": self.optimizer_added_clauses_peak,
            "optimizer_added_literals_peak": self.optimizer_added_literals_peak,
            "optimizer_added_clauses_cumulative": (
                self.optimizer_added_clauses_cumulative
            ),
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

    def _bound_clauses(self, bound: int) -> list[list[int]]:
        lits = self.artifacts.objective_lits
        if bound < 0:
            clauses = [[]]
            top_id = self.artifacts.n_vars
        elif not lits or bound >= len(lits):
            clauses = []
            top_id = self.artifacts.n_vars
        elif bound == 0:
            clauses = [[-lit] for lit in lits]
            top_id = self.artifacts.n_vars
        else:
            encoding = CardEnc.atmost(
                lits=lits,
                bound=bound,
                top_id=self.artifacts.n_vars,
                encoding=EncType.seqcounter,
            )
            clauses = encoding.clauses
            top_id = encoding.nv

        self.n_bound_encodings += 1
        self.optimizer_added_variables_peak = max(
            self.optimizer_added_variables_peak,
            max(0, top_id - self.artifacts.n_vars),
        )
        self.optimizer_added_clauses_peak = max(
            self.optimizer_added_clauses_peak,
            len(clauses),
        )
        self.optimizer_added_literals_peak = max(
            self.optimizer_added_literals_peak,
            sum(len(clause) for clause in clauses),
        )
        self.optimizer_added_clauses_cumulative += len(clauses)
        return clauses

    def solve(
        self,
        verbose: bool = False,
        incumbent_callback: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        with _new_solver(self.artifacts.cnf.clauses, self.solver_name) as solver:
            self.n_optimizer_calls += 1
            if not solver.solve():
                return self._pack_result("UNSAT", None, None)
            initial_model = solver.get_model()

        best_assignment, best_stats, checks = self._evaluate_sat_model(initial_model)
        if checks:
            return self._pack_result("ERROR", best_assignment, best_stats, checks)

        best_objective = best_stats.objective_gap
        if incumbent_callback is not None:
            incumbent_callback(best_objective)
        if verbose:
            print(f"[MultipleSAT] initial IdleRange(P*)={best_objective}")
        if best_objective == 0:
            return self._pack_result(
                "OPTIMAL",
                best_assignment,
                best_stats,
                proven_optimum=0,
            )

        low, high = 0, best_objective - 1
        while low <= high:
            bound = (low + high) // 2
            with _new_solver(self.artifacts.cnf.clauses, self.solver_name) as solver:
                solver.append_formula(self._bound_clauses(bound))
                self.n_optimizer_calls += 1
                satisfiable = solver.solve()
                if verbose:
                    print(
                        f"[MultipleSAT] IdleRange(P*) <= {bound}: "
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
                    if incumbent_callback is not None:
                        incumbent_callback(best_stats.objective_gap)
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
    return B2BMultipleSATSolver(
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
