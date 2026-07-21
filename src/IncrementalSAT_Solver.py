from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from pysat.card import CardEnc, EncType, ITotalizer

from B2B_Instance import B2BInstance, B2BSATModel, B2BSolutionStats, read_instance


def _ensure_instance(instance_or_path: B2BInstance | str | Path) -> B2BInstance:
    return (
        instance_or_path
        if isinstance(instance_or_path, B2BInstance)
        else read_instance(instance_or_path)
    )


def _new_solver(clauses: list[list[int]], preferred: str = "cadical"):
    """Create a SAT solver, falling back to Glucose3 when CaDiCaL is unavailable."""
    solvers = import_module("pysat.solvers")
    if preferred == "glucose":
        return solvers.Glucose3(bootstrap_with=clauses)
    try:
        return solvers.Cadical153(bootstrap_with=clauses)
    except Exception:
        return solvers.Glucose3(bootstrap_with=clauses)


class B2BIncrementalSATSolver:
    """Incremental SAT optimization of the internal-idle-slot range over P*.

    For participant p, B(p) is the number of idle slots strictly between p's
    first and last meetings. The optimized objective is::

        max_{p in P*} B(p) - min_{p in P*} B(p),

    where P* contains participants with at least two meetings.

    Within each optimization phase, one SAT solver is retained while an
    incremental totalizer imposes candidate upper bounds through assumptions.
    The optional second phase fixes the range optimum and minimizes IdleSum.
    """

    def __init__(
        self,
        instance_or_path: B2BInstance | str | Path,
        fairness_limit: int | None = None,
        precedence_mode: str = "traditional",
        encoding_variant: str = "imp12+",
        solver_name: str = "cadical",
        objective_mode: str = "idle-range",
        precedence_edge_mode: str = "direct",
    ) -> None:
        self.inst = _ensure_instance(instance_or_path)
        self.model = B2BSATModel(
            inst=self.inst,
            fairness_limit=fairness_limit,
            precedence_mode=precedence_mode,
            encoding_variant=encoding_variant,
            objective_mode=objective_mode,
            precedence_edge_mode=precedence_edge_mode,
        )
        self.artifacts = self.model.build_base_cnf()
        self.solver_name = solver_name

    def _pack_result(
        self,
        status: str,
        assignment: list[int] | None,
        stats: B2BSolutionStats | None,
        checks: list[str] | None = None,
        *,
        proven_optimum: int | None = None,
        secondary_optimum: int | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "solver": "IncrementalSAT",
            "precedence_mode": self.artifacts.precedence_mode,
            "precedence_edge_mode": self.artifacts.precedence_edge_mode,
            "encoding_variant": self.artifacts.encoding_variant,
            "objective": self.artifacts.objective_name,
            "objective_mode": self.artifacts.objective_mode,
            "objective_participant_count": len(
                self.artifacts.objective_participants
            ),
            "objective_participants": tuple(
                p + 1 for p in self.artifacts.objective_participants
            ),
            "objective_value": (
                stats.fairness_gap if stats is not None else proven_optimum
            ),
            "proven_optimum": proven_optimum,
            "secondary_objective_value": (
                stats.total_internal_idle_slots
                if stats is not None
                and self.artifacts.objective_mode == "lexicographic"
                else None
            ),
            "secondary_proven_optimum": secondary_optimum,
            "hard_fairness_limit": self.artifacts.fairness_limit,
            "assignment": assignment,
            "stats": stats,
            "validation_errors": checks or [],
            "n_vars": self.artifacts.n_vars,
            "n_clauses": self.artifacts.n_clauses,
            "n_objective_lits": len(self.artifacts.objective_lits),
            "n_primary_objective_lits": len(self.artifacts.objective_lits),
            "n_secondary_objective_lits": len(
                self.artifacts.secondary_objective_lits
            ),
            "precedence_direct_edges": self.artifacts.precedence_direct_edges,
            "precedence_source_added_edges": (
                self.artifacts.precedence_source_added_edges
            ),
            "precedence_encoded_edges": self.artifacts.precedence_encoded_edges,
            "precedence_full_closure_edges": (
                self.artifacts.precedence_transitive_edges
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

    @staticmethod
    def _cardinality_bound(
        lits: list[int],
        bound: int,
        top_id: int,
    ) -> tuple[list[list[int]], int]:
        if bound < 0:
            return [[]], top_id
        if not lits or bound >= len(lits):
            return [], top_id
        if bound == 0:
            return [[-lit] for lit in lits], top_id
        encoding = CardEnc.atmost(
            lits=lits,
            bound=bound,
            top_id=top_id,
            encoding=EncType.seqcounter,
        )
        return encoding.clauses, encoding.nv

    def _optimize_secondary(
        self,
        primary_optimum: int,
        *,
        verbose: bool = False,
    ) -> tuple[list[int] | None, B2BSolutionStats | None, list[str], int | None]:
        """Incrementally minimize IdleSum with the optimal range fixed as hard."""
        primary_clauses, primary_top = self._cardinality_bound(
            self.artifacts.objective_lits,
            primary_optimum,
            self.artifacts.n_vars,
        )
        phase2_base = [*self.artifacts.cnf.clauses, *primary_clauses]

        with _new_solver(phase2_base, self.solver_name) as solver:
            if not solver.solve():
                return None, None, [
                    "phase 2 became UNSAT after fixing the phase-1 optimum"
                ], None

            initial_model = solver.get_model()
            best_assignment, best_stats, checks = self._evaluate_sat_model(
                initial_model,
                imposed_bound=primary_optimum,
            )
            checks.extend(
                self.model.secondary_objective_consistency_errors(
                    initial_model,
                    best_stats,
                )
            )
            if checks:
                return best_assignment, best_stats, checks, None

            best_sum = best_stats.total_internal_idle_slots
            if best_sum == 0:
                return best_assignment, best_stats, [], 0

            with ITotalizer(
                lits=self.artifacts.secondary_objective_lits,
                ubound=best_sum,
                top_id=primary_top,
            ) as totalizer:
                solver.append_formula(totalizer.cnf.clauses)
                low, high = 0, best_sum - 1

                while low <= high:
                    bound = (low + high) // 2
                    sat = solver.solve(assumptions=[-totalizer.rhs[bound]])
                    if verbose:
                        print(
                            f"[IncrementalSAT] IdleSum <= {bound}: "
                            f"{'SAT' if sat else 'UNSAT'}"
                        )
                    if sat:
                        sat_model = solver.get_model()
                        (
                            candidate_assignment,
                            candidate_stats,
                            candidate_checks,
                        ) = self._evaluate_sat_model(
                            sat_model,
                            imposed_bound=primary_optimum,
                        )
                        candidate_checks.extend(
                            self.model.secondary_objective_consistency_errors(
                                sat_model,
                                candidate_stats,
                                imposed_bound=bound,
                            )
                        )
                        if candidate_checks:
                            return (
                                candidate_assignment,
                                candidate_stats,
                                candidate_checks,
                                None,
                            )
                        best_assignment = candidate_assignment
                        best_stats = candidate_stats
                        high = bound - 1
                    else:
                        low = bound + 1

        final_checks = self.model.validate_assignment(best_assignment)
        if best_stats.fairness_gap != primary_optimum:
            final_checks.append(
                "lexicographic primary mismatch: "
                f"proven range={primary_optimum}, "
                f"schedule range={best_stats.fairness_gap}"
            )
        if best_stats.total_internal_idle_slots != low:
            final_checks.append(
                "lexicographic secondary mismatch: "
                f"proven IdleSum={low}, "
                f"schedule IdleSum={best_stats.total_internal_idle_slots}"
            )
        return best_assignment, best_stats, final_checks, low

    def solve(self, verbose: bool = False) -> dict[str, Any]:
        with _new_solver(self.artifacts.cnf.clauses, self.solver_name) as solver:
            if not solver.solve():
                return self._pack_result("UNSAT", None, None)

            initial_model = solver.get_model()
            best_assignment, best_stats, checks = self._evaluate_sat_model(initial_model)
            if checks:
                return self._pack_result("ERROR", best_assignment, best_stats, checks)

            best_obj = best_stats.fairness_gap
            if verbose:
                print(f"[IncrementalSAT] initial IdleRange(P*)={best_obj}")

            if best_obj == 0 and self.artifacts.objective_mode == "idle-range":
                return self._pack_result(
                    "OPTIMAL",
                    best_assignment,
                    best_stats,
                    proven_optimum=0,
                )

            objective_lits = self.artifacts.objective_lits
            low = 0
            if best_obj > 0:
                if not objective_lits:
                    checks = [
                        "nonzero schedule gap but the encoder produced no objective literals"
                    ]
                    return self._pack_result("ERROR", best_assignment, best_stats, checks)

                with ITotalizer(
                    lits=objective_lits,
                    ubound=best_obj,
                    top_id=self.artifacts.n_vars,
                ) as totalizer:
                    solver.append_formula(totalizer.cnf.clauses)
                    low, high = 0, best_obj - 1

                    while low <= high:
                        bound = (low + high) // 2
                        # rhs[bound] means at least bound+1 objective literals are true.
                        # Its negation therefore imposes sum(objective_lits) <= bound.
                        sat = solver.solve(assumptions=[-totalizer.rhs[bound]])
                        if verbose:
                            print(
                                "[IncrementalSAT] IdleRange(P*) <= "
                                f"{bound}: {'SAT' if sat else 'UNSAT'}"
                            )

                        if sat:
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
            if best_stats.fairness_gap != low:
                final_checks.append(
                    "optimization mismatch: "
                    f"proven optimum={low}, schedule gap={best_stats.fairness_gap}"
                )

            if self.artifacts.objective_mode == "lexicographic" and not final_checks:
                (
                    best_assignment,
                    best_stats,
                    final_checks,
                    secondary_optimum,
                ) = self._optimize_secondary(low, verbose=verbose)
                status = "OPTIMAL" if not final_checks else "ERROR"
                return self._pack_result(
                    status,
                    best_assignment,
                    best_stats,
                    final_checks,
                    proven_optimum=low,
                    secondary_optimum=secondary_optimum,
                )

            status = "OPTIMAL" if not final_checks else "ERROR"
            return self._pack_result(
                status,
                best_assignment,
                best_stats,
                final_checks,
                proven_optimum=low,
            )


def solve_b2b(
    instance_or_path: B2BInstance | str | Path,
    fairness_limit: int | None = None,
    precedence_mode: str = "traditional",
    encoding_variant: str = "imp12+",
    verbose: bool = False,
    objective_mode: str = "idle-range",
    precedence_edge_mode: str = "direct",
) -> dict[str, Any]:
    return B2BIncrementalSATSolver(
        instance_or_path=instance_or_path,
        fairness_limit=fairness_limit,
        precedence_mode=precedence_mode,
        encoding_variant=encoding_variant,
        objective_mode=objective_mode,
        precedence_edge_mode=precedence_edge_mode,
    ).solve(verbose=verbose)


def solve_b2b_traditional(
    instance_or_path: B2BInstance | str | Path,
    fairness_limit: int | None = None,
    encoding_variant: str = "imp12+",
    verbose: bool = False,
    objective_mode: str = "idle-range",
    precedence_edge_mode: str = "direct",
) -> dict[str, Any]:
    return solve_b2b(
        instance_or_path,
        fairness_limit,
        "traditional",
        encoding_variant,
        verbose,
        objective_mode,
        precedence_edge_mode,
    )


def solve_b2b_staircase(
    instance_or_path: B2BInstance | str | Path,
    fairness_limit: int | None = None,
    encoding_variant: str = "imp12+",
    verbose: bool = False,
    objective_mode: str = "idle-range",
    precedence_edge_mode: str = "direct",
) -> dict[str, Any]:
    return solve_b2b(
        instance_or_path,
        fairness_limit,
        "staircase",
        encoding_variant,
        verbose,
        objective_mode,
        precedence_edge_mode,
    )
