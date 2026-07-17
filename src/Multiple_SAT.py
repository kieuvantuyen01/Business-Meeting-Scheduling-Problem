from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from pysat.card import CardEnc, EncType

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


class B2BMultipleSATSolver:
    """Repeated-SAT optimization of the internal-idle-slot fairness gap.

    Every candidate bound is solved in a fresh SAT solver. The objective literals
    supplied by B2B_Instance encode exactly max_p B(p) - min_p B(p), where B(p)
    counts idle slots strictly between participant p's first and last meetings.
    """

    def __init__(
        self,
        instance_or_path: B2BInstance | str | Path,
        fairness_limit: int | None = None,
        precedence_mode: str = "traditional",
        encoding_variant: str = "imp12+",
        solver_name: str = "cadical",
    ) -> None:
        self.inst = _ensure_instance(instance_or_path)
        self.model = B2BSATModel(
            inst=self.inst,
            fairness_limit=fairness_limit,
            precedence_mode=precedence_mode,
            encoding_variant=encoding_variant,
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
    ) -> dict[str, Any]:
        return {
            "status": status,
            "solver": "MultipleSAT",
            "precedence_mode": self.artifacts.precedence_mode,
            "encoding_variant": self.artifacts.encoding_variant,
            "objective": self.artifacts.objective_name,
            "objective_value": (
                stats.fairness_gap if stats is not None else proven_optimum
            ),
            "proven_optimum": proven_optimum,
            "hard_fairness_limit": self.artifacts.fairness_limit,
            "assignment": assignment,
            "stats": stats,
            "validation_errors": checks or [],
            "n_vars": self.artifacts.n_vars,
            "n_clauses": self.artifacts.n_clauses,
            "n_objective_lits": len(self.artifacts.objective_lits),
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

    def _bound_clauses(self, bound: int) -> list[list[int]]:
        """Encode sum(objective_lits) <= bound for one fresh SAT run."""
        lits = self.artifacts.objective_lits
        if bound < 0:
            return [[]]
        if not lits or bound >= len(lits):
            return []
        if bound == 0:
            return [[-lit] for lit in lits]

        encoding = CardEnc.atmost(
            lits=lits,
            bound=bound,
            top_id=self.artifacts.n_vars,
            encoding=EncType.seqcounter,
        )
        return encoding.clauses

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
            print(f"[MultipleSAT] initial internal-break gap={best_obj}")

        if best_obj == 0:
            return self._pack_result(
                "OPTIMAL",
                best_assignment,
                best_stats,
                proven_optimum=0,
            )

        low, high = 0, best_obj - 1
        while low <= high:
            bound = (low + high) // 2
            bound_clauses = self._bound_clauses(bound)

            with _new_solver(self.artifacts.cnf.clauses, self.solver_name) as solver:
                solver.append_formula(bound_clauses)
                sat = solver.solve()
                if verbose:
                    print(
                        "[MultipleSAT] internal-break gap <= "
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
) -> dict[str, Any]:
    return B2BMultipleSATSolver(
        instance_or_path=instance_or_path,
        fairness_limit=fairness_limit,
        precedence_mode=precedence_mode,
        encoding_variant=encoding_variant,
    ).solve(verbose=verbose)


def solve_b2b_traditional(
    instance_or_path: B2BInstance | str | Path,
    fairness_limit: int | None = None,
    encoding_variant: str = "imp12+",
    verbose: bool = False,
) -> dict[str, Any]:
    return solve_b2b(
        instance_or_path,
        fairness_limit,
        "traditional",
        encoding_variant,
        verbose,
    )


def solve_b2b_staircase(
    instance_or_path: B2BInstance | str | Path,
    fairness_limit: int | None = None,
    encoding_variant: str = "imp12+",
    verbose: bool = False,
) -> dict[str, Any]:
    return solve_b2b(
        instance_or_path,
        fairness_limit,
        "staircase",
        encoding_variant,
        verbose,
    )
