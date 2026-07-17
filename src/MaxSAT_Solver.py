from __future__ import annotations

from pathlib import Path
from typing import Any

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

from B2B_Instance import B2BInstance, B2BSATModel, B2BSolutionStats, read_instance


def _ensure_instance(instance_or_path: B2BInstance | str | Path) -> B2BInstance:
    return (
        instance_or_path
        if isinstance(instance_or_path, B2BInstance)
        else read_instance(instance_or_path)
    )


class B2BMaxSATSolver:
    """MaxSAT optimization of the internal-idle-slot fairness gap.

    B2B_Instance exposes unary objective literals whose true count is exactly
    max_p B(p) - min_p B(p). RC2 minimizes that count using unit soft clauses
    [-lit]. An optional fairness_limit adds a hard upper bound on the same gap.
    """

    def __init__(
        self,
        instance_or_path: B2BInstance | str | Path,
        fairness_limit: int | None = None,
        precedence_mode: str = "traditional",
        encoding_variant: str = "imp12+",
    ) -> None:
        self.inst = _ensure_instance(instance_or_path)
        self.model = B2BSATModel(
            inst=self.inst,
            fairness_limit=fairness_limit,
            precedence_mode=precedence_mode,
            encoding_variant=encoding_variant,
        )
        self.artifacts = self.model.build_base_cnf()

    def _build_wcnf(self) -> WCNF:
        """Use the exact hard and soft clauses produced by the shared encoder."""
        return self.model.build_wcnf()

    def _pack_result(
        self,
        status: str,
        assignment: list[int] | None,
        stats: B2BSolutionStats | None,
        checks: list[str] | None = None,
        *,
        solver_cost: int | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "solver": "MaxSAT",
            "precedence_mode": self.artifacts.precedence_mode,
            "encoding_variant": self.artifacts.encoding_variant,
            "objective": self.artifacts.objective_name,
            "objective_value": (
                stats.fairness_gap if stats is not None else solver_cost
            ),
            "proven_optimum": solver_cost,
            "solver_cost": solver_cost,
            "hard_fairness_limit": self.artifacts.fairness_limit,
            "assignment": assignment,
            "stats": stats,
            "validation_errors": checks or [],
            "n_vars": self.artifacts.n_vars,
            "n_clauses": self.artifacts.n_clauses,
            "n_soft": len(self.artifacts.objective_lits),
            "enabled_constraints": self.artifacts.enabled_constraints,
        }

    def solve(self, verbose: bool = False) -> dict[str, Any]:
        wcnf = self._build_wcnf()

        with RC2(wcnf) as solver:
            sat_model = solver.compute()
            if sat_model is None:
                return self._pack_result("UNSAT", None, None)

            assignment = self.model.decode_assignment(sat_model)
            stats = self.model.compute_stats(assignment)
            solver_cost = int(solver.cost)

            checks = self.model.validate_assignment(assignment)
            checks.extend(
                self.model.objective_consistency_errors(
                    sat_model,
                    stats,
                    solver_cost=solver_cost,
                )
            )

            if verbose:
                print(
                    "[MaxSAT] optimum internal-break gap="
                    f"{stats.fairness_gap} (RC2 cost={solver_cost})"
                )

            status = "OPTIMAL" if not checks else "ERROR"
            return self._pack_result(
                status,
                assignment,
                stats,
                checks,
                solver_cost=solver_cost,
            )


def solve_b2b(
    instance_or_path: B2BInstance | str | Path,
    fairness_limit: int | None = None,
    precedence_mode: str = "traditional",
    encoding_variant: str = "imp12+",
    verbose: bool = False,
) -> dict[str, Any]:
    return B2BMaxSATSolver(
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
