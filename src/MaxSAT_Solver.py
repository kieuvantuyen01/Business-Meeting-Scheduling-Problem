from __future__ import annotations

from pathlib import Path
from typing import Any

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

from B2B_Instance import B2BInstance, B2BSATModel, read_instance


def _ensure_instance(instance_or_path: B2BInstance | str | Path) -> B2BInstance:
    return instance_or_path if isinstance(instance_or_path, B2BInstance) else read_instance(instance_or_path)


class B2BMaxSATSolver:
    """Pure MaxSAT solver using RC2 on the same SAT encoding."""

    def __init__(
        self,
        instance_or_path: B2BInstance | str | Path,
        fairness_limit: int | None = 2,
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

    # =========================================================
    # BUILD WCNF
    # =========================================================

    def _build_wcnf(self) -> WCNF:

        wcnf = WCNF()

        # -----------------------------------------------------
        # HARD CLAUSES
        # -----------------------------------------------------

        for clause in self.artifacts.cnf.clauses:
            wcnf.append(clause)

        # -----------------------------------------------------
        # SOFT CLAUSES
        # minimize number of breaks
        # each objective literal = one break
        # -----------------------------------------------------

        for lit in self.artifacts.objective_lits:
            wcnf.append([-lit], weight=1)

        return wcnf

    # =========================================================
    # RESULT PACKING
    # =========================================================

    def _pack_result(
        self,
        status: str,
        assignment: list[int] | None,
        stats: Any | None,
        checks: list[str] | None = None,
        optimum: int | None = None,
    ) -> dict[str, Any]:

        return {
            "status": status,
            "solver": "MaxSAT",
            "precedence_mode": self.artifacts.precedence_mode,
            "encoding_variant": self.artifacts.encoding_variant,
            "assignment": assignment,
            "stats": stats,
            "validation_errors": checks or [],
            "objective": optimum,
            "n_vars": self.artifacts.n_vars,
            "n_clauses": self.artifacts.n_clauses,
            "n_soft": len(self.artifacts.objective_lits),
            "enabled_constraints": self.artifacts.enabled_constraints,
        }

    # =========================================================
    # SOLVE
    # =========================================================

    def solve(self, verbose: bool = False) -> dict[str, Any]:

        wcnf = self._build_wcnf()

        with RC2(wcnf) as solver:

            model = solver.compute()

            if model is None:
                return self._pack_result("UNSAT", None, None)

            assignment = self.model.decode_assignment(model)

            stats = self.model.compute_stats(assignment)

            checks = self.model.validate_assignment(assignment)

            optimum = solver.cost

            if verbose:
                print(f"[MaxSAT] optimum objective = {optimum}")

            return self._pack_result(
                "OPTIMAL",
                assignment,
                stats,
                checks,
                optimum,
            )


# =========================================================
# CONVENIENCE WRAPPERS
# =========================================================

def solve_b2b(
    instance_or_path: B2BInstance | str | Path,
    fairness_limit: int | None = 2,
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
    fairness_limit: int | None = 2,
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
    fairness_limit: int | None = 2,
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