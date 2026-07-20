from __future__ import annotations

from pathlib import Path
from typing import Any

from pysat.card import CardEnc, EncType
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
    """MaxSAT optimization of the internal-idle-slot range over P*.

    B2B_Instance exposes unary objective literals whose true count is exactly
    max_{p in P*} B(p) - min_{p in P*} B(p), where P* contains participants
    with at least two meetings. RC2 minimizes that count using unit soft clauses
    [-lit]. In ``lexicographic`` mode, a second RC2 run fixes the proven range
    optimum and minimizes the sum of participant internal-idle slots. An optional
    fairness_limit adds a hard upper bound on the same range.
    """

    def __init__(
        self,
        instance_or_path: B2BInstance | str | Path,
        fairness_limit: int | None = None,
        precedence_mode: str = "traditional",
        encoding_variant: str = "imp12+",
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

    def _build_wcnf(self) -> WCNF:
        """Use the exact hard and soft clauses produced by the shared encoder."""
        return self.model.build_wcnf()

    def _build_secondary_wcnf(self, primary_optimum: int) -> WCNF:
        """Fix the optimal range and minimize IdleSum in phase 2."""
        wcnf = WCNF()
        for clause in self.artifacts.cnf.clauses:
            wcnf.append(clause)

        primary_lits = self.artifacts.objective_lits
        if primary_optimum < 0:
            wcnf.append([])
        elif primary_optimum == 0:
            for lit in primary_lits:
                wcnf.append([-lit])
        elif primary_optimum < len(primary_lits):
            bound = CardEnc.atmost(
                lits=primary_lits,
                bound=primary_optimum,
                top_id=self.artifacts.n_vars,
                encoding=EncType.seqcounter,
            )
            for clause in bound.clauses:
                wcnf.append(clause)

        for lit in self.artifacts.secondary_objective_lits:
            wcnf.append([-lit], weight=1)
        return wcnf

    def _pack_result(
        self,
        status: str,
        assignment: list[int] | None,
        stats: B2BSolutionStats | None,
        checks: list[str] | None = None,
        *,
        solver_cost: int | None = None,
        secondary_optimum: int | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "solver": "MaxSAT",
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
                stats.fairness_gap if stats is not None else solver_cost
            ),
            "proven_optimum": solver_cost,
            "solver_cost": solver_cost,
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
            "n_soft": len(self.artifacts.objective_lits),
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

    def solve(self, verbose: bool = False) -> dict[str, Any]:
        wcnf = self._build_wcnf()

        with RC2(wcnf) as solver:
            primary_model = solver.compute()
            if primary_model is None:
                return self._pack_result("UNSAT", None, None)

            assignment = self.model.decode_assignment(primary_model)
            stats = self.model.compute_stats(assignment)
            primary_optimum = int(solver.cost)

            checks = self.model.validate_assignment(assignment)
            checks.extend(
                self.model.objective_consistency_errors(
                    primary_model,
                    stats,
                    solver_cost=primary_optimum,
                )
            )

        if checks:
            return self._pack_result(
                "ERROR",
                assignment,
                stats,
                checks,
                solver_cost=primary_optimum,
            )

        if self.artifacts.objective_mode == "lexicographic":
            secondary_wcnf = self._build_secondary_wcnf(primary_optimum)
            with RC2(secondary_wcnf) as secondary_solver:
                sat_model = secondary_solver.compute()
                if sat_model is None:
                    return self._pack_result(
                        "ERROR",
                        assignment,
                        stats,
                        ["phase 2 became UNSAT after fixing the phase-1 optimum"],
                        solver_cost=primary_optimum,
                    )
                secondary_optimum = int(secondary_solver.cost)

            assignment = self.model.decode_assignment(sat_model)
            stats = self.model.compute_stats(assignment)
            checks = self.model.validate_assignment(assignment)
            checks.extend(
                self.model.objective_consistency_errors(
                    sat_model,
                    stats,
                    imposed_bound=primary_optimum,
                )
            )
            checks.extend(
                self.model.secondary_objective_consistency_errors(
                    sat_model,
                    stats,
                    solver_cost=secondary_optimum,
                )
            )
            if stats.fairness_gap != primary_optimum:
                checks.append(
                    "lexicographic primary mismatch: "
                    f"proven range={primary_optimum}, "
                    f"schedule range={stats.fairness_gap}"
                )

            if verbose:
                print(
                    "[MaxSAT] lexicographic optimum="
                    f"({stats.fairness_gap}, {stats.total_internal_idle_slots})"
                )
            return self._pack_result(
                "OPTIMAL" if not checks else "ERROR",
                assignment,
                stats,
                checks,
                solver_cost=primary_optimum,
                secondary_optimum=secondary_optimum,
            )

        if verbose:
            print(
                "[MaxSAT] optimum IdleRange(P*)="
                f"{stats.fairness_gap} (RC2 cost={primary_optimum})"
            )

        return self._pack_result(
            "OPTIMAL",
            assignment,
            stats,
            checks,
            solver_cost=primary_optimum,
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
    return B2BMaxSATSolver(
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
