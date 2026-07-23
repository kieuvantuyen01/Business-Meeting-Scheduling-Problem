from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from B2B_Instance import B2BInstance
from Exact_Model_Common import exact_result, load_exact_context
from MIP_SpanRange import MIPSpanRangeSpec, build_mip_span_range


class B2BCPLEXMIPSolver:
    """DOcplex.MP adapter for the shared MIP-SpanRange model."""

    solver_backend = "CPLEX-MIP"
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
        self.spec, self.artifacts = build_mip_span_range(self.context)
        self.solver_timeout = solver_timeout
        self.threads = threads
        self.random_seed = random_seed
        self.solver_version = ""
        self.solver_command = (
            f"docplex.mp; threads={threads}; randomseed={random_seed}; "
            "mipgap=0; absmipgap=0"
        )

    @staticmethod
    def _decode_assignment(
        spec: MIPSpanRangeSpec,
        solution: Any,
        variables: dict[str, Any],
        n_meetings: int,
    ) -> list[int]:
        assignment = [-1] * n_meetings
        for (meeting, slot), name in spec.x.items():
            if float(solution.get_value(variables[name])) > 0.5:
                assignment[meeting] = slot
        return assignment

    def solve(self, *, verbose: bool = False) -> dict[str, Any]:
        backend_build_started = time.perf_counter()
        try:
            from docplex.mp.model import Model
        except ImportError as exc:
            raise RuntimeError(
                "CPLEX MIP requires the optional 'docplex' package plus a "
                "local or remote CPLEX runtime; no fallback solver is used"
            ) from exc

        model = Model(name="B2B_MIP_SpanRange")
        try:
            self.solver_version = str(
                getattr(model.environment, "cplex_version", "") or ""
            )
            model.parameters.threads = self.threads
            model.parameters.randomseed = self.random_seed
            model.parameters.mip.tolerances.mipgap = 0.0
            model.parameters.mip.tolerances.absmipgap = 0.0

            variables: dict[str, Any] = {}
            for variable in self.spec.variables.values():
                if variable.vartype == "B":
                    backend_var = model.binary_var(name=variable.name)
                elif variable.vartype == "I":
                    backend_var = model.integer_var(
                        lb=variable.lb,
                        ub=variable.ub,
                        name=variable.name,
                    )
                else:
                    backend_var = model.continuous_var(
                        lb=variable.lb,
                        ub=variable.ub,
                        name=variable.name,
                    )
                variables[variable.name] = backend_var

            for constraint in self.spec.constraints:
                expression = model.linear_expr()
                for name, coefficient in constraint.terms:
                    expression += coefficient * variables[name]
                if constraint.sense == "<=":
                    relation = expression <= constraint.rhs
                elif constraint.sense == ">=":
                    relation = expression >= constraint.rhs
                else:
                    relation = expression == constraint.rhs
                model.add_constraint(relation, ctname=constraint.name)

            objective = model.linear_expr()
            assert self.spec.objective is not None
            for name, coefficient in self.spec.objective.terms:
                objective += coefficient * variables[name]
            model.minimize(objective)
            backend_model_construction_seconds = (
                time.perf_counter() - backend_build_started
            )
            if self.solver_timeout is not None:
                model.parameters.timelimit = max(
                    0.001,
                    self.solver_timeout
                    - backend_model_construction_seconds,
                )
            solution = model.solve(log_output=verbose)
            details = model.solve_details
            details_status = str(getattr(details, "status", "") or "")
            status_lower = details_status.lower()
            hit_limit = bool(
                getattr(details, "has_hit_limit", lambda: False)()
            )
            has_solution = solution is not None
            if has_solution and "optimal" in status_lower:
                status = "OPTIMAL"
            elif "infeasible" in status_lower:
                status = "UNSAT"
            elif hit_limit or "time limit" in status_lower:
                status = "TIMEOUT"
            elif has_solution:
                status = "FEASIBLE"
            else:
                status = "ERROR"

            assignment = (
                self._decode_assignment(
                    self.spec,
                    solution,
                    variables,
                    self.context.inst.n_meetings,
                )
                if has_solution
                else None
            )
            objective_value = (
                solution.objective_value if has_solution else None
            )
            return exact_result(
                self.context,
                self.artifacts,
                status=status,
                solver_backend=self.solver_backend,
                solver_version=self.solver_version,
                assignment=assignment,
                objective_value=objective_value,
                best_bound=getattr(details, "best_bound", None),
                optimality_gap=(
                    getattr(details, "mip_relative_gap", None)
                    if has_solution
                    else None
                ),
                proven_optimum=status == "OPTIMAL",
                solver_message=details_status,
                metrics={
                    "backend_model_construction_seconds": round(
                        backend_model_construction_seconds,
                        6,
                    ),
                    "branch_and_bound_nodes": getattr(
                        details,
                        "nb_nodes_processed",
                        None,
                    ),
                    "n_optimizer_calls": 1,
                },
            )
        finally:
            model.end()
