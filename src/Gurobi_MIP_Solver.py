from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from B2B_Instance import B2BInstance
from Exact_Model_Common import exact_result, load_exact_context
from MIP_SpanRange import MIPSpanRangeSpec, build_mip_span_range


class B2BGurobiMIPSolver:
    """Gurobi adapter for the shared backend-neutral MIP-SpanRange model."""

    solver_backend = "Gurobi-MIP"
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
            f"gurobipy; Threads={threads}; Seed={random_seed}; "
            "MIPGap=0; MIPGapAbs=0"
        )

    @staticmethod
    def _decode_assignment(
        spec: MIPSpanRangeSpec,
        variables: dict[str, Any],
        n_meetings: int,
    ) -> list[int]:
        assignment = [-1] * n_meetings
        for (meeting, slot), name in spec.x.items():
            if float(variables[name].X) > 0.5:
                assignment[meeting] = slot
        return assignment

    def solve(self, *, verbose: bool = False) -> dict[str, Any]:
        backend_build_started = time.perf_counter()
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:
            raise RuntimeError(
                "Gurobi MIP requires the optional 'gurobipy' package and "
                "a usable Gurobi license; no fallback solver is used"
            ) from exc

        version = gp.gurobi.version()
        self.solver_version = ".".join(str(part) for part in version)
        model = gp.Model("B2B_MIP_SpanRange")
        try:
            model.Params.OutputFlag = 1 if verbose else 0
            model.Params.Threads = self.threads
            model.Params.Seed = self.random_seed
            model.Params.MIPGap = 0.0
            model.Params.MIPGapAbs = 0.0
            type_map = {
                "B": GRB.BINARY,
                "I": GRB.INTEGER,
                "C": GRB.CONTINUOUS,
            }
            variables = {
                variable.name: model.addVar(
                    lb=variable.lb,
                    ub=variable.ub,
                    vtype=type_map[variable.vartype],
                    name=variable.name,
                )
                for variable in self.spec.variables.values()
            }
            model.update()

            for constraint in self.spec.constraints:
                expression = gp.LinExpr()
                for name, coefficient in constraint.terms:
                    expression.addTerms(coefficient, variables[name])
                if constraint.sense == "<=":
                    relation = expression <= constraint.rhs
                elif constraint.sense == ">=":
                    relation = expression >= constraint.rhs
                else:
                    relation = expression == constraint.rhs
                model.addConstr(relation, name=constraint.name)

            objective = gp.LinExpr()
            assert self.spec.objective is not None
            for name, coefficient in self.spec.objective.terms:
                objective.addTerms(coefficient, variables[name])
            model.setObjective(objective, GRB.MINIMIZE)
            backend_model_construction_seconds = (
                time.perf_counter() - backend_build_started
            )
            if self.solver_timeout is not None:
                model.Params.TimeLimit = max(
                    0.001,
                    self.solver_timeout
                    - backend_model_construction_seconds,
                )
            model.optimize()

            status_code = model.Status
            has_solution = model.SolCount > 0
            if status_code == GRB.OPTIMAL:
                status = "OPTIMAL"
            elif status_code == GRB.INFEASIBLE:
                status = "UNSAT"
            elif status_code == GRB.TIME_LIMIT:
                status = "TIMEOUT"
            elif has_solution:
                status = "FEASIBLE"
            else:
                status = "ERROR"

            assignment = (
                self._decode_assignment(
                    self.spec,
                    variables,
                    self.context.inst.n_meetings,
                )
                if has_solution
                else None
            )
            objective_value = model.ObjVal if has_solution else None
            best_bound = (
                model.ObjBound
                if status_code not in {GRB.LOADED, GRB.INPROGRESS}
                else None
            )
            optimality_gap = model.MIPGap if has_solution else None
            return exact_result(
                self.context,
                self.artifacts,
                status=status,
                solver_backend=self.solver_backend,
                solver_version=self.solver_version,
                assignment=assignment,
                objective_value=objective_value,
                best_bound=best_bound,
                optimality_gap=optimality_gap,
                proven_optimum=status == "OPTIMAL",
                solver_message=f"Gurobi status code {status_code}",
                metrics={
                    "backend_model_construction_seconds": round(
                        backend_model_construction_seconds,
                        6,
                    ),
                    "branch_and_bound_nodes": float(model.NodeCount),
                    "n_optimizer_calls": 1,
                },
            )
        finally:
            model.dispose()
