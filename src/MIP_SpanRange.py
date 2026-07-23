from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

from Exact_Model_Common import (
    ExactModelArtifacts,
    ExactModelContext,
    make_exact_artifacts,
    with_model_counts,
)


VariableType = Literal["B", "I", "C"]
ConstraintSense = Literal["<=", "==", ">="]


@dataclass(frozen=True)
class LinearVariable:
    name: str
    vartype: VariableType
    lb: float
    ub: float


@dataclass(frozen=True)
class LinearConstraint:
    name: str
    terms: tuple[tuple[str, float], ...]
    sense: ConstraintSense
    rhs: float


@dataclass(frozen=True)
class LinearObjective:
    sense: Literal["min"]
    terms: tuple[tuple[str, float], ...]


@dataclass
class MIPSpanRangeSpec:
    """Backend-neutral linear IR shared by Gurobi MIP and CPLEX MIP."""

    variables: dict[str, LinearVariable] = field(default_factory=dict)
    constraints: list[LinearConstraint] = field(default_factory=list)
    objective: LinearObjective | None = None
    x: dict[tuple[int, int], str] = field(default_factory=dict)
    y: dict[tuple[int, int], str] = field(default_factory=dict)
    prefix: dict[tuple[int, int], str] = field(default_factory=dict)
    suffix: dict[tuple[int, int], str] = field(default_factory=dict)
    idle: dict[int, str] = field(default_factory=dict)
    idle_max: str = ""
    idle_min: str = ""
    idle_range: str = ""

    def add_variable(
        self,
        name: str,
        vartype: VariableType,
        lb: float,
        ub: float,
    ) -> str:
        if name in self.variables:
            raise ValueError(f"duplicate MIP variable name: {name}")
        self.variables[name] = LinearVariable(name, vartype, lb, ub)
        return name

    def add_constraint(
        self,
        name: str,
        terms: dict[str, float] | Iterable[tuple[str, float]],
        sense: ConstraintSense,
        rhs: float,
    ) -> None:
        combined: dict[str, float] = {}
        items = terms.items() if isinstance(terms, dict) else terms
        for variable, coefficient in items:
            if variable not in self.variables:
                raise KeyError(f"unknown MIP variable {variable!r}")
            combined[variable] = combined.get(variable, 0.0) + coefficient
        normalized = tuple(
            (variable, coefficient)
            for variable, coefficient in combined.items()
            if coefficient != 0
        )
        self.constraints.append(
            LinearConstraint(name, normalized, sense, float(rhs))
        )

    @property
    def n_binary_variables(self) -> int:
        return sum(var.vartype == "B" for var in self.variables.values())

    @property
    def n_integer_variables(self) -> int:
        return sum(var.vartype == "I" for var in self.variables.values())

    @property
    def n_continuous_variables(self) -> int:
        return sum(var.vartype == "C" for var in self.variables.values())

    @property
    def n_nonzeros(self) -> int:
        return sum(len(constraint.terms) for constraint in self.constraints)

    def constraint_violations(
        self,
        values: dict[str, float],
        *,
        tolerance: float = 1e-9,
    ) -> list[str]:
        violations: list[str] = []
        for variable in self.variables.values():
            value = values.get(variable.name, 0.0)
            if value < variable.lb - tolerance or value > variable.ub + tolerance:
                violations.append(f"bound:{variable.name}")
            if variable.vartype in {"B", "I"} and abs(value - round(value)) > tolerance:
                violations.append(f"integrality:{variable.name}")
        for constraint in self.constraints:
            lhs = sum(
                coefficient * values.get(variable, 0.0)
                for variable, coefficient in constraint.terms
            )
            if constraint.sense == "<=" and lhs > constraint.rhs + tolerance:
                violations.append(constraint.name)
            elif constraint.sense == ">=" and lhs < constraint.rhs - tolerance:
                violations.append(constraint.name)
            elif constraint.sense == "==" and abs(lhs - constraint.rhs) > tolerance:
                violations.append(constraint.name)
        return violations


MIP_ENABLED_CONSTRAINTS = (
    "exactly one reduced-domain slot per meeting",
    "participant slot all-different",
    "per-slot table capacity",
    "distance-labelled transitive precedence closure",
    "sparse participant occupancy y only on active (p,t) pairs",
    "exact prefix/suffix OR recurrences with constant boundaries",
    "zero-based idle identity I=sum(A)+sum(R)-H-|M_p|",
    "exact max/min range objective over P*",
)


def build_mip_span_range(
    context: ExactModelContext,
) -> tuple[MIPSpanRangeSpec, ExactModelArtifacts]:
    """Build MIP-SpanRange once for both commercial MIP backends."""

    inst = context.inst
    horizon = inst.n_total_slots
    spec = MIPSpanRangeSpec()

    for meeting, domain in enumerate(context.domains):
        for slot in domain:
            name = spec.add_variable(
                f"x_{meeting}_{slot}",
                "B",
                0,
                1,
            )
            spec.x[meeting, slot] = name

    active_slots: dict[int, set[int]] = {}
    for participant in context.objective_participants:
        slots = {
            slot
            for meeting in inst.meetings_by_business[participant]
            for slot in context.domains[meeting]
        }
        active_slots[participant] = slots
        for slot in sorted(slots):
            spec.y[participant, slot] = spec.add_variable(
                f"y_{participant}_{slot}",
                "B",
                0,
                1,
            )
        for slot in range(horizon):
            spec.prefix[participant, slot] = spec.add_variable(
                f"A_{participant}_{slot}",
                "B",
                0,
                1,
            )
            spec.suffix[participant, slot] = spec.add_variable(
                f"R_{participant}_{slot}",
                "B",
                0,
                1,
            )
        idle_upper = max(
            0,
            horizon - len(inst.meetings_by_business[participant]),
        )
        spec.idle[participant] = spec.add_variable(
            f"I_{participant}",
            "I",
            0,
            idle_upper,
        )

    global_idle_upper = max(
        (
            horizon - len(inst.meetings_by_business[participant])
            for participant in context.objective_participants
        ),
        default=0,
    )
    spec.idle_max = spec.add_variable(
        "I_max",
        "I",
        0,
        global_idle_upper,
    )
    spec.idle_min = spec.add_variable(
        "I_min",
        "I",
        0,
        global_idle_upper,
    )
    spec.idle_range = spec.add_variable(
        "Delta",
        "I",
        0,
        global_idle_upper,
    )

    # Every meeting is assigned exactly once. Empty domains deliberately create
    # the contradictory equality 0 == 1, allowing the backend to prove UNSAT.
    for meeting, domain in enumerate(context.domains):
        spec.add_constraint(
            f"assign_{meeting}",
            ((spec.x[meeting, slot], 1) for slot in domain),
            "==",
            1,
        )

    # Participant conflicts are generated on sparse reduced-domain candidates.
    for participant, meetings in enumerate(inst.meetings_by_business):
        for slot in range(horizon):
            candidates = [
                spec.x[meeting, slot]
                for meeting in meetings
                if (meeting, slot) in spec.x
            ]
            if len(candidates) >= 2:
                spec.add_constraint(
                    f"participant_{participant}_{slot}",
                    ((variable, 1) for variable in candidates),
                    "<=",
                    1,
                )

    for slot in range(horizon):
        candidates = [
            variable
            for (meeting, candidate_slot), variable in spec.x.items()
            if candidate_slot == slot
        ]
        if candidates:
            spec.add_constraint(
                f"capacity_{slot}",
                ((variable, 1) for variable in candidates),
                "<=",
                inst.n_tables,
            )

    # Zero-based time variables are represented by sum(t*x[m,t]). Distance d is
    # unchanged by the index shift: tau_pred + d <= tau_post.
    for post, distances in enumerate(context.graph.longest_distance):
        for pred, distance in sorted(distances.items()):
            terms: list[tuple[str, float]] = []
            terms.extend(
                (spec.x[pred, slot], slot)
                for slot in context.domains[pred]
                if slot != 0
            )
            terms.extend(
                (spec.x[post, slot], -slot)
                for slot in context.domains[post]
                if slot != 0
            )
            spec.add_constraint(
                f"precedence_{pred}_{post}",
                terms,
                "<=",
                -distance,
            )

    for participant in context.objective_participants:
        meetings = inst.meetings_by_business[participant]
        for slot in sorted(active_slots[participant]):
            y_name = spec.y[participant, slot]
            terms = [(y_name, 1.0)]
            terms.extend(
                (spec.x[meeting, slot], -1.0)
                for meeting in meetings
                if (meeting, slot) in spec.x
            )
            spec.add_constraint(
                f"occupancy_{participant}_{slot}",
                terms,
                "==",
                0,
            )

        for slot in range(horizon):
            prefix = spec.prefix[participant, slot]
            y_name = spec.y.get((participant, slot))
            previous = (
                spec.prefix[participant, slot - 1]
                if slot > 0
                else None
            )
            if y_name is None:
                terms = [(prefix, 1.0)]
                if previous is not None:
                    terms.append((previous, -1.0))
                spec.add_constraint(
                    f"prefix_zero_{participant}_{slot}",
                    terms,
                    "==",
                    0,
                )
            elif previous is None:
                spec.add_constraint(
                    f"prefix_start_{participant}_{slot}",
                    ((prefix, 1.0), (y_name, -1.0)),
                    "==",
                    0,
                )
            else:
                spec.add_constraint(
                    f"prefix_prev_{participant}_{slot}",
                    ((prefix, 1.0), (previous, -1.0)),
                    ">=",
                    0,
                )
                spec.add_constraint(
                    f"prefix_busy_{participant}_{slot}",
                    ((prefix, 1.0), (y_name, -1.0)),
                    ">=",
                    0,
                )
                spec.add_constraint(
                    f"prefix_or_{participant}_{slot}",
                    (
                        (prefix, 1.0),
                        (previous, -1.0),
                        (y_name, -1.0),
                    ),
                    "<=",
                    0,
                )

        for slot in range(horizon - 1, -1, -1):
            suffix = spec.suffix[participant, slot]
            y_name = spec.y.get((participant, slot))
            following = (
                spec.suffix[participant, slot + 1]
                if slot + 1 < horizon
                else None
            )
            if y_name is None:
                terms = [(suffix, 1.0)]
                if following is not None:
                    terms.append((following, -1.0))
                spec.add_constraint(
                    f"suffix_zero_{participant}_{slot}",
                    terms,
                    "==",
                    0,
                )
            elif following is None:
                spec.add_constraint(
                    f"suffix_end_{participant}_{slot}",
                    ((suffix, 1.0), (y_name, -1.0)),
                    "==",
                    0,
                )
            else:
                spec.add_constraint(
                    f"suffix_next_{participant}_{slot}",
                    ((suffix, 1.0), (following, -1.0)),
                    ">=",
                    0,
                )
                spec.add_constraint(
                    f"suffix_busy_{participant}_{slot}",
                    ((suffix, 1.0), (y_name, -1.0)),
                    ">=",
                    0,
                )
                spec.add_constraint(
                    f"suffix_or_{participant}_{slot}",
                    (
                        (suffix, 1.0),
                        (following, -1.0),
                        (y_name, -1.0),
                    ),
                    "<=",
                    0,
                )

        # With 0-based slots, sum(A)=H-F and sum(R)=L+1, so the same identity
        # I=sum(A)+sum(R)-H-|M_p| remains exact (no off-by-one adjustment).
        idle_terms: list[tuple[str, float]] = [
            (spec.idle[participant], 1.0)
        ]
        idle_terms.extend(
            (spec.prefix[participant, slot], -1.0)
            for slot in range(horizon)
        )
        idle_terms.extend(
            (spec.suffix[participant, slot], -1.0)
            for slot in range(horizon)
        )
        spec.add_constraint(
            f"idle_{participant}",
            idle_terms,
            "==",
            -horizon - len(meetings),
        )
        spec.add_constraint(
            f"maximum_{participant}",
            (
                (spec.idle_max, 1.0),
                (spec.idle[participant], -1.0),
            ),
            ">=",
            0,
        )
        spec.add_constraint(
            f"minimum_{participant}",
            (
                (spec.idle_min, 1.0),
                (spec.idle[participant], -1.0),
            ),
            "<=",
            0,
        )

    spec.add_constraint(
        "idle_range_definition",
        (
            (spec.idle_range, 1.0),
            (spec.idle_max, -1.0),
            (spec.idle_min, 1.0),
        ),
        "==",
        0,
    )
    spec.objective = LinearObjective(
        "min",
        ((spec.idle_range, 1.0),),
    )

    artifacts = make_exact_artifacts(
        context,
        formalism="MIP",
        model_family="MIP-SpanRange",
        formulation_name="MIP-SpanRange-Reduced-DistanceClosure",
        precedence_encoding="native_linear",
        objective_encoding="prefix_suffix_span_range",
        enabled_constraints=MIP_ENABLED_CONSTRAINTS,
    )
    artifacts = with_model_counts(
        artifacts,
        n_vars=len(spec.variables),
        n_primary_variables=len(spec.x),
        n_binary_variables=spec.n_binary_variables,
        n_integer_variables=spec.n_integer_variables,
        n_continuous_variables=spec.n_continuous_variables,
        n_linear_constraints=len(spec.constraints),
        n_nonzeros=spec.n_nonzeros,
    )
    return spec, artifacts


def derive_mip_values(
    context: ExactModelContext,
    spec: MIPSpanRangeSpec,
    assignment: list[int],
) -> dict[str, float]:
    """Construct all MIP variable values implied by a complete schedule."""

    values = {name: 0.0 for name in spec.variables}
    for meeting, slot in enumerate(assignment):
        variable = spec.x.get((meeting, slot))
        if variable is not None:
            values[variable] = 1.0

    idle_values: list[int] = []
    for participant in context.objective_participants:
        meeting_slots = [
            assignment[meeting]
            for meeting in context.inst.meetings_by_business[participant]
        ]
        busy = set(meeting_slots)
        for slot in range(context.inst.n_total_slots):
            y_name = spec.y.get((participant, slot))
            if y_name is not None:
                values[y_name] = float(slot in busy)
            values[spec.prefix[participant, slot]] = float(
                any(meeting_slot <= slot for meeting_slot in meeting_slots)
            )
            values[spec.suffix[participant, slot]] = float(
                any(meeting_slot >= slot for meeting_slot in meeting_slots)
            )
        idle = (
            max(meeting_slots)
            - min(meeting_slots)
            + 1
            - len(meeting_slots)
        )
        values[spec.idle[participant]] = float(idle)
        idle_values.append(idle)

    maximum = max(idle_values, default=0)
    minimum = min(idle_values, default=0)
    values[spec.idle_max] = float(maximum)
    values[spec.idle_min] = float(minimum)
    values[spec.idle_range] = float(maximum - minimum)
    return values
