from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool, WCNF

PrecedenceMode = Literal["traditional", "staircase"]
PrecedenceEncoding = Literal["pairwise", "sparse_suffix"]
PrecedenceGraph = Literal["direct", "distance_closure"]
EncodingVariant = Literal["basic", "imp1", "imp2", "imp12", "imp12+"]
DomainMode = Literal["full", "reduced"]

VALID_PRECEDENCE_MODES = {"traditional", "staircase"}
VALID_PRECEDENCE_ENCODINGS = {"pairwise", "sparse_suffix"}
VALID_PRECEDENCE_GRAPHS = {"direct", "distance_closure"}
VALID_ENCODING_VARIANTS = {"basic", "imp1", "imp2", "imp12", "imp12+"}
VALID_DOMAIN_MODES = {"full", "reduced"}

LEGACY_PRECEDENCE_CONFIGURATIONS: dict[str, tuple[str, str]] = {
    "traditional": ("pairwise", "direct"),
    "staircase": ("sparse_suffix", "distance_closure"),
}


@dataclass(frozen=True)
class B2BInstance:
    """B2B instance using zero-based participants, meetings, and slots."""

    n_business: int
    n_meetings: int
    n_tables: int
    n_total_slots: int
    n_morning_slots: int
    requested: list[tuple[int, int, int]]  # (p1, p2, session), session in {1,2,3}
    meetings_by_business: list[list[int]]
    n_meetings_business: list[int]
    forbidden: list[set[int]]
    fixed: list[int | None]
    precedences: list[set[int]]  # precedences[post] = direct predecessors
    instance_name: str

    @property
    def morning_slots(self) -> range:
        return range(self.n_morning_slots)

    @property
    def afternoon_slots(self) -> range:
        return range(self.n_morning_slots, self.n_total_slots)

    @property
    def max_breaks_per_participant(self) -> int:
        """Legacy upper bound for the number of break groups from the paper."""
        return max(0, (self.n_total_slots - 1) // 2)

    @property
    def max_break_slots_per_participant(self) -> int:
        """Maximum possible idle slots strictly between first and last meetings."""
        return max(0, self.n_total_slots - 2)


@dataclass(frozen=True)
class PrecedenceGraphInfo:
    direct_predecessors: list[set[int]]
    transitive_predecessors: list[set[int]]
    longest_distance: list[dict[int, int]]
    successors: list[set[int]]
    cycle_nodes: tuple[int, ...]
    direct_edge_count: int
    transitive_edge_count: int
    max_chain_distance: int


@dataclass(frozen=True)
class B2BSolutionStats:
    """Schedule statistics for the internal-idle-slot range over P*.

    ``participant_breaks[p]`` is retained for backward compatibility, but now
    stores the total number of idle time slots between consecutive meetings of
    participant p, not the number of contiguous break groups. The objective
    gap is evaluated only over ``P* = {p : |M_p| >= 2}``.
    """

    total_breaks: int
    participant_breaks: list[int]
    objective_gap: int
    all_participant_idle_range: int
    objective_participants: tuple[int, ...]
    meetings_per_slot: list[list[int]]
    busy_participants_per_slot: list[int]

    @property
    def total_internal_idle_slots(self) -> int:
        """Sum of internal idle slots over all participants."""
        return self.total_breaks

    @property
    def participant_internal_idle_slots(self) -> list[int]:
        """Internal idle-slot count B(p) for every participant."""
        return self.participant_breaks

    @property
    def idle_range(self) -> int:
        """Range of internal-idle totals over P*."""
        return self.objective_gap

    @property
    def objective_participant_ids(self) -> tuple[int, ...]:
        """One-based participant IDs in P*."""
        return tuple(participant + 1 for participant in self.objective_participants)

    # Backward-compatible names used by the existing benchmark runner.
    @property
    def total_break_slots(self) -> int:
        return self.total_internal_idle_slots

    @property
    def participant_break_slots(self) -> list[int]:
        return self.participant_internal_idle_slots

@dataclass(frozen=True)
class B2BModelArtifacts:
    cnf: CNF
    objective_lits: list[int]
    objective_name: str
    objective_participants: tuple[int, ...]
    objective_gap_lits: list[int]
    hole_lits_by_participant: list[list[int]]
    sorted_hole_lits_by_participant: list[list[int]]
    n_vars: int
    n_clauses: int
    n_primary_variables: int
    n_auxiliary_variables: int
    n_hard_literals: int
    max_hard_clause_length: int
    n_unit_hard_clauses: int
    n_binary_hard_clauses: int
    n_ternary_hard_clauses: int
    n_long_hard_clauses: int
    encoding_variant: str
    precedence_mode: str
    precedence_encoding: str
    precedence_graph: str
    precedence_configuration: str
    domain_mode: str
    enabled_constraints: list[str]
    full_schedule_candidates: int
    unary_eligible_schedule_candidates: int
    initial_schedule_candidates: int
    reduced_schedule_candidates: int
    active_schedule_candidates: int
    unary_removed_schedule_candidates: int
    preprocessing_removed_schedule_candidates: int
    removed_schedule_candidates: int
    precedence_direct_edges: int
    precedence_transitive_edges: int
    precedence_cycle_nodes: tuple[int, ...]
    precedence_max_distance: int
    precedence_relation_edges: int
    precedence_pairwise_clauses: int
    precedence_sparse_link_clauses: int
    precedence_unique_suffix_cuts: int
    objective_encoding: str

# ---------------------------------------------------------------------------
# MiniZinc .dzn parser
# ---------------------------------------------------------------------------


def _remove_comments(text: str) -> str:
    return "\n".join(
        line
        for raw in text.splitlines()
        if (line := raw.split("%", 1)[0].strip())
    )


def _extract_int(text: str, name: str) -> int:
    match = re.search(rf"\b{name}\s*=\s*(-?\d+)\s*;", text)
    if not match:
        raise ValueError(f"Cannot find integer field {name!r}")
    return int(match.group(1))


def _extract_block(text: str, name: str, open_char: str, close_char: str) -> str:
    match = re.search(rf"\b{name}\s*=\s*{re.escape(open_char)}", text)
    if not match:
        raise ValueError(f"Cannot find block field {name!r}")

    start = match.end()
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    raise ValueError(f"Cannot find end of block field {name!r}")


def _extract_int_list(block: str) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", block)]


def _extract_set_array(block: str) -> list[set[int]]:
    result: list[set[int]] = []
    i = 0
    while i < len(block):
        if block[i] != "{":
            i += 1
            continue
        start = i + 1
        depth = 1
        i += 1
        while i < len(block) and depth:
            if block[i] == "{":
                depth += 1
            elif block[i] == "}":
                depth -= 1
                if depth == 0:
                    result.append({int(x) for x in re.findall(r"-?\d+", block[start:i])})
                    break
            i += 1
        i += 1
    return result


def read_instance(path: str | Path, *, validate_meetingsx_business: bool = True) -> B2BInstance:
    """Read one original B2B .dzn file and convert all indices to zero-based."""

    path = Path(path)
    text = _remove_comments(path.read_text(encoding="utf-8"))

    n_business = _extract_int(text, "nBusiness")
    n_meetings = _extract_int(text, "nMeetings")
    n_tables = _extract_int(text, "nTables")
    n_total_slots = _extract_int(text, "nTotalSlots")
    n_morning_slots = _extract_int(text, "nMorningSlots")

    if n_business < 0 or n_meetings < 0 or n_tables < 0 or n_total_slots < 0:
        raise ValueError("Instance sizes must be non-negative")
    if not 0 <= n_morning_slots <= n_total_slots:
        raise ValueError("nMorningSlots must be between 0 and nTotalSlots")

    requested_nums = _extract_int_list(_extract_block(text, "requested", "[", "]"))
    if len(requested_nums) != 3 * n_meetings:
        raise ValueError(
            f"requested must contain 3*nMeetings values; got {len(requested_nums)} "
            f"for nMeetings={n_meetings}"
        )

    requested: list[tuple[int, int, int]] = []
    for i in range(0, len(requested_nums), 3):
        p1 = requested_nums[i] - 1
        p2 = requested_nums[i + 1] - 1
        session = requested_nums[i + 2]
        m = i // 3
        if not (0 <= p1 < n_business and 0 <= p2 < n_business):
            raise ValueError(f"Meeting {m + 1} contains an invalid participant")
        if p1 == p2:
            raise ValueError(f"Meeting {m + 1} contains the same participant twice")
        if session not in {1, 2, 3}:
            raise ValueError(f"Meeting {m + 1} has invalid session code {session}")
        requested.append((p1, p2, session))

    meetingsx_business_raw = _extract_set_array(
        _extract_block(text, "meetingsxBusiness", "[", "]")
    )
    n_meetings_business = _extract_int_list(
        _extract_block(text, "nMeetingsBusiness", "[", "]")
    )
    forbidden_raw = _extract_set_array(_extract_block(text, "forbidden", "[", "]"))
    fixed_raw = _extract_int_list(_extract_block(text, "fixed", "[", "]"))
    precedences_raw = _extract_set_array(_extract_block(text, "precedences", "[", "]"))

    if len(n_meetings_business) != n_business:
        raise ValueError("nMeetingsBusiness must have nBusiness integers")
    if len(forbidden_raw) != n_business:
        raise ValueError("forbidden must have nBusiness sets")
    if len(fixed_raw) != n_meetings:
        raise ValueError("fixed must have nMeetings integers")
    if len(precedences_raw) != n_meetings:
        raise ValueError("precedences must have nMeetings sets")

    forbidden: list[set[int]] = []
    for p, slots in enumerate(forbidden_raw):
        converted = {slot - 1 for slot in slots if slot > 0}
        invalid = [slot + 1 for slot in converted if not 0 <= slot < n_total_slots]
        if invalid:
            raise ValueError(f"Participant {p + 1} has invalid forbidden slots {invalid}")
        forbidden.append(converted)

    fixed: list[int | None] = []
    for m, slot in enumerate(fixed_raw):
        value = None if slot == 0 else slot - 1
        if value is not None and not 0 <= value < n_total_slots:
            raise ValueError(f"Meeting {m + 1} has invalid fixed slot {slot}")
        fixed.append(value)

    precedences: list[set[int]] = []
    for post, preds in enumerate(precedences_raw):
        converted = {pred - 1 for pred in preds if pred > 0}
        invalid = [pred + 1 for pred in converted if not 0 <= pred < n_meetings]
        if invalid:
            raise ValueError(f"Meeting {post + 1} has invalid predecessors {invalid}")
        precedences.append(converted)

    meetings_by_business: list[list[int]] = [[] for _ in range(n_business)]
    for m, (p1, p2, _) in enumerate(requested):
        meetings_by_business[p1].append(m)
        meetings_by_business[p2].append(m)

    for p, meetings in enumerate(meetings_by_business):
        if len(meetings) != n_meetings_business[p]:
            raise ValueError(
                f"Participant {p + 1}: derived {len(meetings)} meetings but "
                f"nMeetingsBusiness says {n_meetings_business[p]}"
            )

    if validate_meetingsx_business:
        if len(meetingsx_business_raw) != n_business:
            raise ValueError("meetingsxBusiness must have nBusiness sets")
        for p, meetings in enumerate(meetings_by_business):
            # Original data uses dummy value 1 and shifts meeting ids by +1.
            expected = {1} | {m + 2 for m in meetings}
            if meetingsx_business_raw[p] != expected:
                raise ValueError(
                    f"Participant {p + 1}: meetingsxBusiness mismatch; "
                    f"expected {sorted(expected)}, got {sorted(meetingsx_business_raw[p])}"
                )

    return B2BInstance(
        n_business=n_business,
        n_meetings=n_meetings,
        n_tables=n_tables,
        n_total_slots=n_total_slots,
        n_morning_slots=n_morning_slots,
        requested=requested,
        meetings_by_business=meetings_by_business,
        n_meetings_business=n_meetings_business,
        forbidden=forbidden,
        fixed=fixed,
        precedences=precedences,
        instance_name=path.name,
    )


# ---------------------------------------------------------------------------
# Precedence graph and optimized domain reduction
# ---------------------------------------------------------------------------


def build_precedence_graph(precedences: list[set[int]]) -> PrecedenceGraphInfo:
    """Build E*, detect cycles, and compute longest precedence-chain distances.

    ``longest_distance[post][pred] = d`` means that the longest directed path
    from ``pred`` to ``post`` contains ``d`` strict-precedence edges. Therefore
    every feasible schedule satisfies ``time(pred) + d <= time(post)``.
    """

    n = len(precedences)
    direct = [set(preds) for preds in precedences]
    closure = [set(preds) for preds in precedences]

    for k in range(n):
        for post in range(n):
            if k in closure[post]:
                closure[post].update(closure[k])

    successors = [set() for _ in range(n)]
    indegree = [0] * n
    for post, preds in enumerate(direct):
        indegree[post] = len(preds)
        for pred in preds:
            successors[pred].add(post)

    cycle_nodes = tuple(v for v in range(n) if v in closure[v])
    longest_distance: list[dict[int, int]] = [dict() for _ in range(n)]

    if not cycle_nodes:
        ready = [v for v, deg in enumerate(indegree) if deg == 0]
        topo: list[int] = []
        cursor = 0
        while cursor < len(ready):
            node = ready[cursor]
            cursor += 1
            topo.append(node)
            for post in successors[node]:
                indegree[post] -= 1
                if indegree[post] == 0:
                    ready.append(post)

        for post in topo:
            distances = longest_distance[post]
            for pred in direct[post]:
                distances[pred] = max(distances.get(pred, 0), 1)
                for ancestor, distance in longest_distance[pred].items():
                    distances[ancestor] = max(
                        distances.get(ancestor, 0), distance + 1
                    )

    return PrecedenceGraphInfo(
        direct_predecessors=direct,
        transitive_predecessors=closure,
        longest_distance=longest_distance,
        successors=successors,
        cycle_nodes=cycle_nodes,
        direct_edge_count=sum(map(len, direct)),
        transitive_edge_count=sum(map(len, closure)),
        max_chain_distance=max(
            (distance for row in longest_distance for distance in row.values()),
            default=0,
        ),
    )


def _topological_order(graph: PrecedenceGraphInfo) -> list[int]:
    """Return a topological order of an acyclic precedence graph."""

    indegree = [len(preds) for preds in graph.direct_predecessors]
    ready = [node for node, degree in enumerate(indegree) if degree == 0]
    order: list[int] = []
    cursor = 0
    while cursor < len(ready):
        node = ready[cursor]
        cursor += 1
        order.append(node)
        for successor in graph.successors[node]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    return order


def resource_strengthened_precedence_distances(
    inst: B2BInstance,
    graph: PrecedenceGraphInfo,
) -> list[dict[int, int]]:
    """Return exact resource-strengthened lower bounds for every edge in E*.

    For every reachable pair ``pred ->* post``, all meetings that are strict
    descendants of ``pred`` and strict ancestors of ``post`` must be scheduled
    in the open interval between the two endpoints. The interval therefore
    needs enough slots for both the table capacity and every participant's
    capacity-one meetings. These valid lower bounds are then closed under
    max-plus path composition on the precedence DAG.

    The returned constraints are implied by the original B2B semantics; using
    them for domain filtering cannot remove a feasible schedule.
    """

    chain = [dict(distances) for distances in graph.longest_distance]
    if graph.cycle_nodes or graph.transitive_edge_count == 0:
        return chain

    n = inst.n_meetings
    descendants = [set() for _ in range(n)]
    for post, predecessors in enumerate(graph.transitive_predecessors):
        for predecessor in predecessors:
            descendants[predecessor].add(post)

    strengthened = [dict(distances) for distances in chain]
    for post, predecessors in enumerate(graph.transitive_predecessors):
        ancestors_of_post = graph.transitive_predecessors[post]
        for predecessor in predecessors:
            interior = descendants[predecessor].intersection(ancestors_of_post)

            table_layers = 0
            if interior and inst.n_tables > 0:
                table_layers = (
                    len(interior) + inst.n_tables - 1
                ) // inst.n_tables

            participant_loads: dict[int, int] = {}
            for meeting in interior:
                p1, p2, _ = inst.requested[meeting]
                participant_loads[p1] = participant_loads.get(p1, 0) + 1
                participant_loads[p2] = participant_loads.get(p2, 0) + 1
            participant_layers = max(participant_loads.values(), default=0)

            resource_lag = 1 + max(table_layers, participant_layers)
            strengthened[post][predecessor] = max(
                strengthened[post].get(predecessor, 0),
                resource_lag,
            )

    # Every strengthened reachable pair is a valid weighted shortcut. Compose
    # all shortcuts through longest paths on the DAG (max-plus closure).
    weighted_successors: list[dict[int, int]] = [dict() for _ in range(n)]
    for post, distances in enumerate(strengthened):
        for predecessor, distance in distances.items():
            weighted_successors[predecessor][post] = max(
                weighted_successors[predecessor].get(post, 0),
                distance,
            )

    order = _topological_order(graph)
    closed: list[dict[int, int]] = [dict() for _ in range(n)]
    for source in order:
        distance_from_source: dict[int, int] = {source: 0}
        for node in order:
            base = distance_from_source.get(node)
            if base is None:
                continue
            for successor, edge_weight in weighted_successors[node].items():
                candidate = base + edge_weight
                if candidate > distance_from_source.get(successor, -1):
                    distance_from_source[successor] = candidate
        for post, distance in distance_from_source.items():
            if post != source:
                closed[post][source] = distance
    return closed


def _session_slots(inst: B2BInstance, m: int) -> set[int]:
    session = inst.requested[m][2]
    if session == 1:
        return set(inst.morning_slots)
    if session == 2:
        return set(inst.afternoon_slots)
    return set(range(inst.n_total_slots))


def original_eligible_slots(inst: B2BInstance, m: int) -> set[int]:
    """Unary domain from session, fixed meeting, and participant availability."""

    p1, p2, _ = inst.requested[m]
    slots = _session_slots(inst, m)
    fixed = inst.fixed[m]
    if fixed is not None:
        slots.intersection_update({fixed})
    slots.difference_update(inst.forbidden[p1])
    slots.difference_update(inst.forbidden[p2])
    return slots


def _has_full_slot_matching(
    meetings: list[int],
    domains: list[set[int]],
    *,
    fixed_meeting: int | None = None,
    fixed_slot: int | None = None,
) -> bool:
    """Check participant all-different feasibility with one optional fixed edge."""

    if fixed_meeting is not None:
        if fixed_slot is None or fixed_slot not in domains[fixed_meeting]:
            return False
        blocked = {fixed_slot}
        remaining = [m for m in meetings if m != fixed_meeting]
    else:
        blocked = set()
        remaining = list(meetings)

    remaining.sort(key=lambda m: len(domains[m] - blocked))
    slot_owner: dict[int, int] = {}

    def augment(meeting: int, seen: set[int]) -> bool:
        for slot in sorted(domains[meeting]):
            if slot in blocked or slot in seen:
                continue
            seen.add(slot)
            owner = slot_owner.get(slot)
            if owner is None or augment(owner, seen):
                slot_owner[slot] = meeting
                return True
        return False

    return all(augment(meeting, set()) for meeting in remaining)


def _propagate_distance_precedences(
    domains: list[set[int]],
    distances_by_post: list[dict[int, int]],
) -> bool:
    """Apply bounds consistency to every distance-labelled edge in E*."""

    changed = False
    for post, distances in enumerate(distances_by_post):
        for pred, distance in distances.items():
            pred_domain = domains[pred]
            post_domain = domains[post]
            if not pred_domain or not post_domain:
                continue

            latest_pred = max(post_domain) - distance
            new_pred = {slot for slot in pred_domain if slot <= latest_pred}
            if new_pred != pred_domain:
                domains[pred] = pred_domain = new_pred
                changed = True
                if not pred_domain:
                    continue

            earliest_post = min(pred_domain) + distance
            new_post = {slot for slot in post_domain if slot >= earliest_post}
            if new_post != post_domain:
                domains[post] = new_post
                changed = True
    return changed


def _propagate_participant_matchings(
    inst: B2BInstance,
    domains: list[set[int]],
) -> tuple[bool, bool]:
    """Enforce GAC on every participant's all-different slot constraint.

    Returns ``(changed, feasible)``. Candidate ``meeting@slot`` is removed when
    fixing that edge prevents a full matching for the remaining meetings of the
    participant. This preprocessing is exact and does not alter the solution set.
    """

    changed = False
    for meetings in inst.meetings_by_business:
        if len(meetings) <= 1:
            continue
        if any(not domains[m] for m in meetings):
            return changed, False
        if not _has_full_slot_matching(meetings, domains):
            domains[meetings[0]].clear()
            return True, False

        for meeting in meetings:
            if len(domains[meeting]) <= 1:
                continue
            for slot in tuple(sorted(domains[meeting])):
                if not _has_full_slot_matching(
                    meetings,
                    domains,
                    fixed_meeting=meeting,
                    fixed_slot=slot,
                ):
                    domains[meeting].remove(slot)
                    changed = True
                    if not domains[meeting]:
                        return changed, False
    return changed, True


def _propagate_saturated_table_slots(
    inst: B2BInstance,
    domains: list[set[int]],
) -> tuple[bool, bool]:
    """Propagate slots already saturated by fixed/singleton meetings."""

    changed = False
    for slot in range(inst.n_total_slots):
        forced = [m for m, domain in enumerate(domains) if domain == {slot}]
        if len(forced) > inst.n_tables:
            domains[forced[0]].clear()
            return True, False
        if len(forced) != inst.n_tables:
            continue
        for meeting, domain in enumerate(domains):
            if len(domain) > 1 and slot in domain:
                domain.remove(slot)
                changed = True
                if not domain:
                    return changed, False
    return changed, True


def reduce_domains_with_precedence(
    inst: B2BInstance,
    graph: PrecedenceGraphInfo,
) -> tuple[list[list[int]], int, int]:
    """Run exact preprocessing to a fixpoint.

    The fixpoint combines unary restrictions, resource-strengthened E*
    propagation, participant all-different matching GAC, and saturated
    table-slot propagation.
    """

    domains = [original_eligible_slots(inst, m) for m in range(inst.n_meetings)]
    initial_count = sum(len(domain) for domain in domains)

    if graph.cycle_nodes:
        domains[graph.cycle_nodes[0]].clear()
        return [sorted(domain) for domain in domains], initial_count, sum(map(len, domains))

    effective_distances = resource_strengthened_precedence_distances(inst, graph)

    changed = True
    while changed:
        changed = _propagate_distance_precedences(domains, effective_distances)
        matching_changed, feasible = _propagate_participant_matchings(inst, domains)
        changed = changed or matching_changed
        if not feasible:
            break
        capacity_changed, feasible = _propagate_saturated_table_slots(inst, domains)
        changed = changed or capacity_changed
        if not feasible:
            break

    reduced_count = sum(len(domain) for domain in domains)
    return [sorted(domain) for domain in domains], initial_count, reduced_count


def objective_participants(inst: B2BInstance) -> tuple[int, ...]:
    """Return P* = {p: participant p attends at least two meetings}."""

    return tuple(
        participant
        for participant, meetings in enumerate(inst.meetings_by_business)
        if len(meetings) >= 2
    )


def compute_solution_stats(
    inst: B2BInstance,
    assignment: list[int],
    *,
    participants: tuple[int, ...] | None = None,
) -> B2BSolutionStats:
    """Compute IdleRange(P*) statistics independently of any solver model."""

    selected_participants = (
        objective_participants(inst) if participants is None else participants
    )
    meetings_per_slot: list[list[int]] = [
        [] for _ in range(inst.n_total_slots)
    ]
    for meeting, slot in enumerate(assignment):
        if 0 <= slot < inst.n_total_slots:
            meetings_per_slot[slot].append(meeting)

    participant_idle = [0] * inst.n_business
    for participant, meetings in enumerate(inst.meetings_by_business):
        slots = sorted(
            assignment[meeting]
            for meeting in meetings
            if 0 <= meeting < len(assignment) and assignment[meeting] >= 0
        )
        if len(slots) >= 2:
            participant_idle[participant] = (
                slots[-1] - slots[0] + 1 - len(slots)
            )

    objective_values = [
        participant_idle[participant]
        for participant in selected_participants
    ]
    objective_gap = (
        max(objective_values) - min(objective_values)
        if len(objective_values) >= 2
        else 0
    )
    all_participant_gap = (
        max(participant_idle) - min(participant_idle)
        if len(participant_idle) >= 2
        else 0
    )
    return B2BSolutionStats(
        total_breaks=sum(participant_idle),
        participant_breaks=participant_idle,
        objective_gap=objective_gap,
        all_participant_idle_range=all_participant_gap,
        objective_participants=selected_participants,
        meetings_per_slot=meetings_per_slot,
        busy_participants_per_slot=[
            2 * len(meetings) for meetings in meetings_per_slot
        ],
    )


def validate_schedule_assignment(
    inst: B2BInstance,
    assignment: list[int],
    *,
    graph: PrecedenceGraphInfo | None = None,
) -> list[str]:
    """Check a schedule against the original hard B2B semantics."""

    errors: list[str] = []
    if len(assignment) != inst.n_meetings:
        return ["assignment length does not match nMeetings"]

    for meeting, slot in enumerate(assignment):
        if slot not in original_eligible_slots(inst, meeting):
            errors.append(
                f"meeting {meeting + 1} assigned to an ineligible slot "
                f"{slot + 1 if slot >= 0 else slot}"
            )

    for participant, meetings in enumerate(inst.meetings_by_business):
        seen: dict[int, int] = {}
        for meeting in meetings:
            slot = assignment[meeting]
            if slot in seen:
                errors.append(
                    f"participant {participant + 1} collision at slot {slot + 1}: "
                    f"meetings {seen[slot] + 1} and {meeting + 1}"
                )
            seen[slot] = meeting

    for slot in range(inst.n_total_slots):
        count = sum(assigned == slot for assigned in assignment)
        if count > inst.n_tables:
            errors.append(
                f"capacity exceeded at slot {slot + 1}: "
                f"{count}>{inst.n_tables}"
            )

    precedence_graph = graph or build_precedence_graph(inst.precedences)
    for post, preds in enumerate(precedence_graph.direct_predecessors):
        for pred in preds:
            if assignment[pred] >= assignment[post]:
                errors.append(
                    f"precedence violation: meeting {pred + 1} "
                    f"!< meeting {post + 1}"
                )
    return errors


def resolve_precedence_configuration(
    precedence_mode: str | None,
    precedence_encoding: str | None,
    precedence_graph: str | None,
    *,
    default_mode: PrecedenceMode,
) -> tuple[str, str, str]:
    """Resolve independent P/G flags and the deprecated composite mode.

    A caller must either provide both independent flags or neither. The legacy
    modes remain accepted for API compatibility, but conflicting legacy and
    independent values are rejected instead of being silently overridden.
    """

    if precedence_mode is not None:
        if precedence_mode not in VALID_PRECEDENCE_MODES:
            raise ValueError(f"Unknown precedence_mode={precedence_mode!r}")
        legacy_encoding, legacy_graph = LEGACY_PRECEDENCE_CONFIGURATIONS[
            precedence_mode
        ]
        if (
            precedence_encoding is not None
            and precedence_encoding != legacy_encoding
        ):
            raise ValueError(
                "precedence_mode conflicts with precedence_encoding: "
                f"{precedence_mode!r} requires {legacy_encoding!r}"
            )
        if precedence_graph is not None and precedence_graph != legacy_graph:
            raise ValueError(
                "precedence_mode conflicts with precedence_graph: "
                f"{precedence_mode!r} requires {legacy_graph!r}"
            )
        precedence_encoding = legacy_encoding
        precedence_graph = legacy_graph
    elif precedence_encoding is None and precedence_graph is None:
        precedence_mode = default_mode
        precedence_encoding, precedence_graph = (
            LEGACY_PRECEDENCE_CONFIGURATIONS[default_mode]
        )
    elif precedence_encoding is None or precedence_graph is None:
        raise ValueError(
            "precedence_encoding and precedence_graph must be specified together"
        )

    if precedence_encoding not in VALID_PRECEDENCE_ENCODINGS:
        raise ValueError(
            f"Unknown precedence_encoding={precedence_encoding!r}"
        )
    if precedence_graph not in VALID_PRECEDENCE_GRAPHS:
        raise ValueError(f"Unknown precedence_graph={precedence_graph!r}")

    legacy_mode = next(
        (
            mode
            for mode, configuration in LEGACY_PRECEDENCE_CONFIGURATIONS.items()
            if configuration == (precedence_encoding, precedence_graph)
        ),
        "factorial",
    )
    return precedence_encoding, precedence_graph, legacy_mode


# ---------------------------------------------------------------------------
# Combined optimized MaxSAT/SAT encoding
# ---------------------------------------------------------------------------


class B2BSATModel:
    """Optimized exact encoder for the max/min internal-idle-slot gap over P*.

    The only optimization objective is

        minimize max_{p in P*} B(p) - min_{p in P*} B(p),

    where ``B(p)`` is the total number of empty slots strictly between the first
    and last meetings of participant ``p``, and
    ``P* = {p : |M_p| >= 2}``. Participants outside P* always have B(p)=0 and
    are excluded from the range. No hard upper bound on this gap is generated.

    The key identity used by the objective encoding is

        B(p) = last_slot(p) - first_slot(p) + 1 - |M_p|.

    This replaces per-idle-slot variables and quadratic sorting networks with a
    linear prefix/suffix span encoding and exact unary threshold literals.

    ``precedence_encoding`` (P) and ``precedence_graph`` (G) are independent:
    either encoding can consume either the direct distance-one relations or the
    distance-labelled transitive closure. Domain preprocessing intentionally
    uses the same exact resource-strengthened E* bounds for every P/G
    configuration, while the encoded precedence graph remains unchanged.

    ``domain_mode="full"`` creates the complete meeting-slot Cartesian product
    and encodes unary input restrictions as hard clauses. ``"reduced"`` omits
    variables removed by the exact preprocessing fixpoint. Every other encoding
    component follows the same code path in both modes.
    """

    def __init__(
        self,
        inst: B2BInstance,
        precedence_mode: PrecedenceMode | None = None,
        encoding_variant: EncodingVariant = "imp12+",
        domain_mode: DomainMode = "reduced",
        *,
        precedence_encoding: PrecedenceEncoding | None = None,
        precedence_graph: PrecedenceGraph | None = None,
    ) -> None:
        if encoding_variant not in VALID_ENCODING_VARIANTS:
            raise ValueError(f"Unknown encoding_variant={encoding_variant!r}")
        if domain_mode not in VALID_DOMAIN_MODES:
            raise ValueError(f"Unknown domain_mode={domain_mode!r}")

        (
            resolved_precedence_encoding,
            resolved_precedence_graph,
            resolved_precedence_mode,
        ) = resolve_precedence_configuration(
            precedence_mode,
            precedence_encoding,
            precedence_graph,
            default_mode="staircase",
        )

        self.inst = inst
        self.precedence_mode = resolved_precedence_mode
        self.precedence_encoding = resolved_precedence_encoding
        self.precedence_graph = resolved_precedence_graph
        self.precedence_configuration = (
            f"{self.precedence_encoding}+{self.precedence_graph}"
        )
        self.encoding_variant = encoding_variant
        self.domain_mode = domain_mode
        self.objective_participants = tuple(
            participant
            for participant, meeting_count in enumerate(inst.n_meetings_business)
            if meeting_count >= 2
        )
        self.graph = build_precedence_graph(inst.precedences)
        if self.precedence_graph == "direct":
            self._precedence_distances = [
                {pred: 1 for pred in predecessors}
                for predecessors in self.graph.direct_predecessors
            ]
        else:
            self._precedence_distances = [
                dict(distances) for distances in self.graph.longest_distance
            ]
        self._unary_eligible_slots = [
            sorted(original_eligible_slots(inst, meeting))
            for meeting in range(inst.n_meetings)
        ]
        (
            self._reduced_slots,
            reduction_initial_count,
            self.reduced_schedule_candidates,
        ) = reduce_domains_with_precedence(inst, self.graph)
        self.full_schedule_candidates = inst.n_meetings * inst.n_total_slots
        self.unary_eligible_schedule_candidates = sum(
            len(slots) for slots in self._unary_eligible_slots
        )
        if reduction_initial_count != self.unary_eligible_schedule_candidates:
            raise AssertionError("domain-reduction input disagrees with unary domains")

        # Backward-compatible name: historically this field counted candidates
        # after session/fixed/forbidden filtering, before exact propagation.
        self.initial_schedule_candidates = self.unary_eligible_schedule_candidates
        if domain_mode == "full":
            all_slots = list(range(inst.n_total_slots))
            self._eligible_slots = [list(all_slots) for _ in range(inst.n_meetings)]
        else:
            self._eligible_slots = [list(slots) for slots in self._reduced_slots]
        self.active_schedule_candidates = sum(
            len(slots) for slots in self._eligible_slots
        )

        self.vpool = IDPool()
        self.enabled_constraints: list[str] = []
        self._clusters: list[list[int]] | None = None
        self._artifacts: B2BModelArtifacts | None = None
        self._precedence_sparse_suffixes: dict[int, dict[int, int]] = {}
        self._precedence_pairwise_clauses = 0
        self._precedence_sparse_link_clauses = 0
        self._precedence_unique_suffix_cuts = 0
        self._participant_occupancy_strategy: dict[int, str] = {}
        self._objective_prefix_support: dict[tuple[int, int], int] = {}
        self._objective_suffix_support: dict[tuple[int, int], int] = {}

        self._schedule_vars: dict[tuple[int, int], int] = {}
        for meeting, slots in enumerate(self._eligible_slots):
            for slot in slots:
                self._schedule_vars[meeting, slot] = self.vpool.id(
                    ("schedule", meeting, slot)
                )

        self._used_vars: dict[tuple[int, int], int] = {}
        for participant, meetings in enumerate(inst.meetings_by_business):
            possible_slots = {
                slot
                for meeting in meetings
                for slot in self._eligible_slots[meeting]
            }
            for slot in sorted(possible_slots):
                self._used_vars[participant, slot] = self.vpool.id(
                    ("usedSlot", participant, slot)
                )

    # Public variable/domain helpers -----------------------------------

    def eligible_slots(self, m: int) -> list[int]:
        """Return slots for which schedule variables exist in the active mode."""

        return list(self._eligible_slots[m])

    def unary_eligible_slots(self, m: int) -> list[int]:
        """Return slots satisfying session, fixed, and forbidden restrictions."""

        return list(self._unary_eligible_slots[m])

    def reduced_slots(self, m: int) -> list[int]:
        """Return the exact preprocessing fixpoint independently of active mode."""

        return list(self._reduced_slots[m])

    def x(self, m: int, t: int) -> int:
        try:
            return self._schedule_vars[m, t]
        except KeyError as exc:
            raise KeyError(
                f"schedule({m},{t}) does not exist in {self.domain_mode} mode"
            ) from exc

    def x_or_none(self, m: int, t: int) -> int | None:
        return self._schedule_vars.get((m, t))

    def used_or_none(self, p: int, t: int) -> int | None:
        return self._used_vars.get((p, t))

    def prefix_used(self, p: int, t: int) -> int:
        return self.vpool.id(("prefixUsed", p, t))

    def suffix_used(self, p: int, t: int) -> int:
        return self.vpool.id(("suffixUsed", p, t))

    def first_used(self, p: int, t: int) -> int:
        return self.vpool.id(("firstUsed", p, t))

    def break_threshold(self, p: int, k: int) -> int:
        return self.vpool.id(("breakSlotsAtLeast", p, k))

    # Backward-compatible accessors.
    def meeting_before(self, p: int, t: int) -> int:
        if t <= 0:
            raise IndexError("there is no slot strictly before slot 0")
        return self.prefix_used(p, t - 1)

    def meeting_after(self, p: int, t: int) -> int:
        if t >= self.inst.n_total_slots - 1:
            raise IndexError("there is no slot strictly after the final slot")
        return self.suffix_used(p, t + 1)

    def hole(self, p: int, t: int) -> int:
        return self.vpool.id(("legacyBreakSlot", p, t))

    def sorted_hole(self, p: int, k: int) -> int:
        return self.break_threshold(p, k)

    def max_break(self, k: int) -> int:
        return self.vpool.id(("maxBreakSlots", k))

    def min_break(self, k: int) -> int:
        return self.vpool.id(("minBreakSlots", k))

    def dif_break(self, k: int) -> int:
        return self.vpool.id(("difBreakSlots", k))

    def cluster_active(self, c: int, t: int) -> int:
        return self.vpool.id(("clusterActive", c, t))

    @property
    def use_implied_1(self) -> bool:
        return self.encoding_variant in {"imp1", "imp12", "imp12+"}

    @property
    def use_implied_2(self) -> bool:
        return self.encoding_variant in {"imp2", "imp12", "imp12+"}

    @property
    def use_further_improvements(self) -> bool:
        return self.encoding_variant == "imp12+"

    # Cardinality helpers ----------------------------------------------

    @staticmethod
    def _add_pairwise_atmost_one(cnf: CNF, lits: list[int]) -> None:
        for i, left in enumerate(lits):
            for right in lits[i + 1:]:
                cnf.append([-left, -right])

    def _add_exactly_one_commander(
        self,
        cnf: CNF,
        lits: list[int],
        group_size: int = 4,
    ) -> None:
        if not lits:
            cnf.append([])
            return
        if len(lits) == 1:
            cnf.append([lits[0]])
            return
        if len(lits) <= group_size:
            cnf.append(list(lits))
            self._add_pairwise_atmost_one(cnf, lits)
            return

        commanders: list[int] = []
        for start in range(0, len(lits), group_size):
            group = lits[start:start + group_size]
            commander = self.vpool.id(("commander", tuple(group)))
            commanders.append(commander)
            self._add_pairwise_atmost_one(cnf, group)
            for lit in group:
                cnf.append([-lit, commander])
            cnf.append([-commander] + group)
        self._add_exactly_one_commander(cnf, commanders, group_size)

    def _add_atmost_seqcounter(self, cnf: CNF, lits: list[int], bound: int) -> None:
        if bound < 0:
            cnf.append([])
        elif bound == 0:
            cnf.extend([[-lit] for lit in lits])
        elif bound < len(lits):
            encoding = CardEnc.atmost(
                lits=lits,
                bound=bound,
                vpool=self.vpool,
                encoding=EncType.seqcounter,
            )
            cnf.extend(encoding.clauses)

    def _add_exactly_cardnet(self, cnf: CNF, lits: list[int], bound: int) -> None:
        if bound < 0 or bound > len(lits):
            cnf.append([])
        elif bound == 0:
            cnf.extend([[-lit] for lit in lits])
        elif bound == len(lits):
            cnf.extend([[lit] for lit in lits])
        else:
            encoding = CardEnc.equals(
                lits=lits,
                bound=bound,
                vpool=self.vpool,
                encoding=EncType.cardnetwrk,
            )
            cnf.extend(encoding.clauses)

    def _add_atmost_cardnet(self, cnf: CNF, lits: list[int], bound: int) -> None:
        if bound < 0:
            cnf.append([])
        elif bound == 0:
            cnf.extend([[-lit] for lit in lits])
        elif bound < len(lits):
            encoding = CardEnc.atmost(
                lits=lits,
                bound=bound,
                vpool=self.vpool,
                encoding=EncType.cardnetwrk,
            )
            cnf.extend(encoding.clauses)

    # Build entry points ------------------------------------------------

    def build_base_cnf(self) -> B2BModelArtifacts:
        if self._artifacts is not None:
            return self._artifacts

        cnf = CNF()
        self.enabled_constraints = [
            "objective-only optimization: no hard upper bound on objective gap",
            "P/G-independent resource-strengthened E* domain propagation and "
            "cycle detection",
            "precedence configuration: "
            f"P={self.precedence_encoding}, G={self.precedence_graph}",
        ]
        if self.domain_mode == "full":
            self.enabled_constraints.append(
                "Full Domain MxT schedule variables + explicit unary exclusions"
            )
        else:
            self.enabled_constraints.append(
                "Reduced Domain variables after unary filtering + matching GAC + "
                "slot saturation"
            )

        if self.graph.cycle_nodes:
            cnf.append([])
            self.enabled_constraints.append("strict precedence cycle -> UNSAT")

        self._add_assignment(cnf)
        self._add_participant_collision(cnf)
        if self.use_further_improvements:
            self._add_cluster_capacity(cnf)
        else:
            self._add_capacity_over_meetings(cnf)
        self._add_precedences(cnf)

        threshold_lits = self._add_span_break_thresholds(cnf)
        gap_lits = self._add_gap_objective(cnf, threshold_lits)

        n_vars = max(self.vpool.top, cnf.nv)
        n_primary_variables = len(self._schedule_vars)
        clause_lengths = [len(clause) for clause in cnf.clauses]

        self._artifacts = B2BModelArtifacts(
            cnf=cnf,
            objective_lits=gap_lits,
            objective_name="internal_idle_slot_range_pstar",
            objective_participants=self.objective_participants,
            objective_gap_lits=gap_lits,
            hole_lits_by_participant=[[] for _ in range(self.inst.n_business)],
            sorted_hole_lits_by_participant=threshold_lits,
            n_vars=n_vars,
            n_clauses=len(cnf.clauses),
            n_primary_variables=n_primary_variables,
            n_auxiliary_variables=n_vars - n_primary_variables,
            n_hard_literals=sum(clause_lengths),
            max_hard_clause_length=max(clause_lengths, default=0),
            n_unit_hard_clauses=sum(length == 1 for length in clause_lengths),
            n_binary_hard_clauses=sum(length == 2 for length in clause_lengths),
            n_ternary_hard_clauses=sum(length == 3 for length in clause_lengths),
            n_long_hard_clauses=sum(length >= 4 for length in clause_lengths),
            encoding_variant=self.encoding_variant,
            precedence_mode=self.precedence_mode,
            precedence_encoding=self.precedence_encoding,
            precedence_graph=self.precedence_graph,
            precedence_configuration=self.precedence_configuration,
            domain_mode=self.domain_mode,
            enabled_constraints=list(self.enabled_constraints),
            full_schedule_candidates=self.full_schedule_candidates,
            unary_eligible_schedule_candidates=(
                self.unary_eligible_schedule_candidates
            ),
            initial_schedule_candidates=self.initial_schedule_candidates,
            reduced_schedule_candidates=self.reduced_schedule_candidates,
            active_schedule_candidates=self.active_schedule_candidates,
            unary_removed_schedule_candidates=(
                self.full_schedule_candidates
                - self.unary_eligible_schedule_candidates
            ),
            preprocessing_removed_schedule_candidates=(
                self.unary_eligible_schedule_candidates
                - self.reduced_schedule_candidates
            ),
            removed_schedule_candidates=(
                self.initial_schedule_candidates - self.reduced_schedule_candidates
            ),
            precedence_direct_edges=self.graph.direct_edge_count,
            precedence_transitive_edges=self.graph.transitive_edge_count,
            precedence_cycle_nodes=self.graph.cycle_nodes,
            precedence_max_distance=self.graph.max_chain_distance,
            precedence_relation_edges=sum(
                len(distances) for distances in self._precedence_distances
            ),
            precedence_pairwise_clauses=self._precedence_pairwise_clauses,
            precedence_sparse_link_clauses=(
                self._precedence_sparse_link_clauses
            ),
            precedence_unique_suffix_cuts=(
                self._precedence_unique_suffix_cuts
            ),
            objective_encoding=(
                "sparse-support first/last span with exact unary thresholds"
            ),
        )
        return self._artifacts

    def build_wcnf(self) -> WCNF:
        """Build partial MaxSAT minimizing only the idle-slot range over P*."""

        artifacts = self.build_base_cnf()
        wcnf = WCNF()
        for clause in artifacts.cnf.clauses:
            wcnf.append(clause)
        for lit in artifacts.objective_lits:
            wcnf.append([-lit], weight=1)
        return wcnf

    # Feasibility -------------------------------------------------------

    def _add_assignment(self, cnf: CNF) -> None:
        self.enabled_constraints.append(
            f"(20),(22)-(27) exactly once over {self.domain_mode} domains"
        )
        for meeting, slots in enumerate(self._eligible_slots):
            self._add_exactly_one_commander(
                cnf, [self.x(meeting, slot) for slot in slots]
            )

        if self.domain_mode == "full":
            self.enabled_constraints.append(
                "(22)-(27) session/fixed/forbidden exclusions as hard unit clauses"
            )
            for meeting, unary_slots in enumerate(self._unary_eligible_slots):
                allowed = set(unary_slots)
                for slot in range(self.inst.n_total_slots):
                    if slot not in allowed:
                        cnf.append([-self.x(meeting, slot)])

    def _occupancy_strategy(self, participant: int) -> str:
        """Choose a logically equivalent compact occupancy formulation.

        With implied constraint (43), exactly ``|M_p|`` used-slot literals are
        true. Together with exact meeting assignment, either participant
        collision clauses or reverse ``used -> OR(schedule)`` channeling is
        redundant. We retain the cheaper family for each participant.
        """

        cached = self._participant_occupancy_strategy.get(participant)
        if cached is not None:
            return cached
        if not self.use_implied_1:
            strategy = "full"
        else:
            meetings = self.inst.meetings_by_business[participant]
            collision_clauses = 0
            reverse_clauses = 0
            for slot in range(self.inst.n_total_slots):
                count = sum(
                    self.x_or_none(meeting, slot) is not None
                    for meeting in meetings
                )
                collision_clauses += count * (count - 1) // 2
                if count > 0:
                    reverse_clauses += 1

            # Keep binary collisions on ties because they generally propagate
            # better; otherwise emit the family with fewer clauses.
            strategy = (
                "omit_collision"
                if reverse_clauses < collision_clauses
                else "omit_reverse"
            )
        self._participant_occupancy_strategy[participant] = strategy
        return strategy

    def _add_participant_collision(self, cnf: CNF) -> None:
        if self.use_implied_1:
            self.enabled_constraints.append(
                "compact participant occupancy: adaptively omit redundant "
                "collision or reverse-channel clauses"
            )
        else:
            self.enabled_constraints.append("(19) participant atMost-one per slot")
        for participant, meetings in enumerate(self.inst.meetings_by_business):
            if self._occupancy_strategy(participant) == "omit_collision":
                continue
            for slot in range(self.inst.n_total_slots):
                lits = [
                    lit
                    for meeting in meetings
                    if (lit := self.x_or_none(meeting, slot)) is not None
                ]
                if len(lits) > 1:
                    self._add_pairwise_atmost_one(cnf, lits)

    def _add_capacity_over_meetings(self, cnf: CNF) -> None:
        self.enabled_constraints.append(
            f"(21) capacity over {self.domain_mode} schedule variables"
        )
        for slot in range(self.inst.n_total_slots):
            lits = [
                lit
                for meeting in range(self.inst.n_meetings)
                if (lit := self.x_or_none(meeting, slot)) is not None
            ]
            self._add_atmost_seqcounter(cnf, lits, self.inst.n_tables)

    def _compute_meeting_clusters(self) -> list[list[int]]:
        if self._clusters is not None:
            return self._clusters

        unassigned = set(range(self.inst.n_meetings))
        clusters: list[list[int]] = []
        while unassigned:
            best: list[int] = []
            for meetings in self.inst.meetings_by_business:
                candidate = sorted(unassigned.intersection(meetings))
                if len(candidate) > len(best):
                    best = candidate
            if not best:
                best = [min(unassigned)]
            clusters.append(best)
            unassigned.difference_update(best)
        self._clusters = clusters
        return clusters

    def _add_cluster_capacity(self, cnf: CNF) -> None:
        self.enabled_constraints.append(
            "(46),(47) clustered table capacity with compact forward channeling"
        )
        clusters = self._compute_meeting_clusters()

        for slot in range(self.inst.n_total_slots):
            active_clusters: list[int] = []
            for cluster, meetings in enumerate(clusters):
                member_lits = [
                    lit
                    for meeting in meetings
                    if (lit := self.x_or_none(meeting, slot)) is not None
                ]
                if not member_lits:
                    continue
                if len(member_lits) == 1:
                    active_clusters.append(member_lits[0])
                    continue

                cluster_lit = self.cluster_active(cluster, slot)
                active_clusters.append(cluster_lit)
                for lit in member_lits:
                    cnf.append([-lit, cluster_lit])
                # The reverse long clause is intentionally omitted. Forward
                # channeling is sufficient for an at-most capacity constraint.

            self._add_atmost_seqcounter(cnf, active_clusters, self.inst.n_tables)

    # Precedence --------------------------------------------------------

    def _build_sparse_precedence_suffixes(
        self,
        cnf: CNF,
        meeting: int,
        cuts: set[int],
    ) -> dict[int, int]:
        """Build exact suffix ORs only at cuts queried by the selected graph."""

        cached = self._precedence_sparse_suffixes.get(meeting)
        if cached is not None:
            return cached

        slots = self._eligible_slots[meeting]
        valid_cuts = sorted({cut for cut in cuts if 0 <= cut < len(slots)})
        suffixes: dict[int, int] = {}
        next_cut = len(slots)
        next_lit: int | None = None

        for cut in reversed(valid_cuts):
            segment = [self.x(meeting, slots[i]) for i in range(cut, next_cut)]
            if next_lit is None and len(segment) == 1:
                here = segment[0]
            else:
                here = self.vpool.id(("precedenceSparseSuffix", meeting, cut))
                for lit in segment:
                    cnf.append([-lit, here])
                if next_lit is not None:
                    cnf.append([-next_lit, here])
                reverse = [-here] + segment
                if next_lit is not None:
                    reverse.append(next_lit)
                cnf.append(reverse)
            suffixes[cut] = here
            next_cut = cut
            next_lit = here

        self._precedence_sparse_suffixes[meeting] = suffixes
        return suffixes

    def _add_sparse_suffix_precedences(self, cnf: CNF) -> None:
        self.enabled_constraints.append(
            "(28) SparseSuffix precedence over "
            f"{self.precedence_graph} distance-labelled relations"
        )

        links: list[tuple[int, int, int]] = []
        cuts_by_pred: dict[int, set[int]] = {}

        for post, distances in enumerate(self._precedence_distances):
            post_slots = self._eligible_slots[post]
            for pred, distance in sorted(distances.items()):
                pred_slots = self._eligible_slots[pred]
                if not pred_slots or not post_slots:
                    continue

                for post_slot in post_slots:
                    # pred + distance <= post_slot, hence pred <= post_slot-distance.
                    split = bisect_right(pred_slots, post_slot - distance)
                    post_lit = self.x(post, post_slot)
                    if split == 0:
                        cnf.append([-post_lit])
                        self._precedence_sparse_link_clauses += 1
                    elif split < len(pred_slots):
                        cuts_by_pred.setdefault(pred, set()).add(split)
                        links.append((post_lit, pred, split))

        self._precedence_unique_suffix_cuts = sum(
            len(cuts) for cuts in cuts_by_pred.values()
        )
        for pred, cuts in cuts_by_pred.items():
            self._build_sparse_precedence_suffixes(cnf, pred, cuts)

        for post_lit, pred, split in links:
            cnf.append([
                -post_lit,
                -self._precedence_sparse_suffixes[pred][split],
            ])
            self._precedence_sparse_link_clauses += 1

    def _add_pairwise_precedences(self, cnf: CNF) -> None:
        self.enabled_constraints.append(
            "(28) Pairwise precedence over "
            f"{self.precedence_graph} distance-labelled relations"
        )
        for post, distances in enumerate(self._precedence_distances):
            for pred, distance in sorted(distances.items()):
                for pred_slot in self._eligible_slots[pred]:
                    pred_lit = self.x(pred, pred_slot)
                    for post_slot in self._eligible_slots[post]:
                        if pred_slot + distance > post_slot:
                            cnf.append([-pred_lit, -self.x(post, post_slot)])
                            self._precedence_pairwise_clauses += 1

    def _add_precedences(self, cnf: CNF) -> None:
        if self.precedence_encoding == "pairwise":
            self._add_pairwise_precedences(cnf)
            return
        self._add_sparse_suffix_precedences(cnf)

    # Used-slot channeling and span objective --------------------------

    @staticmethod
    def _add_equiv_or(cnf: CNF, out: int, left: int, right: int) -> None:
        cnf.append([-left, out])
        cnf.append([-right, out])
        cnf.append([left, right, -out])

    @staticmethod
    def _add_equiv(cnf: CNF, left: int, right: int) -> None:
        cnf.append([-left, right])
        cnf.append([left, -right])

    def _channel_used_slots(self, cnf: CNF) -> None:
        self.enabled_constraints.append(
            "exact or compact schedule/usedSlot occupancy channeling"
        )
        for participant, meetings in enumerate(self.inst.meetings_by_business):
            strategy = self._occupancy_strategy(participant)
            for slot in range(self.inst.n_total_slots):
                used = self.used_or_none(participant, slot)
                if used is None:
                    continue
                scheduled = [
                    lit
                    for meeting in meetings
                    if (lit := self.x_or_none(meeting, slot)) is not None
                ]
                for lit in scheduled:
                    cnf.append([-lit, used])
                if strategy != "omit_reverse":
                    cnf.append([-used] + scheduled)

            if self.use_implied_1:
                self._add_implied_constraint_1(cnf, participant)

        if self.use_implied_2:
            self._add_implied_constraint_2(cnf)

    def _build_prefix_suffix(
        self,
        cnf: CNF,
        participant: int,
        possible_slots: list[int],
    ) -> None:
        """Build cumulative occupancy only at supported participant slots.

        Prefix/suffix truth values are constant across gaps containing no
        ``usedSlot`` variable. Materializing the chains only where occupancy can
        change removes those duplicate variables and equivalence clauses.
        """

        if not possible_slots:
            return

        previous: int | None = None
        for slot in possible_slots:
            used = self.used_or_none(participant, slot)
            assert used is not None
            if previous is None:
                prefix = used
            else:
                prefix = self.vpool.id(("prefixUsedSupport", participant, slot))
                self._add_equiv_or(cnf, prefix, previous, used)
            self._objective_prefix_support[participant, slot] = prefix
            previous = prefix

        following: int | None = None
        for slot in reversed(possible_slots):
            used = self.used_or_none(participant, slot)
            assert used is not None
            if following is None:
                suffix = used
            else:
                suffix = self.vpool.id(("suffixUsedSupport", participant, slot))
                self._add_equiv_or(cnf, suffix, used, following)
            self._objective_suffix_support[participant, slot] = suffix
            following = suffix

    def _prefix_strictly_before(
        self,
        participant: int,
        possible_slots: list[int],
        slot: int,
    ) -> int | None:
        index = bisect_left(possible_slots, slot) - 1
        if index < 0:
            return None
        support_slot = possible_slots[index]
        return self._objective_prefix_support[participant, support_slot]

    def _suffix_at_or_after(
        self,
        participant: int,
        possible_slots: list[int],
        slot: int,
    ) -> int | None:
        index = bisect_left(possible_slots, slot)
        if index >= len(possible_slots):
            return None
        support_slot = possible_slots[index]
        return self._objective_suffix_support[participant, support_slot]

    def _participant_span_upper_bound(self, participant: int) -> int:
        meetings = self.inst.n_meetings_business[participant]
        possible = sorted(
            slot
            for (p, slot), _ in self._used_vars.items()
            if p == participant
        )
        if meetings <= 1 or not possible:
            return 0
        return max(0, possible[-1] - possible[0] + 1 - meetings)

    def _add_span_break_thresholds(self, cnf: CNF) -> list[list[int]]:
        """Encode ``B(p)>=k`` directly for objective participants in P*."""

        self.enabled_constraints.append(
            "sparse-support P* first/last span encoding: "
            "B(p)=last-first+1-|Mp|"
        )
        self._channel_used_slots(cnf)
        thresholds_by_participant: list[list[int]] = []
        objective_participants = set(self.objective_participants)

        for participant in range(self.inst.n_business):
            meetings = self.inst.n_meetings_business[participant]
            possible_slots = [
                slot
                for slot in range(self.inst.n_total_slots)
                if self.used_or_none(participant, slot) is not None
            ]
            # Participants outside P* have at most one meeting, hence B(p)=0.
            # Skip their prefix/suffix variables entirely to keep the encoding compact.
            if participant not in objective_participants or not possible_slots:
                thresholds_by_participant.append([])
                continue

            self._build_prefix_suffix(cnf, participant, possible_slots)
            first_lits: dict[int, int] = {}
            for slot in possible_slots:
                used = self.used_or_none(participant, slot)
                assert used is not None
                previous = self._prefix_strictly_before(
                    participant,
                    possible_slots,
                    slot,
                )
                if previous is None:
                    first = used
                else:
                    first = self.first_used(participant, slot)
                    # first <-> used AND NOT previous-prefix.
                    cnf.append([-first, used])
                    cnf.append([-first, -previous])
                    cnf.append([-used, previous, first])
                first_lits[slot] = first

            upper = self._participant_span_upper_bound(participant)
            thresholds: list[int] = []
            for amount in range(1, upper + 1):
                threshold = self.break_threshold(participant, amount)
                thresholds.append(threshold)
                for first_slot, first in first_lits.items():
                    target = first_slot + meetings + amount - 1
                    suffix = self._suffix_at_or_after(
                        participant,
                        possible_slots,
                        target,
                    )
                    if suffix is None:
                        cnf.append([-first, -threshold])
                        continue
                    # first -> (threshold <-> suffix[target]).
                    cnf.append([-first, -threshold, suffix])
                    cnf.append([-first, threshold, -suffix])

            for index in range(1, len(thresholds)):
                cnf.append([-thresholds[index], thresholds[index - 1]])
            thresholds_by_participant.append(thresholds)

        return thresholds_by_participant

    def _add_implied_constraint_1(self, cnf: CNF, participant: int) -> None:
        marker = f"(43) exactly |Mp| {self.domain_mode} usedSlot variables"
        if marker not in self.enabled_constraints:
            self.enabled_constraints.append(marker)
        lits = [
            used
            for slot in range(self.inst.n_total_slots)
            if (used := self.used_or_none(participant, slot)) is not None
        ]
        self._add_exactly_cardnet(
            cnf, lits, self.inst.n_meetings_business[participant]
        )

    def _add_implied_constraint_2(self, cnf: CNF) -> None:
        self.enabled_constraints.append("(44) atMost 2|L| busy participants per slot")
        bound = 2 * self.inst.n_tables
        for slot in range(self.inst.n_total_slots):
            lits = [
                used
                for participant in range(self.inst.n_business)
                if (used := self.used_or_none(participant, slot)) is not None
            ]
            if self.use_further_improvements:
                self._add_even_busy_cardinality(cnf, lits, bound)
            else:
                self._add_atmost_cardnet(cnf, lits, bound)

    def _add_even_busy_cardinality(self, cnf: CNF, lits: list[int], bound: int) -> None:
        """Add the useful implied even-busy constraint with a linear parity chain."""

        marker = "(45) exact even number of busy participants"
        if marker not in self.enabled_constraints:
            self.enabled_constraints.append(marker)
        self._add_atmost_cardnet(cnf, lits, bound)
        if not lits:
            return
        if len(lits) == 1:
            cnf.append([-lits[0]])
            return

        parity = lits[0]
        for index, lit in enumerate(lits[1:], start=1):
            next_parity = self.vpool.id(("busyParity", tuple(lits), index))
            cnf.append([parity, lit, -next_parity])
            cnf.append([-parity, -lit, -next_parity])
            cnf.append([parity, -lit, next_parity])
            cnf.append([-parity, lit, next_parity])
            parity = next_parity
        cnf.append([-parity])

    def _add_gap_objective(
        self,
        cnf: CNF,
        thresholds_by_participant: list[list[int]],
    ) -> list[int]:
        """Encode exactly ``max_{p in P*} B(p)-min_{p in P*} B(p)``."""

        self.enabled_constraints.append(
            "exact max/min objective gap over P* with no hard upper bound"
        )
        if len(self.objective_participants) <= 1:
            return []

        global_upper = max(
            (
                len(thresholds_by_participant[participant])
                for participant in self.objective_participants
            ),
            default=0,
        )
        gap_lits: list[int] = []
        max_lits: list[int] = []
        min_lits: list[int] = []

        for amount in range(1, global_upper + 1):
            max_lit = self.max_break(amount)
            min_lit = self.min_break(amount)
            gap_lit = self.dif_break(amount)
            max_lits.append(max_lit)
            min_lits.append(min_lit)
            gap_lits.append(gap_lit)

            present = [
                thresholds_by_participant[participant][amount - 1]
                for participant in self.objective_participants
                if amount <= len(thresholds_by_participant[participant])
            ]

            if present:
                for threshold in present:
                    cnf.append([-threshold, max_lit])
                cnf.append([-max_lit] + present)
            else:
                cnf.append([-max_lit])

            if len(present) != len(self.objective_participants):
                cnf.append([-min_lit])
            else:
                for threshold in present:
                    cnf.append([-min_lit, threshold])
                cnf.append([min_lit] + [-threshold for threshold in present])

            cnf.append([-gap_lit, max_lit])
            cnf.append([-gap_lit, -min_lit])
            cnf.append([-max_lit, min_lit, gap_lit])

        for index in range(1, len(max_lits)):
            cnf.append([-max_lits[index], max_lits[index - 1]])
            cnf.append([-min_lits[index], min_lits[index - 1]])

        return gap_lits

    # Decoding and independent validation ------------------------------

    def decode_assignment(self, sat_model: list[int]) -> list[int]:
        positives = {lit for lit in sat_model if lit > 0}
        assignment = [-1] * self.inst.n_meetings
        for meeting, slots in enumerate(self._eligible_slots):
            chosen = [slot for slot in slots if self.x(meeting, slot) in positives]
            assignment[meeting] = chosen[0] if chosen else -1
        return assignment

    def encoded_objective_value(self, sat_model: list[int]) -> int:
        artifacts = self.build_base_cnf()
        positives = {lit for lit in sat_model if lit > 0}
        return sum(lit in positives for lit in artifacts.objective_lits)

    def objective_consistency_errors(
        self,
        sat_model: list[int],
        stats: B2BSolutionStats,
        *,
        imposed_bound: int | None = None,
        solver_cost: int | None = None,
    ) -> list[str]:
        encoded = self.encoded_objective_value(sat_model)
        expected = stats.objective_gap
        errors: list[str] = []
        if encoded != expected:
            errors.append(
                "objective encoding mismatch: "
                f"encoded gap={encoded}, schedule gap={expected}"
            )
        if imposed_bound is not None and encoded > imposed_bound:
            errors.append(
                "objective-bound violation: "
                f"encoded gap={encoded}, imposed bound={imposed_bound}"
            )
        if solver_cost is not None and solver_cost != encoded:
            errors.append(
                "solver-cost mismatch: "
                f"solver cost={solver_cost}, encoded gap={encoded}"
            )
        return errors

    def compute_stats(self, assignment: list[int]) -> B2BSolutionStats:
        return compute_solution_stats(
            self.inst,
            assignment,
            participants=self.objective_participants,
        )

    def validate_assignment(self, assignment: list[int]) -> list[str]:
        """Check a decoded schedule against the original hard B2B semantics."""

        return validate_schedule_assignment(
            self.inst,
            assignment,
            graph=self.graph,
        )


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build the optimized objective-only B2B SAT/MaxSAT encoding."
        )
    )
    parser.add_argument("instance", help="MiniZinc .dzn instance")
    parser.add_argument(
        "--precedence-mode",
        choices=sorted(VALID_PRECEDENCE_MODES),
        help="deprecated composite alias: traditional or staircase",
    )
    parser.add_argument(
        "--precedence-encoding",
        choices=sorted(VALID_PRECEDENCE_ENCODINGS),
        help="P factor: pairwise or sparse_suffix",
    )
    parser.add_argument(
        "--precedence-graph",
        choices=sorted(VALID_PRECEDENCE_GRAPHS),
        help="G factor: direct or distance_closure",
    )
    parser.add_argument(
        "--encoding-variant",
        choices=sorted(VALID_ENCODING_VARIANTS),
        default="imp12+",
    )
    parser.add_argument(
        "--domain-mode",
        choices=sorted(VALID_DOMAIN_MODES),
        default="reduced",
    )
    parser.add_argument("--write-cnf", type=Path)
    parser.add_argument("--write-wcnf", type=Path)
    parser.add_argument("--skip-meetingsx-validation", action="store_true")
    args = parser.parse_args()
    if args.precedence_mode is not None and (
        args.precedence_encoding is not None or args.precedence_graph is not None
    ):
        parser.error(
            "--precedence-mode cannot be combined with independent P/G flags"
        )
    if (args.precedence_encoding is None) != (args.precedence_graph is None):
        parser.error(
            "--precedence-encoding and --precedence-graph must be used together"
        )

    inst = read_instance(
        args.instance,
        validate_meetingsx_business=not args.skip_meetingsx_validation,
    )
    model = B2BSATModel(
        inst=inst,
        precedence_mode=args.precedence_mode,
        precedence_encoding=args.precedence_encoding,
        precedence_graph=args.precedence_graph,
        encoding_variant=args.encoding_variant,
        domain_mode=args.domain_mode,
    )
    artifacts = model.build_base_cnf()

    print(f"instance={inst.instance_name}")
    print(f"variant={artifacts.encoding_variant}")
    print(f"precedence_mode={artifacts.precedence_mode}")
    print(f"precedence_encoding={artifacts.precedence_encoding}")
    print(f"precedence_graph={artifacts.precedence_graph}")
    print(f"precedence_configuration={artifacts.precedence_configuration}")
    print(f"domain_mode={artifacts.domain_mode}")
    print(
        "schedule_candidates="
        f"active:{artifacts.active_schedule_candidates}, "
        f"full:{artifacts.full_schedule_candidates}, "
        f"unary_eligible:{artifacts.unary_eligible_schedule_candidates}, "
        f"reduced:{artifacts.reduced_schedule_candidates}"
    )
    print(
        "precedence_edges="
        f"direct:{artifacts.precedence_direct_edges}, "
        f"closure:{artifacts.precedence_transitive_edges}, "
        f"max_distance:{artifacts.precedence_max_distance}"
    )
    print(
        "precedence_encoding_metrics="
        f"relations:{artifacts.precedence_relation_edges}, "
        f"pairwise_clauses:{artifacts.precedence_pairwise_clauses}, "
        f"sparse_links:{artifacts.precedence_sparse_link_clauses}, "
        f"unique_cuts:{artifacts.precedence_unique_suffix_cuts}"
    )
    print(
        "precedence_cycle_nodes="
        f"{[node + 1 for node in artifacts.precedence_cycle_nodes]}"
    )
    print(f"vars={artifacts.n_vars}")
    print(f"clauses={artifacts.n_clauses}")
    print(f"objective={artifacts.objective_name}")
    print(
        "objective_participants="
        f"{[participant + 1 for participant in artifacts.objective_participants]}"
    )
    print(f"objective_encoding={artifacts.objective_encoding}")
    print(f"objective_literals={len(artifacts.objective_lits)}")
    print("enabled_constraints=")
    for item in artifacts.enabled_constraints:
        print(f"  - {item}")

    if args.write_cnf is not None:
        artifacts.cnf.to_file(str(args.write_cnf))
        print(f"wrote_cnf={args.write_cnf}")
    if args.write_wcnf is not None:
        model.build_wcnf().to_file(str(args.write_wcnf))
        print(f"wrote_wcnf={args.write_wcnf}")


if __name__ == "__main__":
    _main()
