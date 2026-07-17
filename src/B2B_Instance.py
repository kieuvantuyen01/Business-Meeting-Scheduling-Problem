from __future__ import annotations

"""Compatible SAT/MaxSAT encoder for the B2B meeting-scheduling problem.

The module is intentionally self-contained and keeps the public API used by the
project's IncrementalSAT_Solver.py, Multiple_SAT.py and MaxSAT_Solver.py.

Supported input formats
-----------------------
* Original MiniZinc data files with nParticipants, nMeetings, nTables,
  nTimeSlots, nMorningSlots, meetings, tnForbidden, forbidden,
  indexForbidden and nMeetingsParticipant.
* Refactored aliases such as nBusiness/nBusinesses, nTotalSlots,
  forbiddenSlots, fixed/fixedMeetings and precedence/precedences.

All public meeting, participant and slot indices in .dzn files are interpreted
as one-based and converted to zero-based Python indices.
"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:  # The solver project normally provides python-sat.
    from pysat.formula import CNF, WCNF
except ImportError:  # Lightweight fallback used for parser/encoder smoke tests.
    class CNF:  # type: ignore[no-redef]
        def __init__(self, from_clauses: Iterable[Iterable[int]] | None = None):
            self.clauses: list[list[int]] = []
            self.nv = 0
            if from_clauses is not None:
                self.extend(from_clauses)

        def append(self, clause: Iterable[int]) -> None:
            row = [int(x) for x in clause]
            self.clauses.append(row)
            if row:
                self.nv = max(self.nv, max(abs(x) for x in row))

        def extend(self, clauses: Iterable[Iterable[int]]) -> None:
            for clause in clauses:
                self.append(clause)

        def copy(self) -> "CNF":
            return CNF(from_clauses=self.clauses)

    class WCNF:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.hard: list[list[int]] = []
            self.soft: list[list[int]] = []
            self.wght: list[int] = []
            self.nv = 0

        def append(self, clause: Iterable[int], weight: int | None = None) -> None:
            row = [int(x) for x in clause]
            if weight is None:
                self.hard.append(row)
            else:
                self.soft.append(row)
                self.wght.append(int(weight))
            if row:
                self.nv = max(self.nv, max(abs(x) for x in row))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class B2BInstance:
    """Normalized B2B instance.

    The field order is kept compatible with earlier encoders in this project.
    """

    n_businesses: int
    n_meetings: int
    n_tables: int
    n_total_slots: int
    n_morning_slots: int
    meetings: list[tuple[int, int, int]]
    meetings_by_business: list[list[int]]
    n_meetings_business: list[int]
    forbidden_slots: list[set[int]]
    fixed_meetings: list[int | None]
    precedences: list[set[int]]
    name: str = "instance"

    def __post_init__(self) -> None:
        self.n_businesses = int(self.n_businesses)
        self.n_meetings = int(self.n_meetings)
        self.n_tables = int(self.n_tables)
        self.n_total_slots = int(self.n_total_slots)
        self.n_morning_slots = int(self.n_morning_slots)

        if self.n_businesses < 0 or self.n_meetings < 0:
            raise ValueError("Numbers of participants and meetings must be non-negative")
        if self.n_tables <= 0:
            raise ValueError("nTables must be positive")
        if self.n_total_slots <= 0:
            raise ValueError("nTimeSlots/nTotalSlots must be positive")
        if not 0 <= self.n_morning_slots <= self.n_total_slots:
            raise ValueError("nMorningSlots must be between 0 and nTimeSlots")
        if len(self.meetings) != self.n_meetings:
            raise ValueError(
                f"Expected {self.n_meetings} meetings, found {len(self.meetings)}"
            )
        if len(self.meetings_by_business) != self.n_businesses:
            raise ValueError("meetings_by_business has the wrong length")
        if len(self.forbidden_slots) != self.n_businesses:
            raise ValueError("forbidden_slots has the wrong length")
        if len(self.fixed_meetings) != self.n_meetings:
            raise ValueError("fixed_meetings has the wrong length")
        if len(self.precedences) != self.n_meetings:
            raise ValueError("precedences has the wrong length")

        for m, (p1, p2, session) in enumerate(self.meetings):
            if not 0 <= p1 < self.n_businesses or not 0 <= p2 < self.n_businesses:
                raise ValueError(f"Meeting {m + 1} contains an invalid participant")
            if p1 == p2:
                raise ValueError(f"Meeting {m + 1} has the same participant twice")
            if session not in (1, 2, 3):
                raise ValueError(f"Meeting {m + 1} has invalid session type {session}")

        for p, slots in enumerate(self.forbidden_slots):
            bad = [t for t in slots if not 0 <= t < self.n_total_slots]
            if bad:
                raise ValueError(
                    f"Participant {p + 1} has invalid forbidden slots: {bad}"
                )

        for m, slot in enumerate(self.fixed_meetings):
            if slot is not None and not 0 <= slot < self.n_total_slots:
                raise ValueError(f"Meeting {m + 1} has invalid fixed slot {slot + 1}")

        for after, before_set in enumerate(self.precedences):
            for before in before_set:
                if not 0 <= before < self.n_meetings:
                    raise ValueError("A precedence references an invalid meeting")
                if before == after:
                    raise ValueError(
                        f"Strict self-precedence on meeting {after + 1} is impossible"
                    )

    # Common names used by different revisions of the solver code.
    @property
    def n_participants(self) -> int:
        return self.n_businesses

    @property
    def nParticipants(self) -> int:  # noqa: N802
        return self.n_businesses

    @property
    def nBusiness(self) -> int:  # noqa: N802
        return self.n_businesses

    @property
    def nBusinesses(self) -> int:  # noqa: N802
        return self.n_businesses

    @property
    def nMeetings(self) -> int:  # noqa: N802
        return self.n_meetings

    @property
    def nTables(self) -> int:  # noqa: N802
        return self.n_tables

    @property
    def nTimeSlots(self) -> int:  # noqa: N802
        return self.n_total_slots

    @property
    def nTotalSlots(self) -> int:  # noqa: N802
        return self.n_total_slots

    @property
    def nMorningSlots(self) -> int:  # noqa: N802
        return self.n_morning_slots

    @property
    def meetings_by_participant(self) -> list[list[int]]:
        return self.meetings_by_business

    @property
    def n_meetings_participant(self) -> list[int]:
        return self.n_meetings_business

    @property
    def fixed_slots(self) -> list[int | None]:
        return self.fixed_meetings


@dataclass
class B2BStats:
    total_breaks: int
    fairness_gap: int
    participant_breaks: list[int]
    busy_participants_per_slot: list[int]
    meetings_per_slot: list[list[int]]

    @property
    def breaks(self) -> int:
        return self.total_breaks

    @property
    def break_counts(self) -> list[int]:
        return self.participant_breaks

    @property
    def schedule_by_slot(self) -> list[list[int]]:
        return self.meetings_per_slot


ScheduleStats = B2BStats
Instance = B2BInstance


class _CallableList(list):
    """A list that can also be called, for API compatibility."""

    def __call__(self) -> list[Any]:
        return list(self)



# ---------------------------------------------------------------------------
# MiniZinc .dzn parser
# ---------------------------------------------------------------------------


_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", flags=re.S)
_LINE_COMMENT_RE = re.compile(r"%[^\n]*|//[^\n]*")
_INT_RE = re.compile(r"(?<![A-Za-z_])-?\d+")
_RANGE_RE = re.compile(r"(-?\d+)\s*\.\.\s*(-?\d+)")


def _strip_comments(text: str) -> str:
    text = _BLOCK_COMMENT_RE.sub("", text)
    return _LINE_COMMENT_RE.sub("", text)


def _split_top_level(text: str, delimiter: str = ";") -> list[str]:
    result: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False

    pairs = {"[": "]", "(": ")", "{": "}"}
    openers = set(pairs)
    closers = set(pairs.values())

    for i, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in ("'", '"'):
            quote = char
        elif char in openers:
            depth += 1
        elif char in closers:
            depth = max(0, depth - 1)
        elif char == delimiter and depth == 0:
            part = text[start:i].strip()
            if part:
                result.append(part)
            start = i + 1

    tail = text[start:].strip()
    if tail:
        result.append(tail)
    return result


def _assignment_map(text: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for statement in _split_top_level(_strip_comments(text)):
        if "=" not in statement:
            continue
        lhs, rhs = statement.split("=", 1)
        names = re.findall(r"[A-Za-z_]\w*", lhs)
        if not names:
            continue
        assignments[names[-1]] = rhs.strip()
    return assignments


def _lookup(values: Mapping[str, str], *aliases: str) -> str | None:
    lower = {key.lower(): val for key, val in values.items()}
    for alias in aliases:
        if alias in values:
            return values[alias]
        found = lower.get(alias.lower())
        if found is not None:
            return found
    return None


def _required_int(values: Mapping[str, str], *aliases: str) -> int:
    raw = _lookup(values, *aliases)
    if raw is None:
        raise ValueError(
            "Missing required scalar; expected one of: " + ", ".join(aliases)
        )
    found = _INT_RE.search(raw)
    if found is None:
        raise ValueError(f"Expected an integer for {aliases[0]}, got {raw!r}")
    return int(found.group())


def _optional_int(values: Mapping[str, str], *aliases: str) -> int | None:
    raw = _lookup(values, *aliases)
    if raw is None:
        return None
    found = _INT_RE.search(raw)
    return None if found is None else int(found.group())


def _array_payload(raw: str) -> str:
    """Return the data payload, ignoring array1d/array2d index domains."""

    raw = raw.strip()
    if re.match(r"array\d+d\s*\(", raw, flags=re.I):
        end = raw.rfind("]")
        start = raw.rfind("[", 0, end + 1)
        if start >= 0 and end > start:
            return raw[start : end + 1]
    return raw


def _expanded_ints(raw: str | None) -> list[int]:
    if raw is None:
        return []
    payload = _array_payload(raw)
    payload = re.sub(r"\btrue\b", "1", payload, flags=re.I)
    payload = re.sub(r"\bfalse\b", "0", payload, flags=re.I)

    def replace_range(match: re.Match[str]) -> str:
        first, last = int(match.group(1)), int(match.group(2))
        step = 1 if first <= last else -1
        return ",".join(str(x) for x in range(first, last + step, step))

    payload = _RANGE_RE.sub(replace_range, payload)
    return [int(x) for x in _INT_RE.findall(payload)]


def _matrix_rows(raw: str | None) -> list[list[int]]:
    if raw is None:
        return []
    payload = _array_payload(raw).strip()
    if "[|" in payload:
        body = payload[payload.find("[|") + 2 :]
        if "|]" in body:
            body = body[: body.rfind("|]")]
        return [
            _expanded_ints(row)
            for row in body.split("|")
            if row.strip() and _expanded_ints(row)
        ]
    return []


def _set_rows(raw: str | None) -> list[set[int]]:
    if raw is None:
        return []
    payload = _array_payload(raw)
    groups = re.findall(r"\{([^{}]*)\}", payload)
    return [set(_expanded_ints(group)) for group in groups]


def _normalize_meetings(
    raw: str | None,
    n_meetings: int,
    n_participants: int,
) -> list[tuple[int, int, int]]:
    if raw is None:
        raise ValueError("Missing required meetings array")

    rows = _matrix_rows(raw)
    if not rows:
        flat = _expanded_ints(raw)
        if len(flat) == 2 * n_meetings:
            rows = [flat[2 * i : 2 * i + 2] for i in range(n_meetings)]
        elif len(flat) == 3 * n_meetings:
            rows = [flat[3 * i : 3 * i + 3] for i in range(n_meetings)]
        else:
            raise ValueError(
                "Cannot reshape meetings: expected "
                f"{2 * n_meetings} or {3 * n_meetings} integers, found {len(flat)}"
            )

    if len(rows) != n_meetings:
        raise ValueError(
            f"Expected {n_meetings} meeting rows, found {len(rows)}"
        )

    result: list[tuple[int, int, int]] = []
    for index, row in enumerate(rows):
        if len(row) < 2:
            raise ValueError(f"Meeting row {index + 1} has fewer than two columns")
        p1, p2 = row[0], row[1]
        session = row[2] if len(row) >= 3 else 3
        # DZN indices are one-based. A zero-based custom file is also accepted.
        if 1 <= p1 <= n_participants and 1 <= p2 <= n_participants:
            p1 -= 1
            p2 -= 1
        result.append((p1, p2, session))
    return result


def _parse_forbidden(
    values: Mapping[str, str],
    n_participants: int,
    n_slots: int,
) -> list[set[int]]:
    result = [set() for _ in range(n_participants)]

    # Newer files may directly store one set per participant.
    set_rows = _set_rows(
        _lookup(values, "forbiddenSlots", "forbidden_slots", "participantForbidden")
    )
    if set_rows:
        if len(set_rows) != n_participants:
            raise ValueError(
                "forbiddenSlots must contain one set for every participant"
            )
        for p, slots in enumerate(set_rows):
            result[p] = {slot - 1 if 1 <= slot <= n_slots else slot for slot in slots}
        return result

    flat_raw = _lookup(values, "forbidden", "forbiddenSlot", "forbidden_slots_flat")
    index_raw = _lookup(values, "indexForbidden", "forbiddenIndex", "index_forbidden")
    if flat_raw is None:
        return result

    flat = _expanded_ints(flat_raw)
    total = _optional_int(values, "tnForbidden", "nForbidden", "totalForbidden")
    if total is not None:
        flat = flat[:total]

    if index_raw is not None:
        starts = _expanded_ints(index_raw)
        if len(starts) != n_participants + 1:
            raise ValueError(
                "indexForbidden must contain nParticipants + 1 entries"
            )
        # Original instances use one-based start positions and tnForbidden+1.
        one_based = min(starts, default=1) >= 1
        for p in range(n_participants):
            begin = starts[p] - (1 if one_based else 0)
            end = starts[p + 1] - (1 if one_based else 0)
            slots = flat[max(0, begin) : max(0, end)]
            result[p] = {
                slot - 1 if 1 <= slot <= n_slots else slot for slot in slots
            }
        return result

    # Matrix form with one participant per row.
    rows = _matrix_rows(flat_raw)
    if rows and len(rows) == n_participants:
        for p, row in enumerate(rows):
            result[p] = {
                slot - 1 if 1 <= slot <= n_slots else slot
                for slot in row
                if slot != 0
            }
        return result

    if flat:
        raise ValueError(
            "A flat forbidden array requires indexForbidden; alternatively use "
            "forbiddenSlots=[{...}, ...]"
        )
    return result


def _parse_fixed(
    values: Mapping[str, str],
    n_meetings: int,
    n_slots: int,
) -> list[int | None]:
    result: list[int | None] = [None] * n_meetings
    raw = _lookup(
        values,
        "fixed",
        "fixedMeetings",
        "fixed_meetings",
        "meetingFixedSlot",
        "fixedSlots",
    )
    if raw is None:
        return result

    rows = _matrix_rows(raw)
    if rows and all(len(row) >= 2 for row in rows):
        for meeting, slot, *_ in rows:
            if meeting == 0 or slot == 0:
                continue
            m = meeting - 1 if 1 <= meeting <= n_meetings else meeting
            t = slot - 1 if 1 <= slot <= n_slots else slot
            if not 0 <= m < n_meetings:
                raise ValueError(f"Invalid fixed meeting index {meeting}")
            result[m] = t
        return result

    flat = _expanded_ints(raw)
    if len(flat) == n_meetings:
        for m, slot in enumerate(flat):
            result[m] = None if slot <= 0 else slot - 1
        return result
    if len(flat) % 2 == 0:
        for meeting, slot in zip(flat[0::2], flat[1::2]):
            if meeting <= 0 or slot <= 0:
                continue
            m = meeting - 1
            if not 0 <= m < n_meetings:
                raise ValueError(f"Invalid fixed meeting index {meeting}")
            result[m] = slot - 1
        return result

    raise ValueError("Cannot parse fixed/fixedMeetings data")


def _parse_precedences(
    values: Mapping[str, str],
    n_meetings: int,
) -> list[set[int]]:
    """Return predecessors for every meeting: result[after] contains before."""

    result = [set() for _ in range(n_meetings)]
    raw = _lookup(
        values,
        "precedences",
        "precedence",
        "meetingPrecedences",
        "precedencePairs",
    )
    if raw is None:
        return result

    rows = _matrix_rows(raw)
    if rows and len(rows) == n_meetings and all(
        len(row) == n_meetings for row in rows
    ):
        for before, row in enumerate(rows):
            for after, value in enumerate(row):
                if value:
                    result[after].add(before)
        return result

    if rows and all(len(row) >= 2 for row in rows):
        pairs = [(row[0], row[1]) for row in rows]
    else:
        flat = _expanded_ints(raw)
        if not flat:
            return result
        if len(flat) == n_meetings * n_meetings and all(x in (0, 1) for x in flat):
            for before in range(n_meetings):
                for after in range(n_meetings):
                    if flat[before * n_meetings + after]:
                        result[after].add(before)
            return result
        if len(flat) % 2:
            raise ValueError("precedence/precedences must contain meeting pairs")
        pairs = list(zip(flat[0::2], flat[1::2]))

    for before_value, after_value in pairs:
        if before_value <= 0 or after_value <= 0:
            continue
        before = before_value - 1
        after = after_value - 1
        if not 0 <= before < n_meetings or not 0 <= after < n_meetings:
            raise ValueError(
                f"Invalid precedence pair ({before_value}, {after_value})"
            )
        result[after].add(before)
    return result


def read_instance(instance_or_path: str | Path | B2BInstance) -> B2BInstance:
    if isinstance(instance_or_path, B2BInstance):
        return instance_or_path

    path = Path(instance_or_path)
    text = path.read_text(encoding="utf-8-sig")
    values = _assignment_map(text)

    n_participants = _required_int(
        values,
        "nParticipants",
        "nBusiness",
        "nBusinesses",
        "n_participants",
        "n_businesses",
    )
    n_meetings = _required_int(values, "nMeetings", "n_meetings")
    n_tables = _required_int(values, "nTables", "nLocations", "n_tables")
    n_slots = _required_int(
        values,
        "nTimeSlots",
        "nTotalSlots",
        "nSlots",
        "n_total_slots",
    )
    n_morning = _required_int(values, "nMorningSlots", "n_morning_slots")

    meetings = _normalize_meetings(
        _lookup(values, "meetings", "meeting", "requests"),
        n_meetings,
        n_participants,
    )

    meetings_by_participant = [[] for _ in range(n_participants)]
    for meeting, (p1, p2, _session) in enumerate(meetings):
        meetings_by_participant[p1].append(meeting)
        meetings_by_participant[p2].append(meeting)

    provided_counts = _expanded_ints(
        _lookup(
            values,
            "nMeetingsParticipant",
            "nMeetingsBusiness",
            "n_meetings_participant",
        )
    )
    derived_counts = [len(items) for items in meetings_by_participant]
    if provided_counts and provided_counts != derived_counts:
        raise ValueError(
            "nMeetingsParticipant disagrees with the meetings matrix: "
            f"provided={provided_counts}, derived={derived_counts}"
        )

    return B2BInstance(
        n_businesses=n_participants,
        n_meetings=n_meetings,
        n_tables=n_tables,
        n_total_slots=n_slots,
        n_morning_slots=n_morning,
        meetings=meetings,
        meetings_by_business=meetings_by_participant,
        n_meetings_business=derived_counts,
        forbidden_slots=_parse_forbidden(values, n_participants, n_slots),
        fixed_meetings=_parse_fixed(values, n_meetings, n_slots),
        precedences=_parse_precedences(values, n_meetings),
        name=path.stem,
    )


load_instance = read_instance
parse_instance = read_instance
read_dzn = read_instance
parse_dzn = read_instance
load_dzn = read_instance
read_b2b_instance = read_instance
parse_b2b_instance = read_instance


# ---------------------------------------------------------------------------
# Graph/domain preprocessing
# ---------------------------------------------------------------------------


def compute_transitive_closure(precedences: Sequence[set[int]]) -> list[set[int]]:
    """Compute predecessor closure and reject strict precedence cycles."""

    n = len(precedences)
    closure = [set(row) for row in precedences]
    changed = True
    while changed:
        changed = False
        for after in range(n):
            expanded = set(closure[after])
            for before in tuple(closure[after]):
                expanded.update(closure[before])
            if after in expanded:
                raise ValueError(
                    f"Strict precedence graph contains a cycle through meeting {after + 1}"
                )
            if expanded != closure[after]:
                closure[after] = expanded
                changed = True
    return closure


def original_eligible_slots(instance: B2BInstance, meeting: int) -> set[int]:
    p1, p2, session = instance.meetings[meeting]
    if session == 1:
        domain = set(range(instance.n_morning_slots))
    elif session == 2:
        domain = set(range(instance.n_morning_slots, instance.n_total_slots))
    else:
        domain = set(range(instance.n_total_slots))

    domain.difference_update(instance.forbidden_slots[p1])
    domain.difference_update(instance.forbidden_slots[p2])

    fixed = instance.fixed_meetings[meeting]
    if fixed is not None:
        domain.intersection_update({fixed})
    return domain


def reduce_domains_by_precedence(
    instance: B2BInstance,
    domains: Sequence[set[int]],
) -> tuple[list[set[int]], list[set[int]]]:
    closure = compute_transitive_closure(instance.precedences)
    reduced = [set(domain) for domain in domains]

    # Direct support propagation reaches a fixed point and handles sparse domains.
    changed = True
    while changed:
        changed = False
        for after, before_set in enumerate(instance.precedences):
            for before in before_set:
                before_supported = {
                    t for t in reduced[before] if any(t < u for u in reduced[after])
                }
                after_supported = {
                    u for u in reduced[after] if any(t < u for t in reduced[before])
                }
                if before_supported != reduced[before]:
                    reduced[before] = before_supported
                    changed = True
                if after_supported != reduced[after]:
                    reduced[after] = after_supported
                    changed = True
    return reduced, closure


# ---------------------------------------------------------------------------
# Variable manager and CNF helpers
# ---------------------------------------------------------------------------


class VariableManager:
    def __init__(self) -> None:
        self._top = 0
        self._by_key: dict[tuple[Any, ...], int] = {}
        self._names: dict[int, str] = {}

    def new_var(self, *key: Any, name: str | None = None) -> int:
        normalized = tuple(key)
        existing = self._by_key.get(normalized)
        if existing is not None:
            return existing
        self._top += 1
        self._by_key[normalized] = self._top
        self._names[self._top] = name or ":".join(str(x) for x in normalized)
        return self._top

    def id(self, *key: Any) -> int | None:
        return self._by_key.get(tuple(key))

    @property
    def top(self) -> int:
        return self._top

    @property
    def max_var(self) -> int:
        return self._top

    @property
    def top_id(self) -> int:
        return self._top

    def obj(self, variable: int) -> str | None:
        return self._names.get(abs(variable))


VarManager = VariableManager


def _copy_cnf(cnf: CNF) -> CNF:
    return CNF(from_clauses=[list(clause) for clause in cnf.clauses])


class B2BSATModel:
    VALID_VARIANTS = {"basic", "imp1", "imp2", "imp12", "imp12+"}
    VALID_PRECEDENCE = {"traditional", "staircase"}

    def __init__(
        self,
        instance: B2BInstance | str | Path,
        fairness_limit: int | None = 2,
        precedence_mode: str = "traditional",
        encoding_variant: str = "basic",
        **kwargs: Any,
    ) -> None:
        # Compatibility aliases used by previous drivers.
        if "fairness_bound" in kwargs:
            fairness_limit = kwargs.pop("fairness_bound")
        if "d" in kwargs:
            fairness_limit = kwargs.pop("d")
        if "variant" in kwargs:
            encoding_variant = kwargs.pop("variant")
        if "staircase" in kwargs:
            precedence_mode = "staircase" if kwargs.pop("staircase") else "traditional"
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected B2BSATModel arguments: {unknown}")

        self.instance = read_instance(instance)
        self.fairness_limit = fairness_limit
        self.fairness_bound = fairness_limit
        self.precedence_mode = precedence_mode.lower()
        self.encoding_variant = encoding_variant.lower()
        self.variant = self.encoding_variant

        if self.precedence_mode not in self.VALID_PRECEDENCE:
            raise ValueError(
                f"precedence_mode must be one of {sorted(self.VALID_PRECEDENCE)}"
            )
        if self.encoding_variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"encoding_variant must be one of {sorted(self.VALID_VARIANTS)}"
            )
        if fairness_limit is not None and fairness_limit < 0:
            self.fairness_limit = None
            self.fairness_bound = None

        raw_domains = [
            original_eligible_slots(self.instance, m)
            for m in range(self.instance.n_meetings)
        ]
        self.preprocess_error: str | None = None
        try:
            self.domains, self.precedence_closure = reduce_domains_by_precedence(
                self.instance, raw_domains
            )
        except ValueError as exc:
            self.domains = raw_domains
            self.precedence_closure = [
                set(row) for row in self.instance.precedences
            ]
            self.preprocess_error = str(exc)

        if self.preprocess_error is None:
            for meeting, domain in enumerate(self.domains):
                if not domain:
                    self.preprocess_error = (
                        f"Meeting {meeting + 1} has no eligible time slot after preprocessing"
                    )
                    break

        self.vm = VariableManager()
        self.var_manager = self.vm
        self.vpool = self.vm

        n_m = self.instance.n_meetings
        n_p = self.instance.n_businesses
        n_t = self.instance.n_total_slots
        self.schedule_vars: list[list[int]] = [[0] * n_t for _ in range(n_m)]
        self.busy_vars: list[list[int]] = [[0] * n_t for _ in range(n_p)]
        self.break_vars: list[list[int]] = [[0] * n_t for _ in range(n_p)]
        self.future_busy_vars: list[list[int]] = [[0] * (n_t + 1) for _ in range(n_p)]
        self.sorted_break_vars: list[list[int]] = [[] for _ in range(n_p)]

        self.cnf: CNF | None = None
        self.formula: CNF | None = None
        self._objective: list[int] = []
        self._cardinality_serial = 0
        self._built = False
        self._enabled_constraints = self._constraint_names()

    def _constraint_names(self) -> list[str]:
        result = [
            "exactly-one slot per meeting",
            "participant collision",
            "table capacity",
            "session/fixed/forbidden domain reduction",
            f"precedence ({self.precedence_mode})",
            "exact break reification",
        ]
        if self.fairness_limit is not None:
            result.append(f"fairness gap <= {self.fairness_limit}")
        if self.encoding_variant in {"imp1", "imp12", "imp12+"}:
            result.append("implied participant activity (imp1)")
        if self.encoding_variant in {"imp2", "imp12", "imp12+"}:
            result.append("implied busy-participant capacity (imp2)")
        if self.encoding_variant == "imp12+":
            result.append("clustered adjacent-slot capacity (imp12+)")
        return result

    @property
    def enabled_constraints(self) -> _CallableList:
        return _CallableList(self._enabled_constraints)

    def get_enabled_constraints(self) -> list[str]:
        return list(self._enabled_constraints)

    @property
    def n_vars(self) -> int:
        self._ensure_built()
        return self.vm.top

    @property
    def n_clauses(self) -> int:
        self._ensure_built()
        assert self.cnf is not None
        return len(self.cnf.clauses)

    def schedule_var(self, meeting: int, slot: int) -> int | None:
        self._ensure_built()
        value = self.schedule_vars[meeting][slot]
        return value or None

    def break_var(self, participant: int, slot: int) -> int | None:
        self._ensure_built()
        value = self.break_vars[participant][slot]
        return value or None

    def _new(self, *key: Any) -> int:
        return self.vm.new_var(*key)

    @staticmethod
    def _append(cnf: CNF, clause: Iterable[int]) -> None:
        cnf.append(list(clause))

    def _add_at_most(self, cnf: CNF, literals: Sequence[int], bound: int) -> None:
        lits = list(dict.fromkeys(int(x) for x in literals if x))
        n = len(lits)
        if bound < 0:
            cnf.append([])
            return
        if bound >= n:
            return
        if bound == 0:
            cnf.extend([[-lit] for lit in lits])
            return
        if bound == 1 and n <= 12:
            cnf.extend([[-a, -b] for a, b in combinations(lits, 2)])
            return

        # Exact unary prefix counter. threshold[j] means count >= j + 1.
        self._cardinality_serial += 1
        serial = self._cardinality_serial
        previous: list[int | bool] = [False] * (bound + 1)
        for i, literal in enumerate(lits):
            current: list[int | bool] = []
            for j in range(bound + 1):
                out = self._new("counter", serial, i, j)
                prev_same = previous[j]
                prev_lower: int | bool = True if j == 0 else previous[j - 1]
                self._add_equiv_or_and(cnf, out, prev_same, literal, prev_lower)
                current.append(out)
            previous = current
        overflow = previous[bound]
        assert isinstance(overflow, int)
        cnf.append([-overflow])

    def _add_exactly_one(self, cnf: CNF, literals: Sequence[int]) -> None:
        lits = [int(x) for x in literals if x]
        if not lits:
            cnf.append([])
            return
        cnf.append(lits)
        self._add_at_most(cnf, lits, 1)

    @staticmethod
    def _literal_is_constant(value: int | bool) -> bool:
        return isinstance(value, bool)

    def _add_equiv_or(
        self,
        cnf: CNF,
        out: int,
        terms: Sequence[int | bool],
    ) -> None:
        if any(term is True for term in terms):
            cnf.append([out])
            return
        lits = [int(term) for term in terms if term is not False]
        if not lits:
            cnf.append([-out])
            return
        for literal in lits:
            cnf.append([-literal, out])
        cnf.append([-out, *lits])

    def _add_equiv_and(
        self,
        cnf: CNF,
        out: int,
        terms: Sequence[int | bool],
    ) -> None:
        if any(term is False for term in terms):
            cnf.append([-out])
            return
        lits = [int(term) for term in terms if term is not True]
        if not lits:
            cnf.append([out])
            return
        for literal in lits:
            cnf.append([-out, literal])
        cnf.append([out, *[-literal for literal in lits]])

    def _add_equiv_or_and(
        self,
        cnf: CNF,
        out: int,
        left: int | bool,
        x: int | bool,
        right: int | bool,
    ) -> None:
        """Encode out <-> left OR (x AND right)."""

        and_var = self._new("and-helper", self.vm.top + 1)
        self._add_equiv_and(cnf, and_var, [x, right])
        self._add_equiv_or(cnf, out, [left, and_var])

    def _add_comparator(self, cnf: CNF, a: int, b: int) -> tuple[int, int]:
        high = self._new("sort-high", self.vm.top + 1)
        low = self._new("sort-low", self.vm.top + 1)
        self._add_equiv_or(cnf, high, [a, b])
        self._add_equiv_and(cnf, low, [a, b])
        return high, low

    def _sort_literals(self, cnf: CNF, literals: Sequence[int]) -> list[int]:
        """Fully reified descending insertion sorting network."""

        output: list[int] = []
        for literal in literals:
            carry = literal
            new_output: list[int] = []
            for old in output:
                high, carry = self._add_comparator(cnf, old, carry)
                new_output.append(high)
            new_output.append(carry)
            output = new_output
        return output

    def _allocate_schedule(self) -> None:
        for meeting, domain in enumerate(self.domains):
            for slot in sorted(domain):
                self.schedule_vars[meeting][slot] = self._new(
                    "schedule", meeting, slot
                )

    def _meeting_literals_for_participant_slot(
        self, participant: int, slot: int
    ) -> list[int]:
        return [
            self.schedule_vars[meeting][slot]
            for meeting in self.instance.meetings_by_business[participant]
            if self.schedule_vars[meeting][slot]
        ]

    def _encode_schedule_and_resources(self, cnf: CNF) -> None:
        for meeting in range(self.instance.n_meetings):
            self._add_exactly_one(
                cnf,
                [self.schedule_vars[meeting][t] for t in sorted(self.domains[meeting])],
            )

        # Participant collisions and exact busy variables.
        for participant in range(self.instance.n_businesses):
            for slot in range(self.instance.n_total_slots):
                literals = self._meeting_literals_for_participant_slot(
                    participant, slot
                )
                if not literals:
                    continue
                self._add_at_most(cnf, literals, 1)
                busy = self._new("busy", participant, slot)
                self.busy_vars[participant][slot] = busy
                self._add_equiv_or(cnf, busy, literals)

        # Number of simultaneous meetings cannot exceed the number of tables.
        for slot in range(self.instance.n_total_slots):
            literals = [
                self.schedule_vars[meeting][slot]
                for meeting in range(self.instance.n_meetings)
                if self.schedule_vars[meeting][slot]
            ]
            self._add_at_most(cnf, literals, self.instance.n_tables)

    def _encode_precedence(self, cnf: CNF) -> None:
        for after, before_set in enumerate(self.instance.precedences):
            for before in before_set:
                before_domain = sorted(self.domains[before])
                after_domain = sorted(self.domains[after])
                if self.precedence_mode == "traditional":
                    for t_before in before_domain:
                        x = self.schedule_vars[before][t_before]
                        for t_after in after_domain:
                            if t_before >= t_after:
                                y = self.schedule_vars[after][t_after]
                                cnf.append([-x, -y])
                else:
                    for t_before in before_domain:
                        support = [
                            self.schedule_vars[after][t_after]
                            for t_after in after_domain
                            if t_after > t_before
                        ]
                        cnf.append(
                            [-self.schedule_vars[before][t_before], *support]
                        )
                    for t_after in after_domain:
                        support = [
                            self.schedule_vars[before][t_before]
                            for t_before in before_domain
                            if t_before < t_after
                        ]
                        cnf.append(
                            [-self.schedule_vars[after][t_after], *support]
                        )

    def _busy_literal(self, participant: int, slot: int) -> int | bool:
        if not 0 <= slot < self.instance.n_total_slots:
            return False
        value = self.busy_vars[participant][slot]
        return value if value else False

    def _encode_breaks(self, cnf: CNF) -> None:
        n_slots = self.instance.n_total_slots
        objective: list[int] = []

        for participant in range(self.instance.n_businesses):
            # future[t] <-> participant is busy in some slot >= t.
            future_next: int | bool = False
            for slot in range(n_slots - 1, -1, -1):
                busy = self._busy_literal(participant, slot)
                if busy is False and future_next is False:
                    future: int | bool = False
                else:
                    future_var = self._new("future-busy", participant, slot)
                    self.future_busy_vars[participant][slot] = future_var
                    self._add_equiv_or(cnf, future_var, [busy, future_next])
                    future = future_var
                future_next = future

            participant_breaks: list[int] = []
            for slot in range(1, n_slots - 1):
                previous_busy = self._busy_literal(participant, slot - 1)
                current_busy = self._busy_literal(participant, slot)
                later_busy: int | bool = self.future_busy_vars[participant][slot + 1]
                if not later_busy:
                    later_busy = False

                # One break is counted at the first empty slot after a busy block.
                if previous_busy is False or later_busy is False:
                    continue
                break_var = self._new("break", participant, slot)
                self.break_vars[participant][slot] = break_var
                self._add_equiv_and(
                    cnf,
                    break_var,
                    [previous_busy, -int(current_busy) if current_busy is not False else True, later_busy],
                )
                participant_breaks.append(break_var)
                objective.append(break_var)

            self.sorted_break_vars[participant] = self._sort_literals(
                cnf, participant_breaks
            )

        self._objective = objective

    def _encode_fairness(self, cnf: CNF) -> None:
        if self.fairness_limit is None:
            return
        d = self.fairness_limit
        for p in range(self.instance.n_businesses):
            p_sorted = self.sorted_break_vars[p]
            for q in range(self.instance.n_businesses):
                if p == q:
                    continue
                q_sorted = self.sorted_break_vars[q]
                # count(p) <= count(q) + d
                for threshold in range(d + 1, len(p_sorted) + 1):
                    antecedent = p_sorted[threshold - 1]
                    needed_q = threshold - d
                    if needed_q <= len(q_sorted):
                        cnf.append([-antecedent, q_sorted[needed_q - 1]])
                    else:
                        cnf.append([-antecedent])

    def _encode_implied_constraints(self, cnf: CNF) -> None:
        variant = self.encoding_variant
        if variant in {"imp1", "imp12", "imp12+"}:
            # Every participant involved in a meeting is busy at least once.
            for participant, meetings in enumerate(
                self.instance.meetings_by_business
            ):
                if meetings:
                    busy = [x for x in self.busy_vars[participant] if x]
                    if busy:
                        cnf.append(busy)

        if variant in {"imp2", "imp12", "imp12+"}:
            # Every meeting makes exactly two participants busy. Therefore table
            # capacity implies at most 2*nTables busy participants in each slot.
            # This is compact and useful when many meetings share participants.
            for slot in range(self.instance.n_total_slots):
                busy = [
                    self.busy_vars[p][slot]
                    for p in range(self.instance.n_businesses)
                    if self.busy_vars[p][slot]
                ]
                self._add_at_most(cnf, busy, 2 * self.instance.n_tables)

        if variant == "imp12+":
            self._add_cluster_capacity(cnf)

    def _add_cluster_capacity(self, cnf: CNF) -> None:
        """Safe adjacent-slot clustered capacity strengthening."""

        for first in range(self.instance.n_total_slots - 1):
            literals = [
                self.schedule_vars[m][t]
                for m in range(self.instance.n_meetings)
                for t in (first, first + 1)
                if self.schedule_vars[m][t]
            ]
            self._add_at_most(cnf, literals, 2 * self.instance.n_tables)

    def build_base_cnf(self) -> CNF:
        if self._built:
            assert self.cnf is not None
            return self.cnf

        cnf = CNF()
        if self.preprocess_error is not None:
            cnf.append([])
            self.cnf = cnf
            self.formula = cnf
            self._built = True
            return cnf

        self._allocate_schedule()
        self._encode_schedule_and_resources(cnf)
        self._encode_precedence(cnf)
        self._encode_breaks(cnf)
        self._encode_fairness(cnf)
        self._encode_implied_constraints(cnf)

        # Some CNF implementations do not update nv when only fresh variables
        # occur in no clause; explicitly keep it synchronized where possible.
        try:
            cnf.nv = max(cnf.nv, self.vm.top)
        except (AttributeError, TypeError):
            pass

        self.cnf = cnf
        self.formula = cnf
        self._built = True
        return cnf

    def build_cnf(self, objective_bound: int | None = None) -> CNF:
        if objective_bound is None:
            return self.build_base_cnf()
        return self.cnf_for_bound(objective_bound)

    def cnf_for_bound(self, bound: int) -> CNF:
        base = self.build_base_cnf()
        result = _copy_cnf(base)
        self._add_at_most(result, self._objective, int(bound))
        try:
            result.nv = max(result.nv, self.vm.top)
        except (AttributeError, TypeError):
            pass
        return result

    def add_objective_bound(self, cnf: CNF, bound: int) -> CNF:
        self._ensure_built()
        self._add_at_most(cnf, self._objective, int(bound))
        return cnf

    def objective_bound_clauses(self, bound: int) -> list[list[int]]:
        """Return only the clauses/auxiliaries for sum(breaks) <= bound."""

        self._ensure_built()
        temporary = CNF()
        self._add_at_most(temporary, self._objective, int(bound))
        return [list(clause) for clause in temporary.clauses]

    def build_wcnf(self) -> WCNF:
        base = self.build_base_cnf()
        formula = WCNF()
        for clause in base.clauses:
            formula.append(clause)
        for literal in self._objective:
            formula.append([-literal], weight=1)
        try:
            formula.nv = max(formula.nv, self.vm.top)
        except (AttributeError, TypeError):
            pass
        return formula

    def _ensure_built(self) -> None:
        if not self._built:
            self.build_base_cnf()

    @property
    def objective_literals(self) -> _CallableList:
        self._ensure_built()
        return _CallableList(self._objective)

    def get_objective_literals(self) -> list[int]:
        return list(self.objective_literals)

    def get_break_literals(self) -> list[int]:
        return list(self.objective_literals)

    @property
    def break_literals(self) -> _CallableList:
        return _CallableList(self.objective_literals)

    @property
    def soft_literals(self) -> _CallableList:
        return _CallableList(self.objective_literals)

    @property
    def objective(self) -> _CallableList:
        return _CallableList(self.objective_literals)

    @property
    def objective_vars(self) -> _CallableList:
        return _CallableList(self.objective_literals)

    def decode_assignment(self, sat_model: Sequence[int] | None) -> list[int] | None:
        if sat_model is None:
            return None
        self._ensure_built()
        positive = {literal for literal in sat_model if literal > 0}
        assignment = [-1] * self.instance.n_meetings
        for meeting, domain in enumerate(self.domains):
            for slot in domain:
                variable = self.schedule_vars[meeting][slot]
                if variable in positive:
                    assignment[meeting] = slot
                    break
        return assignment

    decode_model = decode_assignment

    def compute_stats(self, assignment: Sequence[int] | None) -> B2BStats | None:
        if assignment is None:
            return None
        if len(assignment) != self.instance.n_meetings:
            raise ValueError("Assignment has the wrong number of meetings")

        meetings_per_slot = [
            [] for _ in range(self.instance.n_total_slots)
        ]
        for meeting, slot in enumerate(assignment):
            if 0 <= slot < self.instance.n_total_slots:
                meetings_per_slot[slot].append(meeting)

        participant_breaks: list[int] = []
        for participant in range(self.instance.n_businesses):
            used = sorted(
                assignment[m]
                for m in self.instance.meetings_by_business[participant]
                if 0 <= assignment[m] < self.instance.n_total_slots
            )
            participant_breaks.append(
                sum(right > left + 1 for left, right in zip(used, used[1:]))
            )

        busy_per_slot: list[int] = []
        for slot_meetings in meetings_per_slot:
            participants: set[int] = set()
            for meeting in slot_meetings:
                p1, p2, _ = self.instance.meetings[meeting]
                participants.add(p1)
                participants.add(p2)
            busy_per_slot.append(len(participants))

        fairness_gap = (
            max(participant_breaks) - min(participant_breaks)
            if participant_breaks
            else 0
        )
        return B2BStats(
            total_breaks=sum(participant_breaks),
            fairness_gap=fairness_gap,
            participant_breaks=participant_breaks,
            busy_participants_per_slot=busy_per_slot,
            meetings_per_slot=meetings_per_slot,
        )

    get_stats = compute_stats

    def validate_assignment(self, assignment: Sequence[int] | None) -> list[str]:
        errors: list[str] = []
        if assignment is None:
            return ["No assignment"]
        if len(assignment) != self.instance.n_meetings:
            return [
                f"Expected {self.instance.n_meetings} assigned meetings, "
                f"found {len(assignment)}"
            ]

        for meeting, slot in enumerate(assignment):
            if slot not in self.domains[meeting]:
                errors.append(
                    f"Meeting {meeting + 1} is assigned to ineligible slot {slot + 1}"
                )

        for participant, meetings in enumerate(
            self.instance.meetings_by_business
        ):
            seen: dict[int, int] = {}
            for meeting in meetings:
                slot = assignment[meeting]
                if slot in seen:
                    errors.append(
                        f"Participant {participant + 1} has meetings "
                        f"{seen[slot] + 1} and {meeting + 1} in slot {slot + 1}"
                    )
                seen[slot] = meeting

        for slot in range(self.instance.n_total_slots):
            count = sum(value == slot for value in assignment)
            if count > self.instance.n_tables:
                errors.append(
                    f"Slot {slot + 1} uses {count} tables; limit is "
                    f"{self.instance.n_tables}"
                )

        for after, before_set in enumerate(self.instance.precedences):
            for before in before_set:
                if assignment[before] >= assignment[after]:
                    errors.append(
                        f"Precedence violated: meeting {before + 1} must be before "
                        f"meeting {after + 1}"
                    )

        stats = self.compute_stats(assignment)
        if (
            stats is not None
            and self.fairness_limit is not None
            and stats.fairness_gap > self.fairness_limit
        ):
            errors.append(
                f"Fairness gap is {stats.fairness_gap}; limit is "
                f"{self.fairness_limit}"
            )
        return errors

    validate = validate_assignment


SATModel = B2BSATModel
B2BModel = B2BSATModel
B2BEncoder = B2BSATModel
InstanceEncoder = B2BSATModel


def create_model(
    instance_or_path: B2BInstance | str | Path,
    fairness_limit: int | None = 2,
    precedence_mode: str = "traditional",
    encoding_variant: str = "basic",
) -> B2BSATModel:
    return B2BSATModel(
        read_instance(instance_or_path),
        fairness_limit=fairness_limit,
        precedence_mode=precedence_mode,
        encoding_variant=encoding_variant,
    )


__all__ = [
    "B2BInstance",
    "Instance",
    "B2BStats",
    "ScheduleStats",
    "VariableManager",
    "VarManager",
    "B2BSATModel",
    "SATModel",
    "B2BModel",
    "B2BEncoder",
    "InstanceEncoder",
    "read_instance",
    "load_instance",
    "parse_instance",
    "read_dzn",
    "parse_dzn",
    "load_dzn",
    "read_b2b_instance",
    "parse_b2b_instance",
    "create_model",
    "compute_transitive_closure",
    "original_eligible_slots",
    "reduce_domains_by_precedence",
]
