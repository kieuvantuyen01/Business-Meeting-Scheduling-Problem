from __future__ import annotations

from importlib import import_module
from typing import Any


VALID_SAT_BACKENDS = {"cadical", "glucose"}
SAT_BACKEND_LABELS = {
    "cadical": "CaDiCaL",
    "glucose": "Glucose3",
}
SAT_BACKEND_VERSIONS = {
    "cadical": "1.5.3",
    "glucose": "3",
}


def normalize_sat_backend(backend: str) -> str:
    normalized = backend.lower()
    if normalized not in VALID_SAT_BACKENDS:
        raise ValueError(
            f"Unknown SAT backend={normalized!r}; expected one of "
            f"{sorted(VALID_SAT_BACKENDS)}"
        )
    return normalized


def sat_backend_label(backend: str) -> str:
    return SAT_BACKEND_LABELS[normalize_sat_backend(backend)]


def sat_backend_version(backend: str) -> str:
    return SAT_BACKEND_VERSIONS[normalize_sat_backend(backend)]


def create_sat_solver(clauses: list[list[int]], backend: str = "cadical") -> Any:
    """Create exactly the requested SAT backend; never substitute another one."""

    normalized = normalize_sat_backend(backend)
    solvers = import_module("pysat.solvers")
    solver_class = (
        solvers.Cadical153 if normalized == "cadical" else solvers.Glucose3
    )
    try:
        return solver_class(bootstrap_with=clauses)
    except Exception as exc:
        label = sat_backend_label(normalized)
        version = sat_backend_version(normalized)
        raise RuntimeError(
            f"Required SAT backend {label} {version} is unavailable; "
            "automatic solver fallback is disabled"
        ) from exc


def require_sat_backend(backend: str = "cadical") -> None:
    """Fail fast before a benchmark if the requested SAT backend cannot start."""

    with create_sat_solver([], backend):
        pass
