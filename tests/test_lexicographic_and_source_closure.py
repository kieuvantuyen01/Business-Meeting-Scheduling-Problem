from __future__ import annotations

import sys
import unittest
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from B2B_Instance import B2BInstance, B2BSATModel, build_precedence_graph
from IncrementalSAT_Solver import B2BIncrementalSATSolver
from MaxSAT_Solver import B2BMaxSATSolver
from Multiple_SAT import B2BMultipleSATSolver


def _chain_instance() -> B2BInstance:
    requested = [
        (0, 1, 3),
        (2, 3, 3),
        (4, 5, 3),
    ]
    return B2BInstance(
        n_business=6,
        n_meetings=3,
        n_tables=1,
        n_total_slots=5,
        n_morning_slots=2,
        requested=requested,
        meetings_by_business=[[m] for m in range(3) for _ in range(2)],
        n_meetings_business=[1] * 6,
        forbidden=[set() for _ in range(6)],
        fixed=[None] * 3,
        precedences=[set(), {0}, {1}],
        instance_name="source-closure-chain",
    )


def _lexicographic_instance() -> B2BInstance:
    """Two independent participant pairs admit equal-range schedules of unequal sum."""
    return B2BInstance(
        n_business=4,
        n_meetings=4,
        n_tables=2,
        n_total_slots=4,
        n_morning_slots=2,
        requested=[
            (0, 1, 3),
            (0, 1, 3),
            (2, 3, 3),
            (2, 3, 3),
        ],
        meetings_by_business=[[0, 1], [0, 1], [2, 3], [2, 3]],
        n_meetings_business=[2, 2, 2, 2],
        forbidden=[set() for _ in range(4)],
        fixed=[None] * 4,
        precedences=[set() for _ in range(4)],
        instance_name="lexicographic-tie-break",
    )


def _fixed_positive_lexicographic_instance() -> B2BInstance:
    """A fixed feasible schedule with lexicographic value (1, 4)."""
    return B2BInstance(
        n_business=4,
        n_meetings=4,
        n_tables=2,
        n_total_slots=5,
        n_morning_slots=2,
        requested=[
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
            (3, 0, 3),
        ],
        meetings_by_business=[[0, 1, 3], [0, 2], [1, 2], [3]],
        n_meetings_business=[3, 2, 2, 1],
        forbidden=[set() for _ in range(4)],
        fixed=[0, 4, 2, 1],
        precedences=[set() for _ in range(4)],
        instance_name="fixed-positive-lexicographic",
    )


class SourceAnchoredClosureTests(unittest.TestCase):
    def test_source_augmented_edges_keep_e_and_add_only_source_shortcuts(self) -> None:
        graph = build_precedence_graph([set(), {0}, {1}, {2}])

        self.assertEqual(graph.source_nodes, (0,))
        self.assertEqual(
            graph.direct_predecessors,
            [set(), {0}, {1}, {2}],
        )
        self.assertEqual(
            graph.source_augmented_predecessors,
            [set(), {0}, {0, 1}, {0, 2}],
        )
        self.assertEqual(
            graph.transitive_predecessors,
            [set(), {0}, {0, 1}, {0, 1, 2}],
        )
        self.assertEqual(graph.direct_edge_count, 3)
        self.assertEqual(graph.source_added_edge_count, 2)
        self.assertEqual(graph.source_augmented_edge_count, 5)
        self.assertEqual(graph.transitive_edge_count, 6)

    def test_edge_mode_changes_encoded_edges_but_not_reduced_domains(self) -> None:
        inst = _chain_instance()
        direct = B2BSATModel(
            inst,
            precedence_mode="traditional",
            encoding_variant="basic",
            precedence_edge_mode="direct",
        )
        source = B2BSATModel(
            inst,
            precedence_mode="traditional",
            encoding_variant="basic",
            precedence_edge_mode="source-closure",
        )
        direct_artifacts = direct.build_base_cnf()
        source_artifacts = source.build_base_cnf()

        self.assertEqual(
            [direct.eligible_slots(m) for m in range(inst.n_meetings)],
            [source.eligible_slots(m) for m in range(inst.n_meetings)],
        )
        self.assertEqual(direct_artifacts.precedence_encoded_edges, 2)
        self.assertEqual(source_artifacts.precedence_source_added_edges, 1)
        self.assertEqual(source_artifacts.precedence_encoded_edges, 3)
        self.assertEqual(source_artifacts.precedence_transitive_edges, 3)
        self.assertGreater(source_artifacts.n_clauses, direct_artifacts.n_clauses)


class LexicographicObjectiveTests(unittest.TestCase):
    def test_all_three_solvers_find_the_bruteforce_lexicographic_optimum(self) -> None:
        inst = _lexicographic_instance()
        model = B2BSATModel(
            inst,
            precedence_mode="traditional",
            encoding_variant="basic",
            objective_mode="lexicographic",
        )
        domains = [model.eligible_slots(m) for m in range(inst.n_meetings)]
        feasible_values = []
        for assignment in product(*domains):
            candidate = list(assignment)
            if not model.validate_assignment(candidate):
                stats = model.compute_stats(candidate)
                feasible_values.append(
                    (stats.idle_range, stats.total_internal_idle_slots)
                )

        self.assertEqual(min(feasible_values), (0, 0))

        solvers = [
            B2BMaxSATSolver(
                inst,
                precedence_mode="traditional",
                encoding_variant="basic",
                objective_mode="lexicographic",
            ),
            B2BMultipleSATSolver(
                inst,
                precedence_mode="traditional",
                encoding_variant="basic",
                solver_name="glucose",
                objective_mode="lexicographic",
            ),
            B2BIncrementalSATSolver(
                inst,
                precedence_mode="traditional",
                encoding_variant="basic",
                solver_name="glucose",
                objective_mode="lexicographic",
            ),
        ]

        for solver in solvers:
            with self.subTest(solver=type(solver).__name__):
                result = solver.solve()
                self.assertEqual(result["status"], "OPTIMAL")
                self.assertEqual(result["objective_value"], 0)
                self.assertEqual(result["proven_optimum"], 0)
                self.assertEqual(result["secondary_objective_value"], 0)
                self.assertEqual(result["secondary_proven_optimum"], 0)
                self.assertEqual(result["validation_errors"], [])

    def test_positive_secondary_optimum_is_proven_by_all_three_solvers(self) -> None:
        inst = _fixed_positive_lexicographic_instance()
        solvers = [
            B2BMaxSATSolver(
                inst,
                precedence_mode="traditional",
                encoding_variant="basic",
                objective_mode="lexicographic",
            ),
            B2BMultipleSATSolver(
                inst,
                precedence_mode="traditional",
                encoding_variant="basic",
                solver_name="glucose",
                objective_mode="lexicographic",
            ),
            B2BIncrementalSATSolver(
                inst,
                precedence_mode="traditional",
                encoding_variant="basic",
                solver_name="glucose",
                objective_mode="lexicographic",
            ),
        ]

        for solver in solvers:
            with self.subTest(solver=type(solver).__name__):
                result = solver.solve()
                self.assertEqual(result["status"], "OPTIMAL")
                self.assertEqual(result["proven_optimum"], 1)
                self.assertEqual(result["secondary_proven_optimum"], 4)
                self.assertEqual(result["secondary_objective_value"], 4)
                self.assertEqual(result["validation_errors"], [])


if __name__ == "__main__":
    unittest.main()
