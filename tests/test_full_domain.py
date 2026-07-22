from __future__ import annotations

import sys
import unittest
from itertools import product
from pathlib import Path
from random import Random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from B2B_Instance import B2BInstance, B2BSATModel
from IncrementalSAT_Solver import B2BIncrementalSATSolver
from MaxSAT_Solver import B2BMaxSATSolver
from Multiple_SAT import B2BMultipleSATSolver


def _restricted_instance() -> B2BInstance:
    """Feasible instance exercising every unary restriction and propagation."""

    return B2BInstance(
        n_business=4,
        n_meetings=4,
        n_tables=2,
        n_total_slots=5,
        n_morning_slots=2,
        requested=[
            (0, 1, 1),  # morning and fixed at slot 0
            (0, 2, 2),  # afternoon and fixed at slot 4
            (1, 2, 3),  # forbidden slot 3 through participant 1
            (3, 0, 1),  # morning and fixed at slot 1
        ],
        meetings_by_business=[
            [0, 1, 3],
            [0, 2],
            [1, 2],
            [3],
        ],
        n_meetings_business=[3, 2, 2, 1],
        forbidden=[set(), {3}, set(), set()],
        fixed=[0, 4, None, 1],
        precedences=[set(), set(), {0}, set()],
        instance_name="full-domain-restrictions",
    )


def _participant_collision_unsat_instance() -> B2BInstance:
    """Two meetings of one participant are forced into the same slot."""

    return B2BInstance(
        n_business=3,
        n_meetings=2,
        n_tables=2,
        n_total_slots=3,
        n_morning_slots=1,
        requested=[(0, 1, 3), (0, 2, 3)],
        meetings_by_business=[[0, 1], [0], [1]],
        n_meetings_business=[2, 1, 1],
        forbidden=[set(), set(), set()],
        fixed=[0, 0],
        precedences=[set(), set()],
        instance_name="full-domain-unsat-collision",
    )


def _precedence_chain_instance(*, cyclic: bool = False) -> B2BInstance:
    """Three disjoint meetings linked by a distance-two precedence chain."""

    precedences = [{2}, {0}, {1}] if cyclic else [set(), {0}, {1}]
    return B2BInstance(
        n_business=6,
        n_meetings=3,
        n_tables=1,
        n_total_slots=4,
        n_morning_slots=2,
        requested=[(0, 1, 3), (2, 3, 3), (4, 5, 3)],
        meetings_by_business=[[0], [0], [1], [1], [2], [2]],
        n_meetings_business=[1, 1, 1, 1, 1, 1],
        forbidden=[set() for _ in range(6)],
        fixed=[None, None, None],
        precedences=precedences,
        instance_name="precedence-cycle" if cyclic else "precedence-chain",
    )


def _empty_unary_domain_instance() -> B2BInstance:
    """A morning-only meeting is fixed to an afternoon slot."""

    return B2BInstance(
        n_business=2,
        n_meetings=1,
        n_tables=1,
        n_total_slots=3,
        n_morning_slots=1,
        requested=[(0, 1, 1)],
        meetings_by_business=[[0], [0]],
        n_meetings_business=[1, 1],
        forbidden=[set(), set()],
        fixed=[1],
        precedences=[set()],
        instance_name="empty-unary-domain",
    )


def _generated_instance(seed: int) -> B2BInstance:
    """Generate a deterministic small instance for exhaustive enumeration."""

    rng = Random(seed)
    n_business = 4
    n_meetings = 3 + seed % 3
    n_total_slots = 3 + seed % 2
    requested: list[tuple[int, int, int]] = []
    meetings_by_business = [[] for _ in range(n_business)]
    for meeting in range(n_meetings):
        left, right = rng.sample(range(n_business), 2)
        requested.append((left, right, rng.choice([1, 2, 3])))
        meetings_by_business[left].append(meeting)
        meetings_by_business[right].append(meeting)

    forbidden = [
        {
            slot
            for slot in range(n_total_slots)
            if rng.random() < 0.18
        }
        for _ in range(n_business)
    ]
    fixed = [
        rng.randrange(n_total_slots) if rng.random() < 0.3 else None
        for _ in range(n_meetings)
    ]
    precedences = [set() for _ in range(n_meetings)]
    for post in range(n_meetings):
        for pred in range(post):
            if rng.random() < 0.22:
                precedences[post].add(pred)

    return B2BInstance(
        n_business=n_business,
        n_meetings=n_meetings,
        n_tables=1 + seed % 2,
        n_total_slots=n_total_slots,
        n_morning_slots=n_total_slots // 2,
        requested=requested,
        meetings_by_business=meetings_by_business,
        n_meetings_business=list(map(len, meetings_by_business)),
        forbidden=forbidden,
        fixed=fixed,
        precedences=precedences,
        instance_name=f"generated-{seed}",
    )


class FullDomainEncodingTests(unittest.TestCase):
    def test_full_creates_m_times_t_variables_and_explicit_unary_exclusions(self) -> None:
        inst = _restricted_instance()
        full = B2BSATModel(
            inst,
            precedence_mode="traditional",
            encoding_variant="basic",
            domain_mode="full",
        )
        reduced = B2BSATModel(
            inst,
            precedence_mode="traditional",
            encoding_variant="basic",
            domain_mode="reduced",
        )
        full_artifacts = full.build_base_cnf()
        reduced_artifacts = reduced.build_base_cnf()

        self.assertEqual(full_artifacts.domain_mode, "full")
        self.assertEqual(full_artifacts.full_schedule_candidates, 4 * 5)
        self.assertEqual(full_artifacts.active_schedule_candidates, 4 * 5)
        self.assertEqual(
            sum(len(full.eligible_slots(m)) for m in range(inst.n_meetings)),
            4 * 5,
        )
        self.assertEqual(
            full_artifacts.unary_eligible_schedule_candidates,
            reduced_artifacts.unary_eligible_schedule_candidates,
        )
        self.assertEqual(
            full_artifacts.reduced_schedule_candidates,
            reduced_artifacts.reduced_schedule_candidates,
        )
        self.assertEqual(
            reduced_artifacts.active_schedule_candidates,
            reduced_artifacts.reduced_schedule_candidates,
        )
        self.assertGreater(
            full_artifacts.active_schedule_candidates,
            reduced_artifacts.active_schedule_candidates,
        )
        self.assertGreater(full_artifacts.n_vars, reduced_artifacts.n_vars)

        clauses = {tuple(clause) for clause in full_artifacts.cnf.clauses}
        for meeting in range(inst.n_meetings):
            unary_allowed = set(full.unary_eligible_slots(meeting))
            for slot in range(inst.n_total_slots):
                self.assertIsNotNone(full.x_or_none(meeting, slot))
                if slot not in unary_allowed:
                    self.assertIn((-full.x(meeting, slot),), clauses)

    def test_full_and_reduced_have_identical_optimum_across_model_matrix(self) -> None:
        inst = _restricted_instance()
        checker = B2BSATModel(inst, domain_mode="full")
        brute_force_values = [
            checker.compute_stats(list(assignment)).idle_range
            for assignment in product(
                range(inst.n_total_slots), repeat=inst.n_meetings
            )
            if not checker.validate_assignment(list(assignment))
        ]
        self.assertTrue(brute_force_values)
        brute_force_optimum = min(brute_force_values)

        for precedence_mode in ("traditional", "staircase"):
            for encoding_variant in ("basic", "imp1", "imp2", "imp12", "imp12+"):
                results = {}
                for domain_mode in ("full", "reduced"):
                    result = B2BMaxSATSolver(
                        inst,
                        precedence_mode=precedence_mode,
                        encoding_variant=encoding_variant,
                        domain_mode=domain_mode,
                    ).solve()
                    results[domain_mode] = result
                    with self.subTest(
                        precedence=precedence_mode,
                        variant=encoding_variant,
                        domain=domain_mode,
                    ):
                        self.assertEqual(result["status"], "OPTIMAL")
                        self.assertEqual(result["validation_errors"], [])
                        self.assertEqual(result["domain_mode"], domain_mode)
                        self.assertEqual(
                            result["objective_value"], brute_force_optimum
                        )

                self.assertEqual(
                    results["full"]["objective_value"],
                    results["reduced"]["objective_value"],
                )

    def test_all_optimizers_agree_in_both_domain_modes(self) -> None:
        inst = _restricted_instance()
        for domain_mode in ("full", "reduced"):
            solvers = [
                B2BMaxSATSolver(
                    inst,
                    precedence_mode="staircase",
                    encoding_variant="imp12+",
                    domain_mode=domain_mode,
                ),
                B2BMultipleSATSolver(
                    inst,
                    precedence_mode="staircase",
                    encoding_variant="imp12+",
                    solver_name="glucose",
                    domain_mode=domain_mode,
                ),
                B2BIncrementalSATSolver(
                    inst,
                    precedence_mode="staircase",
                    encoding_variant="imp12+",
                    solver_name="glucose",
                    domain_mode=domain_mode,
                ),
            ]
            objective_values = []
            for solver in solvers:
                result = solver.solve()
                with self.subTest(domain=domain_mode, solver=type(solver).__name__):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertEqual(result["validation_errors"], [])
                    self.assertEqual(result["domain_mode"], domain_mode)
                objective_values.append(result["objective_value"])
            self.assertEqual(objective_values, [objective_values[0]] * len(solvers))

    def test_full_and_reduced_agree_on_unsat(self) -> None:
        inst = _participant_collision_unsat_instance()
        for domain_mode in ("full", "reduced"):
            result = B2BMaxSATSolver(
                inst,
                precedence_mode="traditional",
                encoding_variant="basic",
                domain_mode=domain_mode,
            ).solve()
            with self.subTest(domain=domain_mode):
                self.assertEqual(result["status"], "UNSAT")
                self.assertIsNone(result["assignment"])

    def test_distance_two_chain_is_equivalent_and_cycle_is_unsat(self) -> None:
        chain = _precedence_chain_instance()
        for precedence_mode in ("traditional", "staircase"):
            for domain_mode in ("full", "reduced"):
                result = B2BMaxSATSolver(
                    chain,
                    precedence_mode=precedence_mode,
                    encoding_variant="imp12+",
                    domain_mode=domain_mode,
                ).solve()
                with self.subTest(
                    case="distance-two-chain",
                    precedence=precedence_mode,
                    domain=domain_mode,
                ):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertEqual(result["objective_value"], 0)
                    self.assertEqual(result["validation_errors"], [])

        cycle = _precedence_chain_instance(cyclic=True)
        for precedence_mode in ("traditional", "staircase"):
            for domain_mode in ("full", "reduced"):
                result = B2BMaxSATSolver(
                    cycle,
                    precedence_mode=precedence_mode,
                    encoding_variant="basic",
                    domain_mode=domain_mode,
                ).solve()
                with self.subTest(
                    case="cycle",
                    precedence=precedence_mode,
                    domain=domain_mode,
                ):
                    self.assertEqual(result["status"], "UNSAT")

    def test_empty_unary_domain_is_unsat_in_both_modes(self) -> None:
        inst = _empty_unary_domain_instance()
        for domain_mode in ("full", "reduced"):
            model = B2BSATModel(inst, domain_mode=domain_mode)
            self.assertEqual(model.unary_eligible_slots(0), [])
            result = B2BMaxSATSolver(
                inst,
                precedence_mode="staircase",
                encoding_variant="imp12+",
                domain_mode=domain_mode,
            ).solve()
            with self.subTest(domain=domain_mode):
                self.assertEqual(result["status"], "UNSAT")

    def test_generated_micro_instances_match_exhaustive_semantics(self) -> None:
        for seed in range(8):
            inst = _generated_instance(seed)
            checker = B2BSATModel(inst, domain_mode="full")
            feasible_objectives = [
                checker.compute_stats(list(assignment)).idle_range
                for assignment in product(
                    range(inst.n_total_slots), repeat=inst.n_meetings
                )
                if not checker.validate_assignment(list(assignment))
            ]
            expected = min(feasible_objectives) if feasible_objectives else None

            for precedence_mode in ("traditional", "staircase"):
                for domain_mode in ("full", "reduced"):
                    result = B2BMaxSATSolver(
                        inst,
                        precedence_mode=precedence_mode,
                        encoding_variant="basic",
                        domain_mode=domain_mode,
                    ).solve()
                    with self.subTest(
                        seed=seed,
                        precedence=precedence_mode,
                        domain=domain_mode,
                    ):
                        if expected is None:
                            self.assertEqual(result["status"], "UNSAT")
                        else:
                            self.assertEqual(result["status"], "OPTIMAL")
                            self.assertEqual(result["objective_value"], expected)
                            self.assertEqual(result["validation_errors"], [])

    def test_unknown_domain_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            B2BSATModel(_restricted_instance(), domain_mode="expanded")


if __name__ == "__main__":
    unittest.main()
