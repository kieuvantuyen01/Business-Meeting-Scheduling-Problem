# B2B SAT/MaxSAT conference model

This repository implements the conference formulation of the
Business-to-Business Meeting Scheduling Problem. The single optimization
objective is

\[
\operatorname{IdleRange}(P^\star)
= \max_{p\in P^\star} B(p)-\min_{p\in P^\star} B(p),
\qquad
P^\star=\{p:|M_p|\ge 2\},
\]

where `B(p)` is the number of idle slots strictly between participant `p`'s
first and last meetings. Participants with zero or one meeting are excluded
because their internal idle time is structurally zero.

The conference model deliberately has no hard objective cap and no
Lexicographic `IdleSum` objective.

## Main components

- `src/B2B_Instance.py`: parser, exact domain reduction, compact break encoding,
  hard constraints, objective literals, decoding, and independent validation.
- `src/MaxSAT_Solver.py`: unit-weight partial MaxSAT optimization with RC2.
- `src/Multiple_SAT.py`: fresh-SAT binary-search optimization.
- `src/IncrementalSAT_Solver.py`: incremental-SAT optimization with a totalizer.
- `src/Main.py`: timeout-controlled benchmark runner and CSV exporter.
- `src/ORG_new.py`: paper-style ORG MaxSAT baseline using the same
  `IdleRange(P*)` participant set.

## Full versus Reduced Domain

`Full Domain` is the set of meeting-slot candidates after the explicit input
restrictions (session, fixed, and forbidden slots). `Reduced Domain` is the
fixed point obtained after exact precedence-distance, participant-matching, and
slot-capacity propagation. The solver always uses Reduced Domain because the
reduction preserves the feasible schedule set.

Detailed CSV output records `initial_schedule_candidates` (Full Domain) and
`reduced_schedule_candidates` (Reduced Domain), so preprocessing effectiveness
can be reported without treating Full Domain as a separate problem variant.

## Run examples

Run all three optimizers, both precedence encodings, and all compact encoding
variants on one instance:

```bash
python3 src/Main.py \
  --instance data_table03_origin/tic-12.original.dzn \
  --solver all \
  --precedence-mode both \
  --encoding-variant all \
  --timeout 120 \
  --csv output/table3_results.csv
```

Run only MaxSAT with the most compact encoding:

```bash
python3 src/Main.py \
  --data-dir data_table08_prec \
  --solver maxsat \
  --precedence-mode staircase \
  --encoding-variant imp12+ \
  --timeout 7200 \
  --csv output/table8_results.csv
```

Build and inspect one CNF/WCNF directly:

```bash
python3 src/B2B_Instance.py \
  data_table03_origin/tic-12.original.dzn \
  --precedence-mode staircase \
  --encoding-variant imp12+
```

## Test

```bash
python3 -m pytest -q
```
