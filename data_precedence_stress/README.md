# Derived precedence-density stress dataset

This directory contains 60 derived instances:
20 original B2B contents at each of `prec30`, `prec35`, `prec40`. These files are
not part of the official UdG `b2b.zip` archive.

## Construction

For each original instance, the generator first computes the
lexicographically smallest feasible meeting-slot vector using incremental SAT
assumptions. Every generated precedence arc points from an earlier witness slot
to a later witness slot, so the resulting graph is acyclic and the recorded
witness remains feasible.

For a participant with `d` meetings and requested density `gamma`, exactly
`floor(gamma*d/100)` incoming precedence arcs are selected. This is the edge
budget that reproduces the direct-edge counts of every official `prec15` and
`prec25` file. The randomized edge ladders use global seed `20260724` and
are nested: every lower-density edge remains present at all higher levels.

`gamma` is therefore a requested per-participant density parameter. Because of
integer rounding, the realized aggregate density is also recorded explicitly
in `generation_manifest.csv`.

## Files

- `witnesses.csv`: canonical feasible schedules and source hashes.
- `generation_manifest.csv`: generation parameters and structural statistics.
- `instances_manifest.csv`: runner-compatible manifest with blank official
  archive attribution.
- `metadata.json`: dataset-level parameters.

## Reproduce and validate

From the repository root:

```bash
python3 src/Generate_Precedence_Stress.py generate \
  --output-dir /path/to/a/new/output-directory

python3 src/Generate_Precedence_Stress.py validate \
  --data-dir data_precedence_stress
```

The generator refuses to overwrite an existing output directory.

## Pilot run

The controlled four-cell precedence pilot contains
`240` runs:

```bash
python3 src/Main.py \
  --manifest data_precedence_stress/instances_manifest.csv \
  --family precedence \
  --solver maxsat \
  --maxsat-backend uwrmaxsat \
  --uwrmaxsat-bin /absolute/path/to/uwrmaxsat \
  --domain-mode reduced \
  --precedence-encoding both \
  --precedence-graph both \
  --encoding-variant imp12+ \
  --timeout 7200 \
  --csv output/precedence_stress_pilot.csv
```

## Complete Filter-E block

The study-wide Filter-E campaign uses all 60 instances here, not only the 20
`prec40` instances. For each instance it runs all three Boolean engines and all
four P/G cells at Reduced domains:

```bash
UWRMAXSAT_BIN=/absolute/path/to/uwrmaxsat \
./run_filter_e_all_precedence.sh
```

This directory therefore contributes
`60 instances x 3 engines x 2 P x 2 G = 720` new Filter-E runs. The same runner
also covers official `prec15`/`prec25` and stress-high `prec50`/`prec60`.
