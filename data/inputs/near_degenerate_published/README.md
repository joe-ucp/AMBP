# Published Near-Degenerate Input Pack

This folder contains the small seed inputs needed for the public published
`c413` / `c309` live JHTDB lane.

It is intentionally narrow:

- selected scan rows for the published candidates
- the corresponding `material_heat_age_starts.csv`
- the windows CSV used to define the published time grid

It does not include the large upstream cache tree or derived intermediate
artifacts.

The public runner uses this pack automatically for:

```bash
python scripts/run_near_degenerate_jhtdb_pipeline.py --published-cases --require-jhtdb --results-dir outputs/near_degenerate_public_validation
```

When no `JHTDB_TOKEN` is provided, the JHTDB client falls back to the public
testing token defined in `benchmarks/jhtdb_response_family/config.py`. That
path is bounded and may be rate-limited.
