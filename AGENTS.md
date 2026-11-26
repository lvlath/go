# AGENTS.md

## Project identity

- Repository: `github.com/lvlath/go`
- Module path: `github.com/lvlath/go`
- Minimum declared toolchain: Go 1.23.4
- License: AGPL-3.0-only
- Purpose: deterministic, contract-driven algorithm engineering for Go.

This file is operational guidance for coding agents and contributors. It does not replace package GoDoc, repository documentation, tests, or source code. When guidance conflicts, use the following authority order:

1. current exported API and implementation;
2. package `doc.go` and public GoDoc;
3. tests and executable examples;
4. `docs/` specifications;
5. repository-level README and this file.

If these sources disagree, do not guess. Report the inconsistency and propose the smallest correction that restores parity.

## Package map

- `core`: deterministic graph substrate and topology capabilities.
- `builder`: reproducible graph fixtures and generators.
- `bfs`: unweighted traversal, shortest-hop paths, weak components, partial results.
- `dfs`: finish order, cycle witnesses, DAG validation, topological sorting.
- `dijkstra`: non-negative weighted shortest paths and explicit target/path states.
- `mst`: Prim/Kruskal minimum spanning structures and explicit forest policy.
- `flow`: Ford-Fulkerson, Edmonds-Karp, Dinic, residual networks, min-cut evidence.
- `gridgraph`: deterministic grid/lattice graph generation.
- `matrix`: dense algebra, graph adapters, zero-weight policy, APSP, statistics.
- `tsp`: exact, approximate, and local-search routing under explicit assumptions.
- `dtw`: time-series alignment, windows, penalties, memory policy, optional path recovery.
- `atr`: exact signed-candle True Range, Wilder ATR, ATR-only readiness and transactional exact continuation.

## Repository-wide laws

### Determinism

Observable ordering and tie-breaking are API behavior whenever callers can see them. Do not introduce map-order dependence, incidental heap-order dependence, unstable equal-cost choices, or scheduling-dependent publication.

### Explicit domain assumptions

Directedness, weighting, loops, parallel edges, mixed direction, non-negativity, finiteness, metricity, windows, memory modes, and search limits must be validated or explicitly documented. Never silently substitute a weaker algorithm or policy.

### Error identity

Public failure classes use stable package-level sentinel errors and must remain compatible with `errors.Is`. Error text is diagnostic, not a machine protocol. Do not require callers or tests to match error strings.

### Evidence-rich results

When a scalar alone is operationally insufficient, preserve or return the relevant witness: paths, predecessors, cycle witnesses, trees, cuts, residual graphs, tours, alignments, matrices, or incumbent/optimality status.

### Numeric semantics

Never treat `0`, `+Inf`, `NaN`, negative values, absence, unreachability, and disabled tracking as interchangeable.

Important examples:

- In `matrix`, finite `0` can be a real edge weight. Zero-preserving weighted adjacency uses `+Inf` for absence.
- In metric-closure/distance matrices, diagonal `0` and off-diagonal `+Inf` have distance semantics; derived distances are not original graph edges.
- In `dijkstra`, unknown target, known unreachable target, and disabled path tracking are distinct states.

### Ownership and mutation

Document whether returned values are detached copies, caller-owned artifacts, or aliases of live records. Thread safety does not imply snapshot isolation. Do not deepen or weaken ownership guarantees accidentally.

### Documentation parity

Implementation, GoDoc, `docs/`, examples, tests, benchmarks, and README claims must describe the same current library. Do not document future APIs as if implemented.

## Change workflow

1. Identify the package contract affected by the change.
2. Read the package `doc.go`, relevant `docs/` chapter, exported API, and existing tests.
3. Prefer the smallest coherent change.
4. Add validation before publication or mutation whenever practical.
5. Preserve deterministic output and stable error identity.
6. Add or update tests for ordinary, validation, adversarial, and degenerate cases.
7. Update examples and documentation when observable behavior changes.
8. Add or update benchmarks only when the experiment has a defined workload and excludes unrelated setup.

## Go style

- Use idiomatic Go and `gofmt`.
- Prefer explicit, narrow APIs over convenience that hides mathematical policy.
- Do not add package globals for mutable policy or randomness.
- Avoid panic for ordinary invalid public input.
- Use meaningful exported names and complete GoDoc for every exported symbol.
- Keep algorithm packages modular under the single repository module; do not create nested `go.mod` files.

## Required verification

Run the relevant subset while iterating, then the full gates before proposing a repository-wide change:

```bash
gofmt -w <changed-go-files>
go test ./...
go test -race ./...
go vet ./...
golangci-lint run
govulncheck ./...
```

If a tool is unavailable, say so explicitly; do not claim the check passed.

For benchmark-sensitive changes:

```bash
go test -run '^$' -bench . -benchmem ./path/to/package
```

Record Go version, OS/architecture, CPU, workload, and benchmark command when publishing comparisons.

## Testing expectations

Use the layers that match the contract:

- validation and sentinel-error tests;
- deterministic repeated-run tests;
- insertion-order permutation tests;
- independent witness validators;
- mathematical invariants;
- differential oracles where available;
- metamorphic tests;
- fuzzing for public boundaries;
- race tests for documented concurrent surfaces;
- regression fixtures for published behavior.

Never weaken an assertion merely to make a nondeterministic implementation pass.

## Package-specific cautions

### `core`

- Preserve stable vertex, edge, and neighbor publication rules.
- Respect graph capability flags at construction and mutation boundaries.
- Treat live aliased structural records as documented; do not imply retained pointers are snapshots.

### `matrix`

- Use public options/facades; do not construct internal option state manually.
- Use `WithPreserveZeroWeights` when an all-zero weighted graph must retain real zero-cost edges.
- Use `+Inf`, not a large finite constant, for no-edge/no-path distance semantics.
- Do not export metric-closure matrices as literal graph edges.
- Keep fixed loop order and allocation/ownership guarantees.

### `atr`

- Keep Compute/New canonical. ComputePeriod/NewPeriod and ComputeDefault/NewDefault only delegate; the named default is 14, but Options{} remains invalid. Do not add smoothing modes or unrelated risk calculations.
- Preserve TR0=RN64(H0-L0), then the exact prior-close range. Do not invent C[-1], erase warmup TR, or make negative finite prices invalid. Validate nonfinite values before same-bar geometry; ErrBar must precede numerical work.
- Preserve the exact seed of rounded TR values and one nearest-even rounding per Wilder affine step. Never replace the fixed integer kernel with a rounded alpha, delta update, float numerator, or EMA kernel. Prove any optimization against independent rational arithmetic.
- Keep Result.IsNil exactly a pointer-presence check. Preserve owning outputs, nil/empty distinctions, error precedence and P1 identity. Valid belongs to ATR only; a ready zero stays valid.
- Snapshot v1 uses atr/wilder-rn/v1, matching Options, exact unsigned SeedSum in 2^-1074 units, PreviousClose and Count. Maintain grammar, 660-byte and magnitude bounds, phase-specific zero rules, and all-or-nothing Restore/Push. Do not mistake structural validation for provenance.
- One Stream has one owner. Keep Push free of internal heap allocations in warmup, seed completion, maturity, and rejection. No reflect, unsafe, panic, recover, or production math/big; no test-only exports or lint suppression.
- All package tests use package atr_test and public atr calls. Keep reusable helpers and independent math/big.Rat oracles in test_helper_test.go. Protect every reachable outcome with meaningful assertions; do not expose private arithmetic solely to raise coverage.
- Require at least 92% statement coverage overall and for every executable production file. Files with no instrumented statements are N/A. Do not fabricate coverage for physically unreachable allocation-size guards.
- Keep English per-symbol and private-kernel comments synchronized with actual branches, exact bounds, ownership and complexity. Maintain docs/ATR.md (uppercase), tutorial, FAQ, README, root doc.go, llms.txt, and these agent notes. Preserve unrelated content.
- Keep the vendor demo fixture frozen and explicitly sourced; distinguish it from engineered stress histories. Examples must stay offline and contain checked Output blocks. Source cursor persistence, authentication, revision replay and decision policy stay in the application.


### shortest paths, MST, flow, TSP, DTW

- Validate the mathematical input domain.
- Preserve result witnesses and explicit partial/incumbent status.
- Never claim approximation guarantees without all required assumptions and implementation strength.
- Distinguish cancellation, resource exhaustion, unreachability, unknown entities, and unavailable/disabled witnesses.

## AI-generated changes

AI-generated code is subject to the same review standard as human-written code.

Before accepting generated code:

- verify every referenced symbol exists;
- verify the package import path is `github.com/lvlath/go/...`;
- verify complexity and mathematical claims;
- reject invented options, errors, result fields, or future packages;
- inspect edge cases and numerical semantics;
- run the required verification commands;
- update documentation only after behavior exists.
