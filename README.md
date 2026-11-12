<!--
  SPDX-License-Identifier: AGPL-3.0-only
  Copyright (C) 2025-2026 katalvlaran

  lvlath - Repository README

  Purpose:
    This README is the public face of lvlath. It presents the library as a
    deterministic, contract-driven Go toolkit for graph algorithms, flow networks,
    dense graph algebra, routing, exact timestamp grids, causal time-series primitives, fractional PAA,
    symbolic SAX encoding, alignment, and reproducible algorithm engineering. It should give a new reader immediate confidence in
    what lvlath can solve, how packages compose, where the formal contracts live,
    and why this repository is engineered beyond ad-hoc algorithm snippets.

  License:
    lvlath is licensed under AGPL-3.0-only. See LICENSE.
-->

![img.png](img.png)

---

```text
┌─────────────────────────────────┐
│  o       o            o    o    │
│  ║ o   o ║   o───╖  o─╫─o  ║    │
│  ║  \ /  ║   o───╢    ║    ╟──╖ │
│  ╙o  o   ╙o  ╙───o    ╙o   o  o │
└─────────────────────────────────┘
```

---

[![Go Reference](https://pkg.go.dev/badge/github.com/lvlath/go.svg)](https://pkg.go.dev/github.com/lvlath/go)
[![CI](https://github.com/lvlath/go/actions/workflows/go.yml/badge.svg?branch=main)](https://github.com/lvlath/go/actions/workflows/go.yml)
[![Codecov](https://codecov.io/github/lvlath/go/graph/badge.svg?token=QMUZPAO34Y)](https://codecov.io/github/lvlath/go)
[![Go version](https://img.shields.io/badge/Go-%3E%3D1.23.4-00ADD8?logo=go&logoColor=white)](https://go.dev/)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](LICENSE)

[![GitHub contributors](https://img.shields.io/github/contributors/lvlath/go.svg)](https://github.com/lvlath/go/graphs/contributors)
[![GitHub issues](https://img.shields.io/github/issues/lvlath/go.svg)](https://github.com/lvlath/go/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/lvlath/go.svg)](https://github.com/lvlath/go/commits/main)
[![GitHub release](https://img.shields.io/github/release/lvlath/go.svg)](https://github.com/lvlath/go/releases)



---

# lvlath

**lvlath** is a pure-Go algorithm engineering library for graphs, flows, dense graph algebra, routing, exact timestamp grids, causal series transforms, fractional PAA, symbolic SAX time-series signatures, stable statistical moments, fitted feature scaling, covariance PCA, time-series alignment, and reproducible test fixtures.

It is built for people who need algorithmic tools they can explain in a code review, trust in CI, compose across packages, and debug from deterministic artifacts instead of guessing why an answer changed.

**The core promise:**

> Same validated inputs, same explicit policy, same algorithm - same result contract and failure class.

`lvlath` is not just "BFS, DFS, Dijkstra, and friends." It is a small ecosystem where:

* `core.Graph` gives deterministic, thread-safe topology with explicit capabilities;
* traversal packages publish meaningful result artifacts, not incidental internal state;
* weighted routing keeps `+Inf`, unknown targets, disabled path tracking, and unreachable targets separate;
* `matrix` turns topology into dense numeric objects without erasing zero-weight or metric-closure semantics, and provides stable covariance plus reusable pivoted linear-system solvers;
* `series` checks exact timestamp grids and exposes mask-aware causal transformations without interpolation;
* `paa` compresses every sample of a finite series into exact fractional equal-duration averages, without dropped tails or padding;
* `sax` quantizes fractional PAA to compatible symbolic words with Euclidean MINDIST; it never promotes that bound to a DTW pruning certificate;
* `stats` provides scaled Welford/Chan moments and immutable training-only feature standardization without refit;
* `ols` fits full-rank original-unit linear models through pivoted QR, with an explicit zero-origin policy and frozen predictions;
* `ridge` fits strongly correlated or wide linear models with a caller-chosen positive L2 penalty, unpenalized intercept and stable augmented QR coordinates;
* `pca` fits validated covariance principal components and retains immutable training means/scales for leakage-free latent projection and reconstruction;
* flow, MST, TSP, DTW, grid, and builder packages extend the same discipline into optimization, alignment, fixtures, and examples.

---

## Why lvlath exists

A graph or numeric-algorithm library becomes dangerous when the demo works, but the production system loses ordering, topology meaning, numeric semantics, or error identity.

The common failures are familiar:

* Go map iteration leaks into traversal order;
* directed, undirected, mixed, loop, and multi-edge semantics are guessed instead of modeled;
* zero-weight edges disappear inside adjacency matrices;
* `+Inf` is replaced with "large enough" constants and later becomes a fake route;
* BFS is used for weighted routing because it "looks like pathfinding";
* Dijkstra, MST, TSP, and DTW are treated as interchangeable shortest-cost tools;
* DFS discovery order is confused with finish order;
* max-flow examples print only a number and hide residual-network semantics;
* Christofides is claimed without checking metric assumptions or matching strength;
* DTW path output is requested from a memory mode that intentionally discarded history;
* SAX labels are mistaken for compatible thresholds, or Euclidean MINDIST is misused as an exact-DTW pruning bound;
* examples panic, print unstable maps, or silently rely on fictional APIs;
* documentation explains textbook theory but not the actual package contract;
* high-offset variance disappears through raw-moment cancellation, or inference refits on holdout data.

`lvlath` is built against those failure modes. It is an algorithm engineering library: each package publishes a narrow mathematical contract, validates its input domain, returns meaningful result artifacts, and keeps deterministic behavior observable.

### The repository-level laws

1. **Determinism is a package contract**

   `core` provides stable graph surfaces. Algorithm packages either preserve that order or define their own deterministic tie-breaks: BFS processing order, DFS finish order, Dijkstra strict improvement, MST edge ordering, DTW backtracking, TSP matrix order and solver tie-breaks.

2. **Capabilities are declared before topology is trusted**

   Directedness, weighting, loops, multi-edges, and mixed-edge behavior are graph capabilities, not comments. Invalid topology should be rejected at the write or validation boundary, not discovered after an algorithm has already published nonsense.

3. **Every non-trivial algorithm publishes a result artifact**

   BFS `Result`, DFS `Result`, Dijkstra `Result`, MST `Result`, TSP `Result`, DTW `Result`, residual flow graphs, adjacency matrices, incidence matrices, and metric closures are caller-owned artifacts with domain meaning. They are not incidental internal state.

4. **Errors are protocol, not prose**

   Public errors are meant for `errors.Is`. Error text is diagnostic only. A caller must be able to distinguish "unknown target," "known unreachable," "path not tracked," "disconnected strict MST," "invalid matching policy," and "time-limited incumbent."

5. **Policy must be explicit**

   Filters, cutoffs, thresholds, forest mode, metric closure, local search, matching engine, DTW window, memory mode, slope penalty, and TSP time limits are explicit policy choices. No package should silently swap to weaker mathematics because the selected one is inconvenient.

6. **Numeric semantics are not interchangeable**

   `0`, `+Inf`, `NaN`, negative weights, missing edges, zero-cost edges, unreachable states, and local costs mean different things in different packages. `matrix`, `dtw`, `mst`, `dijkstra`, and `tsp` intentionally enforce different numeric laws because they solve different problems.

7. **Graph algorithms and dense algebra compose without erasing meaning**

   `matrix` can convert topology into adjacency, incidence, metric closure, statistics, and factorization workflows. It must preserve zero-weight and `+Inf` semantics instead of flattening them into convenient but false dense arrays.

8. **Time-index contracts must not invent observations**

   `series` checks signed-nanosecond timestamp intervals, keeps sample indices separate from clock time, and publishes validity masks rather than filling missing samples.

9. **Fitted statistics and learned coordinate systems must remain training-only**

   `stats` separates an exact binary-input mean sum from compensated scaled second moments, checks errors before publishing mutations, and freezes fitted feature parameters. PCA similarly freezes its covariance basis, means and optional scales; neither package may recompute training coordinates on a heldout batch. Constant training columns follow explicit policy.

10. **Documentation, examples, tests, and source must describe the same library**

`{package}/doc.go` is the implemented API contract. `docs/{PACKAGE}.md` teaches the mathematics and operational pitfalls. Examples demonstrate real package behavior. Tests defend the contract. None of them should invent future APIs or convenient lies.

---

## Installation

```bash
go get github.com/lvlath/go@latest
```

Requires Go 1.23.4 or newer. Pure Go. No cgo. No external runtime dependencies.

---

## Repository structure

```text
lvlath/
├── core/                  # deterministic graph substrate
│   └── doc.go             # package-level API contract
│
├── bfs/                   # unweighted hop traversal and weak components
├── dfs/                   # post-order traversal, cycle witnesses, topological sort
├── dijkstra/              # full SSSP + Dijkstra/A* target-search family
├── mst/                   # Prim/Kruskal MST/MSF, KMB Steiner networks, tree-covering walks
├── flow/                  # Ford-Fulkerson, Edmonds-Karp, Dinic
├── dtw/                   # temporal alignment, bounded outcomes and exact window search
├── series/                # exact timestamp grids, causal transforms, complete index windows
├── paa/                   # fractional PAA, exact rational geometry, detached averages
├── sax/                   # symbolic PAA words, SHA-256 policy identity, Euclidean MINDIST
├── stats/                 # scaled Welford/Chan moments and immutable fitted scaling
├── ols/                   # QR regression, full-rank gate, frozen physical coefficients
├── ridge/                 # positive-Lambda Tikhonov, balanced augmented QR, immutable fit
├── pca/                   # certified covariance PCA, frozen projection and reconstruction
├── gridgraph/             # 2D lattice graph generation
├── matrix/                # dense graph algebra, stable statistics, pivoted factors and solves
├── tsp/                   # tour construction, local search, exact small-instance routing
├── builder/               # deterministic fixtures and graph generators
│
├── docs/
│   ├── TUTORIAL.md
│   ├── CORE.md
│   ├── BFS.md
│   ├── DFS.md
│   ├── DIJKSTRA.md
│   ├── MST.md
│   ├── FLOW.md
│   ├── DTW.md
│   ├── DTW_SEARCH.md
│   ├── SERIES.md
│   ├── STATS.md
│   ├── OLS.md
│   ├── RIDGE.md
│   ├── PCA.md
│   ├── PAA.md
│   ├── SAX.md
│   ├── GRID_GRAPH.md
│   ├── MATRICES.md     # canonical complete matrix reference
│   ├── TSP.md
│   ├── lvlath_UES.md
│   └── FAQ_&_TIPS.md
│
├── examples/              # runnable scenario programs
├── CONTRIBUTING.md        # contribution workflow and quality gates
├── LICENSE
├── README.md
└── go.mod
```

### Two documentation layers

```text
lvlath/{package}/doc.go
  ↓
  The implemented API contract:
  exported types, options, errors, complexity, ownership, and package guarantees.

lvlath/docs/{PACKAGE}.md
  ↓
  The learning and specification layer:
  mathematics, proofs, pseudocode, diagrams, pitfalls, scenarios, and package selection guidance.
```

---

## Package map: how the system composes

```text
                                      ┌──────────────────────────────────────┐
                                      │              core.Graph              │
                                      │ deterministic topology + capability  │
                                      │ directed / weighted / loops / multi  │
                                      └───────────────────┬──────────────────┘
                                                          │
        ┌───────────────────────────────┬─────────────────┼────────────────┬───────────────────────────────┐
        │                               │                 │                │                               │
  hop / structure                 weighted cost      connectivity      capacity flow                   dense algebra
        │                               │                 │                │                               │
┌───────▼────────┐              ┌───────▼────────┐ ┌──────▼──────┐ ┌───────▼────────┐              ┌───────▼────────┐
│ bfs            │              │ dijkstra       │ │ mst         │ │ flow           │              │ matrix         │
│ hop layers,    │              │ non-negative   │ │ MST/MSF/ST  │ │ max-flow,      │              │ adjacency,     │
│ paths, weak    │              │ shortest paths,│ │ explicit    │ │ min-cut,       │              │ incidence,     │
│ components     │              │ +Inf states    │ │ forest mode │ │ residual graph │              │ APSP, stats    │
└───────┬────────┘              └───────┬────────┘ └──────┬──────┘ └───────┬────────┘              └───────┬────────┘
        │                               │                 │                │                               │
┌───────▼────────┐                      │                 │                │                       ┌───────▼────────┐
│ dfs            │                      │                 │                │                       │ tsp            │
│ finish order,  │                      │                 │                │                       │ matrix-backed  │
│ cycle witness, │                      │                 │                │                       │ tours: exact,  │
│ topo sort      │                      │                 │                │                       │ approximate,   │
└────────────────┘                      │                 │                │                       │ local search   │
                                        │                 │                │                       └────────────────┘
                                        │                 │                │
                           ┌────────────▼──────────┐      │     ┌──────────▼──────────┐
                           │ gridgraph             │      │     │ builder             │
                           │ generated lattice     │      │     │ deterministic       │
                           │ topology for routing  │      │     │ fixtures, examples, │
                           │ and traversal demos   │      │     │ benchmarks          │
                           └───────────────────────┘      │     └─────────────────────┘
                                                          │
                                             ┌────────────▼────────────┐
                                             │ dtw                     │
                                             │ deterministic temporal  │
                                             │ alignment for scalar,   │
                                             │ cost-matrix, and        │
                                             │ multivariate sequences  │
                                             └─────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ series                                     │
                                   │ exact grids / causal masks / index ranges  │
                                   │ standard library; no graph dependencies    │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ paa                                        │
                                   │ rational fractional boundaries / means     │
                                   │ standard library; no graph dependencies    │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ sax                                        │
                                   │ fractional PAA -> symbolic words           │
                                   │ Euclidean MINDIST; NOT a DTW lower bound    │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ stats                                      │
                                   │ stable Welford/Chan moments, fitted scaling│
                                   │ matrix.Matrix input, immutable inference   │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ ols                                        │
                                   │ pivoted QR, full rank, frozen prediction  │
                                   │ physical beta/intercept; train-only scale │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ ridge                                      │
                                   │ positive Lambda / equilibrated augmented QR│
                                   │ wide + collinear, physical beta/intercept  │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ pca                                        │
                                   │ sample covariance -> verified eigenbasis   │
                                   │ train-only scaling / frozen reconstruction │
                                   └────────────────────────────────────────────┘

Composition rule:
  core owns topology.
  matrix owns dense graph/numeric artifacts.
  dtw owns temporal alignment artifacts.
  series owns exact-grid validation and causal array artifacts.
  paa owns fractional step-function compression and exact rational boundaries.
  sax owns compatible symbolic encoding and Euclidean lower-bound dissimilarity.
  stats owns stable scalar moments and immutable matrix feature scaling.
  ols owns full-rank regression artifacts, intercept policy and frozen prediction.
  ridge owns positive-Lambda regression artifacts, unpenalized intercept, and QR equilibration.
  pca owns covariance-based orthogonal factors, strict spectrum admission,
  and immutable train-only reconstruction coordinates.
  mst / flow / tsp own optimization artifacts.
  builder and gridgraph produce reproducible input worlds.
```

---

## Package roles and strengths

| Package     | What it gives you                                                                                                                              | Concrete strength                                                                                                        | Typical scenario                                                                               |
|:------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
| `core`      | Thread-safe in-memory graph with deterministic `Vertices`, `Edges`, and `Neighbors`; explicit directed/weighted/loop/multi/mixed capabilities. | Prevents unstable map order and invalid topology from leaking into algorithms.                                           | Service maps, routing graphs, dependency graphs, test fixtures.                                |
| `bfs`       | Unweighted shortest-hop traversal, path reconstruction, weak components, hooks, filters, partial results.                                      | Distinguishes discovery (`Visited`) from processing (`Order`) and preserves useful partial state.                        | Blast radius, dependency waves, crawler frontiers, weak island discovery.                      |
| `dfs`       | DFS forest, post-order, cycle witnesses, topological sorting, hooks, filters, cancellation.                                                    | Makes finish order explicit and returns deterministic cycle witnesses instead of unstable recursion artifacts.           | Release plans, DAG validation, lock/resource cycle auditing.                                   |
| `dijkstra`  | Full-SSSP Dijkstra plus target Dijkstra, bidirectional Dijkstra, consistent A*, balanced bidirectional A*.                                     | Deterministic original `EdgeIDs`, detached `PathResult`, exact target cutoff/keys and certified stopping.                | Weighted routing, mixed multigraph logistics, grids, network failover.                         |
| `mst`       | Prim/Kruskal exact MST/MSF; original-edge KMB graph Steiner approximation; closed/fixed-end tree-edge covering walks.                          | Shared deterministic Kruskal/DSU selection, original IDs, independent tree validation and distinct network/travel costs. | Fiber/chiplet routing backbones, terminal-only connections, inspection walks.                  |
| `flow`      | Ford-Fulkerson, Edmonds-Karp, and Dinic over `core.Graph`, returning max flow and residual graph.                                              | Preserves residual semantics and supports algorithm selection from simple to high-throughput.                            | Capacity planning, traffic engineering, assignment models, min-cut analysis.                   |
| `dtw`       | General Dynamic Time Warping, bounded classification, certified exact top-K fixed-length squared-scalar window search.                         | Aligns sequences that share a pattern but differ in speed or local timing.                                               | Sensors, gestures, audio contours, time-series similarity.                                     |
| `series`    | Exact signed-nanosecond grid checks, causal lag/difference/returns, and complete index windows.                                                | Detached validity masks, checked integer intervals, and explicit whole-slot gaps without interpolation.                  | Telemetry preparation, causal features, window scheduling, signal-quality gates.               |
| `paa`       | Fractional PAA with exact, unsimplified rational segment geometry and a correctly rounded final binary64 mean.                                 | Never discards remainder samples or fabricates zeros; results are detached.                                              | SAX preparation, telemetry compression, fixed-dimensional sequence features.                   |
| `sax`       | Fixed-length symbolic words, pinned Gaussian/custom thresholds, immutable encoders and Euclidean MINDIST.                                      | Right-bin equality, SHA-256 policy identity, detached results; no false DTW pruning claim.                               | Telemetry signatures, approximate candidate ranking, shape-change screening.                   |
| `ols`       | Full-column-rank least-squares regression, immutable original-unit coefficients and QR diagnostics.                                            | Explicit intercept/scale-only origin policies, rank refusal, training SSE, no heldout refit.                             | CAPM factors, sensor calibration, energy-load forecasting.                                     |
| `ridge`     | Tikhonov L2 regression with mandatory Lambda>0 and immutable physical-unit inference.                                                          | Stable balanced augmented Householder QR, p>n, duplicated predictors, unpenalized intercept and checked objective.       | Correlated multi-factor hedges, wide optical sensors, seasonal AR(12).                         |
| `pca`       | Immutable covariance-based principal components, train-only projection and original-unit reconstruction.                                       | Bounded Jacobi, independent PSD/eigenpair checks, detached loadings and no holdout recentering.                          | Yield-curve factors, turbine residual screening, multichannel telemetry compression.           |
| `stats`     | Stable scalar mean/variance, mergeable Welford/Chan states, train-only fitted feature standardization.                                         | Exact binary-input mean and scaled second moments, detached fitted metadata, immutable inference without refit.          | Telemetry aggregation, distributed metrics, ML feature preparation, anomaly-score calibration. |
| `gridgraph` | 2D lattice graph generation with neighborhood and obstacle-style workflows.                                                                    | Avoids manual wiring for pathfinding maps and teaching graphs.                                                           | Grid routing, maps, demos, benchmark fixtures.                                                 |
| `matrix`    | Dense algebra, graph adapters, scaled covariance/Pearson/L2, pivoted LU/QR, Cholesky, and multi-RHS solves.                                    | Deterministic pivots, detached factors, real condition diagnostics and true residual norms.                              | Graph features, calibration, linear systems and numerical preprocessing.                       |
| `tsp`       | Tour optimization toolkit: practical approximation/local search/exact small-instance strategies.                                               | Bridges exactness and practicality for route planning.                                                                   | Delivery tours, inspection routes, metric routing experiments.                                 |
| `builder`   | Deterministic graph and data generators.                                                                                                       | Produces reproducible examples, tests, and benchmarks.                                                                   | Golden tests, performance fixtures, tutorials.                                                 |


### Series: grid validation before causal features

Use `series` when order, representable timestamp intervals, missing grid slots,
or future-free transformations matter. It is not an alternative DTW alignment
engine: `series` preserves each supplied sample index; `dtw` may align different
indices under an explicit warping policy.

```go
package main

import (
	"errors"
	"fmt"

	"github.com/lvlath/go/series"
)

func main() {
	timesNS := []int64{0, 10, 30}
	observations := []float64{2, 4, 8}

	grid, err := series.Validate(timesNS, observations, series.GridOptions{
		CadenceNS: 10,
		AllowGaps: true,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	changes, err := series.Difference(observations, 1)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println("gaps:", grid.Gaps)
	fmt.Println("differences:", changes.Values, changes.Valid)

	_, err = series.Validate(timesNS, observations, series.GridOptions{CadenceNS: 10})
	fmt.Println("strict grid rejected:", errors.Is(err, series.ErrGap))
}
```

Output:

```text
gaps: [{1 1}]
differences: [0 2 4] [false true true]
strict grid rejected: true
```

A missing timestamp is **not** inserted into `observations`. The difference
at the last supplied index crosses that gap; whether to allow it is an
application decision. Invalid warmup zeros are distinguished by `Valid`,
not by a NaN sentinel. See [`docs/SERIES.md`](docs/SERIES.md) for exact error
precedence, numerical stability, allocation bounds, and ownership guarantees.


### PAA: fractional compression without losing observations

`github.com/lvlath/go/paa` maps finite uniformly sampled `[]float64` observations into a chosen number of **equal-duration**, not floor-divided, output segments. Its rational `Segment` boundaries keep partial contributions at every fractional split. The result owns separate `Values` and `Segments` slices and records the original sample count.

| Need                                                    | API                                                | Why                                                                           |
|:--------------------------------------------------------|:---------------------------------------------------|:------------------------------------------------------------------------------|
| Retain means, exact segment geometry and length         | `paa.Transform(values, segments)`                  | Canonical `*paa.Result` for safe composition with SAX and lower-bound proofs. |
| Reduce one half-open interval without unchecked slicing | `paa.TransformRange(values, start, end, segments)` | The returned boundaries are **local to the selected interval**.               |
| Produce a value-only feature vector                     | `paa.Means(values, segments)`                      | Projects only means; uses the same canonical kernel, but discards geometry.   |

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/paa"
)

func main() {
	result, err := paa.Transform([]float64{0, 2, 4, 6, 8}, 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("PAA: %.1f, %.1f; boundary: %d/%d\n", result.Values[0], result.Values[1], result.Segments[0].RightNumerator, result.Segments[0].Denominator)
	// Output: PAA: 1.6, 6.4; boundary: 5/2
}
```

This is not a calendar-aware resampler: validate timestamps and reject/handle missing grid cells using `series` first. PAA requires `1<=segments<=n`, rejects nonfinite observations, publishes no partial results, and preserves all source interval measure. The signed-zero identity `segments==n` is a detached bitwise copy; all other means use exact fixed-point weighted accumulation followed by nearest-even binary64 rounding. No new Options interface or `matrix` dependency is necessary.

See [`docs/PAA.md`](docs/PAA.md), [`paa/doc.go`](paa/doc.go), and the runnable `go run ./examples/paa_window` scenario.

### SAX: compact symbolic signatures with an explicit Euclidean bound

`github.com/lvlath/go/sax` turns finite, uniformly sampled scalar windows into **fixed-length, fixed-alphabet SAX words**. Unlike PAA, which preserves the averaged numeric values, SAX also quantizes them into unsigned bin indexes and attaches a content-derived `EncodingID` so words are compared only under compatible mathematical policies. The package reuses the **same exact fractional PAA kernel**; it does not drop remainder samples or create a second averaging implementation.

| Need                              | Public surface                                                       | Why it matters                                                                                                       |
|:----------------------------------|:---------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------|
| One normalized window             | `sax.EncodeGaussian(values, segments, alphabet)`                     | Window-local population Z-score, pinned Gaussian quantiles; constant input is rejected by default.                   |
| Preserve absolute units           | `sax.EncodeRaw(values, segments, alphabet)`                          | Identity transform; Gaussian bins are used unchanged, without any claim that the input is Gaussian.                  |
| Calibrate physical thresholds     | `sax.EncodeRawWithBreakpoints(values, segments, breakpoints, label)` | Explicit custom bins with right-boundary equality; label is audit-only.                                              |
| Reuse a fixed policy              | `sax.NewEncoder(opts)` → `encoder.Encode(values)`                    | Immutable copied thresholds and SHA-256 policy identity; detached words per call.                                    |
| Compare compatible symbolic words | `encoder.Mindist(left, right)`                                       | Euclidean lower-bound dissimilarity with fractional factor `sqrt(n/w)`, after independent shape and identity checks. |
| Occasional pairwise bound         | `sax.MindistValues(left, right, opts)`                               | Canonical encode-encode-compare wrapper; no index or recall guarantee.                                               |

For equal-length, equally segmented, compatibly transformed windows, the mathematical invariant in **exact arithmetic** is:

$$ \operatorname{MINDIST}(\hat Q,\hat C)\le D_{\mathrm{PAA}}(\bar Q,\bar C)\le D_{\mathrm{Euclidean}}(Q^*,C^*). $$

![SAX Euclidean lower-bound inequality](https://latex.codecogs.com/svg.image?%5Cmathrm%7BMINDIST%7D%5Cle%20D_%7B%5Cmathrm%7BPAA%7D%7D%5Cle%20D_%7B%5Cmathrm%7BEuclidean%7D%7D%28Q%5E%2A%2CC%5E%2A%29)


The last distance is measured in the **selected transformed space**: original values for `Raw`, separately window-normalized values for `ZPopulation`. The implementation uses float64 and does not provide a directed-rounding numerical pruning certificate. MINDIST is **not a metric on symbol strings**; adjacent but distinct bins can have zero gap.

A runnable, auditable CPU saturation example uses custom percentage boundaries rather than pretending raw percentages follow a Gaussian distribution:

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/sax"
)

func main() {
	baseline := []float64{35, 35, 35, 35, 35, 35, 35, 35}
	saturated := []float64{35, 35, 35, 35, 90, 90, 90, 90}
	opts := sax.Options{
		Segments:      4,
		Alphabet:      5,
		Normalization: sax.Raw,
		Breakpoints:   []float64{20, 40, 60, 80},
		BreakpointID:  "cpu-percent-v1",
	}
	encoder, err := sax.NewEncoder(opts)
	if err != nil {
		fmt.Println(err)
		return
	}
	normal, err := encoder.Encode(baseline)
	if err != nil {
		fmt.Println(err)
		return
	}
	abnormal, err := encoder.Encode(saturated)
	if err != nil {
		fmt.Println(err)
		return
	}
	bound, err := encoder.Mindist(normal, abnormal)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("words: %v vs %v\n", normal.Symbols, abnormal.Symbols)
	fmt.Printf("Euclidean lower bound: %.0f\n", bound)
	// Output:
	// words: [1 1 1 1] vs [1 1 4 4]
	// Euclidean lower bound: 80
}
```

**Critical DTW boundary:** Euclidean SAX MINDIST must **never** be used to prune arbitrary DTW candidates. For `Q=[0,0,1]` and `C=[0,1,1]` with `[0.25,0.75]` thresholds, MINDIST is `0.5` while unconstrained DTW is `0`. SAX can rank an **approximate shortlist** for later `dtw.Align` evaluation, but global DTW nearest-neighbor correctness is not certified; compare retrieval `Recall@K` against exhaustive DTW if it matters. For an exact fixed-length DTW search use the separately certified `dtw.Search` contract.

The immutable Encoder is safe for concurrent inference with stable inputs and Results; `Result.Symbols` and optional `Result.PAA` are detached. Constant Z windows use explicit `RejectConstant` or `ZeroWord` policy (which makes **normalized coordinates**, not necessarily symbol indexes, zero). A matching `EncodingID` proves mathematical policy compatibility, not authenticity of externally mutable Result data.

See [`docs/SAX.md`](docs/SAX.md) for the full fractional Jensen proof, symbol-bin convention, encoding hash serialization, error precedence, numerical limitations, performance and independent regression oracles. Five runnable synthetic operational examples are provided in [`sax/example_test.go`](sax/example_test.go).

### Stats: stable reductions and train-only feature calibration

Use `github.com/lvlath/go/stats` when the application needs **numerically stable one-dimensional moments**, deterministic merging of separately processed partitions, or an **immutable feature standardizer fitted only on training observations**.

| Need                                 | API                                                         | Contract highlight                                               |
|:-------------------------------------|:------------------------------------------------------------|:-----------------------------------------------------------------|
| Stable population/sample statistics  | `stats.SummarizePopulation` / `stats.SummarizeSample`       | Compensated Welford `M2` and exact binary mean sum               |
| Streaming and partitioned statistics | `stats.Moments.Update` / `Merge` / `Summary`                | Error-atomic mutation; scaled Chan merge; checked `uint64` count |
| Train-only feature normalization     | `stats.FitStandardizer`                                     | One fitted mean/scale per matrix column, immutable model         |
| Explicit population/sample fitting   | `stats.FitPopulationStandardizer` / `FitSampleStandardizer` | Fixed DDOF, caller-selected constant policy                      |
| Heldout inference and reconstruction | `Model.Transform` / `InverseTransform`                      | No refit, detached dense outputs, stable affine arithmetic       |

The two numeric layers are deliberate: `Moments` accumulates a **compensated scaled** approximation of the second central moment, while a bounded exact binary sum protects the final mean against arbitrary cancellation of finite binary64 inputs. Therefore an intermediate `M2` outside the `float64` exponent range does not automatically make a *final* variance unrepresentable. Variance remains approximate; no universal error bound is claimed.

This short example demonstrates two independent contracts: sample variance on a large offset and the difference between a constant training feature and a novel heldout value.

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/stats"
)

func main() {
	batch := []float64{1e12 + 1, 1e12 + 2, 1e12 + 3}
	result, err := stats.SummarizeSample(batch)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("sample variance: %.0f\n", result.Variance)

	training, err := matrix.NewDense(3, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	for row := 0; row < 3; row++ {
		if err := training.Set(row, 0, 5); err != nil {
			fmt.Println(err)
			return
		}
	}
	fit, err := stats.FitPopulationStandardizer(training, stats.UnitScaleConstant)
	if err != nil {
		fmt.Println(err)
		return
	}
	heldout, err := matrix.NewDense(1, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := heldout.Set(0, 0, 7); err != nil {
		fmt.Println(err)
		return
	}
	standardized, err := fit.Model.Transform(heldout)
	if err != nil {
		fmt.Println(err)
		return
	}
	novel, err := standardized.At(0, 0)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("heldout constant-feature deviation: %.0f\n", novel)

	// Output:
	// sample variance: 1
	// heldout constant-feature deviation: 2
}
```

`ScaleOptions{}` defaults to population variance and `RejectConstant`; `UnitScaleConstant` must be selected explicitly when retaining a constant feature. Fitted parameter slices (`Means`, `Scales`, `Constant`) are detached from model internals; the model may be used concurrently for inference as long as input matrices remain stable for the call. The mutable `Moments` accumulator is **not** concurrent-safe without external synchronization.

For the full derivations, overflow cases, error precedence, ownership, ASCII diagrams, and integration with `series` masks or multivariate DTW, read [`docs/STATS.md`](docs/STATS.md) and [`stats/doc.go`](stats/doc.go). The five canonical executable examples live in [`stats/example_test.go`](stats/example_test.go).

### Matrix: stable cross-feature statistics and reusable linear solvers

`github.com/lvlath/go/matrix` is both the graph-to-dense representation layer and the numerical layer for full-rank linear systems. Its finite-only statistical kernels are distinct from graph distance matrices that use `+Inf` as an absence marker. For **sample** statistics, rows are observations and columns are features; `Covariance`/`Correlation` require at least two rows, even when the matrix has no feature columns.

| Need                                              | Public API                              | Essential contract                                                              |
|:--------------------------------------------------|:----------------------------------------|:--------------------------------------------------------------------------------|
| Demean many feature columns                       | `matrix.CenterColumns`                  | Scale-aware mean; detached nonempty result; zero-element no-op aliases input    |
| Compute sample cross-feature variability          | `matrix.Covariance`                     | Divides by `n-1` before restoring large scales; bit-identical symmetric cells   |
| Compare correlated features across extreme scales | `matrix.Correlation`                    | Direct Pearson; constant columns have zero row and diagonal                     |
| Unit-length finite feature rows                   | `matrix.NormalizeRowsL2`                | Scaled sum of squares; reports original row norms; true overflow returns error  |
| Solve a square full-rank system                   | `matrix.FactorLU`                       | Deterministic row-pivoted `PA=LU`; reusable and detached                        |
| Fit a tall full-column-rank model                 | `matrix.FactorQR` / `SolveLeastSquares` | Column-pivoted Householder; no normal-equation or minimum-norm fallback         |
| Solve a symmetric SPD system                      | `matrix.FactorCholesky`                 | Explicit symmetry tolerance; positive definiteness checked during factorization |

For every successful new factor solve, `SolveResult` includes a detached solution, numerical rank, one original-coordinate L2 residual per right-hand side, and a **measured** 1-norm condition diagnostic. LU and Cholesky report `kappa_1(A)` of their accepted matrices; QR reports `kappa_1(R)`. Building this diagnostic uses basis-vector solves and is not free. Reusing a factor for successive RHS batches avoids repeatedly paying the factorization cost.

The three current solver domains are intentionally strict. QR rejects wide and rank-deficient inputs; Cholesky rejects matrices outside the accepted symmetry/SPD domain; LU reports numerical rank loss. No silent SVD, ridge, diagonal jitter or alternate objective is introduced. Existing `LUDecompose`/`QRDecompose` are **legacy unpivoted and square-only** interfaces with distinct output conventions.

```go
package main

import (
	"context"
	"fmt"

	"github.com/lvlath/go/matrix"
)

func main() {
	design, err := matrix.NewDense(3, 2)
	if err != nil {
		fmt.Println("design:", err)
		return
	}
	if err := design.Fill([]float64{1, 0, 1, 1, 1, 2}); err != nil {
		fmt.Println("design data:", err)
		return
	}
	rhs, err := matrix.NewDense(3, 1)
	if err != nil {
		fmt.Println("rhs:", err)
		return
	}
	if err := rhs.Fill([]float64{8, 11, 14}); err != nil {
		fmt.Println("rhs data:", err)
		return
	}
	fit, err := matrix.SolveLeastSquares(context.Background(), design, rhs, matrix.FactorOptions{})
	if err != nil {
		fmt.Println("least squares:", err)
		return
	}
	fmt.Printf("unknowns=%d, rank=%d, rhs=%d\n", fit.Solution.Rows(), fit.Rank, len(fit.ResidualNorms))

	// Output: unknowns=2, rank=2, rhs=1
}
```

For larger runnable scenarios with checked errors, see [`docs/MATRICES.md`](docs/MATRICES.md#108-go-usage-examples) and [`matrix/example_test.go`](matrix/example_test.go). For why exact binary-input residuals can be expensive, and where `ErrNaNInf`, `ErrOverflow`, `ErrRankDeficient`, `ErrAsymmetry`, and `ErrNotPositiveDefinite` differ, see the [solver reference](docs/MATRICES.md#1013-error-classification-and-decision-table).

### OLS: stable factor regression with an immutable prediction model

`github.com/lvlath/go/ols` solves **finite, unweighted, full-column-rank
ordinary least squares** by delegating to the existing pivoted Householder
`matrix.SolveLeastSquares` kernel. It provides a regression-specific layer over
`matrix` (numerical solver) and `stats` (population feature scales), without
forming or inverting `XᵀX` or adding a hidden pseudoinverse fallback.

| Need                                      | Public API                                  | Model and failure law                                                     |
|:------------------------------------------|:--------------------------------------------|:--------------------------------------------------------------------------|
| Affine baseline with intercept            | `ols.FitDefault(ctx,x,y)`                   | Explicit ones column, centered predictors, returned physical coefficients |
| Affine model with uneven predictor scales | `ols.FitStandardized(ctx,x,y)`              | Train DDOF=0 scaling; heldout never refits                                |
| Physically mandatory zero origin          | `ols.FitThroughOrigin(ctx,x,y)`             | Exactly positive-zero intercept, no centering                             |
| Origin-constrained and uneven scales      | `ols.FitThroughOriginStandardized(ctx,x,y)` | Scale-only transform, never hidden affine correction                      |
| Custom relative QR rank policy            | `ols.Fit(ctx,x,y,ols.Options{...})`         | Full-rank gate; invalid option/rank/overflow have distinct sentinels      |

A successful `*ols.FitResult` contains `Model`, `Rank`, `ResidualSumSquares`
(training **sum**, not mean squared error) and `ConditionEstimate` (the QR
factor's `kappa_1(R)`, not a statistical confidence measure). `Model.Coefficients()`
returns detached slopes in original feature units and an intercept scalar;
`Model.Predict(ctx,future)` uses those **frozen** parameters without learning
new means or variances. Independent concurrent predictions are safe with
stable input matrices. An `ols.FitResult` also implements the `core.Nilable`
`IsNil()` pointer-presence convention.

```go
package main

import (
    "context"
    "fmt"

    "github.com/lvlath/go/matrix"
    "github.com/lvlath/go/ols"
)

func main() {
    // Each training row is temperature (C) and production-shift count.
    training, err := matrix.NewDense(6, 2)
    if err != nil {
        fmt.Println("training:", err)
        return
    }
    if err = training.Fill([]float64{
        15, 2,
        22, 4,
        20, 3,
        28, 5,
        17, 6,
        30, 1,
    }); err != nil {
        fmt.Println("training data:", err)
        return
    }
    energyKWh := []float64{148, 175, 164, 192, 183, 163}

    fit, err := ols.FitStandardized(context.Background(), training, energyKWh)
    if err != nil {
        fmt.Println("regression:", err)
        return
    }

    future, err := matrix.NewDense(1, 2)
    if err != nil {
        fmt.Println("holdout:", err)
        return
    }
    if err = future.Fill([]float64{26, 3}); err != nil {
        fmt.Println("holdout data:", err)
        return
    }
    prediction, err := fit.Model.Predict(context.Background(), future)
    if err != nil {
        fmt.Println("prediction:", err)
        return
    }
    fmt.Printf("rank=%d, forecast=%.2f kWh\n", fit.Rank, prediction[0])
}
```

OLS **does not** compute p-values, confidence intervals, automatically remove
collinear factors, support weighted/Ridge regression, or guarantee heldout
accuracy. Finance and telemetry may have dependent residuals, so a low training
SSE is not a causal or IID-based inference claim. The five scenario-driven
executable examples include CAPM, industrial calibration, origin-constrained
charging, duplicate-feature rejection and chronological energy-demand holdout.
See [`docs/OLS.md`](docs/OLS.md), [`ols/doc.go`](ols/doc.go) and
[`ols/example_test.go`](ols/example_test.go).

### Ridge: positive-Lambda regularization of correlated and wide factors

`github.com/lvlath/go/ridge` solves a **different statistical objective** from OLS:

![Ridge loss with unpenalized intercept](https://latex.codecogs.com/svg.image?J%28b%2C%5Ctheta%29%3D%5Cfrac%7B1%7D%7B2n%7D%5Csum_%7Bi%3D1%7D%5En%28y_i-b-z_i%5E%5Ctop%5Ctheta%29%5E2%2B%5Cfrac%7B%5Clambda%7D%7B2%7D%5C%7C%5Ctheta%5C%7C_2%5E2%2C%5Cqquad%5Clambda%3E0.)

The intercept `b` is never penalized. `Standardize=true` uses `stats` population scales (`DDOF=0`) while the public `Model.Coefficients` always returns physical-unit coefficients. `NoIntercept=true` forbids centering, so an all-zero physical observation predicts exactly zero. Unlike OLS, `ridge` admits duplicated and very wide original feature sets (`p>n`) when augmented QR accepts their numerical rank.

| Need                                | Public API                                           | Mathematical policy                                  |
|-------------------------------------|------------------------------------------------------|------------------------------------------------------|
| Standard affine shrinkage           | `ridge.FitWithLambda(ctx,x,y,lambda)`                | Physical slopes penalized; free intercept            |
| Scaled correlated factors           | `ridge.FitStandardized(ctx,x,y,lambda)`              | Penalize optimization-coordinate slopes              |
| Physical zero-origin law            | `ridge.FitThroughOrigin(ctx,x,y,lambda)`             | No intercept or centering                            |
| Zero origin with population scaling | `ridge.FitThroughOriginStandardized(ctx,x,y,lambda)` | Scale-only, no implicit affine offset                |
| Custom rank policy                  | `ridge.Fit(ctx,x,y,ridge.Options{...})`              | Mandatory Lambda; `RankTolerance` controls QR cutoff |

The implementation constructs an augmented matrix `[1,Z;0,sqrt(n*Lambda)*I_p]` (or `[Z;sqrt(n*Lambda)*I_p]` without intercept), then applies a **loss-preserving QR preparation**: zero-response penalty rows first, penalized slopes before the intercept, and positive column equilibration. The QR solution is mapped back before publishing `beta`, `SSE`, `Penalty` and `Objective`. This protects extremely small nonzero slopes that a naive Householder RHS update can round to `-0` at high Lambda, without forming a normal equation, changing the penalty or silently falling back to OLS.

```go
package main

import (
    "context"
    "fmt"

    "github.com/lvlath/go/matrix"
    "github.com/lvlath/go/ridge"
)

func main() {
    // Two perfectly correlated synthetic signals plus an independent one.
    train, err := matrix.NewDense(5, 3)
    if err != nil { fmt.Println("matrix:", err); return }
    if err = train.Fill([]float64{
        1, 1, 10,
        2, 2, 12,
        3, 3, 11,
        4, 4, 16,
        5, 5, 14,
    }); err != nil { fmt.Println("training:", err); return }

    fit, err := ridge.FitStandardized(
        context.Background(), train, []float64{3.1, 4.9, 6.8, 9.4, 11.2}, 0.2,
    )
    if err != nil { fmt.Println("fit:", err); return }

    future, err := matrix.NewDense(2, 3)
    if err != nil { fmt.Println("future:", err); return }
    if err = future.Fill([]float64{6, 6, 17, 2, 2, 9}); err != nil {
        fmt.Println("future data:", err)
        return
    }
    forecast, err := fit.Model.Predict(context.Background(), future)
    if err != nil { fmt.Println("forecast:", err); return }
    fmt.Printf("forecasted points=%d objective=%.4f\n", len(forecast), fit.Objective)
}
```

The model learns nothing from future rows. `Lambda` is always caller-controlled and **must be positive and finite**; use `ols` explicitly for unregularized fits. `ResidualSumSquares` is the *original-unit training sum*, while `Penalty` uses *optimization-coordinate* slopes and `Objective=SSE/(2*n)+Penalty`. Treat training diagnostics as model-fit metadata, never as validation or live-market performance claims.

For the Householder cancellation counterexample `Lambda=1e100`, complete arithmetic derivation, strict error precedence, scalar/wide oracles and six operational executable examples, read [`docs/RIDGE.md`](docs/RIDGE.md), [`ridge/doc.go`](ridge/doc.go), [`ridge/example_test.go`](ridge/example_test.go), and [`examples/ridge/main.go`](examples/ridge/main.go).

### PCA: frozen orthogonal factors and auditable reconstruction

`pca` learns orthogonal latent directions from **training rows**, then projects new observations using frozen means, optional training population scales, and canonicalized loading vectors. It is the unsupervised complement of `ols` and `ridge`: those packages fit target responses, while PCA models predictor covariance and reconstructs measured features.

**What makes this PCA implementation deliberately conservative:**

* `pca.Fit(ctx, x, pca.Options{})` uses `matrix.Covariance` plus `matrix.EigenSym`, with a bounded Jacobi rotation budget and an independent eigenpair-accuracy gate. `ThinSVD` is a reserved unsupported enum, not an alternate working kernel or an implicit fallback.
* `FitResult` contains a privately owned immutable `Model`, retained sorted `Eigenvalues`, non-renormalized `ExplainedVarianceRatio`, full detected `Rank`, and the direct training `ReconstructionResidual` in centered/scaled coordinates. `FitResult.IsNil()` implements `core.Nilable`.
* `Model.Transform`, `InverseTransform`, and `Reconstruct` use frozen training parameters and independent per-call buffers; they never recenter a holdout batch or allow a getter to mutate fitted loadings.
* The solver refuses materially negative eigenvalues, invalid symmetry, inaccurate eigenpairs, nonfinite values, overflow and nonconvergence with distinct package sentinels. A covariance-basis numerical rank is **not** an SVD-rank certificate for poorly conditioned data.

A typical application builds **train → calibration → holdout**, fits `k` using training data only, calibrates fault thresholds outside PCA, and evaluates holdout reconstruction against a training-only baseline. Useful examples include yield-curve factor stress, six-channel turbine vibration monitoring, 12-lead ECG denoising, grid-feeder telemetry compression, and mixed-unit water-quality measurements. These executable fixtures are deterministic synthetic engineering demonstrations, not verified market, medical or plant datasets.

For complete executable code, see [`pca/example_test.go`](pca/example_test.go) and [`examples/pca/main.go`](examples/pca/main.go). For formulas, complexity, numeric limits, and exact API/error precedence, read [`docs/PCA.md`](docs/PCA.md).

**Choose by objective, not by the word "factor":** `pca.FitWithComponents` compresses measured features without a response variable; `ols.FitDefault` estimates a full-rank linear response; `ridge.FitStandardized` estimates a penalized response when predictors are correlated or wide. PCA is neither a substitute for a regression target nor a forecasting method.

### Weighted shortest paths: choose the result domain

`github.com/lvlath/go/dijkstra` now exposes **two distinct API families**, not four variants of the same return type:

| Need                                             | Public API                                                            | Result                                                            |
|:-------------------------------------------------|:----------------------------------------------------------------------|:------------------------------------------------------------------|
| All distances from one source                    | `dijkstra.Dijkstra(g, sourceID, opts...)`                             | `*dijkstra.Result` with distance map and optional predecessor map |
| One target, no heuristic                         | `dijkstra.Search(g, sourceID, targetID, opts...)`                     | `*dijkstra.PathResult` with original `EdgeIDs`                    |
| One target, two search directions                | `dijkstra.Bidirectional(g, sourceID, targetID, opts...)`              | `*dijkstra.PathResult` with certified stopping                    |
| One target with a **consistent** target estimate | `dijkstra.AStar(g, sourceID, targetID, hT, opts...)`                  | `*dijkstra.PathResult`                                            |
| Two **consistent** directional estimates         | `dijkstra.BidirectionalAStar(g, sourceID, targetID, hT, hS, opts...)` | `*dijkstra.PathResult`                                            |

`dijkstra.DistanceTo` and `dijkstra.ShortestPathTo` remain legacy **full-SSSP wrappers**, not target early-exit optimizations. `Search` returns only one original-cost route witness and does not expose finalized all-vertex distances. It uses one `core.Clone` per query; the legacy runner retains its live-graph traversal semantics.

Target methods require **exactly representable binary64 stored costs and heuristic keys**; this stricter arithmetic law is not the legacy rounded-distance policy. Reachable results include original edge IDs, while known-unreachable targets return `Distance=+Inf`, `Reachable=false` and nil error. Consistent A* callbacks are verified against all admitted oriented arcs before settlement; arbitrary admissible-only heuristics are not accepted.


### MST, graph Steiner, and covering walks: three distinct results

`github.com/lvlath/go/mst` now supports three related but different objectives:

| Objective                                               | Public call                                | Result / guarantee                                                                                 |
|:--------------------------------------------------------|:-------------------------------------------|:---------------------------------------------------------------------------------------------------|
| Cheapest full-vertex spanning structure                 | `mst.Kruskal(g)` / `mst.Prim(g, root)`     | `*mst.Result`, exact MST; explicit `mst.WithForest()` for MSF                                      |
| Cheap network connecting only named terminals           | `mst.SteinerTree(g, terminals, opts...)`   | `*mst.SteinerResult`, KMB **approximation** of graph Steiner using original `core.Edge.ID` values  |
| Cheapest route covering every edge of one supplied tree | `mst.SteinerWalk(network, startID, endID)` | `*mst.WalkResult`, closed when endpoints agree, fixed-end otherwise; ordered repeated original IDs |

`SteinerTree` requires an undirected weighted graph with finite **nonnegative** edge weights, unlike ordinary Prim/Kruskal, which accept finite negative weights. It validates every original edge (even loops and unrelated islands), permits disconnected components with no required terminal, and publishes only one connected acyclic selected network. `SteinerRecomputePaths` is the default memory mode; `SteinerRetainPaths` retains O(kV) predecessor state to avoid selected-path SSSP recomputations. A network's `TotalWeight` counts **unique installed edges**; a walk's `Distance` counts **every actual passage**.

---

## Current contract documentation

| Layer                   | File                                         | Purpose                                                                                                                         |
|:------------------------|:---------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------|
| Tutorial                | [`docs/TUTORIAL.md`](docs/TUTORIAL.md)       | First guided entry into graph/matrix concepts, package roles, examples, and best practices.                                     |
| Core spec               | [`docs/CORE.md`](docs/CORE.md)               | Graph model, capability law, deterministic enumeration, locks, topology invariants.                                             |
| BFS spec                | [`docs/BFS.md`](docs/BFS.md)                 | Hop-distance math, BFS result semantics, partial results, weak components.                                                      |
| DFS spec                | [`docs/DFS.md`](docs/DFS.md)                 | Post-order semantics, cycle witnesses, DFS forest, topological sort.                                                            |
| Shortest paths spec     | [`docs/DIJKSTRA.md`](docs/DIJKSTRA.md)       | Full SSSP vs four target methods, consistency proofs, balanced potentials, exact arithmetic and runnable scenarios.             |
| MST spec                | [`docs/MST.md`](docs/MST.md)                 | Exact Prim/Kruskal, KMB Steiner approximation, tree-covering walks, original IDs, numeric/error contracts.                      |
| Flow spec               | [`docs/FLOW.md`](docs/FLOW.md)               | Max-flow/min-cut math, residual networks, Ford-Fulkerson, Edmonds-Karp, Dinic.                                                  |
| DTW spec                | [`docs/DTW.md`](docs/DTW.md)                 | Dynamic programming, numeric classification, windows, penalties and path recovery.                                              |
| DTW exact search        | [`docs/DTW_SEARCH.md`](docs/DTW_SEARCH.md)   | Bounded classification, fixed-length top-K, certified pruning, budgets and worker law.                                          |
| Fractional PAA          | [`docs/PAA.md`](docs/PAA.md)                 | Exact step-function geometry, output ownership, nearest-even arithmetic, wrappers and integration.                              |
| Symbolic SAX            | [`docs/SAX.md`](docs/SAX.md)                 | Canonical words, Euclidean MINDIST proof, versioned identity, errors and false-DTW-pruning counterexample.                      |
| Series primitives       | [`docs/SERIES.md`](docs/SERIES.md)           | Exact integer grids, causal transforms, result masks, numeric safeguards, index windows.                                        |
| Stable statistics       | [`docs/STATS.md`](docs/STATS.md)             | Welford/Chan, exact mean sum, scaled M2, fitted scaling, sentinels, immutable inference.                                        |
| Stable Ridge regression | [`docs/RIDGE.md`](docs/RIDGE.md)             | Positive-Lambda augmented QR, strict intercept policy, wide/collinear designs, nonzero tiny coefficients and frozen prediction. |
| Fitted covariance PCA   | [`docs/PCA.md`](docs/PCA.md)                 | Train-only orthogonal projection, checked covariance eigenpairs, numerical rank and leakage-safe reconstruction.                |
| Stable OLS regression   | [`docs/OLS.md`](docs/OLS.md)                 | Pivoted Householder QR, explicit intercept, physical coefficients, SSE, frozen Model and holdout pitfalls.                      |
| Grid spec               | [`docs/GRID_GRAPH.md`](docs/GRID_GRAPH.md)   | Grid/lattice graph modeling and pathfinding-oriented construction.                                                              |
| Matrix spec             | [`docs/MATRICES.md`](docs/MATRICES.md)       | Canonical dense/graph/statistics/factorization reference; [`MATRIX.md`](docs/MATRIX.md) is a navigation alias.                  |
| TSP spec                | [`docs/TSP.md`](docs/TSP.md)                 | Tour optimization, exact vs approximate methods, metric assumptions.                                                            |
| Engineering standard    | [`docs/lvlath_UES.md`](docs/lvlath_UES.md)   | Repository engineering standard and quality expectations.                                                                       |
| FAQ                     | [`docs/FAQ_&_TIPS.md`](docs/FAQ_%26_TIPS.md) | Troubleshooting, common pitfalls, usage tips.                                                                                   |
| Contribution guide      | [`CONTRIBUTING.md`](CONTRIBUTING.md)         | Branching, tests, linting, coverage, PR rules.                                                                                  |

---

## Quick start A: graph "lvlath" - shapes, components, cycles, matrix view

This is the first visual example: a graph is not only "nodes and edges"; it is a controllable topology that can be inspected, transformed, and projected into matrix form.

The old `lvlath` letter sketch is preserved, but the implementation style is updated to current error-returning constructors and deterministic package contracts.

```text
Six disconnected glyph components:

   l₁            v           l₂            a               t             h
  top                       top         tail──top         top           top
   │                         │                 │           |             │
 middle      left right   middle    middleL──circle  left-cross-right  middle──right
   │            \ /          │          │      │           │             │      │
 base──tail    center      base──tail  baseL──base       base──tail    base  baseR

Expected component count: 6
Expected cycle count:     1   // the internal cycle in the letter "a"
```

```go
package main

import (
	"context"
	"fmt"
	"sort"

	"github.com/lvlath/go/bfs"
	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
)

func main() {
	g, err := core.NewGraph()
	if err != nil {
		fmt.Println(err)
		return
	}

	// Static example data. Ignoring AddEdge errors is acceptable here only because
	// all vertices, weights, and capabilities are predetermined. In production,
	// check every returned error.

	// l₁
	_, _ = g.AddEdge("l1_top", "l1_middle", 0)
	_, _ = g.AddEdge("l1_middle", "l1_base", 0)
	_, _ = g.AddEdge("l1_base", "l1_tail_base_right", 0)

	// v
	_, _ = g.AddEdge("v_left", "v_center", 0)
	_, _ = g.AddEdge("v_center", "v_right", 0)

	// l₂
	_, _ = g.AddEdge("l2_top", "l2_middle", 0)
	_, _ = g.AddEdge("l2_middle", "l2_base", 0)
	_, _ = g.AddEdge("l2_base", "l2_tail_base_right", 0)

	// a: the only cyclic glyph.
	_, _ = g.AddEdge("a_tail_top_left", "a_top", 0)
	_, _ = g.AddEdge("a_top", "a_circle_middle", 0)
	_, _ = g.AddEdge("a_circle_middle", "a_circle_base", 0)
	_, _ = g.AddEdge("a_circle_base", "a_circle_base_left", 0)
	_, _ = g.AddEdge("a_circle_base_left", "a_circle_middle_left", 0)
	_, _ = g.AddEdge("a_circle_middle_left", "a_circle_middle", 0)

	// t
	_, _ = g.AddEdge("t_top", "t_cross", 0)
	_, _ = g.AddEdge("t_left", "t_cross", 0)
	_, _ = g.AddEdge("t_right", "t_cross", 0)
	_, _ = g.AddEdge("t_base", "t_cross", 0)
	_, _ = g.AddEdge("t_base", "t_tail_base_right", 0)

	// h
	_, _ = g.AddEdge("h_top", "h_middle", 0)
	_, _ = g.AddEdge("h_middle", "h_base", 0)
	_, _ = g.AddEdge("h_middle", "h_right", 0)
	_, _ = g.AddEdge("h_right", "h_base_right", 0)

	components, err := bfs.Components(context.Background(), g)
	if err != nil {
		fmt.Println(err)
		return
	}

	cycles := g.EdgeCount() - g.VertexCount() + components.Count
	fmt.Println("components:", components.Count)
	fmt.Println("cyclomatic:", cycles)

	// Deterministic induced view of the letter "a".
	aVertices := map[string]bool{
		"a_tail_top_left":     true,
		"a_top":               true,
		"a_circle_middle":     true,
		"a_circle_base":       true,
		"a_circle_base_left":  true,
		"a_circle_middle_left": true,
	}

	sub, err := core.NewGraph()
	if err != nil {
		fmt.Println(err)
		return
	}
	for id := range aVertices {
		_ = sub.AddVertex(id)
	}
	for _, e := range g.Edges() {
		if aVertices[e.From] && aVertices[e.To] {
			_, _ = sub.AddEdge(e.From, e.To, 0)
		}
	}

	mOpts, _ := matrix.NewMatrixOptions(
		matrix.WithUnweighted(),
		matrix.WithAllowLoops(),
	)

	am, err := matrix.NewAdjacencyMatrix(sub, mOpts)
	if err != nil {
		fmt.Println(err)
		return
	}
	degree, err := am.DegreeVector()
	if err != nil {
		fmt.Println(err)
		return
	}

	ids := make([]string, 0, len(am.VertexIndex))
	for id := range am.VertexIndex {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		fmt.Printf("degree[%s]=%.0f\n", id, degree[am.VertexIndex[id]])
	}
}
```
[![Go Playground](https://img.shields.io/badge/Go_Playground-Examplу_№1-blue?logo=go)](https://go.dev/play/p/K1gwwipdNhD)

What this demonstrates:

* `core` gives safe, deterministic topology construction and inspection.
* `bfs.Components` gives weak component discovery without hand-written map iteration.
* The cyclomatic number `|E| - |V| + components` exposes the single cycle in `a`.
* `matrix.NewAdjacencyMatrix` turns a selected subgraph into dense graph algebra.

---

## Quick start B: Hexagram - one weighted graph, many algorithmic views

The classic weighted Hexagram example is the compact demonstration of `lvlath` as a multi-package toolkit: BFS-style shells on an unweighted view, Dijkstra weighted routing, MST backbone extraction, and TSP preparation through a metric matrix.

```text
                               [A]
                              / | \
                  (C-H:9)   3/  |7 \4   (D-F:7)
        (B-G:7)         \   /   |   \   /         (E-G:9)
             [B]────3────[C]──5─┼────[D]────3────[E]
                \       / | \   |  /  | \       /
                6\    7/  | 5\  | /4  |  \6    /7
                  \   /   |   \ |/    |   \   /
                   [F]──3─┼────[G]──5─┼────[H]
                  /   \   |9  / | \   |8  /   \
                2/    6\  |  /4 |  \6 |  /5    \6
                /       \ | /   |   \ | /       \
             [I]────5────[J]──8─┼────[K]────1────[L]
        (I-G:8)         /   \   |8  /   \          (L-G:7)
                  (J-H:7)   2\  |  /3   (K-F:8)
                              \ | /
                               [M]
```

Use this kind of graph when you want one fixture that is complex enough for real algorithms but still small enough to reason about visually.

```go
package main

import (
	"context"
	"fmt"
	"sort"

	"github.com/lvlath/go/bfs"
	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dijkstra"
	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/mst"
)

func main() {
	g, err := core.NewGraph(core.WithWeighted())
	if err != nil {
		fmt.Println(err)
		return
	}

	edges := []struct {
		u, v string
		w    float64
	}{
		{"A", "C", 3}, {"A", "G", 7}, {"A", "D", 4},
		{"B", "C", 3}, {"B", "G", 7}, {"B", "F", 6},
		{"C", "F", 7}, {"C", "J", 9}, {"C", "G", 5}, {"C", "H", 9}, {"C", "D", 5},
		{"D", "F", 7}, {"D", "G", 4}, {"D", "K", 8}, {"D", "H", 6}, {"D", "E", 3},
		{"E", "G", 9}, {"E", "H", 7},
		{"F", "G", 3}, {"F", "K", 8}, {"F", "J", 6}, {"F", "I", 2},
		{"H", "G", 5}, {"H", "L", 6}, {"H", "K", 4}, {"H", "J", 7},
		{"I", "G", 8}, {"I", "J", 5},
		{"J", "G", 4}, {"J", "K", 8}, {"J", "M", 2},
		{"K", "G", 6}, {"K", "L", 1}, {"K", "M", 3},
		{"L", "G", 7},
		{"M", "G", 8},
	}
	for _, e := range edges {
		_, _ = g.AddEdge(e.u, e.v, e.w)
	}

	// BFS is unweighted by contract, so create an unweighted topology view.
	unweighted := core.UnweightedView(g)
	bfsResult, err := bfs.BFS(unweighted, "A")
	if err != nil {
		fmt.Println(err)
		return
	}

	oneHop := make([]string, 0)
	for id, depth := range bfsResult.Depth {
		if depth == 1 {
			oneHop = append(oneHop, id)
		}
	}
	sort.Strings(oneHop)
	fmt.Println("one-hop-from-A:", oneHop)

	// Dijkstra answers cost-sensitive routing.
	d, err := dijkstra.Dijkstra(g, "I", dijkstra.WithPathTracking())
	if err != nil {
		fmt.Println(err)
		return
	}
	distL, _ := d.DistanceTo("L")
	pathL, _ := d.PathTo("L")
	fmt.Println("I-to-L-cost:", distL)
	fmt.Println("I-to-L-path:", pathL)

	// MST gives the cheapest connected backbone.
	mstResult, err := mst.Kruskal(g)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("mst-weight:", mstResult.TotalWeight)

	// Matrix metric closure prepares an all-pairs distance surface for routing or TSP.
	opts, err := matrix.NewMatrixOptions(matrix.WithUndirected(), matrix.WithWeighted())
	if err != nil {
		fmt.Println(err)
		return
	}
	closure, err := matrix.BuildMetricClosure(g, opts)
	if err != nil {
		fmt.Println(err)
		return
	}
	_ = closure
	_ = context.Background()
}
```
[![Go Playground](https://img.shields.io/badge/Go_Playground-Examplу_№1-blue?logo=go)](https://go.dev/play/p/brRONKLM5-9)

---

## Quick start C: weighted service network - routing + flow + matrix policy

This is the practical infrastructure example: a directed weighted network where edge weights can represent latency/cost for routing, and another capacity graph can represent throughput for flow analysis.

```text
Weighted service graph for routing and matrix analysis:

 [edge] ────2────▶ [auth] ──1──▶ [profile] ──2──▶ [db]
    │                │             ▲              │
    │3               │4            │              │0  real zero-weight replication
    ▼                ▼             │              ▼
  [api] ───4────▶ [search] ───2────┘           [backup]
    │                │                            ▲
    │2               │3                           │1
    ▼                ▼                            │
 [cache] ◀──1──── [index] ────────5───────▶ [archive]

Key checks:
  - Dijkstra: cheapest route from edge to db.
  - Matrix: preserve zero-weight db -> backup.
  - Metric closure: publish all-pairs shortest distances.
  - Flow: compute capacity from ingress to storage tier on a related network.
```

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dijkstra"
	"github.com/lvlath/go/flow"
	"github.com/lvlath/go/matrix"
)

func main() {
	routes, err := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	if err != nil {
		fmt.Println(err)
		return
	}

	// Static example data only; check errors in production.
	_, _ = routes.AddEdge("edge", "auth", 2)
	_, _ = routes.AddEdge("edge", "api", 3)
	_, _ = routes.AddEdge("auth", "profile", 1)
	_, _ = routes.AddEdge("auth", "search", 4)
	_, _ = routes.AddEdge("api", "search", 4)
	_, _ = routes.AddEdge("api", "cache", 2)
	_, _ = routes.AddEdge("search", "profile", 2)
	_, _ = routes.AddEdge("search", "index", 3)
	_, _ = routes.AddEdge("index", "cache", 1)
	_, _ = routes.AddEdge("index", "archive", 5)
	_, _ = routes.AddEdge("profile", "db", 2)
	_, _ = routes.AddEdge("db", "backup", 0)
	_, _ = routes.AddEdge("archive", "backup", 1)

	dr, err := dijkstra.Dijkstra(routes, "edge", dijkstra.WithPathTracking())
	if err != nil {
		fmt.Println(err)
		return
	}
	cost, _ := dr.DistanceTo("db")
	path, _ := dr.PathTo("db")
	fmt.Println("route-edge-db-cost:", cost)
	fmt.Println("route-edge-db-path:", path)

	matrixOpts, err := matrix.NewMatrixOptions(
		matrix.WithDirected(),
		matrix.WithWeighted(),
		matrix.WithPreserveZeroWeights(),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	adj, err := matrix.NewAdjacencyMatrix(routes, matrixOpts)
	if err != nil {
		fmt.Println(err)
		return
	}
	db := adj.VertexIndex["db"]
	backup := adj.VertexIndex["backup"]
	zeroReplication, err := adj.Mat.At(db, backup)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("zero-replication-weight:", zeroReplication)

	closure, err := matrix.BuildMetricClosure(routes, matrixOpts)
	if err != nil {
		fmt.Println(err)
		return
	}
	_ = closure

	capacity, err := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	if err != nil {
		fmt.Println(err)
		return
	}
	_, _ = capacity.AddEdge("ingress", "auth", 8)
	_, _ = capacity.AddEdge("ingress", "api", 10)
	_, _ = capacity.AddEdge("auth", "profile", 5)
	_, _ = capacity.AddEdge("api", "search", 7)
	_, _ = capacity.AddEdge("profile", "storage", 6)
	_, _ = capacity.AddEdge("search", "storage", 8)
	_, _ = capacity.AddEdge("api", "cache", 4)
	_, _ = capacity.AddEdge("cache", "storage", 3)

	flowOpts := flow.DefaultOptions()
	maxFlow, residual, err := flow.Dinic(capacity, "ingress", "storage", flowOpts)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("max-flow-ingress-storage:", maxFlow)
	fmt.Println("residual-vertices:", residual.VertexCount())
}
```
[![Go Playground](https://img.shields.io/badge/Go_Playground-Examplу_№1-blue?logo=go)](https://go.dev/play/p/9uHM7u2HD88)

---

## Design notes

### Determinism

Determinism is not "sort something at the end." It is a cross-package contract.

* `core` owns deterministic graph surfaces: vertices, edges, neighbors.
* `bfs` preserves neighbor order and defines observable traversal state through result fields.
* `dfs` defines returned `Order` as finish order and publishes deterministic cycle witnesses.
* `dijkstra` uses deterministic frontier behavior and strict-improvement updates so equal-cost alternatives do not randomly overwrite witnesses.
* `mst` applies `(Weight, Edge.ID)` order to Kruskal selection, uses a deterministic Prim heap, expands KMB terminal links to original edge identities, and validates caller-owned trees before a covering walk.
* `flow` builds and consumes residual state through deterministic algorithm policy.
* `matrix` uses fixed row-major loops and deterministic graph-adapter publication; LU/QR pivot ties prefer the smallest original index.
* `stats` visits scalar samples in input order and matrix cells row-major; a caller fixes the merge-tree order when grouping floating-point moments.
* `ols` validates rows and features ascending, inherits QR deterministic pivot tie rules and predicts with original columns in the given order.
* `dtw` runs row-major DP and backtracks ties as diagonal, then vertical/up, then horizontal/left.
* `series` scans samples in index order, reports gaps and ranges in that order, and preserves length and validity masks.
* `tsp` uses matrix order, deterministic solver tie-breaks, explicit seed-governed local search, and result fields that report exactness, timeout, and approximation status.
* `builder` and `gridgraph` produce reproducible input topologies instead of hiding randomness inside examples.

The purpose is practical: stable examples, golden tests, CI logs, witness paths, component roots, DP paths, MST edges, tours, and failure classes.


### Error handling

Prefer:

```go
if errors.Is(err, dijkstra.ErrTargetNotFound) {
    // unknown target: not the same as known but unreachable
}
```

Avoid:

```go
if strings.Contains(err.Error(), "target") {
    // brittle and not part of the public contract
}
```

### Weighted vs unweighted algorithms

Do not choose by package popularity. Choose by mathematical objective.

```text
Question                                      Correct primitive
────────────────────────────────────────────────────────────────────────────
Fewest edges / hop layers?                     bfs
Deep traversal / finish order / cycles?        dfs
Cheapest non-negative route from one source?   dijkstra
Cheapest acyclic connected backbone?           mst.Kruskal / mst.Prim
Cheapest approximate terminal-only network?    mst.SteinerTree
Cover every installed tree edge and finish?    mst.SteinerWalk
Maximum feasible throughput?                   flow
All-pairs shortest distances?                  matrix.BuildMetricClosure
Dense topology/covariance/spectral features?   matrix
Stable mean/variance and partition merges?     stats
Fit full-rank factor regression?                ols
Fit regularized correlated/wide regression?     ridge
Train-only standardization / no refit?         stats
Exact timestamp grid / causal sample features? series
Symbolic Euclidean screening of windows?       sax
Temporal alignment with phase drift?           dtw
Closed tour through every vertex?              tsp
Generated lattice topology?                    gridgraph
Repeatable benchmark/test fixtures?            builder
```

Important distinctions:

* BFS minimizes edge count, not cost.
* Dijkstra minimizes non-negative path cost, not throughput.
* MST minimizes selected backbone cost, not route distance.
* Flow maximizes capacity, not shortest path.
* Metric closure creates a distance artifact, not original graph topology.
* DTW aligns time-warped sequences, not graph routes.
* `series` operates on supplied sample indices and exact nanosecond checks; it does not fill missing timestamp slots.
* `stats` aggregates finite observations and transforms features with parameters learned on training data only; it does not interpret series validity masks or refit.
* `ols` uses pivoted QR to fit finite, full-rank linear models and returns frozen original-unit coefficients; it never substitutes normal equations, drops rank-deficient factors or refits at inference.
* `ridge` uses loss-preserving augmented QR with mandatory positive Lambda and no intercept penalty; its physical Model remains immutable even when the original design is collinear or wide.
* `sax` quantizes the same fractional PAA geometry into symbolic words; its Euclidean MINDIST is not a metric on words or a certified bound for arbitrary DTW.
* TSP solves a closed tour over a complete finite cost model, not sparse reachability.

---

### Matrix semantics

`matrix` intentionally does not pretend that every dense numeric array means the same thing.

```text
normal adjacency:       0 can mean "no edge"
zero-preserving mode:   finite 0 can mean "real zero-cost edge"; +Inf means absence
metric closure:         cell means shortest-path distance, not an original edge
incidence matrix:       columns are edge identities, not vertex-pair weights
```

### Concurrency

`core` is thread-safe for graph storage and controlled mutation. Algorithm packages should be run against topology that is stable for the duration of the call unless the package explicitly documents otherwise. `matrix.Dense` does not contain locks; concurrent writes require external synchronization. Successfully built LU/QR/Cholesky factors retain detached immutable storage and support concurrent `Solve` with private per-call workspaces when RHS inputs remain stable.

---

## Choosing the right package

### By question type

| You ask                                                                       | Use                                                                | Why                                                                                          |
|:------------------------------------------------------------------------------|:-------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|
| "Which vertices are within `k` hops?"                                         | `bfs.BFS` with depth policy                                        | Hop layers are unweighted traversal.                                                         |
| "Which weak islands exist?"                                                   | `bfs.Components`                                                   | Component membership is not a route-cost problem.                                            |
| "What dependency order is valid?"                                             | `dfs.TopologicalSort`                                              | DAG ordering is finish-order semantics.                                                      |
| "Where is the cycle?"                                                         | `dfs.DetectCycles`                                                 | You need a deterministic witness, not all simple cycles.                                     |
| "What is the cheapest route from this source?"                                | `dijkstra.Dijkstra`                                                | Non-negative weighted shortest path.                                                         |
| "Which targets are outside a runtime budget?"                                 | `dijkstra.WithMaxDistance`                                         | Cutoff policy without deleting topology.                                                     |
| "Which links form the cheapest connected backbone?"                           | `mst.MinimumSpanningTree`, `mst.Kruskal`, `mst.Prim`               | MST/MSF solves acyclic connectivity, not routing.                                            |
| "What if the graph is disconnected but I still need per-component backbones?" | `mst.WithForest`                                                   | Forest mode is explicit, not a hidden fallback.                                              |
| "How do I connect only a required set of vertices?"                           | `mst.SteinerTree`                                                  | KMB approximates a nonnegative graph Steiner tree using original EdgeIDs.                    |
| "How do I cover all edges of the installed tree from A to B?"                 | `mst.SteinerWalk`                                                  | Iterative optimal covering of the given tree, with repeated passage IDs.                     |
| "What is the max source-to-sink capacity?"                                    | `flow.Dinic` or `flow.EdmondsKarp`                                 | Flow algorithms reason over residual capacity.                                               |
| "How do I validate exact sample cadence and missing slots?"                   | `series.Validate`                                                  | Signed-nanosecond grid checks; no interpolation.                                             |
| "How do I compute causal lag, differences, or returns?"                       | `series.Lag` / `Difference` / `SimpleReturns` / `LogReturns`       | Length-preserving values and validity masks.                                                 |
| "How do I compress a window without dropping fractional tails?"               | `paa.Transform` / `paa.TransformRange`                             | Exact fractional PAA, checked integer geometry and a detached result.                        |
| "How do I create comparable symbolic time-series signatures?"                 | `sax.NewEncoder` / `sax.Encode`                                    | Fixed-alphabet symbols, compatible policy hash, detached results.                            |
| "Can SAX MINDIST prune arbitrary DTW candidates?"                             | No                                                                 | Euclidean-only lower bound; shortlist retrieval must report Recall@K.                        |
| "How do I enumerate only complete index windows?"                             | `series.WindowRanges`                                              | Ordered half-open ranges without partial tails.                                              |
| "How do I compare two jittery sensor signatures?"                             | `dtw.Align`                                                        | Scalar DTW aligns timing drift.                                                              |
| "How do I align model-provided frame costs?"                                  | `dtw.AlignCostMatrix`                                              | Caller owns the local-cost surface.                                                          |
| "How do I align multivariate sequences?"                                      | `dtw.AlignMatrix`                                                  | Rows are time steps, columns are features.                                                   |
| "How do I calculate stable population/sample variance?"                       | `stats.Summarize`                                                  | Stable central moments and explicit DDOF denominator.                                        |
| "How do I combine independent streaming statistics?"                          | `stats.Moments.Update` / `Merge`                                   | Error-atomic, count-checked Welford/Chan partition reductions.                               |
| "How do I scale test features using only training data?"                      | `stats.FitStandardizer` followed by `Model.Transform`              | Frozen per-column means and scales, no inference refit.                                      |
| "How do I reduce correlated feature dimensions without a target y?"           | `pca.FitWithComponents` / `pca.FitStandardizedWithComponents`      | Frozen training PCA basis; k cannot exceed detected covariance rank.                         |
| "How do I check a sensor batch against a learned normal subspace?"            | `pca.Model.Reconstruct` + independently calibrated residual policy | Holdout reconstruction uses no batch refit; alerts are application policy.                   |
| "How do I fit stable factors when columns are duplicated or p>n?"             | `ridge.FitStandardized` / `ridge.Fit`                              | Explicit positive-Lambda slope penalty, unpenalized intercept, frozen physical coefficients. |
| "How do I fit reusable linear-factor coefficients with an intercept?"         | `ols.FitDefault` / `ols.FitStandardized`                           | Original-unit beta and b, training SSE, no heldout refit.                                    |
| "How do I force a physical zero intercept?"                                   | `ols.FitThroughOrigin` / `FitThroughOriginStandardized`            | No feature centering; b exactly +0 even when scaling.                                        |
| "How do I fit a full-rank tall calibration model?"                            | `matrix.SolveLeastSquares` / `FactorQR`                            | Pivoted Householder QR, rank refusal and residual diagnostics.                               |
| "How do I solve a reusable SPD coupling system?"                              | `matrix.FactorCholesky`                                            | Symmetry/SPD validation, measured condition and detached solutions.                          |
| "How do I compute topology features?"                                         | `matrix.NewAdjacencyMatrix`, `NewIncidenceMatrix`, `DegreeVector`  | Matrix artifacts preserve topology policy.                                                   |
| "How do I compute all-pairs route distances?"                                 | `matrix.BuildMetricClosure`                                        | Floyd-Warshall distance artifact with `+Inf` semantics.                                      |
| "How do I solve a complete inspection/delivery tour?"                         | `tsp.SolveMatrix`                                                  | TSP kernels consume matrix cost models.                                                      |
| "How do I solve TSP from graph topology?"                                     | `tsp.SolveGraph`                                                   | Graph is adapted at the facade boundary, then solved as a matrix problem.                    |
| "How do I get exact TSP for a small instance?"                                | `tsp.ExactHeldKarp` / `tsp.BranchAndBound` policy                  | Exactness is explicit and guarded.                                                           |
| "How do I get a metric symmetric approximation?"                              | `tsp.Christofides` + `tsp.BlossomMatch`                            | The `1.5` ratio requires exact matching and metric assumptions.                              |
| "How do I build repeatable graphs?"                                           | `builder` / `gridgraph`                                            | Generated fixtures should be deterministic.                                                  |

### By failure mode you need to avoid

| Risk                                                         | lvlath answer                                                                                           |
|:-------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------|
| Map-order flakiness in examples/tests                        | `core` deterministic surfaces and package tie-break laws.                                               |
| Invalid topology reaches algorithms                          | Graph capability validation and package-level input validators.                                         |
| BFS used for weighted routing                                | Separate BFS hop semantics from Dijkstra weighted semantics.                                            |
| DFS order misread as discovery order                         | DFS `Result.Order` is documented finish order.                                                          |
| Unknown target treated as unreachable                        | Dijkstra keeps target-domain errors separate from `+Inf` distances.                                     |
| Runtime route policy deletes graph structure                 | Dijkstra thresholds/cutoffs model the run without mutating topology.                                    |
| Disconnected MST silently becomes forest                     | `mst.WithForest` is explicit; strict MST returns disconnection.                                         |
| Terminal-only optimization confused with full MST            | `mst.SteinerTree` uses required terminals and original edges; it is an approximation, not an exact MST. |
| Network cost mistaken for inspection-route cost              | `SteinerResult.TotalWeight` counts unique links; `WalkResult.Distance` counts actual passages.          |
| Repeated walk EdgeIDs removed as duplicates                  | `SteinerWalk` repeats original IDs by design while covering every selected tree edge.                   |
| Non-finite MST weights corrupt sorting/heap behavior         | MST rejects `NaN`, `+Inf`, `-Inf` before kernel execution.                                              |
| Flow output hides residual state                             | Flow result includes residual-network semantics.                                                        |
| DTW path requested from rolling memory                       | DTW memory/path policy is explicit; path needs full accumulated storage.                                |
| Missing timestamps silently treated as samples               | `series.Validate` reports whole-slot gaps; no values or time slots are synthesized.                     |
| PAA fractional remainder dropped or padded with zeros        | `paa.Transform` uses integer exact-overlap geometry; neither truncation nor padding occurs.             |
| SAX breakpoint labels mistaken for encoding identity         | `EncodingID` hashes actual threshold content and numeric policy; `BreakpointID` is audit-only.          |
| Euclidean SAX MINDIST incorrectly used to prune DTW          | Do not prune; the two distance objectives differ and a positive MINDIST may accompany zero DTW.         |
| Warmup zero mistaken for a computed zero                     | `series.TransformResult.Valid` marks computed cells; retain it with `Values`.                           |
| DTW over-warps a signal                                      | Use window and slope penalty deliberately.                                                              |
| Feature scale dominates multivariate DTW                     | Normalize before `dtw.AlignMatrix`; it uses squared L2 row cost.                                        |
| High-offset variance vanishes or intermediate `M2` overflows | `stats.Moments` uses scaled Welford/Chan; exact binary sum preserves the final mean.                    |
| Holdout data refits a normalizer or constant features vanish | `stats.Standardizer` freezes training parameters; `UnitScaleConstant` preserves novel deviations.       |
| PCA loadings drift after observing a holdout batch           | `pca.Model.Transform` retains train-only origin, scales and loading basis; it does not refit.           |
| Unconverged Jacobi result or negative eigenvalue is hidden   | PCA rejects failed eigenpair/PSD checks and never silently substitutes ThinSVD.                         |
| Zero-weight edge disappears in matrix adjacency              | Use zero-preserving matrix policy; absence becomes `+Inf`.                                              |
| Metric closure exported as fake topology                     | Matrix refuses metric-closure export as ordinary graph topology.                                        |
| TSP receives sparse unresolved `+Inf` matrix                 | Use metric closure only when indirect travel is domain-valid; final kernels need finite completeness.   |
| Christofides ratio claimed with greedy matching              | `tsp.BlossomMatch` is required for the formal `1.5` ratio.                                              |
| Branch-and-Bound timeout is mistaken for optimality          | Inspect TSP `Result.TimedOut` and TSP `Result.Optimal`.                                                 |
| Local-search TSP treated as proof                            | 2-opt/3-opt improve tours; they do not certify global optimum.                                          |
| Randomized tests/benchmarks drift                            | Use fixed seeds or deterministic `builder` fixtures.                                                    |
| Docs describe fictional APIs                                 | `{package}/doc.go`, `docs/{PACKAGE}.md`, examples, and tests must stay synchronized.                    |

---

## Documentation and contribution workflow

Documentation quality is part of the project standard. A package is considered mature when it has:

1. `{package}/doc.go` with implemented API, options, errors, and complexity.
2. `docs/{PACKAGE}.md` with math, proofs/invariants, pseudocode, diagrams, examples, and pitfalls.
3. Scenario-driven examples in `examples/` or `example_test.go`.
4. Validation, medium, and special tests.
5. Benchmarks where performance matters.
6. CI passing: `go test`, `go vet`, and `golangci-lint`.

For contribution rules, branch flow, tests, linting, and coverage expectations, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## License

This project is licensed under **GNU Affero General Public License v3.0 only** (`AGPL-3.0-only`).

Important for server-side and enterprise usage: AGPL-3.0 requires that if you run this software on a server and allow users to interact with it over a network, you must make the corresponding source code available under the license terms.

If you want to use `lvlath` in a proprietary environment without AGPL source-disclosure obligations, contact the maintainer for commercial licensing terms:

**katalvlaran@gmail.com**

---

## Support

* GitHub Issues: `github.com/lvlath/go/issues`
* API reference: `pkg.go.dev/github.com/lvlath/go`
* Maintainer contact: `katalvlaran@gmail.com`

---

**lvlath** - deterministic graph algorithms, stable statistics, fitted feature scaling, dense algebra, and engineering-grade contracts for Go developers who want reproducible answers instead of accidental behavior.
