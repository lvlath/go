<!--
  SPDX-License-Identifier: AGPL-3.0-only
  Copyright (C) 2025-2026 katalvlaran

  File: docs/FAQ_&_TIPS.md
  Package reference: core, matrix, series, stats, paa, sax, ols, ridge, pca, hmm and the wider lvlath package family
  lvlath - FAQ & Tips

  Purpose:
    This document is the practical troubleshooting and best-practices guide for
    lvlath users and contributors. It explains common mistakes, package-selection
    decisions, deterministic behavior, error classification, graph/matrix semantics,
    testing practices, causal time-series contracts, benchmark pitfalls,
    and documentation synchronization rules.
    It is aligned with the lvlath Universal Engineering Standard (UES).

  Contract status:
    This file is guidance, not a replacement for package doc.go contracts. When in
    doubt, package doc.go and current source code are authoritative. This document
    must not mention fictional APIs or stronger guarantees than the implemented
    packages provide.

  Scope:
    - Package selection and algorithm-choice guidance.
    - Common troubleshooting for core, bfs, dfs, dijkstra, matrix, flow, dtw, tsp,
      gridgraph, builder, series, stats, sax, and examples.
    - Testing, benchmarking, documentation, and contribution best practices.
    - Certified PCA covariance factors, leakage-safe reconstruction and spectral diagnostics.
    - Causal hidden-state filtering versus future-conditioned smoothing and MAP-EM training.

  License:
    lvlath is licensed under AGPL-3.0-only. See ../LICENSE.
-->

# FAQ & Tips

> Practical engineering guidance for using and evolving **lvlath** without breaking contracts.

This file answers the questions that usually appear after reading `README.md` and `docs/TUTORIAL.md`: which package should I use, what does this result field mean, why does this error happen, why is `+Inf` intentional, why did a traversal skip something, and how should I test or benchmark a contract-sensitive algorithm?

For formal package contracts, read `{package}/doc.go`. For theory, diagrams, and algorithm walkthroughs, read `docs/{PACKAGE}.md`.

---

## Table of Contents

1. [Choosing the right package](#1-choosing-the-right-package)
2. [Core graph modeling](#2-core-graph-modeling)
3. [BFS FAQ](#3-bfs-faq)
4. [DFS FAQ](#4-dfs-faq)
5. [Dijkstra FAQ](#5-dijkstra-faq)
6. [Matrix FAQ](#6-matrix-faq)
7. [Flow FAQ](#7-flow-faq)
8. [DTW FAQ](#8-dtw-faq)
9. [MST FAQ](#9-mst-faq)
10. [TSP FAQ](#10-tsp-faq)
11. [Series FAQ](#11-series-faq)
12. [Stats FAQ](#12-stats-faq)
13. [PAA FAQ](#13-paa-faq)
    13a. [SAX FAQ](#13a-sax-faq)
    13b. [OLS FAQ](#13b-ols-faq)
    13c. [Ridge FAQ](#13c-ridge-faq)
    13d. [PCA FAQ](#13d-pca-faq)
    13e. [HMM FAQ](#13e-hmm-faq--causal-regimes-smoothing-decoding-and-em)
14. [Builder and gridgraph tips](#14-builder-and-gridgraph-tips)
15. [Errors and diagnostics](#15-errors-and-diagnostics)
16. [Determinism and concurrency](#16-determinism-and-concurrency)
17. [Testing tips](#17-testing-tips)
18. [Benchmark tips](#18-benchmark-tips)
19. [Documentation tips](#19-documentation-tips)
20. [Quick recipes](#20-quick-recipes)

---

## 1. Choosing the right package

### Q1.1. I have a graph. Which package should I start with?

Start with the question, not the algorithm name.

| Question                                                                     | Use                   | Reason                                                                                |
|:-----------------------------------------------------------------------------|:----------------------|:--------------------------------------------------------------------------------------|
| “How many hops away is this vertex?”                                         | `bfs`                 | BFS minimizes edge count.                                                             |
| “Which vertices are in the same weak island?”                                | `bfs.Components`      | Weak membership ignores direction for component grouping.                             |
| “What is a deterministic deep traversal?”                                    | `dfs`                 | DFS gives post-order and forest-style structure.                                      |
| “Does this graph contain a cycle?”                                           | `dfs.DetectCycles`    | Returns deterministic witness cycles.                                                 |
| “What order can I execute a DAG?”                                            | `dfs.TopologicalSort` | Topological order is a directed acyclic graph contract.                               |
| “What is the cheapest route by weight?”                                      | `dijkstra`            | Dijkstra minimizes non-negative total cost.                                           |
| “What is the cheapest connected backbone?”                                   | `mst`                 | MST solves minimum connected spanning structure.                                      |
| “What is the maximum throughput?”                                            | `flow`                | Flow solves capacity-constrained source-to-sink movement.                             |
| “How do I turn graph topology into numeric features?”                        | `matrix`              | Adjacency, incidence, degree vector, metric closure, algebra.                         |
| “How do I model a grid map?”                                                 | `gridgraph`           | Generates lattice topology instead of manual wiring.                                  |
| “How do I build repeatable fixtures?”                                        | `builder`             | Deterministic graph/data generation.                                                  |
| “How do I compress a fixed-size scalar time-series window?”                  | `paa`                 | Exact fractional PAA averages with rational boundaries and no dropped tail.           |
| “How do I encode a fixed-length symbolic signature?”                         | `sax`                 | Fractional PAA + right-bin quantization, compatible Euclidean MINDIST.                |
| “How do I align two time series?”                                            | `dtw`                 | Dynamic Time Warping handles local timing drift.                                      |
| “How do I compute robust moments or merge workers?”                          | `stats`               | Welford/Chan moments, exact mean sum, scaled M2.                                      |
| “How do I normalize heldout features without refit?”                         | `stats`               | Immutable fitted model over matrix columns.                                           |
| “How do I estimate linear factor coefficients and reuse them?”               | `ols`                 | Full-rank QR, original-unit beta/intercept, immutable prediction.                     |
| “How do I fit correlated or wide linear factors with explicit L2 shrinkage?” | `ridge`               | Augmented QR, positive Lambda, unpenalized intercept and frozen physical predictions. |
| “How do I infer a hidden operating regime from time-ordered observations?”   | `hmm`                 | Typed causal filtering, smoothing, Viterbi, and independent-sequence EM.              |
| “How do I reduce multivariate dimensions without a response variable?”       | `pca`                 | Validated covariance spectrum and a frozen loadings model for holdout inference.      |
| “How do I visit all cities in a route?”                                      | `tsp`                 | Tour construction/improvement/exact-small-instance routing.                           |

### Q1.2. How do I avoid using the wrong algorithm?

Keep this mental table:

```text
minimize hops                 -> BFS
inspect depth structure       -> DFS
prove/locate cycle            -> DFS cycle witness
order DAG tasks               -> DFS topological sort
minimize weighted cost        -> Dijkstra
connect all cheaply           -> MST
maximize throughput           -> Flow
compute all-pairs costs       -> Matrix metric closure
validate grid/causal features -> Series
stream/merge stable moments   -> Stats
fit training-only scaling     -> Stats
fit full-rank linear factors  -> OLS (pivoted QR, no refit)
fit correlated/wide factors   -> Ridge (positive Lambda, unpenalized intercept)
compress latent features      -> PCA (train-only covariance basis; no target)
infer hidden temporal regime  -> HMM FilterStep (causal; no future observations)
retrospective regime audit    -> HMM Smooth / Decode (future-conditioned)
encode symbolic SAX words     -> SAX
align time-warped signals     -> DTW
build a closed tour           -> TSP
```

The most common mistake is using BFS on a weighted routing problem. BFS answers “fewest edges,” not “cheapest total weight.”

---

### Q1.3. My input is a time series: should I start with `series` or `dtw`?

Use `series` when the immediate question concerns exact integer-nanosecond
cadence, missing whole grid slots, lagged observations, finite differences,
relative/logarithmic returns, or complete sample-index windows. Its transforms
**preserve the length** of the supplied array and publish a `Valid` mask.

Use `dtw` when you need to compare patterns despite permitted local timing
stretching under a specified alignment cost and window. It does not replace
exact-grid checking. These packages are independent: an application may choose
to validate a series first, but neither package implicitly resamples or fills
gaps for the other. See [`SERIES.md`](SERIES.md) and [Series FAQ](#11-series-faq).

---

## 2. Core graph modeling

### Q2.1. Why does graph construction use options?

Graph options define the mathematical universe of the graph.

```go
g, err := core.NewGraph(
    core.WithDirected(true),
    core.WithWeighted(),
    core.WithLoops(),
)
```

This is not syntax decoration. It means:

* edges are directed by default;
* non-zero weights are legal;
* self-loops are legal;
* the graph will reject operations that violate other capability rules.

### Q2.2. Should I call `AddVertex` before `AddEdge`?

Usually no. `AddEdge` can create endpoints as part of graph construction. For examples and fixtures, prefer direct edge construction when vertices exist only because edges mention them.

Good for static examples:

```go
_, _ = g.AddEdge("gateway", "auth", 0)
_, _ = g.AddEdge("auth", "profile", 0)
```

Use explicit `AddVertex` when an isolated vertex matters:

```go
_ = g.AddVertex("isolated-zone")
```

### Q2.3. When is it acceptable to ignore `AddEdge` errors?

Only in tightly controlled examples with predetermined data and a comment explaining the shortcut.

```go
// Static tutorial data only. In production, check every AddEdge error.
_, _ = g.AddEdge("A", "B", 1)
```

In tests and production code, check errors unless the test is not about graph construction and the impossibility of error is obvious and already guarded by the graph options.

### Q2.4. Why are deterministic graph surfaces important?

Many valid graph answers are not unique. Without stable ordering, the same graph can produce different but still valid BFS paths, DFS orders, equal-cost Dijkstra witnesses, or MST edge orders.

`lvlath` treats order as a contract. If a package returns ordered output, the source of that order should be documented.

### Q2.5. Should I mutate a graph while an algorithm is running?

No, not if you need reproducible results. `core` protects its internal storage, but algorithm packages generally read progressively rather than snapshotting the full graph. Concurrent topology mutation can change what the algorithm sees.

Use a clone/view or run algorithms against stable topology.

---

## 3. BFS FAQ

### Q3.1. Why does BFS reject weighted graphs?

Because BFS computes hop distance:

```text
dist(s, v) = minimum number of edges from s to v
```

If weights represent latency, distance, cost, or risk, BFS is the wrong algorithm. Use `dijkstra` for non-negative weighted costs.

When you intentionally want hop distance over a weighted topology, use an unweighted view rather than rewriting the original graph.

### Q3.2. What is the difference between `Visited` and `Order`?

In BFS:

```text
Visited = discovered and enqueued
Order   = dequeued and processed
```

On a successful full traversal they are often close, but on cancellation or hook failure they can differ. A vertex may be discovered and queued but not yet processed.

This distinction is intentional and useful for diagnostics.

### Q3.3. Why does `PathTo` return one path, not all shortest paths?

BFS records one deterministic parent tree. `PathTo(dst)` reconstructs one shortest-hop witness from that tree. It does not enumerate all shortest paths.

If your domain needs all shortest paths, that is a different algorithmic surface and should not be inferred from `bfs.Result.Parent`.

### Q3.4. Why did BFS skip some nodes?

Common causes:

1. The graph is directed and the relation does not point outward from the current vertex.
2. `WithMaxDepth` stopped expansion at the boundary.
3. `WithFilterNeighbor` rejected the relation.
4. Traversal was canceled or a hook returned an error.
5. The target is in a different weak component.

Check `Depth`, `Visited`, `Order`, and `Skipped` according to the package contract.

### Q3.5. How should I test BFS cancellation?

Use deterministic cancellation from a hook. Avoid timeouts in tests and examples.

Good pattern:

```go
ctx, cancel := context.WithCancel(context.Background())
defer cancel()

res, err := bfs.BFS(
    g,
    "gateway",
    bfs.WithContext(ctx),
    bfs.WithOnVisit(func(id string, depth int) error {
        if id == "ledger" {
            cancel()
        }
    return nil
    }),
)

_ = res
_ = err
```

Then assert partial-result fields exactly as the BFS contract defines.

---

## 4. DFS FAQ

### Q4.1. Why is `dfs.Result.Order` not the order I expected?

Because returned DFS order is finish order, not discovery order.

```text
Discovery: vertex is entered.
Finish:    all reachable descendants from that vertex are completed.
Order:     appended on finish.
```

For a tree:

```text
       A
      / \
     B   C
    / \
   D   E
```

Discovery might be:

```text
A, B, D, E, C
```

Finish order is:

```text
D, E, B, C, A
```

Use an `OnVisit` hook if you need discovery order.

### Q4.2. Why does DFS clear or invalidate order on partial failure?

A partial finish order can be mathematically misleading. DFS may have entered vertices whose descendants were not fully explored. Package docs define which fields remain safe on partial results.

Do not treat partial DFS `Order` as a valid topological or post-order artifact.

### Q4.3. Does cycle detection return all cycles?

No, unless a package explicitly documents exhaustive enumeration. Cycle detection generally returns deterministic witness cycles: enough to prove cyclicity and locate a loop.

This avoids pretending that expensive exhaustive simple-cycle enumeration is a cheap default.

### Q4.4. Why does topological sort require a directed graph?

Topological order is defined over directed acyclic graphs. For an edge `u -> v`, `u` must appear before `v`. Undirected relations do not define this dependency direction.

If your graph contains undirected edges, decide whether they represent missing direction, bidirectional dependency, or a different problem.

---

## 5. Dijkstra FAQ

### Q5.1. When should I use Dijkstra?

Use `lvlath/dijkstra` when the mathematical objective is a minimum accumulated
cost from one source and every graph-edge weight is finite and non-negative.

Good domains include:

* physical route distance;
* network latency;
* monetary or energy cost;
* non-negative risk scores;
* service radius and budget limits;
* deterministic route witnesses.

Do not use Dijkstra when finite negative edge weights are part of the model.
Negative weights violate the finalization argument on which the algorithm relies.

Also do not store `NaN`, `+Inf`, or `-Inf` as `core.Graph` edge weights.
Graph-edge weights belong to the finite storage domain.

`+Inf` is used elsewhere:

* in `Result.Distances`, as known-but-unreachable;
* in `MaxDistance`, as “no cutoff”;
* in `InfEdgeThreshold`, as “no finite edge is blocked”.

Those meanings do not make `+Inf` a valid graph-edge weight.

### Q5.2. What is the difference between unknown and unreachable?

Target-domain state and path-tracking state are separate axes.

| Target state      | Meaning                 | `DistanceTo`        | `HasPathTo`         |
|:------------------|:------------------------|:--------------------|:--------------------|
| Empty target ID   | Invalid query input     | `ErrEmptyTargetID`  | `ErrEmptyTargetID`  |
| Unknown target    | Absent from `Distances` | `ErrTargetNotFound` | `ErrTargetNotFound` |
| Known reachable   | Finite distance         | finite value, nil   | true, nil           |
| Known unreachable | Present with `+Inf`     | `+Inf`, nil         | false, nil          |

`PathTo` additionally depends on witness state:

| Witness state                                      | `PathTo` result           |
|:---------------------------------------------------|:--------------------------|
| `Prev == nil`                                      | `ErrPathTrackingDisabled` |
| Known target with `+Inf`, tracking enabled         | `ErrNoPath`               |
| Valid source-anchored predecessor chain            | deterministic path        |
| Broken, cyclic, or out-of-domain predecessor chain | `ErrNoPath`               |

Do not collapse these cases into one generic “route failed” state.

### Q5.3. Why does Dijkstra publish `+Inf`?

`+Inf` is the canonical **result-domain** value for a vertex that is known to the
result but unreachable under the effective topology and traversal policy.

Use:

```go
cost, err := result.DistanceTo(targetID)
if err != nil {
    // Handle empty or unknown target.
}

if math.IsInf(cost, 1) {
    // The target is known but unreachable under the active policy.
}
```

This `+Inf` is created by the algorithm's result model. It is not copied from an
infinite graph-edge weight.

### Q5.4. Why did `PathTo` fail even though `DistanceTo` worked?

A distance and a path witness require different stored information.

`DistanceTo` reads only `Distances`.

`PathTo` additionally requires:

1. a non-nil `Prev` map;
2. a finite target distance;
3. a predecessor chain that reaches `SourceID`;
4. every predecessor to remain inside `Distances`;
5. no repeated predecessor vertex.

Typical outcomes:

* `ErrPathTrackingDisabled`:
  the run did not enable path tracking;
* `ErrNoPath`:
  the target is unreachable or the stored predecessor chain is invalid;
* `ErrTargetNotFound`:
  the target is not part of the result domain.

Use `WithPathTracking()` when retaining a full `Result`, or use
`ShortestPathTo(...)` when the consumer needs one path and its distance.

### Q5.5. How do wall thresholds and maximum-distance cutoffs differ?

They operate at different mathematical levels.

```text
InfEdgeThreshold:
  edge-local policy

  skip one finite edge when:
    edge.Weight >= threshold

  Typical meaning:
    degraded link, blocked road, prohibited transfer, excessive local risk

MaxDistance:
  path-global policy

  skip a candidate route when:
    accumulatedDistance > max

  Typical meaning:
    service radius, total budget, SLA boundary, search horizon
```

Both policies are inclusive at their boundary:

* `weight == InfEdgeThreshold` is blocked;
* `distance == MaxDistance` is admissible.

Neither policy mutates graph topology.

Positive infinity disables the corresponding finite restriction:

* `MaxDistance = +Inf` means no accumulated-distance cutoff;
* `InfEdgeThreshold = +Inf` means no valid finite edge is blocked.

### Q5.6. Why is equal-cost path selection stable?

The package combines three deterministic laws:

1. relations are consumed in `core.Graph.Neighbors(u)` order;
2. heap entries are ordered by `(candidateDistance, vertexID)`;
3. predecessor state changes only on strict improvement.

Therefore:

```text
candidate < current:
  update distance and predecessor

candidate == current:
  preserve the existing predecessor
```

The package returns one deterministic witness, not all equal-cost shortest paths.

Stability assumes the same graph state, source, package version, and options.
Concurrent graph mutation is outside the reproducibility contract.

### Q5.7. What does `ErrDistanceOverflow` mean?

`ErrDistanceOverflow` means that:

* the current finalized distance was finite;
* the edge weight was finite;
* but their sum exceeded the representable finite `float64` range.

Example:

```text
currentDistance = math.MaxFloat64
edgeWeight      = math.MaxFloat64
candidate       = +Inf
```

This is arithmetic overflow, not ordinary unreachability.

The package therefore returns:

```text
nil Result + ErrDistanceOverflow
```

It does not publish the overflowed route as a `+Inf` unreachable result.

A finite `MaxDistance` may safely reject an already out-of-policy candidate before
performing the overflowing addition.

### Q5.8. Can I mutate `Result.Distances` or `Result.Prev`?

Yes. Published result maps are caller-owned.

However, mutation changes the meaning of that `Result`.

Safe patterns:

* read the result without mutation;
* call `Clone()` before creating an independently mutable variant;
* keep result mutation synchronized if shared across goroutines.

`PathTo` defensively rejects:

* broken chains;
* predecessor cycles;
* predecessors absent from `Distances`.

It returns `ErrNoPath` rather than returning a partial witness or looping forever.

### Q5.9. Does `DistanceTo(...)` stop Dijkstra when the target is reached?

No.

The convenience wrapper:

```go
dijkstra.DistanceTo(graph, sourceID, targetID, options...)
```

runs the same canonical single-source Dijkstra computation and then performs an
O(1) target lookup on the completed `Result`.

It is a point-query convenience API, not a target-directed early-exit kernel.

Use it to reduce caller boilerplate, not as a different asymptotic algorithm.

---

## 6. Matrix FAQ

### Q6.1. Why not just use `[][]float64`?

Because `[][]float64` scatters row allocations and carries no contract for shape, numeric policy, ownership, views, graph semantics, or error classification.

`matrix.Dense` uses row-major storage:

```text
offset(i, j) = i * cols + j
```

That gives predictable memory layout and lets kernels implement fast paths while preserving the safe `Matrix` interface.

### Q6.2. Why does `NewDense(0, n)` succeed?

Zero-shape matrices are valid structural results.

```text
0×0: empty matrix
0×N: no rows, N columns
N×0: N rows, no columns
```

They are useful in feature extraction, empty graph transforms, and degenerate pipeline stages. `At(0,0)` should still fail with an out-of-range error because there is no addressable cell.

### Q6.3. Why does matrix code reject `NaN` or `Inf`?

By default, dirty floats are rejected because they poison downstream algebra and statistics. `+Inf` is allowed only under explicit distance/absence policy, such as metric closure or allow-inf distance modes.

For telemetry or external data, sanitize first:

```go
clean, err := matrix.ReplaceInfNaN(raw, 0)
if err != nil {
return err
}
clean, err = matrix.Clip(clean, 0, 100)
if err != nil {
return err
}
```

### Q6.4. Why did my zero-weight edge disappear?

Classic adjacency often uses `0` as “no edge.” That conflicts with real zero-weight edges.

If a finite zero is a real edge, use a matrix policy that preserves zero weights. In zero-preserving weighted adjacency, absence is represented by `+Inf`.

```go
opts, err := matrix.NewMatrixOptions(
matrix.WithDirected(),
matrix.WithWeighted(),
matrix.WithPreserveZeroWeights(),
)
```

### Q6.5. What is the difference between adjacency and incidence?

```text
Adjacency: V × V
  cell means relation from vertex i to vertex j.
  Good for dense lookup, degree-like row scans, matrix algebra.

Incidence: V × E
  column means one edge identity.
  Good when edge identity, signs, loops, or structural edge participation matter.
```

Do not use dense adjacency as an edge ledger when parallel-edge identity matters.

### Q6.6. Why can’t I export metric closure back to a graph?

Metric closure is not original topology. It stores shortest-path distances.

```text
Original edge A -> C may not exist.
Metric closure D[A,C] may be finite through A -> B -> C.
```

Exporting that as an edge would invent topology. Refusal is correct.

### Q6.7. When should I use `View` vs `Induced`?

Use `View` when you intentionally want shared storage:

```text
View.Set writes into the base Dense.
```

Use `Induced` when the submatrix needs independent lifetime or safe mutation.

`View` now satisfies the `Matrix` interface through `Clone()`: the **view itself** still aliases its parent for `At`/`Set`; `Clone()` publishes a detached owning matrix containing only the selected window. Never infer that simply assigning a View to `matrix.Matrix` snapshots its source.

### Q6.8. Why does covariance require at least two rows, even with no features?

`Covariance` and `Correlation` are **sample** statistics with denominator `rows-1`. Their canonical validation rejects every `rows<2` matrix with `ErrDimensionMismatch`, including `1×0` and `0×0`. A `2×0` (or taller, zero-column) input produces an owning `0×0` result and empty means/deviation slices. This fixes validation precedence before the zero-feature fast path; it does not change the `NewDense` zero-shape law.

### Q6.9. Why does `Covariance([1e308,1e308])` not overflow?

A raw floating sum overflows **before** division, even though the mean `1e308` and variance zero are representable. `matrix.Covariance` uses scaled, compensated centering. Likewise, `[-1e154,0,1e154]` has finite sample variance near `1e308`, although the Gram sum before dividing by `n-1` could overflow. **Only a truly unrepresentable final output is an `ErrOverflow`**, never a successful `+Inf` covariance.

The implementation keeps one upper-triangle accumulator and writes its value into both symmetric positions, preserving bit-identical symmetry. Exceptional cross-moment cancellation is recovered from the original exact binary64 inputs using a `math/big` path. There is no universal correctly-rounded guarantee for every finite input, so numerical oracles remain important.

### Q6.10. Can `Correlation` succeed when `Covariance` fails?

Yes. Correlation is dimensionless and is computed from **normalized centered moments**, not by dividing a materialized covariance matrix. Two perfectly anticorrelated channels of magnitude `1e200` have Pearson coefficient `-1`, even if the covariance exceeds finite binary64. Their standard deviations may still be representable. For a constant feature, the entire correlation row and column—including the diagonal—is exactly zero; for nonconstant features the diagonal is one.

### Q6.11. Does `CenterColumns` always return a detached matrix?

For an input containing addressable elements, yes: it publishes a new owning Dense and detached means. For valid zero-element shapes, the historical no-op policy returns the **original Matrix instance**; only the means slice is fresh. The same aliasing exception applies to `NormalizeRowsL2`. Do not mutate the return as though it were independent when `Rows()==0 || Cols()==0`.

The centered cells may use more accurate internal mean information than the rounded public means. Recomputing `x - means[j]` in your application can lose an ULP-scale deviation.

### Q6.12. How do `ErrNaNInf` and `ErrOverflow` differ?

`ErrNaNInf` means a finite-only operation received invalid **input data**. `ErrOverflow` means the required arithmetic result or diagnostic could not be published as finite binary64. Neither means "missing edge" or "unreachable graph vertex". The `NormalizeRowsL2` error for a truly overflowing norm matches `ErrOverflow` and, for compatibility, the historical `ErrNaNInf` classification under `errors.Is`; do not apply that joined classification to unrelated operations.

### Q6.13. Which solver should I use: LU, QR or Cholesky?

| Input problem                        | API                                         | Refusal instead of fallback                                       |
|:-------------------------------------|:--------------------------------------------|:------------------------------------------------------------------|
| Square, full-rank linear system      | `FactorLU` → `Solve`                        | `ErrRankDeficient` (exact singular LU also matches `ErrSingular`) |
| Tall, full-column-rank least squares | `FactorQR` → `Solve` or `SolveLeastSquares` | `ErrBadShape` for wide; `ErrRankDeficient` for deficient          |
| Symmetric positive-definite system   | `FactorCholesky` → `Solve`                  | `ErrAsymmetry` vs `ErrNotPositiveDefinite`                        |

`LUDecompose` is historical **unpivoted** LU; a zero first pivot may reject a solvable square matrix. Historical `QRDecompose` returns a square-only `A≈QᵀR` convention. Neither should be mistaken for the new reusable factor structs. New QR stores compact Householder reflectors, chooses stable original-column pivots and **does not** form normal equations or a full `Q` in `Solve`.

### Q6.14. How do I interpret `FactorOptions`?

`RankTolerance==0` chooses relative `2^-52*max(rows,cols)`; a positive finite value sets the relative pivoted-diagonal cutoff. Exact equality with the threshold counts as deficiency. `ConditionLimit==0` disables *hard rejection*, not condition measurement; a positive finite value must be at least one. Only an estimate **strictly larger** than the limit triggers `ErrIllConditioned`.

These are numerical policies, not a way to make duplicate predictors independent. Never weaken rank tolerance until a singular model merely happens to return coefficients.

### Q6.15. What does `ConditionEstimate` measure?

For LU and Cholesky it measures `kappa_1(A)` of the accepted system; for QR it measures `kappa_1(R)` of the triangular factor. It computes matrix 1-norms and solves each unit basis RHS to estimate the inverse 1-norm, with **O(n³)** additional work. It is not a diagonal ratio, an exact algebraic certificate, or `kappa_2`.

`ResidualNorms[k]` is the original-coordinate Euclidean norm of `A*x_k-b_k`, computed using an exact binary-input accumulation and a high-precision final square root. For Cholesky with accepted asymmetry it uses the **accepted symmetric midpoint** rather than the raw matrix. A small residual alone does not certify small coefficient error when the system is ill-conditioned.

### Q6.16. Can I share a factor between goroutines?

Yes, a successfully constructed `LUFactor`, `QRFactor`, or `CholeskyFactor` retains private immutable factor data. Each `Solve` invocation owns its workspace and output, so concurrent solves are safe **when RHS inputs are stable throughout each call**. You must not mutate input `Dense` buffers or overlapping `View`s during reads. An immutable factor does not turn `Dense.Set` into a goroutine-safe operation.

Cancellation is cooperative: pass a non-nil context, and on cancellation expect no partial `SolveResult`. A nil context is `ErrInvalidOption`.

### Q6.17. Why does QR use more time or memory than I expected?

The QR constructor recomputes the trailing norms on each pivot; it is *not* a literal LAPACK `DGEQP3` norm-downdate implementation. It also computes `ConditionEstimate` by basis solves (`O(n³)`) before publishing the factor. `Solve` computes high-precision original-coordinate residuals, adding multi-precision work on top of the triangular solves. Benchmark factor construction and repeated solutions **separately**; consider factor reuse rather than recomputing the same calibration model.

### Q6.18. Does a failed Cholesky imply a symmetric matrix is unusable?

Not necessarily. First distinguish `ErrAsymmetry` (the pairwise symmetry contract failed) from `ErrNotPositiveDefinite` (a nonpositive Cholesky pivot). Tolerances `absTol, relTol` only permit *small asymmetry*, not a shift of negative eigenvalues. If the physical matrix is indefinite, changing to a different solver is an explicit modeling decision; do not silently add diagonal jitter or hide the failure.

For mathematics, zero-shape tables, pseudocode, and real calibration examples, read [`MATRICES.md`](MATRICES.md#1011-reusable-factorizations-different-problems-different-contracts).

---

## 7. Flow FAQ

### Q7.1. Which max-flow algorithm should I choose?

| Algorithm      | Use when                                         | Notes                                                               |
|:---------------|:-------------------------------------------------|:--------------------------------------------------------------------|
| Ford-Fulkerson | Small or educational integral-capacity networks. | Simple but can be slow depending on augmenting paths.               |
| Edmonds-Karp   | You want a simple polynomial guarantee.          | Uses BFS shortest augmenting paths.                                 |
| Dinic          | Larger or denser networks.                       | Builds level graphs and blocking flows; usually faster in practice. |

### Q7.2. Why did repeated flow runs give different-looking residual states?

Max-flow algorithms modify residual capacity state. If you need independent runs, rebuild or clone the input network according to the package’s documented ownership/residual behavior.

Do not reuse a mutated residual graph as if it were the original capacity graph.

### Q7.3. Are edge weights capacities or costs?

In `flow`, weights represent capacities. In `dijkstra`, weights represent costs. In MST, weights represent connection cost. The same numeric field can mean different domain quantities depending on the algorithm.

Name variables accordingly in examples:

```go
capacityGraph, _ := core.NewGraph(core.WithDirected(true), core.WithWeighted())
```

### Q7.4. What is a residual graph?

A residual graph represents what capacity remains after current flow and where flow could be canceled or rerouted.

```text
Original capacity: u -> v = 10
Current flow:      u -> v = 6
Residual forward:  u -> v = 4
Residual reverse:  v -> u = 6
```

Residual state is part of the algorithm result, not just debug output.

---

## 8. DTW FAQ

### Q8.1. Which DTW facade should I use: `Align`, `AlignCostMatrix`, or `AlignMatrix`?

Use the facade that matches the shape of your domain data.

| Input shape                   | Use                   | Why                                                                           |
|:------------------------------|:----------------------|:------------------------------------------------------------------------------|
| `[]float64` vs `[]float64`    | `dtw.Align`           | Scalar signal alignment: price impulse, temperature, amplitude, trend score.  |
| precomputed local-cost matrix | `dtw.AlignCostMatrix` | Another model already computed frame-to-frame costs.                          |
| time-step × feature matrices  | `dtw.AlignMatrix`     | Multivariate alignment: OHLC candles, sensor vectors, gesture features.       |
| old tuple-style code          | `dtw.DTW`             | Compatibility wrapper. New code should prefer canonical `Result` facades.     |

`AlignMatrix` uses squared L2 row distance:

$$ c(i,j)=|X_i-Y_j|_2^2 $$

That is powerful but also dangerous if feature scales are not comparable. Normalize or standardize outside DTW when one feature would otherwise dominate the local-cost surface.

### Q8.2. Why is DTW slow on long sequences?

Classic DTW is dynamic programming over an `n × m` grid.

```text
Scalar Align:
  Time:  O(n*m)
  Memory distance-only: O(m) rolling rows
  Memory with path/accumulated matrix: O(n*m)

AlignMatrix:
  Local-cost construction: O(n*m*d)
  DP recurrence:           O(n*m)
```

A window can restrict admissible cells, but do not oversell it as a universal complexity fix. The general `Align` DP still scans `O(n*m)` cells, while fixed-length `Search` runs a true band-only scorer with `O(L*b)` worst-case DP work per candidate, `b=min(L,2r+1)` or `L` unconstrained. Certification may reduce evaluated windows but does not guarantee universal speedup.

Use a smaller window when domain knowledge says the alignment should stay near the diagonal. Use distance-only mode when you do not need a path or diagnostic matrices.

### Q8.3. Why does `WithWindow(0)` not mean “no window”?

Because the window has three distinct meanings:

```text
WithWindow(-1):
  no Sakoe-Chiba constraint.

WithWindow(0):
  strict diagonal alignment only.
  No temporal warping.

WithWindow(w), w > 0:
  allow cells where |i-j| <= w.
```

This is one of the easiest DTW mistakes. `0` is not “unlimited”; it is “no warping”.

### Q8.4. Why can’t I recover a path in rolling-row mode?

Path reconstruction needs predecessor history. Rolling rows intentionally discard most accumulated state to reduce memory.

Use:

```go
res, err := dtw.Align(
	a,
	b,
	dtw.WithReturnPath(true),
	dtw.WithMemoryMode(dtw.FullMatrix),
)
```

Rules of thumb:

* distance-only scan → rolling rows;
* explain one alignment → `FullMatrix` + `WithReturnPath(true)`;
* debug DP behavior → add `WithReturnAccumulated(true)`;
* inspect feature/cost scaling → add `WithReturnLocalCost(true)`.

A path is one deterministic optimal path, not all optimal paths. Backtracking tie-break is diagonal first, then vertical/up, then horizontal/left.

### Q8.5. Why did DTW return `Reachable=false` or `+Inf` distance?

`+Inf` in a successful final `Result` means the endpoint is structurally unreachable. An internal reachable prefix may also overflow, but **final optimal overflow returns `ErrOverflow`**, joined with `ErrNaNInf`; it is never silently published as unreachable.

Common causes:

* window too narrow;
* strict diagonal mode for unequal or phase-shifted sequences;
* local-cost matrix shape and the selected window admit no endpoint-to-endpoint route;
* path requested but policy makes the endpoint unreachable;
* the endpoint lies outside the admissible window (penalty changes costs, not structural reachability).

A good debugging progression:

```text
1. Run with WithWindow(-1).
2. Verify the distance is finite.
3. Enable WithReturnPath(true) only when you need explanation.
4. Tighten the window gradually.
5. Add WithReturnAccumulated(true) if you need to see where reachability breaks.
```

### Q8.6. Why does DTW reject `NaN`, `Inf`, or negative costs?

DTW separates local costs from accumulated reachability states.

Valid local costs:

```text
finite
non-negative
not NaN
not +Inf
not -Inf
```

`+Inf` is prohibited in caller-provided local costs. A full accumulated artifact may represent structural absence or an overflowing reachable intermediate prefix with `+Inf`; final overflow is a classified error. `AlignCostMatrix` therefore rejects `NaN`, infinities, and negative costs. `AlignMatrix` validates finite matrix inputs. `Align` validates scalar inputs by default.

### Q8.7. Should I normalize time-series data before DTW?

Usually yes.

DTW aligns timing; it does not magically fix feature scale. For scalar series, normalize when amplitude scale is not the signal you want to measure. For `AlignMatrix`, normalize columns/features before alignment when dimensions have different units. Use a `stats.Standardizer` fitted on training data when the **same fixed scaling** should be applied to both sequences; DTW does not fit its own scaler.

Examples:

```text
Good DTW inputs:
  normalized amplitude contours
  standardized sensor features
  log-cost surfaces from probabilistic models
  domain-weighted local-cost matrices

Risky DTW inputs:
  raw heterogeneous features
  probabilities treated as distances
  one huge-scale feature mixed with small-scale features
```

If a classifier gives probabilities, consider converting them to costs intentionally, for example with `-log(p)`, instead of passing probabilities directly as distances.

### Q8.8. What is the correct way to read `Result`?

Think of `dtw.Result` as an alignment artifact, not just a number.

Important fields and meaning:

```text
Distance:
  accumulated DTW cost.

Reachable:
  whether the final cell can be reached under active policy.

PathTracked:
  whether path tracking was requested.

Path:
  deterministic path only when tracking was requested and result is reachable.

Window / SlopePenalty / MemoryMode:
  policy that actually ran.

Accumulated:
  optional DP matrix artifact.

LocalCost:
  optional local-cost matrix artifact.
```

Use `PathOrError` when you need to distinguish nil result, unreachable result, and path-not-tracked state.

### Q8.9. When should I use `AlignBounded` instead of `Align`?

Use `AlignBounded(ctx,a,b,limit,opts...)` when the caller needs an explicit
inclusive cutoff classification. `BoundExact` has a finite exact Distance;
`BoundAboveLimit` sets Distance to `NaN` and publishes a certified LowerBound
above limit; `BoundUnreachable` uses `+Inf` for both fields. The current
bounded API still performs the full canonical recurrence. It rejects
unpublishable local/accumulated matrix artifacts rather than silently ignoring
those options. A final overflow remains an error, not a cutoff.

### Q8.10. Is `Search` a globally exact subsequence search?

`Search` is **globally exact over fixed-length candidate windows** of length
`len(query)` at admitted stride-aligned starts. It is *not* open-boundary
subsequence DTW. Scoring uses squared local differences, no callback, and no
slope penalty. ExcludedStarts are half-open **start-index** ranges, not sample
overlap exclusions. A complete result is sorted by `(SquaredDistance,Start,End)`.
A partial result ranks only completed candidates and is not globally certified.

### Q8.11. Can LB_Keogh be used to discard candidates near a cutoff?

Only with a numerical certificate tied to the scorer. The public
`LBKeoghSquared` is a rounded diagnostic; comparing its value directly with
cutoff is not safe in general. `Search` builds its own radius-matched envelope,
proves finite local-domain and diagonal-path representability, accumulates a
monotone machine bound, and prunes only on **strictly greater** than cutoff.
Uncertain cases are evaluated normally. A tight MaxWorkingBytes can disable
optional pruning without changing the exact mathematical result.

### Q8.12. What do `CandidatesCompleted`, `CandidatesPruned` and `CellsVisited` mean?

`CandidatesCompleted` counts windows with fully scored finite DTW objectives.
`CandidatesPruned` counts candidates excluded with a valid machine certificate.
`CellsVisited` counts only actual DP cells evaluated, including incomplete
work before a budget or cancellation error. On complete search,
`CandidatesCompleted+CandidatesPruned=CandidatesTotal`; worker scheduling may
change these performance counters but not the final sorted Hits.

---

## 9. MST FAQ

### Q9.1. Why does MST reject my graph?

MST is defined for weighted, undirected graph connectivity. The package intentionally rejects graph models that would change the mathematical problem.

Common causes:

* nil graph input;
* graph is unweighted;
* graph is directed;
* graph has directed edge-level overrides;
* graph is empty;
* graph is disconnected in strict tree mode;
* an edge weight is `NaN`, `+Inf`, or `-Inf`;
* Prim was requested without a required root in strict mode;
* Prim root does not exist.

MST is not a shortest-path algorithm. It does not answer “how do I travel from A to B?” It answers:

> Which finite-cost acyclic backbone connects all vertices with minimum total selected edge weight?

### Q9.2. Why does MST require a weighted undirected graph?

Because MST optimizes a finite edge-cost backbone over undirected connectivity.

```text
Required:
  weighted graph policy
  undirected graph policy
  finite edge weights

Rejected:
  unweighted graph
  directed graph
  directed edge-level overrides
  NaN / +Inf / -Inf weights
```

Negative finite weights are valid for MST. Unlike Dijkstra, MST does not require non-negative weights. It only needs a finite ordering of candidate edges.

Self-loops are ignored because they cannot connect two components. Parallel edges are allowed because the lighter useful candidate may be selected.

### Q9.3. Why does strict MST fail on disconnected graphs instead of returning a forest?

Because strict MST and minimum spanning forest are different publication contracts.

```text
Strict MST:
  expects one connected component.
  selected edges = |V| - 1.
  disconnected graph => ErrDisconnected.

Explicit MSF:
  enabled with WithForest().
  computes one MST per connected component.
  selected edges = |V| - componentCount.
```

The package must not silently downgrade strict tree semantics into forest semantics. If you want a forest, request it explicitly.

This matters in production: a disconnected network is often an incident, not a successful smaller tree.

### Q9.4. Kruskal or Prim?

Use either under the shared MST contract, but choose based on how you want to reason about the graph.

```text
Kruskal:
  Think globally.
  Sort all finite candidate edges by weight.
  Add the next safe edge if it joins two different components.

  Best mental model:
    "Build the cheapest backbone from a sorted edge ledger."

Prim:
  Think from a root/frontier.
  Start at one vertex.
  Repeatedly take the cheapest edge leaving the grown tree.

  Best mental model:
    "Grow one connected service area from a chosen root."
```

Current complexity contract:

```text
Kruskal:
  O(E log E + E*α(V)) time
  O(E + V) space

Prim:
  O(E log E) time
  O(E + V) space
```

Do not document Prim as `O(E log V)` unless the implementation changes to a vertex-key decrease-key policy.

### Q9.5. What does `mst.Result` actually guarantee?

`mst.Result` is the canonical result artifact.

It tells you:

```text
Algorithm:
  Kruskal or Prim.

Mode:
  strict tree or explicit forest.

Root:
  meaningful for Prim; deterministic component root in forest contexts.

Edges:
  detached selected edge values.

TotalWeight:
  sum of selected finite edge weights.

VertexCount:
  number of vertices in the validated snapshot.

ComponentCount:
  1 for strict MST success, >1 for explicit forest.

ComponentRoots:
  deterministic public roots for components.
```

The result does not retain live `*core.Edge` pointers into the source graph. Mutating the graph after the call must not mutate the published result.

### Q9.6. Why does MST snapshot the graph?

The package snapshots vertices and edges before kernel execution so the result can publish detached selected edges and stable metadata. This protects result ownership, but it is not a transaction against concurrent graph writers.

Do not mutate the graph while snapshot construction is happening.

### Q9.7. Why are `NaN` and infinities rejected instead of treated as walls?

Because MST needs a total finite ordering of candidate edges. Infinities are not MST wall semantics in this package.

Use another package or preprocess your graph if you need wall/unreachable semantics. For MST:

```text
finite negative weight: accepted
finite zero weight:     accepted
finite positive weight: accepted
NaN:                    rejected
+Inf / -Inf:            rejected
```

---

## 10. TSP FAQ

### Q10.1. Why does TSP reject my matrix?

Final TSP kernels require a complete, finite, square distance model.

Common validation failures:

* matrix is nil;
* matrix is not square;
* `n < 2`;
* diagonal is not approximately zero;
* a finite weight is negative;
* a cell is `NaN`;
* a cell is `-Inf`;
* a `+Inf` missing edge remains after optional metric closure;
* selected algorithm requires symmetry but the matrix is asymmetric;
* `StartVertex` is out of range;
* IDs do not match matrix size.

TSP is matrix-backed in the current contract. Graph input is adapted at the facade boundary; solver kernels operate on matrix indices.

### Q10.2. Should I use `SolveMatrix` or `SolveGraph`?

Use `SolveMatrix` when your true domain is already a distance/cost matrix.

Use `SolveGraph` when your source object is a `core.Graph` and you want the package to adapt it into the matrix-solving boundary.

```text
SolveMatrix:
  canonical matrix facade.
  Best for routing matrices, metric closure output, distance tables,
  warehouse/city/inspection cost surfaces.

SolveGraph:
  graph adapter facade.
  Best when topology starts as core.Graph.
  After adaptation, solver semantics are matrix semantics.
```

Do not send `core.Graph` into internal solver thinking. TSP kernels solve over matrix coordinates.

### Q10.3. Christofides, ExactHeldKarp, BranchAndBound, TwoOptOnly, or ThreeOptOnly?

Choose by contract strength and runtime regime.

| Algorithm        | Use when                                                                  | Result meaning                                                               |
|:-----------------|:--------------------------------------------------------------------------|:-----------------------------------------------------------------------------|
| `ExactHeldKarp`  | Small guarded instances where proof of optimality matters.                | Exact optimal result if accepted and completed.                              |
| `BranchAndBound` | Exact search with pruning, good incumbents, and optional time budget.     | Exact only when completed; time limit may return incumbent + `ErrTimeLimit`. |
| `Christofides`   | Symmetric complete metric TSP where an approximation certificate matters. | `1.5` ratio only with exact `BlossomMatch`.                                  |
| `TwoOptOnly`     | You have a feasible tour and want deterministic local improvement.        | Heuristic local optimum in 2-opt/2-opt* neighborhood.                        |
| `ThreeOptOnly`   | Symmetric local search where deeper neighborhood is worth more time.      | Heuristic local optimum; no global certificate.                              |

Mental model:

```text
Need proof and n is small:
  ExactHeldKarp.

Need exact search with pruning / timeout governance:
  BranchAndBound.

Need scalable symmetric metric approximation:
  Christofides + BlossomMatch.

Need fast deterministic improvement:
  TwoOptOnly.

Need stronger local search and can pay more:
  ThreeOptOnly.
```

Do not hide exact-to-heuristic fallback. If a caller selects exactness, failure or timeout must remain visible.

### Q10.4. What is the Blossom vs Greedy matching law in Christofides?

Christofides needs a minimum-weight perfect matching over odd-degree MST vertices.

```text
BlossomMatch:
  exact MWPM.
  Supports publishing the formal 1.5 approximation ratio.

GreedyMatch:
  deterministic weaker matching.
  Useful when explicitly selected.
  Does not publish the Christofides 1.5 ratio.
```

Never describe Greedy matching as preserving the Christofides proof. It may be useful engineering policy, but it is not the same theorem.

### Q10.5. Why does TSP need a complete distance matrix?

A TSP tour must be able to move from every selected vertex to the next vertex in the cycle. Missing edges break that assumption.

If your source graph is sparse, first decide what missing edges mean:

```text
Missing edge means truly impossible:
  TSP should reject or fail validation.

Missing edge means indirect travel is allowed:
  apply metric closure before final solving.
```

When `RunMetricClosure` is enabled, `+Inf` can act as a missing-edge sentinel before closure. Final solver kernels must still receive a complete finite matrix.

### Q10.6. Why do approximation guarantees require metric assumptions?

Christofides’ shortcut step relies on triangle inequality. Shortcutting an Eulerian circuit is safe only because a direct edge is no more expensive than the path it replaces.

```text
Metric requirement:
  c(a,c) <= c(a,b) + c(b,c)

If this fails:
  shortcutting may increase cost unpredictably.
  the 1.5 guarantee does not apply.
```

The package should not publish an approximation ratio unless the selected algorithm and matching policy justify it.

### Q10.7. What does `tsp.Result` tell me beyond `Tour` and `Cost`?

`tsp.Result` is the result contract. Use it instead of inferring behavior from the selected options.

Important fields:

```text
Tour:
  closed vertex-index cycle.

Cost:
  stabilized directed cycle cost.

Algorithm:
  selected algorithm after option finalization.

Exact:
  whether the algorithm is exact by design.

Optimal:
  whether this specific result is certified optimal.

TimedOut:
  whether a governed time limit stopped the search.

ApproximationRatio:
  formal ratio when valid; otherwise 0.

MetricClosureApplied:
  whether +Inf closure preparation was used.

IDs:
  detached labels aligned with matrix order.
```

Low cost is not proof of optimality. Read `Optimal`.

### Q10.8. How should I handle Branch-and-Bound time limits?

Branch-and-Bound is exact only when the search completes. With a time limit, it may return a feasible incumbent together with `ErrTimeLimit`.

Correct handling:

```go
res, err := tsp.SolveMatrix(dist, ids, opts)
if errors.Is(err, tsp.ErrTimeLimit) {
    // res may contain the best incumbent found so far.
    // It is feasible, but not certified optimal.
    if res != nil && res.TimedOut && !res.Optimal {
        // Decide whether the incumbent is acceptable for your application.
    }
}
```

Do not discard the result automatically, and do not mark it optimal.

### Q10.9. What is different about ATSP local search?

In ATSP, direction matters:

$$ c(u,v) \ne c(v,u) $$

That changes what local moves are legal. Symmetric 2-opt can reverse a segment. Directed 2-opt* preserves orientation by swapping tails. ATSP 3-opt in the package is restricted 3-opt*, not full arbitrary directed 3-opt.

Do not document ATSP local search as if it were the symmetric TSP move set.

### Q10.10. Why are costs rounded/stabilized?

TSP compares many candidate tours and local-search moves. Tiny floating-point drift can create meaningless observable differences.

The package stabilizes published costs so deterministic output does not depend on irrelevant last-bit noise. This does not mean invalid numeric input is tolerated: `NaN`, `-Inf`, negative finite weights, and unresolved final `+Inf` remain validation failures.

---

## 11. Series FAQ

### Q11.1. Does `series.Validate` infer cadence from timestamps?

No. `GridOptions{}` or `GridOptions{CadenceNS: 0}` validates strict order
and representable positive timestamp differences only. It deliberately does
**not** derive a frequency, reconstruct a calendar, or report missing slots.

For positive `CadenceNS = c`, every adjacent interval `d` must satisfy
`d > 0`, `d <= math.MaxInt64`, and `d % c == 0`. `AllowGaps` changes only
what happens to **whole** missing slots; it cannot permit a nonmultiple.

```text
[0, 10, 20], cadence 0  -> order-only success
[0, 10, 20], cadence 10 -> exact-grid success
[0, 10, 30], cadence 10 -> one gap if AllowGaps=true
[0, 10, 25], cadence 10 -> ErrGap even if AllowGaps=true
```

### Q11.2. Does a reported `Gap` insert a missing observation?

No. `Gap{AfterIndex: i, MissingCount: m}` reports `m` absent lattice
slots between supplied observations `i` and `i+1`. The returned
`ValidationResult.Samples` counts **only supplied pairs**, and the original
`[]float64` remains unchanged. `ValidationResult.Gaps` is a detached,
non-nil slice on success, even when empty.

If a model requires uninterrupted sampling, use `AllowGaps: false` or handle
reported gaps explicitly before calling it. Calling `Lag` or `SimpleReturns`
on the original array uses adjacent **supplied indices**, including pairs
separated by a timestamp gap.

### Q11.3. Why is a returned zero insufficient to tell whether a result exists?

All transforms preserve input length. Positions lacking history contain
`Values[i] == 0` and `Valid[i] == false`; valid computations can also
legitimately produce zero. These cases are different:

```text
values = [2, 2, 4]
Difference(values, 1)
  Values = [0, 0, 2]
  Valid  = [false, true, true]
              ^
         valid zero difference
```

Keep the two result slices together. A mask is not a NaN fill, a timestamp
gap report, or a reason to mutate observations. Both slices are detached
from caller input.

### Q11.4. How do the domains of simple and log returns differ?

`SimpleReturns` computes the adjacent relative change
`x[i]/x[i-1] - 1`, requiring only **used previous denominators** to be
nonzero. Negative finite values are allowed mathematically; a zero in the
last position is permitted because no successor uses it as a denominator.
A singleton zero is also accepted and produces one invalid warmup cell.

`LogReturns` computes `ln(x[i]) - ln(x[i-1])` and requires **all** input
values to be strictly positive, even for a singleton. It therefore rejects
zero, negative zero, and negative observations with `ErrDomain`.
Both operations reject NaN and infinity with `ErrNonFinite` before domain
checks and return nil results on errors.

### Q11.5. Why does `LogReturns` not compute `math.Log(curr/prev)`?

The intermediate ratio can overflow or underflow when both operands are
valid, finite, strictly positive `float64` values. For `prev=1e-300` and
`curr=1e300`, the ratio overflows but the correct log difference is finite.

`LogReturns` uses the difference of logarithms for distant positive pairs,
and a stable `math.Log1p` relative-change path for nearby values. The latter
also avoids losing a very small nonzero change when two separately rounded
logarithms would be equal. This improves numerical behavior without claiming
exact real-number arithmetic or introducing an epsilon validity threshold.

### Q11.6. What errors should I expect, and which one takes precedence?

`series` owns `ErrInvalidOption`, `ErrNilInput`, `ErrEmptyInput`,
`ErrShape`, `ErrNonFinite`, `ErrOrder`, `ErrGap`, `ErrDomain`, and
`ErrOverflow`. Classify through `errors.Is`.

`Validate` checks options first, then nil inputs, mismatched lengths,
equally empty inputs, and **all** nonfinite observations before walking
consecutive timestamps. Timestamp pairs are then checked in index order:
strict order, signed delta representability, exact cadence, and gap policy.

For `Lag`/`Difference`, lag validation precedes nil/empty/finite checks.
`SimpleReturns` checks all denominators before computing any pair;
`LogReturns` checks the entire positive domain first. Hence a domain error
takes precedence over an arithmetic overflow elsewhere in that call.
There is no error-producing partial result or silent fallback.

### Q11.7. What does `ErrOverflow` mean when all inputs are finite?

It can describe an unrepresentable **result** rather than invalid input:

* Signed timestamps `math.MinInt64 -> 0` are ordered, but their difference
  exceeds `math.MaxInt64` nanoseconds.
* A finite difference such as `math.MaxFloat64 - (-math.MaxFloat64)`
  overflows float64.
* Extreme simple returns can overflow division from a small nonzero base.
* `WindowRanges` can count a mathematically valid number of windows whose
  required `[]Range` byte size is not representable as a Go `int`.

This does not mean `series` guarantees that all representable allocations
will succeed under an operating-system memory limit. It checks arithmetic,
not available RAM.

### Q11.8. Are these operations causal and safe to call concurrently?

Each transform at index `i` reads only elements at indices at most `i`,
so appending future observations cannot change a successful output prefix
**when both original and extended calls succeed**. For example, appending
after a final zero can make a subsequent `SimpleReturns` fail because that
zero becomes a denominator; the package then publishes no partial result.

Independent calls can use the same input concurrently **if the caller never
mutates it during those calls**. Returned gap, values, mask, and range
slices are caller-owned, detached memory. Concurrent writes to the same
result must be externally synchronized. There is no mutable streaming
accumulator or implicit snapshot isolation.

### Q11.9. Are `WindowRanges` outputs durations or sample positions?

Sample positions only. `WindowRanges(8,3,2)` yields `[0,3)`, `[2,5)`,
`[4,7)`, all complete and half-open; there is no trailing partial range.
`WindowRanges(0,3,1)` returns a **non-nil empty** slice. Stride larger
than width is valid and emits separated windows. To interpret indices in
clock time, the application must separately inspect or validate timestamps.

### Q11.10. Can I pipe the output directly into DTW, statistics, or a model?

Only after checking domain requirements of the downstream consumer.
`series` has no dependency on `dtw` or `matrix` and does not build a model.
If a transform has warmup cells, a downstream algorithm must not consume
those zero placeholders as genuine data. Likewise, a grid validated with
`AllowGaps: true` is not automatically a uniformly sampled sequence.

For the full proofs, numeric examples, error tables, and tested workflows,
read [`SERIES.md`](SERIES.md), [`series/doc.go`](../series/doc.go), and
[`series/example_test.go`](../series/example_test.go).

---

## 12. Stats FAQ

### Q12.1. When should I use `stats` instead of statistics already present in `matrix`?

Use `stats` for **stable scalar streaming moments**, **mergeable partitions**, and **a fitted standardizer reused on heldout data**. `matrix` remains responsible for dense storage, graph adapters, covariance/correlation and its own documented algebraic transformations. A matrix normalization routine that computes parameters on its current input is not necessarily an immutable train-only preprocessing model.

```text
Need an ordered scalar batch mean/variance?    -> stats.Summarize
Need ongoing or partitioned scalar reduction?  -> stats.Moments
Need frozen per-column training scaling?       -> stats.FitStandardizer
Need covariance or matrix algebra?             -> matrix
Need timestamp cadence / Valid masks?          -> series
Need fractional length reduction, not warping? -> paa
Need timing alignment, not scaling?            -> dtw
```

The packages complement one another. `stats` imports `matrix` and the `core.Nilable` contract; `matrix` does not import `stats`.

### Q12.2. Why not compute variance as `mean(x*x) - mean(x)*mean(x)`?

That identity is exact in real arithmetic but numerically unstable for high-offset, low-spread data. The two terms can become equal after binary64 rounding, and their difference becomes zero even when the true variance is positive.

`stats` uses Welford updates and Chan merges, both based on centered residuals. The `M2` state has a compensated significand and independent exponent, so large intermediate squares do not require a `float64` representation. The final mean is calculated separately from an exact binary-input sum; **variance remains an approximate floating-point quantity**.

For `[1e12+1, 1e12+2, 1e12+3]`, sample variance (`ddof=1`) is 1. See [the Welford/Chan derivations](STATS.md#4-mathematics-welford-and-chan).

### Q12.3. Can `Summary` succeed when the internal `M2` is larger than `math.MaxFloat64`?

Yes. With `[-1e154,0,1e154]`, the unnormalized `M2` is around `2e308`, but after dividing by `3-1` the sample variance is around `1e308`, which is representable. `Summary` keeps the moment scaled until the division.

**Do not confuse avoidable intermediate overflow with final overflow.** If the requested published variance itself cannot be represented as finite binary64, `ErrOverflow` is the correct result. `Update`/`Merge` are not required to reject an intermediate scaled `M2` just because a plain float64 representation would overflow.

### Q12.4. Is the mean exact? Is the variance exact?

The mean is rounded once from the exact sum of the original finite binary64 observations, divided by their exact count. This is stronger than simply compensating an ordinary `float64` sum. For `[1e154,1,-1e154]`, the correct mean rounds from `1/3` instead of disappearing to zero.

Variance uses compensated scaled Welford/Chan arithmetic. It is *not* an arbitrary-precision rational central moment; output error depends on the input magnitudes and update/merge grouping. Independent `big.Rat`-based test oracles help establish practical accuracy on adversarial inputs, but they do not constitute a universal error theorem.

### Q12.5. How should I choose between `ddof=0` and `ddof=1`?

`ddof=0` divides the unnormalized central moment by `n` (population variance). `ddof=1` divides by `n-1` (sample variance) and requires at least two samples. The package does not infer which statistical interpretation fits your domain.

Call `stats.SummarizePopulation(values)` for the first convention or
`stats.SummarizeSample(values)` for the second. Both wrappers select the
corresponding fixed `ddof` and delegate to the same Welford kernel.

`Summary` rejects a count no larger than `ddof` with `ErrInsufficientSamples`. An unsupported `ddof` produces `ErrInvalidOption` before even inspecting a nil slice or nil receiver.

### Q12.6. Are `Moments.Update` and `Moments.Merge` safe to call concurrently?

Not on the same accumulator. They are **transactional on a returned error**, not synchronized mutators. Use one `Moments` per goroutine, finish all updates, then merge in a fixed order after the workers stop. Do not mutate the source during `Merge`.

An empty source is a no-op. Merging into an empty destination copies the source's complete state by value. `m.Merge(m)` is rejected with `ErrInvalidOption`, even for an empty receiver. Count overflow is checked before the merge and leaves both states unchanged.

### Q12.7. Can changing the order of partitions change the answer?

For the *exact binary sum* used by the final mean, partition grouping does not lose the original input bits. But the compensated floating-point `M2` is not bitwise associative; different update or merge trees can produce slightly different variance approximations.

Keep partition contents, in-partition order, and merge-tree order deterministic if you need reproducible results. Do not force bit-for-bit variance equality between independent groupings unless the specific dataset warrants it.

### Q12.8. Why does `FitStandardizer` reject a constant column by default?

`ScaleOptions{}` means `DDOF=0` and `ConstantPolicy=RejectConstant`. A zero internal `M2` yields `ErrZeroVariance`; division by a fitted scale of zero is undefined.

If retaining the feature is intentional, use `stats.UnitScaleConstant`. The fit then preserves its training mean, assigns scale exactly one, and marks `Constant[j]=true`. For training `[5,5,5]` and heldout `[7]`, the standardized value is **2**, not forced zero. An epsilon such as `1e-5` is *not* used to declare features constant.

### Q12.9. Why can `Summary.Variance` be zero while a training feature is not constant?

A positive but very small variance can underflow when it is converted to the public `float64` field. The fitter does **not** classify a feature by that rounded field. Instead it checks its scaled internal `M2` for zero and computes the standard deviation in the scaled domain.

This matters twice: a tiny positive variance should not be relabeled constant, and a variance greater than `MaxFloat64` can still have a finite square root that is a valid fitted scale. Do not compute fitting parameters by first calling `Summary()` and then taking `math.Sqrt` of the public variance.

### Q12.10. Can I reuse a `Standardizer` on test data or across goroutines?

Yes: that is the intended inference model. Fit only on training rows, persist your application-level feature order and split provenance, and call `fit.Model.Transform(heldout)` without refitting.

The model's private means/scales are immutable after successful fit. Separate concurrent `Transform` or `InverseTransform` calls use independent output matrices. **Input matrices must remain unchanged during each call**: inference scans each input twice and is not a snapshot transaction.

### Q12.11. What happens if the training and inference dimensions differ?

Fit requires at least one row and one column; `rows <= DDOF` is `ErrInsufficientSamples` after input finite checks. For inference the **column count must equal the fitted feature count** or `ErrShape` is returned. A `0×d` inference batch is legal when `d` matches training and returns a detached `0×d` dense matrix; a `0×d` *training* input yields `ErrEmptyInput`.

The interface may carry a typed-nil matrix pointer. `stats` uses `matrix.ValidateNotNil`, relying on the `core.Nilable` convention for supported types. Custom pointer-backed Matrix types should provide `IsNil()`; reflection is not needed.

### Q12.12. Why use `math.FMA` in `InverseTransform`?

The intended inverse is `z*scale + trainingMean`. A separate product `z*scale` may overflow even when the final sum would be finite after cancellation with the mean. `math.FMA` performs the combined operation with one final rounding and can preserve that finite result.

This is not a guarantee that every `InverseTransform(Transform(x))` is bit-identical to `x`. Fitted parameters and forward transformations are still rounded. True unrepresentable output values correctly produce `ErrOverflow`.

### Q12.13. Which error wins if options, input data, and sample count are all wrong?

Precedence is operation-specific. `Summarize` checks `ddof`, nil, and empty slice before scanning observations; data are scanned before `Summary` detects insufficient count. `FitStandardizer` checks options, matrix presence/shape, and empty extent before reading all training cells; finite checks precede sample-count sufficiency. A first encountered `Matrix.At` error can precede a nonfinite cell at a later coordinate.

`Transform` and `InverseTransform` validate the model, input presence, and feature count, then check all input cells before computing output. On any error, no partial matrix is returned. Always branch with `errors.Is`; matrix-origin errors are preserved through wrapping. Consult the [complete precedence table](STATS.md#8-validation-order-and-sentinel-error-law).

### Q12.14. Can I pass `series.TransformResult.Values` directly into `stats.Summarize`?

Only if **every position you pass is a valid observation** under your chosen causal and timestamp policy. `series` deliberately stores zero placeholders in invalid warmup positions, with `Valid[i] == false`; a computed zero with `Valid[i] == true` is different. `stats` accepts finite numeric values and does not inspect `Valid` or timestamp gaps.

Filter or segment the data explicitly. The same applies to `series.WindowRanges`: they are sample-index partitions, not a promise that a timestamp gap was repaired. See the [complete mask-aware usage scenario](STATS.md#12-complete-runnable-go-scenarios).

### Q12.15. What do the `IsNil()` methods check?

`(*stats.Summary).IsNil`, `(*stats.ScaleFitResult).IsNil`, and `(*stats.Standardizer).IsNil` implement `core.Nilable` by checking whether the receiver pointer itself is nil. They do **not** validate sample count, fitted state, or contents. A non-nil `&stats.Standardizer{}` is still unfitted; calling `Transform` on it returns `ErrNilInput`.

### Q12.16. How should I test and benchmark the numeric contract?

Use independent rational oracles on the exact original binary64 input values for large offsets, cancellation, subnormals, and true final overflow. Test `Update` and `Merge` error atomicity, `MaxUint64` count overflow, constant holdout deviations, input/result aliasing, 0-row inference, and concurrent inference on one model. Keep all tests in `package stats_test` and use `errors.Is`, not string parsing.

Benchmark `Update`, `Merge`, `FitStandardizer`, and `Transform` as **separate** regimes; preparation belongs outside `b.N`. Include realistic feature dimensions and report toolchain, CPU, allocations, and actual workload shape. `Update`'s hot loop may avoid heap allocation, while final mean materialization and owning matrix outputs legitimately allocate.

For derivations, runnable examples, data-flow diagrams, and file boundaries, read [`STATS.md`](STATS.md), [`stats/doc.go`](../stats/doc.go), and [`stats/example_test.go`](../stats/example_test.go).

---

## 13. PAA FAQ

### Q13.1. When should I use `paa` instead of `series`, `stats`, or `dtw`?

`series` validates the exact time grid and can produce complete sample-index windows or length-preserving causal features. `paa` **compresses** one finite, uniformly sampled scalar series to `w` equal-duration averages, preserving the entire step-function domain. `stats` calculates global moments and fitted feature scaling; `dtw` aligns sequences under a warping policy. None of them implicitly repairs gaps for another package.

### Q13.2. What goes wrong with `floor(n/w)` or padding zeros?

With `values=[0,2,4,6,8]`, `w=2`, the correct means are `[1.6,6.4]`; their exact rational boundaries are `[0/2,5/2)` and `[5/2,10/2)`. A floor-block implementation would lose the last sample or misassign the fractional middle cell. Padding would invent measurements. `paa.Transform` distributes each crossing sample according to the exact integer overlap of its source interval with each output cell.

### Q13.3. Why are `Segment` coordinates rational and local to the selected window?

The coordinates describe **sample-time positions**, not rounded index ranges or real timestamps. `Segment{LeftNumerator,RightNumerator,Denominator}` keeps `Denominator=w` unsimplified, so equality and indexing are deterministic even for fractional boundaries. `TransformRange(values,start,end,w)` delegates to `Transform(values[start:end],w)` and reports geometry relative to `start`; its `OriginalLength` is `end-start`.

Do not add `start` to the numerator without multiplying it by the denominator. Absolute coordinates are `(start*w+numerator)/w`; check additional integer operations before using them on huge indices.

### Q13.4. Should I use `Transform` or `Means`?

Use `Transform` when original length, exact segment coordinates or downstream SAX distances matter. Use `Means` only when the consumer already knows the geometry and wants a caller-owned `[]float64` vector. `Means` projects `Result.Values` from the canonical implementation; its temporary `Segments` allocation is not removed. There are no optional modes that secretly change the math.

### Q13.5. Can I safely call `TransformRange` with untrusted indices?

Yes, it checks the half-open interval before forming a slice: nonpositive `w` produces `ErrInvalidOption` first; then come `ErrNilInput`, `ErrEmptyInput`, and only then invalid bounds (`start<0`, `start>=end`, `end>len(values)`) or excessive segment counts. Only values inside the accepted window are checked for `ErrNonFinite`. All failures return a nil result; classify through `errors.Is`.

```go
result, err := paa.TransformRange(samples, start, end, segments)
if errors.Is(err, paa.ErrInvalidOption) {
    // Reject malformed count or half-open indices before slicing.
}
_ = result
```

### Q13.6. Why does `[1e308,1e308]` not overflow? Is the mean only approximate?

The kernel records the original finite IEEE-754 values as exact binary mantissas and integer overlap weights in two fixed-width sign accumulators. It cancels signs exactly, divides by `n`, then rounds **once** to nearest-even binary64. The mean stays in the convex hull of the contributing samples, even if a naive unscaled sum would overflow. At the subnormal/normal transition, both the quotient's discarded bits and its nonzero division remainder affect rounding. This is stronger than ordinary Kahan/Neumaier compensation.

The exact real weighted-conservation identity holds mathematically; recomputing an integral from separately rounded output means may still introduce a small floating-point discrepancy. For testing, compare against independent exact `math/big.Rat` references rather than against overflow-prone test-side float sums.

### Q13.7. Is `w=n` a view? Can I mutate results or call it concurrently?

No view is published. `w=n` returns a detached **bit-preserving** identity, including `-0`. Both result slices are caller-owned, and one call retains no input. Calls on shared immutable inputs are safe concurrently; simultaneous caller mutation of the source is unsupported. Changing an earlier result must not change future calls.

### Q13.8. Does PAA alone guarantee `LB_PAA <= DTW` or SAX `MINDIST` correctness?

No. With matched series lengths and segment grids, Jensen's inequality gives a scaled Euclidean lower bound for exact PAA averages. SAX quantization and `MINDIST` need their own proof and compatible breakpoints; general DTW lower bounds depend on the precise warping/window constraints. Do not treat generic PAA compression as a certified DTW pruning rule.

### Q13.9. How should I verify and benchmark it?

Use external `package paa_test` tests, `errors.Is`, exact rational overlap oracles, bitwise rounding comparisons, fractional `n%w!=0` fixtures, and detached output checks. Benchmark global mean (`w=1`), small fixed `w`, identity (`w=n`), checked subranges and the `Means` projection separately. For the complete contract, examples, arithmetic proof and pitfalls, see [`PAA.md`](PAA.md), [`paa/doc.go`](../paa/doc.go) and [`paa/example_test.go`](../paa/example_test.go).

---

## 13a. SAX FAQ

### Q13a.1. What can SAX answer that `paa` cannot?

`paa.Transform` returns real-valued equal-duration fractional averages and exact rational interval geometry. `sax` takes those averages, maps them into an alphabet of 2–256 bins and returns a `Result` containing `Symbols`, source geometry, normalization metadata and `EncodingID`. It also computes a compatible **Euclidean MINDIST lower-bound dissimilarity**. SAX is useful when you need stable symbolic signatures and coarse ranking; PAA is useful when you need the actual aggregate values and geometry. Neither package builds a search index or repairs timestamps.

### Q13a.2. Should I choose `ZPopulation` or `Raw`?

Use `ZPopulation` for **relative shape** when removing each window's offset and amplitude is appropriate. Every window is fitted independently with the population divisor `n`, so it is not a reusable training-only `stats.Standardizer` and it erases absolute level changes. Use `Raw` when physical units (CPU percentage, millihertz deviation, temperature or return basis points) and amplitude matter. If you choose Raw Gaussian bins, remember their standard-normal thresholds may not fit the input units; calibrated custom breakpoints are usually clearer.

### Q13a.3. Why did a constant input produce `ErrZeroVariance`?

`DefaultOptions` chooses `ZPopulation` and `RejectConstant`. A constant window has population `sigma=0`; Z-score division would be undefined, so `Encoder.Encode` returns nil with `ErrZeroVariance`. Set `opts.ConstantPolicy=sax.ZeroWord` only if a constant shape should be represented. Then `Mean` retains the original constant level, `StdDev=0`, `Degenerate=true` and the normalized vector consists of zero coordinates. **ZeroWord is not symbol code zero:** with a Gaussian two-symbol alphabet, zero belongs to right bin `1`. Raw accepts constant inputs without the zero-variance policy.

### Q13a.4. Why does a nonconstant tiny-spread input return `ErrOverflow` instead of `ZeroWord`?

The implementation distinguishes **all observations exactly identical** from nonconstant observations whose positive sigma cannot be published as finite nonzero binary64. Reclassifying such a window as constant would silently lose information and violate the public `Degenerate` contract. The stable normalization path returns `ErrOverflow` rather than a false zero-variance result; do not add an arbitrary epsilon to turn the failure into a successful ZeroWord.

### Q13a.5. Why does equality to a custom breakpoint go to the right bin?

The bins are `(-Inf,b0), [b0,b1), ... [b_(a-2),+Inf)`. Hence a value `v==b[k]` belongs to bin `k+1`; the last tail is still encoded. The kernel searches for the **first breakpoint strictly greater than** `v`, equivalent to counting boundaries `<=v`.

```text
Breakpoints [0,1], input [-1,0,0.5,1,2], n=w=5, Raw:
Symbols = [0,1,1,2,2]
```

### Q13a.6. Can two words with the same `BreakpointID` be compared?

Not necessarily. `BreakpointID` is audit text, not a mathematical identifier. `EncodingID` is a canonical SHA-256 of `Segments`, `Alphabet`, normalization/constant policy, normalization/table version and actual breakpoint binary64 bits (with `-0.0` canonicalized to `+0.0`). `Mindist` requires both result IDs to agree with its encoder and verifies the source lengths and shapes separately. Two arrays both labeled `"custom"` but containing different thresholds produce incompatible IDs (`ErrIncompatible`). Changing `ReturnPAA` or the audit label alone does not change the identity.

### Q13a.7. Does `EncodingID` authenticate a stored Result?

No. `Result.EncodingID` and `Result.Symbols` are exported writable fields. The hash establishes **configuration compatibility** under honest construction, not authenticity of an untrusted record. `Mindist` validates accessible dimensions, symbols, finite numeric metadata and optional PAA consistency; it cannot prove that a hand-built word represents particular original observations. Authenticate or sign persisted data in the application if your threat model requires it.

### Q13a.8. Why did `Mindist` return `ErrIncompatible`, `ErrShape`, or `ErrNonFinite`?

These errors protect different boundaries. `ErrShape` rejects malformed slice lengths, out-of-range symbol codes or contradictory fit/PAA metadata. `ErrNonFinite` rejects publicly modified NaN or infinity in PAA/mean/stddev. `ErrIncompatible` rejects mismatched original source lengths, segmentation, alphabet, EncodingID or normalization version (as well as missing identity metadata). Result structure is validated **before** compatible-policy comparison, so a malformed hand-built Result may report `ErrShape` even when its ID is also mismatched. Classify through `errors.Is`, not text matching.

### Q13a.9. Is MINDIST a true metric? Does zero mean the series match?

Neither. With thresholds `[0,1]`, the gap of bins `(0,1)` is `0`, `(1,2)` is `0`, yet `(0,2)` is `1`. Therefore distinct symbolic words can have zero dissimilarity and triangle inequality can fail. Even a zero MINDIST with identical words does not prove identical PAA averages, raw windows, or DTW alignments. Treat MINDIST as a **lower-bound dissimilarity**, not an ordinary metric suitable for metric-tree assumptions.

### Q13a.10. Can SAX MINDIST safely prune DTW candidates?

**No.** `Q=[0,0,1]`, `C=[0,1,1]`, `w=3`, `a=3`, thresholds `[0.25,0.75]` produce words `[0,0,2]` and `[0,2,2]`, so Euclidean MINDIST is `0.5`; an unrestricted DTW path matches equal values and has **zero** cost. A positive MINDIST does not certify rejection under nonlinear warping. Use SAX to build an explicitly **approximate** candidate shortlist, then calculate DTW on those original windows and measure `Recall@K` against exhaustive DTW. For globally exact fixed-length DTW search, study `dtw.Search` and its DTW-specific machine certificates.

### Q13a.11. Does the Euclidean lower bound remain a certified floating-point cutoff?

The inequality `MINDIST <= D_PAA <= D_2` follows in **exact arithmetic** for compatible transformed windows with fractional segment weighting `n/w`. Implemented `float64` normalizations, PAA values and norm calculations can round; the package does not publish a directed-downward machine certificate. If correctness depends on excluding every candidate below a Euclidean cutoff, derive and verify conservative numerical error bounds separately. A casual `epsilon` does not supply that proof.

### Q13a.12. What data does `ReturnPAA=true` return? Can I mutate it?

It returns a detached slice of **transformed** PAA means. Under `Raw`, these are physical-unit averages; under `ZPopulation`, they are means of window-local Z coordinates. Mutating that slice or `Symbols` changes your Result but never the immutable Encoder or another successful result. Do not mutate the same result concurrently with `Encoder.Mindist`, because the package does not snapshot publicly writable Results. `ReturnPAA` does not affect Symbols or EncodingID; it can increase validation work when the PAA slice is present.

### Q13a.13. How do I choose a facade for a production monitoring pipeline?

Use `sax.EncodeGaussian(values,w,a)` for an occasional normalized Gaussian word, `sax.EncodeRaw(values,w,a)` when supplied values are already in appropriate Raw units, and `sax.EncodeRawWithBreakpoints(values,w,boundaries,label)` for an occasional domain-calibrated word. Use `NewEncoder(opts)` once followed by repeated `encoder.Encode(window)` for many windows; this avoids repeatedly cloning/hashing the same mathematical configuration. `sax.MindistValues(left,right,opts)` is a one-shot pair comparison that encodes both sides on each call; it does **not** retain words or build an index.

### Q13a.14. How do I connect `series`, SAX and DTW safely?

Validate cadence and gaps with `series.Validate` first if you have timestamps, choose complete sample-index windows with `series.WindowRanges`, and explicitly reject or segment windows crossing missing timestamp slots. `sax.Encode` neither checks timestamps nor consumes `series.TransformResult.Valid` masks; a warmup placeholder zero must not be mistaken for a genuine sample. Encode windows of equal length with one compatible policy and retain the original observations if DTW refinement is required. Ranking a SAX shortlist followed by exact per-candidate `dtw.Align` is only **approximate global retrieval**. For error tables, proofs and five executable synthetic scenarios see [`SAX.md`](SAX.md), [`sax/doc.go`](../sax/doc.go) and [`sax/example_test.go`](../sax/example_test.go).

---

## 13b. OLS FAQ

### Q13b.1. Why do I need `ols` if `matrix.SolveLeastSquares` already exists?

`matrix` solves a **given** rectangular linear system using pivoted QR.
`ols` defines the regression model around that solver: intercept semantics,
training-only feature scaling, rank gate, back-conversion to physical
coefficients, independent training SSE and a frozen `Model.Predict` API.

Use `stats` when you need moments or a stand-alone fitted feature transform,
`matrix` for general linear systems or reusable QR factors, and `ols` when a
response vector must be explained by a stable linear factor model. These
packages do not silently substitute for one another.

### Q13b.2. Why does `FitDefault` center features even though Standardize is false?

The explicit intercept basis is `design=[ones, x-trainingMean]` with unit
scales; this is a numerically useful *change of coordinates*, not a change of
the class of affine models. `ols` restores the physical intercept via
`b=thetaIntercept-sum(trainingMean[j]*beta[j])`. Disabling `Standardize`
disables population scaling but **does not disable intercept centering**.

When `NoIntercept=true`, centering is completely forbidden. With
`Standardize=true` the design is `x/trainScale` and the returned intercept is
exactly positive zero. This prevents the hidden-constant defect of fitting
`x-mean` while publicly advertising an origin-constrained model.

### Q13b.3. Why did `Predict` return `2.999999999999999` for an exact line?

Because Householder QR, coefficient back-conversion and FMA calculations
round in IEEE-754 `float64`. A few ULPs of error are normal; the exact
mathematical line `y=1+2*x` does not imply bit-identical computed predictions.

For a fixture with expected value `3`, write a scale-aware check such as:

```go
const tolerance = 1e-12
if math.Abs(predicted[0]-3) > tolerance*math.Max(1, math.Abs(3)) {
    t.Fatalf("prediction %.17g not close to 3", predicted[0])
}
```

The test snippet assumes an existing `predicted` slice, `t *testing.T`, and
`math` import. **Do not round internal coefficients**, weaken the QR solver,
or alter an exact `NoIntercept`/rank/ownership assertion to placate a bad
float-equality test.

### Q13b.4. Why does a constant or duplicate feature return `ErrRankDeficient`?

OLS v0.1 requires **full numerical column rank**. Duplicate predictors do
not identify separate coefficients; a constant predictor alongside an
intercept duplicates the ones column. `stats.UnitScaleConstant` sets scale
one for a constant predictor but does not erase the column, so the QR solver
correctly rejects the deficient design.

Fix the **caller feature schema** deliberately, then refit if appropriate.
Do not silently drop a predictor, introduce a pseudoinverse, or reduce the
rank tolerance until a dependent model produces coefficients. A nonzero
constant *without* an intercept can be a valid rank-one predictor.

### Q13b.5. Why are `ConditionEstimate` and `ResidualSumSquares` not p-values?

`ConditionEstimate` measures the induced 1-norm condition estimate
`kappa_1(R)` of the **transformed QR triangular factor**; standardizing
features changes that diagnostic. It is not the spectral condition of raw X,
a coefficient error bound, or proof of statistical independence.

`ResidualSumSquares` is `sum((y_i-Predict(x_i))²)` on **training data** in
response units squared, not an RMS error, holdout score, or significance
test. Financial and industrial data may be autocorrelated or
heteroskedastic; OLS intentionally does not invent IID Gaussian t/F tests,
p-values, or confidence intervals.

### Q13b.6. Can I standardize test data again before calling `Predict`?

No. `FitStandardized` learned training means and population scales
(`DDOF=0`) once. It then converted slopes and intercept back to physical
units. `Model.Predict` already expects the **original feature units** and
the exact training column order. Independently normalizing or refitting on
test data changes the model and leaks future information into inference.

A 0-row inference matrix with the trained number of columns is valid and
returns a **non-nil empty** prediction slice; a different feature count is
`ErrShape`. Input matrices must remain unchanged during each prediction.

### Q13b.7. Can I mutate `FitResult.Model`, a coefficient slice, or input matrix?

The exported `FitResult` fields are caller-owned metadata. Replacing
`result.Model` does not mutate the previously retained `*ols.Model`.
`Coefficients()` returns a detached slope slice; editing it cannot affect
predictions. `Predict()` returns a new output slice per call. On the other
hand, mutating a passed `matrix.Dense` or a write-through `matrix.View`
while a `Fit`/`Predict` call is reading it is unsupported; an immutable
Model is not a snapshot transaction over caller data.

`(*ols.FitResult).IsNil()` checks the receiver pointer only. A non-nil
zero-value result is still present; check the error returned by `Fit` to
establish training success. Nil `Model.Predict` returns `ErrNilInput`, while
nil `Model.Coefficients` returns `(nil,0)` because it has no error result.

### Q13b.8. What happens if all inputs are finite but SSE overflows?

An individual residual near `1e200` is finite, but its square may exceed
`math.MaxFloat64`. `ols` uses a scaled sum-of-squares accumulator so it does
not falsely overflow on intermediate squaring; if the **required final SSE**
cannot be represented, Fit returns `nil, ErrOverflow`. Failing Fit never
publishes a partial model, and a canceled Predict never publishes a partial
prediction slice.

### Q13b.9. What is the public validation and cancellation order?

`Fit`: context → RankTolerance → nil/typed-nil → shape/count → finite input
cells → QR rank → coefficient/SSE computation. A canceled context takes
precedence over invalid options, and invalid options precede data reads.
Use `errors.Is(err, ols.ErrRankDeficient)`, not string comparisons;
wrapped QR/stats roots remain available to `errors.Is`. Cancellation is
cooperative, including after the non-context-taking stats fitter returns.

### Q13b.10. Which realistic regression scenarios are included?

Five executable `// Output:` examples cover a 24-month CAPM beta and alpha
with a stress scenario, a 24-point industrial sensor calibration, a strict
zero-origin two-channel charging model, rank-deficient duplicate telemetry
with caller-driven schema correction, and a chronological six-day holdout
for energy-load forecasting. Their data are **synthetic**, not live feeds.
See [`OLS.md`](OLS.md), [`ols/doc.go`](../ols/doc.go), and
[`ols/example_test.go`](../ols/example_test.go) for the proofs and real API.

---

## 13c. Ridge FAQ

### Q13c.1. When should I use `ridge` rather than `ols` or a raw matrix QR solve?

Use `ols` when the unregularized linear factor model is identifiable under its full-column-rank contract. Use `ridge` when your **explicit modeling decision** is to control slope magnitude and accept shrinkage bias, especially with duplicated/correlated predictors or `p>n`. Use `matrix.SolveLeastSquares` for a general rectangular least-squares system that already has its design and RHS and does not need a regression-specific training transform or public objective diagnostics.

The canonical Ridge objective is **`SSE/(2*n) + Lambda/2*sum(theta_j²)`**. `Lambda` must be finite and strictly positive. `Lambda=0` returns `ridge.ErrInvalidOption`; it does not invoke OLS. The package does not choose Lambda, cross-validate, compute p-values or provide a statistical uncertainty model.

### Q13c.2. Which API fits my data, and what does the Lambda argument mean?

| Need                                 | Call                                                                | Important contract                                       |
|--------------------------------------|---------------------------------------------------------------------|----------------------------------------------------------|
| Affine, original feature units       | `ridge.FitWithLambda(ctx,X,y,lambda)`                               | Penalize only physical slopes; intercept is free         |
| Affine, population-scaled predictors | `ridge.FitStandardized(ctx,X,y,lambda)`                             | Penalize optimization slopes theta; return physical beta |
| Physical zero-origin calibration     | `ridge.FitThroughOrigin(ctx,X,y,lambda)`                            | Intercept exactly +0; no centering                       |
| Zero-origin with balanced predictors | `ridge.FitThroughOriginStandardized(ctx,X,y,lambda)`                | Scale-only, no implicit offset                           |
| Custom relative QR cutoff            | `ridge.Fit(ctx,X,y,ridge.Options{Lambda: ..., RankTolerance: ...})` | Positive Lambda remains mandatory                        |

These are wrappers over the one canonical `Fit`. None performs automatic hyperparameter selection or changes the penalty when a solve fails. In particular, a standardized `theta` and the physical `beta` differ: `theta_j=beta_j*trainingScale_j`. Therefore the same numeric Lambda does not define an equivalent objective with `Standardize=true` versus false.

### Q13c.3. Why is the intercept never penalized?

The affine model is `y≈b+X*beta`. An intercept represents the output level when the selected feature reference is zero. Shrinking it with L2 would incorrectly pull that level toward zero. Ridge therefore creates **exactly p penalty rows**, with zeros in the intercept column, and puts `sqrt(n*Lambda)` only on slope diagonals. An original constant feature is **not the same as an intercept**: its slope remains penalized, and the explicit unpenalized intercept can absorb the constant level.

At `X=[-1,1]`, `y=[4,4]`, with any finite positive Lambda, the optimal intercept is `4`, both residuals vanish, and the slope is zero. This is a useful oracle for catching accidental intercept penalization.

### Q13c.4. Why must `NoIntercept + Standardize` be scale-only?

Centering would change the physical prediction law. If `z=(x-mean)/scale` and an apparently zero-intercept model fits `theta*z`, then in original units the output is `x*(theta/scale) - mean*(theta/scale)`: **the second term is a hidden intercept**. The origin-constrained mode must instead use `z=x/scale`, ensuring that `Predict([0,...,0])` returns exactly positive zero. The population scales (`DDOF=0`, `stats.UnitScaleConstant`) are learned only on training rows and are never fitted again during inference.

### Q13c.5. Why can a correct huge-Lambda coefficient become `-0` in naive QR?

A safe `sqrt(n)*sqrt(Lambda)` prevents intermediate multiplication overflow, but it does **not** prevent loss of precision during `Qᵀ*y` in Householder QR. With `NoIntercept=true`, `X=[1,2]`, `y=[1,2]`, and `Lambda=1e100`, the exact optimum is `5/(5+2e100) ≈ 2.5e-100`. This is a representable, nonzero coefficient. If the augmented matrix is fed to QR in the naive row order `[1,2,sqrt(2*Lambda)]ᵀ`, subtracting large RHS components during a reflector can destroy the tiny projection, producing `-0`.

`ridge` protects this by constructing the canonical augmented model, then passing a **mathematically equivalent QR representation** that moves the *zero-response penalty rows first*, places slope columns before the optional intercept, and equilibrates columns by strictly positive divisors. If `P` is a row permutation, `C` a column permutation and `D` positive diagonal, the solver sees `P*A*C*D^-1` and `P*yAug`; the recovered canonical coefficients are `C*D^-1*gamma`. This is preconditioning, **not** a silent change of Lambda. Full mathematics and the independent oracles are documented in [`RIDGE.md`](RIDGE.md#21-why-the-qr-representation-needs-its-own-coordinate-system).

### Q13c.6. Does strict Lambda>0 guarantee the numerical QR solver will always succeed?

No. The penalty establishes positive definiteness of the **exact slope Hessian**. With an intercept and at least one row, the exact augmented matrix has full column rank. But float64 can still underflow an extremely small penalty contribution or fail the QR relative rank cutoff. `RankTolerance=0` selects the `matrix` solver's dimension-dependent automatic cutoff; positive tolerances are relative to the pivoted QR diagonal of the *equilibrated* system. A numerical rank failure returns `ErrRankDeficient` (also matching `matrix.ErrRankDeficient` when reported by that dependency). Never decrease tolerance or increase Lambda automatically to make a test pass.

### Q13c.7. Can Ridge fit `p>n` or duplicated columns?

Yes, when the augmented QR rank is accepted. The original `n × p` matrix may be wide or have repeated columns; the additional `p` diagonal penalty rows make the regularized slope problem algebraically identifiable. `ridge` does **not** drop duplicates, expose a pseudoinverse, or change into OLS. For perfectly duplicated inputs and equal penalty weights, the exact optimum assigns equal slopes in the matching symmetric coordinates. Unit-scale constant features remain present and penalized, not silently discarded.

### Q13c.8. How should I interpret `ResidualSumSquares`, `Penalty`, and `Objective`?

`ResidualSumSquares` is **unscaled** original-unit training SSE. `Penalty` is the already-normalized term `Lambda*sum(theta_j²)/2` in training optimization coordinates; it never includes the intercept. `Objective` equals `ResidualSumSquares/(2*n) + Penalty`. With `Standardize=false`, `theta=beta`; otherwise use the learned training scales to interpret the implicit penalty. The public `Coefficients()` method returns original-unit beta, so recomputing penalty from its output after standardized training is generally **wrong**.

A low training Objective is not a cross-validated loss, holdout MAE, confidence interval, or portfolio-risk guarantee. For a performance claim, measure an independent, suitably ordered validation dataset with exactly the same frozen model and feature schema.

### Q13c.9. How do I choose Lambda without leaking future information?

This package deliberately has no automatic selection policy. In financial, industrial and temporal settings, define training/validation splits **in time order** and choose Lambda using a separate application-controlled sweep. Fit population means/scales only on the candidate's training fold; compute validation error using physical heldout features and the fitted `Model.Predict`. For the final model, refit on the intended training period with the selected explicit Lambda, then test once on untouched evaluation data. A good result for a synthetic fixture is not evidence of performance on live measurements.

### Q13c.10. What do empty batches, nil receivers, and typed nils mean?

A successful fit returns a non-nil `*ridge.FitResult` implementing `core.Nilable`. `(*FitResult)(nil).IsNil()` is true, including behind an interface; `(&FitResult{}).IsNil()` is false because this tests pointer **presence**, not training success. Nil `(*Model).Coefficients()` returns `(nil,0)`, while nil or unfitted `Model.Predict` returns `ErrNilInput`. `Predict` accepts a detached, non-nil empty result for a `0 × trainedFeatureCount` matrix; an incompatible feature width returns `ErrShape`. A nil context produces `ErrInvalidOption`; cancellation preserves `ctx.Err()`.

### Q13c.11. What is the failure order, and can an output be partially returned?

`Fit`: context → Lambda and rank options → nil/typed-nil → shape/checked allocation → entire input finiteness → transformation/QR rank → physical coefficients/SSE/Penalty/Objective. `Predict`: context → model/matrix presence → shape/size → *all* input finiteness → row calculation. `ErrNonFinite` means an invalid source observation; `ErrOverflow` means unrepresentable derived arithmetic or size. Matrix/stats root causes remain available via `errors.Is` when wrapped.

Both APIs return **nil output on error**, not partially filled results. Neither retains or mutates caller-owned training matrices. Do not mutate a matrix during either training or inference; successful Model inference can be called concurrently with stable inputs and independent per-call buffers.

### Q13c.12. What real-world scenarios and numeric oracles are executable?

Six deterministic `// Output:` examples model a chronological multi-factor portfolio holdout plus macro stress tests, 64-channel optical spectroscopy trained on only ten mixtures, AR(12) monthly energy demand against a seasonal-naive baseline, strict-zero dosimetry with an illustrative alarm threshold, the inability to identify 32 channel sensitivities from a single observation, and nonzero extreme-Lambda coefficients. All data are **synthetic engineering fixtures**, not market feeds or certified safety measurements. The independent test suite also checks correlated factors, wide systems, large Lambda with and without intercept, `Lambda=math.MaxFloat64`, original-unit predictions, and `core.Nilable`.

For the precise contract, see [`RIDGE.md`](RIDGE.md), [`ridge/doc.go`](../ridge/doc.go), [`ridge/example_test.go`](../ridge/example_test.go), and the runnable [`examples/ridge/main.go`](../examples/ridge/main.go) scenario.

---

## 13d. PCA FAQ

### Q13d.1. Why does PCA reject `ThinSVD` instead of silently running the covariance solver?

`ThinSVD` is a reserved, **unimplemented** enum value. `pca.Fit` returns `ErrUnsupportedSolver` before reading matrix data. A covariance eigensolver squares the conditioning burden and cannot be relabeled a direct SVD. Use `CovarianceEigen` only when its numeric acceptance and validation requirements are appropriate; there is no hidden fallback.

### Q13d.2. How are `Components=0` and `RankTolerance=0` interpreted?

`Components=0` retains all **detected numerical-rank** directions. `RankTolerance=0` selects `epsilon*max(n,d)` relative to the **covariance eigenvalue spectrum**. If a caller explicitly requests more directions than the detected rank, the result is nil with `ErrRankDeficient`. A smaller threshold does not magically restore information lost when forming covariance.

### Q13d.3. Does the result's explained-variance ratio sum to one?

Only when the retained eigenvalues account for the full accepted nonnegative covariance trace, within floating-point rounding. The denominator always uses the **complete accepted spectrum**. For `k<rank`, a positive omitted tail means the sum of published ratios is below one. Do not renormalize truncated ratios or treat high training variance retention as proven holdout reconstruction quality.

### Q13d.4. Why do I get `ErrNotConverged` or `ErrNumerical` for apparently simple data?

These errors describe **different** failures. `ErrNotConverged` means Jacobi exhausted the configured rotation budget; `errors.Is(err,matrix.ErrMatrixEigenFailed)` also remains true. `ErrNumerical` means the returned covariance/eigenpairs fail PSD, symmetry, orthogonality or equation-residual acceptance. A very loose `EigenTolerance` can let Jacobi stop prematurely but **cannot** relax the independent PCA publication check. Never suppress either error or clip all negative eigenvalues.

### Q13d.5. Why does Transform not center each incoming test batch?

Re-centering changes the coordinate origin and introduces holdout leakage. `Model.Transform(ctx,x)` uses **only** fitted training means and (if enabled) training population scales. A single out-of-range observation must retain a nonzero score rather than being translated to zero because its one-row batch mean equals itself.

### Q13d.6. Is `Components()` a writable view? Is `FitResult.IsNil()` a fit-status check?

No. `Components()` returns a detached `d×k` matrix whose mutation cannot change the fitted model. `FitResult.IsNil()` satisfies `core.Nilable`: a nil receiver is absent, while `&pca.FitResult{}` is present but unfitted. Check `Fit`'s error and the `Model` returned on success. Concurrent read-only inference on the same fitted model is safe because each call owns its workspace; caller mutation of input matrices during calls is not.

### Q13d.7. How do I use PCA for industrial SPE or Hotelling T² alerts?

Compute residuals from the frozen model, then independently calibrate a domain-specific threshold using **healthy validation data excluded from Fit**. Equal physical units permit an unweighted physical-unit SPE; mixed units require justified scaling or a single-channel diagnostic. The current `pca` package **does not** publish `SPE`, `T²`, threshold fitting, Nelson–Siegel estimation, anomaly root-cause attribution or automatic safety decisions. Do not mistake a demonstration's calibrated engineering threshold for a certified alarm limit.

### Q13d.8. How should I interpret repeated eigenvalues, very high condition numbers, or rotation limits?

Signs are canonicalized by the first maximum-absolute coordinate, but repeated eigenspaces permit rotations of the entire basis. Compare projectors only across **completely retained** degenerate subspaces. Covariance rank is more fragile than direct SVD for nearly dependent features; do not force a full-rank result with a smaller tolerance. `MaxRotations=0` means checked `50*d*d` Jacobi rotations, not sweeps. The underlying `matrix.EigenSym` has no context-aware internal cancellation hook.

### Q13d.9. Can I use PCA with constant features, or interpret all zero-rank outputs as valid?

Without standardization, a constant feature is allowed if other columns provide positive total covariance variance; its centered contribution vanishes. With `Standardize=true`, `stats.RejectConstant` rejects **any exactly constant training column** through `ErrZeroVariance`, rather than silently removing it. A constant *entire* training batch is also `ErrZeroVariance`. A nonzero spectrum can still have no eigenvalue above an excessively high user rank cutoff: this is `ErrRankDeficient`, not an acceptable zero-column fitted Model. No successful Fit publishes a `Model` with `k=0`.

### Q13d.10. How should I select k and compare alternative PCA models without leakage?

Fix the chronological/entity partition first. Fit means, scales and eigenvectors on **training rows**; choose `k`, rank policy and application-specific thresholds using only training and designated calibration/validation data, then keep an untouched final holdout for reporting. A high in-sample `ExplainedVarianceRatio` is not proof of good holdout reconstruction, anomaly detection recall or future yield-curve forecasting. Compare model variants in the same feature units and against a training-only baseline.

### Q13d.11. How does physical residual energy differ from the model's training residual?

`FitResult.ReconstructionResidual` is a **single Frobenius norm over all training rows in prepared coordinates**, not a per-observation SPE. `Model.Reconstruct(ctx,observations)` returns **original-unit** approximations. For a standardized model, summing raw squared differences between original and reconstructed columns produces a **physical-unit weighted mixture**, not automatically the standardized training objective. `Model` intentionally exposes no public scale snapshot; if a downstream score needs a dimensionless metric, define and freeze a separate, audited weighting policy. For equal-unit sensor channels an original-unit SPE is meaningful once calibrated separately.

### Q13d.12. Does a PCA yield-curve factor represent Nelson–Siegel level, slope, or curvature?

Not automatically. PCA eigenvectors diagonalize **sample covariance** for the selected maturities and observation window. Their signs are fixed for display, but economically named level/slope/curvature interpretations require inspecting maturity loadings and checking stability on holdout curves. Nelson–Siegel uses a separate parametric yield-curve model, which `pca` does not fit; PCA scores also do not predict tomorrow's yields.

### Q13d.13. What must I validate before admitting PCA to a production analytics pipeline?

Use the exact repository toolchain; check `go test ./pca -count=1`, `go test -race ./pca`, `go test ./...`, `go vet ./...`, and `golangci-lint run`. Independently test extreme covariance construction, residual identities, repeated-subspace projectors, holdout-only evaluation, and numeric error classification through `errors.Is`. Benchmark **real** dependencies, not a compatibility stand-in. See [`PCA.md`](PCA.md) for the full API and [`pca/example_test.go`](../pca/example_test.go) for executable workflows.

---

### 13e. HMM FAQ — causal regimes, smoothing, decoding and EM

#### Q13e.1. When do I use `FilterStep`, `FilterWithHistory`, `Smooth` or `Decode`?

They return different mathematical objects. `FilterStep(ctx,prev,x)` returns the causal `P(z_t|x_0:t)` for one new observation and a detachable `FilterState`. `FilterWithHistory` returns those same *causal* distributions at each time, `Smooth` returns `P(z_t|x_0:T-1)` using **future** observations, and `Decode` returns the maximum-joint-probability sequence of states. Viterbi is not the per-row argmax of smoothed probabilities. For online alerts and historically faithful backtests, only the **causal** APIs are appropriate.

#### Q13e.2. Can adding tomorrow's observation change today's posterior?

For a successful causal filter, **no**. At time `t` the recursion reads only the previous normalized log posterior, fixed transition parameters and the current emission score. Appending future observations leaves all preceding filtered rows unchanged (subject to the original and appended calls both succeeding). Smoothing intentionally *can* revise them; in an absorbing two-state model, an initial reassuring event can later be reclassified after successive alarms. See [`HMM.md`](HMM.md#4-causal-forward-and-stateful-replay).

#### Q13e.3. Why does `LogLikelihood` remain finite on 100,000 small-probability observations?

The package never multiplies hundreds of raw probabilities together. It works in log space and normalizes each Forward step with `LogSumExp`. `LogSumExp(all -Inf)` returns exactly `-Inf` and is classified as `ErrImpossibleSequence` if no categorical state can produce an observation. Gaussian log densities are **not** probabilities and can be positive. Finite inputs that lead to an unrepresentable log-density or total yield `ErrOverflow`, not `ErrImpossibleSequence`.

#### Q13e.4. What are the accepted categorical and Gaussian shapes?

Categorical construction requires `Prior[K]`, `Transition[K,K]` and `Emission[K,A]` where `K,A>=1`; observations are `[]uint32` with each symbol `<A`. Gaussian construction requires `Prior[K]`, `Transition[K,K]`, `Means[K,D]`, and `Variances[K,D]` with `D>=1`, finite means and strictly positive finite variances. Gaussian observations are `matrix.Matrix` with **T rows, D columns**. Inputs with missing values, unknown symbols, NaN and infinities are rejected. A numeric signal cannot be used directly as a categorical index without a stable encoder.

#### Q13e.5. Why does a model with zero transitions or emissions fail at runtime?

Exact zero probabilities are legal and remain zero after constructor validation. If at some step every hidden state has zero possible path mass, the sequence is **impossible under the chosen model**, not numerically too small. `Filter`, `Smooth` and `Decode` return nil result plus `ErrImpossibleSequence`; they do not synthesize tiny epsilon mass or a `NaN` posterior. For Gaussian densities with positive variances, the mathematical support is the whole real feature space; nonrepresentable arithmetic is an overflow problem instead.

#### Q13e.6. Why is a previous `FilterState` rejected after retraining?

`FilterState.ModelID` is a SHA-256 digest of normalized parameters, emission family, dimensions and algorithm revision. `FilterStep` checks the ID, K log posterior elements, `Steps>=1`, finite accumulated log evidence, and log normalization within `1e-12`. Retraining changes parameters and normally changes the ID. **Never** fix an `ErrStateVersion` by editing the public string: initialize a new filter or explicitly migrate the application state under an external, mathematically justified policy. Copying a Go struct does **not** deep-copy its `LogPosterior` slice; call `append([]float64(nil), old.LogPosterior...)` when an independent editable copy is required.

#### Q13e.7. What does `Pseudocount` do in Baum–Welch?

`Pseudocount=lambda>=0` corresponds to symmetric Dirichlet MAP prior `alpha=1+lambda` over prior/transition rows and categorical emission rows. Its optimization target is **data log likelihood plus `lambda * sum(log(simplex cell))`**. With `lambda=0`, the penalty is disabled completely, including exact zero cells. Gaussian means and variances are not Dirichlet-regularized. Never test only whether raw log likelihood increases when `lambda>0`; the objective used for convergence has changed.

#### Q13e.8. Why can't I merge independent training sessions into one long series?

With independent sequences, each one gets its **own** initial hidden state drawn from Prior. Prior counts sum `gamma_0` over **every** sequence; transition counts sum `xi_t` over **only** `t<T_s-1` inside that sequence. Concatenating sessions invents transitions across boundaries and changes the statistical model. Two independent length-one sessions provide exactly zero observed within-sequence transitions, though both contribute prior and emission responsibilities.

#### Q13e.9. Why does `FitGaussian` reject a positive initial variance below the floor?

`NewGaussian` accepts any strictly positive finite diagonal variance. `FitGaussian` additionally requires **initial variance >= effective `VarianceFloor`**, which defaults to `1e-8` in *feature²* units. The constrained M-step uses `max(empiricalVariance,VarianceFloor)`—it does not add the floor as noise. If the seed is outside the constrained feasible domain, it is `ErrInvalidOption`, not a silently repaired start. Scaling training features changes the physical interpretation of the variance floor; freeze preprocessing units first.

#### Q13e.10. What happens to a dead hidden state with zero posterior occupancy?

No state is deleted or randomly reseeded. With `lambda=0`, a categorical emission row or transition row with no expected counts keeps its previous values. Gaussian means and variances for a zero-occupancy state remain unchanged. With `lambda>0`, zero-count categorical/transition simplexes are normalized from the pseudocount and become uniform. `Diagnostics.Occupancy` reports final posterior expected visits; its length remains K.

#### Q13e.11. Does an EM error always mean a nil model?

**No.** Invalid options, invalid input, impossible observations or context cancellation return nil. `ErrNotConverged` means the iteration budget was exhausted and returns the **last accepted immutable model** with `Converged=false`; `ErrNumerical` on objective decline returns the last accepted model for diagnosis. Neither constitutes a successful fit. The current code rejects *any* negative finite penalized-objective change (including tiny roundoff) rather than allowing a tolerance band; callers must not silently reinterpret that as convergence. Use `errors.Is` and inspect `Diagnostics.Objective`, `Iterations`, `Converged`, and `Occupancy`.

#### Q13e.12. What does `IsNil()` test, and is an empty result valid?

`FilterResult`, `SmoothResult`, `DecodeResult`, `CategoricalFitResult`, and `GaussianFitResult` each implement `core.Nilable`. Their `IsNil()` checks **only whether the receiver pointer is nil**, including typed nil held in an interface. A non-nil zero-valued result is not necessarily a valid inference artifact. A successful `FilterResult` may intentionally have `Posterior==nil` when `ReturnHistory=false`; check `Last` instead. Do not treat a nil history matrix as a failed filter.

#### Q13e.13. How are input matrices validated without `reflect`?

Supported pointer-backed lvlath matrix types implement the nil-safe `IsNil()` convention. The HMM checks the interface for nil, optionally calls `IsNil`, validates dimensions and finite cells via `Matrix.At`, then copies values into private row-major buffers. A custom pointer-backed matrix must implement `IsNil()` to make its typed-nil discoverable without reflection. `*matrix.Dense` is useful for efficient reads, but the O(TK²) transition hot path already operates on private flat parameters and does not access another package's unexported backing slice.

#### Q13e.14. Is immutable model inference safe from multiple goroutines?

Independent `Filter`, `Smooth`, `Decode`, and `FilterStep` calls use per-call workspaces and do not mutate fitted parameters. They are safe concurrently **only while caller-owned input matrices/slices remain unchanged**. Public `FilterState.LogPosterior`, histories, and diagnostics are mutable by their owners and require synchronization if shared for writes. There is no snapshot isolation across concurrent `matrix.Set` calls and no public mutable streaming accumulator.

#### Q13e.15. How much memory do long sequences use?

Categorical `FilterDefault` requires O(K) live workspace plus immutable model; `FilterWithHistory` returns a T×K matrix. `Smooth` uses O(TK) messages and a detached T×K result; Viterbi stores T×K predecessor indexes. EM additionally materializes `(T_s-1)*K²` pair responsibilities for one sequence at a time. Gaussian batch inference also copies T×D observation cells on input validation. These are **peak live space**, not a promise of zero per-observation allocations: current batch filtering still creates per-step snapshots. Memory budgets are caller-owned even when integer allocation products are checked.

#### Q13e.16. Where can I see practical, executable scenarios and independent proofs?

[`hmm/example_test.go`](../hmm/example_test.go) exercises clickstream intent, live market regime monitoring, retrospective incident interpretation, industrial state decoding, and independent-session EM. [`examples/hmm`](../examples/hmm) (when installed) shows a standalone synthetic streaming regime scenario. The `hmm_test` suite checks exact tiny hidden-path enumeration, prefix invariance, state detachment, long-sequence log evidence, Viterbi ties, MAP objective behavior and concurrent inference. The worked formulas and decision boundaries are in [`HMM.md`](HMM.md). Real deployments must separately validate regime semantics, policy thresholds and out-of-sample calibration.

---

## 14. Builder and gridgraph tips

### Q14.1. When should I use `builder`?

Use `builder` when you need reproducible graph shapes for:

* examples;
* tests;
* benchmarks;
* algorithm demonstrations;
* golden outputs.

Builder-generated fixtures should encode meaningful topology, not hide arbitrary setup.

### Q14.2. When should I use `gridgraph`?

Use `gridgraph` when your domain is naturally a lattice:

* grid maps;
* pathfinding demos;
* obstacle experiments;
* 4-neighbor/8-neighbor movement;
* teaching BFS/Dijkstra visually.

Do not manually wire large grids unless the example is specifically about low-level graph construction.

### Q14.3. How should random fixtures be handled?

Use fixed seeds and document the shape.

Bad:

```go
rand.Seed(time.Now().UnixNano())
```

Good:

```go
const seed = 42
```

Reproducibility is part of the benchmark/test contract.

---

## 15. Errors and diagnostics

### Q15.1. Why are error sentinels so important?

They let callers branch safely while still preserving helpful context.

```go
if errors.Is(err, matrix.ErrDimensionMismatch) {
    // stable protocol branch
}
```

Error strings are for humans. They may change without changing the protocol.

### Q15.2. How should wrapped errors behave?

Wrapping should preserve sentinel identity.

Good:

```go
return fmt.Errorf("build adjacency: %w", matrix.ErrUnknownVertex)
```

If an adapter needs to expose both an algorithm-level error and an underlying root cause, it should preserve both through the package’s chosen wrapping/join strategy.

### Q15.3. How do I debug a failing algorithm call?

Ask in this order:

1. Is the graph/matrix nil or wrong shape?
2. Are options valid and finalized once?
3. Does the graph capability match the algorithm?
4. Are weights/numeric values legal?
5. Is the target known but unreachable, or unknown?
6. Is path tracking enabled when asking for a path?
7. Is the result partial by contract?
8. Are docs/examples synchronized with current API?

---

## 16. Determinism and concurrency

### Q16.1. What does “deterministic” mean in lvlath?

It means the observable result is stable for the same input state, options, and package version.

Examples:

* BFS order follows the documented neighbor order.
* DFS result order means documented finish order.
* Dijkstra equal-cost witnesses follow documented tie/strict-improvement rules.
* Matrix adapters use stable vertex/edge order and fixed loop order.
* Components are sorted according to their contract.

### Q16.2. Does determinism mean every mathematically valid answer is identical across packages?

No. It means each package publishes its own deterministic choice among valid answers. For example, one shortest path may be selected as a witness, not all possible shortest paths.

### Q16.3. Is `core.Graph` safe for concurrency?

`core` is designed with internal locking for graph storage and controlled mutations. Algorithm reproducibility still requires stable topology while an algorithm runs unless the package explicitly documents snapshot isolation.

### Q16.4. Is `matrix.Dense` safe for concurrency?

Concurrent read-only access is safe when no goroutine mutates the matrix. Concurrent writes or read/write races require external synchronization. Views share storage with their base matrix.

### Q16.5. Is `stats.Standardizer` safe for concurrent inference?

Yes, once successfully fitted. Its parameter slices are private and immutable, and each `Transform` or `InverseTransform` allocates a detached output. Input matrices must remain unchanged during the entire call, including both read passes. `stats.Moments` is **not** safe for simultaneous updates/merges without external synchronization.

### Q16.6. Can multiple goroutines share one fitted PCA Model?

Yes. `pca.Model.Transform`, `InverseTransform`, and `Reconstruct` use immutable private training parameters and per-call output/workspace. Keep every caller-owned input matrix stable for each call, and do not infer that `matrix.Dense` itself permits concurrent mutation. `Components()` returns a detached copy, not writable access to the trained basis. The underlying `matrix.EigenSym` is a fit-time operation and has no mid-rotation cancellation hook.

---

## 17. Testing tips

### Q17.1. What should a good package test suite contain?

At minimum:

```text
validation tests:
  nil input, bad options, invalid shape, invalid numeric values

medium tests:
  real non-trivial graph/matrix/math scenarios

special tests:
  zero-shape, loops, mixed edges, zero-weight edges, cancellation,
  equal-cost ties, partial-result behavior

determinism tests:
  stable order, stable witnesses, stable component sorting

error tests:
  errors.Is checks for every expected sentinel
```

### Q17.2. Should tests use `testify`?

No. Use the standard library and package-specific helpers. This keeps failure messages precise and avoids dependency weight.

### Q17.3. Should matrix tests use reflection or `reflect.DeepEqual`?

No. The `matrix` codebase follows a zero-reflection policy, including external `matrix_test` tests and fixtures. Use explicit shape comparisons, `At`-based element checks, `math.Float64bits` for required bitwise symmetry/determinism, and magnitude-aware numerical oracles. Concrete `*Dense` fast paths use type assertions, not runtime reflection.

Other packages may have their own testing policy, but reflection is never required to prove matrix contracts. Explicit failure messages are more useful than a generic structural-difference dump.

### Q17.4. How should numeric tests compare floats?

Use tolerances for computations that can accumulate floating-point error.

```go
if math.Abs(got-want) > tol {
    t.Fatalf("got %.12g, want %.12g", got, want)
}
```

For matrix-wide comparison, use the package’s tolerant comparison surface where appropriate.

For `stats`, compare the mean against an independent exact binary-input rational oracle; compare compensated `M2`/variance against a separate high-precision central-moment oracle using an explicitly justified magnitude-aware error budget. Do not use the production kernel as its own oracle.

For `pca`, verify the original covariance with a high-precision oracle on small matrices; check `C*v=lambda*v`, `VᵀV≈I`, heldout invariance under batch splitting, and `||Y-YVVᵀ||_F²≈(n-1)*sum(discarded eigenvalues)`. Compare projectors over complete repeated-eigenvalue subspaces, not individual eigenvectors. Nonconvergence and material PSD violations must remain classified errors.

For `series`, compare integer timestamps, gap counts, index ranges, and Valid masks **exactly**. For return-value or log-transformation oracles, base high-precision reference calculations on the original binary64 input values, not rounded printed decimals; use justified magnitude-aware error budgets rather than making epsilon a validity policy.

### Q17.5. How should tests handle map output?

Do not print or compare maps through unstable iteration. Extract keys, sort them, then compare or print stable output.

### Q17.6. How should cancellation be tested?

Use deterministic hooks or contexts. Avoid timeouts as the primary mechanism in examples or unit tests.

### Q17.7. What is a regression anchor?

A regression anchor is a test named after the bug or contract it protects.

Example:

```go
func TestAdjacencyMatrix_ToGraphPreservesZeroWeightEdgeByDefaultInInfEncoding(t *testing.T) {
// ...
}
```

A bug fix without a regression test is incomplete unless there is a clear reason no test can be written.

---

## 18. Benchmark tips

### Q18.1. What should be inside the benchmark loop?

Only the measured operation.

Good:

```go
g := buildGraph(b)
b.ReportAllocs()
b.ResetTimer()
for i := 0; i < b.N; i++ {
    _, err := runAlgorithm(g)
    if err != nil {
        b.Fatal(err)
    }
}
```

Bad:

```go
for i := 0; i < b.N; i++ {
    g := buildGraph(b) // measures setup, not just algorithm
    _, _ = runAlgorithm(g)
}
```

### Q18.2. What is a benchmark regime?

A regime is the real shape or pressure you are measuring.

Examples:

* sparse chain;
* dense graph;
* mixed-edge endpoint resolution;
* path tracking on/off;
* cutoff policy overhead;
* flow level graph rebuild pressure;
* matrix fast path vs generic fallback;
* stats scalar Update/Merge versus fitted train/inference matrix regimes;
* DTW full matrix vs rolling memory.

A benchmark name should reveal the regime.

### Q18.3. Why must setup errors be checked in benchmarks?

Because graph policy can reject edges. If setup silently fails, the benchmark may measure a smaller or different graph than intended.

This is one of the easiest ways to produce meaningless performance numbers.

### Q18.4. Should benchmarks use random data?

Only with fixed seed and documented shape. Randomness without reproducibility is not acceptable for regression-sensitive performance work.

---

## 19. Documentation tips

### Q19.1. Which file should explain what?

| File                | Responsibility                                         |
|:--------------------|:-------------------------------------------------------|
| `README.md`         | Public face, package map, strengths, key examples.     |
| `docs/TUTORIAL.md`  | First learning path and fundamental concepts.          |
| `{package}/doc.go`  | Package charter and implemented API contract.          |
| Exported GoDoc      | Per-symbol behavior, errors, determinism, complexity.  |
| `docs/{PACKAGE}.md` | Theory, diagrams, pseudocode, pitfalls, examples.      |
| `examples/*.go`     | Larger scenario programs.                              |
| `example_test.go`   | Executable package examples with deterministic output. |
| `CONTRIBUTING.md`   | Contribution workflow and quality gates.               |

### Q19.2. What is documentation drift?

Documentation drift happens when docs, examples, tests, and code describe different behavior.

Common drift patterns:

* docs mention removed APIs;
* examples use old constructors;
* README claims Go version or license incorrectly;
* package docs promise stronger guarantees than tests protect;
* tests lock behavior that contradicts mathematical correctness.

Fix drift immediately. Do not leave “almost correct” docs in place.

### Q19.3. Should docs include formulas?

Yes, when the formula explains real implemented semantics.

Good formulas:

* BFS hop distance;
* Dijkstra relaxation;
* Floyd-Warshall update;
* covariance/correlation;
* max-flow constraints;
* DTW recurrence;
* TSP objective.

Bad formulas are decorative and should be removed.

### Q19.4. Should docs include ASCII diagrams?

Yes, when they clarify topology, memory layout, residual flow, traversal order, or matrix encoding.

A good ASCII diagram should make the next paragraph easier to understand.

---

## 20. Quick recipes

### Recipe A. Hop blast radius

Use BFS with a depth limit. Group by `Depth`.

```text
Use when:
  incident response, service dependency waves, local exploration.
Avoid when:
  edge weights matter.
```

### Recipe B. Weighted failover route

Use Dijkstra with path tracking and, if needed, wall thresholds.

```text
Use when:
  links have latency/cost/risk.
Remember:
  known unreachable = +Inf, not missing target.
```

### Recipe C. Detect dependency cycles

Use DFS cycle detection.

```text
Use when:
  release pipeline, lock graph, task dependency validation.
Remember:
  witness cycles are not necessarily exhaustive cycle enumeration.
```

### Recipe D. Build a topology feature matrix

Use `matrix.NewAdjacencyMatrix`, `DegreeVector`, incidence, or metric closure.

```text
Use when:
  analytics, spectral methods, route matrices, graph features.
Remember:
  choose zero-weight policy deliberately.
```

### Recipe E. Model throughput

Use flow algorithms on a directed weighted capacity graph.

```text
Use when:
  capacity planning, assignment, bottleneck cuts.
Remember:
  residual graph is algorithm state, not the original graph.
```

### Recipe F. Compare time-warped signals

Use DTW with explicit memory mode and window policy.

```text
Use when:
  same pattern, different speed or local timing.
Remember:
  path recovery requires enough stored matrix state.
```

### Recipe G. Build reproducible tests

Use `builder` or manually fixed topology. Sort output before printing. Check errors with `errors.Is`. Avoid time-based cancellation.

### Recipe H. Validate a telemetry grid before causal transforms

Use `series.Validate` with an explicit positive `CadenceNS` and reject gaps unless the downstream consumer has a clear gap policy. Then use `Lag`, `Difference`, `SimpleReturns`, or `LogReturns` with their returned `Valid` masks. Use `WindowRanges` for complete index windows. Never substitute zeros for skipped timestamps or invalid warmup.

### Recipe I. Stable moments from partitioned telemetry

Stream finite observations into one `stats.Moments` per completed partition. Merge states only after workers stop, in a fixed order. Materialize the requested `ddof` through `Summary`; do not convert scaled `M2` before its denominator. If the source was `series`, retain only explicitly valid entries and apply your gap policy first.

### Recipe J. Train-only scaling before multivariate DTW

Fit `stats.FitStandardizer` on a stable training `matrix.Matrix` with explicit `ConstantPolicy`. Apply the **same immutable model** to both heldout matrices before `dtw.AlignMatrix`; never independently refit the candidate sequence. Preserve feature-column ordering and use `errors.Is` to classify failures.


### Recipe K. Calibrate a PCA residual detector without touching the final holdout

Fit `pca.FitWithComponents` on the healthy **training** matrix and retain the immutable `Model`. Reconstruct a separate healthy **calibration** matrix and derive a documented threshold from its per-row residuals in consistent units. Score untouched holdout/production rows with the same model; do not refit PCA or recompute feature means for the incoming batch. A threshold exceedance is an **investigation signal**, not a hardware shutdown or clinical decision. For mixed-unit channels, calibrate physical-channel residuals separately or define an independently fitted weighting rule; the PCA Model does not expose its internal scales.

See [`PCA.md`](PCA.md) and [`pca/example_test.go`](../pca/example_test.go) for the actual public API and deterministic examples.

---

## Final rule

When in doubt, ask:

```text
What is the mathematical object?
What is the package contract?
What is the deterministic order source?
What sentinel class should failure expose?
Who owns the result?
What does this value mean under this policy?
```

If you cannot answer those questions, the code, test, example, or documentation is not ready.

