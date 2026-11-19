<!--
  SPDX-License-Identifier: AGPL-3.0-only
  Copyright (C) 2025-2026 katalvlaran

  File: docs/FAQ_&_TIPS.md
  Package reference: core, matrix, series, stats, paa, sax, ols, ridge, pca, hmm, rsi, macd, mfi, obv, ad, montecarlo, gbm, jumpdiffusion, gjrgarch, arima, lstm and the wider lvlath package family
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
    - Fixed-window ACF estimator conventions, numeric safeguards, pair-count diagnostics, and noncausal provenance.
    - Reproducible block-bootstrap indices, unbiased random integers, partial row publication, and resource admission.
    - Wilder RSI warmup masks, deterministic batch/stream semantics, state rollback and checkpoint boundaries.
    - MACD SMA-seeded EMAs, signed features, three readiness masks, checkpoint identity and hot-path ownership.
    - MFI fixed-transition OHLCV windows, zero-volume retention, compensated rebasing and checkpoint authentication.
    - OBV first-bar zero, cumulative Neumaier correction, zero-volume continuity and checkpoint trust boundaries.
    - Chaikin A/D bounded multiplier, first-bar contribution, global batch error precedence and checkpoint replay integrity.
    - Monte Carlo per-path RNG replay, 2W scheduling bounds, stable moments, exact quantiles, byte admission and interruption prefixes.
    - GBM exact-in-distribution lognormal paths, explicit time units, safe wide-number increments, and complete-prefix semantics.
    - Merton jump diffusion with compensated drift, bounded exact Poisson kernels, stable RNG order, and complete-prefix safety.
    - GJR-GARCH causal state, asymmetric shock coefficients, stationary forecast, Gaussian QMLE and explicit nonconvergence.
    - Fixed-order conditional Gaussian ARIMA, Schur roots, burn-in, anchored integration and marginal innovation variance.
    - Frozen PyTorch IFGO LSTM weights, exact-zero batch masking, ModelID-bound streaming state and numeric overflow classification.

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
    13e. [HMM FAQ](#13e-hmm-faq---causal-regimes-smoothing-decoding-and-em)
    13f. [ACF FAQ](#13f-acf-faq---estimators-scaled-numerics-and-fixed-windows)
    13g. [Bootstrap FAQ](#13g-bootstrap-faq---dependent-data-blocks-replay-and-safe-partial-results)
    13h. [RNG FAQ](#13h-rng-faq-replay-ownership-and-numeric-safety)
    13i. [RSI FAQ](#13i-rsi-faq-wilder-streaming-readiness-and-checkpoints)
    13j. [MACD FAQ](#13j-macd-faq---sma-seeding-signed-values-batchstream-replay-and-feature-safety)
    13k. [MFI FAQ](#13k-mfi-faq-volume-weighted-transitions-compensated-windows-and-authenticated-replay)
    13l. [OBV FAQ](#13l-obv-faq-cumulative-signed-volume-and-replay-safety)
    13m. [AD FAQ](#13m-ad-faq-close-location-volume-precision-and-checkpoint-safety)
    13n. [Monte Carlo FAQ](#13n-monte-carlo-faq-replay-scheduling-numerics-and-partial-results)
    13o. [GBM FAQ](#13o-gbm-faq-exact-lognormal-steps-units-overflow-and-complete-paths)
    13p. [Jump Diffusion FAQ](#13p-jump-diffusion-faq-merton-compensation-poisson-counts-and-replay)
    13q. [GJR-GARCH FAQ](#13q-gjr-garch-faq-causal-variance-leverage-qmle-and-model-release)
    13r. [ARIMA FAQ](#13r-arima-faq-conditional-gaussian-fit-schur-constraints-and-anchored-forecasts)
    13s. [LSTM FAQ](#13s-lstm-faq-ifgo-inference-masking-and-streaming-replay)
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

| Question                                                                                                      | Use                                   | Reason                                                                                                |
|:--------------------------------------------------------------------------------------------------------------|:--------------------------------------|:------------------------------------------------------------------------------------------------------|
| “How many hops away is this vertex?”                                                                          | `bfs`                                 | BFS minimizes edge count.                                                                             |
| “Which vertices are in the same weak island?”                                                                 | `bfs.Components`                      | Weak membership ignores direction for component grouping.                                             |
| “What is a deterministic deep traversal?”                                                                     | `dfs`                                 | DFS gives post-order and forest-style structure.                                                      |
| “Does this graph contain a cycle?”                                                                            | `dfs.DetectCycles`                    | Returns deterministic witness cycles.                                                                 |
| “What order can I execute a DAG?”                                                                             | `dfs.TopologicalSort`                 | Topological order is a directed acyclic graph contract.                                               |
| “What is the cheapest route by weight?”                                                                       | `dijkstra`                            | Dijkstra minimizes non-negative total cost.                                                           |
| “What is the cheapest connected backbone?”                                                                    | `mst`                                 | MST solves minimum connected spanning structure.                                                      |
| “What is the maximum throughput?”                                                                             | `flow`                                | Flow solves capacity-constrained source-to-sink movement.                                             |
| “How do I turn graph topology into numeric features?”                                                         | `matrix`                              | Adjacency, incidence, degree vector, metric closure, algebra.                                         |
| “How do I model a grid map?”                                                                                  | `gridgraph`                           | Generates lattice topology instead of manual wiring.                                                  |
| “How do I build repeatable fixtures?”                                                                         | `builder`                             | Deterministic graph/data generation.                                                                  |
| “How do I compress a fixed-size scalar time-series window?”                                                   | `paa`                                 | Exact fractional PAA averages with rational boundaries and no dropped tail.                           |
| “How do I encode a fixed-length symbolic signature?”                                                          | `sax`                                 | Fractional PAA + right-bin quantization, compatible Euclidean MINDIST.                                |
| “How do I align two time series?”                                                                             | `dtw`                                 | Dynamic Time Warping handles local timing drift.                                                      |
| “How do I compute robust moments or merge workers?”                                                           | `stats`                               | Welford/Chan moments, exact mean sum, scaled M2.                                                      |
| “How do I normalize heldout features without refit?”                                                          | `stats`                               | Immutable fitted model over matrix columns.                                                           |
| “How do I estimate linear factor coefficients and reuse them?”                                                | `ols`                                 | Full-rank QR, original-unit beta/intercept, immutable prediction.                                     |
| “How do I fit correlated or wide linear factors with explicit L2 shrinkage?”                                  | `ridge`                               | Augmented QR, positive Lambda, unpenalized intercept and frozen physical predictions.                 |
| “What serial dependence or cycle repeats at a fixed sample-index lag?”                                        | `acf`                                 | One full-window mean, explicit denominator, exact pair counts.                                        |
| “How strong are smoothed up/down changes in an ordered positive price feed?”                                  | `rsi.ComputePeriod` / `Stream.Push`   | Wilder oscillator with a validity mask or zero-allocation, single-owner recurrence.                   |
| “How do fast and slow price trends diverge and how does that separation change?”                              | `macd.ComputeDefault` / `NewDefault`  | SMA-seeded EMA, three signed aligned series, independent readiness masks and replayable Stream.       |
| “How much rising versus falling typical-price flow occurred over the most recent candles?”                    | `mfi.ComputeDefault` / `NewDefault`   | Volume-weighted last-P-transition MFI, warmup mask, compensated rolling ring and detached checkpoint. |
| “What signed volume has accumulated since a chosen session origin?”                                           | `obv.ComputeCandles` / `New`          | Closing-price direction, seed zero, Neumaier compensation, transactional replayable stream.           |
| “How does close location inside each candle weight cumulative volume?”                                        | `ad.ComputeCandles` / `New`           | Chaikin multiplier times current base volume, first-bar inclusion, Neumaier and snapshot replay.      |
| “How do I build deterministic Monte Carlo, GBM or sensor-noise paths?”                                        | `rng.Factory.Stream`                  | Pinned PCG-DXSM, logical path IDs and cached normals; not a cryptographic PRNG.                       |
| “How do I execute many independent stochastic paths with identical full-run statistics across worker counts?” | `montecarlo.Run` / `EstimateParallel` | Per-path RNG, ordered `stats.Moments`, bounded 2W dispatch and strict error/prefix law.               |
| “How do I sample positive GBM prices at irregular explicit times?”                                            | `gbm.Simulate` / `SimulateSeeded`     | Exact-in-distribution lognormal endpoints, no implicit annualization, complete or valid prefix.       |
| “How do I simulate compensated Poisson jump risk over GBM?”                                                   | `jumpdiffusion.Simulate`              | Discrete Poisson counts and aggregated normal log jumps; experimental scenario model.                 |
| “How does a negative return shock change the next conditional variance?”                                      | `gjrgarch`                            | Immutable asymmetric variance, causal `Step`/`Filter`, first-shock forecast and constrained QMLE.     |
| “How do I estimate and forecast fixed-order nonseasonal ARIMA levels?”                                        | `arima.Fit` / `Model.Forecast`        | Conditional Gaussian estimand, strict Schur roots, anchored levels and marginal innovation variance.  |
| “How do I extract recurrent features from a frozen network over variable-length sessions?”                    | `lstm.Import` / `Model.InferSequence` | Immutable IFGO weights, length-masked `[B,T,D]`, zero padding, detached `Final`.                      |
| “How do I build dependent-data bootstrap index ensembles?”                                                    | `bootstrap.SampleIndices`             | Moving/Circular/Stationary blocks, replay ID and only complete rows.                                  |
| “How do I infer a hidden operating regime from time-ordered observations?”                                    | `hmm`                                 | Typed causal filtering, smoothing, Viterbi, and independent-sequence EM.                              |
| “How do I reduce multivariate dimensions without a response variable?”                                        | `pca`                                 | Validated covariance spectrum and a frozen loadings model for holdout inference.                      |
| “How do I visit all cities in a route?”                                                                       | `tsp`                                 | Tour construction/improvement/exact-small-instance routing.                                           |

### Q1.2. How do I avoid using the wrong algorithm?

Keep this mental table:

```text
minimize hops                          -> BFS
inspect depth structure                -> DFS
prove/locate cycle                     -> DFS cycle witness
order DAG tasks                        -> DFS topological sort
minimize weighted cost                 -> Dijkstra
connect all cheaply                    -> MST
maximize throughput                    -> Flow
compute all-pairs costs                -> Matrix metric closure
validate grid/causal features          -> Series
stream/merge stable moments            -> Stats
fit training-only scaling              -> Stats
fit full-rank linear factors           -> OLS (pivoted QR, no refit)
fit correlated/wide factors            -> Ridge (positive Lambda, unpenalized intercept)
compress latent features               -> PCA (train-only covariance basis; no target)
infer hidden temporal regime           -> HMM FilterStep (causal; no future observations)
detect fixed-window serial dependence  -> ACF ComputeDefault (not causal)
compute causal Wilder price strength   -> RSI ComputePeriod / Stream.Push
compute signed trend-separation factor -> MACD ComputeDefault / Stream.Push
compute volume-weighted OHLCV flow     -> MFI ComputeDefault / Stream.Push
compute cumulative signed volume       -> OBV ComputeCandles / Stream.Push (no warmup)
compute candle close-location flow     -> AD ComputeCandles / Stream.Push (first bar contributes)
drive independent stochastic trials    -> MONTECARLO Run / EstimateParallel (ordered vector moments)
simulate exact lognormal GBM paths     -> GBM Simulate / SimulateSeeded (explicit time units, positive prices)
simulate compensated Merton jumps      -> JUMPDIFFUSION Simulate / SimulateSeeded (no calibration)
measure causal asymmetric variance     -> GJRGARCH Filter/Step / ForecastVariance (no price simulation)
fit a fixed-order conditional mean     -> ARIMA Fit / Model.Forecast (not exact diffuse MLE)
extract frozen recurrent embeddings     -> LSTM Import / Model.InferSequence (not training)
replay a recurrent telemetry stream     -> LSTM Model.InferChunk with local lengths and Final state
fit conditional Gaussian QMLE          -> GJRGARCH Fit with explicit Initial (check convergence)
compute exact empirical path quantiles -> MONTECARLO RunWithQuantiles (budgeted raw storage)
draw dependent-data index replicates   -> BOOTSTRAP SampleIndices (no statistic or CI)
retrospective regime audit             -> HMM Smooth / Decode (future-conditioned)
encode symbolic SAX words              -> SAX
align time-warped signals              -> DTW
build a closed tour                    -> TSP
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

Yes. Correlation is dimensionless and is computed from **normalized centered moments**, not by dividing a materialized covariance matrix. Two perfectly anticorrelated channels of magnitude `1e200` have Pearson coefficient `-1`, even if the covariance exceeds finite binary64. Their standard deviations may still be representable. For a constant feature, the entire correlation row and column-including the diagonal-is exactly zero; for nonconstant features the diagonal is one.

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

Re-centering changes the coordinate origin and introduces holdout leakage. `Model.Transform(ctx,x)` uses **only** the fitted population parameters (if enabled) plus the independent frozen two-part covariance center in prepared coordinates. A single out-of-range observation must retain a nonzero score rather than being translated to zero because its one-row batch mean equals itself.

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

### Q13d.14. Why can an accurate covariance still produce incorrectly centered scores?

A single `float64` mean cannot always encode the training origin used by
`matrix.Covariance` to accumulate stable moments. For `base=2^450` and
`next=Nextafter(base,+Inf)`, the exact center is `base+g/2` where `g=next-base`.
Correct one-component scores are `[-g/2,+g/2]`; naive subtraction of a rounded
mean instead produces `[0,g]`. The fitted PCA model stores a first-row anchor
and high/low exact binary mean displacement **of the prepared training matrix**,
separately from the original-unit population-standardizer parameters.
`Transform`, `InverseTransform`, and the training residual use that frozen origin;
there is no refit or recentering on a future batch. This stabilizes the model
coordinate system, not the underlying covariance eigensolver's approximation.

### Q13d.15. Why does an overflowing raw infinity norm not imply PCA overflow?

The raw absolute row sum of covariance entries is an internal scale diagnostic,
not a published PCA eigenvalue. A rank-one covariance can have a finite leading
eigenvalue near `1.608e308` even though one absolute row sum exceeds
`MaxFloat64`. PCA validates the original covariance, scales every entry by
an exact binary exponent, computes the norm and Jacobi decomposition in bounded
coordinates, and restores each eigenvalue with `math.Ldexp`. It still returns
`ErrOverflow` for a genuinely unrepresentable required eigenvalue. This
correction does not weaken PSD acceptance, rank selection, eigenpair equations,
or the `ErrUnsupportedSolver` policy for `ThinSVD`.

---

### 13e. HMM FAQ - causal regimes, smoothing, decoding and EM

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

`NewGaussian` accepts any strictly positive finite diagonal variance. `FitGaussian` additionally requires **initial variance >= effective `VarianceFloor`**, which defaults to `1e-8` in *feature²* units. The constrained M-step uses `max(empiricalVariance,VarianceFloor)`-it does not add the floor as noise. If the seed is outside the constrained feasible domain, it is `ErrInvalidOption`, not a silently repaired start. Scaling training features changes the physical interpretation of the variance floor; freeze preprocessing units first.

#### Q13e.10. What happens to a dead hidden state with zero posterior occupancy?

No state is deleted or randomly reseeded. With `lambda=0`, a categorical emission row or transition row with no expected counts keeps its previous values. Gaussian means and variances for a zero-occupancy state remain unchanged. With `lambda>0`, zero-count categorical/transition simplexes are normalized from the pseudocount and become uniform. `Diagnostics.Occupancy` reports final posterior expected visits; its length remains K.

#### Q13e.11. Does an EM error always mean a nil model?

**No.** Invalid options, invalid input, impossible observations or context cancellation return nil. `ErrNotConverged` means the iteration budget was exhausted and returns the **last accepted immutable model** with `Converged=false`; `ErrNumerical` on objective decline returns the last accepted model for diagnosis. Neither constitutes a successful fit. A lower candidate is **never accepted**. An objective decline of at most two binary64 ULPs that also meets the caller's convergence threshold classifies the previously accepted model as numerically stationary; no declining score is appended. Larger declines still produce `ErrNumerical`. Callers must never silently accept a degraded model. Use `errors.Is` and inspect `Diagnostics.Objective`, `Iterations`, `Converged`, and `Occupancy`.

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

#### Q13e.17. Why can a normalized log posterior fail after a very large Gaussian observation?

When all states share an emission score near `-5e15`, calculating `log(p)+emission` in raw coordinates may erase the small `log(p)` distinctions. Subtracting a rounded full likelihood then fails to recover them: even equal states can sum to `0.7357` instead of `1`. The current Forward kernel removes the common emission **before** prediction is added, normalizes locally, and accumulates log evidence separately. `FilterStep` retains the original strict `1e-12` state normalization rule; do not increase that threshold to hide a damaged checkpoint.

#### Q13e.18. How can `x-mean` overflow when the Gaussian log density is finite?

For `x=+1.1e308`, `mean=-1.1e308`, `variance=MaxFloat64`, the physical difference is too large for float64, but the standardized residual is finite. The kernel uses direct subtraction whenever it is finite, otherwise `x/sqrt(variance)-mean/sqrt(variance)`; it then calculates the negative half-square in a safe order. A genuinely unrepresentable complete density still returns `ErrOverflow`.

#### Q13e.19. Why must Viterbi differentiate an impossible path from log-score overflow?

A categorical zero transition makes a path **structurally impossible**. Two finite Gaussian log densities can instead add to less than `-MaxFloat64` while the hidden trajectory remains mathematically supported. Viterbi tracks reachability independently of score, with exceptional wide arithmetic for overflowed candidates. If its winning supported path has an unrepresentable final cost, expect `ErrOverflow`, not `ErrImpossibleSequence`. A losing overflowed branch is not allowed to corrupt a finite winner.

#### Q13e.20. Why is integer-safe `rows*cols` still insufficient for a matrix buffer?

A Go slice also needs addressable **bytes**. For `float64` the checked bound is `rows*cols <= MaxInt/8`; for Viterbi backpointers it is `MaxInt/(IntSize/8)`. Custom virtual matrices with enormous declared shapes are rejected with `ErrOverflow` before their first `At()` read. This prevents the arithmetic `makeslice` panic, but it cannot guarantee available physical memory for every representable buffer.

#### Q13e.21. How does the Gaussian M-step avoid cancellation at `1e8±1`?

The E-step computes responsibilities under the same immutable old model in two deterministic passes. The first pass obtains a weighted new mean using compensated scaled differences; the second evaluates squared deviations **around that new mean**, normalized before summing. Therefore `[1e8-1,1e8+1]` has fitted population variance `1`, and `[-1e154,0,1e154]` can produce the finite population variance about `6.67e307`. A changed scalar feature scale requires a consistent external preprocessing policy and a corresponding squared-unit `VarianceFloor`.

#### Q13e.22. Does `Pseudocount=1e308` necessarily make categorical EM invalid?

No. A 2-symbol row with finite expected counts `[1,1]` and a symmetric Dirichlet pseudocount `1e308` still has exact probabilities `[0.5,0.5]`. The package scales counts and lambda before summation rather than computing the overflowing raw denominator `total+2*lambda`. An *actually* unrepresentable penalized objective is a separate numerical failure; it is not grounds for manufacturing a zero-probability row.

---

### 13f. ACF FAQ - estimators, scaled numerics and fixed windows

#### Q13f.1. What exactly does `acf` compute, and which public facade should I select?

`acf` computes the sample autocorrelation of one **fixed, uniformly indexed**
finite `[]float64` window for every lag `k` from zero through inclusive
`maxLag`. It does not interpret timestamps or infer missing samples. The
canonical API is `acf.Compute(values,maxLag,opts)`; use
`acf.ComputeDefault` for centered `CommonDenominator`,
`acf.ComputeLagAdjusted` for centered pair-count correction,
`acf.ComputeUncentered` for raw common-denominator second moments, and
`acf.ComputeUncenteredLagAdjusted` when both raw origin and pair adjustment
are explicitly required. All shortcuts delegate to `Compute`.

#### Q13f.2. Why are `CommonDenominator` and `LagAdjusted` not interchangeable?

With `c_i=x_i-mean` centered **once over the full window** (or `c_i=x_i` under
`NoCenter`), define `S_k=sum(c_i*c_(i-k), i=k..n-1)`. Common gives
`r_k=S_k/S_0`; adjusted gives `r_k=n/(n-k)*S_k/S_0`. The correction does not
merely improve precision; it changes the estimator. A downstream method that
needs Toeplitz PSD in exact arithmetic must select **common**, not infer the
convention from magnitude or silently substitute adjusted coefficients.

#### Q13f.3. Can a valid sample autocorrelation really be less than -1 or greater than 1?

**Yes for `LagAdjusted`.** Centered `[1,2,3]` has `S_0=2`, `S_2=-1` and one
lag-two pair, giving `CommonDenominator=-0.5` but `LagAdjusted=-1.5`.
Clamping to `-1` would publish an answer to a *different* mathematical
question. For an adjusted Toeplitz matrix with diagonal 1 and lag-two
entries `-1.5`, the 3-by-3 eigenvalues include `-0.5`; it is not PSD.
The common estimator has a Gram/sum-of-squares PSD proof in **exact
arithmetic**, not a universal binary64 PSD certificate.

#### Q13f.4. Why does the kernel fit only one mean, not one mean per lag?

Every `S_k` uses the **same** window-wide centered deviations. Pairwise
re-fitting changes the input coordinate system at each lag, breaking the
common Gram structure and producing another estimator. Late lags have fewer
products (`PairCounts[k]=n-k`), but the fitted mean and zero-lag energy remain
the **full-window** quantities. Setting `NoCenter=true` is the only public
switch that intentionally changes the origin.

#### Q13f.5. Is `NoCenter` an option to suppress errors for constant signals?

No. It selects **raw second moments around a meaningful physical zero**.
For `[2,2,2]`, centered ACF has zero energy and returns `ErrZeroVariance`;
uncentered common produces `[1,2/3,1/3]`, uncentered adjusted `[1,1,1]`.
An uncentered `[0,0,0]` still has zero second moment and fails. A
successful `NoCenter` result always has `Mean=0` **as mode metadata**, even
if the actual signal offset is nonzero. Raw and centered ACFs should not be
substituted for one another in a residual diagnostics pipeline.

#### Q13f.6. How do I interpret the lag-zero coefficient and `PairCounts`?

`Values[0]` is **exactly `1.0`** on every successful call, including
`maxLag=0`. `PairCounts[0]==n`; every later count is `n-k`. These are the
number of actually multiplied pairs, *not* confidence limits, IID degrees of
freedom or a guarantee of independent observations. The last permissible
lag uses one pair, often making adjusted estimates volatile. Select a
physically defensible maximum lag and inspect counts before drawing an
operational conclusion.

#### Q13f.7. Why does `acf.ComputeDefault([]float64{1},0)` fail while the uncentered call succeeds?

A singleton has zero **centered** variation by construction; the
normalized covariance ratio is undefined and produces `ErrZeroVariance`.
A nonzero singleton `[]float64{1}` with `NoCenter=true` has positive raw
second moment and returns `Values=[1]`, `PairCounts=[1]`, `Mean=0`. An
all-zero singleton fails in both modes. `maxLag=0` does not bypass input
finiteness or zero-energy validation.

#### Q13f.8. What is the exact error precedence, including simultaneously invalid input and options?

Options are checked before data: invalid `Denominator` or negative `maxLag`
return `ErrInvalidOption`, even with a nil slice. The subsequent order is
**nil** (`ErrNilInput`) → **non-nil empty** (`ErrEmptyInput`) →
`maxLag>=n` (`ErrInvalidOption`) → **any nonfinite sample**
(`ErrNonFinite`) → **zero moment** (`ErrZeroVariance`) →
**unrepresentable result** (`ErrOverflow`). All failures return a nil result;
no partial coefficients or missing-value repair are published. Always test
with `errors.Is(err, acf.ErrXxx)`, not string parsing. See the full table in
[`ACF.md`](ACF.md#6-validation-sentinel-errors-and-degenerate-inputs).

#### Q13f.9. How do inputs around `1e154` or `1e308` avoid false overflow?

The kernel does **not** first accumulate raw `sum(c_i*c_i)` or `sum(x_i)`.
It chooses an overflow-aware midpoint, scales deviations, computes one
compensated global mean shift, rescales, and forms compensated signed lag
dots. `math.FMA` reconstructs the reported mean with one final rounding,
including a tested subnormal boundary case. This prevents specific
avoidable intermediate failures; it is **not** a guarantee that every
possible binary64 coefficient is correctly rounded. A genuinely
unrepresentable public output is classified with `ErrOverflow`.

#### Q13f.10. Why does subtracting `Result.Mean` from my huge-offset samples not reproduce ACF's internal deviations?

`Result.Mean` is one rounded binary64 scalar. The direct kernel preserves a
scaled global mean shift before that final rounding. Computing
`values[i]-result.Mean` afresh can erase small representable differences or
change the last bits. The public result is the computed `Values`, not a
promise that its internal workspace can be reconstructed from metadata.
Use `stats` separately if your application needs an independent moment
summary with its own specified numeric guarantees.

#### Q13f.11. Is ACF translation/scale/reversal invariant? Can I append new samples without changing earlier lags?

In **exact arithmetic**, both conventions are invariant to sequence
reversal and nonzero common scaling; centered ACF is also translation
invariant. Uncentered ACF is **not** translation invariant because it uses
raw second moments. Finite `float64` operations only approximate these
identities when transformed observations remain representable. Appending
new data changes the entire fitted window, generally its center, its
`S_0`, and earlier coefficients. Unlike `series.Lag` or `hmm.FilterStep`,
ACF has **no causal prefix-invariance promise**.

#### Q13f.12. How do I use it after `series.Difference`, `series.LogReturns` or timestamp validation?

First establish a complete and equally spaced grid (for example with
`series.Validate` under a strict cadence policy). Then handle invalid
warmups using `TransformResult.Valid`; do not treat placeholder zeros as
observed values. If gaps exist, reject or segment into intact fixed
windows-blindly filtering invalid positions can create false adjacency.
Finally, compute ACF over the selected window. The resulting coefficients
are retrospective diagnostics and may be useful for **training-window
model selection**, not as leakage-free historical online features.

#### Q13f.13. May I mutate or share `Result.Values` and `PairCounts` across goroutines?

The slices are detached from input and kernel scratch and belong to the
caller. `Result.Clone()` creates independent copies of both slices.
`(*Result).IsNil()` safely checks the receiver pointer, including typed nil
behind `core.Nilable`; `&acf.Result{}` is **not nil**, but is not thereby a
valid completed result. Independent `Compute` calls are safe over stable
read-only input; concurrent caller mutation of any input or unsynchronized
reads/writes of a shared returned `Result` are unsupported.

#### Q13f.14. Is ACF a substitute for `stats`, `dtw`, HMM filtering, PACF, or spectral density?

No. `stats` computes moments and reusable training scalers; `series` checks
grids and produces causal masks; `dtw` aligns *two* sequences under time
warping; `acf` examines one fixed sequence at integer lags. HMM
`FilterStep` produces a genuinely causal posterior using observations
through the current time only. PACF, Ljung–Box tests, confidence bands,
FFT/spectral density and inference about stationarity are **not present**
in `acf`; they require distinct assumptions, algorithms, validation and
error contracts. No unseen PACF method should be advertised in docs.

#### Q13f.15. How expensive are late lags, and how do I benchmark the real operation?

With `n` samples and inclusive maximum lag `m`, the direct algorithm costs
`O(n*(m+1))` time and `O(n+m)` additional memory. It computes
`n+sum(n-k,k=1..m)` dot-product terms. There is **no FFT fallback**.
A request for `m=n-1` at `n=100000` implies billions of pair products;
constrain the horizon to the domain question and measure `ns/op`, `B/op`,
and `allocs/op` using the shipped benchmarks on the actual Go toolchain.
Do not publish a universal performance speedup from one host or compare
benchmarks measured on different CPUs/toolchains.

#### Q13f.16. What is a realistic, testable end-to-end ACF use case?

Compare six hours of one-minute retry-count telemetry from an incident window
and a fixed healthy reference. Both samples must share an explicit cadence,
window length and preprocessing policy. `acf.ComputeDefault(window,45)`
returns lag coefficients and actual pair counts; compare a suspected
15-minute recurrence, then *investigate* rather than claiming a diagnosed
root cause. Run [`examples/acf/main.go`](../examples/acf/main.go) with
`go run ./examples/acf`. The six independent executable examples in
[`acf/example_test.go`](../acf/example_test.go) also cover refrigerated
loads, volatility amplitudes, pitch, vibration impacts, calibration, and
standby current. All fixtures are synthetic and the review thresholds are
illustrative, not p-values.

---

## 13g. Bootstrap FAQ - dependent-data blocks, replay, and safe partial results

### Q13g.1. What does `bootstrap` return, and why only indices?

`bootstrap.SampleIndices(ctx, rows, plan)` returns `*bootstrap.Result` with
`Indices [][]int`, `Completed int` and `RNG AlgorithmID`. Every index
points to one observation position in the **same original, caller-owned
window**. The package does not receive a data slice, matrix or estimator.
This prevents hidden copying of market data, value normalization or
model fitting during sampling. Clients gather their chosen aligned rows
and compute statistics in a separate stage.

### Q13g.2. Which block mode should I choose?

- `MovingBlocks`: one uniform start in `0..rows-B` for each fixed block,
  then strictly consecutive indices **without wrap**. Choose when the
  physical end of the original history must not connect to its beginning.
- `CircularBlocks`: one uniform start in `0..rows-1` for each fixed block;
  within the block, `rows-1` continues to 0. Choose when a cyclic-boundary
  approximation is defensible for your data.
- `StationaryBlocks`: initial uniform index, then for each next index a
  new uniform start with probability `1/B`, or cyclic continuation with
  probability `1-1/B`. `B` is the **mean** geometric length, not a maximum.

All modes validate `1<=B<=rows` and use the same 256-MiB result budget.
Changing `Kind` changes the probability law; there is no implicit fallback.

### Q13g.3. Does `BlockLength=1` always mean IID sampling?

Yes. The three modes sample one uniformly chosen original index for each
output position when `B=1`. Given the same RNG factory, `Seed`, and logical
replicate ID, the emitted rows are identical because the stationary kernel
skips its otherwise redundant Bernoulli trial. For `B>1`, do not replace
blocks with IID draws: the time-dependence objective would be lost.

### Q13g.4. Can the output row be longer than the available history?

Yes. `Length` controls the requested number of **output indices**, while
`rows` controls the allowable source index domain. Multiple independently
selected blocks can repeat original observations. Every returned position
still satisfies `0<=index<rows`. Huge lengths must pass checked output-size
and budget validation before allocation.

### Q13g.5. Why is raw `uint64 % bound` forbidden?

For a uniform 8-bit source with 256 possible words and bound=3, naive
modulo counts the residues 86/85/85 times. Rejection discards exactly
`256 % 3 = 1` word, yielding equal 85/85/85 accepted counts.
For 64-bit words the adapter computes `threshold=(-b)%b` in unsigned
arithmetic, rejects `word<threshold`, and only then evaluates `word%b`.
The accepted count is an exact multiple of `b`; no floating-point
probability is involved. A `word%b` *after* rejection is correct.

### Q13g.6. Can a `uint64` accepted remainder overflow native `int`?

Under the validated public contract, no: `1<=bound<=MaxInt`, so
`0<=word%uint64(bound)<bound<=MaxInt`. The adapter nevertheless performs an
**explicit native-range guard** before narrowing. This protects the
conversion boundary and makes its safety condition auditable independently
of the unsigned-mapping proof. `ErrOverflow` normally classifies the
separate output-byte arithmetic before sampling.

### Q13g.7. Is `Seed=0` allowed? How do I replay a particular ensemble?

Zero is an ordinary reproducible seed. Each replicate `r` obtains the
separate logical stream `Factory.Stream(Seed,uint64(r))`, not a stream
assigned by goroutine or execution order. Persist `rows`, the entire
`Plan`, `Result.RNG`, the original dataset revision and split provenance.
A replay requires the same RNG factory version and bootstrap mapping
revision; a seed alone is not enough.

### Q13g.8. If I increase `Replicates`, do earlier rows change?

No. With the same `rows`, `Length`, `BlockLength`, `Kind`, `Seed`, and accepted
RNG revision, increasing `Replicates` from 2 to 5 must preserve the **exact
first two rows**. This is stronger than merely preserving distributional
properties: it is byte-for-byte integer equality. Separate invocations
can run concurrently because each facade owns its result and streams.

### Q13g.9. Can I split a large ensemble into independent calls?

Not with the current API while claiming exactly the same path IDs.
Every call begins at replicate `r=0`. Two calls with identical `Seed`
repeat the same streams; changing the seed creates a *different* ensemble,
not logical replicates `r=N..2N-1` from the original Plan. A true chunking
contract would require an explicit replicate offset; none exists here.

### Q13g.10. Why does `SampleIndices` sometimes return both Result and error?

Only **completed** replicate rows are published. If context cancellation
interrupts replicate `r`, its partially written row is discarded; the result
retains exactly `r` complete rows and `Completed==len(Indices)`, together
with `context.Canceled` or `context.DeadlineExceeded`. Cancellation after
all rows have committed can still return the full Result **and an error**
if observed at the final context gate. Never report a non-nil Result as
full success without checking `err`.

### Q13g.11. What if cancellation occurs before the first result row?

Two distinct cases are intentional. An already-canceled context at API
admission returns `(nil,ctx.Err())`, as does any invalid plan (with its
corresponding error). Once computation has begun, cancellation before a
complete first row returns a **non-nil `Result`** with `Completed=0` and
`Indices` a **non-nil empty slice**, plus the native context error.
Tests trigger these transitions by deterministic context polling, never
by fragile sleeps or timing assumptions.

### Q13g.12. What is checked by the 256-MiB resource ceiling?

`MaxResultBytes` is the logical sum of `Replicates * Length * sizeof(int)`
and `Replicates * sizeof([]int)` (row headers on the native architecture).
Integer arithmetic is checked for overflow **before** the budget comparison.
`ErrOverflow` means an unrepresentable size; `ErrBudget` means a
representable size exceeds 256 MiB. The ceiling does not include all Go
heap metadata or guarantee that an admitted allocation survives system OOM.
No panic/recover-based OOM recovery is provided.

### Q13g.13. Why are some block transitions discontinuous?

Every new Moving/Circular block starts independently. A large jump at a
boundary is legal; within one fixed block, Moving is strictly consecutive,
while Circular advances through the ring. A stationary restart may appear
to be a smooth continuation, so detecting boundaries by examining indices
alone is unsound. Tests must use the known fixed block boundaries or
independent transition-law oracles, not visual guesses.

### Q13g.14. Does `StationaryBlocks` make my original data stationary?

No. The **resampling index chain** has a uniform invariant distribution
when started uniformly, and its transition kernel is time-homogeneous.
The original observations may still contain trends, structural breaks,
calendar effects and long memory. Index-process stationarity neither
proves those assumptions away nor guarantees a valid confidence interval
for an arbitrary source process or estimator.

### Q13g.15. Can I use the result directly with OLS or Ridge?

Use each returned index to gather the **same source row** from the feature
matrix X and response y. Then fit a new model per replicate outside
`bootstrap` if a pairs-bootstrap analysis is appropriate. Independent
resampling of X and y destroys their joint relationship. Block bootstrap
is not Newey-West HAC covariance; residual bootstrap and pairs bootstrap
are also distinct inferential procedures requiring explicit assumptions.

### Q13g.16. Can I bootstrap a `series` with timestamp gaps?

Only after **explicitly** deciding what adjacency means. `series.Validate`
reports missing time-grid slots but does not interpolate them. The bootstrap
kernel sees contiguous *supplied indices*, not elapsed time. If adjacent
indices are separated by missing timestamps, bootstrap blocks may not be
contiguous in clock time. Reject, segment or repair the grid in application
code under a justified policy before resampling.

### Q13g.17. Can I obtain Ljung-Box null p-values by simply bootstrapping ACF?

No. Block resampling an autocorrelated window tends to preserve some of its
serial dependence; it does not enforce the null hypothesis of zero
autocorrelation. `acf.ComputeDefault` evaluates fixed-window coefficients
and is not a hypothesis-test engine. An empirical null generator must be
separately specified and validated for that hypothesis and data model.

### Q13g.18. What should I do after a sampling error?

Use `errors.Is(err, bootstrap.ErrInvalidOption)`, `ErrOverflow`, `ErrBudget`,
`context.Canceled` or `context.DeadlineExceeded`, as appropriate. An invalid
request is always a nil result; cancellation after computation starts can
carry complete prefix rows. Decide explicitly whether your application can
use a partial ensemble. Do **not** silently retry with shorter blocks,
truncate input data or change the RNG seed, since each alters the experiment.

### Q13g.19. What are the useful executables and documentation sources?

`bootstrap/example_test.go` provides five real-world synthetic pipelines:
portfolio 99% VaR / fractional-tail ES, paired dependent regression, turbine
vibration threshold calibration, deterministic regulatory replay and budget
admission. `go run ./examples/bootstrap` performs a standalone portfolio
review with source-risk estimates and replay metadata.
For the exact laws and all limitations, read [`BOOTSTRAP.md`](BOOTSTRAP.md)
and [`../bootstrap/doc.go`](../bootstrap/doc.go). These examples do not claim
to be live financial or certified industrial measurements.

---

## 13h. RNG FAQ: replay, ownership and numeric safety

### Q13h.1. Why does lvlath need a dedicated `rng` package?

A simulation using an unpinned standard-library or process-global PRNG may
change when a toolchain changes, or when a different number of worker goroutines
consumes values in a different order. `rng` fixes the integer recurrence,
SHA-derived initial state and transformation/consumption rules behind an immutable
factory. The goal is reproducible **numerical experiments**, not random secrets.

### Q13h.2. What do `seed` and `pathID` mean, and can I use zero?

Both are unrestricted `uint64`. `seed=0` is a fully valid, reproducible seed.
`pathID` is the logical trajectory, bootstrap replication, sensor session or
Monte Carlo sample number. It must **not** be a goroutine ID, physical worker
slot or CPU number; those are scheduler details, not experiment identity.

```go
factory := rng.NewFactory()
first := factory.Stream(0, 42)
replay := factory.Stream(0, 42)
fmt.Println(first.Uint64() == replay.Uint64()) // true
```

### Q13h.3. Is `Factory` thread-safe? Is `RNG` thread-safe?

`Factory` is stateless and safe for simultaneous calls, including a nil or
zero-value receiver. Each returned stream is an independent **mutable**
object; concurrent calls on the same stream are unsupported. Use one stream
per logical trial and write each trial's result to a disjoint output index.

### Q13h.4. What does `Version()` actually protect?

It returns `pcg-dxsm128-sha256le-u52-boxmuller-v1` and identifies the
pinned raw state recurrence, seed encoding, 52-bit uniform conversion and
cosine-first cached Box-Muller protocol. Persist this alongside the seed,
logical path mapping, input dataset revision, model parameters and reduction
order. A bare seed is insufficient for a long-term reproducibility record.

### Q13h.5. Does SHA-256 guarantee independent or disjoint streams?

No. SHA-256 derives reproducible starting states from a fixed 38-byte
preimage, but independent/disjoint PCG subsequences are not mathematically
certified by hashing. Statistical acceptance and model assumptions belong to
the consuming experiment.

### Q13h.6. Why exactly 52 bits for `UniformOpen`?

The 52-bit midpoint conversion has the **exact** interval
`[2^-53,1-2^-53]` in binary64. It therefore never returns `0` or `1`.
Using the analogous 53-bit midpoint conversion can round the maximum code to
**exactly `1.0`**. `UniformOpenFromUint64(0)` and
`UniformOpenFromUint64(^uint64(0))` expose the two endpoint cases to black-box
tests without inventing test-only production hooks.

### Q13h.7. Can I call `Uint64()` between two `Normal()` calls?

Yes, and its effect is defined. On the first `Normal`, two uniforms are
consumed and the sine half is cached. An intervening `Uint64()` consumes one
raw word but **does not** invalidate the spare. The next `Normal` returns the
cached sine, consuming zero raw words. Thus `Normal -> Uint64 -> Normal`
consumes precisely **three** raw draws. `UniformOpen` interleaving has the
same cache-preservation rule.

### Q13h.8. Why does `FillNormal` not continue the previous batch?

Every `FillNormal(seed,pathID,dst)` creates a **new** stream, fills its caller-
owned buffer, and then discards that temporary state. For an odd length, the
unused second normal from the final pair disappears with the stream. To
continue across chunks, retain one `RNG` and call `Normal()` repeatedly.
`FillUint64` and `FillUniformOpen` follow the same fresh-stream-per-call rule.

### Q13h.9. What does `Factory.IsNil()` mean? Where is `rng.Result`?

`Factory.IsNil()` reports whether the pointer is nil. A nil factory is still
fully functional because Stream/Version do not dereference it. The private
stream supports optional `IsNil()` without adding it to the three-method
`RNG` interface, keeping custom stream implementations compatible. The
package produces mutable streams, **not** a successful/partial `Result`
artifact; inventing `Result` or Options would violate the narrow ABI.

### Q13h.10. Are the RNG outputs cryptographically secure?

No. PCG-DXSM is for fast deterministic numerical draws; SHA-256 for state
derivation does not make the resulting PCG bitstream secure. Use
`crypto/rand` directly for keys, salts, tokens and security-sensitive
randomness in an appropriate application layer.

### Q13h.11. Does an unchanged path guarantee a bit-identical final Monte Carlo price?

Not by itself. If workers add partial float64 totals in completion order,
non-associativity can change the last bits. Publish `payoffs[path]` into a
disjoint slot, join workers, then reduce by ascending path index. Also keep
model parameters, draw-call order and the platform's math implementation in
scope: normal transcendental last bits are not universally portable.

### Q13h.12. Can I sample arbitrary block starts using `Uint64()%n`?

Not if unbiased integer sampling is required. Modulo is biased when `n` does
not divide `2^64`. The RNG examples use a **32-element** population, where
masking the low five bits is exactly unbiased. For arbitrary lengths, use a
correctly tested rejection sampler; the existing `bootstrap` package already
owns its own index and replay contract. `rng` does not introduce a separate
bounded-integer API in this revision.

### Q13h.13. Does this package already change `bootstrap.SampleIndices`?

No consumer migration is implied by delivering `rng`. The existing bootstrap
contract has index laws, cancellation semantics and a published `Result.RNG`
replay identifier. Changing its stream provider requires a **separate**
source-level compatibility audit and golden-vector decision; silently changing
its historical replicated indices would be a breaking change.

### Q13h.14. How do I validate the package efficiently?

Use fixed goldens for SHA/state/raw words; independent `math/big.Int` unsigned
128-bit arithmetic; direct forced endpoint values; exact Box-Muller
consumption/cache tests; fixed indexed path-order permutations; the race
detector on simultaneous **factory** usage; and allocation tests on a single
stream's draw loop. A million-value fixed-seed moment smoke is useful but
cannot certify the distribution or cryptographic strength.

```bash
go test ./rng -count=1
go test ./rng -race -count=1
go test ./rng -cover -count=1
go test -run '^$' -bench . -benchmem ./rng
```

For full code, six executable operational examples and mathematical proofs,
see [`RNG.md`](RNG.md), [`rng/doc.go`](../rng/doc.go) and
[`rng/example_test.go`](../rng/example_test.go).

---

## 13i. RSI FAQ: Wilder streaming, readiness and checkpoints

### Q13i.1. When should I use `rsi` instead of `series`, `acf` or `stats`?

Use `rsi` for the **relative strength of smoothed positive versus negative
price changes**, computed causally in supplied price order. `series` validates
clock cadence or computes individual causal changes and returns; `acf` estimates
serial correlation over one *entire* supplied window; `stats` summarizes
moments or fits reusable feature scales. None of these is an interchangeable
RSI kernel. Unlike `acf`, future valid prices never revise previously
published RSI values, provided both complete evaluations succeed.

`rsi` has no trading recommendation, fixed 30/70 rule, automatic period
selection, timestamp interpretation or source freshness check.

### Q13i.2. Why does RSI(14) need 15 prices before its first valid output?

The first price defines the baseline. Each **subsequent** price contributes
one signed difference, and Wilder's initial means require exactly `P`
differences. Therefore the first valid batch index is `P`, or accepted stream
`Count=P+1`. The period is the number of changes, not the number of samples.

```text
P=2, close=[1,2,3,2]
Values=[0,0,100,50]
Valid =[F,F,T,T]
```

A nonempty short history returns successfully with all `Valid=false`. It is
not an `ErrEmptyInput`: only an allocated slice of length zero is empty.

### Q13i.3. Is RSI=0 always a price-collapse signal?

No. `Values[i]==0` with `Valid[i]==false` means **no computed RSI** during
warmup. `Values[i]==0` with `Valid[i]==true` can be a mathematically valid
loss-only result. A flat valid history has neither gains nor losses and returns
50. Inspect the mask before any chart, feature extraction or downstream
statistical calculation; no NaN warmup marker is used.

### Q13i.4. Which API should I call, and is `Latest` O(1) memory?

| Task                                                    | API                                                                          |
|---------------------------------------------------------|------------------------------------------------------------------------------|
| Whole period-configured history                         | `rsi.Compute(close, rsi.Options{Period:p})`                                  |
| Standard 14-change history                              | `rsi.ComputeDefault(close)`                                                  |
| Explicit period shorthand                               | `rsi.ComputePeriod(close, p)`                                                |
| Only the latest observation of a complete input history | `rsi.Latest(close, opts)`                                                    |
| Stateful processing of an ordered feed                  | `rsi.New(opts)` / `rsi.NewDefault()` / `rsi.NewPeriod(p)` followed by `Push` |

`Latest` delegates to the complete `Compute`, returning the final value and
validity flag. It therefore costs **O(n) time and O(n) temporary output
memory**. For persistent feeds, `Push` is O(1) time and state, with zero heap
allocations per call after stream construction.

### Q13i.5. What happens when `Period` is zero, negative or extremely large?

`Period=0` normalizes to 14; an explicitly positive `Period` is used exactly.
Negative values and any period whose required `Period+1` overflows Go `int`
return `ErrInvalidOption` before the batch examines prices. There is no
arbitrary maximum such as 14/28/1000. A large valid period may require an
impractically long warmup, but is not quietly reduced. Streams with default
and explicit 14 share the same normalized checkpoint `Config`.

### Q13i.6. Why does the package not use `100*g/(g+loss)` or clamp to [0,100]?

The sum `g+loss` can overflow even when the ratio is a finite, valid RSI.
Computing `gain/loss` first can also overflow for a tiny positive loss.
The implementation first handles exact zero combinations and then divides
the **smaller** average by the larger. Each nonzero ratio is at most one,
with denominator between one and two. This avoids avoidable intermediate
overflow without altering Wilder's mathematical objective. A genuinely
nonfinite, negative or invalid internal value is `ErrOverflow`, not a
manufactured boundary RSI. It is never repaired with a clamp or an
application-specific numerical epsilon.

### Q13i.7. Is the initial arithmetic mean exact in `float64`?

The seed is mathematically the mean of the first P gains and of the first P
losses. The implementation computes it incrementally with divisor `k` at seed
step `k`, avoiding a raw sum of potentially enormous observations. Binary64
rounding can still discard sub-ULP contributions. This is **not** the
exact-binary-input mean algorithm from `stats`, nor is a universal
correctly-rounded mean promised. Changing the seed accumulator would change
observable batch/stream bits and checkpoint continuity.

### Q13i.8. Are batch and stream results bitwise identical?

For the **same valid input sequence**, the same normalized period and the
same floating-point execution environment, yes: both invoke the same private
`stepWilder` transition in ascending order. A batch validates its **entire**
history before producing a result, whereas `Push` validates one price per
call. On a bad batch price, no partial Result is published; on a rejected
stream price, successful earlier steps remain committed. These error-publication
contracts differ intentionally and do not contradict successful-output parity.

Do not expect bitwise identity after changing close representation, running a
different algorithm version/toolchain, resetting only one side, or comparing
under different floating-point environments.

### Q13i.9. Does a rejected `Push` corrupt the next delta or trigger a reset?

No. `Push` returns an error **without updating** the previous price, Count,
gain mean or loss mean. The next valid price is compared with the **last
accepted** price. Invalid `NaN`/infinity is `ErrNonFinite`; finite `0`, `-0`
and negative prices are `ErrDomain`. A rejected price is not automatically a
missing observation. If the caller *also* knows that a feed sequence or clock
interval was lost, it must handle that discontinuity explicitly, normally by
calling `Reset` and rebuilding the seed.

### Q13i.10. Can I use RSI over an irregular or timestamped feed?

`rsi` accepts only price numbers, in caller-defined sequence order. It has no
timestamp field, bar calendar or gap interpolator. Validate the sampling grid
with `series.Validate` where a known cadence matters; use source sequence IDs
or exchange offsets where relevant. A reported gap is metadata, **not** an
inserted close. If a discontinuity invalidates the intended consecutive-change
interpretation, start a fresh stream with `Reset()` or compute on independently
segmented, validated windows. Never feed manufactured zeros as gaps.

### Q13i.11. Which errors take precedence?

`Compute`: options → nil history → non-nil empty history → first invalid
price **by ascending index** (per-element nonfinite before domain) →
calculation. The `ComputeDefault`, `ComputePeriod` and `Latest` wrappers
preserve these classifications.

`Push`: nil receiver → uninitialized `Stream{}` → bad price → arithmetic,
including `Count==MaxUint64`. Therefore an invalid price beats counter overflow
in the same call. `Restore`: nil receiver → uninitialized stream → snapshot
version/configuration/structure; all malformed snapshot classes return
`ErrStateVersion`. Classify with `errors.Is`, never error message text.

### Q13i.12. What exactly does a Snapshot contain?

`Snapshot.Algorithm` is `"rsi/wilder/v1"`; `Config` is lowercase hex SHA-256 of
those UTF-8 bytes followed by **LE64(normalized period)**. `Count` is the
number of accepted prices, `FloatState` is **exactly**
`[previous,avgGain,avgLoss]`, and `IntState` has length zero. The caller owns
these arrays; mutating them after `Snapshot` does not change the stream, and
`Restore` does not retain them. `Reset` keeps the configured period/config
but clears Count and averages.

The checkpoint does **not** include timestamps, exchange sequence IDs,
bar-price definitions or the complete price history. Persist that provenance
at the application boundary.

### Q13i.13. Does `Config` make a checkpoint tamper-proof?

No. It identifies the **algorithm version and normalized period**, not the
mutable Count or mean values. It has no secret key and is not an HMAC.
Authenticate the *entire serialized snapshot* with an application-managed
MAC or signature before trusting externally transported state; also ensure
serialization preserves the exact binary64 bits. The
[`ExampleStream_Snapshot`](../rsi/example_test.go) demonstrates a deterministic
HMAC-protected transfer with an illustrative key, not production key management.

### Q13i.14. What does `Restore` reject, and can it prove historical provenance?

It rejects foreign algorithm IDs, mismatched configuration hashes, a
`FloatState` length other than three, nonempty `IntState`, nonfinite or
negative averages, a nonpositive previous price when Count>0, nonzero
state when Count=0, nonzero averages when Count=1, and both averages positive
when Count=2 (one price change cannot be simultaneously positive and
negative). Failure leaves the destination unchanged.

The v1 shape cannot prove the actual previous price history for arbitrary
later counts. `Restore` enforces **necessary structural/numeric invariants**;
it does not authenticate the source or certify that every accepted snapshot
was ever produced by valid historical calls.

### Q13i.15. What happens with nil, zero-value and exhausted streams?

`(*Stream)(nil).Push` and `.Restore` return `ErrNilInput`;
`(&Stream{}).Push` and `.Restore` return `ErrStateVersion`.
`Reset(nil)` is safe, and `Snapshot(nil)` and `Snapshot(&Stream{})` return
`Snapshot{}`, not a valid initialized checkpoint. A properly configured
stream with `Count=MaxUint64` rejects the **next valid** `Push` with
`ErrOverflow` and leaves its state unchanged. `(*Result)(nil).IsNil()`
returns true without reflection or a panic.

### Q13i.16. Can multiple goroutines share a Stream?

Not without external synchronization. `Stream` owns private mutable recurrence
state and no internal lock. Its `Push`, `Reset`, `Snapshot` and `Restore`
methods must not race with one another. Give each independent instrument/period
one owned stream and feed it in the authoritative order. Independent streams
can run concurrently. `Compute` may operate concurrently on read-only stable
input arrays, but concurrent mutation of those arrays by the caller is
unsupported. Successful result and snapshot slices are detached and
caller-owned.

### Q13i.17. How should I test and benchmark RSI?

Test independent manual and high-precision reference fixtures, period-one and
flat cases, short warmup, extreme finite prices, all invalid price categories,
per-prefix batch/stream bit identity, rejected-Push rollback, snapshot
boundaries `Count={0,1,P,P+1,P+2}`, `Count=MaxUint64`, aliasing, nilable
receivers and Reset after source gaps. **Parity alone cannot prove correct
mathematics.** Keep tests in external `package rsi_test` and exercise only
`rsi.` exports; do not add test-only production hooks or `reflect`.

Bench separately: batch `n=10^3` and `n=10^6`, `New`, and `Push` after
construction for representative periods and seed/steady/error branches.
`Push` should report **0 B/op and 0 allocs/op**; Snapshot may allocate a
small detached float slice and is deliberately outside the hot path.

```bash
go test ./rsi -count=1 -cover
go test -race ./rsi -count=1
go test ./rsi -run '^$' -bench . -benchmem
go vet ./rsi
```

For exact formulas, error precedence and snapshot ABI, see
[`RSI.md`](RSI.md), [`../rsi/doc.go`](../rsi/doc.go), and six executable
operational scenarios in [`../rsi/example_test.go`](../rsi/example_test.go).

---

## 13j. MACD FAQ - SMA seeding, signed values, batch/stream replay and feature safety

### Q13j.1. What does a MACD Result represent?

`macd.Result` holds three **separate, index-aligned** `Series` values:
`Line = fast price EMA - slow price EMA`, `Signal = EMA(valid Line)`, and
`Histogram = Line - Signal`. Each `Series` has `Values` and `Valid` slices
of the exact input length. It is a descriptive trend-separation feature,
**not** an order signal, volatility model, forecast or confidence interval.
The package does not fetch quotes, select candle boundaries or choose a risk
threshold for a trading engine.

### Q13j.2. Why is the seed an SMA rather than the first price?

The first `P` samples of **each EMA's own input sequence** establish an
ordinary arithmetic mean. Only after that does EMA smoothing use the factor
`2/(P+1)`. Fast and Slow both start with the *same first close*; the Signal
seed starts at the **first valid Line**, not the first close. Seeding from
`close[0]` for a period greater than one changes every early value and cannot
be silently substituted as an optimization. `Fast=1` is the legitimate
single-sample special case.

### Q13j.3. Is MACD smoothing the same as RSI's Wilder smoothing?

No. The conventional MACD recurrence uses **`alpha=2/(P+1)`** whereas
`rsi` uses Wilder's **`alpha=1/P`** for its gain/loss averages *after*
its seed. Their inputs also differ: MACD smooths price and signed Line;
RSI smooths the separate nonnegative gain/loss magnitudes of price changes.
Same period integers and the word "EMA" do not make their warmup or numerical
outputs interchangeable. See [MACD mathematics](MACD.md#3-mathematics-and-warmup-proof)
and [RSI mathematics](RSI.md).

### Q13j.4. When does each output become valid?

For normalized `(Fast, Slow, Signal)`, the first `Line.Valid=true` is at
**index `Slow-1`**, after `Slow` accepted closes. `Signal.Valid` and
`Histogram.Valid` first become true at **index `Slow+Signal-2`**, after
`Slow+Signal-1` accepted closes. Thus default 12/26/9 needs 26 closes
for Line and **34 closes** for all three outputs. A shorter nonempty batch
is successful, but its unready fields remain `(Values=0, Valid=false)`.

### Q13j.5. Does a zero histogram mean the signal is missing?

No. An invalid warmup cell is `Histogram.Values[i]==0` **and**
`Histogram.Valid[i]==false`. A fully computed histogram can also be exactly
zero. For `Signal=1`, Signal equals the valid Line and every valid Histogram
is exactly `+0.0`; those zeros are **real computed values**, not missing
values. In streaming mode Histogram has no dedicated validity flag:
`Point.SignalValid` is its validity flag. Check this boolean rather than
`Histogram==0` before using a point.

### Q13j.6. Which prices are legal, and may the indicator be negative?

All *input closes* must be strictly positive finite `float64` values,
including positive subnormals and very large finite numbers. Zero, negative
zero and negative finite closes are `ErrDomain`; NaN and both infinities are
`ErrNonFinite`. In contrast, `Line`, `Signal` and `Histogram` are signed.
A negative Signal is a normal mathematical state. Do not apply the price
validator to any intermediate or feed signed returns into `Compute` as if
they were closing prices.

### Q13j.7. Which validation error wins when a batch has multiple bad prices?

`Compute` checks options first, then nil input, empty input, **all nonfinite
prices**, and finally **all nonpositive finite prices**. As a result,
`[]float64{-1, 2, math.NaN()}` produces `ErrNonFinite`, even though the
negative price appears earlier. A `Push` validates only its one price after
checking nil/uninitialized receiver state. Both APIs classify errors via
`errors.Is`; neither returns usable partial output on an error.

### Q13j.8. Can finite prices still cause arithmetic overflow?

Yes. `ErrOverflow` is not the same as `ErrNonFinite`: an individually valid
close may contribute to an unrepresentable **derived** Line or Histogram
under an extreme state. The SMA warmup uses a bounded online mean rather
than summing `P` large prices; signed Signal updates avoid raw subtraction
of opposite-sign large operands. This avoids *avoidable* overflow but does
not promise exact real-number arithmetic. If any required derived value is
not finite, the entire `Push` is rejected without committing fast, slow,
signal or count.

### Q13j.9. Why can the same periods disagree with a charting service?

Check the source candle history, price adjustments, first accepted index,
missing-bar decisions and seed convention. Some libraries seed price EMAs
from different offsets or the first price rather than the same initial
SMA windows. Even mathematically equivalent recurrences can differ in the
last binary64 bits because arithmetic order matters. `Compute` and `Push`
in *this* package share one kernel, but their bitwise parity is guaranteed
only for the same input/order/options and floating-point execution
conditions, not for arbitrary external implementations.

### Q13j.10. Does failed Push partially advance the fast EMA?

No. A `Push` computes fast, slow, Signal and Histogram in a **private
candidate** and commits only after all checks succeed. Whether failure
comes from the supplied price, total counter, signal calculation or final
histogram, `Point{}` is returned and the existing stream is unchanged.
The caller may correct and resend a price, but must separately guard
source-sequence IDs to prevent a duplicate accepted close.

### Q13j.11. What exactly does Snapshot preserve?

For an initialized Stream, `Algorithm` is exactly `"macd/sma-ema/v1"`;
`Config` is lowercase SHA-256 of that ASCII version plus LE64-encoded
normalized Fast, Slow and Signal. `Count` counts **accepted closes**.
`FloatState` is the three current fast/slow/signal SMA-or-EMA means;
`IntState` is the three corresponding *saturated seed counts* in that order.
Both slices are detached copies. Snapshot works **during warmup** as well
as after full readiness; no historical prices are needed to continue.
An uninitialized or nil Stream produces a zero Snapshot.

### Q13j.12. Does the SHA-256 Config authenticate my checkpoint?

**No.** It fingerprints only algorithm/version/period configuration and
is not an HMAC, digital signature, checksum of the mutable means, or proof
of original prices. `Restore` checks version, config, exact slice lengths,
count/phase consistency, finite values and inactive-state canonical zeros.
A mathematically plausible but forged mean can still pass. Authenticate
the **whole checkpoint plus source offset** in the application and do not
assume `Restore` prevents replay or record replacement by a malicious peer.

### Q13j.13. How should I restart after a crash or feed gap?

For failover: create `New` with the **same normalized options**, securely
load the checkpoint, `Restore`, and submit the **first unseen** close after
the separately persisted source offset. Do not repeat the already committed
last close. A missing time slot or new instrument session is different: the
package has no clock or feed-ID validator. Decide upstream whether to
reject the feed, reconstruct explicitly with provenance, or call `Reset`
and wait through a new warmup. Do **not** inject zero or synthetic prices
into MACD as a hidden gap policy.

### Q13j.14. May several goroutines share one Stream?

No. `Stream` has no locks; `Push`, `Reset`, `Restore` and `Snapshot` may not
race with mutation. One logical instrument/cadence/parameter stream has
one owner at a time. Independent Streams and concurrently **read-only**
completed Results are fine while their input/output slices remain stable.
`Push` uses O(1) state and has no heap allocation in its hot path after
construction, but Snapshot allocates two small detached slices; parsing,
network IO, authentication and caller logging are outside the 0-allocation
promise.

### Q13j.15. How do I combine MACD with RSI, ACF or learned models?

Validate timestamp cadence with an explicit `series` policy, construct
**positive closes** without inserting placeholder zeros, then compute MACD
and keep *all* masks. RSI becomes ready on its own (different) schedule;
for a common feature row, require every selected feature to be ready at
that **same original index**. `acf` is retrospective over a *completed*
window and cannot act as an online causal feature without a separate
provenance decision. Fit `stats`, `ols`, `ridge`, `pca` or `hmm` only under
their own training and holdout rules; a causal MACD column does not make
an otherwise leaky pipeline safe.

### Q13j.16. Can I compare raw histograms across instruments or trade a crossover?

MACD values inherit **price units**. A 0.5 histogram on a $10 asset is not
numerically equivalent to 0.5 on a $1,000 asset. Any normalization, spread
filter, latency gate, hysteresis, cross-asset ranking or trading execution
belongs to the caller. A crossover detector must have **two consecutive
valid histograms** and an explicit zero policy; the first valid histogram
is not automatically a sign change. No MACD result certifies future returns.

### Q13j.17. Is `Result.IsNil()` a validity check?

No. It implements the `core.Nilable` pointer-presence convention:
`var result *macd.Result; result.IsNil()` is true, while
`(&macd.Result{}).IsNil()` is false. It does not inspect series lengths,
readiness masks or data provenance. On a batch error `Compute` returns
a **nil Result**, not a partially filled object. On successful short
history it returns an allocated Result with unready masks.

### Q13j.18. What must MACD tests and benchmarks prove independently?

Verify SMA seeds and warmup against exact-rational oracles, not solely
batch/stream self-comparison. Then test per-index bitwise parity, extreme
finite constants, signed Line/Signal, Signal=1, error precedence, rejected
Push rollback, per-phase Snapshot/Restore, alias isolation, nil receivers,
zero-count canonical state, and invalid hash/version/shape. External tests
belong to `package macd_test` and call public `macd.` symbols only. Benchmark
batch `n=10^3` and `10^6` separately from a steady-state `Push`, with input
setup outside the timer and `b.ReportAllocs()`. Do not claim a full-service
latency bound from one kernel's `ns/op`.

```bash
go test ./macd -count=1 -cover
go test -race ./macd -count=1
go test ./macd -run '^Example' -count=1 -v
go test ./macd -run '^$' -bench . -benchmem
go vet ./macd
golangci-lint run
```

The full mathematical contract and operational workflows are in
[`MACD.md`](MACD.md), [`../macd/doc.go`](../macd/doc.go), and
[`../macd/example_test.go`](../macd/example_test.go).

---

## 13k. MFI FAQ: Volume-weighted transitions, compensated windows and authenticated replay

### Q13k.1. What problem does MFI solve that RSI and MACD do not?

`mfi` measures the proportion of recent **volume-weighted typical-price flow**
attached to rising rather than falling typical prices. RSI smooths closing-price
gains/losses without volume; MACD compares price EMAs and publishes three signed
series. These indicators answer different numerical questions and are not substitutes.
MFI neither classifies buy/sell-initiated orders nor identifies institutional actors.

| Feature | Public surface                                   | Crucial difference                                                                  |
|---------|--------------------------------------------------|-------------------------------------------------------------------------------------|
| MFI     | `mfi.ComputeDefault`, `mfi.Stream.Push(Bar)`     | Positive/negative current `Typical*Volume` over the most recent `P` **transitions** |
| RSI     | `rsi.ComputeDefault`, `rsi.Stream.Push(close)`   | Wilder smoothing of closing-price gains and losses                                  |
| MACD    | `macd.ComputeDefault`, `macd.Stream.Push(close)` | SMA-seeded fast/slow price EMAs and a separate Signal EMA                           |
| ACF     | `acf.ComputeDefault`                             | Noncausal, whole-window lag correlation; not an online momentum oscillator          |

### Q13k.2. Why does MFI(14) need **15** bars?

The first valid bar establishes a previous typical price but creates no transition.
Each subsequent bar contributes exactly one positive, negative or zero pair. The
first valid output occurs at **index `P`**, after `P+1` accepted bars, and not at
`P-1`. Before that point `Values` are exactly zero and `Valid` is false.
A batch shorter than the warmup is still successful; it is not `ErrEmptyInput`.

### Q13k.3. Are unchanged-price and zero-volume candles dropped from the ring?

**Never.** Both create a `(0,0)` pair, consume one ring slot and expire the correct
old transition. A zero-volume candle with a changed typical price also updates
`previousTypical`, so the *next* bar is compared with the correct price.
Filtering these observations changes the clock of the rolling window.

For `P=2`, typical prices `[1,2,1]` with unit volumes yield MFI `200/3`
at index 2. Appending another price `1` with volume `100` creates an unchanged
transition, evicts the old positive flow `2` and produces MFI **0**. The reported
zero is a valid negative-only window, not a warmup placeholder.

### Q13k.4. Why can MFI return exactly 50, 0 or 100?

- **50:** both effective rolling sums are zero (flat prices, no volume or a mixture of zero contributions).
- **100:** positive flow is nonzero and negative flow is zero.
- **0:** negative flow is nonzero and positive flow is zero.

A valid 0 must not be confused with a warmup 0. Always check the `Valid` mask or
the `valid` return from `Push`. The package does not clip an invalid sum into
`[0,100]`, nor does it add an arbitrary epsilon to force a ratio.

### Q13k.5. Why does the package validate `Open` if it is not in Typical Price?

Because `Bar` is an OHLCV candle, not merely a HLC triple. The contract requires
finite positive `Open`, `High`, `Low` and `Close`, with
`Low <= Open,Close <= High`; `Volume` must be finite and nonnegative.
Rejecting impossible candle geometry prevents a malformed upstream record from
silently participating in a mathematically valid-looking feature.

### Q13k.6. How do the volume units affect an MFI value?

Use the same **base-asset quantity unit** for all bars of one instrument.
A positive uniform scaling of volume preserves the *real-arithmetic* ratio as
long as flows/sums remain representable; it need not preserve all float64 bits.
Do not mix base-asset quantity, quote-currency notional or contract counts in one
window without an explicit caller-side normalization. `mfi` does not perform
instrument metadata lookup, multi-exchange aggregation or currency conversion.

### Q13k.7. Why avoid `(High+Low+Close)/3`?

The intermediate three-price sum can overflow even when its mean is finite.
The kernel uses the High-anchored convex expression:

![Stable MFI typical price](https://latex.codecogs.com/svg.image?TP%3DH%2B%5Cfrac%7BL-H%7D%7B3%7D%2B%5Cfrac%7BC-H%7D%7B3%7D)

Both offsets are nonpositive for a valid candle, avoiding this avoidable
positive overflow. It remains binary64 arithmetic, not an exact-rational
average. If a required output is unrepresentable, `ErrOverflow` is returned.

### Q13k.8. Why not calculate `positiveSum/negativeSum` first?

That money-flow ratio becomes unbounded when the negative sum approaches zero.
Even `positiveSum + negativeSum` can overflow despite each signed sum being
finite. The implementation divides the **smaller** sum by the **larger** and
uses a denominator between 1 and 2; it handles all zero cases explicitly.
This does not extend the numeric domain to arbitrary-precision money flows:
every individual flow and both rolling sums must still be representable.

### Q13k.9. Why use Neumaier compensation and a full-ring rebase?

Repeated subtraction of expired contributions can make a naïve floating sum
drift away from a sum of nonnegative ring entries. Two compensated accumulators
reduce that loss. Every `P` successful transitions a deterministic `O(P)` scan
rebuilds the candidate sums from the oldest slot to the newest.
One extra scan is permitted on negative drift between scheduled boundaries;
a second such incident returns `ErrOverflow` rather than silently clamping
or performing unlimited rescans.

A single `Push` therefore has **worst-case `O(P)` time**, although the bounded
number of rebase scans makes its amortized time `O(1)`. A zero-allocation hot
loop is not a worst-case latency guarantee.

### Q13k.10. What is the maximum period and memory cost?

`Period=0` selects `DefaultPeriod=14`; explicit periods from 1 through
`MaxPeriod=1<<20` (1,048,576) are allowed. A larger or negative period
returns `ErrInvalidOption` **before** ring allocation. Two rings at the maximum
contain 2,097,152 float64 cells, or 16 MiB without structure overhead.
The limit cannot guarantee that the operating system has memory available.

### Q13k.11. In what order are invalid inputs rejected?

For `Compute`: **period -> nil columns -> empty Open -> mismatched lengths**,
all before any element read. Then bars are inspected in increasing index order.
Within **one** bar, `ErrNonFinite` has precedence over `ErrDomain`.
An `ErrDomain` in an earlier bar can therefore precede an `ErrNonFinite` in a
later bar; do **not** assume the package performs a whole-batch nonfinite scan.

For `Push`, nil/uninitialized receivers take precedence; after that the current
bar is validated before count saturation or arithmetic. On failure, `Push`
returns `(0,false,error)` and leaves all state unchanged. Classify with
`errors.Is`, never by inspecting error strings.

### Q13k.12. Is `Restore` safe against corrupted or forged checkpoints?

`Restore` checks algorithm/config ID, array shapes, Count-dependent head/used,
rebase-epoch flag, finite and nonnegative slot pairs, unused zeros and effective
sum consistency. The receiver is unchanged on rejection. The public wire fields
are `uint64`, but they must fit native `int` indexes before accessing a Go slice;
external coordinates cannot be wrapped or reduced modulo the period.

**This is a mathematical format validator, not authentication.** `Config` is
SHA-256 of the public algorithm ID concatenated with LE64(period); it does not
hash `FloatState`, `IntState` or `Count`. Authenticate the **entire** canonical
serialized checkpoint with a key or signature before `Restore`, and record the
exchange source offset independently to prevent duplicate-bar replay.

### Q13k.13. Will an accepted restored stream produce identical future values?

Under the same package build, floating-point environment and future input
sequence, yes: `Snapshot` retains physical ring order, stored sums,
compensations, previous typical price, head, occupancy, epoch flag and Count.
`Restore` checks a recomputed ring sum but installs the **saved** recurrence
state rather than replacing it with a newly rounded accumulator.
Do not round the exported floating fields, reorder arrays or discard the
compensation terms in checkpoint storage.

### Q13k.14. How does a real timestamp gap differ from a zero-volume bar?

A real zero-volume bar is an observed candle with a known time slot; it is a
valid transition. A missing bar is absent input, and `mfi` cannot detect it
from OHLCV alone. Validate cadence/sequence outside the package (for example
using `series.Validate` on a known time grid). On a genuine discontinuity, call
`stream.Reset()` and require a new `P+1`-bar warmup. A snapshot does not include
exchange timestamps or guarantee that the source replay position is correct.

### Q13k.15. Which API should a market-data service use?

| Source/request            | Prefer                                             | Reason                                                                |
|---------------------------|----------------------------------------------------|-----------------------------------------------------------------------|
| Existing five columns     | `mfi.Compute` or `mfi.ComputeDefault`              | No row-to-column transposition                                        |
| Existing decoder `[]Bar`  | `mfi.ComputeCandles`                               | Clear adapter; uses `O(n)` additional temporary columns               |
| One last historical value | `mfi.Latest`                                       | Convenience projection, but still a full `O(n+P)` batch computation   |
| High-rate ongoing feed    | `mfi.NewPeriod` + `Push`                           | Retains a bounded ring, no heap allocation per successful steady Push |
| Primary/standby replay    | `Snapshot` + authenticated persistence + `Restore` | Preserves compensation and ring topology                              |

Run one independent stream per instrument and coherent volume unit. The mutable
`Stream` does not support unsynchronized operations from multiple goroutines;
return slices and detached snapshots are caller-owned.

### Q13k.16. What does `IsNil()` mean for `mfi.Result`?

`var r *mfi.Result` has `r.IsNil()==true`, including when stored as an
`interface{ IsNil() bool }`. An allocated `&mfi.Result{}` has `IsNil()==false`
even if its slices are nil. A successful short Compute returns a **non-nil**
Result whose `Valid` entries may all be false. Never infer pointer nilness
from the length or content of the output arrays.

### Q13k.17. Which tests are required before production integration?

Independent analytic fixtures and exact-rational oracles must validate the
math, not just a batch/stream comparison through one kernel. Verify `P+1`
warmup, zero-volume and equal-price expiration, volume scaling within finite
range, overflow atomicity, invalid Snapshot state, every prefix split,
no allocation in steady Push and worst-case rebase cost. All package tests
are external (`package mfi_test`).

```bash
go test ./mfi -count=1 -cover
go test -race ./mfi -count=1
go test ./mfi -run '^Example' -count=1 -v
go test ./mfi -run '^$' -bench . -benchmem
go vet ./...
golangci-lint run
```

See [`MFI.md`](MFI.md), [`../mfi/doc.go`](../mfi/doc.go), and the five
execution-checked scenarios in [`../mfi/example_test.go`](../mfi/example_test.go).

---

## 13l. OBV FAQ: cumulative signed volume and replay safety

### Q13l.1. What does `obv` compute that `mfi`, `rsi` and `macd` do not?

`obv` accumulates **signed current-candle volume** according to successive
closing-price directions. Its result is an unbounded cumulative feature in
the supplied volume units. `mfi` instead computes a **bounded rolling
[0,100] typical-price money-flow ratio**, `rsi` measures smoothed changes
in price strength, and `macd` measures differences between SMA-seeded EMAs.
These results and their warmup policies are not interchangeable.

### Q13l.2. Why is the first `obv.Result.Values[0]` zero and already valid?

There is no earlier close against which to sign the initial candle's volume.
The first bar **only** seeds `previousClose`, and the first accepted output is
positive `0.0` with `Valid[0]=true`. For a singleton with even
`Volume=math.MaxFloat64`, `Compute` must return `[0]` without overflow.
Unlike MFI, there is no period and no unready prefix.

### Q13l.3. Can I remove unchanged-price or zero-volume candles?

No. Both are accepted observations, update `previousClose`, and increment
`Count`; their contribution is zero. Dropping an unchanged-close bar can
leave the displayed balance and next comparison unchanged, but it still
corrupts the accepted Count and feed-offset correspondence. Dropping a
zero-volume bar **whose close changed** can additionally reverse the sign
of a later contribution. A **missing** candle is a different state: the
upstream feed must detect discontinuity and choose an explicit reset/replay
policy. OBV itself has no timestamps.

### Q13l.4. Does OBV examine Open, High and Low if only Close signs volume?

Yes. `Bar` and `Bars` must satisfy complete finite-positive OHLC geometry:
`Low<=Open,Close<=High`, `High>=Low`, `Volume>=0`. Every input field is
validated to reject malformed market records. `Open>High` returns
`ErrDomain` even if Close and Volume are plausible.

### Q13l.5. Which API should I use for batch, row-oriented feed and streaming?

| Input/use                         | Public call                                         | Ownership                                    |
|-----------------------------------|-----------------------------------------------------|----------------------------------------------|
| Five aligned OHLCV columns        | `obv.Compute(obv.Bars{...})`                        | Read-only input; detached Values/Valid       |
| Decoded chronological candle rows | `obv.ComputeCandles([]obv.Bar{...})`                | No five-column transposition or retention    |
| Single live candle                | `stream := obv.New(); stream.Push(bar)`             | Mutable single-owner state                   |
| New known epoch                   | `stream.Reset()`                                    | Starts from a new first-bar zero             |
| Controlled handoff                | `state := stream.Snapshot(); backup.Restore(state)` | Detached checkpoint; external authentication |

`Stream{}` is as usable as `New()`. There are no `Options`, `Context` or hidden
period modes in the OBV package.

### Q13l.6. What error takes precedence for malformed columns and OHLCV?

First: nil any-column → `ErrNilInput`; non-nil empty Open →
`ErrEmptyInput`; unequal column lengths → `ErrShape`. Only after shape
validation are individual rows inspected in increasing index order. The
first failing row wins, with nonfinite input (`ErrNonFinite`) taking priority
over domain geometry (`ErrDomain`) *within that row*. Numeric Count or
accumulation failure is `ErrOverflow`. Always use `errors.Is`.

### Q13l.7. Why not use `sum += volume` or `sum -= volume` directly?

At `1e16` magnitude, adding an exact `1` can round away the increment.
When a later opposite-signed `1e16` arrives, ordinary accumulation reports
zero instead of `1`. OBV retains a Neumaier correction; the displayed value
is the finite `sum+compensation`. Both components are needed for correct
future cancellation. This improves accuracy, not arbitrary precision.

### Q13l.8. Can a valid finite pair of volumes still trigger `ErrOverflow`?

Yes. The cumulative result may exceed the representable binary64 range.
The kernel attempts cancellation-first rebalance for intermediate overflow
that might otherwise conceal a finite result; it never silently clips,
ignores a bar or invents infinity as a normal OBV. If the operation fails,
`Push` returns no valid observation and leaves **all** stream fields intact.
A late batch overflow returns `(nil,error)` rather than a usable prefix.

### Q13l.9. How do I prove `Push` rejected a candle without partial mutation?

Capture `before := stream.Snapshot()` and compare **all** fields with
`stream.Snapshot()` directly after the failed Push. Compare `FloatState`
using `math.Float64bits` because compensation or even a signed zero may
change while the displayed total appears identical. Then replay a corrected
valid bar and check that it compares against the last accepted close.

### Q13l.10. What exactly is inside the v1 Snapshot?

`Algorithm="obv/neumaier/v1"`, `Config=lowercase
SHA256(ASCII Algorithm)`, `Count`, `FloatState=[previousClose,sum,compensation]`
and `IntState=[]` of length zero. `Count=0` requires zero coordinates;
`Count=1` requires a positive previous close and zero contributions.
`Count=MaxUint64` can be restored, but the next Push rejects with
`ErrOverflow`. Nil `Snapshot()` returns `Snapshot{}`; nil `Restore` reports
`ErrNilInput` and nil `Reset()` is harmless.

### Q13l.11. Does Config protect against forged state or replay?

No. It is SHA-256 of a **fixed public algorithm string**, not a hash of
the mutable fields and not a keyed MAC. `Restore` protects against malformed
or incompatible **structure**, not an authenticated market history. Sign or
MAC the *entire serialized snapshot* together with the instrument, volume
unit, source feed identity and last committed offset, and enforce replay
protection outside `obv`.

### Q13l.12. Are returned Result slices and checkpoints safe to modify?

They are detached and caller-owned. `Restore` copies values into scalar
stream fields and retains no input slice. Concurrent calls mutating one
Stream are **unsupported**; use one owner or external synchronization.
`Result.IsNil()` checks the result pointer, not whether slices have length
zero. An allocated empty `Result{}` is not a successful computation.

### Q13l.13. Can I sum OBV across different instruments or claim “smart money”?

Not without a separately justified unit conversion and aggregation rule.
Some feeds report base assets, others contracts or quote notional. A growing
OBV records volume attached to upward close transitions; it cannot identify
buyers, institutions, executed order types, or guarantee a future breakout.
Any screening threshold or strategy belongs to the caller, not this package.

### Q13l.14. How should I verify this package before production?

Start with independent hand or rational oracles, not only batch/stream parity
through the shared kernel. Cover OBV01–OBV08: first-bar zero, cancellation
`(+1e16,+1,-1e16)`, equal close with huge volume, true overflow and state
rollback, volume scaling, malformed OHLC, and checkpoint continuation at
count zero/one and after cancellation. Test nonfinite fields, nil receivers,
source ownership and nonzero-delta allocation behavior. The tests live
strictly in `package obv_test`.

```bash
go test ./obv -count=1 -cover
go test ./obv -race -count=1
go test ./obv -run '^Example' -count=1 -v
go test ./obv -run '^$' -bench . -benchmem
go test ./...
go vet ./...
golangci-lint run
```

For complete mathematics, ABI, code examples and the trust model, see
[`OBV.md`](OBV.md), [`../obv/doc.go`](../obv/doc.go) and the five
[`../obv/example_test.go`](../obv/example_test.go) scenarios.

---

## 13m. AD FAQ: Close-Location Volume, Precision and Checkpoint Safety

### Q13m.1. Is `ad` Advance/Decline, an OBV synonym, or a trading signal?

No. `ad` is **Chaikin Accumulation/Distribution**, the cumulative sum of
current-candle *close-location multiplier* times current-candle **base
volume**. Advance/Decline is a market-breadth concept; automatic
differentiation is unrelated. `obv` signs a candle's volume by comparing its
close with the previous accepted close. `mfi` produces a bounded rolling
Money Flow Index from typical-price transitions. `ad` neither generates
thresholds nor identifies traders or future returns.

### Q13m.2. Why is the first A/D output nonzero and already valid?

The accumulator starts at zero **before** index zero, but the first candle
has enough information to compute its own position in its High/Low range.
For `Open=1.5, High=2, Low=1, Close=2, Volume=3`, the multiplier is `+1`
and `Values[0]=3`, `Valid[0]=true`. `obv` deliberately seeds its first close
without adding volume; transferring that rule to `ad` would be incorrect.
There is no period and no warmup for any successful A/D batch.

### Q13m.3. Why not compute `(2*Close-High-Low)/(High-Low)`?

It can overflow **before** division if Close approaches
`math.MaxFloat64`, even for a perfectly valid positive candle with a
multiplier between `-1` and `1`. The implemented calculation forms
`left=Close-Low`, `right=High-Close`, `spread=High-Low`, and divides
`(left-right)/spread`. Positive finite accepted prices ensure these
distances and spread are finite. There is no automatic multiplier clamp or
price-epsilon repair; changing the formula changes numerical behavior.

### Q13m.4. What if `High == Low` and Volume is enormous?

A valid zero-spread candle must have all four OHLC prices equal. The
multiplier is explicitly **zero**, so the contribution is **zero** even if
Volume is `math.MaxFloat64`. The bar is still accepted and increments
`Count`. No division occurs. A nonflat Open or Close outside the equal
boundaries remains `ErrDomain`; flatness is not an exemption from geometry.

### Q13m.5. Why validate Open when the formula only uses High, Low, Close?

Open is part of the candle's public integrity contract. A record with
`Open<Low` or `Open>High` is malformed and returns `ErrDomain`, even if
its multiplier could be computed. The same rule applies to `obv` and
`mfi` inputs. Prices must be finite and **strictly positive**; volumes
must be finite and **nonnegative**. The package never repairs vendor data.

### Q13m.6. Which shape and numeric error takes precedence?

For `Compute(ad.Bars{...})`:

```text
any nil column -> ErrNilInput
then non-nil empty Open -> ErrEmptyInput
then unequal column lengths -> ErrShape
then any nonfinite field anywhere -> ErrNonFinite
then any price/volume/geometry violation anywhere -> ErrDomain
then unrepresentable output allocation or arithmetic -> ErrOverflow
```

Both numeric categories are scanned across the **entire** history. Thus a
NaN at the final index takes precedence over an invalid finite Open at
index zero. This differs from the per-row precedence documented for `obv`.
`ComputeCandles([]Bar)` has no five-column shape check but has the same
global numeric-category ordering. `Stream.Push` validates just its current
bar after checking the receiver. Classify errors using `errors.Is`.

### Q13m.7. Which public facade fits my data layout?

| Workload                                  | API                                          | Key cost or guarantee                                |
|-------------------------------------------|----------------------------------------------|------------------------------------------------------|
| Five aligned price/volume columns         | `ad.Compute(ad.Bars{...})`                   | O(n) time/space, detached `Result`                   |
| Already-decoded row slice                 | `ad.ComputeCandles(candles)`                 | No five-column transposition; O(n) output            |
| Last value from whole history             | `ad.Latest(bars)` / `ad.LatestCandles(rows)` | Delegate to complete batch; O(n) time **and memory** |
| Each incoming completed candle            | `stream := ad.New(); stream.Push(bar)`       | Transactional O(1), zero steady-state heap allocs    |
| New explicitly chosen accumulation origin | `stream.Reset()`                             | Clears count, sum, compensation                      |
| Controlled service handoff                | `Snapshot()` / `Restore(state)`              | Detached and version-checked, not authenticated      |

No `Options`, `NewPeriod`, `context.Context`, epsilon, or hidden fast mode exists.
A zero-valued `ad.Stream{}` works exactly like `ad.New()`.

### Q13m.8. What does `Valid=true` tell me?

Only that the A/D observation was successfully computed. A valid A/D of
zero may be a midpoint, a flat candle, a zero-volume candle, or cancellation
of earlier contributions. All accepted bars have `Valid=true`, including
bar zero. It does **not** certify feed authenticity or a statistically
significant signal. `(*ad.Result).IsNil()` checks **pointer presence** for
`core.Nilable`; an allocated empty `Result{}` is not a valid Compute result.

### Q13m.9. Is the multiplier always in `[-1,1]`? Is A/D bounded too?

Validated candle geometry guarantees the multiplier is in `[-1,1]` without
clipping. Therefore a **single** contribution has magnitude at most the
bar's Volume. The **cumulative** balance is not limited to `[-1,1]`, to
`[0,100]`, or to any individual bar's Volume. Do not confuse the
multiplier with the A/D line or the bounded MFI oscillator.

### Q13m.10. Does a positive affine change of prices preserve the answer?

In ideal arithmetic, any positive scale plus offset that preserves the
candle's order preserves its relative close location. With binary64 inputs,
that is only reliable when the transformed High/Low/Close differences remain
representable. If rounding collapses a nonzero range, the library sees a
flat bar instead. Do not promise exact affine invariance under arbitrary
huge offsets or tiny spreads; test representable fixtures and rational
oracles when validating numerical changes.

### Q13m.11. Why keep both Neumaier components?

Adding `+1e16`, `+1`, `-1e16` with ordinary float64 accumulation can lose
the middle `1`. Neumaier stores a rounding correction; `sum+compensation`
returns `1` for that fixture. Checkpoints **must not** replace the pair
with the rounded displayed total, because later cancellation depends on
both coordinates. The implementation still has finite `float64` limits.

### Q13m.12. What does `ErrOverflow` mean, and is `Push` atomic?

`ErrOverflow` means the fixed recurrence cannot represent the next
compensated state as finite, or `Count` is exhausted; batch allocation
byte-product overflow also uses this sentinel. An intermediate direct
addition may overflow while an opposing stored correction makes a finite
rebase possible, so the kernel tries a **cancellation-first rebase** before
rejecting. On a failed `Push`, `value=0`, `valid=false`, and the stream's
sum, correction, and Count are all unchanged. On any batch error, `Result`
is nil with **no** published prefix. Runtime operating-system OOM is not a
recoverable ErrOverflow guarantee.

### Q13m.13. How can I check rollback without accessing private fields?

Capture `before := stream.Snapshot()`, deliberately attempt an invalid
candle or an unrepresentable contribution, and compare `before` to the
snapshot after failure. Compare `FloatState` with `math.Float64bits` and
check `Algorithm`, `Config`, `Count`, and both slice lengths. A check of
only the displayed total can miss corrupted correction state. The tests
remain in external `package ad_test`; no internal test hooks are required.

### Q13m.14. What is the exact v1 checkpoint schema?

```text
Algorithm   "ad/chaikin-neumaier/v1"
Config      lowercase SHA-256 of ASCII Algorithm
Count       number of accepted bars
FloatState  [sum, compensation] (exactly 2 finite float64 values)
IntState    [] (length zero)
```

`sum+compensation` must be finite. `Count=0` requires both coordinates
numerically zero and restores the canonical `Stream{}` state. A restored
`Count=math.MaxUint64` is valid but the next Push fails with
`ErrOverflow`. There is **no previousClose**, unlike the OBV snapshot.
Nil `Snapshot` returns `Snapshot{}`; nil `Reset` does nothing; nil `Push`
and `Restore` report `ErrNilInput`. A bad version/config/layout/numeric
state returns `ErrStateVersion` **before modifying** the receiver.

### Q13m.15. Can Config authenticate a checkpoint or prevent replay?

No. `Config` hashes a **public fixed algorithm string**. An attacker can
compute the same hash for a forged state, so it is only a compatibility
fingerprint. Authenticate the **entire serialized snapshot** together with
feed identity, asset, base-unit convention, cumulative epoch and last
committed source offset. Enforce exactly-once replay externally and resume
from the next unprocessed candle after Restore. The package does not read
or validate that offset or use clocks or network I/O.

### Q13m.16. What if a candle is missing, duplicated, or zero-volume?

`ad` does not see timestamps. A present zero-volume candle is valid,
produces zero contribution, and increases Count. A **missing** candle is
not reconstructed; it can change the cumulative history and provenance.
Use `series.Validate` or another explicit upstream integrity check to
establish cadence. On a known gap, choose reject/replay or `Reset` to begin
an entirely new origin. Comparing raw A/D levels across reset epochs or
assets with incompatible base-volume units is not meaningful by itself.

### Q13m.17. Are batch Results, snapshots and streams safe across goroutines?

Returned `Values`, `Valid` and snapshot slices are detached, caller-owned
memory; Restore copies two scalar coordinates and retains no input arrays.
A **single mutable Stream** has no locks and must have one owner or
external synchronization for Push, Snapshot, Restore and Reset. Separate
streams can run simultaneously on stable inputs. Mutating a caller-owned
batch column or row slice during Compute is unsupported and can race.

### Q13m.18. Does price/A/D divergence prove accumulation or a breakout?

No. A/D is a reproducible price-position-weighted **volume feature**, not a
divergence detector, order-book model, causal attribution, or significance
test. An application can compare A/D movement to price movement, but must
define comparable windows/origins, volume units, data quality, a historical
training/holdout split, and an independently validated decision rule. The
`ad` package intentionally has no threshold or signal API.

### Q13m.19. What must tests and benchmarks verify before integration?

Use independent high/low/midpoint/flat-candle oracles, near-MaxFloat64 and
large-offset cases, accurate cancellation and true overflow, exact global
error precedence, failed-Push rollback, snapshot corruption, detached
ownership, and checkpoint continuation at **every** prefix. Verify per-bar
bitwise parity among Compute, ComputeCandles and Stream.Push, but never use
parity as a substitute for independent mathematics. Benchmarks must drive
nonflat candles with varying signed volume and check `Push` allocs/op.

```sh
go test ./ad -count=1 -cover
go test ./ad -race -count=1
go test ./ad -run '^Example' -count=1 -v
go test ./ad -run '^$' -bench 'Benchmark(Compute|Push)' -benchmem
go test ./...
go vet ./...
golangci-lint run
```

See [`AD.md`](AD.md), [`../ad/doc.go`](../ad/doc.go) and the six
executable usage cases in [`../ad/example_test.go`](../ad/example_test.go).

---

## 13n. Monte Carlo FAQ: Replay, Scheduling, Numerics and Partial Results

### Q13n.1. When should I use `montecarlo` instead of `rng`, `bootstrap` or `stats`?

Use `rng` when you need a **single version-pinned mutable stream** and will write your own execution orchestration. Use `montecarlo` when you need to execute **many independent model paths** with bounded parallelism, canonical path-ordered summaries and optional exact quantiles. Use `bootstrap` when you want **block-resampled index rows from an existing dependent sequence**; it does not sample a financial/physical model or estimate your statistic. Use `stats` for scalar moments that your application updates directly, not for a versioned trial scheduler.

```text
rng.Factory.Stream(seed,pathID) -> one stream per logical path
           |
caller Sampler + montecarlo.Run -> ordered vector moments / exact quantiles

bootstrap.SampleIndices -> existing-data index replicates (different objective)
stats.Moments            -> caller-controlled scalar updates / merges (no driver)
```

See [`RNG.md`](RNG.md), [`BOOTSTRAP.md`](BOOTSTRAP.md), [`STATS.md`](STATS.md) and [`MONTECARLO.md`](MONTECARLO.md).

### Q13n.2. What is the canonical API, and what do the convenience wrappers change?

`montecarlo.Run(ctx,plan,sampler)` is canonical. `Estimate(ctx,paths,seed,sampler)` selects serial mean-only execution. `EstimateParallel(ctx,paths,seed,workers,sampler)` selects mean-only parallel execution. `RunWithQuantiles(ctx,plan,levels,sampler)` **replaces** `plan.Quantiles` with its `levels` argument; it does not combine old and new levels. `RunWithSamples(ctx,plan,sampler)` enables `RetainSamples`. Every wrapper delegates to `Run`, preserving validation, errors, memory policy and result semantics.

### Q13n.3. How should I implement `Sampler` without breaking replay?

Implement the two-method interface:

```go
type Sampler interface {
    Dimension() int
    Sample(ctx context.Context, pathID uint64, random rng.RNG) ([]float64, error)
}
```

Keep the output dimension constant and positive, and return precisely that many finite components on every successful call. Treat `random` as exclusively owned by that one path invocation. Use only explicitly fixed model parameters, `pathID` and the passed stream for stochastic decisions. Avoid global `math/rand`, mutable shared draw counters, worker-derived seeds, externally changing state and clock-dependent branches.

A sampler used with `Workers>1` must support concurrent `Sample` calls. If the model needs mutable state, move it to path-local variables or synchronize and prove that the state cannot affect draw sequences.

### Q13n.4. Why is the RNG stream constructed per path rather than per worker?

Worker scheduling is nondeterministic. Deriving streams from worker slot numbers changes which random words a path receives when you change parallelism. The driver uses `rng.Factory.Stream(Plan.Seed,pathID)` for **every** issued logical path. Persist `Result.RNG` (the factory's version), base seed, model parameters and model revision if you need replay across deployments.

### Q13n.5. What exactly is bitwise invariant across `Workers=1,2,4,8`?

For **complete successful runs** with the same deterministic concurrency-safe sampler, seed, path count, platform and underlying implementation: retained path values, every feature's mean/variance/SE, and requested empirical quantiles are bitwise reproducible. The coordinator uses path IDs in ascending order for every `stats.Moments.Update`; workers never merge partial moments. This is not a cross-platform floating-point equivalence guarantee, and partial-run `Completed` can vary with timing.

### Q13n.6. What prevents an OOM if path zero runs much slower than every other path?

The dispatcher issues only IDs `i` satisfying `nextReduce <= i < nextReduce+2W`, with the upper end capped by `Paths` and `W=min(Paths,max(1,Workers))`. If path zero has not reduced, `nextReduce` cannot advance and the pool cannot issue path 8 at `W=4`. Accepted outcomes and the pending reorder map remain `O(W*d)` in addition to application model costs. There is no Paths-sized job queue or one goroutine per path.

### Q13n.7. Can I compute variance independently in every worker and merge it later?

Not while preserving this contract. Although `stats.Moments.Merge` is valid for intentionally partitioned statistics, different worker counts induce different merge trees and floating-point rounding. `montecarlo` deliberately reduces through one coordinator in **ascending `pathID` and feature index**. Worker-local accumulation would violate complete-run bitwise worker-count parity.

### Q13n.8. What do `Mean`, `SampleVariance`, and `MeanStandardError` each mean?

- `Mean[j]`: arithmetic sample mean of outcome coordinate `j` across `Completed` committed paths.
- `SampleVariance[j]`: Bessel-corrected **sample** variance with denominator `Completed-1`, not population variance.
- `MeanStandardError[j]`: `sqrt(SampleVariance[j])/sqrt(Completed)`, measuring Monte Carlo sampling precision of the **estimated mean**, assuming independent trials.

The last value does **not** measure variability of a physical payoff itself, uncertainty in model assumptions or a validated forecast error bound. Use the sample standard deviation `sqrt(SampleVariance[j])` if your question concerns spread of the simulated outcomes.

### Q13n.9. Why use `stats.Moments` rather than `mean(x*x)-mean(x)*mean(x)`?

Raw second-moment subtraction loses small variance at large mean offsets; squaring large finite numbers can also overflow before the final variance is divided by `n-1`. The existing `stats.Moments` kernel uses compensated scaled Welford updates and its independently protected mean. A constant `math.MaxFloat64` input can have finite mean `MaxFloat64` and exact zero variance. Variance is still a floating-point approximation, not an unlimited-precision rational quantity.

### Q13n.10. Why can a finite variance produce a nonzero SE when `variance/n` underflows to zero?

The driver evaluates `sqrt(variance)/sqrt(n)` instead of computing `sqrt(variance/n)` in that order. The two expressions are mathematically equal, but dividing a tiny subnormal variance by large `n` may round to zero **before** square root, even though the actual standard error is representable. This ordering avoids that preventable intermediate underflow; it is not a guarantee of correct rounding in every binary64 case.

### Q13n.11. Are `Quantiles` exact, and must the levels be sorted?

They are exact **empirical order-statistic quantiles** over every simulated sample, without histogram approximation or reservoir subsampling. Interpolation is still performed in binary64. The caller must provide strictly increasing, distinct, finite levels in `[0,1]`; duplicates, descending values, `NaN`, infinities and out-of-range probabilities are `ErrInvalidOption`. The driver never silently sorts levels. `QuantileLevels` is a detached copy on execution results.

### Q13n.12. Why not use `a+f*(b-a)` for all interpolated quantiles?

For opposite-sign values `a=-math.MaxFloat64`, `b=+math.MaxFloat64`, `(b-a)` overflows, even though `Q(0.5)=0` is representable. For opposite signs the kernel uses `(1-f)*a+f*b`; for same signs it can safely use the bounded difference form. Exact `p=0` and `p=1` endpoints preserve observed extremes. This avoids the specific overflowing subtraction but does not turn float64 into arbitrary precision.

### Q13n.13. Why can a run fail even though the requested median is representable?

`Run` must publish **all required moments** on complete success. For observations `[-MaxFloat64,+MaxFloat64]`, the type-7 median is zero, but sample variance is not representable as finite binary64. The run returns `ErrOverflow`, `Complete=false` and no quantile matrix; it must not masquerade as a successful partial set of statistics.

### Q13n.14. Why did `RetainSamples=false` still trigger `ErrBudget`?

A request for exact quantiles requires storing all `Paths*d` values even when the public `Samples` matrix is omitted. The checked optional-buffer budget accounts for one raw sample backing, one `Paths`-length sort column when quantiles exist, and a `q*d` quantile output:

![Monte Carlo budget](https://latex.codecogs.com/svg.image?B%3D8Nd%5Cmathbf%7B1%7D_%7B%5Cmathrm%7Bretain%7D%5Clor%20q%3E0%7D%2B8N%5Cmathbf%7B1%7D_%7Bq%3E0%7D%2B8qd)

`MaxSamplesBytes=0` selects the default **64 MiB**, not unlimited memory. The limit is about these buffers, not process RSS or allocations within the sampler. Check for `ErrBudget` before increasing a ceiling or dropping exact quantiles intentionally.

### Q13n.15. Is the memory ceiling a guarantee that `Run` cannot run out of RAM?

No. It checks optional raw/sort/quantile buffer arithmetic **before the first sampler call** and protects against impossible sizes. It excludes sampler-managed arrays, per-path RNG/returned vectors, `stats.Moments`, channel/map metadata, goroutine stacks and runtime overhead. Even representable admitted `make` calls may fail at the OS/runtime level. Total application memory planning remains the caller's responsibility.

### Q13n.16. How does a partial result preserve an exact prefix?

`Completed=k` means exactly IDs `[0,k)` were accepted and reduced in that order. A successful later path `k+2` is never counted if path `k` failed or did not finish. A terminal sampler/context error stops future dispatch; the coordinator drains accepted jobs and can still reduce earlier successes as their gaps close. The exact `k` after cancellation depends on timing. Never interpret it as “the number of successful outcomes anywhere in the window.”

### Q13n.17. Are the moment vectors and quantiles always available after cancellation?

`Quantiles` is **nil on any unsuccessful run**, even if `Completed>=2`. For `Completed<2`, `Mean`, `SampleVariance` and `MeanStandardError` are nil. With `Completed>=2`, a successful final summary may describe that exact prefix; but if materializing required numeric statistics fails, the vectors can remain nil. Consumers must inspect `err`, `Complete`, `Completed` and field presence instead of guessing.

### Q13n.18. Can `Completed==Requested` occur when `Complete==false`?

Yes. All path vectors may have been reduced before a final `stats.Moments.Summary(1)` reports an unrepresentable variance, or context may cancel during final quantile sorting. `Completed` means *paths reduced*, not *all requested statistics published*. The only successful full experiment is `err==nil && result!=nil && result.Complete`.

### Q13n.19. What exactly is `Samples`, and is it detached from the sampler?

On a complete retained run, `Samples` is a `matrix.Matrix` backed by an owning `*matrix.Dense` with rows in path-ID order. On an interrupted retained run with `Completed>0`, it is a private `matrix.Matrix` prefix view exposing exactly `Completed × d`; its originally allocated full `Paths × d` Dense backing remains retained, but inaccessible rows cannot be queried through the view. `Samples.Clone()` produces an independently owned compact matrix for those visible rows. The driver copies returned sampler slices before publication, so result matrices and vectors do not alias a sampler's output buffer.

### Q13n.20. Can my sampler reuse the same `[]float64` buffer across concurrent paths?

Only with a genuinely race-free ownership protocol. The worker copies the returned slice *after* `Sample` returns. If another goroutine mutates that same backing array while the worker copies, it is a race and the resulting vector may be corrupted. Returning a fresh per-path slice is the simplest safe approach; a model may use a pool only if its transfer and reuse policy prevents overlapping read/write access.

### Q13n.21. Does `Result.IsNil()` tell me that the experiment was completed?

No. The method implements the `core.Nilable` convention and returns `true` **only** for a nil `*Result` receiver. A non-nil partial or numerically failed result is still non-nil. `Complete` and `err` are the separate success protocol. The result's public slices and matrices are caller-owned and must not be mutated concurrently without synchronization.

### Q13n.22. What are the exact validation and runtime error classes?

Preflight order: `ctx` presence/pre-cancellation -> `Paths`/`Workers`/`Quantiles` -> sampler presence (including `IsNil()` if exposed) -> positive `Dimension()` -> checked integer sizes and budget -> sampling. Classify with `errors.Is`: `ErrInvalidOption`, `ErrNilInput`, `ErrOverflow`, `ErrBudget` before work; `ErrShape`/`ErrNonFinite` for invalid successful sample vectors; `ErrDomain`/`ErrOverflow` for dependency/statistic failures; original context and user-sampler causes are preserved. Competing runtime error causes are first-*observed*, not ordered by path ID.

### Q13n.23. What happens if my `Sampler.Sample` panics?

A panic is not swallowed, returned as `ErrDomain`, converted to a zero vector, or automatically propagated to the `Run` caller's goroutine. `Sample` executes inside a worker goroutine and the package uses neither `recover()` nor `panic()` nor `reflect`. An unhandled sampler panic has the normal Go process-terminating semantics. It does **not** guarantee orderly pool cleanup. Return an error for recoverable model failures; if you need to isolate untrusted panic-prone simulation code, use a supervised separate process.

### Q13n.24. Can cancellation forcibly stop a slow or blocked sampler?

No. The driver cancels its child context, stops issuing new work, drains accepted outcomes and waits for its workers. A sampler that never checks `ctx.Done()` or never returns can prevent `Run` from returning. A timeout on the **caller context** is a cooperative cancellation request, not a Go goroutine-kill primitive. Structure long path simulations to check `ctx.Err()` at meaningful stages.

### Q13n.25. How do I compute VaR and Expected Shortfall correctly?

For a sampler returning **signed losses**, request the desired high quantile such as `0.99` through `RunWithQuantiles` and interpret it in the same loss units as the sampler. A quantile is **not** Expected Shortfall. One exact empirical tail calculation can replay the same per-path streams with a second sampler that returns `[lossIfInTail,tailIndicator]` under an explicit `loss>=VaR` tie convention; `mean(tailLoss)/mean(tailIndicator)` is then the conditional tail mean if the count is nonzero. That two-pass pipeline is demonstrated in [`../montecarlo/example_test.go`](../montecarlo/example_test.go). Its statistical validity still depends on the chosen loss model and the simulation ensemble.

### Q13n.26. How should I benchmark and verify a production Monte Carlo integration?

Test analytical oracles for means, Bessel variance, SE and type-7 quantiles; full-success bit parity for workers 1/2/4/8; a blocked path-zero barrier verifying the `2W` issue limit; cancellation with strict prefix/quantiles-nil semantics; budget/overflow checks **before Sample**; wrong shape/nonfinite/error wrapping; and detached ownership. Use only external `package montecarlo_test` tests, `errors.Is`, race checks, `go vet`, and the repository `golangci-lint`. Bench separately with dimension `1` versus `32`, paths `1,000` versus `100,000`, workers `1` versus `4`, and mean-only/retained/quantiles modes. Report `allocs/op` and real model work; avoid treating timing-based cancellations as deterministic unit-test oracles. See [`MONTECARLO.md`](MONTECARLO.md#13-verification-and-benchmark-regimes).

---

## 13o. GBM FAQ: Exact Lognormal Steps, Units, Overflow and Complete Paths

### Q13o.1. When should I choose `gbm` rather than `rng` or `montecarlo`?

Use `rng` to obtain reproducible normal draws, `gbm` to transform those draws into one **positive scalar geometric Brownian motion path**, and `montecarlo` to run and summarize many independent paths. `gbm.Simulate` owns the model transition, not parameter estimation, dividend calendars, instrument-specific payouts, parallel scheduling or statistical confidence intervals. For a complete cross-package example, see [GBM's Monte Carlo bridge](GBM.md#10-monte-carlo-option-pricing-bridge).

### Q13o.2. Is the simulated process really exact, or is this Euler–Maruyama?

The requested endpoints use **exact-in-distribution conditional lognormal transitions** for constant `mu` and `sigma`:

$$
\Delta\log S=(\mu-\sigma^2/2)\Delta t+\sigma\sqrt{\Delta t}\,Z.
$$

![GBM conditional lognormal transition](https://latex.codecogs.com/svg.image?%5Cdisplaystyle%20%5CDelta%5Clog%20S%3D%28%5Cmu-%5Csigma%5E2%2F2%29%5CDelta%20t%2B%5Csigma%5Csqrt%7B%5CDelta%20t%7DZ)

There is no Euler–Maruyama time-discretization bias for these endpoints. There *is* finite-precision arithmetic error, RNG approximation, and model risk. Simulating only discrete endpoints also says nothing about a continuous-time barrier crossing between them.

### Q13o.3. Is `TimeYears` the automatic default when `Unit` is omitted?

`TimeYears` has enum value zero, so a zero-valued `TimeUnit` happens to identify years. **This is not permission to forget units.** Set `Unit` explicitly in both `Parameters` and `TimeGrid`, especially in multi-source applications. Invalid enums or mismatched units are `ErrInvalidOption`; GBM does not infer seconds from a small timestamp or infer years from a large drift.

### Q13o.4. Why can't I combine annual drift with second-based grid timestamps?

The drift has units `1/Unit`, while volatility has units `1/sqrt(Unit)`. Applying annual coefficients to seconds silently multiplies modeled growth and variance by the wrong time scale. `Parameters.Unit != TimeGrid.Unit` is therefore rejected before any normal draw. There is no hidden 252-session, 365-day or 365.25-day conversion.

### Q13o.5. How do I deliberately convert annual parameters to seconds?

Choose and document a calendar convention in **caller code**. If `secondsPerYear` expresses that convention, use

```go
const secondsPerYear = 365.25 * 24 * 60 * 60
secondsDrift := annualDrift / secondsPerYear
secondsVolatility := annualVolatility / math.Sqrt(secondsPerYear)
// Use TimeSeconds for both parameters and the elapsed-second grid.
```

Both coefficients must be converted. Dividing volatility by `secondsPerYear` instead of its square root changes the diffusion distribution. The library never chooses this constant.

### Q13o.6. Are Unix timestamps or arbitrary nonuniform grids valid?

Irregular spacing is valid; **epoch timestamps are not directly suitable** because the first grid entry must be numerically zero. Rebase observations to elapsed time and use a consistent unit, such as seconds since the first tick. Every later value must be finite and strictly greater than its predecessor. `gbm` does not sort, resample, interpolate or create missing observations. A large `int64` nanosecond feed should be checked and carefully converted before use; adjacent distinct integers can collapse to equal float64 timestamps.

### Q13o.7. Why is `[0]` a valid grid?

It represents a request for the known initial observation only. On success, the result contains exactly `Times=[0]`, `Values=[Initial]`, `Complete=true`, and **zero** `Normal()` calls even when `Volatility>0`. The argument `random` must still pass non-nil validation. `SimulateTerminal(..., horizon=0, ...)` is the corresponding wrapper.

### Q13o.8. Why does zero volatility still require a non-nil RNG?

The canonical public validation law does not change with model parameters. A missing generator is `ErrNilInput` for both deterministic and stochastic paths. Once validated, `sigma==0` follows the analytic `Initial*exp(mu*t)` path and does not call `Normal`, `UniformOpen` or `Uint64`. This both saves entropy and makes draw-count replay straightforward.

### Q13o.9. How many Gaussian draws will a stochastic path consume?

A successful `n`-timestamp path with `sigma>0` calls `random.Normal()` exactly `n-1` times in increasing step order. A step that produces a nonfinite draw or unrepresentable price may consume its own draw but is excluded from the published result. Cancellation detected before a step consumes no draw for that step. The underlying pinned RNG's *raw word* consumption is different because Box–Muller caches a spare Gaussian variate.

### Q13o.10. Why not compute `S * exp(increment)` directly?

Because the intermediate `exp(increment)` can overflow even when `S * exp(increment)` is mathematically representable. For example, a very small `Initial=1e-300` and a log increase of about `1000` yield a finite positive result despite `math.Exp(1000)` being infinite. The implementation accumulates log growth relative to the unchanged starting value and converts to a `float64` price only when publishing a complete step.

### Q13o.11. Why can `sigma*sigma*dt` also fail for valid inputs?

For `sigma=1e160` and `dt=1e-320`, the direct `sigma*sigma` intermediate exceeds `MaxFloat64` even though `sigma*sqrt(dt)` is approximately one. The GBM kernel computes the interval and its square root in extended-exponent arithmetic, then constructs the Itô correction from the scale. An **actually unrepresentable output** remains `ErrOverflow`; the method does not promise exact-real arithmetic or unlimited exponents.

### Q13o.12. Are positive subnormal prices permitted?

Yes. All **strictly positive, finite** float64 values, including subnormals, are valid published prices. If a computed new price rounds to exactly zero or infinity, the step fails with `ErrOverflow`. The package never substitutes `math.SmallestNonzeroFloat64` or an arbitrary epsilon as a fabricated price floor.

### Q13o.13. Is `Result.Values[len-1]` always the requested terminal price?

Only when `err==nil`, `result!=nil` and `result.Complete==true`. A runtime failure returns only the completed prefix, including time zero, with `Times` and `Values` of equal length. **The failed time step is not included**, so the last published value may be an earlier observation. In particular, do not price a maturity payoff or estimate full-horizon risk from an incomplete prefix.

### Q13o.14. Does GBM return a partial result for invalid inputs?

No. Nil context, units mismatch, nil/empty grid, duplicate times, nonfinite coefficients and physical-domain errors fail **preflight** with a nil `*Result`. Only failures **after** preflight may return a non-nil prefix: cancellation observed before a step, nonfinite `Normal()` or an unrepresentable computed price. `Complete=false` accompanies the runtime error; there is no successful incomplete simulation.

### Q13o.15. Can cancellation interrupt `random.Normal()` immediately?

No. `gbm` checks `ctx.Err()` before each new step; the `rng.RNG.Normal` interface itself takes no context. A cancellation occurring during `Normal` or numeric arithmetic is normally noticed at a later iteration boundary, not forcibly in the middle of an operation. A cancellation that arrives after the last step's check can coexist with a complete result. Pre-canceled contexts instead return `nil, ctx.Err()` before any simulation work.

### Q13o.16. Which error wins when several inputs are invalid?

The order is **context → units → RNG presence → grid presence/length → origin and ascending timestamp scan → parameter finiteness → parameter domain → runtime**. In the timestamp scan, each point is checked for NaN/Inf before strict order. Consequently a nonfinite grid point is `ErrNonFinite`, a finite duplicate is `ErrOrder`, and a unit mismatch wins before either. For the exact validation matrix, see [`GBM.md`](GBM.md#5-validation-and-error-precedence).

### Q13o.17. Should I compare `err == gbm.ErrOverflow`?

No. Sentinels are classified through `errors.Is`, which remains correct when an adapter wraps the error:

```go
if errors.Is(err, gbm.ErrOverflow) {
    // Discard the failed endpoint; retain the completed prefix only if useful.
}
if errors.Is(err, context.Canceled) {
    // Preserve the calling operation's cancellation classification.
}
```

Do not branch on `err.Error()` or string fragments. The package defines distinct `ErrInvalidOption`, `ErrNilInput`, `ErrEmptyInput`, `ErrOrder`, `ErrNonFinite`, `ErrDomain` and `ErrOverflow` sentinels.

### Q13o.18. Which of the four public facades should I use?

| Goal                                 | API                                                        |
|:-------------------------------------|:-----------------------------------------------------------|
| Full, explicitly modeled path        | `gbm.Simulate(ctx,p,grid,stream)`                          |
| Scalar parameters with elapsed times | `gbm.SimulatePath(ctx,initial,mu,sigma,unit,times,stream)` |
| Terminal-only payoff                 | `gbm.SimulateTerminal(ctx,p,horizon,stream)`               |
| One replayable standalone scenario   | `gbm.SimulateSeeded(ctx,p,grid,seed,pathID)`               |

`SimulateTerminal` still returns `*Result`, not a bare price. `SimulateSeeded` is **not** appropriate for creating a new RNG inside `montecarlo.Sampler.Sample`, because the driver already provides the logical path's stream.

### Q13o.19. How do I bridge a GBM path into `montecarlo.Sampler` safely?

Implement `Dimension() int` and `Sample(ctx,pathID,random)` in your application. Pass the **supplied** `random rng.RNG` to `gbm.SimulateTerminal` or `Simulate`. Return the GBM error unchanged; guard `result==nil || !result.Complete` before constructing a payoff; validate the resulting payoff as finite. Then `montecarlo.Run` can reduce full-horizon samples by path ID. Do not convert `Complete=false` into a successful row. A [complete, runnable call-pricing sampler](GBM.md#10-monte-carlo-option-pricing-bridge) shows the interface.

### Q13o.20. Can a terminal-only simulation price an Asian or barrier option?

Not when the payoff depends on intermediate values. `SimulateTerminal` returns `[S0,S(T)]` for positive `T`; it cannot describe observed path extrema or averages. Use an explicit time grid for discretely monitored payoffs. **Even then**, a GBM endpoint grid does not certify that a continuous barrier was never crossed between two observations. Continuous monitoring requires additional bridge mathematics outside this package.

### Q13o.21. Does this package turn exchange sessions or nanosecond `series` grids into GBM time?

No. `series.Validate` uses signed integer nanoseconds and optional cadence checks; GBM uses elapsed `float64` timestamps and **three explicit unit labels**. When composing them, the caller is responsible for checking source integrity, choosing a time origin, converting to elapsed years/days/seconds, and ensuring strict ordering survives floating conversion. There is no automatic trading calendar, interpolation or 252-session annualization.

### Q13o.22. What drift should I supply for a risk-neutral dividend-paying call?

Under a **caller-chosen** constant-rate risk-neutral model, use `Parameters.Drift=r-q` where `r` is the interest rate and `q` the continuous dividend yield, both per chosen time unit. The transition already subtracts `sigma²/2` in **log coordinates**. Double-subtracting it from the input drift is wrong. Discount a terminal call payoff externally by `exp(-r*T)`. The package neither decides whether the measure is risk-neutral nor fits the model.

### Q13o.23. Does a simulated GBM path prove future market prices are lognormal?

No. Many observed series exhibit jumps, stochastic volatility, leverage effects, serial dependence, calendar discontinuities or changes in regime. GBM is useful as an explicit baseline or scenario generator, not a universal market model. Calibrate and backtest outside GBM; distinguish estimation uncertainty, Monte Carlo sampling error, market-model risk and liquidity/transaction-cost effects.

### Q13o.24. Are `TimeGrid` inputs and returned slices independent?

The caller owns the input `TimeGrid.Values`, which is borrowed read-only during the operation; GBM does not retain or modify it. `Result.Times` and `Result.Values` are detached caller-owned slices, including on runtime failure. Each simulation has its own numeric workspace. Different goroutines may share a stable grid and copied model values but **must not share the same mutable `rng.RNG` concurrently**. Concurrent mutation of an input slice while it is read is unsupported.

### Q13o.25. Is `Result.IsNil()` equivalent to `Complete`?

No. `(*Result)(nil).IsNil()` returns true; a non-nil incomplete Result returns false. `IsNil` implements `core.Nilable` pointer-presence semantics, not mathematical success, valid-horizon status, ownership, or a price-domain check. Always inspect **both** the returned error and `Complete`.

### Q13o.26. How do I make standalone path replay stable?

Use `gbm.SimulateSeeded` or one explicit `rng.Factory.Stream(seed,pathID)` for each logical path. Persist the same model coefficients and units, ordered grid, base seed, logical path ID, RNG revision (`Factory.Version()`) and GBM implementation version when strict audit replay matters. Never derive path ID from a worker slot or reseed after partial failure expecting the prior stream's advanced state to roll back. Identical results across Go/math backends are not promised to the last bit.

### Q13o.27. What do meaningful GBM tests and benchmarks measure?

Use independent `sigma=0` analytic solutions, zero-shock Itô-correction oracles, controlled `Normal` sequences, irregular grids, zero-RNG consumption, exact prefix lengths, all sentinel precedence families, positive subnormal boundaries, extreme sigma/tiny-dt cancellation, ownership isolation and Monte Carlo worker 1/4 replay. Keep tests in `package gbm_test` through public API. Benchmark grid lengths **100 and 10,000** with `sigma=0` and `sigma>0`, after prebuilding fixtures; record allocations as well as time. A fixed-precision `math/big` square root has meaningful constant cost even though path complexity is `O(n)`. See [GBM verification](GBM.md#12-testing-benchmarking-and-limitations).

---

## 13p. Jump Diffusion FAQ: Merton Compensation, Poisson Counts and Replay

### Q13p.1. When should I use `jumpdiffusion` instead of `gbm`?

Use `gbm` when a continuous lognormal diffusion with constant drift and
volatility is the actual assumed model. Use `jumpdiffusion` when you
**explicitly** want independent Poisson arrival counts with lognormally
multiplicative discontinuities at observation-grid endpoints. Both return
positive-price `Result` paths; jumpdiffusion is still **experimental** as a
modeling component and does not calibrate itself to observed crash tails.
The diffusion uses `gbm.Parameters`, not a duplicate parameter type.

### Q13p.2. Does this model actually simulate the times of individual jumps?

No. For `a=lambda*dt`, it draws the integer count `N~Poisson(a)` and one
aggregate normal log jump `N*m+sqrt(N)*s*Z` when needed. The returned
`Times` are **only the requested grid timestamps**. A path may hide a jump
and subsequent recovery inside an interval. Continuous-barrier probability,
intrastep extrema, order-book dynamics and event-by-event sequences need a
different simulator.

### Q13p.3. Why is `LogJumpMean` not the drift compensator?

`m=E[Y]` is the mean of the *logarithm* of one price multiplier. The price
jumps by `e^Y`, so its expected **relative arithmetic** jump is
`k=E[e^Y-1]=expm1(m+s^2/2)`. Setting `k=m` biases the expected price, even
if the error looks small on one sample. At `s>0`, Jensen's inequality already
shows why the extra `s^2/2` matters. `math.Expm1` preserves precision when
the exponent is near zero.

### Q13p.4. What exactly do Compensated and Uncompensated mean?

`Compensated` (zero-value enum) subtracts `lambda*k` from the proportional
drift before the Itô `sigma^2/2` correction. The mathematical expected price
is `S0*exp(mu*t)`. `Uncompensated` does not subtract the jump mean and yields
`S0*exp((mu+lambda*k)*t)` when the expectations exist. Both simulate the
same sampled jump/normal sequences for identical RNG states; the deterministic
drift differs. **Neither setting chooses a real-world versus risk-neutral
probability measure automatically.**

### Q13p.5. Can I put `r-q` into `Diffusion.Drift` and directly price an option?

Only if you have **separately selected and justified** a risk-neutral jump
measure consistent with the model. Under that assumption, compensated
`Diffusion.Drift=r-q` preserves the desired discounted stock expectation.
The caller still computes the contractual payoff and discounts it by
`exp(-r*T)`; `jumpdiffusion` does not price options or fit a volatility smile.
See the checked `ExampleSimulateTerminal_optionPricing` for a caller-owned
`montecarlo.Sampler` and a put–call-parity diagnostic.

### Q13p.6. Why not replace a large Poisson count by a Gaussian approximation?

Because `N` is an **integer random variable**. Rounded normals change low
count probabilities and the compound-jump distribution, and can destroy tail
risk comparisons. The package uses product inversion at `0<a<30` and
Hörmann PTRS at `30<=a<=2^40`. There is **no Gaussian fallback**.

### Q13p.7. Why does the Poisson algorithm switch at `a=30`?

For small means, product inversion typically takes `O(a+1)` open uniforms;
for large means, this is wasteful. PTRS proposes a count with transformed
rejection and a squeeze test, using two open uniforms per attempt with
bounded expected work in the intended regime. The threshold is a
**versioned reproducibility boundary**: different branches consume different
RNG sequences. A statistical distribution check does not prove byte-level
sequence parity across the boundary.

### Q13p.8. Can the rejection loop hang indefinitely on a bad RNG?

No. Both methods are bounded by `2^20` attempts **per interval**, with a
context check on each attempt. Exhaustion returns `ErrBudget` and the already
completed path prefix. `UniformOpen` values must be finite and strictly in
`(0,1)`; out-of-range endpoints return `ErrDomain`, while NaN/Inf returns
`ErrNonFinite`. A custom RNG that blocks inside its method cannot be forcibly
interrupted by a Go context—this is a limitation of the external interface.

### Q13p.9. What are the mean and count limits?

The per-interval Poisson mean is restricted to `0<=lambda*dt<=2^40`.
An accepted count must be at most `2^53-1`, so it converts to a float64
integer without loss of integer precision. Too large an interval mean is
`ErrDomain` **before entropy consumption for that interval**. Values are
checked per interval, not only for the model's total horizon.

### Q13p.10. Why is `lambda*dt` evaluated with extended precision?

Subtracting two large but distinct finite timestamps and then multiplying
by a very small intensity can lose a positive result in binary64. The
kernel forms the exact-input interval and product with a 2304-bit workspace
before the `2^40` range check. If the positive product cannot survive its
required float64 representation (rounds to zero), the simulator returns
`ErrOverflow`; it does **not** pretend the model had `lambda=0`.

### Q13p.11. What happens when `Intensity=0`?

After validating both jump coefficients (`m` and `s` still must be finite,
and `s>=0`), `Simulate` calls **`gbm.Simulate` directly**. This preserves
GBM floating-point bits and its RNG consumption on the same starting stream.
There are zero Poisson uniforms and zero jump normals. The result is a new
`jumpdiffusion.Result` wrapper over the detached GBM path, not the same
`*gbm.Result` pointer.

### Q13p.12. When do stochastic draws occur, exactly?

For each positive-intensity admitted step: `Normal()` for diffusion only if
`sigma>0`; then `UniformOpen()` attempts for Poisson; then **one** jump
`Normal()` only if `N>0 && s>0`. When `sigma=0` or `s=0`, its corresponding
normal call is skipped. One-point `[0]` grids consume no interval draws.
Because `rng` may cache a Box–Muller normal, **method order**, not just the
normal count, is part of replay identity.

### Q13p.13. Why does the package not generate `N` Gaussian jump draws?

The sum of `N` independent `Normal(m,s^2)` marks is conditionally
`Normal(N*m,N*s^2)`. The implementation uses `N*m+sqrt(N)*s*Z` for
`N>0`, so cost does not scale with the realized count. If `s=0` it
produces exactly `N*m` with **no jump normal draw**. This preserves the
fixed-grid *distribution*, not the path of individually timestamped marks.

### Q13p.14. Is `s=0, m=0` really a no-jump simulation?

The sampled count still exists and consumes Poisson uniforms for every
positive-intensity interval, but its price multiplier is exactly one. With
`sigma=0`, the price is the deterministic drift trajectory, independent of
the realized counts. Do **not** remove those count draws as an optimization:
it would change the documented RNG consumption protocol. To suppress all
jump entropy, use **`Intensity=0`**, which delegates to GBM.

### Q13p.15. Which of the five facades should I use?

| Need                                          | Public method                                 |
|:----------------------------------------------|:----------------------------------------------|
| An explicit irregular-grid path               | `Simulate(ctx,p,grid,random)`                 |
| Timestamps already in the model unit          | `SimulatePath(ctx,p,times,random)`            |
| A single positive expiry or zero-horizon mark | `SimulateTerminal(ctx,p,T,random)`            |
| Reproducible stand-alone full path            | `SimulateSeeded(ctx,p,grid,seed,pathID)`      |
| Reproducible stand-alone maturity             | `SimulateTerminalSeeded(ctx,p,T,seed,pathID)` |

All return the canonical `*Result` with `Times`, `Values` and `Complete`.
There is no alternate solver or automatic policy selection in the wrappers.

### Q13p.16. What physical time unit is `Intensity` in?

It is **expected jumps per `Diffusion.Unit`**. If the grid is in seconds,
`Intensity=50` means 50 expected arrivals *per second*. If the grid is in
years, `Intensity=2` means 2 expected arrivals *per model year*.
`Diffusion.Drift` scales as inverse time and `Volatility` as inverse square
root time. The package never assumes 252 trading days, 365 calendar days or
an exchange schedule; choose a conversion externally and apply it to the
grid, drift, volatility **and** intensity.

### Q13p.17. Why can valid finite jump parameters still cause `ErrOverflow`?

A finite `m,s` can make `exp(m+s^2/2)-1` unrepresentable as a finite
float64. `Compensated` computes and validates `k` before simulation and
returns a nil result with `ErrOverflow`. Wide arithmetic avoids premature
`s*s` overflow, but cannot make an unrepresentable *final* compensator
valid. `Uncompensated` skips this calculation; a simulated price may still
fail at its publication boundary.

### Q13p.18. Why is a positive subnormal price accepted but zero rejected?

The model starts at `S0>0`; a strictly positive subnormal is still a valid
binary64 price. Zero is not. `jumpdiffusion` accumulates anchored log growth
in extended arithmetic and publishes only finite `price>0`. If the rounded
candidate becomes `0` or infinity, it returns `ErrOverflow` with the
previously completed prefix. There is **no epsilon clamp** and no
`S_previous*exp(increment)` intermediate that can overflow unnecessarily.

### Q13p.19. What is an incomplete `Result`, and can I price with its last value?

No. A runtime failure publishes the completed pairs only, with
`len(Times)==len(Values)>=1`, `Complete=false` and non-nil error.
The unfinished interval and its candidate timestamp are absent. An
invalid input or pre-canceled context instead returns **nil result**.
`Result.IsNil()` is pointer identity, not a completed-horizon check.
A `montecarlo.Sampler` must reject a failed/incomplete path, not use its
last completed mark as if it were the requested expiry.

### Q13p.20. What error category takes precedence for invalid inputs?

Nil context, pre-canceled context, unit/convention mismatch, RNG/grid
presence, grid origin/order and nonfinite timestamps, parameter finiteness,
physical domain, then compensator arithmetic. Timestamps are scanned in
ascending order; a first-index error is not overridden by a different error
at a later index. `ErrBudget` and runtime `ErrDomain`/`ErrOverflow` carry
completed prefixes. Use `errors.Is`, never error-string matching. Zero
intensity preserves relevant underlying `gbm` sentinels through `errors.Is`.

### Q13p.21. How should I implement `montecarlo.Sampler` correctly?

Use the path-specific `rng.RNG` supplied by `Sampler.Sample`; call
`SimulateTerminal` or `Simulate` with an immutable model; check **both**
`err` and `Complete`; then return one new fixed-length finite payoff vector.
Do not create `SimulateSeeded` inside the Sampler: the driver already keys
streams by `(Seed,pathID)`. Store seed, path ID semantics, RNG version,
model coefficients and grid for replay; scheduler worker IDs are not path IDs.

### Q13p.22. Is a modeled flash crash frequency a validated real-world probability?

No. A fixed-seed stress scenario describes **outcomes under chosen parameters**.
Confidence intervals or Monte Carlo standard errors measure sampling error,
not errors in jump arrival assumptions, jump magnitudes, market volatility,
liquidity, clustering or calibration. The model has independent homogeneous
Poisson arrivals and Gaussian log-jump sizes, not self-exciting arrival rates.

### Q13p.23. Does the package require NumPy or Python?

**No.** The Go implementation only calls methods of `rng.RNG` and Go
standard-library numerics. `poisson.go` cites the published PTRS parameterization
cross-checked against the NumPy 2.2.0 source, but neither NumPy nor Python is
linked, imported or installed to run the model. `THIRD_PARTY_NOTICES.txt` records
source attribution and the NumPy BSD-3-Clause notice, **not** a dependency.
Do not remove it without a provenance/licensing review of the adapted PTRS
source. See [`JUMPDIFFUSION.md`](JUMPDIFFUSION.md#13-algorithm-provenance-and-license-notices).

### Q13p.24. What tests protect the mathematical contract?

The external-package suite covers JD01–JD08, fixed-seed compensated moment
checks, independent Poisson frequencies, exact count/draw-order fixtures,
large-mean acceptance thresholds, partial publication, numerical underflow,
context cancellation, result ownership and `core.Nilable`. `bench_test.go`
measures 100/10,000 intervals at per-step means 0.1/50. A passing statistical
test **cannot prove** distributional correctness by itself; independent
acceptance oracles and algebraic expectation identities are also necessary.

For exact formulas, sample code and the complete error matrix, see
[`docs/JUMPDIFFUSION.md`](JUMPDIFFUSION.md),
[`jumpdiffusion/doc.go`](../jumpdiffusion/doc.go),
[`jumpdiffusion/example_test.go`](../jumpdiffusion/example_test.go)
and [`GBM FAQ`](#13o-gbm-faq-exact-lognormal-steps-units-overflow-and-complete-paths).

## 13q. GJR-GARCH FAQ: Causal Variance, Leverage, QMLE and Model Release

### Q13q.1. When should I use GJR-GARCH rather than GBM, Merton jumps, ACF or HMM?

Use `gjrgarch` to **measure conditional return variance and its sign-dependent
persistence**. `gbm` and `jumpdiffusion` simulate price-path distributions
under explicitly chosen stochastic assumptions; `acf` measures
*fixed-window* lag correlation and is not a causal volatility filter; `hmm`
models latent states with filtering and future-conditioned smoothing. None of
these packages automatically establishes an empirical GJR fit. Combining them
requires an application-owned modeling assumption, not a hidden adapter.

### Q13q.2. Why does `Filter` return the initial variance as its first element?

The contract defines `h[t]` as the variance **assigned to return[t] before
return[t] is observed**. Thus `h[0]=initialVariance`, and its first residual
may influence only `h[1]`. Using `epsilon[0]` inside `h[0]` would leak the
observation into its own risk estimate. Appending a valid future suffix cannot
revise an existing variance prefix when the starting variance is fixed.

### Q13q.3. What is stored in `State` and how do I resume an online filter?

`InitialState(firstReturn,h0)` constructs `State{LastResidual:epsilon[0],
LastVariance:h[0],Steps:1,ModelID:...}`. `Step(previous,nextReturn)` consumes
exactly one later return and creates the next independent value. Copy the
previous `State` and persist the feed's source offset **outside this package**.
On restart, validate the feed offset and model version before continuing.
A zero `State`, mismatched ID, nonfinite variance/residual, `h<=0` or `Steps=0`
returns `ErrStateVersion`; do not invent a posterior from zero-value fields.
The identity is a **model-compatibility hash**, not authentication of a mutable
external checkpoint.

### Q13q.4. Does the current return determine its own predicted variance?

**No.** For each call:

```text
old State(epsilon[t],h[t])
    -> h[t+1] = Omega + shock(epsilon[t]) + Beta*h[t]
    -> epsilon[t+1] = nextObservedReturn - Mean
    -> new State(epsilon[t+1],h[t+1])
```

The incoming residual must be recorded only after the next variance is fixed.
`ForecastNextVariance(oldState)` uses the same prior shock without changing
state; it can be queried *before* the next market observation arrives.

### Q13q.5. Is a negative Gamma invalid or a mistake?

Not necessarily. Valid parameters satisfy `Omega>0`, `Alpha>=0`, `Beta>=0`,
`Alpha+Gamma>=0`, and `rho=Alpha+Beta+Gamma/2<1`. Negative `Gamma`
is legitimate when downside shocks produce a **smaller** subsequent conditional
variance response than equal-sized positive shocks. The indicator is strictly
`epsilon<0`, so `+0` and `-0` use the nonnegative-shock coefficient.

### Q13q.6. Why is persistence `Alpha+Beta+Gamma/2` instead of `Alpha+Beta+Gamma`?

For **symmetric**, unit-variance standardized innovations,
`E[z²*I(z<0)]=1/2`. Only half the asymmetric shock contribution enters
the unconditional second-moment recurrence. The resulting stationary level
is `Omega/(1-rho)`. For skewed innovation distributions the expectation must
be rederived; GJR-GARCH v1 deliberately does not support a skewed likelihood.

### Q13q.7. Why is the first forecast different from the remaining horizons?

The first forecast is based on the **known** last residual: its exact sign and
magnitude are in `State`. Later residuals are unknown. `ForecastVariance`
therefore uses the expected recursion `Omega+rho*h` from the second forecast
onward. `ForecastVolatility` returns `sqrt(E[h])`, not `E[sqrt(h)]` or a
full predictive distribution. A 10-day Gaussian `z*sqrt(sum(forecastVariances))`
is a planning proxy, not a calibrated tail-risk or VaR/ES guarantee.

### Q13q.8. Does a variance floor repair an otherwise invalid fit?

**No.** `VarianceFloor` is a hard feasibility bound in **squared return
units**. Fit requires `Omega>=VarianceFloor` and nonnegative shock/lag
coefficients; a computed variance below the floor causes candidate rejection.
Clamping the likelihood's `h[t]` after computation would change the objective,
its derivatives, and the stochastic model. `New` has no QMLE floor requirement;
therefore constructing a valid `Model` does **not** prove that the model is a
feasible candidate for a particular Fit policy.

### Q13q.9. How is QMLE different from maximum likelihood and from the fit result?

The Gaussian conditional log-likelihood is used as a **quasi**-likelihood:
innovations need not actually be Gaussian for it to be a useful estimation
criterion under additional assumptions. The objective is *negative average*
log-likelihood, while `FitResult.LogLikelihood` is the **total**, nonnegated
quantity. `Diagnostics.Objective[0]` corresponds to the requested initial
point; later entries correspond only to accepted optimizer moves. A lower
training objective does not certify predictive accuracy or standard errors.

### Q13q.10. Why does the initial variance affect the analytic gradient?

Every evaluated parameter candidate uses `h[0]=Omega/(1-rho)`, not a frozen
variance from the initial solver iterate. Changing `Omega`, `Alpha`, `Gamma`
or `Beta` changes both the current h[0] and every later recurrence. The
analytic score differentiates those dependencies and the explicit
`epsilon[t]=r[t]-Mean` derivative. Dropping the initial derivative may still
produce superficially decreasing objectives, but it is **not** the derivative
of the documented likelihood.

### Q13q.11. Can `EvaluateGaussianQMLE` score held-out data?

It evaluates a **fixed candidate** with a fresh stationary `h[0]` and returns
`Objective`, `LogLikelihood`, `Gradient`, and `Observations`; it neither fits
parameters nor continues an earlier posterior. This is useful for an
independently initialized window or numeric-gradient audit, **not** for an
already-live conditional log-score. For a chronological holdout, continue
from the last training State and score each observation using the variance
forecast *before* updating with that same observation. The runnable pipeline
is in [`GJRGARCH.md`](GJRGARCH.md#5-conditional-gaussian-qmle-and-analytic-score).

### Q13q.12. What if `Fit` returns a model together with `ErrNotConverged`?

Keep the prior production model. A non-nil `*FitResult` can be a **diagnostic
candidate**, not an approved replacement. Inspect both `errors.Is(err,
gjrgarch.ErrNotConverged)` and `result.Diagnostics.Converged`; `IsNil()`
checks pointer presence only. A canceled context before initial evaluation
returns `nil,ctx.Err()`; a later cancellation can return the last validated
candidate with `ctx.Err()`. No false success or unmarked fallback is allowed.

### Q13q.13. Which input errors take precedence?

For Fit the order is context, option validation including `Initial`, nil
input, empty input, full nonfinite scan, then numerical computation. A nil
context is `ErrInvalidOption`; a previously canceled context returns its
original context error. `ErrNonFinite` is invalid **caller input**;
`ErrOverflow` is unrepresentable **arithmetic**. A foreign state or invalid
posterior h is `ErrStateVersion`. Classify every branch with `errors.Is`;
never match error-message text.

### Q13q.14. Are models shareable, and do returned arrays alias caller data?

`Model` is immutable and safe for independent concurrent read-only inference
with stable inputs. `State` is a copied value; the caller must serialize
updates of its own mutable state variable. Input return slices are not
mutated or retained, but concurrent mutation during a call is unsupported.
`Filter`/forecast slices are detached; `FitResult.Clone()` deep-copies the
mutable diagnostics while sharing the safe immutable model. A successful
`Step` and `ForecastNextVariance` avoid result-slice allocations.

### Q13q.15. What do I need before using a GJR estimate for collateral or VaR?

Validate a continuous return sampling frequency and mask-aware price
conversion; retain raw source timestamps; choose the economic shock law and
parameter start explicitly; check `err==nil` and convergence; compare with a
symmetric GARCH and fixed-volatility baseline on **strictly held-out** data;
inspect standardized residual dynamics and tails; assess parameter stability
and forecast calibration; obtain independent model-risk approval. GJR's
conditional variance is one input to a risk framework, not its release gate.

**Further reading:** [`GJRGARCH.md`](GJRGARCH.md),
[`gjrgarch/doc.go`](../gjrgarch/doc.go),
[`gjrgarch/example_test.go`](../gjrgarch/example_test.go) and
[`examples/gjrgarch/main.go`](../examples/gjrgarch/main.go).

---

## 13r. ARIMA FAQ: Conditional Gaussian Fit, Schur Constraints and Anchored Forecasts

### Q13r.1. When is ARIMA preferable to ACF, GJR-GARCH, GBM or HMM?

Use `arima` for an explicitly chosen fixed-order **conditional mean model**
for a single, uniformly sampled scalar series, with future level predictions
and marginal Gaussian innovation variances. `acf` measures retrospective
fixed-window correlations; it does not fit an ARIMA model. `gjrgarch` models
causal conditional **variance** of returns, including asymmetric shocks;
ARIMA's single innovation variance is constant under the fitted model.
`gbm` and `jumpdiffusion` generate different assumed price-path laws;
`hmm` models latent regime probabilities. Do not turn any of these into a
joint pipeline without stating the additional modeling assumption.

### Q13r.2. Which data and orders are allowed?

`Order{AR:p,Difference:d,MA:q}` requires `0<=p,q<=8`, `0<=d<=2`, with
`Order{}` an ordinary white-noise model. All data must be finite and in
ascending **sample** order. Time units, exact spacing and missing dates are
not encoded by `[]float64`: validate those separately (for example via
`series.Validate`) and reject/segment gaps before using ARIMA.

After differencing, `N=n-d`, the conditioning span is `b=max(p,q)`, and
`EffectiveN=N-b`. Fitting requires the **strict** threshold
`EffectiveN > p+q+I(IncludeDrift)+1`. A separately constructed model can
forecast from `n>=d+b+1` levels even when that is too short to estimate
all of its coefficients from scratch.

### Q13r.3. Does `Drift` mean unconditional level growth?

No. It is the literal intercept `c` in
`y[t]=c+sum(phi_i*y[t-i])+epsilon[t]+sum(theta_j*epsilon[t-j])`, where
`y=Delta^d x`. In stationary differenced coordinates,
`E[y]=c/(1-sum(phi))`. Thus an AR(1) with `phi=0.5,c=1` has stationary
expected increment `2`, **not** `1`, when `d=1`. With `d=2`, a nonzero
intercept can represent an average acceleration, not a constant trend.

`FitWithoutDrift` fixes `c=0`; `FitWithDrift` fits it. `New` does neither:
it validates explicitly supplied `Parameters.Drift` without an inclusion flag.

### Q13r.4. Why does ARIMA not have an exact Kalman log likelihood?

This package implements **conditional Gaussian** likelihood with
`b=max(p,q)`. It sets `epsilon[t]=0` for `t<b`, uses those samples only
as conditioning lag values, and forms SSE solely from `t>=b`.
`InnovationVariance=SSE/EffectiveN` is the profiled MLE denominator, not
`n`, `n-1` or a sample-variance degrees-of-freedom correction.
The reported objective is the **negative total** conditional Gaussian
log likelihood; `FitResult.LogLikelihood` has the opposite sign.

Exact diffuse state-space/Kalman MLE would use a different likelihood
initialization and is **not** a hidden fallback. Comparisons with
statsmodels/R require matching this conditioning, MA sign and drift
convention first.

### Q13r.5. Why does a constant series fail instead of producing zero variance?

Exactly zero effective SSE would require `log(0)` in a profiled Gaussian
likelihood and a nonpositive innovation variance. This version has no
separate deterministic zero-noise model. It returns `ErrSingular` instead
of inventing a variance floor or marking an undefined objective as success.
For caller-supplied `New` parameters, `InnovationVariance` must be finite
and strictly positive.

### Q13r.6. What happens to an AR/MA root at exactly one?

`New` checks the AR polynomial `1-sum(phi_i*z^i)` and MA polynomial
`1+sum(theta_j*z^j)` using a Schur step-down recurrence. Every reflection
coefficient must satisfy **strictly** `abs(k)<1`; equality is invalid.
For AR(1), `phi=1` gives `ErrNonStationary`; for MA(1), `theta=1` gives
`ErrNonInvertible`. The solver uses `k=tanh(u)` but rejects a candidate
rounded to exactly +/-1 rather than silently clipping it.
The MA check uses **negative** theta as its stable-polynomial coefficients;
reversing this sign would change the model.

### Q13r.7. Where do the integration anchors come from?

From the original observed `history` passed to *this particular* Forecast
call. For `d=1`, the first predicted level adds the predicted difference
to `history[n-1]`. For `d=2`, first accumulate against the last **observed
slope** `history[n-1]-history[n-2]`, then against the last actual level.

```text
history = [1, 3, 6], d=2, future second differences = [0, 0, 0]
last level = 6; last slope = 6-3 = 3
future levels = [9, 12, 15]
```

Do not initialize either anchor at zero. Do not pass pre-differenced data
to a model that still specifies `Difference>0`.

### Q13r.8. How is the MA part forecast when future errors are unknown?

`Forecast` recomputes historical conditional residuals using the same
burn-in as `Fit`. A historical innovation at index `t` can enter a future
MA term while its lag remains within `q`. **All future innovations are set
to conditional mean zero**, not recycled from the most recently observed
error. For `theta=+0.5`, `history=[0,2]` and `b=1`, we obtain
`epsilon[0]=0`, `epsilon[1]=2`, and the first two predicted means `[1,0]`.
The sign of the first forecast is a useful independent test.

### Q13r.9. Why are integrated forecast variances increasing?

The stationary ARMA impulse response satisfies `psi[0]=1` and
`psi[j]=theta[j]+sum(phi_i*psi[j-i])`; apply one cumulative sum for `d=1`
or two for `d=2` to obtain `Psi[j]`. The marginal horizon-h variance is
`sigmaSquared*sum_{j=0}^{h-1}Psi[j]^2`. For a random walk with
`sigmaSquared=2`, the first three values are `[2,4,6]`; for a twice
integrated unit-variance white noise they are `[1,5,14]`.

These are **innovation-only marginal** uncertainties. There is no
`H x H` forecast covariance, estimated-parameter uncertainty, joint
simultaneous-horizon band or tail-risk calibration guarantee.

### Q13r.10. Which convenience wrapper should I select?

| Purpose                                      | Canonical facade                                    |
|----------------------------------------------|-----------------------------------------------------|
| Fixed `(p,d,q)` and custom solver controls   | `arima.Fit(ctx, values, arima.Options{...})`        |
| Explicit differenced-space intercept         | `arima.FitWithDrift(ctx, values, order)`            |
| Explicitly zero intercept                    | `arima.FitWithoutDrift(ctx, values, order)`         |
| Stationary ARMA with `d=0`                   | `arima.FitARMA(ctx, values, p, q, includeDrift)`    |
| Integrated white noise (with optional drift) | `arima.FitIntegrated(ctx, values, d, includeDrift)` |
| One-step level **and variance**              | `model.ForecastNext(ctx, history)`                  |
| Several horizons                             | `model.Forecast(ctx, history, horizon)`             |

Wrappers preserve the result type and error classes; none invents an
alternative likelihood or silently chooses a model order.

### Q13r.11. What does `ErrNotConverged` publish?

A **last accepted** `FitResult` with `Diagnostics.Converged=false` and
`ErrNotConverged`. It may be inspected for model development, but is not a
successful deployment candidate. `Diagnostics.Iterations` counts accepted
L-BFGS steps, and `Objective[0]` is the initialized loss (later values are
accepted losses only). A failed search does not publish its rejected trial.
A pre-canceled or subsequently canceled context returns `nil,ctx.Err()`;
there is no cancellation partial result.

Use:

```go
fit, err := arima.Fit(ctx, observed, opts)
if errors.Is(err, arima.ErrNotConverged) {
    // fit can be a diagnostic candidate; do not deploy it automatically.
    return
}
if err != nil || fit == nil || !fit.Diagnostics.Converged {
    return
}
// The application still needs a holdout and model-approval policy.
```

This snippet assumes `ctx`, `observed` and `opts` are already defined;
for a complete runnable pipeline use [`ARIMA.md`](ARIMA.md).

### Q13r.12. What is the error precedence?

For `Fit`: context, order/solver options, nil input, empty input,
effective sample length, then nonfinite input and numerical evaluation.
For `Forecast`: context, initialized model and positive horizon, nil/empty
history, minimum length, nonfinite history, allocation admission, then
computation. For `New`: order/parameter lengths, finite scalars, positive
variance, finite coefficient arrays, then Schur stability/invertibility.

A nil context returns `ErrInvalidOption`, while an already canceled context
returns `ctx.Err()` **before** other validation. Use `errors.Is`; no failure
is diagnosed through substrings in error text. `ErrNonFinite` describes
invalid **caller data**; `ErrOverflow` describes invalid **computed data**.

### Q13r.13. Are returned arrays detached and is the model thread-safe?

Yes. `New` owns copies of input coefficient slices. `Model.Parameters()`
returns new AR/MA slices; returned forecast means/variances and accepted
objective histories are detached. A `Model` is immutable and independent
Forecast calls can execute concurrently with stable caller-owned inputs.
The caller may **not** mutate a history slice during a Forecast call.

`FitResult.IsNil()` and `ForecastResult.IsNil()` test **pointer presence
only**, including when a typed-nil pointer is stored in `core.Nilable`.
They do not test convergence or field completeness.

### Q13r.14. Is `Model.ID()` proof of source-data integrity?

No. It is a deterministic SHA-256 **content identifier** of the model
method/version, order, canonical binary64 parameters, drift and variance.
`+0` and `-0` have identical canonical floating encoding. It does not
include the original training feed, timestamp ledger, data permissions,
authorization or external checkpoint content. Authenticate externally and
keep a separate model registry/source-lineage record. A nil or zero-value
model returns an empty ID; a zero-value model cannot forecast.

### Q13r.15. Can I append future observations and still claim prefix causality?

A forecast at a fixed origin reads only the history supplied to that call.
To evaluate a chronological holdout, fit on a **frozen training prefix**;
for each next observation, call `ForecastNext` on the prefix ending just
*before* it, record the prediction, and only then reveal the observation.
Using the whole dataset before splitting or conditioning on a future
observation invalidates a predictive backtest. A new origin intentionally
has more information and therefore may produce a different forecast.

### Q13r.16. Is the forecast computation linear in horizon memory?

Yes. The horizon impulse buffer and marginal variance arrays require
`O(H)` space. Historical residual reconstruction adds `O(N)` workspace,
and total forecast time is `O((N+H)*(p+q+d+1))`. The implementation
never forms a dense `H x H` covariance matrix. Avoid interpreting this as
a guaranteed memory limit for arbitrarily large H; allocation-size guards
cannot predict operating-system OOM.

### Q13r.17. What should I check before approving an ARIMA baseline?

Establish cadence and missing-data policy; justify `p,d,q` and drift;
require `err==nil` **and** convergence; test that residual behavior is
adequately modeled on independent chronological holdout data; compare to
a naive persistence or seasonal baseline; inspect stability across
recalibration windows and extreme numerical regimes; evaluate uncertainty
coverage **without** treating innovation variance as full parameter risk;
and record an externally authenticated model/data revision. Application
thresholds, economic consequences and approvals remain outside ARIMA.

**Further reading:** [`ARIMA.md`](ARIMA.md),
[`arima/doc.go`](../arima/doc.go),
[`arima/example_test.go`](../arima/example_test.go),
[`examples/arima/main.go`](../examples/arima/main.go),
[`ACF.md`](ACF.md) and [`SERIES.md`](SERIES.md).

---

## 13s. LSTM FAQ: IFGO Inference, Masking and Streaming Replay

### Q13s.1. When should I use `lstm`, and what is it actually predicting?

Use `lstm` to compute **deterministic hidden features** from a single frozen,
previously trained, one-direction LSTM layer. The package does **not** train
an LSTM, supply its parameters, decode event categories, estimate failure
probabilities, or predict future values by itself. A supervised output head,
loss function, calibration, feature normalization, and decision thresholds
are separate application responsibilities.

Choose the simpler contract when possible: `arima` for an explicit fixed-order
conditional mean forecast, `gjrgarch` for asymmetric conditional variance,
`hmm` for typed probabilistic regimes, `pca` for static orthogonal projection,
or `dtw` to align two existing sequences. LSTM is appropriate only when a
trained recurrent representation has established value on a chronological
holdout under the correct feature schema.

### Q13s.2. Which gate order can be passed to `Import`?

**Exactly `i, f, g, o`**, the PyTorch single-layer gate order:

```text
Input [4H,D]       Recurrent [4H,H]      Bias [4H]
rows [0,H)     -> input gate  i -> sigmoid
rows [H,2H)    -> forget gate f -> sigmoid
rows [2H,3H)   -> candidate   g -> tanh
rows [3H,4H)   -> output gate o -> sigmoid
```

Some exported neural formats use another order. Do not pass them unchanged:
a shape-compatible gate permutation can produce plausible but mathematically
wrong hidden vectors with no validation error. The external exporter must
translate and independently verify its exact ABI before numerical import.

### Q13s.3. Why is there only one `Bias`, not PyTorch's two bias vectors?

`Weights.Bias` holds an **already-combined** `4H`-vector:

`Bias[k] = bias_ih_l0[k] + bias_hh_l0[k]`.

`lstm` does not perform this translation, distinguish two bias sources, or
import PyTorch checkpoint objects. Even when one source bias is zero, supply
the full nonnil `Bias` array. A missing Bias is `ErrNilInput`; an incorrectly
sized nonnil Bias is `ErrShape`. The combined vector is copied and checked
for NaN/Inf during `Import`.

### Q13s.4. What is the correct shape of `Tensor3` and `State`?

Input `Tensor3` is **batch-first** `[B,T,D]`, and its flat address is
`(b*T+t)*D+d`. A returned sequence has `[B,T,H]`. `Final.Hidden` and
`Final.Cell` each have shape `[B,H]`, tagged with `Final.ModelID`.

PyTorch's single-layer batched initial/final hidden-state tensors have
`[1,B,H]` shape; a trusted external adapter must remove the leading layer
dimension when building a Go `State`. It must not transpose batch and hidden
axes. `B>=1`, `D>=1`, and `T>=0`; feature count `D` must equal the imported
model's width. A zero-time tensor is valid and can have nil Data.

### Q13s.5. What does `lengths=nil` mean? Can I infer lengths from zero-filled padding?

`lengths=nil` means **all `T` steps in every batch row are valid**. A nonnil
mask has exactly `B` entries, and each `lengths[b]` is an integer in
`[0,T]`. The package never guesses a valid prefix from zero-valued samples:
zero is a legitimate temperature deviation, vibration component, or other
physical feature after calibration.

Supply explicit lengths when rows end early. A mask with wrong length or
out-of-range entries returns `ErrInvalidOption`, not a best-effort result.

### Q13s.6. Why are padded `Result.Sequence` cells zero instead of the last hidden state?

Padding is a **non-observation**, not an extra recurrent transition.
For `t>=lengths[b]`, both `h` and `c` remain unchanged; the optional
`Sequence.Data` entries are exactly positive zero. Repeating the last hidden
value would falsely turn missing time slots into valid step outputs.

Use `Result.Final.Hidden` and `.Cell` when you need the state after the last
valid step. Do not read the last row of `Sequence` as the final state when a
batch contains padding. `ReturnSequence=false` gives an entirely zero-valued
`Tensor3`, while Final is still published.

### Q13s.7. Why does a NaN in masked padding reject the entire batch?

Input validation covers **every element of `input.Data`**, not just active
steps. The mask changes the recurrence, not the validity of the supplied
payload. An invalid masked sample must not remain silently stored in an
otherwise successful inference record. `NaN`, `+Inf`, or `-Inf` in any
position returns `nil, ErrNonFinite`, even when that row's length is zero.

Fix data ingestion and source-quality gates upstream; do not use masking as
an excuse to accept corrupt serialized numeric payloads.

### Q13s.8. What happens when `T=0` or one row has `lengths[b]=0`?

No gate step runs for that row. Both hidden and cell states remain equal to
its supplied initial state (or all zeros when `initial==nil`). For `T=0`,
`Sequence` has zero data length even when `ReturnSequence=true`, with tensor
shape `[B,0,H]`. For positive `T` but a zero-length row, its entire sequence
slice is exactly zero.

The **full input** still must pass dimensional, mask, and numeric validation.
A zero-time call does not authorize a wrong ModelID or malformed state.

### Q13s.9. How do I split a session into chunks without changing its result?

Pass a successfully completed result to the next call:

```go
first, err := model.InferFinal(ctx, firstChunk, firstLocalLengths, nil)
if err != nil { return err }
next, err := model.InferChunk(ctx, secondChunk, secondLocalLengths, first, lstm.Options{})
if err != nil { return err }
_ = next.Final
```

This excerpt assumes variables are in scope and the containing function
returns `error`. `InferChunk` requires nonnil `previous` and uses **both**
`previous.Final.Hidden` and `previous.Final.Cell`; it does not reset the cell.

For global valid length `L`, chunk start `s`, and chunk width `C`, calculate
`localLength = min(C, max(0, L-s))` for **each** batch row. Reusing global
lengths on later chunks is incorrect. The same Go kernel with identical
ordered valid observations, ModelID and initial state yields **bit-identical**
full versus chunked hidden/cell values; independent PyTorch backends are a
separate numerical comparison, not this replay contract.

### Q13s.10. What makes a checkpoint compatible with a Model?

`State.ModelID` must exactly equal `model.ID()`. The fingerprint hashes a
version tag, D/H dimensions, and all normalized binary64 weight and bias
words in canonical row-major order. Imported negative zero is normalized to
positive zero for storage and hashing. A different ID yields
`ErrStateVersion` **before** the initial matrices are inspected.

A SHA-256 identity is not a digital signature or a provenance guarantee.
For production restart, authenticate the complete model artifact, fixed
feature schema, normalization policy, checkpoint content and external
source-offset mapping. Neither `ModelID` nor `InferChunk` authenticates a
transport stream.

### Q13s.11. Does an overflowing affine combination simply saturate sigmoid to 1?

**No.** `sigmoid(1000)` may legitimately round to one for finite input;
`sigmoid(+Inf)` following an overflowed `Wx+Uh+b` is not a correct
inference step. The kernel uses fixed-order compensated reductions and
checks exceptional high-magnitude sums with exact rational arithmetic.
An exact magnitude above the largest finite binary64 returns `ErrOverflow`,
including a near-limit sum that would round back to a finite MaxFloat64.

Do not catch and replace `ErrOverflow` with zeros, 1s, or a fabricated
probability. Reassess the feature scales, imported weights and model
conditioning, and reject the failed inference atomically.

### Q13s.12. Is the package guaranteed bitwise identical to PyTorch CPU double?

No universal cross-framework bitwise guarantee is made. The weight layout
and mathematical recurrence match PyTorch's supported single-layer IFGO
configuration; different CPU kernels, summation orders and transcendental
implementations can legitimately differ in low bits. The included
`pytorch_golden_test.go` anchors a **specific synthetic reference** from
PyTorch **2.10.0+cpu**, `torch.float64`, single-layer, unidirectional,
`batch_first=true`, `.eval()`, deterministic algorithms enabled and MKLDNN
disabled, with **absolute tolerance `1e-12`** on a small conditioned batch.

A version/hash mismatch is not fixed by silently increasing test tolerance.
Keep the source fixture, architecture and backend recorded, and treat new
exports as a reviewed compatibility event.

### Q13s.13. Why is there no Python export script or parser in the Go package?

No Python runtime, checkpoint loader or PyTorch package is needed to run
or test `lstm`. The external reference data are **frozen Go literals**
with recorded generation environment; a separately implemented scalar
oracle in `package lstm_test` independently checks LSTM arithmetic.

To deploy another trained model, create a trusted, explicit adapter outside
the package that validates architecture, converts gate order/biases and
provides finite `matrix.Matrix` and `[]float64` values. Do **not** deserialize
arbitrary pickle data inside numerical inference or treat a file name as
weight identity.

### Q13s.14. What do `ErrNilInput`, `ErrShape`, `ErrInvalidOption`, `ErrStateVersion`, `ErrNonFinite`, and `ErrOverflow` mean?

| Sentinel           | Typical failure                                                                                   |
|:-------------------|:--------------------------------------------------------------------------------------------------|
| `ErrNilInput`      | Nil model, absent required weights/initial matrices, or nil Data with positive expected count     |
| `ErrShape`         | Wrong `[B,T,D]`/`[B,H]`/weights geometry, wrong Data length, or overflowing element-count product |
| `ErrInvalidOption` | Nil context, invalid mask size or per-row valid lengths                                           |
| `ErrStateVersion`  | Initial state has the wrong canonical ModelID                                                     |
| `ErrNonFinite`     | A supplied weight, hidden/cell value, or input element (including padding) is NaN/Inf             |
| `ErrOverflow`      | Finite arithmetic operands produce an unrepresentable affine/cell/hidden outcome                  |

Callers must use `errors.Is`, not string comparisons. Context cancellation
returns `ctx.Err()`; a pre-canceled context is checked before input geometry.
Initial ModelID/shape/finiteness checks precede the full Data finite scan.
A failure returns **nil Result**; there is no usable partial batch or stream.
Lower-level `Matrix.At` read errors retain their own identity alongside
`ErrShape` via `errors.Join`.

### Q13s.15. What does `(*Result).IsNil()` actually guarantee?

It implements the `core.Nilable` convention using only `return r==nil`.
It is safe on a typed-nil `*Result`, and does **not** inspect whether
`Final.Hidden`/`.Cell` exist or whether numerical output is valid. Infer
errors return nil result, but an independently constructed `&lstm.Result{}`
will report `IsNil()==false` and is not a successful computation artifact.
Always inspect the returned `error` before using results.

### Q13s.16. Can one imported Model serve many goroutines?

Yes: `Import` deeply copies the weights, and `Infer` allocates its own
workspace per call, without mutable global caches or locks. Concurrent
read-only inference against stable, caller-owned inputs and initial states
is supported. It is **not** safe for the caller to write an input slice,
state matrix, or previous `Result.Final` while a goroutine is reading it.
Successful results are detached; later caller-side mutations cannot change
the Model or future calls. A model fingerprint does not implement a lock.

### Q13s.17. Why is `ReturnSequence=false` not a zero-allocation call?

It avoids the optional `[B,T,H]` output buffer but still creates detached
`[B,H]` hidden and cell matrices and per-call working state. There are no
regular **per-timestep** heap allocations in the recurrent kernel; there
can be allocations per invocation, and exceptional exact-rational overflow
rescue also allocates. Import copies weights once via `Matrix.At`; the hot
loop reads private contiguous slices instead of dispatching matrix methods.

Measure your own workload with `go test -run '^$' -bench BenchmarkInfer -benchmem ./lstm`;
use realistic B/T/D/H and compare both sequence policies. Avoid concluding
"zero allocations" from a single favorable benchmark shape.

### Q13s.18. How should I connect LSTM to `series`, `stats` and other models?

Validate timestamp spacing and choose complete windows using `series`
*before* forming the row-major tensor. Fit feature scaling on the training
split only (for example with the immutable `stats` standardizer) and reuse
**exactly that** feature order, units and transformation on holdout/live
observations. LSTM accepts only finite numeric vectors; it does not inspect
`series.Valid`, auto-handle missing grid slots, or fit a Standardizer.

Keep the downstream trained head and alarm policy external. Compare against
ARIMA/HMM/linear baselines on chronological holdout, and audit false alarms
before taking automated production action. For the complete bridge, read
[`LSTM.md`](LSTM.md), the [Tutorial LSTM walkthrough](TUTORIAL.md#11t-lstm-foundations-frozen-ifgo-inference-and-masked-streaming)
and [`lstm/example_test.go`](../lstm/example_test.go).

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

### Q16.7. Is the `montecarlo` worker pool deterministic and safe with a shared Sampler?

The *mathematical full-run result* is deterministic for a fixed platform/backend **only when the sampler is itself deterministic and concurrency-safe**. Streams are private to logical paths and the coordinator alone updates moments; neither property automatically makes an arbitrary shared model object safe. Do not mutate model parameters during a run or share one mutable RNG across paths. Partial-result length after cancellation is timing-dependent. See [Monte Carlo FAQ](#13n-monte-carlo-faq-replay-scheduling-numerics-and-partial-results).

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

### Recipe L. Investigate fixed-window retry or vibration periodicity

Validate the sampling cadence and choose two **non-overlapping or otherwise
explicitly defined** analysis windows. After handling any `series` validity
masks, call `acf.ComputeDefault` for the same scientific lag horizon in each
window. Compare coefficients **and** `PairCounts`, and persist the estimator
convention and window boundaries with the incident record. A repeat peak is
an investigation lead, not a root-cause diagnosis or a p-value. For
physical-zero raw signals use `acf.ComputeUncentered` deliberately; for
lag-adjusted covariance ratios use `acf.ComputeLagAdjusted` without clipping.
Never substitute `LagAdjusted` into a Toeplitz PSD-sensitive solver.

See [`ACF.md`](ACF.md) and the reproducible `go run ./examples/acf` program.

---

### Recipe M. Replay an index-only block-bootstrap audit

```go
plan := bootstrap.Plan{
    Kind: bootstrap.StationaryBlocks,
    Length: trainingRows,
    BlockLength: 8,
    Replicates: 200,
    Seed: 0,
}
result, err := bootstrap.SampleIndices(ctx, trainingRows, plan)
if err != nil {
    // A non-nil partial Result can contain completed rows; do not silently
    // accept those rows as the requested full 200-replicate ensemble.
    return err
}
// Persist plan, trainingRows and result.RNG before gathering domain values.
_ = result
```

This snippet belongs inside a function returning `error` with `ctx` and
`trainingRows` supplied by the caller. `bootstrap` does not read observations,
select block length or estimate confidence intervals. See
[`BOOTSTRAP.md`](BOOTSTRAP.md) for exact mathematics and memory limits.

---

### Recipe N. Build a chronological supervised table from MACD without false warmup rows

Use a historical training block with known **future labels**, but construct
features only from closes available at each feature index. The last training
feature index must be earlier than the last training close because its
next-close label is unavailable otherwise. Never perform this label lookup
in a live `Push` callback.

```go
package main

import (
    "fmt"

    "github.com/lvlath/go/macd"
)

func main() {
    closes := make([]float64, 48)
    for index := range closes {
        closes[index] = 80 + 0.08*float64(index) + 0.17*float64(index%7-3)
    }
    result, err := macd.ComputeDefault(closes)
    if err != nil {
        fmt.Println("MACD:", err)
        return
    }

    // A fixed chronological training block ends before close index 40.
    const trainingEnd = 40
    type trainingRow struct {
        index int
        histogram float64
        nextCloseHigher bool
    }
    var rows []trainingRow
    for index := 0; index+1 < trainingEnd; index++ {
        if !result.Histogram.Valid[index] {
            continue
        }
        rows = append(rows, trainingRow{
            index: index,
            histogram: result.Histogram.Values[index],
            nextCloseHigher: closes[index+1] > closes[index],
        })
    }
    if len(rows) == 0 {
        fmt.Println("no ready training rows")
        return
    }
    fmt.Printf("training rows=%d first=%d last=%d\n",
        len(rows), rows[0].index, rows[len(rows)-1].index)
    // Output:
    // training rows=6 first=33 last=38
}
```

The `nextCloseHigher` label is intentionally historical and **not** a MACD
prediction. The remaining closes (indices 40–47) are outside the training
block; an application may reserve them for holdout after fixing its entire
feature and label policy. The package itself does not split datasets, fit
models, produce probabilities or promise positive trading performance.

See [`MACD.md`](MACD.md) and the streaming/failover example in the
[Tutorial](TUTORIAL.md#11k-macd-foundations-sma-seeded-ema-signed-momentum-and-checkpoint-replay).

---

### Recipe O. Replay Monte Carlo percentiles without publishing the raw paths

Use a fixed, versioned `Sampler` and a stable seed. Call `montecarlo.RunWithQuantiles(ctx, montecarlo.Plan{Paths: 100_000, Seed: seed, Workers: 4}, []float64{0.95, 0.99}, sampler)` for complete empirical quantiles with `Samples==nil`. A private `Paths*d` buffer is still budgeted and allocated; `MaxSamplesBytes=0` selects the 64 MiB optional-buffer ceiling. Inspect `err`, `Complete`, `Quantiles` and `QuantileLevels` before consuming any quantile. Persist the model revision/parameters and `Result.RNG`, not just `Seed`; never interpret the Monte Carlo SE as certainty about the stochastic model. For a two-pass VaR/Expected Shortfall pipeline, see [`../montecarlo/example_test.go`](../montecarlo/example_test.go) and [`MONTECARLO.md`](MONTECARLO.md).

---

### Recipe P. Replay a complete GBM path without silent time conversion

Construct `gbm.Parameters` with an explicitly chosen `TimeUnit`, and `gbm.TimeGrid{Values: elapsedTimes, Unit: params.Unit}` whose first value is zero and whose entries are strictly increasing. For an independent synthetic scenario call `gbm.SimulateSeeded(ctx,params,grid,seed,pathID)`; check **both** `err==nil` and `result.Complete` before extracting the horizon price. If a `montecarlo.Sampler` needs the model, call `gbm.Simulate(ctx,params,grid,random)` with the **driver-supplied** path stream; forward errors instead of reducing an incomplete prefix. Changing from annual model coefficients to elapsed seconds requires external `mu/C` and `sigma/sqrt(C)` conversion, not matching enum labels without changing numeric values. See [`GBM.md`](GBM.md) and [`../examples/gbm/main.go`](../examples/gbm/main.go).

---

### Recipe Q. Reproduce a jump-aware collateral scenario without a truncated payoff

Construct `jumpdiffusion.Parameters{Diffusion: gbm.Parameters{...}, Intensity: lambda,
LogJumpMean: m, LogJumpStdDev: s, Convention: jumpdiffusion.Compensated}` with
explicit matched time units. Call `jumpdiffusion.SimulateSeeded(ctx,p,grid,seed,pathID)`
for a single reproducible scenario. Check **both** `err` and `Result.Complete`
before computing terminal exposure or observed-grid drawdown. In a Monte Carlo
sampler, use the supplied path-owned RNG rather than this seeded wrapper.
Use `errors.Is(err, jumpdiffusion.ErrBudget)` or another sentinel to
classify a failure; never count an incomplete prefix as an ordinary successful
scenario. See [`JUMPDIFFUSION.md`](JUMPDIFFUSION.md).

---

### Recipe R. Continue a frozen telemetry LSTM without interpreting padding as data

Import verified, trusted IFGO weights once and keep the resulting immutable
`Model`. Before the first call, validate the source timestamp grid and
normalize every feature with frozen training-time parameters. For the first
packet use `model.InferFinal(ctx, chunk, localLengths, nil)`; for each following
packet use `model.InferChunk(ctx, chunk, localLengths, previous, lstm.Options{})`
and retain its successful `Final`. Convert global valid session lengths to
**chunk-local** masks. Do not update state on padding or infer valid lengths
from numeric zeros. Check `err` before reading any result, persist an
externally authenticated source cursor and artifact, and treat `ModelID` as
parameter identity, not an authorization token. The recurrent feature is
**not** a calibrated equipment fault probability.

See [`LSTM.md`](LSTM.md#6-streaming-chunks-and-exact-replay) and the
[canonical executable telemetry example](../lstm/example_test.go).

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

