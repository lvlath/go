<!--
  SPDX-License-Identifier: AGPL-3.0-only
  Copyright (C) 2025-2026 katalvlaran

  lvlath - FAQ & Tips

  Purpose:
    This document is the practical troubleshooting and best-practices guide for
    lvlath users and contributors. It explains common mistakes, package-selection
    decisions, deterministic behavior, error classification, graph/matrix semantics,
    testing practices, benchmark pitfalls, and documentation synchronization rules.
    It is aligned with the lvlath Universal Engineering Standard (UES).

  Contract status:
    This file is guidance, not a replacement for package doc.go contracts. When in
    doubt, package doc.go and current source code are authoritative. This document
    must not mention fictional APIs or stronger guarantees than the implemented
    packages provide.

  Scope:
    - Package selection and algorithm-choice guidance.
    - Common troubleshooting for core, bfs, dfs, dijkstra, matrix, flow, dtw, tsp,
      gridgraph, builder, and examples.
    - Testing, benchmarking, documentation, and contribution best practices.

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
11. [Builder and gridgraph tips](#10-builder-and-gridgraph-tips)
12. [Errors and diagnostics](#11-errors-and-diagnostics)
13. [Determinism and concurrency](#12-determinism-and-concurrency)
14. [Testing tips](#13-testing-tips)
15. [Benchmark tips](#14-benchmark-tips)
16. [Documentation tips](#15-documentation-tips)
17. [Quick recipes](#16-quick-recipes)

---

## 1. Choosing the right package

### Q1.1. I have a graph. Which package should I start with?

Start with the question, not the algorithm name.

| Question                                              | Use                   | Reason                                                        |
|:------------------------------------------------------|:----------------------|:--------------------------------------------------------------|
| “How many hops away is this vertex?”                  | `bfs`                 | BFS minimizes edge count.                                     |
| “Which vertices are in the same weak island?”         | `bfs.Components`      | Weak membership ignores direction for component grouping.     |
| “What is a deterministic deep traversal?”             | `dfs`                 | DFS gives post-order and forest-style structure.              |
| “Does this graph contain a cycle?”                    | `dfs.DetectCycles`    | Returns deterministic witness cycles.                         |
| “What order can I execute a DAG?”                     | `dfs.TopologicalSort` | Topological order is a directed acyclic graph contract.       |
| “What is the cheapest route by weight?”               | `dijkstra`            | Dijkstra minimizes non-negative total cost.                   |
| “What is the cheapest connected backbone?”            | `mst`                 | MST solves minimum connected spanning structure.              |
| “What is the maximum throughput?”                     | `flow`                | Flow solves capacity-constrained source-to-sink movement.     |
| “How do I turn graph topology into numeric features?” | `matrix`              | Adjacency, incidence, degree vector, metric closure, algebra. |
| “How do I model a grid map?”                          | `gridgraph`           | Generates lattice topology instead of manual wiring.          |
| “How do I build repeatable fixtures?”                 | `builder`             | Deterministic graph/data generation.                          |
| “How do I align two time series?”                     | `dtw`                 | Dynamic Time Warping handles local timing drift.              |
| “How do I visit all cities in a route?”               | `tsp`                 | Tour construction/improvement/exact-small-instance routing.   |

### Q1.2. How do I avoid using the wrong algorithm?

Keep this mental table:

```text
minimize hops             -> BFS
inspect depth structure   -> DFS
prove/locate cycle        -> DFS cycle witness
order DAG tasks           -> DFS topological sort
minimize weighted cost    -> Dijkstra
connect all cheaply       -> MST
maximize throughput       -> Flow
compute all-pairs costs   -> Matrix metric closure
align time-warped signals -> DTW
build a closed tour       -> TSP
```

The most common mistake is using BFS on a weighted routing problem. BFS answers “fewest edges,” not “cheapest total weight.”

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

### Q6.8. Why does covariance require at least two rows for positive features?

Sample covariance divides by `rows - 1`. With positive feature count and one row, sample covariance is not defined under that formula.

For zero features (`cols == 0`), a valid `0×0` covariance is a structural result.

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

DTW aligns timing; it does not magically fix feature scale. For scalar series, normalize when amplitude scale is not the signal you want to measure. For `AlignMatrix`, normalize columns/features before alignment when dimensions have different units.

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

## 11. Builder and gridgraph tips

### Q11.1. When should I use `builder`?

Use `builder` when you need reproducible graph shapes for:

* examples;
* tests;
* benchmarks;
* algorithm demonstrations;
* golden outputs.

Builder-generated fixtures should encode meaningful topology, not hide arbitrary setup.

### Q11.2. When should I use `gridgraph`?

Use `gridgraph` when your domain is naturally a lattice:

* grid maps;
* pathfinding demos;
* obstacle experiments;
* 4-neighbor/8-neighbor movement;
* teaching BFS/Dijkstra visually.

Do not manually wire large grids unless the example is specifically about low-level graph construction.

### Q11.3. How should random fixtures be handled?

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

## 12. Errors and diagnostics

### Q12.1. Why are error sentinels so important?

They let callers branch safely while still preserving helpful context.

```go
if errors.Is(err, matrix.ErrDimensionMismatch) {
// stable protocol branch
}
```

Error strings are for humans. They may change without changing the protocol.

### Q12.2. How should wrapped errors behave?

Wrapping should preserve sentinel identity.

Good:

```go
return fmt.Errorf("build adjacency: %w", matrix.ErrUnknownVertex)
```

If an adapter needs to expose both an algorithm-level error and an underlying root cause, it should preserve both through the package’s chosen wrapping/join strategy.

### Q12.3. How do I debug a failing algorithm call?

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

## 13. Determinism and concurrency

### Q13.1. What does “deterministic” mean in lvlath?

It means the observable result is stable for the same input state, options, and package version.

Examples:

* BFS order follows the documented neighbor order.
* DFS result order means documented finish order.
* Dijkstra equal-cost witnesses follow documented tie/strict-improvement rules.
* Matrix adapters use stable vertex/edge order and fixed loop order.
* Components are sorted according to their contract.

### Q13.2. Does determinism mean every mathematically valid answer is identical across packages?

No. It means each package publishes its own deterministic choice among valid answers. For example, one shortest path may be selected as a witness, not all possible shortest paths.

### Q13.3. Is `core.Graph` safe for concurrency?

`core` is designed with internal locking for graph storage and controlled mutations. Algorithm reproducibility still requires stable topology while an algorithm runs unless the package explicitly documents snapshot isolation.

### Q13.4. Is `matrix.Dense` safe for concurrency?

Concurrent read-only access is safe when no goroutine mutates the matrix. Concurrent writes or read/write races require external synchronization. Views share storage with their base matrix.

---

## 14. Testing tips

### Q14.1. What should a good package test suite contain?

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

### Q14.2. Should tests use `testify`?

No. Use the standard library and package-specific helpers. This keeps failure messages precise and avoids dependency weight.

### Q14.3. When is `reflect.DeepEqual` acceptable?

Use explicit structural helpers when the domain matters. For example, prefer path comparison helpers, matrix cell checks, sorted domain checks, and numeric tolerance helpers.

`reflect.DeepEqual` is acceptable only when it genuinely expresses the whole contract and produces understandable failure output.

### Q14.4. How should numeric tests compare floats?

Use tolerances for computations that can accumulate floating-point error.

```go
if math.Abs(got-want) > tol {
t.Fatalf("got %.12g, want %.12g", got, want)
}
```

For matrix-wide comparison, use the package’s tolerant comparison surface where appropriate.

### Q14.5. How should tests handle map output?

Do not print or compare maps through unstable iteration. Extract keys, sort them, then compare or print stable output.

### Q14.6. How should cancellation be tested?

Use deterministic hooks or contexts. Avoid timeouts as the primary mechanism in examples or unit tests.

### Q14.7. What is a regression anchor?

A regression anchor is a test named after the bug or contract it protects.

Example:

```go
func TestAdjacencyMatrix_ToGraphPreservesZeroWeightEdgeByDefaultInInfEncoding(t *testing.T) {
// ...
}
```

A bug fix without a regression test is incomplete unless there is a clear reason no test can be written.

---

## 15. Benchmark tips

### Q15.1. What should be inside the benchmark loop?

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

### Q15.2. What is a benchmark regime?

A regime is the real shape or pressure you are measuring.

Examples:

* sparse chain;
* dense graph;
* mixed-edge endpoint resolution;
* path tracking on/off;
* cutoff policy overhead;
* flow level graph rebuild pressure;
* matrix fast path vs generic fallback;
* DTW full matrix vs rolling memory.

A benchmark name should reveal the regime.

### Q15.3. Why must setup errors be checked in benchmarks?

Because graph policy can reject edges. If setup silently fails, the benchmark may measure a smaller or different graph than intended.

This is one of the easiest ways to produce meaningless performance numbers.

### Q15.4. Should benchmarks use random data?

Only with fixed seed and documented shape. Randomness without reproducibility is not acceptable for regression-sensitive performance work.

---

## 16. Documentation tips

### Q16.1. Which file should explain what?

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

### Q16.2. What is documentation drift?

Documentation drift happens when docs, examples, tests, and code describe different behavior.

Common drift patterns:

* docs mention removed APIs;
* examples use old constructors;
* README claims Go version or license incorrectly;
* package docs promise stronger guarantees than tests protect;
* tests lock behavior that contradicts mathematical correctness.

Fix drift immediately. Do not leave “almost correct” docs in place.

### Q16.3. Should docs include formulas?

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

### Q16.4. Should docs include ASCII diagrams?

Yes, when they clarify topology, memory layout, residual flow, traversal order, or matrix encoding.

A good ASCII diagram should make the next paragraph easier to understand.

---

## 17. Quick recipes

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

