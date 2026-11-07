<!--
  SPDX-License-Identifier: AGPL-3.0-only
  Copyright (C) 2025-2026 katalvlaran

  lvlath - Repository Tutorial

  Purpose:
    This tutorial is the first educational entry point for lvlath. It introduces
    fundamental graph and matrix ideas, explains how lvlath maps mathematical
    algorithms into deterministic Go packages, and helps readers move from basic
    graph concepts to package-specific documentation. It does not duplicate every
    package specification; instead it teaches the mental model, core entities,
    options, result surfaces, and engineering discipline needed before reading
    docs/{PACKAGE}.md and {package}/doc.go files.

  License:
    lvlath is licensed under AGPL-3.0-only. See ../LICENSE.
-->

# lvlath Tutorial

> **hard math on an easy level**

Welcome to **lvlath**, a deterministic Go library for graph algorithms, dense graph algebra, flows, routing, tours, fixtures, and time-series alignment.

This tutorial is the beginning of the learning path. It introduces the objects and ideas that most packages share: graphs, matrices, options, result artifacts, error contracts, and deterministic algorithm execution.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Documentation Map](#2-documentation-map)
3. [Installation](#3-installation)
4. [Mental Model: How lvlath Is Built](#4-mental-model-how-lvlath-is-built)
5. [Graph Foundations](#5-graph-foundations)
6. [First Rich Graph: Build, Inspect, Traverse](#6-first-rich-graph-build-inspect-traverse)
7. [Graph Algorithms on the Same Topology](#7-graph-algorithms-on-the-same-topology)
8. [Matrix Foundations](#8-matrix-foundations)
9. [Graph-to-Matrix Example: Routing, Closure, and Spectral-Ready Data](#9-graph-to-matrix-example-routing-closure-and-spectral-ready-data)
10. [Algorithm Selection Guide](#10-algorithm-selection-guide)
11. [Professional Best Practices](#11-professional-best-practices)
12. [Where To Go Next](#12-where-to-go-next)

---

## 1. Introduction

### What is lvlath?

`lvlath` is a deterministic Go library for graph algorithms, dense graph algebra, routing, network optimization, time-series alignment, tour planning, and reproducible algorithm fixtures.

It is built around a simple engineering idea:

> Algorithms should publish results you can explain, test, reproduce, and safely compose.

The library covers:

* `core` graph modeling with deterministic topology and explicit capabilities;
* `bfs` hop-distance traversal and weak components;
* `dfs` finish-order traversal, cycle witnesses, and topological sorting;
* `dijkstra` weighted full-SSSP and four target-search methods (Dijkstra, bidirectional Dijkstra, consistent A*, balanced A*), original EdgeIDs, and exact target-cutoff policy;
* `mst` exact Prim/Kruskal MST/MSF, KMB original-edge graph Steiner approximation, and closed/fixed-end edge-covering walks;
* `flow` max-flow / residual-network analysis;
* `matrix` dense graph algebra: adjacency, incidence, metric closure, statistics, factorization, sanitation;
* `dtw` deterministic Dynamic Time Warping, bounded outcomes, and globally exact fixed-length squared-DTW top-K window search;
* `tsp` matrix-backed tour optimization with exact, approximate, matching, and local-search regimes;
* `gridgraph` generated lattice topology;
* `builder` deterministic fixtures for examples, tests, and benchmarks.

```text
Mathematical object        Package surface             Contract idea
────────────────────────────────────────────────────────────────────────────
G = (V,E)                  *core.Graph                 deterministic topology
hop layers                 *bfs.Result                 depth / parent / visited
DFS forest                 *dfs.Result                 finish order, not discovery
weighted SSSP              *dijkstra.Result            all known distances / optional predecessor
weighted target            *dijkstra.PathResult        finite cost / +Inf / original EdgeIDs
full-vertex backbone       *mst.Result                 strict MST or explicit MSF
terminal-only network      *mst.SteinerResult          original EdgeIDs, KMB approximation
cover edges of one tree    *mst.WalkResult             repeated passage IDs, closed/fixed endpoints
flow network               residual *core.Graph        capacity state after flow
dense operator             *matrix.Dense               row-major numeric artifact
warping alignment          *dtw.Result                 distance, reachability, path
closed tour                *tsp.Result                 cost, exactness, timeout, ratio
```

### Why lvlath?

Because real algorithm work fails at the edges, not in the textbook recurrence.

A shortest-path formula is easy. A reliable Go package must also answer:

* Is traversal order stable enough for CI and golden tests?
* Is this graph really directed, weighted, loop-enabled, or multi-edge capable?
* Is a missing target different from a known but unreachable target?
* Is a zero edge a real zero-cost edge or an absent edge?
* In which numeric domain is `+Inf` being used: graph storage, runtime policy, dense representation, or algorithm result?
* Is a disconnected MST request an error or an explicit forest request?
* Am I connecting every vertex (MST), only required terminals (graph Steiner), or covering every edge of an existing tree (SteinerWalk)?
* Is this cost a unique installed-network total or the distance of a route with repeated traversals?
* Is a DTW path recoverable under the selected memory mode?
* Is a TSP result exact, approximate, locally improved, or merely a timed-out incumbent?
* Does the returned result own its data, or does it alias the input?

`lvlath` treats those questions as the API, not as afterthoughts.

### What makes lvlath different?

1. **Contract-first packages**

   Each package defines its mathematical domain, result artifact, error law, determinism law, and non-goals. A function is not “just an implementation”; it is a public contract.

2. **Deterministic result surfaces**

   Stable graph order, deterministic tie-breaks, strict improvement laws, fixed DP traversal, deterministic matching/local-search policy, and detached result artifacts make outputs reproducible.

3. **Explicit modeling instead of hidden fallback**

   `mst` does not silently turn strict MST into a forest. `tsp` does not silently replace Blossom matching with Greedy matching. `dtw` does not pretend rolling-row memory can reconstruct a path.

4. **Numeric domains are explicit**

   `core.Graph` edge weights are finite. Individual algorithms may impose stricter
   input subsets. `+Inf` may still be valid as an unbounded option, an unreachable
   result, or a dense absence sentinel. Arithmetic overflow is a separate error and
   must not be silently converted into another domain state.

5. **Graph and matrix workflows compose**

   `core` owns topology. `matrix` turns topology into dense artifacts. `dijkstra`, `mst`, `flow`, `tsp`, and `dtw` solve different mathematical questions without collapsing their result meanings.

---

## 2. Documentation Map

Read these files as a learning path. `doc.go` files define implemented package contracts; `docs/*.md` files teach the theory, diagrams, pitfalls, and recipes.

| Step | Document                                   | What you learn                                                                                                             |
|:-----|:-------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|
| 1    | [`CORE.md`](CORE.md)                       | Graph capabilities, deterministic topology, ownership, locks, graph invariants.                                            |
| 2    | [`BFS.md`](BFS.md)                         | Hop-distance traversal, `Result`, components, filters, partial-result behavior.                                            |
| 3    | [`DFS.md`](DFS.md)                         | Finish order, DFS forest, deterministic cycle witnesses, topological sort.                                                 |
| 4    | [`DIJKSTRA.md`](DIJKSTRA.md)               | Full SSSP vs target Dijkstra/A*, consistent/balanced potentials, original EdgeIDs, exact cutoff, walls and error contracts. |
| 5    | [`MST.md`](MST.md)                         | Exact Prim/Kruskal MST/MSF, KMB Steiner networks, closed/fixed-end tree walks, original IDs and separate numeric contracts. |
| 6    | [`FLOW.md`](FLOW.md)                       | Capacity networks, residual graph state, Ford-Fulkerson, Edmonds-Karp, Dinic.                                              |
| 7    | [`MATRICES.md`](MATRICES.md)               | Dense storage, graph adapters, zero-weight policy, metric closure, statistics, numeric sanitation.                         |
| 8    | [`DTW.md`](DTW.md)                         | `Align`, `AlignCostMatrix`, `AlignMatrix`, `AlignBounded`, exact `Search`, numeric and path artifacts.                     |
| 8a   | [`DTW_SEARCH.md`](DTW_SEARCH.md)            | Fixed-length exact top-K, certified LB_Keogh, workers, budgets, and partial-result law.                                  |
| 9    | [`TSP.md`](TSP.md)                         | Matrix-backed tours, exact regimes, Christofides, Blossom matching, local search, timeouts.                                |
| 10   | [`GRID_GRAPH.md`](GRID_GRAPH.md)           | Lattice topology for traversal/routing demos and pathfinding fixtures.                                                     |
| 11   | [`FAQ_&_TIPS.md`](FAQ_%26_TIPS.md)         | Troubleshooting, package selection, testing/benchmark/documentation pitfalls.                                              |
| 12   | [`lvlath_UES.md`](lvlath_UES.md)           | Universal engineering standard for package quality and contribution discipline.                                            |
| 13   | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Branching, tests, linting, examples, docs, benchmark, and PR workflow.                                                     |

Suggested route:

```text
New to lvlath:
  CORE -> BFS -> DFS -> DIJKSTRA -> MATRICES

Working on optimization:
  DIJKSTRA -> MST -> FLOW -> TSP

Working on numeric/time-series workflows:
  MATRICES -> DTW -> TSP

Contributing:
  lvlath_UES -> CONTRIBUTING -> package doc.go -> docs/{PACKAGE}.md
```

---

## 3. Installation

```bash
go get github.com/lvlath/go@latest
```

Requires Go 1.23.4 or newer. Pure Go. No cgo. No external runtime dependencies.

---

## 4. Mental Model: How lvlath Is Built

Think of `lvlath` as a contract stack.

```text
                                    ┌──────────────────────────────────┐
                                    │            core.Graph            │
                                    │ deterministic topology substrate │
                                    └────────────────┬─────────────────┘
                                                     │
       ┌──────────────────────┬──────────────────────┼──────────────────────┬──────────────────────┐
       │                      │                      │                      │                      │
 traversal structure     weighted routing      connectivity design      capacity network       dense artifacts
       │                      │                      │                      │                      │
  ┌────▼────┐       ┌─────────▼──────────┐      ┌────▼────┐            ┌────▼────┐            ┌────▼──────┐
  │ bfs     │       │ dijkstra           │      │ mst     │            │ flow    │            │ matrix    │
  │ dfs     │       │ finite weights     │      │ MST/ST  │            │ residual│            │ graph     │
  └────┬────┘       │ +Inf result/policy │      └────┬────┘            └────┬────┘            │ algebra   │
       │            └─────────┬──────────┘           │                      │                 └────┬──────┘
       │                      │                      │                      │                      │
       └──────────────────────┴──────────────────────┴──────────────────────┴────────────┬─────────┘
                                                                                         │
                                                                             ┌───────────▼───────────┐
                                                                             │ tsp                   │
                                                                             │ matrix-backed tours   │
                                                                             └───────────────────────┘

                         ┌──────────────────────────┐       ┌──────────────────────────┐
                         │ dtw                      │       │ builder / gridgraph      │
                         │ temporal alignment       │       │ reproducible input worlds│
                         └──────────────────────────┘       └──────────────────────────┘
```

### Repository laws

1. **Topology is modeled before algorithms run**

   Directedness, weights, loops, multi-edges, and mixed edges are graph capabilities. They are not loose comments around a map.

2. **Different algorithms answer different questions**

   BFS minimizes hops. Dijkstra minimizes non-negative route cost. MST minimizes selected backbone cost. Flow maximizes feasible throughput. DTW minimizes temporal alignment cost. TSP minimizes closed-tour cost.

3. **Result artifacts carry meaning**

   Do not reduce results to “a slice and a number.” A result may contain reachability, path tracking, exactness, approximation ratio, timeout state, component roots, residual capacity, matrix policy, or detached IDs. In `mst`, `Result` spans all vertices, `SteinerResult` selects one original-edge terminal network, and `WalkResult` records repeated travel passages. Their costs are not interchangeable.

4. **Policy is explicit and local to the run**

   Filters, thresholds, forest mode, metric closure, local search, matching algorithm, DTW memory mode, and time limits are explicit options. Hidden fallback is not a feature.

5. **Numeric values need domain context**

   `0`, `+Inf`, `NaN`, negative weights, and missing edges are interpreted differently by different packages. Correct code respects the package contract instead of forcing one universal meaning.

---

## 5. Graph Foundations

A graph is:

$$ G = (V, E) $$

where:

* `V` is a set of vertices;
* `E` is a set of edges between vertices.

In `lvlath/core`, vertices are identified by string IDs, and edges carry direction, weight, ID, and policy-dependent behavior.

### Core capabilities

| Capability           | What it controls                           | Typical use                                              |
|:---------------------|:-------------------------------------------|:---------------------------------------------------------|
| `WithDirected(true)` | Default orientation of edges.              | Dependency graphs, one-way roads, call graphs.           |
| `WithWeighted()`     | Non-zero `float64` edge values.            | Cost, distance, latency, capacity.                       |
| `WithLoops()`        | Self-edges such as `A -> A`.               | State machines, roundabouts, internal transitions.       |
| `WithMultiEdges()`   | Parallel edges between the same endpoints. | Road + rail, multiple network links, redundant channels. |
| `WithMixedEdges()`   | Per-edge direction overrides.              | City maps with one-way streets and two-way roads.        |

### Storage intuition

```text
+------------------------------------------------------------------+
| core.Graph                                                       |
|------------------------------------------------------------------|
| vertices:       map[string]*Vertex                               |
| edges:          map[string]*Edge                                 |
| adjacency:      map[fromID]map[toID]map[edgeID]struct{}          |
| config:         directed / weighted / loops / multi / mixed      |
| locks:          vertex/config lock + edge/adjacency lock         |
+------------------------------------------------------------------+
            │                         │
            │                         └── edge IDs preserve identity
            │
            └── vertex IDs are the stable public coordinate system
```

### Directed vs undirected

```text
Directed edge A -> B:

   [A] ──▶ [B]

Storage intent:
   A can reach B through this edge.
   B does not automatically reach A.

Undirected edge A -- B:

   [A] ◀──▶ [B]

Storage intent:
   Traversal can move both ways.
```

### Weight is not just decoration

```text
Unweighted graph:
  all edges have weight 0;
  BFS/DFS are natural tools.

Weighted graph:
  edge values carry cost/capacity/distance;
  Dijkstra, MST, flow, matrix closure, and TSP become meaningful.
```

---

## 6. First Rich Graph: Build, Inspect, Traverse

The first example should be visually useful, not a toy line. We use a 10-vertex directed service graph with a central loop, two branches, and a storage tail.

```text
                         ┌────────────┐
                         │  incident  │
                         └──────┬─────┘
                                │
                                ▼
                         ┌────────────┐
                         │  gateway   │
                         └───┬────┬───┘
                             │    │
                ┌────────────┘    └──────────┐
                ▼                            ▼
            ┌────────┐                  ┌────────┐
            │  auth  │                  │  api   │
            └───┬────┘                  └────┬───┘
                │                            │
                ▼                            ▼
          ┌──────────┐                 ┌──────────┐
          │ profile  │◀──────┐         │ search   │
          └─────┬────┘       │         └─────┬────┘
                │            │               │
                ▼            │               ▼
           ┌────────┐        │         ┌──────────┐
           │   db   │        └─────────│  index   │
           └────┬───┘                  └─────┬────┘
                │                            │
                ▼                            ▼
           ┌────────┐                  ┌──────────┐
           │ backup │                  │  cache   │
           └────────┘                  └──────────┘

Features:
  - main dependency chain: incident -> gateway -> auth -> profile -> db -> backup
  - parallel branch: gateway -> api -> search -> index -> cache
  - cross-loop: index -> profile
  - 10 vertices, visible depth layers, one meaningful feedback-style structural join
```

```go
package main

import (
	"context"
	"fmt"
	"sort"

	"github.com/lvlath/go/bfs"
	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dfs"
)

func main() {
	g, err := core.NewGraph(core.WithDirected(true))
	if err != nil {
		fmt.Println(err)
		return
	}

	// Static tutorial data. Ignoring AddEdge errors is acceptable only here because
	// the topology is predetermined and the graph is unweighted. Check errors in
	// production and tests.
	_, _ = g.AddEdge("incident", "gateway", 0)
	_, _ = g.AddEdge("gateway", "auth", 0)
	_, _ = g.AddEdge("gateway", "api", 0)
	_, _ = g.AddEdge("auth", "profile", 0)
	_, _ = g.AddEdge("profile", "db", 0)
	_, _ = g.AddEdge("db", "backup", 0)
	_, _ = g.AddEdge("api", "search", 0)
	_, _ = g.AddEdge("search", "index", 0)
	_, _ = g.AddEdge("index", "cache", 0)
	_, _ = g.AddEdge("index", "profile", 0)

	// BFS: hop-distance layers from incident.
	bfsResult, err := bfs.BFS(g, "incident")
	if err != nil {
		fmt.Println(err)
		return
	}

	layers := map[int][]string{}
	for id, depth := range bfsResult.Depth {
		layers[depth] = append(layers[depth], id)
	}
	for depth := 0; depth <= 5; depth++ {
		sort.Strings(layers[depth])
		fmt.Printf("depth %d: %v\n", depth, layers[depth])
	}

	path, err := bfsResult.PathTo("backup")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("path-to-backup:", path)

	// DFS: finish order, useful for dependency reasoning.
	dfsResult, err := dfs.DFS(g, "incident")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("dfs-finish-order:", dfsResult.Order)

	// Components: direction ignored for weak membership.
	components, err := bfs.Components(context.Background(), g)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("weak-components:", components.Count)
}
```

### What to notice

* `BFS.Depth` is a shortest-hop certificate.
* `BFS.PathTo` reconstructs one deterministic shortest path from `StartID`.
* `DFS.Order` is finish order, not discovery order.
* `Components` answers weak connectivity and should not be confused with strongly connected components.

### Layer visualization

```text
Depth 0:   incident
              │
Depth 1:   gateway
           /      \
Depth 2: auth     api
          │        │
Depth 3: profile  search
          │        │
Depth 4: db       index
          │       /    \
Depth 5: backup  cache  profile(already discovered earlier)
```

---

## 7. Graph Algorithms on the Same Topology

The fastest way to misuse graph libraries is to treat every algorithm as “pathfinding.” `lvlath` keeps the objective visible.

### BFS: hop-distance and frontier waves

Use BFS when every edge has equal cost and you care about layers.

```text
Objective:
  minimize number of edges

Result meaning:
  Depth[v] = shortest hop count from start
  Parent[v] = deterministic shortest-hop witness tree
```

Good questions:

* “Which services are within 2 hops?”
* “What is the blast radius wave by wave?”
* “Which weak components exist?”

### DFS: structure, finish order, cycles

Use DFS when completion structure matters.

```text
Objective:
  inspect depth-first structure

Important law:
  dfs.Result.Order = finish order, not discovery order
```

Good questions:

* “Can this dependency graph be topologically sorted?”
* “Where is a deterministic cycle witness?”
* “What is the post-order structure of this graph?”

### Dijkstra and consistent A*: non-negative weighted routing

Use the **full SSSP** method when one source needs distances to many destinations, and a **target method** when only one original-cost route is required.

```text
Full Dijkstra:     Dijkstra(...)         -> Result, all known distances
One target:        Search(...)           -> PathResult, original EdgeIDs
Two frontiers:     Bidirectional(...)    -> certified meeting and lower bound
Consistent A*:    AStar(...)            -> frozen target estimates, no reopening
Balanced A*:      BidirectionalAStar(...) -> two frozen directional estimates
```

All target methods operate on one `core.Clone`, retain original edge identities and have stricter exact-binary64 cost/key requirements than the legacy SSSP runner. A target result contains one complete witness, **not** a map of finalized distances.

Good questions:

* “What are the shortest costs from ingress to every service?” — full `Dijkstra`.
* “What single route uses these particular original multigraph links?” — `Search` / `Bidirectional`.
* “Can my grid Manhattan estimate legally accelerate settlement?” — consistent `AStar`.
* “Do I have reliable, directionally consistent estimates from both endpoints?” — balanced `BidirectionalAStar`.

Run the [three standalone shortest-path scenarios](DIJKSTRA.md#59-compilable-scenarios-and-their-runner-tests) to compare public APIs and exact witness semantics.

### MST, graph Steiner, and tree-covering walks: three different objectives

Use exact MST for **every** graph vertex; KMB graph Steiner for an **explicit required subset**; and `SteinerWalk` when the goal is to **traverse every edge of one validated tree**.

```text
Strict MST, Prim/Kruskal:
  connect every original vertex; Result.TotalWeight counts selected unique links.

Explicit MSF:
  one exact MST per connected component; WithForest() must be requested.

SteinerTree (KMB approximation):
  connect only requested terminals through original graph vertices/edge IDs.
  Every original edge weight must be finite and nonnegative, even on loops/islands.
  SteinerResult.TotalWeight counts unique selected original edges, not closure links.

SteinerWalk:
  closed start=end: cover every tree edge twice.
  fixed start!=end: cover unique spine edges once, off-spine edges twice.
  WalkResult.Distance sums actual repeated original-ID passages.
```

On a simple hub with `AX=BX=CX=1` and direct terminal connections costing 3, `SteinerTree(g, []string{"A", "B", "C"})` chooses original `AX,BX,CX`, total **3**, while the terminal closure MST totals **4**. A closed walk from A costs **6**, a fixed-end walk from A to B costs **4**. Neither is a proof of globally optimal original-graph routing.

The KMB kernel uses source-grouped public Dijkstra SSSP queries and reuses Kruskal selection twice, then prunes unused nonterminal leaves. `SteinerRecomputePaths` favors bounded live predecessor history; `SteinerRetainPaths` retains O(kV) source history for fewer repeated SSSP queries. The walk validates the caller-owned tree independently, including correct `TotalWeight` in original Edge.ID order.

Good questions:

* “What is the cheapest cable, fiber, or relay backbone spanning **all** nodes?” — MST.
* “Which required sensor or chiplet pins can share original relay links?” — `SteinerTree`.
* “How do I inspect every link of the selected tree and finish at a specified endpoint?” — `SteinerWalk`.
* “Are disconnected islands allowed when they contain no required terminal?” — yes, for graph Steiner.

### Flow: maximum feasible throughput

Use flow when weights are capacities.

```text
Objective:
  maximize source-to-sink flow

Residual graph:
  records remaining forward capacity and possible reverse cancellation.
```

Good questions:

* “How much traffic can reach storage?”
* “Which cut limits throughput?”
* “What residual capacity remains after routing maximum flow?”

### TSP: closed tour over a complete cost model

Use TSP when the result must visit every required stop exactly once and return to start.

```text
Objective:
  minimize closed Hamiltonian tour cost

Input model:
  complete finite matrix after optional metric closure
```

Good questions:

* “What is the best inspection/delivery route?”
* “Do I need exact optimality or a heuristic/local-search result?”
* “Can I claim a Christofides ratio, or did I choose weaker matching?”

### DTW: temporal alignment, not graph routing

Use DTW when two sequences describe the same process at different speeds.

```text
Objective:
  minimize accumulated alignment cost through a time-time grid

Path meaning:
  diagonal = both advance
  vertical/horizontal = one side stretches
```

Good questions:

* “Do these two vibration signatures match despite phase drift?”
* “Which phase was stretched?”
* “Can I afford path recovery, or do I only need distance?”
* “Which complete fixed-length windows are my exact top-K matches?” — use `dtw.Search`.
* “Is a score within a known inclusive limit?” — use `dtw.AlignBounded`.

---

## 8. Matrix Foundations

Graphs describe relationships. Matrices make those relationships computable.

`lvlath/matrix` is the dense layer for:

* adjacency and incidence artifacts;
* metric closure / all-pairs shortest paths;
* degree vectors and reductions;
* row-major dense algebra;
* LU, QR, inverse, symmetric eigen decomposition;
* covariance, correlation, row normalization;
* numeric sanitation and tolerant comparison.

### Matrix core capabilities

| Capability                 | What it controls                        | Why it matters                                                    |
|:---------------------------|:----------------------------------------|:------------------------------------------------------------------|
| `Dense` row-major storage  | One flat `[]float64` buffer.            | Predictable memory layout and fast loops.                         |
| `Matrix` interface         | `Rows`, `Cols`, `At`, `Set`, `Clone`.   | Kernels can accept dense and wrapped/fallback inputs.             |
| `NewPreparedDense` options | Finite/`+Inf` admissibility.            | Numeric policy is explicit.                                       |
| `View`                     | Shared no-copy submatrix.               | Block assembly and write-through workflows.                       |
| `Induced`                  | Copied submatrix by selected rows/cols. | Independent result artifact.                                      |
| Graph adapters             | Adjacency, incidence, metric closure.   | Topology becomes numeric data without losing policy.              |
| Sanitizers                 | `ReplaceInfNaN`, `Clip`.                | Raw telemetry becomes safe for statistics/algebra.                |
| Comparators                | `AllClose`.                             | Tests compare numeric results by tolerance, not fragile equality. |

### Storage intuition

A dense matrix with `R` rows and `C` columns stores `(i,j)` at:

$$ offset(i,j)=i \times C+j $$

```text
Logical 3×4 matrix:

      c0   c1   c2   c3
r0  [ a00  a01  a02  a03 ]
r1  [ a10  a11  a12  a13 ]
r2  [ a20  a21  a22  a23 ]

Physical row-major storage:

[ a00 a01 a02 a03 | a10 a11 a12 a13 | a20 a21 a22 a23 ]
  row 0 contiguous   row 1 contiguous   row 2 contiguous
```

### Graph-to-matrix semantics

```text
Adjacency matrix: V × V
  cell(i,j) means relation from vertex i to vertex j.

Incidence matrix: V × E
  column(k) means structural contribution of edge k.

Metric closure: V × V
  cell(i,j) means shortest-path distance, not original edge existence.
```

### Zero and infinity law

```text
Classic adjacency:
  0 may mean “no edge”.

Zero-preserving weighted adjacency:
  finite 0 may mean “real zero-cost edge”;
  +Inf means absence.

Metric closure:
  finite value means shortest distance;
  +Inf means unreachable.

Sanitation/statistics:
  NaN and invalid Inf values must be cleaned or rejected before analysis.
```

### Matrix as bridge to other packages

`matrix` is not isolated from graph algorithms:

* Dijkstra-style route outputs can become dense distance surfaces.
* TSP consumes matrix-backed complete cost models.
* Spectral analysis starts from adjacency or symmetrized adjacency.
* Flow/cut analysis can use incidence-style structural thinking.
* DTW can align matrix rows as multivariate time steps through `dtw.AlignMatrix`.

---

## 9. Graph-to-Matrix Example: Incident Routing, Closure, and Spectral-Ready Data

This example models a small incident-response service network.

The story is intentionally realistic:

* `edge` receives user traffic;
* requests split into authentication and API paths;
* `profile`, `search`, and `index` represent service dependencies;
* `db -> backup` is a real zero-cost replication edge;
* `logs` and `archive` preserve operational evidence;
* the same topology can feed routing, dense matrices, spectral analysis, MST backbone planning, terminal-only Steiner network selection, tree-edge covering walks, flow capacity analysis, TSP inspection tours, and DTW telemetry comparison.

The goal of this section is not to run every package in one oversized code block. The goal is to show the **graph-to-matrix foundation** that later packages can reuse.

```text
Incident-response service graph
weights = latency / operational cost

                   
        ┌──────────2────────────▶ auth ─────1────▶ profile ─────2────▶ db
        │                           └─4─┐                               │
        │                               │                               │0
        │                               ▼                               ▼
      edge ────3────▶ api ─────4────▶ search ─────3────▶ logs ───2──▶ backup
        │              │                 │
        │              │2                │1
        │              ▼                 ▼
        └─────6────▶ cache             index ─────5────▶ archive

Important semantics:
  - `db -> backup = 0` is a real zero-cost replication link.
  - missing edges are not zero-cost edges.
  - metric closure answers shortest reachable distance, not original topology.
  - symmetrized adjacency is a derived operator for spectral inspection.
  
```

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
)

func main() {
	g, err := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	if err != nil {
		fmt.Println(err)
		return
	}

	// Static tutorial data only. In production and tests, check AddEdge errors.
	_, _ = g.AddEdge("edge", "auth", 2)
	_, _ = g.AddEdge("edge", "api", 3)
	_, _ = g.AddEdge("edge", "cache", 6)
	_, _ = g.AddEdge("auth", "profile", 1)
	_, _ = g.AddEdge("auth", "search", 4)
	_, _ = g.AddEdge("api", "search", 4)
	_, _ = g.AddEdge("api", "cache", 2)
	_, _ = g.AddEdge("search", "index", 1)
	_, _ = g.AddEdge("search", "logs", 3)
	_, _ = g.AddEdge("index", "archive", 5)
	_, _ = g.AddEdge("logs", "backup", 2)
	_, _ = g.AddEdge("profile", "db", 2)
	_, _ = g.AddEdge("db", "backup", 0)

	opts, err := matrix.NewMatrixOptions(
		matrix.WithDirected(),
		matrix.WithWeighted(),
		matrix.WithPreserveZeroWeights(),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	// NewAdjacencyMatrix gives the graph a deterministic dense coordinate system:
	// string vertex IDs become stable matrix row/column indices.
	adj, err := matrix.NewAdjacencyMatrix(g, opts)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("vertices:", len(adj.VertexIndex))

	// DegreeVector is a compact topology signal. In this weighted directed example,
	// it summarizes outgoing structure under the adjacency policy.
	degree, err := adj.DegreeVector()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("degree-vector-len:", len(degree))

	// PreserveZeroWeights is essential here: db -> backup is a real edge with cost 0,
	// not an absent relation. Missing edges are represented by +Inf under this policy.
	db := adj.VertexIndex["db"]
	backup := adj.VertexIndex["backup"]
	replicationCost, err := adj.Mat.At(db, backup)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("replication-cost:", replicationCost)

	// BuildMetricClosure converts direct-edge costs into all-pairs shortest distances.
	// This is a distance artifact. It must not be confused with original topology.
	closure, err := matrix.BuildMetricClosure(g, opts)
	if err != nil {
		fmt.Println(err)
		return
	}

	edge := adj.VertexIndex["edge"]
	distToBackup, err := closure.Mat.At(edge, backup)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("distance-edge-backup:", distToBackup)

	// Symmetrize builds an undirected-style operator from directed adjacency.
	// That derived operator is suitable for symmetric spectral routines.
	sym, err := matrix.Symmetrize(adj.Mat)
	if err != nil {
		fmt.Println(err)
		return
	}

	_, _, err = matrix.EigenSym(sym, 1e-10, 200)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("spectral-ready:", true)

	// Output:
	// vertices: 11
	// degree-vector-len: 11
	// replication-cost: 0
	// distance-edge-backup: 5
	// Eigen: ValidateSymmetric: diagonal[0,0]=+Inf: matrix: NaN or Inf encountered
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-TUTORIAL_Graph_to_Matrix_Example-blue?logo=go)](https://go.dev/play/p/NkN7e4dOt_D)

### What this example proves

* `core.NewGraph` models topology and capability: directed + weighted.
* `matrix.NewMatrixOptions` is the graph-to-matrix policy boundary.
* `matrix.NewAdjacencyMatrix` gives string vertices deterministic dense coordinates.
* `matrix.WithPreserveZeroWeights` keeps a real zero-cost edge distinct from absence.
* `matrix.BuildMetricClosure` turns topology into shortest-distance data.
* `matrix.Symmetrize` prepares a derived operator for symmetric spectral analysis.
* `matrix.EigenSym` demonstrates that graph-derived matrices can become numeric-analysis inputs.

### How the same incident graph connects to other lvlath packages

| Package     | Incident-response use                                                                      |
|:------------|:-------------------------------------------------------------------------------------------|
| `bfs`       | Hop blast radius: “which services are within 2 calls of `edge`?”                           |
| `dfs`       | Dependency cycle audit before deployment.                                                  |
| `dijkstra`  | Cheapest failover route from ingress to storage or backup.                                 |
| `mst`       | Exact full-vertex MST/MSF, or KMB for required service terminals; walk covers all edges of a selected tree. |
| `flow`      | Maximum capacity from ingress tier to storage tier on a capacity version of this topology. |
| `matrix`    | Adjacency, incidence, closure, spectral features, statistics, sanitation.                  |
| `dtw`       | Compare incident telemetry signature against historical outages.                           |
| `tsp`       | Plan a closed inspection / maintenance route over a complete distance matrix.              |
| `gridgraph` | Model the same incident over a datacenter floor or network zone map.                       |
| `builder`   | Generate repeatable larger incident fixtures for tests and benchmarks.                     |

### Spectral interpretation

The eigen step is not “magic centrality.” It is a compact structural signal over a derived symmetric operator.

Use it as:

* a ranking hint for central services;
* a seed for graph clustering;
* a sanity check before heavier analytics;
* a feature source for dashboards or anomaly models.

Do not use it as:

* proof of causality;
* replacement for shortest-path or flow analysis;
* direct evidence that an edge exists in the original directed graph.

---

## 10. Algorithm Selection Guide

### If the input is a graph

| Problem                                | Package                  | Main output                                                                             |
|:---------------------------------------|:-------------------------|:----------------------------------------------------------------------------------------|
| Reachability by hop count              | `bfs`                    | BFS `Result` with depth, parent, visited/order state.                                   |
| Weak components                        | `bfs`                    | component artifact and deterministic membership.                                        |
| DFS forest / finish order              | `dfs`                    | DFS `Result` with finish-order semantics.                                               |
| Cycle witness                          | `dfs`                    | deterministic witness, not exhaustive enumeration.                                      |
| Topological execution order            | `dfs`                    | DAG order or cycle error.                                                               |
| Weighted route from one source         | `dijkstra.Dijkstra`      | all known distances, `+Inf` unreachable, optional predecessor tracking                |
| One weighted target / consistent A*    | `dijkstra.Search` family | one original-cost `PathResult` and ordered original `EdgeIDs`                         |
| Runtime weighted walls/cutoffs         | `dijkstra`               | `+Inf` publication without mutating topology.                                           |
| Strict connected backbone              | `mst.Kruskal` / `mst.Prim` | Exact full-vertex MST `Result`.                                                          |
| Component-wise backbones               | `mst` with forest policy | Explicit exact minimum spanning forest.                                                 |
| Connect only required graph terminals  | `mst.SteinerTree`        | KMB approximate original-edge `SteinerResult` on nonnegative graph.                     |
| Cover every edge in a selected tree    | `mst.SteinerWalk`        | Closed/fixed-end `WalkResult` with repeated original EdgeIDs and actual passage cost.   |
| Max throughput                         | `flow`                   | max-flow value and residual graph.                                                      |
| Grid/lattice topology                  | `gridgraph`              | generated graph world.                                                                  |
| Dense topology artifacts               | `matrix`                 | adjacency, incidence, degree vector, metric closure.                                    |
| Complete tour over graph-derived costs | `tsp.SolveGraph`         | TSP `Result` after graph-to-matrix adaptation.                                          |
| Reproducible fixture graph             | `builder`                | deterministic generated graph.                                                          |

### If the input is numeric data

| Problem                            | Package               | Main output                                   |
|:-----------------------------------|:----------------------|:----------------------------------------------|
| Dense algebra                      | `matrix`              | matrix result with shape/error policy.        |
| Numeric cleaning                   | `matrix`              | sanitized matrix.                             |
| Feature centering/normalization    | `matrix`              | transformed matrix plus metadata.             |
| Covariance/correlation             | `matrix`              | statistical matrix artifact.                  |
| Symmetric spectral analysis        | `matrix`              | eigenvalues/eigenvectors.                     |
| Scalar sequence alignment          | `dtw.Align`           | DTW `Result` with distance/reachability.      |
| Inclusive-limit scalar alignment    | `dtw.AlignBounded`    | Exact/above-limit/unreachable status.        |
| Globally exact window top-K         | `dtw.Search`          | Sorted fixed-length squared-DTW matches.     |
| Precomputed local-cost alignment   | `dtw.AlignCostMatrix` | DTW over caller-owned finite cost surface.    |
| Multivariate time-series alignment | `dtw.AlignMatrix`     | squared-L2 row-cost alignment result.         |
| Complete route matrix optimization | `tsp.SolveMatrix`     | tour, cost, exactness/timeout/ratio metadata. |

### If the goal is documentation or testing

| Goal                          | Package/file         |
|:------------------------------|:---------------------|
| Reproducible generated graphs | `builder`            |
| Grid pathfinding fixtures     | `gridgraph`          |
| Package API contract          | `{package}/doc.go`   |
| Theory, diagrams, recipes     | `docs/{PACKAGE}.md`  |
| Contribution discipline       | `CONTRIBUTING.md`    |
| Engineering standard          | `docs/lvlath_UES.md` |

---

## 11. Professional Best Practices

### 11.1. Start from the objective, not the function name

```text
min hops                          -> bfs
post-order / cycle / DAG order    -> dfs
min non-negative route cost       -> dijkstra
min acyclic full backbone cost    -> mst.Kruskal / mst.Prim
approx terminal-only backbone     -> mst.SteinerTree
cover all edges of a given tree   -> mst.SteinerWalk
max feasible throughput           -> flow
all-pairs distances               -> matrix metric closure
dense topology/statistics         -> matrix
time-warped sequence similarity   -> dtw
closed tour cost                  -> tsp
```

### 11.2. Model graph capabilities honestly

Do not enable options “just in case.” A directed weighted multi-edge graph is not more correct than a simple graph; it is a different domain model.

### 11.3. Keep topology stable while algorithms run

`core` protects storage. Algorithm meaning still assumes the topology is not being changed underneath the run. Clone, freeze, or synchronize if reproducibility matters.

### 11.4. Prefer policy over destructive graph edits

* Use filters, wall thresholds, max-distance cutoffs, forest mode, metric closure, matrix options, DTW windows, and TSP time limits instead of rewriting input data to simulate one run.
* Use Dijkstra finite wall thresholds for degraded links; do not encode a failed link as a `+Inf` graph-edge weight.

### 11.5. Preserve result meaning

Do not collapse rich result artifacts into anonymous tuples. Inspect the fields that matter:

* BFS: `Depth`, `Parent`, `Visited`, `Order`;
* DFS: finish `Order` and cycle/topology state;
* Dijkstra: known/unreachable/path-tracked states;
* MST/MSF: `Mode`, `ComponentCount`, `ComponentRoots`;
* Graph Steiner: selected `Terminals`, `Vertices`, original `Edges`, unique `TotalWeight`;
* Tree walk: ordered `Vertices` and repeated `EdgeIDs`, distinct `NetworkWeight` and `Distance`;
* Flow: residual graph;
* DTW: `Reachable`, `PathTracked`, `MemoryMode`;
* TSP: `Exact`, `Optimal`, `TimedOut`, `ApproximationRatio`.

### 11.6. Separate numeric domains explicitly

The same `float64` value can have different meanings in different package domains.

* `core.Graph` edge weights must be finite.
* Dijkstra accepts finite non-negative edge weights.
* Dijkstra may publish `+Inf` for a known unreachable vertex.
* Dijkstra options may use `+Inf` to mean “unlimited”.
* Matrix policies may use `+Inf` for absence or unreachable pairs.
* Arithmetic overflow must remain an error, not an unreachable sentinel.

Never transfer a sentinel from one domain into another implicitly.

### 11.7. Sanitize before statistics and spectral workflows

Raw telemetry can contain `NaN`, `Inf`, spikes, and unit-scale drift. Use sanitation and normalization before covariance, correlation, eigen analysis, or multivariate DTW.

### 11.8. Preserve zero-weight edges deliberately

A zero-cost replication or transfer edge is real topology. Use zero-preserving matrix policy when absence must be distinguishable from finite zero.

### 11.9. Be honest about exactness and guarantees

* Prim/Kruskal MST is exact for its full-vertex finite-weight domain.
* KMB graph Steiner is a polynomial-time approximation for finite nonnegative edges, not an exact optimality certificate.
* `SteinerWalk` is optimal for covering the given tree with fixed endpoints, not necessarily for routes allowed to leave it.
* Dijkstra is exact for non-negative single-source routing.
* Flow algorithms solve max-flow under capacity semantics.
* Held-Karp / completed Branch-and-Bound are exact TSP regimes.
* Branch-and-Bound with timeout may return a feasible incumbent, not proof.
* Christofides’ `1.5` ratio requires metric input and exact Blossom matching.
* 2-opt/3-opt are improvements, not optimality certificates.
* DTW distance is not generally a metric.

### 11.10. Make examples deterministic and inspectable

Avoid unstable map printing, time-flaky cancellation, hidden random seeds, and helper-heavy examples that obscure public calls. A good example shows:

```text
build input -> choose policy -> run algorithm -> inspect result fields
```

### 11.11. Benchmark the shape you claim

Sparse graph traversal, dense matrix kernels, flow residual pressure, DTW full-matrix mode, TSP exact search, and local search are different regimes. Benchmark them separately and name the regime clearly.


## 12. Where To Go Next

Use this tutorial as the entry point, then move into package-specific material:

> If you are modeling topology first:
>
>    [`CORE.md`](CORE.md) -> [`BFS.md`](BFS.md) -> [`DFS.md`](DFS.md) -> [`DIJKSTRA.md`](DIJKSTRA.md)

> If you are doing routing/optimization:
>
>   [`DIJKSTRA.md`](DIJKSTRA.md) -> [`MST.md`](MST.md) -> [`FLOW.md`](FLOW.md) -> [`TSP.md`](TSP.md)

> If you are doing numeric graph analytics:
>
>   [`CORE.md`](CORE.md) -> [`MATRICES.md`](MATRICES.md) -> examples/matrix_*.go

> If you are doing sequence comparison:
>
>   [`DTW.md`](DTW.md) -> [`DTW_SEARCH.md`](DTW_SEARCH.md) -> matrix sanitation/statistics sections

> If you are contributing:
>
>   [`lvlath_UES.md`](lvlath_UES.md) -> [`CONTRIBUTING.md`](../CONTRIBUTING.md) -> package doc.go -> tests/examples

---

**lvlath** teaches algorithms through contracts: every graph option, traversal field, numeric policy, and matrix cell should have a reason you can explain.

