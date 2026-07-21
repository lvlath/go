<!--
  SPDX-License-Identifier: AGPL-3.0-only
  Copyright (C) 2025-2026 katalvlaran

  lvlath - Repository README

  Purpose:
    This README is the public face of lvlath. It presents the library as a
    deterministic, contract-driven Go toolkit for graph algorithms, flow networks,
    dense graph algebra, routing, time-series alignment, and reproducible
    algorithm engineering. It should give a new reader immediate confidence in
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

---

# lvlath

**lvlath** is a pure-Go algorithm engineering library for graphs, flows, dense graph algebra, routing, time-series alignment, and reproducible test fixtures.

It is built for people who need algorithmic tools they can explain in a code review, trust in CI, compose across packages, and debug from deterministic artifacts instead of guessing why an answer changed.

**The core promise:**

> Same graph, same options, same algorithm - same result surface, same witness semantics, same failure class.

`lvlath` is not just “BFS, DFS, Dijkstra, and friends.” It is a small ecosystem where:

* `core.Graph` gives deterministic, thread-safe topology with explicit capabilities;
* traversal packages publish meaningful result artifacts, not incidental internal state;
* weighted routing keeps `+Inf`, unknown targets, disabled path tracking, and unreachable targets separate;
* `matrix` turns topology into dense numeric objects without erasing zero-weight or metric-closure semantics;
* flow, MST, TSP, DTW, grid, and builder packages extend the same discipline into optimization, alignment, fixtures, and examples.

---

## Why lvlath exists

A graph or numeric-algorithm library becomes dangerous when the demo works, but the production system loses ordering, topology meaning, numeric semantics, or error identity.

The common failures are familiar:

* Go map iteration leaks into traversal order;
* directed, undirected, mixed, loop, and multi-edge semantics are guessed instead of modeled;
* zero-weight edges disappear inside adjacency matrices;
* `+Inf` is replaced with “large enough” constants and later becomes a fake route;
* BFS is used for weighted routing because it “looks like pathfinding”;
* Dijkstra, MST, TSP, and DTW are treated as interchangeable shortest-cost tools;
* DFS discovery order is confused with finish order;
* max-flow examples print only a number and hide residual-network semantics;
* Christofides is claimed without checking metric assumptions or matching strength;
* DTW path output is requested from a memory mode that intentionally discarded history;
* examples panic, print unstable maps, or silently rely on fictional APIs;
* documentation explains textbook theory but not the actual package contract.

`lvlath` is built against those failure modes. It is an algorithm engineering library: each package publishes a narrow mathematical contract, validates its input domain, returns meaningful result artifacts, and keeps deterministic behavior observable.

### The repository-level laws

1. **Determinism is a package contract**

   `core` provides stable graph surfaces. Algorithm packages either preserve that order or define their own deterministic tie-breaks: BFS processing order, DFS finish order, Dijkstra strict improvement, MST edge ordering, DTW backtracking, TSP matrix order and solver tie-breaks.

2. **Capabilities are declared before topology is trusted**

   Directedness, weighting, loops, multi-edges, and mixed-edge behavior are graph capabilities, not comments. Invalid topology should be rejected at the write or validation boundary, not discovered after an algorithm has already published nonsense.

3. **Every non-trivial algorithm publishes a result artifact**

   BFS `Result`, DFS `Result`, Dijkstra `Result`, MST `Result`, TSP `Result`, DTW `Result`, residual flow graphs, adjacency matrices, incidence matrices, and metric closures are caller-owned artifacts with domain meaning. They are not incidental internal state.

4. **Errors are protocol, not prose**

   Public errors are meant for `errors.Is`. Error text is diagnostic only. A caller must be able to distinguish “unknown target,” “known unreachable,” “path not tracked,” “disconnected strict MST,” “invalid matching policy,” and “time-limited incumbent.”

5. **Policy must be explicit**

   Filters, cutoffs, thresholds, forest mode, metric closure, local search, matching engine, DTW window, memory mode, slope penalty, and TSP time limits are explicit policy choices. No package should silently swap to weaker mathematics because the selected one is inconvenient.

6. **Numeric semantics are not interchangeable**

   `0`, `+Inf`, `NaN`, negative weights, missing edges, zero-cost edges, unreachable states, and local costs mean different things in different packages. `matrix`, `dtw`, `mst`, `dijkstra`, and `tsp` intentionally enforce different numeric laws because they solve different problems.

7. **Graph algorithms and dense algebra compose without erasing meaning**

   `matrix` can convert topology into adjacency, incidence, metric closure, statistics, and factorization workflows. It must preserve zero-weight and `+Inf` semantics instead of flattening them into convenient but false dense arrays.

8. **Documentation, examples, tests, and source must describe the same library**

   `{package}/doc.go` is the implemented API contract. `docs/{PACKAGE}.md` teaches the mathematics and operational pitfalls. Examples demonstrate real package behavior. Tests defend the contract. None of them should invent future APIs or convenient lies.

---

## Installation

```bash
go get github.com/lvlath/go@latest
```

Requires Go 1.23 or newer. Pure Go. No cgo. No external runtime dependencies.

---

## Repository structure

```text
lvlath/
├── core/                  # deterministic graph substrate
│   └── doc.go             # package-level API contract
│
├── bfs/                   # unweighted hop traversal and weak components
├── dfs/                   # post-order traversal, cycle witnesses, topological sort
├── dijkstra/              # weighted single-source shortest paths
├── mst/                   # minimum spanning tree algorithms
├── flow/                  # Ford-Fulkerson, Edmonds-Karp, Dinic
├── dtw/                   # dynamic time warping for numeric sequences
├── gridgraph/             # 2D lattice graph generation
├── matrix/                # dense graph algebra and statistics
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
│   ├── GRID_GRAPH.md
│   ├── MATRICES.md
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
│ hop layers,    │              │ non-negative   │ │ strict MST  │ │ max-flow,      │              │ adjacency,     │
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

Composition rule:
  core owns topology.
  matrix owns dense graph/numeric artifacts.
  dtw owns temporal alignment artifacts.
  mst / flow / tsp own optimization artifacts.
  builder and gridgraph produce reproducible input worlds.
```

---

## Package roles and strengths

| Package     | What it gives you                                                                                                                                   | Concrete strength                                                                                              | Typical scenario                                                             |
|:------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------|
| `core`      | Thread-safe in-memory graph with deterministic `Vertices`, `Edges`, and `Neighbors`; explicit directed/weighted/loop/multi/mixed capabilities.      | Prevents unstable map order and invalid topology from leaking into algorithms.                                 | Service maps, routing graphs, dependency graphs, test fixtures.              |
| `bfs`       | Unweighted shortest-hop traversal, path reconstruction, weak components, hooks, filters, partial results.                                           | Distinguishes discovery (`Visited`) from processing (`Order`) and preserves useful partial state.              | Blast radius, dependency waves, crawler frontiers, weak island discovery.    |
| `dfs`       | DFS forest, post-order, cycle witnesses, topological sorting, hooks, filters, cancellation.                                                         | Makes finish order explicit and returns deterministic cycle witnesses instead of unstable recursion artifacts. | Release plans, DAG validation, lock/resource cycle auditing.                 |
| `dijkstra`  | Single-source shortest paths on non-negative weighted graphs, path witnesses, wall thresholds, max-distance cutoff, `+Inf` unreachable publication. | Separates unknown target, known unreachable target, tracking disabled, and no path.                            | Logistics routing, network failover, service-radius queries.                 |
| `mst`       | Minimum spanning tree construction through Prim/Kruskal.                                                                                            | Uses greedy MST structure for deterministic backbones and clustering cuts.                                     | Cable layout, transport backbones, clustering by removing heavy MST edges.   |
| `flow`      | Ford-Fulkerson, Edmonds-Karp, and Dinic over `core.Graph`, returning max flow and residual graph.                                                   | Preserves residual semantics and supports algorithm selection from simple to high-throughput.                  | Capacity planning, traffic engineering, assignment models, min-cut analysis. |
| `dtw`       | Dynamic Time Warping with window, slope penalty, memory modes, and optional path recovery.                                                          | Aligns sequences that share a pattern but differ in speed or local timing.                                     | Sensors, gestures, audio contours, time-series similarity.                   |
| `gridgraph` | 2D lattice graph generation with neighborhood and obstacle-style workflows.                                                                         | Avoids manual wiring for pathfinding maps and teaching graphs.                                                 | Grid routing, maps, demos, benchmark fixtures.                               |
| `matrix`    | Dense row-major matrices, adjacency/incidence, metric closure, APSP, algebra, LU/QR/Eigen, covariance/correlation, sanitation.                      | Connects graph topology to numeric workflows without losing zero/`+Inf`/metric semantics.                      | Spectral analysis, graph features, routing matrices, risk/ML preprocessing.  |
| `tsp`       | Tour optimization toolkit: practical approximation/local search/exact small-instance strategies.                                                    | Bridges exactness and practicality for route planning.                                                         | Delivery tours, inspection routes, metric routing experiments.               |
| `builder`   | Deterministic graph and data generators.                                                                                                            | Produces reproducible examples, tests, and benchmarks.                                                         | Golden tests, performance fixtures, tutorials.                               |

---

## Current contract documentation

| Layer                | File                                         | Purpose                                                                                     |
|:---------------------|:---------------------------------------------|:--------------------------------------------------------------------------------------------|
| Tutorial             | [`docs/TUTORIAL.md`](docs/TUTORIAL.md)       | First guided entry into graph/matrix concepts, package roles, examples, and best practices. |
| Core spec            | [`docs/CORE.md`](docs/CORE.md)               | Graph model, capability law, deterministic enumeration, locks, topology invariants.         |
| BFS spec             | [`docs/BFS.md`](docs/BFS.md)                 | Hop-distance math, BFS result semantics, partial results, weak components.                  |
| DFS spec             | [`docs/DFS.md`](docs/DFS.md)                 | Post-order semantics, cycle witnesses, DFS forest, topological sort.                        |
| Dijkstra spec        | [`docs/DIJKSTRA.md`](docs/DIJKSTRA.md)       | Weighted routing, `+Inf`, strict improvement, path tracking, wall/cutoff policy.            |
| MST spec             | [`docs/MST.md`](docs/MST.md)                 | Cut/cycle properties, Kruskal/Prim, deterministic MST construction.                         |
| Flow spec            | [`docs/FLOW.md`](docs/FLOW.md)               | Max-flow/min-cut math, residual networks, Ford-Fulkerson, Edmonds-Karp, Dinic.              |
| DTW spec             | [`docs/DTW.md`](docs/DTW.md)                 | Dynamic programming alignment, windows, penalties, memory modes, path recovery.             |
| Grid spec            | [`docs/GRID_GRAPH.md`](docs/GRID_GRAPH.md)   | Grid/lattice graph modeling and pathfinding-oriented construction.                          |
| Matrix spec          | [`docs/MATRICES.md`](docs/MATRICES.md)       | Dense matrix model, graph adapters, metric closure, zero-shape/statistics/numeric policy.   |
| TSP spec             | [`docs/TSP.md`](docs/TSP.md)                 | Tour optimization, exact vs approximate methods, metric assumptions.                        |
| Engineering standard | [`docs/lvlath_UES.md`](docs/lvlath_UES.md)   | Repository engineering standard and quality expectations.                                   |
| FAQ                  | [`docs/FAQ_&_TIPS.md`](docs/FAQ_%26_TIPS.md) | Troubleshooting, common pitfalls, usage tips.                                               |
| Contribution guide   | [`CONTRIBUTING.md`](CONTRIBUTING.md)         | Branching, tests, linting, coverage, PR rules.                                              |

---

## Quick start A: graph “lvlath” - shapes, components, cycles, matrix view

This is the first visual example: a graph is not only “nodes and edges”; it is a controllable topology that can be inspected, transformed, and projected into matrix form.

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
Expected cycle count:     1   // the internal cycle in the letter “a”
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

	// Deterministic induced view of the letter “a”.
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

---

## Design notes

### Determinism

Determinism is not “sort something at the end.” It is a cross-package contract.

* `core` owns deterministic graph surfaces: vertices, edges, neighbors.
* `bfs` preserves neighbor order and defines observable traversal state through result fields.
* `dfs` defines returned `Order` as finish order and publishes deterministic cycle witnesses.
* `dijkstra` uses deterministic frontier behavior and strict-improvement updates so equal-cost alternatives do not randomly overwrite witnesses.
* `mst` snapshots graph data, Kruskal stable-sorts finite edges, and Prim orders heap candidates by finite weight plus edge identity.
* `flow` builds and consumes residual state through deterministic algorithm policy.
* `matrix` uses fixed row-major loops and deterministic graph-adapter publication.
* `dtw` runs row-major DP and backtracks ties as diagonal, then vertical/up, then horizontal/left.
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
Fewest edges / hop layers?                    bfs
Deep traversal / finish order / cycles?       dfs
Cheapest non-negative route from one source?  dijkstra
Cheapest acyclic connected backbone?          mst
Maximum feasible throughput?                  flow
All-pairs shortest distances?                 matrix.BuildMetricClosure
Dense topology/statistics/spectral features?  matrix
Temporal alignment with phase drift?          dtw
Closed tour through every vertex?             tsp
Generated lattice topology?                   gridgraph
Repeatable benchmark/test fixtures?           builder
```

Important distinctions:

* BFS minimizes edge count, not cost.
* Dijkstra minimizes non-negative path cost, not throughput.
* MST minimizes selected backbone cost, not route distance.
* Flow maximizes capacity, not shortest path.
* Metric closure creates a distance artifact, not original graph topology.
* DTW aligns time-warped sequences, not graph routes.
* TSP solves a closed tour over a complete finite cost model, not sparse reachability.

---

### Matrix semantics

`matrix` intentionally does not pretend that every dense numeric array means the same thing.

```text
normal adjacency:       0 can mean “no edge”
zero-preserving mode:   finite 0 can mean “real zero-cost edge”; +Inf means absence
metric closure:         cell means shortest-path distance, not an original edge
incidence matrix:       columns are edge identities, not vertex-pair weights
```

### Concurrency

`core` is thread-safe for graph storage and controlled mutation. Algorithm packages should be run against topology that is stable for the duration of the call unless the package explicitly documents otherwise. `matrix.Dense` does not contain locks; concurrent writes require external synchronization.

---

## Choosing the right package

### By question type

| You ask                                                                       | Use                                                               | Why                                                                       |
|:------------------------------------------------------------------------------|:------------------------------------------------------------------|:--------------------------------------------------------------------------|
| “Which vertices are within `k` hops?”                                         | `bfs.BFS` with depth policy                                       | Hop layers are unweighted traversal.                                      |
| “Which weak islands exist?”                                                   | `bfs.Components`                                                  | Component membership is not a route-cost problem.                         |
| “What dependency order is valid?”                                             | `dfs.TopologicalSort`                                             | DAG ordering is finish-order semantics.                                   |
| “Where is the cycle?”                                                         | `dfs.DetectCycles`                                                | You need a deterministic witness, not all simple cycles.                  |
| “What is the cheapest route from this source?”                                | `dijkstra.Dijkstra`                                               | Non-negative weighted shortest path.                                      |
| “Which targets are outside a runtime budget?”                                 | `dijkstra.WithMaxDistance`                                        | Cutoff policy without deleting topology.                                  |
| “Which links form the cheapest connected backbone?”                           | `mst.MinimumSpanningTree`, `mst.Kruskal`, `mst.Prim`              | MST/MSF solves acyclic connectivity, not routing.                         |
| “What if the graph is disconnected but I still need per-component backbones?” | `mst.WithForest`                                                  | Forest mode is explicit, not a hidden fallback.                           |
| “What is the max source-to-sink capacity?”                                    | `flow.Dinic` or `flow.EdmondsKarp`                                | Flow algorithms reason over residual capacity.                            |
| “How do I compare two jittery sensor signatures?”                             | `dtw.Align`                                                       | Scalar DTW aligns timing drift.                                           |
| “How do I align model-provided frame costs?”                                  | `dtw.AlignCostMatrix`                                             | Caller owns the local-cost surface.                                       |
| “How do I align multivariate sequences?”                                      | `dtw.AlignMatrix`                                                 | Rows are time steps, columns are features.                                |
| “How do I compute topology features?”                                         | `matrix.NewAdjacencyMatrix`, `NewIncidenceMatrix`, `DegreeVector` | Matrix artifacts preserve topology policy.                                |
| “How do I compute all-pairs route distances?”                                 | `matrix.BuildMetricClosure`                                       | Floyd-Warshall distance artifact with `+Inf` semantics.                   |
| “How do I solve a complete inspection/delivery tour?”                         | `tsp.SolveMatrix`                                                 | TSP kernels consume matrix cost models.                                   |
| “How do I solve TSP from graph topology?”                                     | `tsp.SolveGraph`                                                  | Graph is adapted at the facade boundary, then solved as a matrix problem. |
| “How do I get exact TSP for a small instance?”                                | `tsp.ExactHeldKarp` / `tsp.BranchAndBound` policy                 | Exactness is explicit and guarded.                                        |
| “How do I get a metric symmetric approximation?”                              | `tsp.Christofides` + `tsp.BlossomMatch`                           | The `1.5` ratio requires exact matching and metric assumptions.           |
| “How do I build repeatable graphs?”                                           | `builder` / `gridgraph`                                           | Generated fixtures should be deterministic.                               |

### By failure mode you need to avoid

| Risk                                                 | lvlath answer                                                                                         |
|:-----------------------------------------------------|:------------------------------------------------------------------------------------------------------|
| Map-order flakiness in examples/tests                | `core` deterministic surfaces and package tie-break laws.                                             |
| Invalid topology reaches algorithms                  | Graph capability validation and package-level input validators.                                       |
| BFS used for weighted routing                        | Separate BFS hop semantics from Dijkstra weighted semantics.                                          |
| DFS order misread as discovery order                 | DFS `Result.Order` is documented finish order.                                                        |
| Unknown target treated as unreachable                | Dijkstra keeps target-domain errors separate from `+Inf` distances.                                   |
| Runtime route policy deletes graph structure         | Dijkstra thresholds/cutoffs model the run without mutating topology.                                  |
| Disconnected MST silently becomes forest             | `mst.WithForest` is explicit; strict MST returns disconnection.                                       |
| Non-finite MST weights corrupt sorting/heap behavior | MST rejects `NaN`, `+Inf`, `-Inf` before kernel execution.                                            |
| Flow output hides residual state                     | Flow result includes residual-network semantics.                                                      |
| DTW path requested from rolling memory               | DTW memory/path policy is explicit; path needs full accumulated storage.                              |
| DTW over-warps a signal                              | Use window and slope penalty deliberately.                                                            |
| Feature scale dominates multivariate DTW             | Normalize before `dtw.AlignMatrix`; it uses squared L2 row cost.                                      |
| Zero-weight edge disappears in matrix adjacency      | Use zero-preserving matrix policy; absence becomes `+Inf`.                                            |
| Metric closure exported as fake topology             | Matrix refuses metric-closure export as ordinary graph topology.                                      |
| TSP receives sparse unresolved `+Inf` matrix         | Use metric closure only when indirect travel is domain-valid; final kernels need finite completeness. |
| Christofides ratio claimed with greedy matching      | `tsp.BlossomMatch` is required for the formal `1.5` ratio.                                            |
| Branch-and-Bound timeout is mistaken for optimality  | Inspect TSP `Result.TimedOut` and TSP `Result.Optimal`.                                               |
| Local-search TSP treated as proof                    | 2-opt/3-opt improve tours; they do not certify global optimum.                                        |
| Randomized tests/benchmarks drift                    | Use fixed seeds or deterministic `builder` fixtures.                                                  |
| Docs describe fictional APIs                         | `{package}/doc.go`, `docs/{PACKAGE}.md`, examples, and tests must stay synchronized.                  |

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

**lvlath** - deterministic graph algorithms, dense graph algebra, and engineering-grade contracts for Go developers who want reproducible answers instead of accidental behavior.
