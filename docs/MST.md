<!--
  lvlath - Repository Documentation

  Purpose:
    This document is the repository-level specification, tutorial, and contract map
    for lvlath/mst. It defines the mathematical MST/MSF problem, maps the
    public API to the implementation architecture, explains deterministic Kruskal
    and Prim behavior, documents graph-policy validation, numeric discipline,
    sentinel-first errors, result ownership, and production usage patterns.

  Contract status:
    - Public API signatures described here are part of the public contract.
    - Option semantics described here are part of the public contract.
    - Determinism rules described here are part of the public contract.
    - Numeric-policy rules described here are part of the public contract.
    - Error-classification rules described here are part of the public contract.
    - Result ownership and nilability rules described here are part of the public contract.
    - Any incompatible change must be explicit, documented, tested, and versioned.

  Scope:
    - Exact MST and explicit MSF through deterministic Kruskal and Prim kernels.
    - Original-edge graph Steiner approximation through KMB for required terminals.
    - Closed/fixed-end edge-covering walks on caller-owned validated trees.
    - Distinct numeric domains, reproducible witnesses, and detached results.
    - Original-edge KMB graph Steiner network design for required terminals.
    - Closed and fixed-end edge-covering walks over validated SteinerResult trees.

  License:
    The lvlath repository is licensed under AGPL-3.0-only. See LICENSE.
-->

# 6. Minimum Spanning Trees and Forests

> **Package:** `lvlath/mst` | **Focus:** Prim/Kruskal MST/MSF · KMB Graph Steiner Networks · Closed/Fixed-End Tree-Covering Walks · Original Edge IDs

The `lvlath/mst` package provides three mathematically distinct connectivity operations over `core.Graph`: exact minimum spanning trees/forests (Prim and Kruskal), polynomial-time original-edge graph Steiner approximation (Kou–Markowsky–Berman), and optimal edge-covering walks **on a supplied tree**. Its public results represent different objects: `Result` for a full-vertex backbone, `SteinerResult` for a unique-edge terminal network, and `WalkResult` for ordered, potentially repeated traversals. The package validates capabilities, separates numeric domains, preserves original identities, and suppresses partial results.

The package answers one foundational question:

> What is the minimum finite-cost acyclic backbone that connects every vertex, or each connected component when explicit forest mode is requested?

Two exact greedy MST kernels remain available under one existing contract:

- `Kruskal`: global finite-weight ordering plus disjoint-set component merging.
- `Prim`: root-driven frontier expansion through a deterministic edge heap.

`SteinerTree` adds a **different optimization problem** with a nonnegative-weight domain, and `SteinerWalk` adds a fixed-tree traversal problem. Their specific laws appear in [§6.12](#612-graph-steiner-tree-kmb-original-edge-network-design) and [§6.13](#613-steinerwalk-optimal-edge-covering-on-a-given-tree). Do not transfer their guarantees or errors to ordinary MST/MSF.

---

## 6.1. What & Why: The Laws of `mst`

Minimum spanning tree computation is a graph-theoretic compression tool: it removes every unnecessary cycle while preserving connectivity with the least possible total finite edge cost. This makes it valuable in domains where each retained edge means money, latency, metal area, risk, power loss, or operational complexity.

### 6.1.1. Why MST matters

1. **Network design**
    - MST provides the cheapest connected backbone before redundancy, bandwidth, or reliability overlays are added.
    - Examples: fiber backbones, microwave relays, satellite control links, campus cabling, data-center interconnects, sensor networks, and emergency response grids.

2. **VLSI and physical design**
    - MST is a deterministic baseline for pin connection, global-routing sketches, net clustering, clock-tree preplanning, and channel reservation before Steiner-tree or timing-aware refinement.
    - It is not final sign-off routing, but it is a stable mathematical backbone for the first pass.

3. **Smart grid and green energy systems**
    - MST helps identify minimum-cost connectivity between substations, batteries, renewable generators, and critical loads.
    - Explicit forest mode represents islanded microgrids honestly instead of hiding disconnection.

4. **Big data and machine learning**
    - MST is a standard primitive for clustering, graph-based segmentation, similarity backbones, outlier analysis, and “cut the heaviest selected bridges” pipelines.

5. **Algorithmic subroutines**
    - MST appears in approximation algorithms, graph sparsification, partitioning, topology simplification, and as a diagnostic baseline for more constrained optimization problems.

### 6.1.2. The Six Laws of `mst`

#### Law 1: Graph-Semantics Fidelity

The package computes MST/MSF only over weighted, undirected graph semantics.

It rejects:
- nil graph input,
- unweighted graph policy,
- directed graph policy,
- directed edge-level overrides,
- empty graph input for tree publication.

Self-loops are ignored because they cannot connect two components. Parallel edges are preserved as independent candidates because the lighter useful representative may be part of the optimum.

#### Law 2: Determinism

The same graph and the same option list produce the same result.

Determinism comes from the composition of these implementation decisions:
- options are applied in caller-provided order,
- vertices are read from `core.Vertices()`,
- edges are read from `core.Edges()`,
- snapshot storage contains detached `core.Edge` values,
- Kruskal uses the strict comparator `(Weight, Edge.ID)` over finite weights,
- equal-weight Kruskal candidates follow unique original `Edge.ID`,
- Prim heap ordering uses finite `Weight`, then unique `Edge.ID`,
- component roots are normalized into deterministic public roots.

No public output may depend on Go map iteration order.

#### Law 3: Numeric Discipline

Weights are `float64`, but only finite values are valid.

Accepted:
- finite negative weights,
- finite zero weights,
- finite positive weights.

Rejected:
- `NaN`,
- `+Inf`,
- `-Inf`.

The package does not treat infinity as a wall, shortcut, disconnected marker, unknown value, or unreachable relation. Non-finite values are rejected before sorting or heap operations.

#### Law 4: Endpoint Resolution

Undirected edges are mirrored into both endpoint adjacency lists. Prim must resolve the opposite endpoint relative to the current source vertex.

The forbidden shortcut is:
```text
next = edge.To
```

The correct law is:
```text
if source == edge.From: next = edge.To
if source == edge.To:   next = edge.From
```

This is required for roots that start at the stored `To` endpoint of an undirected edge.

#### Law 5: Result Ownership

`Result` is the canonical result artifact.

It stores detached `core.Edge` values and does not retain live `*core.Edge` pointers into `core.Graph`. Callers can clone, serialize, reorder copies, or consume metadata without mutating the source graph.

#### Law 6: Partial-Result Suppression

On validation failure or strict-tree kernel failure, the package returns:
```go
nil, err
```

It does not publish partial trees. A disconnected strict-tree attempt is not a smaller successful MST; it is `ErrDisconnected`. A forest is published only when the caller explicitly requests `WithForest()`.

---

## 6.2. Domain Scope & Non-Goals

### 6.2.1. What the package solves

The package solves these problems:
1. strict MST over a connected weighted undirected `*core.Graph`,
2. explicit MSF over disconnected weighted undirected `*core.Graph`,
3. deterministic Kruskal selection through global candidate ordering and DSU merging,
4. deterministic Prim selection through root-based frontier expansion,
5. graph Steiner **approximation** through `SteinerTree` for a nonempty terminal set,
6. minimum-distance **tree-edge covering** through `SteinerWalk`, closed or fixed-end,
7. detached publication through three distinct results (`Result`, `SteinerResult`, `WalkResult`).

### 6.2.2. Supported graph model

The input graph must satisfy:
- `graph != nil`,
- `graph.Weighted() == true`,
- `graph.Directed() == false`,
- `graph.HasDirectedEdges() == false`,
- every candidate edge weight is finite,
- graph topology remains stable during snapshot construction.

### 6.2.3. Non-goals

The package does not implement:
- directed optimum branching,
- directed arborescence,
- independent shortest-path kernels (KMB reuses `dijkstra`),
- all-pairs routing,
- **exact general graph** Steiner optimization or geometric Steiner-point placement,
- globally optimal original-graph inspection routes (walks cover only supplied trees),
- dynamic MST maintenance,
- capacity-constrained MST,
- reliability-constrained redundant design,
- epsilon-based ordering,
- infinite-weight wall semantics,
- partial-result publication,
- concurrent mutation safety during execution,
- matrix-specific APIs unless those exact APIs exist in source code.

If a workflow needs directed reachability, dependency ordering, route distance, or a graph with one-way semantics, this package is not the correct abstraction.

### 6.2.4. Snapshot boundary

The adapter reads vertices and edges before the kernel starts and stores detached edge values. This protects the algorithm from later edge-value aliasing in the result, but it is not an isolation transaction against concurrent writers.

Concurrent mutation during snapshot construction is unsupported and voids correctness.

---

## 6.3. Mathematical Formulation

### 6.3.1. Graph and weight model

Let:
$$ G = (V, E, w) $$
where:
$$ V = \text{finite vertex set} $$
$$ E \subseteq \{\{u,v\} \mid u,v \in V, u \ne v\} $$
$$ w : E \to \mathbb{R} $$

The runtime requires every `w(e)` to be finite:
$$ \forall e \in E: w(e) \in \mathbb{R},\quad w(e) \ne \mathrm{NaN},\quad w(e) \ne +\infty,\quad w(e) \ne -\infty $$

Negative finite weights are valid. MST optimality does not require non-negative weights; it requires a finite total ordering over candidate edges.

### 6.3.2. Strict MST objective

A strict spanning tree `T` must satisfy:
$$ T \subseteq E $$
$$ (V,T) \text{ is connected} $$
$$ (V,T) \text{ is acyclic} $$
$$ |T| = |V| - 1 $$

The objective is:
$$ \operatorname{MST}(G) = \arg\min_{T \subseteq E}\;\sum_{e \in T} w(e) $$

subject to:
$$ T \text{ spans } V \quad \land \quad T \text{ is acyclic} \quad \land \quad |T| = |V| - 1 $$

CodeCogs view:
$$ \operatorname{MST}(G)=\arg\min_{T\subseteq E}\sum_{e\in T}w(e),\quad |T|=|V|-1 $$

### 6.3.3. Minimum spanning forest objective

For a disconnected graph with connected components:
$$ \mathcal{C}(G)=\{C_1,C_2,\ldots,C_k\} $$

explicit forest mode publishes:
$$ \operatorname{MSF}(G)=\bigcup_{i=1}^{k}\operatorname{MST}(C_i) $$

The result cardinality law is:
$$ |E_{result}| = |V| - k $$
and:
$$ \texttt{ComponentCount} = k $$

Forest mode is a different publication domain, not a fallback after strict MST fails.

### 6.3.4. Cut and cycle properties

The correctness of Kruskal and Prim is based on two classical greedy laws.

**Cut property:** for any non-empty proper subset `S ⊂ V`, if `e` is a minimum-weight edge crossing the cut `(S, V \ S)`, then `e` belongs to at least one MST.

$$ \delta(S)=\{(u,v)\in E\mid u\in S,\;v\in V\setminus S\} $$
$$ e \in \arg\min_{x\in\delta(S)}w(x) \Rightarrow e \text{ is safe for some MST} $$

**Cycle property:** for any cycle, a strictly heaviest edge on that cycle cannot belong to any MST.

$$ e \in C \land w(e) > w(x)\;\forall x\in C\setminus\{e\}\Rightarrow e \notin \operatorname{MST}(G) $$

When equal weights exist, multiple MST representatives may be valid. Determinism chooses a stable representative; it does not change the optimality proof.

### 6.3.5. Kruskal update law

Kruskal maintains a DSU partition `P` of vertices.

For a candidate edge `e=(u,v)` in sorted finite-weight order:
$$ \operatorname{Accept}(e) \iff \operatorname{Find}(u) \ne \operatorname{Find}(v) $$

If accepted:
$$ P \leftarrow \operatorname{Union}(P,u,v) $$
and:
$$ T \leftarrow T \cup \{e\} $$

Every accepted edge reduces the number of DSU components by one. Every rejected edge would form a cycle.

### 6.3.6. Prim frontier update law

Prim maintains a visited set `S` and a frontier heap of edges crossing from visited to unvisited vertices.

At each step:
$$ e^* = \arg\min_{e\in\delta(S)}(w(e), id(e)) $$
where `id(e)` is `Edge.ID` and is used only as a deterministic tie-breaker after finite weight.

Then:
$$ S \leftarrow S \cup \{target(e^*)\} $$
$$ T \leftarrow T \cup \{e^*\} $$

Endpoint resolution is source-relative:

$$
target(e,source)=
\begin{cases}
edge.To, & source=edge.From\\
edge.From, & source=edge.To
\end{cases}
$$

There are no runtime thresholds or cutoffs in the current source. The only admissibility gates are graph policy, finite numeric validation, root policy, and explicit strict-tree versus forest publication mode.

### 6.3.7. Complexity by phase

Let:
- `V = |vertices|`,
- `E = |non-loop candidate edges|`,
- `C = component count`,
- `k = number of options`,
- `α(V) = inverse Ackermann factor`.

| Phase            | Time                        | Space        | Notes                                                                                             |
|:-----------------|:----------------------------|:-------------|:--------------------------------------------------------------------------------------------------|
| Option assembly  | $$O(k)$$                    | $$O(1)$$     | Applies options in caller order and validates policy values.                                      |
| Snapshot adapter | $$O(V\log(V+1) + E\log(E+1))$$ | $$O(V + E)$$ | Includes sorting by `core.Vertices()` and `core.Edges()`, copying, loop exclusion, and validation. |
| Kruskal kernel   | $$O(E\log E + E\alpha(V))$$ | $$O(E + V)$$ | Strict `(Weight, Edge.ID)` order plus shared DSU; publication folds only selected edges. |
| Prim kernel      | $$O(E\log E)$$              | $$O(E + V)$$ | Edge-frontier heap; stale candidates may be discarded.                                            |
| Result clone     | $$O(E + C)$$                | $$O(E + C)$$ | Deep-copies selected edges and component roots.                                                   |
| EdgeValues       | $$O(E)$$                    | $$O(E)$$     | Returns caller-owned edge slice.                                                                  |

The current Prim implementation is an edge-frontier heap implementation. It must not be documented as `O(E log V)` unless the kernel changes to a vertex-key decrease-key design.

---

## 6.4. Public API & Result Contract

### 6.4.1. Public entry points

```go
func MinimumSpanningTree(graph *core.Graph, opts ...Option) (*Result, error)

func Kruskal(graph *core.Graph) (*Result, error)

func Prim(graph *core.Graph, root string) (*Result, error)
```

`MinimumSpanningTree` is the canonical facade. It assembles options, validates the graph through the snapshot adapter, dispatches to exactly one kernel, and returns `Result` on success.

`Kruskal` and `Prim` are focused wrappers over the facade. They do not contain independent algorithm logic.

### 6.4.2. Public policy types

```go
type Algorithm string

const (
    AlgorithmKruskal Algorithm = "kruskal"
    AlgorithmPrim    Algorithm = "prim"
)

type Mode string

const (
    ModeStrictTree Mode = "strict_tree"
    ModeForest     Mode = "forest"
)

type Option func(*Options) error

type Options struct {
    Algorithm Algorithm
    Mode      Mode
    Root      string
}
```

### 6.4.3. Public option helpers

```go
func DefaultOptions() Options

func ApplyOptions(user ...Option) (Options, error)

func WithAlgorithm(algorithm Algorithm) Option

func WithRoot(root string) Option

func WithForest() Option

func WithStrictTree() Option
```

### 6.4.4. Result artifact

```go
type Result struct {
    Algorithm Algorithm
    Mode      Mode
    Root      string

    Edges       []core.Edge
    TotalWeight float64

    VertexCount    int
    ComponentCount int
    ComponentRoots []string
}
```

Helper methods:

```go
func (r *Result) IsNil() bool
func (r *Result) Clone() *Result
func (r *Result) EdgeValues() ([]core.Edge, error)
```

### 6.4.5. Field contract

| Field            | Contract                                                    |
|:-----------------|:------------------------------------------------------------|
| `Algorithm`      | Algorithm selected by option assembly and dispatch.         |
| `Mode`           | Result publication mode: strict tree or explicit forest.    |
| `Root`           | Meaningful for Prim; empty for normal Kruskal strict calls. |
| `Edges`          | Detached selected `core.Edge` values.                       |
| `TotalWeight`    | Sum of selected edge weights.                               |
| `VertexCount`    | Number of vertices in the validated snapshot.               |
| `ComponentCount` | `1` for successful strict MST; `k` for forest mode.         |
| `ComponentRoots` | Deterministic public roots for result components.           |

### 6.4.6. Result-domain states

Ordinary MST/MSF has no target vertex and does not expose path-style states such as “unknown target” or “known but unreachable”. `SteinerTree` does distinguish missing/unreachable **required terminals**, while `SteinerWalk` requires explicit endpoints on a structurally valid supplied tree. Their independent result/error contracts are specified below.

| State                    | Result               | Error                                      | Meaning                                         |
|:-------------------------|:---------------------|:-------------------------------------------|:------------------------------------------------|
| Valid strict tree        | non-nil `*Result` | nil                                        | Connected graph successfully spanned.           |
| Valid explicit forest    | non-nil `*Result` | nil                                        | Forest requested and published.                 |
| Invalid graph policy     | nil                  | sentinel error                             | Graph cannot enter MST snapshot domain.         |
| Missing Prim root        | nil                  | `ErrEmptyRoot` or `core.ErrVertexNotFound` | Prim root policy failed.                        |
| Disconnected strict tree | nil                  | `ErrDisconnected`                          | Strict tree cannot span all vertices.           |
| Nil result helper access | nil data             | `ErrNilResult`                             | Helper requiring result data was called on nil. |

### 6.4.7. Wrapper equivalence

```go
Kruskal(graph)
```

is equivalent to:

```go
MinimumSpanningTree(graph, WithAlgorithm(AlgorithmKruskal))
```

```go
Prim(graph, root)
```

is equivalent to:

```go
MinimumSpanningTree(graph, WithAlgorithm(AlgorithmPrim), WithRoot(root))
```

---

## 6.5. Options, Numeric Policy & Error Law

### 6.5.1. Option assembly law

`ApplyOptions` starts from:

```go
Options{
    Algorithm: AlgorithmKruskal,
    Mode:      ModeStrictTree,
    Root:      "",
}
```

Then every option is applied in caller-provided order.

The option model is last-writer-wins. `WithForest()` may be overridden later by `WithStrictTree()`. `WithAlgorithm()` may be supplied more than once; the last valid algorithm wins.

### 6.5.2. Option behavior

| Option                            | Effect                             | Failure                           |
|:----------------------------------|:-----------------------------------|:----------------------------------|
| `WithAlgorithm(AlgorithmKruskal)` | Select Kruskal.                    | none                              |
| `WithAlgorithm(AlgorithmPrim)`    | Select Prim.                       | none                              |
| `WithAlgorithm(unknown)`          | Reject unsupported algorithm.      | `ErrUnsupportedAlgorithm`         |
| `WithRoot(root)`                  | Set explicit Prim root.            | `ErrEmptyRoot` when root is empty |
| `WithForest()`                    | Publish MSF instead of strict MST. | none                              |
| `WithStrictTree()`                | Restore strict MST publication.    | none                              |
| nil option                        | Invalid option value.              | `ErrNilOption`                    |

Prim strict tree mode requires a root during option assembly. Prim forest mode may omit root and grow components in deterministic vertex order.

### 6.5.3. Numeric policy

The numeric policy is finite-only:

$$ w(e) \in \mathbb{R}\quad\text{and}\quad w(e) \notin \{\mathrm{NaN}, +\infty, -\infty\} $$

There is no threshold option, no cutoff option, and no infinite-wall option in the current public API.

### 6.5.4. Comparator law

Kruskal candidate ordering:

```text
Primary key: finite Weight ascending
Tie law:     original Edge.ID ascending
Implementation: copy candidates; sort by strict (Weight, Edge.ID) tuple
```

Because `core.Edges()` is already ordered by original ID, the tuple comparator preserves the previous stable weight-sort selection. The shared selector intentionally does **not** sum selected costs: KMB closure/union selections need only their edges. Ordinary MST separately performs a checked ordered cost fold **after** connectivity is established.

Prim candidate ordering:

```text
Primary key: finite Weight ascending
Tie law:     unique Edge.ID ascending
```

Epsilon comparisons are forbidden in sort and heap ordering because they can violate strict weak ordering.

### 6.5.5. Sentinel-first error law

Use `errors.Is`. Do not parse error strings.

Current sentinels:

```go
var ErrInvalidGraph error
var ErrNilGraph error
var ErrDirectedGraph error
var ErrDirectedEdge error
var ErrUnweightedGraph error
var ErrEmptyGraph error
var ErrDisconnected error
var ErrEmptyRoot error
var ErrUnsupportedAlgorithm error
var ErrInvalidOption error
var ErrNilOption error
var ErrNaNInfWeight error
var ErrNilResult error
```

Graph-policy errors may be joined:

```go
errors.Join(ErrInvalidGraph, ErrNilGraph)
errors.Join(ErrInvalidGraph, ErrUnweightedGraph)
errors.Join(ErrInvalidGraph, ErrDirectedGraph)
errors.Join(ErrInvalidGraph, ErrDirectedEdge)
errors.Join(ErrDisconnected, ErrEmptyGraph)
```

A non-empty Prim root absent from the validated snapshot is classified through `core.ErrVertexNotFound`.

### 6.5.6. Panic policy

Public validation is error-returning.

Nil option values return `ErrNilOption`. Nil result data access returns `ErrNilResult`. The package does not use panic-based validation for normal user errors.

---

## 6.6. Algorithmic Architecture & Pseudocode

### 6.6.1. Runtime layers

```text
┌───────────────────────────┐
│ Public facade / wrappers  │
│ MinimumSpanningTree       │
│ Kruskal / Prim            │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ Option assembly           │
│ DefaultOptions            │
│ ApplyOptions              │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ Snapshot adapter          │
│ graph policy + edge copy  │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ Algorithm kernel          │
│ Kruskal or Prim           │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ Result publication     │
│ detached result artifact  │
└───────────────────────────┘
```

### 6.6.2. Snapshot adapter model

The adapter validates graph policy and builds:

```go
type mstSnapshot struct {
    vertices []string
    edges    []core.Edge
    adj      map[string][]core.Edge
}
```

Snapshot laws:

- `vertices` is a detached copy of `core.Vertices()` output,
- `edges` stores detached non-loop candidate edges,
- `adj` stores undirected mirrored adjacency lists,
- every candidate edge is copied by value,
- no live `*core.Edge` pointer is retained.

### 6.6.3. Facade pseudocode

```text
FUNCTION MinimumSpanningTree(graph, opts...):
    cfg, err = ApplyOptions(opts...)
    IF err != nil:
        RETURN nil, err

    snapshot, err = newMSTSnapshot(graph)
    IF err != nil:
        RETURN nil, err

    SWITCH cfg.Algorithm:
        CASE AlgorithmKruskal:
            RETURN kruskalKernel(snapshot, cfg)
        CASE AlgorithmPrim:
            RETURN primKernel(snapshot, cfg)
        DEFAULT:
            RETURN nil, ErrUnsupportedAlgorithm
```

### 6.6.4. Snapshot pseudocode

```text
FUNCTION newMSTSnapshot(graph):
    IF graph == nil:
        RETURN nil, Join(ErrInvalidGraph, ErrNilGraph)

    IF graph.Weighted() == false:
        RETURN nil, Join(ErrInvalidGraph, ErrUnweightedGraph)

    IF graph.Directed() == true:
        RETURN nil, Join(ErrInvalidGraph, ErrDirectedGraph)

    IF graph.HasDirectedEdges() == true:
        RETURN nil, Join(ErrInvalidGraph, ErrDirectedEdge)

    vertices = graph.Vertices()
    IF len(vertices) == 0:
        RETURN nil, Join(ErrDisconnected, ErrEmptyGraph)

    snapshot.vertices = copy(vertices)
    snapshot.edges = empty edge-value list
    snapshot.adj = map vertex -> edge-value list

    FOR each edge pointer in graph.Edges():
        IF edge.Weight is NaN or +Inf or -Inf:
            RETURN nil, ErrNaNInfWeight

        IF edge.Directed:
            RETURN nil, Join(ErrInvalidGraph, ErrDirectedEdge)

        IF edge.From == edge.To:
            CONTINUE

        candidate = copy edge value
        append candidate to snapshot.edges
        append candidate to snapshot.adj[candidate.From]
        append candidate to snapshot.adj[candidate.To]

    RETURN snapshot, nil
```

### 6.6.5. Kruskal kernel pseudocode

```text
FUNCTION kruskalKernel(snapshot, cfg):
    result = Result{
        Algorithm: AlgorithmKruskal
        Mode: cfg.Mode
        VertexCount: len(snapshot.vertices)
        Edges: capacity max(0, V-1)
    }

    IF V == 1:
        result.ComponentCount = 1
        result.ComponentRoots = [snapshot.vertices[0]]
        RETURN result, nil

    edges, roots, err = selectKruskalEdges(background, snapshot.vertices, snapshot.edges, cfg.Mode)
    IF err != nil:
        RETURN nil, err

    # Shared DSU selection has already resolved strict-tree connectivity.
    # Closure/union consumers do not need a scalar total at all.
    total, err = checkedMSTEdgeTotal(edges)
    IF err != nil:
        RETURN nil, err

    RETURN Result{Algorithm: Kruskal, Mode: cfg.Mode, Edges: edges,
                  TotalWeight: total, VertexCount: V,
                  ComponentCount: len(roots), ComponentRoots: roots}, nil
```

### 6.6.6. Prim kernel pseudocode

```text
FUNCTION primKernel(snapshot, cfg):
    result = Result{
        Algorithm: AlgorithmPrim
        Mode: cfg.Mode
        Root: cfg.Root
        VertexCount: len(snapshot.vertices)
        Edges: capacity max(0, V-1)
    }

    IF V == 1:
        root = cfg.Root
        IF root == "":
            root = snapshot.vertices[0]

        IF snapshot.hasVertex(root) == false:
            RETURN nil, core.ErrVertexNotFound

        result.Root = root
        result.ComponentCount = 1
        result.ComponentRoots = [root]
        RETURN result, nil

    IF cfg.Root != "" AND snapshot.hasVertex(cfg.Root) == false:
        RETURN nil, core.ErrVertexNotFound

    visited = empty set

    IF cfg.Root != "":
        growPrimComponent(snapshot, cfg.Root, visited, result)

    IF cfg.Mode == ModeStrictTree:
        IF len(result.Edges) != V - 1:
            RETURN nil, ErrDisconnected
        result.ComponentCount = 1
        IF result.ComponentRoots is empty:
            result.ComponentRoots = [cfg.Root]
    ELSE:
        FOR each root in snapshot.vertices:
            IF visited[root]:
                CONTINUE
            growPrimComponent(snapshot, root, visited, result)
        result.ComponentCount = len(result.ComponentRoots)
        IF result.Root == "" AND result.ComponentRoots is not empty:
            result.Root = result.ComponentRoots[0]

    # Cost is published only after structural success, in acceptance order.
    total, err = checkedMSTEdgeTotal(result.Edges)
    IF err != nil:
        RETURN nil, err
    result.TotalWeight = total
    RETURN result, nil
```

### 6.6.7. Prim component expansion pseudocode

```text
FUNCTION growPrimComponent(snapshot, root, visited, result):
    append root to result.ComponentRoots

    frontier = empty heap ordered by Weight, then Edge.ID

    visited[root] = true
    enqueuePrimSnapshotFrontier(snapshot, root, visited, frontier)

    WHILE frontier is not empty:
        candidate = heap.Pop(frontier)

        IF visited[candidate.target]:
            CONTINUE

        visited[candidate.target] = true
        append candidate.edge to result.Edges
        # Do not sum yet: strict-tree connectivity has precedence over overflow.

        enqueuePrimSnapshotFrontier(snapshot, candidate.target, visited, frontier)
```

### 6.6.8. Endpoint resolution pseudocode

```text
FUNCTION enqueuePrimSnapshotFrontier(snapshot, source, visited, frontier):
    FOR each edge in snapshot.adj[source]:
        IF source == edge.From:
            target = edge.To
        ELSE IF source == edge.To:
            target = edge.From
        ELSE:
            CONTINUE

        IF visited[target]:
            CONTINUE

        heap.Push(frontier, {edge: edge, target: target})
```

---

## 6.7. ASCII Diagrams

This section is intentionally diagram-heavy. Every non-trivial topology diagram uses explicit edge IDs and a complete edge catalog, so the drawing never forces readers to guess which channel connects which endpoints.

Diagram notation:

- `E##`, `F##`, `D##`, `U##`, and `K##` are edge IDs used only inside this document section.
- Edge labels in drawings are IDs, not weights.
- The edge catalog below each drawing is the authority for endpoints and weights.
- Box-drawing lines show visual structure; the catalog removes ambiguity.

### 6.7.1. VLSI global-routing candidate graph

This graph has ten blocks and eighteen candidate channels. It models a global-routing baseline: the MST is the minimum-cost acyclic skeleton that connects all blocks, not the final timing-aware, congestion-aware, or Steiner-optimized route.

```text
╔══════════════════════════════════════════════════════════════════════════════╗
║                         VLSI FLOORPLAN CONNECTIVITY                          ║
║            Edge IDs are candidate channel reservations, not wires.           ║
╚══════════════════════════════════════════════════════════════════════════════╝
  
                        ┌───────┐
         E14┌───────────┤  PLL  │──────────────┐
  ┌─────────┴───────┐   └───────┘              │E01
  │      PMIC       │                          │
  └───┬────────┬────┘                          │
      │        │E16                            │
      │        └───────────────────────┐       │
      │E15                          ┌──┴───────┴──────────┐
      │       E12┌──────────────────┤         NoC         ├────────────┐
      │      ┌───┴───┐              └─┬────┬──────┬──────┬┘            │
      └──────┤  PHY  ┝━━━━━┓          │E02 │      │E09   │E03          │E04
             └───┬───┘     ┃E18   ┌───┘    │  ┌───┴───┐  │             │
                 │         ┗━━━━━━┿━━━━━━━━┿━━┥  ISP  │  │             │
                 │E11             │        │  └───┬───┘  │             │
             ┌───┴───┐  E13  ┌────┴────┐   │   E10│ ┌────┴────┐    ┌───┴───┐
             │  DDR  ├───────┤   CPU   │   │      └─┤   GPU   │    │  NPU  │
             └───┬───┘       └────┬────┘   │        └────┬────┘    └───┬───┘
                 │                │E06     │E05          │E07          │E08
                 │                └─────┐  │            ┌┘             │
                 │                      │  │            │              │
                 │E17               ┌───┴──┴────────────┴─┐            │
                 └──────────────────┤         SRAM        ├────────────┘
                                    └─────────────────────┘
```

The drawing intentionally repeats `NoC` and `SRAM` labels as floorplan anchors. They refer to the same logical vertices, shown in multiple locations to keep the orthogonal wiring readable.

Complete VLSI edge catalog:

| Edge ID | Endpoint A | Endpoint B | Weight | Interpretation                               |
|:--------|:-----------|:-----------|-------:|:---------------------------------------------|
| `E01`   | `PLL`      | `NoC`      |      4 | Clock-control channel into the NoC hub.      |
| `E02`   | `NoC`      | `CPU`      |      3 | Low-cost CPU fabric attachment.              |
| `E03`   | `NoC`      | `GPU`      |      5 | Candidate GPU fabric channel.                |
| `E04`   | `NoC`      | `NPU`      |      4 | Candidate AI accelerator fabric channel.     |
| `E05`   | `NoC`      | `SRAM`     |      2 | Cheapest direct hub-to-SRAM channel.         |
| `E06`   | `CPU`      | `SRAM`     |      6 | Local CPU-to-SRAM fallback.                  |
| `E07`   | `GPU`      | `SRAM`     |      3 | Low-cost GPU memory channel.                 |
| `E08`   | `NPU`      | `SRAM`     |      4 | NPU memory-side channel.                     |
| `E09`   | `ISP`      | `NoC`      |      7 | Image pipeline to fabric.                    |
| `E10`   | `ISP`      | `GPU`      |      6 | ISP-to-GPU processing shortcut.              |
| `E11`   | `DDR`      | `PHY`      |      5 | DDR physical interface channel.              |
| `E12`   | `PHY`      | `NoC`      |      8 | PHY-to-fabric fallback channel.              |
| `E13`   | `DDR`      | `CPU`      |      9 | Expensive CPU-to-DDR candidate.              |
| `E14`   | `PMIC`     | `PLL`      |      3 | Power-management to clock-control link.      |
| `E15`   | `PMIC`     | `PHY`      |      4 | Power-management to physical interface link. |
| `E16`   | `PMIC`     | `NoC`      |     10 | Expensive PMIC-to-fabric fallback.           |
| `E17`   | `DDR`      | `SRAM`     |      7 | Memory-subsystem bridge candidate.           |
| `E18`   | `ISP`      | `PHY`      |      8 | ISP-to-physical interface fallback.          |

Result interpretation:

```text
Strict MST law:
  VertexCount = 10
  len(Edges)  = 9

Kruskal:
  sort all E01..E18 by finite Weight,
  accept only edges that merge different DSU components.

Prim:
  start from Root,
  push incident candidate channels,
  repeatedly select the minimum frontier edge by (Weight, Edge.ID).
```

### 6.7.2. Strict MST versus explicit MSF publication

The following graph has three connected components. Strict tree mode must reject it because there is no spanning tree over all vertices. Forest mode must publish one tree per component.

```text
╔════════════════════════════════ MODE COMPARISON ════════════════════════════╗
║        The topology is valid for MSF but invalid for strict MST output.     ║
╚═════════════════════════════════════════════════════════════════════════════╝

  ┌────────────────────────────── COMPONENT 1 ──────────────────────────────┐
  │                                                                         │
  │        ┌──────────── F01:3 ────────────┐                                │
  │        │                               │                                │
  │  ┌─────┴─────┐                   ┌─────┴─────┐                          │
  │  │  NorthA   │                   │  NorthB   │                          │
  │  └─────┬─────┘                   └─────┬─────┘                          │
  │        │ F03:7                         │ F02:2                          │
  │        │                               │                                │
  │  ┌─────┴─────┐                         │                                │
  │  │  NorthC   │◀────────────────────────┘                                │
  │  └───────────┘                                                          │
  └─────────────────────────────────────────────────────────────────────────┘
  
  ┌────────────────────────────── COMPONENT 2 ──────────────────────────────┐
  │                                                                         │
  │        ┌──────────── F04:4 ────────────┐                                │
  │        │                               │                                │
  │  ┌─────┴─────┐                   ┌─────┴─────┐                          │
  │  │  SouthA   │                   │  SouthB   │                          │
  │  └─────┬─────┘                   └─────┬─────┘                          │
  │        │ F06:6                         │ F05:1                          │
  │        │                               │                                │
  │  ┌─────┴─────┐                         │                                │
  │  │  SouthC   │◀────────────────────────┘                                │
  │  └───────────┘                                                          │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────── COMPONENT 3 ──────────────────────────────┐
  │                                                                         │
  │                              ┌─────────────┐                            │
  │                              │ Warehouse   │                            │
  │                              │ isolated    │                            │
  │                              └─────────────┘                            │
  └─────────────────────────────────────────────────────────────────────────┘
```

Complete forest edge catalog:

| Edge ID | Endpoint A  | Endpoint B | Weight | Component | Forest role                                                            |
|:--------|:------------|:-----------|-------:|:----------|:-----------------------------------------------------------------------|
| `F01`   | `NorthA`    | `NorthB`   |      3 | 1         | Selected together with `F02`; together they span the component at cost 5. |
| `F02`   | `NorthB`    | `NorthC`   |      2 | 1         | Selected.                                                              |
| `F03`   | `NorthA`    | `NorthC`   |      7 | 1         | Rejected: `F01` + `F02` already connects all three vertices. |
| `F04`   | `SouthA`    | `SouthB`   |      4 | 2         | Selected.                                                              |
| `F05`   | `SouthB`    | `SouthC`   |      1 | 2         | Selected.                                                              |
| `F06`   | `SouthA`    | `SouthC`   |      6 | 2         | Rejected; forms a more expensive cycle alternative.                    |
| none    | `Warehouse` | none       |      0 | 3         | Isolated component root, contributes no edge.                          |

Publication law:

```text
ModeStrictTree:
  result = nil
  error  = ErrDisconnected

ModeForest:
  result != nil
  ComponentCount = 3
  len(Edges)     = |V| - 3
  ComponentRoots = deterministic public roots
```

### 6.7.3. Validation and numeric policy walls

There are no runtime thresholds, cutoffs, blocked-edge policies, or infinite-wall semantics in the current API. The only walls are option validation, graph semantics, and finite numeric policy.

```text
╔══════════════════════════════════════════════════════════════════════════════╗
║                         MINIMUM SPANNING TREE CALL                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. Option assembly                                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ nil option                       │ ErrNilOption                              │
│ unsupported algorithm            │ ErrUnsupportedAlgorithm                   │
│ invalid mode                     │ ErrInvalidOption                          │
│ Prim + strict tree + empty root  │ ErrEmptyRoot                              │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│ 2. Graph policy wall                                                               │
├────────────────────────────────────────────────────────────────────────────────────┤
│ nil graph                       │ errors.Join(ErrInvalidGraph, ErrNilGraph)        │
│ unweighted graph                │ errors.Join(ErrInvalidGraph, ErrUnweightedGraph) │
│ directed graph                  │ errors.Join(ErrInvalidGraph, ErrDirectedGraph)   │
│ directed edge-level override    │ errors.Join(ErrInvalidGraph, ErrDirectedEdge)    │
│ empty graph                     │ errors.Join(ErrDisconnected, ErrEmptyGraph)      │
└──────────────────────────────────────┬─────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. Edge catalog scan                                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│ NaN                            │ ErrNaNInfWeight                             │
│ +Inf                           │ ErrNaNInfWeight                             │
│ -Inf                           │ ErrNaNInfWeight                             │
│ finite negative weight         │ accepted                                    │
│ finite zero weight             │ accepted                                    │
│ finite positive weight         │ accepted                                    │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │ Kruskal or Prim kernel   │
                         └──────────────────────────┘
```

Numeric interpretation catalog:

| Weight class    | Runtime meaning                                               |
|:----------------|:--------------------------------------------------------------|
| finite negative | Valid. MST can prefer negative-cost edges.                    |
| finite zero     | Valid. Useful for free links or already-provisioned channels. |
| finite positive | Valid. Standard cost model.                                   |
| `NaN`           | Invalid. Ordering is undefined.                               |
| `+Inf`          | Invalid. This package has no unreachable-wall semantics.      |
| `-Inf`          | Invalid. This package has no free-superedge semantics.        |

### 6.7.4. Equal-cost determinism tie-break

Equal weights are mathematically valid. Public instability is not. This diagram uses a complete four-vertex graph where every edge has weight `1`; the chosen representative must still be reproducible.

```text
╔══════════════════════════ EQUAL-WEIGHT COMPLETE GRAPH ══════════════════════╗
║                        Every listed edge has Weight = 1.                    ║
╚═════════════════════════════════════════════════════════════════════════════╝

                         ┌─────────────D03─────────────┐       
                         │                             │       
                   ┌─────┴─────┐                 ┌─────┴─────┐       
                   │     A     ├─────────┐       │     D     ├─┐     
                   └─────┬─────┘         │       └─────┬─────┘ │      
                         │D01            │D02          │D06    │       
                         │               └─────────┐   │       │        
                   ┌─────┴─────┐                 ┌─┴───┴─────┐ │      
                   │     B     ├──────D05────────┤     C     │ │      
                   └─────┬─────┘                 └───────────┘ │      
                         └─────────────D04─────────────────────┘                                            
                                                                       

Additional cross edge:
  D04 = B──D, Weight = 1
```

Complete equal-weight edge catalog:

| Edge ID | Endpoint A | Endpoint B | Weight | Deterministic role               |
|:--------|:-----------|:-----------|-------:|:---------------------------------|
| `D01`   | `A`        | `B`        |      1 | Earliest equal-weight candidate. |
| `D02`   | `A`        | `C`        |      1 | Candidate with same cost.        |
| `D03`   | `A`        | `D`        |      1 | Candidate with same cost.        |
| `D04`   | `B`        | `D`        |      1 | Candidate with same cost.        |
| `D05`   | `B`        | `C`        |      1 | Candidate with same cost.        |
| `D06`   | `C`        | `D`        |      1 | Candidate with same cost.        |

Tie-break law:

```text
Kruskal:
  strict sort by (Weight, Edge.ID)
  equal weights follow original Edge.ID

Prim:
  heap key = (Weight, Edge.ID)
  Edge.ID closes the tie because IDs are unique

Forbidden:
  epsilon comparator inside sort/heap
  map iteration as tie-break source
  source/target tie-break after unique Edge.ID
```

### 6.7.5. Endpoint Resolution Law

Undirected adjacency is mirrored. Prim must resolve the opposite endpoint relative to the current source. The stored edge direction fields are not traversal direction.

```text
╔════════════════════════════ STORED UNDIRECTED EDGE ═════════════════════════╗
║ U01: edge.From = "A"                                                        ║
║      edge.To   = "B"                                                        ║
║      Weight    = 7                                                          ║
╚═════════════════════════════════════════════════════════════════════════════╝

Snapshot adjacency after mirroring:

┌─────────────────────┬──────────────────────────────────────────────────────┐
│ Adjacency bucket    │ Stored edge values                                   │
├─────────────────────┼──────────────────────────────────────────────────────┤
│ adj["A"]            │ U01: core.Edge{From:"A", To:"B", Weight:7}           │
│ adj["B"]            │ U01: core.Edge{From:"A", To:"B", Weight:7}           │
└─────────────────────┴──────────────────────────────────────────────────────┘

Correct endpoint resolution:

┌───────────────┬─────────────────────────┬──────────────────────────────────┐
│ Current source│ Condition               │ Target                           │
├───────────────┼─────────────────────────┼──────────────────────────────────┤
│ "A"           │ source == edge.From     │ edge.To   = "B"                  │
│ "B"           │ source == edge.To       │ edge.From = "A"                  │
└───────────────┴─────────────────────────┴──────────────────────────────────┘

Failure caused by the wrong shortcut:

┌──────────────────────────────────────────────────────────────────────────────┐
│ Wrong rule: target = edge.To                                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│ root = "B"                                                                   │
│ source = "B"                                                                 │
│ edge.To = "B"                                                                │
│ target appears already visited                                               │
│ frontier fails to cross B──A                                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

Complete endpoint-law edge catalog:

| Edge ID | Stored `From` | Stored `To` | Weight | Correct target from `A` | Correct target from `B` |
|:--------|:--------------|:------------|-------:|:------------------------|:------------------------|
| `U01`   | `A`           | `B`         |      7 | `B`                     | `A`                     |

### 6.7.6. Kruskal merge timeline

This eight-vertex graph shows how sorted edges and DSU merges construct a strict MST. Every listed edge is undirected and finite.

```text
╔════════════════════════════ KRUSKAL MERGE TIMELINE ═════════════════════════╗
║            An accepted edge must merge two different DSU components.        ║
╚═════════════════════════════════════════════════════════════════════════════╝

Initial partition:
  {A}  {B}  {C}  {D}  {E}  {F}  {G}  {H}

Topology sketch:

        ┌──────── K01:2 ────────┐
        │                       │
  ┌─────┴─────┐           ┌─────┴─────┐           ┌───────────┐
  │     A     │           │     B     ├──K02:3────┤     C     ├──┐
  └─────┬─────┘           └─────┬─────┘           └───────────┘  │
        │ K08:9                 │ K09:7                          │
        │                       │                                │
  ┌─────┴─────┐           ┌─────┴─────┐           ┌───────────┐  │
  │     H     ├──K07:6────┤     G     ├──K06:5────┤     F     │  │
  └───────────┘           └─────┬─────┘           └─────┬─────┘  │
                                │ K10:8                 │ K05:5  │
                                │                       │        │
                          ┌─────┴─────┐           ┌─────┴─────┐  │
                          │     D     ├──K03:3────┤     E     │  │
                          └─────┬─────┘           └───────────┘  │
                                └─────────K04:4──────────────────┘
```

Complete Kruskal candidate catalog:

| Edge ID | Endpoint A | Endpoint B | Weight | DSU outcome in the shown timeline                         |
|:--------|:-----------|:-----------|-------:|:----------------------------------------------------------|
| `K01`   | `A`        | `B`        |      2 | Accepted; merges `{A}` and `{B}`.                         |
| `K02`   | `B`        | `C`        |      3 | Accepted; merges `{A,B}` and `{C}`.                       |
| `K03`   | `D`        | `E`        |      3 | Accepted; merges `{D}` and `{E}`.                         |
| `K04`   | `C`        | `D`        |      4 | Accepted; merges `{A,B,C}` and `{D,E}`.                   |
| `K05`   | `E`        | `F`        |      5 | Accepted; merges `F` into the main component.             |
| `K06`   | `F`        | `G`        |      5 | Accepted; merges `G` into the main component.             |
| `K07`   | `G`        | `H`        |      6 | Accepted; merges `H` into the main component.             |
| `K08`   | `A`        | `H`        |      9 | Rejected if reached; forms a cycle.                       |
| `K09`   | `B`        | `G`        |      7 | Rejected if reached after `K06`; forms a cycle.           |
| `K10`   | `D`        | `G`        |      8 | Rejected if reached after `K04` and `K06`; forms a cycle. |

Merge table:

| Step | Edge  | Weight | Action                       | Accepted edges count |
|-----:|:------|-------:|:-----------------------------|---------------------:|
|    1 | `K01` |      2 | accept                       |                    1 |
|    2 | `K02` |      3 | accept                       |                    2 |
|    3 | `K03` |      3 | accept                       |                    3 |
|    4 | `K04` |      4 | accept                       |                    4 |
|    5 | `K05` |      5 | accept                       |                    5 |
|    6 | `K06` |      5 | accept                       |                    6 |
|    7 | `K07` |      6 | accept; strict tree complete |                    7 |

Completion law:

$$ |T| = |V| - 1 = 8 - 1 = 7 $$

## 6.8. Go Example Scenarios

The examples below use production-style error checking for graph construction and algorithm execution. They are scenario-driven and intentionally use non-trivial graphs with 8 or more vertices.

### 6.8.1. VLSI global-routing baseline

The graph models candidate channels between chip blocks. MST gives a deterministic low-cost skeleton for later physical design passes.

```go
package main

import (
    "fmt"

    "github.com/lvlath/go/core"
    "github.com/lvlath/go/mst"
)

func main() {
    graph, _ := core.NewGraph(core.WithWeighted())
	_, _ = graph.AddEdge("PLL", "NoC", 4)
	_, _ = graph.AddEdge("NoC", "CPU", 3)
	_, _ = graph.AddEdge("NoC", "GPU", 5)
	_, _ = graph.AddEdge("NoC", "NPU", 4)
	_, _ = graph.AddEdge("NoC", "SRAM", 2)
	_, _ = graph.AddEdge("CPU", "SRAM", 6)
	_, _ = graph.AddEdge("GPU", "SRAM", 3)
	_, _ = graph.AddEdge("NPU", "SRAM", 4)
	_, _ = graph.AddEdge("ISP", "NoC", 7)
	_, _ = graph.AddEdge("ISP", "GPU", 6)
	_, _ = graph.AddEdge("DDR", "PHY", 5)
	_, _ = graph.AddEdge("PHY", "NoC", 8)
	_, _ = graph.AddEdge("DDR", "CPU", 9)
	_, _ = graph.AddEdge("PMIC", "PLL", 3)
	_, _ = graph.AddEdge("PMIC", "PHY", 4)
	_, _ = graph.AddEdge("PMIC", "NoC", 10)
	_, _ = graph.AddEdge("DDR", "SRAM", 7)
	_, _ = graph.AddEdge("ISP", "PHY", 8)

    result, err := mst.MinimumSpanningTree(graph)
    if err != nil {
        fmt.Println("mst:", err)
        return
    }

    fmt.Printf(
        "algorithm=%s mode=%s blocks=%d wires=%d components=%d total=%.0f\n",
        result.Algorithm,
        result.Mode,
        result.VertexCount,
        len(result.Edges),
        result.ComponentCount,
        result.TotalWeight,
    )

    // Output:
    // algorithm=kruskal mode=strict_tree blocks=10 wires=9 components=1 total=34
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-MST_VLSI_global_routing_baseline-blue?logo=go)](https://go.dev/play/p/03e_35hMHgC)

### 6.8.2. Space-mesh minimum spanning forest

The graph models disconnected orbital communication groups. Strict MST would be false; explicit forest mode publishes one minimum tree per reachable mesh.

```go
package main

import (
    "fmt"

    "github.com/lvlath/go/core"
    "github.com/lvlath/go/mst"
)

func main() {
    graph, err := core.NewGraph(core.WithWeighted())
	_, _ = graph.AddEdge("Arctic-1", "Arctic-2", 6)
	_, _ = graph.AddEdge("Arctic-2", "Arctic-3", 4)
	_, _ = graph.AddEdge("Arctic-3", "Arctic-Gateway", 5)
	_, _ = graph.AddEdge("Arctic-1", "Arctic-Gateway", 9)
	_, _ = graph.AddEdge("Arctic-2", "Arctic-Gateway", 8)

	_, _ = graph.AddEdge("Equator-1", "Equator-2", 3)
	_, _ = graph.AddEdge("Equator-2", "Equator-3", 4)
	_, _ = graph.AddEdge("Equator-3", "Equator-4", 3)
	_, _ = graph.AddEdge("Equator-1", "Equator-4", 8)
	_, _ = graph.AddEdge("Equator-2", "Equator-4", 5)
	_, _ = graph.AddEdge("Equator-1", "Equator-3", 6)

	_, _ = graph.AddEdge("Pacific-Relay", "Pacific-1", 7)
	_, _ = graph.AddEdge("Pacific-Relay", "Pacific-2", 6)
	_, _ = graph.AddEdge("Pacific-1", "Pacific-2", 10)

    result, err := mst.MinimumSpanningTree(
        graph,
        mst.WithAlgorithm(mst.AlgorithmPrim),
        mst.WithForest(),
    )
    if err != nil {
        fmt.Println("msf:", err)
        return
    }

    fmt.Printf(
        "algorithm=%s mode=%s satellites=%d links=%d components=%d roots=%v total=%.0f\n",
        result.Algorithm,
        result.Mode,
        result.VertexCount,
        len(result.Edges),
        result.ComponentCount,
        result.ComponentRoots,
        result.TotalWeight,
    )

    // Output:
    // algorithm=prim mode=forest satellites=11 links=8 components=3 roots=[Arctic-1 Equator-1 Pacific-1] total=38
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-MST_Space_mesh_minimum_spanning_forest-blue?logo=go)](https://go.dev/play/p/zgSGcKIVerE)

### 6.8.3. Embedding clustering by cutting MST bridges

The graph represents distances between embedding prototypes. Kruskal builds the similarity backbone; cutting the two largest selected edges yields three clusters.

```go
package main

import (
    "fmt"
    "sort"

    "github.com/lvlath/go/core"
    "github.com/lvlath/go/mst"
)

func main() {
    graph, _ := core.NewGraph(core.WithWeighted())
	_, _ = graph.AddEdge("vision-01", "vision-02", 0.12)
	_, _ = graph.AddEdge("vision-02", "vision-03", 0.15)
	_, _ = graph.AddEdge("vision-01", "vision-03", 0.22)

	_, _ = graph.AddEdge("speech-01", "speech-02", 0.10)
	_, _ = graph.AddEdge("speech-02", "speech-03", 0.17)
	_, _ = graph.AddEdge("speech-01", "speech-03", 0.25)

	_, _ = graph.AddEdge("fraud-01", "fraud-02", 0.08)
	_, _ = graph.AddEdge("fraud-02", "fraud-03", 0.13)
	_, _ = graph.AddEdge("fraud-01", "fraud-03", 0.21)

	_, _ = graph.AddEdge("vision-03", "speech-01", 0.94)
	_, _ = graph.AddEdge("speech-03", "fraud-01", 1.08)
	_, _ = graph.AddEdge("vision-02", "speech-02", 1.21)
	_, _ = graph.AddEdge("speech-01", "fraud-02", 1.35)
	_, _ = graph.AddEdge("vision-01", "fraud-03", 1.62)

    result, err := mst.Kruskal(graph)
    if err != nil {
        fmt.Println("mst:", err)
        return
    }

    selected := result.Clone().Edges
    sort.SliceStable(selected, func(i, j int) bool {
        return selected[i].Weight > selected[j].Weight
    })

    cut1 := selected[0]
    cut2 := selected[1]
    remaining := result.TotalWeight - cut1.Weight - cut2.Weight

    fmt.Printf(
        "mst_edges=%d mst_total=%.2f cut1=%s-%s:%.2f cut2=%s-%s:%.2f remaining=%.2f clusters=%d\n",
        len(result.Edges),
        result.TotalWeight,
        cut1.From,
        cut1.To,
        cut1.Weight,
        cut2.From,
        cut2.To,
        cut2.Weight,
        remaining,
        3,
    )

    // Output:
    // mst_edges=8 mst_total=2.77 cut1=speech-03-fraud-01:1.08 cut2=vision-03-speech-01:0.94 remaining=0.75 clusters=3
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-MST_Embedding_clustering_by_cutting_MST_bridges-blue?logo=go)](https://go.dev/play/p/df1IH3FAYFl)

### 6.8.4. Smart-grid sentinel classification

The graph mixes undirected physical power links with a directed telemetry/control relation. MST rejects the directed edge-level override before tree construction.

```go
package main

import (
    "errors"
    "fmt"

    "github.com/lvlath/go/core"
    "github.com/lvlath/go/mst"
)

func main() {
    graph, _ := core.NewGraph(core.WithWeighted(), core.WithMixedEdges())
	_, _ = graph.AddEdge("SolarFarm-A", "Substation-A", 4)
	_, _ = graph.AddEdge("WindPark-B", "Substation-B", 5)
	_, _ = graph.AddEdge("Substation-A", "Substation-B", 3)
	_, _ = graph.AddEdge("Substation-B", "BatteryHub", 2)
	_, _ = graph.AddEdge("BatteryHub", "HospitalLoop", 6)
	_, _ = graph.AddEdge("HospitalLoop", "DowntownLoad", 4)
	_, _ = graph.AddEdge("DowntownLoad", "IndustrialLoad", 5)
	_, _ = graph.AddEdge("IndustrialLoad", "Substation-A", 7)
	_, _ = graph.AddEdge("MicrogridIsland", "BatteryHub", 8)

    if _, err := graph.AddEdge("ControlCenter", "Substation-A", 1, core.WithEdgeDirected(true)); err != nil {
        fmt.Println("directed edge:", err)
        return
    }

    _, err := mst.MinimumSpanningTree(graph)

    fmt.Printf(
        "invalid=%t directed_edge=%t\n",
        errors.Is(err, mst.ErrInvalidGraph),
        errors.Is(err, mst.ErrDirectedEdge),
    )

    // Output:
    // invalid=true directed_edge=true
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-MST_Smart_grid_sentinel_classification-blue?logo=go)](https://go.dev/play/p/N0nkAyGUd17)


---

## 6.9. Laws of Ownership, Concurrency & Partial Results

### 6.9.1. Ownership Law

The public result is detached.

`Result.Edges` stores `core.Edge` values, not `*core.Edge` pointers.

The snapshot adapter also copies edge values before kernel execution.

Consequences:

- mutating returned edge values does not mutate the source graph,
- `Result.Clone()` deep-copies slice fields,
- `Result.EdgeValues()` returns a caller-owned slice,
- callers may serialize or transform results safely after return.

### 6.9.2. Clone Law

`Clone()` is nil-safe.

```go
var result *mst.Result
clone := result.Clone()
fmt.Println(clone == nil)
```

Output:

```text
true
```

For non-nil results, `Clone()` preserves scalar metadata and deep-copies:

- `Edges`,
- `ComponentRoots`.

### 6.9.3. Nil-result helper law

`EdgeValues()` classifies nil receiver access with `ErrNilResult`.

```go
var result *mst.Result
edges, err := result.EdgeValues()
```

Expected state:

```text
edges == nil
errors.Is(err, mst.ErrNilResult) == true
```

### 6.9.4. Concurrency Law

The package snapshots graph vertices and edges before kernel execution.

Concurrent graph reads are acceptable only when the underlying `core.Graph` usage remains read-only during the algorithm call.

Concurrent mutation during snapshot construction is unsupported.

Do not add, remove, or rewrite graph vertices or edges while `MinimumSpanningTree`, `Kruskal`, or `Prim` is executing on the same graph.

### 6.9.5. Partial-Result Law

The package suppresses partial results.

For any validation failure or strict-tree kernel failure:

```text
result == nil
err != nil
```

Examples:

- nil graph,
- unweighted graph,
- directed graph,
- directed edge override,
- empty graph,
- non-finite weight,
- missing Prim root,
- disconnected graph in strict mode.

Forest mode is not a partial result. It is an explicit publication mode selected by `WithForest()`.

---

## 6.10. Pitfalls & Best Practices

### 6.10.1. Anti-patterns

>#### Anti-pattern 1: Parsing error strings
>
>Do not branch on `err.Error()`.
>
>Use:
>
>```go
>errors.Is(err, mst.ErrDisconnected)
>```

> #### Anti-pattern 2: Treating `edge.To` as the Prim destination
>
> Undirected edge values are mirrored into both adjacency lists. The next vertex must be resolved relative to the current source.

> #### Anti-pattern 3: Using epsilon inside MST comparators
>
> Do not compare MST candidate weights as “approximately less”. Epsilon-based comparison can be non-transitive and can break sort/heap contracts.

> #### Anti-pattern 4: Expecting strict MST on disconnected input
>
> Strict mode requires one spanning tree over all vertices. Use `WithForest()` when disconnected components are expected and meaningful.

> #### Anti-pattern 5: Treating `+Inf` as unreachable
>
> This package has no infinite-wall semantics. All non-finite weights are rejected.

> #### Anti-pattern 6: Reusing partial results after errors
>
> There are no partial MST results. If `err != nil`, treat the result as absent.

> #### Anti-pattern 7: Mutating graph topology during execution
>
> Do not add/remove vertices or edges while MST/MSF computation is running.

> #### Anti-pattern 8: Assuming Prim and Kruskal must always select identical edges
>
> They must agree on optimal total weight and structural validity. Equal-weight graphs can have multiple valid MST representatives.

> #### Anti-pattern 9: Discarding result metadata too early
>
> Do not immediately convert `Result` into a raw edge slice unless the downstream stage truly does not need algorithm, mode, root, component, or total-weight metadata.

### 6.10.2. Best practices

> #### Best practice 1: Model publication policy through options
>
> Use `WithForest()` for naturally disconnected systems. Do not delete vertices or edges to force strict tree success.

> #### Best practice 2: Use the canonical facade for policy-rich calls
>
> ```go
> MinimumSpanningTree(graph, WithAlgorithm(AlgorithmPrim), WithRoot("A"), WithForest())
> ```
>
> Use focused wrappers only when strict tree mode and algorithm choice are fixed.

> #### Best practice 3: Preserve edge IDs for auditability
>
> Stable edge IDs make deterministic MST output traceable back to source topology.

> #### Best practice 4: Validate data ingestion upstream
>
> The package rejects non-finite values, but data pipelines should still validate numeric inputs before graph construction.

> #### Best practice 5: Clone before destructive local transforms
>
> If a downstream stage sorts selected edges by descending weight, clusters by cuts, or filters edges for reporting, use `Clone()` or `EdgeValues()` first.

> #### Best practice 6: Prefer forest mode for islanded systems
>
> Satellite groups, microgrids, offline regions, disconnected data clusters, and segmented infrastructure often have meaningful disconnected components. Use explicit forest mode.

> #### Best practice 7: Treat strict mode failure as a topology signal
>
> `ErrDisconnected` in strict mode is not an inconvenience; it is a mathematical statement that the graph cannot be spanned as one tree.

> #### Best practice 8: Keep graph topology stable during execution
>
> Run MST/MSF over a stable graph. For mutation-heavy systems, build or clone a graph snapshot before invoking the package.

---

## 6.11. Practical Recipes

### 6.11.1. Cheapest connected backbone

Use the default facade:

```go
result, err := mst.MinimumSpanningTree(graph)
```

Default policy is Kruskal + strict tree.

### 6.11.2. Rooted expansion from an operational hub

Use Prim when the root is meaningful to the workflow:

```go
result, err := mst.Prim(graph, "HQ")
```

This is useful when logs, diagrams, or operational explanations should describe growth from a known site.

### 6.11.3. Islanded network publication

Use explicit forest mode:

```go
result, err := mst.MinimumSpanningTree(graph, mst.WithForest())
```

This is correct for disconnected domains where each component has independent meaning.

### 6.11.4. ML clustering through bridge cuts

Run Kruskal, clone selected edges, sort by descending weight, and remove the largest selected bridges. The MST gives a sparse backbone where heavy selected edges often represent inter-cluster bridges.

### 6.11.5. Sentinel-classified validation

Use `errors.Is`:

```go
if errors.Is(err, mst.ErrDirectedEdge) {
    // The graph contains a directed edge-level override and is outside MST domain.
}
```

---

## 6.12. Graph Steiner Tree: KMB Original-Edge Network Design

### 6.12.1. Mathematical problem and exactness boundary

A strict MST spans **every vertex**, even an operationally irrelevant relay. In the graph Steiner problem, the input is a finite weighted undirected graph `G=(V,E,w)` with `w(e) >= 0` and an explicit nonempty set of required vertices `T ⊆ V` (terminals). An admissible result is a connected, acyclic subgraph of **original edges** that includes all terminals. Nonterminal original vertices may be used as junctions; no new geometric point is inserted.

$$ \operatorname{OPT}_{ST}(G,T)=\min_{H\subseteq G,\ H\text{ connected},\ T\subseteq V(H)}\ \sum_{e\in E(H)}w(e). $$

The general graph Steiner problem is NP-hard. `SteinerTree` implements the Kou–Markowsky–Berman (**KMB**) polynomial-time approximation, not Dreyfus–Wagner exact dynamic programming. In exact arithmetic, for `k=|T|>1`, the result satisfies the conservative bound:

$$ \operatorname{cost}(T_{KMB})\le 2\left(1-\frac{1}{k}\right)\operatorname{OPT}_{ST}. $$

![KMB approximation bound](https://latex.codecogs.com/svg.image?%5Coperatorname%7Bcost%7D%28T_%7BKMB%7D%29%5Cleq2%5Cleft%281-%5Cfrac1k%5Cright%29%5Coperatorname%7BOPT%7D)

The stronger classic statement uses the number of leaves of a suitable optimal Steiner tree. The library does not compute the optimum, leaf count, or achieved ratio. In particular, ordinary `float64` arithmetic, including legacy Dijkstra's rounded SSSP labels, is **not** a proof that the exact-real approximation inequality holds for every possible finite input. On small exactly represented integer fixtures, an independent exhaustive oracle can check it.

A successful result certifies a **terminal-covering original-edge tree**, not a globally optimal network and not an original-graph tour.

### 6.12.2. One original-edge hub and the complete KMB witness

```text
                  A
                  | AX:1
                  |
       C ---CX:1--X--BX:1--- B

Additional original terminal links:
  AB:3, BC:3, AC:3

Required terminals: A, B, C
Intermediate original vertex: X

Terminal shortest distances:
  d(A,B)=2 via AX + BX
  d(A,C)=2 via AX + CX
  d(B,C)=2 via BX + CX

Terminal metric closure MST:
  2 + 2 = 4  (synthetic terminal links; NOT published)

Expanded union, second Kruskal, and pruning:
  AX, BX, CX  (three original IDs, three unique edges)
  Final network weight = 1+1+1 = 3
```

| Original ID | Stored endpoints | Weight | Final status |
|:------------|:-----------------|-------:|:-------------|
| `AX`        | `A`, `X`         |      1 | Selected     |
| `BX`        | `B`, `X`         |      1 | Selected     |
| `CX`        | `C`, `X`         |      1 | Selected     |
| `AB`        | `A`, `B`         |      3 | Unused       |
| `BC`        | `B`, `C`         |      3 | Unused       |
| `AC`        | `A`, `C`         |      3 | Unused       |

This example exposes two common errors: returning closure weight 4 as `TotalWeight`, or publishing synthetic terminal-to-terminal edges instead of `AX`, `BX`, and `CX`.

### 6.12.3. Working public API

```go
func SteinerTree(g *core.Graph, terminals []string, opts ...SteinerOption) (*SteinerResult, error)

type SteinerPathStorage uint8
const (
    SteinerRecomputePaths SteinerPathStorage = iota
    SteinerRetainPaths
)

type SteinerOptions struct {
    Context      context.Context
    PathStorage  SteinerPathStorage
    MaxVertices  int
    MaxEdges     int
    MaxTerminals int
}

type SteinerOption func(*SteinerOptions) error

func DefaultSteinerOptions() SteinerOptions
func WithSteinerContext(ctx context.Context) SteinerOption
func WithSteinerPathStorage(mode SteinerPathStorage) SteinerOption
func WithSteinerMaxVertices(limit int) SteinerOption
func WithSteinerMaxEdges(limit int) SteinerOption
func WithSteinerMaxTerminals(limit int) SteinerOption

type SteinerResult struct {
    Terminals   []string
    Vertices    []string
    Edges       []core.Edge
    TotalWeight float64
}

func (r *SteinerResult) IsNil() bool
func (r *SteinerResult) Clone() *SteinerResult
func (r *SteinerResult) EdgeValues() ([]core.Edge, error)
```

`Terminals` is a **sorted independent copy**; the input terminal slice is never sorted in place. `Vertices` is the set of vertices **used in the chosen tree**, not all graph vertices. `Edges` is a detached, original-ID-sorted list of unique original `core.Edge` values; parallel physical records stay distinguishable by `ID`. The published `TotalWeight` is a checked `float64` left-fold of the unique **final** edges sorted by original `Edge.ID`, never of the metric closure.

A non-nil result has one connected acyclic network, all terminals covered, no self-loops or nonterminal leaves after KMB pruning, and `len(Edges)=len(Vertices)-1`. The singleton fast-path returns one vertex, zero edges, and zero total, but **only after** complete graph/domain validation. A two-terminal request returns one original-edge shortest-path tree through one tracked full-SSSP query.

### 6.12.4. Full KMB pipeline and what each selection proves

```text
Input graph G, requested terminals T
  |
  +-- applySteinerOptions: context / memory strategy / input-count caps
  |
  +-- validate weighted undirected capability and canonical terminal IDs
  |
  +-- one coherent core.Clone; detached original ID -> Edge inventory
  |      validate every edge: finite, nonnegative, non-directed, including
  |      loops and nonterminal disconnected islands; only then ignore loops
  |
  +-- k>=3: run public Dijkstra SSSP from k-1 terminal sources
  |      build only k*(k-1)/2 terminal-pair distances; no V*V matrix
  |
  +-- shared selectKruskalEdges on metric terminal closure
  |      select k-1 terminal pairs; do NOT sum closure MST weight
  |
  +-- for each selected terminal connection:
  |      reconstruct a source-to-target public vertex path
  |      map each step to a real original edge by min (Weight, Edge.ID)
  |      check incidence, original identity and every distance prefix
  |
  +-- union recovered edges by original Edge.ID, not endpoint pairs
  |      use only vertices touched by the paths and required terminals
  |
  +-- shared selectKruskalEdges on original-edge union
  |      remove cycles; do NOT sum an unnecessary union-tree total
  |
  +-- iterative degree queue removes all nonterminal leaves
  |
  +-- verify connected / acyclic / terminal-covering / original identities
  |
  +-- sort final originals by Edge.ID, check finite left-fold prefixes
  |
  +-- publish detached SteinerResult or (nil, error)
```

The single DSU/Kruskal selection kernel is shared with ordinary `Kruskal`. Prim keeps its separate edge-frontier heap. Neither ordinary Kruskal nor Prim is forced into the Steiner nonnegative-weight domain: **finite negative weights remain legal for ordinary MST/MSF**.

### 6.12.5. Full SSSP integration, complexity, and two memory modes

The implemented KMB adapter uses `dijkstra.Dijkstra` (full-SSSP), optionally `dijkstra.WithPathTracking()`. The new `dijkstra.Search` does expose original `EdgeIDs`, but also clones the graph for **each** target query and imposes stricter exact-binary64 intermediate-cost semantics. KMB therefore deliberately uses the same rounded full-SSSP objective for both closure and witness expansion. Do **not** substitute `Search` in one mode only or claim the two arithmetic domains are interchangeable.

The available `dijkstra.Result.PathTo(target)` returns vertex IDs, not predecessor edge IDs. Each traversed pair is resolved to an original undirected edge by the strict key `(Weight, Edge.ID)` and checked against every reconstructed SSSP prefix. That is a verified original-edge path; it is **not a claim** to expose the legacy Dijkstra kernel's private predecessor-edge event.

| Mode                              | Closure                        | Expansion                                                                   | Live predecessor history |
|:----------------------------------|:-------------------------------|:----------------------------------------------------------------------------|:-------------------------|
| `SteinerRecomputePaths` (default) | k−1 distance-only SSSP sources | Up to `g` additional tracked SSSP sources grouped by selected closure links | O(V)                     |
| `SteinerRetainPaths`              | k−1 tracked SSSP sources       | Reuse retained source results                                               | O(kV)                    |

`g` is the count of distinct lower-index closure sources used for expansion (`g ≤ k−1`). For `k=2`, **both modes** execute one tracked SSSP. Both return the same mathematical result and use the same numeric domain; neither silently switches to another strategy under memory pressure. `SteinerOptions.Context` supports cooperative cancellation **between** legacy SSSP calls and in the KMB-owned loops, but cannot interrupt the synchronous legacy Dijkstra kernel internally.

Let `Tsssp(V,E)` include Dijkstra validation and sorted graph accesses, `Eu,Vu` describe the expanded original-edge union, and `L` the total length of selected reconstructed paths. For `k≥3`, the end-to-end time is approximately:

$$ O((k-1+g)T_{SSSP}(V,E)+k^2\\log k+E_u\\log E_u+L+V_u+E_u), $$

plus the one original input Clone, sorted catalogs, and linear pruning. Live memory is O(V+E+k²) for recompute (besides temporary Dijkstra work) and O(V+E+k²+kV) for retain. These are algorithmic bounds, **not a hard process RSS ceiling**.

### 6.12.6. Domain, sentinels, numeric failures, and no partial publication

| Condition                                                         | Outcome                                                        |
|:------------------------------------------------------------------|:---------------------------------------------------------------|
| Nil/unweighted/directed graph                                     | Existing `ErrInvalidGraph` joined with precise policy sentinel |
| Empty terminal set                                                | `ErrEmptyTerminals`                                            |
| Duplicate terminal                                                | `ErrDuplicateTerminal`                                         |
| Empty or missing terminal ID                                      | `core.ErrEmptyVertexID` / `core.ErrVertexNotFound`             |
| Finite negative original weight (even a loop or unrelated island) | `ErrNegativeWeight`                                            |
| NaN or infinity anywhere in original edge catalog                 | `ErrNaNInfWeight`                                              |
| Terminals in different connected components                       | `ErrDisconnectedTerminals`                                     |
| Original-ID/path witness mismatch                                 | `ErrInvalidWitness`; lower-level causes remain wrapped         |
| Invalid final tree / malformed supplied tree                      | `ErrInvalidSteinerResult`                                      |
| Non-finite published network or walk left-fold                    | `ErrWeightOverflow`                                            |
| Explicit positive input-count limit exceeded                      | `ErrResourceLimit`                                             |
| Unrepresentable element count or slice byte size                  | `ErrSizeOverflow`                                              |
| Cancelled context in KMB-owned phase                              | Original `context.Canceled` or `context.DeadlineExceeded`      |

A true Dijkstra arithmetic overflow remains a Dijkstra numeric error; it is **not** converted to `ErrDisconnectedTerminals`. The first applicable admission phase determines error precedence. All successful results own their slices; **any failure returns `nil, err`**. `Clone` preserves scalar fields and deep-copies mutable slices; `EdgeValues` returns an independent slice and classifies nil-receiver access with `ErrNilResult`.

**Floating-point counterexample:** suppose `AX=BX=CX=w`, where `w=2^1022` in binary64. Each terminal-pair distance `2w` is finite. Summing two closure-MST edges gives `4w` and overflows, but the final original network weighs `3w`, which is finite. Therefore closure/union selections **must never require intermediate scalar totals**. Conversely, a truly overflowing final original-edge fold returns `ErrWeightOverflow`, not a successful `+Inf`.

Canonical final summation and shortest-path prefix summation deliberately use **different edge orders**. With fractional weights they need not have identical rounded bits, even when the recovered original edges and checked prefix distances are valid; only the respective contracts are asserted.

---

## 6.13. SteinerWalk: Optimal Edge-Covering on a Given Tree

### 6.13.1. Mathematical object and two endpoint regimes

`SteinerWalk` consumes a `SteinerResult` as a **caller-owned, potentially modified value**, not as an immutable KMB certificate. It validates the entire network before travel. The input may be any structurally valid undirected finite nonnegative tree covering its declared terminals; unlike a newly pruned KMB tree, **nonterminal leaves are allowed and must be traversed**. The walk does not ask the source graph for shortcuts.

```go
func SteinerWalk(tree *SteinerResult, startID, endID string) (*WalkResult, error)

type WalkResult struct {
    StartID       string
    EndID         string
    Vertices      []string
    EdgeIDs       []string
    NetworkWeight float64
    Distance      float64
}

func (r *WalkResult) IsNil() bool
func (r *WalkResult) Clone() *WalkResult
```

- **Closed:** `startID == endID`. Every tree edge is traversed **twice** (once in each direction).
- **Fixed-end:** `startID != endID`. The unique tree `start→end` spine is traversed **once**; all off-spine tree edges are traversed **twice**.
- Both endpoint IDs are **mandatory, nonempty, and present** in the supplied tree. Empty `endID` does not mean closed mode.
- For a one-vertex tree, equal endpoints yield one visited vertex, no passages, zero weight, zero distance.

For a tree of unique-edge weight `W` with fixed endpoints `s,t`, the exact arithmetic identity is:

$$ L_{closed}=2W,\qquad L_{fixed}(s,t)=2W-d_{tree}(s,t). $$

![Tree walk cost law](https://latex.codecogs.com/svg.image?L_%7Bclosed%7D%3D2W%2C%5Cquad%20L_%7Bfixed%7D%28s%2Ct%29%3D2W-d_%7Btree%7D%28s%2Ct%29)

Each off-spine tree edge forms a cut that can only be crossed by a return to its attachment, hence two passages. On the unique start-to-end spine, one passage is necessary and sufficient. This proves **optimality for covering all edges of the specified tree with the specified endpoints**. It does not prove global optimality among routes allowed to use additional edges in the original graph.

### 6.13.2. Original-edge walk witness: hub closed and fixed-end

```text
Original tree:

  A ----AX:1---- X ----BX:1---- B
                 |
                CX:1
                 |
                 C

Closed A -> A:
  Vertices = [A X B X C X A]
  EdgeIDs  = [AX BX BX CX CX AX]
  Unique network weight W = 3
  Actual passage distance  = 6

Fixed-end A -> B:
  Vertices = [A X C X B]
  EdgeIDs  = [AX CX CX BX]
  Unique network weight W = 3
  Actual passage distance  = 4
```

Every element `EdgeIDs[i]` connects `Vertices[i]` with `Vertices[i+1]`, allowing travel in either orientation of the original undirected edge. In particular:

```text
len(WalkResult.Vertices) == len(WalkResult.EdgeIDs) + 1
WalkResult.Vertices[0] == WalkResult.StartID
WalkResult.Vertices[last] == WalkResult.EndID
```

Repeating the same original ID in `EdgeIDs` is essential traversal evidence, **not duplicate installation**. `NetworkWeight` is the cost of unique original tree edges; `Distance` sums the actual recorded passages. In the runtime, `Distance` is checked by an ordered left-fold over the published passage witness and is **not computed as an unchecked `2*W-d` expression**.

### 6.13.3. Caller-owned mutable tree validation

Before traversing, `SteinerWalk` copies and canonicalizes the supplied `Terminals`, `Vertices`, and `Edges` locally. It validates the following without consulting the original graph:

1. Nonempty, unique terminal and vertex IDs; terminals are all present in vertices.
2. Unique, nonempty original edge IDs, and two declared endpoints per edge.
3. Finite, nonnegative undirected weights; no loops or self-connections.
4. All edge endpoints exist in the declared vertex set; cycles and disconnected components are forbidden.
5. The tree cardinality `|Edges|=|Vertices|-1` and one connected DSU component.
6. `TotalWeight` exactly matches the checked sum of copied edges sorted by `Edge.ID`.

Valid permutations of caller-owned slices are accepted; `SteinerWalk` never mutates them or repairs a stale total. Any structural, numeric, or endpoint failure returns `nil, err`. `ErrInvalidSteinerResult` describes malformed topology or stale cost; `ErrWeightOverflow` describes a nonfinite actual passage prefix. `core.ErrEmptyVertexID`, `core.ErrVertexNotFound`, `core.ErrEmptyEdgeID`, `ErrDirectedEdge`, `ErrNegativeWeight` and `ErrNaNInfWeight` can additionally classify the underlying invalid field via `errors.Is`.

`SteinerWalk` **cannot** establish whether the IDs ever belonged to a particular historical `core.Graph` or whether the input tree came from KMB; it can only validate the supplied value. Callers must not mutate the tree concurrently with the call.

### 6.13.4. Iterative traversal, ordering, and numeric overflow

```text
FUNCTION SteinerWalk(tree, startID, endID):
    require non-nil tree, nonempty explicit endpoints
    copy and sort caller-owned ID/edge slices; verify canonical tree cost
    require both endpoints in the validated vertex set

    spine = unique tree path from start to end (iterative BFS)
    mark each spine Edge.ID
    steps = E + (E - len(spine_edges))    # checked integer addition
    allocate route witness for steps passages and steps+1 vertices

    FOR each spine vertex in path order:
        FOR each off-spine neighbor in (neighbor ID, Edge.ID) order:
            iterative DFS of that branch using explicit stack frames:
                emit outward original-ID step and checked cost
                emit return original-ID step after child subtree completes
        IF next spine vertex exists:
            emit the one spine-edge step (no return)

    require len(EdgeIDs)=steps, len(Vertices)=steps+1, last=endID
    RETURN detached WalkResult or nil/error
```

The local graph preparation sorts copied edges by ID and adjacency by neighbor ID then original edge ID. Its traversal uses no recursive call, making a 30,000-vertex chain a valid deep-path workload. Total complexity includes validation and copies:

$$ O(V\log V + E\log E + \sum_v \deg(v)\log(deg(v)+1) + S) $$

in time and O(V+E+S) space, where `S ≤ 2E` is the number of passages. No context cancellation API exists for `SteinerWalk`.

**Important numeric distinction:** if a tree has two spokes of weight `w=2^1022`, its unique network weight `2w` is finite. A closed walk requires `4w` and fails with `ErrWeightOverflow`, while a fixed-end walk along the two-spoke spine covers each edge once and succeeds with finite `2w`. The implementation checks actual passages, so it does not falsely overflow merely because an intermediate theoretical `2W` would overflow for an otherwise finite fixed-end itinerary.

---

## 6.14. Complete Go Usage: Build → Optimize → Traverse

This compact self-contained executable demonstrates both public operations on the original hub graph. The heavier chiplet, emergency-relay and fiber multigraph scenarios live in [`examples/mst_steiner_chiplet_routing.go`](../examples/mst_steiner_chiplet_routing.go), [`examples/mst_steiner_emergency_mesh.go`](../examples/mst_steiner_emergency_mesh.go), and [`examples/mst_steiner_fiber_audit.go`](../examples/mst_steiner_fiber_audit.go).

```go
package main

import (
    "fmt"
    "github.com/lvlath/go/core"
    "github.com/lvlath/go/mst"
)

func main() {
    graph, err := core.NewGraph(core.WithWeighted())
    if err != nil { panic(err) }

    for _, e := range []struct {
        id, from, to string
        weight float64
    }{
        {"AX", "A", "X", 1}, {"BX", "B", "X", 1}, {"CX", "C", "X", 1},
        {"AB", "A", "B", 3}, {"BC", "B", "C", 3}, {"AC", "A", "C", 3},
    } {
        if _, err := graph.AddEdge(e.from, e.to, e.weight, core.WithID(e.id)); err != nil {
            panic(err)
        }
    }

    network, err := mst.SteinerTree(graph, []string{"C", "A", "B"})
    if err != nil { panic(err) }
    walk, err := mst.SteinerWalk(network, "A", "B")
    if err != nil { panic(err) }

    fmt.Println(network.Terminals, network.TotalWeight, len(network.Edges))
    fmt.Println(walk.Vertices)
    fmt.Println(walk.EdgeIDs, walk.NetworkWeight, walk.Distance)
    // Output:
    // [A B C] 3 3
    // [A X C X B]
    // [AX CX CX BX] 3 4
}
```

The runnable package examples [`mst/example_test.go`](../mst/example_test.go) additionally demonstrate original parallel identities, memory strategy parity, a caller-provided tree with nonterminal leaves, and fixed-end passage reproduction. The three standalone programs include explicit construction/error checks and provide audit-ready output. Run them **separately**, since each is an independent `package main`:

```bash
go run ./examples/mst_steiner_chiplet_routing.go
go run ./examples/mst_steiner_emergency_mesh.go
go run ./examples/mst_steiner_fiber_audit.go

go test ./mst -run '^TestMSTSteinerScenarios_Run$' -count=1
```

Do not fabricate Go Playground permalink badges for these files: a URL is useful only after the exact code has been uploaded and verified at that location.

---

## 6.15. Performance and Verification Regimes

The public end-to-end benchmarks deliberately **change the dominant algorithmic pressure**, rather than inventing private per-phase timing APIs. Graph fixtures and terminals are constructed before the timed loop. Each benchmark uses `b.ReportAllocs`, preserves the returned result, and does not call formatters or construct topology in `b.N`.

| Benchmark regime                                      | Dominant work                                                         | What is included in `ns/op`                                          |
|:------------------------------------------------------|:----------------------------------------------------------------------|:---------------------------------------------------------------------|
| `BenchmarkSteinerTree_AdmissionSingleton`             | One coherent Clone, catalog/terminal admission, singleton publication | Entire public call, not isolated Clone                               |
| `BenchmarkSteinerTree_SSSPAndClosure/pair-k02`        | One tracked shortest-path run                                         | Entire two-terminal call                                             |
| `.../closure-k04-recompute`, `.../closure-k04-retain` | Terminal closure + selected original paths                            | Full KMB, both memory policies                                       |
| `.../closure-k12-recompute`, `.../closure-k12-retain` | k−1 repeated SSSP source passes and O(k²) closure                     | Full KMB with larger terminal set                                    |
| `BenchmarkSteinerTree_CycleUnionAndPruning`           | Both Kruskal phases and nonterminal leaf pruning                      | Full six-edge structural fixture; not a large-scale throughput claim |
| `BenchmarkSteinerWalk_ClosedAndFixed`                 | Mutable-result validation plus O(S) actual passage output             | Closed/fixed-end walks over the same deep-spine branch tree          |

Benchmark the package with a compatible Go toolchain and the actual source tree:

```bash
go test ./mst -run '^$' -bench '^BenchmarkSteiner(Tree|Walk)_' -benchmem -count=3
go test ./mst -count=1
go test -race ./mst -count=1
go test ./core ./dijkstra ./mst -count=1
go test ./...
go vet ./...
golangci-lint run
git diff --check
```

`golangci-lint` must use the version/configuration accepted by the repository. Benchmark numbers are machine-specific: report actual `ns/op`, `B/op`, `allocs/op`, Go version, CPU, terminal count, graph shape, and storage mode. Do not report phase-only performance improvements from differences between full-call timings.

### Engineering pitfalls to avoid

1. **MST vs Steiner:** exact full-vertex minimum-cost spanning and approximate terminal-only network design solve different objectives.
2. **Network vs walk:** `SteinerResult.TotalWeight` sums unique original installed edges; `WalkResult.Distance` sums repeat visits.
3. **Legacy SSSP vs target Search:** edge-aware `PathResult` is valuable but cannot be substituted silently into KMB when its exact-binary64 storage/clone semantics differ.
4. **Original identity:** never publish closure IDs, deduplicate parallel links by endpoint pair, or infer next vertex blindly from `core.Edge.To`.
5. **Floating-point arithmetic:** do not impose epsilon comparisons on sorting or heap order, or require closure aggregate sums that may overflow while the final network remains finite.
6. **Caller-owned trees:** no KMB provenance is guaranteed on a mutable `SteinerResult`; `SteinerWalk` must reject cycles, disconnectedness, stale totals and invalid weights.
7. **Route optimality:** the fixed-tree covering formula does not optimize a route over unused original graph edges.
8. **Complexity:** use representative sparse/dense original graphs and growing terminal counts. Larger k expands O(k²) closure storage and multiplies SSSP work.

---

**lvlath/mst**: exact Prim/Kruskal MST/MSF, original-edge graph Steiner approximation, and independently validated closed/fixed-end covering walks—with distinct result and numeric contracts.

> Next: [7. Max-Flow: Ford-Fulkerson / Edmonds-Karp / Dinic →](FLOW.md)
