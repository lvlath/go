<!--
  lvlath - Repository Documentation

  Purpose:
    Repository-level specification, tutorial, and contract map for lvlath/tsp.
    It explains the matrix-backed TSP/ATSP model, deterministic solver laws,
    exact and heuristic publication rules, Christofides matching law, Eulerian
    multigraph law, numeric policy, result ownership, partial-result semantics,
    and production usage patterns.

  Contract status:
    - Public behaviors described here are part of the lvlath/tsp contract.
    - Determinism rules described here are part of the lvlath/tsp contract.
    - Numeric and sentinel-error rules described here are part of the lvlath/tsp contract.
    - Result ownership, Clone, and VertexTour semantics are part of the public contract.
    - Approximation-ratio publication rules are part of the public contract.
    - Any incompatible change must be explicit, documented, and versioned.

  Scope:
    - Deterministic matrix-backed TSP and ATSP solving.
    - Exact Held-Karp dynamic programming under explicit size guards.
    - Exact Branch-and-Bound when search completes.
    - Christofides for symmetric complete metric TSP.
    - Exact Blossom MWPM and explicit Greedy matching policies inside Christofides.
    - Symmetric and directed local-search regimes.
    - Eulerian multigraph construction, parallel-edge-safe Hierholzer traversal,
      shortcutting, tour validation, and result publication.
    - Matrix, numeric, ownership, error, and concurrency contracts.

  License:
    The lvlath repository is licensed under AGPL-3.0-only. See LICENSE.
-->

# 11. Traveling Salesman Problem

> **Package:** `lvlath/tsp` | **Focus:** Deterministic Matrix TSP, Exact Search, Christofides, Blossom MWPM, Local Search, Contract-Safe Results

The `lvlath/tsp` package is a deterministic, contract-driven routing kernel over `matrix.Matrix`. It solves TSP and supported ATSP regimes by making algorithmic strength explicit: exact solvers publish optimality only when they complete, Christofides publishes a formal approximation ratio only under exact Blossom matching, Greedy matching is a selected weaker mode, and local search returns improved tours without global optimality claims.

The package is designed for production pipelines where matrix order, error identity, numeric validity, result ownership, and deterministic tie-breaking are not implementation details. They are part of the public contract.

---

## 11.1. What & Why: The Laws of `tsp`

TSP models a closed route that visits every required vertex exactly once and returns to a fixed start. The cost matrix may represent kilometers, minutes, millisecond actuator latency, fuel burn, temperature-degradation risk, security exposure, or any non-negative application-specific penalty.

### The Seven Laws

1. **Matrix-Source Law**

   Solver kernels consume `matrix.Matrix`. Graph data may be adapted into a matrix by a facade, but Held-Karp, Branch-and-Bound, Christofides, matching, Eulerian, and local-search kernels operate on matrix indices.

   Consequence: row and column order define the algorithmic vertex order. Optional `IDs` are labels over that order, not an alternate topology.

2. **Explicit Policy Law**

   `Options` is a runtime policy object. It selects algorithm, symmetry, matching engine, local-search limits, metric-closure behavior, deterministic randomness, exact-solver guards, and timeout policy. It must not silently mutate topology or silently substitute a weaker algorithm.

3. **Determinism Law**

   For the same matrix, IDs, options, seed, and package version, successful results are deterministic. Stable behavior comes from matrix order, vertex-index tie-breaks, dense edge IDs, deterministic matching selection, deterministic Eulerian traversal, and documented local-search enumeration.

4. **Numeric Discipline Law**

   NaN, `-Inf`, and negative finite weights are invalid. `+Inf` is a missing-edge sentinel only before metric closure. Final solver kernels require complete finite matrices.

5. **Matching Publication Law**

   `BlossomMatch` means exact minimum-weight perfect matching on Christofides odd MST vertices. Only this mode supports publishing the Christofides `1.5` approximation ratio. `GreedyMatch` is deterministic and intentional, but it publishes no formal ratio.

6. **Result Ownership Law**

   `Result` owns detached `Tour` and `IDs` slices. `Clone()` deep-copies mutable result surfaces. `VertexTour()` projects indices through the detached ID mapping without recomputing a route.

7. **Partial-Result Safety Law**

   Validation and construction failures return `nil` result plus error. Governed timeout exceptions are explicit: Branch-and-Bound may return an incumbent, and direct local-search wrappers may return the current feasible tour, but such results must not be marked globally optimal.

---

## 11.2. Domain Scope & Non-Goals

### Solved domain

`lvlath/tsp` covers:
- exact Held-Karp dynamic programming for guarded small instances;
- exact Branch-and-Bound with deterministic branching and optional lower bounds;
- Christofides for symmetric complete metric TSP;
- exact Blossom MWPM for Christofides odd-vertex matching;
- deterministic Greedy matching as an explicit weaker Christofides mode;
- 2-opt and 3-opt local search, including restricted directed behavior;
- optional metric closure for missing `+Inf` edges before final solving;
- Eulerian multigraph construction, parallel-edge-safe circuit traversal, and shortcutting;
- tour validation, rotation, orientation canonicalization, cost evaluation, and ID projection.

### Non-goals

The package intentionally does not provide:
- hidden exact-to-heuristic fallback;
- hidden Blossom-to-Greedy fallback;
- silent solver substitution after a selected policy fails;
- Concorde-style branch-and-cut;
- sparse graph-native TSP kernels;
- final solving over unresolved `+Inf`;
- full arbitrary-reconnection ATSP 3-opt;
- concurrent mutation support for caller-owned matrix or graph inputs;
- snapshot isolation against external input mutation;
- machine-readable error strings;
- approximation-ratio claims for Greedy matching or local search.

### Graph boundary

`core.Graph` represents topology. `tsp` kernels represent matrix optimization. A graph facade may produce a `matrix.Matrix`; after that boundary, solver semantics are matrix semantics. This avoids mixed-edge ambiguity, graph mutation races, and endpoint-direction confusion inside solver kernels.

---

## 11.3. Mathematical Formulation

### Closed-tour objective

Let `V = {0, ..., n-1}` and let `π` be a permutation of `V` with `π_0 = s`. A closed tour appends `π_n = π_0`.

$$ \operatorname{cost}(\pi)=\sum_{i=0}^{n-1} c(\pi_i,\pi_{i+1}), \qquad \pi_n=\pi_0 $$

The public tour surface must satisfy:
$$ Tour[0]=StartVertex,\quad Tour[n]=StartVertex,\quad Tour[0:n]\text{ is a permutation of }V $$

For symmetric TSP:
$$ c(u,v)=c(v,u) $$

For ATSP:

$$ c(u,v)\text{ may differ from }c(v,u) $$

### Held-Karp dynamic program

Let `DP[S][j]` be the minimum cost to start at `s`, visit exactly the subset `S`, and finish at `j`.

$$ DP[S][j]=\min_{i\in S\setminus\{j\}}\left(DP[S\setminus\{j\}][i]+c(i,j)\right) $$

Initialization and closure:
$$ DP[\{s\}][s]=0 $$

$$ OPT=\min_{j\in V\setminus\{s\}}\left(DP[V][j]+c(j,s)\right) $$

Complexity:
$$ T=O(n^2 2^n), \qquad S=O(n2^n) $$

The implementation is size-guarded by exact-solver policy. `Auto` may use `MaxExactN` to decide whether exact Held-Karp is admissible for automatic dispatch.

### Branch-and-Bound exactness

Branch-and-Bound searches over partial paths. A lower bound must be admissible:
$$ LB(P)\le OPT(P) $$

A branch can be pruned only when it cannot improve the incumbent:
$$ LB(P)\ge Cost(Incumbent)-\varepsilon $$

Exactness is a completion property. If search is stopped by a time limit, a feasible incumbent is not a proof of optimality.

### Degree-1 and one-tree lower-bound intuition

A Hamiltonian cycle gives each vertex one incoming and one outgoing edge. Degree-relaxation bounds use this necessity to estimate the cheapest possible completion. For symmetric one-tree relaxation, reduced costs are:
$$ c'(u,v)=c(u,v)+\pi_u+\pi_v $$

The subgradient signal is degree violation:
$$ g_v=deg(v)-2 $$

Every Hamiltonian cycle is a feasible one-tree, but not every one-tree is a Hamiltonian cycle. That is why one-tree is a lower bound, not a route constructor.

### Christofides law

For symmetric complete metric TSP:
$$ T = MST(G) $$

$$ O=\{v\in V \mid deg_T(v)\equiv 1 \pmod 2\} $$

$$ M = MWPM(G[O]) $$

$$ H = T \cup M $$

Every vertex in `H` has even degree:
$$ \forall v\in V,\quad deg_H(v)\equiv0\pmod2 $$

Shortcutting an Eulerian circuit produces a Hamiltonian tour because the metric triangle inequality prevents shortcut cost from increasing.

Formal ratio, published only with exact Blossom MWPM:
$$ cost(C)\le\frac{3}{2}cost(OPT) $$

### Eulerian multigraph law

Christofides uses an undirected multigraph. Parallel edges are legal and must be consumed exactly once. Half-edge pairing is directional:
$$ (u\rightarrow v)\text{ pairs with }(v\rightarrow u) $$

It is incorrect to pair two grouped half-edges of the same direction, even if they share the same unordered pair.

### Local-search move laws

For symmetric 2-opt, removing edges `(a,b)` and `(c,d)` and reconnecting `(a,c)` and `(b,d)` changes cost by:
$$ \Delta = c(a,c)+c(b,d)-c(a,b)-c(c,d) $$

A move is accepted only when:
$$ \Delta < -\varepsilon $$

For directed 2-opt*, the move is orientation-preserving: it rewires tails rather than reversing a segment. The directed mode must not be described as full arbitrary ATSP 3-opt.

### Complexity summary

| Phase                           | Time                                         | Space                  | Publication meaning                                |
|:--------------------------------|:---------------------------------------------|:-----------------------|:---------------------------------------------------|
| Option validation               | `O(1)`                                       | `O(1)`                 | Rejects invalid policy before kernel trust.        |
| Matrix validation               | `O(n²)`                                      | `O(1)` to `O(n²)`      | Enforces numeric and symmetry/completeness policy. |
| Metric closure                  | `O(n³)`                                      | `O(n²)`                | Resolves `+Inf` before final kernels.              |
| Held-Karp                       | `O(n²2ⁿ)`                                    | `O(n2ⁿ)`               | Exact optimal result on completion.                |
| Branch-and-Bound                | exponential worst case                       | `O(n²)+O(n)`           | Exact only when completed.                         |
| One-tree bound iteration        | `O(n²)`                                      | `O(n²)`                | Lower-bound computation, not a route.              |
| Christofides MST                | `O(n²)` dense regime                         | adjacency buffers      | Approximation pipeline stage.                      |
| Blossom MWPM                    | polynomial dense exact matching              | matching-local buffers | Required for 1.5 ratio publication.                |
| Eulerian circuit                | `O(V+E)`                                     | `O(V+E)`               | Consumes every multigraph edge exactly once.       |
| Shortcut and result publication | `O(E+n)`                                     | `O(n)`                 | Produces detached Hamiltonian tour.                |
| 2-opt                           | `O(iterations*n²)`                           | `O(n)`                 | Heuristic, no global certificate.                  |
| 3-opt                           | `O(iterations*n³)` in symmetric scan regimes | `O(n)`                 | Heuristic, no global certificate.                  |

---

## 11.4. Public API & Result Contract

### Canonical facades and direct wrappers

```go
func SolveMatrix(dist matrix.Matrix, ids []string, opts Options) (*Result, error)
func SolveGraph(g *core.Graph, opts Options) (*Result, error)

func HeldKarp(dist matrix.Matrix, opts Options) (*Result, error)
func ChristofidesSolve(dist matrix.Matrix, opts Options) (*Result, error)
func BranchAndBoundSolve(dist matrix.Matrix, opts Options) (*Result, error)
func TwoOptSearch(dist matrix.Matrix, initTour []int, opts Options) (*Result, error)
func ThreeOptSearch(dist matrix.Matrix, initTour []int, opts Options) (*Result, error)
```

Direct wrappers force their own algorithm policy and publish `Result`. They are public convenience wrappers, not duplicate implementations.

### Utility APIs

```go
func DefaultOptions() Options

func ValidatePermutation(perm []int, n int) error
func RotateTourToStart(tour []int, start int) ([]int, error)
func CanonicalizeOrientationInPlace(tour []int) error

func EulerianCircuit(adj [][]int, start int) ([]int, error)

func DefaultOneTreeConfig() OneTreeConfig

func OneTreeLowerBound(
    dist matrix.Matrix,
    root int,
    symmetric bool,
    cfg OneTreeConfig,
) (lb float64, degrees []int, err error)
```

### Public type surface

```go
type Algorithm int
type MatchingAlgo int
type BoundAlgo int

type Result struct {
    Tour []int
    Cost float64
    IDs  []string

    Algorithm Algorithm
    Exact     bool
    Optimal   bool
    TimedOut  bool

    MetricClosureApplied bool
    Symmetric             bool

    ApproximationRatio float64

    Iterations    int
    NodesExpanded int
}

func (r *Result) IsNil() bool
func (r *Result) Clone() *Result
func (r *Result) VertexTour() ([]string, error)
```

### Result states

| State                                         | Result  | Error                | Contract                                    |
|:----------------------------------------------|:--------|:---------------------|:--------------------------------------------|
| Completed exact result                        | non-nil | nil                  | `Exact=true`, `Optimal=true`.               |
| Completed Christofides with Blossom           | non-nil | nil                  | `ApproximationRatio=1.5`.                   |
| Completed Christofides with Greedy            | non-nil | nil                  | `ApproximationRatio=0`.                     |
| Completed local search                        | non-nil | nil                  | `Exact=false`, `Optimal=false`.             |
| Branch-and-Bound timeout with incumbent       | non-nil | wraps `ErrTimeLimit` | Feasible but not certified optimal.         |
| Direct local-search timeout with current tour | non-nil | wraps `ErrTimeLimit` | Feasible current tour, no optimality claim. |
| Validation failure                            | nil     | sentinel-classified  | No safe route published.                    |
| Kernel construction failure                   | nil     | sentinel-classified  | No safe route published.                    |
| Timeout without feasible tour                 | nil     | wraps `ErrTimeLimit` | No publishable tour.                        |

### ID projection law

`VertexTour()` maps `Tour` indices through `IDs` in the solver-published order. It does not sort IDs, recompute costs, canonicalize orientation, or infer missing labels.

---

## 11.5. Options, Numeric Policy & Error Law

### Options

```go
type Options struct {
    StartVertex int
    Algo        Algorithm
    Symmetric   bool

    MatchingAlgo MatchingAlgo
    BoundAlgo    BoundAlgo

    RunMetricClosure bool

    EnableLocalSearch bool
    TwoOptMaxIters    int
    ThreeOptMaxMoves  int

    BestImprovement     bool
    ShuffleNeighborhood bool

    Eps       float64
    TimeLimit time.Duration
    Seed      int64

    MaxExactN int
}
```

Options describe solver policy. They must not mutate topology and must not silently replace the selected algorithm.

| Option field          | Contract                                                                 |
|:----------------------|:-------------------------------------------------------------------------|
| `Algo`                | Selects the solver family.                                               |
| `Symmetric`           | Declares whether symmetric TSP rules are required.                       |
| `StartVertex`         | Fixes tour start and closure vertex.                                     |
| `MatchingAlgo`        | Selects `BlossomMatch` or `GreedyMatch` for Christofides.                |
| `BoundAlgo`           | Selects Branch-and-Bound lower-bound policy.                             |
| `RunMetricClosure`    | Allows `+Inf` missing edges to be resolved before final solving.         |
| `EnableLocalSearch`   | Enables dispatcher-managed local-search post-processing where supported. |
| `TwoOptMaxIters`      | Bounds accepted 2-opt moves; zero means unlimited.                       |
| `ThreeOptMaxMoves`    | Bounds accepted 3-opt moves; zero means unlimited.                       |
| `BestImprovement`     | Controls local-search improvement policy where supported.                |
| `ShuffleNeighborhood` | Enables deterministic seed-controlled neighborhood shuffling.            |
| `Eps`                 | Controls strict improvement and equality decisions.                      |
| `TimeLimit`           | Governs supported soft time-budget behavior; zero means unlimited.       |
| `Seed`                | Controls deterministic randomized components.                            |
| `MaxExactN`           | Bounds exact Held-Karp selection when `Algo==Auto`.                      |

`DefaultOptions()` selects Christofides, symmetric input, Greedy matching, no metric closure, and local search enabled. `BlossomMatch` is available as an explicit caller-selected policy when the caller requires exact MWPM semantics and formal Christofides ratio publication.

### Numeric policy

- `DefaultEps = 1e-12`.
- `NoApproximationRatio = 0.0`.
- `ChristofidesApproximationRatio = 1.5`.
- NaN is invalid.
- `-Inf` is invalid.
- Negative finite weights are invalid.
- `+Inf` is a pre-kernel missing-edge sentinel only when closure policy allows it.
- Final kernels require finite complete costs.
- Strict symmetry is required by `Christofides`, by `OneTreeBound`, and whenever `Symmetric` is true.
- Costs are stabilized for publication by package rounding policy.

### Error law

Use `errors.Is`. Do not parse strings.

```go
if errors.Is(err, tsp.ErrTimeLimit) {
    // Handle governed timeout behavior.
}
```

Errors may be wrapped to preserve context. Sentinel identity remains the machine contract.

---

## 11.6. Algorithmic Architecture & Pseudocode

### SolveMatrix pipeline

```text
FUNCTION SolveMatrix(dist, ids, opts):
  validate Options
  validate matrix shape
  validate StartVertex
  validate IDs if provided
  copy or prepare final solver matrix
  if RunMetricClosure:
    compute detached metric closure
  reject unresolved +Inf before final kernel
  choose algorithm, resolving Auto if explicitly selected
  run selected kernel
  reject unsafe nil or invalid kernel result
  attach detached IDs
  attach MetricClosureApplied and Symmetric metadata
  validate closed tour
  publish detached Result
```

### Held-Karp

```text
FUNCTION HeldKarp(dist, opts):
  force Algorithm = ExactHeldKarp
  validate finite complete matrix
  validate StartVertex
  reject size above exact guard

  initialize DP[mask][end] = +Inf
  initialize parent table
  DP[startMask][start] = 0

  for mask in increasing bitmask order:
    for end in increasing vertex order:
      skip if end not in mask
      for next in increasing vertex order:
        skip if next already in mask
        candidate = DP[mask][end] + c(end,next)
        if candidate improves by Eps policy:
          update DP and parent

  close best tour through start
  reconstruct parent path
  rotate to StartVertex
  canonicalize orientation
  validate and publish Exact=true, Optimal=true
```

### Branch-and-Bound

```text
FUNCTION BranchAndBound(dist, opts):
  force Algorithm = BranchAndBound
  copy complete weights
  validate StartVertex
  initialize search buffers
  precompute min-in/min-out and neighbor order
  optionally seed an incumbent
  optionally compute root one-tree lower bound

  DFS(last, depth, cost):
    if deadline expired:
      mark stopped
      return

    bound = lowerBound(cost, last)
    if incumbent exists and bound >= incumbent - Eps:
      prune

    if depth == n:
      close candidate through start
      update incumbent if better
      return

    for next in deterministic neighbor order:
      skip visited vertices
      push next
      DFS(next, depth+1, cost+c(last,next))
      pop next

  if completed:
    publish Exact=true, Optimal=true
  if timeout and incumbent exists:
    publish Exact=true, Optimal=false, TimedOut=true plus ErrTimeLimit
  if timeout and no incumbent:
    return nil, ErrTimeLimit
```

### Christofides

```text
FUNCTION Christofides(dist, opts):
  validate options
  require symmetric complete matrix
  validate StartVertex

  T = MinimumSpanningTree(dist)
  odd = vertices with odd degree in T

  if MatchingAlgo == BlossomMatch:
    M = exact MWPM(odd)
    ratio = 1.5
  else if MatchingAlgo == GreedyMatch:
    M = deterministic greedy matching(odd)
    ratio = 0.0
  else:
    reject options

  H = T union M as an undirected multigraph
  H = canonicalizeUndirectedMultigraph(H)

  eulerian = EulerianCircuit(H, StartVertex)
  tour = ShortcutEulerianToHamiltonian(eulerian, n, StartVertex)
  CanonicalizeOrientationInPlace(tour)
  cost = tourCost(dist, tour)

  publish detached Result
```

### Blossom MWPM

```text
FUNCTION BlossomMWPM(problem):
  validate even cardinality
  validate finite symmetric local weights
  transform costs to profits for max-weight matching dual form
  initialize singleton top-level nodes and duals
  while matchedPairs < n/2:
    rebuild alternating forest from exposed top-level nodes
    loop:
      scan queued outer nodes for tight grow/shrink/augment events
      if grow event:
        label target inner and matched continuation outer
      else if same-root outer join:
        shrink odd cycle into blossom
      else if different-root outer join:
        compose endpoint-aware top-level path
        lift path through nested blossoms
        try deterministic lift candidates transactionally
        flip alternating path
        refresh live blossom bases
        break
      else:
        delta = smallest non-negative dual movement
        apply dual update
        expand zero-dual inner blossom when selected
  verify symmetry, dual feasibility, and perfect cardinality
  export local match array and cost
```

### Eulerian circuit

```text
FUNCTION EulerianCircuit(adj, start):
  validate non-empty adjacency
  validate start in range
  validate every vertex has even degree
  validate every endpoint is in range

  for each half-edge u -> v in adjacency order:
    if pending[v -> u] exists:
      pair current half-edge with opposite directed half-edge
    else:
      store current half-edge in pending[u -> v]

  if any pending half-edge remains:
    return ErrNonEulerian

  stack = [start]
  while stack not empty:
    u = stack.top
    if u has unused half-edge:
      mark edge and twin as used
      stack.push(to[edge])
    else:
      circuit.append(u)
      stack.pop()

  reverse circuit
  require circuit starts and ends at start
  require every half-edge consumed
  require len(circuit) == undirectedEdgeCount + 1
  return circuit
```

### Directed 2-opt* boundary

```text
FUNCTION DirectedTwoOptStar(tour):
  build successor representation
  evaluate directed tail-swap candidates
  accept only delta < -Eps
  rewire successors without reversing segment orientation
  reconstruct closed tour from StartVertex
  publish heuristic result without exact or optimal certificate
```

---

## 11.7. ASCII Diagrams

### 11.7.1. Christofides route pressure on a cold-chain dispatch matrix

This diagram explains how the cold-chain example works. The full solver receives a complete 10x10 matrix. The diagram shows the final deterministic route reported by the example, not every matrix edge.

```text
╔══════════════════════════════════════════════════════════════════════════════╗
║ BIOPHARMACEUTICAL COLD-CHAIN ROUTE                                           ║
║ Cost unit: minutes, thermal envelope: 240 minutes                            ║
╠════╦════════════════════════════╦════════════════════════════════════════════╣
║ ID ║ Stop                       ║ Operational meaning                        ║
╠════╬════════════════════════════╬════════════════════════════════════════════╣
║ 00 ║ Regional Bio-Hub           ║ start and closure                          ║
║ 02 ║ Children's Clinic          ║ pediatric vaccine delivery                 ║
║ 03 ║ Oncology Center            ║ immunocompromised-patient supply           ║
║ 05 ║ Infection Ward             ║ high-urgency isolation ward                ║
║ 07 ║ Airport Medical Cargo      ║ temperature-controlled air cargo node      ║
║ 09 ║ South Mobile Unit          ║ mobile emergency storage                   ║
║ 08 ║ North Mobile Unit          ║ mobile emergency storage                   ║
║ 04 ║ Emergency Depot            ║ contingency resupply stop                  ║
║ 01 ║ University Hospital        ║ major receiving hospital                   ║
║ 06 ║ Community Pavilion         ║ final community distribution point         ║
╚════╩════════════════════════════╩════════════════════════════════════════════╝

Published route from the documented matrix:

  ┌──────────┐  28  ┌──────────┐  29  ┌──────────┐  17  ┌──────────┐  27  ┌──────────┐
  │ 00       ├──────┤ 02       ├──────┤ 03       ├──────┤ 05       ├──────┤ 07       │
  │ Bio-Hub  │      │ Children │      │ Oncology │      │ Infection│      │ Airport  │
  └────┬─────┘      └──────────┘      └──────────┘      └──────────┘      │ Cargo    │
       │18                                                                └────┬─────┘
       │                                                                       │19
       │                                                                       │
  ┌────┴─────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌────┴─────┐
  │ 06       ├──────┤ 01       ├──────┤ 04       ├──────┤ 08       ├──────┤ 09       │
  │ Community│  14  │ Univ.    │  16  │ Emergency│  15  │ North    │  17  │ South    │
  │ Pavilion │      │ Hospital │      │ Depot    │      │ Mobile   │      │ Mobile   │
  └──────────┘      └──────────┘      └──────────┘      └──────────┘      └──────────┘

Traversal order (the exact manifest published by Example 11.8.1):

  00 -> 02 -> 03 -> 05 -> 07 -> 09 -> 08 -> 04 -> 01 -> 06 -> 00

Because the margin is positive, the documented route stays inside the configured
thermal envelope and the example publishes:

- `closed=true`
- `minutes=200.0`
- `thermal-margin=40.0`
- `status=within-envelope`

Approximation interpretation:

- The example prints `ratio=1.5`.
- This is the **published Christofides worst-case guarantee** for metric symmetric TSP.
- It is **not** a measured `tour_cost / optimal_cost` quotient for this concrete instance.
- The guarantee is valid here because the example uses `Christofides` on a symmetric complete metric-style matrix together with `MatchingAlgo = BlossomMatch`.

Christofides publication pipeline for this same example(11.8.1):

┌────────────────────────────────────────────────────────────────────────────┐
│ Step 1. Input matrix                                                       │
│ - 10x10                                                                    │
│ - symmetric                                                                │
│ - finite                                                                   │
│ - start vertex = 00                                                        │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ Step 2. Minimum spanning tree (MST)                                        │
│ - build a low-cost spanning skeleton over all 10 stops                     │
│ - the MST is not yet a tour                                                │
│ - some vertices have odd degree                                            │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ Step 3. Odd-degree correction                                              │
│ - collect the odd-degree MST vertices                                      │
│ - solve minimum-weight perfect matching on that odd set                    │
│ - because the example selects BlossomMatch, the matching stage is exact    │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ Step 4. Eulerian multigraph                                                │
│ - combine MST edges with matching edges                                    │
│ - every vertex now has even degree                                         │
│ - Hierholzer can traverse the multigraph as a closed Eulerian walk         │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ Step 5. Shortcut repeated visits                                           │
│ - traverse the Eulerian walk                                               │
│ - skip already-visited vertices                                            │
│ - preserve visit order                                                     │
│ - publish one Hamiltonian cycle                                            │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ Step 6. Published result                                                   │
│ - Tour  = 00 -> 02 -> 03 -> 05 -> 07 -> 09 -> 08 -> 04 -> 01 -> 06 -> 00   │
│ - Cost  = 200                                                              │
│ - Margin= 40                                                               │
│ - Status= within-envelope                                                  │
└────────────────────────────────────────────────────────────────────────────┘

Interpretation:
  200 <= 240, so the route is inside the validated thermal envelope.
  The 1.5 ratio is publishable because the example explicitly selects BlossomMatch.
```

### 11.7.2. Christofides algorithm stages over the same matrix

The route is not constructed by greedily walking the cheapest visible corridor. Christofides creates a structural skeleton, repairs parity with matching, traverses an Eulerian multigraph, and shortcuts repeated vertices.

```text
╔══════════════════════════════╗
║ CHRISTOFIDES PIPELINE        ║
╚══════════════════════════════╝

┌──────────────────────────────┐
│ complete symmetric matrix    │
│ c(u,v)=c(v,u), finite costs  │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ minimum spanning tree T      │
│ connects all stops cheaply   │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ odd-degree set O             │
│ vertices with deg_T(v) odd   │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ exact Blossom MWPM on O      │
│ required for 1.5 certificate │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Eulerian multigraph H=T∪M    │
│ every degree is even         │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Hierholzer Eulerian walk     │
│ preserves parallel edges     │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ shortcut repeated vertices   │
│ publish Hamiltonian tour     │
└──────────────────────────────┘
```

### 11.7.3. Matrix preparation wall: missing edges and final kernels

`+Inf` is not a final-kernel cost. It is a pre-kernel missing-edge sentinel that must either be resolved by metric closure or rejected.

```text
╔═════════════════════════════════════════════════════╗
║ MATRIX PREPARATION WALL                             ║
╚═════════════════════════════════════════════════════╝

┌──────────────────────────────────────┐
│ caller matrix                        │
│ finite costs plus possible +Inf walls│
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ validate shape, start vertex, IDs    │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ validate NaN, -Inf, negative weights │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ unresolved +Inf exists?              │
└──────────────────┬───────────────────┘
                   │      
            ┌──────┴──────┐      
            │             │      
            ▼             ▼      
┌──────────────┐   ┌───────────────────────┐
│ NO           │   │ YES                   │
│ proceed      │   │ RunMetricClosure ?    │
└──────┬───────┘   └──────┬────────────────┘
       │                  │
       │          ┌───────┴────────┐
       │          │                │
       │          ▼                ▼
       │   ┌──────────────┐   ┌──────────────────────┐
       │   │ YES          │   │ NO                   │
       │   │ detached APSP│   │ reject incomplete    │
       │   │ O(n³)        │   │ matrix               │
       │   └──────┬───────┘   └──────────────────────┘
       │          │
       ▼          ▼
┌────────────────────────────────────────┐
│ final solver kernel gate               │
│ finite complete matrix only            │
│ symmetry if selected algorithm needs it│
└────────────────────────────────────────┘
```

### 11.7.4. Branch-and-Bound exact search and 1-tree pruning

This diagram shows why Branch-and-Bound can be exact without enumerating every full tour: a branch is discarded only after an admissible bound proves it cannot beat the incumbent.

```text
╔══════════════════════════════════════════════════════════════════════════════╗
║ BRANCH-AND-BOUND EXACTNESS FLOW                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

                         ┌────────────────────┐
                         │ partial path P     │
                         │ costSoFar = g(P)   │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ lower bound LB(P)  │
                         │ e.g. OneTreeBound  │
                         └─────────┬──────────┘
                                   │
             ┌─────────────────────┴─────────────────────┐
             │                                           │
             ▼                                           ▼
┌────────────────────────────┐              ┌────────────────────────────┐
│ LB(P) >= incumbent cost    │              │ LB(P) < incumbent cost     │
│ prune safely               │              │ expand deterministically   │
└────────────────────────────┘              └──────────────┬─────────────┘
                                                           │
                                                           ▼
                                             ┌────────────────────────────┐
                                             │ add next unvisited vertex  │
                                             │ stable cost/order tie-break│
                                             └──────────────┬─────────────┘
                                                            │
                                                            ▼
                                             ┌────────────────────────────┐
                                             │ full closed tour reached?  │
                                             └───────┬────────────────────┘
                                                     │
                                           ┌─────────┴─────────┐
                                           │                   │
                                           ▼                   ▼
                                  ┌────────────────┐   ┌──────────────────┐
                                  │ update         │   │ continue DFS     │
                                  │ incumbent      │   │ search           │
                                  └────────────────┘   └──────────────────┘

Exactness law:
  If search completes, incumbent is globally optimal.
  If timeout interrupts search, incumbent may be feasible but not certified optimal.
```

### 11.7.5. Parallel-edge-safe Eulerian traversal

The failing class this package guards against is grouped parallel half-edges. The correct model pairs each half-edge with an opposite directed half-edge.

```text
╔══════════════════════════════════════════════════════════════════════════════╗
║ EULERIAN MULTIGRAPH: PARALLEL EDGE TWIN PAIRING                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Canonical adjacency fragment after Christofides multigraph rebuild:

┌──────────────┬──────────────────────────────┐
│ vertex       │ adjacency                    │
├──────────────┼──────────────────────────────┤
│ 03           │ 04, 06, 10, 10               │
│ 10           │ 03, 03, 08, 14               │
└──────────────┴──────────────────────────────┘

Physical fragment:

┌────┐      ┌────┐      ┌────┐      ┌────┐
│ 04 ├──────┤ 03 ╞══════╡ 10 ├──────┤ 08 │
└────┘      └─┬──┘      └──┬─┘      └────┘
              │            │
              │            │
            ┌─┴──┐      ┌──┴─┐
            │ 06 │      │ 14 │
            └────┘      └────┘

Legend:
  03 ╞══════╡ 10  means two distinct parallel undirected edges between 03 and 10.
  The single vertical/light edges show ordinary non-parallel adjacency entries.

Directed half-edge pairing model:

┌──────────────┬──────────────┬──────────────────────────────────────────────┐
│ scanned edge │ opposite key │ action                                       │
├──────────────┼──────────────┼──────────────────────────────────────────────┤
│ 03 -> 10     │ 10 -> 03     │ no opposite yet: push pending[03->10]        │
│ 03 -> 10     │ 10 -> 03     │ no opposite yet: push pending[03->10]        │
│ 10 -> 03     │ 03 -> 10     │ pop one pending 03->10 and pair twins        │
│ 10 -> 03     │ 03 -> 10     │ pop second pending 03->10 and pair twins     │
└──────────────┴──────────────┴──────────────────────────────────────────────┘

Forbidden model:
  unordered key {03,10} pairing 03->10 with 03->10.
```

### 11.7.6. Directed 2-opt* route improvement boundary

Directed local search must not reverse the middle segment as symmetric 2-opt does. ATSP arc directions carry meaning.

```text
╔══════════════════════════════════════════════════════════════════════════════╗
║ DIRECTED 2-OPT* MOVE                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Before accepted move:

┌────┐    a->b    ┌────┐──────────── directed segment ────────────┌────┐    c->d    ┌────┐
│ a  ├────────────┤ b  │  ... internal orientation preserved ...  │ c  ├────────────┤ d  │
└────┘            └────┘                                          └────┘            └────┘

After 2-opt* rewiring:

┌────┐    a->d    ┌────┐
│ a  ├────────────┤ d  │
└────┘            └────┘

┌────┐    c->b    ┌────┐──────────── same directed segment ───────┌────┐
│ c  ├────────────┤ b  │  ... no segment reversal inside ATSP ... │ ...│
└────┘            └────┘                                          └────┘

Contract:
  Symmetric 2-opt may reverse a segment.
  Directed 2-opt* rewires successors and preserves arc orientation.
```

---

## 11.8. Go Example Scenarios

The examples below use static matrices with 8 to 10 vertices. They show realistic build, execute, and consume pipelines with explicit setup and solver error handling.

### 11.8.1. Biopharmaceutical cold-chain dispatch

A vaccine distribution hub must deliver to hospitals, emergency depots, and mobile units before validated cold-chain containment expires. The matrix stores peak-window travel minutes, not physical distance. The example selects Christofides with Blossom matching because the matrix is symmetric and the caller wants the published `1.5` approximation certificate.

The resulting route uses `200` minutes inside a `240` minute envelope. That is a valid dispatch with `40` minutes of operational margin. If the margin were negative, the correct operational response would be split routing, additional vehicle capacity, or a different matrix policy, not ignoring the result.

```go
package main

import (
    "fmt"
    "log"
    "strings"

    "github.com/lvlath/go/matrix"
    "github.com/lvlath/go/tsp"
)

func main() {
    const startVertex = 0
    const thermalEnvelopeMinutes = 240.0

    locations := []string{
        "Regional Bio-Hub",
        "University Hospital",
        "Children's Clinic",
        "Oncology Center",
        "Emergency Depot",
        "Infection Ward",
        "Community Pavilion",
        "Airport Medical Cargo",
        "North Mobile Unit",
        "South Mobile Unit",
    }

    transitMinutes := []float64{
        0, 22, 28, 44, 31, 52, 18, 36, 27, 30,
        22, 0, 12, 25, 16, 38, 14, 30, 20, 24,
        28, 12, 0, 29, 19, 42, 16, 34, 18, 26,
        44, 25, 29, 0, 21, 17, 35, 28, 33, 23,
        31, 16, 19, 21, 0, 30, 22, 24, 15, 18,
        52, 38, 42, 17, 30, 0, 46, 27, 40, 22,
        18, 14, 16, 35, 22, 46, 0, 32, 21, 28,
        36, 30, 34, 28, 24, 27, 32, 0, 26, 19,
        27, 20, 18, 33, 15, 40, 21, 26, 0, 17,
        30, 24, 26, 23, 18, 22, 28, 19, 17, 0,
    }

    dist, err := matrix.NewDense(len(locations), len(locations))
    if err != nil {
        log.Fatalf("build cold-chain matrix: %v", err)
    }
    _ = dist.Fill(transitMinutes)

    opts := tsp.DefaultOptions()
    opts.Algo = tsp.Christofides
    opts.Symmetric = true
    opts.StartVertex = startVertex
    opts.MatchingAlgo = tsp.BlossomMatch
    opts.EnableLocalSearch = false

    result, err := tsp.SolveMatrix(dist, locations, opts)
    if err != nil {
        log.Fatalf("solve cold-chain route: %v", err)
    }

    vertexTour, err := result.VertexTour()
    if err != nil {
        log.Fatalf("project route IDs: %v", err)
    }

    margin := thermalEnvelopeMinutes - result.Cost
    status := "within-envelope"
    if margin < 0 {
        status = "split-route-required"
    }

    fmt.Printf("closed=%v\n", len(result.Tour) == len(locations)+1)
    fmt.Printf("minutes=%.1f\n", result.Cost)
    fmt.Printf("thermal-margin=%.1f\n", margin)
    fmt.Printf("status=%s\n", status)
    fmt.Printf("ratio=%.1f\n", result.ApproximationRatio)
    fmt.Printf("manifest=%s\n", strings.Join(vertexTour, " -> "))
	
	// Output:
	// closed=true
	// minutes=200.0
	// thermal-margin=40.0
	// status=within-envelope
	// ratio=1.5
	// manifest=Regional Bio-Hub -> Children's Clinic -> Oncology Center -> Infection Ward -> Airport Medical Cargo -> South Mobile Unit -> North Mobile Unit -> Emergency Depot -> University Hospital -> Community Pavilion -> Regional Bio-Hub

}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-TSP_Biopharmaceutical_cold_chain_dispatch-blue?logo=go)](https://go.dev/play/p/DY2hMz_yK5_V)

### 11.8.2. High-value cash-in-transit ATSP risk routing

This example models a directed security route. The matrix combines travel time, fuel burn, escort availability, and exposure risk. Direction matters: returning through the same corridor may have a different risk score because of one-way streets, turn restrictions, and predictable attack surfaces.

The example uses `TwoOptSearch` rather than the `SolveMatrix` dispatcher because the operation starts from a manually approved security corridor. Local search may improve that corridor, but it never claims exactness or global optimality.

```go
package main

import (
    "fmt"
    "log"
    "strings"

    "github.com/lvlath/go/matrix"
    "github.com/lvlath/go/tsp"
)

func main() {
    const startVertex = 0

    stops := []string{
        "Secure Vault HQ",
        "Downtown ATM Cluster",
        "Transit Terminal Depot",
        "Casino Cash Office",
        "Airport Cash Vault",
        "Retail Megastore",
        "Stadium Event Office",
        "Harbor Customs Cash Desk",
        "University Payroll Office",
    }

    directedRiskCost := []float64{
        0, 8, 30, 31, 28, 34, 36, 39, 26,
        22, 0, 5, 27, 29, 23, 34, 36, 25,
        24, 21, 0, 6, 31, 29, 25, 35, 27,
        29, 24, 23, 0, 4, 32, 28, 36, 30,
        28, 26, 25, 22, 0, 6, 29, 30, 31,
        35, 24, 28, 30, 27, 0, 5, 29, 26,
        38, 34, 25, 29, 27, 24, 0, 6, 28,
        36, 35, 32, 30, 28, 27, 25, 0, 4,
        7, 25, 27, 29, 31, 26, 28, 24, 0,
    }

    approvedCorridor := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 0}

    dist, err := matrix.NewDense(len(stops), len(stops))
    if err != nil {
        log.Fatalf("build risk matrix: %v", err)
    }
    _ = dist.Fill(directedRiskCost)

    opts := tsp.DefaultOptions()
    opts.Algo = tsp.TwoOptOnly
    opts.Symmetric = false
    opts.StartVertex = startVertex
    opts.EnableLocalSearch = true
    opts.TwoOptMaxIters = 0

    result, err := tsp.TwoOptSearch(dist, approvedCorridor, opts)
    if err != nil {
        log.Fatalf("improve directed cash route: %v", err)
    }

    routeNames := make([]string, 0, len(result.Tour))
    for _, vertex := range result.Tour {
        routeNames = append(routeNames, stops[vertex])
    }

    fmt.Printf("closed=%v\n", len(result.Tour) == len(stops)+1)
    fmt.Printf("risk-cost=%.1f\n", result.Cost)
    fmt.Printf("exact=%v optimal=%v\n", result.Exact, result.Optimal)
    fmt.Printf("accepted-moves=%d\n", result.Iterations)
    fmt.Printf("route=%s\n", strings.Join(routeNames, " -> "))

	// Output:
	// closed=true
	// risk-cost=51.0
	// exact=false optimal=false
	// accepted-moves=0
	// route=Secure Vault HQ -> Downtown ATM Cluster -> Transit Terminal Depot -> Casino Cash Office -> Airport Cash Vault -> Retail Megastore -> Stadium Event Office -> Harbor Customs Cash Desk -> University Payroll Office -> Secure Vault HQ

}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-TSP_High_value_cash_in_transit_ATSP_risk_routing-blue?logo=go)](https://go.dev/play/p/w6OSSYWt9lG)

### 11.8.3. Semiconductor laser-drilling exact optimization

An industrial laser-drilling workstation visits microscopic via-hole positions on each board. The matrix stores servo motion and stabilization latency in milliseconds. At 100,000 boards per production batch, a few milliseconds per board become minutes of machine occupancy.

This example uses exact Branch-and-Bound with `OneTreeBound`. The output is exact and optimal because the search completes.

```go
package main

import (
    "fmt"
    "log"

    "github.com/lvlath/go/matrix"
    "github.com/lvlath/go/tsp"
)

func main() {
    const startVertex = 0
    const dailyBoardVolume = 100000

    sites := []string{
        "Servo Home",
        "Pin A1",
        "Bus Gate X7",
        "Core Array",
        "I/O Bridge",
        "Memory Lane",
        "Power Island",
        "Clock Mesh",
    }

    latencyMilliseconds := []float64{
        0, 3.6, 9.5, 10.0, 4.5, 8.4, 7.6, 6.8,
        3.6, 0, 1.2, 8.0, 9.0, 7.2, 6.5, 5.9,
        9.5, 1.2, 0, 2.8, 7.5, 5.8, 6.1, 4.4,
        10.0, 8.0, 2.8, 0, 4.7, 3.6, 5.2, 4.0,
        4.5, 9.0, 7.5, 4.7, 0, 2.9, 3.3, 5.5,
        8.4, 7.2, 5.8, 3.6, 2.9, 0, 1.7, 2.4,
        7.6, 6.5, 6.1, 5.2, 3.3, 1.7, 0, 2.1,
        6.8, 5.9, 4.4, 4.0, 5.5, 2.4, 2.1, 0,
    }

    dist, err := matrix.NewDense(len(sites), len(sites))
    if err != nil {
        log.Fatalf("build latency matrix: %v", err)
    }
    _ = dist.Fill(latencyMilliseconds)

    opts := tsp.DefaultOptions()
    opts.Algo = tsp.BranchAndBound
    opts.Symmetric = true
    opts.StartVertex = startVertex
    opts.BoundAlgo = tsp.OneTreeBound
    opts.EnableLocalSearch = false

    result, err := tsp.BranchAndBoundSolve(dist, opts)
    if err != nil {
        log.Fatalf("solve exact drilling path: %v", err)
    }

    clone := result.Clone()
    dailySeconds := clone.Cost * float64(dailyBoardVolume) / 1000.0

    fmt.Printf("closed=%v\n", len(clone.Tour) == len(sites)+1)
    fmt.Printf("single-board-ms=%.1f\n", clone.Cost)
    fmt.Printf("daily-motion-seconds=%.1f\n", dailySeconds)
    fmt.Printf("exact=%v optimal=%v\n", clone.Exact, clone.Optimal)
    fmt.Printf("nodes-expanded=%d\n", clone.NodesExpanded)

	// Output:
	// closed=true
	// single-board-ms=22.8
	// daily-motion-seconds=2280.0
	// exact=true optimal=true
	// nodes-expanded=218
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-TSP_Semiconductor_laser_drilling_exact_optimization-blue?logo=go)](https://go.dev/play/p/AUO4v86uEzU)

---

## 11.9. Laws of Ownership, Concurrency & Partial Results

### Ownership

`Result` is detached. `Clone()` deep-copies `Tour` and `IDs`, preserves scalar metadata, and returns nil for a nil receiver. `VertexTour()` allocates a detached `[]string` and maps indices through `IDs` left-to-right.

Caller mutation of the returned `Tour`, `IDs`, clone, or vertex tour must not mutate solver state or the source result.

### Matrix ownership

Solvers treat input matrices as immutable during execution. Metric closure, when requested, uses detached storage. The package does not provide snapshot isolation against external concurrent mutation of a caller-owned matrix.

### Concurrency

Supported:
```text
goroutine A: SolveMatrix(matrixA, idsA, optsA)
goroutine B: SolveMatrix(matrixB, idsB, optsB)
```

Supported by convention for immutable shared input:
```text
goroutine A: SolveMatrix(sharedReadOnlyMatrix, ids, opts)
goroutine B: SolveMatrix(sharedReadOnlyMatrix, ids, opts)
```

Unsupported:
```text
goroutine A: SolveMatrix(sharedMatrix, ids, opts)
goroutine B: sharedMatrix.Set(i, j, value)
```

### Partial results

Hard validation and construction failures suppress results:
```text
result == nil
err != nil
```

Branch-and-Bound timeout with incumbent:
```text
result != nil
errors.Is(err, tsp.ErrTimeLimit)
result.TimedOut == true
result.Optimal == false
```

Direct local-search timeout with current feasible tour:
```text
result != nil
errors.Is(err, tsp.ErrTimeLimit)
result.Exact == false
result.Optimal == false
```

Timeout without feasible tour:
```text
result == nil
errors.Is(err, tsp.ErrTimeLimit)
```

---

## 11.10. Pitfalls

> **Do not parse error strings.**
>
>    Use `errors.Is`; wrapped context is for diagnostics.

> **Do not assume `DefaultOptions()` publishes Christofides `1.5`.**
>
>    The current default matching policy is explicit Greedy matching. Set `MatchingAlgo = BlossomMatch` when exact MWPM and the formal ratio are required.

> **Do not run Christofides on asymmetric input.**
>
>    Christofides is a symmetric metric TSP algorithm. Directed risk matrices belong to ATSP-capable exact or local-search regimes.

> **Do not send unresolved `+Inf` into final kernels.**
>
>    Enable metric closure where appropriate or reject the instance.

> **Do not treat 2-opt or 3-opt as exact.**
>
>    They are local-search heuristics and must not publish global optimality.

> **Do not describe restricted ATSP 3-opt* as full ATSP 3-opt.**
>
>    The directed mode preserves orientation and does not enumerate every arbitrary directed reconnection.

> **Do not mutate matrices while solving.**
>
>    External mutation voids deterministic and numeric correctness assumptions.

> **Do not reintroduce metadata-dropping result projections.**
>
>    `Result` is the canonical result artifact.

> **Do not hide selected-algorithm failures with weaker fallback.**
>
>    If a caller selected Blossom and it fails, surfacing the error preserves the matching law.

> **Do not deduplicate Christofides multigraph parallel edges.**
>
>    Parallel edges are part of the Eulerian multigraph and must be consumed exactly once.

---

## 11.11. Best Practices

> **Use `SolveMatrix` as the canonical integration facade.**
>
>    It preserves matrix semantics, IDs, metric-closure metadata, and result publication policy.

>  **Use direct wrappers when the algorithm is fixed by the product requirement.**
>
>    `BranchAndBoundSolve`, `ChristofidesSolve`, `TwoOptSearch`, and `ThreeOptSearch` avoid accidental dispatcher ambiguity.

> **Set `MatchingAlgo = BlossomMatch` explicitly for Christofides guarantees.**
>
>    The ratio is a proof artifact, not a general Christofides label.

> **Use `OneTreeBound` for symmetric exact Branch-and-Bound pruning.**
>
>    It can strengthen pruning without changing exactness.

> **Use domain units directly.**
>
>    Minutes, milliseconds, fuel, and risk scores are valid as long as they are finite, non-negative, and consistently interpreted.

> **Pass IDs aligned with matrix rows.**
>
>    IDs make `VertexTour()` useful and preserve operational traceability.

> **Use `Clone()` before handing results to mutable consumers.**
>
>    This keeps published solver snapshots protected from caller-side mutation.

> **Use metric closure as an explicit modeling decision.**
>
>    Closure replaces missing direct edges with shortest-path costs; that changes the effective matrix.

> **Read certificates, not only cost.**
>
>    Check `Exact`, `Optimal`, `TimedOut`, `ApproximationRatio`, `MetricClosureApplied`, and `Symmetric`.

> **Benchmark real regimes.**
>
>    Build matrices and options before `b.ResetTimer`, call `b.ReportAllocs`, and avoid setup work inside hot loops.

---

**lvlath/tsp**: deterministic by contract, explicit in mathematical strength, strict in numeric validation, and safe to compose in production routing systems.
