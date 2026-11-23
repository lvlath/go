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
    - Independent `match` Blossom MWPM and explicit Greedy matching policies consumed by Christofides.
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

For directed 2-opt, the kernel reverses the middle segment to preserve one Hamiltonian cycle. Unlike symmetric 2-opt, the directed move delta includes the changed direction of **every internal segment arc**:

$$ \Delta = c(a,c)+c(b,d)-c(a,b)-c(c,d) + \sum_{t=i}^{k-1}\big[c(\pi_{t+1},\pi_t)-c(\pi_t,\pi_{t+1})\big] $$

Here `a=T[i-1]`, `b=T[i]`, `c=T[k]`, and `d=T[k+1]`. Pure tail swapping of two arcs in **one directed cycle** without reversing the middle segment would create two disconnected subtours. The directed 3-opt neighborhood is separately restricted and does not implement arbitrary ATSP 3-opt.

### Complexity summary

| Phase                           | Time                                                                                | Space                          | Publication meaning                                 |
|:--------------------------------|:------------------------------------------------------------------------------------|:-------------------------------|:----------------------------------------------------|
| Option validation               | `O(1)`                                                                              | `O(1)`                         | Rejects invalid policy before kernel trust.         |
| Matrix validation               | `O(n²)`                                                                             | `O(1)` to `O(n²)`              | Enforces numeric and symmetry/completeness policy.  |
| Metric closure                  | `O(n³)`                                                                             | `O(n²)`                        | Resolves `+Inf` before final kernels.               |
| Held-Karp                       | `O(n²2ⁿ)`                                                                           | `O(n2ⁿ)`                       | Exact optimal result on completion.                 |
| Branch-and-Bound                | exponential worst case                                                              | `O(n²)+O(n)`                   | Exact only when completed.                          |
| One-tree bound iteration        | `O(n²)`                                                                             | `O(n²)`                        | Lower-bound computation, not a route.               |
| Christofides MST                | `O(n²)` dense regime                                                                | adjacency buffers              | Approximation pipeline stage.                       |
| Blossom MWPM                    | exact dense search; global bound not certified for this implementation              | `O(k²)` dense matching storage | Required for 1.5 ratio publication on metric input. |
| Eulerian circuit                | `O(V+E)`                                                                            | `O(V+E)`                       | Consumes every multigraph edge exactly once.        |
| Shortcut and result publication | `O(E+n)`                                                                            | `O(n)`                         | Produces detached Hamiltonian tour.                 |
| 2-opt                           | symmetric `O(iterations*n²)`; directed `O(iterations*n³)` worst-case candidate scan | `O(n)`                         | Heuristic, no global certificate.                   |
| 3-opt                           | `O(iterations*n³)` in symmetric scan regimes                                        | `O(n)`                         | Heuristic, no global certificate.                   |

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

`DefaultOptions()` selects Christofides, symmetric input, **BlossomMatch**, no metric closure and local search enabled. GreedyMatch is an explicitly requested weaker mode. Note that selecting BlossomMatch does not prove the input is metric; establishing triangle inequality remains the caller's responsibility.

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
  canonicalize orientation only when symmetric
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
  if Symmetric:
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

### Directed 2-opt reversal boundary

```text
FUNCTION DirectedTwoOptReversal(tour, i, k):
  evaluate delta = changed boundary arcs + changed internal directions
  accept only when delta < -Eps
  reverse tour[i:k+1] to preserve one connected Hamiltonian cycle
  re-evaluate the final published tour cost from source weights
  preserve directed tour orientation (never mirror it for canonical appearance)
  publish Exact=false, Optimal=false
```

Directed 2-opt candidate evaluation includes `O(k-i)` arc differences and
therefore up to `O(n³)` work per complete neighborhood scan. The internal
reversal is a valid directed Hamiltonian cycle; **a two-arc tail swap without
reversal would yield two subtours in a single-cycle TSP**.

---

## 11.7. ASCII Diagrams

### 11.7.1. Cold-chain distribution: a verified six-stop metric fixture

The current executable `tsp/example_test.go` uses a **6 × 6** symmetric road
travel-time matrix. All 216 ordered triangle inequalities have been checked
for this fixed fixture; the approximation hypothesis is met rather than
merely asserted. The route starts and ends at the regional bio-hub.

```text
Regional Bio-Hub (0)
     | 34 min
     v
Metropolitan Infection Ward (4)
     | 16 min
     v
Oncology Treatment Center (2)
     | 20 min
     v
Emergency Red Cross Depot (3)
     | 12 min
     v
Children's Clinical Hospital (1)
     | 14 min
     v
Community Health Pavilion (5)
     | 18 min
     v
Regional Bio-Hub (0)

Total: 34+16+20+12+14+18 = 114 minutes.
Thermal window: 120 minutes; margin: 6 minutes.
```

Christofides builds an MST, matches its odd-degree vertices through the
independent `match.Blossom` engine, adds each reciprocal edge to an undirected
multigraph, traverses a complete Eulerian circuit and shortcuts repeated
visits. The `1.5` ratio is the **worst-case theorem for metric instances**, not
an observed ratio `114 / OPT`. If the caller selects GreedyMatch or violates
triangle inequality, the theorem is not available.

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

### 11.7.6. Directed 2-opt preserves a single cycle

```text
Original directed tour:
  ... -> a -> b -> [ internal b -> ... -> c ] -> d -> ...

Selected reversal:
  ... -> a -> c -> [ internal c -> ... -> b ] -> d -> ...

Boundary changes: (a,b),(c,d) -> (a,c),(b,d)
Internal change:   each u->v becomes v->u in the reversed segment

Delta = new boundary - old boundary + sum(internal reverse - forward)
```

Both the boundary and the internal direction changes must be charged. The
algorithm rejects moves with `delta >= -Eps`, never reconstructs a tour by
following a pair of tail-swapped successors, and retains the actual directed
orientation on publication. The final result cost is recomputed from its
published tour to avoid floating-point delta accumulation drift.


---

## 11.8. Executable operational scenarios

All three listings below come directly from the tested `tsp/example_test.go`.
They are deterministic fixtures, **not live routing, security or manufacturing
measurements**. They are written as `package tsp_test` Go examples, so `go test
./tsp -run '^ExampleSolveMatrix_' -count=1` checks the `// Output:` contracts.
Use a real graph-derived or measured matrix after independent source validation.

| Fixture | Physical units and operational release | Algorithmic certificate |
|:--|:--|:--|
| Cold-chain hospitals, six sites | 114 min; 120 min thermal envelope, 6 min margin | Christofides + exact Blossom; formal 1.5 only because matrix is metric |
| Armored transport, five directed stops | Risk cost 34; direction-dependent return arcs | Directed 2-opt local optimum, no exactness |
| Semiconductor drilling, five sites | 16.8 ms per board; 840 s per 50,000 boards | Completed exact Branch-and-Bound using one-tree lower bound |

### 11.8.1–11.8.3. Full working Go examples

The functions are shown verbatim from the package's executable examples,
including checked matrix construction, full finite data and the exact expected
output. Import order and error handling follow the actual source.

```go
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test demonstrates canonical lvlath/tsp usage through production-shaped
// operational scenarios.
//
// The examples use static matrices instead of generated toy graphs. Each matrix
// represents a business cost model: minutes, directed security risk, or mechanical
// positioning latency. The static fixtures are intentionally embedded in the
// examples so readers can audit the exact optimization problem without chasing helpers.
package tsp_test

import (
	"fmt"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

const (
	exampleStartVertex = 0
	exampleEps         = 1e-12
)

// ExampleSolveMatrix_coldChainLogistics demonstrates symmetric metric TSP for
// biopharmaceutical cold-chain distribution.
//
// Business story:
// A regional bio-hub must deliver temperature-sensitive mRNA vaccine containers
// across a hospital network and return to the hub. The containers remain inside
// the validated thermal envelope for 120 minutes. A route beyond that window
// risks product degradation and emergency rescheduling.
//
// Algorithm choice:
// Christofides with BlossomMatch is selected because the instance is symmetric,
// complete, and metric. The exact Blossom MWPM stage is required to publish the
// formal 1.5 approximation ratio.
//
// This example intentionally keeps the dispatch matrix static. In production,
// these values would come from a traffic-time service, but the solver contract is
// identical: matrix rows and columns define the stop order.
func ExampleSolveMatrix_coldChainLogistics() {
	const thermalEnvelopeMinutes = 120.0

	locationNames := []string{
		"Regional Bio-Hub",
		"Children's Clinical Hospital",
		"Oncology Treatment Center",
		"Emergency Red Cross Depot",
		"Metropolitan Infection Ward",
		"Community Health Pavilion",
	}

	// Static symmetric metric transit-time matrix, measured in minutes.
	//
	// The values model a weighted urban road network after travel-time smoothing.
	// Row/column order follows locationNames. The intended interpretation is time,
	// not geographic distance.
	coldChainTransitMinutes := []float64{
		0, 32, 50, 44, 34, 18,
		32, 0, 32, 12, 48, 14,
		50, 32, 0, 20, 16, 46,
		44, 12, 20, 0, 36, 26,
		34, 48, 16, 36, 0, 48,
		18, 14, 46, 26, 48, 0,
	}

	dist, err := matrix.NewDense(len(locationNames), len(locationNames))
	if err != nil {
		fmt.Println("cold-chain matrix allocation failed:", err)
		return
	}
	if err = dist.Fill(coldChainTransitMinutes); err != nil {
		fmt.Println("cold-chain matrix initialization failed:", err)
		return
	}

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.StartVertex = exampleStartVertex
	opts.MatchingAlgo = tsp.BlossomMatch
	opts.EnableLocalSearch = false
	opts.Eps = exampleEps

	result, err := tsp.SolveMatrix(dist, nil, opts)
	if err != nil {
		fmt.Printf("cold-chain route failed: %v\n", err)
		return
	}
	if result == nil {
		fmt.Println("unexpected nil route result")
		return
	}

	fmt.Println("Cold-chain vaccine dispatch")
	fmt.Printf("closed-loop=%v\n", len(result.Tour) == len(locationNames)+1 &&
		result.Tour[0] == exampleStartVertex &&
		result.Tour[len(locationNames)] == exampleStartVertex)
	fmt.Printf("route-minutes=%.1f\n", result.Cost)
	fmt.Printf("formal-ratio=%.1f\n", result.ApproximationRatio)
	fmt.Printf("thermal-margin=%.1f minutes\n", thermalEnvelopeMinutes-result.Cost)

	fmt.Print("manifest=")
	for step, vertex := range result.Tour {
		if step > 0 {
			fmt.Print(" -> ")
		}
		fmt.Print(locationNames[vertex])
	}
	fmt.Println()

	// Output:
	// Cold-chain vaccine dispatch
	// closed-loop=true
	// route-minutes=114.0
	// formal-ratio=1.5
	// thermal-margin=6.0 minutes
	// manifest=Regional Bio-Hub -> Metropolitan Infection Ward -> Oncology Treatment Center -> Emergency Red Cross Depot -> Children's Clinical Hospital -> Community Health Pavilion -> Regional Bio-Hub
}

// ExampleSolveMatrix_cashInTransitAsymmetric demonstrates directed local-search
// routing for armored cash-in-transit operations.
//
// Business story:
// An armored vehicle replenishes high-value ATM and cash-depot sites. One-way
// streets, turn restrictions, timed security escorts, and high-risk intersections
// make A->B different from B->A. The matrix therefore stores directed composite
// risk/fuel costs, not symmetric distance.
//
// Algorithm choice:
// TwoOptOnly with Symmetric=false evaluates the costs of every reversed directed arc.
// This is a scalable local-search mode, not an exact ATSP certificate. It is
// useful when dispatch must react quickly and the cost model is directional.
func ExampleSolveMatrix_cashInTransitAsymmetric() {
	stopNames := []string{
		"Secure Bank Vault",
		"Downtown Retail Mega-Mall",
		"Financial District Transit Terminal",
		"Westside Casino Hub",
		"Central Airport Cash-Vault",
	}

	// Static asymmetric risk matrix.
	//
	// Each value combines transit time, fuel burn, guard exposure, and predictable
	// interception risk. The asymmetry is intentional: for example, Mall->Casino
	// can use a guarded arterial route, while Casino->Mall crosses a slower
	// exposure-heavy corridor.
	directedRiskCost := []float64{
		0, 8, 19, 21, 17,
		14, 0, 5, 9, 18,
		18, 11, 0, 6, 20,
		16, 8, 12, 0, 4,
		11, 15, 13, 10, 0,
	}

	dist, err := matrix.NewDense(len(stopNames), len(stopNames))
	if err != nil {
		fmt.Println("directed-security matrix allocation failed:", err)
		return
	}
	if err = dist.Fill(directedRiskCost); err != nil {
		fmt.Println("directed-security matrix initialization failed:", err)
		return
	}

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.TwoOptOnly
	opts.Symmetric = false
	opts.StartVertex = exampleStartVertex
	opts.EnableLocalSearch = true
	opts.Eps = exampleEps

	result, err := tsp.SolveMatrix(dist, nil, opts)
	if err != nil {
		fmt.Printf("cash-in-transit route failed: %v\n", err)
		return
	}
	if result == nil {
		fmt.Println("unexpected nil route result")
		return
	}

	fmt.Println("Cash-in-transit directed route")
	fmt.Printf("closed-loop=%v\n", len(result.Tour) == len(stopNames)+1 &&
		result.Tour[0] == exampleStartVertex &&
		result.Tour[len(stopNames)] == exampleStartVertex)
	fmt.Printf("directed-risk-cost=%.1f\n", result.Cost)
	fmt.Printf("exact=%v optimal=%v\n", result.Exact, result.Optimal)

	fmt.Print("route=")
	for step, vertex := range result.Tour {
		if step > 0 {
			fmt.Print(" -> ")
		}
		fmt.Print(stopNames[vertex])
	}
	fmt.Println()

	// Output:
	// Cash-in-transit directed route
	// closed-loop=true
	// directed-risk-cost=34.0
	// exact=false optimal=false
	// route=Secure Bank Vault -> Downtown Retail Mega-Mall -> Financial District Transit Terminal -> Westside Casino Hub -> Central Airport Cash-Vault -> Secure Bank Vault
}

// ExampleSolveMatrix_semiconductorLaserDrilling demonstrates exact matrix TSP
// for high-precision semiconductor manufacturing.
//
// Business story:
// A CNC-guided laser drilling workstation must visit microscopic via-hole
// positions on each board. A few saved milliseconds per board can become hours
// of recovered production time across large daily batches. In this offline
// programming stage, near-optimal is not enough: the route must be globally
// optimal for the provided latency matrix.
//
// Algorithm choice:
// BranchAndBound with OneTreeBound performs exact search with an admissible
// lower bound. The result is exact and optimal when the search completes.
func ExampleSolveMatrix_semiconductorLaserDrilling() {
	const dailyBoardVolume = 50000

	siteNames := []string{
		"Servo Home Anchor",
		"Pin A1",
		"Bus Gate X7",
		"Core Array",
		"I/O Bridge",
	}

	// Static symmetric mechanical latency matrix, measured in milliseconds.
	//
	// The values include servo-axis travel, deceleration, settling latency, and
	// laser stabilization overhead between drilling sites.
	servoLatencyMilliseconds := []float64{
		0, 3.6, 9.5, 10.0, 4.5,
		3.6, 0, 1.2, 8.0, 9.0,
		9.5, 1.2, 0, 2.8, 7.5,
		10.0, 8.0, 2.8, 0, 4.7,
		4.5, 9.0, 7.5, 4.7, 0,
	}

	dist, err := matrix.NewDense(len(siteNames), len(siteNames))
	if err != nil {
		fmt.Println("laser-drilling matrix allocation failed:", err)
		return
	}
	if err = dist.Fill(servoLatencyMilliseconds); err != nil {
		fmt.Println("laser-drilling matrix initialization failed:", err)
		return
	}

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.BranchAndBound
	opts.Symmetric = true
	opts.StartVertex = exampleStartVertex
	opts.BoundAlgo = tsp.OneTreeBound
	opts.EnableLocalSearch = false
	opts.Eps = exampleEps

	result, err := tsp.SolveMatrix(dist, nil, opts)
	if err != nil {
		fmt.Printf("laser drilling optimization failed: %v\n", err)
		return
	}
	if result == nil {
		fmt.Println("unexpected nil route result")
		return
	}

	dailyRuntimeSeconds := result.Cost * float64(dailyBoardVolume) / 1000.0

	fmt.Println("Semiconductor laser-drilling route")
	fmt.Printf("closed-loop=%v\n", len(result.Tour) == len(siteNames)+1 &&
		result.Tour[0] == exampleStartVertex &&
		result.Tour[len(siteNames)] == exampleStartVertex)
	fmt.Printf("single-board-latency=%.1f ms\n", result.Cost)
	fmt.Printf("daily-laser-motion=%.1f seconds\n", dailyRuntimeSeconds)
	fmt.Printf("exact=%v optimal=%v\n", result.Exact, result.Optimal)

	fmt.Print("path=")
	for step, vertex := range result.Tour {
		if step > 0 {
			fmt.Print(" -> ")
		}
		fmt.Print(siteNames[vertex])
	}
	fmt.Println()

	// Output:
	// Semiconductor laser-drilling route
	// closed-loop=true
	// single-board-latency=16.8 ms
	// daily-laser-motion=840.0 seconds
	// exact=true optimal=true
	// path=Servo Home Anchor -> Pin A1 -> Bus Gate X7 -> Core Array -> I/O Bridge -> Servo Home Anchor
}
```

**Operational interpretation.** A thermal window is a separate release policy,
not a parameter magically enforced by Christofides. Similarly, a directed
risk score must be calibrated by the caller; a low score does not certify
physical security. The drilling example establishes the optimum for the
provided static machine-latency model, not for an unmodeled industrial plant.

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

> **Do not confuse default Blossom selection with verified metric input.**
>
>    `DefaultOptions()` already selects `BlossomMatch`; the 1.5 theorem additionally requires a complete symmetric **metric** matrix. Selecting GreedyMatch deliberately forfeits the formal bound.

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

> **Use `BlossomMatch` with a verified metric matrix for Christofides guarantees.**
>
>    It is the default matching policy. The 1.5 ratio is a conditional theorem, not an empirical certificate of a particular tour.

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
