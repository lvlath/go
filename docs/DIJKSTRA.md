<!--
  lvlath - Repository Documentation

  Purpose:
    This document is the repository-level specification, tutorial, and contract map
    for lvlath/dijkstra. It explains the weighted shortest-path kernel, numeric
    policy, determinism laws, traversal-policy gates, result-surface semantics,
    and safe production usage.

  Contract status:
    - Determinism rules (graph-surface order, heap tie-break, strict improvement)
      are part of the public contract.
    - Numeric policy (finite non-negative graph-edge weights, `+Inf` result and
      option semantics, non-finite input rejection, and explicit arithmetic-overflow
      classification) is part of the public contract.
    - Sentinel-error governance and errors.Is-first matching are part of the
      public contract.
    - Result-surface semantics (detached ownership, tracking-enabled path witness,
      partial-result suppression) are part of the public contract.
    - Any incompatible change must be explicit, documented, and versioned.

  Scope:
    - Deterministic single-source shortest paths over weighted core.Graph.
    - Runtime traversal policy through MaxDistance and InfEdgeThreshold.
    - Deterministic reconstruction of one shortest-path witness.
    - Directed, undirected, and mixed-edge endpoint resolution laws.

  License:
    The lvlath repository is licensed under AGPL-3.0-only. See LICENSE.
-->

# 5. Shortest Paths: Dijkstra

> **Package:** `lvlath/dijkstra` | **Focus:** Determinism, Numeric Discipline, Result Contracts, Traversal Policy, Stable Witnesses

The `lvlath/dijkstra` package provides a deterministic, contract-driven implementation of single-source shortest paths over `core.Graph`. It is designed not as a throwaway priority-queue snippet, but as a reusable weighted shortest-path kernel for downstream packages and applications that require reproducible costs, stable witnesses, explicit unreachable semantics, strict numeric governance, and detached caller-owned result artifacts.

Unlike naive implementations that silently drift under unstable ordering, blur unknown-vs-unreachable states, or become semantically unsafe under concurrent graph mutation, `lvlath/dijkstra` treats traversal as a strict mathematical protocol.

---

## 5.1. What & Why

Why use `lvlath/dijkstra` instead of a custom heap loop?

### The Five Laws of Dijkstra

1. **Deterministic Weighted Routing**
    *   *Others:* Return one of several equal-cost paths depending on incidental adjacency map order or unstable heap behavior.
    *   *Lvlath:* Preserves the deterministic graph-surface order from `core`, adds an explicit heap tie-break on vertex ID, and applies strict-improvement-only predecessor updates. The chosen witness is binary-stable for the same graph state and runtime policy.
2. **Explicit Numeric-Domain Separation**
    *   *Others:* Reuse one numeric sentinel across storage, options, arithmetic, and results, making it impossible to distinguish corrupted input from valid unreachability.
    *   *Lvlath:* Requires finite non-negative graph-edge weights. `+Inf` is valid only as an unbounded option value or a known-unreachable result value. If finite distance arithmetic overflows to `+Inf`, the package returns `ErrDistanceOverflow` and publishes no partial result.
3. **Traversal Policy Without Topology Mutation**
    *   *Others:* Force callers to physically delete edges or clone the graph to model "blocked" roads or "out-of-radius" conditions.
    *   *Lvlath:* Applies `WithInfEdgeThreshold` and `WithMaxDistance` as runtime policy gates. The topology remains unchanged by the call.
4. **Detached Result Surface**
    *   *Others:* Return ad-hoc internal maps or couple result semantics to implementation detail.
    *   *Lvlath:* Publishes a detached `*Result` that exposes explicit distance, reachability, and path-witness queries.
5. **Graph-Semantics Fidelity**
    *   *Others:* Assume the traversable destination is always `edge.To`, which silently breaks on undirected or mixed-edge relations.
    *   *Lvlath:* Resolves the effective opposite endpoint relative to the current vertex. Mixed and undirected traversal therefore remains mathematically correct even when internal edge storage orientation differs from the traversal direction.

---

## 5.2. Domain Scope & Non-Goals

### 5.2.1. Domain Scope
The package answers weighted shortest-path questions such as:
*   "What is the minimum total cost from this source to every known vertex?"
*   "What is the minimum cost to this one target?"
*   "Is this known target reachable under the active traversal policy?"
*   "What is one deterministic shortest-path witness to this target?"

### 5.2.2. Explicit Non-Goals
`lvlath/dijkstra` is intentionally **not**:
*   **A negative-weight solver:** If any traversed edge can have a negative finite weight, Dijkstra is mathematically the wrong tool. The package actively rejects this input.
*   **An all-pairs solver:** It does not provide Floyd-Warshall or Johnson-style orchestration.
*   **An exhaustive enumerator:** It returns exactly *one* deterministic witness, not a catalog of all possible shortest paths.
*   **A snapshot-isolated engine:** It does not materialize an isolated graph image before traversal.

---

## 5.3. Mathematical Formulation

The package differentiates between classic theoretical definitions and our strict runtime application. Let $G = (V, E)$ be a graph whose edge-weight function is:

$$
w: E \rightarrow \mathbb{R}_{\ge 0}
$$

with the additional storage law:

$$
\forall e \in E,\quad w(e)\ \text{is finite}
$$

The extended result distance belongs to:

$$
dist(s,v) \in \mathbb{R}_{\ge 0} \cup \{+\infty\}
$$

Here `+Inf` belongs to the result domain and means that `v` is known but no
admissible representable path was published.

### 5.3.1. Classical Shortest-Path Objective
For a source $s$ and target $v$, the weighted shortest-path objective is the infimum of the sum of edge weights over all possible paths $\mathcal{P}(s,v)$:
$$ dist(s,v) = \inf_{P \in \mathcal{P}(s,v)} \sum_{e \in P} w(e) $$
If no admissible path exists, $dist(s,v) = +\infty$.

### 5.3.2. Lvlath Implementation (Strict-Improvement)
For the known result domain, initialization dictates $dist(s,s) = 0$ and $dist(s,v) = +\infty$ for all $v \neq s$.
During relaxation of a relation from $u$ to $v$ via edge $e$:
$$ candidate = dist(s,u) + w(e) $$
The package updates state **only** on strict improvement:
$$ \text{If } candidate < dist(s,v) \implies dist(s,v) \leftarrow candidate, \quad \pi(v) \leftarrow u $$
If $candidate == dist(s,v)$, predecessor state is intentionally **not** overwritten.

### 5.3.3. Traversal-Policy Impact (Cutoff Publication Law)
Options geometrically restrict the admissible graph and the publication domain at runtime.

*   **Wall-Threshold Admissibility (`WithInfEdgeThreshold` $\tau$):**
    $$ E_{\text{admissible}} = \{ e \in E \mid w(e) < \tau \} $$
    Edges where $w(e) \ge \tau$ are skipped.
*   **Cutoff Publication Law (`WithMaxDistance` $M$):**
    Candidate paths $> M$ are not explored. The publication domain enforces:
    $$ dist_{\text{pub}}(s,v) = \begin{cases} dist(s,v), & \text{if } v \text{ is finalized under effective policy within } M \\ +\infty, & \text{otherwise} \end{cases} $$

### 5.3.4. Deterministic Tie-Break Law
Frontier priority is resolved using a secondary lexicographical key:
$$ Priority(v) = \langle candidateDistance(v), \text{VertexID}(v) \rangle $$

### 5.3.5. Representability Law
Two valid finite operands can still exceed the representable `float64` range:
$$ finite(a) \land finite(b) \not\Rightarrow finite(a+b) $$

When:
$$ dist(s,u) + w(u,v) = +\infty $$

for an otherwise admissible candidate, the run fails with `ErrDistanceOverflow`.

Overflow is not converted into ordinary unreachable state.

### 5.3.6. Effective Complexity
*   **Validation Phase:** Depends on option validation and $O(E \log E)$ graph-surface numeric pre-scans.
*   **Kernel Phase:** Effective time $\mathcal{O}(V \log V + E \log V + E_{\text{surface}})$. Space $\mathcal{O}(V + E_{\text{heap}})$.
*   **Result Surface:** `DistanceTo` is $\mathcal{O}(1)$. `PathTo` is $\mathcal{O}(k)$ where $k$ is path length.

---

## 5.4. Public API & Result Contract

### 5.4.1. Public Entry Points & Wrapper Semantics
```go
// 1. Canonical Execution
func Dijkstra(g *core.Graph, sourceID string, opts ...Option) (*Result, error)

// 2. Convenience Wrappers (Facades over the canonical result model)
func Distances(g *core.Graph, sourceID string, opts ...Option) (map[string]float64, error)
func DistanceTo(g *core.Graph, sourceID, targetID string, opts ...Option) (float64, error)
func ShortestPathTo(g *core.Graph, sourceID, targetID string, opts ...Option) ([]string, float64, error)
```
*   `Distances(...)` publishes a detached distance map only.
*   `DistanceTo(...)` is a point-query wrapper for isolated distance checks.
*   `ShortestPathTo(...)` forces `WithPathTracking` internally and returns one witness.

### 5.4.2. Canonical Result Artifact
```go
type Result struct {
	SourceID  string
	Distances map[string]float64
	Prev      map[string]string
}
```

Field semantics:

* `SourceID` is the explicit source vertex identifier used for the run.
* `Distances` stores finalized shortest-path costs for the known result domain.
* `Prev` stores predecessor links only when path tracking is enabled.
* `Prev == nil` means path tracking was disabled, not that the graph has no reachable paths.


### 5.4.3. Target-Domain States
For a queried vertex, domain membership determines the primary outcome:

| Target State          | Meaning                                        | Result Action                         |
|:----------------------|:-----------------------------------------------|:--------------------------------------|
| **Unknown Target**    | Vertex is absent from the result domain.       | Returns `ErrTargetNotFound`           |
| **Known Reachable**   | Exists in domain, reachable under policy.      | Returns finite `float64`, `nil` error |
| **Known Unreachable** | Exists in domain, walled off or out-of-bounds. | Returns `+Inf`, `nil` error           |

### 5.4.4. Path-Tracking States
Path reconstruction depends on a separate, independent axis:

| Tracking State               | Meaning                                 | `PathTo` Action                            |
|:-----------------------------|:----------------------------------------|:-------------------------------------------|
| **Enabled (`Prev != nil`)**  | Tracked during kernel run.              | Yields path (if reachable), or `ErrNoPath` |
| **Disabled (`Prev == nil`)** | Explicitly disabled to save allocation. | Returns `ErrPathTrackingDisabled`          |

---

## 5.5. Options, Numeric & Error Policies

### 5.5.1. Numeric Domain Policy

Numeric values have different meanings in different package domains.

| Domain              | Valid values            | Meaning of `+Inf`         |
|:--------------------|:------------------------|:--------------------------|
| `core.Edge.Weight`  | finite values           | invalid                   |
| Dijkstra edge input | finite `w >= 0`         | invalid                   |
| `MaxDistance`       | finite `>= 0` or `+Inf` | no distance cutoff        |
| `InfEdgeThreshold`  | finite `> 0` or `+Inf`  | no finite edge is blocked |
| `Result.Distances`  | finite `>= 0` or `+Inf` | known but unreachable     |

Edge classification:

```text
NaN             -> ErrInvalidWeight
+Inf edge       -> ErrInvalidWeight
-Inf edge       -> ErrInvalidWeight
finite w < 0    -> ErrNegativeWeight
finite w >= 0   -> accepted
```

Arithmetic classification:

```text
finite current distance
+ finite edge weight
= +Inf
    -> ErrDistanceOverflow
    -> nil Result
```

`InfEdgeThreshold` is not permission to store infinite edges. It filters valid
finite edges by policy.

### 5.5.2. Option Governance
Options are explicit runtime policy inputs, not hidden mutable state.

- options are applied sequentially,
- last-writer-wins per field,
- nil options are invalid,
- invalid explicit option input fails before traversal begins.

Public policy surface:
```go
WithPathTracking()
WithMaxDistance(max)
WithInfEdgeThreshold(threshold)
```

Default runtime policy:

* `TrackPaths       = false`
* `MaxDistance      = +Inf`
* `InfEdgeThreshold = +Inf`

Important separation:

* `sourceID` is **not** an option,
* `sourceID` is an explicit required public API argument.

Because `Option` is publicly constructible, canonical assembly revalidates the
entire `Options` state after every option. A custom option cannot bypass
`MaxDistance` or `InfEdgeThreshold` numeric invariants merely by returning `nil`.

### 5.5.3. Error Law

Exported sentinels are the single source of truth for protocol matching.

Callers must use:

```
errors.Is(err, ...)
```

and must not parse error strings.

Primary sentinels include:

* `ErrNilGraph`
* `ErrEmptySourceID`
* `ErrSourceNotFound`
* `ErrTargetNotFound`
* `ErrUnweightedGraph`
* `ErrNilOption`
* `ErrBadMaxDistance`
* `ErrBadInfEdgeThreshold`
* `ErrNegativeWeight`
* `ErrInvalidWeight`
* `ErrDistanceOverflow`
* `ErrPathTrackingDisabled`
* `ErrNoPath`
* `ErrEmptyTargetID`
* `ErrNilResult`

Error-class separation is intentional:

* missing source != missing target,
* invalid numeric input != negative finite weight,
* tracking disabled != no path,
* unknown target != known unreachable target.

When additional context is attached, the sentinel must be preserved with `%w`.

`ErrDistanceOverflow` is intentionally distinct from both `ErrInvalidWeight`
and a published `+Inf` distance:

* `ErrInvalidWeight` means the edge input itself is non-finite;
* `ErrDistanceOverflow` means valid finite operands produced an unrepresentable sum;
* published `+Inf` means no admissible representable route was found.

Panic-based option validation is forbidden.

---


## 5.6. Determinism Law

Determinism is a package-level contract.

### 5.6.1. Vertex-Domain Initialization
The known result domain is initialized by iterating:
```go
g.Vertices()
```

as surfaced by `core.Graph`.

### 5.6.2. Relation Enumeration

Relaxation processes candidate relations in the exact order returned by:

```go
g.Neighbors(u)
```

The package does not inject a second hidden neighbor-sorting layer.

### 5.6.3. Heap Ordering

Equal-distance frontier items are ordered by:
$$
(candidateDistance,\ vertexID)
$$

That means:

* smaller candidate distance wins,
* equal candidate distance is broken by lexicographically smaller vertex IDs.

### 5.6.4. Strict-Improvement Predecessor Law

Relaxation updates predecessor state only when:
$$
candidate < currentDistance
$$

Equal-cost candidates do **not** overwrite predecessor state.

### 5.6.5. Consequence

For the same:

* graph state,
* `sourceID`,
* and runtime policy,

the package publishes stable:

* finalized distances,
* predecessor selection,
* and reconstructed path witnesses.

---


## 5.7. Algorithmic Architecture & Pseudocode

The implementation utilizes a **Lazy Decrease-Key Min-Heap** bounded by a strict **Visited-Finalization** loop. Duplicates are allowed in the heap; non-authoritative frontier entries are discarded cleanly by the visited-finalization model.

```text
FUNCTION Dijkstra(g, sourceID, opts...):

  Stage 1: Validate admission
    - reject nil graph
    - reject empty or missing source
    - require weighted graph
    - apply options in call order
    - revalidate finalized state after every option
    - pre-scan all edges in deterministic Edge.ID order:
        reject NaN
        reject +Inf and -Inf
        reject finite negative weights

  Stage 2: Initialize result domain
    for each vertex v in g.Vertices():
      distances[v] = +Inf
      visited[v] = false

    distances[sourceID] = 0

    if TrackPaths:
      allocate Prev
    else:
      Prev = nil

  Stage 3: Seed frontier
    push (sourceID, 0)

    heap order:
      (candidateDistance, vertexID)

  Stage 4: Visited-finalization loop
    while frontier is not empty:
      item = pop minimum

      if visited[item.id]:
        continue

      if item.dist > MaxDistance:
        break

      visited[item.id] = true

      for each edge in g.Neighbors(item.id):
        neighborID, ok = otherEndpoint(edge, item.id)

        if not ok or visited[neighborID]:
          continue

        weight = edge.Weight

        reclassify weight defensively:
          reject NaN
          reject +/-Inf
          reject finite negative value

        if weight >= InfEdgeThreshold:
          continue

        currentDistance = distances[item.id]

        if MaxDistance is finite and
           weight > MaxDistance - currentDistance:
          continue

        candidate = currentDistance + weight

        if candidate == +Inf:
          return nil Result + ErrDistanceOverflow

        if candidate < distances[neighborID]:
          distances[neighborID] = candidate

          if TrackPaths:
            Prev[neighborID] = item.id

          push (neighborID, candidate)

  Stage 5: Publish
    return detached Result only after complete success
```

---

## 5.8. ASCII Diagrams: Routing Mechanics

### 5.8.1. Complex 11-Vertex Logistics & Policy Network
This diagram demonstrates how thresholds, cutoffs, and +Inf semantics interact with a complex topology.

```text
         [Gateway](1) ──2──▶[Hub:N](2) ──3──▶ [Sort:River](4) ──2──▶[City:Aurora](8)
              │                   │                   │
              4                   5                   2
              │                   ▼                   ▼
              └────────────▶ [Hub:S](3) ──2──▶ [Sort:Valley](5) ──3─▶ [City:Delta](9)
                                  │                   │
                     (Wall >= 8) 10                   3
                                  ▼                   ▼
                            [Backup:1](6) ──2─▶[City:Ember](10)
                                  │
                                  1
                                  ▼
                            [Backup:2](7) ──2─▶ [City:Isolated](11)
```

*   **Scenario A (Unlimited):** All vertices reached. `City:Ember` is reached via `Sort:Valley` (Cost: $4+2+3 = 9$).
*   **Scenario B (`WithInfEdgeThreshold(8.0)`):** `Hub:S -> Backup:1` is a wall. `Backup:1`, `Backup:2`, and `City:Isolated` are now Known but Unreachable (`+Inf`).
*   **Scenario C (`WithMaxDistance(7.0)`):** `Gateway -> Hub:N -> Sort:River -> City:Aurora` finishes at cost $7.0$. `City:Delta` requires cost $9.0$, which exceeds $7.0$, so it remains `+Inf` (Policy Cutoff).

### 5.8.2. Equal-Cost Determinism
```text
                [relay:alpha]
               /             \
(c=2)         / (c=2)        (c=2)
             /                  \
          [start]           [target]
             \                  /
(c=2)         \ (c=2)        (c=2)
               \             /
                 [relay:beta]
```
Equal candidate distance to `target` (cost $4$) occurs through both `relay:alpha` and `relay:beta`. The heap tie-break on `VertexID` processes `alpha` first. When `beta` is processed, $4 \not< 4$ (Strict Improvement Law). `Prev` is **not** overwritten. The witness `[start -> relay:alpha -> target]` is universally stable.

### 5.8.3. Endpoint Law on Undirected Storage
```text
Stored edge record in core.Graph:
  From: "B", To: "A", Directed: false

Current finalized vertex context in Dijkstra:
  u = "A"

Wrong interpretation (Silent failure):
  v = edge.To = "A"   // Incorrectly points back to self

Correct interpretation (lvlath endpoint law):
  v = otherEndpoint(edge, "A") = "B"
```

### 5.8.4. Result-State Distinctions
```text
Result query priority:

  [receiver == nil]
        -> ErrNilResult

  [targetID == ""]
        -> ErrEmptyTargetID

  [target absent from Distances]
        -> ErrTargetNotFound

  [DistanceTo / HasPathTo]
        |
        +-- finite distance
        |      -> DistanceTo = finite
        |      -> HasPathTo  = true
        |
        +-- distance is +Inf
               -> DistanceTo = +Inf
               -> HasPathTo  = false

  [PathTo]
        |
        +-- Prev == nil
        |      -> ErrPathTrackingDisabled
        |
        +-- target distance is +Inf
        |      -> ErrNoPath
        |
        +-- valid source-anchored predecessor chain
        |      -> deterministic path
        |
        +-- missing, cyclic, or out-of-domain predecessor
               -> ErrNoPath
               -> nil path
```

---

## 5.9. Go Example Scenarios

The package examples are intentionally scenario-driven, modeling real pipelines: `build -> dijkstra -> consume`.

### 5.9.1. The Unified Master Pipeline (Expanded)
Demonstrates graph construction, dynamic wall thresholds, rigorous error handling, and stable witness extraction.

```go
package main

import (
	"fmt"
	"math"
	"strings"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dijkstra"
)

func main() {
	// 1. Build a mixed-edge distribution network.
	g, _ := core.NewGraph(core.WithWeighted(), core.WithMixedEdges())

	// Explicitly check AddEdge errors in production
	if _, err := g.AddEdge("gateway", "hub:north", 2.0, core.WithEdgeDirected(true)); err != nil {
		fmt.Println("Build error:", err)
		return
	}
	_, _ = g.AddEdge("gateway", "hub:south", 4.0, core.WithEdgeDirected(true))
	_, _ = g.AddEdge("hub:north", "sort:river", 3.0, core.WithEdgeDirected(true))

	// Degraded Highway (Cost: 10.0)
	_, _ = g.AddEdge("hub:south", "backup:node", 10.0, core.WithEdgeDirected(true))
	_, _ = g.AddEdge("backup:node", "city:isolated", 1.0, core.WithEdgeDirected(true))

	// Local Streets (Undirected)
	_, _ = g.AddEdge("sort:river", "city:aurora", 2.0, core.WithEdgeDirected(false))
	_, _ = g.AddEdge("hub:south", "city:aurora", 4.0, core.WithEdgeDirected(false))

	// 2. Runtime Policy Assembly
	opts := []dijkstra.Option{
		dijkstra.WithPathTracking(),        // Required for PathTo()
		dijkstra.WithInfEdgeThreshold(8.0), // The "Wall"
	}

	// 3. Kernel Execution
	result, err := dijkstra.Dijkstra(g, "gateway", opts...)
	if err != nil {
		fmt.Println("Traversal failed:", err)
		return
	}

	// 4. Result Consumption: Reachable target
	target := "city:aurora"
	cost, _ := result.DistanceTo(target)
	path, _ := result.PathTo(target)
	fmt.Printf("Route to %s: %s (Cost: %.0f)\n", target, strings.Join(path, " -> "), cost)

	// 5. Result Consumption: Unreachable target (+Inf semantics)
	isolatedTarget := "city:isolated"
	isolatedCost, _ := result.DistanceTo(isolatedTarget)
	if math.IsInf(isolatedCost, 1) {
		fmt.Printf("Target %q is known but walled off by policy (+Inf).\n", isolatedTarget)
	}

	// Output:
	// Route to city:aurora: gateway -> hub:north -> sort:river -> city:aurora (Cost: 7)
	// Target "city:isolated" is known but walled off by policy (+Inf).
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-Dijkstra_Unified_Master_Pipeline-blue?logo=go)](https://go.dev/play/p/FoZ3eIIvaYM)

### 5.9.2. Scenario Capsules

The package examples are intentionally scenario-driven and follow the same practical pipeline:

```text
build -> dijkstra -> consume
```

### 1) Logistics Routing

**Story:** central warehouse, hubs, sort centers, and destination cities.

**Demonstrates:**

* `Dijkstra(...)`
* `WithPathTracking()`
* `DistanceTo(...)`
* `PathTo(...)`

**Why it matters:**
This is the canonical “cost + one deterministic route” workflow.

### 2) Failover Network

**Story:** weighted network routing with degraded links treated as walls.

**Demonstrates:**

* `WithInfEdgeThreshold(...)`
* known-but-unreachable target state
* explicit `+Inf` publication
* `HasPathTo(...)`

### 3) Service Radius Cutoff

**Story:** bounded delivery or field-service radius.

**Demonstrates:**

* `WithMaxDistance(...)`
* graph path exists, but traversal policy forbids reaching it
* `+Inf` as a cutoff-induced outcome

### 4) Mixed Transit Graph

**Story:** one-way avenues plus two-way streets in one city graph.

**Demonstrates:**

* `core.WithMixedEdges()`
* practical mixed-edge routing
* endpoint-law correctness through a real route

### 5) Equal-Cost Determinism

**Story:** two equal-cost corridors to the same destination.

**Demonstrates:**

* `WithPathTracking()`
* deterministic witness selection
* strict-improvement stability under equal-cost competition


> The full runnable scenario set lives in `dijkstra/examples_test.go`.

---

## 5.10. Ownership, Concurrency & Partial-Result Laws

### 5.10.1. Detached Ownership Law
Published results are detached and caller-owned.

This means:
- the package does not retain a live mutable link from `Result` back to the graph,
- the package does not mutate published result maps after return,
- callers may read, clone, cache, or transform the result after return.

The graph itself remains externally owned by the caller.

### 5.10.2. Concurrency Law
- concurrent reads through the `core` contract are expected usage,
- concurrent topology mutation during execution is unsupported,
- the package does not materialize a snapshot-isolated graph image before traversal.

If correctness and reproducibility matter, graph topology must remain stable for the duration of the call.

### 5.10.3. Partial-Result Suppression Law
On failure, the package does **not** publish a partial `Result`.

Publication rule:
- validation failure $\implies$ `nil` result + error
- runtime kernel failure $\implies$ `nil` result + error
- successful completion $\implies$ detached finalized result publication

A non-nil published `Result` therefore means the run completed successfully.

### 5.10.4. Caller-Owned Witness Integrity

`Result.Distances` and `Result.Prev` belong to the caller after publication.

This ownership permits mutation, but mutation can invalidate result invariants.

`PathTo` therefore fails closed:

* every predecessor must exist in `Distances`;
* every non-source path vertex must have a non-empty predecessor;
* no predecessor vertex may repeat;
* the chain must terminate at `SourceID`;
* no partial path is returned on failure.

Broken, cyclic, or out-of-domain witness state returns `ErrNoPath`.

`Clone()` provides independent map ownership, but cloning does not repair
semantically invalid caller-created data.

---

## 5.11. Pitfalls & Best Practices (Architectural Mastery)

### 5.11.0. Operational pitfalls
* **Do not store `+Inf` as a graph-edge wall.**
  Store only finite edge weights. Use `WithInfEdgeThreshold` to block finite heavy
  edges, or omit the relation from topology.

* **Do not reinterpret overflow as unreachable.**
  `ErrDistanceOverflow` means a representable path cost could not be produced.
  It is not equivalent to a successful `+Inf` result.

* **Do not mutate `Prev` casually.**
  It is caller-owned, but `PathTo` will reject cycles, foreign IDs, and broken chains.

### 5.11.1. Do not parse error strings
Use `errors.Is` with exported sentinels only.
String matching is not part of the contract.

### 5.11.2. Do not run Dijkstra on negative-weight graphs
Finite negative weights are mathematically invalid for Dijkstra and are rejected by the package.
If the domain allows negative weights, switch algorithms rather than forcing this package to answer the wrong question.

### 5.11.3. Do not treat `Prev == nil` as “no path exists”
`Prev == nil` means path tracking was disabled.
Reachability must be queried independently through the result surface.

### 5.11.4. Do not collapse unknown targets and unreachable targets
These are different operational states:
- missing target -> `ErrTargetNotFound`
- known unreachable target -> `+Inf` with `nil` error

### 5.11.5. Do not simplify mixed or undirected traversal to `edge.To`
The effective traversable endpoint must be resolved relative to the current vertex.
Reducing traversal to `edge.To` breaks endpoint correctness on undirected and mixed-edge graphs.

### 5.11.6. Use policy gates instead of topology mutation when the problem is operational
Use:
- `WithInfEdgeThreshold(...)` for degraded or impassable links,
- `WithMaxDistance(...)` for bounded traversal radius or bounded weighted reachability.

Do not mutate graph topology merely to simulate runtime routing policy.

### 5.11.7. Keep `+Inf` as semantic data
Do not rewrite it into synthetic finite sentinels such as `-1` or arbitrary “large” numbers.
`+Inf` is part of the numeric contract.

### 5.11.8. Use convenience wrappers when the consumer only needs a point query
Use:
- `DistanceTo(...)` for one-target distance lookup,
- `ShortestPathTo(...)` when the caller needs one witness and its distance,
- `Distances(...)` when only the detached distance map is required.

### 5.11.9. Keep examples and documentation contract-faithful
Examples should:
- use only current public API,
- keep graph-construction errors explicit,
- print deterministic ordered output,
- and avoid implying stronger guarantees than the package actually provides.

---

> Next:[6. Minimum Spanning Trees: Prim & Kruskal ->](MST.md)
