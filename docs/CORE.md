<!--
  lvlath - Repository Documentation

  Purpose:
    This document is the repository-level specification, tutorial, and contract map
    for lvlath/core. It explains the graph model, deterministic enumeration laws,
    configuration capabilities, concurrency rules, topology invariants, sentinel
    error protocol, and operational patterns that all downstream algorithms rely on.

  Contract status:
    - Public ordering rules described here are part of the core contract.
    - Capability and policy rules described here are part of the core contract.
    - Sentinel-error semantics described here are part of the core contract.
    - Clone, view, and topology-preservation rules described here are part of the core contract.
    - Any incompatible change must be explicit, documented, and versioned.

  Scope:
    - Deterministic in-memory graph representation.
    - Explicit graph capabilities: directedness, weights, loops, multi-edges, mixed edges.
    - Stable topology mutation and query semantics.
    - Concurrency model for safe reads and controlled mutations.
    - Clone/view/subgraph derivation for downstream algorithm pipelines.

  License:
    The lvlath repository is licensed under AGPL-3.0-only. See LICENSE.
-->

# Core Package: Architectural Specification

> **Package:** `lvlath/core` | **Focus:** Determinism, Concurrency, Correctness

The `core` package provides a **foundational, thread-safe, in-memory graph** implementation. It is engineered to serve as the bedrock for complex algorithms where **reproducibility** and **mathematical precision** are non-negotiable.

---

## 2.1. What & Why
Why choose `lvlath/core` over standard `gonum` or `map[int][]int` implementations?
### The Three Laws of Core:

1.  **Absolute Determinism (Binary Reproducibility)**
    *   *Others:* Rely on Go's random map iteration. Running a Greedy Coloring algorithm twice yields different results.
    *   *Lvlath:* Enforces **strict lexicographical ordering** at every API boundary (`Vertices`, `Edges`, `Neighbors`). Your tests, logs, and algorithms behave identically on every run, on every machine.

2.  **Topological Atomicity (Transaction Safety)**
    *   *Others:* Often suffer from "Gap Conditions" where a vertex is deleted while an edge is being added, leading to panics or dangling pointers.
    *   *Lvlath:* Implements a **Global Transactional Lock Model**. Operations like `AddEdge` lock the entire topology. It is mathematically impossible to add an edge to a non-existent vertex.

3.  **Explicit Capability Model (No Magic)**
    *   *Others:* Implicitly allow self-loops or multi-edges, causing algorithms to fail silently later.
    *   *Lvlath:* Capabilities are **Opt-In**. If you didn't enable `WithLoops()`, the graph will reject a loop with a clear error. The constraints are enforced at the write-barrier.
---

## 2.2. Math Formulation

The graph $G$ is defined as $G = (V, E)$, adhering to strict graph theory conventions regarding connectivity.

### Degree Calculation Policy

Let $\deg(v)$ be the total degree of vertex $v$. The implementation of `Degree(v)` calculates:

$$ \deg(v) = \deg_{in}(v) + \deg_{out}(v) + \deg_{undir}(v) $$

**Contribution Table:**

| Edge Type           | Notation  | Contribution Rule                  | Math Justification                                                |
|:--------------------|:----------|:-----------------------------------|:------------------------------------------------------------------|
| **Directed**        | $u \to v$ | $u$: $+1$ out<br>$v$: $+1$ in      | Standard incidence.                                               |
| **Undirected**      | $u - v$   | $u$: $+1$ undir<br>$v$: $+1$ undir | Symmetric incidence.                                              |
| **Directed Loop**   | $v \to v$ | $v$: **$+1$ in, $+1$ out**         | The edge leaves $v$ and enters $v$. Total degree contribution: 2. |
| **Undirected Loop** | $v - v$   | $v$: **$+2$ undir**                | The edge is incident to $v$ twice. Total degree contribution: 2.  |

> **Note:** The `Degree` method performs an $O(E)$ scan to calculate $\deg_{in}$ correctly without maintaining an expensive reverse-index during mutations.

---

## 2.3. Constructor Patterns

The package utilizes the **Functional Options Pattern** to enforce immutability of configuration flags after construction.

```go
// Pattern A: "The Social Graph" (Simple)
// Use Case: Users and Friendships.
// Behavior: Undirected (friendship is mutual), Unweighted, No Loops (cannot friend self).
socialGraph, _ := core.NewGraph()

_ = socialGraph.AddVertex("Alice")
_ = socialGraph.AddVertex("Bob")
_, _ = socialGraph.AddEdge("Alice", "Bob", 0) // Friendship established


// Pattern B: "The CI/CD Pipeline" (Weighted DAG)
// Scenario: A task execution system where time is money.
// Behavior: Directed (Dependency flow). Weighted (Execution time in seconds). No Loops (Circular dependencies are fatal).
pipeline, _ := core.NewGraph(
    core.WithDirected(true), // Dependencies flow one way
    core.WithWeighted(),     // Edge weight = Duration
)
// Defining the Critical Path
_, _ = pipeline.AddEdge("git_clone", "build_core", 45.0)
_, _ = pipeline.AddEdge("build_core", "run_tests", 120.0)
_, _ = pipeline.AddEdge("build_core", "build_ui", 60.0) // Parallel task
_, _ = pipeline.AddEdge("run_tests", "deploy", 15.0)

// Note: AddEdge would return ErrLoopNotAllowed if we tried "deploy" -> "git_clone"


// Pattern C: "The Urban Traffic System" (Complex Multigraph)
// Scenario: A realistic city map with roads, flight paths, and logistics.
// Behavior:
//    Mixed: Some streets are one-way (Directed), avenues are two-way (Undirected).
//    Multi: A tunnel and a bridge can connect the same two districts (Parallel edges).
//    Loops: Roundabouts (Internal routing).
cityMap, _ := core.NewGraph(
    core.WithMixedEdges(), // Essential for mixing one-way and two-way streets
    core.WithMultiEdges(), // Allows Tunnel AND Bridge between same points
    core.WithLoops(),      // Roundabouts
    core.WithWeighted(),   // Distance in km
)

// 1. Two-way Avenue (Undirected Override)
idMain, _ := cityMap.AddEdge("Downtown", "Uptown", 5.2, core.WithEdgeDirected(false))
// 2. One-way Express Lane (Directed Override) - Parallel to Main St!
idExpress, _ := cityMap.AddEdge("Downtown", "Uptown", 3.1, core.WithEdgeDirected(true))
// 3. Roundabout inside Downtown (Self-Loop)
_, _ = cityMap.AddEdge("Downtown", "Downtown", 0.5, core.WithEdgeDirected(true))

fmt.Printf("Route 1 (Main St): %s\nRoute 2 (Express): %s\n", idMain, idExpress)
```

---

## 2.4. Core API & Complexity

We prioritize **correctness** and **safety** over raw $O(1)$ speed in specific mutation scenarios.

| Method          | Complexity    | Analysis                                                                                                                        |
|:----------------|:--------------|:--------------------------------------------------------------------------------------------------------------------------------|
| `AddVertex`     | $O(1)$        | Amortized map insertion.                                                                                                        |
| `AddEdge`       | $O(1)$        | **Transactional.** Locks `muVert` then `muEdgeAdj`. Creation of endpoints is included.                                          |
| `RemoveEdge`    | $O(1)^*$      | Deletion + cleanup of empty adjacency buckets (minor overhead).                                                                 |
| `RemoveVertex`  | **$O(E)$**    | **Heavy Operation.** Must scan the entire edge catalog to remove all incident edges (incoming & outgoing) to preserve topology. |
| `Degree(id)`    | **$O(E)$**    | **Correctness Trade-off.** Scans edges to count in-degree accurately for directed graphs.                                       |
| `Neighbors(id)` | $O(d \log d)$ | $d$ = degree. $\log d$ cost is for enforcing deterministic sorting of pointers.                                                 |
| `Vertices`      | $O(V \log V)$ | Snapshot + Sort.                                                                                                                |
| `Edges`         | $O(E \log E)$ | Snapshot + Sort.                                                                                                                |
| `Clone`         | $O(V+E)$      | Atomic deep copy of topology.                                                                                                   |

---

## 2.5. Pseudocode: The Locking Hierarchy

To prevent deadlocks while ensuring atomic updates, `core` enforces a strict **Global Lock Order**:

1.  **Level 1 (Highest):** `muVert` (Vertices & Configuration)
2.  **Level 2 (Lowest):** `muEdgeAdj` (Edges & Adjacency)

**The `AddEdge` Transaction Logic:**

```text
FUNCTION AddEdge(u, v, weight):
    // 1. Validate Statics
    IF weight != 0 AND NOT g.Weighted: RETURN ErrBadWeight

    // 2. Begin Topology Transaction (Level 1)
    LOCK(muVert)
        // Ensure endpoints exist (Atomic creation)
        IF NOT Exists(u): CreateVertex(u)
        IF NOT Exists(v): CreateVertex(v)

        // 3. Lock Adjacency (Level 2)
        LOCK(muEdgeAdj)
            // Critical Section: No one can remove vertices or edges now
            CHECK MultiEdge Policy
            CREATE Edge Object
            ASSIGN Monotonic ID (e.g., "e105")
            
            INSERT to Edges Map
            UPDATE Adjacency[u][v]
            IF Undirected: UPDATE Adjacency[v][u]
        UNLOCK(muEdgeAdj)
    UNLOCK(muVert)
    // Transaction Complete
    RETURN EdgeID
```

---

## 2.6. ASCII Diagrams: Memory Layout & Configuration Visualized

Understanding how `NewGraph` options affect topology and storage.

### Option: `WithDirected(bool)`
Defines the default orientation of edges.

**Case A: Directed (`true`)**
Edges act as a one-way stream. Storage is efficient (single entry).
```text
   [A] ──(e1)──▶ [B]

   Storage:
   adj[A][B] = "e1"  (only)
   adj[B] has no record of A
```

**Case B: Undirected (`false`)**
Edges act as a two-way bond. Storage is mirrored for $O(1)$ bidirectional traversal.
```text
   [A] ◀──(e1)──▶ [B]

   Storage:
   adj[A][B] = "e1"
   adj[B][A] = "e1"  (Mirror)
```

---

### Option: `WithWeighted()`
Controls whether edges carry a numerical cost (payload).

**Case A: Weighted (`true`)**
Edges carry a float64 payload. Useful for distances, latency, or capacity.
```text
         (weight: 15.5)
   [A] ───────15.5──────▶ [B]

   Edge Struct: { ID: "e1", Weight: 15.5, ... }
```

**Case B: Unweighted (Default)**
Weight is strictly `0`. Attempting to add weight returns `ErrBadWeight`.
```text
         (weight: 0)
   [A] ─────────────────▶ [B]

   Edge Struct: { ID: "e1", Weight: 0.0, ... }
```

---

### Option: `WithLoops()`
Controls self-referential edges ($v \to v$).

**Case A: Loops Enabled (`true`)**
Useful for state machines (remain in state) or network routing (internal processing).
```text
      ╭──(e1)──╮
      │        │
      ▼        │
     [A] ──────╯

   Degree Contribution:
   Directed:   In+1, Out+1 (Total 2)
   Undirected: Undir+2     (Total 2)
```

**Case B: Loops Disabled (Default)**
Enforces strict simple graphs.
```text
     [A] --X--> [A]

   Result: Returns `ErrLoopNotAllowed`
```

---

### Option: `WithMultiEdges()`
Controls parallel edges between the same two vertices.

**Case A: Multi-Edges Enabled (`true`)**
Allows multiple distinct connections (e.g., "Flight" and "Train" between cities).
```text
        ┌──────(e1)──────┐
   [A] ═╡                ╞═▶ [B]
        └──────(e2)──────┘

   Storage (Nested Map):
   adj[A][B] = { "e1":{}, "e2":{} }
```

**Case B: Simple Graph (Default)**
Enforces strict singularity.
```text
   [A] ──(e1)──▶ [B]
   [A] --(e2)-X- [B]

   Result: Second AddEdge returns `ErrMultiEdgeNotAllowed`
```

---

### Option: `WithMixedEdges()`
Controls the ability to override direction on a per-edge basis.

**Case A: Mixed Mode (`true`)**
Coexistence of directed and undirected edges in one instance.
```text
   [A] ──(e1: Road)── [B]   <-- Undirected (Override)
    │
    ╰──(e2: River)──▶ [C]   <-- Directed
```

**Case B: Uniform Mode (Default)**
All edges must follow the global `Directed` setting.
```text
   Graph Default: Directed
   Attempt: AddEdge(..., WithEdgeDirected(false))

   Result: Returns `ErrMixedEdgesNotAllowed`
```

### Memory Layout & Separation of Concerns
The architecture separates **Policy** (Rules) from **Topology** (Data).

```text
      [ USER CALL ] -> g.AddEdge("A", "B")
            │
            ▼
+---------------------------------------------------------------+
| GRAPH INSTANCE (The Fortress)                                 |
|---------------------------------------------------------------|
| 🔒 muVert (Policy Lock)                                       |
|    Protects the "Universe" of vertices and immutable rules.   |
|                                                               |
|    [ CONFIG ] Directed: Mixed | Weighted: Yes | Loops: No     |
|    [ STATE  ] nextEdgeID: 105                                 |
|    [ DATA   ] vertices map:                                   |
|               "A" -> 0x1234 (Vertex Ptr)                      |
|               "B" -> 0x5678 (Vertex Ptr)                      |
|---------------------------------------------------------------|
| 🔒 muEdgeAdj (Topology Lock)                                  |
|    Protects the connections. Can only be taken if muVert      |
|    is already held (prevents deadlocks).                      |
|                                                               |
|    [ DATA   ] edges map:                                      |
|               "e105" -> Edge{From:"A", To:"B", W:10}          |
|                                                               |
|    [ INDEX  ] adjacencyList (Nested Map Trie):                |
|               map[FromID]                                     |
|                  └─ map[ToID]                                 |
|                       └─ map[EdgeID]struct{}                  |
+---------------------------------------------------------------+
```

Visualizing how `Graph` stores and manages data:
```text
+-------------------------------------------------------------+
|                        GRAPH INSTANCE                       |
|-------------------------------------------------------------|
| Config (Immutable): Directed, Weighted, Loops, ...          |
| Locks: muVert (RWMutex), muEdgeAdj (RWMutex)                |
| State: nextEdgeID (uint64)                                  |
+-------------------------------------------------------------+
       |                           |
       v (Guarded by muVert)       v (Guarded by muEdgeAdj)
+----------------------+    +---------------------------------+
| VERTICES CATALOG     |    | EDGES CATALOG                   |
| map[string]*Vertex   |    | map[string]*Edge                |
+----------------------+    +---------------------------------+
| "A" -> {ID: "A"}     |    | "e1" -> {From:"A", To:"B"}      |
| "B" -> {ID: "B"}     |    | "e2" -> {From:"B", To:"A"}      |
+----------------------+    +---------------------------------+
          ^                                |
          | (Referenced by ID)             |
          +--------------------------------+
          |
          v (Guarded by muEdgeAdj)
+-------------------------------------------------------------+
| ADJACENCY LIST (Optimized for Lookup)                       |
| map[from_id] map[to_id] map[edge_id] struct{}               |
+-------------------------------------------------------------+
| "A": {                                                      |
|    "B": { "e1": {} }   <-- O(1) Check: HasEdge("A","B")     |
| }                                                           |
+-------------------------------------------------------------+
```
---

## 2.7. Go Example

A complete lifecycle demonstration.
```go
package main

import (
    "fmt"
    "github.com/lvlath/go/core"
)

func main() {
    // 1. Initialize Ukraine Transport Map
    // Needs Weights (Distance) and Mixed Edges (Roads vs One-ways)
    g, _ := core.NewGraph(
        core.WithWeighted(),
        core.WithMultiEdges(),
        core.WithMixedEdges(),
    )

    // 2. Add Undirected Roads (Distance in km)
    // AddEdge atomically creates vertices "Kyiv", "Lviv", etc.
    id1, _ := g.AddEdge("Kyiv", "Lviv", 537)
    id2, _ := g.AddEdge("Kyiv", "Odesa", 508)
    id3, _ := g.AddEdge("Kyiv", "Lutsk", 423)
    id4, _ := g.AddEdge("Kyiv", "Vinnytsia", 230)
    
    // Cross connections
    _, _ = g.AddEdge("Lviv", "Vinnytsia", 377)
    _, _ = g.AddEdge("Odesa", "Yuzhnoukrainsk", 148)

    fmt.Printf("Network initialized. Edge IDs: %s, %s, %s, %s\n", id1, id2, id3, id4)

    // 3. Inspect Neighbors (Deterministic Order)
    // Neighbors() sorts edges by ID, ensuring this print order is stable.
    fmt.Println("\n--- Connections from Kyiv ---")
    nbs, _ := g.Neighbors("Kyiv")
    for _, e := range nbs {
        fmt.Printf(" -> To: %-10s | Dist: %.0f km | EdgeID: %s\n", e.To, e.Weight, e.ID)
    }

    // 4. Atomic Snapshot
    // Create an unweighted view to count hops instead of distance
    hopsView := core.UnweightedView(g)

	origEdge, err := g.GetEdge(id1)
	if err != nil {
		panic(err)
	}
	viewEdge, err := hopsView.GetEdge(id1)
	if err != nil {
		panic(err)
	}

	fmt.Printf("\nOriginal Weight (Kyiv-Lviv): %.0f\n", origEdge.Weight) // 537
	fmt.Printf("View Weight (Kyiv-Lviv): %.0f\n", viewEdge.Weight) // 0
}
```
[![Playground - Core](https://img.shields.io/badge/Go_Playground-Core-blue?logo=go)](https://go.dev/play/p/sjcxoMWV4SF)
---

## 2.8. Pitfalls & Best Practices (Architectural Mastery)

This section details how to leverage `core` for high-load systems.

### ️ 1. The ID Correlation Pattern
**Use `WithID` for Distributed Tracing.**
While auto-IDs (`e1`, `e2`) are great for simple tests, distributed systems should enforce their own UUIDs to correlate graph edges with database rows.
```go
// Recommended for Production:
g.AddEdge("UserA", "ProductB", 0, core.WithID("uuid-550e-8400..."))
```
*Benefit:* `Clone()` and `Views` preserve these IDs, allowing you to map results back to your DB seamlessly.

### 2. The View-Propagation Strategy
**Zero-Copy Logic via Views.**
If you need to run a standard algorithm (like BFS) on a weighted graph, do not manually strip weights. Use `core.UnweightedView(g)`.
*   **Why?** It acts as an $O(V+E)$ copy, but it guarantees safety.
*   **Bonus:** The `nextEdgeID` counter is carried over. If you add *new* edges to the View, they will strictly follow the sequence of the original graph (`e100` -> `e101`), preventing collision confusion during debugging.

### 3. The Metadata Ownership Model
**Aliasing Awareness.**
`Vertex.Metadata` is a `map`. `Clone()` performs a shallow copy of the map pointer.
*   **The Power:** Extremely fast cloning. Memory efficient.
*   **The Risk:** Modifying metadata in a clone affects the original.
*   **The Fix:** If you need transactional isolation on metadata, treat the map inside `Vertex` as **Immutable**. Replace the whole map pointer if you need to update data, rather than mutating keys inside.

### 4. Avoiding the $O(E)$ Trap
**Degree vs. Neighbors.**
*   **Don't:** Call `Degree(v)` inside a hot loop (like checking termination conditions in a simulation). It scans all edges.
*   **Do:** Use `len(Neighbors(v))` if you only care about outgoing connections (in directed graphs). It is $O(d)$.

---
**lvlath/core**: Designed for precision. Built for scale.

> Next: [3. BFS (Breadth-First Search) →](BFS.md)