<!--
  lvlath - Repository Documentation

  Purpose:
    This document is the repository-level specification, tutorial, and contract map
    for lvlath/bfs. It explains what the package guarantees, how the mathematics maps
    to the implementation, where determinism comes from, what partial results mean,
    and how to use BFS and Components correctly in production pipelines.

  Contract status:
    - Behaviors described here are part of the public contract.
    - Determinism rules described here are part of the public contract.
    - Error-classification rules described here are part of the public contract.
    - Result semantics for Result, PathTo, and Components are part of the public contract.
    - Any incompatible change must be explicit, documented, and versioned.

  Scope:
    - Deterministic unweighted BFS from a start vertex.
    - Deterministic weak connectivity discovery.
    - Partial-result semantics for cancellation and runtime interruption.
    - Practical engineering guidance for downstream algorithms.

  License:
    The lvlath repository is licensed under AGPL-3.0-only. See LICENSE.
-->

# 3. BFS (Breadth-First Search)

> **Package:** `lvlath/bfs` | **Focus:** Determinism, Unweighted Shortest Paths, Partial Results, Weak Connectivity, Memory-Safe Traversal

The `lvlath/bfs` package provides a deterministic, contract-driven implementation of Breadth-First Search over `core.Graph`.
It is engineered not as a throwaway helper, but as a foundational algorithmic primitive for packages that need stable hop-distance traversal, reproducible tie-break behavior, and debuggable interruption semantics.

Unlike casual BFS snippets that silently depend on map iteration, blur the meaning of discovery versus visit, or discard useful state on cancellation, `lvlath/bfs` treats traversal as a disciplined protocol:

- the graph layer defines the deterministic neighbor relation,
- the BFS layer preserves and interprets it without hidden policy,
- the result layer exposes stable, mathematically meaningful artifacts.

---

## 3.1. What & Why

Why use `lvlath/bfs` instead of a handwritten queue loop or a generic helper?

### The Four Laws of BFS

1. **Deterministic Frontier Expansion**
    *   *Others:* Frequently inherit unstable order from maps, ad-hoc adjacency containers, or edge enumeration side effects.
    *   *Lvlath:* Expands neighbors in the exact order returned by `core.Graph.NeighborIDs(u)`. BFS itself does not sort neighbors and does not invent a second tie-break rule.

2. **Visit-Time Semantics Are Explicit**
    *   *Others:* Blur the distinction between discovered, queued, and visited, which makes hooks and partial results ambiguous.
    *   *Lvlath:* Defines `visit = dequeue`. `Order` is dequeue order, `Visited` is discovery-at-enqueue, and `OnVisit` runs only when a vertex is actually processed.

3. **Partial Result Is a Valid Outcome**
    *   *Others:* Fail or cancel mid-traversal and throw away useful state.
    *   *Lvlath:* Returns a partial `Result` plus the error on context cancellation, neighbor enumeration failure, or hook failure.

4. **Queue Retention Is Engineered Away**
    *   *Others:* Use `queue = queue[1:]`, which may retain the backing array and keep old references alive longer than intended.
    *   *Lvlath:* Uses a head-index queue with slot clearing, preserving BFS mathematics while improving operational behavior.

### What BFS returns
`lvlath/bfs` returns a result object that is simultaneously:

- a deterministic traversal trace,
- an unweighted shortest-path certificate,
- a policy and observability artifact.

Concretely:

- `Order` answers: in what exact order were vertices processed?
- `Depth` answers: how many hops away is each discovered vertex?
- `Parent` answers: through which predecessor was each vertex first discovered?
- `PathTo` answers: what is one shortest path from the start to a target?
- `Skipped` answers: how many neighbor relations were rejected by traversal policy?

### Why this matters in practice
BFS is the correct primitive when the domain question is fundamentally about fewest hops, layer boundaries, or reachability under a deterministic relation:

- blast-radius analysis in service graphs,
- dependency-wave expansion,
- policy-gated discovery without mutating topology,
- deterministic audit crawlers and scanners,
- weak component discovery via `Components`.

> **Non-goal:** `lvlath/bfs` does not compute weighted shortest paths.
> If the path objective depends on latency, cost, capacity, or risk, BFS is mathematically the wrong tool.

---

## 3.2. Math Formulation

The BFS package operates on an unweighted neighbor relation induced by `core.Graph.NeighborIDs`.
Distances are measured in edge count, not stored edge weight.

### 3.2.1. Hop Distance
For a source vertex `s` and target vertex `v`, BFS computes the minimum number of edges on any valid traversal path:

$$
dist(s,v) = min_{P in Paths(s,v)} |P|
$$

Equivalent edge-sum form:

$$
dist(s,v) = min_{paths s -> v} sum_{e in P} 1
$$

This is the exact quantity represented by `Depth[v]` when `v` is reachable from `s`.

### 3.2.2. BFS Layer Partition
BFS partitions the discovered region into discrete layers:

$$
L_d = { v in V | dist(s,v) = d }
$$

Traversal proceeds in non-decreasing `d`, which is why the first time a vertex is discovered, its recorded depth is already shortest.

### 3.2.3. Parent Certificate
For every non-root discovered vertex `v` with parent `p = Parent[v]`, BFS maintains:

$$
Depth[v] = Depth[p] + 1
$$

This recurrence is the correctness certificate behind `PathTo(dst)`: following parent links reconstructs one shortest hop path.

### 3.2.4. Discovery Invariant
A vertex is marked visited at enqueue time, not at dequeue time:

$$
Visited[v] = true  =>  v is enqueued at most once
$$

This is the key invariant that prevents duplicate frontier entries and preserves linear-time traversal.

### 3.2.5. Inclusive Radius Bound
With an inclusive maximum depth `D`, the reported reachable set is:

$$
V_{<=D} = { v in V | Depth[v] <= D }
$$

Vertices at depth `D` are still visited and appear in `Order`, but their outgoing neighbor relations are not expanded.

### 3.2.6. Weak Connectivity
For `Components(ctx, g)`, direction is ignored for membership.
Define a symmetric relation `~w` by:

$$
u ~w v  iff  there exists an undirected path between u and v
$$

The weak components are exactly the equivalence classes of `~w`.

### 3.2.7. Forest Semantics Under Full Traversal
With `WithFullTraversal()`, BFS becomes a deterministic forest traversal.
Each secondary root `r` starts a new tree with:

$$
Depth[r] = 0
$$

and no parent entry.

This preserves local shortest-hop depth inside each connected region while intentionally keeping `PathTo` anchored to the original `StartID`.

### 3.2.8. Complexity
For a single-source BFS traversal:

$$
T_BFS = O(|V| + |E|)
$$

$$
S_BFS = O(|V|)
$$

For `WithFullTraversal()`, BFS additionally scans all vertices to select deterministic secondary roots:

$$
T_Forest = O(|V| + |E|) + O(|V|_scan)
$$

For `Components(ctx, g)`, total cost includes deterministic adjacency construction and deterministic output ordering:

$$
T_Components = O(|V| + |E|) + sorting overhead required by stable output
$$

> [!NOTE]
> Exact constants depend on the deterministic snapshot behavior of the underlying `core` APIs, but the algorithmic backbone remains linear in graph size.

---

## 3.3. Public API & Result Contract

In lvlath, the returned result is not a secondary implementation detail. It is part of the public protocol.
This section intentionally combines package entry points, error priority, result meaning, and interpretation rules.

### 3.3.1. Public Entry Points

```go
func BFS(g *core.Graph, startID string, opts ...Option) (*Result, error)
func Components(ctx context.Context, g *core.Graph) (*ComponentsResult, error)
```

### 3.3.2. Validation Order and Error Priority
The public `BFS` facade validates in the following order:

1. `g == nil` -> `ErrGraphNil`
2. option application failure -> `ErrOptionViolation`
3. `g.Weighted() == true` -> `ErrWeightedGraph`
4. `!g.HasVertex(startID)` -> `ErrStartVertexNotFound`
5. runtime interruption:
    - `ctx.Err()`
    - `ErrNeighbors` on neighbor enumeration failure, double-wrapped with the underlying cause
    - any `OnVisit` error, wrapped with `%w`

This priority is intentional. It keeps diagnosis stable and predictable.

### 3.3.3. Result

```go
type Result struct {
    StartID string
    Order   []string
    Depth   map[string]int
    Parent  map[string]string
    Visited map[string]bool
    Skipped int
}
```

**Field semantics and invariants:**

| Field | Meaning | Safe on partial result? | Key invariant |
|------:|---------|------------------------:|---------------|
| `StartID` | start vertex for this BFS invocation | yes | anchors `PathTo` under forest mode |
| `Order` | dequeue and visit order | yes | visit means dequeue |
| `Depth` | hop distance in edge count | yes | shortest on first discovery |
| `Parent` | BFS tree or forest links | yes | roots have no parent entry |
| `Visited` | discovery set, marked at enqueue | yes | may be a superset of `Order` on early exit |
| `Skipped` | rejected neighbor relations | yes | counts `(currID, nbrID)`, not edge IDs |

### 3.3.4. PathTo

```go
func (r *Result) PathTo(dst string) ([]string, error)
```

`PathTo(dst)` reconstructs one shortest path from `StartID` to `dst`.

- success: returns `[StartID ... dst]`
- failure: returns `(nil, ErrNoPath)`

**Forest safety:**
With `WithFullTraversal()`, `Visited` may contain vertices from other components.
`PathTo` therefore enforces `StartID` anchoring to prevent false paths.

### 3.3.5. ComponentsResult

```go
type ComponentsResult struct {
    Components     [][]string
    Count          int
    UndirectedView bool
}
```

`ComponentsResult` describes weakly-connected components under an undirected relation.
Its determinism contract is:

- each component is lex-sorted,
- the list of components is sorted by a stable key.

### 3.3.6. Partial-Result Contract
On any non-nil error returned after traversal begins, BFS returns a non-nil partial `*Result`.
This is not an implementation accident. It is a documented package guarantee.

> [!IMPORTANT]
> `Visited` is a discovery set, while `Order` is a processing trace. On early exit, they are intentionally not the same thing.

---

## 3.4. Options & Policy Surface

Options are applied sequentially and are last-writer-wins.
Invalid options are rejected before traversal allocates its working sets.

### 3.4.1. MaxDepth

| Value | Meaning | Behavioral effect |
|------:|---------|-------------------|
| `MaxDepthUnlimited` which is `-1` | unlimited traversal | expand all reachable layers |
| `0` | root-only | visit `StartID`, do not expand neighbors |
| `d > 0` | inclusive limit | visit depth `d`, do not expand from it |

### 3.4.2. FilterNeighbor
`FilterNeighbor` is a relation-level policy:

```go
func(currID, nbrID string) bool
```

It is evaluated on the neighbor relations surfaced by `NeighborIDs(currID)`.
If it returns `false`, BFS does not enqueue `nbrID` from `currID` and increments `Skipped`.

### 3.4.3. Hooks
Hooks are deterministic observers:

- `OnEnqueue` runs when a vertex is discovered,
- `OnDequeue` runs immediately before visit,
- `OnVisit` runs at visit time and can abort traversal.

### 3.4.4. FullTraversal
`WithFullTraversal()` turns single-source BFS into a deterministic forest traversal:

1. BFS explores the component of `StartID`.
2. BFS enumerates `g.Vertices()` in deterministic order.
3. Every still-unvisited vertex becomes a new root with depth zero and no parent.

> [!NOTE]
> Full traversal is for deterministic coverage and indexing, not for multi-source shortest paths.

---

## 3.5. Pseudocode

### 3.5.1. BFS Kernel

```text
FUNCTION BFS(g, startID, opts...):

  Stage 1: Validate inputs
    - graph is non-nil
    - options are valid
    - graph is unweighted
    - start exists

  Stage 2: Allocate once
    - Order, Depth, Parent, Visited sized from VertexCount
    - queue allocated once
    - head index starts at 0

  Stage 3: Seed root
    - mark StartID visited
    - set Depth[StartID] = 0
    - enqueue StartID

  Stage 4: Core loop
    while head < len(queue):
      if ctx.Done():
        return partial result + ctx.Err()

      u = queue[head]
      clear queue[head]
      head++

      OnDequeue(u)
      append u to Order

      if OnVisit(u) returns error:
        return partial result + error

      if maxDepth is set and Depth[u] >= maxDepth:
        continue

      nbrs = NeighborIDs(u)
      if NeighborIDs fails:
        return partial result + ErrNeighbors + underlying cause

      for each v in nbrs:
        if ctx.Done():
          return partial result + ctx.Err()

        if FilterNeighbor(u, v) rejects:
          Skipped++
          continue

        if v already visited:
          continue

        mark v visited
        Depth[v] = Depth[u] + 1
        Parent[v] = u
        OnEnqueue(v)
        enqueue v

  Stage 5: Optional forest continuation
    if fullTraversal:
      iterate Vertices() in deterministic order
      start a new root for each unvisited vertex

  return result
```

### 3.5.2. Components Kernel

```text
FUNCTION Components(ctx, g):

  Stage 1: Snapshot deterministic vertex order
  Stage 2: Build an undirected adjacency view
  Stage 3: Traverse each unvisited root in deterministic order
  Stage 4: Sort each component and sort the component list

  On ctx cancellation:
    return partial ComponentsResult + ctx.Err()
```

---

## 3.6. ASCII Diagrams

### 3.6.1. Layered Frontier Expansion

```text
Depth 0:            [S]
                  /  |  \
Depth 1:        [A] [B] [C]
                /         \
Depth 2:      [D]         [E]

Order is dequeue order:
[S, A, B, C, D, E]
```

### 3.6.2. Deterministic Tie-Break and Parent Selection

```text
      S
     / \
    A   B
     \ /
      C

NeighborIDs(S) = [A, B]
NeighborIDs(A) = [C]
NeighborIDs(B) = [C]

C is discovered from A first, so:
  Parent[C] = A
  PathTo(C) = [S, A, C]
```

### 3.6.3. MaxDepth Inclusive Cut

```text
S -- A -- D
 \  |
  \ B -- E
   \
    C

MaxDepth = 1

Visited and ordered:
  S, A, B, C

Not expanded:
  A, B, C

Not reached:
  D, E
```

### 3.6.4. Discovery Versus Visit on Early Interruption

```text
Queue state before interruption:
  queue = [S, A, B, C]
  head  = 2

Meaning:
  Order   = [S, A]
  Visited = {S, A, B, C}

Interpretation:
  B and C were discovered and queued,
  but they were not yet visited.
```

### 3.6.5. FullTraversal Forest

```text
Component #1:  A -- B -- C
Component #2:  M -- N
Component #3:  X -- Y -- Z

StartID = A

Forest order:
  [A, B, C, M, N, X, Y, Z]

Depths by root:
  A=0  B=1  C=2
  M=0  N=1
  X=0  Y=1  Z=2
```

### 3.6.6. Weak Components

```text
Directed edges:
  A -> B -> C
  X -> Y

Weak components ignore direction for membership:
  {A, B, C}
  {X, Y}

This is weak connectivity,
not strong connectivity.
```

---

## 3.7. Go Example

A production-style example: service dependency exploration with a deterministic early-stop policy.

```go
package main

import (
    "context"
    "errors"
    "fmt"

    "github.com/lvlath/go/bfs"
    "github.com/lvlath/go/core"
)

func main() {
    g, _ := core.NewGraph(core.WithDirected(true))

    edges := [][2]string{
        {"gateway", "auth"},
        {"gateway", "billing"},
        {"gateway", "search"},
        {"auth", "users"},
        {"auth", "sessions"},
        {"billing", "ledger"},
        {"billing", "db"},
        {"search", "index"},
        {"users", "db"},
        {"sessions", "cache"},
        {"ledger", "archive"},
        {"index", "cache"},
    }

    for _, e := range edges {
        _, _ = g.AddEdge(e[0], e[1], 0)
    }

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

    fmt.Println("canceled:", errors.Is(err, context.Canceled))
    fmt.Println("order:", res.Order)
    fmt.Println("depth to db:", res.Depth["db"])

    if path, pathErr := res.PathTo("db"); pathErr == nil {
        fmt.Println("path to db:", path)
    } else {
        fmt.Println("path to db unavailable:", pathErr)
    }
}
```
[![Go Playground](https://img.shields.io/badge/Go_Playground-BFS-blue?logo=go)](https://go.dev/play/p/C8qwk3-CVHW)

**What this example teaches:**
- deterministic traversal from `gateway`,
- cancellation via deterministic hook trigger,
- partial result consumption after cancellation,
- shortest-hop path reconstruction when the target has already been discovered.

---

## 3.8. Pitfalls & Best Practices (Architectural Mastery)

### 1. Do not parse error strings
Use `errors.Is` with sentinels such as `bfs.ErrNoPath`, `bfs.ErrNeighbors`, and `bfs.ErrOptionViolation`.
String parsing is brittle and is not part of the contract.

### 2. Do not use timeouts for deterministic examples or tests
Use `context.WithCancel` and cancel from a deterministic hook condition.
Timeout-based examples are flaky under CI load and distort traversal semantics.

### 3. Understand the difference between `Visited` and `Order`
- `Visited` means discovered and enqueued.
- `Order` means dequeued and processed.

On partial results, `Visited` may strictly contain more vertices than `Order`.

### 4. Do not confuse FullTraversal with multi-source shortest paths
`WithFullTraversal()` builds a deterministic forest for coverage and indexing.
It does not redefine `PathTo` into a global forest-path operator.

### 5. Keep hooks deterministic and light
Hooks run in hot paths.
They should not allocate heavily, block unpredictably, or mutate graph topology.

### 6. Use FilterNeighbor for policy, not topology mutation
If the traversal policy says a relation is forbidden, reject it in `FilterNeighbor` and use `Skipped` as an audit counter.
Do not mutate the underlying graph just to express traversal policy.

### 7. Weighted graphs are intentionally rejected
BFS answers a hop-distance question.
If the graph is weighted and the business question depends on weights, switch algorithms rather than forcing BFS to answer the wrong question.

### 8. Treat determinism as a feature, not as an accident
If traversal order changes, that is usually either:
- a change in graph neighbor ordering,
- a change in vertex ordering for full traversal,
- or a real regression.

Downstream algorithms should rely on this reproducibility.

---

## 3.9. Practical Recipes

### Recipe A. Blast Radius
Run `BFS(start, WithMaxDepth(k))` and group vertices by `Depth`.
This yields deterministic k-hop waves around the source.

### Recipe B. Policy Firewall
Use `WithFilterNeighbor` to block forbidden relations while leaving topology intact.
Use `Skipped` as a reproducible audit signal.

### Recipe C. Deterministic Crawler
Use `WithContext(ctx)` and cancel from `OnVisit` when a domain rule triggers.
Consume the partial `Order`, `Visited`, and `Depth` maps for diagnostics.

### Recipe D. Weak Island Discovery
Use `Components(ctx, g)` to enumerate weakly-connected islands.
This is the correct primitive when direction should not split membership.

### Recipe E. Forest Indexing
Use `WithFullTraversal()` when you need deterministic coverage of every connected region from a known primary start vertex.

---

**lvlath/bfs**: deterministic by contract, strict in semantics, and safe to compose.

> Next: [4. DFS (Depth-First Search) ->](DFS.md)

