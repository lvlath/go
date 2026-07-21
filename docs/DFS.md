<!--
  lvlath - Repository Documentation

  Purpose:
    This document is the repository-level specification, tutorial, and contract map
    for lvlath/dfs. It explains what the package guarantees, how the mathematics maps
    to the implementation, where determinism comes from, what partial results mean,
    and how to use DFS, Cycle Detection, and Topological Sorting correctly in
    production pipelines.

  Contract status:
    - Behaviors described here are part of the public contract.
    - Determinism rules described here are part of the public contract.
    - Error-classification rules described here are part of the public contract.
    - Result semantics for Result and CycleDetectionResult are part of the public contract.
    - Any incompatible change must be explicit, documented, and versioned.

  Scope:
    - Deterministic Depth-First Search traversal and DFS forests.
    - Cycle witness detection and minimal-rotation canonicalization.
    - Directed acyclic graph (DAG) topological sorting.
    - Partial-result semantics and context cancellation.

  License:
    The lvlath repository is licensed under AGPL-3.0-only. See LICENSE.
-->

# 4. DFS (Depth-First Search)

> **Package:** `lvlath/dfs` | **Focus:** Determinism, Post-Order Semantics, Witness Cycles, Directed Topological Ordering

The `lvlath/dfs` package provides a deterministic, contract-driven implementation of Depth-First Search over `core.Graph`. It acts as the algorithmic backbone for workflows requiring deep path exploration, cycle detection, and dependency ordering.

Unlike casual textbook recursion that silently breaks under edge cases or relies on unstable map iteration, `lvlath/dfs` is engineered for high-load, mathematically exact, and debuggable graph analytics.

---

## 4.1. What & Why

Why use `lvlath/dfs` instead of a handwritten recursive loop?

### The Four Laws of DFS

1. **Absolute Determinism**
    *   *Others:* Yield different traversal paths on identical graphs depending on map hashing, unstable adjacency containers, or accidental traversal tie-breaks.
    *   *Lvlath:* Expands candidate relations in the exact order returned by `core.Graph.Neighbors(u)`. The package does not inject a hidden fallback sort during traversal. Cycle canonicalization yields byte-for-byte stable witness signatures, so the same graph produces the same traversal and witness output on every run.
2. **Post-Order Semantics by Contract**
    *   *Others:* Blur the distinction between discovery, entry, and completion.
    *   *Lvlath:* DFS `Result.Order` is strictly the *finish order* (post-order). Pre-order visibility is securely delegated to explicit `OnVisit` hooks.
3. **Witness Sets over Theorem Provers**
    *   *Others:* Attempt NP-hard exhaustive simple-cycle enumeration, leading to catastrophic exponential memory/time blowups in dense graphs.
    *   *Lvlath:* Cycle detection returns a deterministic *witness set* of canonical cycles. It proves cyclicity and locates logical loops without halting your production system.
4. **Strict Safety & Fail-Fast Policies**
    *   *Others:* Fail mid-traversal or return corrupted arrays on context timeout.
    *   *Lvlath:* Options are validated *before* state is allocated. If execution aborts, partial metadata (`Depth`, `Visited`, `Parent`) is safely retained, but the incomplete `Order` is explicitly cleared to prevent mathematical falsehoods downstream.

---

## 4.2. Mathematical Formulation

The `lvlath/dfs` package models strict graph-theoretic properties. Distances are hop-counts within the constructed DFS tree, and directed paths enforce strict edge orientation.

### 4.2.1. DFS-Tree Depth
For every visited vertex $v$ that is not a root of a DFS tree, its depth is exactly one greater than the vertex from which it was first entered:
$$ \forall v \in V \setminus \{Roots\}, \quad Depth[v] = Depth[Parent[v]] + 1 $$

### 4.2.2. Finish-Order Semantics
Let $entry(v)$ and $finish(v)$ be the timestamps of entering and leaving $v$. A vertex $v$ is appended to `Order` exactly at $finish(v)$. Thus, if $u$ is an ancestor of $v$ in the DFS tree, the invariant holds:
$$ finish(v) < finish(u) $$
This means descendants always appear *before* their ancestors in the `Order` slice.

### 4.2.3. Forest Semantics under FullTraversal
With `WithFullTraversal()`, the graph is partitioned into a DFS forest. For each new tree root $r$ selected by the deterministic graph iteration:
$$ \forall r \in Roots, \quad Depth[r] = 0 \quad \text{and} \quad r \notin keys(Parent) $$

### 4.2.4. Cycle Witness Formulation
A cycle is extracted from the active path stack when a `Gray` $\to$ `Gray` back-edge is encountered. The witness is returned as a closed sequence:
$$ W =[v_0, v_1, ..., v_k, v_0] $$

### 4.2.5. Canonicalization: Directed vs Undirected
To deduplicate cycles starting at different vertices but representing the same loop, `lvlath` uses Booth's minimal rotation algorithm.
$$ C_{canon} = \begin{cases} \min_{lex}(Rotations(C)) & \text{if } G \text{ is directed} \\ \min_{lex}\Big(\min_{lex}(Rotations(C)), \min_{lex}(Rotations(Reverse(C)))\Big) & \text{if } G \text{ is undirected} \end{cases} $$
*Directed cycles strictly preserve their orientation. Undirected cycles consider reverse flow to be equivalent.*

### 4.2.6. Topological Order Invariant
For a directed acyclic graph (DAG), the final topological order guarantees that every directed edge points forward:
$$ \forall (u \to v) \in E_{directed}, \quad \text{Index}(u) < \text{Index}(v) \text{ in Output} $$

### 4.2.7. Complexity Summary
*   **DFS Traversal / Forest:** $T = O(V + E)$, $S = O(V)$, excluding hook and filter cost.
*   **Topological Sort:** $T = O(V + E)$, $S = O(V)$.
*   **Cycle Detection:** $T = O(V + E + \sum |witness_i|)$, $S = O(V + L_{max} + \sum |witness_i|)$.
---

## 4.3. Public API & Result Contract

### 4.3.1. Public Entry Points

```go
func DFS(g *core.Graph, startID string, opts ...Option) (*Result, error)
func Forest(g *core.Graph, opts ...Option) (*Result, error)

func DetectCycles(g *core.Graph) (*CycleDetectionResult, error)
func HasCycle(g *core.Graph) (bool, error)

func TopologicalSort(g *core.Graph, options ...TopoOption) ([]string, error)
func TopologicalSortContext(ctx context.Context, g *core.Graph) ([]string, error)
```

### 4.3.2. Result Semantics

```go
type Result struct {
	Order            []string
	Depth            map[string]int
	Parent           map[string]string
	Visited          map[string]bool
	SkippedNeighbors int
}
```

| Field              | Meaning                                     | Safe on Partial Result? | Key Invariant                                  |
|:-------------------|:--------------------------------------------|:-----------------------:|:-----------------------------------------------|
| `Order`            | Finish order (post-order).                  |   **NO (Set to nil)**   | A partial post-order is mathematically unsafe. |
| `Depth`            | DFS-tree depth from the root.               |           YES           | Records depth upon entry.                      |
| `Parent`           | The predecessor that discovered the vertex. |           YES           | DFS-tree roots do not appear in this map.      |
| `Visited`          | Vertices actually entered.                  |           YES           | Used to track coverage.                        |
| `SkippedNeighbors` | Count of explicitly filtered neighbors.     |           YES           | Diagnostic for policy boundaries.              |

> [!IMPORTANT]
> **Partial-Result Contract:** If traversal aborts due to `context.Canceled` or a hook error, DFS `Result` is returned alongside the error. The `Order` field is cleared (`nil`), but `Visited`, `Depth`, and `Parent` retain the structural progress up to the exact point of failure.

### 4.3.3. CycleDetectionResult

```go
type CycleDetectionResult struct {
	HasCycle bool
	Cycles   [][]string
}
```
*   `HasCycle`: the summary boolean exposed directly in the result object.
*   `HasCycle(g)`: a convenience facade for callers that need only the boolean classification and not the witness payload.
*   `Cycles`: The deterministic, canonicalized witness list. Note that `len(Cycles) == 0` when `HasCycle == false`.

### 4.3.4. Error Protocol and Validation Priority
The `DFS` facade evaluates preconditions in a strict, predictable order:
1. `g == nil` $\to$ `ErrGraphNil`
2. Explicit invalid options $\to$ `ErrOptionViolation`
3. Single-source start vertex absent $\to$ `ErrStartVertexNotFound`
4. Runtime failures (Context, Hooks, Neighbor fetch).

### 4.3.5. Result Ownership

Returned result slices and maps belong to the caller after the function returns.

This has three practical consequences:
*   the package does not retain them for later mutation,
*   callers may safely copy, transform, or serialize them after return,
*   the package remains an algorithm layer rather than a topology-storage or builder layer.
---

## 4.4. Options & Policy Surface

Options implement a last-writer-wins functional assembly. Invalid explicit options (e.g., `WithContext(nil)`) fail-fast before allocating graph traversal state.

### 4.4.1. DFS Options
*   `WithContext(ctx)`: Ties the DFS recursion to standard cancellation rules.
*   `WithOnVisit(fn)`: Executes *immediately* when a vertex is entered (pre-order). Returning an error halts traversal.
*   `WithOnExit(fn)`: Executes right before a vertex is marked Black and pushed to `Order`.
*   `WithMaxDepth(limit)`: The horizon cut. Enforced *before* entering deeper vertices. Use `NoDepthLimit (-1)` for unrestricted.
*   `WithFilterNeighbor(fn)`: A policy firewall. Returning `false` blocks traversal into the candidate, treating the edge as non-existent for the current traversal.
*   `WithFullTraversal()`: Transforms single-source search into a graph-wide DFS forest. Used implicitly by `Forest()`.

### 4.4.2. Topological-Sort Options
*   `WithCancelContext(ctx)`: Topo-sort is specialized and uses a narrower option surface `TopoOption`.

---

## 4.5. Algorithmic Architecture

### 4.5.1. DFS Runtime Model (`dfsWalker`)
The `dfsWalker` maintains the traversal state strictly isolated from the option structs. It parses per-edge semantics using `neighborFromEdge`-this means in mixed-edge graphs, directed edges flow one way, while undirected edges behave bidirectionally, interpreted locally per-edge.

### 4.5.2. Cycle Detector Runtime Model (`cycleDetector`)
It employs the classic 3-color vertex state:
*   `White (0)`: Unvisited.
*   `Gray (1)`: Discovered, currently in the active DFS recursion path.
*   `Black (2)`: Fully explored and exited.
    A `Gray` $\to$ `Gray` edge triggers cycle reconstruction by scanning the active path stack, followed by $O(L)$ Booth's canonicalization.

### 4.5.3. Topological Sorter Runtime Model (`topoSorter`)
Valid only for directed graphs (`g.Directed() == true`). It leverages the 3-color model to simultaneously detect directed cycles (`ErrCycleDetected`) and record post-order. Once all vertices are `Black`, it performs an $O(V)$ in-place array reversal to yield the exact topological execution pipeline. Mixed-edge environments are handled strictly: undirected edges are bypassed entirely.

---

## 4.6. Pseudocode

### 4.6.1. DFS Traversal Kernel
```text
FUNCTION Traverse(u, depth):
  if ctx.Done():
    clear authoritative finish order
    return ctx.Err()

  if MaxDepth is bounded and depth > MaxDepth:
    return OK

  Visited[u] = true
  Depth[u] = depth

  if OnVisit is defined:
    OnVisit(u)

  neighbors = g.Neighbors(u)

  for each edge in neighbors:
    v, ok = ResolveNeighbor(edge, u)
    if not ok:
      continue

    if v == u and loops are disallowed by graph policy:
      continue

    if FilterNeighbor is defined and FilterNeighbor(v) == false:
      SkippedNeighbors++
      continue

    if Visited[v]:
      continue

    Parent[v] = u
    Traverse(v, depth + 1)

  if OnExit is defined:
    OnExit(u)

  append u to Order
```

### 4.6.2. DetectCycles Witness Extraction
```text
FUNCTION VisitCycle(u):
  State[u] = Gray
  push u to ActivePath

  for each edge in Neighbors(u):
    v = neighborFromEdge(edge, u)
    if Undirected Parent Backtrack: continue

    if State[v] == White:
       VisitCycle(v)
    else if State[v] == Gray:
       // Witness Detected!
       cycleBase = ActivePath[indexOf(v) ... end]
       canon = Canonicalize(cycleBase)
       SaveWitness(canon)

  pop u from ActivePath
  State[u] = Black
```

---

## 4.7. ASCII Diagrams

### 4.7.1. Recursion and Finish Order (Post-Order)
The most common mistake is expecting `Order` to track discovery. It strictly tracks *completion*.
```text
       [A]
      /   \
    [B]   [C]
   /   \
 [D]   [E]

Discovery (Pre-Order via OnVisit): A -> B -> D -> E -> C
Finish Order (Returned DFS Order): D -> E -> B -> C -> A
```

### 4.7.2. Forest Traversal (WithFullTraversal)
A full traversal iterates through the graph's deterministic vertex catalog, spinning up new trees for unvisited disconnected components.
```text
Component 1:   [A] -> [B]       Depth: A=0, B=1
Component 2:   [X]              Depth: X=0
Component 3:   [M] -> [N]       Depth: M=0, N=1

Roots: A, X, M (Absent from Parent map)
```

### 4.7.3. MaxDepth Horizon Limit
A hard firewall enforcing bounding radius without mutating the graph.
```text
MaxDepth = 1

    [S] (depth 0)
   /   \
 [A]   [B] (depth 1 - Visited, appended to Order)
  |     |
 [X]   [Y] (depth 2 - Ignored completely, White)
```

### 4.7.4. Cycle Extraction (Gray Back-Edge)
Extracting the loop from the active stack.
```text
Active Stack: [ S -> A -> B -> C ]
Current Node: C
Neighbors(C) sees 'A', which is GRAY (already in stack).

Extraction: Suffix from A to C:[A, B, C]
Closed Output: [A, B, C, A]
```

### 4.7.5. Topological Reversal
Topological sorting reverses the finish order exactly once at the very end.
```text
DAG:  [Build] -> [Test] -> [Deploy]

Finish Order (Black): [Deploy] -> [Test] -> [Build]
Reversal (O(V)):      [Build]  -> [Test] -> [Deploy]
```

---

## 4.8. Go Example Scenarios

The examples below demonstrate production-grade usage of `lvlath/dfs`.
They are intentionally scenario-driven: each one models a realistic graph, uses the package as an actual pipeline, and explains how to consume the returned result safely and meaningfully.

> [!NOTE]
> The snippets below are repository-level teaching bridges: they stay faithful to the real public API, avoid flaky output, and show how a caller should actually consume DFS results in operational code.

### 4.8.1. Infrastructure Inspection

```go
package main

import (
	"fmt"
	"strings"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dfs"
)

// Scenario:
//   A platform team is auditing a service dependency tree rooted at the ingress gateway.
//   One branch leads into a quarantine zone that must remain present in topology,
//   but must not be traversed during this inspection run.
//
// What this example demonstrates:
//   - deterministic single-source DFS,
//   - policy filtering via WithFilterNeighbor,
//   - entry tracing via WithOnVisit,
//   - finish tracing via WithOnExit,
//   - stable consumption of Result.Order as post-order,
//   - reproducible skipped-neighbor diagnostics.
func main() {

	// Stage 1: Build a directed infrastructure dependency graph.
	// The graph intentionally includes a quarantine branch that stays visible in topology.
	graph, _ := core.NewGraph(core.WithDirected(true))

	_, _ = graph.AddEdge("gateway", "auth", 0)
	_, _ = graph.AddEdge("gateway", "billing", 0)
	_, _ = graph.AddEdge("gateway", "cache", 0)
	_, _ = graph.AddEdge("gateway", "edge", 0)

	_, _ = graph.AddEdge("auth", "profile", 0)
	_, _ = graph.AddEdge("auth", "secrets", 0)
	_, _ = graph.AddEdge("profile", "db", 0)

	_, _ = graph.AddEdge("billing", "ledger", 0)
	_, _ = graph.AddEdge("billing", "reports", 0)
	_, _ = graph.AddEdge("ledger", "db", 0)
	_, _ = graph.AddEdge("reports", "archive", 0)

	_, _ = graph.AddEdge("cache", "replicas", 0)
	_, _ = graph.AddEdge("cache", "warmer", 0)
	_, _ = graph.AddEdge("warmer", "metrics", 0)

	_, _ = graph.AddEdge("edge", "quarantine-lab", 0)
	_, _ = graph.AddEdge("edge", "traffic", 0)
	_, _ = graph.AddEdge("traffic", "metrics", 0)

	// Stage 2: Prepare deterministic audit collectors.
	// entered records pre-order entry events; finished records post-order exit events.
	entered := make([]string, 0, graph.VertexCount())
	finished := make([]string, 0, graph.VertexCount())

	// Stage 3: Define traversal policy.
	// The quarantine node remains in the graph, but the traversal policy blocks entry into it.
	filter := func(vertexID string) bool {
		return !strings.HasPrefix(vertexID, "quarantine-")
	}

	// Record deterministic entry order.
	onVisit := func(vertexID string) error {
		entered = append(entered, vertexID)
		return nil
	}

	// Record deterministic finish order.
	onExit := func(vertexID string) error {
		finished = append(finished, vertexID)
		return nil
	}

	// Stage 4: Run DFS from the infrastructure ingress.
	result, err := dfs.DFS(
		graph,
		"gateway",
		dfs.WithFilterNeighbor(filter),
		dfs.WithOnVisit(onVisit),
		dfs.WithOnExit(onExit),
	)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	// Stage 5: Consume the result as an operational audit report.
	// entered shows pre-order discovery, finished shows completion order,
	// and result.Order is the authoritative DFS post-order returned by the package.
	fmt.Println("entered:", strings.Join(entered, " -> "))
	fmt.Println("finished:", strings.Join(finished, " -> "))
	fmt.Println("skipped:", result.SkippedNeighbors)
	fmt.Println("postorder:", strings.Join(result.Order, " -> "))

	// Output:
	// entered: gateway -> auth -> profile -> db -> secrets -> billing -> ledger -> reports -> archive -> cache -> replicas -> warmer -> metrics -> edge -> traffic
	// finished: db -> profile -> secrets -> auth -> ledger -> archive -> reports -> billing -> replicas -> metrics -> warmer -> cache -> traffic -> edge -> gateway
	// skipped: 1
	// postorder: db -> profile -> secrets -> auth -> ledger -> archive -> reports -> billing -> replicas -> metrics -> warmer -> cache -> traffic -> edge -> gateway
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-DFS_Infrastructure_Inspection-blue?logo=go)](https://go.dev/play/p/svbZa_HnnRK)

**Why the result looks like this:**
- `entered` is a deterministic pre-order trace produced by `WithOnVisit`.
- `finished` is a deterministic completion trace produced by `WithOnExit`.
- `postorder` matches `finished` exactly because `Result.Order` is a finish-order artifact, not a discovery-order artifact.
- `quarantine-lab` remains part of the topology, but the filter blocks traversal into it; that rejected relation is reported through `SkippedNeighbors`.
- This pattern is useful when policy must affect traversal behavior without mutating the underlying graph.

---

### 4.8.2. Full Traversal Inventory Sweep

```go
package main

import (
	"fmt"
	"sort"
	"strings"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dfs"
)

// Scenario:
//   An inventory job must sweep all disconnected service islands in one run.
//   The platform team wants deterministic DFS-forest coverage, explicit roots,
//   and proof that depth resets independently for each disconnected region.
//
// What this example demonstrates:
//   - Forest for disconnected topology,
//   - deterministic forest traversal,
//   - root detection via absence from Parent,
//   - depth reset per forest root,
//   - caller-side stabilization of map-derived reporting.
func main() {

	// Stage 1: Build three disconnected operational islands.
	graph, _ := core.NewGraph(core.WithDirected(true))

	_, _ = graph.AddEdge("zone-a:0-gw", "zone-a:1-api", 0)
	_, _ = graph.AddEdge("zone-a:0-gw", "zone-a:1-auth", 0)
	_, _ = graph.AddEdge("zone-a:1-api", "zone-a:2-db", 0)
	_, _ = graph.AddEdge("zone-a:1-auth", "zone-a:2-cache", 0)
	_, _ = graph.AddEdge("zone-a:1-auth", "zone-a:2-queue", 0)

	_, _ = graph.AddEdge("zone-m:0-gw", "zone-m:1-batch", 0)
	_, _ = graph.AddEdge("zone-m:0-gw", "zone-m:1-etl", 0)
	_, _ = graph.AddEdge("zone-m:1-batch", "zone-m:2-warehouse", 0)
	_, _ = graph.AddEdge("zone-m:1-etl", "zone-m:2-lake", 0)
	_, _ = graph.AddEdge("zone-m:1-etl", "zone-m:2-report", 0)

	_, _ = graph.AddEdge("zone-z:0-gw", "zone-z:1-web", 0)
	_, _ = graph.AddEdge("zone-z:0-gw", "zone-z:1-worker", 0)
	_, _ = graph.AddEdge("zone-z:1-web", "zone-z:2-cdn", 0)
	_, _ = graph.AddEdge("zone-z:1-worker", "zone-z:2-jobs", 0)

	// Stage 2: Run full-coverage DFS across all disconnected components.
	result, err := dfs.Forest(graph)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	// Stage 3: Derive forest roots from the public contract:
	// visited vertices without parent entries are forest roots.
	roots := make([]string, 0, 3)
	for vertexID := range result.Visited {
		if _, hasParent := result.Parent[vertexID]; !hasParent {
			roots = append(roots, vertexID)
		}
	}

	// Stage 4: Stabilize caller-owned reporting.
	// The traversal itself is deterministic, but Go map iteration is not a printable contract.
	sort.Strings(roots)

	// Stage 5: Print a compact inventory summary.
	fmt.Println("roots:", strings.Join(roots, ", "))
	fmt.Printf(
		"root-depths: %s=%d, %s=%d, %s=%d\n",
		roots[0], result.Depth[roots[0]],
		roots[1], result.Depth[roots[1]],
		roots[2], result.Depth[roots[2]],
	)
	fmt.Println(
		"parent-samples:",
		"zone-a:2-db<-"+result.Parent["zone-a:2-db"]+",",
		"zone-m:2-lake<-"+result.Parent["zone-m:2-lake"]+",",
		"zone-z:2-jobs<-"+result.Parent["zone-z:2-jobs"],
	)
	fmt.Println("visited:", len(result.Visited))

	// Output:
	// roots: zone-a:0-gw, zone-m:0-gw, zone-z:0-gw
	// root-depths: zone-a:0-gw=0, zone-m:0-gw=0, zone-z:0-gw=0
	// parent-samples: zone-a:2-db<-zone-a:1-api, zone-m:2-lake<-zone-m:1-etl, zone-z:2-jobs<-zone-z:1-worker
	// visited: 17
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-DFS_Full_Traversal_Inventory_Sweep-blue?logo=go)](https://go.dev/play/p/eXn8Nbnn8av)

**Why the result looks like this:**
- `Forest` covers every disconnected island, not just one component.
- Each new forest root starts with depth `0`.
- Root vertices do not appear in `Parent`; that is the correct public way to identify forest roots from the result.
- Sorting the derived `roots` slice is the caller’s responsibility because the roots are extracted from a map, while the package itself remains deterministic at traversal level.
- This pattern is useful for indexers, inventory crawlers, and full-environment audits.

---

### 4.8.3. Depth-Limited Blast Radius

```go
package main

import (
	"fmt"
	"strings"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dfs"
)

// Scenario:
//   An incident commander wants a bounded blast-radius analysis around a compromised edge.
//   The objective is to inspect the near field of the dependency tree while proving that
//   deeper systems remain untouched by this particular run.
//
// What this example demonstrates:
//   - DFS with WithMaxDepth,
//   - inclusive depth-limit semantics,
//   - deterministic bounded post-order,
//   - operational use of Visited and Depth to audit the traversal frontier.
func main() {

	// Stage 1: Build a directed dependency graph around the incident entry point.
	graph, _ := core.NewGraph(core.WithDirected(true))

	_, _ = graph.AddEdge("incident-gateway", "auth", 0)
	_, _ = graph.AddEdge("incident-gateway", "billing", 0)
	_, _ = graph.AddEdge("incident-gateway", "edge", 0)

	_, _ = graph.AddEdge("auth", "profile", 0)
	_, _ = graph.AddEdge("auth", "secrets", 0)
	_, _ = graph.AddEdge("profile", "db", 0)

	_, _ = graph.AddEdge("billing", "ledger", 0)
	_, _ = graph.AddEdge("billing", "reports", 0)
	_, _ = graph.AddEdge("ledger", "archive", 0)

	_, _ = graph.AddEdge("edge", "queue", 0)
	_, _ = graph.AddEdge("edge", "traffic", 0)
	_, _ = graph.AddEdge("queue", "workers", 0)
	_, _ = graph.AddEdge("traffic", "metrics", 0)

	// Stage 2: Run a bounded DFS.
	// MaxDepth(2) means depth-2 vertices are still entered and finished,
	// but traversal does not expand beyond them.
	result, err := dfs.DFS(graph, "incident-gateway", dfs.WithMaxDepth(2))
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	// Stage 3: Consume the result as a blast-radius report.
	fmt.Println("visited:", len(result.Visited))
	fmt.Println("postorder:", strings.Join(result.Order, " -> "))
	fmt.Printf(
		"depths: auth=%d, ledger=%d, traffic=%d\n",
		result.Depth["auth"],
		result.Depth["ledger"],
		result.Depth["traffic"],
	)
	fmt.Printf(
		"outer-ring-entered: archive=%t, db=%t, metrics=%t, workers=%t\n",
		result.Visited["archive"],
		result.Visited["db"],
		result.Visited["metrics"],
		result.Visited["workers"],
	)

	// Output:
	// visited: 10
	// postorder: profile -> secrets -> auth -> ledger -> reports -> billing -> queue -> traffic -> edge -> incident-gateway
	// depths: auth=1, ledger=2, traffic=2
	// outer-ring-entered: archive=false, db=false, metrics=false, workers=false
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-DFS_DepthLimited_Blast_Radius-blue?logo=go)](https://go.dev/play/p/vBQ2NvCWjOr)

**Why the result looks like this:**
- Depth limit `2` is inclusive: depth-2 vertices such as `ledger`, `reports`, `queue`, and `traffic` are still entered and appear in finish order.
- Their children are not entered, which is why `archive`, `db`, `workers`, and `metrics` remain outside the visited set.
- This gives a clean operational radius without mutating the graph or manually pruning topology.
- The example also reinforces that DFS depth is a DFS-tree depth, not a shortest-path distance claim.

---

### 4.8.4. Release Pipeline

```go
package main

import (
	"fmt"
	"strings"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dfs"
)

// Scenario:
//   A release orchestration system models specification, policy, code generation,
//   testing, packaging, and promotion stages as a DAG.
//   The engineering team needs one deterministic execution plan that is stable enough
//   for CI logs, reviews, reproducible dry-runs, and deployment audits.
//
// What this example demonstrates:
//   - TopologicalSort on a realistic DAG,
//   - deterministic ordering under the package traversal law,
//   - dependency convergence across multiple prerequisite branches,
//   - direct consumption of the resulting release plan.
func main() {

	// Stage 1: Build a directed acyclic release graph.
	graph, _ := core.NewGraph(core.WithDirected(true))

	_, _ = graph.AddEdge("01-spec-freeze", "03-schema-lock", 0)
	_, _ = graph.AddEdge("01-spec-freeze", "02-threat-model", 0)

	_, _ = graph.AddEdge("02-threat-model", "04-auth-rules", 0)

	_, _ = graph.AddEdge("03-schema-lock", "05-api-compat", 0)
	_, _ = graph.AddEdge("03-schema-lock", "06-codegen", 0)

	_, _ = graph.AddEdge("04-auth-rules", "07-config-render", 0)
	_, _ = graph.AddEdge("05-api-compat", "06-codegen", 0)

	_, _ = graph.AddEdge("06-codegen", "08-unit-tests", 0)
	_, _ = graph.AddEdge("07-config-render", "08-unit-tests", 0)

	_, _ = graph.AddEdge("08-unit-tests", "09-image-build", 0)
	_, _ = graph.AddEdge("09-image-build", "10-smoke-tests", 0)
	_, _ = graph.AddEdge("10-smoke-tests", "11-deploy-staging", 0)
	_, _ = graph.AddEdge("11-deploy-staging", "12-promote-prod", 0)

	// Stage 2: Compute the release order.
	order, err := dfs.TopologicalSort(graph)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	// Stage 3: Print the execution plan as a deterministic deployment pipeline.
	fmt.Println("release-plan:", strings.Join(order, " -> "))

	// Output:
	// release-plan: 01-spec-freeze -> 02-threat-model -> 04-auth-rules -> 07-config-render -> 03-schema-lock -> 05-api-compat -> 06-codegen -> 08-unit-tests -> 09-image-build -> 10-smoke-tests -> 11-deploy-staging -> 12-promote-prod
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-DFS_Release_Pipeline-blue?logo=go)](https://go.dev/play/p/NUx4m50lmp4)

**Why the result looks like this:**
- The graph is a DAG, so a topological plan exists.
- The output is suitable for operational replay because every edge points forward in the final sequence.
- The order is not presented as “some arbitrary valid order” here; it is consumed as the deterministic result produced by the current package contract.
- This makes the snippet relevant for real release systems, migration planners, and staged deployment frameworks.

---

### 4.8.5. Escalation Loop Witness

```go
package main

import (
	"fmt"
	"strings"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dfs"
)

// Scenario:
//   A governance workflow routes cases through triage, security, approvals,
//   finance, legal, and procurement lanes.
//   The organization needs to know whether escalation logic contains loops,
//   and if so, it needs deterministic witness cycles that operators can inspect immediately.
//
// What this example demonstrates:
//   - DetectCycles on a realistic workflow graph,
//   - deterministic witness-set reporting,
//   - canonical closed-cycle output,
//   - honest consumption of CycleDetectionResult without implying exhaustive enumeration.
func main() {

	// Stage 1: Build a directed workflow graph with two independent loop zones.
	graph, _ := core.NewGraph(core.WithDirected(true))

	_, _ = graph.AddEdge("intake", "triage", 0)
	_, _ = graph.AddEdge("triage", "security", 0)
	_, _ = graph.AddEdge("security", "approvals", 0)
	_, _ = graph.AddEdge("approvals", "triage", 0)

	_, _ = graph.AddEdge("approvals", "audit", 0)
	_, _ = graph.AddEdge("audit", "archive", 0)
	_, _ = graph.AddEdge("audit", "billing", 0)

	_, _ = graph.AddEdge("billing", "finance", 0)
	_, _ = graph.AddEdge("finance", "legal", 0)
	_, _ = graph.AddEdge("legal", "procurement", 0)
	_, _ = graph.AddEdge("procurement", "finance", 0)

	_, _ = graph.AddEdge("legal", "notary", 0)
	_, _ = graph.AddEdge("notary", "vault", 0)

	// Stage 2: Detect cyclicity and retrieve canonical witness cycles.
	result, err := dfs.DetectCycles(graph)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	// Stage 3: Print the summary and the stable witness list.
	fmt.Println("hasCycle:", result.HasCycle)
	fmt.Println("witness-count:", len(result.Cycles))
	fmt.Println("witness-1:", strings.Join(result.Cycles[0], " -> "))
	fmt.Println("witness-2:", strings.Join(result.Cycles[1], " -> "))

	// Output:
	// hasCycle: true
	// witness-count: 2
	// witness-1: approvals -> triage -> security -> approvals
	// witness-2: finance -> legal -> procurement -> finance
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-DFS_Escalation_Loop_Witness-blue?logo=go)](https://go.dev/play/p/Y016CxIXaz_o)

**Why the result looks like this:**
- `DetectCycles` returns a deterministic witness set, not an exhaustive listing of every simple-cycle encoding.
- Each witness is returned as a closed cycle, which makes the loop explicit for human operators and log pipelines.
- Canonicalization rotates each cycle to a stable representative, which is why the first cycle is printed from `approvals` rather than from another vertex in the same loop.
- This is the right operational trade-off for workflow debugging, approval-loop audits, and deadlock-style escalation analysis.

---

These scenarios illustrate the intended professional use of `lvlath/dfs`:
- use `DFS` when you need deterministic depth-first structural exploration,
- use `Forest` when coverage must span disconnected regions,
- use `WithMaxDepth` to bound operational blast radius,
- use `TopologicalSort` for deterministic dependency execution plans,
- use `DetectCycles` for fast, auditable loop witnesses rather than exhaustive cycle enumeration.

---

## 4.9. Pitfalls & Best Practices (Architectural Mastery)

To use `lvlath/dfs` correctly in production pipelines, treat its traversal artifacts and witness contracts as first-class mathematical outputs rather than incidental implementation leftovers.

> [!CAUTION]
> **1. Discovery vs. Finish Order**
> `Result.Order` strictly records the **Post-Order** (when vertices *finish* traversal).
> *   **The Pitfall:** Do not confuse this with entry/discovery order.
> *   **The Solution:** If you need to track the discovery sequence, use the `WithOnVisit` hook to append vertices to your own slice.

> [!CAUTION]
> **2. Results on Aborting Failure**
> When a traversal is aborted (e.g., via `context.Cancellation`), the `Result.Order` is set to `nil`.
> *   **Why:** A partial post-order is mathematically meaningless for most algorithms.
> *   **The Practice:** Read `Visited`, `Depth`, or `Parent` instead to analyze the state of the graph reached before the crash.

> [!IMPORTANT]
> **3. Traversal Policy vs. Topology Mutation**
> Never mutate the underlying graph structure just to simulate a runtime boundary or "quarantine" a node.
> *   **The Way:** Use the `FilterNeighbor` option to enforce traversal policies. This keeps your core topology stable and thread-safe while allowing dynamic branch pruning.

> [!IMPORTANT]
> **4. Error Handling: Sentinels over Strings**
> Never parse error strings. `lvlath/dfs` wraps user hooks and timeouts using `%w`, keeping the chain intact.
> *   **The Rule:** Always use `errors.Is(err, dfs.ErrCycleDetected)` or `errors.Is(err, dfs.ErrOptionViolation)` for logic branching.

> [!WARNING]
> **5. Witness Discovery vs. Exhaustive Sets**
> `DetectCycles` is an optimized tool for proving cyclicity, not a search engine for all possible paths.
> *   **The Contract:** It returns a deterministic **witness set** to prove a loop exists. It does not promise a mathematically exhaustive catalog of all simple cycles (which is an NP-hard task).

> [!CAUTION]
> **6. The Directed-Graph Gate in Topological Sorting**
> Topological sorting is defined only for directed graphs.
> *   **The Gate:** If the graph fails the package’s directed-policy check, `TopologicalSort` returns `ErrGraphNotDirected`.
> *   **The Constraint:** Only directed outgoing relations contribute to the ordering. Undirected relations are ignored in the final topological plan.

> [!IMPORTANT]
> **7. Stability & Concurrency**
> The package reads the graph progressively and does not take a global immutable snapshot.
> *   **The Requirement:** Do not mutate graph topology concurrently during traversal. For reproducible results, the graph must remain topology-stable during execution.

> [!NOTE]
> **8. Ownership & Determinism**
> *   **Ownership:** Once returned, the maps and slices inside `Result` belong entirely to the caller. The package does not retain or mutate them.
> *   **Determinism:** Treat it as a **contract**. If the output order changes for the same graph, it indicates a change in graph ordering or a regression in the implementation.

---

## 4.10. Practical Recipes

*   **Blast Radius (Impact Analysis):** Run `DFS(start, WithMaxDepth(k))` to bound the analysis area around an incident node without scanning the entire topology.
*   **Deterministic Crawler:** Use `WithContext(ctx)` and trigger `cancel()` from within an `OnVisit` hook once a domain condition is met. Process the partial `Visited` map.
*   **Release Planning:** Model CI/CD steps as a directed graph. Execute `TopologicalSort`. If `ErrCycleDetected` is returned, halt the pipeline to prevent circular deadlocks.
*   **Cycle Auditing:** Use `DetectCycles` on resource-allocation graphs. The `Cycles` output provides the exact sequence of locks forming a deadlock.
*   **Performance Hooks:** Keep `OnVisit` and `OnExit` hooks lightweight and side-effect aware, as they execute directly in the hot traversal paths.

---
**lvlath/dfs**: deterministic by contract, strict in witness semantics, and safe to compose.

> Next: [5. Shortest Paths: Dijkstra ->](DIJKSTRA.md)
