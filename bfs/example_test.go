// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package bfs_test

import (
	"context"
	"errors"
	"fmt"
	"sort"

	"github.com/lvlath/go/bfs"
	"github.com/lvlath/go/core"
)

// AI-HINTS (file):
//   - Examples must be deterministic and CI-stable: never use time-based cancellation.
//   - Prefer context.WithCancel + deterministic trigger from a hook (Nth visit / specific vertex).
//   - Do not print map iteration directly; print slices or selected keys only.
//   - Neighbor tie-break is delegated to core.Graph.NeighborIDs(): unique neighbor IDs in lex order.
//   - Components() computes weak connectivity (undirected relation), not SCC.

// ExampleBFS_IncidentRunbook demonstrates shortest-hop routing in a service mesh.
//
// Scenario:
//
//	An incident commander needs a deterministic hop-minimal route from "pager" to "db".
//	The mesh has multiple competing routes, but BFS must deterministically pick one
//	based on core neighbor enumeration order.
//
// Implementation:
//   - Stage 1: Build a directed service mesh (12+ edges).
//   - Stage 2: Run BFS from "pager".
//   - Stage 3: Reconstruct the chosen shortest path to "db" and print a small Depth audit.
//
// Behavior highlights:
//   - Determinism: tie-break is governed by core.NeighborIDs lex order.
//   - PathTo: reconstructs one shortest path using Parent links.
//
// Inputs:
//   - None (deterministic topology).
//
// Returns:
//   - None (prints path + selected depths).
//
// Errors:
//   - Any unexpected error is printed and the example returns early.
//
// Determinism:
//   - NeighborIDs are unique + lex-sorted; BFS preserves that ordering.
//
// Complexity:
//   - Time O(|V|+|E|), Space O(|V|).
//
// Notes:
//   - This example shows a real pipeline: build topology -> traverse -> extract path & distances.
//
// AI-Hints:
//   - If you need a different tie-break, enforce it at the graph layer (neighbor ordering / IDs).
func ExampleBFS_IncidentRunbook() {
	g, _ := core.NewGraph(core.WithDirected(true))

	// Mesh edges (>= 12):
	// pager -> api-gw -> auth -> user -> db
	// pager -> api-gw -> billing -> db
	// pager -> edge -> cache -> db
	// pager -> edge -> user (alternate)
	// Extra observability branches:
	// api-gw -> metrics, auth -> audit, user -> search, billing -> ledger
	edges := [][2]string{
		{"pager", "api-gw"},
		{"pager", "edge"},

		{"api-gw", "auth"},
		{"api-gw", "billing"},
		{"api-gw", "metrics"},

		{"auth", "user"},
		{"auth", "audit"},

		{"user", "db"},
		{"user", "search"},

		{"billing", "db"},
		{"billing", "ledger"},

		{"edge", "cache"},
		{"edge", "user"},
		{"cache", "db"},
	}

	for _, e := range edges {
		if _, err := g.AddEdge(e[0], e[1], 0); err != nil {
			fmt.Println("error:", err)
			return
		}
	}

	res, err := bfs.BFS(g, "pager")
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	path, err := res.PathTo("db")
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	// Print a compact “runbook” view.
	fmt.Println("path:", path)
	fmt.Println("depth:", "api-gw", res.Depth["api-gw"], "edge", res.Depth["edge"], "db", res.Depth["db"])
	// Output:
	// path: [pager api-gw billing db]
	// depth: api-gw 1 edge 1 db 3
}

// ExampleBFS_BlastRadius demonstrates inclusive MaxDepth as a practical "blast radius" tool.
//
// Scenario:
//
//	Starting from an affected service, list everything within 2 hops (inclusive semantics).
//
// Implementation:
//   - Stage 1: Build a directed dependency graph (12+ edges).
//   - Stage 2: Run BFS with WithMaxDepth(2).
//   - Stage 3: Print Order (visited-at-dequeue) as the deterministic radius expansion trace.
//
// Behavior highlights:
//   - MaxDepth is inclusive: depth==2 vertices are visited but not expanded.
//
// Inputs:
//   - None (deterministic graph).
//
// Returns:
//   - None (prints traversal order).
//
// Errors:
//   - Any unexpected error is printed and the example returns early.
//
// Determinism:
//   - NeighborIDs order is deterministic; BFS preserves it.
//
// Complexity:
//   - Time O(|V|+|E|), Space O(|V|).
//
// Notes:
//   - WithMaxDepth(0) visits the root only; WithMaxDepth(MaxDepthUnlimited) is unlimited.
//
// AI-Hints:
//   - Prefer bfs.MaxDepthUnlimited for unlimited traversal; avoid magic -1.
func ExampleBFS_BlastRadius() {
	g, _ := core.NewGraph(core.WithDirected(true))

	// Dependency fan-out from "auth" (>= 12 edges).
	edges := [][2]string{
		{"auth", "user"},
		{"auth", "audit"},
		{"auth", "session"},

		{"user", "db"},
		{"user", "cache"},
		{"user", "search"},

		{"session", "cache"},
		{"session", "metrics"},

		{"audit", "ledger"},
		{"audit", "archive"},

		{"db", "backup"},
		{"db", "replica"},

		{"search", "indexer"},
	}
	for _, e := range edges {
		if _, err := g.AddEdge(e[0], e[1], 0); err != nil {
			fmt.Println("error:", err)
			return
		}
	}

	res, err := bfs.BFS(g, "auth", bfs.WithMaxDepth(2))
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println(res.Order)
	// Output:
	// [auth audit session user archive ledger cache metrics db search]
}

// ExampleBFS_PolicyFirewall demonstrates relation-level filtering with Skipped accounting.
//
// Scenario:
//
//	A company enforces network segmentation: "prod" services must not reach "staging".
//	We simulate this by filtering neighbor relations based on ID prefixes and count what was blocked.
//
// Implementation:
//   - Stage 1: Build a mixed environment graph (14+ edges).
//   - Stage 2: Apply WithFilterNeighbor to block prod -> staging relations.
//   - Stage 3: Print traversal order and Skipped count.
//
// Behavior highlights:
//   - FilterNeighbor is relation-level (currID,nbrID); topology is not mutated.
//   - Skipped counts blocked relations deterministically.
//
// Inputs:
//   - None (deterministic topology + policy).
//
// Returns:
//   - None (prints order + skipped).
//
// Errors:
//   - Any unexpected error is printed and the example returns early.
//
// Determinism:
//   - NeighborIDs order is deterministic; filter decisions are pure and stable.
//
// Complexity:
//   - Time O(|V|+|E|), Space O(|V|).
//
// AI-Hints:
//   - Use filters for policy simulation without mutating the graph.
func ExampleBFS_PolicyFirewall() {
	g, _ := core.NewGraph(core.WithDirected(true))

	// IDs embed environment: prod:* and stg:*.
	// Build >= 14 edges, including cross-env edges that must be blocked.
	edges := [][2]string{
		{"prod:gw", "prod:auth"},
		{"prod:gw", "prod:billing"},
		{"prod:gw", "stg:gw"}, // must be blocked
		{"prod:auth", "prod:user"},
		{"prod:auth", "stg:audit"}, // must be blocked
		{"prod:user", "prod:db"},
		{"prod:user", "prod:cache"},
		{"prod:billing", "prod:db"},
		{"prod:billing", "stg:db"}, // must be blocked

		{"stg:gw", "stg:auth"},
		{"stg:auth", "stg:user"},
		{"stg:user", "stg:db"},

		{"prod:db", "prod:backup"},
		{"stg:db", "stg:backup"},
	}
	for _, e := range edges {
		if _, err := g.AddEdge(e[0], e[1], 0); err != nil {
			fmt.Println("error:", err)
			return
		}
	}

	// Block prod -> stg relations deterministically.
	isProd := func(id string) bool { return len(id) >= 5 && id[:5] == "prod:" }
	isStg := func(id string) bool { return len(id) >= 4 && id[:4] == "stg:" }

	filter := func(currID, nbrID string) bool {
		if isProd(currID) && isStg(nbrID) {
			return false
		}
		return true
	}

	res, err := bfs.BFS(g, "prod:gw", bfs.WithFilterNeighbor(filter))
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("order:", res.Order)
	fmt.Println("skipped:", res.Skipped)
	// Output:
	// order: [prod:gw prod:auth prod:billing prod:user prod:db prod:cache prod:backup]
	// skipped: 3
}

// ExampleBFS_CancellationPartial demonstrates deterministic cancellation and partial results.
//
// Scenario:
//
//	A crawler scans reachable services but stops immediately when it sees a "tripwire" node,
//	returning a partial traversal trace for diagnostics.
//
// Implementation:
//   - Stage 1: Build a directed graph (12+ edges) with a deterministic order.
//   - Stage 2: Use context.WithCancel and cancel from OnVisit when visiting "tripwire".
//   - Stage 3: Print cancellation classification and partial order.
//
// Behavior highlights:
//   - Cancellation is deterministic (no timeouts).
//   - Partial result is returned alongside context.Canceled.
//
// Inputs:
//   - None.
//
// Returns:
//   - None (prints canceled flag and order).
//
// Errors:
//   - context.Canceled is expected.
//
// Determinism:
//   - Trigger is a specific vertex ID; order is deterministic by core neighbor ordering.
//
// Complexity:
//   - Time proportional to visited subset.
//
// AI-Hints:
//   - Never use WithTimeout in examples; cancel based on graph events instead.
func ExampleBFS_CancellationPartial() {
	g, _ := core.NewGraph(core.WithDirected(true))

	edges := [][2]string{
		{"root", "a"},
		{"root", "b"},
		{"root", "c"},

		{"a", "d"},
		{"a", "e"},
		{"b", "f"},
		{"b", "tripwire"},
		{"c", "g"},

		{"d", "h"},
		{"e", "i"},
		{"f", "j"},
		{"g", "k"},
	}
	for _, e := range edges {
		if _, err := g.AddEdge(e[0], e[1], 0); err != nil {
			fmt.Println("error:", err)
			return
		}
	}

	ctx, cancel := context.WithCancel(context.Background())

	onVisit := func(id string) error {
		if id == "tripwire" {
			cancel()
		}
		return nil
	}

	res, err := bfs.BFS(
		g, "root",
		bfs.WithContext(ctx),
		bfs.WithOnVisit(onVisit),
	)

	fmt.Println("canceled:", errors.Is(err, context.Canceled))
	fmt.Println("order:", res.Order)
	// Output:
	// canceled: true
	// order: [root a b c d e f tripwire]
}

// ExampleComponents_DiscoveryAndForest demonstrates weak connectivity discovery and full forest traversal.
//
// Scenario:
//
//	A platform team wants:
//	  1) the list of weakly-connected "islands" in the environment,
//	  2) a deterministic forest traversal order for indexing.
//
// Implementation:
//   - Stage 1: Build a directed graph with 3 weak components (12+ edges total).
//   - Stage 2: Compute weakly-connected components via Components().
//   - Stage 3: Run BFS with WithFullTraversal to produce a forest order.
//   - Stage 4: Demonstrate that PathTo remains anchored to StartID under FullTraversal.
//
// Behavior highlights:
//   - Components uses an undirected relation: directed chains still form one component.
//   - FullTraversal visits all vertices, but PathTo still enforces StartID anchoring.
//
// Inputs:
//   - None.
//
// Returns:
//   - None (prints components + forest order + ErrNoPath classification).
//
// Errors:
//   - Any unexpected error is printed and the example returns early.
//
// Determinism:
//   - Components are lex-sorted internally and sorted by stable key.
//
// Complexity:
//   - Components: O(V+E) plus sorting overhead.
//
// AI-Hints:
//   - Components ≠ SCC; it is weak connectivity.
//   - Under FullTraversal, Visited != reachable-from-StartID; PathTo enforces correctness.
func ExampleComponents_DiscoveryAndForest() {
	g, _ := core.NewGraph(core.WithDirected(true))

	// Component #1: A -> B -> C -> D
	// Component #2: M -> N, and M -> O
	// Component #3: X -> Y -> Z
	edges := [][2]string{
		{"A", "B"},
		{"B", "C"},
		{"C", "D"},
		{"A", "C"},

		{"M", "N"},
		{"M", "O"},
		{"N", "O"},
		{"O", "N"},

		{"X", "Y"},
		{"Y", "Z"},
		{"X", "Z"},

		// Extra directed edges inside components to increase realism.
		{"B", "D"},
		{"Y", "X"},
	}
	for _, e := range edges {
		if _, err := g.AddEdge(e[0], e[1], 0); err != nil {
			fmt.Println("error:", err)
			return
		}
	}

	// Stage 2: Weak components.
	compRes, err := bfs.Components(context.Background(), g)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	// Print components deterministically.
	// The result is already sorted by contract, but we only print stable slices.
	fmt.Println("components:", compRes.Components)

	// Stage 3: Forest traversal from "A".
	forestRes, err := bfs.BFS(g, "A", bfs.WithFullTraversal())
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println("forest:", forestRes.Order)

	// Stage 4: Anchoring law: D is reachable from A, Z is not.
	_, err = forestRes.PathTo("Z")
	fmt.Println("pathToZ_isNoPath:", errors.Is(err, bfs.ErrNoPath))

	// For visibility, show one internal sanity signal: first ID of each component.
	firstIDs := make([]string, 0, len(compRes.Components))
	for _, c := range compRes.Components {
		firstIDs = append(firstIDs, c[0])
	}
	sort.Strings(firstIDs)
	fmt.Println("component-roots:", firstIDs)

	// Output:
	// components: [[A B C D] [M N O] [X Y Z]]
	// forest: [A B C D M N O X Y Z]
	// pathToZ_isNoPath: true
	// component-roots: [A M X]
}
