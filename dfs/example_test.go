// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package dfs_test

import (
	"fmt"
	"sort"
	"strings"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dfs"
)

// AI-HINTS (file):
//   - Examples must demonstrate real current API only.
//   - Example outputs must be stable and CI-safe.
//   - Do not imply stronger guarantees than the package contract provides.
//   - Use scenario-driven pipelines: build -> algorithm -> consume.
//   - Keep all printed data deterministic; never print map iteration directly.

// ExampleDFS_infrastructureInspection MAIN DESCRIPTION (2+ lines, no marketing).
// Demonstrates a deterministic infrastructure-inspection pipeline over a directed dependency graph:
// build a service topology -> traverse with quarantine filtering and audit hooks -> consume trace and post-order.
//
// Implementation:
//   - Stage 1: Build a deterministic directed infrastructure graph with multiple operational zones.
//   - Stage 2: Run DFS from the ingress gateway with a quarantine filter and entry/exit hooks.
//   - Stage 3: Print the inspection trace, skipped-neighbor diagnostics, and stable post-order.
//
// Behavior highlights:
//   - Uses WithFilterNeighbor to simulate a quarantine boundary without mutating topology.
//   - Uses WithOnVisit and WithOnExit as deterministic audit hooks.
//   - Prints Result.Order, which is finish order (post-order), not discovery order.
//
// Inputs:
//   - None (the topology is deterministic and hard-coded).
//
// Returns:
//   - None (prints a stable infrastructure-inspection report).
//
// Errors:
//   - Any unexpected traversal error is printed and the example returns early.
//
// Determinism:
//   - Stable for identical graph construction order and the current package traversal contract.
//
// Complexity:
//   - Time O(V+E), Space O(V).
//
// Notes:
//   - The quarantine branch remains present in the graph but is blocked by policy at traversal time.
//   - The printed post-order is identical to Result.Order and to the exit hook sequence.
//
// AI-Hints:
//   - Hooks are observers here; keep them deterministic and side-effect aware.
//   - FilterNeighbor affects reachability and diagnostics, not graph structure.
//   - Order is post-order finish order, not discovery order.
func ExampleDFS_infrastructureInspection() {
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

	entered := make([]string, 0, graph.VertexCount())
	finished := make([]string, 0, graph.VertexCount())

	filter := func(vertexID string) bool {
		return !strings.HasPrefix(vertexID, "quarantine-")
	}

	onVisit := func(vertexID string) error {
		entered = append(entered, vertexID)
		return nil
	}

	onExit := func(vertexID string) error {
		finished = append(finished, vertexID)
		return nil
	}

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

	fmt.Println("entered:", joinExamplePath(entered))
	fmt.Println("finished:", joinExamplePath(finished))
	fmt.Println("skipped:", result.SkippedNeighbors)
	fmt.Println("postorder:", joinExamplePath(result.Order))

	// Output:
	// entered: gateway -> auth -> profile -> db -> secrets -> billing -> ledger -> reports -> archive -> cache -> replicas -> warmer -> metrics -> edge -> traffic
	// finished: db -> profile -> secrets -> auth -> ledger -> archive -> reports -> billing -> replicas -> metrics -> warmer -> cache -> traffic -> edge -> gateway
	// skipped: 1
	// postorder: db -> profile -> secrets -> auth -> ledger -> archive -> reports -> billing -> replicas -> metrics -> warmer -> cache -> traffic -> edge -> gateway
}

// ExampleDFS_fullTraversalInventorySweep MAIN DESCRIPTION (2+ lines, no marketing).
// Demonstrates deterministic DFS-forest traversal for an inventory sweep across disconnected service islands:
// build three isolated topology islands -> run WithFullTraversal -> consume forest-root, depth, and parent semantics.
//
// Implementation:
//   - Stage 1: Build three disconnected directed service islands with deterministic vertex identifiers.
//   - Stage 2: Run DFS with WithFullTraversal so the traversal becomes a DFS forest.
//   - Stage 3: Print forest roots, root depths, selected parent relations, and visited count.
//
// Behavior highlights:
//   - FullTraversal creates a DFS forest rather than a single rooted tree.
//   - Depth resets at each new forest root.
//   - Parent is assigned only for vertices actually entered as non-roots.
//
// Inputs:
//   - None (the topology is deterministic and hard-coded).
//
// Returns:
//   - None (prints a stable forest-sweep summary).
//
// Errors:
//   - Any unexpected traversal error is printed and the example returns early.
//
// Determinism:
//   - Stable under the package contract for full traversal and deterministic root selection.
//
// Complexity:
//   - Time O(V+E), Space O(V).
//
// Notes:
//   - The example intentionally leaves startID empty because full traversal determines roots from the graph itself.
//   - Forest roots are derived from visited vertices that do not appear in Parent.
//
// AI-Hints:
//   - In full traversal mode, Depth is measured from each tree root, not from a single global origin.
//   - Root vertices must not appear in Parent.
//   - FullTraversal is appropriate for inventory/indexing sweeps over disconnected topology.
func ExampleDFS_fullTraversalInventorySweep() {
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
	_, _ = graph.AddEdge("zone-z:1-web", "zone-z:2-edge-cache", 0)
	_, _ = graph.AddEdge("zone-z:1-worker", "zone-z:2-jobs", 0)
	_, _ = graph.AddEdge("zone-z:1-worker", "zone-z:2-metrics", 0)

	result, err := dfs.DFS(graph, "", dfs.WithFullTraversal())
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	roots := make([]string, 0, len(result.Visited))
	for vertexID := range result.Visited {
		if _, hasParent := result.Parent[vertexID]; !hasParent {
			roots = append(roots, vertexID)
		}
	}
	sort.Strings(roots)

	rootDepths := make([]string, 0, len(roots))
	for _, root := range roots {
		rootDepths = append(rootDepths, fmt.Sprintf("%s=%d", root, result.Depth[root]))
	}

	parentSamples := []string{
		fmt.Sprintf("zone-a:2-db=%s", result.Parent["zone-a:2-db"]),
		fmt.Sprintf("zone-m:2-lake=%s", result.Parent["zone-m:2-lake"]),
		fmt.Sprintf("zone-z:2-jobs=%s", result.Parent["zone-z:2-jobs"]),
	}

	_, rootHasParent := result.Parent["zone-a:0-gw"]

	fmt.Println("roots:", strings.Join(roots, ", "))
	fmt.Println("root-depths:", strings.Join(rootDepths, ", "))
	fmt.Println("parent-samples:", strings.Join(parentSamples, ", "))
	fmt.Println("root-has-parent:", rootHasParent)
	fmt.Println("visited:", len(result.Visited))

	// Output:
	// roots: zone-a:0-gw, zone-m:0-gw, zone-z:0-gw
	// root-depths: zone-a:0-gw=0, zone-m:0-gw=0, zone-z:0-gw=0
	// parent-samples: zone-a:2-db=zone-a:1-api, zone-m:2-lake=zone-m:1-etl, zone-z:2-jobs=zone-z:1-worker
	// root-has-parent: false
	// visited: 18
}

// ExampleDFS_depthLimitedBlastRadius MAIN DESCRIPTION (2+ lines, no marketing).
// Demonstrates a depth-limited blast-radius analysis from an incident root:
// build a deterministic dependency graph -> run DFS with WithMaxDepth -> consume reachable-node and depth output.
//
// Implementation:
//   - Stage 1: Build an incident-centered directed dependency graph with a controlled outer ring.
//   - Stage 2: Run DFS from the incident root with WithMaxDepth(2).
//   - Stage 3: Print the exact visited post-order, selected depths, and whether the outer ring was entered.
//
// Behavior highlights:
//   - MaxDepth is enforced before entry into deeper vertices.
//   - Depth values remain tree-depth values, not shortest-path distances.
//   - Outer-ring vertices remain unvisited when they would require depth 3 or more.
//
// Inputs:
//   - None (the topology is deterministic and hard-coded).
//
// Returns:
//   - None (prints a stable blast-radius summary).
//
// Errors:
//   - Any unexpected traversal error is printed and the example returns early.
//
// Determinism:
//   - Stable for the current package traversal contract and deterministic graph topology.
//
// Complexity:
//   - Time O(V+E) on the traversed portion, Space O(V).
//
// Notes:
//   - The example demonstrates controlled horizon analysis without mutating the graph.
//
// AI-Hints:
//   - WithMaxDepth limits traversal reachability, not graph structure.
//   - Depth-limited DFS is useful for bounded blast-radius and local impact analysis.
func ExampleDFS_depthLimitedBlastRadius() {
	graph, _ := core.NewGraph(core.WithDirected(true))

	_, _ = graph.AddEdge("incident", "api", 0)
	_, _ = graph.AddEdge("incident", "audit", 0)
	_, _ = graph.AddEdge("incident", "queue", 0)

	_, _ = graph.AddEdge("api", "auth", 0)
	_, _ = graph.AddEdge("api", "profile", 0)

	_, _ = graph.AddEdge("auth", "db", 0)
	_, _ = graph.AddEdge("profile", "cache", 0)

	_, _ = graph.AddEdge("audit", "events", 0)
	_, _ = graph.AddEdge("audit", "ledger", 0)
	_, _ = graph.AddEdge("events", "archive", 0)

	_, _ = graph.AddEdge("queue", "worker", 0)
	_, _ = graph.AddEdge("worker", "mail", 0)
	_, _ = graph.AddEdge("worker", "search", 0)
	_, _ = graph.AddEdge("mail", "smtp", 0)
	_, _ = graph.AddEdge("search", "index", 0)

	result, err := dfs.DFS(graph, "incident", dfs.WithMaxDepth(2))
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	selectedDepths := []string{
		fmt.Sprintf("api=%d", result.Depth["api"]),
		fmt.Sprintf("worker=%d", result.Depth["worker"]),
		fmt.Sprintf("ledger=%d", result.Depth["ledger"]),
	}

	fmt.Println("visited:", joinExamplePath(result.Order))
	fmt.Println("depths:", strings.Join(selectedDepths, ", "))
	fmt.Println("outer-ring-entered:", result.Visited["db"] || result.Visited["cache"] || result.Visited["archive"] || result.Visited["smtp"] || result.Visited["index"])

	// Output:
	// visited: auth -> profile -> api -> events -> ledger -> audit -> worker -> queue -> incident
	// depths: api=1, worker=2, ledger=2
	// outer-ring-entered: false
}

// ExampleTopologicalSort_releasePipeline MAIN DESCRIPTION (2+ lines, no marketing).
// Demonstrates a deterministic release-plan build for a directed deployment pipeline:
// build a real DAG -> compute a topological order -> consume the resulting exact execution plan.
//
// Implementation:
//   - Stage 1: Build a deterministic directed release DAG with explicit stage identifiers.
//   - Stage 2: Run TopologicalSort over the full graph.
//   - Stage 3: Print the exact execution plan as a stable ordered pipeline.
//
// Behavior highlights:
//   - Uses the public TopologicalSort API only.
//   - Demonstrates a deterministic exact output, not an invariant-only sample.
//   - Models a real release pipeline rather than a toy diamond graph.
//
// Inputs:
//   - None (the topology is deterministic and hard-coded).
//
// Returns:
//   - None (prints a stable release plan).
//
// Errors:
//   - Any unexpected sorting error is printed and the example returns early.
//
// Determinism:
//   - The release plan is stable under the current package determinism law and graph topology.
//
// Complexity:
//   - Time O(V+E), Space O(V).
//
// Notes:
//   - This example intentionally uses a uniquely constrained stage chain so the exact output is authoritative.
//
// AI-Hints:
//   - This algorithm is only valid for directed graphs.
//   - Example output is exact here because the constructed DAG fixes one deterministic plan.
func ExampleTopologicalSort_releasePipeline() {
	graph, _ := core.NewGraph(core.WithDirected(true))

	_, _ = graph.AddEdge("01-spec-freeze", "02-schema-lock", 0)
	_, _ = graph.AddEdge("02-schema-lock", "03-codegen", 0)
	_, _ = graph.AddEdge("03-codegen", "04-unit-tests", 0)
	_, _ = graph.AddEdge("04-unit-tests", "05-security-scan", 0)
	_, _ = graph.AddEdge("05-security-scan", "06-package", 0)
	_, _ = graph.AddEdge("06-package", "07-image-sign", 0)
	_, _ = graph.AddEdge("07-image-sign", "08-staging-deploy", 0)
	_, _ = graph.AddEdge("08-staging-deploy", "09-smoke-tests", 0)
	_, _ = graph.AddEdge("09-smoke-tests", "10-approval", 0)
	_, _ = graph.AddEdge("10-approval", "11-prod-deploy", 0)
	_, _ = graph.AddEdge("11-prod-deploy", "12-post-checks", 0)
	_, _ = graph.AddEdge("12-post-checks", "13-closeout", 0)

	order, err := dfs.TopologicalSort(graph)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("release-plan:", joinExamplePath(order))

	// Output:
	// release-plan: 01-spec-freeze -> 02-schema-lock -> 03-codegen -> 04-unit-tests -> 05-security-scan -> 06-package -> 07-image-sign -> 08-staging-deploy -> 09-smoke-tests -> 10-approval -> 11-prod-deploy -> 12-post-checks -> 13-closeout
}

// ExampleDetectCycles_escalationLoopWitness MAIN DESCRIPTION (2+ lines, no marketing).
// Demonstrates deterministic witness-cycle reporting for an escalation and approval workflow:
// build a directed workflow with multiple loop-zones -> run DetectCycles -> consume the witness summary and canonical cycles.
//
// Implementation:
//   - Stage 1: Build a deterministic directed workflow graph with two escalation loops.
//   - Stage 2: Run DetectCycles over the full graph.
//   - Stage 3: Print the summary flag, witness count, and canonical witness cycles.
//
// Behavior highlights:
//   - Demonstrates the witness contract honestly: the function reports deterministic witnesses, not an exhaustive theorem prover.
//   - Directed-cycle canonicalization preserves orientation.
//   - The printed cycles are closed canonical sequences.
//
// Inputs:
//   - None (the topology is deterministic and hard-coded).
//
// Returns:
//   - None (prints a stable witness summary).
//
// Errors:
//   - Any unexpected cycle-detection error is printed and the example returns early.
//
// Determinism:
//   - Stable under deterministic graph traversal order and canonical witness normalization.
//
// Complexity:
//   - Time O(V+E+W*L), Space O(V+W*L), where W is witness count and L is witness length.
//
// Notes:
//   - The result is a deterministic witness set discovered by the implemented algorithm.
//   - The example does not claim exhaustive enumeration of every mathematically representable simple cycle.
//
// AI-Hints:
//   - DetectCycles returns witness cycles, not an exhaustive simple-cycle catalogue.
//   - Directed cycle canonicalization must preserve edge orientation.
func ExampleDetectCycles_escalationLoopWitness() {
	graph, _ := core.NewGraph(core.WithDirected(true))

	_, _ = graph.AddEdge("entry", "triage", 0)
	_, _ = graph.AddEdge("triage", "security", 0)
	_, _ = graph.AddEdge("security", "approvals", 0)
	_, _ = graph.AddEdge("approvals", "finance", 0)
	_, _ = graph.AddEdge("finance", "legal", 0)
	_, _ = graph.AddEdge("legal", "security", 0)

	_, _ = graph.AddEdge("triage", "ops", 0)
	_, _ = graph.AddEdge("ops", "change", 0)
	_, _ = graph.AddEdge("change", "qa", 0)
	_, _ = graph.AddEdge("qa", "ops", 0)

	_, _ = graph.AddEdge("finance", "archive", 0)
	_, _ = graph.AddEdge("ops", "notifications", 0)
	_, _ = graph.AddEdge("notifications", "archive", 0)
	_, _ = graph.AddEdge("archive", "sink", 0)

	result, err := dfs.DetectCycles(graph)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("hasCycle:", result.HasCycle)
	fmt.Println("witness-count:", len(result.Cycles))
	fmt.Println("witness-1:", joinExamplePath(result.Cycles[0]))
	fmt.Println("witness-2:", joinExamplePath(result.Cycles[1]))

	// Output:
	// hasCycle: true
	// witness-count: 2
	// witness-1: approvals -> finance -> legal -> security -> approvals
	// witness-2: change -> qa -> ops -> change
}

// joinExamplePath converts an ordered vertex slice into stable printable output.
//
// Implementation:
//   - Stage 1: Join the ordered slice with a stable string separator.
//
// Behavior highlights:
//   - Preserves the exact order provided by the caller.
//   - Produces stable CI-safe output.
//
// Inputs:
//   - values: ordered vertex identifiers.
//
// Returns:
//   - string: joined ordered representation.
//
// Errors:
//   - N/A.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(total bytes), Space O(total bytes).
//
// Notes:
//   - The helper is intentionally presentation-only.
//
// AI-Hints:
//   - Never print map iteration directly in examples; convert to stable ordered forms first.
func joinExamplePath(values []string) string {
	return strings.Join(values, " -> ")
}
