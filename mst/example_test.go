// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package mst_test

import (
	"errors"
	"fmt"
	"slices"
	"sort"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/mst"
)

// ExampleMinimumSpanningTree_vlsiGlobalRouting demonstrates MST as a deterministic
// global-routing baseline for VLSI-style physical design.
//
// Scenario:
//   - A chip floorplan contains compute blocks, memory blocks, PHY blocks, power management,
//     and a NoC hub.
//   - Each weighted edge is a candidate metal route / channel reservation cost between blocks.
//   - The goal is not final sign-off routing; the goal is a cheap, acyclic connectivity skeleton
//     that can be refined later by timing-aware, congestion-aware, or Steiner-style tools.
//
// Process:
//   - Build a fixed weighted undirected graph of candidate interconnect channels.
//   - Run the canonical facade with default Kruskal strict-tree policy.
//   - Consume stable invariants: selected wires, connected components, and total routing cost.
//
// Static fixture note:
//   - Construction errors are ignored only because this package example uses fixed, audited data.
//   - Production code must check every core.NewGraph/AddEdge error.
func ExampleMinimumSpanningTree_vlsiGlobalRouting() {
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
		fmt.Println("error:", err)
		return
	}

	fmt.Printf(
		"vlsi algorithm=%s mode=%s blocks=%d wires=%d components=%d total=%.0f\n",
		result.Algorithm,
		result.Mode,
		result.VertexCount,
		len(result.Edges),
		result.ComponentCount,
		result.TotalWeight,
	)

	// Output:
	// vlsi algorithm=kruskal mode=strict_tree blocks=10 wires=9 components=1 total=34
}

// ExampleMinimumSpanningTree_spaceMeshForest demonstrates explicit forest mode for
// a satellite / orbital relay mesh.
//
// Scenario:
//   - Ground control receives topology candidates from several disconnected orbital groups.
//   - A strict MST would be mathematically wrong because not every satellite can currently
//     reach every other group.
//   - WithForest asks for the cheapest safe control backbone inside every reachable component,
//     while preserving the fact that the constellation is still split.
//
// Process:
//   - Build three disconnected weighted mesh components.
//   - Select AlgorithmPrim to demonstrate component growth from deterministic roots.
//   - Enable WithForest so the result is an MSF, not a hidden strict-tree fallback.
//
// Static fixture note:
//   - Construction errors are ignored only because this package example uses fixed, audited data.
//   - Production code must check every core.NewGraph/AddEdge error.
func ExampleMinimumSpanningTree_spaceMeshForest() {
	graph, _ := core.NewGraph(core.WithWeighted())

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
		fmt.Println("error:", err)
		return
	}

	fmt.Printf(
		"space_mesh algorithm=%s mode=%s satellites=%d links=%d components=%d roots=%v total=%.0f\n",
		result.Algorithm,
		result.Mode,
		result.VertexCount,
		len(result.Edges),
		result.ComponentCount,
		result.ComponentRoots,
		result.TotalWeight,
	)

	// Output:
	// space_mesh algorithm=prim mode=forest satellites=11 links=8 components=3 roots=[Arctic-1 Equator-1 Pacific-1] total=38
}

// ExampleKruskal_embeddingClustering demonstrates MST consumption in a machine-learning
// clustering workflow.
//
// Scenario:
//   - Each vertex is a vector cluster prototype from an embedding index.
//   - Edge weights are distances between prototypes.
//   - The MST exposes the cheapest global similarity backbone.
//   - Cutting the largest selected MST edges is a classic way to split the backbone into
//     a chosen number of clusters.
//
// Process:
//   - Build a fixed graph with three dense local groups and expensive cross-group bridges.
//   - Run Kruskal to obtain a deterministic MST.
//   - Sort the selected MST edges by descending weight and cut the two largest links.
//
// Static fixture note:
//   - Construction errors are ignored only because this package example uses fixed, audited data.
//   - Production code must check every core.NewGraph/AddEdge error.
func ExampleKruskal_embeddingClustering() {
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
		fmt.Println("error:", err)
		return
	}

	cutCandidates := append([]core.Edge(nil), result.Edges...)
	sort.SliceStable(cutCandidates, func(i, j int) bool {
		return cutCandidates[i].Weight > cutCandidates[j].Weight
	})

	cut1 := cutCandidates[0]
	cut2 := cutCandidates[1]
	remaining := result.TotalWeight - cut1.Weight - cut2.Weight

	fmt.Printf(
		"ml_clustering mst_edges=%d mst_total=%.2f cut1=%s-%s:%.2f cut2=%s-%s:%.2f remaining=%.2f clusters=%d\n",
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
	// ml_clustering mst_edges=8 mst_total=2.77 cut1=speech-03-fraud-01:1.08 cut2=vision-03-speech-01:0.94 remaining=0.75 clusters=3
}

// ExampleMinimumSpanningTree_smartGridProtectionPolicy demonstrates sentinel-first rejection
// of a directed edge inside an otherwise weighted smart-grid topology.
//
// Scenario:
//   - A planning graph mixes physical power links with a one-way telemetry/control channel.
//   - The power links are undirected connectivity candidates.
//   - The telemetry edge is directed and must not be treated as a physical bidirectional
//     grid tie by MST.
//   - The package rejects this graph before any tree construction starts.
//
// Process:
//   - Build a mixed-edge graph with realistic grid assets.
//   - Insert several undirected power links and one directed control edge.
//   - Use errors.Is to classify the failure without matching error strings.
//
// Static fixture note:
//   - Construction errors are ignored only because this package example uses fixed, audited data.
//   - Production code must check every core.NewGraph/AddEdge error.
func ExampleMinimumSpanningTree_smartGridProtectionPolicy() {
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

	_, _ = graph.AddEdge("ControlCenter", "Substation-A", 1, core.WithEdgeDirected(true))

	_, err := mst.MinimumSpanningTree(graph)

	fmt.Printf(
		"smart_grid_policy invalid=%t directed_edge=%t\n",
		errors.Is(err, mst.ErrInvalidGraph),
		errors.Is(err, mst.ErrDirectedEdge),
	)

	// Output:
	// smart_grid_policy invalid=true directed_edge=true
}

// ExampleSteinerTree_chipletBackbone demonstrates multi-terminal interconnect design.
// A shared fabric hub can connect four required processing blocks more cheaply than direct links.
// Implementation:
//   - Stage 1: Create a weighted undirected candidate-channel graph with original edge IDs.
//   - Stage 2: Run the KMB approximation for only the four required blocks.
//   - Stage 3: Inspect canonical original edges and the unique-edge network cost.
//
// Behavior highlights:
//   - Closure distances are not published as graph edges or as TotalWeight.
//
// Inputs:
//   - Fixed candidate channels and four explicitly named terminal blocks.
//
// Returns:
//   - Deterministic terminal coverage, original edge IDs, and a checked total.
//
// Errors:
//   - Graph construction or SteinerTree failure is reported rather than silently ignored.
//
// Determinism:
//   - Original edge IDs are published in lexical order.
//
// Complexity:
//   - Dominated by three terminal-source SSSP queries and O(k^2 log k) closure selection.
//
// Notes:
//   - The resulting network is an approximation, not a timing- or congestion-aware layout.
//
// AI-Hints:
//   - Do not compare the cost of synthetic closure links with the cost of final original links.
func ExampleSteinerTree_chipletBackbone() {
	graph, err := core.NewGraph(core.WithWeighted())
	if err != nil {
		fmt.Println("graph:", err)
		return
	}

	// The four low-cost hub attachments outperform the six direct terminal links.
	for _, candidate := range []struct {
		id, from, to string
		cost         float64
	}{
		{"CPU-NOC", "CPU", "NoC", 2},
		{"DSP-NOC", "DSP", "NoC", 2},
		{"GPU-NOC", "GPU", "NoC", 2},
		{"NPU-NOC", "NPU", "NoC", 2},
		{"CPU-DSP", "CPU", "DSP", 5},
		{"CPU-GPU", "CPU", "GPU", 5},
		{"CPU-NPU", "CPU", "NPU", 5},
		{"DSP-GPU", "DSP", "GPU", 5},
		{"DSP-NPU", "DSP", "NPU", 5},
		{"GPU-NPU", "GPU", "NPU", 5},
		{"NOC-RELAY", "NoC", "Relay", 7},
		{"RELAY-GPU", "Relay", "GPU", 8},
		{"RELAY-DSP", "Relay", "DSP", 9},
	} {
		if _, err := graph.AddEdge(candidate.from, candidate.to, candidate.cost, core.WithID(candidate.id)); err != nil {
			fmt.Println("edge:", err)
			return
		}
	}

	network, err := mst.SteinerTree(graph, []string{"GPU", "CPU", "NPU", "DSP"})
	if err != nil {
		fmt.Println("steiner:", err)
		return
	}
	ids := make([]string, 0, len(network.Edges))
	for _, edge := range network.Edges {
		ids = append(ids, edge.ID)
	}
	fmt.Printf("terminals=%v cost=%.0f original=%v\n", network.Terminals, network.TotalWeight, ids)
	// Output:
	// terminals=[CPU DSP GPU NPU] cost=8 original=[CPU-NOC DSP-NOC GPU-NOC NPU-NOC]
}

// ExampleSteinerTree_fiberParallelRoutes demonstrates auditable multigraph edge identity.
// Two zero-cost parallel fiber links must remain distinct even when their endpoints match.
// Implementation:
//   - Stage 1: Build a multi-edge network with physical original connection IDs.
//   - Stage 2: Request a terminal-covering network for HQ, LAB, and DR.
//   - Stage 3: Confirm that the deterministic lowest-ID parallel link is retained.
//
// Behavior highlights:
//   - Zero-weight links are real connections, not absent edges.
//
// Inputs:
//   - Twelve candidate links, including identical-cost parallel fiber records.
//
// Returns:
//   - Original, traceable IDs and the cost of three unique selected links.
//
// Errors:
//   - Construction and shortest-path failures are handled explicitly.
//
// Determinism:
//   - Parallel ties use (Weight, Edge.ID), independently of stored endpoint direction.
//
// Complexity:
//   - KMB O((k-1)Tsssp(V,E)+k^2 log k+Eu log Eu) for k=3.
//
// Notes:
//   - Parallel-link identity is preserved even though Dijkstra paths are reconstructed by vertex.
//
// AI-Hints:
//   - Never deduplicate the final network by endpoint pairs; use original Edge.ID.
func ExampleSteinerTree_fiberParallelRoutes() {
	graph, err := core.NewGraph(core.WithWeighted(), core.WithMultiEdges())
	if err != nil {
		fmt.Println("graph:", err)
		return
	}
	for _, candidate := range []struct {
		id, from, to string
		cost         float64
	}{
		{"z-supply", "HQ", "SW", 0},
		{"a-supply", "SW", "HQ", 0},
		{"lab", "LAB", "SW", 2},
		{"backup", "SW", "DR", 3},
		{"HQ-LAB", "HQ", "LAB", 7},
		{"HQ-DR", "HQ", "DR", 9},
		{"LAB-DR", "LAB", "DR", 8},
		{"HQ-POP", "HQ", "POP", 9},
		{"LAB-POP", "LAB", "POP", 9},
		{"DR-POP", "DR", "POP", 9},
		{"POP-PEER", "POP", "PEER", 2},
		{"PEER-SW", "PEER", "SW", 9},
	} {
		if _, err := graph.AddEdge(candidate.from, candidate.to, candidate.cost, core.WithID(candidate.id)); err != nil {
			fmt.Println("edge:", err)
			return
		}
	}

	network, err := mst.SteinerTree(graph, []string{"LAB", "DR", "HQ"})
	if err != nil {
		fmt.Println("steiner:", err)
		return
	}
	ids := make([]string, 0, len(network.Edges))
	for _, edge := range network.Edges {
		ids = append(ids, edge.ID)
	}
	fmt.Printf("cost=%.0f selected=%v\n", network.TotalWeight, ids)
	// Output:
	// cost=5 selected=[a-supply backup lab]
}

// ExampleSteinerTree_pathStorageModes compares the two supported KMB memory strategies.
// Recomputing path witnesses and retaining predecessor maps must preserve the same objective.
// Implementation:
//   - Stage 1: Build a graph with multiple candidate terminal connections and detours.
//   - Stage 2: Run independent KMB queries in recompute and retain modes.
//   - Stage 3: Compare detached original-edge output, not intermediate SSSP state.
//
// Behavior highlights:
//   - Retain trades O(kV) predecessor storage against repeated source-grouped SSSP calls.
//
// Inputs:
//   - Four terminals and an undirected weighted graph with an extra relay branch.
//
// Returns:
//   - Equal final cost and network witness under both modes.
//
// Errors:
//   - Any failed mode is reported rather than being silently replaced.
//
// Determinism:
//   - Canonical IDs and Kruskal tie-order keep both runs reproducible.
//
// Complexity:
//   - The recompute mode may perform up to k-1 additional SSSP runs; retain uses O(kV) history.
//
// Notes:
//   - Memory strategy does not select a different exact or heuristic Steiner solver.
//
// AI-Hints:
//   - Choose storage by workload measurement, not by assuming retained state is always faster.
func ExampleSteinerTree_pathStorageModes() {
	graph, err := core.NewGraph(core.WithWeighted())
	if err != nil {
		fmt.Println("graph:", err)
		return
	}
	for _, candidate := range []struct {
		id, from, to string
		cost         float64
	}{
		{"AX", "A", "X", 1}, {"BX", "B", "X", 2},
		{"CX", "C", "X", 2}, {"DX", "D", "X", 3},
		{"AB", "A", "B", 7}, {"AC", "A", "C", 7},
		{"AD", "A", "D", 8}, {"BC", "B", "C", 7},
		{"BD", "B", "D", 8}, {"CD", "C", "D", 8},
		{"XY", "X", "Y", 7}, {"AY", "A", "Y", 10},
		{"BY", "B", "Y", 10}, {"CY", "C", "Y", 10},
	} {
		if _, err := graph.AddEdge(candidate.from, candidate.to, candidate.cost, core.WithID(candidate.id)); err != nil {
			fmt.Println("edge:", err)
			return
		}
	}

	terminals := []string{"D", "B", "A", "C"}
	recomputed, err := mst.SteinerTree(graph, terminals,
		mst.WithSteinerPathStorage(mst.SteinerRecomputePaths))
	if err != nil {
		fmt.Println("recompute:", err)
		return
	}
	retained, err := mst.SteinerTree(graph, terminals,
		mst.WithSteinerPathStorage(mst.SteinerRetainPaths))
	if err != nil {
		fmt.Println("retain:", err)
		return
	}
	fmt.Printf("recompute=%.0f retain=%.0f same=%t\n",
		recomputed.TotalWeight, retained.TotalWeight, slices.Equal(recomputed.Edges, retained.Edges))
	// Output:
	// recompute=8 retain=8 same=true
}

// ExampleSteinerWalk_closedFacilityAudit demonstrates the actual traversal of a supplied tree.
// All original edges, including branches ending at nonterminal maintenance locations, are covered.
// Implementation:
//   - Stage 1: Prepare a valid caller-owned star tree with reversed input edge order.
//   - Stage 2: Request an explicit closed walk from the facility hub to itself.
//   - Stage 3: Check actual passage count independently from unique network cost.
//
// Behavior highlights:
//   - A closed walk traverses every tree edge twice, including zero/branch edges when present.
//
// Inputs:
//   - A twelve-link inspection network with valid terminals and nonterminal leaves.
//
// Returns:
//   - Complete route length and passage count; the input tree remains unchanged.
//
// Errors:
//   - Invalid mutable result or overflowing travel cost prevents walk publication.
//
// Determinism:
//   - Walk adjacency is canonicalized independently of caller edge order.
//
// Complexity:
//   - Validation/sorting O(V log V+E log E); traversal O(E) steps and O(V+E) memory.
//
// Notes:
//   - This is optimal only for covering the given tree, not a route through external shortcuts.
//
// AI-Hints:
//   - Nonterminal leaves remain necessary if the requested walk must cover every supplied edge.
func ExampleSteinerWalk_closedFacilityAudit() {
	const inspectionLinks = 12
	vertices := []string{"Hub"}
	edges := make([]core.Edge, 0, inspectionLinks)
	for index := inspectionLinks; index >= 1; index-- {
		vertex := fmt.Sprintf("Station%02d", index)
		vertices = append(vertices, vertex)
		edges = append(edges, core.Edge{
			ID: fmt.Sprintf("cable-%02d", index), From: "Hub", To: vertex, Weight: 1,
		})
	}
	// The public result is intentionally supplied in noncanonical order.
	tree := &mst.SteinerResult{
		Terminals:   []string{"Station01", "Hub", "Station02"},
		Vertices:    vertices,
		Edges:       edges,
		TotalWeight: inspectionLinks,
	}
	walk, err := mst.SteinerWalk(tree, "Hub", "Hub")
	if err != nil {
		fmt.Println("walk:", err)
		return
	}
	fmt.Printf("network=%.0f distance=%.0f passages=%d\n",
		walk.NetworkWeight, walk.Distance, len(walk.EdgeIDs))
	// Output:
	// network=12 distance=24 passages=24
}

// ExampleSteinerWalk_fixedTerminalInspection demonstrates KMB-to-walk composition.
// A fixed endpoint avoids retracing the unique tree spine while still inspecting every branch.
// Implementation:
//   - Stage 1: Construct an original-link network with expensive unused alternatives.
//   - Stage 2: Run SteinerTree for three required facilities.
//   - Stage 3: Run SteinerWalk from A to B and publish each original passage in order.
//
// Behavior highlights:
//   - The off-spine C branch is visited and returned; the A-to-B spine is traversed once.
//
// Inputs:
//   - Three required facilities and a weighted candidate network of twelve links.
//
// Returns:
//   - Actual vertex itinerary, repeated original EdgeIDs, and separate network/walk costs.
//
// Errors:
//   - Failures from graph construction, KMB, and mutable-tree verification are not suppressed.
//
// Determinism:
//   - The selected original edges and walk branch order are sorted by ID/neighbor.
//
// Complexity:
//   - KMB O((k-1)Tsssp+closure+union); walk O(V log V+E log E+S).
//
// Notes:
//   - No original-graph shortcut is introduced in the edge-covering walk.
//
// AI-Hints:
//   - NetworkWeight counts three unique spokes; Distance counts four real passages.
func ExampleSteinerWalk_fixedTerminalInspection() {
	graph, err := core.NewGraph(core.WithWeighted())
	if err != nil {
		fmt.Println("graph:", err)
		return
	}
	for _, candidate := range []struct {
		id, from, to string
		cost         float64
	}{
		{"AX", "A", "X", 1}, {"BX", "B", "X", 1}, {"CX", "C", "X", 1},
		{"AB", "A", "B", 3}, {"BC", "B", "C", 3}, {"AC", "A", "C", 3},
		{"AY", "A", "Y", 10}, {"BY", "B", "Y", 10},
		{"CY", "C", "Y", 10}, {"XY", "X", "Y", 10},
		{"YZ", "Y", "Z", 10}, {"BZ", "B", "Z", 10},
	} {
		if _, err := graph.AddEdge(candidate.from, candidate.to, candidate.cost, core.WithID(candidate.id)); err != nil {
			fmt.Println("edge:", err)
			return
		}
	}
	network, err := mst.SteinerTree(graph, []string{"C", "B", "A"})
	if err != nil {
		fmt.Println("steiner:", err)
		return
	}
	walk, err := mst.SteinerWalk(network, "A", "B")
	if err != nil {
		fmt.Println("walk:", err)
		return
	}
	fmt.Printf("network=%.0f distance=%.0f vertices=%v edges=%v\n",
		walk.NetworkWeight, walk.Distance, walk.Vertices, walk.EdgeIDs)
	// Output:
	// network=3 distance=4 vertices=[A X C X B] edges=[AX CX CX BX]
}
