// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package mst_test

import (
	"errors"
	"fmt"
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
