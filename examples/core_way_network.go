// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package main demonstrates lvlath/core graph construction over a predefined
// Ukraine transportation dataset.
//
// Scenario:
//
//	A platform team wants one deterministic in-memory transportation graph that
//
// can support both topology inspection and downstream shortest-path analysis.
//
//	The example builds a road-only graph, then a multimodal graph containing road,
//	rail, and air links. It demonstrates graph size inspection, deterministic
//	neighbor listing, empty topology cloning, policy filtering, and then delegates
//	to the Dijkstra route-analysis scenario.
//
// Demonstrates:
//   - Weighted road graph construction.
//   - Weighted mixed multi-edge multimodal graph construction.
//   - Deterministic neighborhood inspection through core.Graph.Neighbors.
//   - CloneEmpty for preserving the vertex domain without edges.
//   - FilterEdges for policy-style graph reduction in a copied/demo graph.
//   - Safe handoff to the Dijkstra shortest-path scenario.
//
// Playground: https://go.dev/play/p/ios0HUK5fry
//
// Notes:
//   - The dataset in ukrainian_map_data.go is deterministic and bundled with
//     this example package.
//   - AddEdge errors are intentionally ignored only while loading this trusted
//     static demo dataset. In production ingestion pipelines, always handle
//     AddEdge errors explicitly and fail fast with useful context.
package main

import (
	"fmt"

	"github.com/lvlath/go/core"
)

const (
	// CAPITAL is the central source city used by both topology and shortest-path examples.
	CAPITAL = Kyiv

	// edgeAuditLimitKM is the maximum edge length retained by the filtering demonstration.
	edgeAuditLimitKM = 300.0

	// neighborPreviewLimit bounds terminal output while still proving deterministic ordering.
	neighborPreviewLimit = 12
)

// WaySegment describes one weighted transportation link between two cities.
//
// Implementation:
//   - Stage 1: Stores endpoint identifiers from the predefined city constants.
//   - Stage 2: Stores the link length in kilometers.
//
// Behavior highlights:
//   - This is a plain dataset carrier used by RoadNetwork, RailwayNetwork, and AirNetwork.
//   - Direction is not stored here; graph insertion policy decides whether a segment is directed.
//
// Inputs:
//   - From: source city identifier.
//   - To: destination city identifier.
//   - KM: link length in kilometers.
//
// Returns:
//   - N/A.
//
// Errors:
//   - N/A.
//
// Determinism:
//   - Deterministic when dataset slices are iterated in their declared order.
//
// Complexity:
//   - Field access is O(1).
//
// Notes:
//   - Air routes are inserted as directed edges in BuildFullUkraineGraph.
//   - Road and rail routes are inserted with the graph default, which is undirected.
//
// AI-Hints:
//   - Do not attach graph-mode semantics to this struct; graph construction owns that policy.
type WaySegment struct {
	From string
	To   string
	KM   float64
}

// ExampleCore_wayNetwork runs the Ukraine transportation topology scenario and then delegates to
// the Dijkstra shortest-path scenario.
//
// Implementation:
//   - Stage 1: Build and inspect the road-only graph.
//   - Stage 2: Demonstrate CloneEmpty and FilterEdges on road topology.
//   - Stage 3: Build and inspect the multimodal graph.
//   - Stage 4: Demonstrate CloneEmpty and FilterEdges on multimodal topology.
//   - Stage 5: Pass a fresh multimodal graph into the Dijkstra scenario.
//
// Behavior highlights:
//   - Keeps graph construction deterministic.
//   - Uses bounded neighbor previews to keep example output readable.
//   - Avoids mutating the graph that is passed into the Dijkstra scenario.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - This example uses a trusted static dataset; AddEdge errors are ignored only
//     inside the dataset-loading builders.
//   - Neighbor lookup errors are printed and terminate the scenario early.
//
// Determinism:
//   - Stable for the same dataset and core.Graph ordering contract.
//
// Complexity:
//   - Road graph construction is O(|RoadNetwork|).
//   - Full graph construction is O(|RoadNetwork| + |RailwayNetwork| + |AirNetwork|).
//   - Neighbors(CAPITAL) is governed by the core.Graph.Neighbors complexity.
//   - FilterEdges is O(E + cleanup cost) over the graph being filtered.
//
// Notes:
//   - Filtering is shown on demo graph instances; do not filter the graph you
//     intend to reuse for the Dijkstra scenario unless that mutation is desired.
//
// AI-Hints:
//   - Keep this example focused on core topology operations.
//   - Route analysis belongs in dijkstra_map_operations.go.
func ExampleCore_wayNetwork() {
	// [Run on Go Playground](https://go.dev/play/p/x1RZBJkPw7W)
	fmt.Println("Ukraine transportation topology audit")
	fmt.Println("-------------------------------------")

	roadGraph := BuildUkraineRoads()
	fmt.Printf("RoadGraph: vertices=%d edges=%d\n", roadGraph.VertexCount(), roadGraph.EdgeCount())

	roadNeighbors, err := roadGraph.Neighbors(CAPITAL)
	if err != nil {
		fmt.Printf("error: neighbors(%s): %v\n", CAPITAL, err)
		return
	}

	fmt.Printf("Road neighbors of %s: total=%d preview=%d\n", CAPITAL, len(roadNeighbors), minInt(len(roadNeighbors), neighborPreviewLimit))
	for index, edge := range roadNeighbors {
		if index >= neighborPreviewLimit {
			break
		}

		otherEndpoint := edge.To
		if otherEndpoint == CAPITAL {
			otherEndpoint = edge.From
		}

		fmt.Printf("  %02d. %-20s %.1f km\n", index+1, otherEndpoint, edge.Weight)
	}

	emptyRoadDomain := roadGraph.CloneEmpty()
	fmt.Printf("Road CloneEmpty: vertices=%d edges=%d\n", emptyRoadDomain.VertexCount(), emptyRoadDomain.EdgeCount())

	roadAuditGraph := BuildUkraineRoads()
	_, err = roadAuditGraph.FilterEdges(func(edge core.Edge) bool {
		return edge.Weight <= edgeAuditLimitKM
	})
	if err != nil {
		fmt.Printf("error: FilterEdges: %v\n", err)
		return
	}
	fmt.Printf("RoadGraph after keeping links <= %.0f km: edges=%d\n", edgeAuditLimitKM, roadAuditGraph.EdgeCount())

	fullGraphForAudit := BuildFullUkraineGraph()
	fmt.Printf("\nFullGraph: vertices=%d edges=%d\n", fullGraphForAudit.VertexCount(), fullGraphForAudit.EdgeCount())

	fullNeighbors, err := fullGraphForAudit.Neighbors(CAPITAL)
	if err != nil {
		fmt.Printf("error: neighbors(%s): %v\n", CAPITAL, err)
		return
	}

	fmt.Printf("FullGraph neighbors of %s: total=%d preview=%d\n", CAPITAL, len(fullNeighbors), minInt(len(fullNeighbors), neighborPreviewLimit))
	for index, edge := range fullNeighbors {
		if index >= neighborPreviewLimit {
			break
		}

		otherEndpoint := edge.To
		if otherEndpoint == CAPITAL {
			otherEndpoint = edge.From
		}

		fmt.Printf("  %02d. %-20s %.1f km\n", index+1, otherEndpoint, edge.Weight)
	}

	emptyFullDomain := fullGraphForAudit.CloneEmpty()
	fmt.Printf("FullGraph CloneEmpty: vertices=%d edges=%d\n", emptyFullDomain.VertexCount(), emptyFullDomain.EdgeCount())

	fullAuditGraph := BuildFullUkraineGraph()
	_, err = fullAuditGraph.FilterEdges(func(edge core.Edge) bool {
		return edge.Weight <= edgeAuditLimitKM
	})
	if err != nil {
		fmt.Printf("error: FilterEdges: %v\n", err)
	}
	fmt.Printf("FullGraph after keeping links <= %.0f km: edges=%d\n", edgeAuditLimitKM, fullAuditGraph.EdgeCount())

	//fmt.Println()
	//ExampleDijkstra_mapOperations(BuildFullUkraineGraph())
}

// BuildUkraineRoads constructs the road-only Ukraine graph.
//
// Implementation:
//   - Stage 1: Create a weighted graph with the default undirected edge policy.
//   - Stage 2: Insert every predefined road segment from RoadNetwork.
//   - Stage 3: Return the constructed graph.
//
// Behavior highlights:
//   - Roads are modeled as undirected weighted links.
//   - Vertices are created implicitly by AddEdge.
//   - The trusted static dataset is expected to satisfy the graph policy.
//
// Inputs:
//   - None.
//
// Returns:
//   - *core.Graph: weighted road graph.
//
// Errors:
//   - AddEdge errors are intentionally ignored only for this trusted static example dataset.
//
// Determinism:
//   - Deterministic for the same RoadNetwork slice order and core.Graph insertion semantics.
//
// Complexity:
//   - Time O(|RoadNetwork|), excluding graph internal ordering costs.
//   - Space O(V_road + E_road).
//
// Notes:
//   - Production data loaders must validate every AddEdge error.
//
// AI-Hints:
//   - Do not copy this ignored-error ingestion pattern into untrusted data pipelines.
func BuildUkraineRoads() *core.Graph {
	graph, _ := core.NewGraph(core.WithWeighted())

	for _, segment := range RoadNetwork {
		_, _ = graph.AddEdge(segment.From, segment.To, segment.KM)
	}

	return graph
}

// BuildFullUkraineGraph constructs the multimodal Ukraine transportation graph.
//
// Implementation:
//   - Stage 1: Create a weighted graph with mixed-edge and multi-edge support.
//   - Stage 2: Insert road segments as undirected links.
//   - Stage 3: Insert rail segments as undirected parallel-capable links.
//   - Stage 4: Insert air segments as directed links.
//
// Behavior highlights:
//   - Multi-edge support preserves distinct transport links between the same city pair.
//   - Mixed-edge support allows directed air corridors alongside undirected ground links.
//   - The graph remains a single route-analysis domain for downstream algorithms.
//
// Inputs:
//   - None.
//
// Returns:
//   - *core.Graph: weighted multimodal graph.
//
// Errors:
//   - AddEdge errors are intentionally ignored only for this trusted static example dataset.
//
// Determinism:
//   - Deterministic for the same dataset slice order and core.Graph insertion semantics.
//
// Complexity:
//   - Time O(|RoadNetwork| + |RailwayNetwork| + |AirNetwork|), excluding graph internal ordering costs.
//   - Space O(V_total + E_total).
//
// Notes:
//   - Real ingestion should record the mode as metadata or a parallel domain table
//     if callers need mode-specific route explanation.
//
// AI-Hints:
//   - WithMixedEdges is required because air links are inserted as directed while
//     road and rail links remain undirected.
func BuildFullUkraineGraph() *core.Graph {
	graph, _ := core.NewGraph(core.WithWeighted(), core.WithMixedEdges(), core.WithMultiEdges())

	for _, segment := range RoadNetwork {
		_, _ = graph.AddEdge(segment.From, segment.To, segment.KM)
	}

	for _, segment := range RailwayNetwork {
		_, _ = graph.AddEdge(segment.From, segment.To, segment.KM)
	}

	for _, segment := range AirNetwork {
		_, _ = graph.AddEdge(segment.From, segment.To, segment.KM, core.WithEdgeDirected(true))
	}

	return graph
}

// minInt returns the smaller of two integers.
//
// Implementation:
//   - Stage 1: Compare both values directly.
//   - Stage 2: Return the smaller value.
//
// Behavior highlights:
//   - Used only to keep example output bounded and readable.
//
// Inputs:
//   - left: first integer.
//   - right: second integer.
//
// Returns:
//   - int: smaller input value.
//
// Errors:
//   - None.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper is intentionally tiny but prevents repeated inline conditional output logic.
//
// AI-Hints:
//   - Keep preview bounds explicit in examples that inspect large deterministic datasets.
func minInt(left, right int) int {
	if left < right {
		return left
	}

	return right
}
