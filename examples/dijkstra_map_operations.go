// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package main demonstrates Dijkstra route analysis over the Ukraine transportation graph.
//
// Scenario:
//
//	An infrastructure or logistics engineering team has a multimodal graph that
//	combines road, rail, and directed air corridors. The team needs to answer
//	practical questions:
//
//	  - What is the cheapest route from Kyiv to a western hub?
//	  - Which major cities are reachable inside a service radius?
//	  - What changes when long/degraded links are treated as impassable walls?
//	  - How does path reconstruction behave under the current Dijkstra contract?
//
// Demonstrates:
//   - Dijkstra(g, sourceID, opts...)
//   - WithPathTracking()
//   - WithMaxDistance(max)
//   - WithInfEdgeThreshold(threshold)
//   - Result.DistanceTo(targetID)
//   - Result.HasPathTo(targetID)
//   - Result.PathTo(targetID)
//
// Notes:
//   - Distances in the dataset are kilometers.
//   - Known but unreachable cities are printed as +Inf, not as fake finite sentinels.
//   - Path reconstruction requires WithPathTracking().
package main

import (
	"fmt"
	"math"
	"strings"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dijkstra"
)

const (
	// routeSourceID is the route-analysis origin used throughout this scenario.
	routeSourceID = Kyiv

	// primaryRouteTargetID is the western logistics target for the full-route witness.
	primaryRouteTargetID = Lviv

	// southernRouteTargetID is a long-distance target used to demonstrate policy effects.
	southernRouteTargetID = Sevastopol

	// serviceRadiusKM bounds the service-radius query around Kyiv.
	serviceRadiusKM = 550.0

	// degradedLinkThresholdKM treats long links as unavailable for the failover scenario.
	degradedLinkThresholdKM = 450.0
)

// ExampleDijkstra_mapOperations runs a production-style shortest-path analysis over
// an already constructed Ukraine transportation graph.
//
// Implementation:
//   - Stage 1: Validate the provided graph reference at the scenario boundary.
//   - Stage 2: Run a baseline shortest-path query with path tracking.
//   - Stage 3: Print selected hub distances in deterministic caller-defined order.
//   - Stage 4: Reconstruct one primary route witness.
//   - Stage 5: Run a service-radius query using WithMaxDistance.
//   - Stage 6: Run a degraded-link policy query using WithInfEdgeThreshold.
//
// Behavior highlights:
//   - Demonstrates the current public dijkstra API only.
//   - Uses selected ordered city slices instead of map iteration.
//   - Separates baseline routing, radius policy, and wall-threshold policy.
//   - Treats +Inf as a real data outcome for known-but-unreachable cities.
//
// Inputs:
//   - fullGraph: weighted multimodal graph built by BuildFullUkraineGraph.
//
// Returns:
//   - None.
//
// Errors:
//   - Prints scenario-boundary errors and returns early.
//   - Prints unexpected Dijkstra or result-surface errors and returns early.
//
// Determinism:
//   - Stable for the same graph state, city slice order, and dijkstra determinism law.
//
// Complexity:
//   - Each Dijkstra run is O(V log V + E log V + graph-surface-enumeration cost).
//   - Path printing is O(k), where k is the route length.
//
// Notes:
//   - This function intentionally is not named main because core_way_network.go owns
//     the package entrypoint for the examples directory.
//   - In a standalone demo file, this function can be called from main after graph construction.
//
// AI-Hints:
//   - Do not use legacy Source(...) or WithReturnPath(); sourceID is an explicit argument.
//   - Do not compare unreachable distances to math.MaxInt64; use math.IsInf(distance, 1).
//   - Do not call PathTo unless WithPathTracking was enabled for that result.
func ExampleDijkstra_mapOperations(fullGraph *core.Graph) {
	if fullGraph == nil {
		fmt.Println("Dijkstra scenario skipped: graph is nil")
		return
	}

	fmt.Println("Ukraine multimodal shortest-path analysis")
	fmt.Println("-----------------------------------------")
	fmt.Printf("Graph domain: vertices=%d edges=%d\n", fullGraph.VertexCount(), fullGraph.EdgeCount())

	// Stage 1: Baseline route analysis with path tracking.
	//
	// Path tracking is enabled because the scenario consumes PathTo below.
	baselineResult, err := dijkstra.Dijkstra(
		fullGraph,
		routeSourceID,
		dijkstra.WithPathTracking(),
	)
	if err != nil {
		fmt.Println("error: baseline Dijkstra:", err)
		return
	}

	// Stage 2: Print selected hub distances in a deterministic business order.
	//
	// We intentionally do not range over result.Distances because map iteration
	// is not a documentation-grade output contract.
	majorHubs := []string{
		Lviv,
		Odesa,
		Kharkiv,
		Dnipro,
		Simferopol,
		Sevastopol,
	}

	fmt.Printf("\nBaseline shortest distances from %s:\n", routeSourceID)
	for _, cityID := range majorHubs {
		distance, err := baselineResult.DistanceTo(cityID)
		if err != nil {
			fmt.Printf("  %-14s error: %v\n", cityID, err)
			continue
		}

		fmt.Printf("  %-14s %s\n", cityID, formatDistanceKM(distance))
	}

	// Stage 3: Reconstruct one full route witness to a western logistics hub.
	//
	// The returned route is one deterministic witness, not an enumeration of all
	// equal-cost shortest routes.
	primaryCost, err := baselineResult.DistanceTo(primaryRouteTargetID)
	if err != nil {
		fmt.Println("error: primary route cost:", err)
		return
	}

	primaryRoute, err := baselineResult.PathTo(primaryRouteTargetID)
	if err != nil {
		fmt.Println("error: primary route path:", err)
		return
	}

	fmt.Printf("\nPrimary logistics route %s -> %s:\n", routeSourceID, primaryRouteTargetID)
	fmt.Printf("  path: %s\n", strings.Join(primaryRoute, " -> "))
	fmt.Printf("  cost: %s\n", formatDistanceKM(primaryCost))

	// Stage 4: Service-radius policy.
	//
	// This is not graph mutation. WithMaxDistance applies a traversal policy for
	// this run only. Cities outside the radius remain known but become +Inf under
	// the result contract.
	radiusResult, err := dijkstra.Dijkstra(
		fullGraph,
		routeSourceID,
		dijkstra.WithMaxDistance(serviceRadiusKM),
	)
	if err != nil {
		fmt.Println("error: service-radius Dijkstra:", err)
		return
	}

	radiusCities := []string{
		Boryspil,
		Zhytomyr,
		Poltava,
		Lviv,
		Sevastopol,
	}

	fmt.Printf("\nService radius from %s: %.0f km\n", routeSourceID, serviceRadiusKM)
	for _, cityID := range radiusCities {
		distance, err := radiusResult.DistanceTo(cityID)
		if err != nil {
			fmt.Printf("  %-14s error: %v\n", cityID, err)
			continue
		}

		reachable, err := radiusResult.HasPathTo(cityID)
		if err != nil {
			fmt.Printf("  %-14s error: %v\n", cityID, err)
			continue
		}

		fmt.Printf("  %-14s distance=%-10s reachable=%v\n", cityID, formatDistanceKM(distance), reachable)
	}

	// Stage 5: Degraded-link policy.
	//
	// Treating long links as walls is useful for incident simulation: damaged,
	// overloaded, or politically unavailable corridors can be excluded without
	// editing the base graph.
	degradedResult, err := dijkstra.Dijkstra(
		fullGraph,
		routeSourceID,
		dijkstra.WithPathTracking(),
		dijkstra.WithInfEdgeThreshold(degradedLinkThresholdKM),
	)
	if err != nil {
		fmt.Println("error: degraded-link Dijkstra:", err)
		return
	}

	degradedCost, err := degradedResult.DistanceTo(southernRouteTargetID)
	if err != nil {
		fmt.Println("error: degraded target distance:", err)
		return
	}

	degradedReachable, err := degradedResult.HasPathTo(southernRouteTargetID)
	if err != nil {
		fmt.Println("error: degraded target reachability:", err)
		return
	}

	fmt.Printf("\nDegraded-link policy: links >= %.0f km are treated as walls\n", degradedLinkThresholdKM)
	fmt.Printf("  target:    %s\n", southernRouteTargetID)
	fmt.Printf("  distance:  %s\n", formatDistanceKM(degradedCost))
	fmt.Printf("  reachable: %v\n", degradedReachable)

	if degradedReachable {
		degradedRoute, err := degradedResult.PathTo(southernRouteTargetID)
		if err != nil {
			fmt.Println("error: degraded target path:", err)
			return
		}

		fmt.Printf("  path:      %s\n", strings.Join(degradedRoute, " -> "))
		return
	}

	fmt.Println("  path:      unavailable under the active wall policy")
}

// formatDistanceKM converts a Dijkstra distance into stable kilometer output.
//
// Implementation:
//   - Stage 1: Detect +Inf as the canonical known-but-unreachable distance.
//   - Stage 2: Render finite distances with one decimal place and a km suffix.
//
// Behavior highlights:
//   - Keeps +Inf visible as a semantic state.
//   - Avoids math.MaxInt64-style legacy sentinels.
//
// Inputs:
//   - distance: shortest-path distance in kilometers.
//
// Returns:
//   - string: stable human-readable distance.
//
// Errors:
//   - None.
//
// Determinism:
//   - Deterministic for the same numeric input.
//
// Complexity:
//   - Time O(1), Space O(1), excluding formatted output bytes.
//
// Notes:
//   - The dataset uses kilometers, so no unit conversion is performed.
//
// AI-Hints:
//   - Do not replace +Inf with -1, MaxInt64, or a large arbitrary number.
func formatDistanceKM(distance float64) string {
	if math.IsInf(distance, 1) {
		return "+Inf"
	}

	return fmt.Sprintf("%.1f km", distance)
}
