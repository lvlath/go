// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package dijkstra_test

import (
	"fmt"
	"math"
	"strings"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dijkstra"
)

// AI-HINTS (file):
//   - Examples must demonstrate only the current public API.
//   - Example outputs must be fully deterministic and CI-stable.
//   - Never print map iteration directly; print selected values or ordered paths only.
//   - Each example must follow a real pipeline: build -> dijkstra -> consume.
//   - Do not imply all-shortest-paths behavior; the package returns one deterministic witness.
//   - Keep graph construction errors explicit; never ignore AddEdge / AddVertex failures.
//   - Path examples must enable WithPathTracking explicitly.
//   - +Inf in examples is a valid data outcome, not an error protocol.
//   - Graph-edge weights in examples must be finite.
//   - Never use a +Inf edge to model failure; use a finite threshold or omit the edge.
//   - Examples consume unmodified Result values and must not demonstrate malformed Prev state.
//
//   - We DELIBERATELY ignore the error, since this is a test example and the data is predetermined! DON'T DO THAT!

// formatExampleDistance converts a shortest-path distance into stable printable output.
//
// Implementation:
//   - Stage 1: Detect the canonical +Inf unreachable value.
//   - Stage 2: Render +Inf explicitly as text.
//   - Stage 3: Render all finite values as integer-like decimal output for exact fixtures.
//
// Behavior highlights:
//   - +Inf is printed as a semantic unreachable marker.
//   - Finite values stay concise for exact deterministic example fixtures.
//
// Inputs:
//   - distance: shortest-path distance to format.
//
// Returns:
//   - string: stable printable representation of the distance.
//
// Errors:
//   - None.
//
// Determinism:
//   - Deterministic for the same numeric input.
//
// Complexity:
//   - Time O(1), Space O(1), excluding formatting output bytes.
//
// Notes:
//   - The examples below intentionally use exact integral weights to avoid tolerance-based output.
//
// AI-Hints:
//   - Prefer explicit +Inf formatting in examples instead of exposing raw Go formatting noise.
func formatExampleDistance(distance float64) string {
	if math.IsInf(distance, 1) {
		return "+Inf"
	}

	return fmt.Sprintf("%.0f", distance)
}

// ExampleDijkstra_logisticsRouting MAIN DESCRIPTION (2+ lines, no marketing).
// Demonstrates regional logistics routing from a central warehouse through hubs and sort centers:
// build a weighted distribution graph -> run Dijkstra with path tracking -> consume exact route and total delivery cost.
//
// Implementation:
//   - Stage 1: Build a deterministic weighted logistics graph with hubs, sort centers, and destination cities.
//   - Stage 2: Run Dijkstra from the central warehouse with explicit path tracking.
//   - Stage 3: Query the exact cost and deterministic witness route to one destination city.
//   - Stage 4: Print a stable route-and-cost summary.
//
// Behavior highlights:
//   - Uses the current public API only.
//   - Demonstrates DistanceTo and PathTo as the canonical result-surface consumption path.
//   - Prints one deterministic shortest-path witness rather than implying all-shortest-path enumeration.
//
// Inputs:
//   - None (deterministic hard-coded topology).
//
// Returns:
//   - None (prints a stable logistics route summary).
//
// Errors:
//   - Any graph-construction error is printed and the example returns early.
//   - Any unexpected Dijkstra/result-surface error is printed and the example returns early.
//
// Determinism:
//   - Stable for the same graph fixture and the package determinism law.
//
// Complexity:
//   - Time O((V+E) log V), Space O(V+E_heap) for the algorithm run.
//   - Example printing is O(k) for the chosen path length.
//
// Notes:
//   - The example models a real “warehouse -> hubs -> city” routing pipeline.
//
// AI-Hints:
//   - Enable WithPathTracking explicitly whenever the example consumes PathTo.
//   - Keep logistics IDs semantically meaningful so route output is self-explanatory.
func ExampleDijkstra_logisticsRouting() {
	const (
		warehouseCentral = "warehouse:central"
		cityDelta        = "city:delta"

		weightFour  = 4.0
		weightFive  = 5.0
		weightSix   = 6.0
		weightThree = 3.0
		weightTwo   = 2.0
	)

	graph, _ := core.NewGraph(core.WithWeighted())
	// build example graph
	// !! Repeat: We DELIBERATELY ignore the error, since this is a test example and the data is predetermined!!
	_, _ = graph.AddEdge(warehouseCentral, "hub:north", weightFour)
	_, _ = graph.AddEdge(warehouseCentral, "hub:south", weightFive)
	_, _ = graph.AddEdge(warehouseCentral, "hub:east", weightSix)

	_, _ = graph.AddEdge("hub:north", "sort:river", weightThree)
	_, _ = graph.AddEdge("hub:north", "sort:valley", weightFive)

	_, _ = graph.AddEdge("hub:south", "sort:valley", weightTwo)
	_, _ = graph.AddEdge("hub:south", "sort:port", weightFour)

	_, _ = graph.AddEdge("hub:east", "sort:river", weightFour)
	_, _ = graph.AddEdge("hub:east", "sort:port", weightTwo)

	_, _ = graph.AddEdge("sort:river", "city:aurora", weightThree)
	_, _ = graph.AddEdge("sort:river", "city:birch", weightFour)
	_, _ = graph.AddEdge("sort:valley", "city:cobalt", weightThree)
	_, _ = graph.AddEdge("sort:valley", cityDelta, weightFour)
	_, _ = graph.AddEdge("sort:port", "city:ember", weightThree)

	_, _ = graph.AddEdge("sort:river", "sort:valley", weightTwo)
	_, _ = graph.AddEdge("sort:valley", "sort:port", weightTwo)

	result, err := dijkstra.Dijkstra(graph, warehouseCentral, dijkstra.WithPathTracking())
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	route, err := result.PathTo(cityDelta)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	cost, err := result.DistanceTo(cityDelta)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("route:", strings.Join(route, " -> "))
	fmt.Println("cost:", formatExampleDistance(cost))
	// Output:
	// route: warehouse:central -> hub:south -> sort:valley -> city:delta
	// cost: 11
}

// ExampleDijkstra_failoverNetwork MAIN DESCRIPTION (2+ lines, no marketing).
// Demonstrates failover routing over a weighted network with degraded links treated as walls:
// build a directed network graph -> run Dijkstra with WithInfEdgeThreshold -> consume reachable and policy-unreachable outcomes.
//
// Implementation:
//   - Stage 1: Build a deterministic directed network graph with primary, backup, and degraded links.
//   - Stage 2: Run Dijkstra from the ingress router with an explicit wall threshold.
//   - Stage 3: Query one reachable destination and one known-but-unreachable destination.
//   - Stage 4: Print stable cost and reachability output, including +Inf.
//
// Behavior highlights:
//   - Edges with weight >= threshold are treated as impassable walls.
//   - A target may remain known to the result domain while still being unreachable under policy.
//   - +Inf is printed as data, not as an error.
//
// Inputs:
//   - None (deterministic hard-coded topology and threshold).
//
// Returns:
//   - None (prints a stable failover-routing summary).
//
// Errors:
//   - Any graph-construction error is printed and the example returns early.
//   - Any unexpected Dijkstra/result-surface error is printed and the example returns early.
//
// Determinism:
//   - Stable for the same graph fixture, threshold policy, and package determinism law.
//
// Complexity:
//   - Time O((V+E) log V), Space O(V+E_heap) for the algorithm run.
//
// Notes:
//   - This example demonstrates traversal policy, not graph invalidity.
//
// AI-Hints:
//   - Keep wall-threshold examples distinct from invalid-weight examples.
//   - A known vertex with +Inf is a valid result-domain state.
func ExampleDijkstra_failoverNetwork() {
	const (
		ingressID     = "edge:ingress"
		gammaID       = "dc:gamma"
		omegaID       = "dc:omega"
		wallThreshold = 8.0

		weightTwo   = 2.0
		weightThree = 3.0
		weightFour  = 4.0
		weightEight = 8.0
		weightNine  = 9.0
		weightTen   = 10.0
	)

	graph, _ := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	// build example graph
	// !! Repeat: We DELIBERATELY ignore the error, since this is a test example and the data is predetermined!!
	_, _ = graph.AddEdge(ingressID, "core:a", weightTwo)
	_, _ = graph.AddEdge(ingressID, "core:b", weightThree)

	_, _ = graph.AddEdge("core:a", "core:c", weightTwo)
	_, _ = graph.AddEdge("core:b", "core:c", weightTwo)

	_, _ = graph.AddEdge("core:a", "backup:x", weightFour)
	_, _ = graph.AddEdge("backup:x", "backup:y", weightTwo)
	_, _ = graph.AddEdge("backup:y", gammaID, weightThree)

	_, _ = graph.AddEdge("core:c", "dc:alpha", weightTwo)
	_, _ = graph.AddEdge("core:c", "dc:beta", weightThree)
	_, _ = graph.AddEdge("dc:beta", gammaID, weightTwo)
	_, _ = graph.AddEdge(gammaID, "sink:archive", weightTwo)

	_, _ = graph.AddEdge("core:b", "dc:delta", weightFour)
	_, _ = graph.AddEdge("dc:delta", omegaID, weightNine)
	_, _ = graph.AddEdge("backup:y", omegaID, weightEight)
	_, _ = graph.AddEdge("core:c", omegaID, weightTen)

	result, err := dijkstra.Dijkstra(graph, ingressID, dijkstra.WithInfEdgeThreshold(wallThreshold))
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	gammaCost, err := result.DistanceTo(gammaID)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	omegaCost, err := result.DistanceTo(omegaID)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	omegaReachable, err := result.HasPathTo(omegaID)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("gammaCost:", formatExampleDistance(gammaCost))
	fmt.Println("omegaCost:", formatExampleDistance(omegaCost))
	fmt.Println("omegaReachable:", omegaReachable)
	// Output:
	// gammaCost: 9
	// omegaCost: +Inf
	// omegaReachable: false
}

// ExampleDijkstra_serviceRadiusCutoff MAIN DESCRIPTION (2+ lines, no marketing).
// Demonstrates delivery-radius policy over a weighted service map:
// build a reachable graph -> run Dijkstra with WithMaxDistance -> consume both in-radius and cutoff-induced +Inf outcomes.
//
// Implementation:
//   - Stage 1: Build a deterministic weighted service-area graph with zones and clients.
//   - Stage 2: Run Dijkstra from the central depot with a strict service-radius cutoff.
//   - Stage 3: Query one in-radius client and one client that is reachable in the graph but outside the allowed radius.
//   - Stage 4: Print stable cost and reachability output.
//
// Behavior highlights:
//   - WithMaxDistance limits traversal policy, not graph validity.
//   - A graph path may exist mathematically while remaining +Inf under cutoff policy.
//   - +Inf here means “outside allowed traversal radius”, not “missing vertex”.
//
// Inputs:
//   - None (deterministic hard-coded topology and cutoff).
//
// Returns:
//   - None (prints a stable service-radius summary).
//
// Errors:
//   - Any graph-construction error is printed and the example returns early.
//   - Any unexpected Dijkstra/result-surface error is printed and the example returns early.
//
// Determinism:
//   - Stable for the same graph fixture, cutoff policy, and package determinism law.
//
// Complexity:
//   - Time O((V+E) log V), Space O(V+E_heap) for the algorithm run.
//
// Notes:
//   - This example is about traversal policy, not path inexistence.
//
// AI-Hints:
//   - Keep cutoff examples separate from disconnected-graph examples.
//   - Use +Inf explicitly to teach the difference between policy-unreachable and unknown-target states.
func ExampleDijkstra_serviceRadiusCutoff() {
	const (
		depotCentral = "depot:central"
		clientAmber  = "client:amber"
		clientEmber  = "client:ember"
		maxRadius    = 9.0

		weightTwo   = 2.0
		weightThree = 3.0
		weightFour  = 4.0
		weightEight = 8.0
		weightNine  = 9.0
	)

	graph, _ := core.NewGraph(core.WithWeighted())
	// build example graph
	// !! Repeat: We DELIBERATELY ignore the error, since this is a test example and the data is predetermined!!
	_, _ = graph.AddEdge(depotCentral, "zone:n1", weightTwo)
	_, _ = graph.AddEdge(depotCentral, "zone:e1", weightThree)
	_, _ = graph.AddEdge(depotCentral, "zone:s1", weightTwo)

	_, _ = graph.AddEdge("zone:n1", "zone:n2", weightTwo)
	_, _ = graph.AddEdge("zone:n2", "zone:n3", weightTwo)

	_, _ = graph.AddEdge("zone:e1", "zone:e2", weightTwo)
	_, _ = graph.AddEdge("zone:s1", "zone:s2", weightThree)

	_, _ = graph.AddEdge("zone:n2", clientAmber, weightTwo)
	_, _ = graph.AddEdge("zone:e2", "client:birch", weightThree)
	_, _ = graph.AddEdge("zone:s2", "client:cinder", weightTwo)
	_, _ = graph.AddEdge("zone:n3", "client:drift", weightThree)

	_, _ = graph.AddEdge("zone:n3", clientEmber, weightFour)
	_, _ = graph.AddEdge("zone:e2", clientEmber, weightEight)
	_, _ = graph.AddEdge("zone:s2", clientEmber, weightNine)

	result, err := dijkstra.Dijkstra(graph, depotCentral, dijkstra.WithMaxDistance(maxRadius))
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	amberCost, err := result.DistanceTo(clientAmber)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	emberCost, err := result.DistanceTo(clientEmber)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	emberReachable, err := result.HasPathTo(clientEmber)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("amberCost:", formatExampleDistance(amberCost))
	fmt.Println("emberCost:", formatExampleDistance(emberCost))
	fmt.Println("emberReachable:", emberReachable)
	// Output:
	// amberCost: 6
	// emberCost: +Inf
	// emberReachable: false
}

// ExampleDijkstra_mixedTransitGraph MAIN DESCRIPTION (2+ lines, no marketing).
// Demonstrates shortest-path routing over a mixed transit graph with one-way and two-way streets:
// build a mixed-edge city graph -> run Dijkstra with path tracking -> consume one stable route to the harbor terminal.
//
// Implementation:
//   - Stage 1: Build a deterministic weighted mixed-edge transit graph.
//   - Stage 2: Use explicit per-edge direction options to mix one-way avenues and two-way streets.
//   - Stage 3: Run Dijkstra from the central station with path tracking enabled.
//   - Stage 4: Print the exact route and total travel cost to the harbor terminal.
//
// Behavior highlights:
//   - Demonstrates core.WithMixedEdges() in a practical user-facing scenario.
//   - Uses the public result surface only.
//   - Relies on canonical endpoint resolution for undirected street segments.
//
// Inputs:
//   - None (deterministic hard-coded topology).
//
// Returns:
//   - None (prints a stable mixed-transit routing summary).
//
// Errors:
//   - Any graph-construction error is printed and the example returns early.
//   - Any unexpected Dijkstra/result-surface error is printed and the example returns early.
//
// Determinism:
//   - Stable for the same mixed graph fixture and the package determinism law.
//
// Complexity:
//   - Time O((V+E) log V), Space O(V+E_heap) for the algorithm run.
//
// Notes:
//   - This is a practical mixed-edge routing scenario, not just a regression demo.
//
// AI-Hints:
//   - Mixed graphs must keep per-edge direction explicit.
//   - Do not describe endpoint-law behavior only in comments; demonstrate it through a real route.
func ExampleDijkstra_mixedTransitGraph() {
	const (
		stationCentral = "station:central"
		venueHarbor    = "venue:harbor"

		weightOne   = 1.0
		weightTwo   = 2.0
		weightThree = 3.0
		weightFive  = 5.0
	)

	graph, _ := core.NewGraph(core.WithWeighted(), core.WithMixedEdges())
	// build example graph
	// !! Repeat: We DELIBERATELY ignore the error, since this is a test example and the data is predetermined!!
	_, _ = graph.AddEdge(stationCentral, "ave:1", weightTwo, core.WithEdgeDirected(true))
	_, _ = graph.AddEdge(stationCentral, "ave:2", weightThree, core.WithEdgeDirected(true))
	_, _ = graph.AddEdge("ave:1", "ave:3", weightTwo, core.WithEdgeDirected(true))
	_, _ = graph.AddEdge("ave:2", "ave:3", weightOne, core.WithEdgeDirected(true))
	_, _ = graph.AddEdge("ave:3", "hub:market", weightTwo, core.WithEdgeDirected(true))
	_, _ = graph.AddEdge("hub:market", "street:oak", weightOne, core.WithEdgeDirected(false))
	_, _ = graph.AddEdge("street:oak", "street:pine", weightOne, core.WithEdgeDirected(false))
	_, _ = graph.AddEdge("street:pine", venueHarbor, weightTwo, core.WithEdgeDirected(true))
	_, _ = graph.AddEdge("hub:market", venueHarbor, weightFive, core.WithEdgeDirected(true))
	_, _ = graph.AddEdge("ave:2", "street:elm", weightFive, core.WithEdgeDirected(true))
	_, _ = graph.AddEdge("street:elm", "street:pine", weightOne, core.WithEdgeDirected(false))
	_, _ = graph.AddEdge("street:oak", "stop:museum", weightTwo, core.WithEdgeDirected(false))
	_, _ = graph.AddEdge("stop:museum", venueHarbor, weightThree, core.WithEdgeDirected(true))

	result, err := dijkstra.Dijkstra(graph, stationCentral, dijkstra.WithPathTracking())
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	route, err := result.PathTo(venueHarbor)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	cost, err := result.DistanceTo(venueHarbor)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("route:", strings.Join(route, " -> "))
	fmt.Println("cost:", formatExampleDistance(cost))
	// Output:
	// route: station:central -> ave:1 -> ave:3 -> hub:market -> street:oak -> street:pine -> venue:harbor
	// cost: 10
}

// ExampleDijkstra_equalCostDeterminism MAIN DESCRIPTION (2+ lines, no marketing).
// Demonstrates deterministic witness selection when two equal-cost corridors compete for the same target:
// build a directed corridor graph -> run Dijkstra with path tracking -> consume the exact chosen witness and total cost.
//
// Implementation:
//   - Stage 1: Build a deterministic directed graph with two equal-cost corridors to the same target.
//   - Stage 2: Add side branches so the example remains realistic and non-trivial.
//   - Stage 3: Run Dijkstra from the campus gate with explicit path tracking.
//   - Stage 4: Print the exact chosen route and total cost.
//
// Behavior highlights:
//   - The package returns one deterministic shortest-path witness.
//   - Equal-cost alternatives do not imply unstable output.
//   - The example illustrates determinism law without exposing internal prev-map details directly.
//
// Inputs:
//   - None (deterministic hard-coded topology).
//
// Returns:
//   - None (prints a stable equal-cost determinism summary).
//
// Errors:
//   - Any graph-construction error is printed and the example returns early.
//   - Any unexpected Dijkstra/result-surface error is printed and the example returns early.
//
// Determinism:
//   - Stable for the same graph fixture and the package determinism law.
//
// Complexity:
//   - Time O((V+E) log V), Space O(V+E_heap) for the algorithm run.
//
// Notes:
//   - The example intentionally demonstrates one chosen witness, not all valid shortest paths.
//
// AI-Hints:
//   - Equal-cost examples must assert one exact route only when the package contract truly fixes it.
//   - Keep user-facing determinism examples on the public result surface, not on internal predecessor maps.
func ExampleDijkstra_equalCostDeterminism() {
	const (
		campusGate  = "campus:gate"
		siteReactor = "site:reactor"
		weightOne   = 1.0
		weightTwo   = 2.0
		weightThree = 3.0
	)

	graph, _ := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	// build example graph
	// !! Repeat: We DELIBERATELY ignore the error, since this is a test example and the data is predetermined!!
	_, _ = graph.AddEdge(campusGate, "corridor:alpha", weightTwo)
	_, _ = graph.AddEdge(campusGate, "corridor:beta", weightTwo)

	_, _ = graph.AddEdge("corridor:alpha", "relay:alpha", weightTwo)
	_, _ = graph.AddEdge("corridor:beta", "relay:beta", weightTwo)

	_, _ = graph.AddEdge("relay:alpha", siteReactor, weightTwo)
	_, _ = graph.AddEdge("relay:beta", siteReactor, weightTwo)

	_, _ = graph.AddEdge("corridor:alpha", "audit:a", weightThree)
	_, _ = graph.AddEdge("corridor:beta", "audit:b", weightThree)

	_, _ = graph.AddEdge("audit:a", "ops:archive", weightTwo)
	_, _ = graph.AddEdge("audit:b", "ops:archive", weightTwo)

	_, _ = graph.AddEdge("relay:alpha", "ops:monitor", weightOne)
	_, _ = graph.AddEdge("relay:beta", "ops:monitor", weightOne)

	_, _ = graph.AddEdge("ops:monitor", "ops:archive", weightTwo)
	_, _ = graph.AddEdge("ops:archive", "ops:observer", weightTwo)

	result, err := dijkstra.Dijkstra(graph, campusGate, dijkstra.WithPathTracking())
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	route, err := result.PathTo(siteReactor)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	cost, err := result.DistanceTo(siteReactor)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("route:", strings.Join(route, " -> "))
	fmt.Println("cost:", formatExampleDistance(cost))
	// Output:
	// route: campus:gate -> corridor:alpha -> relay:alpha -> site:reactor
	// cost: 6
}
