// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package main demonstrates production-style medical drone dispatch planning
// with lvlath/tsp over a deterministic Euclidean distance matrix.
package main

import (
	"fmt"
	"log"
	"math"
	"strings"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

// ExampleTSP_ApproxDrones plans one closed medical-drone dispatch route over an
// emergency urban waypoint set.
//
// Scenario:
//
//	A medical drone starts at a hospital logistics roof and must visit multiple
//	time-sensitive drop points: trauma kits, blood-bank samples, dialysis supplies,
//	pharmacy payloads, and rooftop emergency landing zones. The drone must return
//	to the same base so the operator can swap battery packs, reload payload racks,
//	and keep the route auditable as one closed mission cycle.
//
//	The coordinates below are simplified city-grid kilometers from the base.
//	They are not random: each waypoint is named, fixed, and domain-relevant.
//	The resulting Euclidean matrix is symmetric and metric-compatible, which makes
//	Christofides a suitable fast approximation strategy.
//
// Playground: https://go.dev/play/p/hea0Q15eHJ2
//
// Algorithm choice:
//
//	We select tsp.Christofides with tsp.BlossomMatch. Christofides is appropriate
//	for symmetric complete metric TSP. BlossomMatch is selected explicitly because
//	the published 1.5 approximation guarantee depends on exact minimum-weight
//	perfect matching over odd MST vertices.
//
// Implementation:
//   - Stage 1: Define a fixed operational waypoint catalog with stable names.
//   - Stage 2: Build a complete symmetric Euclidean distance matrix in kilometers.
//   - Stage 3: Configure Christofides with exact Blossom matching.
//   - Stage 4: Execute tsp.SolveMatrix through the canonical public matrix facade.
//   - Stage 5: Project vertex indices back to waypoint names with Result.VertexTour.
//   - Stage 6: Interpret the route against a validated one-sortie distance envelope.
//
// Behavior highlights:
//   - Uses current public tsp APIs instead of legacy tsp.TSPApprox.
//   - Uses matrix.NewDense and matrix.Fill as the matrix construction boundary.
//   - Keeps route labels aligned with matrix row/column order.
//   - Demonstrates a real approximation certificate: ratio is meaningful only
//     because MatchingAlgo is BlossomMatch.
//   - Prints operational interpretation, not only raw indices.
//
// Inputs:
//   - None. The waypoint catalog and coordinates are embedded as deterministic
//     scenario data.
//
// Returns:
//   - None. The function prints a route summary and mission interpretation.
//
// Errors:
//   - Fails fast on matrix construction errors.
//   - Fails fast on matrix filling errors.
//   - Fails fast on tsp.SolveMatrix errors.
//   - Fails fast if result.VertexTour cannot project route indices to names.
//
// Determinism:
//   - Stable for the same waypoint order, coordinates, tsp.Options, and lvlath/tsp
//     deterministic tie-breaking policy.
//
// Complexity:
//   - Matrix construction is O(n²) time and O(n²) space.
//   - Christofides dense matrix preparation is O(n²) for MST-style matrix scans,
//     plus exact Blossom MWPM over the odd-degree MST set.
//   - Eulerian traversal and shortcut publication are O(V+E) over the Christofides
//     multigraph.
//   - Result printing is O(n).
//
// Notes:
//   - The distance unit is kilometers because coordinates are kilometers.
//   - The 1.5 ratio is a worst-case approximation guarantee, not the measured
//     result.Cost divided by the unknown optimum of this concrete scenario.
//   - For airspace with wind, no-fly zones, or asymmetric corridors, this symmetric
//     Euclidean model is not sufficient; use an asymmetric cost model and a solver
//     mode that supports directed costs.
//
// AI-Hints:
//   - Do not replace BlossomMatch with GreedyMatch if you want the Christofides
//     1.5 guarantee to remain publishable.
//   - Do not reorder waypointNames without applying the same order to waypoints.
//   - Do not compare route feasibility only by stop count; compare result.Cost
//     against the physical drone range or battery envelope.
//   - Do not treat local-search ATSP examples as equivalent to this symmetric
//     Christofides example.
func ExampleTSP_ApproxDrones() {
	const (
		startVertex                    = 0
		validatedOneSortieKilometers   = 42.0
		emergencyReserveKilometers     = 4.0
		minOperationalMarginKilometers = emergencyReserveKilometers
	)

	waypointNames := []string{
		"Central Hospital Drone Roof",
		"Trauma Center North",
		"Blood Bank Lab",
		"Children's Clinic",
		"Dialysis Unit East",
		"Emergency Pharmacy",
		"Firehouse Relay Pad",
		"Stadium Triage Zone",
		"Harbor Rescue Pier",
		"University Medical Lab",
		"Westside Nursing Home",
		"South Mobile Clinic",
	}

	waypoints := [][2]float64{
		{0.0, 0.0},   // Central Hospital Drone Roof.
		{2.2, 3.4},   // Trauma Center North.
		{4.8, 1.2},   // Blood Bank Lab.
		{6.5, 4.7},   // Children's Clinic.
		{8.2, 2.0},   // Dialysis Unit East.
		{7.4, -1.3},  // Emergency Pharmacy.
		{4.9, -3.0},  // Firehouse Relay Pad.
		{1.8, -4.2},  // Stadium Triage Zone.
		{-1.6, -3.5}, // Harbor Rescue Pier.
		{-3.8, -0.9}, // University Medical Lab.
		{-2.7, 2.6},  // Westside Nursing Home.
		{0.8, 4.9},   // South Mobile Clinic.
	}

	if len(waypointNames) != len(waypoints) {
		log.Fatalf("scenario data mismatch: names=%d coordinates=%d", len(waypointNames), len(waypoints))
	}

	dist, err := matrix.NewDense(len(waypoints), len(waypoints))
	if err != nil {
		log.Fatalf("build drone distance matrix: %v", err)
	}

	distances := make([]float64, len(waypoints)*len(waypoints))
	for from := range waypoints {
		for to := range waypoints {
			if from == to {
				continue
			}

			dx := waypoints[from][0] - waypoints[to][0]
			dy := waypoints[from][1] - waypoints[to][1]
			distances[from*len(waypoints)+to] = math.Hypot(dx, dy)
		}
	}

	if err = dist.Fill(distances); err != nil {
		log.Fatalf("fill drone distance matrix: %v", err)
	}

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.StartVertex = startVertex
	opts.MatchingAlgo = tsp.BlossomMatch
	opts.EnableLocalSearch = false

	result, err := tsp.SolveMatrix(dist, waypointNames, opts)
	if err != nil {
		log.Fatalf("solve medical drone route: %v", err)
	}

	vertexTour, err := result.VertexTour()
	if err != nil {
		log.Fatalf("project route waypoint names: %v", err)
	}

	margin := validatedOneSortieKilometers - result.Cost
	status := "one-sortie-approved"
	if margin < minOperationalMarginKilometers {
		status = "split-route-or-battery-swap-required"
	}

	fmt.Println("Medical drone dispatch route")
	fmt.Printf("closed=%v\n", len(result.Tour) == len(waypoints)+1 &&
		result.Tour[0] == startVertex &&
		result.Tour[len(waypoints)] == startVertex)
	fmt.Printf("waypoints=%d\n", len(waypoints))
	fmt.Printf("distance-km=%.2f\n", result.Cost)
	fmt.Printf("range-margin-km=%.2f\n", margin)
	fmt.Printf("status=%s\n", status)
	fmt.Printf("ratio=%.1f\n", result.ApproximationRatio)
	fmt.Printf("exact=%v optimal=%v\n", result.Exact, result.Optimal)
	fmt.Printf("route=%s\n", strings.Join(vertexTour, " -> "))

	// Output:
	// Medical drone dispatch route
	// closed=true
	// waypoints=12
	// distance-km=43.81
	// range-margin-km=-1.81
	// status=split-route-or-battery-swap-required
	// ratio=1.5
	// exact=false optimal=false
	// route=Central Hospital Drone Roof -> Westside Nursing Home -> University Medical Lab -> Harbor Rescue Pier -> Stadium Triage Zone -> Firehouse Relay Pad -> Emergency Pharmacy -> Children's Clinic -> Dialysis Unit East -> Blood Bank Lab -> Trauma Center North -> South Mobile Clinic -> Central Hospital Drone Roof
}
