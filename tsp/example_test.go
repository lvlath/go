// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test demonstrates canonical lvlath/tsp usage through production-shaped
// operational scenarios.
//
// The examples use static matrices instead of generated toy graphs. Each matrix
// represents a business cost model: minutes, directed security risk, or mechanical
// positioning latency. The static fixtures are intentionally embedded in the
// examples so readers can audit the exact optimization problem without chasing helpers.
package tsp_test

import (
	"fmt"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

const (
	exampleStartVertex = 0
	exampleEps         = 1e-12
)

// ExampleSolveMatrix_coldChainLogistics demonstrates symmetric metric TSP for
// biopharmaceutical cold-chain distribution.
//
// Business story:
// A regional bio-hub must deliver temperature-sensitive mRNA vaccine containers
// across a hospital network and return to the hub. The containers remain inside
// the validated thermal envelope for 120 minutes. A route beyond that window
// risks product degradation and emergency rescheduling.
//
// Algorithm choice:
// Christofides with BlossomMatch is selected because the instance is symmetric,
// complete, and metric. The exact Blossom MWPM stage is required to publish the
// formal 1.5 approximation ratio.
//
// This example intentionally keeps the dispatch matrix static. In production,
// these values would come from a traffic-time service, but the solver contract is
// identical: matrix rows and columns define the stop order.
func ExampleSolveMatrix_coldChainLogistics() {
	const thermalEnvelopeMinutes = 120.0

	locationNames := []string{
		"Regional Bio-Hub",
		"Children's Clinical Hospital",
		"Oncology Treatment Center",
		"Emergency Red Cross Depot",
		"Metropolitan Infection Ward",
		"Community Health Pavilion",
	}

	// Static symmetric metric transit-time matrix, measured in minutes.
	//
	// The values model a weighted urban road network after travel-time smoothing.
	// Row/column order follows locationNames. The intended interpretation is time,
	// not geographic distance.
	coldChainTransitMinutes := []float64{
		0, 32, 50, 44, 34, 18,
		32, 0, 32, 12, 48, 14,
		50, 32, 0, 20, 16, 46,
		44, 12, 20, 0, 36, 26,
		34, 48, 16, 36, 0, 48,
		18, 14, 46, 26, 48, 0,
	}

	dist, _ := matrix.NewDense(len(locationNames), len(locationNames))
	// Static example data is fixed and auditable. Do not ignore setup errors for
	// dynamic data in production code.
	_ = dist.Fill(coldChainTransitMinutes)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.StartVertex = exampleStartVertex
	opts.MatchingAlgo = tsp.BlossomMatch
	opts.EnableLocalSearch = false
	opts.Eps = exampleEps

	result, err := tsp.SolveMatrix(dist, nil, opts)
	if err != nil {
		fmt.Printf("cold-chain route failed: %v\n", err)
		return
	}

	fmt.Println("Cold-chain vaccine dispatch")
	fmt.Printf("closed-loop=%v\n", len(result.Tour) == len(locationNames)+1 &&
		result.Tour[0] == exampleStartVertex &&
		result.Tour[len(locationNames)] == exampleStartVertex)
	fmt.Printf("route-minutes=%.1f\n", result.Cost)
	fmt.Printf("formal-ratio=%.1f\n", result.ApproximationRatio)
	fmt.Printf("thermal-margin=%.1f minutes\n", thermalEnvelopeMinutes-result.Cost)

	fmt.Print("manifest=")
	for step, vertex := range result.Tour {
		if step > 0 {
			fmt.Print(" -> ")
		}
		fmt.Print(locationNames[vertex])
	}
	fmt.Println()

	// Output:
	// Cold-chain vaccine dispatch
	// closed-loop=true
	// route-minutes=114.0
	// formal-ratio=1.5
	// thermal-margin=6.0 minutes
	// manifest=Regional Bio-Hub -> Metropolitan Infection Ward -> Oncology Treatment Center -> Emergency Red Cross Depot -> Children's Clinical Hospital -> Community Health Pavilion -> Regional Bio-Hub
}

// ExampleSolveMatrix_cashInTransitAsymmetric demonstrates directed local-search
// routing for armored cash-in-transit operations.
//
// Business story:
// An armored vehicle replenishes high-value ATM and cash-depot sites. One-way
// streets, turn restrictions, timed security escorts, and high-risk intersections
// make A->B different from B->A. The matrix therefore stores directed composite
// risk/fuel costs, not symmetric distance.
//
// Algorithm choice:
// TwoOptOnly with Symmetric=false uses the package's directed 2-opt* behavior.
// This is a scalable local-search mode, not an exact ATSP certificate. It is
// useful when dispatch must react quickly and the cost model is directional.
func ExampleSolveMatrix_cashInTransitAsymmetric() {
	stopNames := []string{
		"Secure Bank Vault",
		"Downtown Retail Mega-Mall",
		"Financial District Transit Terminal",
		"Westside Casino Hub",
		"Central Airport Cash-Vault",
	}

	// Static asymmetric risk matrix.
	//
	// Each value combines transit time, fuel burn, guard exposure, and predictable
	// interception risk. The asymmetry is intentional: for example, Mall->Casino
	// can use a guarded arterial route, while Casino->Mall crosses a slower
	// exposure-heavy corridor.
	directedRiskCost := []float64{
		0, 8, 19, 21, 17,
		14, 0, 5, 9, 18,
		18, 11, 0, 6, 20,
		16, 8, 12, 0, 4,
		11, 15, 13, 10, 0,
	}

	dist, _ := matrix.NewDense(len(stopNames), len(stopNames))
	// Static example data is fixed and auditable. Do not ignore setup errors for
	// dynamic data in production code.
	_ = dist.Fill(directedRiskCost)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.TwoOptOnly
	opts.Symmetric = false
	opts.StartVertex = exampleStartVertex
	opts.EnableLocalSearch = true
	opts.Eps = exampleEps

	result, err := tsp.SolveMatrix(dist, nil, opts)
	if err != nil {
		fmt.Printf("cash-in-transit route failed: %v\n", err)
		return
	}

	fmt.Println("Cash-in-transit directed route")
	fmt.Printf("closed-loop=%v\n", len(result.Tour) == len(stopNames)+1 &&
		result.Tour[0] == exampleStartVertex &&
		result.Tour[len(stopNames)] == exampleStartVertex)
	fmt.Printf("directed-risk-cost=%.1f\n", result.Cost)
	fmt.Printf("exact=%v optimal=%v\n", result.Exact, result.Optimal)

	fmt.Print("route=")
	for step, vertex := range result.Tour {
		if step > 0 {
			fmt.Print(" -> ")
		}
		fmt.Print(stopNames[vertex])
	}
	fmt.Println()

	// Output:
	// Cash-in-transit directed route
	// closed-loop=true
	// directed-risk-cost=34.0
	// exact=false optimal=false
	// route=Secure Bank Vault -> Downtown Retail Mega-Mall -> Financial District Transit Terminal -> Westside Casino Hub -> Central Airport Cash-Vault -> Secure Bank Vault
}

// ExampleSolveMatrix_semiconductorLaserDrilling demonstrates exact matrix TSP
// for high-precision semiconductor manufacturing.
//
// Business story:
// A CNC-guided laser drilling workstation must visit microscopic via-hole
// positions on each board. A few saved milliseconds per board can become hours
// of recovered production time across large daily batches. In this offline
// programming stage, near-optimal is not enough: the route must be globally
// optimal for the provided latency matrix.
//
// Algorithm choice:
// BranchAndBound with OneTreeBound performs exact search with an admissible
// lower bound. The result is exact and optimal when the search completes.
func ExampleSolveMatrix_semiconductorLaserDrilling() {
	const dailyBoardVolume = 50000

	siteNames := []string{
		"Servo Home Anchor",
		"Pin A1",
		"Bus Gate X7",
		"Core Array",
		"I/O Bridge",
	}

	// Static symmetric mechanical latency matrix, measured in milliseconds.
	//
	// The values include servo-axis travel, deceleration, settling latency, and
	// laser stabilization overhead between drilling sites.
	servoLatencyMilliseconds := []float64{
		0, 3.6, 9.5, 10.0, 4.5,
		3.6, 0, 1.2, 8.0, 9.0,
		9.5, 1.2, 0, 2.8, 7.5,
		10.0, 8.0, 2.8, 0, 4.7,
		4.5, 9.0, 7.5, 4.7, 0,
	}

	dist, _ := matrix.NewDense(len(siteNames), len(siteNames))
	// Static example data is fixed and auditable. Do not ignore setup errors for
	// dynamic data in production code.
	_ = dist.Fill(servoLatencyMilliseconds)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.BranchAndBound
	opts.Symmetric = true
	opts.StartVertex = exampleStartVertex
	opts.BoundAlgo = tsp.OneTreeBound
	opts.EnableLocalSearch = false
	opts.Eps = exampleEps

	result, err := tsp.SolveMatrix(dist, nil, opts)
	if err != nil {
		fmt.Printf("laser drilling optimization failed: %v\n", err)
		return
	}

	dailyRuntimeSeconds := result.Cost * float64(dailyBoardVolume) / 1000.0

	fmt.Println("Semiconductor laser-drilling route")
	fmt.Printf("closed-loop=%v\n", len(result.Tour) == len(siteNames)+1 &&
		result.Tour[0] == exampleStartVertex &&
		result.Tour[len(siteNames)] == exampleStartVertex)
	fmt.Printf("single-board-latency=%.1f ms\n", result.Cost)
	fmt.Printf("daily-laser-motion=%.1f seconds\n", dailyRuntimeSeconds)
	fmt.Printf("exact=%v optimal=%v\n", result.Exact, result.Optimal)

	fmt.Print("path=")
	for step, vertex := range result.Tour {
		if step > 0 {
			fmt.Print(" -> ")
		}
		fmt.Print(siteNames[vertex])
	}
	fmt.Println()

	// Output:
	// Semiconductor laser-drilling route
	// closed-loop=true
	// single-board-latency=16.8 ms
	// daily-laser-motion=840.0 seconds
	// exact=true optimal=true
	// path=Servo Home Anchor -> Pin A1 -> Bus Gate X7 -> Core Array -> I/O Bridge -> Servo Home Anchor
}
