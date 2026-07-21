// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package flow_test

import (
	"fmt"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/flow"
)

func ExampleMaxFlow_dataCenterTrafficEngineering() {
	// Scenario:
	// A data center needs to know how much traffic can be safely pushed from the
	// internet ingress tier into a latency-critical service. Capacities are Gbps.
	//
	// Interpretation:
	//   - InternetGW is the external ingress boundary.
	//   - EdgeA/EdgeB are edge aggregation switches.
	//   - Core1/Core2 are core fabric switches.
	//   - RackSearch/RackAPI are service pools.
	//   - Service is the logical demand sink.
	//
	// Expected engineering question:
	// "Can the current fabric deliver at least 180 Gbps to the service before
	// the next traffic peak?"
	g, err := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	if err != nil {
		fmt.Println(err)
		return
	}

	edges := []struct {
		from     string
		to       string
		capacity float64
	}{
		{from: "InternetGW", to: "EdgeA", capacity: 120},
		{from: "InternetGW", to: "EdgeB", capacity: 100},

		{from: "EdgeA", to: "Core1", capacity: 80},
		{from: "EdgeA", to: "Core2", capacity: 40},
		{from: "EdgeB", to: "Core1", capacity: 50},
		{from: "EdgeB", to: "Core2", capacity: 60},

		{from: "Core1", to: "RackSearch", capacity: 70},
		{from: "Core1", to: "RackAPI", capacity: 50},
		{from: "Core2", to: "RackSearch", capacity: 30},
		{from: "Core2", to: "RackAPI", capacity: 60},

		{from: "RackSearch", to: "Service", capacity: 100},
		{from: "RackAPI", to: "Service", capacity: 90},
	}

	for _, edge := range edges {
		if _, err = g.AddEdge(edge.from, edge.to, edge.capacity); err != nil {
			fmt.Println(err)
			return
		}
	}

	result, err := flow.MaxFlow(
		g,
		"InternetGW",
		"Service",
		flow.WithAlgorithm(flow.AlgorithmDinic),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("scenario=data-center-traffic\n")
	fmt.Printf("maxThroughputGbps=%g\n", result.Value)
	fmt.Printf("meets180GbpsTarget=%v\n", result.Value >= 180)
	fmt.Printf("algorithm=%s\n", result.Algorithm)
	fmt.Printf("residualPublished=%v\n", result.Residual != nil)
	fmt.Printf("partial=%v\n", result.Partial)

	// Output:
	// scenario=data-center-traffic
	// maxThroughputGbps=190
	// meets180GbpsTarget=true
	// algorithm=dinic
	// residualPublished=true
	// partial=false
}

func ExampleMaxFlow_smartGridMinCutCertificate() {
	// Scenario:
	// A smart-grid controller estimates how much power can be delivered from a
	// generation portfolio to a demand region. Capacities are MW.
	//
	// The scalar max-flow value is useful, but the operationally important proof
	// is the min-cut: it identifies the transmission boundary that caps delivery.
	//
	// This example computes the max flow and then verifies the theorem:
	//
	//     max-flow value == capacity of the returned min-cut
	//
	// using the original graph and MaxFlowResult.CutSourceSide.
	g, err := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	if err != nil {
		fmt.Println(err)
		return
	}

	edges := []struct {
		from     string
		to       string
		capacity float64
	}{
		{from: "Grid", to: "SolarNorth", capacity: 90},
		{from: "Grid", to: "WindCoast", capacity: 70},
		{from: "Grid", to: "HydroEast", capacity: 60},
		{from: "Grid", to: "BatteryReserve", capacity: 40},

		{from: "SolarNorth", to: "SubstationA", capacity: 80},
		{from: "SolarNorth", to: "SubstationB", capacity: 10},
		{from: "WindCoast", to: "SubstationA", capacity: 30},
		{from: "WindCoast", to: "SubstationB", capacity: 40},
		{from: "HydroEast", to: "SubstationB", capacity: 60},
		{from: "BatteryReserve", to: "SubstationA", capacity: 20},
		{from: "BatteryReserve", to: "SubstationB", capacity: 20},

		{from: "SubstationA", to: "DemandRegion", capacity: 100},
		{from: "SubstationB", to: "DemandRegion", capacity: 120},
	}

	for _, edge := range edges {
		if _, err = g.AddEdge(edge.from, edge.to, edge.capacity); err != nil {
			fmt.Println(err)
			return
		}
	}

	result, err := flow.MaxFlow(
		g,
		"Grid",
		"DemandRegion",
		flow.WithAlgorithm(flow.AlgorithmEdmondsKarp),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	sourceSide := make(map[string]bool, len(result.CutSourceSide))
	for _, vertexID := range result.CutSourceSide {
		sourceSide[vertexID] = true
	}

	cutCapacity := 0.0
	for _, edge := range g.Edges() {
		if sourceSide[edge.From] && !sourceSide[edge.To] {
			cutCapacity += edge.Weight
		}
	}

	fmt.Printf("scenario=smart-grid-dispatch\n")
	fmt.Printf("maxDeliverableMW=%g\n", result.Value)
	fmt.Printf("minCutCapacityMW=%g\n", cutCapacity)
	fmt.Printf("theoremVerified=%v\n", cutCapacity == result.Value)
	fmt.Printf("partial=%v\n", result.Partial)

	// Output:
	// scenario=smart-grid-dispatch
	// maxDeliverableMW=220
	// minCutCapacityMW=220
	// theoremVerified=true
	// partial=false
}

func ExampleCapacityMatrix_regionalBackboneDiagnostics() {
	// Scenario:
	// A network operations team wants a deterministic capacity matrix for a
	// regional two-way fiber backbone before running higher-level analysis.
	//
	// The graph is undirected because each physical link is modeled as usable in
	// both directions for this capacity snapshot. CapacityMatrix applies the same
	// adapter law as MaxFlow: an undirected edge becomes two directed capacities.
	g, err := core.NewGraph(core.WithDirected(false), core.WithWeighted())
	if err != nil {
		fmt.Println(err)
		return
	}

	links := []struct {
		a        string
		b        string
		capacity float64
	}{
		{a: "Frankfurt", b: "Warsaw", capacity: 160},
		{a: "Frankfurt", b: "Prague", capacity: 120},
		{a: "Warsaw", b: "Kyiv", capacity: 90},
		{a: "Prague", b: "Kyiv", capacity: 70},
		{a: "Kyiv", b: "Tbilisi", capacity: 55},
	}

	for _, link := range links {
		if _, err = g.AddEdge(link.a, link.b, link.capacity); err != nil {
			fmt.Println(err)
			return
		}
	}

	capacityMatrix, order, err := flow.CapacityMatrix(g)
	if err != nil {
		fmt.Println(err)
		return
	}

	index := make(map[string]int, len(order))
	for i, vertexID := range order {
		index[vertexID] = i
	}

	frankfurtToWarsaw, err := capacityMatrix.At(index["Frankfurt"], index["Warsaw"])
	if err != nil {
		fmt.Println(err)
		return
	}

	warsawToFrankfurt, err := capacityMatrix.At(index["Warsaw"], index["Frankfurt"])
	if err != nil {
		fmt.Println(err)
		return
	}

	kyivToTbilisi, err := capacityMatrix.At(index["Kyiv"], index["Tbilisi"])
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("scenario=regional-backbone-matrix\n")
	fmt.Printf("vertices=%v\n", order)
	fmt.Printf("Frankfurt->Warsaw=%g\n", frankfurtToWarsaw)
	fmt.Printf("Warsaw->Frankfurt=%g\n", warsawToFrankfurt)
	fmt.Printf("Kyiv->Tbilisi=%g\n", kyivToTbilisi)

	// Output:
	// scenario=regional-backbone-matrix
	// vertices=[Frankfurt Kyiv Prague Tbilisi Warsaw]
	// Frankfurt->Warsaw=160
	// Warsaw->Frankfurt=160
	// Kyiv->Tbilisi=55
}
