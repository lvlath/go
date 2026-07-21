// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package main

import (
	"fmt"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/flow"
)

// ExampleMaxFlow_dinicCDN runs a production-style CDN throughput analysis with Dinic.
// The scenario models peak video delivery from a viewer-demand boundary into a
// service tier through edge PoPs, regional backbone nodes, and origin-cache pools.
//
// Scenario:
//
//			A CDN operator prepares for a live-event traffic spike. The question is not
//			"which single edge is large?", but "how much end-to-end traffic can the whole
//			delivery fabric push before some tier becomes the bottleneck?"
//
//	 The network is directed and weighted:
//	 ┌──────────────┐           ┌───────────┐
//	 │ ViewerDemand ├───────────┤ TLSGateway│
//	 └──────────────┘           └─┬───┬───┬─┘
//	                              │   │   │
//	           ┌───────────┐      │   │   │      ┌───────────┐
//	           │  PoP_EU   ├──────┘   │   └──────┤  PoP_US   │
//	           └───┬───┬───┘    ┌─────┴─────┐    └───┰───┬───┘
//	               │   │   ┏━━━━┥ PoP_APAC  ├────┐   ┃   │
//	               │   │   ┃    └───────────┘    │   ┃   │
//	               │   └───╂──────────────────┐  │   ┃   │
//	               │       ┃  ┏━━━━━━━━━━━━━━━┿━━┿━━━┛   │
//	           ┌───┴───────┸─┐┃               │┌─┴───────┴───┐
//	           │ Regional_EU ┝┛               └┤ Regional_US │
//	           └──────┬─────┬┘                 └┰─────┬──────┘
//	                  │     │         ┏━━━━━━━━━┛     │
//	                  │     └─────────╂─────────┐     │
//	                  │     ┏━━━━━━━━━┛         │     │
//	            ┌─────┴─────┸──┐             ┌──┴─────┴─────┐
//	            │ OriginCacheA │             │ OriginCacheB │
//	            └────────┬─────┘             └─────┬────────┘
//	                     │                         │
//	                     │     ┌──────────────┐    │
//	                     └─────┤ VideoService ├────┘
//	                           └──────────────┘
//
// Capacities are in Gbps.
// The final service-facing cache egress is 95 + 90 = 185 Gbps, while upstream
// capacity is higher. Dinic should therefore compute 185 Gbps and the min-cut
// certificate should identify OriginCacheA→VideoService and
// OriginCacheB→VideoService as the saturated bottleneck boundary.
//
// Playground: https://go.dev/play/p/guL-0bqHfCS
//
// Implementation:
//   - Stage 1: Build a directed weighted core.Graph where edge weights are capacities.
//   - Stage 2: Add a deterministic, production-shaped CDN capacity topology.
//   - Stage 3: Run flow.MaxFlow with AlgorithmDinic.
//   - Stage 4: Read MaxFlowResult.Value as the maximum deliverable throughput.
//   - Stage 5: Use CutSourceSide/CutSinkSide to compute and print the min-cut boundary.
//   - Stage 6: Compare the computed throughput against an operational target.
//
// Behavior highlights:
//   - Demonstrates the canonical MaxFlow API, not legacy tuple wrappers.
//   - Uses stable caller-defined edge order for deterministic reporting.
//   - Shows the max-flow/min-cut theorem as an operational bottleneck explanation.
//   - Avoids direct dependency on residual graph internals.
//
// Inputs:
//   - None. The example constructs its own deterministic CDN capacity graph.
//
// Returns:
//   - None. The example prints a compact throughput and bottleneck report.
//
// Errors:
//   - Prints graph-construction or max-flow errors and returns early.
//   - Does not panic or call log.Fatal, so the example remains test-friendly.
//
// Determinism:
//   - Stable for the same edge list, core.Graph ordering, and flow traversal law.
//
// Complexity:
//   - Graph construction is O(E).
//   - Dinic is O(V²·A) worst-case on general networks, where A is residual adjacency count.
//   - Min-cut reporting scans the fixed edge list once: O(E).
//
// Notes:
//   - The min-cut capacity is computed from the original graph, not from residual edges.
//   - MaxFlowResult.Residual is still available for deeper diagnostics, but the example
//     focuses on the business-facing certificate: "what is the bottleneck?"
//
// AI-Hints:
//   - Prefer flow.MaxFlow(..., flow.WithAlgorithm(flow.AlgorithmDinic)) for new examples.
//   - Do not use residual.AdjacencyList() as a public proof surface.
//   - Do not claim a throughput value without verifying the min-cut certificate.
func ExampleMaxFlow_dinicCDN() {
	g, err := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	if err != nil {
		fmt.Println(err)
		return
	}

	_, err = g.AddEdge("ViewerDemand", "TLSGateway", 220)

	_, err = g.AddEdge("TLSGateway", "PoP_EU", 90)
	_, err = g.AddEdge("TLSGateway", "PoP_US", 80)
	_, err = g.AddEdge("TLSGateway", "PoP_APAC", 70)

	_, err = g.AddEdge("PoP_EU", "Regional_EU", 70)
	_, err = g.AddEdge("PoP_EU", "Regional_US", 20)
	_, err = g.AddEdge("PoP_US", "Regional_EU", 25)
	_, err = g.AddEdge("PoP_US", "Regional_US", 65)
	_, err = g.AddEdge("PoP_APAC", "Regional_EU", 10)
	_, err = g.AddEdge("PoP_APAC", "Regional_US", 50)

	_, err = g.AddEdge("Regional_EU", "OriginCacheA", 75)
	_, err = g.AddEdge("Regional_EU", "OriginCacheB", 20)
	_, err = g.AddEdge("Regional_US", "OriginCacheA", 25)
	_, err = g.AddEdge("Regional_US", "OriginCacheB", 85)

	_, err = g.AddEdge("OriginCacheA", "VideoService", 95)
	_, err = g.AddEdge("OriginCacheB", "VideoService", 90)

	result, err := flow.MaxFlow(
		g,
		"ViewerDemand",
		"VideoService",
		flow.WithAlgorithm(flow.AlgorithmDinic),
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

	fmt.Printf("CDN peak throughput: %.0f Gbps\n", result.Value)
	fmt.Printf("SLO target 180 Gbps met: %v\n", result.Value >= 180)
	fmt.Println("Min-cut bottleneck boundary:")

	for _, edge := range g.Edges() {
		if !sourceSide[edge.From] || sourceSide[edge.To] {
			continue
		}

		cutCapacity += edge.Weight
		fmt.Printf("  %s -> %s: %.0f Gbps\n", edge.From, edge.To, edge.Weight)
	}

	fmt.Printf("Min-cut capacity: %.0f Gbps\n", cutCapacity)
	fmt.Printf("Certificate verified: %v\n", cutCapacity == result.Value)
	fmt.Printf("Residual graph published: %v\n", result.Residual != nil)

	// Output:
	// CDN peak throughput: 185 Gbps
	// SLO target 180 Gbps met: true
	// Min-cut bottleneck boundary:
	//   OriginCacheA -> VideoService: 95 Gbps
	//   OriginCacheB -> VideoService: 90 Gbps
	// Min-cut capacity: 185 Gbps
	// Certificate verified: true
	// Residual graph published: true
}
