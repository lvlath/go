// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// This standalone example demonstrates a balanced bidirectional A* routing query.
// All labels and the returned witness refer to original graph costs and edge IDs.
package main

import (
	"fmt"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dijkstra"
)

func main() {
	graph, err := core.NewGraph(core.WithWeighted(), core.WithDirected(true))
	if err != nil {
		panic(err)
	}

	// The first meeting at 101 must not hide the optimal route at 52.
	for _, edge := range []struct {
		from, to, id string
		cost         float64
	}{
		{"S", "A", "startA", 1},
		{"A", "T", "finishA", 100},
		{"S", "B", "startB", 50},
		{"B", "T", "finishB", 2},
	} {
		if _, err := graph.AddEdge(edge.from, edge.to, edge.cost, core.WithID(edge.id)); err != nil {
			panic(err)
		}
	}

	// Both callbacks are consistent on every original directed arc.
	toward := map[string]float64{"S": -100, "A": -100, "B": -100, "T": 0}
	from := map[string]float64{"S": 0, "A": -100, "B": -100, "T": -100}
	toTarget := func(id string) (float64, error) { return toward[id], nil }
	fromSource := func(id string) (float64, error) { return from[id], nil }

	result, err := dijkstra.BidirectionalAStar(
		graph, "S", "T", toTarget, fromSource,
		dijkstra.WithSearchMaxDistance(52),
	)
	if err != nil {
		panic(err)
	}
	vertices, edges, err := result.Path()
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Method, result.Distance, vertices, edges)
	// Output: 3 52 [S B T] [startB finishB]
}
