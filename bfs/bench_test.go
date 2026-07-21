// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package bfs_test

import (
	"fmt"
	"math/rand"
	"strconv"
	"testing"

	"github.com/lvlath/go/bfs"
	"github.com/lvlath/go/core"
)

// AI-HINTS (file):
//   - Benchmark setup must be fully completed before b.ResetTimer().
//   - Never ignore AddEdge errors in setup: otherwise you benchmark an unknown topology.
//   - If random generation can create duplicates/loops, enforce a strict law:
//       - either enable loops/multi-edges,
//       - or generate edges "until success" to insert exactly E valid edges.
//   - The hot loop must contain only BFS (+ minimal err check).

const (
	benchWeightZero = 0.0
)

// BenchmarkBFS_Chain measures BFS on a directed chain of size N+1 vertices and N edges.
//
// Setup integrity:
//   - Exactly V=N+1 vertices are reachable from the start due to directed edges (v0 -> v1 -> ...).
func BenchmarkBFS_Chain(b *testing.B) {
	const (
		N = 10_000
	)

	ids := make([]string, N+1)
	for i := 0; i <= N; i++ {
		ids[i] = "v" + strconv.Itoa(i)
	}

	g, _ := core.NewGraph(core.WithDirected(true))
	for i := 0; i < N; i++ {
		_, err := g.AddEdge(ids[i], ids[i+1], benchWeightZero)
		if err != nil {
			b.Fatalf("setup AddEdge(%q,%q) failed: %v", ids[i], ids[i+1], err)
		}
	}

	V := N + 1
	E := N
	startID := ids[0]

	b.ReportAllocs()
	b.SetBytes(int64(V + E))

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := bfs.BFS(g, startID)
		if err != nil {
			b.Fatalf("BFS failed: %v", err)
		}
	}
}

// BenchmarkBFS_BinaryTree runs BFS on a complete binary tree of depth D.
//
// Setup integrity:
//   - Exactly V=2^D-1 vertices and E=V-1 edges are inserted.
//   - Directed edges are parent -> children, so all nodes are reachable from root "1".
func BenchmarkBFS_BinaryTree(b *testing.B) {
	const (
		depth = 10 // V = 2^10 - 1 = 1023, E = 1022
	)

	nodeCount := (1 << depth) - 1
	edgeCount := nodeCount - 1

	g, _ := core.NewGraph(core.WithDirected(true))

	// Build edges: parent -> children. Vertices are created implicitly by AddEdge.
	for i := 1; i <= (nodeCount-1)/2; i++ {
		p := strconv.Itoa(i)

		_, err := g.AddEdge(p, strconv.Itoa(2*i), benchWeightZero)
		if err != nil {
			b.Fatalf("setup AddEdge failed: %v", err)
		}
		_, err = g.AddEdge(p, strconv.Itoa(2*i+1), benchWeightZero)
		if err != nil {
			b.Fatalf("setup AddEdge failed: %v", err)
		}
	}

	b.ReportAllocs()
	b.SetBytes(int64(nodeCount + edgeCount))

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := bfs.BFS(g, "1")
		if err != nil {
			b.Fatalf("BFS failed: %v", err)
		}
	}
}

// BenchmarkBFS_Grid runs BFS on an M×M grid graph.
//
// Setup integrity:
//   - Uses directed edges (right and down) so the start corner reaches all vertices.
//   - Exactly E = 2*M*(M-1) edges are inserted.
func BenchmarkBFS_Grid(b *testing.B) {
	const (
		M = 100
	)

	V := M * M
	E := 2 * M * (M - 1)

	g, _ := core.NewGraph(core.WithDirected(true))

	// Precompute IDs to avoid repeated formatting logic noise in edge insertion.
	ids := make([][]string, M)
	for i := 0; i < M; i++ {
		row := make([]string, M)
		for j := 0; j < M; j++ {
			row[j] = fmt.Sprintf("%d_%d", i, j)
		}
		ids[i] = row
	}

	// Add directed edges: (i,j) -> (i,j+1) and (i,j) -> (i+1,j).
	for i := 0; i < M; i++ {
		for j := 0; j < M; j++ {
			if j+1 < M {
				_, err := g.AddEdge(ids[i][j], ids[i][j+1], benchWeightZero)
				if err != nil {
					b.Fatalf("setup AddEdge failed: %v", err)
				}
			}
			if i+1 < M {
				_, err := g.AddEdge(ids[i][j], ids[i+1][j], benchWeightZero)
				if err != nil {
					b.Fatalf("setup AddEdge failed: %v", err)
				}
			}
		}
	}

	startID := ids[0][0]

	b.ReportAllocs()
	b.SetBytes(int64(V + E))

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := bfs.BFS(g, startID)
		if err != nil {
			b.Fatalf("BFS failed: %v", err)
		}
	}
}

// BenchmarkBFS_RandomSparse measures BFS on a sparse undirected graph with exact V/E.
//
// Setup integrity law (chosen):
//   - Loops are forbidden and duplicates are forbidden.
//   - We generate edges "until success" to insert exactly E valid edges.
//   - We also build a deterministic backbone chain first to guarantee connectivity from "n0",
//     so BFS actually traverses ~V vertices (stable workload).
func BenchmarkBFS_RandomSparse(b *testing.B) {
	const (
		V = 5000
		E = 10_000
	)

	if E < V-1 {
		b.Fatalf("invalid benchmark constants: need E >= V-1 for backbone connectivity, got V=%d E=%d", V, E)
	}

	ids := make([]string, V)
	for i := 0; i < V; i++ {
		ids[i] = "n" + strconv.Itoa(i)
	}

	g, _ := core.NewGraph(core.WithDirected(false))

	// Stage 1: Deterministic connected backbone (V-1 edges).
	for i := 0; i < V-1; i++ {
		_, err := g.AddEdge(ids[i], ids[i+1], benchWeightZero)
		if err != nil {
			b.Fatalf("setup backbone AddEdge failed: %v", err)
		}
	}

	// Stage 2: Add remaining random edges until exactly E edges exist.
	//
	// We enforce:
	//   - no loops (u != v)
	//   - no duplicates (both directions considered duplicates for undirected graphs)
	rnd := rand.New(rand.NewSource(42))

	targetExtra := E - (V - 1)
	addedExtra := 0

	for addedExtra < targetExtra {
		u := rnd.Intn(V)
		v := rnd.Intn(V)
		if u == v {
			continue
		}

		from := ids[u]
		to := ids[v]

		// Prevent duplicates in the current graph state.
		//
		// For undirected graphs HasEdge works in both directions by contract.
		if g.HasEdge(from, to) {
			continue
		}

		_, err := g.AddEdge(from, to, benchWeightZero)
		if err != nil {
			// Defensive: if this ever triggers, setup would become non-deterministic.
			// We fail the benchmark immediately to avoid measuring an unknown topology.
			b.Fatalf("setup random AddEdge failed: %v", err)
		}

		addedExtra++
	}

	startID := ids[0]

	b.ReportAllocs()
	b.SetBytes(int64(V + E))

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := bfs.BFS(g, startID)
		if err != nil {
			b.Fatalf("BFS failed: %v", err)
		}
	}
}

// BenchmarkBFS_HookOverhead compares BFS with no hook vs a CPU-heavy OnVisit hook.
//
// Setup integrity:
//   - Graph is built once.
//   - Hot loop contains only BFS (+ minimal err check).
func BenchmarkBFS_HookOverhead(b *testing.B) {
	const (
		N = 1000
	)

	ids := make([]string, N+1)
	for i := 0; i <= N; i++ {
		ids[i] = "v" + strconv.Itoa(i)
	}

	g, _ := core.NewGraph(core.WithDirected(true))
	for i := 0; i < N; i++ {
		_, err := g.AddEdge(ids[i], ids[i+1], benchWeightZero)
		if err != nil {
			b.Fatalf("setup AddEdge failed: %v", err)
		}
	}

	V := N + 1
	E := N
	startID := ids[0]

	// No-op hook variant.
	b.Run("NoHook", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(V + E))

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_, err := bfs.BFS(g, startID)
			if err != nil {
				b.Fatalf("BFS failed: %v", err)
			}
		}
	})

	// CPU-heavy OnVisit hook variant.
	//
	// AI-HINT:
	//   - Keep the hook itself deterministic and allocation-free to measure pure CPU overhead.
	heavy := func(_ string) error {
		sum := 0
		for i := 0; i < 100; i++ {
			sum += i
		}
		_ = sum
		return nil
	}

	b.Run("HeavyVisitHook", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(V + E))

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_, err := bfs.BFS(g, startID, bfs.WithOnVisit(heavy))
			if err != nil {
				b.Fatalf("BFS failed: %v", err)
			}
		}
	})
}
