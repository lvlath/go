// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package main demonstrates spectral graph analysis with lvlath/matrix.
//
// Scenario:
//
//	We model a small infrastructure network where two dense service zones are
//	connected through a narrow bridge and a storage tail. This is exactly the kind
//	of topology where local degree alone is not enough: a node may have a modest
//	number of direct links but still dominate global connectivity because many
//	paths must pass through it.
//
//	The example builds the graph, converts it into an adjacency matrix, then uses
//	matrix operations to answer practical engineering questions:
//
//	  - Which services have the largest local adjacency mass?
//	  - Which services are emphasized by the principal spectral mode?
//	  - Does the Jacobi eigen result satisfy A*v ≈ λ*v?
//	  - What is the shortest failover distance from the web tier to backup storage?
//
// Graph:
//
//	 ┌─────auth──────┐
//	 │       │       │
//	 │       │       │
//	edge─────┼──────api──────┐
//	 │       │       │       │
//	 │       │       │       │
//	 └──────web──────┘ ┌───cache
//	         │         │     │
//	         ├─────────┘     │
//	         │               │
//	         └──────┐ ┌──────┘
//	  db───────────bridge
//	   │
//	backup
//
// Dense service cluster:
//
//	edge, auth, api, web, cache
//
// Critical connector:
//
//	bridge
//
// Storage tail:
//
//	db, backup
//
// Playground: https://go.dev/play/p/TptinFiaIkD
//
// Why this matters:
//
//	The principal eigenvalue/eigenvector of an adjacency operator is a compact
//	spectral fingerprint of graph structure. In operations, observability,
//	routing, and resilience analysis, it is useful as a centrality signal and as
//	a seed for clustering or anomaly detection.
//
// Complexity:
//   - Graph-to-adjacency build: O(V + E) plus dense storage O(V²).
//   - Symmetrization: O(V²).
//   - EigenSym: O(maxIter · V³).
//   - Power iteration: O(steps · V²).
//   - Metric closure: O(V³).
package main

import (
	"fmt"
	"log"
	"math"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
)

const (
	spectralTol     = 1e-10
	spectralMaxIter = 200
	powerSteps      = 24
)

func ExampleMatrix_spectralAnalysis() {
	// NewGraph creates the source topology. This graph is undirected because
	// infrastructure links are treated as bidirectional reachability channels,
	// and weighted because the matrix layer should preserve numeric edge costs.
	g, err := core.NewGraph(core.WithDirected(false), core.WithWeighted())
	if err != nil {
		log.Fatalf("new graph: %v", err)
	}

	// This is a deterministic example with static data and compatible graph
	// capabilities. Ignoring AddEdge errors here keeps the scenario readable.
	// Do not repeat this in production code: real systems must check every
	// returned error from graph mutation.
	_, _ = g.AddEdge("edge", "auth", 1)
	_, _ = g.AddEdge("edge", "api", 1)
	_, _ = g.AddEdge("edge", "web", 1)
	_, _ = g.AddEdge("auth", "api", 1)
	_, _ = g.AddEdge("auth", "web", 1)
	_, _ = g.AddEdge("api", "web", 1)
	_, _ = g.AddEdge("api", "cache", 1)
	_, _ = g.AddEdge("web", "cache", 1)
	_, _ = g.AddEdge("web", "bridge", 1)
	_, _ = g.AddEdge("cache", "bridge", 1)
	_, _ = g.AddEdge("bridge", "db", 1)
	_, _ = g.AddEdge("db", "backup", 1)

	// NewMatrixOptions freezes the matrix interpretation of the graph:
	// undirected cells are mirrored, and weights are copied into adjacency.
	opts, err := matrix.NewMatrixOptions(
		matrix.WithUndirected(),
		matrix.WithWeighted(),
	)
	if err != nil {
		log.Fatalf("matrix options: %v", err)
	}

	// NewAdjacencyMatrix maps service IDs to deterministic matrix coordinates
	// and materializes the graph as a dense adjacency operator.
	am, err := matrix.NewAdjacencyMatrix(g, opts)
	if err != nil {
		log.Fatalf("adjacency: %v", err)
	}

	// DegreeVector summarizes local adjacency mass. In this unweighted graph it
	// behaves as ordinary service degree; with weights it becomes weighted row mass.
	degrees, err := am.DegreeVector()
	if err != nil {
		log.Fatalf("degree vector: %v", err)
	}

	// Symmetrize enforces the operator required by EigenSym. The graph is already
	// undirected, but this explicit repair step documents the spectral precondition:
	// symmetric input for Jacobi eigen-decomposition.
	symA, err := matrix.Symmetrize(am.Mat)
	if err != nil {
		log.Fatalf("symmetrize: %v", err)
	}

	// EigenSym computes the deterministic symmetric eigen-decomposition. The first
	// eigenpair is used here as the principal spectral connectivity signal.
	eigenvalues, eigenvectors, err := matrix.EigenSym(symA, spectralTol, spectralMaxIter)
	if err != nil {
		log.Fatalf("eigen: %v", err)
	}

	// Power iteration is intentionally implemented inline to show how the public
	// MatVecMul facade can be used to build higher-level iterative methods without
	// reaching into Dense internals.
	n := symA.Rows()
	vPower := make([]float64, n)
	for i := range vPower {
		vPower[i] = 1 / math.Sqrt(float64(n))
	}

	lambdaPower := 0.0
	for iter := 0; iter < powerSteps; iter++ {
		// MatVecMul computes w = A*v using the package kernel rather than manual
		// indexing. This keeps shape validation and dense fast-path behavior inside
		// the matrix package.
		w, err := matrix.MatVecMul(symA, vPower)
		if err != nil {
			log.Fatalf("power iteration MatVecMul: %v", err)
		}

		normSquared := 0.0
		for i := range w {
			normSquared += w[i] * w[i]
		}
		norm := math.Sqrt(normSquared)
		if norm == 0 {
			log.Fatal("power iteration: zero vector")
		}

		for i := range vPower {
			vPower[i] = w[i] / norm
		}

		// A second MatVecMul evaluates the Rayleigh quotient λ ≈ vᵀAv after
		// normalization. This gives a stable scalar estimate of the dominant mode.
		AvPower, err := matrix.MatVecMul(symA, vPower)
		if err != nil {
			log.Fatalf("power iteration Rayleigh MatVecMul: %v", err)
		}

		lambdaPower = 0
		for i := range vPower {
			lambdaPower += vPower[i] * AvPower[i]
		}
	}

	// BuildMetricClosure turns graph adjacency into all-pairs shortest distances.
	// The function forces distance semantics internally: +Inf for unreachable pairs,
	// zero diagonal, and Floyd-Warshall closure.
	closure, err := matrix.BuildMetricClosure(g, opts)
	if err != nil {
		log.Fatalf("metric closure: %v", err)
	}

	fmt.Printf("vertices=%d\n", len(am.VertexIndex))
	fmt.Printf("principalEigenvalue=%.4f\n", eigenvalues[0])
	fmt.Printf("powerEigenvalue=%.4f\n", lambdaPower)

	nodeOrder := []string{"edge", "auth", "api", "web", "cache", "bridge", "db", "backup"}

	fmt.Println("degree:")
	for _, id := range nodeOrder {
		idx := am.VertexIndex[id]
		fmt.Printf("  %-7s %.4f\n", id, degrees[idx])
	}

	fmt.Println("powerCentrality:")
	for _, id := range nodeOrder {
		idx := am.VertexIndex[id]
		fmt.Printf("  %-7s %.4f\n", id, vPower[idx])
	}

	web := am.VertexIndex["web"]
	backup := am.VertexIndex["backup"]

	// At reads a distance from the metric-closure matrix without panics. The
	// value is the shortest failover length from web to backup.
	distWebBackup, err := closure.Mat.At(web, backup)
	if err != nil {
		log.Fatalf("distance web->backup: %v", err)
	}
	fmt.Printf("distance(web,backup)=%.0f\n", distWebBackup)

	// Eigenvectors are returned as matrix columns. Extract the leading vector and
	// verify the defining equation A*v ≈ λ*v using MatVecMul and a deterministic
	// scalar tolerance.
	leading := make([]float64, n)
	for row := range leading {
		v, err := eigenvectors.At(row, 0)
		if err != nil {
			log.Fatalf("leading eigenvector row %d: %v", row, err)
		}
		leading[row] = v
	}

	// MatVecMul computes the left side A*v.
	Av, err := matrix.MatVecMul(symA, leading)
	if err != nil {
		log.Fatalf("A*v: %v", err)
	}

	// The right side λ*v is evaluated explicitly so the residual check is visible.
	okEigenResidual := true
	for i := range leading {
		want := eigenvalues[0] * leading[i]
		if math.Abs(Av[i]-want) > 1e-8 {
			okEigenResidual = false
			break
		}
	}
	fmt.Printf("okEigenResidual=%v\n", okEigenResidual)
}
