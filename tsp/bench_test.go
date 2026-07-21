// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test benchmarks real public tsp algorithm regimes.
//
// Benchmark policy:
//   - Fixed deterministic seeds.
//   - All setup is outside the timed section.
//   - Every benchmark reports allocations.
//   - Errors are fail-fast.
//   - No fmt/log work in hot loops.
//   - Names encode algorithm, size, and regime.
package tsp_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

const (
	benchmarkSeed       = int64(20260707)
	benchmarkStart      = 0
	benchmarkEps        = 1e-12
	benchmarkPointScale = 10_000.0
)

var benchmarkResultSink *tsp.Result

// benchmarkDenseMatrix is a detached dense matrix fixture for benchmark setup.
// It satisfies matrix.Matrix without depending on test helper files.
//
// Complexity:
//   - Access O(1), storage O(n^2).
type benchmarkDenseMatrix struct {
	data [][]float64
}

var _ matrix.Matrix = benchmarkDenseMatrix{}

// Rows returns the number of rows.
//
// Complexity:
//   - Time O(1), Space O(1).
func (m benchmarkDenseMatrix) Rows() int {
	return len(m.data)
}

// Cols returns the number of columns.
//
// Complexity:
//   - Time O(1), Space O(1).
func (m benchmarkDenseMatrix) Cols() int {
	if len(m.data) == 0 {
		return 0
	}

	return len(m.data[0])
}

// At returns one matrix cell with bounds validation.
//
// Complexity:
//   - Time O(1), Space O(1).
func (m benchmarkDenseMatrix) At(row int, col int) (float64, error) {
	if row < 0 || row >= m.Rows() || col < 0 || col >= m.Cols() {
		return 0, matrix.ErrIndexOutOfBounds
	}

	return m.data[row][col], nil
}

// Set writes one matrix cell with bounds validation.
//
// Complexity:
//   - Time O(1), Space O(1).
func (m benchmarkDenseMatrix) Set(row int, col int, value float64) error {
	if row < 0 || row >= m.Rows() || col < 0 || col >= m.Cols() {
		return matrix.ErrIndexOutOfBounds
	}

	m.data[row][col] = value

	return nil
}

// Clone returns a detached copy.
//
// Complexity:
//   - Time O(n^2), Space O(n^2).
func (m benchmarkDenseMatrix) Clone() matrix.Matrix {
	copyData := make([][]float64, len(m.data))

	for row := range m.data {
		copyData[row] = append([]float64(nil), m.data[row]...)
	}

	return benchmarkDenseMatrix{data: copyData}
}

// benchmarkMetricComplete builds a deterministic Euclidean complete metric.
//
// Complexity:
//   - Time O(n^2), Space O(n^2).
func benchmarkMetricComplete(n int, seed int64) matrix.Matrix {
	rng := rand.New(rand.NewSource(seed))

	points := make([][2]float64, n)
	for vertex := 0; vertex < n; vertex++ {
		points[vertex] = [2]float64{
			rng.Float64() * benchmarkPointScale,
			rng.Float64() * benchmarkPointScale,
		}
	}

	data := make([][]float64, n)
	for row := 0; row < n; row++ {
		data[row] = make([]float64, n)
	}

	for row := 0; row < n; row++ {
		for col := row + 1; col < n; col++ {
			dx := points[row][0] - points[col][0]
			dy := points[row][1] - points[col][1]
			value := math.Hypot(dx, dy)

			data[row][col] = value
			data[col][row] = value
		}
	}

	return benchmarkDenseMatrix{data: data}
}

// benchmarkSolve runs SolveMatrix and stores the result in a package sink so the
// compiler cannot eliminate the call.
//
// Complexity:
//   - Solver-dependent.
func benchmarkSolve(b *testing.B, dist matrix.Matrix, opts tsp.Options) {
	b.Helper()

	result, err := tsp.SolveMatrix(dist, nil, opts)
	if err != nil {
		b.Fatalf("SolveMatrix failed: %v", err)
	}

	benchmarkResultSink = result
}

func BenchmarkHeldKarp_N12Dense(b *testing.B) {
	dist := benchmarkMetricComplete(12, benchmarkSeed)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.ExactHeldKarp
	opts.Symmetric = true
	opts.StartVertex = benchmarkStart
	opts.EnableLocalSearch = false
	opts.Eps = benchmarkEps

	b.ReportAllocs()
	b.ResetTimer()

	for iteration := 0; iteration < b.N; iteration++ {
		benchmarkSolve(b, dist, opts)
	}
}

func BenchmarkBranchBound_N14MetricSeeded(b *testing.B) {
	dist := benchmarkMetricComplete(14, benchmarkSeed)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.BranchAndBound
	opts.Symmetric = true
	opts.StartVertex = benchmarkStart
	opts.BoundAlgo = tsp.OneTreeBound
	opts.EnableLocalSearch = false
	opts.Eps = benchmarkEps

	b.ReportAllocs()
	b.ResetTimer()

	for iteration := 0; iteration < b.N; iteration++ {
		benchmarkSolve(b, dist, opts)
	}
}

func BenchmarkChristofides_N200MetricBlossom(b *testing.B) {
	dist := benchmarkMetricComplete(200, benchmarkSeed)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.StartVertex = benchmarkStart
	opts.MatchingAlgo = tsp.BlossomMatch
	opts.EnableLocalSearch = false
	opts.Eps = benchmarkEps

	b.ReportAllocs()
	b.ResetTimer()

	for iteration := 0; iteration < b.N; iteration++ {
		benchmarkSolve(b, dist, opts)
	}
}

func BenchmarkChristofides_N200MetricGreedy(b *testing.B) {
	dist := benchmarkMetricComplete(200, benchmarkSeed)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.StartVertex = benchmarkStart
	opts.MatchingAlgo = tsp.GreedyMatch
	opts.EnableLocalSearch = false
	opts.Eps = benchmarkEps

	b.ReportAllocs()
	b.ResetTimer()

	for iteration := 0; iteration < b.N; iteration++ {
		benchmarkSolve(b, dist, opts)
	}
}

func BenchmarkTwoOpt_N500TrivialRingSeed(b *testing.B) {
	dist := benchmarkMetricComplete(500, benchmarkSeed)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.TwoOptOnly
	opts.Symmetric = true
	opts.StartVertex = benchmarkStart
	opts.EnableLocalSearch = true
	opts.ShuffleNeighborhood = false
	opts.Seed = benchmarkSeed
	opts.Eps = benchmarkEps

	b.ReportAllocs()
	b.ResetTimer()

	for iteration := 0; iteration < b.N; iteration++ {
		benchmarkSolve(b, dist, opts)
	}
}

func BenchmarkThreeOpt_N200Symmetric(b *testing.B) {
	dist := benchmarkMetricComplete(200, benchmarkSeed)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.ThreeOptOnly
	opts.Symmetric = true
	opts.StartVertex = benchmarkStart
	opts.EnableLocalSearch = true
	opts.BestImprovement = false
	opts.ShuffleNeighborhood = false
	opts.Seed = benchmarkSeed
	opts.Eps = benchmarkEps

	b.ReportAllocs()
	b.ResetTimer()

	for iteration := 0; iteration < b.N; iteration++ {
		benchmarkSolve(b, dist, opts)
	}
}
