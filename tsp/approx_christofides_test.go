// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test verifies the Christofides approximation pipeline through the
// public result-native API. The tests focus on symmetric-metric validation,
// Blossom-vs-Greedy matching policy, deterministic output, and local-search
// non-worsening behavior.
package tsp_test

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

// mstWeight returns the dense MST total used by the Christofides sanity bound.
// It intentionally calls the exported MST helper so the test observes the same
// matrix firewall as the production approximation pipeline.
func mstWeight(t *testing.T, dist matrix.Matrix) float64 {
	t.Helper()

	weight, _, err := tsp.MinimumSpanningTree(dist)
	mustNoError(t, err)

	return weight
}

func TestChristofidesBlossomHexagon_ValidTourAndRatio(t *testing.T) {
	t.Parallel()

	const n = 6

	points := [][2]float64{
		{1, 0},
		{0.5, math.Sqrt(3) / 2},
		{-0.5, math.Sqrt(3) / 2},
		{-1, 0},
		{-0.5, -math.Sqrt(3) / 2},
		{0.5, -math.Sqrt(3) / 2},
	}

	dist := euclid(points)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.StartVertex = startV
	opts.MatchingAlgo = tsp.BlossomMatch
	opts.EnableLocalSearch = false

	result, err := tsp.ChristofidesSolve(dist, opts)
	mustNoError(t, err)
	mustValidResult(t, result, n, startV, tsp.Christofides)

	limit := tsp.ChristofidesApproximationRatio * mstWeight(t, dist)
	if result.Cost > limit+epsLoose {
		t.Fatalf("Christofides exceeded 1.5*MST: cost=%.12f limit=%.12f", result.Cost, limit)
	}
	if result.ApproximationRatio != tsp.ChristofidesApproximationRatio {
		t.Fatalf("ratio mismatch: got %.3f want %.3f", result.ApproximationRatio, tsp.ChristofidesApproximationRatio)
	}
}

func TestChristofidesBlossomDeterministicRepeat(t *testing.T) {
	t.Parallel()

	const n = 8

	points := make([][2]float64, n)
	for vertex := 0; vertex < n; vertex++ {
		theta := 2 * math.Pi * float64(vertex) / float64(n)
		radius := 1.0 + 0.02*math.Sin(3*theta)
		points[vertex] = [2]float64{radius * math.Cos(theta), radius * math.Sin(theta)}
	}

	dist := euclid(points)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.StartVertex = startV
	opts.MatchingAlgo = tsp.BlossomMatch
	opts.EnableLocalSearch = false

	first, err := tsp.ChristofidesSolve(dist, opts)
	mustNoError(t, err)
	mustValidResult(t, first, n, startV, tsp.Christofides)

	firstOpen := normalizeClosedToOpen(t, first.Tour)

	Repeat(t, 5, func(t *testing.T) {
		got, runErr := tsp.ChristofidesSolve(dist, opts)
		mustNoError(t, runErr)
		mustValidResult(t, got, n, startV, tsp.Christofides)

		gotOpen := normalizeClosedToOpen(t, got.Tour)
		if !slices.Equal(gotOpen, firstOpen) {
			t.Fatalf("nondeterministic tour: got %v want %v", gotOpen, firstOpen)
		}
		mustEqualFloat(t, got.Cost, first.Cost, epsTiny, "deterministic Christofides cost")
	})
}

func TestChristofidesRejectsAsymmetry(t *testing.T) {
	t.Parallel()

	const n = 7

	points := make([][2]float64, n)
	for vertex := 0; vertex < n; vertex++ {
		theta := 2 * math.Pi * float64(vertex) / float64(n)
		points[vertex] = [2]float64{math.Cos(theta), math.Sin(theta)}
	}

	dist := euclidAsym(points, 0.2)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.StartVertex = startV
	opts.EnableLocalSearch = false

	_, err := tsp.ChristofidesSolve(dist, opts)
	if !errors.Is(err, tsp.ErrAsymmetry) {
		t.Fatalf("want ErrAsymmetry, got %v", err)
	}
}

func TestSolveMatrixChristofidesLocalSearchNotWorseThanRawChristofides(t *testing.T) {
	t.Parallel()

	const n = 10

	points := make([][2]float64, n)
	for vertex := 0; vertex < n; vertex++ {
		theta := 2 * math.Pi * float64(vertex) / float64(n)
		radius := 1.0 + 0.03*math.Cos(4*theta)
		points[vertex] = [2]float64{radius * math.Cos(theta), radius * math.Sin(theta)}
	}

	dist := euclid(points)

	raw := tsp.DefaultOptions()
	raw.Algo = tsp.Christofides
	raw.Symmetric = true
	raw.StartVertex = startV
	raw.MatchingAlgo = tsp.BlossomMatch
	raw.EnableLocalSearch = false

	rawResult, err := tsp.ChristofidesSolve(dist, raw)
	mustNoError(t, err)

	polished := raw
	polished.EnableLocalSearch = true
	polished.BestImprovement = false
	polished.Eps = epsTiny

	polishedResult, err := tsp.SolveMatrix(dist, nil, polished)
	mustNoError(t, err)
	mustValidResult(t, polishedResult, n, startV, tsp.Christofides)

	if polishedResult.Cost > rawResult.Cost+epsTiny {
		t.Fatalf("local search worsened result: raw=%.12f polished=%.12f", rawResult.Cost, polishedResult.Cost)
	}
}

func TestChristofidesGreedyPublishesNoApproximationRatio(t *testing.T) {
	t.Parallel()

	dist := seededSymmetricComplete(12, 777)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.StartVertex = startV
	opts.MatchingAlgo = tsp.GreedyMatch
	opts.EnableLocalSearch = false

	result, err := tsp.ChristofidesSolve(dist, opts)
	mustNoError(t, err)
	mustValidResult(t, result, 12, startV, tsp.Christofides)

	mustEqualFloat(t, result.ApproximationRatio, tsp.NoApproximationRatio, epsTiny, "greedy Christofides ratio")
}

func TestChristofidesBlossomPublishesOnePointFiveRatio(t *testing.T) {
	t.Parallel()

	dist := seededSymmetricComplete(12, 7)
	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.MatchingAlgo = tsp.BlossomMatch
	opts.EnableLocalSearch = false

	result, err := tsp.ChristofidesSolve(dist, opts)
	mustNoError(t, err)
	mustValidResult(t, result, 12, 0, tsp.Christofides)
	mustEqualFloat(t, result.ApproximationRatio, 1.5, epsTiny, "approximation ratio")
}

func TestChristofidesBlossomDeterministicAcrossRuns(t *testing.T) {
	t.Parallel()

	dist := seededSymmetricComplete(32, 42)
	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.MatchingAlgo = tsp.BlossomMatch
	opts.EnableLocalSearch = false

	first, err := tsp.ChristofidesSolve(dist, opts)
	mustNoError(t, err)

	for run := 0; run < 10; run++ {
		got, runErr := tsp.ChristofidesSolve(dist, opts)
		mustNoError(t, runErr)

		mustEqualInts(t, got.Tour, first.Tour)
		mustEqualFloat(t, got.Cost, first.Cost, epsTiny, "deterministic blossom cost")
	}
}

func TestChristofidesBlossomLargeKNoSizeRefusal(t *testing.T) {
	t.Parallel()

	dist := seededSymmetricComplete(128, 1001)
	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.MatchingAlgo = tsp.BlossomMatch
	opts.EnableLocalSearch = false

	result, err := tsp.ChristofidesSolve(dist, opts)
	mustNoError(t, err)
	mustValidResult(t, result, 128, 0, tsp.Christofides)
}

func TestChristofidesBlossomLargeDenseK128K192(t *testing.T) {
	cases := []struct {
		name  string
		k     int
		seed  int64
		start int
	}{
		{name: "k128", k: 128, seed: 128, start: 0},
		{name: "k192", k: 192, seed: 192, start: 7},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dist := seededSymmetricComplete(tc.k, tc.seed)

			opts := tsp.DefaultOptions()
			opts.Algo = tsp.Christofides
			opts.Symmetric = true
			opts.StartVertex = tc.start
			opts.MatchingAlgo = tsp.BlossomMatch
			opts.EnableLocalSearch = false

			result, err := tsp.ChristofidesSolve(dist, opts)
			mustNoError(t, err)
			mustValidResult(t, result, tc.k, tc.start, tsp.Christofides)
			mustEqualFloat(t, result.ApproximationRatio, tsp.ChristofidesApproximationRatio, epsTiny, "ratio")
		})
	}
}

func TestChristofidesBlossomLargeDenseRegressionK128Seed424242(t *testing.T) {
	dist := seededSymmetricComplete(128, 424242)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.MatchingAlgo = tsp.BlossomMatch
	opts.EnableLocalSearch = false

	result, err := tsp.ChristofidesSolve(dist, opts)
	mustNoError(t, err)
	mustValidResult(t, result, 128, 0, tsp.Christofides)
	mustEqualFloat(t, result.ApproximationRatio, tsp.ChristofidesApproximationRatio, epsTiny, "ratio")
}

func TestChristofidesBlossomLargeDenseRegressionK192(t *testing.T) {
	dist := seededSymmetricComplete(192, 192)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.StartVertex = 7
	opts.MatchingAlgo = tsp.BlossomMatch
	opts.EnableLocalSearch = false

	result, err := tsp.ChristofidesSolve(dist, opts)
	mustNoError(t, err)
	mustValidResult(t, result, 192, 7, tsp.Christofides)
	mustEqualFloat(t, result.ApproximationRatio, tsp.ChristofidesApproximationRatio, epsTiny, "ratio")
}

func TestChristofidesBlossomLargeDenseK256Smoke(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large Christofides smoke in short mode")
	}

	dist := seededSymmetricComplete(256, 256)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.MatchingAlgo = tsp.BlossomMatch
	opts.EnableLocalSearch = false
	opts.StartVertex = 11

	result, err := tsp.ChristofidesSolve(dist, opts)
	mustNoError(t, err)
	mustValidResult(t, result, 256, 11, tsp.Christofides)
	mustEqualFloat(t, result.ApproximationRatio, tsp.ChristofidesApproximationRatio, epsTiny, "ratio")
}

func TestChristofidesBlossomLargeDenseDeterministicRepeatK128(t *testing.T) {
	dist := seededSymmetricComplete(128, 424242)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.MatchingAlgo = tsp.BlossomMatch
	opts.EnableLocalSearch = false

	first, err := tsp.ChristofidesSolve(dist, opts)
	mustNoError(t, err)
	mustValidResult(t, first, 128, 0, tsp.Christofides)

	for run := 0; run < 10; run++ {
		got, runErr := tsp.ChristofidesSolve(dist, opts)
		mustNoError(t, runErr)
		mustValidResult(t, got, 128, 0, tsp.Christofides)

		mustEqualInts(t, got.Tour, first.Tour)
		mustEqualFloat(t, got.Cost, first.Cost, epsTiny, "deterministic large Christofides Blossom cost")
	}
}
