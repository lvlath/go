// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test exercises the public Christofides route after matching extraction.
// It checks the existing algorithm and approximation metadata without private hooks.
package tsp_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/match"
	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

// tspLineMatrix constructs a true metric dense fixture from 1D station positions.
func tspLineMatrix(t *testing.T, n int) *matrix.Dense {
	t.Helper()
	m, err := matrix.NewDense(n, n)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			w := i - j
			if w < 0 {
				w = -w
			}
			if err := m.Set(i, j, float64(w)); err != nil {
				t.Fatal(err)
			}
		}
	}
	return m
}

// TestChristofides_IntegrationWithMatch proves both policies remain selectable.
// The optimal 1D metric tour has exact cost 2*(n-1), independent of orientation.
func TestChristofides_IntegrationWithMatch(t *testing.T) {
	m := tspLineMatrix(t, 6)
	for _, policy := range []struct {
		name  string
		algo  tsp.MatchingAlgo
		ratio float64
	}{
		{name: "Blossom", algo: tsp.BlossomMatch, ratio: tsp.ChristofidesApproximationRatio},
		{name: "Greedy", algo: tsp.GreedyMatch, ratio: tsp.NoApproximationRatio},
	} {
		t.Run(policy.name, func(t *testing.T) {
			opts := tsp.DefaultOptions()
			opts.Algo = tsp.Christofides
			opts.MatchingAlgo = policy.algo
			opts.EnableLocalSearch = false
			first, err := tsp.ChristofidesSolve(m, opts)
			if err != nil {
				t.Fatal(err)
			}
			if err = tsp.ValidateTour(first.Tour, 6, opts.StartVertex); err != nil {
				t.Fatalf("invalid route: %v", err)
			}
			if first.Cost != 10 || first.ApproximationRatio != policy.ratio {
				t.Fatalf("route cost/ratio drift: result=%+v", first)
			}
			second, err := tsp.ChristofidesSolve(m, opts)
			if err != nil {
				t.Fatal(err)
			}
			if first.Cost != second.Cost || len(first.Tour) != len(second.Tour) {
				t.Fatalf("nondeterministic route result: first=%+v second=%+v", first, second)
			}
			for i := range first.Tour {
				if first.Tour[i] != second.Tour[i] {
					t.Fatalf("nondeterministic tour at %d", i)
				}
			}
		})
	}
}

// TestChristofides_DomainAndDefaults preserves TSP's enum/default values.
// It also proves the legacy result is not replaced by match.Result.
func TestChristofides_DomainAndDefaults(t *testing.T) {
	if tsp.GreedyMatch != 0 || tsp.BlossomMatch != 1 || tsp.DefaultOptions().MatchingAlgo != tsp.BlossomMatch {
		t.Fatal("legacy TSP enum/default has changed")
	}
	m := tspLineMatrix(t, 4)
	options := tsp.DefaultOptions()
	options.EnableLocalSearch = false
	result, err := tsp.ChristofidesSolve(m, options)
	if err != nil {
		t.Fatal(err)
	}
	if result.Algorithm != tsp.Christofides || result.Cost != 6 || math.IsNaN(result.Cost) {
		t.Fatalf("route metadata drift: %+v", result)
	}
	invalid := options
	invalid.MatchingAlgo = tsp.MatchingAlgo(255)
	failed, err := tsp.ChristofidesSolve(m, invalid)
	if failed != nil || !errors.Is(err, tsp.ErrInvalidOptions) {
		t.Fatalf("unknown strategy error: result=%+v err=%v", failed, err)
	}
}

// TestChristofides_BlossomErrorPreservesBothPackageClasses checks adapter provenance.
// The whole matrix is symmetric within legacy structural symTol, while the
// explicitly smaller Blossom Eps rejects the odd-subset local edge difference.
func TestChristofides_BlossomErrorPreservesBothPackageClasses(t *testing.T) {
	const n = 4
	data := make([][]float64, n)
	for i := range data {
		data[i] = make([]float64, n)
		for j := range data[i] {
			delta := i - j
			if delta < 0 {
				delta = -delta
			}
			data[i][j] = float64(delta)
		}
	}
	// The line MST has odd endpoints 0 and 3, so this accepted near-asymmetry
	// appears in the selected local two-vertex Blossom input.
	data[0][3] += 5e-13
	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.EnableLocalSearch = false
	opts.Eps = 1e-14
	result, err := tsp.ChristofidesSolve(testDense{a: data}, opts)
	if result != nil || !errors.Is(err, tsp.ErrAsymmetry) || !errors.Is(err, match.ErrAsymmetry) {
		t.Fatalf("Blossom cause was not retained: result=%+v err=%v", result, err)
	}
}
