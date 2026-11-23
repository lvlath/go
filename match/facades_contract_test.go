// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match_test checks that every convenience facade delegates to Perfect.
// These are public-boundary tests; no numerical kernel or internal state is exposed.
package match_test

import (
	"errors"
	"testing"

	"github.com/lvlath/go/match"
)

// TestFacadeOptions_DelegatesToTheCanonicalKernel checks both real public wrappers.
// The options-value facade must reject invalid policies instead of repairing them.
func TestFacadeOptions_DelegatesToTheCanonicalKernel(t *testing.T) {
	matrix := newFixtureMatrix(t, [][]float64{
		{0, 2, 2, 1},
		{2, 0, 100, 2},
		{2, 100, 0, 3},
		{1, 2, 3, 0},
	})
	subset := []int{0, 1, 2, 3}
	blossom, err := match.PerfectWithOptions(matrix, subset, match.DefaultOptions())
	if err != nil || blossom == nil || blossom.Cost != 4 || !blossom.Optimal {
		t.Fatalf("Blossom value policy: result=%+v err=%v", blossom, err)
	}
	explicitGreedy, err := match.GreedyPerfect(matrix, subset)
	if err != nil || explicitGreedy == nil || explicitGreedy.Cost != 101 || explicitGreedy.Optimal {
		t.Fatalf("Greedy facade: result=%+v err=%v", explicitGreedy, err)
	}
	throughValue, err := match.PerfectWithOptions(matrix, subset, match.Options{
		Algorithm: match.Greedy,
		Eps:       match.DefaultEpsilon,
	})
	if err != nil || throughValue == nil || throughValue.Cost != explicitGreedy.Cost {
		t.Fatalf("Greedy value policy: result=%+v err=%v", throughValue, err)
	}
	for _, invalid := range []match.Options{
		{},
		{Algorithm: match.Algorithm(255), Eps: match.DefaultEpsilon},
	} {
		result, err := match.PerfectWithOptions(matrix, subset, invalid)
		if result != nil || !errors.Is(err, match.ErrInvalidOptions) {
			t.Fatalf("invalid policy %+v: result=%+v err=%v", invalid, result, err)
		}
	}
}
