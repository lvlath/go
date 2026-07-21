// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test verifies Blossom matching through the public Christofides API.
// Internal Blossom invariants live in package tsp tests while this file confirms
// external exact-matching behavior, determinism, and large odd-set acceptance.
package tsp_test

import (
	"testing"

	"github.com/lvlath/go/tsp"
)

func TestBlossomMatch_PublicChristofides_LargeDeterministicK64(t *testing.T) {
	t.Parallel()

	dist := seededSymmetricComplete(64, 64)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.StartVertex = startV
	opts.MatchingAlgo = tsp.BlossomMatch
	opts.EnableLocalSearch = false

	first, err := tsp.ChristofidesSolve(dist, opts)
	mustNoError(t, err)
	mustValidResult(t, first, 64, startV, tsp.Christofides)

	for run := 0; run < 5; run++ {
		got, runErr := tsp.ChristofidesSolve(dist, opts)
		mustNoError(t, runErr)
		mustValidResult(t, got, 64, startV, tsp.Christofides)

		mustEqualInts(t, got.Tour, first.Tour)
		mustEqualFloat(t, got.Cost, first.Cost, epsTiny, "large Blossom deterministic cost")
	}
}

func TestBlossomMatch_PublicChristofides_NoSizeBasedRefusalK128(t *testing.T) {
	t.Parallel()

	dist := seededSymmetricComplete(128, 128)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.StartVertex = startV
	opts.MatchingAlgo = tsp.BlossomMatch
	opts.EnableLocalSearch = false

	result, err := tsp.ChristofidesSolve(dist, opts)
	mustNoError(t, err)
	mustValidResult(t, result, 128, startV, tsp.Christofides)
}

func TestBlossomMatch_PublicChristofides_PublishesFormalRatio(t *testing.T) {
	t.Parallel()

	dist := seededSymmetricComplete(20, 2020)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.StartVertex = startV
	opts.MatchingAlgo = tsp.BlossomMatch
	opts.EnableLocalSearch = false

	result, err := tsp.ChristofidesSolve(dist, opts)
	mustNoError(t, err)
	mustValidResult(t, result, 20, startV, tsp.Christofides)

	mustEqualFloat(t, result.ApproximationRatio, tsp.ChristofidesApproximationRatio, epsTiny, "Blossom ratio")
}
