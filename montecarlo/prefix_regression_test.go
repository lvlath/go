// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo_test verifies public partial-result and matrix ownership laws.
// These scenarios never construct the private prefix backend directly.
package montecarlo_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/montecarlo"
	"github.com/lvlath/go/rng"
)

// TestPublishedPrefixCompleteIndexDomain exercises both coordinate bounds and
// the dependency's numeric write policy using only the returned matrix.Matrix.
// Implementation: terminate a serial experiment at a known path, clone the
// committed rows, then check independent writes and immutable summary values.
// Determinism: the first three valid path IDs form the exact prefix.
// Complexity: O(Completed*d) copy and verification; no hidden tail is read.
// AI-Hints: an invalid tail coordinate must fail even when backing storage exists.
func TestPublishedPrefixCompleteIndexDomain(t *testing.T) {
	const dimension = 2
	const completed = 3
	const requested = 11
	const seed = 19
	cause := errors.New("measurement service unavailable")
	sampler := &funcSampler{dimension, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		if id == completed {
			return nil, cause
		}
		return []float64{float64(id), 100 + 2*float64(id)}, nil
	}}
	result, err := montecarlo.Run(context.Background(), montecarlo.Plan{
		Paths: requested, Seed: seed, Workers: 1, RetainSamples: true, Quantiles: []float64{.5},
	}, sampler)
	mustErrorIs(t, err, cause)
	mustNilState(t, result, false, "partial result")
	if result.Requested != requested || result.Completed != completed || result.Complete || result.Seed != seed {
		t.Fatalf("unexpected prefix metadata: %+v", result)
	}
	mustNilState(t, result.Quantiles, true, "partial quantiles")
	mustEqualString(t, result.RNG, rng.NewFactory().Version(), "RNG revision")
	mustEqualInt(t, len(result.QuantileLevels), 1, "level count")
	mustEqualBool(t, result.QuantileLevels[0] == .5, true, "level value")
	prefix := result.Samples
	mustNilState(t, prefix, false, "prefix")
	mustEqualInt(t, prefix.Rows(), completed, "public rows")
	mustEqualInt(t, prefix.Cols(), dimension, "public columns")

	// Check every addressable cell against the original logical path identity.
	for row := 0; row < completed; row++ {
		mustMatrixBits(t, prefix, row, 0, float64(row))
		mustMatrixBits(t, prefix, row, 1, 100+2*float64(row))
	}
	// Negative indices, exact upper bounds and very large indices are distinct guards.
	for _, coordinate := range [][2]int{{-1, 0}, {completed, 0}, {requested - 1, 0}, {0, -1}, {0, dimension}, {math.MaxInt, 0}, {0, math.MaxInt}} {
		_, readErr := prefix.At(coordinate[0], coordinate[1])
		mustErrorIs(t, readErr, matrix.ErrOutOfRange)
		mustErrorIs(t, prefix.Set(coordinate[0], coordinate[1], 7), matrix.ErrOutOfRange)
	}
	// Invalid writes cannot corrupt an otherwise valid committed value.
	for _, invalid := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		mustErrorIs(t, prefix.Set(0, 0, invalid), matrix.ErrNaNInf)
		mustMatrixBits(t, prefix, 0, 0, 0)
	}

	clone := prefix.Clone()
	mustNilState(t, clone, false, "cloned prefix")
	mustEqualInt(t, clone.Rows(), completed, "clone rows")
	mustEqualInt(t, clone.Cols(), dimension, "clone columns")
	for row := 0; row < completed; row++ {
		mustMatrixBits(t, clone, row, 0, float64(row))
		mustMatrixBits(t, clone, row, 1, 100+2*float64(row))
	}
	// Mutations must remain local in both directions, including the last cell.
	if err := clone.Set(completed-1, dimension-1, 700); err != nil {
		t.Fatal(err)
	}
	mustMatrixBits(t, prefix, completed-1, dimension-1, 104)
	if err := prefix.Set(0, 1, -700); err != nil {
		t.Fatal(err)
	}
	mustMatrixBits(t, clone, 0, 1, 100)
	closeFloat(t, result.Mean[0], 1)
	closeFloat(t, result.Mean[1], 102)
	closeFloat(t, result.SampleVariance[0], 1)
	closeFloat(t, result.SampleVariance[1], 4)
}

// TestPrefixInvalidLastCoordinateIsAtomic verifies that invalid samples do not
// update earlier valid coordinates or leak a row into the retained prefix.
// Implementation: inject several malformed rows at ID2, compare the prefix's
// moments to the independent two-path analytical reference.
// Determinism: one worker and an explicit failing path, not a timeout.
// Complexity: constant fixture size; each failure tests a distinct public cause.
// AI-Hints: a sampler-provided error takes precedence over any accompanying slice.
func TestPrefixInvalidLastCoordinateIsAtomic(t *testing.T) {
	cause := errors.New("upstream instrument rejected trial")
	for _, tc := range []struct {
		name string
		row  []float64
		err  error
		want error
	}{
		{"too-long", []float64{9, 8, 7}, nil, montecarlo.ErrShape},
		{"late-nan", []float64{1e100, math.NaN()}, nil, montecarlo.ErrNonFinite},
		{"late-negative-infinity", []float64{1e100, math.Inf(-1)}, nil, montecarlo.ErrNonFinite},
		{"external-error-before-shape", nil, fmt.Errorf("sample failed: %w", cause), cause},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sampler := &funcSampler{2, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
				if id == 2 {
					return tc.row, tc.err
				}
				return []float64{float64(id), 2 * float64(id)}, nil
			}}
			result, err := montecarlo.Run(context.Background(), montecarlo.Plan{
				Paths: 6, Workers: 1, RetainSamples: true, Quantiles: []float64{.5},
			}, sampler)
			mustErrorIs(t, err, tc.want)
			mustNilState(t, result, false, "partial result")
			if result.Completed != 2 || result.Complete || result.Quantiles != nil {
				t.Fatalf("wrong prefix: %+v", result)
			}
			mustNilState(t, result.Samples, false, "retained prefix")
			mustEqualInt(t, result.Samples.Rows(), 2, "retained rows")
			for column, wantMean := range []float64{.5, 1} {
				closeFloat(t, result.Mean[column], wantMean)
				closeFloat(t, result.SampleVariance[column], 2*wantMean*wantMean)
				closeFloat(t, result.MeanStandardError[column], wantMean)
			}
			mustMatrixBits(t, result.Samples, 1, 0, 1)
			mustMatrixBits(t, result.Samples, 1, 1, 2)
		})
	}
}

// TestPrefixOneCommittedRowWithoutMoments distinguishes present samples from
// an insufficient sample count for Bessel-corrected statistics.
// Determinism: path1 returns a terminal error after the sole valid row.
// Complexity: O(d) storage for the visible prefix, full backing retained by Run.
// AI-Hints: never treat Completed==1 as an admissible sample variance.
func TestPrefixOneCommittedRowWithoutMoments(t *testing.T) {
	cause := errors.New("second trial unavailable")
	sampler := &funcSampler{1, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		if id != 0 {
			return nil, cause
		}
		return []float64{42}, nil
	}}
	result, err := montecarlo.Run(context.Background(), montecarlo.Plan{
		Paths: 5, Workers: 1, RetainSamples: true, Quantiles: []float64{0, 1},
	}, sampler)
	mustErrorIs(t, err, cause)
	mustNilState(t, result, false, "partial result")
	if result.Completed != 1 || result.Complete || result.Mean != nil ||
		result.SampleVariance != nil || result.MeanStandardError != nil || result.Quantiles != nil {
		t.Fatalf("invalid one-row prefix: %+v", result)
	}
	mustNilState(t, result.Samples, false, "one retained row")
	mustEqualInt(t, result.Samples.Rows(), 1, "one-row prefix")
	mustMatrixBits(t, result.Samples, 0, 0, 42)
	mustMatrixBits(t, result.Samples.Clone(), 0, 0, 42)
}

// TestCancellationWithAllRowsStillIncomplete exercises the distinction between
// successful reduction count and successful experiment finalization.
// Implementation: cancel inside the final sampler invocation, then return its
// valid vector. The coordinator drains that outcome but must withhold quantiles.
// Determinism: synchronous cancellation on a fixed logical path; no sleeps.
// Complexity: O(Paths) work for the small fixture.
// AI-Hints: Completed==Requested is not sufficient evidence of Complete.
func TestCancellationWithAllRowsStillIncomplete(t *testing.T) {
	const paths = 4
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sampler := &funcSampler{1, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		if id == paths-1 {
			cancel()
		}
		return []float64{float64(id)}, nil
	}}
	result, err := montecarlo.Run(ctx, montecarlo.Plan{
		Paths: paths, Workers: 1, RetainSamples: true, Quantiles: []float64{.5},
	}, sampler)
	mustErrorIs(t, err, context.Canceled)
	mustNilState(t, result, false, "canceled result")
	if result.Completed != paths || result.Requested != paths || result.Complete || result.Quantiles != nil {
		t.Fatalf("invalid canceled full prefix: %+v", result)
	}
	mustNilState(t, result.Samples, false, "retained rows")
	mustEqualInt(t, result.Samples.Rows(), paths, "all committed rows")
	closeFloat(t, result.Mean[0], 1.5)
	closeFloat(t, result.SampleVariance[0], 5.0/3)
}
