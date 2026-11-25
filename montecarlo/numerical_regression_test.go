// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo_test checks mathematical and floating-point invariants.
// All statistical and quantile oracles consume the public simulation result.
package montecarlo_test

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/montecarlo"
	"github.com/lvlath/go/rng"
	"github.com/lvlath/go/stats"
)

// TestReductionRationalExtremeScale compares ordered reduction against an
// independently computed rational mean and sample variance.
// The second central moment may exceed MaxFloat64 before division by n-1.
// Implementation: use three exact binary64 observations with catastrophic raw-sum
// cancellation and extreme spread; compare finite summaries for multiple workers.
// Determinism: the sampler depends only on path ID; exact mean uses a bit oracle.
// Complexity: O(n) driver updates, bounded arbitrary-precision test arithmetic.
// AI-Hints: an overflowing intermediate M2 is not a final variance overflow.
func TestReductionRationalExtremeScale(t *testing.T) {
	const tolerance = 5e-14
	for _, tc := range []struct {
		name   string
		values []float64
	}{
		{"finite-variance-overflowing-m2", []float64{-1e154, 0, 1e154}},
		{"exact-mean-survives-cancellation", []float64{1e154, 1, -1e154}},
		{"high-offset-small-spread", []float64{1e12 + 1, 1e12 + 2, 1e12 + 3}},
		{"tiny-normal-variance", []float64{-1e-150, 0, 1e-150}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wantMean, wantVariance := rationalSampleMoments(t, tc.values)
			// The direct quotient is representable for each selected oracle.
			wantSE := math.Sqrt(wantVariance / float64(len(tc.values)))
			sampler := &funcSampler{1, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
				return []float64{tc.values[id]}, nil
			}}
			for _, workers := range []int{1, 2, 4} {
				result, err := montecarlo.Run(context.Background(), montecarlo.Plan{
					Paths: uint64(len(tc.values)), Workers: workers,
				}, sampler)
				if err != nil || result == nil || !result.Complete {
					t.Fatalf("workers=%d: result=%+v err=%v", workers, result, err)
				}
				if math.Float64bits(result.Mean[0]) != math.Float64bits(wantMean) {
					t.Fatalf("mean %.17g want exact rounded %.17g", result.Mean[0], wantMean)
				}
				mustRelativeFinite(t, result.SampleVariance[0], wantVariance, tolerance, "sample variance")
				mustRelativeFinite(t, result.MeanStandardError[0], wantSE, tolerance, "mean standard error")
			}
		})
	}
}

// TestReductionDistinctColumnsAcrossBlocks guards independent accumulator ownership.
// Analytic [scale,2*scale,3*scale] columns cross allocation-block boundaries.
// Implementation: vary dimensions around a common machine-cache-sized block and
// verify every coordinate, not only the first and last.
// Determinism: expected mean=2*scale, sample variance=scale^2 and median=2*scale.
// Complexity: O(3*d) reduction and O(d) state; exact quantiles use a tiny scratch.
// AI-Hints: a pointer table must never make neighboring columns share a Moments.
func TestReductionDistinctColumnsAcrossBlocks(t *testing.T) {
	for _, dimension := range []int{1, 63, 64, 65, 128, 129} {
		t.Run(fmt.Sprintf("dimension-%d", dimension), func(t *testing.T) {
			sampler := &funcSampler{dimension, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
				row := make([]float64, dimension)
				for column := range row {
					row[column] = float64(id+1) * float64(column+1)
				}
				return row, nil
			}}
			result, err := montecarlo.Run(context.Background(), montecarlo.Plan{
				Paths: 3, Workers: 3, RetainSamples: true, Quantiles: []float64{.5},
			}, sampler)
			if err != nil || result == nil || !result.Complete {
				t.Fatalf("result=%+v err=%v", result, err)
			}
			mustEqualInt(t, len(result.Mean), dimension, "mean dimension")
			mustEqualInt(t, len(result.SampleVariance), dimension, "variance dimension")
			mustEqualInt(t, len(result.MeanStandardError), dimension, "SE dimension")
			for column := 0; column < dimension; column++ {
				scale := float64(column + 1)
				closeFloat(t, result.Mean[column], 2*scale)
				closeFloat(t, result.SampleVariance[column], scale*scale)
				closeFloat(t, result.MeanStandardError[column], scale/math.Sqrt(3))
				mustMatrixBits(t, result.Quantiles, 0, column, 2*scale)
				mustMatrixBits(t, result.Samples, 2, column, 3*scale)
			}
		})
	}
}

// TestQuantileSubnormalHalfwayRounding is a regression for premature product
// rounding during interpolation. All variances in this fixture round to zero,
// which is a valid summary and must not suppress representable quantiles.
// Implementation: exercise equal-sign and opposite-sign adjacent subnormals
// at both endpoints and the midpoint, in both optional storage modes.
// Determinism: exact binary64 tie-to-even expectations, including signed zero.
// Complexity: O(d) arithmetic for two rows; no approximate quantile sketch.
// AI-Hints: use Float64bits; a conventional absolute tolerance hides this defect.
func TestQuantileSubnormalHalfwayRounding(t *testing.T) {
	u := math.SmallestNonzeroFloat64
	// Rows are deliberately not sorted in every column.
	// Normal values adjacent to the underflow boundary need the same care.
	const minNormal = 0x1p-1022
	first := []float64{u, -u, -u, u, -u, -(minNormal + u)}
	second := []float64{2 * u, -2 * u, 2 * u, -2 * u, u, minNormal + 2*u}
	lower := []float64{u, -2 * u, -u, -2 * u, -u, -(minNormal + u)}
	upper := []float64{2 * u, -u, 2 * u, u, u, minNormal + 2*u}
	midpoint := []float64{2 * u, -2 * u, 0, math.Copysign(0, -1), 0, 0}
	for _, retain := range []bool{false, true} {
		t.Run(fmt.Sprintf("retain-%t", retain), func(t *testing.T) {
			sampler := &funcSampler{len(first), func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
				if id == 0 {
					return first, nil
				}
				return second, nil
			}}
			result, err := montecarlo.Run(context.Background(), montecarlo.Plan{
				Paths: 2, Workers: 2, RetainSamples: retain, Quantiles: []float64{0, .5, 1},
			}, sampler)
			if err != nil || result == nil || !result.Complete {
				t.Fatalf("result=%+v err=%v", result, err)
			}
			for column := range first {
				mustMatrixBits(t, result.Quantiles, 0, column, lower[column])
				mustMatrixBits(t, result.Quantiles, 1, column, midpoint[column])
				mustMatrixBits(t, result.Quantiles, 2, column, upper[column])
				if result.SampleVariance[column] != 0 || result.MeanStandardError[column] != 0 {
					t.Fatal("subnormal squared variance must round to zero")
				}
				if retain {
					mustMatrixBits(t, result.Samples, 0, column, first[column])
					mustMatrixBits(t, result.Samples, 1, column, second[column])
				}
			}
			mustNilState(t, result.Samples, !retain, "samples publication")
		})
	}
}

// TestReductionOverflowPreservesStatsCause tests the actual dependency error,
// not merely a matching Monte Carlo sentinel or a fabricated accumulator state.
// The first coordinate is representable; the last coordinate's variance is not.
// Determinism: all observations are valid, and failure occurs during summary.
// Complexity: O(d) finalization and a two-row sample fixture.
// AI-Hints: failure after all rows are reduced must not publish quantiles or
// partially initialized moment vectors.
func TestReductionOverflowPreservesStatsCause(t *testing.T) {
	sampler := &funcSampler{2, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		if id == 0 {
			return []float64{5, -math.MaxFloat64}, nil
		}
		return []float64{9, math.MaxFloat64}, nil
	}}
	result, err := montecarlo.Run(context.Background(), montecarlo.Plan{
		Paths: 2, RetainSamples: true, Quantiles: []float64{.5},
	}, sampler)
	mustErrorIs(t, err, montecarlo.ErrOverflow)
	mustErrorIs(t, err, stats.ErrOverflow)
	mustNilState(t, result, false, "failed summary result")
	if result.Completed != 2 || result.Requested != 2 || result.Complete || result.Quantiles != nil ||
		result.Mean != nil || result.SampleVariance != nil || result.MeanStandardError != nil {
		t.Fatalf("numeric failure leaked a statistic: %+v", result)
	}
	mustMatrixBits(t, result.Samples, 0, 1, -math.MaxFloat64)
	mustMatrixBits(t, result.Samples, 1, 1, math.MaxFloat64)
}

// TestPublicModesOrderStatisticOracle validates independent quantile ranks and
// moment semantics across the canonical Run and every convenience facade.
// The fixture has shuffled, sign-crossing and repeated order statistics.
// Implementation:
//   - Stage 1: run mean-only wrappers and validate rational moment references.
//   - Stage 2: request quantiles alone and samples with quantiles through wrappers.
//   - Stage 3: verify exact type-7 order statistics and the original row order.
//
// Behavior highlights: wrappers must preserve the caller's seed and unrelated
// plan options; quantile levels supplied to RunWithQuantiles replace old levels.
// Inputs: four deterministic finite rows, seven dyadic probabilities.
// Returns: no value; checks result ownership and publication, not error strings.
// Errors: any unexpected execution or matrix access failure fails the test.
// Determinism: path-based fixture and independently derived exact ranks.
// Complexity: O(n*d log n + q*d) for quantiles; fixed small fixture.
// AI-Hints: do not compare two wrappers alone: both may share the same bug.
func TestPublicModesOrderStatisticOracle(t *testing.T) {
	const paths = 4
	const seed = 73
	const tolerance = 5e-14
	rows := [paths][2]float64{{9, 11}, {-9, 7}, {3, 11}, {-3, 7}}
	levels := []float64{0, .125, .25, .5, .75, .875, 1}
	// Sorted columns are [-9,-3,3,9] and [7,7,11,11]. For n=4, h=3*p.
	wantQuantiles := [7][2]float64{{-9, 7}, {-6.75, 7}, {-4.5, 7}, {0, 9}, {4.5, 11}, {6.75, 11}, {9, 11}}
	sampler := &funcSampler{2, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		return []float64{rows[id][0], rows[id][1]}, nil
	}}
	ctx := context.Background()
	plan := montecarlo.Plan{Paths: paths, Seed: seed, Workers: 3, Quantiles: levels}
	for _, tc := range []struct {
		name      string
		call      func() (*montecarlo.Result, error)
		retained  bool
		quantiles bool
	}{
		{"estimate", func() (*montecarlo.Result, error) {
			return montecarlo.Estimate(ctx, paths, seed, sampler)
		}, false, false},
		{"estimate-parallel", func() (*montecarlo.Result, error) {
			return montecarlo.EstimateParallel(ctx, paths, seed, 2, sampler)
		}, false, false},
		{"quantiles-wrapper", func() (*montecarlo.Result, error) {
			// The supplied levels must replace the invalid levels in this copy.
			replaced := plan
			replaced.Quantiles = []float64{.5, .5}
			return montecarlo.RunWithQuantiles(ctx, replaced, levels, sampler)
		}, false, true},
		{"samples-wrapper-preserves-quantiles", func() (*montecarlo.Result, error) {
			return montecarlo.RunWithSamples(ctx, plan, sampler)
		}, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, err := tc.call()
			if err != nil || result == nil || !result.Complete || result.Requested != paths || result.Completed != paths || result.Seed != seed {
				t.Fatalf("unexpected mode result: %+v %v", result, err)
			}
			mustEqualString(t, result.RNG, rng.NewFactory().Version(), "RNG revision")
			for column := 0; column < 2; column++ {
				values := make([]float64, paths)
				for row := range rows {
					values[row] = rows[row][column]
				}
				mean, variance := rationalSampleMoments(t, values)
				mustRelativeFinite(t, result.Mean[column], mean, tolerance, "mean")
				mustRelativeFinite(t, result.SampleVariance[column], variance, tolerance, "variance")
				mustRelativeFinite(t, result.MeanStandardError[column], math.Sqrt(variance/paths), tolerance, "SE")
			}
			mustNilState(t, result.Samples, !tc.retained, "sample mode")
			mustNilState(t, result.Quantiles, !tc.quantiles, "quantile mode")
			if tc.retained {
				mustEqualInt(t, result.Samples.Rows(), paths, "sample rows")
				mustEqualInt(t, result.Samples.Cols(), 2, "sample columns")
				for row, values := range rows {
					for column, want := range values {
						mustMatrixBits(t, result.Samples, row, column, want)
					}
				}
			}
			if tc.quantiles {
				mustEqualInt(t, len(result.QuantileLevels), len(levels), "level count")
				mustEqualInt(t, result.Quantiles.Rows(), len(levels), "quantile rows")
				mustEqualInt(t, result.Quantiles.Cols(), 2, "quantile columns")
				for row, values := range wantQuantiles {
					mustEqualBool(t, result.QuantileLevels[row] == levels[row], true, "level identity")
					for column, want := range values {
						mustMatrixBits(t, result.Quantiles, row, column, want)
					}
				}
			} else {
				mustEqualInt(t, len(result.QuantileLevels), 0, "disabled levels")
			}
		})
	}
}
