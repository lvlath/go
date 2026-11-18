// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo reduces vector observations in strict path-index order.
// This file delegates stable central moments to the existing stats dependency.
package montecarlo

import (
	"errors"
	"math"

	"github.com/lvlath/go/stats"
)

// reducer owns one compensated, extended-range accumulator per vector coordinate.
// A single coordinator alone mutates it, and no worker owns partial moments.
type reducer struct {
	columns []stats.Moments
	count   uint64
}

// newReducer allocates fixed vector state once, independent of Paths.
// It preserves the same stats.Moments zero-state used by sequential Summarize.
//
// Inputs: dimension is a validated strictly positive feature count.
// Returns: private zero-state reducer, owned by the coordinator.
// Determinism: columns are assigned in ascending coordinate order.
// Complexity: Time O(d), Space O(d).
// AI-Hints: no additional full-sample storage belongs in reducer.
func newReducer(dimension int) *reducer {
	return &reducer{columns: make([]stats.Moments, dimension)}
}

// add incorporates one finite vector using the dependency's scaled Welford law.
// Row order comes exclusively from nextReduce; the accumulator does not know
// workers, completion times, or quantile storage details.
//
// Inputs: row is a fixed-dimensional finite worker-detached vector.
// Returns: nil after the complete row; stats errors preserve dependency identity.
// Determinism: ascending coordinate updates within ascending logical paths.
// Complexity: Time O(d), Space O(1) additional.
// AI-Hints: do not use ordinary sums or merge independent worker accumulators.
func (r *reducer) add(row []float64) error {
	for column, value := range row {
		if err := r.columns[column].Update(value); err != nil {
			if errors.Is(err, stats.ErrOverflow) {
				return errors.Join(ErrOverflow, err)
			}
			if errors.Is(err, stats.ErrNonFinite) {
				return errors.Join(ErrNonFinite, err)
			}
			return errors.Join(ErrDomain, err)
		}
	}
	r.count++
	return nil
}

// summarize converts all coordinate moments to detached vectors atomically.
// It never publishes a partial vector when any coordinate is unrepresentable;
// standard error uses sqrt(variance)/sqrt(count) to avoid intermediate underflow.
//
// Implementation:
//   - Stage 1: preserve nil moment vectors for fewer than two reduced paths.
//   - Stage 2: obtain sample variance (ddof=1) for each coordinate from stats.
//   - Stage 3: publish all three vectors together only after every check succeeds.
//
// Inputs: out is the newly constructed, coordinator-owned Result.
// Returns: nil on success, or a sentinel-preserving wrapped stats error.
// Determinism: ascending coordinate materialization and one fixed rounding path.
// Complexity: Time O(d), Space O(d) temporary and returned vectors.
// Notes: MeanStandardError is sampling uncertainty, not physical model uncertainty.
// AI-Hints: sqrt(variance/count) can round to zero before sqrt at subnormal scale.
func (r *reducer) summarize(out *Result) error {
	if r.count < minimumPaths {
		return nil
	}

	// Temporary arrays remain private until every coordinate is successful.
	dimension := len(r.columns)
	means := make([]float64, dimension)
	variances := make([]float64, dimension)
	standardErrors := make([]float64, dimension)
	countScale := math.Sqrt(float64(r.count))

	for column := range r.columns {
		summary, err := r.columns[column].Summary(1)
		if err != nil {
			if errors.Is(err, stats.ErrOverflow) {
				return errors.Join(ErrOverflow, err)
			}
			return errors.Join(ErrDomain, err)
		}
		means[column] = summary.Mean
		variances[column] = summary.Variance
		standardErrors[column] = math.Sqrt(summary.Variance) / countScale
		if !finiteScalar(standardErrors[column]) {
			return ErrOverflow
		}
	}

	out.Mean = means
	out.SampleVariance = variances
	out.MeanStandardError = standardErrors
	return nil
}
