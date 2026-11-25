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

// reducerBlockColumns bounds each independently allocated accumulator block.
// The feature table then needs only one pointer per coordinate, avoiding an
// unchecked dimension*sizeof(stats.Moments) allocation without unsafe.Sizeof.
const reducerBlockColumns = 64

// reducer owns one compensated, extended-range accumulator per vector coordinate.
// A single coordinator alone mutates it, and no worker owns partial moments.
// Implementation:
//   - Stage 1: allocate a checked-size feature pointer table and bounded blocks.
//   - Stage 2: update the same accumulator for each feature in path order.
//   - Stage 3: materialize all statistics atomically after reduction.
//
// Behavior highlights: no per-path accumulator allocation or partition merging.
// Complexity: O(d) state; O(d) work for each row.
// AI-Hints: pointer targets must remain distinct; never reuse one Moments object.
type reducer struct {
	columns []*stats.Moments // Distinct accumulator per original feature index.
	count   uint64           // Number of fully committed path vectors.
}

// newReducer allocates fixed vector state once, independent of Paths.
// Bounded contiguous blocks preserve zero-valued stats.Moments and avoid one
// dimension-scaled allocation whose element size belongs to another package.
// Implementation:
//   - Stage 1: allocate the pointer table; preflight has checked d*8 bytes.
//   - Stage 2: allocate at most reducerBlockColumns moments per block.
//   - Stage 3: attach distinct pointers in ascending feature order.
//
// Behavior highlights:
//   - No unsafe or reflection is needed to discover the dependency's struct size.
//   - Only one pointer-table allocation and ceil(d/block size) block allocations.
//
// Inputs: dimension is a validated strictly positive feature count.
// Returns: private zero-state reducer, owned by the coordinator.
// Errors: none after arithmetic admission; physical allocation is runtime-owned.
// Determinism: columns are assigned in ascending coordinate order.
// Complexity: Time O(d), Space O(d), including the pointer table.
// Notes: the pointer table adds one pointer per feature, not per sample.
// AI-Hints: do not allocate one full sample matrix or share accumulator pointers.
func newReducer(dimension int) *reducer {
	// A pointer is at most eight bytes on supported 32-/64-bit Go platforms.
	columns := make([]*stats.Moments, dimension)
	for offset := 0; offset < dimension; {
		// Subtract before adding so the final, possibly short block cannot wrap.
		blockLength := min(reducerBlockColumns, dimension-offset)
		block := make([]stats.Moments, blockLength)
		// Taking each element's address preserves its unique zero-valued state.
		for index := range block {
			columns[offset+index] = &block[index]
		}
		offset += blockLength
	}
	return &reducer{columns: columns}
}

// add incorporates one finite vector using the dependency's scaled Welford law.
// Row order comes exclusively from nextReduce; the accumulator does not know
// workers, completion times, or quantile storage details.
// Implementation:
//   - Stage 1: consume the already validated, detached row in feature order.
//   - Stage 2: preserve dependency error identity if an invariant fails.
//   - Stage 3: advance the committed count after the entire row succeeds.
//
// Inputs: row is a fixed-dimensional finite worker-detached vector.
// Returns: nil after the complete row; stats errors preserve dependency identity.
// Errors: joined ErrOverflow, ErrNonFinite or ErrDomain for dependency failures.
// Determinism: ascending coordinate updates within ascending logical paths.
// Complexity: Time O(d), Space O(1) additional.
// Notes: worker validation and bounded path IDs exclude ordinary Update errors.
// AI-Hints: do not use ordinary sums or merge independent worker accumulators.
func (r *reducer) add(row []float64) error {
	for column, value := range row {
		// Each column has exactly the same count before accepting this path.
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
	// No valid path can increment an already maximal uint64 count.
	r.count++
	return nil
}

// summarize converts all coordinate moments to detached vectors atomically.
// It never publishes a partial vector when any coordinate is unrepresentable;
// standard error uses sqrt(variance)/sqrt(count) to avoid intermediate underflow.
// Implementation:
//   - Stage 1: preserve nil moment vectors for fewer than two reduced paths.
//   - Stage 2: obtain sample variance (ddof=1) for each coordinate from stats.
//   - Stage 3: publish all three vectors together only after every check succeeds.
//
// Inputs: out is the newly constructed, coordinator-owned Result.
// Returns: nil on success, or a sentinel-preserving wrapped stats error.
// Errors: ErrOverflow for unrepresentable summary or SE; ErrDomain for other dependency failures.
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
		// Summary uses the exact count and the sample-variance denominator.
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

	// No coordinate is published until all dependency summaries are accepted.
	out.Mean = means
	out.SampleVariance = variances
	out.MeanStandardError = standardErrors
	return nil
}
