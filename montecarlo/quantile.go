// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo implements exact, sign-aware empirical vector quantiles.
// This file sorts one feature at a time without reordering retained path rows.
package montecarlo

import (
	"context"
	"errors"
	"math"
	"sort"

	"github.com/lvlath/go/matrix"
)

// interpolated computes a convex combination of two ordered finite samples.
// Unlike a+f*(b-a), it avoids overflowing the opposite-sign difference;
// for equal-sign values it uses a nonoverflowing difference for accuracy.
//
// Inputs: a<=b are finite order statistics; f is the finite fraction in [0,1].
// Returns: the finite linear interpolation result, up to float64 rounding.
// Determinism: branches depend only on input values and their sign bits.
// Complexity: Time O(1), Space O(1).
// AI-Hints: for [-MaxFloat,+MaxFloat] at f=0.5, b-a is infinite but median is 0.
func interpolated(a, b, fraction float64) float64 {
	if fraction == 0 || a == b {
		return a
	}
	if fraction == 1 {
		return b
	}
	if math.Signbit(a) != math.Signbit(b) {
		// Multiplication is bounded by each finite endpoint; their weighted sum
		// cannot overflow, even when the raw difference b-a would overflow.
		return (1-fraction)*a + fraction*b
	}
	// For same-sign ordered samples |b-a| <= MaxFloat64; interpolation remains
	// between endpoints without an overflowing intermediate.
	return a + fraction*(b-a)
}

// quantileMatrix calculates exact type-7 quantiles for independently sorted features.
// It uses h=(n-1)*p, i=floor(h), f=h-i and interpolates adjacent order statistics.
// One reusable n-element scratch column is counted in the preflight budget.
//
// Implementation:
//   - Stage 1: allocate a q*d Dense and one n-element scratch column.
//   - Stage 2: gather, sort and interpolate each dimension independently.
//   - Stage 3: publish the matrix only after all columns pass numeric checks.
//
// Inputs:
//   - ctx: parent-linked cancellation context (checked per sorted column).
//   - paths: checked positive int row count; dimension: checked positive columns.
//   - levels: private strictly increasing finite interpolation probabilities.
//   - valueAt: owning-storage reader returning a finite scalar or an error.
//
// Returns: detached q*d matrix; nil on any materialization or backend failure.
// Errors: ErrOverflow for unrepresentable quantiles, ErrDomain joined with any
//
//	storage backend error; constructor errors are joined with ErrOverflow.
//
// Determinism: column ascending, row ascending before deterministic numeric sort,
//
//	quantile levels in original ascending order; retained row indices are unchanged.
//
// Complexity: Time O(d*paths*log(paths)+d*q), Space O(paths+d*q).
// Notes: exact quantiles are never computed for partial experiments.
// AI-Hints: do not sort the retained Samples matrix itself or approximate silently.
func quantileMatrix(ctx context.Context, paths, dimension int, levels []float64,
	valueAt func(row, column int) (float64, error),
) (matrix.Matrix, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out, err := matrix.NewDense(len(levels), dimension)
	if err != nil {
		return nil, errors.Join(ErrOverflow, err)
	}
	scratch := make([]float64, paths)

	for column := 0; column < dimension; column++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		// Sorting a private column does not permute original path identifiers.
		for row := 0; row < paths; row++ {
			value, readErr := valueAt(row, column)
			if readErr != nil {
				return nil, errors.Join(ErrDomain, readErr)
			}
			scratch[row] = value
		}
		sort.Float64s(scratch)
		// Standard-library sorting is atomic at the column level; honor a
		// cancellation immediately after its deterministic completion.
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		for quantileIndex, level := range levels {
			position := float64(paths-1) * level
			lower := int(math.Floor(position))
			if lower >= paths-1 {
				// This also protects the p=1 endpoint from lower+1 access.
				lower = paths - 1
			}
			fraction := position - float64(lower)
			value := scratch[lower]
			if lower < paths-1 {
				value = interpolated(value, scratch[lower+1], fraction)
			}
			if !finiteScalar(value) {
				return nil, ErrOverflow
			}
			if err := out.Set(quantileIndex, column, value); err != nil {
				return nil, errors.Join(ErrDomain, err)
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return out, nil
}
