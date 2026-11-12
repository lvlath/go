// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge constructs the regularized rectangular QR problem.
// It owns the exact zero-intercept-penalty and scale-only design laws.
package ridge

import (
	"context"
	"errors"
	"math"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/stats"
)

// augmentedWeight evaluates sqrt(n*Lambda) without forming n*Lambda first.
// This avoids a spurious infinity or zero from intermediate binary64 multiplication.
//
// Implementation:
//   - Stage 1: take the square root of the positive integer observation count.
//   - Stage 2: take the square root of the positive regularization coefficient.
//   - Stage 3: multiply both roots and validate the representable final result.
//
// Behavior highlights:
//   - The result is strictly positive for all accepted positive inputs.
//
// Inputs:
//   - n: validated positive training observation count.
//   - lambda: validated finite strictly positive regularization coefficient.
//
// Returns:
//   - float64: positive finite augmented diagonal weight, or zero/error.
//
// Errors:
//   - ErrOverflow: the final weight cannot be represented as a positive finite float64.
//
// Determinism:
//   - Fixed operation order: sqrt(n), sqrt(lambda), multiplication, classification.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - A representable weight does not guarantee numerical rank in Householder QR.
//
// AI-Hints:
//   - Never replace this with math.Sqrt(float64(n)*lambda): n*lambda may overflow.
//   - Very small lambda must not be silently increased to satisfy the rank criterion.
func augmentedWeight(n int, lambda float64) (float64, error) {
	weight := math.Sqrt(float64(n)) * math.Sqrt(lambda)
	if !finite(weight) || weight == 0 {
		return 0, ErrOverflow
	}
	return weight, nil
}

// normalized constructs a feature coordinate from physical value, center and scale.
// It preserves finite standardized differences even when value-mean overflows.
//
// Implementation:
//   - Stage 1: evaluate the difference before division for normal precision.
//   - Stage 2: only if that difference is unrepresentable, divide both terms first.
//   - Stage 3: reject nonfinite normalized values rather than publish invalid design.
//
// Behavior highlights:
//   - NoIntercept supplies mean=0, so the same function implements scale-only.
//
// Inputs:
//   - value: finite feature in physical units; mean: permitted train center.
//   - scale: finite positive train standard deviation, or the unit constant scale.
//
// Returns:
//   - float64: finite transformed feature, or ErrOverflow.
//
// Errors:
//   - ErrOverflow: the resulting normalized coordinate cannot fit in float64.
//
// Determinism:
//   - Fixed difference-first path, with division-first only on subtraction overflow.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Division-first loses some low-order bits; it is a last-resort finite branch.
//
// AI-Hints:
//   - Checking finiteness only after dividing an overflowing numerator is insufficient.
func normalized(value, mean, scale float64) (float64, error) {
	delta := value - mean
	var result float64
	if finite(delta) {
		result = delta / scale
	} else {
		// The two quotients are representable more often than the raw
		// opposite-sign difference; this branch is entered only when needed.
		result = value/scale - mean/scale
	}
	if !finite(result) {
		return 0, ErrOverflow
	}
	return result, nil
}

// trainingTransform fits deterministic feature centers and population scales.
// It freezes the hypothesis-space distinction between affine and origin fits.
//
// Implementation:
//   - Stage 1: initialize zero centers and unit scales for every predictor.
//   - Stage 2: fit stats scales when requested, copying centers only with intercept.
//   - Stage 3: otherwise center affine predictors to decouple them from the ones column.
//
// Behavior highlights:
//   - NoIntercept never subtracts a training mean, including when Standardize=true.
//   - Standardize uses stats.UnitScaleConstant with DDOF=0; constants are retained.
//
// Inputs:
//   - ctx: active context; x: validated finite n-by-p read-only Matrix.
//   - n,p: positive training dimensions; opts: validated intercept/scaling policy.
//
// Returns:
//   - means, scales: independent p-vectors, with positive finite scales.
//
// Errors:
//   - ErrOverflow with stats.ErrOverflow on an unrepresentable fitted scale.
//   - ErrOverflow for a nonfinite manually fitted mean or scale.
//   - ctx.Err() or retained stats/Matrix.At errors.
//
// Determinism:
//   - Stats traverses rows/columns in its fixed order; unscaled means scan columns then rows.
//
// Complexity:
//   - Time O(n*p), Space O(p) excluding the stats fit's temporary storage.
//
// Notes:
//   - stats.FitStandardizer has no ctx; cancellation is checked immediately afterward.
//   - Means and scales are copied again into Model before publication.
//
// AI-Hints:
//   - Do not invoke Standardizer.Transform with NoIntercept: that would center predictors.
//   - Do not drop constant columns; positive ridge penalty makes them identifiable.
func trainingTransform(ctx context.Context, x matrix.Matrix, n, p int, opts Options) ([]float64, []float64, error) {
	means := make([]float64, p)
	scales := make([]float64, p)
	for j := range scales {
		scales[j] = 1
	}

	if opts.Standardize {
		// stats owns the population moments and constant-feature policy.
		fitted, err := stats.FitStandardizer(x, stats.ScaleOptions{
			DDOF:           0,
			ConstantPolicy: stats.UnitScaleConstant,
		})
		if err != nil {
			if errors.Is(err, stats.ErrOverflow) {
				return nil, nil, errors.Join(ErrOverflow, err)
			}
			return nil, nil, err
		}
		if err = ctx.Err(); err != nil {
			return nil, nil, err
		}
		// The dependency promises one fitted mean and scale per feature.
		// Defend this boundary explicitly so malformed metadata cannot
		// silently retain a unit scale or create a wrong physical model.
		if fitted == nil || len(fitted.Means) != p || len(fitted.Scales) != p {
			return nil, nil, ErrShape
		}
		copy(scales, fitted.Scales)
		if !opts.NoIntercept {
			copy(means, fitted.Means)
		}
	} else if !opts.NoIntercept {
		// Internal centering is legal with an explicit intercept even when
		// standardization is disabled: the original affine model is recovered
		// after solving. FMA avoids an unnecessary (value-mean) overflow.
		for j := 0; j < p; j++ {
			mean := 0.0
			for i := 0; i < n; i++ {
				if i%contextCheckInterval == 0 {
					if err := ctx.Err(); err != nil {
						return nil, nil, err
					}
				}
				value, err := x.At(i, j)
				if err != nil {
					return nil, nil, err
				}
				count := float64(i + 1)
				mean = math.FMA(-mean, 1/count, mean) + value/count
				if !finite(mean) {
					return nil, nil, ErrOverflow
				}
			}
			means[j] = mean
		}
	}
	for _, scale := range scales {
		if !finite(scale) || scale <= 0 {
			return nil, nil, ErrOverflow
		}
	}
	return means, scales, nil
}

// buildDesign assembles the one canonical (n+p)-by-q regularized QR system.
// The penalty block always has zeros in the explicit intercept column.
//
// Implementation:
//   - Stage 1: learn training transform and a finite sqrt(n*Lambda) weight.
//   - Stage 2: allocate independent augmented design and one-column response matrices.
//   - Stage 3: fill real rows then the diagonal slope penalties, keeping RHS tail zero.
//
// Behavior highlights:
//   - With intercept, top block is [1, Z] and bottom is [0, sqrt(n*Lambda)*I].
//   - Without intercept, top block is Z and bottom is sqrt(n*Lambda)*I.
//   - Every lower RHS element remains exactly +0.0 from zero-initialized storage.
//
// Inputs:
//   - ctx: active context; x/y: validated read-only finite n-row observations.
//   - n,p: validated positive dimensions with checked augmented allocation size.
//   - opts: validated regularization, intercept and standardization selections.
//
// Returns:
//   - a,b: independent owning *matrix.Dense systems, and detached means/scales.
//
// Errors:
//   - ErrOverflow from weight/transform; ctx.Err(), stats/matrix errors preserved.
//
// Determinism:
//   - Actual rows i ascending, feature columns j ascending; then diagonal j ascending.
//
// Complexity:
//   - Time O(n*p+p), Space O((n+p)*q+n+p+p) for owning design and RHS.
//
// Notes:
//   - The solver additionally snapshots this design; no raw Dense internals are accessed.
//   - No complete Gram matrix is formed and no OLS branch is available.
//
// AI-Hints:
//   - Never put sqrt(n*Lambda) in column zero when the intercept exists.
//   - Avoid undocumented raw-data accessors: matrix.Dense owns and validates its storage.
func buildDesign(ctx context.Context, x matrix.Matrix, y []float64, n, p int, opts Options) (*matrix.Dense, *matrix.Dense, []float64, []float64, error) {
	if err := checkAugmentedSize(n, p, opts.NoIntercept); err != nil {
		return nil, nil, nil, nil, err
	}
	means, scales, err := trainingTransform(ctx, x, n, p, opts)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	weight, err := augmentedWeight(n, opts.Lambda)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	q, offset := p, 0
	if !opts.NoIntercept {
		q += interceptColumns
		offset = interceptColumns
	}
	rows := n + p
	a, err := matrix.NewDense(rows, q)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	b, err := matrix.NewDense(rows, 1)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	// Only the original n rows carry observations; no row is reweighted.
	for i := 0; i < n; i++ {
		if i%contextCheckInterval == 0 {
			if err = ctx.Err(); err != nil {
				return nil, nil, nil, nil, err
			}
		}
		if offset != 0 {
			if err = a.Set(i, 0, 1); err != nil {
				return nil, nil, nil, nil, err
			}
		}
		for j := 0; j < p; j++ {
			if j%contextCheckInterval == 0 {
				if err = ctx.Err(); err != nil {
					return nil, nil, nil, nil, err
				}
			}
			value, readErr := x.At(i, j)
			if readErr != nil {
				return nil, nil, nil, nil, readErr
			}
			value, err = normalized(value, means[j], scales[j])
			if err != nil {
				return nil, nil, nil, nil, err
			}
			if err = a.Set(i, j+offset, value); err != nil {
				return nil, nil, nil, nil, err
			}
		}
		if err = b.Set(i, 0, y[i]); err != nil {
			return nil, nil, nil, nil, err
		}
	}

	// Explicitly write only slope columns: the intercept's p penalty
	// entries and the response tail stay at their exact initialized zero.
	for j := 0; j < p; j++ {
		if j%contextCheckInterval == 0 {
			if err = ctx.Err(); err != nil {
				return nil, nil, nil, nil, err
			}
		}
		if err = a.Set(n+j, j+offset, weight); err != nil {
			return nil, nil, nil, nil, err
		}
	}
	return a, b, means, scales, nil
}
