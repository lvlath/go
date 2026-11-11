// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols builds the explicit rectangular regression design and parameter map.
// This file isolates training coordinate transforms from the single QR kernel.
package ols

import (
	"context"
	"errors"
	"math"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/stats"
)

// design constructs the complete training basis and learns optional feature scales.
// It keeps the affine and origin-constrained hypothesis spaces mathematically
// distinct: intercept designs center, while origin-constrained designs never do.
//
// Implementation:
//   - Stage 1: initialize unit scales and, if requested, fit population stats scales.
//   - Stage 2: learn means for unscaled intercept models, or retain the stats means.
//   - Stage 3: allocate n-by-q and fill the explicit basis in stable row-major order.
//
// Behavior highlights:
//   - With an intercept, each feature becomes (x-mean)/scale after a column of ones.
//   - NoIntercept keeps exactly p columns; Standardize transforms only x/scale.
//   - Exact-constant columns use stats.UnitScaleConstant; QR owns rank rejection.
//
// Inputs:
//   - ctx: previously validated active context.
//   - x: finite, validated read-only n-by-p source; unretained after return.
//   - n, p, q: validated positive dimensions, q=p or p+1.
//   - opts: selected intercept, scale, and rank policy (rank used by caller).
//
// Returns:
//   - *matrix.Dense: detached owning design, n rows by q columns.
//   - means/scales: independent train-only p-vectors; zero means for NoIntercept.
//
// Errors:
//   - ErrOverflow: an accepted finite input produces an unrepresentable basis cell.
//   - Wrapped stats.ErrOverflow and ErrOverflow: unrepresentable fitted scale.
//   - ctx.Err(), matrix and stats errors: canceled operation or dependency failure.
//
// Determinism:
//   - Feature centers ascend j then training rows i; design filling is row-major.
//
// Complexity:
//   - Time O(n*p), Space O(n*q+p) including the owning design and fitted vectors.
//
// Notes:
//   - stats.FitStandardizer has no context argument; cancellation is polled after fitting.
//   - A constant feature alongside an intercept is not silently discarded.
//
// AI-Hints:
//   - Never call Standardizer.Transform for NoIntercept: it centers the predictors.
//   - Keep the explicit intercept column instead of centering y or hiding b.
func design(ctx context.Context, x matrix.Matrix, n, p, q int, opts Options) (*matrix.Dense, []float64, []float64, error) {
	means := make([]float64, p)
	scales := make([]float64, p)
	for j := range scales {
		scales[j] = 1
	}

	if opts.Standardize {
		// Use stats as the canonical source of population variance and constant policy.
		fitted, err := stats.FitStandardizer(x, stats.ScaleOptions{DDOF: 0, ConstantPolicy: stats.UnitScaleConstant})
		if err != nil {
			if errors.Is(err, stats.ErrOverflow) {
				return nil, nil, nil, errors.Join(ErrOverflow, err)
			}
			return nil, nil, nil, err
		}
		if err = ctx.Err(); err != nil {
			return nil, nil, nil, err
		}
		copy(scales, fitted.Scales)
		if !opts.NoIntercept {
			copy(means, fitted.Means)
		}
	} else if !opts.NoIntercept {
		// Even with Standardize=false, the explicit intercept basis is
		// centered so the ones column is not mixed with large feature offsets.
		// The online mean recurrence is mean+=(value-mean)/count. Evaluating
		// its two divided terms separately avoids overflowing value-mean for
		// finite observations of opposite signs; FMA rounds the retained mean
		// part only once. No response centering is required because the ones
		// column remains explicit in the subsequent QR design.
		for j := 0; j < p; j++ {
			mean := 0.0
			for i := 0; i < n; i++ {
				if i%contextCheckRowInterval == 0 {
					if err := ctx.Err(); err != nil {
						return nil, nil, nil, err
					}
				}
				value, err := x.At(i, j)
				if err != nil {
					return nil, nil, nil, err
				}
				count := float64(i + 1)
				// Algebraically (1-1/count)*mean + value/count; avoid the
				// unrepresentable intermediate value-mean when magnitudes are huge.
				mean = math.FMA(-mean, 1/count, mean) + value/count
				if !finite(mean) {
					return nil, nil, nil, ErrOverflow
				}
			}
			means[j] = mean
		}
	}

	// The design must own storage so factoring it cannot change caller inputs.
	a, err := matrix.NewDense(n, q)
	if err != nil {
		return nil, nil, nil, err
	}
	for i := 0; i < n; i++ {
		if i%contextCheckRowInterval == 0 {
			if err = ctx.Err(); err != nil {
				return nil, nil, nil, err
			}
		}
		offset := 0
		if !opts.NoIntercept {
			if err = a.Set(i, 0, 1); err != nil {
				return nil, nil, nil, err
			}
			offset = interceptDesignColumns
		}
		for j := 0; j < p; j++ {
			value, e := x.At(i, j)
			if e != nil {
				return nil, nil, nil, e
			}
			normalized, err := normalizePredictor(value, means[j], scales[j])
			if err != nil {
				return nil, nil, nil, err
			}
			if err = a.Set(i, j+offset, normalized); err != nil {
				return nil, nil, nil, err
			}
		}
	}
	return a, means, scales, nil
}

// normalizePredictor computes (value-mean)/scale without avoidable subtraction
// overflow. Its direct path preserves cancellation precision for close values;
// only an already overflowed numerator uses an equivalent split quotient.
//
// Implementation:
//   - Stage 1: subtract finite value and training mean in their original units.
//   - Stage 2: if subtraction is representable, divide the exact rounded delta.
//   - Stage 3: otherwise divide both operands before subtracting, checking output.
//
// Behavior highlights:
//   - The split fallback preserves a finite standardized value when x-mean is
//     mathematically outside float64 although (x-mean)/scale is representable.
//
// Inputs: finite value/mean in physical units; finite positive scale in same units.
// Returns: finite normalized scalar, or zero/error when float64 cannot represent it.
// Errors: ErrOverflow for a nonfinite result in either mathematically equivalent path.
// Determinism: one fixed branch decided by IEEE finiteness of the subtraction.
// Complexity: Time O(1), Space O(1), no allocation.
// Notes: A split quotient may lose relative precision; it is used ONLY when
//
//	the more accurate numerator-first calculation is not representable.
//
// AI-Hints: Checking after `delta/scale` is too late if `delta` already overflowed.
func normalizePredictor(value, mean, scale float64) (float64, error) {
	delta := value - mean
	normalized := 0.0
	if finite(delta) {
		normalized = delta / scale
	} else {
		// Overflow in x-mean does not imply overflow after normalization.
		normalized = value/scale - mean/scale
	}
	if !finite(normalized) {
		return 0, ErrOverflow
	}
	return normalized, nil
}

// convertCoefficients maps the solved QR parameters into original predictor units.
// It algebraically removes train means and scales without introducing an
// intercept for an origin-constrained regression.
//
// Implementation:
//   - Stage 1: read the explicit QR intercept when present.
//   - Stage 2: divide each theta slope by its positive training scale.
//   - Stage 3: subtract sum_j(mean_j*beta_j) from the affine offset using FMA.
//
// Behavior highlights:
//   - NoIntercept forces positive zero without any mean correction.
//   - Every unrepresentable division or correction produces ErrOverflow.
//
// Inputs:
//   - theta: read-only q-by-1 QR solution, guaranteed finite by matrix solver.
//   - p: original number of predictors; means/scales: detached p-vectors.
//   - noIntercept: whether the explicit design omitted its affine column.
//
// Returns: independent p-vector beta, original-unit intercept b, or nil/zero on error.
// Errors: ErrOverflow for unrepresentable coefficients; wrapped theta.At errors.
// Determinism: feature j increasing; the affine correction uses the same order.
// Complexity: Time O(p), Space O(p) for the returned coefficients.
// Notes: Original-unit beta and b must themselves fit in float64 to publish a Model.
// AI-Hints: Never return theta's scaled slopes as the public physical coefficients.
func convertCoefficients(theta matrix.Matrix, p int, means, scales []float64, noIntercept bool) ([]float64, float64, error) {
	beta := make([]float64, p)
	offset := 0
	intercept := 0.0
	if !noIntercept {
		value, err := theta.At(0, 0)
		if err != nil {
			return nil, 0, err
		}
		intercept = value
		offset = interceptDesignColumns
	}
	for j := 0; j < p; j++ {
		value, err := theta.At(j+offset, 0)
		if err != nil {
			return nil, 0, err
		}
		beta[j] = value / scales[j]
		if !finite(beta[j]) {
			return nil, 0, ErrOverflow
		}
	}
	if !noIntercept {
		for j := 0; j < p; j++ {
			intercept = math.FMA(-means[j], beta[j], intercept)
			if !finite(intercept) {
				return nil, 0, ErrOverflow
			}
		}
	}
	return beta, intercept, nil
}
