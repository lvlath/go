// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf factors the Toeplitz prediction system with Durbin-Levinson recursion.
// This file owns the canonical reflection-coefficient and error-variance kernel.
package pacf

import (
	"context"
	"math"
)

// levinson calculates a PACF curve and physical prediction-error variances.
// Each reflection coefficient solves the next Yule-Walker order using a private
// previous coefficient row; no update can overwrite an operand of that same order.
//
// Implementation:
//   - Stage 1: allocate result and two disjoint coefficient rows; publish lag zero privately.
//   - Stage 2: for m=1..L, compute kappa from gamma and the previous coefficients.
//   - Stage 3: update the next row, decrease positive variance, and swap row ownership.
//   - Stage 4: check cancellation before publishing the complete detached Result.
//
// Behavior highlights:
//   - kappa[m]=(gamma[m]-sum_j(a[m-1,j]*gamma[m-j]))/V[m-1].
//   - a[m,j]=a[m-1,j]-kappa[m]*a[m-1,m-j] for j<m; a[m,m]=kappa[m].
//   - V[m]=V[m-1]*(1-|kappa[m]|)*(1+|kappa[m]|).
//   - No variance clipping, fallback estimators, or partial-result publication exist.
//
// Inputs:
//   - ctx: active context checked at lag boundaries and in both quadratic loops.
//   - gamma: private finite common-denominator covariances with gamma[0]>0.
//   - scale: positive physical scale used by the covariance frame.
//   - count: number of original samples; retained as Result.Count.
//
// Returns:
//   - *Result: complete detached coefficients and positive physical variances.
//   - error: nil on success; nil *Result on any singular, overflow, or context failure.
//
// Errors:
//   - ErrZeroVariance: invalid nonpositive initial lag-zero energy.
//   - ErrSingular: nonpositive prediction variance or reflection magnitude >= 1.
//   - ErrOverflow: nonfinite coefficient arithmetic or unrepresentable physical variance.
//   - ctx.Err(): cancellation at any cooperative checkpoint, without output publication.
//
// Determinism:
//   - Orders m ascend, numerator products scan j ascending, and new coefficients scan j ascending.
//   - The previous row is read-only during each order; buffers swap only after full success.
//
// Complexity:
//   - Time O(L^2), Space O(L+1) for gamma-external output and two coefficient rows.
//   - Total package cost including covariance construction is O(n*(L+1)+L^2).
//
// Notes:
//   - Exact Toeplitz PSD follows from a common denominator; binary64 near-singularity
//     may still force conservative ErrSingular rather than publishing unreliable values.
//   - The solver is deliberately unexported; public users supply samples through Compute.
//
// AI-Hints:
//   - Never update prev in place: j and m-j share previous-row dependencies.
//   - Do not replace (1-|kappa|)*(1+|kappa|) with a rounded 1-kappa*kappa near one.
//   - Do not infer exact finite-sample AR(p) cutoff from a theoretical AR(p) model.
func levinson(ctx context.Context, gamma []float64, scale float64, count int) (*Result, error) {
	maxLag := len(gamma) - 1
	result := &Result{
		Values:             make([]float64, maxLag+1),
		PredictionVariance: make([]float64, maxLag+1),
		Count:              count,
	}

	// Ping-pong rows avoid overwriting a coefficient referenced by its mirror index.
	// Index zero is unused: row positions correspond to AR order indices 1..m.
	previous := make([]float64, maxLag+1)
	next := make([]float64, maxLag+1)

	// Lag zero is defined by the whole-window covariance, not an estimator choice.
	variance := gamma[0]
	if variance <= 0 || math.IsNaN(variance) || math.IsInf(variance, 0) {
		return nil, ErrZeroVariance
	}
	physicalVariance, err := restoreVariance(variance, scale)
	if err != nil {
		return nil, err
	}
	result.Values[0] = 1
	result.PredictionVariance[0] = physicalVariance

	for order := 1; order <= maxLag; order++ {
		// A failed earlier order cannot be bypassed by asking for a higher-order PACF.
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if variance <= 0 || math.IsNaN(variance) || math.IsInf(variance, 0) {
			return nil, ErrSingular
		}

		// The reflection numerator is a compensated Yule-Walker residual.
		// Every coefficient comes from the completed previous row.
		var numerator compensation
		numerator.add(gamma[order])
		for j := 1; j < order; j++ {
			if j%cancellationPollInterval == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			product := previous[j] * gamma[order-j]
			if math.IsNaN(product) || math.IsInf(product, 0) {
				return nil, ErrOverflow
			}
			numerator.add(-product)
		}

		residual := numerator.value()
		if math.IsNaN(residual) || math.IsInf(residual, 0) {
			return nil, ErrOverflow
		}
		reflection := residual / variance
		if math.IsNaN(reflection) || math.IsInf(reflection, 0) {
			return nil, ErrOverflow
		}
		magnitude := math.Abs(reflection)
		if magnitude >= 1 {
			// A zero or negative next variance must never be made positive by clamping.
			return nil, ErrSingular
		}

		// Factor the dangerous difference of squares to avoid cancellation near one.
		factor := (1 - magnitude) * (1 + magnitude)
		nextVariance := variance * factor
		if nextVariance <= 0 || math.IsNaN(nextVariance) || math.IsInf(nextVariance, 0) {
			return nil, ErrSingular
		}

		// Build the next row while leaving the previous row unchanged.
		// FMA rounds the coefficient difference once, improving near-cancellation.
		for j := 1; j < order; j++ {
			if j%cancellationPollInterval == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			next[j] = math.FMA(-reflection, previous[order-j], previous[j])
			if math.IsNaN(next[j]) || math.IsInf(next[j], 0) {
				return nil, ErrOverflow
			}
		}
		next[order] = reflection

		// Restore physical variance before publishing this order in the private Result.
		// Failure discards the complete artifact; no successful prefix escapes.
		physicalVariance, err = restoreVariance(nextVariance, scale)
		if err != nil {
			return nil, err
		}
		result.Values[order] = reflection
		result.PredictionVariance[order] = physicalVariance
		variance = nextVariance

		// At the next order, the just-completed row becomes immutable input.
		previous, next = next, previous
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}
