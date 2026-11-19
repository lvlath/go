// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima profiles conditional Gaussian likelihood over innovation variance.
// This file differentiates the same innovations used for forecasting without dense covariance.
package arima

import (
	"context"
	"math"
)

// profile evaluates the negative conditional Gaussian log likelihood and gradient.
// Only t>=max(p,q) enters SSE; the first b observations are conditioning history.
// Derivatives propagate through every lagged MA residual in physical order
// [AR..., MA..., Drift], including a drift coordinate fixed by a caller policy.
//
// Implementation:
//   - Stage 1: allocate contiguous N-by-(p+q+1) sensitivities with overflow guard.
//   - Stage 2: calculate innovations and derivatives in causal ascending time order.
//   - Stage 3: scale SSE, profile sigma^2=SSE/EffectiveN, and normalize the gradient.
//
// Behavior highlights:
//   - Original residuals before b=max(p,q) remain exactly zero.
//   - Scaled SSE avoids directly squaring huge finite residuals, unlike a naive sum.
//
// Inputs:
//   - ctx: active context, checked during both residual and gradient passes.
//   - y: validated differenced observations, read-only, with positive EffectiveN.
//   - parameters: strictly stable/invertible physical coefficients and c.
//
// Returns:
//   - objective: negative profiled conditional Gaussian log likelihood.
//   - variance: positive finite profiled sigma^2; gradient: owned physical derivatives.
//
// Errors:
//   - ErrSingular: all effective residuals vanish, making log(SSE) undefined.
//   - ErrOverflow: unrepresentable residual, derivative, variance or allocation product.
//   - ctx.Err(): cancellation without partial likelihood or derivatives.
//
// Determinism:
//   - Samples, lagged residuals and coordinates are scanned in ascending index order.
//
// Complexity:
//   - Time O((N-b)*(p+q+1)*(q+1)), Space O(N*(p+q+1)).
//
// Notes:
//   - This is conditional Gaussian MLE, not exact state-space likelihood.
//
// AI-Hints:
//   - Do not add the first b residuals to SSE or substitute denominator N.
//   - Do not differentiate sigma^2 as an independent optimizer parameter.
func profile(ctx context.Context, y []float64, parameters Parameters) (float64, float64, []float64, error) {
	burnIn := maxN(parameters.Order.AR, parameters.Order.MA)
	effectiveN := len(y) - burnIn
	dimension := parameters.Order.AR + parameters.Order.MA + 1
	if len(y) > int(^uint(0)>>1)/dimension {
		return 0, 0, nil, ErrOverflow
	}

	residuals := make([]float64, len(y))
	// Flatten the sensitivity matrix: unlike [][]float64, one allocation covers all rows.
	sensitivities := make([]float64, len(y)*dimension)
	scale, scaledSSE := 0.0, 1.0
	for t := burnIn; t < len(y); t++ {
		if t&contextCheckMask == 0 {
			if err := ctx.Err(); err != nil {
				return 0, 0, nil, err
			}
		}

		residual := innovationAt(t, y, residuals, parameters)
		if !finite(residual) {
			return 0, 0, nil, ErrOverflow
		}
		residuals[t] = residual
		current := sensitivities[t*dimension : (t+1)*dimension]

		for coordinate := 0; coordinate < dimension; coordinate++ {
			derivative := -1.0 // Intercept c contributes -1 to its residual derivative.
			switch {
			case coordinate < parameters.Order.AR:
				derivative = -y[t-coordinate-1]
			case coordinate < parameters.Order.AR+parameters.Order.MA:
				derivative = -residuals[t-(coordinate-parameters.Order.AR)-1]
			}
			for j, theta := range parameters.MA {
				previous := (t-j-1)*dimension + coordinate
				derivative -= theta * sensitivities[previous]
			}
			if !finite(derivative) {
				return 0, 0, nil, ErrOverflow
			}
			current[coordinate] = derivative
		}

		// LAPACK-style scale/SSQ accumulation: scale^2*scaledSSE = Σ(e_t^2).
		absolute := math.Abs(residual)
		if absolute > 0 {
			if scale < absolute {
				ratio := scale / absolute
				scaledSSE = 1 + scaledSSE*ratio*ratio
				scale = absolute
			} else {
				ratio := absolute / scale
				scaledSSE += ratio * ratio
			}
		}
	}
	if scale == 0 { // All effective residuals vanish; Gaussian variance is undefined.
		return 0, 0, nil, ErrSingular
	}

	logSSE := 2*math.Log(scale) + math.Log(scaledSSE)
	logVariance := logSSE - math.Log(float64(effectiveN))
	objective := 0.5 * float64(effectiveN) * (math.Log(2*math.Pi) + 1 + logVariance)
	if !finite(objective) || !finite(logVariance) {
		return 0, 0, nil, ErrOverflow
	}
	variance := math.Exp(logVariance)
	if variance <= 0 || !finite(variance) {
		return 0, 0, nil, ErrOverflow
	}

	gradient := make([]float64, dimension)
	for t := burnIn; t < len(y); t++ {
		if t&contextCheckMask == 0 {
			if err := ctx.Err(); err != nil {
				return 0, 0, nil, err
			}
		}
		// Dividing both factors by scale keeps large SSE derivatives representable.
		normalized := residuals[t] / scale
		current := sensitivities[t*dimension : (t+1)*dimension]
		for coordinate, derivative := range current {
			gradient[coordinate] += normalized * (derivative / scale)
		}
	}
	factor := float64(effectiveN) / scaledSSE
	for coordinate := range gradient {
		gradient[coordinate] *= factor
		if !finite(gradient[coordinate]) {
			return 0, 0, nil, ErrOverflow
		}
	}
	return objective, variance, gradient, nil
}
