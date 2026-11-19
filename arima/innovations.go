// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima computes causal conditional innovations from differenced samples.
// This file supplies the single signed ARMA residual equation used by fitting and forecasting.
package arima

import "context"

// innovationAt computes one signed conditional ARMA innovation from known history.
// It is the single residual equation used by both profile fitting and the
// historical-origin reconstruction performed before each forecast.
//
// Implementation:
//   - Stage 1: subtract the differenced-equation intercept from observed y[t].
//   - Stage 2: subtract increasing-lag AR terms and known MA innovations.
//   - Stage 3: return the innovation for caller-side finiteness admission.
//
// Behavior highlights:
//   - The public MA convention is y[t]=...+epsilon[t]+sum(theta*epsilon[t-j]);
//     therefore theta terms enter the residual computation with a minus sign.
//
// Inputs:
//   - t: validated sample index t>=max(p,q).
//   - y: complete differenced history, read-only.
//   - e: earlier innovations, with e[t]=0 before conditional burn-in.
//   - parameters: validated physical coefficients, including intercept c.
//
// Returns:
//   - float64: current residual; caller checks representability before retaining it.
//
// Errors:
//   - None directly; the caller turns nonfinite arithmetic into ErrOverflow.
//
// Determinism:
//   - AR and MA coefficients are each scanned in increasing lag order.
//
// Complexity:
//   - Time O(p+q), Space O(1).
//
// Notes:
//   - The helper does not infer presample innovations, allocate, or mutate y.
//
// AI-Hints:
//   - Keep residual signs identical in training and forecasting; a positive
//     MA coefficient is not the same as the stable polynomial coefficient.
func innovationAt(t int, y, e []float64, parameters Parameters) float64 {
	value := y[t] - parameters.Drift
	for i, phi := range parameters.AR {
		value -= phi * y[t-i-1]
	}
	for j, theta := range parameters.MA {
		value -= theta * e[t-j-1]
	}
	return value
}

// innovations reconstructs conditional residuals for an observed differenced prefix.
// Its initialization is exactly the training initialization, so subsequent
// forecasts do not use a different, hidden state-space approximation.
//
// Implementation:
//   - Stage 1: allocate an owned zero-initialized residual vector of length N.
//   - Stage 2: leave the first b=max(p,q) residuals at zero and evaluate
//     innovationAt only for t>=b, with cooperative context checks.
//   - Stage 3: reject nonfinite arithmetic or publish the complete residual vector.
//
// Behavior highlights:
//   - The first b differenced observations remain available as AR lag values.
//   - The first b residuals are conditioning values and are never estimated.
//
// Inputs:
//   - ctx: non-nil active context; cancellation aborts without residual publication.
//   - y: finite differenced samples in sample-index order, read-only.
//   - parameters: validated AR, MA, intercept and degree metadata.
//
// Returns:
//   - []float64: owned complete innovation history, including b leading zeros.
//
// Errors:
//   - ErrOverflow: the signed recurrence is not finite at an effective sample.
//   - ctx.Err(): cancellation at a deterministic sampled checkpoint.
//
// Determinism:
//   - Samples are visited from t=b to N-1; each lag is increasing.
//
// Complexity:
//   - Time O((N-b)*(p+q)), Space O(N).
//
// Notes:
//   - The returned buffer never aliases the original caller history or Model.
//
// AI-Hints:
//   - Do not include presample/conditioning residuals in the SSE even if
//     a more elaborate state-space model could estimate them.
func innovations(ctx context.Context, y []float64, parameters Parameters) ([]float64, error) {
	residuals := make([]float64, len(y))
	burnIn := maxN(parameters.Order.AR, parameters.Order.MA)
	for t := burnIn; t < len(y); t++ {
		if t&contextCheckMask == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		value := innovationAt(t, y, residuals, parameters)
		if !finite(value) {
			return nil, ErrOverflow
		}
		residuals[t] = value
	}
	return residuals, nil
}
