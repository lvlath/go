// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima computes causal conditional innovations from differenced samples.
// This file supplies the single signed ARMA residual equation used by fitting and forecasting.
package arima

import "context"

// innovationAt evaluates one conditional ARMA residual at sample index t.
// Its inputs are already differenced y, a residual prefix e with zeros before
// b=max(p,q), and physical coefficients with positive-theta MA convention.
// This helper does not allocate; the caller enforces finiteness and t>=b.
// Time O(p+q), Space O(1).
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

// innovations reconstructs the conditional residuals of a known historical prefix.
// Observations before b=max(p,q) are conditioning data: their innovations are
// set to exactly zero and never synthesized from unavailable presample history.
// Returns an owned residual slice or nil on cancellation/arithmetic overflow.
// Time O((N-b)*(p+q)), Space O(N).
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
