// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch implements the known-shock and expected-variance forecast.
// Forecast recursion follows the same initial kernel as causal online filtering.
package gjrgarch

// forecastKernel computes H variance forecasts without constructing future residuals.
// The first point is conditional on the observed shock; later points integrate
// over hypothetical symmetric innovations through the fixed persistence rho.
//
// Implementation:
//   - Stage 1: Calculate exact known-shock h[T+1].
//   - Stage 2: Allocate H results.
//   - Stage 3: Advance expected recursion for k>=2.
//
// Behavior highlights:
//   - First horizon depends on last residual sign; later horizons use rho only.
//
// Inputs:
//   - state: validated posterior; horizon: positive allocation-checked future count.
//
// Returns:
//   - Detached []float64 of expected conditional variances, nil on error.
//
// Errors:
//   - ErrOverflow for known shock or unrepresentable expected variance.
//
// Determinism:
//   - Horizon index ascends without sampling or inferring future innovations.
//
// Complexity:
// Time O(H), Space O(H); a single detached result allocation.
//
// Notes:
//   - Returns E[h], not an exact variance-of-volatility or Gaussian predictive-tail law.
//
// AI-Hints:
//   - Do not use the last observed residual again at the second horizon.
func (m *Model) forecastKernel(state State, horizon int) ([]float64, error) {
	first, err := nextVariance(m.p, state.LastResidual, state.LastVariance)
	if err != nil {
		return nil, err
	}

	// Allocate only AFTER computing the validated first variance.
	out := make([]float64, horizon)
	out[0] = first
	for i := 1; i < horizon; i++ {
		// The next innovation is unobserved: expected squared shock is rho*h.
		variance := m.p.Omega + m.rho*out[i-1]
		if !finite(variance) || variance <= 0 {
			return nil, ErrOverflow
		}
		out[i] = variance
	}
	return out, nil
}
