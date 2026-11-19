// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch implements the known-shock and expected-variance forecast.
// Forecast recursion follows the same initial kernel as causal online filtering.
package gjrgarch

// forecastKernel computes H variance forecasts without constructing future residuals.
// The first point is conditional on the observed shock; later points integrate
// over hypothetical symmetric innovations through the fixed persistence rho.
//
// Complexity: Time O(H), Space O(H); a single detached result allocation.
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
