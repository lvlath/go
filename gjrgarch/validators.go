// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch centralizes finite-domain, state, and stationarity validation.
// Arithmetic overflow is classified separately from invalid caller inputs.
package gjrgarch

import (
	"context"
	"math"
)

// stationarityMargin is the 64-ulp safety distance from the nonstationary face.
// New itself permits any representable rho<1; projected QMLE stays farther inside.
const stationarityMargin = 64 * (0x1p-52)

// finite accepts exactly the non-NaN, non-infinite subset of IEEE binary64.
// It imposes no sign restriction; domain-specific validators handle those.
func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

// stationary checks the symmetric-innovation GJR-GARCH parameter domain.
// It evaluates rho as Beta + Alpha/2 + (Alpha+Gamma)/2 to avoid cancellation
// between a nonnegative Alpha and a potentially negative Gamma.
//
// Errors:
//   - ErrNonFinite for caller-supplied nonfinite fields.
//   - ErrDomain for sign or stationarity violations.
//   - ErrOverflow for non-representable derived coefficients or long-run variance.
//
// Complexity: Time O(1), Space O(1); no allocations.
func stationary(p Parameters) (float64, error) {
	if !finite(p.Mean) || !finite(p.Omega) || !finite(p.Alpha) || !finite(p.Gamma) || !finite(p.Beta) {
		return 0, ErrNonFinite
	}
	if p.Omega <= 0 || p.Alpha < 0 || p.Beta < 0 {
		return 0, ErrDomain
	}

	// Validate the shock coefficient before any stationarity arithmetic.
	negative := p.Alpha + p.Gamma
	if !finite(negative) {
		return 0, ErrOverflow
	}
	if negative < 0 {
		return 0, ErrDomain
	}

	// Sum nonnegative terms; the expression cannot lose a large opposing term.
	rho := p.Beta + p.Alpha*0.5 + negative*0.5
	if !finite(rho) {
		return 0, ErrOverflow
	}
	if rho >= 1 {
		return 0, ErrDomain
	}

	// A stationary model must have a representable unconditional variance.
	unconditional := p.Omega / (1 - rho)
	if !finite(unconditional) || unconditional <= 0 {
		return 0, ErrOverflow
	}
	return rho, nil
}

// stateValid rejects corrupted or foreign state before recurrence evaluation.
// State-level invalidity always maps to ErrStateVersion, even for nonfinite fields.
func (m *Model) stateValid(s State) error {
	if m == nil || s.ModelID != m.id || s.Steps == 0 ||
		!finite(s.LastResidual) || !finite(s.LastVariance) || s.LastVariance <= 0 {
		return ErrStateVersion
	}
	return nil
}

// checkContext classifies a nil context before inspecting other arguments.
// Canceled contexts retain their original context error for errors.Is.
func checkContext(ctx context.Context) error {
	if ctx == nil {
		return ErrInvalidOption
	}
	return ctx.Err()
}

// checkReturns enforces the read-only, nonempty, finite observation contract.
// The scan is ascending and publishes no partial state on failure.
func checkReturns(returns []float64) error {
	if returns == nil {
		return ErrNilInput
	}
	if len(returns) == 0 {
		return ErrEmptyInput
	}
	for _, value := range returns {
		if !finite(value) {
			return ErrNonFinite
		}
	}
	return nil
}
