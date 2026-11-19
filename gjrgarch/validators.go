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
//
// Implementation:
//   - Stage 1: Classify IEEE-754 NaN and signed infinities using math predicates.
//
// Behavior highlights:
//   - Accepts ordinary negative values and both signed zeros; sign constraints are separate.
//
// Inputs:
//   - x: arbitrary float64.
//
// Returns:
//   - bool: true for finite binary64 only.
//
// Errors:
//   - None.
//
// Determinism:
//   - Same classification for identical IEEE-754 bits.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper is not a domain validator for positivity or stationarity.
//
// AI-Hints:
//   - Do not equate finite values with valid variances.
func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

// stationary checks the symmetric-innovation GJR-GARCH parameter domain.
// It evaluates rho as Beta + Alpha/2 + (Alpha+Gamma)/2 to avoid cancellation
// between a nonnegative Alpha and a potentially negative Gamma.
//
// Implementation:
//   - Stage 1: Validate all fields.
//   - Stage 2: Form nonnegative downside coefficient.
//   - Stage 3: Sum nonnegative rho terms.
//   - Stage 4: Validate long-run variance.
//
// Behavior highlights:
//   - The stable rho form avoids cancellation of Alpha with negative Gamma.
//
// Inputs:
//   - p: finite coefficients in return and squared-return units.
//
// Returns:
//   - float64: persistence rho in [0,1), or zero and classified error.
//
// Errors:
//   - ErrNonFinite for caller-supplied nonfinite fields.
//   - ErrDomain for sign or stationarity violations.
//   - ErrOverflow for non-representable derived coefficients or long-run variance.
//
// Determinism:
//   - Fixed field scan and positive-term addition order.
//
// Complexity:
// Time O(1), Space O(1); no allocations.
//
// Notes:
//   - The gamma/2 expectation depends on symmetric standardized innovations.
//
// AI-Hints:
//   - Do not silently change to a skew-innovation persistence formula.
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
//
// Implementation:
//   - Stage 1: Check model presence, exact ID, step count and finite positive stored h.
//
// Behavior highlights:
//   - All corrupted-state categories map to one ErrStateVersion sentinel.
//
// Inputs:
//   - m: possibly nil receiver; s: caller-owned posterior value.
//
// Returns:
//   - error: nil for model-compatible State, otherwise ErrStateVersion.
//
// Errors:
//   - ErrStateVersion for missing/corrupted/foreign state.
//
// Determinism:
//   - Predicate order is fixed and does not modify s.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Does not authenticate bytes or external event offsets.
//
// AI-Hints:
//   - Persist ModelID together with feed cursor and source integrity metadata.
func (m *Model) stateValid(s State) error {
	if m == nil || s.ModelID != m.id || s.Steps == 0 ||
		!finite(s.LastResidual) || !finite(s.LastVariance) || s.LastVariance <= 0 {
		return ErrStateVersion
	}
	return nil
}

// checkContext classifies a nil context before inspecting other arguments.
// Canceled contexts retain their original context error for errors.Is.
//
// Implementation:
//   - Stage 1: Reject nil context.
//   - Stage 2: Otherwise return the context cancellation error unchanged.
//
// Behavior highlights:
//   - Pre-canceled operations stop before inspecting other caller data.
//
// Inputs:
//   - ctx: context.Context, possibly nil.
//
// Returns:
//   - error: nil for active context or classified/context error.
//
// Errors:
//   - ErrInvalidOption for nil; ctx.Err() for canceled/deadline context.
//
// Determinism:
//   - No data-dependent work after a failed context check.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Long-running kernels perform later cooperative cancellation polls.
//
// AI-Hints:
//   - Do not replace a context error with ErrNotConverged.
func checkContext(ctx context.Context) error {
	if ctx == nil {
		return ErrInvalidOption
	}
	return ctx.Err()
}

// checkReturns enforces the read-only, nonempty, finite observation contract.
// The scan is ascending and publishes no partial state on failure.
//
// Implementation:
//   - Stage 1: Check nil, then empty.
//   - Stage 2: Scan every supplied observation for NaN/Inf.
//
// Behavior highlights:
//   - A nonnil zero-length slice is ErrEmptyInput, distinct from ErrNilInput.
//
// Inputs:
//   - returns: caller-owned, read-only sample sequence.
//
// Returns:
//   - error: nil only for nonempty all-finite observations.
//
// Errors:
//   - ErrNilInput, ErrEmptyInput or ErrNonFinite in that precedence.
//
// Determinism:
//   - Validation reads elements in ascending index order.
//
// Complexity:
//   - Time O(T), Space O(1).
//
// Notes:
//   - Timestamp continuity and price-to-return conversion are external checks.
//
// AI-Hints:
//   - Never treat a warmup invalid zero from series.TransformResult as an observed return.
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
