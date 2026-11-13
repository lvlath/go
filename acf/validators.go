// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package acf

// validators.go validates policy, data shape, and finite observations.
// Its order fixes the public sentinel precedence before arithmetic begins.

import "math"

// validateInput enforces the stable configuration-to-data validation order.
// It checks observable failure categories without allocating a numerical workspace
// and ensures that direct may assume strictly positive pair counts at every lag.
//
// Implementation:
//   - Stage 1: reject an unknown estimator and negative maximum lag before reading data.
//   - Stage 2: distinguish nil, empty, and out-of-range lag shape in that order.
//   - Stage 3: scan all observations for NaN and infinity in ascending input order.
//
// Behavior highlights:
//   - maxLag == n is invalid; maxLag == 0 with n == 1 is structurally valid.
//   - Domain errors such as zero variance are left to the direct kernel.
//
// Inputs:
//   - values: caller-owned read-only sample window, never retained or mutated.
//   - maxLag: inclusive lag bound in integer observations.
//   - opts: explicit estimator; its zero value is supported.
//
// Returns:
//   - error: nil only when configuration, shape, and all observations are valid.
//
// Errors:
//   - ErrInvalidOption: unsupported estimator or negative/out-of-range lag.
//   - ErrNilInput: nil observation slice; ErrEmptyInput: allocated empty slice.
//   - ErrNonFinite: first encountered NaN or infinity in the input scan.
//
// Determinism:
//   - Options precede nil; nil precedes empty; shape precedes the ascending finite scan.
//
// Complexity:
//   - Time O(n), Space O(1), with no allocations.
//
// Notes:
//   - Avoid computing maxLag+1 before maxLag < len(values) has been checked.
//
// AI-Hints:
//   - Do not move the NaN check ahead of the shape gate; callers depend on error precedence.
func validateInput(values []float64, maxLag int, opts Options) error {
	if (opts.Denominator != CommonDenominator && opts.Denominator != LagAdjusted) || maxLag < 0 {
		return ErrInvalidOption
	}
	if values == nil {
		return ErrNilInput
	}
	if len(values) == 0 {
		return ErrEmptyInput
	}
	if maxLag >= len(values) {
		return ErrInvalidOption
	}
	for _, observation := range values {
		if math.IsInf(observation, 0) || math.IsNaN(observation) {
			return ErrNonFinite
		}
	}
	return nil
}
