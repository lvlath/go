// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rsi defines independent validation boundaries for price data and checkpoint state.
// This file classifies caller input errors without leaking data and validates state before publication.
package rsi

import "math"

// finite reports whether a float64 has a representable, non-NaN magnitude.
// The classification is intentionally sign-independent; price and mean domains are checked separately.
// Complexity: Time O(1), Space O(1), with no allocation or side effects.
func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// validatePrice enforces the input-price domain before a history or stream transition.
// Non-finite values have precedence over the positivity check; zero is a real domain error.
//
// Errors:
//   - ErrNonFinite: NaN, positive infinity, or negative infinity.
//   - ErrDomain: any finite price <= 0, including negative zero.
//
// Complexity:
//   - Time O(1), Space O(1); no allocations.
//
// AI-Hints:
//   - Never interpret a missing quote encoded as zero as a valid no-change observation.
func validatePrice(price float64) error {
	if !finite(price) {
		return ErrNonFinite
	}
	if price <= 0 {
		return ErrDomain
	}
	return nil
}

// validateSnapshot checks the complete v1 state shape and phase-dependent invariants.
// Checkpoint failure always maps to ErrStateVersion, never to a price-input sentinel.
//
// Implementation:
//   - Stage 1: compare the algorithm/config identifiers and exact slice lengths.
//   - Stage 2: require finite previous/means and nonnegative gain/loss averages.
//   - Stage 3: validate the empty, baseline, and one-transition phase invariants.
//
// Behavior highlights:
//   - Count=0 permits only zero numeric state; Count=1 cannot have seeded means.
//   - Count=2 represents exactly one price change, so gain and loss cannot both be positive.
//   - For later counts the schema cannot prove the historical origin of the means.
//
// Inputs:
//   - state: caller-owned, potentially corrupted or foreign checkpoint.
//   - expectedConfig: stream-owned SHA-256 fingerprint of the normalized period and ABI.
//
// Returns:
//   - nil for structurally and numerically admissible v1 checkpoints.
//
// Errors:
//   - ErrStateVersion: incompatible metadata, shape, numeric domain, or phase state.
//
// Determinism:
//   - Checks metadata first, then float cells in fixed ABI order, then Count phase.
//
// Complexity:
//   - Time O(1), Space O(1), independent of the number of accepted prices.
//
// Notes:
//   - This is structural validation, not proof of state provenance or authentication.
//   - The function must never mutate state or any slice it references.
//
// AI-Hints:
//   - Do not weaken Count=2 exclusivity to allow an impossible first price difference.
//   - A correct config digest does not establish that FloatState came from genuine prices.
func validateSnapshot(state Snapshot, expectedConfig string) error {
	// Reject the entire version/configuration/shape mismatch before indexing slices.
	if state.Algorithm != algorithmID || state.Config != expectedConfig ||
		len(state.FloatState) != snapshotFloatStateSize || len(state.IntState) != 0 {
		return ErrStateVersion
	}

	previous := state.FloatState[snapshotPreviousIndex]
	gain := state.FloatState[snapshotGainIndex]
	loss := state.FloatState[snapshotLossIndex]

	// The oscillator's averages are nonnegative by construction. A nonfinite
	// checkpoint cannot be rescued by the ratio formula and must be rejected.
	if !finite(previous) || !finite(gain) || !finite(loss) || gain < 0 || loss < 0 {
		return ErrStateVersion
	}

	switch state.Count {
	case 0:
		// No accepted close means neither a price baseline nor any gain/loss exists.
		if previous != 0 || gain != 0 || loss != 0 {
			return ErrStateVersion
		}
	case 1:
		// The baseline is the first accepted price; no difference has been seen.
		if previous <= 0 || gain != 0 || loss != 0 {
			return ErrStateVersion
		}
	case 2:
		// Exactly one signed delta is available. It cannot produce a positive gain
		// and positive loss simultaneously, for any admissible smoothing period.
		if previous <= 0 || (gain > 0 && loss > 0) {
			return ErrStateVersion
		}
	default:
		// Later seed/recurrence phases may legitimately have both positive means.
		// The remaining necessary numeric invariant is a positive previous price.
		if previous <= 0 {
			return ErrStateVersion
		}
	}
	return nil
}
