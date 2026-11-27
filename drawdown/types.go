// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package drawdown defines the caller-owned depth and witness artifact.
// This file fixes field invariants, ownership, and nil-receiver semantics.
package drawdown

// Result contains the sampled depth profile and one exact maximum episode.
// It describes observed equity, not an intrabar or cashflow-adjusted model.
//
// Implementation:
//   - Compute creates an aligned depth slice and selects the witness in input order.
//   - The first subsequent recovery to the witness peak finalizes the summary.
//
// Behavior highlights:
//   - Depth values are nearest-even rounded exact ratios in [0,1]; zero is +0.
//   - Maximum is bit-identical to Depth[TroughIndex] for successful results.
//   - Distinct exact ratios can have identical rounded depths; the greater exact
//     ratio still wins. Exact ties retain the earliest trough and equal peak.
//
// Inputs:
//   - Construct through Compute from finite, ordered, nonnegative equity.
//
// Returns:
//   - Depth is detached and writable by the caller; no input alias or retention.
//   - Indices are zero-based observation positions, not calendar durations.
//   - RecoveryIndex=-1 means the selected episode is unrecovered in the input.
//
// Errors:
//   - Compute never returns a partially valid Result on error.
//
// Determinism:
//   - Strict peak increases and strict exact depth improvements replace anchors.
//   - No drawdown means Maximum=+0 and all three indices are 0.
//
// Complexity:
//   - O(n) depth storage; O(1) summary storage.
//
// Notes:
//   - The zero-value Result is not a computed history. IsNil checks pointer only.
//   - Copying a Result by value copies its slice header, not its underlying data.
//   - Concurrent writes to shared result data require caller synchronization.
//   - Caller mutation may invalidate the relationship between fields.
//
// AI-Hints:
//   - Never reconstruct the witness by argmax over rounded Depth.
//   - Do not treat recovery=-1 as a negative duration or the absence of drawdown.
//   - A rounded Maximum of 1 does not prove that equity actually reached zero.
type Result struct {
	// Depth contains one nonnegative relative depth per input observation.
	// Each element is RN64((runningPeak-equity)/runningPeak) in [0,1],
	// including +0 at the origin and at every observation equal to its running peak.
	// The caller owns this non-nil slice on success; it never aliases equity.
	Depth []float64
	// Maximum is the rounded depth at the exactly selected trough.
	// It is a dimensionless fraction, not a percentage; it equals Depth[TroughIndex]
	// bit for bit when returned by Compute, even if several depths round alike.
	Maximum float64
	// PeakIndex is the first occurrence of the running peak for the witness.
	// It is zero-based and does not move to a later equal peak after recovery.
	PeakIndex int
	// TroughIndex is the earliest observation attaining the exact maximum depth.
	// For a positive drawdown, PeakIndex < TroughIndex < len(Depth).
	// A zero trough index is reserved for a history with no drawdown.
	TroughIndex int
	// RecoveryIndex is the first recovery after the trough, -1 if absent,
	// or 0 when no drawdown occurred. It is strictly greater than TroughIndex
	// for a recovered positive drawdown; it is not a time duration.
	RecoveryIndex int
}

// IsNil reports whether r is a nil result pointer.
// It implements the reflect-free nilable result convention without inspecting fields.
//
// Implementation:
//   - Stage 1: compare the receiver pointer to nil, without dereferencing it.
//   - Stage 2: return the pointer comparison unchanged.
//
// Behavior highlights:
//   - A nil receiver is safe; a non-nil zero-value Result is not nil.
//   - A typed-nil Result remains detectable behind a compatible interface.
//
// Inputs:
//   - r: any *Result, including nil or a caller-constructed value.
//
// Returns:
//   - bool: true only when r is nil.
//
// Errors:
//   - None; this predicate neither validates fields nor classifies computation failures.
//
// Determinism:
//   - Pure pointer identity check; no observable state transition.
//
// Complexity:
//   - Time O(1), Space O(1); no allocation or slice access.
//
// Notes:
//   - Compatible with core.Nilable by Go's structural interface rules; no core import is needed.
//   - A non-nil result is not evidence of successful computation without checking err.
//
// AI-Hints:
//   - Do not substitute len(Depth)==0 for pointer nilability or use IsNil as a validity mask.
func (r *Result) IsNil() bool {
	return r == nil // Do not dereference a possibly nil receiver.
}
