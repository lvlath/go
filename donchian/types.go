// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package donchian defines explicit window policy and owning output channels.
// This file separates pointer nilability, per-row readiness, and buffer ownership.
package donchian

// Options selects the number of observations and the channel's information set.
// ExcludeCurrent determines whether the current bar contributes to its bounds.
//
// Implementation:
//   - Stage 1: Compute validates Period before shape or price values.
//   - Stage 2: the kernel applies exactly one explicitly selected window policy.
//   - Stage 3: the result retains input alignment through its Valid mask.
//
// Behavior highlights:
//   - The zero value is invalid; there is no implicit period or mode switch.
//
// Inputs:
//   - Period: number of observations, not calendar days; must be positive.
//   - ExcludeCurrent: false includes the current bar; true uses preceding bars.
//
// Returns:
//   - Passed by value; Compute never mutates the caller's settings.
//
// Errors:
//   - ErrInvalidOption for Period < 1, after context validation.
//
// Determinism:
//   - Equal options select identical window boundaries and readiness indices.
//
// Complexity:
//   - Time O(1), Space O(1) to validate and copy options.
//
// Notes:
//   - Period larger than the history is valid and does not allocate by Period.
//
// AI-Hints:
//   - Use ExcludeCurrent for a current-close versus previous-range comparison.
//   - IncludeCurrent is causal at bar close, but is not a prior-bar threshold.
type Options struct {
	// Period counts ordered observations in a complete window, not calendar days.
	// Every positive int is allowed; insufficient history is masked, not an error.
	Period int

	// ExcludeCurrent selects the information set of the row at index i.
	// False uses [i-Period+1,i] and first becomes ready at Period-1.
	// True uses [i-Period,i-1] and first becomes ready at Period.
	ExcludeCurrent bool
}

// Result owns the aligned Donchian boundaries and their shared readiness mask.
// All four channels have input length; an unavailable row has three positive zeros.
//
// Implementation:
//   - Stage 1: Compute validates the complete input before output allocation.
//   - Stage 2: one float backing array is partitioned into disjoint capped slices.
//   - Stage 3: the complete result is published only after a final context check.
//
// Behavior highlights:
//   - A valid zero remains valid. Empty successful results have non-nil slices.
//   - Upper and Lower retain the selected observation's bits, including zero sign.
//   - Middle is the nearest-even mean; exact zero is canonical positive zero.
//
// Inputs:
//   - Created by Compute; caller-created Result values need not satisfy these laws.
//
// Returns:
//   - Upper, Middle, Lower and Valid: caller-owned, non-overlapping channels.
//
// Errors:
//   - No methods validate manually constructed channels; Compute errors return nil.
//
// Determinism:
//   - Equal extrema select the latest input index, including ties between zero signs.
//
// Complexity:
//   - Time O(n), Space O(n) to allocate and populate all returned channels.
//
// Notes:
//   - Channels never alias inputs. Their capacities equal their lengths.
//   - The numeric channels share disjoint regions of one backing array; retaining
//     one retains storage for all three. Copy a channel to retain only its own data.
//   - The package retains no result or input references after return.
//   - Appending to one channel cannot overwrite another. Concurrent caller writes
//     to the same result still require caller synchronization.
//
// AI-Hints:
//   - Inspect Valid before reading a row; warmup zeros are not price observations.
//   - Do not infer calendar availability or a trade execution price from a result.
type Result struct {
	// Upper is the maximum observed high in the selected complete window.
	// Equal values select the latest index, preserving that input's zero sign.
	// An unavailable row contains positive zero, regardless of input values.
	Upper []float64

	// Middle is the nearest-even mean of the exact binary64 Upper and Lower.
	// Exact cancellation is positive zero; negative nonzero underflow may be -0.
	// It is not a moving average of closes, a typical price, or an execution price.
	Middle []float64

	// Lower is the minimum observed low in the selected complete window.
	// Ties use the same latest-index policy as Upper; masked payloads are +0.
	Lower []float64

	// Valid indicates that all Period observations needed by this row exist.
	// False masks all three payloads; true is retained even for a zero-width channel.
	// Length and capacity equal the input length, including non-nil empty output.
	Valid []bool
}

// IsNil reports whether the result pointer itself is absent.
// A non-nil empty or entirely masked result is not nil.
//
// Implementation:
//   - Stage 1: inspect pointer presence without dereferencing it.
//   - Stage 2: compare the receiver with nil.
//   - Stage 3: return that comparison unchanged.
//
// Behavior highlights:
//   - Safe on a typed nil receiver, including through an interface.
//
// Inputs:
//   - r: any *Result, including nil or a caller-created value.
//
// Returns:
//   - bool: true exactly when r == nil.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure pointer-presence test, independent of channel contents.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Implements the structural interface { IsNil() bool } without a core import.
//
// AI-Hints:
//   - Never substitute len(r.Valid) == 0 for this nilability contract.
func (r *Result) IsNil() bool {
	return r == nil
}
