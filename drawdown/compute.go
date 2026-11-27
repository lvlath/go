// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package drawdown computes correctly rounded observed depth and its maximum witness.
// This file owns the private profile scan, recovery scan, and publication boundary.
package drawdown

import (
	"context"
	"math/bits"
)

// noRecoveryIndex denotes a positive drawdown with no recovery in the history.
// The no-drawdown convention is instead the zero witness 0/0/0.
const noRecoveryIndex = -1

// roundedDepth returns RN64((peak-equity)/peak) for 0<=equity<peak.
// It rounds the exact difference ratio rather than subtracting a rounded quotient.
//
// Implementation:
//   - Stage 1: return exact depth 1 for either sign of zero equity.
//   - Stage 2: use exact subtraction and one division when equity is near the peak.
//   - Stage 3: normalize positive operands and round the deep loss on a fixed grid.
//
// Behavior highlights:
//   - Sterbenz subtraction is exact when peak/2 <= equity < peak.
//   - A rounded half-peak test at the subnormal boundary remains safe: there
//     the difference fits exactly on the integer subnormal grid.
//
// Inputs:
//   - peak: positive finite running maximum; equity: finite nonnegative value < peak.
//
// Returns:
//   - float64: nearest-even rounding of the exact positive depth, in (0,1].
//
// Determinism:
//   - Both paths implement the same binary64 rounding law, not two estimators.
//
// Complexity:
//   - Time O(1), Space O(1); no dynamic arithmetic workspace.
//
// Notes:
//   - A strict decrease cannot round to zero: neighboring positive binary64
//     values have a relative separation of at least 2^-53.
//
// AI-Hints:
//   - Neither 1-equity/peak nor an unconditional rounded subtraction/division
//     implements the exact deep-loss contract.
func roundedDepth(peak, equity float64) float64 {
	if equity == 0 {
		return 1 // No division is needed for exact total loss.
	}
	if equity >= peak*0.5 {
		return (peak - equity) / peak // Exact difference, then one rounding.
	}
	// Deep differences may not be representable before division; round the ratio directly.
	return deepDepth(splitBinary(peak), splitBinary(equity))
}

// deepDepth rounds the exact difference ratio on the binary64 grid in [0.5,1].
// With even N=2^53, rounding the exact rational N*(P-E)/P is equivalent to
// N-roundEven(N*E/P), including ties; the complement is an integer, not a float ratio.
//
// Implementation:
//   - Stage 1: compute the normalized exponent gap; resolve residuals below half
//     a grid unit and the gap-54 midpoint case without division.
//   - Stage 2: form the scaled residual numerator as two integer words and divide
//     by the normalized peak significand, retaining quotient and remainder.
//   - Stage 3: round quotient by remainder and parity, complement on the even
//     grid, and scale exactly by 2^-53 for publication.
//
// Behavior highlights:
//   - For the remaining gaps 1..53, shift=53-gap lies in [0,52]. The high
//     numerator word is <2^42 and the denominator is >=2^52, so Div64 cannot
//     overflow or divide by zero. Twice the remainder fits within 54 bits.
//   - The integer complement is <=2^53, exactly convertible to float64.
//
// Inputs:
//   - peak, equity: normalized positive operands with exact equity < peak/2.
//
// Returns:
//   - float64: correctly rounded exact depth in [0.5,1].
//
// Determinism:
//   - Midpoint decisions use the integer quotient's parity, never an epsilon.
//
// Complexity:
//   - Time O(1), Space O(1); at most one bounded Div64 operation.
//
// AI-Hints:
//   - Do not replace quotient/remainder rounding with floating E/P or drop the
//     parity condition at equality; near-total losses can round to exactly 1.
func deepDepth(peak, equity binaryValue) float64 {
	gap := peak.exponent - equity.exponent // Strict deep loss implies gap >= 1.
	if gap > significandBits+1 {
		// E*N/P < 1/2; nearest-even residual integer is zero.
		return 1
	}
	if gap == significandBits+1 {
		// E*N/P = mE/(2*mP), in (1/4,1); an exact half rounds to zero.
		if equity.mantissa > peak.mantissa {
			return 1 - depthStep // Strictly above half rounds to residual integer one.
		}
		return 1 // At or below half, the even integer zero wins.
	}
	// Construct the exact scaled numerator; shift zero naturally gives hi=0.
	shift := significandBits - gap
	hi := equity.mantissa >> (productWordBits - shift)
	lo := equity.mantissa << shift
	// The normalized denominator dominates the high word, satisfying Div64's precondition.
	quotient, remainder := bits.Div64(hi, lo, peak.mantissa)
	twice := remainder << 1 // Compare against half the divisor without fractional arithmetic.
	if twice > peak.mantissa || (twice == peak.mantissa && quotient&1 != 0) {
		quotient++ // Above half or a midpoint with an odd quotient rounds upward.
	}
	// Complement on the exact even integer grid, then convert and scale exactly.
	return float64(depthGrid-quotient) * depthStep
}

// scanDepth validates later values and calculates the profile and exact witness.
// The initial observation is already admitted; the output remains private until
// recovery and the final context check succeed.
//
// Implementation:
//   - Stage 1: initialize the earliest positive peak at index zero.
//   - Stage 2: scan later indices, polling context and validating before use;
//     select strict peak increases or compute the rounded depth of a lower value.
//   - Stage 3: replace the witness only on a strict exact depth improvement and
//     copy the selected rounded depth into Maximum.
//
// Behavior highlights:
//   - Fresh zeroed storage supplies canonical +0 at peaks, including the origin.
//   - Equal peaks preserve their earliest index, including after a recovery.
//
// Inputs:
//   - ctx: usable context; equity: nonempty history with valid initial equity.
//   - result: fresh private Result with zeroed input-length Depth and zero indices.
//
// Returns:
//   - nil after a complete profile scan; otherwise the first detected failure.
//
// Errors:
//   - ctx.Err(), ErrNonFinite, ErrEquity; a partially written result stays private.
//
// Determinism:
//   - Ascending index order, finite-before-sign at each index, strict exact improvements.
//
// Complexity:
//   - Time O(n), Space O(1) beyond the already allocated O(n) output.
//
// AI-Hints:
//   - Do not skip validation of later peaks or use a rounded maximum as a comparator.
func scanDepth(ctx context.Context, equity []float64, result *Result) error {
	peakIndex := 0 // The initial positive observation establishes the running peak.
	for i := 1; i < len(equity); i++ {
		// Poll relative to the original history, with entry already checked by the facade.
		if i%cancellationStride == 0 {
			if err := ctx.Err(); err != nil {
				return err // The unpublished profile will be discarded by compute.
			}
		}
		value := equity[i] // Read the original sample; do not normalize input storage.
		if err := validateEquity(i, value); err != nil {
			return err // No later observation can change this detected input error.
		}
		peak := equity[peakIndex] // The first index attaining the current maximum.
		if value >= peak {
			if value > peak {
				peakIndex = i // Only a strict increase establishes a new peak.
			}
			continue // Preallocated +0; equal peaks retain their earliest index.
		}
		// The numeric artifact and witness ordering have separate responsibilities.
		result.Depth[i] = roundedDepth(peak, value)
		if result.TroughIndex == 0 || deeper(peak, value, equity[result.PeakIndex], equity[result.TroughIndex]) {
			result.PeakIndex = peakIndex // Preserve the anchor associated with this candidate.
			result.TroughIndex = i       // Never replace an exact equal-depth incumbent.
		}
	}
	// The summary must match the selected published element bit for bit, including +0.
	result.Maximum = result.Depth[result.TroughIndex]
	return nil
}

// scanRecovery finds the earliest later observation that reaches the witness peak.
// It recovers the selected historical episode, not the final underwater episode.
//
// Implementation:
//   - Stage 1: preserve the 0/0/0 witness if no drawdown occurred.
//   - Stage 2: mark the selected positive drawdown unrecovered and scan strictly
//     after its trough, polling context on suffix-relative stride boundaries.
//   - Stage 3: record the first equity >= the selected peak, or retain -1.
//
// Behavior highlights:
//   - Equality recovers; a new higher peak also recovers. No later index can
//     replace the first recovery. An empty suffix is a valid unrecovered episode.
//
// Inputs:
//   - ctx: usable context; equity: unchanged fully validated history.
//   - result: private output with the exact selected PeakIndex and TroughIndex.
//
// Returns:
//   - nil for recovered, unrecovered or no-drawdown cases; otherwise ctx.Err().
//
// Errors:
//   - Cancellation only; input validation is complete.
//
// Determinism:
//   - Increasing original index order starting at TroughIndex+1.
//
// Complexity:
//   - Time O(n-trough) worst case, Space O(1); early exit on first recovery.
//
// AI-Hints:
//   - Do not search before the selected trough or use the final running peak as the target.
func scanRecovery(ctx context.Context, equity []float64, result *Result) error {
	if result.TroughIndex == 0 {
		return nil // Preserve the explicit no-drawdown witness convention.
	}
	start := result.TroughIndex + 1        // Recovery is strictly after the selected trough.
	peak := equity[result.PeakIndex]       // Fixed selected peak, not the final running peak.
	result.RecoveryIndex = noRecoveryIndex // Positive drawdown is unrecovered until proved otherwise.
	for i := start; i < len(equity); i++ {
		if (i-start)%cancellationStride == 0 {
			if err := ctx.Err(); err != nil {
				return err // Cancellation invalidates the whole private result.
			}
		}
		if equity[i] >= peak {
			result.RecoveryIndex = i // The first qualifying observation is authoritative.
			break
		}
	}
	return nil // An absent recovery is successful evidence, not a domain error.
}

// compute owns the detached output allocation, ordered scans, and publication.
// It prevents any intermediate state from escaping through a failed call.
//
// Implementation:
//   - Stage 1: allocate the zeroed aligned output once after initial admission.
//   - Stage 2: scan depth/validation and the selected recovery suffix.
//   - Stage 3: check cancellation immediately before publishing the complete artifact.
//
// Behavior highlights:
//   - No per-observation heap allocation on success; no state or input retention.
//
// Inputs:
//   - ctx: already admitted usable context; equity: nonempty history with valid origin.
//
// Returns:
//   - *Result: detached and complete on success; nil on every error.
//
// Errors:
//   - Propagates scan validation and context errors unchanged.
//
// Determinism:
//   - Profile scan precedes recovery; no output is visible before final context check.
//
// Complexity:
//   - Time O(n), Space O(n) output and O(1) auxiliary, including allocation zeroing.
//
// Notes:
//   - Allocation is not interruptible; physical memory availability is not guaranteed.
//
// AI-Hints:
//   - Keep error returns nil; never expose the partially populated result after a failed scan.
func compute(ctx context.Context, equity []float64) (*Result, error) {
	result := &Result{Depth: make([]float64, len(equity))} // Own the aligned profile, initialized to +0.
	if err := scanDepth(ctx, equity, result); err != nil {
		return nil, err // Includes late invalid data and first-pass cancellation.
	}
	if err := scanRecovery(ctx, equity, result); err != nil {
		return nil, err // Recovery cancellation must not publish even a finished profile.
	}
	if err := ctx.Err(); err != nil {
		return nil, err // Final check also covers no-drawdown and empty-suffix cases.
	}
	return result, nil // The caller now owns the complete independent artifact.
}
