// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema provides exact affine exponential moving average smoothing.
// This file validates public inputs and bounded snapshot integers without reflection.
package ema

import (
	"context"
	"math"
)

const (
	// maxSeedText bounds the entire canonical seed spelling, including a minus sign.
	// Rejecting longer input before parsing keeps Restore work and storage bounded.
	maxSeedText = 660
	// contextCheckInterval caps uninterrupted observations in each batch pass.
	// Push instead checks context on entry and immediately before its commit.
	contextCheckInterval = 1024
	// resultBytesPerObservation accounts for one float64 and one bool payload.
	// It guards address arithmetic, not physical memory availability or allocator overhead.
	resultBytesPerObservation = 8 + 1
)

// checkContext gives missing or canceled context precedence at a public boundary.
// It does not inspect observations, options, or the stream receiver.
//
// Implementation:
//   - Stage 1: reject a nil context interface.
//   - Stage 2: return the context error without wrapping or replacing its identity.
//
// Behavior highlights:
//   - A live context returns nil; cancellation cause customization is not substituted for Err.
//
// Inputs:
//   - ctx: context interface; nil is explicitly supported as a rejected input.
//
// Returns:
//   - error: ErrNilContext, ctx.Err(), or nil.
//
// Errors:
//   - ErrNilContext: no context was supplied; otherwise the original context error.
//
// Determinism:
//   - The context state observed at this call determines cancellation.
//
// Complexity:
//   - Time/space O(1) excluding work implemented by a caller-supplied Err method.
//
// Notes:
//   - A typed-nil custom Context must honor the context interface itself; no reflection is used.
//
// AI-Hints:
//   - Do not replace nil context with Background, which would silently remove cancellation.
func checkContext(ctx context.Context) error {
	// A missing cancellation boundary is an explicit contract failure.
	if ctx == nil {
		return ErrNilContext
	}
	// Preserve the standard context sentinel and any caller context behavior.
	return ctx.Err()
}

// validateOptions admits the explicit seed policy and exact integer period.
// Signed range checking happens before conversion to unsigned coefficient storage.
//
// Implementation:
//   - Stage 1: require 1<=Period<MaxInt.
//   - Stage 2: reject enum values other than SeedSMA and SeedFirst.
//   - Stage 3: return the exact unsigned period without rounding or defaulting.
//
// Behavior highlights:
//   - Options{} remains invalid; wrappers do not change that rule.
//
// Inputs:
//   - opts: copied public configuration; its fields are not mutated.
//
// Returns:
//   - uint64: exact positive period on success; zero with an error.
//
// Errors:
//   - ErrInvalidOption: out-of-range period or unrecognized seed mode.
//
// Determinism:
//   - Period is checked before Seed, with no dependence on floating-point precision.
//
// Complexity:
//   - Time O(1), Space O(1), no allocations.
//
// Notes:
//   - MaxInt is target-dependent; admitted Period+1 fits the target int.
//
// AI-Hints:
//   - Keep the positivity check before the unsigned conversion.
//   - Never infer a default period from a zero-valued Options.
func validateOptions(opts Options) (uint64, error) {
	// Keep signed input until both range limits have been verified.
	period := opts.Period
	// Reserve room for period+1 and reject every nonpositive period.
	if period < 1 || period > math.MaxInt-1 {
		return 0, ErrInvalidOption
	}
	// No unknown policy or hidden fallback may select the seed mechanics.
	if opts.Seed != SeedSMA && opts.Seed != SeedFirst {
		return 0, ErrInvalidOption
	}
	// The preceding signed bound proves this conversion preserves the exact integer.
	return uint64(period), nil
}

// finite admits every finite binary64 observation and rejects only NaN and infinities.
// Negative values, zero signs, and subnormal values are legitimate EMA inputs.
//
// Implementation:
//   - Stage 1: test for NaN.
//   - Stage 2: exclude either infinity sign.
//
// Behavior highlights:
//   - No positivity or epsilon-domain restriction is applied.
//
// Inputs:
//   - x: arbitrary binary64 bits at an input boundary.
//
// Returns:
//   - bool: true exactly for finite represented values.
//
// Errors:
//   - None; callers classify a false observation as ErrNonFinite.
//
// Determinism:
//   - Only the represented value classification is used.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Snapshot current validation additionally enforces a positive-zero sign.
//
// AI-Hints:
//   - Do not import a positive-price validator from MACD or RSI.
func finite(x float64) bool {
	// Reject nonfinite values without rejecting signed finite measurements.
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

// canonicalZero maps either numerical zero to the canonical positive-zero bit pattern.
// All nonzero values are returned unchanged, including subnormal magnitudes.
//
// Implementation:
//   - Stage 1: compare x numerically to zero.
//   - Stage 2: return +0 for zero or preserve x otherwise.
//
// Behavior highlights:
//   - No small-value threshold is applied.
//
// Inputs:
//   - x: finite observation selected directly by a seed or P1 transition.
//
// Returns:
//   - float64: same value, with any negative-zero sign removed.
//
// Errors:
//   - None; callers already validated finiteness.
//
// Determinism:
//   - The only bit change is the sign of an input negative zero.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The exact integer path already canonicalizes zero through roundUnits.
//
// AI-Hints:
//   - Do not implement zero canonicalization using abs: negative nonzero inputs must survive.
func canonicalZero(x float64) float64 {
	// The numeric comparison matches both signs but no nonzero subnormal.
	if x == 0 {
		return 0
	}
	return x
}

// validSeedText recognizes the complete bounded canonical signed-decimal grammar.
// It checks ASCII bytes directly and never allocates an integer or a normalized string.
//
// Implementation:
//   - Stage 1: reject empty or oversized input before indexing any byte.
//   - Stage 2: accept only the unique zero spelling or an optional minus with a nonzero first digit.
//   - Stage 3: require every remaining byte to be an ASCII decimal digit.
//
// Behavior highlights:
//   - Whitespace, plus signs, negative zero, leading zeros, and Unicode digits are rejected.
//
// Inputs:
//   - text: caller-owned candidate SeedSum, measured in bytes rather than runes.
//
// Returns:
//   - bool: whether the whole string matches 0|-?[1-9][0-9]* within maxSeedText.
//
// Errors:
//   - None; Restore maps false to ErrStateVersion.
//
// Determinism:
//   - Bytes are examined left-to-right with no locale or normalization rules.
//
// Complexity:
//   - Time O(D), Space O(1), D<=660; oversized input rejects in O(1).
//
// Notes:
//   - Syntax admission does not imply magnitude admission.
//
// AI-Hints:
//   - Keep the length test before indexing or parsing hostile text.
func validSeedText(text string) bool {
	// Check size before indexing and before doing any multiword arithmetic.
	if len(text) == 0 || len(text) > maxSeedText {
		return false
	}
	// Zero has exactly one canonical decimal spelling.
	if text == zeroSeedText {
		return true
	}
	// Track the first digit without slicing or copying the caller string.
	start := 0
	// A minus sign is the only permitted prefix.
	if text[0] == '-' {
		start = 1
	}
	// Reject a lone minus, leading zero, and any nondecimal initial byte.
	if start == len(text) || text[start] < '1' || text[start] > '9' {
		return false
	}
	// All remaining bytes must be digits; this excludes hidden suffixes and NUL bytes.
	for i := start + 1; i < len(text); i++ {
		if text[i] < '0' || text[i] > '9' {
			return false
		}
	}
	return true
}

// parseSeed decodes canonical decimal units into fixed-size exact signed storage.
// Neither valid input nor malformed input creates a heap-backed integer.
//
// Implementation:
//   - Stage 1: apply the bounded ASCII grammar gate.
//   - Stage 2: record an optional negative sign.
//   - Stage 3: accumulate digits with checked multiply-by-ten plus digit.
//
// Behavior highlights:
//   - A carry beyond 2176 bits rejects the candidate immediately.
//
// Inputs:
//   - text: untrusted SeedSum spelling; no caller mutation or retention occurs.
//
// Returns:
//   - exactSum, bool: canonical decoded units and true, or zero and false.
//
// Errors:
//   - No error allocation; Restore reports ErrStateVersion on false.
//
// Determinism:
//   - Digits are folded in original most-significant-to-least order.
//
// Complexity:
//   - Time O(D*W), Space O(W), D<=660, W=34; no heap allocation.
//
// Notes:
//   - The count-dependent physical bound is enforced separately by seedWithinBound.
//
// AI-Hints:
//   - Never return the truncated accumulator after a multiplyAdd overflow.
func parseSeed(text string) (exactSum, bool) {
	// Grammar validation also protects text[0] from an empty-string access.
	if !validSeedText(text) {
		return exactSum{}, false
	}
	// The entire accumulator is a detached fixed-size value.
	var result exactSum
	// Track the first digit without slicing or copying the caller string.
	start := 0
	// A minus sign is the only permitted prefix.
	if text[0] == '-' {
		result.negative = true
		start = 1
	}
	// Checked Horner accumulation preserves every decimal digit in exact units.
	for i := start; i < len(text); i++ {
		// validSeedText proved this byte subtraction is in [0,9].
		if result.mag.multiplyAdd(decimalRadix, uint64(text[i]-'0')) {
			return exactSum{}, false
		}
	}
	return result, true
}

// seedWithinBound checks whether a seed magnitude is structurally possible for Count.
// It compares exact units against Count times the largest finite binary64 magnitude.
//
// Implementation:
//   - Stage 1: decompose MaxFloat64 exactly.
//   - Stage 2: multiply its magnitude by Count.
//   - Stage 3: compare the candidate magnitude against the inclusive limit.
//
// Behavior highlights:
//   - Count zero admits only a zero magnitude, irrespective of seed phase.
//
// Inputs:
//   - sum: non-nil canonical signed units; count: successful observation count.
//
// Returns:
//   - bool: true exactly when |sum|<=Count*MaxBinary64Units.
//
// Errors:
//   - None; Restore classifies a false result as ErrStateVersion.
//
// Determinism:
//   - Pure integer comparison with an inclusive boundary.
//
// Complexity:
//   - Time O(W), Space O(W), W=34; no allocations.
//
// Notes:
//   - Even MaxUint64 times the largest finite units is below 2^2162 and fits.
//   - This bound is necessary, not proof that the sum came from a particular history.
//
// AI-Hints:
//   - Do not form this limit in float64: it can overflow even when validation is meaningful.
func seedWithinBound(sum *exactSum, count uint64) bool {
	// Start from the exact largest finite magnitude in subnormal units.
	limit := binaryUnits(math.MaxFloat64)
	// The full uint64 count bound fits the 34-limb storage without discarded carry.
	limit.mag.multiplyAdd(count, 0)
	// Accept equality; reject either sign only when its magnitude is too large.
	return sum.mag.compare(&limit.mag) <= 0
}

// validateCurrent recognizes a finite canonical snapshot measurement.
// Negative nonzero currents are allowed, but a negative-zero spelling is not.
//
// Implementation:
//   - Stage 1: require finiteness.
//   - Stage 2: for numerical zero, require the all-zero binary64 representation.
//
// Behavior highlights:
//   - Zero sign is part of the continuation format, not a display preference.
//
// Inputs:
//   - current: untrusted scalar from a snapshot.
//
// Returns:
//   - bool: whether the scalar is finite and zero-canonical.
//
// Errors:
//   - None; Restore maps false to ErrStateVersion rather than ErrNonFinite.
//
// Determinism:
//   - Numeric classification and an exact bit check determine admission.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Warmup and Count-zero phase constraints are checked separately in Restore.
//
// AI-Hints:
//   - Do not silently normalize a malformed persisted Current; reject the snapshot atomically.
func validateCurrent(current float64) bool {
	// An all-zero bit pattern is the sole accepted representation of numerical zero.
	return finite(current) && (current != 0 || math.Float64bits(current) == 0)
}

// validateHistory checks finite observations in their original ascending index order.
// It completes input admission before any EMA arithmetic or result allocation.
//
// Implementation:
//   - Stage 1: poll context at the start of each bounded block.
//   - Stage 2: reject the first nonfinite observation encountered.
//   - Stage 3: poll context at the end, including for an empty history.
//
// Behavior highlights:
//   - No sorting, gap filling, filtering, or input mutation occurs.
//
// Inputs:
//   - ctx: non-nil context admitted by checkContext.
//   - x: caller-owned read-only series, including nil or empty.
//
// Returns:
//   - error: nil after complete admission, otherwise no result is created.
//
// Errors:
//   - ErrNonFinite: a visited observation is NaN or infinite.
//   - ctx.Err(): cancellation observed before, during, or after the scan.
//
// Determinism:
//   - Ascending indexes; concurrent cancellation and input corruption have first-observed precedence.
//
// Complexity:
//   - Time O(n), Space O(1), excluding caller context behavior.
//
// Notes:
//   - Context is checked at least every contextCheckInterval observations.
//
// AI-Hints:
//   - Do not fuse this scan into publication of a partially usable Result.
func validateHistory(ctx context.Context, x []float64) error {
	// Retain input order so the first detected input failure has a stable index.
	for i, value := range x {
		// Bound cancellation latency without introducing per-observation allocation.
		if i%contextCheckInterval == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		// Fail before the calculation phase; never repair NaN by inserting a zero.
		if !finite(value) {
			return ErrNonFinite
		}
	}
	// Preserve the standard context sentinel and any caller context behavior.
	return ctx.Err()
}
