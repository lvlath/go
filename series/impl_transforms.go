// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package series computes length-preserving transformations from earlier observations.
// This file implements lag, differences, and numerically guarded simple/log returns.
package series

import "math"

// stableRelativeChangeLimit defines the region where curr-prev is safely
// representable and a relative difference avoids cancellation from a quotient
// rounded to one. Its strict bound is a dimensionless half-unit change.
const stableRelativeChangeLimit = 0.5

// lagTransform copies each available predecessor into its corresponding output.
// All earlier indices remain zero/false; a lag of zero is a detached identity
// and a lag beyond the data length leaves every output position invalid.
//
// Implementation:
//   - Stage 1: Validate lag before validating the complete finite input.
//   - Stage 2: Allocate two detached length-preserving output slices.
//   - Stage 3: Fill valid predecessor positions in ascending order and publish.
//
// Behavior highlights:
//   - A lag >= n performs no indexed reads but still validates every input value.
//
// Inputs:
//   - values: finite read-only observations, indexed from zero.
//   - lag: nonnegative number of earlier sample positions, not nanoseconds.
//
// Returns:
//   - *TransformResult: input-length values/mask with exact zero/false warmup.
//
// Errors:
//   - ErrInvalidOption: negative lag, before input validation.
//   - ErrNilInput, ErrEmptyInput, ErrNonFinite: shared unary input policy.
//
// Determinism:
//   - Writes increasing indices i from lag through n-1, reading values[i-lag].
//
// Complexity:
//   - Time O(n), space O(n), including input validation and result allocation.
//
// Notes:
//   - Inputs are not mutated; returned storage is detached from the caller.
//
// AI-Hints:
//   - Never shortcut lag >= n before validating nonfinite warmup data.
func lagTransform(values []float64, lag int) (*TransformResult, error) {
	// Validation order is part of the error protocol: options come first.
	if lag < 0 {
		return nil, ErrInvalidOption
	}
	if err := validateValues(values); err != nil {
		return nil, err
	}

	// All missing-history cells start with exactly zero/false.
	result := newTransform(len(values))

	// Each valid cell reads only a predecessor, so no future append can
	// change an already computed value or its validity bit.
	for i := lag; i < len(values); i++ {
		result.Values[i] = values[i-lag]
		result.Valid[i] = true
	}
	return result, nil
}

// differenceTransform subtracts a predecessor lag positions before the current cell.
// It produces values[i]-values[i-lag] for i >= lag and reserves zero/false
// positions for missing history without shortening the result array.
//
// Implementation:
//   - Stage 1: Reject a nonpositive lag and scan all values for finiteness.
//   - Stage 2: Allocate detached buffers and compute each admissible subtraction.
//   - Stage 3: Reject an unrepresentable difference before publishing results.
//
// Behavior highlights:
//   - If lag >= n, every output remains zero/false and no subtraction runs.
//
// Inputs:
//   - values: read-only nonempty finite float64 observations.
//   - lag: strictly positive sample-index distance.
//
// Returns:
//   - *TransformResult: finite differences and exact validity mask; nil on error.
//
// Errors:
//   - ErrInvalidOption: lag <= 0.
//   - ErrNilInput, ErrEmptyInput, ErrNonFinite: invalid input array.
//   - ErrOverflow: subtraction produces an infinite or NaN float64 result.
//
// Determinism:
//   - Subtractions are performed in increasing index order without reassociation.
//
// Complexity:
//   - Time O(n), space O(n), including validation and detached output.
//
// Notes:
//   - Even finite opposite-sign operands can overflow during subtraction.
//
// AI-Hints:
//   - Do not substitute Inf for arithmetic failure or publish a partial result.
func differenceTransform(values []float64, lag int) (*TransformResult, error) {
	// Difference has no mathematical meaning for a zero or negative lag here.
	if lag < 1 {
		return nil, ErrInvalidOption
	}
	if err := validateValues(values); err != nil {
		return nil, err
	}

	result := newTransform(len(values))
	for i := lag; i < len(values); i++ {
		// Both operands are finite, but their difference may exceed MaxFloat64.
		v := values[i] - values[i-lag]
		if !finiteResult(v) {
			return nil, ErrOverflow
		}
		// Publish validity only after the corresponding arithmetic succeeded.
		result.Values[i], result.Valid[i] = v, true
	}
	return result, nil
}

// simpleReturnsTransform calculates changes relative to the previous observation.
// The mathematical return is curr/prev-1, but sufficiently close operands
// use (curr-prev)/prev to avoid losing a small change to quotient rounding.
//
// Implementation:
//   - Stage 1: Validate every observation, then every used denominator.
//   - Stage 2: Allocate detached result storage after domain checks.
//   - Stage 3: Select a stable relative-change or quotient path for each pair.
//   - Stage 4: Reject unrepresentable returns and publish only on full success.
//
// Behavior highlights:
//   - Negative values are allowed; a final zero is valid until used as denominator.
//   - A singleton has no denominator and yields one zero/false cell.
//
// Inputs:
//   - values: caller-owned finite observations, read-only during the call.
//
// Returns:
//   - *TransformResult: finite adjacent returns in input-length detached buffers.
//
// Errors:
//   - ErrNilInput, ErrEmptyInput, ErrNonFinite: shared unary input policy.
//   - ErrDomain: a zero at an index used as a previous value.
//   - ErrOverflow: computed return cannot be represented as finite float64.
//
// Determinism:
//   - Denominators are checked first, then pairs processed in ascending order.
//
// Complexity:
//   - Time O(n), space O(n), including two validation passes and output.
//
// Notes:
//   - A stable relative-difference branch preserves returns near machine epsilon.
//
// AI-Hints:
//   - Validate every denominator before computation to preserve domain precedence.
//   - Do not force nonnegative values: this is mathematics, not trading policy.
func simpleReturnsTransform(values []float64) (*TransformResult, error) {
	// Structural and nonfinite errors take precedence over domain failures.
	if err := validateValues(values); err != nil {
		return nil, err
	}

	// Only values with a successor act as denominators. Checking the whole
	// denominator domain before arithmetic guarantees ErrDomain precedence
	// even when an earlier pair could otherwise overflow.
	for i := 1; i < len(values); i++ {
		if values[i-1] == 0 {
			return nil, ErrDomain
		}
	}

	result := newTransform(len(values))
	for i := 1; i < len(values); i++ {
		// The validated previous sample is nonzero; the current one may be zero.
		prev, curr := values[i-1], values[i]

		// Prefer cancellation-resistant subtraction when its relative change
		// is small; otherwise the quotient also covers opposite-sign extremes
		// whose direct subtraction could overflow while the ratio remains finite.
		v, stable := smallRelativeChange(prev, curr)
		if !stable {
			v = curr/prev - 1
		}
		if !finiteResult(v) {
			return nil, ErrOverflow
		}

		result.Values[i], result.Valid[i] = v, true
	}
	return result, nil
}

// logReturnsTransform computes natural-log differences for positive observations.
// Close pairs use log1p((curr-prev)/prev) to preserve sub-ULP changes in
// log(curr)-log(prev); distant pairs use log differences to avoid ratio overflow.
//
// Implementation:
//   - Stage 1: Validate finiteness and strict positivity of all observations.
//   - Stage 2: Allocate detached length-preserving values and masks.
//   - Stage 3: Evaluate log1p for close pairs or direct log differences otherwise.
//   - Stage 4: Verify finite results and publish only after every pair succeeds.
//
// Behavior highlights:
//   - Full-domain validation includes a singleton and all warmup observations.
//   - Extreme ratios never form curr/prev as an intermediate float64.
//
// Inputs:
//   - values: read-only nonempty, strictly positive finite float64 observations.
//
// Returns:
//   - *TransformResult: input-length finite log changes with zero/false first cell.
//
// Errors:
//   - ErrNilInput, ErrEmptyInput, ErrNonFinite: shared unary input policy.
//   - ErrDomain: any observation <= 0, including a singleton.
//   - ErrOverflow: computed log return fails finite-result checks.
//
// Determinism:
//   - Full positive-domain scan precedes ascending adjacent-pair evaluation.
//
// Complexity:
//   - Time O(n), space O(n), including validation and detached output.
//
// Notes:
//   - Natural logarithms and log1p use standard IEEE-754 float64 rounding.
//
// AI-Hints:
//   - Never evaluate math.Log(curr/prev): extreme ratios can overflow or underflow.
//   - A difference of rounded logs can erase real small changes; retain log1p.
func logReturnsTransform(values []float64) (*TransformResult, error) {
	if err := validateValues(values); err != nil {
		return nil, err
	}

	// Strict positivity is an input-domain law, not a condition on computed pairs.
	// Therefore a zero singleton is rejected even though no return is evaluated.
	for _, v := range values {
		if v <= 0 {
			return nil, ErrDomain
		}
	}

	result := newTransform(len(values))
	for i := 1; i < len(values); i++ {
		prev, curr := values[i-1], values[i]

		// A small relative difference is accurately represented as a fraction
		// of prev; log1p retains the change when the two separately rounded
		// logarithms would be indistinguishable at this scale.
		relative, stable := smallRelativeChange(prev, curr)
		var v float64
		if stable {
			v = math.Log1p(relative)
		} else {
			// Taking logs before subtraction avoids materializing a ratio
			// such as 1e300/1e-300 that would overflow in binary64.
			v = math.Log(curr) - math.Log(prev)
		}
		if !finiteResult(v) {
			return nil, ErrOverflow
		}

		result.Values[i], result.Valid[i] = v, true
	}
	return result, nil
}

// smallRelativeChange identifies a safe close-pair region and computes delta/prev.
// When |(curr-prev)/prev| < stableRelativeChangeLimit, nearby same-sign
// operands satisfy the exact-subtraction region of Sterbenz's lemma; the
// resulting relative change can be used directly or as input to math.Log1p.
//
// Implementation:
//   - Stage 1: Compute delta and reject overflow; callers require nonzero prev.
//   - Stage 2: Divide by prev and check finite magnitude against the limit.
//   - Stage 3: Return the accurate relative change or signal the caller fallback.
//
// Behavior highlights:
//   - False requests a caller-specific formula, not a blanket arithmetic error.
//   - The same branch rule is shared by simple and logarithmic returns.
//
// Inputs:
//   - prev: previous finite observation, nonzero under caller validation.
//   - curr: current finite observation, positive for log-return callers.
//
// Returns:
//   - float64: relative change only when bool is true; otherwise zero.
//   - bool: true if the relative subtraction branch is safe and appropriate.
//
// Determinism:
//   - Uses a fixed strict inequality, independent of data length or scheduling.
//
// Complexity:
//   - Time O(1), additional space O(1), without allocation.
//
// Notes:
//   - Both public transforms preserve their own domain and overflow semantics.
//
// AI-Hints:
//   - Do not replace the strict bound with an epsilon acceptance tolerance.
//   - Fallback must be formula-specific: quotient for simple, log difference for log.
func smallRelativeChange(prev, curr float64) (float64, bool) {
	// An overflowing subtraction cannot be used for a relative-difference path.
	delta := curr - prev
	if !finiteResult(delta) {
		return 0, false
	}

	// The relative change is dimensionless. An overflowed division is not
	// a usable close-pair representation, even though input operands are finite.
	relative := delta / prev
	if !finiteResult(relative) || math.Abs(relative) >= stableRelativeChangeLimit {
		return 0, false
	}
	return relative, true
}
