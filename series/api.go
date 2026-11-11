// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package series provides finite, causal transformations of ordered observations.
// This file exposes the validated public facades; mathematical kernels remain private.
package series

// Validate checks the correspondence between observations and strictly increasing
// signed nanosecond timestamps. It verifies an exact integer grid only when
// GridOptions.CadenceNS is positive; missing samples are never synthesized.
//
// Implementation:
//   - Stage 1: Check options, nil inputs, equal lengths, nonempty input, and finiteness.
//   - Stage 2: Traverse adjacent timestamps and classify order, overflow, and grid gaps.
//   - Stage 3: Return sample count and a detached, non-nil gap slice on success.
//
// Behavior highlights:
//   - CadenceNS == 0 enforces order and representable deltas without inferring a grid.
//   - A nonmultiple interval is ErrGap even if AllowGaps is true.
//
// Inputs:
//   - timesNS: read-only signed int64 nanosecond timestamps, one per observation.
//   - values: read-only, finite float64 observations with matching indices.
//   - opts: cadence and gap policy; the zero value checks order only.
//
// Returns:
//   - *ValidationResult: detached gap records, one sample count, and non-nil Gaps.
//   - error: nil on success; failure always returns a nil result, never a partial grid.
//
// Errors:
//   - ErrInvalidOption: negative cadence, before inspecting inputs.
//   - ErrNilInput, ErrShape, ErrEmptyInput: in that order for input structure.
//   - ErrNonFinite: NaN or infinity anywhere in values, before timestamp checks.
//   - ErrOrder: equal or decreasing consecutive timestamps.
//   - ErrOverflow: a positive nanosecond delta not representable as int64.
//   - ErrGap: nonmultiple interval or an unpermitted missing grid slot.
//
// Determinism:
//   - Timestamps are visited in ascending input-index order; gaps follow that order.
//
// Complexity:
//   - Time O(n), additional space O(g), for n observations and g reported gaps.
//
// Notes:
//   - Inputs are neither sorted nor mutated; concurrent caller mutation is unsupported.
//   - Gaps describe absent grid positions, not inserted values or elapsed calendar time.
//
// AI-Hints:
//   - Do not skip delta-overflow checking when CadenceNS == 0.
//   - Do not interpret AllowGaps as permission for off-grid timestamps.
func Validate(timesNS []int64, values []float64, opts GridOptions) (*ValidationResult, error) {
	return validateGrid(timesNS, values, opts)
}

// Lag shifts observations backward by a nonnegative number of sample positions.
// It preserves length and places zero/false warmup cells where no predecessor exists.
//
// Implementation:
//   - Stage 1: Validate lag and every input observation, including warmup-only data.
//   - Stage 2: Allocate detached values/mask buffers and copy available predecessors.
//   - Stage 3: Publish the complete result; no partial output is returned on error.
//
// Behavior highlights:
//   - lag == 0 is a detached identity, while lag >= len(values) marks all cells invalid.
//
// Inputs:
//   - values: read-only nonempty slice of finite float64 observations.
//   - lag: number of earlier positions, not a duration; must be >= 0.
//
// Returns:
//   - *TransformResult: len(values) Values/Valid cells; invalid cells are zero/false.
//
// Errors:
//   - ErrInvalidOption: negative lag, prior to input access.
//   - ErrNilInput, ErrEmptyInput, ErrNonFinite: shared input validation.
//
// Determinism:
//   - Valid cells are populated in increasing output-index order from values[i-lag].
//
// Complexity:
//   - Time O(n), space O(n), including full input validation and detached output.
//
// Notes:
//   - Neither the result nor its slices alias values; concurrent input mutation is unsafe.
//
// AI-Hints:
//   - A zero warmup value is not an observation: consult Valid before consuming it.
//   - lag >= n still validates every input for non-finite values.
func Lag(values []float64, lag int) (*TransformResult, error) {
	return lagTransform(values, lag)
}

// Difference computes an exact-lag finite difference for each available pair.
// For i >= lag, the mathematical operation is values[i]-values[i-lag]; earlier
// positions are zero/false rather than shorter output or synthetic samples.
//
// Implementation:
//   - Stage 1: Reject lag < 1, then validate the entire input for finiteness.
//   - Stage 2: Subtract each available pair and classify unrepresentable float64 results.
//   - Stage 3: Publish detached values/mask buffers only after the full computation.
//
// Behavior highlights:
//   - lag >= len(values) yields an all-invalid result without performing subtraction.
//
// Inputs:
//   - values: read-only nonempty finite float64 observations.
//   - lag: positive count of sample positions; must be >= 1.
//
// Returns:
//   - *TransformResult: one finite difference per valid cell, with input length.
//
// Errors:
//   - ErrInvalidOption: lag < 1, before input validation.
//   - ErrNilInput, ErrEmptyInput, ErrNonFinite: invalid input slice or observation.
//   - ErrOverflow: subtraction produces infinity or NaN from finite operands.
//
// Determinism:
//   - Input is scanned from i == lag to n-1 without reordering or parallel reduction.
//
// Complexity:
//   - Time O(n), space O(n), including validation and result buffers.
//
// Notes:
//   - Floating-point subtraction is rounded to float64; the input is never mutated.
//
// AI-Hints:
//   - Finite operands do not guarantee a finite difference; check the result.
//   - A lag is measured in indices, not nanoseconds or missing grid slots.
func Difference(values []float64, lag int) (*TransformResult, error) {
	return differenceTransform(values, lag)
}

// SimpleReturns computes one-step relative changes for finite observations.
// The mathematical return at i >= 1 is values[i]/values[i-1]-1; only a previous
// observation used as a denominator must be nonzero, so a final zero is valid.
//
// Implementation:
//   - Stage 1: Validate all finite inputs and every denominator before arithmetic.
//   - Stage 2: Use relative subtraction for nearby values, otherwise the quotient.
//   - Stage 3: Reject nonfinite results and publish detached length-preserving output.
//
// Behavior highlights:
//   - Negative observations are mathematically supported; financial meaning is caller-owned.
//   - A singleton returns a zero/false warmup cell, without requiring a denominator.
//
// Inputs:
//   - values: read-only nonempty slice of finite float64 observations.
//
// Returns:
//   - *TransformResult: first cell zero/false; later valid values are finite.
//
// Errors:
//   - ErrNilInput, ErrEmptyInput, ErrNonFinite: shared input validation.
//   - ErrDomain: values[i-1] == 0 for any computed pair.
//   - ErrOverflow: an unrepresentable computed return.
//
// Determinism:
//   - Denominators and output pairs are inspected in ascending index order.
//
// Complexity:
//   - Time O(n), space O(n), including validation and detached results.
//
// Notes:
//   - No timestamps or gap interpolation are considered; caller mutation is unsupported.
//
// AI-Hints:
//   - Reject only used zero denominators; do not reject a trailing zero on its own.
//   - Do not replace relative subtraction with division-minus-one for close values.
func SimpleReturns(values []float64) (*TransformResult, error) {
	return simpleReturnsTransform(values)
}

// LogReturns computes adjacent differences of natural logarithms.
// All observations must be strictly positive, including singleton data; for
// i >= 1 the mathematical value is ln(values[i])-ln(values[i-1]).
//
// Implementation:
//   - Stage 1: Validate the complete finite and strictly positive input domain.
//   - Stage 2: Use log1p for small relative changes and log differences otherwise.
//   - Stage 3: Check finiteness and return detached length-preserving output.
//
// Behavior highlights:
//   - Differences of logarithms avoid overflowing or underflowing an extreme ratio.
//   - Nearby distinct high-magnitude floats retain returns smaller than one log ULP.
//
// Inputs:
//   - values: read-only nonempty slice of strictly positive finite float64 values.
//
// Returns:
//   - *TransformResult: first cell zero/false; subsequent valid cells are finite.
//
// Errors:
//   - ErrNilInput, ErrEmptyInput, ErrNonFinite: shared input validation.
//   - ErrDomain: a nonpositive observation anywhere, including a singleton.
//   - ErrOverflow: a computed result fails the finite-output postcondition.
//
// Determinism:
//   - Inputs are validated and adjacent log returns computed in increasing index order.
//
// Complexity:
//   - Time O(n), space O(n), including complete validation and detached result.
//
// Notes:
//   - The results use IEEE-754 float64 rounding; inputs cannot be concurrently mutated.
//
// AI-Hints:
//   - Never compute log(curr/prev) for extreme inputs: the intermediate ratio overflows.
//   - Direct log subtraction may round tiny changes to zero; preserve the log1p path.
func LogReturns(values []float64) (*TransformResult, error) {
	return logReturnsTransform(values)
}

// WindowRanges enumerates all complete sample-index windows with fixed stride.
// The returned [Start,End) intervals have width elements, start at index zero,
// and never include a trailing incomplete window.
//
// Implementation:
//   - Stage 1: Validate dimension parameters and compute a division-safe window count.
//   - Stage 2: Reject unrepresentable slice storage, then allocate the output once.
//   - Stage 3: Emit ordered half-open ranges without overflowing start or end.
//
// Behavior highlights:
//   - length == 0 or width > length returns an empty but non-nil slice.
//   - stride > width is allowed; windows need not overlap or touch.
//
// Inputs:
//   - length: number of indexed samples, >= 0; unrelated to timestamp units.
//   - width: positive count of samples per window.
//   - stride: positive increment between consecutive start indices.
//
// Returns:
//   - []Range: ordered, detached half-open windows; never nil on success.
//
// Errors:
//   - ErrInvalidOption: negative length, nonpositive width or stride.
//   - ErrOverflow: complete-window storage size is not representable as int.
//
// Determinism:
//   - Start values increase by stride; the final complete window is included.
//
// Complexity:
//   - Time O(w), space O(w), where w is the number of emitted windows.
//
// Notes:
//   - Integer-size checks do not guarantee memory availability against system OOM.
//
// AI-Hints:
//   - Use a half-open End, not an inclusive final element or a partial tail.
//   - Check byte-size representability as well as the number of windows.
func WindowRanges(length, width, stride int) ([]Range, error) {
	return windowRanges(length, width, stride)
}
