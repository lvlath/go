// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 katalvlaran

package series

// Validate checks one finite observation per strictly increasing timestamp,
// measured in signed int64 nanoseconds. A zero cadence checks order only;
// a positive cadence enforces exact intervals, optionally reporting gaps.
// Inputs remain unchanged. Time O(n); additional space O(number of gaps).
// Errors: ErrInvalidOption, ErrNilInput, ErrShape, ErrEmptyInput,
// ErrNonFinite, ErrOrder, ErrOverflow, or ErrGap, in validation order.
func Validate(timesNS []int64, values []float64, opts GridOptions) (*ValidationResult, error) {
	return validateGrid(timesNS, values, opts)
}

// Lag copies values from lag positions earlier; lag zero is a detached copy.
// Missing prefix positions have value zero and validity false. A lag >= len
// produces only invalid cells. Time/space O(n). Errors: ErrInvalidOption,
// ErrNilInput, ErrEmptyInput, or ErrNonFinite.
func Lag(values []float64, lag int) (*TransformResult, error) {
	return lagTransform(values, lag)
}

// Difference calculates values[i]-values[i-lag] for lag >= 1.
// Missing prefix cells are zero/false. Time/space O(n).
// Errors: ErrInvalidOption, ErrNilInput, ErrEmptyInput, ErrNonFinite,
// or ErrOverflow if a difference cannot be represented as finite float64.
func Difference(values []float64, lag int) (*TransformResult, error) {
	return differenceTransform(values, lag)
}

// SimpleReturns calculates values[i]/values[i-1]-1 for i >= 1.
// Negative observations are permitted, but every used previous value must
// be nonzero. The first cell is zero/false. Time/space O(n).
// Errors: ErrNilInput, ErrEmptyInput, ErrNonFinite, ErrDomain, ErrOverflow.
func SimpleReturns(values []float64) (*TransformResult, error) {
	return simpleReturnsTransform(values)
}

// LogReturns calculates log(values[i])-log(values[i-1]) for i >= 1.
// Every observation, including a singleton, must be strictly positive.
// Using differences of logarithms avoids overflow from forming large ratios.
// The first cell is zero/false. Time/space O(n).
// Errors: ErrNilInput, ErrEmptyInput, ErrNonFinite, ErrDomain, ErrOverflow.
func LogReturns(values []float64) (*TransformResult, error) {
	return logReturnsTransform(values)
}

// WindowRanges enumerates complete half-open intervals of width samples,
// starting at multiples of stride. length may be zero; width and stride
// must be positive. Width > length yields an empty non-nil slice.
// Time/space O(number of ranges). Errors: ErrInvalidOption, ErrOverflow.
func WindowRanges(length, width, stride int) ([]Range, error) {
	return windowRanges(length, width, stride)
}
