// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 katalvlaran

package series

// GridOptions controls timestamp-grid validation.
// CadenceNS is a nonnegative interval in nanoseconds: zero checks strictly
// increasing order only; a positive value requires exact multiples.
// AllowGaps permits missing complete grid slots, but not nonmultiples.
// The zero value is valid and does not infer a cadence.
type GridOptions struct {
	CadenceNS int64
	AllowGaps bool
}

// Gap reports complete missing cadence slots between input indices
// AfterIndex and AfterIndex+1. MissingCount is strictly positive.
type Gap struct {
	AfterIndex   int
	MissingCount int64
}

// ValidationResult describes a validated input grid. Gaps is always non-nil
// on success, even if no cadence was specified or no gaps were found.
// The result and its slices are detached from the inputs.
type ValidationResult struct {
	Samples int
	Gaps    []Gap
}

// TransformResult describes a length-preserving causal transform.
// Values[i] is meaningful only when Valid[i] is true. Invalid warmup cells
// have Values[i] == 0. Both slices have exactly the input length and belong
// exclusively to the caller; valid values are always finite.
type TransformResult struct {
	Values []float64
	Valid  []bool
}

// Range is a half-open sample interval [Start, End), with 0 <= Start < End.
type Range struct {
	Start int
	End   int
}
