// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package acf

// types.go defines estimator policy, configuration, and result metadata.
// Public slices are owned by callers; the computational workspace remains private.

// Denominator selects the divisor convention for sample autocorrelation.
// It distinguishes Toeplitz-PSD-compatible common normalization from
// pair-count-adjusted normalization, which is generally not PSD compatible.
//
// Behavior highlights:
//   - Its zero value means CommonDenominator, not an automatic estimator choice.
//   - Values other than the named constants are rejected before input inspection.
//
// Complexity:
//   - Storage O(1); selection adds O(1) work per computed lag.
//
// AI-Hints:
//   - A denominator convention is a mathematical policy, not a numerical optimization.
//   - Do not infer it from the magnitude of computed coefficients.
type Denominator uint8

const (
	// CommonDenominator divides all lag cross-products by the same S0.
	// It equals S_k/S_0, and its finite-sample Toeplitz sequence is PSD in exact arithmetic.
	CommonDenominator Denominator = iota

	// LagAdjusted corrects cross-products for the n-k available pairs at lag k.
	// It equals (n/(n-k))*(S_k/S_0), may exceed [-1,1], and has no general PSD guarantee.
	LagAdjusted
)

// Options configures a single fixed-window autocorrelation computation.
// The zero value specifies centered CommonDenominator; its fields are copied
// by value into Compute, so callers need not allocate a configuration object.
//
// Behavior highlights:
//   - Both conventions use one global sample mean when NoCenter is false.
//   - NoCenter makes the raw physical zero the origin and sets Result.Mean to zero.
//
// Inputs:
//   - Denominator: one of CommonDenominator or LagAdjusted; the zero value is common.
//   - NoCenter: true to retain the original zero origin, false to fit a global sample mean.
//
// Errors:
//   - Compute returns ErrInvalidOption for unknown Denominator values.
//
// Complexity:
//   - O(1) configuration storage; kernel complexity is unchanged.
//
// AI-Hints:
//   - NoCenter is not a switch for stationarity tests or trend removal.
//   - Use an explicit pair-count estimator only if its mathematical interpretation fits the task.
type Options struct {
	// Denominator records the normalization applied to every nonzero lag.
	// The field is public to support value-literal configuration without mutable globals.
	Denominator Denominator

	// NoCenter disables subtraction of the whole-window sample mean.
	// It does not change the observation window, validation rules, or time origin.
	NoCenter bool
}

// Result contains the computed ACF and its exact effective sample-pair counts.
// It is the canonical public artifact for Compute, with no pointers to its input
// or internal scratch space and no partial publication on failure.
//
// Behavior highlights:
//   - For positive energy Values[0]==1 and PairCounts[k]==n-k for every reported lag.
//   - Mean stores the rounded full-window mean when centered; it is exactly zero otherwise.
//   - Denominator states the actual convention, not a default inferred by the reader.
//
// Returns:
//   - Both slices are detached, independently allocated, and caller-owned after success.
//
// Determinism:
//   - Values and PairCounts are indexed by ascending lag, including lag zero.
//
// Complexity:
//   - O(maxLag+1) retained storage; Clone copies both slices in O(maxLag+1).
//
// Notes:
//   - The struct is mutable by its owner; concurrent mutation and reads require caller synchronization.
//   - Mean is rounded metadata; scaled centering may retain more precision than recomputing x-Mean.
//   - A nil *Result is a failed or absent result, and IsNil is safe on that receiver.
//
// AI-Hints:
//   - Pass Result.Denominator along with Values to avoid silently mixing estimators.
//   - Do not infer the observation count only from len(Values); consult PairCounts[0].
type Result struct {
	// Values holds r_0 through r_maxLag inclusive, with r_0 exactly one on success.
	// The caller may mutate this slice without affecting future Compute calls.
	Values []float64

	// PairCounts holds n-k for each k; its length exactly matches Values.
	// These are structural pair counts, not effective statistical degrees of freedom.
	PairCounts []int

	// Mean is the fitted full-sample mean or exactly zero when NoCenter was enabled.
	// It is expressed in the original observation units and rounded to binary64.
	Mean float64

	// Denominator is the estimator convention used for every nonzero lag.
	// Store it with Values when exchanging results with another package.
	Denominator Denominator
}

// IsNil reports whether the Result pointer is nil, including typed nil interfaces.
// This method structurally implements the core.Nilable convention without
// importing the graph substrate into this independent scalar package.
//
// Implementation:
//   - Stage 1: compare the receiver pointer with nil; do not dereference it.
//   - Stage 2: return true for a nil receiver and false for an allocated one.
//
// Behavior highlights:
//   - An allocated zero-value Result is not nil, even if its slices are nil.
//
// Inputs:
//   - r: potentially nil *Result receiver, including a typed nil in an interface.
//
// Returns:
//   - bool: true exactly when r == nil; no allocations or side effects.
//
// Errors:
//   - None; nil receivers are supported without panics.
//
// Determinism:
//   - Pure pointer comparison, independent of Result field contents.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This tests pointer presence, not whether the stored coefficients are valid.
//
// AI-Hints:
//   - Do not use reflection or inspect slices to implement this nil-state protocol.
func (r *Result) IsNil() bool {
	return r == nil
}

// Clone returns a detached deep copy of the caller-owned ACF result slices.
// It preserves coefficient order, effective pair counts, mean, and estimator
// convention while allowing independent mutation of the copy.
//
// Implementation:
//   - Stage 1: return nil for a nil receiver without dereferencing it.
//   - Stage 2: copy scalar metadata and allocate copies of any non-nil result slices.
//   - Stage 3: return the independent copy; no source/result buffers are shared.
//
// Behavior highlights:
//   - An allocated zero-value Result clones to a non-nil zero-value Result.
//   - Nil and non-nil empty slices retain their distinction after cloning.
//
// Inputs:
//   - r: immutable for the duration of Clone; may be nil.
//
// Returns:
//   - *Result: deep copy of both slices, or nil if r is nil.
//
// Errors:
//   - None; slice allocation is bounded by existing slice lengths.
//
// Determinism:
//   - Slices are copied in their existing index order without re-sorting or reduction.
//
// Complexity:
//   - Time O(k), Space O(k), k=len(Values)+len(PairCounts).
//
// Notes:
//   - Concurrent mutation of the source Result during Clone is unsupported.
//
// AI-Hints:
//   - A shallow struct copy would alias exported slices and violate the ownership guarantee.
func (r *Result) Clone() *Result {
	if r == nil {
		return nil
	}

	copyResult := *r
	if r.Values != nil {
		copyResult.Values = append([]float64{}, r.Values...)
	}
	if r.PairCounts != nil {
		copyResult.PairCounts = append([]int{}, r.PairCounts...)
	}
	return &copyResult
}
