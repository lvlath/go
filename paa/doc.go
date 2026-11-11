// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package paa computes deterministic fractional Piecewise Aggregate Approximation
// for uniformly spaced finite float64 observations. It reduces a step-function
// series from n original sample-time intervals to w equal-duration averages.
//
// Domain and mathematical laws:
//   - The input value at i describes [i,i+1); segment j describes
//     [j*n/w,(j+1)*n/w), with exact unsimplified rational boundaries.
//   - All source interval measure is conserved; there is no tail truncation,
//     input resampling, zero-padding, sorting, or smoothing.
//   - Each output average is a convex combination of intersecting input values.
//     Its represented float64 is finite and inside their closed value range.
//   - The scalar mean is rounded once from an exact binary weighted numerator
//     to nearest-even float64; w=n preserves original float64 bits, including -0.
//   - (n/w)*sum(output) == sum(input) in real arithmetic; float64 operations
//     are rounded, so consumers should test conservation with scaled tolerances.
//
// Public surface:
//   - Transform(values,w) returns the canonical Result with exact geometry.
//   - TransformRange(values,start,end,w) checks a half-open source interval and
//     returns identical window-local geometry without copying source samples.
//   - Means(values,w) returns only detached averaged values; its geometry and
//     source length metadata are deliberately discarded.
//   - The requested segment count is explicit; there is no Options mode or
//     default partition size that could change the geometric contract.
//
// Shape, errors and partial-result law:
//   - Required 1<=w<=n; w=0 is invalid, not a default or empty output.
//   - Transform validation order: w<=0, nil input, empty input, w>n,
//     NaN/Inf, checked int64 products/output sizes, then compute.
//   - TransformRange checks nonpositive w before nil/empty source; checked
//     bounds and window size follow source presence, before numeric validation.
//     Only observations inside an accepted range are checked for finiteness.
//   - Stable sentinels are ErrInvalidOption, ErrNilInput, ErrEmptyInput,
//     ErrNonFinite and ErrOverflow. Test them with errors.Is.
//   - Errors return a nil *Result; partial results are never published.
//
// Nilability:
//   - *Result implements core.Nilable through IsNil without importing core.
//   - A nil *Result reports true even when held by an interface. Any non-nil
//     *Result (including its zero value) reports false; this does not validate
//     result completeness or numerics.
//
// Ownership, determinism, concurrency and complexity:
//   - Results own detached slices; the input slice is never mutated or retained.
//   - A single ordered left-to-right integer-overlap sweep defines the order;
//     segment numbering is ascending and boundaries are compared exactly.
//   - Calls have no shared mutable state. Concurrent calls are supported if the
//     caller does not concurrently change any input being read.
//   - Time O(n+w), auxiliary memory O(w) for the two output slices;
//     for a range, n is the selected window length. Means also allocates the
//     temporary O(w) segment geometry because it delegates to Transform.
//
// Non-goals:
//   - No irregular-time aggregation, missing-data imputation, normalization,
//     symbolic discretization, indexing, LB_Keogh, DTW, or distance calculation.
//   - The input float64 bit patterns are treated exactly; only the final
//     segment mean is rounded to nearest, ties to even.
//
// AI-Hints:
//   - Never cast fractional segment boundaries to input sample indices.
//   - Do not silently change w, ignore the last observations, or fill with zeros.
//   - Exact geometry does not mean arithmetic on the published rounded means
//     preserves the total integral bit-for-bit.
//   - Segment positions returned by TransformRange are local to the window;
//     do not treat them as absolute indexes or timestamps.
package paa
