// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File: normalization.go
// Role: Anchor-preserving population Z scores and finite fitted diagnostics.
// AI-Hint: Never round a large absolute mean before subtracting it from samples.

package sax

import (
	"math"

	"github.com/lvlath/go/paa"
)

// compensatedSum accumulates bounded coordinates in a fixed traversal order.
// Neumaier compensation retains small terms when the running sum is larger,
// including cancellation between positive and negative observations.
// Its zero value is ready for use; it owns no shared or heap-allocated state.
type compensatedSum struct {
	sum        float64 // Rounded primary sum.
	correction float64 // Low-order terms lost by primary additions.
}

// add incorporates one finite bounded term into the compensated reduction.
// Inputs and all partial sums are bounded by the caller's scaled domain.
// Time O(1), space O(1); deterministic in the supplied term order.
func (s *compensatedSum) add(term float64) {
	next := s.sum + term // Ordinary addition may round away the smaller operand.
	if math.Abs(s.sum) >= math.Abs(term) {
		s.correction += (s.sum - next) + term // Recover the smaller term's residual.
	} else {
		s.correction += (term - next) + s.sum // Keep the residual when term dominates.
	}
	s.sum = next // Preserve the primary rounded sum for the next observation.
}

// total combines the primary sum and its accumulated correction once.
// Time O(1), space O(1); the result is an approximate binary64 reduction.
func (s *compensatedSum) total() float64 {
	return s.sum + s.correction
}

// smallestNormalFloat64 marks the boundary below which binary scaling may
// round away significand bits. Normal powers-of-two scaling is exact.
const smallestNormalFloat64 = 0x1p-1022

// normalizationFrame describes a local affine coordinate system x -> (x-origin)*2^-exponent.
// The power-of-two scale is represented by its exponent, never by a potentially
// overflowing reciprocal. It keeps the entire normalized range bounded.
// An origin of zero is safe for opposite-signed inputs, including overflowing spans.
type normalizationFrame struct {
	origin   float64 // Safe original-coordinate anchor.
	exponent int     // Binary exponent from Frexp of the nonzero span or max magnitude.
}

// newNormalizationFrame selects a finite subtraction and a bounded binary scale.
// It uses the minimum for same-sign windows and zero for sign-crossing windows.
// The latter choice also avoids needless cancellation when the true mean is near zero.
// Inputs are finite extrema of a nonconstant validated window. Time/space O(1).
func newNormalizationFrame(minimum, maximum float64) normalizationFrame {
	if minimum < 0 && maximum > 0 {
		// Zero is inside this interval. Subtracting it is exact and cannot overflow,
		// even when maximum-minimum itself is outside the finite float64 domain.
		_, exponent := math.Frexp(math.Max(-minimum, maximum))
		return normalizationFrame{exponent: exponent}
	}
	// Same-sign subtraction is finite; for nearby large observations it is
	// exact by Sterbenz's lemma. Scaling this difference preserves their spread.
	_, exponent := math.Frexp(maximum - minimum)
	return normalizationFrame{origin: minimum, exponent: exponent}
}

// normalizeWindow performs population Z normalization after anchoring the input.
// It keeps the small original-coordinate differences before binary scaling, and
// computes both dispersion and Z coordinates from one compensated local center.
//
// Implementation:
//   - Stage 1: find finite extrema and apply the explicit exact-constant policy.
//   - Stage 2: choose a safe affine frame; accumulate scaled offsets and their
//     centered squares in source order using compensated reductions.
//   - Stage 3: restore finite original-unit metadata and divide detached
//     residuals by the internal scaled population standard deviation.
//
// Behavior highlights:
//   - Nearby same-sign observations are subtracted before scaling. A zero
//     anchor avoids overflowing differences and cancellation across signs.
//   - A binary exponent keeps offsets and squares bounded without dividing
//     raw observations by an arbitrary maximum magnitude.
//   - ZeroWord is reserved for exact equality, not small variance. A positive
//     sigma rounded to zero is ErrOverflow, regardless of constant policy.
//
// Inputs:
//   - values: validated nonempty finite binary64 samples, read-only for the call.
//   - policy: validated exact-constant policy.
//
// Returns:
//   - []float64: detached window-local Z coordinates (exact zeros for ZeroWord).
//   - mean/stddev: finite original-unit metadata; positive sigma for nonconstants.
//   - bool: true only for the exact-constant ZeroWord case.
//   - error: nil on success; otherwise no coordinates or partial metadata.
//
// Errors:
//   - ErrZeroVariance: an exactly constant RejectConstant window.
//   - ErrOverflow: positive population sigma rounds to zero, or an exceptional
//     exact-mean reduction cannot complete.
//
// Determinism:
//   - Fixed extrema selection, ascending scans, and compensated reductions.
//     There is no cross-window mutable state or parallel reduction tree.
//
// Complexity:
//   - Time O(n), space O(n) including one detached coordinate workspace.
//     A rare extreme-exponent diagnostic mean uses one additional O(n) PAA
//     reduction (one segment, O(1) auxiliary storage).
//
// Notes:
//   - The affine frame avoids losing a representable small spread merely due to
//     a large common offset. It cannot recover information lost before entry.
//   - Very small terms may still round during exponent scaling or reduction.
//     The fitted public mean is rounded and is not the internal center.
//   - Population variance is approximate binary64 arithmetic, not an exact
//     rational or directed-rounding certificate.
//
// AI-Hints:
//   - Never use x-Result.Mean to reconstruct the normalized coordinates.
//   - Do not replace the binary frame with observation/maxAbs before centering.
//   - Keep the population divisor n; do not introduce an epsilon constant test.
func normalizeWindow(values []float64, policy ConstantPolicy) ([]float64, float64, float64, bool, error) {
	// Determine the finite hull without subtracting potentially opposite extremes.
	minimum, maximum := values[0], values[0]
	for _, observation := range values[1:] {
		minimum = math.Min(minimum, observation)
		maximum = math.Max(maximum, observation)
	}
	// Exact numerical equality includes either sign of zero. Preserve the first
	// original value in constant metadata, matching the previous ownership law.
	if minimum == maximum {
		if policy == RejectConstant {
			return nil, 0, 0, false, ErrZeroVariance
		}
		return make([]float64, len(values)), values[0], 0, true, nil
	}

	// The same-sign span is positive and finite. For crossed signs the safe
	// zero-origin maximum magnitude is positive and finite as well.
	frame := newNormalizationFrame(minimum, maximum)
	coordinates := make([]float64, len(values))
	var offsets compensatedSum
	var subnormalCoordinate bool
	for index, observation := range values {
		delta := observation - frame.origin // Subtract BEFORE any scaling.
		coordinate := math.Ldexp(delta, -frame.exponent)
		coordinates[index] = coordinate
		offsets.add(coordinate)
		// Downscaling a nonzero term into the subnormal domain can lose bits
		// here or during the later division by n, even if this scale was exact.
		// This is not a constancy test: the extrema still prove nonzero spread.
		if frame.origin == 0 && frame.exponent > 0 && delta != 0 &&
			math.Abs(coordinate) < smallestNormalFloat64 {
			subnormalCoordinate = true
		}
	}
	count := float64(len(values))
	offsetSum := offsets.total()
	center := offsetSum / count // Local population center, not rounded Mean.

	// Each offset has magnitude at most one; residual magnitude is at most two.
	// Thus squared terms are bounded by four and their sum cannot overflow for
	// any valid Go slice length. One endpoint residual has magnitude >= 1/4,
	// because the scaled nonconstant hull has diameter at least 1/2. Therefore
	// the scaled population sigma is strictly positive and representable.
	var squares compensatedSum
	for index, coordinate := range coordinates {
		residual := coordinate - center
		coordinates[index] = residual // Reuse the detached normalization workspace.
		squares.add(residual * residual)
	}
	sigma := math.Sqrt(squares.total() / count)

	// Restore the center in one scaled addition, then one exponent conversion.
	// Adding an already rounded original-unit offset would double-round means
	// at the subnormal boundary. The hull is a mathematical bound on the mean.
	mean := math.Ldexp(math.Ldexp(frame.origin, -frame.exponent)+center, frame.exponent)
	mean = math.Max(minimum, math.Min(maximum, mean))
	if subnormalCoordinate || (frame.origin == 0 && frame.exponent > 0 &&
		offsetSum != 0 && math.Abs(center) < smallestNormalFloat64) {
		// Extreme dynamic-range windows can lose tiny terms during binary scaling
		// even though their original-unit mean is representable after cancellation.
		// Use the existing exact binary-weighted PAA mean only for this diagnostic;
		// do not feed a rounded public Mean back into the internal Z calculation.
		exactMean, err := paa.Means(values, 1)
		if err != nil {
			return nil, 0, 0, false, ErrOverflow
		}
		mean = exactMean[0]
	}
	// Population sigma cannot exceed max(|min|,|max|). Clamp only a rounded
	// overshoot of that proven finite bound; no actual variance is suppressed.
	stddev := math.Min(math.Ldexp(sigma, frame.exponent), math.Max(math.Abs(minimum), math.Abs(maximum)))
	if stddev == 0 {
		return nil, 0, 0, false, ErrOverflow // Nonconstant sigma is unpublishable.
	}

	// Use internal residuals and internal sigma, never the rounded metadata.
	// Their quotient is bounded by sqrt(n) up to normal floating-point rounding.
	for index := range coordinates {
		coordinates[index] /= sigma
	}
	return coordinates, mean, stddev, false, nil
}
