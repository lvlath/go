// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package acf

// numeric.go implements scaled reductions for mean, energy, and cross-products.
// No estimator policy is selected or modified inside numerical helpers.

import "math"

// compensatedSum records a Neumaier correction to a deterministic signed reduction.
// It retains low-order contributions that ordinary summation would lose when
// adding small terms to much larger accumulated values.
//
// Complexity:
//   - O(1) state; each add and value evaluation take O(1) time.
//
// AI-Hints:
//   - Inputs must be scaled before multiplication; compensation cannot repair overflow.
type compensatedSum struct{ sum, correction float64 }

// add incorporates one finite summand using Neumaier's two-order error formula.
// The larger-magnitude operand determines which rearrangement recovers low bits;
// the branch does not change the fixed order in which samples are traversed.
//
// Inputs:
//   - x: finite signed summand, already scaled to a safe accumulation range.
//
// Complexity:
//   - Time O(1), Space O(1).
func (s *compensatedSum) add(x float64) {
	combined := s.sum + x
	if math.Abs(s.sum) >= math.Abs(x) {
		s.correction += (s.sum - combined) + x
	} else {
		s.correction += (x - combined) + s.sum
	}
	s.sum = combined
}

// value returns the compensated binary64 reduction result without modifying it.
// The final addition is rounded once at the public accumulation precision.
//
// Complexity:
//   - Time O(1), Space O(1).
func (s compensatedSum) value() float64 {
	return s.sum + s.correction
}

// midpoint selects an in-range origin for two ordered finite values.
// It avoids the overflow of lo+hi, uses a single rounding for finite spans,
// and handles an overflowing range through two individually halved operands.
//
// Inputs:
//   - lo, hi: finite extremes of one observation window, lo <= hi.
//
// Returns:
//   - float64: rounded midpoint used only as a numerical centering origin.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Do not replace the fallback by (lo+hi)/2; even finite data can overflow.
func midpoint(lo, hi float64) float64 {
	span := hi - lo
	if !math.IsInf(span, 0) {
		return math.FMA(0.5, span, lo)
	}
	return lo/2 + hi/2
}

// normalizedDeviations builds one globally centered, scale-free observation vector.
// It computes c_i/max(abs(c)) for centered ACF, or x_i/max(abs(x)) when
// NoCenter is set, without forming raw centered differences that may overflow.
//
// Implementation:
//   - Stage 1: for raw moments, scale the original data by its largest magnitude.
//   - Stage 2: for centering, choose a bounded midpoint origin, then scale deviations.
//   - Stage 3: sum normalized offsets with compensation, subtract that one global
//     mean shift, rescale residuals, and publish the rounded fitted mean.
//
// Behavior highlights:
//   - A representable tiny spread around a huge common offset remains observable.
//   - NoCenter publishes Mean=0 exactly, including for a nonzero constant signal.
//   - No input sample is replaced, removed, or copied into any persistent model.
//
// Inputs:
//   - values: finite nonempty observations validated by the facade, read-only.
//   - noCenter: preserve the physical raw zero rather than fit a sample mean.
//
// Returns:
//   - []float64: fresh normalized deviations, privately owned by the caller kernel.
//   - float64: rounded global sample mean, or exactly positive zero if noCenter.
//   - error: ErrZeroVariance for zero normalized energy, ErrOverflow if the mean is unrepresentable.
//
// Determinism:
//   - Min/max, scale, compensated mean, and final rescale each scan samples ascending.
//
// Complexity:
//   - Time O(n), Space O(n), with one n-element workspace allocation.
//
// Notes:
//   - The published rounded Mean may not reproduce the full internal centering
//     precision if a consumer subsequently subtracts it from large observations.
//
// AI-Hints:
//   - Never compute sum(x) or x-Mean directly for extreme-offset observations.
//   - Scaling must happen before squares; compensating an overflowed product is too late.
func normalizedDeviations(values []float64, noCenter bool) ([]float64, float64, error) {
	n := len(values)
	out := make([]float64, n)

	if noCenter {
		// A common positive scale cancels from S_k/S_0. Avoid unscaled squares
		// while keeping the original signed second-moment interpretation.
		scale := 0.0
		for _, v := range values {
			if a := math.Abs(v); a > scale {
				scale = a
			}
		}
		if scale == 0 {
			return nil, 0, ErrZeroVariance
		}
		for i, v := range values {
			out[i] = v / scale
		}
		return out, 0, nil
	}

	// Computing a midpoint reduces the spread before the mean reduction.
	// This preserves resolvable unit-scale differences near large offsets.
	lo, hi := values[0], values[0]
	for _, x := range values[1:] {
		if x < lo {
			lo = x
		}
		if x > hi {
			hi = x
		}
	}
	base := midpoint(lo, hi)

	// The in-range base makes every subtraction representable. Capture the
	// largest absolute difference so normalized terms lie within [-1,1].
	scale := 0.0
	for i, x := range values {
		deviation := x - base
		out[i] = deviation
		if magnitude := math.Abs(deviation); magnitude > scale {
			scale = magnitude
		}
	}
	if scale == 0 {
		return nil, 0, ErrZeroVariance
	}

	// The normalized mean shift is shared by every lag. Neumaier summation
	// keeps signed contributions from being lost in cancellation or large offsets.
	var aggregate compensatedSum
	for i, deviation := range out {
		out[i] = deviation / scale
		aggregate.add(out[i])
	}
	meanShift := aggregate.value() / float64(n)
	// FMA rounds the reconstructed mean once, even when meanShift*scale
	// alone would underflow at the subnormal boundary.
	mean := math.FMA(meanShift, scale, base)
	if math.IsInf(mean, 0) || math.IsNaN(mean) {
		return nil, 0, ErrOverflow
	}

	// Subtract the normalized shift, not the rounded public Mean. This protects
	// binary64 differences even if Result.Mean cannot encode the full offset.
	devScale := 0.0
	for i, normalized := range out {
		centered := normalized - meanShift
		out[i] = centered
		if magnitude := math.Abs(centered); magnitude > devScale {
			devScale = magnitude
		}
	}
	if devScale == 0 {
		return nil, 0, ErrZeroVariance
	}

	// A second scale bounds the centered coordinates after subtracting the
	// common mean; this scale also cancels exactly from all mathematical ratios.
	for i := range out {
		out[i] /= devScale
	}
	return out, mean, nil
}

// dotScaled computes one compensated cross-product sum for normalized samples.
// It forms S_k using exactly the n-k valid endpoint pairs and never recomputes
// the window mean or materializes unscaled products that can overflow.
//
// Implementation:
//   - Stage 1: initialize an independent signed Neumaier accumulator.
//   - Stage 2: scan i=lag through n-1 and accumulate values[i]*values[i-lag].
//   - Stage 3: return the compensated scalar sum at the normalized scale.
//
// Inputs:
//   - values: private finite normalized deviations with magnitude <= 1.
//   - lag: prevalidated integer in [0,n), selecting exactly n-lag pairs.
//
// Returns:
//   - float64: scaled signed cross-product sum for this lag.
//
// Determinism:
//   - Ascending endpoint i and sequential IEEE-754 multiplication/addition.
//
// Complexity:
//   - Time O(n-lag), Space O(1), without additional allocations.
//
// AI-Hints:
//   - Do not use pairwise means here; all lags share normalizedDeviations' single center.
func dotScaled(values []float64, lag int) float64 {
	var sum compensatedSum
	for i := lag; i < len(values); i++ {
		sum.add(values[i] * values[i-lag])
	}
	return sum.value()
}
