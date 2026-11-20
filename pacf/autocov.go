// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf constructs stable biased sample autocovariances without n-sized buffers.
// This file owns one whole-window centering frame, compensated reductions, and scaling.
package pacf

import (
	"context"
	"math"
)

// compensation performs a sequential Neumaier reduction of signed finite terms.
// It recovers low-order contributions when the running sum and new term differ in scale.
//
// Implementation:
//   - Stage 1: preserve the ordinary running sum and a separate correction.
//   - Stage 2: recover the floating-point addition error according to relative magnitudes.
//   - Stage 3: return the corrected total after all terms have been added.
//
// Behavior highlights:
//   - Every addition keeps the caller's traversal order, ensuring deterministic output.
//
// Inputs:
//   - Summands are already scaled into a bounded finite arithmetic domain.
//
// Returns:
//   - value() returns the compensated scalar in the same units as the summands.
//
// Errors:
//   - None directly; kernels guard input ranges and the final finite total.
//
// Determinism:
//   - Explicit sequential updates in the given sample or recurrence index order.
//
// Complexity:
//   - Time O(1), Space O(1) per addition and finalization.
//
// Notes:
//   - Compensation reduces rounding error but cannot rescue nonfinite products.
//
// AI-Hints:
//   - Never compute unscaled squared samples before calling add.
type compensation struct {
	sum        float64
	correction float64
}

// add incorporates one finite summand with a Neumaier error correction.
// It records the low-order part lost when ordinary floating-point addition rounds.
//
// Implementation:
//   - Stage 1: compute the rounded provisional sum.
//   - Stage 2: recover lost bits using the larger-magnitude argument as anchor.
//   - Stage 3: retain the rounded sum and correction for the next addition.
//
// Inputs:
//   - x: one finite, scaled summand; caller guarantees a bounded total.
//
// Returns:
//   - No value; updates only the accumulator receiver.
//
// Errors:
//   - None; a caller detects unrepresentable arithmetic where it can arise.
//
// Determinism:
//   - Exact caller-provided summand order and fixed magnitude branch.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Replacing this with ordinary sum += x loses small signed contributions.
func (s *compensation) add(x float64) {
	combined := s.sum + x
	if math.Abs(s.sum) >= math.Abs(x) {
		s.correction += (s.sum - combined) + x
	} else {
		s.correction += (x - combined) + s.sum
	}
	s.sum = combined
}

// value combines the correction with the running sum after a complete reduction.
// It performs one final rounded addition and does not mutate the accumulator.
//
// Inputs:
//   - s: compensated accumulator containing a completed deterministic reduction.
//
// Returns:
//   - float64: final signed sum, in the units of its summands.
//
// Errors:
//   - None directly; callers validate finite arithmetic where required.
//
// Determinism:
//   - One binary64 addition, independent of any external state.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Call only after the full reduction; do not round every correction in a hot loop.
func (s compensation) value() float64 {
	return s.sum + s.correction
}

// frame defines the common coordinate system of all lag-product reductions.
// It stores a stable origin, a positive common magnitude scale, and the fitted
// dimensionless shift, without allocating an n-element deviation vector.
//
// Behavior highlights:
//   - For Center=false, origin=shift=0 and scale=max(abs(values)).
//   - For Center=true, origin is a bounded midpoint and shift is the normalized mean.
//
// Determinism:
//   - The same frame is used at every lag; no per-lag recentering occurs.
//
// Complexity:
//   - Storage O(1); centered evaluates in O(1) for each sample.
//
// AI-Hints:
//   - center subtracts shift AFTER dividing by scale; never subtract a rounded mean first.
type frame struct {
	origin float64
	scale  float64
	shift  float64
}

// centered converts one observation to a normalized deviation from the global mean.
// It avoids forming unscaled centered differences that can overflow or cancel.
//
// Inputs:
//   - x: one previously validated finite sample; f is the same frame for every lag.
//
// Returns:
//   - float64: dimensionless (x-origin)/scale-shift, bounded near [-2,2].
//
// Errors:
//   - None; prepare guarantees f.scale>0 and bounded finite frame components.
//
// Determinism:
//   - One division and subtraction in a fixed order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - A rounded public mean can erase resolvable large-offset deviations; use the frame.
func (f frame) centered(x float64) float64 {
	return (x-f.origin)/f.scale - f.shift
}

// midpoint returns a finite midpoint of two ordered finite observations.
// It avoids overflow in lo+hi and handles even an unrepresentable full span.
//
// Inputs:
//   - lo, hi: finite minimum and maximum input values, lo<=hi.
//
// Returns:
//   - float64: rounded origin between the observed endpoints.
//
// Errors:
//   - None: a midpoint of finite ordered values is representable.
//
// Determinism:
//   - An explicit span-overflow branch selects a fixed arithmetic path.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Do not simplify the overflow branch to (lo+hi)/2.
func midpoint(lo, hi float64) float64 {
	span := hi - lo
	if math.IsInf(span, 0) {
		// Both separately halved endpoints are representable even when hi-lo is not.
		return lo/2 + hi/2
	}
	// FMA performs the affine combination with one final rounding.
	return math.FMA(0.5, span, lo)
}

// prepare learns one scale-safe reference frame from the full training window.
// It preserves representable local fluctuations around large global offsets
// and forms the mean shift using compensated normalized deviations.
//
// Implementation:
//   - Stage 1: for centered data, scan extrema to select a finite midpoint origin.
//   - Stage 2: scan the maximum absolute deviation from that origin for scaling.
//   - Stage 3: if centered, accumulate normalized deviations and divide by n once.
//
// Behavior highlights:
//   - Uncentered nonzero constants are valid because their scale is nonzero.
//   - A centered constant has exactly zero scale and returns ErrZeroVariance.
//   - No n-element centered buffer is created or retained.
//
// Inputs:
//   - ctx: active context polled throughout all sample scans.
//   - x: nonempty finite chronological samples, read-only until completion.
//   - center: whether the unique whole-window sample mean is fitted.
//
// Returns:
//   - frame: finite origin/shift and positive physical scale.
//   - error: nil on success; zero frame on computational failure is not usable.
//
// Errors:
//   - ErrZeroVariance: no nonzero raw energy or no spread after chosen origin.
//   - ctx.Err(): cancellation during any stage, before covariance construction.
//
// Determinism:
//   - The extrema, scale, and mean reductions scan sample indices ascending.
//
// Complexity:
//   - Time O(n), Space O(1), n=len(x).
//
// Notes:
//   - Means may lie between representable samples; normalized shift retains sub-ULP detail.
//
// AI-Hints:
//   - Never compute mean as sum(x)/n on large-offset data or mean via E[x*y]-mean^2.
//   - Summing normalized coordinates before dividing by n prevents early tiny-term loss.
func prepare(ctx context.Context, x []float64, center bool) (frame, error) {
	var f frame
	if center {
		// A midpoint avoids choosing an unnecessarily distant origin for large offsets.
		lo, hi := x[0], x[0]
		for i, value := range x {
			if i%cancellationPollInterval == 0 {
				if err := ctx.Err(); err != nil {
					return frame{}, err
				}
			}
			if value < lo {
				lo = value
			}
			if value > hi {
				hi = value
			}
		}
		f.origin = midpoint(lo, hi)
	}

	// For uncentered data, origin remains the exact physical zero.
	// The largest centered/raw deviation bounds every normalized coordinate.
	for i, value := range x {
		if i%cancellationPollInterval == 0 {
			if err := ctx.Err(); err != nil {
				return frame{}, err
			}
		}
		if magnitude := math.Abs(value - f.origin); magnitude > f.scale {
			f.scale = magnitude
		}
	}
	if f.scale == 0 {
		return frame{}, ErrZeroVariance
	}

	if center {
		// Summation happens in the normalized domain, so no raw offset can overflow.
		// A single division at the end avoids prematurely underflowing small terms.
		var average compensation
		for i, value := range x {
			if i%cancellationPollInterval == 0 {
				if err := ctx.Err(); err != nil {
					return frame{}, err
				}
			}
			average.add((value - f.origin) / f.scale)
		}
		f.shift = average.value() / float64(len(x))
	}

	// A final cancellation check prevents preparing a successful frame after cancellation.
	if err := ctx.Err(); err != nil {
		return frame{}, err
	}
	return f, nil
}

// autocov calculates biased sample autocovariance at every lag from zero to L.
// It always divides by the same n and uses prepare's one shared centering frame.
//
// Implementation:
//   - Stage 1: allocate L+1 scaled covariances after public validation.
//   - Stage 2: scan each lag k and its n-k observed endpoint pairs in ascending order.
//   - Stage 3: compensate signed products, reject nonfinite totals and zero lag energy.
//
// Behavior highlights:
//   - All lags use the single 1/n denominator, never 1/(n-k).
//   - Products are formed after both deviations are scaled, not before division.
//   - The returned covariance vector is private workspace, not a public result artifact.
//
// Inputs:
//   - ctx: non-nil context polled at each lag and periodically over observations.
//   - x: finite input samples, never modified or retained.
//   - lag: inclusive maximum lag, already validated in [0,len(x)-1].
//   - f: whole-window frame with positive finite scale.
//
// Returns:
//   - []float64: dimensionless gamma[k] for k=0..lag, sharing a common physical scale.
//   - error: nil on success; any error discards the entire private covariance vector.
//
// Errors:
//   - ErrZeroVariance: normalized gamma[0] is zero for the selected estimand.
//   - ErrOverflow: a required covariance sum is nonfinite.
//   - ctx.Err(): canceled scan, including final publication check.
//
// Determinism:
//   - Lags k increase from zero; endpoint t increases from k to n-1.
//
// Complexity:
//   - Time O(n*(lag+1)), additional Space O(lag+1), without an n-element copy.
//
// Notes:
//   - gamma is rescaled to physical units only when publishing prediction variances.
//
// AI-Hints:
//   - Never recenter separate lag pairs or substitute an n-k denominator.
func autocov(ctx context.Context, x []float64, lag int, f frame) ([]float64, error) {
	gamma := make([]float64, lag+1)
	invN := 1 / float64(len(x))

	for k := 0; k <= lag; k++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var sum compensation

		// Every product uses the same global frame; only the number of pairs changes.
		for t := k; t < len(x); t++ {
			if t%cancellationPollInterval == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			left := f.centered(x[t])
			right := f.centered(x[t-k])
			sum.add((left * right) * invN)
		}

		gamma[k] = sum.value()
		if math.IsNaN(gamma[k]) || math.IsInf(gamma[k], 0) {
			return nil, ErrOverflow
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if gamma[0] <= 0 {
		return nil, ErrZeroVariance
	}
	return gamma, nil
}

// restoreVariance converts a positive scaled variance into squared input units.
// It decomposes both the variance and physical scale into binary exponents,
// avoiding premature overflow and underflow during their squared product.
//
// Implementation:
//   - Stage 1: decompose scaled v and scale into bounded binary significands.
//   - Stage 2: multiply bounded significands, then combine the integer exponents.
//   - Stage 3: use Ldexp to round once at the desired physical magnitude.
//
// Behavior highlights:
//   - A tiny normalized variance may still produce a finite physical variance.
//   - There is no epsilon clamp for zero, negative, or unrepresentable results.
//
// Inputs:
//   - v: strictly positive finite normalized prediction variance.
//   - scale: strictly positive finite common scale in original observation units.
//
// Returns:
//   - float64: finite positive prediction variance in squared original units.
//   - error: nil on success, otherwise zero and ErrOverflow.
//
// Errors:
//   - ErrOverflow: output is infinite, NaN, zero or otherwise unrepresentable.
//
// Determinism:
//   - The same Frexp/Ldexp decomposition and binary64 operations for all lags.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The required public result uses float64 and cannot express positive sub-minnormal values.
//
// AI-Hints:
//   - Do not compute scale*scale before multiplying v, nor (v*fraction)*fraction.
//   - Combining both exponents first preserves small v whose physical variance is large.
func restoreVariance(v, scale float64) (float64, error) {
	if v <= 0 || math.IsNaN(v) || math.IsInf(v, 0) ||
		scale <= 0 || math.IsNaN(scale) || math.IsInf(scale, 0) {
		return 0, ErrOverflow
	}

	// Each fraction lies in [0.5,1), so their product is always representable.
	varianceFraction, varianceExponent := math.Frexp(v)
	scaleFraction, scaleExponent := math.Frexp(scale)
	physicalFraction := (varianceFraction * scaleFraction) * scaleFraction
	physical := math.Ldexp(physicalFraction, varianceExponent+2*scaleExponent)

	// Reject both overflow and underflow-to-zero: callers require positive variances.
	if physical <= 0 || math.IsNaN(physical) || math.IsInf(physical, 0) {
		return 0, ErrOverflow
	}
	return physical, nil
}
