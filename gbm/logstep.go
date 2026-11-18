// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gbm computes lognormal increments and safe price publication.
// This file owns extended-range arithmetic; simulation order lives in simulate.go.
package gbm

import (
	"math"
	"math/big"
)

// workingBits retains meaningful cancellation when drift, diffusion variance,
// and shocks contain products of extreme but finite float64 inputs. A fixed
// precision keeps numeric policy independent of path length and worker count.
// The supported floating-point result is still an approximation to real GBM.
const workingBits uint = 2304

// expBits controls the rare high-precision conversion of cumulative log growth
// into a finite/subnormal float64 price. It exceeds float64's 53-bit significand
// sufficiently to resolve boundary rounding without using math.Log(initial).
const expBits uint = 384

const (
	// Safe fast-region limits keep both math.Exp(growth) and initial*exp(growth)
	// far from binary64 overflow and underflow; the generic converter handles
	// everything outside these limits without premature exponential overflow.
	fastInitialMin   = 1e-200
	fastInitialMax   = 1e200
	fastLogGrowthMax = 100.0
	// Even starting at the smallest positive float64, growth > 1500 exceeds
	// MaxFloat64; growth < -1500 underflows even from MaxFloat64.
	maxRelevantGrowth = 1500.0
	// Reducing |growth| to at most 1/8 makes the exponential Taylor series
	// rapidly convergent before repeated squaring restores the full exponent.
	expReductionLimit = 0.125
)

// wide constructs an independent, extended-exponent floating-point operand.
// It represents every finite float64 exactly; callers may mutate the result.
//
// Implementation:
//   - Stage 1: allocate a 2304-bit significand.
//   - Stage 2: copy the exact binary value of the source float64.
//   - Stage 3: return caller-owned arithmetic state.
//
// Inputs: x must be finite; no validation is done inside the numeric kernel.
// Returns: a detached *big.Float with a much wider exponent range than float64.
// Determinism: fixed precision and default ties-to-even rounding.
// Complexity: O(workingBits) memory and arithmetic initialization.
// AI-Hints: never convert a huge intermediate back to float64 to test progress.
func wide(x float64) *big.Float {
	return new(big.Float).SetPrec(workingBits).SetFloat64(x)
}

// newWide allocates scratch arithmetic state at the kernel's fixed precision.
// Every caller owns its instance; no mutable *big.Float is shared across paths.
func newWide() *big.Float {
	return new(big.Float).SetPrec(workingBits)
}

// stepArithmetic owns reusable precision-fixed operands for one trajectory.
// Reuse matters because wide square roots and exponent-correction arithmetic
// otherwise allocate fresh buffers for every observation in a long path.
// All fields are private, mutable, and strictly owned by one simulation call.
type stepArithmetic struct {
	previous   *big.Float
	current    *big.Float
	interval   *big.Float
	mu         *big.Float
	sigma      *big.Float
	drift      *big.Float
	scale      *big.Float
	correction *big.Float
	shock      *big.Float
	z          *big.Float
	half       *big.Float
	increment  *big.Float
}

// newStepArithmetic initializes scratch operands once for a validated path.
// It retains coefficients only within this call and never caches mutable state
// globally or aliases a caller-owned slice.
//
// Implementation:
//   - Stage 1: allocate mutable operands at the fixed wide precision.
//   - Stage 2: copy the exact binary coefficient values into owned buffers.
//   - Stage 3: return an independent workspace for sequential kernel steps.
//
// Inputs: finite mu and finite nonnegative sigma already validated publicly.
// Returns: isolated workspace with no dependencies on caller memory.
// Errors: none; input validation happens before allocation.
// Determinism: each workspace begins with the same precision and operands.
// Complexity: O(1) time and fixed-precision scratch memory.
// AI-Hints: workspaces must never be shared between concurrent trajectories.
func newStepArithmetic(mu, sigma float64) *stepArithmetic {
	work := &stepArithmetic{
		previous: newWide(), current: newWide(), interval: newWide(),
		mu: wide(mu), sigma: wide(sigma), drift: newWide(),
		scale: newWide(), correction: newWide(), shock: newWide(),
		z: newWide(), half: wide(0.5), increment: newWide(),
	}
	return work
}

// logIncrement evaluates one exact-distribution Ito logarithmic increment.
// The increment remains within workspace memory until the caller accumulates it.
// No float64 intermediary can overflow while building drift, Ito correction,
// or the Gaussian shock, even for extreme but individually finite operands.
//
// Implementation:
//   - Stage 1: subtract exact binary timestamps at wide precision.
//   - Stage 2: compute sigma*sqrt(dt) and its half-square at wide precision.
//   - Stage 3: combine drift, negative Ito correction, and scale*z in order.
//
// Behavior highlights:
//   - The wide square root avoids inaccurate cancellation at huge horizons.
//   - Every scratch operand is overwritten, not freshly allocated per step.
//
// Inputs:
//   - previous, current: validated consecutive timestamps, current>previous.
//   - z: one finite standard-normal draw already obtained by the driver.
//
// Returns: workspace-owned log increment, valid only until the next call.
// Errors: none; all public input failures are handled by Simulate.
// Determinism: exact timestamp order, fixed operand order and rounding mode.
// Complexity: fixed-size arithmetic O(1) per step; O(1) scratch memory.
// Notes: a returned pointer must not be retained beyond the current step.
// AI-Hints: avoid math.Sqrt(float64(interval)) and sigma*sigma in float64.
func (w *stepArithmetic) logIncrement(previous, current, z float64) *big.Float {
	w.previous.SetFloat64(previous)
	w.current.SetFloat64(current)
	w.interval.Sub(w.current, w.previous)
	w.drift.Mul(w.mu, w.interval)

	// Only stochastic calls enter this method. The analytic sigma=0 path
	// evaluates mu*t independently, with no Gaussian draw or square root.
	w.scale.Sqrt(w.interval)
	w.scale.Mul(w.scale, w.sigma)
	w.correction.Mul(w.scale, w.scale)
	w.correction.Mul(w.correction, w.half)
	w.z.SetFloat64(z)
	w.shock.Mul(w.scale, w.z)
	return w.increment.Sub(w.drift, w.correction).Add(w.increment, w.shock)
}

// priceFromGrowth publishes a positive float64 using an anchored log state.
// Growth is ln(S_t/S_0), rather than math.Log(S_0)+growth: retaining the exact
// initial float64 avoids irreversible log/exp roundtrips at price boundaries.
//
// Implementation:
//   - Stage 1: preserve zero growth exactly and exclude enormous log changes.
//   - Stage 2: use ordinary math.Exp only within a provably safe dynamic range.
//   - Stage 3: otherwise evaluate the exponential with extended-range arithmetic.
//
// Behavior highlights:
//   - A valid positive subnormal is accepted; a rounded zero or Inf is rejected.
//   - No per-step S_previous*exp(increment) update is performed.
//
// Inputs:
//   - initial: validated finite price >0, held constant throughout the path.
//   - growth: cumulative wide log growth; caller retains its ownership.
//
// Returns: positive finite float64 price, or zero and ErrOverflow.
// Errors: ErrOverflow when no positive finite price is representable.
// Determinism: one fixed fast-region criterion; otherwise fixed-precision series.
// Complexity: O(1) bounded arithmetic per price (precision constants are fixed).
// Notes: final float64 rounding is inevitable and must not be epsilon-clamped.
// AI-Hints: a finite growth does not ensure a representable final price.
func priceFromGrowth(initial float64, growth *big.Float) (float64, error) {
	if growth.Sign() == 0 {
		return initial, nil
	}
	approxGrowth, _ := growth.Float64()
	if !finite(approxGrowth) || math.Abs(approxGrowth) > maxRelevantGrowth {
		return 0, ErrOverflow
	}

	// This region is intentionally conservative: even the worst possible
	// multiplication is hundreds of decimal orders from a float64 boundary.
	if initial >= fastInitialMin && initial <= fastInitialMax &&
		math.Abs(approxGrowth) <= fastLogGrowthMax {
		price := initial * math.Exp(approxGrowth)
		if price > 0 && finite(price) {
			return price, nil
		}
	}

	return extendedPrice(initial, growth)
}

// extendedPrice converts an otherwise dangerous exponential without forcing
// either its intermediate value or the initial price through an invalid range.
// A high-precision exponential series is used only outside the cheap safe region.
//
// Implementation:
//   - Stage 1: halve log growth until its absolute value is at most 1/8.
//   - Stage 2: sum exp(x)'s convergent Taylor series at 384-bit precision.
//   - Stage 3: square back, multiply by exact initial, and round only once.
//
// Inputs: finite initial >0 and finite growth with magnitude <=1500.
// Returns: positive finite rounded price, or zero with ErrOverflow.
// Errors: ErrOverflow when final binary64 rounding yields 0 or infinity.
// Determinism: fixed reduction, ascending Taylor term, then ordered squaring.
// Complexity: O(1) time and scratch at the declared fixed precision.
// Notes: this converter is not a second GBM kernel; it only publishes a price.
// AI-Hints: do not replace the squaring with math.Pow or math.Exp(1500).
func extendedPrice(initial float64, growth *big.Float) (float64, error) {
	reduced := new(big.Float).SetPrec(expBits).Set(growth)
	limit := new(big.Float).SetPrec(expBits).SetFloat64(expReductionLimit)
	minusLimit := new(big.Float).SetPrec(expBits).Neg(limit)
	half := new(big.Float).SetPrec(expBits).SetFloat64(0.5)

	// Reduction avoids enormous intermediate exponentials; the original
	// magnitude is at most 1500, so at most 14 halvings are necessary.
	squarings := 0
	for reduced.Cmp(limit) > 0 || reduced.Cmp(minusLimit) < 0 {
		reduced.Mul(reduced, half)
		squarings++
	}

	one := new(big.Float).SetPrec(expBits).SetFloat64(1)
	sum := new(big.Float).SetPrec(expBits).Set(one)
	term := new(big.Float).SetPrec(expBits).Set(one)
	previous := new(big.Float).SetPrec(expBits)
	denominator := new(big.Float).SetPrec(expBits)

	// With |reduced|<=1/8, |term_n| <= (1/8)^n/n!: each successive
	// term shrinks by a factor <=1/(8*n). The loop necessarily reaches
	// a term that cannot change the 384-bit sum, including for x<0.
	for n := int64(1); ; n++ {
		term.Mul(term, reduced)
		denominator.SetInt64(n)
		term.Quo(term, denominator)
		previous.Set(sum)
		sum.Add(sum, term)
		if sum.Cmp(previous) == 0 {
			break
		}
	}

	for i := 0; i < squarings; i++ {
		sum.Mul(sum, sum)
	}
	initialWide := new(big.Float).SetPrec(expBits).SetFloat64(initial)
	priceWide := new(big.Float).SetPrec(expBits).Mul(initialWide, sum)
	price, _ := priceWide.Float64()
	if !finite(price) || price == 0 {
		return 0, ErrOverflow
	}
	return price, nil
}
