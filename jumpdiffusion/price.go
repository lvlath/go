// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion converts anchored log growth into finite positive prices.
// This file follows the extended-exponentiation numeric policy of gbm/logstep.go.
package jumpdiffusion

import (
	"math"
	"math/big"
)

const (
	// growthPrecision keeps products, tiny elapsed times, and near-canceling
	// diffusion/jump terms out of the binary64 arithmetic domain.
	growthPrecision uint = 2304

	// expPrecision supplies guard bits for exponentiation near float64 limits.
	expPrecision uint = 384

	// expReductionLimit bounds the magnitude of the reduced exponential argument.
	expReductionLimit = 0.125

	// maxRelevantGrowth bounds all log changes capable of producing a positive
	// finite price from any positive finite float64 initial value.
	maxRelevantGrowth = 1500.0

	// Fast-path limits avoid overflow and underflow in initial*math.Exp(growth).
	fastInitialMin = 1e-200
	fastInitialMax = 1e200
	fastLogGrowth  = 100.0

	// maxExponentSeriesTerms is a defensive bound for the 384-bit Taylor kernel.
	// With |x|<=1/8, the series stagnates far before reaching this budget.
	maxExponentSeriesTerms = 1024
)

// wide constructs a caller-owned 2304-bit floating operand from one finite value.
// Conversion preserves every bit of the source's 53-bit significand.
//
// Inputs: validated finite float64; Returns: independent mutable *big.Float.
// Errors: none; Determinism: fixed precision and ties-to-even rounding.
// Complexity: O(1) fixed-precision storage/time.
// AI-Hints: don't replace this with a float64 intermediate in log arithmetic.
func wide(value float64) *big.Float {
	return new(big.Float).SetPrec(growthPrecision).SetFloat64(value)
}

// newWide allocates an uninitialized fixed-precision working operand.
// Each caller owns the result and may mutate it without synchronization.
//
// Returns: fresh 2304-bit *big.Float; Errors: none.
// Determinism: fixed precision; Complexity: O(1) fixed-size storage/time.
// AI-Hints: do not retain workspace pointers in public Result fields.
func newWide() *big.Float {
	return new(big.Float).SetPrec(growthPrecision)
}

// positivePrice publishes a price from exact initial float64 and wide log growth.
// Retaining the initial price as an anchor avoids Log(initial) round-trip damage.
//
// Implementation:
//   - Stage 1: preserve exact zero growth and reject log changes too extreme.
//   - Stage 2: use math.Exp only where intermediate and final ranges are safe.
//   - Stage 3: otherwise compute an extended-precision exponential and round once.
//
// Behavior highlights: positive float64 subnormals are accepted; zero is not.
// Inputs: initial>0 and finite; growth is a finite wide log(S(t)/S0) candidate.
// Returns: positive finite price, or zero with ErrOverflow.
// Errors: ErrOverflow for underflow to zero or overflow to infinity.
// Determinism: fixed branch bounds and original input anchor.
// Complexity: Time O(1), Space O(1) fixed-precision scratch.
// Notes: relative differences below half an ulp may round to unchanged price.
// AI-Hints: never compute math.Exp(growth) at 1000 and multiply by a tiny S0.
func positivePrice(initial float64, growth *big.Float) (float64, error) {
	if growth.Sign() == 0 {
		return initial, nil
	}

	approximate, _ := growth.Float64()
	if !finite(approximate) || math.Abs(approximate) > maxRelevantGrowth {
		return 0, ErrOverflow
	}

	if initial >= fastInitialMin && initial <= fastInitialMax && math.Abs(approximate) <= fastLogGrowth {
		price := initial * math.Exp(approximate)
		if price > 0 && finite(price) {
			return price, nil
		}
	}
	return extendedPrice(initial, growth)
}

// extendedPrice computes initial*exp(growth) without a binary64 intermediate.
// It uses argument halving, a 384-bit Taylor sum, then repeated squaring.
//
// Implementation:
//   - Stage 1: reduce |growth| to at most 1/8 by multiplying by 1/2.
//   - Stage 2: sum the convergent Taylor series until it stops changing.
//   - Stage 3: square back, multiply by the exact initial, and round once.
//
// Inputs: validated initial>0 and growth bounded in magnitude by 1500.
// Returns: positive finite rounded float64, or ErrOverflow.
// Errors: ErrOverflow for conversion failure or defensive convergence bound.
// Determinism: fixed reduction, ascending Taylor terms, sequential squaring.
// Complexity: O(1) bounded precision/time/space with at most 1024 series terms.
// Notes: this boundary is not a second stochastic simulation implementation.
// AI-Hints: never replace squaring with an unbounded float64 exponential.
func extendedPrice(initial float64, growth *big.Float) (float64, error) {
	reduced := new(big.Float).SetPrec(expPrecision).Set(growth)
	upper := new(big.Float).SetPrec(expPrecision).SetFloat64(expReductionLimit)
	lower := new(big.Float).SetPrec(expPrecision).Neg(upper)
	half := new(big.Float).SetPrec(expPrecision).SetFloat64(0.5)

	// Reducing by powers of two guarantees a rapidly convergent Taylor sum.
	// The reduction count is bounded since |growth|<=1500.
	squarings := 0
	for reduced.Cmp(upper) > 0 || reduced.Cmp(lower) < 0 {
		reduced.Mul(reduced, half)
		squarings++
	}

	sum := new(big.Float).SetPrec(expPrecision).SetFloat64(1)
	term := new(big.Float).SetPrec(expPrecision).SetFloat64(1)
	previous := new(big.Float).SetPrec(expPrecision)
	divisor := new(big.Float).SetPrec(expPrecision)
	converged := false
	for degree := int64(1); degree <= maxExponentSeriesTerms; degree++ {
		term.Mul(term, reduced)
		divisor.SetInt64(degree)
		term.Quo(term, divisor)
		previous.Set(sum)
		sum.Add(sum, term)
		if sum.Cmp(previous) == 0 {
			converged = true
			break
		}
	}
	if !converged {
		return 0, ErrOverflow
	}

	for index := 0; index < squarings; index++ {
		sum.Mul(sum, sum)
	}
	initialWide := new(big.Float).SetPrec(expPrecision).SetFloat64(initial)
	priceWide := new(big.Float).SetPrec(expPrecision).Mul(initialWide, sum)
	price, _ := priceWide.Float64()
	if !finite(price) || price == 0 {
		return 0, ErrOverflow
	}
	return price, nil
}
