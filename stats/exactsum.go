// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats accumulates exact binary sums independently of central M2.
// This file implements a bounded two's-complement superaccumulator for means.
package stats

import (
	"encoding/binary"
	"math"
	"math/big"
	"math/bits"
)

const (
	// exactSumWords is sufficient to hold the signed sum of MaxUint64
	// finite float64 values without overflow, in units of 2^-1074:
	// the maximum occupied magnitude bit is 2097+64=2161, below the
	// 2175th two's-complement sign bit of 34 * 64-bit words.
	exactSumWords = 34

	// exactSumWordBits is the number of binary places in one accumulator word.
	exactSumWordBits = 64

	// floatFractionBits is the number of explicit IEEE-754 mantissa bits.
	floatFractionBits = 52

	// floatExponentMask selects the 11 IEEE-754 exponent bits.
	floatExponentMask = uint64(0x7ff)

	// floatFractionMask selects the explicit mantissa, excluding sign/exponent.
	floatFractionMask = (uint64(1) << floatFractionBits) - 1

	// exactSumScaleBits is the fixed-point exponent of the smallest float64.
	// Every finite float64 is an integer multiple of 2^-1074.
	exactSumScaleBits = 1074
)

// exactSum holds the exact signed sum of all committed finite float64 inputs.
// The representation is a fixed 2176-bit two's-complement integer measured in
// units of 2^-1074, so a large positive and negative pair can cancel without
// discarding even an intervening subnormal observation.
//
// The width covers at most MaxUint64 finite observations, all <=MaxFloat64
// in magnitude. Every operation is bounded, allocation-free, and independent
// of the number of processed samples; the zero value represents exact zero.
// Complexity: O(1) bounded storage (272 bytes), O(34) worst-case word work.
// AI-Hints: this exact sum is for Mean only; M2 must remain centered Welford/Chan.
type exactSum struct {
	// words contains little-endian unsigned limbs of the signed integer.
	// Negative numbers use two's complement; carries propagate toward higher words.
	words [exactSumWords]uint64
}

// addFloat adds the full binary value of a finite observation without rounding.
// Normal IEEE-754 values have an implicit leading mantissa bit; subnormals do
// not, so their shift relative to the fixed 2^-1074 unit differs by one.
// Implementation:
//   - Stage 1: unpack sign, biased exponent, and exact significand bits.
//   - Stage 2: align the 53-bit mantissa with its absolute fixed-point bit index.
//   - Stage 3: add or subtract at most two nonzero limbs and propagate carry.
//
// Inputs: value is a validated finite float64, including signed zero/subnormals.
// Returns: none; mutates only this accumulator's private words.
// Errors: none; public count capacity is checked before this operation.
// Determinism: fixed low-to-high carry/borrow propagation.
// Complexity: O(1) to O(exactSumWords) time, O(1) space; no heap allocation.
// Notes: the caller must apply this to a candidate state before publishing Update.
// AI-Hints: do not convert values to decimal; decode Float64bits exactly.
// nolint:gosec
func (sum *exactSum) addFloat(value float64) {
	// Stage 1: an IEEE signed zero contributes no integer bits.
	encoded := math.Float64bits(value)
	mantissa := encoded & floatFractionMask
	biasedExponent := int((encoded >> floatFractionBits) & floatExponentMask)
	if biasedExponent == 0 && mantissa == 0 {
		return
	}

	// For normal numbers x=mantissa*2^(biasedExponent-1075), so the
	// alignment to unit 2^-1074 is shift=biasedExponent-1.
	// For subnormals the mantissa already counts exact 2^-1074 units.
	shift := 0
	if biasedExponent != 0 {
		mantissa |= uint64(1) << floatFractionBits
		shift = biasedExponent - 1
	}
	wordIndex := shift / exactSumWordBits
	bitIndex := shift % exactSumWordBits
	low := mantissa << bitIndex
	var high uint64
	if bitIndex != 0 {
		high = mantissa >> (exactSumWordBits - bitIndex)
	}

	// Stage 2: the fixed-width integer has proven spare sign bits for
	// every allowed sample count; carries cannot exceed allocated width.
	negative := encoded>>(exactSumWordBits-1) != 0
	if negative {
		var borrow uint64
		sum.words[wordIndex], borrow = bits.Sub64(sum.words[wordIndex], low, 0)
		if wordIndex+1 < exactSumWords {
			sum.words[wordIndex+1], borrow = bits.Sub64(sum.words[wordIndex+1], high, borrow)
		}
		for index := wordIndex + 2; borrow != 0 && index < exactSumWords; index++ {
			sum.words[index], borrow = bits.Sub64(sum.words[index], 0, borrow)
		}
		return
	}

	// Addition uses the same two-limb alignment. After the first nonzero
	// limbs only a carry, not the whole original mantissa, can propagate.
	var carry uint64
	sum.words[wordIndex], carry = bits.Add64(sum.words[wordIndex], low, 0)
	if wordIndex+1 < exactSumWords {
		sum.words[wordIndex+1], carry = bits.Add64(sum.words[wordIndex+1], high, carry)
	}
	for index := wordIndex + 2; carry != 0 && index < exactSumWords; index++ {
		sum.words[index], carry = bits.Add64(sum.words[index], 0, carry)
	}
}

// add combines two independent exact sums without reducing numeric precision.
// A limb-by-limb two's-complement addition is sufficient for negative and
// positive operands; a checked count sum guarantees sufficient signed range.
// Implementation:
//   - Stage 1: start with carry zero at the least significant limb.
//   - Stage 2: add corresponding limbs and propagate a carry through all words.
//   - Stage 3: keep the resulting signed integer in the receiver's own array.
//
// Inputs: other is an immutable exactSum value, not an aliased slice.
// Returns: none; the destination is mutated and source remains unchanged.
// Errors: none, given the caller has checked combined Count capacity.
// Determinism: strict little-endian word order.
// Complexity: O(exactSumWords) time, O(1) additional space.
// AI-Hints: do not round individual partition sums before merging them.
func (sum *exactSum) add(other exactSum) {
	var carry uint64
	for index := 0; index < exactSumWords; index++ {
		sum.words[index], carry = bits.Add64(sum.words[index], other.words[index], carry)
	}
}

// signedInteger materializes the exact fixed-point integer for final rounding.
// No big.Int object is created by Update/Merge; arbitrary precision is used
// only once at a public Mean boundary, where correct cancellation matters.
// Implementation:
//   - Stage 1: inspect the highest sign bit for two's-complement negativity.
//   - Stage 2: convert every word into a big-endian local byte buffer.
//   - Stage 3: recover magnitude and apply sign to the detached big.Int.
//
// Inputs: receiver is an immutable, internally consistent exactSum value.
// Returns: detached *big.Int equal to the exact signed fixed-point sum.
// Errors: none.
// Determinism: fixed word order, with no map iteration.
// Complexity: O(exactSumWords) time/space, excluding allocated big.Int limbs.
// AI-Hints: conversion must not mutate the original accumulator words.
func (sum exactSum) signedInteger() *big.Int {
	negative := (sum.words[exactSumWords-1] >> (exactSumWordBits - 1)) != 0
	var encoded [exactSumWords * (exactSumWordBits / 8)]byte

	// Negative values are decoded by inverting all limbs and adding one.
	// All intermediate words live in the local encoded byte array.
	var carry uint64
	if negative {
		carry = 1
	}
	for index := 0; index < exactSumWords; index++ {
		word := sum.words[index]
		if negative {
			word, carry = bits.Add64(^word, 0, carry)
		}
		offset := (exactSumWords - index - 1) * (exactSumWordBits / 8)
		binary.BigEndian.PutUint64(encoded[offset:], word)
	}

	integer := new(big.Int).SetBytes(encoded[:])
	if negative {
		integer.Neg(integer)
	}
	return integer
}

// meanFloat returns the correctly rounded arithmetic mean of committed data.
// The numerator is an exact 2176-bit signed sum; the denominator combines
// Count and the fixed-point scale, so no cancellation can lose a small term.
// Implementation:
//   - Stage 1: reconstruct the detached exact binary sum integer.
//   - Stage 2: divide exactly by Count*2^1074 in big.Rat.
//   - Stage 3: round the final rational once to finite float64.
//
// Inputs: count is the positive, validated number of committed observations.
// Returns: nearest-even float64 arithmetic mean.
// Errors: ErrOverflow if the rounded mean cannot be represented as finite.
// Determinism: exact integer/rational arithmetic, then one IEEE conversion.
// Complexity: O(1) in observations; bounded ~2176-bit work and allocation.
// Notes: only summary/fitting calls pay this cost; per-sample updates allocate zero.
// AI-Hints: never compute mean by summing rounded partition means.
func (sum exactSum) meanFloat(count uint64) (float64, error) {
	// The Count>0 precondition is checked at public Summary/Fit boundaries.
	denominator := new(big.Int).SetUint64(count)
	denominator.Lsh(denominator, exactSumScaleBits)
	ratio := new(big.Rat).SetFrac(sum.signedInteger(), denominator)
	value, _ := ratio.Float64()
	if !finite(value) {
		return 0, ErrOverflow
	}
	return value, nil
}
