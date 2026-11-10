// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix provides finite statistical reductions with exact rescue for
// cancellation-sensitive float64 inputs. This file uses binary integer
// superaccumulators, avoiding a per-observation big.Rat allocation during the
// rare cancellation path and retaining the original float64 values exactly.
package matrix

import (
	"math"
	"math/big"
)

const (
	// statisticsBinaryFractionBits is the fraction width in IEEE-754 binary64.
	statisticsBinaryFractionBits = 52

	// statisticsExactInputShift maps every finite binary64 value to an exact
	// signed integer by multiplying it by 2^1074 (the least subnormal unit).
	statisticsExactInputShift = 1074

	// statisticsBinaryExponentMask selects the eleven stored exponent bits.
	statisticsBinaryExponentMask uint64 = 0x7ff

	// statisticsBinaryFractionMask selects the stored fraction without sign or
	// exponent. Normal numbers also acquire the implicit leading mantissa bit.
	statisticsBinaryFractionMask uint64 = (1 << statisticsBinaryFractionBits) - 1

	// statisticsExactPrecision is used only to convert an exact rational cross
	// numerator to a dimensionless Pearson coefficient. The binary integer
	// sums and products remain exact regardless of their exponent range.
	statisticsExactPrecision uint = 512
)

// statisticsFloatToInteger converts a validated finite binary64 observation into
// an exact integer on the shared 2^-1074 lattice, without decimal conversion.
// Every finite float64 is an integer multiple of the smallest positive subnormal.
// Implementation:
//   - Stage 1: inspect IEEE-754 sign, stored exponent, and fraction bits.
//   - Stage 2: restore the implicit leading bit for normal values and shift the
//     significand by its stored exponent minus one.
//   - Stage 3: apply the original sign and overwrite the destination big.Int.
//
// Behavior highlights:
//   - Positive and negative zero both map to exact integer zero.
//   - No per-observation rational object is allocated in the reduction loops.
//
// Inputs:
//   - destination: non-nil caller-owned big.Int workspace, overwritten on entry.
//   - value: finite original float64, already validated by statisticsReader.
//
// Returns:
//   - No return value; destination holds value * 2^1074 exactly.
//
// Errors:
//   - None; nonfinite input is rejected before the conversion boundary.
//
// Determinism:
//   - Direct bit decomposition with no input-dependent iteration or float summation.
//
// Complexity:
//   - O(B) bit work and O(B) destination storage, B at most 2098 for a
//     finite binary64 observation; no extra heap-backed rational object.
//
// Notes:
//   - This is a fixed binary lattice encoding, not a floating-point scaling
//     operation and not a format that callers should publish as matrix data.
//
// AI-Hints:
//   - Do not convert through decimal strings or a rounded big.Float; doing so
//     would destroy the small exact terms needed for covariance cancellation.
//   - This helper assumes finite input and is intentionally private.
func statisticsFloatToInteger(destination *big.Int, value float64) {
	bits := math.Float64bits(value)
	storedExponent := (bits >> statisticsBinaryFractionBits) & statisticsBinaryExponentMask
	mantissa := bits & statisticsBinaryFractionMask
	shift := uint(0)
	if storedExponent != 0 {
		// For normal numbers, the stored exponent starts at one while
		// the implicit leading mantissa bit contributes another 2^52.
		mantissa |= 1 << statisticsBinaryFractionBits
		shift = uint(storedExponent - 1)
	}

	// Multiplying by the common 2^1074 scale makes every shifted value
	// integral, including the smallest positive subnormal observation.
	destination.SetUint64(mantissa)
	destination.Lsh(destination, shift)
	if bits>>63 != 0 {
		destination.Neg(destination)
	}
}

// statisticsExactColumnMean recovers a column mean from the exact binary64 inputs.
// It also supplies the sub-float64 remainder required when materializing
// centered values whose true mean lies between adjacent float64 numbers.
// Implementation:
//   - Stage 1: convert each observation into the common 2^-1074 integer lattice.
//   - Stage 2: accumulate the complete signed integer sum in row order.
//   - Stage 3: divide by n*2^1074 exactly, round the public mean once, then
//     calculate the residual of the exact mean relative to that rounded value.
//
// Behavior highlights:
//   - Opposite enormous terms cancel before rounding, preserving a surviving
//     small observation rather than reporting a false zero mean.
//
// Inputs:
//   - reader: validated immutable matrix reader with at least one row.
//   - column: valid original feature index; observations are not modified.
//
// Returns:
//   - float64: rounded original-coordinate mean, required to be finite.
//   - float64: rounded low-order remainder to retain during centering.
//
// Errors:
//   - ErrOverflow: a required mean is not representable in finite float64.
//   - Wrapped Matrix.At error: generic backend fails at an accessed cell.
//
// Determinism:
//   - Row index increases monotonically; all additions before final rounding
//     are exact signed integer operations.
//
// Complexity:
//   - O(n*B) big-integer operations plus one final rational division, with
//     B bounded by the binary64 exponent range and log2(n); O(B) workspace.
//
// Notes:
//   - The low component is not a second public mean. It is private precision
//     retained only while materializing the centered matrix.
//
// AI-Hints:
//   - Never drop the low component for cancellation-sensitive columns; an
//     otherwise harmless rounded mean can erase a small centered observation.
//   - Do not allocate one big.Rat per sample; reuse the integer workspace.
func statisticsExactColumnMean(reader statisticsReader, column int) (float64, float64, error) {
	var sum, observation big.Int
	for i := 0; i < reader.rows; i++ {
		value, err := reader.value(i, column)
		if err != nil {
			return 0, 0, err
		}
		statisticsFloatToInteger(&observation, value)
		sum.Add(&sum, &observation)
	}

	// Mean = integerSum / (n*2^1074); n is positive for every caller.
	denominator := new(big.Int).Lsh(big.NewInt(int64(reader.rows)), statisticsExactInputShift)
	mean := new(big.Rat).SetFrac(&sum, denominator)
	published, _ := mean.Float64()
	if math.IsNaN(published) || math.IsInf(published, 0) {
		return 0, 0, ErrOverflow
	}

	// Retain the part lost in the public binary64 mean when the caller
	// explicitly needs materialized centered values.
	residual := new(big.Rat).Sub(mean, new(big.Rat).SetFloat64(published))
	low, _ := residual.Float64()
	return published, low, nil
}

// statisticsExactCovariance reconstructs one sample cross-covariance as a
// rational number from the original binary64 observations, not from rounded
// means or already-normalized floating-point moment accumulators.
// Implementation:
//   - Stage 1: encode both columns in the shared integer 2^-1074 lattice.
//   - Stage 2: accumulate exact signed sums and pairwise integer products.
//   - Stage 3: evaluate (n*sumXY-sumX*sumY)/(n*(n-1)*2^2148)
//     as a reduced rational number, postponing float64 publication.
//
// Behavior highlights:
//   - Protects a tiny cross-moment left after cancellation of enormous terms.
//   - No O(n*d) centered matrix or floating Gram product is materialized.
//
// Inputs:
//   - reader: validated read-only matrix with rows >= 2.
//   - j, k: original feature indices whose cross-covariance is requested.
//
// Returns:
//   - *big.Rat: newly owned exact covariance, including values too large for float64.
//
// Errors:
//   - Wrapped Matrix.At error from a custom input implementation.
//
// Determinism:
//   - Rows are processed in ascending order; integer reductions are exact.
//
// Complexity:
//   - O(n) bounded-width big-integer multiplications/additions and O(B) live
//     integer storage, where B grows only with float exponent width and log2(n).
//
// Notes:
//   - The outer caller decides whether the final covariance is representable;
//     this helper never substitutes an infinity or silently clips a result.
//
// AI-Hints:
//   - Do not replace the exact numerator by a floating-point Gram subtraction.
//   - This is an exceptional cancellation rescue, not the common O(1)-arithmetic
//     path for each observation pair.
func statisticsExactCovariance(reader statisticsReader, j, k int) (*big.Rat, error) {
	var sumJ, sumK, sumJK big.Int
	var left, right, product big.Int
	for i := 0; i < reader.rows; i++ {
		x, err := reader.value(i, j)
		if err != nil {
			return nil, err
		}
		y, err := reader.value(i, k)
		if err != nil {
			return nil, err
		}
		statisticsFloatToInteger(&left, x)
		statisticsFloatToInteger(&right, y)
		sumJ.Add(&sumJ, &left)
		sumK.Add(&sumK, &right)
		product.Mul(&left, &right)
		sumJK.Add(&sumJK, &product)
	}

	// Subtract after exact multiplication. A float64 Gram matrix would
	// overflow or erase a tiny surviving term in this same expression.
	count := big.NewInt(int64(reader.rows))
	numerator := new(big.Int).Mul(&sumJK, count)
	crossMeans := new(big.Int).Mul(&sumJ, &sumK)
	numerator.Sub(numerator, crossMeans)

	// Both the sample denominator and the common binary scaling factor
	// are combined as integers, avoiding a floating intermediate entirely.
	denominator := new(big.Int).Sub(count, big.NewInt(1))
	denominator.Mul(denominator, count)
	denominator.Lsh(denominator, 2*statisticsExactInputShift)
	return new(big.Rat).SetFrac(numerator, denominator), nil
}

// statisticsExactPearson evaluates Pearson correlation using three exact
// sample moments reconstructed from the original binary64 input values.
// This avoids reusing rounded normalized column lengths on the very path that
// was selected because cancellation made ordinary floating reduction unsafe.
// Implementation:
//   - Stage 1: require strictly positive exact sample variances; zero variance
//     is handled by the caller's zero-correlation policy before this method.
//   - Stage 2: convert the exact rational covariance and variances to fixed
//     512-bit binary floats and form sqrt(varianceJ * varianceK) without a
//     float64 intermediate or premature exponent overflow.
//   - Stage 3: divide the exact-numerator approximation by the extended-range
//     denominator, round the dimensionless answer to float64 and validate it.
//
// Behavior highlights:
//   - Unrepresentable sample covariance and variances can still yield a finite
//     Pearson coefficient in [-1,1]. Only the final coefficient is converted.
//
// Inputs:
//   - cross: exact sample cross-covariance, read-only for this operation.
//   - varianceJ, varianceK: positive exact sample variances, read-only.
//
// Returns: finite correlation coefficient computed with 512-bit precision.
// Errors: ErrOverflow when input variance is non-positive, denominator collapses,
// or the final float64 coefficient is not representable as a finite number.
// Determinism: fixed operand order and fixed math/big binary precision.
// Complexity: O(1) fixed-precision big.Float operations and O(1) workspace
// with respect to rows and columns; actual bit cost depends on sqrt/division;
// the per-column exact moments are computed and memoized by the caller.
// Notes: Sqrt and division are rounded at 512 bits; mathematical exactness of
// the input moments does not imply a universal correctly-rounded Pearson result.
// AI-Hints:
//   - Never reconstruct the denominator from the approximate column lengths.
//   - Never convert the individual covariance or variance to float64 first:
//     their magnitudes may exceed finite binary64 even when correlation fits.
func statisticsExactPearson(cross, varianceJ, varianceK *big.Rat) (float64, error) {
	if varianceJ.Sign() <= 0 || varianceK.Sign() <= 0 {
		return 0, ErrOverflow
	}

	makeFloat := func() *big.Float {
		return new(big.Float).SetPrec(statisticsExactPrecision).SetMode(big.ToNearestEven)
	}
	numerator := makeFloat().SetRat(cross)
	product := makeFloat().Mul(makeFloat().SetRat(varianceJ), makeFloat().SetRat(varianceK))
	denominator := makeFloat().Sqrt(product)
	if denominator.Sign() == 0 {
		return 0, ErrOverflow
	}
	coefficient, _ := makeFloat().Quo(numerator, denominator).Float64()
	if math.IsNaN(coefficient) || math.IsInf(coefficient, 0) {
		return 0, ErrOverflow
	}
	return coefficient, nil
}
