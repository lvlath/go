// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pca freezes the high-precision center of the covariance coordinates.
// This file keeps training and inference in the same affine coordinate system
// even when a mean lies between adjacent representable float64 observations.
package pca

import (
	"context"
	"math"
	"math/big"

	"github.com/lvlath/go/matrix"
)

// fittedCenter represents a prepared-feature mean as anchor+offset+low.
// The anchor is the first prepared training observation, not a rounded mean;
// offset and low are the correctly rounded high/low parts of its exact mean
// displacement. Keeping the offset separately preserves sub-ULP centers.
//
// Implementation:
//   - Stage 1: choose the fixed first training value as the difference origin.
//   - Stage 2: accumulate exact binary64 differences, divided by sample count.
//   - Stage 3: freeze two finite displacement parts for inference and inversion.
//
// Behavior highlights:
//   - A mean between adjacent large float64 values remains representable as
//     a pair of floats even when no single rounded mean stores it correctly.
//
// Inputs:
//   - Created internally from an unmodified, finite training matrix.
//
// Returns:
//   - Its center and restore operations use this immutable fitted origin.
//
// Errors:
//   - ErrOverflow when a necessary finite center displacement is unrepresentable.
//
// Determinism:
//   - The anchor is row zero; subsequent differences are accumulated in row order.
//
// Complexity:
//   - Persistent O(1) float64 state per feature; exact fitting costs
//     O(n) bounded-width exact binary-integer operations for that feature.
//
// Notes:
//   - A covariance produced from rounded standardized observations must be
//     centered in those prepared coordinates, not in original feature units.
//
// AI-Hints:
//   - A published float64 mean alone is not sufficient near large offsets.
//   - Never recompute this structure from heldout or production observations.
type fittedCenter struct {
	anchor float64 // First finite prepared training coordinate.
	offset float64 // High part of exact (mean-anchor), in prepared units.
	low    float64 // Low part of exact (mean-anchor), in prepared units.
}

// fitCovarianceCenters freezes one two-part mean per prepared feature column.
// It computes exact binary-integer offsets to avoid cancellation against a
// large anchor and to preserve the covariance coordinate origin after Fit.
//
// Implementation:
//   - Stage 1: read the first prepared row as independent feature anchors.
//   - Stage 2: accumulate exact binary integers, then subtract n*anchor.
//   - Stage 3: divide by n, split the result into two floats, and publish.
//
// Behavior highlights:
//   - Exact fixed-point accumulation is used only when fitting, never per inference.
//   - The public covariance API still owns covariance construction and errors.
//
// Inputs:
//   - ctx: active context; prepared: finite, unmodified training matrix.
//   - d: validated feature width; at least two rows exist.
//
// Returns:
//   - []fittedCenter: detached private means, length d, or nil on error.
//
// Errors:
//   - ctx.Err(): cancellation during fitting; ErrOverflow: the displacement
//     cannot be represented; Matrix.At error: preserved read failure.
//
// Determinism:
//   - Features ascend, then rows ascend; no map iteration or rounding order drift.
//
// Complexity:
//   - O(n*d) exact binary-integer operations, O(d) persistent center state.
//   - Integer intermediate bit lengths are bounded by float64 exponents and
//     O(log n); the integer-operation cost is not constant-time.
//
// Notes:
//   - Input preparation and matrix.Covariance read the same prepared matrix.
//   - Exact fitted centers avoid an additional public matrix/stats API change.
//
// AI-Hints:
//   - Do not average rounded per-sample centered floats and call that a center.
//   - Keep the covariance mean in prepared units, even for standardized models.
func fitCovarianceCenters(ctx context.Context, prepared matrix.Matrix, d int) ([]fittedCenter, error) {
	rows := prepared.Rows()
	// A fittedCenter owns three float64 slots per feature. The independent
	// check matters because a valid n*d float64 matrix could still exceed
	// the byte-size bound of the center slice on a pathological shape.
	if d > math.MaxInt/(3*8) {
		return nil, ErrOverflow
	}
	centers := make([]fittedCenter, d)

	// Every finite binary64 observation is an exact integer multiplied by
	// 2^-1074. A reusable fixed-point integer avoids allocating one big.Rat
	// per observation while preserving every bit of a large-offset mean.
	count := new(big.Int).SetInt64(int64(rows))
	denominator := new(big.Int).Lsh(count, exactBinaryFractionExponent)
	var sum, term, anchorInteger, displacement, scaledAnchor big.Int
	var offsetRat, highRat, remainder big.Rat
	for feature := 0; feature < d; feature++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		anchor, err := prepared.At(0, feature)
		if err != nil {
			return nil, err
		}
		if !finite(anchor) {
			return nil, ErrNumerical
		}

		// Accumulate the exact sum in a shared binary fixed-point domain.
		// Scanning every row, including row zero, also preserves signed
		// cancellation when a column straddles positive and negative values.
		sum.SetInt64(0)
		for row := 0; row < rows; row++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			value, readErr := prepared.At(row, feature)
			if readErr != nil {
				return nil, readErr
			}
			if !finite(value) {
				return nil, ErrNumerical
			}
			addExactBinary64(&sum, &term, value)
		}

		// offset = exactMean-anchor. The large baseline cancels in integer
		// space BEFORE the result is converted to float64. This preserves
		// half-ULP means such as the midpoint of two adjacent huge numbers.
		anchorInteger.SetInt64(0)
		addExactBinary64(&anchorInteger, &term, anchor)
		scaledAnchor.Mul(&anchorInteger, count)
		displacement.Sub(&sum, &scaledAnchor)
		offsetRat.SetFrac(&displacement, denominator)
		hi, _ := offsetRat.Float64()
		if !finite(hi) {
			return nil, ErrOverflow
		}
		highRat.SetFloat64(hi)
		remainder.Sub(&offsetRat, &highRat)
		lo, _ := remainder.Float64()
		if !finite(lo) {
			return nil, ErrOverflow
		}
		centers[feature] = fittedCenter{anchor: anchor, offset: hi, low: lo}
	}
	return centers, nil
}

const (
	// All finite binary64 values are integer multiples of 2^-1074.
	exactBinaryFractionExponent = 1074
	// The significand is an explicit 52 bits plus the normal hidden bit.
	exactBinaryMantissaBits = 52
	// Binary64 exponent bits start after the 52 fraction bits.
	exactBinaryExponentMask = 0x7ff
	// The exponent field has eleven bits; all fit in uint even on 32-bit Go.
	exactBinaryExponentBits = 11
	// The highest bit marks the sign independently of the exponent field.
	exactBinarySignShift = 63
)

// addExactBinary64 accumulates one finite binary64 value as an exact integer.
// It aligns every input exponent to the common 2^-1074 quantum, so a sum
// involving 2^450 and its next float retains the one-ULP difference exactly.
//
// Implementation:
//   - Stage 1: extract IEEE754 fraction, exponent and sign by math.Float64bits.
//   - Stage 2: construct the exact integer significand shifted to 2^-1074.
//   - Stage 3: add or subtract it to the private accumulator in original order.
//
// Behavior highlights:
//   - An input of either signed zero contributes the exact integer zero.
//   - A normal exponent maps to shift exponentBits-1; subnormals shift zero.
//
// Inputs:
//   - sum: mutable exact accumulator; term: reusable workspace; value: finite.
//
// Returns:
//   - None; sum is updated without changing the input observation.
//
// Errors:
//   - None; caller rejects nonfinite input before decoding its bits.
//
// Determinism:
//   - One fixed shift and exact integer add/subtract per observed cell.
//
// Complexity:
//   - O(1) bounded-width binary-integer operations per float64 on a fixed
//     word size, with bit lengths bounded by 2098+log2(n), not arbitrary input.
//
// Notes:
//   - sum and term must be distinct big.Int values; no rational allocations.
//
// AI-Hints:
//   - Do not compute value-anchor in float64 before updating the accumulator.
//   - Never drop the implicit normal significand bit when forming the integer.
func addExactBinary64(sum, term *big.Int, value float64) {
	bits := math.Float64bits(value)
	fraction := bits & ((uint64(1) << exactBinaryMantissaBits) - 1)
	exponent := (bits >> exactBinaryMantissaBits) & exactBinaryExponentMask
	shift := uint64(0)
	if exponent != 0 {
		fraction |= uint64(1) << exactBinaryMantissaBits
		shift = exponent - 1
	}
	term.SetUint64(fraction)
	// Decompose the bounded exponent into eleven machine-independent bits.
	// This avoids an unchecked uint64-to-uint conversion on 32-bit targets.
	var wordShift uint
	for bit := 0; bit < exactBinaryExponentBits; bit++ {
		if shift&(uint64(1)<<bit) != 0 {
			wordShift |= uint(1) << bit
		}
	}
	term.Lsh(term, wordShift)
	if bits>>exactBinarySignShift != 0 {
		sum.Sub(sum, term)
	} else {
		sum.Add(sum, term)
	}
}

// centered subtracts the frozen composite prepared-space covariance center.
// Compensated subtraction preserves small observations even when a large
// anchor and displacement cancel; exact rescue resolves sensitive sums.
//
// Implementation:
//   - Stage 1: order the observation and negated center terms.
//   - Stage 2: compensate the first two subtractions before applying the low part.
//   - Stage 3: use exact binary arithmetic when compensation is not sufficient.
//
// Behavior highlights:
//   - For adjacent input floats separated by g, the symmetric two-row scores
//     become -g/2 and +g/2 instead of 0 and g.
//
// Inputs:
//   - c: frozen prepared-coordinate origin; value: finite prepared observation.
//
// Returns:
//   - float64: centered prepared coordinate, or zero on failure.
//
// Errors:
//   - ErrOverflow: the final centered value cannot fit in float64.
//
// Determinism:
//   - Fixed anchor, high, low evaluation order; fallback is exact binary rational.
//
// Complexity:
//   - O(1) float64 fast path; bounded rational arithmetic in rare rescue.
//
// Notes:
//   - The fitted Model never changes this center after fitting.
//
// AI-Hints:
//   - Do not combine anchor and offset into one rounded float64 first.
//   - Do not classify a finite final difference as overflow from a partial sum.
//   - Do not discard the rounding error of value-anchor when anchor and offset cancel.
func (c fittedCenter) centered(value float64) (float64, error) {
	return centerAffine(value, -c.anchor, -c.offset, -c.low)
}

// restore adds the frozen composite covariance origin to one prepared score.
// It uses a compensated three-part addition so a small residual is not lost
// merely because the original mean was very large or rounded between ulps.
//
// Implementation:
//   - Stage 1: add score and high mean displacement, retaining sum roundoff.
//   - Stage 2: add the anchor and combine low-order corrections.
//   - Stage 3: use exact binary-rational addition on sensitive cancellation or overflow.
//
// Behavior highlights:
//   - A reconstructed training input regains original prepared coordinates;
//     final rounding is to a representable float64 input value.
//
// Inputs:
//   - c: immutable fitted origin; value: prepared-space reconstructed score.
//
// Returns:
//   - float64: reconstructed prepared observation, or zero on failure.
//
// Errors:
//   - ErrOverflow: a nonfinite projection or unrepresentable final prepared observation.
//
// Determinism:
//   - Exactly two error-free additions and a fixed low-component correction.
//
// Complexity:
//   - O(1) fast path; bounded exact-rational rescue when compensation is unsafe.
//
// Notes:
//   - This does not recover discarded principal components.
//
// AI-Hints:
//   - Undo the covariance center BEFORE undoing population standardization.
//   - Do not compute a rounded single mean and add that directly to scores.
func (c fittedCenter) restore(value float64) (float64, error) {
	return centerAffine(value, c.offset, c.anchor, c.low)
}

// centerAffine evaluates the four signed terms of a frozen affine center.
// It preserves addition roundoff before accepting a well-conditioned result;
// near cancellation or intermediate overflow is resolved over exact rationals.
//
// Implementation:
//   - Stage 1: use TwoSum twice to retain the roundoff of the first three terms.
//   - Stage 2: apply the final term and recovered roundoff in fixed order.
//   - Stage 3: accept only a finite safe result, otherwise use exact rescue.
//
// Behavior highlights:
//   - The allocation-free path does not silently drop a small heldout coordinate.
//   - If both additions are exact, only one final rounding remains.
//
// Inputs:
//   - a,b,c,d: signed affine terms in prepared feature units.
//
// Returns:
//   - A finite prepared coordinate, or zero on failure.
//
// Errors:
//   - ErrOverflow: a nonfinite operand or unrepresentable final sum.
//
// Determinism:
//   - A fixed compensated order, followed by deterministic exact arithmetic.
//
// Complexity:
//   - O(1) float64 work; bounded rational work only on the rescue path.
//
// Notes:
//   - The dominance guard is a conditioning check, not a rank tolerance or
//     a guarantee of correctly rounded output on the compensated fast path.
//
// AI-Hints:
//   - Reuse this law for forward and inverse centering; do not cache a rounded
//     single mean or discard TwoSum corrections before subtracting the offset.
func centerAffine(a, b, c, d float64) (float64, error) {
	// Three correction terms can oppose the rounded main sum. Requiring the
	// latter to exceed four times their largest magnitude leaves a positive
	// margin even if all three corrections have the same opposing sign.
	const correctionDominance = 4
	first, firstErr := twoSum(a, b)
	if finite(first) {
		second, secondErr := twoSum(first, c)
		if finite(second) {
			// Exact partial sums need only the final correctly rounded addition.
			if firstErr == 0 && secondErr == 0 {
				if result := second + d; finite(result) {
					return result, nil
				}
			} else {
				// Check individual magnitudes, not their potentially canceled sum.
				largest := max(math.Abs(firstErr), math.Abs(secondErr), math.Abs(d))
				if math.Abs(second)/correctionDominance > largest {
					if result := second + ((firstErr + secondErr) + d); finite(result) {
						return result, nil
					}
				}
			}
		}
	}
	// The exact path also rejects any nonfinite projected input.
	return centerRational(a, b, c, d)
}

// twoSum returns a rounded sum together with the rounding error of two floats.
// The result is exact as sum+error when the rounded sum is finite and the
// elementary operations do not underflow outside their IEEE754 guarantees.
//
// Implementation:
//   - Stage 1: add the operands and recover the second contribution.
//   - Stage 2: recover the first and second rounding residuals.
//   - Stage 3: return both terms without altering inputs.
//
// Behavior highlights:
//   - Compensates cancellation of large positive and negative terms.
//
// Inputs:
//   - a,b: finite IEEE754 binary64 operands.
//
// Returns:
//   - float64,float64: rounded sum and signed low-order correction.
//
// Errors:
//   - None; callers check a non-finite sum and use exact fallback.
//
// Determinism:
//   - Pure fixed operation sequence, independent of scheduling.
//
// Complexity:
//   - O(1) time and space.
//
// Notes:
//   - This is the Knuth TwoSum identity, not a general compensated dot product.
//
// AI-Hints:
//   - Do not evaluate only a+b if adding a later fitted center correction.
func twoSum(a, b float64) (float64, float64) {
	sum := a + b
	recovered := sum - a
	return sum, (a - (sum - recovered)) + (b - recovered)
}

// centerRational resolves cancellation-sensitive or overflowing affine sums.
// It tests the final binary-rational value rather than any partial float64 sum.
//
// Implementation:
//   - Stage 1: convert each finite original float64 operand to exact rationals.
//   - Stage 2: sum the four terms in the caller's deterministic order.
//   - Stage 3: round the final exact result to float64 and reject overflow.
//
// Behavior highlights:
//   - A huge pair of canceling terms does not imply an overflowing result.
//
// Inputs:
//   - a,b,c,d: finite binary64 terms in prepared feature units.
//
// Returns:
//   - float64: rounded final sum, or zero on failure.
//
// Errors:
//   - ErrOverflow: a nonfinite operand or an unrepresentable exact final result.
//
// Determinism:
//   - Four binary-to-rational conversions in fixed argument order.
//
// Complexity:
//   - O(1) bounded-size rational operations in the float64 exponent domain.
//
// Notes:
//   - No rational workspace is retained in the fitted model.
//
// AI-Hints:
//   - Only the final finite value determines overflow, never partial sums.
func centerRational(a, b, c, d float64) (float64, error) {
	var sum, term big.Rat
	for _, value := range [4]float64{a, b, c, d} {
		// SetFloat64 returns nil for NaN/Inf and does not replace the receiver.
		// Reject that input instead of accidentally reusing a previous term.
		if term.SetFloat64(value) == nil {
			return 0, ErrOverflow
		}
		sum.Add(&sum, &term)
	}
	result, _ := sum.Float64()
	if !finite(result) {
		return 0, ErrOverflow
	}
	return result, nil
}

// preprocessFeature maps one original observation to the frozen PCA coordinates.
// Population standardization uses training means/scales, then covariance
// centering removes the second prepared-space mean, with no heldout fitting.
//
// Implementation:
//   - Stage 1: scale the original observation with fitted population parameters.
//   - Stage 2: on intermediate subtraction overflow, use exact rational division.
//   - Stage 3: subtract the independently frozen prepared covariance center.
//
// Behavior highlights:
//   - Non-standardized fits have mean zero and scale one in the first stage.
//   - All training and inference paths share this same internal kernel.
//
// Inputs:
//   - m: valid immutable Model; value: finite original feature observation.
//   - feature: validated original feature index in [0,m.d).
//
// Returns:
//   - float64: finite centered (and optionally scaled) coordinate, or zero.
//
// Errors:
//   - ErrOverflow: an essential fitted affine result is unrepresentable.
//
// Determinism:
//   - Fixed subtraction/division and frozen center order per original feature.
//
// Complexity:
//   - O(1) float64 fast path; bounded big.Rat rescue on early overflow.
//
// Notes:
//   - Never mutates Model; per-call temporaries are local to the operation.
//
// AI-Hints:
//   - Do not call stats.FitStandardizer from inference or trainingResidual.
//   - A covariance mean is not interchangeable with population scaling means.
func (m *Model) preprocessFeature(value float64, feature int) (float64, error) {
	mean, scale := m.means[feature], m.scales[feature]
	shifted := value - mean
	if finite(shifted) {
		prepared := shifted / scale
		if !finite(prepared) {
			return 0, ErrOverflow
		}
		return m.centers[feature].centered(prepared)
	}
	// Extreme opposite-sign inputs can overflow the unscaled difference;
	// the exact final quotient may nevertheless be representable.
	var numerator, meanRat, denominator big.Rat
	numerator.SetFloat64(value)
	meanRat.SetFloat64(mean)
	denominator.SetFloat64(scale)
	numerator.Sub(&numerator, &meanRat)
	numerator.Quo(&numerator, &denominator)
	prepared, _ := numerator.Float64()
	if !finite(prepared) {
		return 0, ErrOverflow
	}
	return m.centers[feature].centered(prepared)
}
