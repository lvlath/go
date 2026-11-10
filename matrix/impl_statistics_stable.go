// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix provides deterministic, finite-data statistical reductions.
// This file contains the shared scale-aware kernels used by column centering,
// sample covariance, Pearson correlation, and row-wise L2 normalization.
// The kernels never allocate a centered copy for covariance or correlation.
package matrix

import (
	"fmt"
	"math"
)

const (
	// statisticsPearsonRoundoff bounds harmless overshoots of the Cauchy-Schwarz
	// interval caused by a few float64 rounding operations. Larger violations
	// indicate a numerical failure rather than a value suitable for clipping.
	statisticsPearsonRoundoff = 32 * (1.0 / (1 << 52))

	// statisticsTargetRelativeError governs the error budget used to decide
	// whether signed, nearly canceling moments deserve exact evaluation.
	// It is a numeric-stability trigger, not a promise of correctly-rounded
	// results for every input in the entire float64 exponent range.
	statisticsTargetRelativeError = 1e-12

	// statisticsCancellationBound derives from the local float64 rounding
	// unit and the target relative error. A moment smaller than this share
	// of its absolute terms may lose meaningful digits through subtraction,
	// even when Neumaier summation recovers the additions themselves.
	statisticsCancellationBound = 8 * (1.0 / (1 << 52)) / statisticsTargetRelativeError
)

// statisticsReader fixes the shape and access policy of an immutable input Matrix.
// It makes the concrete *Dense fast path explicit while ensuring that generic
// Matrix.At failures and non-finite input values cannot disappear in hot loops.
// Implementation:
//   - Stage 1: retain the original Matrix interface and validated dimensions.
//   - Stage 2: select direct row-major data access only for a concrete *Dense.
//   - Stage 3: reject invalid floating-point values at the point of every read.
//
// Behavior highlights:
//   - Reader instances are private, value-copyable metadata, not owned matrices.
//   - No mutable or shared numerical workspace is stored in this type.
//
// Inputs: An existing Matrix whose mutation must not overlap any reduction call.
// Returns: O(1) per-element values or the original wrapped access failure.
// Errors: ErrNilMatrix, ErrInvalidDimensions, ErrNaNInf or Matrix.At's cause.
// Determinism: Indexing follows the consumer's explicit row/column order.
// Complexity: O(1) metadata; each value read is O(1) for standard Dense/At.
// Notes: The optional empty-domain aliasing rule is owned by public kernels.
// AI-Hints: Never use reflect for the fast path; a direct *Dense type assertion
// is sufficient and avoids virtual dispatch inside quadratic reductions.
type statisticsReader struct {
	input Matrix // Original caller-owned Matrix; never mutated by reductions.
	dense *Dense // Direct row-major path when the concrete input is *Dense.
	rows  int    // Validated observation count.
	cols  int    // Validated feature count.
}

// newStatisticsReader selects a safe input-access path without copying data.
// It establishes dimension arithmetic and type-based Dense access before any
// statistical workspace is allocated or observations are inspected.
// Implementation:
//   - Stage 1: call the canonical nil validator, including recognized typed nils.
//   - Stage 2: reject negative dimensions or a signed-int rows*cols overflow.
//   - Stage 3: capture *Dense direct access, or retain Matrix.At fallback.
//
// Behavior highlights: Shape validation always precedes value validation.
// Inputs: input is read-only during the call; zero-element shapes are legal.
// Returns: Private reader holding input by reference for the duration of one call.
// Errors: ErrNilMatrix and ErrInvalidDimensions; no partial reader escapes.
// Determinism: The function never scans elements or reorders observations.
// Complexity: Time O(1), Space O(1).
// Notes: Concurrent mutation by the caller is unsupported, including via views.
// AI-Hints: Checking rows*cols protects both the flat offset and later output sizes.
func newStatisticsReader(input Matrix) (statisticsReader, error) {
	if err := ValidateNotNil(input); err != nil {
		return statisticsReader{}, err
	}

	rows, cols := input.Rows(), input.Cols()
	if _, err := matrixCheckedElementCount(rows, cols); err != nil {
		return statisticsReader{}, err
	}

	// This type assertion is deliberately not reflection. It removes interface
	// dispatch from the O(rows*cols) scans of the common owning representation.
	dense, _ := input.(*Dense)
	if dense == nil {
		return statisticsReader{input: input, rows: rows, cols: cols}, nil
	}
	return statisticsReader{input: input, dense: dense, rows: rows, cols: cols}, nil
}

// value returns one finite observation through the already selected access path.
// Dense reads bypass interface dispatch; other Matrix implementations preserve
// every original At error under an index-bearing wrapper.
// Implementation:
//   - Stage 1: read directly from the bounded Dense buffer or through At(i,j).
//   - Stage 2: reject NaN and both infinities before numerical arithmetic.
//   - Stage 3: return the unmodified binary64 observation.
//
// Behavior highlights: No sanitation, replacement, or implicit zeroing occurs.
// Inputs: i,j are valid indices within the validated reader shape.
// Returns: Original finite float64 coordinate; no retained aliases.
// Errors: ErrNaNInf or wrapped backend error including its sentinel identity.
// Determinism: No data-dependent reordering; caller fixes the visitation order.
// Complexity: Time O(1), Space O(1).
// Notes: Fallback Matrix.At must itself obey the public Matrix bounds contract.
// AI-Hints: Never swallow custom At errors; tests rely on errors.Is for the cause.
func (r statisticsReader) value(i, j int) (float64, error) {
	var value float64
	if r.dense != nil {
		value = r.dense.data[i*r.cols+j]
	} else {
		var err error
		value, err = r.input.At(i, j)
		if err != nil {
			return 0, fmt.Errorf("At(%d,%d): %w", i, j, err)
		}
	}
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, ErrNaNInf
	}
	return value, nil
}

// statisticsSum stores one compensated, strictly ordered scalar reduction.
// Neumaier correction tracks low-order information lost by adding terms of
// different magnitudes, without a per-observation allocation.
// Implementation:
//   - Stage 1: keep the rounded running total in sum.
//   - Stage 2: record the lost low-order contribution in correction.
//   - Stage 3: publish sum+correction only after a completed reduction.
//
// Behavior highlights: No sorting or arbitrary reassociation of input terms.
// Inputs: Each term is finite and already normalized to a bounded scale.
// Returns: An improved, though not universally correctly rounded, float64 sum.
// Errors: Not returned; the caller controls its bounded numeric domain.
// Determinism: Exact call order determines the update sequence.
// Complexity: Time O(1) per term, Space O(1).
// Notes: Cancellation-sensitive moments may still require exact evaluation.
// AI-Hints: Compensation cannot recover information that was rounded away
// before add was called; preserve scaled input operands until accumulation.
type statisticsSum struct {
	sum        float64 // Rounded high component of the ordered reduction.
	correction float64 // Low-order Neumaier compensation for the same reduction.
}

// add incorporates one bounded term into a Neumaier compensated sum.
// The larger-operand branch reconstructs the rounding loss from a floating
// addition; it works for both increasing and decreasing running magnitudes.
// Implementation:
//   - Stage 1: compute a rounded sum in the caller-determined order.
//   - Stage 2: add the algebraic roundoff correction using the larger operand.
//   - Stage 3: replace the current sum without publishing a partial result.
//
// Inputs: term is a finite bounded scalar from the normalized numerical kernel.
// Returns: Mutable private accumulator; no errors or allocations.
// Determinism: Each call is executed in its exact scan order.
// Complexity: Time O(1), Space O(1).
// Notes: Not a replacement for arbitrary-precision rescue of cross-moments.
// AI-Hints: Do not move the compensation after scaling back to original units.
func (a *statisticsSum) add(term float64) {
	next := a.sum + term
	if math.Abs(a.sum) >= math.Abs(term) {
		a.correction += (a.sum - next) + term
	} else {
		a.correction += (term - next) + a.sum
	}
	a.sum = next
}

// value returns the compensated total after a completed ordered reduction.
// It is used for normalized moments that must retain low-order summed contributions.
// Implementation:
//   - Stage 1: Read both high and correction components.
//   - Stage 2: Add once at the publication point.
//   - Stage 3: Return the rounded binary64 total.
//
// Behavior highlights:
//   - No mutation or extra allocation occurs.
//
// Inputs:
//   - a is the private completed Neumaier accumulator.
//
// Returns:
//   - One finite float64 total.
//
// Errors:
//   - None when bounded scaled inputs obey their contract.
//
// Determinism:
//   - Caller-established reduction order is retained.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Compensation is not an exact rational sum.
//
// AI-Hints:
//   - Do not read before all terms have been added or drop correction.
func (a statisticsSum) value() float64 {
	return a.sum + a.correction
}

// statisticsColumn stores the normalized and original-coordinate feature state.
// The anchor and scale are fixed in the first scan, enabling consistent
// centered products even when a published float64 mean loses a fraction of an ulp.
// Implementation:
//   - Stage 1: fix the first observation as anchor and maximum magnitude as scale.
//   - Stage 2: accumulate bounded offsets into the compensated acc.
//   - Stage 3: publish mean and, only on dangerous cancellation, meanLow.
//
// Behavior highlights: Each column's state is private to one reduction call.
// Inputs: Finite column observations in ascending row order.
// Returns: Private feature metadata, not a caller-visible fitted model.
// Errors: Reconstruction or exact rescue can fail with ErrOverflow or At errors.
// Determinism: The anchor is always row 0; no random pivot is selected.
// Complexity: O(1) per-feature state and O(n) updates per feature.
// Notes: meanLow is the residual of a high-precision mean after float64 rounding.
// AI-Hints: Do not reconstruct every centered element by x-mean when mean
// may have rounded away small but statistically meaningful deviations.
type statisticsColumn struct {
	anchor     float64       // First observation, retained as a stable difference origin.
	scale      float64       // Largest absolute original observation in the feature.
	meanOffset float64       // Mean of (x-anchor)/scale in normalized coordinates.
	mean       float64       // Caller-visible rounded mean in original feature units.
	meanLow    float64       // Rounded exact-mean remainder after the high component.
	precise    bool          // Exact rescue ran because cancellation made a float mean unsafe.
	acc        statisticsSum // Compensated sum of anchor-relative normalized observations.
}

// statisticsRelativeDifference computes a scaled anchor-relative coordinate.
// It protects opposite-sign, extreme-magnitude subtraction from overflowing before scaling.
// Implementation:
//   - Stage 1: Reject the trivial scale==0 case as a zero offset.
//   - Stage 2: Use direct subtraction for like signs or divide before subtraction otherwise.
//   - Stage 3: Return the bounded relative difference.
//
// Behavior highlights:
//   - Input values remain unmodified.
//
// Inputs:
//   - observation and anchor are finite data; scale is their positive absolute bound.
//
// Returns:
//   - Finite normalized coordinate, with zero for scale==0.
//
// Errors:
//   - None on validated finite observations.
//
// Determinism:
//   - Pure fixed arithmetic branch determined only by operand signs.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The bounded coordinate can still suffer rounding for widely separated exponents.
//
// AI-Hints:
//   - Do not replace opposite-sign branch with (observation-anchor)/scale.
func statisticsRelativeDifference(observation, anchor, scale float64) float64 {
	if scale == 0 {
		return 0
	}
	if (observation >= 0 && anchor >= 0) || (observation <= 0 && anchor <= 0) {
		return (observation - anchor) / scale
	}
	return observation/scale - anchor/scale
}

// statisticsProduct3 restores three finite scale factors without early overflow.
// It allows a representable final covariance after intermediate scale products exceed float64.
// Implementation:
//   - Stage 1: Return exact zero when any operand is zero.
//   - Stage 2: Decompose each factor with Frexp and multiply bounded mantissas.
//   - Stage 3: Restore the combined exponent once and check the final result.
//
// Behavior highlights:
//   - No unbounded intermediate multiply of original magnitudes occurs.
//
// Inputs:
//   - a,b,c are finite caller-validated scaling factors.
//
// Returns:
//   - One representable binary64 product or no value on error.
//
// Errors:
//   - ErrOverflow only if the restored final result is not finite.
//
// Determinism:
//   - The three input factors are always visited in argument order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Very small final products obey binary64 subnormal rounding.
//
// AI-Hints:
//   - Multiplying scaleA*scaleB before reducing the moment reintroduces Gram overflow.
func statisticsProduct3(a, b, c float64) (float64, error) {
	if a == 0 || b == 0 || c == 0 {
		return 0, nil
	}
	ma, ea := math.Frexp(a)
	mb, eb := math.Frexp(b)
	mc, ec := math.Frexp(c)

	// Normalize the mantissa product a second time so Ldexp receives a
	// mantissa of manageable magnitude even when three factors are involved.
	mantissa, adjustment := math.Frexp((ma * mb) * mc)
	value := math.Ldexp(mantissa, ea+eb+ec+adjustment)
	if math.IsInf(value, 0) || math.IsNaN(value) {
		return 0, ErrOverflow
	}

	return value, nil
}

// statisticsPrepareColumns finds stable column means without summing raw data.
//
// Implementation:
//   - Stage 1: scan rows in row-major order, select a first-row anchor and a
//     maximum absolute scale for every column, rejecting non-finite inputs.
//   - Stage 2: sum bounded, anchor-relative coordinates with compensation.
//   - Stage 3: reconstruct each mean, using a normalized fallback when the
//     offset product overflows despite a representable final mean.
//
// The original input is never changed. Returned states are private, per-call
// storage and cannot escape into a caller-owned result.
// Complexity: O(n*d) time, O(d) extra space.
func statisticsPrepareColumns(reader statisticsReader) ([]statisticsColumn, error) {
	columns := make([]statisticsColumn, reader.cols)
	if reader.rows == 0 || reader.cols == 0 {
		return columns, nil
	}

	// The first pass also validates every observation. The first observation
	// in a column becomes its anchor; using the same anchor in all later
	// passes keeps the numerical reduction deterministic.
	for i := 0; i < reader.rows; i++ {
		for j := 0; j < reader.cols; j++ {
			observation, err := reader.value(i, j)
			if err != nil {
				return nil, err
			}
			if i == 0 {
				columns[j].anchor = observation
			}
			if magnitude := math.Abs(observation); magnitude > columns[j].scale {
				columns[j].scale = magnitude
			}
		}
	}

	// Scaling makes every term lie in [-2,2], avoiding sum overflow even
	// when the unscaled observations are close to MaxFloat64.
	for i := 0; i < reader.rows; i++ {
		for j := 0; j < reader.cols; j++ {
			observation, err := reader.value(i, j)
			if err != nil {
				return nil, err
			}
			column := &columns[j]
			column.acc.add(statisticsRelativeDifference(observation, column.anchor, column.scale))
		}
	}

	// Reconstruct only after dividing the normalized sum by the row count.
	// A separately overflowing offset is not proof that the mean overflows:
	// the scaled fallback includes the anchor before restoring magnitude.
	count := float64(reader.rows)
	for j := range columns {
		column := &columns[j]
		column.meanOffset = column.acc.value() / count
		candidate := column.anchor + column.meanOffset*column.scale
		if math.IsInf(candidate, 0) {
			candidate = column.scale * (column.anchor/column.scale + column.meanOffset)
		}
		if math.IsNaN(candidate) || math.IsInf(candidate, 0) {
			return nil, ErrOverflow
		}
		column.mean = candidate
		// When the anchor and mean offset nearly cancel, the normalized
		// representation may have lost a small original-coordinate sum.
		// Re-read that column exactly before publishing misleading zeroes.
		if column.scale > 0 && math.Abs(candidate) <= statisticsCancellationBound*column.scale {
			precise, low, exactErr := statisticsExactColumnMean(reader, j)
			if exactErr != nil {
				return nil, exactErr
			}
			column.mean = precise
			column.meanLow = low
			column.precise = true
		}
	}

	return columns, nil
}

// statisticsCenterColumns returns finite centered values and detached means.
//
// Implementation:
//   - Stage 1: validate the shape and prepare scale-aware means.
//   - Stage 2: calculate each centered cell in the reference coordinate system
//     before scaling back, preserving tiny deviations near a huge offset.
//   - Stage 3: publish a fresh Dense and a newly allocated means slice.
//
// Zero-element shapes retain the existing no-op matrix alias rule; their
// metadata is still a fresh slice. An unrepresentable centered value fails
// explicitly instead of storing an infinity.
// Complexity: O(n*d) time, O(n*d+d) space including the output.
func statisticsCenterColumns(input Matrix) (Matrix, []float64, error) {
	reader, err := newStatisticsReader(input)
	if err != nil {
		return nil, nil, err
	}
	if reader.rows == 0 || reader.cols == 0 {
		return input, make([]float64, reader.cols), nil
	}

	columns, err := statisticsPrepareColumns(reader)
	if err != nil {
		return nil, nil, err
	}
	centered, err := NewDense(reader.rows, reader.cols)
	if err != nil {
		return nil, nil, err
	}
	means := make([]float64, reader.cols)
	for j := range means {
		means[j] = columns[j].mean
	}

	// Preserve the normalized difference until after subtracting the mean
	// offset; subtracting two separately rounded huge float64 means would
	// incorrectly erase meaningful ulp-level spread.
	for i := 0; i < reader.rows; i++ {
		for j := 0; j < reader.cols; j++ {
			observation, readErr := reader.value(i, j)
			if readErr != nil {
				return nil, nil, readErr
			}
			column := columns[j]
			var value float64
			if column.precise {
				// A canceled column mean was rebuilt from exact binary
				// rationals; subtract its high and low parts in original
				// coordinates so tiny surviving observations are not erased.
				value = (observation - column.mean) - column.meanLow
				if math.IsNaN(value) || math.IsInf(value, 0) {
					return nil, nil, ErrOverflow
				}
			} else {
				// Otherwise anchor-relative centering preserves sub-ulp
				// deviations that a rounded public mean cannot express.
				difference := statisticsRelativeDifference(observation, column.anchor, column.scale) - column.meanOffset
				var scaleErr error
				value, scaleErr = statisticsProduct3(difference, column.scale, 1)
				if scaleErr != nil {
					return nil, nil, scaleErr
				}
			}
			centered.data[i*reader.cols+j] = value
		}
	}

	return centered, means, nil
}

// statisticsMoment retains a signed compensated cross-moment and its magnitude.
// The absolute-term total is a cancellation diagnostic, not a second covariance;
// it triggers exact binary-rational rescue only for unsafe signed reductions.
// Implementation:
//   - Stage 1: accumulate products once into acc, in increasing row order.
//   - Stage 2: accumulate absolute product magnitudes into absolute.
//   - Stage 3: compare the signed-to-absolute ratio against the named bound.
//
// Behavior highlights: Only one state exists for each unordered feature pair.
// Inputs: Products of normalized centered feature observations.
// Returns: A private compact upper-triangle moment, never an exposed Matrix.
// Errors: None at this stage; final restoration classifies overflow.
// Determinism: Fixed upper-triangle index and row-major reduction order.
// Complexity: O(1) storage and update cost per feature pair.
// AI-Hints: A small signed sum is not proof of zero covariance when both
// positive and negative terms were enormous before scale normalization.
type statisticsMoment struct {
	acc      statisticsSum // Compensated signed sum of normalized cross-products.
	absolute float64       // Sum of absolute term magnitudes for cancellation detection.
}

// needsExact identifies dangerously canceled normalized cross-moments.
// The absolute sum bounds how much signed information could have been lost.
// Implementation:
//   - Stage 1: Read the compensated signed sum.
//   - Stage 2: Compare its magnitude to the named roundoff-risk bound times the absolute sum.
//   - Stage 3: Request exact rescue when the signed result is insufficiently resolved.
//
// Behavior highlights:
//   - A zero moment may still require exact evaluation when terms cancel.
//
// Inputs:
//   - m contains private signed and absolute cross-product accumulations.
//
// Returns:
//   - Boolean indicating whether an exact rational recomputation is required.
//
// Errors:
//   - None; the returned decision does not publish a covariance.
//
// Determinism:
//   - Pure comparison with fixed coefficients and known scan history.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This trigger is an engineering error budget, not a universal rounding certificate.
//
// AI-Hints:
//   - Do not assume an accumulator result of zero proves zero covariance.
func (m statisticsMoment) needsExact() bool {
	return math.Abs(m.acc.value()) <= statisticsCancellationBound*m.absolute
}

// statisticsTriangularIndex maps an unordered feature pair to compact storage.
// It ensures a single accumulator exists for each covariance pair.
// Implementation:
//   - Stage 1: Require the caller to pass 0<=j<=k<cols.
//   - Stage 2: Subtract the triangular prefix and add the within-row offset.
//   - Stage 3: Return the unique upper-triangle index.
//
// Behavior highlights:
//   - No map lookup, dynamic sorting, or allocation is performed.
//
// Inputs:
//   - j,k are ascending feature indices; cols is the validated feature count.
//
// Returns:
//   - An int offset in the compact triangular moment slice.
//
// Errors:
//   - None; validated index/dimension invariants are caller responsibilities.
//
// Determinism:
//   - Fixed algebraic mapping identical across Dense and fallback paths.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Caller checks the quadratic dimension bound before computing this index.
//
// AI-Hints:
//   - Never accumulate symmetric pairs separately; that loses bitwise symmetry.
func statisticsTriangularIndex(j, k, cols int) int {
	return j*cols - j*(j-1)/2 + (k - j)
}

// statisticsCenteredMoments accumulates the one authoritative cross-moment
// for each unordered pair of columns. Every pair is evaluated exactly once.
//
// Implementation:
//   - Stage 1: check dimensions and the sample-size domain.
//   - Stage 2: prepare original-coordinate means and normalized centers.
//   - Stage 3: scan rows in fixed order, accumulating only the upper triangle
//     with Neumaier compensation and a single reusable O(d) row buffer.
//
// Covariance and correlation share these exact moments but finalize them
// independently. Neither operation needs a materialized centered matrix,
// transpose, Gram matrix, or an unrepresentable intermediate covariance.
// Complexity: O(n*d*d) time, O(d*d+d) auxiliary space.
func statisticsCenteredMoments(input Matrix) ([]statisticsColumn, []statisticsMoment, int, error) {
	reader, err := newStatisticsReader(input)
	if err != nil {
		return nil, nil, 0, err
	}
	// Sample statistics are unavailable with fewer than two observations,
	// including the zero-feature case; validate this before the c==0 fast path.
	if reader.rows < 2 {
		return nil, nil, 0, ErrDimensionMismatch
	}

	// Fail before allocating a quadratic covariance workspace or result.
	maxInt := int(^uint(0) >> 1)
	if reader.cols > 0 && reader.cols > maxInt/reader.cols {
		return nil, nil, 0, ErrInvalidDimensions
	}
	columns, err := statisticsPrepareColumns(reader)
	if err != nil {
		return nil, nil, 0, err
	}
	if reader.cols == 0 {
		return columns, []statisticsMoment{}, reader.rows, nil
	}

	// n*(d*d) operations are unavoidable for dense pairwise moments, but
	// only d*(d+1)/2 accumulators are retained at any instant.
	moments := make([]statisticsMoment, reader.cols*(reader.cols+1)/2)
	row := make([]float64, reader.cols)
	for i := 0; i < reader.rows; i++ {
		for j := 0; j < reader.cols; j++ {
			observation, readErr := reader.value(i, j)
			if readErr != nil {
				return nil, nil, 0, readErr
			}
			column := columns[j]
			row[j] = statisticsRelativeDifference(observation, column.anchor, column.scale) - column.meanOffset
		}
		for j := 0; j < reader.cols; j++ {
			for k := j; k < reader.cols; k++ {
				index := statisticsTriangularIndex(j, k, reader.cols)
				product := row[j] * row[k]
				moments[index].acc.add(product)
				moments[index].absolute += math.Abs(product)
			}
		}
	}

	return columns, moments, reader.rows, nil
}

// statisticsCovariance returns a sample covariance computed from compact
// scaled cross-moments, not from an overflowing intermediate Gram matrix.
//
// Implementation:
//   - Stage 1: derive column means and normalized upper-triangle moments.
//   - Stage 2: divide each normalized moment by n-1 before multiplying scales.
//   - Stage 3: publish the value once in each symmetric position, bit-identical.
//
// An unrepresentable final covariance returns ErrOverflow and a nil result.
// Complexity: O(n*d*d+d*d) time, O(d*d+d) space for workspace and output.
func statisticsCovariance(input Matrix) (Matrix, []float64, error) {
	columns, moments, rows, err := statisticsCenteredMoments(input)
	if err != nil {
		return nil, nil, err
	}
	cols := len(columns)
	covarianceD, err := NewDense(cols, cols)
	if err != nil {
		return nil, nil, err
	}
	means := make([]float64, cols)
	for j := range columns {
		means[j] = columns[j].mean
	}

	// Division precedes exponent restoration. This matters for data such as
	// [-1e154,0,1e154], whose sample variance is representable although
	// the sum of squared original observations is not.
	denominator := float64(rows - 1)
	reader, err := newStatisticsReader(input)
	if err != nil {
		return nil, nil, err
	}
	for j := 0; j < cols; j++ {
		for k := j; k < cols; k++ {
			moment := moments[statisticsTriangularIndex(j, k, cols)]
			var value float64
			// Constant features have an identically zero centered product.
			// Do not invoke the expensive exact-rescue path for these pairs:
			// a completely constant d-column dataset otherwise triggers
			// O(n*d*d) unnecessary big-integer reconstructions.
			constantFeature := moments[statisticsTriangularIndex(j, j, cols)].acc.value() == 0 ||
				moments[statisticsTriangularIndex(k, k, cols)].acc.value() == 0
			if moment.needsExact() && j != k && !constantFeature {
				// If nearly canceling contributions lose significance,
				// recover the cross-covariance from exact binary rationals.
				exact, exactErr := statisticsExactCovariance(reader, j, k)
				if exactErr != nil {
					return nil, nil, exactErr
				}
				value, _ = exact.Float64()
				if math.IsNaN(value) || math.IsInf(value, 0) {
					return nil, nil, ErrOverflow
				}
			} else {
				// Division occurs before restoring the original scales.
				var scaleErr error
				value, scaleErr = statisticsProduct3(columns[j].scale, columns[k].scale, moment.acc.value()/denominator)
				if scaleErr != nil {
					return nil, nil, scaleErr
				}
			}
			covarianceD.data[j*cols+k] = value
			covarianceD.data[k*cols+j] = value
		}
	}

	return covarianceD, means, nil
}

// statisticsCorrelation computes Pearson coefficients directly from the
// scaled centered moments and independently reconstructs sample deviations.
//
// Implementation:
//   - Stage 1: compute normalized centered moments without Covariance.
//   - Stage 2: convert normalized diagonal moments to L2 lengths and sample
//     standard deviations without squaring large original coordinates.
//   - Stage 3: normalize off-diagonal moments, mirror each pair, and set the
//     diagonal to exactly one only for columns with positive variance.
//
// A constant column retains an all-zero row and column, including diagonal.
// The original input and all returned slices are detached from workspaces.
// Complexity: O(n*d*d+d*d) time, O(d*d+d) workspace and output.
func statisticsCorrelation(input Matrix) (Matrix, []float64, []float64, error) {
	columns, moments, rows, err := statisticsCenteredMoments(input)
	if err != nil {
		return nil, nil, nil, err
	}
	cols := len(columns)
	correlationD, err := NewDense(cols, cols)
	if err != nil {
		return nil, nil, nil, err
	}
	means := make([]float64, cols)
	stds := make([]float64, cols)
	lengths := make([]float64, cols)
	denominator := float64(rows - 1)

	// Diagonal moments are nonnegative by construction: each is a sum of
	// squared normalized centers. Zero identifies constant observed data.
	for j := range columns {
		means[j] = columns[j].mean
		diagonal := moments[statisticsTriangularIndex(j, j, cols)].acc.value()
		if diagonal < 0 || math.IsNaN(diagonal) || math.IsInf(diagonal, 0) {
			return nil, nil, nil, ErrOverflow
		}
		lengths[j] = math.Sqrt(diagonal)
		stds[j], err = statisticsProduct3(columns[j].scale, lengths[j], 1/math.Sqrt(denominator))
		if err != nil {
			return nil, nil, nil, err
		}
	}

	// Divide by lengths independently. A huge unscaled covariance or
	// underflowed variance is not needed to obtain a valid correlation.
	reader, err := newStatisticsReader(input)
	if err != nil {
		return nil, nil, nil, err
	}
	for j := 0; j < cols; j++ {
		for k := j; k < cols; k++ {
			if lengths[j] == 0 || lengths[k] == 0 {
				continue // The destination is already zero-initialized.
			}
			moment := moments[statisticsTriangularIndex(j, k, cols)]
			var value float64
			if moment.needsExact() && j != k {
				// An exact small cross numerator is indispensable even
				// when the covariance itself is outside float64 range.
				exact, exactErr := statisticsExactCovariance(reader, j, k)
				if exactErr != nil {
					return nil, nil, nil, exactErr
				}
				value, err = statisticsExactPearson(exact, columns[j].scale, columns[k].scale, lengths[j], lengths[k], rows)
				if err != nil {
					return nil, nil, nil, err
				}
			} else {
				value = moment.acc.value() / lengths[j] / lengths[k]
			}
			if j == k {
				value = 1 // Positive variance fixes a mathematically exact diagonal.
			} else if value > 1 || value < -1 {
				// Only correct an overshoot within the explicitly bounded
				// floating-point budget; never silently clip a broken result.
				if math.Abs(value) > 1+statisticsPearsonRoundoff {
					return nil, nil, nil, ErrOverflow
				}
				value = math.Copysign(1, value)
			}
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, nil, nil, ErrOverflow
			}
			correlationD.data[j*cols+k] = value
			correlationD.data[k*cols+j] = value
		}
	}

	return correlationD, means, stds, nil
}

// statisticsNormalizeRowsL2 returns the original L2 norms and independently
// normalized rows using the Blue/Golub scaled sum-of-squares recurrence.
//
// Implementation:
//   - Stage 1: validate the Matrix and its shape, and handle empty-domain no-ops.
//   - Stage 2: scan every row for finite coordinates and a representable norm,
//     retaining only one scale and sqrt(ssq) per row.
//   - Stage 3: allocate the Dense output after all numeric validation succeeds;
//     divide each cell by its scale and sqrt(ssq), never by a tiny reciprocal.
//
// Zero vectors remain zero. An empty-domain input Matrix is returned by alias
// as specified by the existing no-op contract, while norms are fresh slices.
// Complexity: O(n*d) time, O(n*d+n) space including outputs.
func statisticsNormalizeRowsL2(input Matrix) (Matrix, []float64, error) {
	reader, err := newStatisticsReader(input)
	if err != nil {
		return nil, nil, err
	}
	norms := make([]float64, reader.rows)
	if reader.rows == 0 || reader.cols == 0 {
		return input, norms, nil
	}

	// A failed finite-value or overflow check must not consume an n*d
	// destination allocation. Two O(n) staging vectors suffice for the
	// second pass that materializes normalized rows.
	scales := make([]float64, reader.rows)
	lengths := make([]float64, reader.rows)
	for i := 0; i < reader.rows; i++ {
		// The scale tracks the largest magnitude seen so far; ssq stays
		// near the column count, avoiding the raw-square overflow defect.
		scale, ssq := 0.0, 1.0
		for j := 0; j < reader.cols; j++ {
			observation, readErr := reader.value(i, j)
			if readErr != nil {
				return nil, nil, readErr
			}
			absolute := math.Abs(observation)
			if absolute == 0 {
				continue
			}
			if absolute > scale {
				// Rescale prior contributions when a larger value arrives.
				ratio := scale / absolute
				ssq = 1 + ssq*ratio*ratio
				scale = absolute
			} else {
				// A smaller contribution cannot overflow its ratio square.
				ratio := absolute / scale
				ssq += ratio * ratio
			}
		}
		if scale == 0 {
			continue // An all-zero source row has an all-zero destination.
		}

		length := math.Sqrt(ssq)
		norm, scaleErr := statisticsProduct3(scale, length, 1)
		if scaleErr != nil {
			return nil, nil, scaleErr
		}
		norms[i] = norm
		scales[i] = scale
		lengths[i] = length
	}

	// Only validated inputs and representable row norms reach the materialize
	// stage. The output owns its storage and the input is never overwritten.
	out, err := NewDense(reader.rows, reader.cols)
	if err != nil {
		return nil, nil, err
	}
	for i := 0; i < reader.rows; i++ {
		if scales[i] == 0 {
			continue
		}
		for j := 0; j < reader.cols; j++ {
			observation, readErr := reader.value(i, j)
			if readErr != nil {
				return nil, nil, readErr
			}
			out.data[i*reader.cols+j] = (observation / scales[i]) / lengths[i]
		}
	}

	return out, norms, nil
}
