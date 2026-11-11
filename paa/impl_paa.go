// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package paa implements fractional Piecewise Aggregate Approximation using
// one monotone integer-overlap sweep. This file owns segment construction and
// delegates exact numeric accumulation and rounding to numeric.go.

package paa

// transformKernel constructs the complete PAA result from an already validated
// finite input and an exact checked endpoint grid. The input step function and
// equal-duration segment partition are intersected without rounding boundaries.
//
// Implementation:
//   - Stage 1: allocate independent O(w) result slices and prepare int64 units.
//   - Stage 2: use a detached identity copy when each segment is one full sample.
//   - Stage 3: sweep the source and output grids, adding each exact overlap.
//   - Stage 4: finalize one correctly-rounded mean at each segment boundary.
//   - Stage 5: return only the complete artifact, never an incomplete result.
//
// Behavior highlights:
//   - Source cell i occupies [i*w,(i+1)*w) in common integer units.
//   - Output cell j occupies [j*n,(j+1)*n) in the same integer units.
//   - Each event advances i, j, or both; the sweep visits at most n+w-1
//     positive overlaps (exactly n+w-gcd(n,w)).
//   - Samples crossing fractional segment boundaries contribute to both cells
//     with the correct positive integer overlap, never by duplication or padding.
//
// Inputs:
//   - values: read-only, nonempty finite samples with stable storage for call.
//   - w: required segment count in [1,n], already checked with n*w<=MaxInt64.
//
// Returns:
//   - *Result: detached finite averages, exact Segment metadata and n.
//   - error: nil on success; nil Result and ErrOverflow for arithmetic failure
//     or an impossible internal geometric invariant.
//
// Determinism:
//   - Monotone increasing source index i, segment index j, exact integer
//     endpoint comparisons and one finalization per segment; no sorting.
//
// Complexity:
//   - Time O(n+w) including output copying; additional space O(w) for the
//     result and O(1) fixed-width workspace per in-progress segment.
//
// Notes:
//   - Only the validated canonical facade may enter this kernel; this private
//     method must not silently reassign missing values or alter endpoint units.
//
// AI-Hints:
//   - Never implement nested loops over all (i,j) pairs: only adjacent
//     overlapping cells are visited in the monotone sweep.
//   - When inputRight==segmentRight, advance both indices, or a zero-overlap
//     boundary event may duplicate work or corrupt geometry.
//   - The identity path must copy float64 bits, including negative zero,
//     rather than reconstructing those values through an averaging operation.
func transformKernel(values []float64, w int) (*Result, error) {
	n := len(values)
	out := &Result{
		Values:         make([]float64, w),
		Segments:       make([]Segment, w),
		OriginalLength: n,
	}

	// The validator proved n*w fits int64. Keep both grids in that integer
	// domain to avoid converting any fractional boundary into float64.
	ni, wi := int64(n), int64(w)

	// Identity partitions provide one complete sample for every output cell.
	// Copy rather than re-rounding: this also preserves both IEEE-754 zero signs.
	// Populate canonical rational geometry as on the general scan path so every
	// caller observes the same Segment representation regardless of fast path.
	if w == n {
		copy(out.Values, values)
		for j := range out.Segments {
			left := int64(j) * ni
			out.Segments[j] = Segment{
				LeftNumerator:  left,
				RightNumerator: left + ni,
				Denominator:    wi,
			}
		}
		return out, nil
	}

	// i identifies the current original sample and j the current output cell.
	// Their endpoints partition the common [0,n*w) domain with no gaps.
	// The private accumulator holds only the *current* segment's weighted mass.
	var i, j int64
	var accumulator weightedAccumulator

	for j < wi {
		// Step 1: construct the two active half-open intervals. Both right
		// endpoints are at most n*w and therefore safe in checked int64.
		inputLeft, inputRight := i*wi, (i+1)*wi
		segmentLeft, segmentRight := j*ni, (j+1)*ni

		// Step 2: intersect them using integer max(lefts) and min(rights).
		// At each event both active intervals overlap: after an endpoint is
		// reached, its corresponding pointer advances to the next interval.
		left, right := inputLeft, inputRight
		if segmentLeft > left {
			left = segmentLeft
		}
		if segmentRight < right {
			right = segmentRight
		}
		overlap := right - left
		if overlap <= 0 {
			// A nonpositive overlap contradicts the contiguous grid and pointer
			// ordering invariants. Never publish partially filled result slices.
			return nil, ErrOverflow
		}

		// Step 3: accumulate x[i]*overlap in exact fixed-point form. The
		// weighted mean will divide by n *once* when this output cell ends;
		// no float64 product can overflow before that division.
		accumulator.add(values[int(i)], overlap)

		// Step 4: publish only when the segment's entire interval has been
		// covered. A sample crossing the boundary remains active for the
		// next output segment, but this segment's accumulator is reset.
		if inputRight >= segmentRight {
			mean, err := accumulator.finish(ni)
			if err != nil {
				return nil, err
			}
			out.Values[int(j)] = mean
			out.Segments[int(j)] = Segment{
				LeftNumerator:  segmentLeft,
				RightNumerator: segmentRight,
				Denominator:    wi,
			}
			accumulator = weightedAccumulator{}
			j++
		}

		// Step 5: retire an original sample when its right endpoint has
		// been fully consumed. Equal endpoints advance BOTH i and j;
		// otherwise the still-overlapping sample participates in the next
		// segment. Either branch makes monotone progress in O(1) work.
		if inputRight <= segmentRight {
			i++
		}
	}

	// Every output cell has been finalized. The two separately allocated
	// slices are now caller-owned and have no aliases into input or workspace.
	return out, nil
}
