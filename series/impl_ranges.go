// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package series enumerates complete, half-open sample-index windows.
// This file owns count arithmetic, checked allocation, and stable range order.
package series

import (
	"math"
	"unsafe"
)

// windowRanges creates complete [Start,End) index windows without partial tails.
// The first range starts at zero; each successive start advances by stride
// while the full width continues to fit within length.
//
// Implementation:
//   - Stage 1: Validate dimensions and handle empty successful output.
//   - Stage 2: Count windows through subtraction and division, then check bytes.
//   - Stage 3: Allocate once and append ordered, half-open index pairs.
//
// Behavior highlights:
//   - Width greater than length returns an empty non-nil slice, not an error.
//   - Stride need not divide width and may be greater than width.
//
// Inputs:
//   - length: available sample count, a nonnegative int.
//   - width: strictly positive sample count per complete window.
//   - stride: strictly positive increment of consecutive starting indices.
//
// Returns:
//   - []Range: non-nil, independent slice with End-Start == width per entry.
//
// Errors:
//   - ErrInvalidOption: invalid dimension or stride arguments.
//   - ErrOverflow: representable count requires unrepresentable slice storage.
//
// Determinism:
//   - Emits starts 0, stride, 2*stride, ... in strictly increasing order.
//
// Complexity:
//   - Time O(w), space O(w), for w complete windows.
//
// Notes:
//   - Byte-size checks prevent arithmetic overflow, not system-level OOM.
//   - The size bound follows the actual architecture's two-int Range layout.
//
// AI-Hints:
//   - Never evaluate start+width before proving the window fits into length.
//   - Checking only count <= MaxInt does not prove count*sizeof(Range) fits.
func windowRanges(length, width, stride int) ([]Range, error) {
	// No slice is present, so a zero input length is valid; width/stride are not.
	if length < 0 || width <= 0 || stride <= 0 {
		return nil, ErrInvalidOption
	}

	// An incomplete first window implies no complete windows at all.
	// Preserve the required non-nil empty-result distinction.
	if width > length {
		return make([]Range, 0), nil
	}

	// Subtract before dividing: 0 <= length-width <= MaxInt.
	// Therefore 1+(length-width)/stride stays representable as int.
	count := 1 + (length-width)/stride

	// unsafe.Sizeof is a compile-time layout query; no unsafe pointer is used.
	// The byte-size guard rejects allocations that cannot be represented by int
	// even though their element count is itself a valid int.
	if count > math.MaxInt/int(unsafe.Sizeof(Range{})) {
		return nil, ErrOverflow
	}

	// Preallocate exactly one backing array; slices are returned caller-owned.
	ranges := make([]Range, 0, count)
	start := 0

	for i := 0; i < count; i++ {
		// By the count formula, start <= length-width, so End <= length
		// without integer overflow; End remains exclusive.
		ranges = append(ranges, Range{Start: start, End: start + width})

		// Avoid calculating an unused final start. For every earlier index,
		// the next start also satisfies nextStart <= length-width, so the
		// increment is mathematically bounded by length.
		if i+1 < count {
			start += stride
		}
	}

	return ranges, nil
}
