// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 katalvlaran

package series

// windowRanges uses division before multiplication to avoid overflow in
// range-count calculation. Every computed end is bounded by length.
func windowRanges(length, width, stride int) ([]Range, error) {
	if length < 0 || width <= 0 || stride <= 0 {
		return nil, ErrInvalidOption
	}
	if width > length {
		return make([]Range, 0), nil
	}
	count := 1 + (length-width)/stride
	ranges := make([]Range, 0, count)
	start := 0
	for i := 0; i < count; i++ {
		ranges = append(ranges, Range{Start: start, End: start + width})
		if i+1 < count {
			// The subsequent start is at most length-width and thus fits int.
			start += stride
		}
	}
	return ranges, nil
}
