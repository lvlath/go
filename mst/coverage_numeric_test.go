// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst_test verifies integer admission at the actual allocation-size boundaries.
// The tests pass scalar counts to test-only bridges and never allocate impossible slices.
package mst_test

import (
	"math"
	"strconv"
	"testing"
	"unsafe"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/mst"
)

// TestSteinerNumeric_CheckedPairCount separates pair cardinality from backing-store size.
// Both overflow guards must reject before the quadratic closure is allocated.
// Implementation:
//   - Stage 1: Check zero, singleton, even and odd terminal cardinalities.
//   - Stage 2: Use architecture-aware counts for product and backing-store overflow.
//   - Stage 3: Require the precise count or zero with ErrSizeOverflow.
//
// Behavior highlights:
//   - Large invalid counts are integers only; no corresponding graph or slice is allocated.
//
// Inputs:
//   - t: active test context; counts are independent scalar fixtures.
//
// Returns:
//   - None; successful counts and error identities are asserted exactly.
//
// Errors:
//   - ErrSizeOverflow for negative counts, overflowing products, or excessive bytes.
//
// Determinism:
//   - Integer arithmetic uses the target architecture's int width and core.Edge size.
//
// Complexity:
//   - O(1) time and space per bounded case.
//
// Notes:
//   - The byte-overflow fixture has a representable pair count on both 32- and 64-bit Go.
//
// AI-Hints:
//   - Do not allocate a huge terminal slice to test a scalar admission guard.
func TestSteinerNumeric_CheckedPairCount(t *testing.T) {
	// The product fits int, but pairs of this cardinality cannot fit a core.Edge slice.
	const backingOverflowTerminals = 1 << (strconv.IntSize/2 - 1)
	cases := []struct {
		name  string
		count int
		want  int
		err   error
	}{
		{"negative", -1, 0, mst.ErrSizeOverflow},
		{"zero", 0, 0, nil},
		{"singleton", 1, 0, nil},
		{"pair", 2, 1, nil},
		{"odd", 3, 3, nil},
		{"even", 12, 66, nil},
		{"backing bytes", backingOverflowTerminals, 0, mst.ErrSizeOverflow},
		{"pair product", math.MaxInt, 0, mst.ErrSizeOverflow},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// The production helper must report its exact scalar outcome without allocating.
			got, err := mst.CheckedSteinerPairCountForTest(tc.count)
			if tc.err != nil {
				mustErrorIs(t, err, tc.err, "pair count admission")
			} else {
				mustNoError(t, err, "pair count admission")
			}
			mustEqualInt(t, got, tc.want, "pair count")
		})
	}
}

// TestSteinerNumeric_CheckedCapacityAndCountAdd verifies exact integer boundary behavior.
// The largest admissible backing size and integer sum succeed; the next count is rejected.
// Implementation:
//   - Stage 1: Derive an architecture-aware edge-slice element limit.
//   - Stage 2: Check invalid widths/counts, the exact limit, and the next element.
//   - Stage 3: Check signed-invalid and overflowing composite counts separately.
//
// Behavior highlights:
//   - Representability is not confused with available memory or caller resource limits.
//
// Inputs:
//   - t: active test context; counts and widths are scalars, not allocated buffers.
//
// Returns:
//   - None; the exact valid counts and zero-valued failures are asserted.
//
// Errors:
//   - ErrSizeOverflow for negative counts, zero width, and overflowing products/sums.
//
// Determinism:
//   - Uses math.MaxInt and unsafe.Sizeof only; no timing or allocation pressure.
//
// Complexity:
//   - O(1) time and space per case.
//
// Notes:
//   - unsafe.Sizeof is a compile-time type-width query, not unsafe memory access.
//
// AI-Hints:
//   - Avoid testing with a fabricated slice header or actual out-of-memory allocation.
func TestSteinerNumeric_CheckedCapacityAndCountAdd(t *testing.T) {
	const edgeBytes = unsafe.Sizeof(core.Edge{})
	const edgeLimit = math.MaxInt / int(edgeBytes)
	for _, tc := range []struct {
		name  string
		count int
		width uintptr
		want  int
		err   error
	}{
		{"negative", -1, edgeBytes, 0, mst.ErrSizeOverflow},
		{"zero width", 0, 0, 0, mst.ErrSizeOverflow},
		{"empty", 0, edgeBytes, 0, nil},
		{"exact limit", edgeLimit, edgeBytes, edgeLimit, nil},
		{"one above limit", edgeLimit + 1, edgeBytes, 0, mst.ErrSizeOverflow},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := mst.CheckedSteinerSliceCapacityForTest(tc.count, tc.width)
			if tc.err != nil {
				mustErrorIs(t, err, tc.err, "slice capacity")
			} else {
				mustNoError(t, err, "slice capacity")
			}
			mustEqualInt(t, got, tc.want, "capacity count")
		})
	}
	for _, tc := range []struct {
		name        string
		left, right int
		want        int
		err         error
	}{
		{"negative left", -1, 0, 0, mst.ErrSizeOverflow},
		{"negative right", 0, -1, 0, mst.ErrSizeOverflow},
		{"zero sum", 0, 0, 0, nil},
		{"max sum", math.MaxInt - 1, 1, math.MaxInt, nil},
		{"overflow sum", math.MaxInt, 1, 0, mst.ErrSizeOverflow},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := mst.CheckedSteinerCountAddForTest(tc.left, tc.right)
			if tc.err != nil {
				mustErrorIs(t, err, tc.err, "count addition")
			} else {
				mustNoError(t, err, "count addition")
			}
			mustEqualInt(t, got, tc.want, "composite count")
		})
	}
}
