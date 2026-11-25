// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// External validation tests for planned PAA shapes without allocating huge
// source slices or synthesizing invalid slice headers with unsafe/reflect.

package paa_test

import (
	"math"
	"strconv"
	"testing"

	"github.com/lvlath/go/paa"
)

const (
	// The independent storage oracle follows Segment's three exported int64
	// fields, not production-private allocation constants.
	testSegmentFields = 3
	testInt64Bytes    = 64 / 8
	testSegmentBytes  = testSegmentFields * testInt64Bytes
)

// TestValidateDimensions_Domain checks every public option error and ensures
// that the length-only preflight does not impersonate slice-based validation.
func TestValidateDimensions_Domain(t *testing.T) {
	cases := []struct {
		name     string
		length   int
		segments int
		want     error
	}{
		{name: "negativeLength", length: -1, segments: 1, want: paa.ErrInvalidOption},
		{name: "zeroLength", length: 0, segments: 1, want: paa.ErrInvalidOption},
		{name: "negativeSegments", length: 5, segments: -2, want: paa.ErrInvalidOption},
		{name: "zeroSegments", length: 5, segments: 0, want: paa.ErrInvalidOption},
		{name: "segmentsExceedLength", length: 5, segments: 6, want: paa.ErrInvalidOption},
		{name: "zeroPair", length: 0, segments: 0, want: paa.ErrInvalidOption},
		{name: "fractional", length: 5, segments: 2},
		{name: "singleSegment", length: 5, segments: 1},
		{name: "identity", length: 5, segments: 5},
		{name: "largestIntWithSingleSegment", length: math.MaxInt, segments: 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := paa.ValidateDimensions(tc.length, tc.segments)
			if tc.want != nil {
				mustErrorIs(t, err, tc.want)
			} else if err != nil {
				t.Fatalf("valid planned PAA dimensions rejected: %v", err)
			}
		})
	}
}

// TestValidateDimensions_StorageOverflow proves that overflow in the Segment
// output byte count is classified without allocating a single input sample.
// The sizing rule is meaningful on both 32-bit and 64-bit Go targets.
func TestValidateDimensions_StorageOverflow(t *testing.T) {
	// This is the first w at which 24*w cannot be represented by Go int.
	segments := math.MaxInt/testSegmentBytes + 1
	mustErrorIs(t, paa.ValidateDimensions(segments, segments), paa.ErrOverflow)

	// When w is inadmissible for n, the structural option must win over the
	// later storage overflow regardless of the configured word size.
	mustErrorIs(t, paa.ValidateDimensions(segments-1, segments), paa.ErrInvalidOption)
}

// TestValidateDimensions_CoordinateOverflow exercises n*w overflow without
// fabricating an enormous []float64. On 32-bit, two positive int counts cannot
// multiply past MaxInt64, so this representability error is unreachable there.
func TestValidateDimensions_CoordinateOverflow(t *testing.T) {
	if strconv.IntSize != 64 {
		t.Skip("signed 32-bit dimensions cannot overflow the int64 coordinate product")
	}

	// On 64-bit, the squared count exceeds MaxInt64 while its 24*w storage
	// byte count still fits the platform int. The two overflow checks therefore
	// exercise different failure classes inside the shared size gate.
	segments := int(1) << (strconv.IntSize/2 + 1)
	mustErrorIs(t, paa.ValidateDimensions(segments, segments), paa.ErrOverflow)
}

// TestValidateDimensions_MatchesTransform checks the preflight's agreement
// with actual successful Transform dimensions across small fractional grids.
func TestValidateDimensions_MatchesTransform(t *testing.T) {
	for length := 1; length <= 31; length++ {
		values := make([]float64, length)
		for j := range values {
			values[j] = float64(j - length/2)
		}
		for segments := 1; segments <= length; segments++ {
			if err := paa.ValidateDimensions(length, segments); err != nil {
				t.Fatalf("length=%d, w=%d preflight: %v", length, segments, err)
			}
			res, err := paa.Transform(values, segments)
			if err != nil {
				t.Fatalf("length=%d, w=%d Transform: %v", length, segments, err)
			}
			assertExactSegments(t, res, length, segments)
		}
	}
}

// TestValidateDimensions_NoAllocations ensures the constant-time preflight
// does not allocate even for dimensions that could never be materialized.
func TestValidateDimensions_NoAllocations(t *testing.T) {
	allocs := testing.AllocsPerRun(1000, func() {
		_ = paa.ValidateDimensions(math.MaxInt/testSegmentBytes+1, math.MaxInt/testSegmentBytes+1)
	})
	if allocs != 0 {
		t.Fatalf("preflight allocated %.1f times per call; want zero", allocs)
	}
}
